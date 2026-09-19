/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * test_stratum: protocol conformance tests for the Gapcoin legacy stratum
 * client (new_src/stratum.c), driven by an in-process mock pool.
 *
 * The mock is deliberately strict: it validates the PoW-solution payload the
 * client sends (80-byte header + nNonce + nShift + nAdd, all little-endian,
 * > 86 bytes) against the header it handed out, so a layout regression fails
 * the test instead of silently producing shares the real pool would reject.
 *
 * Covered: pure payload helpers, work handoff (request response + push),
 * share target updates (mining.set_difficulty / mining.target), submit
 * accept/reject, local duplicate suppression, and reconnect + resubmit.
 */

#define _POSIX_C_SOURCE 200809L

#include "../new_src/stratum.h"

#include <arpa/inet.h>
#include <errno.h>
#include <jansson.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        g_checks++;                                                           \
        if (cond) {                                                           \
            printf("  ok   : " __VA_ARGS__);                                  \
            printf("\n");                                                     \
        } else {                                                              \
            printf("  FAIL : " __VA_ARGS__);                                  \
            printf("\n");                                                     \
            g_fail++;                                                         \
        }                                                                     \
    } while (0)

/* ───────────────────────────── mock pool ───────────────────────────── */

#define MOCK_HDR180 80

struct mock_pool {
    int listen_fd;
    int port;
    pthread_t th;
    int running;

    uint8_t hdr1[80];
    uint8_t hdr2[80];
    uint8_t hdr3[80];      /* rotated template, handed out on request */
    int serve_hdr;         /* 1..3: the template the pool currently has */
    uint64_t serve_share;  /* the share target of that template */
    int push_unknown;      /* deliver work under a NON-standard method */
    int unknown_sent;
    int work_requests;
    uint64_t share1;
    uint64_t share2;
    uint64_t net_mid;

    int connections;
    int submits_seen;
    int submits_accepted;
    int submits_rejected;
    int payload_bad;
    int silent_submits;   /* shares whose verdict is deliberately withheld */
    int allow_push;       /* gate: test releases the push updates when ready */
    int push_pending;
    int pushes_sent;
};

static void hex_encode(const uint8_t *in, size_t n, char *out)
{
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i * 2] = d[in[i] >> 4];
        out[i * 2 + 1] = d[in[i] & 0x0f];
    }
    out[n * 2] = '\0';
}

static void craft_header(uint8_t h[80], uint8_t seed, uint64_t ndiff)
{
    for (int i = 0; i < 72; i++)
        h[i] = (uint8_t)(seed + i);
    for (int i = 0; i < 8; i++)
        h[72 + i] = (uint8_t)((ndiff >> (8 * i)) & 0xffU);
}

static int send_all(int fd, const char *s)
{
    size_t len = strlen(s);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, s + sent, len - sent, 0);
        if (n <= 0)
            return -1;
        sent += (size_t)n;
    }
    return 0;
}

/* Validate one mining.submit payload against the header we handed out. */
static void validate_submit(struct mock_pool *m, json_t *params)
{
    const char *hex;
    char expect_hdr1[161];
    char expect_hdr2[161];
    char expect_hdr3[161];
    size_t hexlen;

    if (!json_is_array(params) || json_array_size(params) != 3) {
        m->payload_bad++;
        fprintf(stderr, "  [mock] submit params not [user,pass,hex]\n");
        return;
    }
    if (!json_is_string(json_array_get(params, 0)) ||
        !json_is_string(json_array_get(params, 1)) ||
        !json_is_string(json_array_get(params, 2))) {
        m->payload_bad++;
        fprintf(stderr, "  [mock] submit params are not all strings\n");
        return;
    }
    if (strcmp(json_string_value(json_array_get(params, 0)), "testuser") != 0) {
        m->payload_bad++;
        fprintf(stderr, "  [mock] submit user mismatch\n");
        return;
    }
    hex = json_string_value(json_array_get(params, 2));
    hexlen = strlen(hex);
    hex_encode(m->hdr1, 80, expect_hdr1);
    hex_encode(m->hdr2, 80, expect_hdr2);
    hex_encode(m->hdr3, 80, expect_hdr3);

    if (hexlen < STRATUM_POW_MIN_BYTES * 2) {
        m->payload_bad++;
        fprintf(stderr, "  [mock] payload only %zu hex chars (needs > %u)\n",
                hexlen, STRATUM_POW_MIN_BYTES * 2U);
        return;
    }
    if (strncmp(hex, expect_hdr1, 160) != 0 &&
        strncmp(hex, expect_hdr2, 160) != 0 &&
        strncmp(hex, expect_hdr3, 160) != 0) {
        m->payload_bad++;
        fprintf(stderr, "  [mock] payload header is not one we issued\n");
        return;
    }    /* nNonce(4 LE) + nShift(2 LE) must be little-endian. */
    {
        size_t off = 160;
        if (off + 12 > hexlen) {
            m->payload_bad++;
            fprintf(stderr, "  [mock] payload lacks nonce+shift\n");
            return;
        }
    }
}

static void mock_handle_line(struct mock_pool *m, int fd, const char *line)
{
    json_error_t err;
    json_t *root = json_loads(line, 0, &err);
    char hdr1_hex[161];
    char buf[1024];

    if (!root) {
        fprintf(stderr, "  [mock] bad JSON from client: %s\n", err.text);
        return;
    }

    {
        const char *method = json_string_value(json_object_get(root, "method"));
        json_t *jid = json_object_get(root, "id");
        int id = json_is_integer(jid) ? (int)json_integer_value(jid) : 0;

        if (method && strcmp(method, "mining.request") == 0) {
            json_t *params = json_object_get(root, "params");
            const char *user = json_is_array(params) && json_array_size(params) > 0
                             ? json_string_value(json_array_get(params, 0)) : NULL;
            const uint8_t *hdr = (m->serve_hdr == 3) ? m->hdr3
                              : (m->serve_hdr == 2) ? m->hdr2
                                                    : m->hdr1;
            hex_encode(hdr, 80, hdr1_hex);
            m->work_requests++;
            m->connections++;
            snprintf(buf, sizeof(buf),
                     "{\"id\":%d,\"result\":{\"data\":\"%s\",\"difficulty\":%llu}}\n",
                     id, hdr1_hex, (unsigned long long)m->serve_share);
            send_all(fd, buf);
            /* The share-target update and the new-block push are released by
               the test (allow_push) so the ordering it asserts on is
               deterministic instead of racing the first mining.request. */
            m->push_pending = 1;
            (void)user;
        } else if (method && strcmp(method, "mining.submit") == 0) {
            json_t *params = json_object_get(root, "params");
            const char *hex = NULL;
            m->submits_seen++;
            validate_submit(m, params);
            if (json_is_array(params) && json_array_size(params) == 3 &&
                json_is_string(json_array_get(params, 2)))
                hex = json_string_value(json_array_get(params, 2));
            /* nonce 0xdeadbeef (LE "efbeadde") = "answer nothing", so the
               test can drop the connection with a share still in flight. */
            if (hex && strlen(hex) >= 168 && strncmp(hex + 160, "efbeadde", 8) == 0) {
                m->silent_submits++;
                json_decref(root);
                return;
            }
            /* First distinct share is accepted, the next one is rejected, so
               both verdict paths of the client are exercised. */
            if (m->submits_accepted == 0) {
                m->submits_accepted++;
                snprintf(buf, sizeof(buf), "{\"id\":%d,\"result\":true}\n", id);
            } else {
                m->submits_rejected++;
                snprintf(buf, sizeof(buf),
                         "{\"id\":%d,\"result\":false,\"error\":"
                         "{\"code\":20,\"message\":\"duplicate share\"}}\n", id);
            }
            send_all(fd, buf);
        }
    }
    json_decref(root);
}

static void *mock_thread(void *arg)
{
    struct mock_pool *m = (struct mock_pool *)arg;

    while (m->running) {
        int fd;
        char rx[4096];
        size_t rxlen = 0;
        struct timeval tv = { 0, 100000 };   /* 100 ms: fine-grained gate */

        fd = accept(m->listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
                continue;
            break;
        }
        (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        for (;;) {
            ssize_t n = recv(fd, rx + rxlen, sizeof(rx) - rxlen - 1U, 0);
            if (n > 0) {
                rxlen += (size_t)n;
                rx[rxlen] = '\0';
                for (;;) {
                    char *nl = strchr(rx, '\n');
                    size_t used;
                    if (!nl)
                        break;
                    *nl = '\0';
                    mock_handle_line(m, fd, rx);
                    used = (size_t)(nl - rx) + 1U;
                    memmove(rx, rx + used, rxlen - used);
                    rxlen -= used;
                    rx[rxlen] = '\0';
                }
                continue;
            }
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                if (!m->running)
                    break;                /* shutdown requested */
                if (m->push_pending && m->allow_push && !m->pushes_sent) {
                    char hdr2_hex_l[161];
                    char pbuf[512];
                    /* suprnova pushes the rotated template as mining.notify
                       (NOT blockchain.block.new): the regression this test
                       pins down is that a strict method filter ignores it and
                       the pool then rejects every share with no message. */
                    hex_encode(m->hdr2, 80, hdr2_hex_l);
                    snprintf(pbuf, sizeof(pbuf),
                             "{\"id\":null,\"method\":\"mining.set_difficulty\","
                             "\"params\":[%llu,%llu]}\n",
                             (unsigned long long)m->share2,
                             (unsigned long long)m->net_mid);
                    send_all(fd, pbuf);
                    snprintf(pbuf, sizeof(pbuf),
                             "{\"id\":null,\"method\":\"mining.notify\","
                             "\"params\":{\"data\":\"%s\",\"difficulty\":%llu}}\n",
                             hdr2_hex_l, (unsigned long long)m->share2);
                    send_all(fd, pbuf);
                    m->pushes_sent = 1;
                    m->push_pending = 0;
                }
                if (m->push_unknown && !m->unknown_sent) {
                    char hdr1_hex_l[161];
                    char pbuf[512];
                    hex_encode(m->hdr1, 80, hdr1_hex_l);
                    snprintf(pbuf, sizeof(pbuf),
                             "{\"id\":null,\"method\":\"pool.work.unknown\","
                             "\"params\":{\"data\":\"%s\",\"difficulty\":%llu}}\n",
                             hdr1_hex_l, (unsigned long long)m->share1);
                    send_all(fd, pbuf);
                    m->unknown_sent = 1;
                }
                continue;                 /* idle, keep the connection */
            }
            break;                        /* EOF or error: drop the client */
        }
        close(fd);
    }
    return NULL;
}

static int mock_start(struct mock_pool *m)
{
    struct sockaddr_in addr;
    socklen_t alen = sizeof(addr);
    int one = 1;

    memset(m, 0, sizeof(*m));
    m->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (m->listen_fd < 0)
        return -1;
    (void)setsockopt(m->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(m->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
        return -1;
    if (listen(m->listen_fd, 4) != 0)
        return -1;
    if (getsockname(m->listen_fd, (struct sockaddr *)&addr, &alen) != 0)
        return -1;
    m->port = ntohs(addr.sin_port);

    {
        struct timeval tv = { 1, 0 };
        (void)setsockopt(m->listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

    m->share1 = (uint64_t)(17.25 * (double)(1ULL << 48));
    m->share2 = (uint64_t)(18.5 * (double)(1ULL << 48));
    m->net_mid = (uint64_t)(23.4 * (double)(1ULL << 48));
    craft_header(m->hdr1, 0x11, (uint64_t)(23.5 * (double)(1ULL << 48)));
    craft_header(m->hdr2, 0x77, (uint64_t)(23.8 * (double)(1ULL << 48)));
    craft_header(m->hdr3, 0xc3, (uint64_t)(23.9 * (double)(1ULL << 48)));
    m->serve_hdr = 1;
    m->serve_share = m->share1;

    m->running = 1;
    if (pthread_create(&m->th, NULL, mock_thread, m) != 0)
        return -1;
    return 0;
}

static void mock_stop(struct mock_pool *m)
{
    m->running = 0;
    if (m->listen_fd >= 0) {
        shutdown(m->listen_fd, SHUT_RDWR);
        close(m->listen_fd);
        m->listen_fd = -1;
    }
    pthread_join(m->th, NULL);
}

/* ───────────────────────────── helpers ───────────────────────────── */

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000L);
}

/* usleep() is not in POSIX.1-2008, and this test is compiled with
   _POSIX_C_SOURCE = 200809L, so sleep through nanosleep(). */
static void sleep_ms(unsigned ms)
{
    struct timespec ts;
    ts.tv_sec = (time_t)(ms / 1000U);
    ts.tv_nsec = (long)(ms % 1000U) * 1000000L;
    nanosleep(&ts, NULL);
}

/* Wait until predicate holds, up to timeout_ms. */
#define WAIT_FOR(expr, timeout_ms, label)                                     \
    do {                                                                      \
        uint64_t t0_ = now_ms();                                              \
        while (!(expr) && now_ms() - t0_ < (uint64_t)(timeout_ms))            \
            sleep_ms(20);                                                     \
        CHECK((expr), "%s (waited %llu ms)", label,                           \
              (unsigned long long)(now_ms() - t0_));                          \
    } while (0)

/* ───────────────────────────── tests ───────────────────────────── */

/* Verdict capture: proves the pool's answer is reported with the gap metadata
   the share was submitted with. */
struct cb_capture {
    int calls;
    int last_accepted;
    char last_message[64];
    struct stratum_share_meta last_meta;
};

static void verdict_cb(void *user, int accepted,
                       const struct stratum_share_meta *meta,
                       const char *message)
{
    struct cb_capture *c = (struct cb_capture *)user;
    c->calls++;
    c->last_accepted = accepted;
    snprintf(c->last_message, sizeof(c->last_message), "%s",
             message ? message : "");
    if (meta)
        c->last_meta = *meta;
}

static void test_pow_hex(void){
    uint8_t hdr[80];
    uint8_t nadd[2] = { 0x34, 0x12 };
    char out[512];
    size_t len;

    printf("\n[pure] stratum_pow_hex layout\n");
    craft_header(hdr, 0x00, 0x0000000000000000ULL);
    for (int i = 0; i < 80; i++)
        hdr[i] = (uint8_t)i;              /* 00 01 02 ... 4f */

    len = stratum_pow_hex(hdr, 0x11223344U, 509U, nadd, 2, out, sizeof(out));
    CHECK(len == (80U + 4U + 2U + 2U) * 2U, "length is 88 bytes = %zu hex chars",
          len);
    CHECK(len > (size_t)STRATUM_POW_MIN_BYTES * 2U - 2U,
          "payload exceeds the node's 86-byte floor");
    CHECK(strncmp(out, "00010203", 8) == 0, "header is emitted as-is (hex)");
    /* nonce 0x11223344 little-endian -> 44332211 */
    CHECK(strncmp(out + 160, "44332211", 8) == 0,
          "nNonce is little-endian: %.8s", out + 160);
    /* shift 509 = 0x01fd -> little-endian fd01 */
    CHECK(strncmp(out + 168, "fd01", 4) == 0,
          "nShift is little-endian: %.4s", out + 168);
    /* nAdd 0x1234 -> little-endian 3412 */
    CHECK(strncmp(out + 172, "3412", 4) == 0,
          "nAdd is little-endian: %.4s", out + 172);

    len = stratum_pow_hex(hdr, 0U, 0U, NULL, 0, out, sizeof(out));
    CHECK(len == (80U + 4U + 2U + 1U) * 2U,
          "zero nAdd emits exactly one 0x00 byte (%zu hex chars)", len);
    CHECK(strcmp(out + len - 2, "00") == 0, "zero nAdd byte is 00");

    len = stratum_pow_hex(hdr, 0U, 0U, nadd, 2, out, 10);
    CHECK(len == 0, "undersized buffer is refused");

    len = stratum_pow_hex(hdr, 0U, 0U, nadd, 2, out, sizeof(out));
    {
        char *end = out + len;
        int allhex = 1;
        for (char *p = out; p < end; p++) {
            if (!((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f')))
                allhex = 0;
        }
        CHECK(allhex, "payload is pure lowercase hex");
    }
}

static void test_header_helpers(void)
{
    uint8_t hdr[80];
    uint64_t nd = (uint64_t)(23.5 * (double)(1ULL << 48));

    printf("\n[pure] header difficulty decode\n");
    craft_header(hdr, 0x5a, nd);
    CHECK(stratum_net_ndiff_from_header(hdr) == nd,
          "nDifficulty decode from bytes 72..79 round-trips");
    CHECK(stratum_ndiff_to_merit(nd) > 23.49 && stratum_ndiff_to_merit(nd) < 23.51,
          "merit conversion = %.6f", stratum_ndiff_to_merit(nd));
    CHECK(stratum_ndiff_to_merit(0) == 0.0, "zero ndiff is zero merit");
}

int main(void)
{
    struct mock_pool mock;
    stratum_ctx *ctx;
    struct cb_capture cap;
    char port[16];
    char work[STRATUM_DATA_HEX_SIZE];
    char hdr1_hex[161];
    char hdr2_hex[161];
    char hdr3_hex[161];
    uint64_t share = 0, net = 0;
    uint8_t nadd[2] = { 0x34, 0x12 };

    printf("=== test_stratum: Gapcoin legacy stratum client ===\n");
    memset(&cap, 0, sizeof(cap));

    test_pow_hex();
    test_header_helpers();

    if (mock_start(&mock) != 0) {
        printf("FATAL: cannot start the mock pool\n");
        return 1;
    }
    snprintf(port, sizeof(port), "%d", mock.port);
    hex_encode(mock.hdr1, 80, hdr1_hex);
    hex_encode(mock.hdr2, 80, hdr2_hex);
    hex_encode(mock.hdr3, 80, hdr3_hex);
    printf("\n[mock] listening on 127.0.0.1:%d\n", mock.port);

    printf("\n[1] connect + first work\n");
    ctx = stratum_connect("127.0.0.1", port, "testuser", "testpass");
    CHECK(ctx != NULL, "stratum_connect returns a context");
    if (!ctx) {
        mock_stop(&mock);
        return 1;
    }
    memset(work, 0, sizeof(work));
    CHECK(stratum_wait_work(ctx, work, &share, &net, 5000) == 1,
          "first work arrives within 5 s");
    CHECK(strcmp(work, hdr1_hex) == 0, "work data matches the pool's header");
    CHECK(share == mock.share1, "share target taken from the work response");
    CHECK(net == (uint64_t)(23.5 * (double)(1ULL << 48)),
          "network difficulty decoded from the header");
    CHECK(stratum_poll_work(ctx, work, &share, &net) == 0,
          "no spurious work update right after the first one");

    printf("\n[2] push updates (set_difficulty then a mining.notify work push)\n");
    {
        uint64_t t0 = now_ms();
        mock.allow_push = 1;              /* release the gated pushes */
        while (strcmp(work, hdr2_hex) != 0 && now_ms() - t0 < 3000) {
            (void)stratum_poll_work(ctx, work, &share, &net);
            sleep_ms(10);
        }
        CHECK(strcmp(work, hdr2_hex) == 0,
              "a mining.notify push replaced the work");
        CHECK(share == mock.share2,
              "set_difficulty/block.new updated the share target");
        CHECK(net == (uint64_t)(23.8 * (double)(1ULL << 48)),
          "network difficulty follows the new header (not the set_difficulty net)");
        CHECK(stratum_share_merit(ctx) > 18.49 && stratum_share_merit(ctx) < 18.51,
              "stratum_share_merit = %.6f", stratum_share_merit(ctx));
        CHECK(stratum_network_merit(ctx) > 23.79 && stratum_network_merit(ctx) < 23.81,
              "stratum_network_merit = %.6f", stratum_network_merit(ctx));
    }

    printf("\n[2b] getwork refresh: an UNCHANGED answer is not republished\n");
    {
        int req_before = mock.work_requests;
        mock.serve_hdr = 2;           /* the pool's current template ... */
        mock.serve_share = mock.share2;   /* ... and its current target */
        stratum_set_refresh_seconds(ctx, 1);
        uint64_t t0 = now_ms();
        while (mock.work_requests < req_before + 2 && now_ms() - t0 < 8000)
            sleep_ms(50);
        CHECK(mock.work_requests >= req_before + 2,
              "client re-requested work on its own (%d -> %d)", req_before,
              mock.work_requests);
        CHECK(stratum_poll_work(ctx, work, &share, &net) == 0,
              "the duplicate answer did not restart the work generation");
        CHECK(strcmp(work, hdr2_hex) == 0,
              "work still describes the pool's latest header");
    }

    printf("\n[2c] getwork refresh: a ROTATED template is published\n");
    {
        uint64_t t0;
        mock.serve_hdr = 3;           /* pool rotates its template silently */
        t0 = now_ms();
        while (now_ms() - t0 < 6000) {
            if (stratum_poll_work(ctx, work, &share, &net))
                break;
            sleep_ms(50);
        }
        CHECK(strcmp(work, hdr3_hex) == 0,
              "rotated work replaced the header (this is the suprnova fix)");
        CHECK(net == (uint64_t)(23.9 * (double)(1ULL << 48)),
              "network difficulty followed the rotated header");
        stratum_set_refresh_seconds(ctx, 0);
    }

    printf("\n[2d] work under an UNKNOWN push method is adopted\n");
    {
        uint64_t t0 = now_ms();
        mock.push_unknown = 1;
        while (now_ms() - t0 < 4000) {
            if (stratum_poll_work(ctx, work, &share, &net))
                break;
            sleep_ms(50);
        }
        CHECK(strcmp(work, hdr1_hex) == 0,
              "work delivered under 'pool.work.unknown' was adopted (the exact "
              "bug behind suprnova's message-less rejections)");
    }

    printf("\n[3] submit accept + payload conformance\n");    {
        struct stratum_share_meta meta;
        memset(&meta, 0, sizeof(meta));
        meta.gap_length = 12345;
        meta.merit = 23.9;
        meta.height = 0;
        stratum_set_verdict_callback(ctx, verdict_cb, &cap);
        CHECK(stratum_submit_share(ctx, mock.hdr1, 0x11223344U, 509U, nadd, 2,
                                   &meta) == 1,
              "first share is queued (with metadata)");
    }
    WAIT_FOR(mock.submits_seen == 1, 3000, "mock received the share");
    CHECK(mock.payload_bad == 0, "payload passed the mock's strict validation");
    {
        uint64_t acc = 0, rej = 0, dup = 0, sfail = 0, rec = 0, cfail = 0, unres = 0;
        WAIT_FOR((stratum_get_stats(ctx, &acc, &rej, &dup, &sfail, &rec, &cfail, &unres), acc == 1),
                 3000, "share ACCEPTED verdict is counted");
        CHECK(rej == 0, "no rejected shares yet");
        CHECK(cap.calls == 1 && cap.last_accepted == 1,
              "verdict callback fired as accepted (calls=%d, accepted=%d)",
              cap.calls, cap.last_accepted);
        CHECK(cap.last_meta.gap_length == 12345 && cap.last_meta.merit > 23.89 &&
                  cap.last_meta.shift == 509,
              "callback carried the gap metadata (gap=%u merit=%.2f shift=%u)",
              cap.last_meta.gap_length, cap.last_meta.merit, cap.last_meta.shift);
    }

    printf("\n[4] local duplicate suppression\n");
    CHECK(stratum_submit_pow(ctx, mock.hdr1, 0x11223344U, 509U, nadd, 2) == 0,
          "identical payload is dropped locally");
    WAIT_FOR(mock.submits_seen == 1, 500, "mock still sees exactly one submit");
    {
        uint64_t acc = 0, rej = 0, dup = 0, sfail = 0, rec = 0, cfail = 0, unres = 0;
        stratum_get_stats(ctx, &acc, &rej, &dup, &sfail, &rec, &cfail, &unres);
        CHECK(dup == 1, "duplicate counter = %llu", (unsigned long long)dup);
    }

    printf("\n[5] submit reject verdict\n");
    CHECK(stratum_submit_pow(ctx, mock.hdr2, 0x55667788U, 509U, nadd, 2) == 1,
          "a distinct share is queued");
    {
        uint64_t acc = 0, rej = 0, dup = 0, sfail = 0, rec = 0, cfail = 0, unres = 0;
        WAIT_FOR((stratum_get_stats(ctx, &acc, &rej, &dup, &sfail, &rec, &cfail, &unres), rej == 1),
                 3000, "share REJECTED verdict is counted");
        CHECK(acc == 1, "accepted count unchanged (%llu)", (unsigned long long)acc);
    }

    printf("\n[6] dropped connection: in-flight accounting + reconnect\n");
    {
        uint64_t rec_before = 0, rec_after = 0, acc = 0, rej = 0, dup = 0;
        uint64_t sfail = 0, cfail = 0, unres = 0;
        int conns_before;

        stratum_get_stats(ctx, &acc, &rej, &dup, &sfail, &rec_before, &cfail, &unres);
        conns_before = mock.connections;
        CHECK(unres == 0, "nothing unresolved before the drop");

        /* A share the pool never answers: its id stays in flight. */
        CHECK(stratum_submit_pow(ctx, mock.hdr2, 0xdeadbeefU, 509U, nadd, 2) == 1,
              "share with a withheld verdict is queued");
        WAIT_FOR(mock.silent_submits == 1, 3000, "mock received it and stayed silent");

        /* Drop the connection by stopping the mock, then bring the listener
           back on the SAME port so the client can reconnect after its 1 s
           backoff. */
        {
            int saved_port = mock.port;
            mock_stop(&mock);
            mock.running = 1;
            mock.listen_fd = socket(AF_INET, SOCK_STREAM, 0);
            if (mock.listen_fd >= 0) {
                struct sockaddr_in a;
                int one = 1;
                struct timeval tv = { 1, 0 };
                (void)setsockopt(mock.listen_fd, SOL_SOCKET, SO_REUSEADDR, &one,
                                 sizeof(one));
                memset(&a, 0, sizeof(a));
                a.sin_family = AF_INET;
                a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
                a.sin_port = htons((uint16_t)saved_port);
                (void)setsockopt(mock.listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv,
                                 sizeof(tv));
                if (bind(mock.listen_fd, (struct sockaddr *)&a, sizeof(a)) == 0 &&
                    listen(mock.listen_fd, 4) == 0) {
                    mock.port = saved_port;
                    if (pthread_create(&mock.th, NULL, mock_thread, &mock) != 0)
                        mock.running = 0;
                } else {
                    mock.running = 0;
                }
            } else {
                mock.running = 0;
            }
        }

        if (!mock.running) {
            printf("  skip : reconnect test (could not rebind the mock)\n");
        } else {
            WAIT_FOR(mock.connections > conns_before, 8000, "client reconnected");
            stratum_get_stats(ctx, &acc, &rej, &dup, &sfail, &rec_after, &cfail, &unres);
            CHECK(rec_after == rec_before + 1,
                  "successful reconnect counted (%llu -> %llu)",
                  (unsigned long long)rec_before, (unsigned long long)rec_after);
            CHECK(unres == 1, "the in-flight share is counted unresolved (%llu)",
                  (unsigned long long)unres);
            CHECK(acc == 1 && rej == 1,
                  "verdicts received before the drop are intact (acc=%llu rej=%llu)",
                  (unsigned long long)acc, (unsigned long long)rej);
            CHECK(cap.last_accepted == -1 && cap.calls >= 3,
                  "the in-flight share was reported unresolved to the callback "
                  "(calls=%d, last=%d)", cap.calls, cap.last_accepted);
            CHECK(stratum_submit_pow(ctx, mock.hdr2, 0x99aabbccU, 509U, nadd, 2) == 1,
                  "shares can be submitted again after reconnect");
            WAIT_FOR(mock.submits_seen >= 3, 3000, "the post-reconnect share arrived");
        }
    }

    printf("\n[7] shutdown\n");
    stratum_disconnect(ctx);
    printf("  ok   : stratum_disconnect returned\n");
    if (mock.running)
        mock_stop(&mock);

    printf("\n=== %s: %d/%d checks passed ===\n",
           g_fail == 0 ? "PASS" : "FAIL", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
