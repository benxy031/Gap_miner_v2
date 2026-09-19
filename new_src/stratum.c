/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gapcoin legacy stratum (getwork-over-JSON) pool client.
 * Protocol reference: see stratum.h and docs/POOL_STRATUM.md.
 *
 * Design notes that matter:
 *  - The receive path is a dedicated thread (no select() in the mining hot
 *    path), so a slow pool can never stall the workers.
 *  - Work is published under work_lock and signalled, exactly like the RPC
 *    thread's template handoff in main.c.
 *  - Shares are queued non-blocking: the caller (BPSW-verified gap in main.c)
 *    must never block on a socket.
 *  - Duplicate payloads are dropped locally.  The pool rejects them, and a
 *    re-submitted nAdd after a header rotation is a duplicate by definition
 *    when the payload is byte-identical.
 *  - In-flight submit ids are lost on reconnect, so their verdicts are counted
 *    as "unresolved" (logged once) instead of being mis-credited.
 */

#define _POSIX_C_SOURCE 200809L

#include "stratum.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <jansson.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define STRATUM_HOST_MAX 256
#define STRATUM_PORT_MAX 16
#define STRATUM_USER_MAX 128
#define STRATUM_PASS_MAX 128

#define STRATUM_RECV_BUF_INIT 4096U
#define STRATUM_RECV_BUF_MAX (256U * 1024U)

#define STRATUM_SUBMIT_ID_RING 64
#define STRATUM_DEDUP_RING 256

#define STRATUM_CONNECT_TIMEOUT_S 10
#define STRATUM_RECV_TIMEOUT_S 1

/* Default work-refresh period.  suprnova accepts work for roughly 20 s and
   then rejects shares built on the old header without any message, so 10 s
   keeps a comfortable margin while costing nothing when the work is unchanged
   (the response is compared and identical work is not republished). */
#define STRATUM_REFRESH_DEFAULT_S 10U

/* One queued share: its JSON id plus the gap it belongs to, so the verdict can
   be logged against the right candidate. */
struct stratum_pending {
    int id;
    struct stratum_share_meta meta;
};

struct stratum_ctx {
    char host[STRATUM_HOST_MAX];
    char port[STRATUM_PORT_MAX];
    char user[STRATUM_USER_MAX];
    char pass[STRATUM_PASS_MAX];

    int sock;
    int running;
    int connected;
    int ever_connected;
    pthread_t recv_thread;
    int recv_thread_started;

    pthread_mutex_t send_lock;   /* socket writes, pending ring, dedup ring */
    int msg_id;
    struct stratum_pending pending[STRATUM_SUBMIT_ID_RING];
    int pending_count;

    stratum_verdict_cb verdict_cb;
    void *verdict_user;

    pthread_mutex_t work_lock;   /* published work + cond */
    pthread_cond_t work_cond;
    char work_data[STRATUM_DATA_HEX_SIZE];
    uint64_t share_ndiff;
    uint64_t net_ndiff;
    int work_ready;
    int work_new;

    uint64_t dedup[STRATUM_DEDUP_RING];
    int dedup_count;

    unsigned refresh_s;          /* 0 = never re-request work */
    time_t last_request;         /* when work was last requested */
    int consecutive_rejects;     /* forces an immediate refresh */
    unsigned work_changes;       /* times the pool actually changed its work */

    uint64_t accepted;
    uint64_t rejected;
    uint64_t duplicates;
    uint64_t send_failures;
    uint64_t reconnects;
    uint64_t connect_failures;
    uint64_t unresolved;

    char *recv_buf;
    size_t recv_buf_len;
    size_t recv_buf_cap;
};

/* ───────────────────────────── helpers ───────────────────────────── */

static int stratum_debug_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("STRATUM_DEBUG");
        cached = (v && *v && strcmp(v, "0") != 0) ? 1 : 0;
    }
    return cached;
}

static void stratum_log(const char *fmt, ...)
{
    va_list ap;
    fputs("[stratum] ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static uint64_t fnv1a64(const char *s)
{
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t stratum_net_ndiff_from_header(const uint8_t hdr80[STRATUM_HDR80_SIZE])
{
    uint64_t nd = 0;
    if (!hdr80)
        return 0;
    for (int i = 7; i >= 0; i--)
        nd = (nd << 8) | (uint64_t)hdr80[72 + i];
    return nd;
}

double stratum_ndiff_to_merit(uint64_t ndiff)
{
    return (double)ndiff / (double)(1ULL << 48);
}

size_t stratum_pow_hex(const uint8_t hdr80[STRATUM_HDR80_SIZE], uint32_t nonce,
                       uint16_t shift, const uint8_t *nadd, size_t nadd_len,
                       char *out, size_t out_cap)
{
    static const char hexd[] = "0123456789abcdef";
    size_t n = nadd_len;
    size_t need;

    if (!hdr80 || !out)
        return 0;
    if (n == 0)
        n = 1;                     /* nAdd is at least one byte */
    need = (STRATUM_HDR80_SIZE + 4U + 2U + n) * 2U + 1U;
    if (out_cap < need)
        return 0;

    {
        uint8_t shift_le[2] = { (uint8_t)(shift & 0xffU), (uint8_t)(shift >> 8) };
        uint8_t nonce_le[4] = { (uint8_t)(nonce & 0xffU),
                                (uint8_t)((nonce >> 8) & 0xffU),
                                (uint8_t)((nonce >> 16) & 0xffU),
                                (uint8_t)((nonce >> 24) & 0xffU) };
        size_t pos = 0;

        for (size_t i = 0; i < STRATUM_HDR80_SIZE; i++) {
            out[pos++] = hexd[hdr80[i] >> 4];
            out[pos++] = hexd[hdr80[i] & 0x0f];
        }
        for (size_t i = 0; i < 4; i++) {
            out[pos++] = hexd[nonce_le[i] >> 4];
            out[pos++] = hexd[nonce_le[i] & 0x0f];
        }
        for (size_t i = 0; i < 2; i++) {
            out[pos++] = hexd[shift_le[i] >> 4];
            out[pos++] = hexd[shift_le[i] & 0x0f];
        }
        /* nAdd is serialized little-endian; a caller-supplied length of 0
           means "the value zero" and is emitted as a single 0x00 byte. */
        if (nadd_len == 0) {
            out[pos++] = '0';
            out[pos++] = '0';
        } else {
            for (size_t i = 0; i < n; i++) {
                out[pos++] = hexd[nadd[i] >> 4];
                out[pos++] = hexd[nadd[i] & 0x0f];
            }
        }
        out[pos] = '\0';
        return pos;
    }
}

/* ───────────────────────────── socket layer ───────────────────────────── */

static void sock_close_safe(int *s)
{
    if (*s >= 0) {
        shutdown(*s, SHUT_RDWR);
        close(*s);
        *s = -1;
    }
}

/* Non-blocking connect with a bounded wait, so a black-holed pool host cannot
   hang the miner for the kernel's default ~2 minute SYN timeout. */
static int tcp_connect_timeout(const char *host, const char *port, int timeout_s)
{
    struct addrinfo hints;
    struct addrinfo *res = NULL;
    int sock = -1;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0 || !res) {
        stratum_log("resolve failed for %s:%s", host, port);
        return -1;
    }

    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int flags;
        int rc;
        fd_set wfds;
        struct timeval tv;

        sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (sock < 0)
            continue;

        flags = fcntl(sock, F_GETFL, 0);
        if (flags >= 0)
            (void)fcntl(sock, F_SETFL, flags | O_NONBLOCK);

        rc = connect(sock, ai->ai_addr, ai->ai_addrlen);
        if (rc != 0 && errno != EINPROGRESS) {
            sock_close_safe(&sock);
            continue;
        }
        if (rc == 0)
            goto connected;

        FD_ZERO(&wfds);
        FD_SET(sock, &wfds);
        tv.tv_sec = timeout_s;
        tv.tv_usec = 0;
        rc = select(sock + 1, NULL, &wfds, NULL, &tv);
        if (rc <= 0) {
            stratum_log("connect timeout to %s:%s", host, port);
            sock_close_safe(&sock);
            continue;
        }
        {
            int soerr = 0;
            socklen_t slen = sizeof(soerr);
            if (getsockopt(sock, SOL_SOCKET, SO_ERROR, &soerr, &slen) != 0 ||
                soerr != 0) {
                sock_close_safe(&sock);
                continue;
            }
        }
    connected:
        if (flags >= 0)
            (void)fcntl(sock, F_SETFL, flags);
        {
            int one = 1;
            (void)setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }
        {
            struct timeval rtv = { STRATUM_RECV_TIMEOUT_S, 0 };
            (void)setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));
        }
        break;
    }

    freeaddrinfo(res);
    if (sock < 0)
        stratum_log("connect failed for %s:%s", host, port);
    return sock;
}

static int stratum_send_raw(stratum_ctx *ctx, const char *buf, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(ctx->sock, buf + sent, len - sent, 0);
        if (n > 0) {
            sent += (size_t)n;
            continue;
        }
        if (n < 0 && (errno == EINTR))
            continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            fd_set wfds;
            struct timeval tv = { STRATUM_RECV_TIMEOUT_S, 0 };
            FD_ZERO(&wfds);
            FD_SET(ctx->sock, &wfds);
            if (select(ctx->sock + 1, NULL, &wfds, NULL, &tv) > 0)
                continue;
        }
        return -1;
    }
    return 0;
}

/* Blocking single line read (bounded by SO_RCVTIMEO, which yields -1/EAGAIN
   so the caller can re-check ctx->running).  Returns line length, or -1. */
static int stratum_recv_line(stratum_ctx *ctx, char **out)
{
    if (!ctx->recv_buf) {
        ctx->recv_buf_cap = STRATUM_RECV_BUF_INIT;
        ctx->recv_buf = malloc(ctx->recv_buf_cap);
        if (!ctx->recv_buf)
            return -1;
        ctx->recv_buf_len = 0;
    }

    for (;;) {
        for (size_t i = 0; i < ctx->recv_buf_len; i++) {
            if (ctx->recv_buf[i] == '\n') {
                size_t linelen = i;
                char *line = malloc(linelen + 1);
                if (!line)
                    return -1;
                memcpy(line, ctx->recv_buf, linelen);
                line[linelen] = '\0';
                if (linelen > 0 && line[linelen - 1] == '\r')
                    line[linelen - 1] = '\0';
                {
                    size_t remaining = ctx->recv_buf_len - i - 1;
                    if (remaining > 0)
                        memmove(ctx->recv_buf, ctx->recv_buf + i + 1, remaining);
                    ctx->recv_buf_len = remaining;
                }
                *out = line;
                return (int)linelen;
            }
        }

        if (ctx->recv_buf_len + 1024U > ctx->recv_buf_cap) {
            size_t new_cap;
            char *tmp;
            if (ctx->recv_buf_cap >= STRATUM_RECV_BUF_MAX) {
                stratum_log("line exceeds %u bytes; dropping connection",
                            (unsigned)STRATUM_RECV_BUF_MAX);
                return -1;
            }
            new_cap = ctx->recv_buf_cap * 2U;
            if (new_cap > STRATUM_RECV_BUF_MAX)
                new_cap = STRATUM_RECV_BUF_MAX;
            tmp = realloc(ctx->recv_buf, new_cap);
            if (!tmp)
                return -1;
            ctx->recv_buf = tmp;
            ctx->recv_buf_cap = new_cap;
        }

        {
            ssize_t n = recv(ctx->sock, ctx->recv_buf + ctx->recv_buf_len,
                             ctx->recv_buf_cap - ctx->recv_buf_len - 1U, 0);
            if (n > 0) {
                ctx->recv_buf_len += (size_t)n;
                continue;
            }
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                return -2;         /* idle tick, not an error */
            if (n < 0 && errno == EINTR)
                continue;
            return -1;
        }
    }
}

/* ───────────────────────────── work / dedup state ───────────────────────── */

static void dedup_reset_locked(stratum_ctx *ctx)
{
    ctx->dedup_count = 0;
}

static int dedup_seen_locked(stratum_ctx *ctx, uint64_t h)
{
    for (int i = 0; i < ctx->dedup_count; i++) {
        if (ctx->dedup[i] == h)
            return 1;
    }
    if (ctx->dedup_count < STRATUM_DEDUP_RING) {
        ctx->dedup[ctx->dedup_count++] = h;
    } else {
        /* Ring full: replace the oldest entry (position 0) and rotate. */
        memmove(&ctx->dedup[0], &ctx->dedup[1],
                sizeof(ctx->dedup[0]) * (STRATUM_DEDUP_RING - 1));
        ctx->dedup[STRATUM_DEDUP_RING - 1] = h;
    }
    return 0;
}

static void parse_work_params(stratum_ctx *ctx, json_t *params)
{
    json_t *jdata;
    json_t *jdiff;
    const char *data;
    uint64_t ndiff;
    uint64_t net_ndiff = 0;

    if (!json_is_object(params)) {
        stratum_log("work: params is not an object");
        return;
    }
    jdata = json_object_get(params, "data");
    jdiff = json_object_get(params, "difficulty");
    if (!json_is_string(jdata)) {
        stratum_log("work: missing 'data'");
        return;
    }
    data = json_string_value(jdata);
    if (!data || strlen(data) < 160) {
        stratum_log("work: 'data' too short (%zu hex chars)",
                    data ? strlen(data) : 0U);
        return;
    }
    if (json_is_integer(jdiff)) {
        ndiff = (uint64_t)json_integer_value(jdiff);
    } else if (json_is_number(jdiff)) {
        ndiff = (uint64_t)json_number_value(jdiff);
    } else {
        stratum_log("work: missing numeric 'difficulty'");
        return;
    }

    {
        uint8_t hdr80[STRATUM_HDR80_SIZE];
        for (size_t i = 0; i < STRATUM_HDR80_SIZE; i++) {
            unsigned int bv = 0;
            if (sscanf(data + i * 2, "%2x", &bv) != 1) {
                stratum_log("work: 'data' is not hex");
                return;
            }
            hdr80[i] = (uint8_t)bv;
        }
        net_ndiff = stratum_net_ndiff_from_header(hdr80);
    }

    /*
     * Publishing rule: only a REAL change is published.  A periodic refresh
     * that returns the same header must not bump the work generation, because
     * every generation bump restarts the chains and drops queued shares (the
     * old header's solutions become "stale" by our own policy).  A difficulty
     * change alone is still published, because it moves the threshold.
     */
    pthread_mutex_lock(&ctx->work_lock);
    if (ctx->work_ready &&
        strncmp(ctx->work_data, data, 160) == 0 &&
        ctx->share_ndiff == ndiff && ctx->net_ndiff == net_ndiff) {
        ctx->work_ready = 1;   /* same work: nothing to republish */
        pthread_mutex_unlock(&ctx->work_lock);
        if (stratum_debug_enabled())
            stratum_log("work refresh: unchanged");
        return;
    }
    if (ctx->work_ready)
        ctx->work_changes++;
    memcpy(ctx->work_data, data, 160);
    ctx->work_data[160] = '\0';
    ctx->share_ndiff = ndiff;
    ctx->net_ndiff = net_ndiff;
    ctx->work_ready = 1;
    ctx->work_new = 1;
    pthread_cond_broadcast(&ctx->work_cond);
    pthread_mutex_unlock(&ctx->work_lock);

    pthread_mutex_lock(&ctx->send_lock);
    dedup_reset_locked(ctx);
    pthread_mutex_unlock(&ctx->send_lock);

    stratum_log("new work: share=%.6f merit, network=%.6f merit%s",
                stratum_ndiff_to_merit(ndiff),
                stratum_ndiff_to_merit(net_ndiff),
                (ctx->work_changes > 0) ? " (pool rotated its template)" : "");
}

static void parse_set_difficulty(stratum_ctx *ctx, json_t *params)
{
    uint64_t share = 0;
    uint64_t net = 0;
    int got_share = 0;
    int got_net = 0;

    if (json_is_array(params) && json_array_size(params) >= 1) {
        json_t *v = json_array_get(params, 0);
        if (json_is_integer(v)) {
            share = (uint64_t)json_integer_value(v);
            got_share = 1;
        } else if (json_is_number(v)) {
            share = (uint64_t)json_number_value(v);
            got_share = 1;
        }
        if (json_array_size(params) >= 2) {
            json_t *v2 = json_array_get(params, 1);
            if (json_is_integer(v2)) {
                net = (uint64_t)json_integer_value(v2);
                got_net = 1;
            } else if (json_is_number(v2)) {
                net = (uint64_t)json_number_value(v2);
                got_net = 1;
            }
        }
    } else if (json_is_object(params)) {
        json_t *v = json_object_get(params, "difficulty");
        if (!v)
            v = json_object_get(params, "target");
        if (json_is_integer(v)) {
            share = (uint64_t)json_integer_value(v);
            got_share = 1;
        } else if (json_is_number(v)) {
            share = (uint64_t)json_number_value(v);
            got_share = 1;
        }
        v = json_object_get(params, "network_difficulty");
        if (!v)
            v = json_object_get(params, "network");
        if (!v)
            v = json_object_get(params, "ndiff");
        if (json_is_integer(v)) {
            net = (uint64_t)json_integer_value(v);
            got_net = 1;
        } else if (json_is_number(v)) {
            net = (uint64_t)json_number_value(v);
            got_net = 1;
        }
    }

    if (!got_share || share == 0) {
        stratum_log("set_difficulty: no usable share target");
        return;
    }

    pthread_mutex_lock(&ctx->work_lock);
    ctx->share_ndiff = share;
    if (got_net && net > 0)
        ctx->net_ndiff = net;
    ctx->work_new = 1;
    pthread_cond_broadcast(&ctx->work_cond);
    pthread_mutex_unlock(&ctx->work_lock);

    stratum_log("pool set share target: %.6f merit%s", stratum_ndiff_to_merit(share),
                (got_net && net > 0) ? " (network value updated too)" : "");
}

/* Called with send_lock held.  Returns 1 if id was a pending share, copying
   its metadata into *out. */
static int pending_take_locked(stratum_ctx *ctx, int id,
                               struct stratum_share_meta *out)
{
    for (int i = 0; i < ctx->pending_count; i++) {
        if (ctx->pending[i].id == id) {
            if (out)
                *out = ctx->pending[i].meta;
            ctx->pending_count--;
            for (int j = i; j < ctx->pending_count; j++)
                ctx->pending[j] = ctx->pending[j + 1];
            return 1;
        }
    }
    return 0;
}

/* Drop every in-flight share and report them to the callback as unresolved.
   Called with send_lock held; the callback is invoked WITHOUT the lock. */
static void pending_flush_unresolved_locked(stratum_ctx *ctx,
                                            struct stratum_pending *saved,
                                            int *saved_n)
{
    *saved_n = ctx->pending_count;
    for (int i = 0; i < ctx->pending_count; i++)
        saved[i] = ctx->pending[i];
    ctx->pending_count = 0;
    ctx->unresolved += (uint64_t)*saved_n;
}

static void handle_line(stratum_ctx *ctx, const char *line)
{
    json_error_t err;
    json_t *root;

    if (stratum_debug_enabled())
        stratum_log("<< %s", line);

    root = json_loads(line, 0, &err);
    if (!root) {
        stratum_log("bad JSON from pool: %s", err.text);
        return;
    }
    if (!json_is_object(root)) {
        json_decref(root);
        return;
    }

    {
        json_t *jid = json_object_get(root, "id");
        json_t *jresult = json_object_get(root, "result");
        json_t *jerror = json_object_get(root, "error");
        json_t *jmethod = json_object_get(root, "method");
        json_t *jparams = json_object_get(root, "params");

        if (json_is_integer(jid)) {
            int id = (int)json_integer_value(jid);
            int is_submit;
            struct stratum_share_meta meta;
            memset(&meta, 0, sizeof(meta));
            pthread_mutex_lock(&ctx->send_lock);
            is_submit = pending_take_locked(ctx, id, &meta);
            pthread_mutex_unlock(&ctx->send_lock);

            if (is_submit) {
                if (json_is_true(jresult)) {
                    uint64_t v;
                    pthread_mutex_lock(&ctx->send_lock);
                    v = ++ctx->accepted;
                    pthread_mutex_unlock(&ctx->send_lock);
                    stratum_log("share ACCEPTED (id=%d) [accepted=%llu]", id,
                                (unsigned long long)v);
                    if (ctx->verdict_cb)
                        ctx->verdict_cb(ctx->verdict_user, 1, &meta, "accepted");
                    ctx->consecutive_rejects = 0;
                } else if (json_is_boolean(jresult) || json_is_null(jresult)) {
                    uint64_t v;
                    const char *msg = json_is_object(jerror)
                        ? json_string_value(json_object_get(jerror, "message"))
                        : NULL;
                    pthread_mutex_lock(&ctx->send_lock);
                    v = ++ctx->rejected;
                    pthread_mutex_unlock(&ctx->send_lock);
                    if (msg) {
                        stratum_log("share REJECTED (id=%d): %s [rejected=%llu]",
                                    id, msg, (unsigned long long)v);
                    } else {
                        stratum_log("share REJECTED (id=%d) [rejected=%llu]",
                                    id, (unsigned long long)v);
                    }
                    if (ctx->verdict_cb)
                        ctx->verdict_cb(ctx->verdict_user, 0, &meta,
                                        msg ? msg : "rejected");
                    /* Two rejections in a row look like a rotated template
                       (the pool answers with no message, so this is the only
                       signal available): drop the refresh timer to force an
                       immediate work request on the next loop pass. */
                    if (++ctx->consecutive_rejects >= 2)
                        ctx->last_request = 0;
                } else if (json_is_object(jresult)) {
                    /* Some pool builds answer a submit with the work object. */
                    parse_work_params(ctx, jresult);
                } else {
                    stratum_log("share verdict unrecognized (id=%d)", id);
                }
            } else if (json_is_object(jresult)) {
                parse_work_params(ctx, jresult);
            } else if (json_is_boolean(jresult) || json_is_null(jresult)) {
                stratum_log("response id=%d with result=%s (no pending submit?)",
                            id, json_is_null(jresult) ? "null" : "bool");
            }

            if (json_is_object(jerror)) {
                const char *msg = json_string_value(json_object_get(jerror, "message"));
                const char *code = NULL;
                json_t *jc = json_object_get(jerror, "code");
                if (json_is_integer(jc))
                    code = json_dumps(jc, JSON_COMPACT);
                stratum_log("pool error (id=%d): %s%s%s", id,
                            msg ? msg : "?", code ? " code=" : "",
                            code ? code : "");
                free((void *)code);
            }
        } else if (json_is_string(jmethod)) {
            const char *method = json_string_value(jmethod);
            if (strcmp(method, "mining.set_difficulty") == 0 ||
                strcmp(method, "mining.target") == 0) {
                parse_set_difficulty(ctx, jparams);
            } else if (json_is_object(jparams)) {
                /*
                 * Any push that carries a work object IS work.
                 *
                 * The legacy Gapcoin protocol documents blockchain.block.new,
                 * but suprnova pushes the rotated template as
                 * "mining.notify" -- while its mining.request keeps answering
                 * with the template cached at session start.  A client that
                 * filters strictly on the method name therefore never learns
                 * the pool moved on, and the failure is SILENT: the pool
                 * rejects every later share with no error message, which looks
                 * like a difficulty or payload problem and is neither
                 * (measured 2026-09-19: 108 accepted, then 202 rejected in a
                 * row, with six ignored mining.notify pushes in between; the
                 * work requests in that same session returned one single
                 * template).  The reference client accepts any object-params
                 * push, which is why it works on this pool.
                 */
                parse_work_params(ctx, jparams);
            } else if (json_is_array(jparams) && json_array_size(jparams) >= 1 &&
                       json_is_object(json_array_get(jparams, 0))) {
                /* Some pool builds wrap the work object in a 1-element array. */
                parse_work_params(ctx, json_array_get(jparams, 0));
            } else if (method && strcmp(method, "blockchain.block.new") == 0) {
                stratum_log("work push '%s' carried no work object", method);
            } else if (stratum_debug_enabled()) {
                stratum_log("push method '%s' carried no work object", method);
            }
        }
    }

    json_decref(root);
}

/* ───────────────────────────── protocol writers ───────────────────────────── */

static int stratum_send_getwork_locked(stratum_ctx *ctx)
{
    char buf[512];
    int id = ctx->msg_id++;
    int len = snprintf(buf, sizeof(buf),
                       "{\"id\":%d,\"method\":\"mining.request\","
                       "\"params\":[\"%s\",\"%s\"]}\n",
                       id, ctx->user, ctx->pass);
    if (len < 0 || (size_t)len >= sizeof(buf))
        return -1;
    if (stratum_debug_enabled())
        stratum_log(">> %s", buf);
    return stratum_send_raw(ctx, buf, (size_t)len);
}

/* ───────────────────────────── receive thread ───────────────────────────── */

static int stratum_reconnect_locked(stratum_ctx *ctx)
{
    int sock;
    int was_ever_connected = ctx->ever_connected;
    sock = tcp_connect_timeout(ctx->host, ctx->port, STRATUM_CONNECT_TIMEOUT_S);
    if (sock < 0) {
        ctx->connect_failures++;
        return 0;
    }

    ctx->sock = sock;
    {
        struct stratum_pending lost[STRATUM_SUBMIT_ID_RING];
        int lost_n = 0;
        stratum_verdict_cb cb;
        void *cb_user;

        pending_flush_unresolved_locked(ctx, lost, &lost_n);
        cb = ctx->verdict_cb;
        cb_user = ctx->verdict_user;
        if (lost_n > 0)
            stratum_log("%d in-flight share(s) lost to reconnect "
                        "(counted unresolved)", lost_n);
        pthread_mutex_unlock(&ctx->send_lock);
        if (cb) {
            for (int i = 0; i < lost_n; i++)
                cb(cb_user, -1, &lost[i].meta, "connection lost");
        }
        pthread_mutex_lock(&ctx->send_lock);
    }
    ctx->recv_buf_len = 0;
    if (stratum_send_getwork_locked(ctx) != 0) {
        sock_close_safe(&ctx->sock);
        ctx->connect_failures++;
        return 0;
    }
    ctx->last_request = time(NULL);
    ctx->consecutive_rejects = 0;
    ctx->connected = 1;
    ctx->ever_connected = 1;
    if (was_ever_connected)
        ctx->reconnects++;   /* a real reconnect, not the initial connect */
    return 1;
}

static void *stratum_recv_thread(void *arg)
{
    stratum_ctx *ctx = (stratum_ctx *)arg;
    unsigned backoff_ms = 0;

    while (ctx->running) {
        char *line = NULL;
        int rc;

        if (!ctx->connected) {
            int ok;
            pthread_mutex_lock(&ctx->send_lock);
            ok = stratum_reconnect_locked(ctx);
            pthread_mutex_unlock(&ctx->send_lock);
            if (!ok) {
                if (backoff_ms == 0)
                    backoff_ms = 1000;
                else if (backoff_ms < 30000)
                    backoff_ms *= 2;
                if (backoff_ms > 30000)
                    backoff_ms = 30000;
                stratum_log("reconnect failed; retrying in %u ms (host=%s:%s)",
                            backoff_ms, ctx->host, ctx->port);
                {
                    struct timespec ts;
                    ts.tv_sec = (time_t)(backoff_ms / 1000U);
                    ts.tv_nsec = (long)(backoff_ms % 1000U) * 1000000L;
                    nanosleep(&ts, NULL);
                }
                continue;
            }
            backoff_ms = 0;
            stratum_log("connected to %s:%s as '%s'", ctx->host, ctx->port,
                        ctx->user);
        }

        rc = stratum_recv_line(ctx, &line);
        if (rc == -2) {
            /* Idle tick: the only place a timer can fire, because the receive
               loop is otherwise blocked in recv(). */
            if (ctx->connected && ctx->refresh_s > 0) {
                time_t now = time(NULL);
                if ((unsigned)(now - ctx->last_request) >= ctx->refresh_s) {
                    pthread_mutex_lock(&ctx->send_lock);
                    if (ctx->connected &&
                        stratum_send_getwork_locked(ctx) == 0) {
                        ctx->last_request = now;
                        ctx->consecutive_rejects = 0;
                    }
                    pthread_mutex_unlock(&ctx->send_lock);
                }
            }
            continue;
        }
        if (rc < 0) {
            stratum_log("disconnected from %s:%s", ctx->host, ctx->port);
            pthread_mutex_lock(&ctx->send_lock);
            sock_close_safe(&ctx->sock);
            ctx->connected = 0;
            pthread_mutex_unlock(&ctx->send_lock);
            continue;
        }
        handle_line(ctx, line);
        free(line);
    }

    pthread_mutex_lock(&ctx->send_lock);
    sock_close_safe(&ctx->sock);
    ctx->connected = 0;
    pthread_mutex_unlock(&ctx->send_lock);
    return NULL;
}

/* ───────────────────────────── public API ───────────────────────────── */

stratum_ctx *stratum_connect(const char *host, const char *port,
                             const char *user, const char *pass)
{
    stratum_ctx *ctx;

    if (!host || !*host || !port || !*port || !user) {
        fprintf(stderr, "[stratum] connect: host/port/user are required\n");
        return NULL;
    }

    ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;

    ctx->sock = -1;
    ctx->running = 1;
    ctx->msg_id = 1;
    ctx->refresh_s = STRATUM_REFRESH_DEFAULT_S;
    {
        const char *env = getenv("STRATUM_REFRESH_S");
        if (env && *env)
            ctx->refresh_s = (unsigned)strtoul(env, NULL, 10);
    }
    snprintf(ctx->host, sizeof(ctx->host), "%s", host);
    snprintf(ctx->port, sizeof(ctx->port), "%s", port);
    snprintf(ctx->user, sizeof(ctx->user), "%s", user);
    snprintf(ctx->pass, sizeof(ctx->pass), "%s", pass ? pass : "x");

    pthread_mutex_init(&ctx->send_lock, NULL);
    pthread_mutex_init(&ctx->work_lock, NULL);
    pthread_cond_init(&ctx->work_cond, NULL);

    if (pthread_create(&ctx->recv_thread, NULL, stratum_recv_thread, ctx) != 0) {
        fprintf(stderr, "[stratum] cannot start receive thread\n");
        pthread_mutex_destroy(&ctx->send_lock);
        pthread_mutex_destroy(&ctx->work_lock);
        pthread_cond_destroy(&ctx->work_cond);
        free(ctx);
        return NULL;
    }
    ctx->recv_thread_started = 1;

    /* Do not block here: the caller keeps mining and picks work up as soon as
       the receive thread publishes it (the miner has nothing to do before). */
    return ctx;
}

void stratum_disconnect(stratum_ctx *ctx)
{
    if (!ctx)
        return;

    ctx->running = 0;
    pthread_mutex_lock(&ctx->send_lock);
    sock_close_safe(&ctx->sock);
    pthread_mutex_unlock(&ctx->send_lock);

    pthread_mutex_lock(&ctx->work_lock);
    pthread_cond_broadcast(&ctx->work_cond);
    pthread_mutex_unlock(&ctx->work_lock);

    if (ctx->recv_thread_started)
        pthread_join(ctx->recv_thread, NULL);

    pthread_mutex_destroy(&ctx->send_lock);
    pthread_mutex_destroy(&ctx->work_lock);
    pthread_cond_destroy(&ctx->work_cond);
    free(ctx->recv_buf);
    free(ctx);
}

int stratum_wait_work(stratum_ctx *ctx, char data_hex[STRATUM_DATA_HEX_SIZE],
                      uint64_t *share_ndiff, uint64_t *net_ndiff,
                      unsigned timeout_ms)
{
    struct timespec deadline;
    int rc = 0;

    if (!ctx || !data_hex)
        return 0;

    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += (time_t)(timeout_ms / 1000U);
    deadline.tv_nsec += (long)(timeout_ms % 1000U) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_nsec -= 1000000000L;
        deadline.tv_sec += 1;
    }

    pthread_mutex_lock(&ctx->work_lock);
    while (ctx->running && !ctx->work_ready) {
        rc = pthread_cond_timedwait(&ctx->work_cond, &ctx->work_lock, &deadline);
        if (rc == ETIMEDOUT)
            break;
    }
    if (ctx->work_ready) {
        memcpy(data_hex, ctx->work_data, STRATUM_DATA_HEX_SIZE);
        if (share_ndiff)
            *share_ndiff = ctx->share_ndiff;
        if (net_ndiff)
            *net_ndiff = ctx->net_ndiff;
        ctx->work_new = 0;
        rc = 1;
    } else {
        rc = 0;
    }
    pthread_mutex_unlock(&ctx->work_lock);
    return rc;
}

int stratum_poll_work(stratum_ctx *ctx, char data_hex[STRATUM_DATA_HEX_SIZE],
                      uint64_t *share_ndiff, uint64_t *net_ndiff)
{
    int got = 0;

    if (!ctx || !data_hex)
        return 0;

    pthread_mutex_lock(&ctx->work_lock);
    if (ctx->work_new && ctx->work_ready) {
        memcpy(data_hex, ctx->work_data, STRATUM_DATA_HEX_SIZE);
        if (share_ndiff)
            *share_ndiff = ctx->share_ndiff;
        if (net_ndiff)
            *net_ndiff = ctx->net_ndiff;
        ctx->work_new = 0;
        got = 1;
    }
    pthread_mutex_unlock(&ctx->work_lock);
    return got;
}

double stratum_share_merit(stratum_ctx *ctx)
{
    double m;
    if (!ctx)
        return 0.0;
    pthread_mutex_lock(&ctx->work_lock);
    m = stratum_ndiff_to_merit(ctx->share_ndiff);
    pthread_mutex_unlock(&ctx->work_lock);
    return m;
}

double stratum_network_merit(stratum_ctx *ctx)
{
    double m;
    if (!ctx)
        return 0.0;
    pthread_mutex_lock(&ctx->work_lock);
    m = stratum_ndiff_to_merit(ctx->net_ndiff);
    pthread_mutex_unlock(&ctx->work_lock);
    return m;
}

int stratum_submit_pow(stratum_ctx *ctx, const uint8_t hdr80[STRATUM_HDR80_SIZE],
                       uint32_t nonce, uint16_t shift, const uint8_t *nadd,
                       size_t nadd_len)
{
    return stratum_submit_share(ctx, hdr80, nonce, shift, nadd, nadd_len, NULL);
}

int stratum_submit_share(stratum_ctx *ctx, const uint8_t hdr80[STRATUM_HDR80_SIZE],
                         uint32_t nonce, uint16_t shift, const uint8_t *nadd,
                         size_t nadd_len,
                         const struct stratum_share_meta *meta)
{
    char hex[2 * (STRATUM_HDR80_SIZE + 4 + 2 + STRATUM_NADD_MAX) + 1];
    size_t hexlen;
    char *buf = NULL;
    size_t buf_cap;
    int len;
    int id;
    uint64_t h;
    int rc;

    if (!ctx || !hdr80)
        return 0;
    if (nadd_len > STRATUM_NADD_MAX) {
        stratum_log("submit: nAdd of %zu bytes exceeds the pool payload "
                    "metadata buffer (%u)", nadd_len,
                    (unsigned)STRATUM_NADD_MAX);
        return 0;
    }

    hexlen = stratum_pow_hex(hdr80, nonce, shift, nadd, nadd_len, hex, sizeof(hex));
    if (hexlen == 0) {
        stratum_log("submit: cannot serialize the PoW solution");
        return 0;
    }

    h = fnv1a64(hex);

    pthread_mutex_lock(&ctx->send_lock);
    if (!ctx->connected) {
        pthread_mutex_unlock(&ctx->send_lock);
        stratum_log("submit skipped: not connected (%zu-byte solution)",
                    hexlen / 2U);
        return 0;
    }
    if (dedup_seen_locked(ctx, h)) {
        ctx->duplicates++;
        pthread_mutex_unlock(&ctx->send_lock);
        return 0;
    }
    id = ctx->msg_id++;
    if (ctx->pending_count < STRATUM_SUBMIT_ID_RING) {
        struct stratum_share_meta m;
        memset(&m, 0, sizeof(m));
        if (meta)
            m = *meta;
        m.header_nonce = nonce;
        m.shift = shift;
        m.nadd_len = (nadd_len == 0) ? 1U : nadd_len;
        if (nadd && nadd_len > 0)
            memcpy(m.nadd, nadd, nadd_len);
        ctx->pending[ctx->pending_count].id = id;
        ctx->pending[ctx->pending_count].meta = m;
        ctx->pending_count++;
    }
    pthread_mutex_unlock(&ctx->send_lock);

    buf_cap = strlen(ctx->user) + strlen(ctx->pass) + hexlen + 128U;
    buf = malloc(buf_cap);
    if (!buf) {
        pthread_mutex_lock(&ctx->send_lock);
        ctx->send_failures++;
        pthread_mutex_unlock(&ctx->send_lock);
        return 0;
    }
    len = snprintf(buf, buf_cap,
                   "{\"id\":%d,\"method\":\"mining.submit\","
                   "\"params\":[\"%s\",\"%s\",\"%s\"]}\n",
                   id, ctx->user, ctx->pass, hex);

    if (len < 0 || (size_t)len >= buf_cap) {
        free(buf);
        pthread_mutex_lock(&ctx->send_lock);
        ctx->send_failures++;
        pthread_mutex_unlock(&ctx->send_lock);
        return 0;
    }

    pthread_mutex_lock(&ctx->send_lock);
    rc = stratum_send_raw(ctx, buf, (size_t)len);
    if (rc != 0) {
        struct stratum_share_meta ignored;
        ctx->send_failures++;
        (void)pending_take_locked(ctx, id, &ignored);
    }
    pthread_mutex_unlock(&ctx->send_lock);

    free(buf);
    if (rc != 0) {
        stratum_log("submit id=%d FAILED (%zu-byte solution, %d hex chars)",
                    id, hexlen / 2U, (int)hexlen);
        return 0;
    }

    if (stratum_debug_enabled())
        stratum_log(">> submit id=%d %zu-byte solution", id, hexlen / 2U);
    return 1;
}

void stratum_set_verdict_callback(stratum_ctx *ctx, stratum_verdict_cb cb,
                                  void *user)
{
    if (!ctx)
        return;
    pthread_mutex_lock(&ctx->send_lock);
    ctx->verdict_cb = cb;
    ctx->verdict_user = user;
    pthread_mutex_unlock(&ctx->send_lock);
}

void stratum_set_refresh_seconds(stratum_ctx *ctx, unsigned seconds)
{
    if (!ctx)
        return;
    pthread_mutex_lock(&ctx->send_lock);
    ctx->refresh_s = seconds;
    ctx->last_request = time(NULL);
    pthread_mutex_unlock(&ctx->send_lock);
}

int stratum_request_work(stratum_ctx *ctx)
{
    int rc;
    if (!ctx)
        return 0;
    pthread_mutex_lock(&ctx->send_lock);
    if (!ctx->connected) {
        pthread_mutex_unlock(&ctx->send_lock);
        return 0;
    }
    rc = stratum_send_getwork_locked(ctx);
    if (rc == 0) {
        ctx->last_request = time(NULL);
        ctx->consecutive_rejects = 0;
    }
    pthread_mutex_unlock(&ctx->send_lock);
    return rc == 0;
}

void stratum_get_stats(stratum_ctx *ctx, uint64_t *accepted,
                       uint64_t *rejected, uint64_t *duplicates,
                       uint64_t *send_failures, uint64_t *reconnects,
                       uint64_t *connect_failures, uint64_t *unresolved)
{
    if (!ctx)
        return;
    pthread_mutex_lock(&ctx->send_lock);
    if (accepted)
        *accepted = ctx->accepted;
    if (rejected)
        *rejected = ctx->rejected;
    if (duplicates)
        *duplicates = ctx->duplicates;
    if (send_failures)
        *send_failures = ctx->send_failures;
    if (reconnects)
        *reconnects = ctx->reconnects;
    if (connect_failures)
        *connect_failures = ctx->connect_failures;
    if (unresolved)
        *unresolved = ctx->unresolved;
    pthread_mutex_unlock(&ctx->send_lock);
}

int stratum_is_connected(stratum_ctx *ctx)
{
    int c;
    if (!ctx)
        return 0;
    pthread_mutex_lock(&ctx->send_lock);
    c = ctx->connected;
    pthread_mutex_unlock(&ctx->send_lock);
    return c;
}
