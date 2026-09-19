/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GapMiner V2 — Main Entry Point (Phase 5 Stub)
 *
 * Command-line interface:
 *   gapminer --pool <url> --worker <name> --gpus <count>
 *
 * Workflow:
 * 1. Parse CLI arguments
 * 2. Initialize CRT engine and GPU workers
 * 3. Connect to mining pool
 * 4. Spawn worker threads (one per GPU)
 * 5. Process work and submit gaps
 * 6. Graceful shutdown on SIGINT
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <sys/stat.h>
#include <gmp.h>
#include "miner_farm.h"
#include "gapcoin_rpc.h"
#include "gapcoin_work.h"
#include "worker_gpu.h"
#include "merit_records.h"
#include "record_log.h"
#include "crt_runtime.h"
#include "halfclass.h"
#include "gap_dist.h"
#include "gap_hunt.h"
#include "stratum.h"
#ifdef WITH_CUDA
#include "gpu/gpu_fermat.h"
#endif

/* Global farm and RPC for signal handling */
static struct miner_farm *g_farm = NULL;
static struct gapcoin_rpc *g_rpc = NULL;

/* Submission counters (only meaningful when --enable-submission is set) */
static uint64_t g_submit_attempts = 0;
static uint64_t g_submit_accepted = 0;
static uint64_t g_submit_rejected = 0;
static uint64_t g_submit_stale = 0;
/* Assembly failures are NOT staleness: the gap was never offered to the node.
   They used to be added to g_submit_stale, which made a local buffer bug look
   like a node rejection (2026-09-18). */
static uint64_t g_submit_assemble_failed = 0;

/* RPC polling thread state */
static volatile int g_rpc_running = 0;
static volatile int g_new_block_ready = 0;
static pthread_t g_rpc_thread;
static struct block_template *g_rpc_template = NULL;
static uint32_t g_rpc_prev_height = 0;  /* Track previous height to avoid duplicate signaling */
static pthread_mutex_t g_rpc_lock = PTHREAD_MUTEX_INITIALIZER;

/* Pool (Gapcoin legacy stratum) mode.  g_stratum is NULL unless --stratum is
   given; in that mode no local node is contacted at all. */
static struct stratum_ctx *g_stratum = NULL;
static uint64_t g_pool_queued = 0;       /* shares handed to the pool client */
static uint64_t g_pool_send_failed = 0;  /* shares the client refused to queue */

static void signal_handler(int sig) {
    printf("\n[Main] Received signal %d, shutting down...\n", sig);
    if (g_farm) {
        miner_farm_stop(g_farm);
    }
}

/* Even-length hex, non-empty, bounded to a sane scriptPubKey size. */
static int is_valid_hex_script(const char *hex) {
    if (!hex) return 0;
    size_t len = strlen(hex);
    if (len == 0 || len % 2 != 0 || len > 1024) return 0;
    for (size_t i = 0; i < len; i++) {
        char c = hex[i];
        int is_hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                     (c >= 'A' && c <= 'F');
        if (!is_hex) return 0;
    }
    return 1;
}

/* Minimal env-flag parser matching worker_env_enabled(): "1"/"yes"/"true"
   (case-insensitive) enable; empty/"0"/"no"/"off"/"false" disable. */
static int main_env_enabled(const char *name) {
    const char *v = getenv(name);
    if (!v) return 0;
    while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') v++;
    if (*v == '\0' || *v == '0' || *v == 'n' || *v == 'N' ||
        *v == 'f' || *v == 'F') return 0;
    return 1;
}

/* Decimal string for a queue entry's nAdd (handles >64-bit CRT offsets). */
static char *gap_entry_nadd_dec(const struct gap_queue_entry *e) {
    if (!e) return NULL;
    if (e->nadd_len > 0) {
        mpz_t z;
        mpz_init(z);
        mpz_import(z, e->nadd_len, -1, 1, 0, 0, e->nadd_bytes);
        char *s = mpz_get_str(NULL, 10, z);
        mpz_clear(z);
        return s;
    }
    char *s = (char *)malloc(24);
    if (s) snprintf(s, 24, "%llu", (unsigned long long)e->nadd);
    return s;
}

/* RPC polling thread - continuously fetches new block template */
static void *rpc_poll_thread_func(void *arg) {
    (void)arg;  /* unused */
    
    printf("[RPC Thread] Started, polling every 500 ms\n");
    
    while (g_rpc_running) {
        struct block_template *new_tmpl = gapcoin_rpc_get_block_template(g_rpc);
        
        if (new_tmpl) {
            pthread_mutex_lock(&g_rpc_lock);
            
            /* Only signal if block height CHANGED (avoids duplicate signaling) */
            if (new_tmpl->height != g_rpc_prev_height) {
                /* Free old template and store new one */
                if (g_rpc_template) {
                    block_template_free(g_rpc_template);
                }
                g_rpc_template = new_tmpl;
                g_rpc_prev_height = new_tmpl->height;
                g_new_block_ready = 1;  /* Signal main loop only on height change */
                
                printf("[RPC Thread] New block detected: height=%u\n", new_tmpl->height);
            } else {
                /* Same height, just discard */
                block_template_free(new_tmpl);
            }
            
            pthread_mutex_unlock(&g_rpc_lock);
        }
        
        /* Poll every 500 ms: cuts average new-block detection latency from
           ~1 s to ~0.25 s (~0.85% -> ~0.2% lost hashing time per 118 s block). */
        {
            struct timespec poll_ts = {0, 500 * 1000 * 1000};
            nanosleep(&poll_ts, NULL);
        }
    }
    
    printf("[RPC Thread] Stopped\n");
    return NULL;
}

/* ─────────────── Pool mode (Gapcoin legacy stratum) ─────────────── */

/* Accepts "host:port", "stratum+tcp://host:port", "stratum://host:port" and
   "tcp://host:port".  Returns 0 on a malformed endpoint. */
static int parse_stratum_endpoint(const char *endpoint, char *host,
                                  size_t host_cap, char *port, size_t port_cap)
{
    const char *p = endpoint;
    const char *colon;
    size_t host_len;

    if (!endpoint || !*endpoint)
        return 0;
    if (strncmp(p, "stratum+tcp://", 14) == 0) {
        p += 14;
    } else if (strncmp(p, "stratum://", 10) == 0) {
        p += 10;
    } else if (strncmp(p, "tcp://", 6) == 0) {
        p += 6;
    }
    colon = strrchr(p, ':');
    if (!colon || colon == p)
        return 0;
    host_len = (size_t)(colon - p);
    if (host_len == 0 || host_len >= host_cap)
        return 0;
    if (colon[1] == '\0' || strlen(colon + 1) >= port_cap)
        return 0;
    memcpy(host, p, host_len);
    host[host_len] = '\0';
    snprintf(port, port_cap, "%s", colon + 1);
    return 1;
}

/*
 * Worker name and password for the pool.
 *
 * The password is NEVER taken from the command line: it would be visible in
 * `ps` to every user on the box.  It comes from --stratum-auth-file (worker on
 * line 1, password on line 2) or from GAPMINER_STRATUM_PASS.  The worker name
 * may come from the file, --stratum-user or GAPMINER_STRATUM_USER.  When no
 * password is available "x" is used, which pools accept for wallet logins
 * (the legacy Gapcoin pool protocol also accepts mining anonymously with the
 * wallet address as the worker name).
 */
static int resolve_stratum_auth(const char *auth_file, const char *cli_user,
                                char *user, size_t user_cap, char *pass,
                                size_t pass_cap)
{
    const char *env_user = getenv("GAPMINER_STRATUM_USER");
    const char *env_pass = getenv("GAPMINER_STRATUM_PASS");

    user[0] = '\0';
    pass[0] = '\0';

    if (auth_file && *auth_file) {
        struct stat st;
        FILE *f = fopen(auth_file, "r");
        char line[256];
        if (!f) {
            fprintf(stderr, "[Main] Cannot open --stratum-auth-file %s\n",
                    auth_file);
            return 0;
        }
        if (stat(auth_file, &st) == 0 && (st.st_mode & 0077))
            fprintf(stderr,
                    "[Main] WARNING: %s is readable by group/others; chmod 600 "
                    "is recommended (it holds the pool password)\n",
                    auth_file);
        if (fgets(line, sizeof(line), f)) {
            line[strcspn(line, "\r\n")] = '\0';
            snprintf(user, user_cap, "%.*s", (int)user_cap - 1, line);
        }
        if (fgets(line, sizeof(line), f)) {
            line[strcspn(line, "\r\n")] = '\0';
            snprintf(pass, pass_cap, "%.*s", (int)pass_cap - 1, line);
        }
        fclose(f);
    }
    if (user[0] == '\0' && cli_user && *cli_user)
        snprintf(user, user_cap, "%s", cli_user);
    if (user[0] == '\0' && env_user && *env_user)
        snprintf(user, user_cap, "%s", env_user);
    if (pass[0] == '\0' && env_pass && *env_pass)
        snprintf(pass, pass_cap, "%s", env_pass);
    if (user[0] == '\0') {
        fprintf(stderr, "[Main] Pool mode needs a worker name: use "
                        "--stratum-user, --stratum-auth-file or "
                        "GAPMINER_STRATUM_USER\n");
        return 0;
    }
    if (pass[0] == '\0')
        snprintf(pass, pass_cap, "x");
    return 1;
}

/* Hex (>=160 chars) -> 80 header bytes.  Returns 0 on malformed input. */
static int decode_hex80(const char *hex, uint8_t out[80])
{
    if (!hex || strlen(hex) < 160)
        return 0;
    for (size_t i = 0; i < 80; i++) {
        unsigned int bv = 0;
        if (sscanf(hex + i * 2, "%2x", &bv) != 1)
            return 0;
        out[i] = (uint8_t)bv;
    }
    return 1;
}

/* Little-endian nAdd bytes -> decimal string (record log field). */
static void nadd_bytes_to_dec(const uint8_t *bytes, size_t len, char *out,
                              size_t out_cap)
{
    mpz_t z;
    char *s;

    if (len == 0) {
        snprintf(out, out_cap, "0");
        return;
    }
    mpz_init(z);
    mpz_import(z, len, -1, 1, 0, 0, bytes);
    s = mpz_get_str(NULL, 10, z);
    snprintf(out, out_cap, "%s", s ? s : "?");
    free(s);
    mpz_clear(z);
}

/*
 * Pool verdict -> record log.  Runs on the stratum receive thread, so it never
 * blocks a worker; record_log_write_outcome_big() is mutex-protected.
 *
 * The worker that found the gap already logged status=queued, so this adds the
 * terminal state: accepted, rejected, or unresolved (the connection dropped
 * with the share in flight, which is NOT a rejection and must not be counted
 * as one).  The line is keyed by pool share metadata, because the verdict
 * arrives asynchronously and carries only a JSON message id.
 */
static void pool_verdict_cb(void *user, int accepted,
                            const struct stratum_share_meta *meta,
                            const char *message)
{
    char nadd_dec[192];
    (void)user;

    if (!meta)
        return;
    nadd_bytes_to_dec(meta->nadd, meta->nadd_len, nadd_dec, sizeof(nadd_dec));

    record_log_write_outcome_big(meta->height, meta->shift, meta->header_nonce,
                                 nadd_dec, meta->gap_length, meta->merit,
                                 (accepted == 1) ? "accepted" :
                                 (accepted == 0) ? "rejected" : "unresolved");

    if (accepted == 1) {
        printf("[Main] Pool ACCEPTED share: gap=%u merit=%.2f nAdd=%s\n",
               meta->gap_length, meta->merit, nadd_dec);
        fflush(stdout);
    } else {
        fprintf(stderr, "[Main] Pool %s share: gap=%u merit=%.2f nAdd=%s%s%s\n",
                (accepted == 0) ? "REJECTED" : "UNRESOLVED",
                meta->gap_length, meta->merit, nadd_dec,
                message ? " - " : "", message ? message : "");
    }
}

void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("  --host <addr>         Gapcoin RPC host (default: 127.0.0.1)\n");
    printf("  --port <port>         Gapcoin RPC port (default: 31397)\n");
    printf("  --user <name>         RPC username (default: benxy031)\n");
    printf("  --pass <pass>         RPC password (default: xx)\n");
    printf("  --threads <count>     Number of worker threads (default: 1)\n");
    printf("  --shift <value>       Non-CRT shift (20-1024; default: 26); window scales through shift 26\n");
    printf("  --crt-file <path>     CRT covering file (text format from gen_crt); enables CRT\n");
    printf("                        mode: workers hash strided header nonces in parallel at the\n");
    printf("                        file's shift (--shift is ignored; pair with --threads N,\n");
    printf("                        --enable-gpu-fermat and --enable-submission)\n");
    printf("  --sieve-primes <n>    Sieve prime limit (default: 50000; in non-CRT\n");
    printf("                        mode with --enable-gpu-fermat an adaptive\n");
    printf("                        bit-scaled default (cover..20M); 100k in\n");
    printf("                        CRT+GPU mode; 2M in CRT+GPU+FUSED_GPU mode;\n");
    printf("                        10M in CRT CPU mode)\n");
    printf("  --merit <value>       Merit threshold override (default: node difficulty)\n");
    printf("                         Validated candidates are logged; submission is disabled\n");
    printf("                         unless --enable-submission is given.\n");
    printf("  --enable-submission   Submit BPSW-verified gaps via RPC submitblock (default: off)\n");
    printf("  --coinbase-script-hex <hex>\n");
    printf("                        Coinbase payout scriptPubKey, hex-encoded (default: none;\n");
    printf("                        falls back to an anyone-can-spend OP_TRUE output with a warning)\n");
    printf("  --stratum <host:port> Mine on a Gapcoin pool instead of a local node.\n");
    printf("                        Accepts stratum+tcp://host:port.\n");
    printf("                        Uses the LEGACY Gapcoin pool protocol (suprnova port\n");
    printf("                        2434; port 2433 is a private dialect only suprnova's\n");
    printf("                        own miners speak). The pool supplies the header and\n");
    printf("                        the share target, which becomes the merit threshold,\n");
    printf("                        and solutions go back as mining.submit PoW payloads\n");
    printf("                        (80-byte header + nNonce + nShift + nAdd) - no local\n");
    printf("                        node, no coinbase, no block assembly.\n");
    printf("                        The share target is used instead of the network\n");
    printf("                        difficulty so the pool sees share traffic; override\n");
    printf("                        with --merit.\n");
    printf("  --stratum-user <name> Pool worker, or the wallet address for anonymous mining\n");
    printf("                        (fallback: GAPMINER_STRATUM_USER)\n");
    printf("  --stratum-auth-file <path>\n");
    printf("                        File with the worker name on line 1 and the password on\n");
    printf("                        line 2 (chmod 600). The password is never read from the\n");
    printf("                        command line; GAPMINER_STRATUM_PASS is the fallback.\n");
    printf("  --enable-gpu-fermat   Batch-test sieve candidates on the CUDA GPU (base-2\n");
    printf("                        Miller-Rabin) as the primality filter, skipping the CPU\n");
    printf("                        Euler test (default: off; requires a WITH_CUDA=1 build;\n");
    printf("                        falls back to CPU on GPU failure).\n");
    printf("                        Optional GPU bitmap sieve is off by default; enable with\n");
    printf("                        GPU_SIEVE=1 (multi-GPU only).\n");
    printf("                        GPU_SIEVE batch size is configurable via\n");
    printf("                        GPU_SIEVE_BATCH=<windows> (default: 1024,\n");
    printf("                        max 4096).\n");
    printf("                        Dense GPU marking (p <= window) uses chunked\n");
    printf("                        work items; GPU_MARK_SPLIT=0 restores the\n");
    printf("                        per-prime walk and GPU_MARK_SPLIT_SLOTS=<n>\n");
    printf("                        sets the chunk size in odd slots\n");
    printf("                        (default: 160, clamped 32..4096).\n");
    printf("                        On the fused path the no-test chain is ON by\n");
    printf("                        default (MINING_JUMP2=0 restores the full scan;\n");
    printf("                        inert without FUSED_GPU=1).\n");
    printf("  --record-log <path>   Log every BPSW-verified candidate with full parameters\n");
    printf("                        (default: gapminer_records.log, or\n");
    printf("                        gapminer_pool_records.log in pool/--stratum mode)\n");
    printf("  --merit-records <path>\n");
    printf("                        Reference gap-length -> best-known-merit table used to flag\n");
    printf("                        new records in the record log (default: data/prime_gap_merits.txt,\n");
    printf("                        from https://primegaps.cloudygo.com/merits.txt)\n");
    printf("  --gap-hunt            Run standalone record-hunting walk (requires\n");
    printf("                        --crt-file; exits instead of starting the miner)\n");
    printf("  --gap-hunt-start <hex>  Base anchor hex (default: 2^762)\n");
    printf("  --gap-hunt-min-merit <m>  Report gaps with merit >= m (default 15)\n");
    printf("  --gap-hunt-state <path>  Resume state file\n");
    printf("  --gap-hunt-out <path>    Results file (default: stdout only)\n");
    printf("  --gap-hunt-device <n>    CUDA device id (default 0; multi-GPU fleets)\n");
    printf("  --help                Show this help message\n");
}

int main(int argc, char *argv[]) {
    printf("================================================\n");
    printf("GapMiner V2 — Prime Gap Mining Engine (Phase 6)\n");
    printf("================================================\n\n");
    
    /* Parse command-line arguments */
    const char *rpc_host = "127.0.0.1";
    uint16_t rpc_port = 31397;
    const char *rpc_user = "benxy031";
    const char *rpc_pass = "xx";
    uint32_t num_threads = 1;        /* Number of worker threads */
    uint32_t user_shift = 26;        /* Default supported non-CRT shift value */
    const char *crt_file = NULL;     /* CRT covering file (enables CRT mode) */
    uint32_t sieve_primes = 50000;   /* Sieve prime limit */
    int sieve_primes_overridden = 0; /* Set when --sieve-primes is given explicitly */
    double merit_threshold = 0.0;    /* Resolved from node difficulty unless overridden */
    int merit_threshold_overridden = 0;
    int enable_submission = 0;       /* Real submitblock RPC calls; off by default */
    const char *coinbase_script_hex = NULL; /* Real payout scriptPubKey; NULL = OP_TRUE */
    int enable_gpu_fermat = 0;       /* GPU Fermat pre-filter; off by default */
    int half_class_active = 0;       /* HALF_CLASS two-pass scan (non-CRT) */
    const char *record_log_path = "gapminer_records.log";
    int record_log_overridden = 0;   /* Set when --record-log is given explicitly */
    const char *merit_records_path = "data/prime_gap_merits.txt";
    /* GAP_HUNT standalone mode */
    int gap_hunt_enabled = 0;
    const char *gap_hunt_start = NULL;
    double gap_hunt_min_merit = 15.0;
    const char *gap_hunt_state = NULL;
    const char *gap_hunt_out = NULL;
    int gap_hunt_device = 0;
    /* Pool (legacy stratum) mode: --stratum enables it and no node is used */
    const char *stratum_endpoint = NULL;
    const char *stratum_user = NULL;
    const char *stratum_auth_file = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            rpc_host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            rpc_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--user") == 0 && i + 1 < argc) {
            rpc_user = argv[++i];
        } else if (strcmp(argv[i], "--pass") == 0 && i + 1 < argc) {
            rpc_pass = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--shift") == 0 && i + 1 < argc) {
            user_shift = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--crt-file") == 0 && i + 1 < argc) {
            crt_file = argv[++i];
        } else if (strcmp(argv[i], "--sieve-primes") == 0 && i + 1 < argc) {
            sieve_primes = atoi(argv[++i]);
            sieve_primes_overridden = 1;
        } else if (strcmp(argv[i], "--merit") == 0 && i + 1 < argc) {
            merit_threshold = atof(argv[++i]);
            if (merit_threshold <= 0.0) {
                fprintf(stderr, "[Main] --merit must be greater than zero\n");
                return 1;
            }
            merit_threshold_overridden = 1;
        } else if (strcmp(argv[i], "--enable-submission") == 0) {
            enable_submission = 1;
        } else if (strcmp(argv[i], "--coinbase-script-hex") == 0 && i + 1 < argc) {
            coinbase_script_hex = argv[++i];
            if (!is_valid_hex_script(coinbase_script_hex)) {
                fprintf(stderr,
                        "[Main] --coinbase-script-hex must be non-empty, even-length hex "
                        "(max 1024 characters)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--enable-gpu-fermat") == 0) {
            enable_gpu_fermat = 1;
        } else if (strcmp(argv[i], "--record-log") == 0) {
            if (i + 1 >= argc || argv[i + 1][0] == '-') {
                fprintf(stderr, "[Main] --record-log requires a file path\n");
                return 1;
            }
            record_log_path = argv[++i];
            record_log_overridden = 1;
        } else if (strcmp(argv[i], "--merit-records") == 0) {
            if (i + 1 >= argc || argv[i + 1][0] == '-') {
                fprintf(stderr, "[Main] --merit-records requires a file path\n");
                return 1;
            }
            merit_records_path = argv[++i];
        } else if (strcmp(argv[i], "--stratum") == 0 && i + 1 < argc) {
            stratum_endpoint = argv[++i];
        } else if (strcmp(argv[i], "--stratum-user") == 0 && i + 1 < argc) {
            stratum_user = argv[++i];
        } else if (strcmp(argv[i], "--stratum-auth-file") == 0 && i + 1 < argc) {
            stratum_auth_file = argv[++i];
        } else if (strcmp(argv[i], "--gap-hunt") == 0) {
            gap_hunt_enabled = 1;
        } else if (strcmp(argv[i], "--gap-hunt-start") == 0 && i + 1 < argc) {
            gap_hunt_start = argv[++i];
        } else if (strcmp(argv[i], "--gap-hunt-min-merit") == 0 && i + 1 < argc) {
            gap_hunt_min_merit = atof(argv[++i]);
        } else if (strcmp(argv[i], "--gap-hunt-state") == 0 && i + 1 < argc) {
            gap_hunt_state = argv[++i];
        } else if (strcmp(argv[i], "--gap-hunt-out") == 0 && i + 1 < argc) {
            gap_hunt_out = argv[++i];
        } else if (strcmp(argv[i], "--gap-hunt-device") == 0 && i + 1 < argc) {
            gap_hunt_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (gap_hunt_enabled) {
        if (!crt_file) {
            fprintf(stderr, "[Main] --gap-hunt requires --crt-file\n");
            return 1;
        }
        struct gap_hunt_config gh;
        memset(&gh, 0, sizeof(gh));
        gh.crt_file = crt_file;
        gh.start_hex = gap_hunt_start;
        gh.min_merit = gap_hunt_min_merit;
        gh.state_path = gap_hunt_state;
        gh.out_path = gap_hunt_out;
        /* GAP_HUNT sieve depth: 2M (was 10M).  Re-measured 2026-09-15 AFTER the
           chunked mark split made marking ~9x cheaper and with the chain
           (GAP_HUNT_JUMP2) on: the curve is now FLAT then falling -- shift507:
           0.5M=1893, 1M=1908, 2M=1897, 5M=1819, 10M=1736, 20M=1558 win/s;
           shift1017: 2M=599, 5M=596, 10M=593, 20M=538 win/s.  The old "deeper
           is better" advice held when the mark kernel dominated; now the
           per-prime per-window residue/mark cost outweighs the shrinking
           survivor set.  2M is equal-best at both shifts with 5x less prime
           table (16 MB vs 80 MB) and faster startup.  Record parity verified
           at 2M vs 10M (identical gap/merit/start over the same k range). */
        gh.sieve_primes = sieve_primes_overridden ? sieve_primes : 2000000U;
        gh.device = gap_hunt_device;
        return gap_hunt_run(&gh);
    }

    struct crt_runtime crt_rt;
    int crt_mode = 0;
    if (crt_file) {
        if (!crt_runtime_load(&crt_rt, crt_file)) {
            fprintf(stderr, "[Main] Failed to load CRT covering file: %s\n", crt_file);
            return 1;
        }
        crt_mode = 1;
        user_shift = crt_rt.shift;
        printf("[Main] CRT mode: %s (shift=%u, %u primes, gap_target=%llu, "
               "design merit %.2f)\n",
               crt_file, crt_rt.shift, crt_rt.n_primes,
               (unsigned long long)crt_rt.gap_target, crt_rt.merit);
    }

    if (!crt_mode && (user_shift < 20 || user_shift > 1024)) {
        fprintf(stderr,
                "[Main] Non-CRT scanning supports --shift values from 20 through 1024\n");
        return 1;
    }

    /* Gap-distribution health check: magnitude for the Hardy-Littlewood
       density correction.  Set once before any worker starts. */
    gap_dist_reset();
    gap_dist_set_logbase((256.0 + (double)user_shift) * log(2.0));
    if (crt_mode) {
        /* In CRT mode the covering conditions the gap distribution inside
           [1, gap_target) relative to the aligned base.  Restrict the health
           check to honest (unconditioned) gaps: both endpoints before the
           aligned base or beyond gap_target.  back_limit mirrors the worker.
           GAP_DIST_EXCL_HI overrides the upper boundary for diagnostics. */
        uint64_t back_limit_hl =
            (uint64_t)ceil(2.0 * (256.0 + (double)user_shift) * log(2.0));
        if (back_limit_hl < 4096) back_limit_hl = 4096;
        uint64_t excl_hi = back_limit_hl + crt_rt.gap_target;
        const char *excl_env = getenv("GAP_DIST_EXCL_HI");
        if (excl_env) {
            excl_hi = back_limit_hl + strtoull(excl_env, NULL, 10);
        }
        gap_dist_set_excluded(back_limit_hl, excl_hi);
    } else {
        gap_dist_set_excluded(0, 0);   /* no exclusion in non-CRT mode */
    }

    /* HALF_CLASS two-pass scan: the visible classes {1,7,11,13,17,19,23,29}
       mod 60 are sieved and primality-tested, the hidden classes are
       verified on demand.  Non-CRT verifies them with a mini-sieve; CRT
       additionally pre-filters them with the covering template.  Visible
       gaps are not consecutive-prime gaps, so the Hardy-Littlewood health
       histogram is disabled in this mode. */
    if (main_env_enabled("HALF_CLASS")) {
        half_class_active = 1;
        gap_dist_set_enabled(0);
        printf("[Main] HALF_CLASS scan active: visible classes "
               "{1,7,11,13,17,19,23,29} mod 60 only; hidden classes verified "
               "on demand (%s); gap-dist health disabled\n",
               crt_mode ? "via CRT covering template" : "mini-sieve");
    }

    /* QUARTER_CLASS: hide 12 of the 16 coprime classes (visible = 4).
       Generalizes HALF_CLASS; implies it.  The containment lemma guarantees
       no true qualifying gap is lost, so the GPU MR load halves while the
       blocks-per-window yield stays identical. */
    if (main_env_enabled("QUARTER_CLASS")) {
        halfclass_set_quarter(1);
        half_class_active = 1;
        gap_dist_set_enabled(0);
        printf("[Main] QUARTER_CLASS scan active: visible classes "
               "{1,7,11,13} mod 60 only (12 hidden verified on demand); "
               "gap-dist health disabled\n");
    }

    uint32_t owned_window_size = non_crt_owned_window_size(user_shift);

    /* Non-CRT GPU: the CGBN MR kernel is the bottleneck and sieve survivors
       oversubscribe it (acc/wall ~10.9 at shift 55 with the window+halo
       default).  The optimal sieve depth grows with the MR bit length:
       measured at 282-bit (shift 26): 1.06M = 292 win/s, 2M = 286, 5M = 285
       (optimum at the old default); at 311-bit (shift 55): 1.06M = 291,
       5M = 307, 10M = 321, 20M = 332 (+14%), 30M = 285, 50M = 209 (the CPU
       sieve becomes the bottleneck past 20M).  Smart default interpolates
       log2(depth) between those two anchors, clamped to [cover, 20M]. */
    if (!sieve_primes_overridden && enable_gpu_fermat) {
        uint32_t cover = owned_window_size + NON_CRT_LOOKAHEAD_SIZE;
        double bits = 256.0 + (double)user_shift;
        /* Full mode anchors: 282-bit -> cover (1.06M), 311-bit -> 20M.
           HALF_CLASS halves the MR load, which shifts the optimum down
           (measured at shift 55: 5M = 609 win/s vs 20M = 391; 4M/6M both
           lower): 311-bit anchor -> 5M.  Clamped to [cover, 20M]. */
        double slope = 4.247927513443585 / 29.0;   /* log2(20M)-20 per 29 bits */
        if (half_class_active) {
            slope = (22.2534966642115 - 20.0) / 29.0;   /* log2(5M)-20 */
        }
        double l2 = 20.0 + (bits - 282.0) * slope;
        uint64_t deep = (uint64_t)pow(2.0, l2);
        if (deep < cover) deep = cover;
        if (deep > 20000000U) deep = 20000000U;
        sieve_primes = (uint32_t)deep;
    }

    /* CRT+GPU: a deep CPU sieve saturates the weak 4C/8T host (~66ns/prime
       incl. marking => 9.9ms/window at 2M primes, dropping windows/s ~30%).
       Keep the shallow 100K default for the hybrid H2D GPU path: the GPU MR
       test is the bottleneck and smart-scan already trims candidates.  The
       fused pipeline (FUSED_GPU=1) computes residues + marking + extraction
       entirely on-device; the measured optimum is 2M primes on the
       production host (shift475 live merit: 500K=3183, 1M=3197, 2M=3324,
       5M=3205 win/s; older dev-host runs: 1M=3106 vs 5M=2814 at shift258,
       1M=1971 vs 5M=1905 at shift509).  Deeper than 2M costs more GPU
       marking time than the MR savings of the shrinking survivor set). */
    if (!sieve_primes_overridden && crt_mode) {
        if (enable_gpu_fermat) {
            sieve_primes = main_env_enabled("FUSED_GPU") ? 2000000 : 100000;
        } else {
            sieve_primes = 10000000;
        }
    }

    uint64_t windows_per_header = non_crt_windows_per_header(user_shift);
    if (!crt_mode && windows_per_header == 0) {
        /* shift >= 64: the nAdd search space is far larger than the 32-bit
           window counter can enumerate in a single block.  The work range is
           treated as effectively unlimited and is refreshed by new blocks. */
        printf("[Main] --shift %u: search space exceeds the 32-bit window "
               "counter; work refreshes on new blocks only\n", user_shift);
    } else if (!crt_mode && windows_per_header >= UINT32_MAX) {
        /* shift in [50, 63]: the window counter enumerates only the first
           2^50 adders per block, far more than any block time can consume. */
        printf("[Main] --shift %u: window counter covers only the first "
               "2^50 adders per block (effectively unlimited)\n", user_shift);
    }
    
    printf("[Main] Configuration:\n");
    if (stratum_endpoint) {
        /* Pool mode contacts no node, and the pool's share target (read after
           connecting) is the threshold - saying "RPC" or "live node
           difficulty" here would describe a source that is not in use. */
        printf("  Work source: pool %s\n", stratum_endpoint);
        printf("  Payout: the pool account (the pool owns the coinbase at payout)\n");
    } else {
        printf("  RPC: %s:%u\n", rpc_host, rpc_port);
        printf("  User: %s\n", rpc_user);
    }
    printf("  Threads: %u\n", num_threads);
    printf("  User shift: %u\n", user_shift);
    if (!crt_mode) {
        printf("  Owned window: %u nAdd | Windows/header: %llu\n",
            owned_window_size, (unsigned long long)windows_per_header);
    }
    if (half_class_active) {
        printf("  HALF_CLASS scan: ON (hidden classes verified on demand; gap-dist health disabled)\n");
    }
    printf("  Sieve primes: %u\n", sieve_primes);
    if (merit_threshold_overridden) {
        printf("  Merit threshold: %.2f (CLI override)\n", merit_threshold);
    } else if (stratum_endpoint) {
        printf("  Merit threshold: pool share target (read right after connecting)\n");
    } else {
        printf("  Merit threshold: live node difficulty\n");
    }
    printf("  Mode: %s %s scan (%s)\n",
           stratum_endpoint ? "POOL" : "live",
           crt_mode ? "CRT" : "non-CRT",
           enable_submission ? "submission enabled" : "dry-run");
    printf("\n");

    if (coinbase_script_hex) {
        gapcoin_work_set_payout_script_hex(coinbase_script_hex);
    }
    if (enable_submission) {
        worker_set_submission_enabled(1);
        /* In pool mode the coinbase belongs to the pool, so the OP_TRUE
           fallback warning would describe a payout path that does not exist
           here (and --coinbase-script-hex is ignored). */
        if (!coinbase_script_hex && !stratum_endpoint) {
            fprintf(stderr,
                    "[Main] WARN: --enable-submission is set without --coinbase-script-hex; "
                    "the coinbase payout will use an anyone-can-spend OP_TRUE output, so any "
                    "reward from an accepted block can be spent by anyone\n");
        }
    }

    if (enable_gpu_fermat) {
#ifdef WITH_CUDA
        /* The GPU Fermat kernel is compiled for GPU_NLIMBS × 64 bits; beyond
           that the candidate limbs no longer fit and every batch would fall
           back to the CPU, so disable it up front with a clear warning. */
        if ((256ULL + user_shift) > (uint64_t)GPU_NLIMBS * 64ULL) {
            fprintf(stderr,
                    "[Main] WARN: --shift %u produces %llu-bit candidates but "
                    "this build's GPU Fermat is limited to %u bits; "
                    "disabling --enable-gpu-fermat (CPU Euler will be used)\n",
                    user_shift, 256ULL + (uint64_t)user_shift, GPU_NLIMBS * 64U);
        } else {
            worker_set_gpu_fermat_enabled(1);
            int limbs = gpu_fermat_limbs_for_bits(256U + user_shift);
            printf("[Main] GPU Fermat pre-filter enabled (CUDA, compiled for "
                   "%u-bit candidates / %u limbs)\n",
                   GPU_NLIMBS * 64U, (unsigned)GPU_NLIMBS);
            printf("[Main]   shift %u -> %llu-bit candidates -> %d active limb%s "
                   "-> kernel: %s\n",
                   user_shift, 256ULL + (uint64_t)user_shift, limbs,
                   limbs == 1 ? "" : "s", gpu_fermat_kernel_label(limbs));
        }
        printf("[Main] GPU bitmap sieve: %s (GPU_SIEVE=1 to enable, multi-GPU only)\n",
               getenv("GPU_SIEVE") ? getenv("GPU_SIEVE") : "off");
        printf("[Main] GPU bitmap sieve batch target: %s windows (env GPU_SIEVE_BATCH, default 1024)\n",
               getenv("GPU_SIEVE_BATCH") ? getenv("GPU_SIEVE_BATCH") : "1024");
#else
        fprintf(stderr,
                "[Main] WARN: --enable-gpu-fermat has no effect; this binary was built "
                "without WITH_CUDA=1\n");
#endif
    }

    merit_records_load(merit_records_path);

    /*
     * Mode switch: a local gapcoind (default) or a Gapcoin POOL over the legacy
     * stratum protocol.  Pool mode contacts no node at all: the pool hands out
     * the 80-byte header prefix, its share target becomes the merit threshold
     * (so the pool sees share traffic, not only the rare network-difficulty
     * gaps), and solutions go back as mining.submit PoW payloads instead of
     * assembled blocks.  The pool owns the template -- its merkle root is inside
     * the header it gave us -- so no coinbase and no block assembly are needed.
     */
    int stratum_mode = (stratum_endpoint != NULL);

    /*
     * Pool runs get their own record log.  A pool share is NOT a node-accepted
     * gap: the verdict comes from the pool (accepted/rejected/unresolved), the
     * merit threshold is the pool's share target, and height is always 0 -- so
     * appending both into gapminer_records.log would silently change what that
     * file means (its lines carry no field saying which work source they came
     * from).  --record-log overrides this in both modes.
     */
    if (stratum_mode && !record_log_overridden)
        record_log_path = "gapminer_pool_records.log";
    record_log_init(record_log_path);

    struct block_template *tmpl = NULL;
    struct gapcoin_gbt_work active_work;
    uint8_t initial_h256[32];
    uint32_t work_difficulty = 0;   /* integer merit handed to the workers */
    uint32_t work_height = 0;       /* the pool protocol carries no height (0) */
    char pool_hdr80_hex[STRATUM_DATA_HEX_SIZE];
    uint64_t pool_share_ndiff = 0;
    uint64_t pool_net_ndiff = 0;

    memset(pool_hdr80_hex, 0, sizeof(pool_hdr80_hex));

    if (stratum_mode) {
        char s_host[256];
        char s_port[16];
        char s_user[128];
        char s_pass[128];

        if (!parse_stratum_endpoint(stratum_endpoint, s_host, sizeof(s_host),
                                    s_port, sizeof(s_port))) {
            fprintf(stderr, "[Main] --stratum expects host:port (optionally "
                            "stratum+tcp://host:port), got '%s'\n",
                    stratum_endpoint);
            return 1;
        }
        if (!resolve_stratum_auth(stratum_auth_file, stratum_user, s_user,
                                  sizeof(s_user), s_pass, sizeof(s_pass)))
            return 1;

        printf("[Main] POOL MODE: %s:%s as '%s' (legacy Gapcoin stratum)\n",
               s_host, s_port, s_user);
        printf("  No local node is used: the pool supplies work and the share target.\n");

        g_stratum = stratum_connect(s_host, s_port, s_user, s_pass);
        if (!g_stratum) {
            fprintf(stderr, "[Main] Failed to start the stratum client\n");
            return 1;
        }
        stratum_set_verdict_callback(g_stratum, pool_verdict_cb, NULL);

        if (!stratum_wait_work(g_stratum, pool_hdr80_hex, &pool_share_ndiff,
                               &pool_net_ndiff, 20000)) {
            fprintf(stderr, "[Main] No work from the pool within 20 s.  Check the "
                            "worker name, the password and the port: 2434 is the "
                            "legacy protocol the official miners use, while 2433 "
                            "is a private dialect only suprnova's miners speak.\n");
            stratum_disconnect(g_stratum);
            g_stratum = NULL;
            return 1;
        }
        printf("  Pool share target: %.4f merit | network difficulty: %.4f merit\n",
               stratum_ndiff_to_merit(pool_share_ndiff),
               stratum_ndiff_to_merit(pool_net_ndiff));
        if (!merit_threshold_overridden) {
            merit_threshold = stratum_ndiff_to_merit(pool_share_ndiff);
            if (merit_threshold <= 0.0) {
                fprintf(stderr, "[Main] Pool returned an unusable share target\n");
                stratum_disconnect(g_stratum);
                g_stratum = NULL;
                return 1;
            }
        }
        printf("  Active merit threshold: %.4f (%s)\n", merit_threshold,
               merit_threshold_overridden ? "CLI override" : "pool share target");
        printf("\n");
    } else {
        /* Phase 2: Connect to real Gapcoin node */
        g_rpc = gapcoin_rpc_connect(rpc_host, rpc_port, rpc_user, rpc_pass);
        if (!g_rpc) {
            fprintf(stderr, "[Main] Failed to connect to Gapcoin RPC\n");
            return 1;
        }

        /* Get initial mining info */
        struct mining_info *info = gapcoin_rpc_get_mining_info(g_rpc);
        if (!info) {
            fprintf(stderr, "[Main] Failed to get mining info\n");
            gapcoin_rpc_free(g_rpc);
            return 1;
        }

        printf("[Main] Gapcoin info:\n");
        printf("  Height: %u\n", info->blocks);
        printf("  Difficulty: %.2f\n", info->difficulty);
        printf("  Network power: %.2fM mH/s\n",
               info->networkminingpower / 1000000.0);
        if (!merit_threshold_overridden) {
            if (info->difficulty <= 0.0) {
                fprintf(stderr, "[Main] Node returned an invalid difficulty\n");
                mining_info_free(info);
                gapcoin_rpc_free(g_rpc);
                return 1;
            }
            merit_threshold = info->difficulty;
        }
        printf("  Active merit threshold: %.2f (%s)\n", merit_threshold,
               merit_threshold_overridden ? "CLI override" : "live node difficulty");
        printf("\n");

        mining_info_free(info);
    }

    struct miner_farm_config farm_config = {
        .num_gpus = num_threads,
        .crt_rt = crt_mode ? &crt_rt : NULL,
        .merit_threshold = merit_threshold,
        .sieve_prime_limit = sieve_primes
    };
    g_farm = miner_farm_create(&farm_config);
    if (!g_farm) {
        fprintf(stderr, "[Main] Failed to create miner farm\n");
        gapcoin_rpc_free(g_rpc);
        return 1;
    }

    if (stratum_mode) {
        /* Materialize the pool header: decode the 80-byte prefix and pick a
           nonce whose hash meets Gapcoin's 256-bit requirement (the nonce is
           part of the PoW payload, so the pool re-checks it itself). */
        active_work.nonce = 0;
        if (!decode_hex80(pool_hdr80_hex, active_work.header_prefix) ||
            gapcoin_gbt_work_hash(&active_work, initial_h256) != 0) {
            fprintf(stderr, "[Main] Cannot materialize the pool work header\n");
            miner_farm_free(g_farm);
            stratum_disconnect(g_stratum);
            g_stratum = NULL;
            return 1;
        }
        work_difficulty = (uint32_t)merit_threshold;
        work_height = 0;
    } else {
        tmpl = gapcoin_rpc_get_block_template(g_rpc);
        if (!tmpl) {
            fprintf(stderr, "[Main] Failed to fetch the initial block template\n");
            miner_farm_free(g_farm);
            gapcoin_rpc_free(g_rpc);
            return 1;
        }
        fprintf(stderr, "[Main] Loaded GBT: height=%u curtime=%u tx_count=%zu version=%u\n",
                tmpl->height, tmpl->curtime, tmpl->transaction_count, tmpl->version);

        if (gapcoin_gbt_work_init(&active_work, tmpl) != 0 ||
            (!crt_mode && gapcoin_gbt_work_hash(&active_work, initial_h256) != 0)) {
            fprintf(stderr, "[Main] Cannot materialize the initial GBT header\n");
            block_template_free(tmpl);
            miner_farm_free(g_farm);
            gapcoin_rpc_free(g_rpc);
            return 1;
        }
        work_difficulty = (uint32_t)tmpl->difficulty;
        work_height = tmpl->height;
        g_rpc_prev_height = tmpl->height;
    }

    if (crt_mode) {
        miner_farm_update_work_crt(g_farm, work_height, user_shift,
                                   work_difficulty,
                                   active_work.header_prefix, active_work.nonce);
    } else {
        miner_farm_update_work(g_farm, work_height, user_shift,
                               work_difficulty, initial_h256,
                               active_work.nonce);
    }
    uint64_t header_bases = 1;
    if (!crt_mode) {
        printf("[Main] %s header base: nonce=%u, shift=%u, window=%u\n",
               stratum_mode ? "Pool" : "GBT",
               active_work.nonce, user_shift, owned_window_size);
    }
    
    /* 3. Set up signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* 4. Start workers */
    if (miner_farm_start(g_farm) != 0) {
        fprintf(stderr, "[Main] Failed to start workers\n");
        block_template_free(tmpl);
        miner_farm_free(g_farm);
        gapcoin_rpc_free(g_rpc);
        return 1;
    }
    
    printf("[Main] Workers started, scanning active (%s)...\n",
           enable_submission ? "submission enabled" : "submission disabled");
    if (stratum_mode) {
        printf("[Main] Pool client running, waiting for new work pushes\n");
    } else {
        printf("[Main] Connected to %s:%u, waiting for block template updates\n",
               rpc_host, rpc_port);
    }

    /* 5. Start the RPC polling thread (node mode only: in pool mode the work
          arrives on the stratum client's receive thread instead). */
    if (!stratum_mode) {
        g_rpc_running = 1;
        if (pthread_create(&g_rpc_thread, NULL, rpc_poll_thread_func, NULL) != 0) {
            fprintf(stderr, "[Main] Failed to create RPC polling thread\n");
            return 1;
        }
    }
    
    /* 6. Main loop: monitor workers and print rolling scan statistics. */
    time_t start_time = time(NULL);  /* For uptime tracking */
    uint64_t prev_total_nonces = 0;  /* For throughput calculation */
    double prev_throughput = 0.0;  /* For rolling average throughput */
    int stats_count = 0;  /* Number of stats samples */
    uint32_t prev_height = 0;  /* Track previous block height for new block detection */
    time_t last_stats_time = time(NULL);  /* Track last stats print time */
    uint64_t prev_gpu_accounted_us = 0;  /* For acc/wall GPU-utilization metric */
    
    while (!g_farm->stop_flag) {
        /* Check if RPC thread has new block template (node mode; pool mode gets
           its work from the stratum client, see the block below). */
        pthread_mutex_lock(&g_rpc_lock);
        if (!stratum_mode && g_new_block_ready && g_rpc_template) {
            if (tmpl) {
                block_template_free(tmpl);
            }
            tmpl = g_rpc_template;
            g_rpc_template = NULL;
            g_new_block_ready = 0;
            
            /* New block detected by RPC thread */
            if (tmpl->height != prev_height) {
                printf("\n");
                printf("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n");
                printf("★ NEW BLOCK ★ Height: %u | Starting new work (shift: %u)\n", 
                       tmpl->height, user_shift);
                printf("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n");
                printf("\n");
                prev_height = tmpl->height;
            }

            if (gapcoin_gbt_work_init(&active_work, tmpl) != 0) {
                fprintf(stderr, "[Main] Stopping: cannot materialize new GBT header\n");
                miner_farm_stop(g_farm);
            } else {
                if (crt_mode) {
                    miner_farm_update_work_crt(g_farm, tmpl->height, user_shift,
                                               (uint32_t)tmpl->difficulty,
                                               active_work.header_prefix,
                                               active_work.nonce);
                } else {
                    uint8_t h256[32];
                    if (gapcoin_gbt_work_hash(&active_work, h256) != 0) {
                        fprintf(stderr, "[Main] Stopping: cannot materialize new GBT header\n");
                        miner_farm_stop(g_farm);
                    } else {
                        miner_farm_update_work(g_farm, tmpl->height, user_shift,
                                               (uint32_t)tmpl->difficulty, h256,
                                               active_work.nonce);
                    }
                }
                header_bases++;
                if (!crt_mode) {
                    printf("[Main] GBT header base: nonce=%u, shift=%u, window=%u\n",
                           active_work.nonce, user_shift, owned_window_size);
                }
            }
        }
        pthread_mutex_unlock(&g_rpc_lock);

        /* Pool mode: work is pushed by the pool.  A new work item (or a share
           target change) restarts the search on the new header; the share
           target keeps the chain threshold current when the pool moves it. */
        if (stratum_mode) {
            char new_hdr_hex[STRATUM_DATA_HEX_SIZE];
            uint64_t s_nd = 0;
            uint64_t n_nd = 0;
            if (stratum_poll_work(g_stratum, new_hdr_hex, &s_nd, &n_nd)) {
                double new_share_merit = stratum_ndiff_to_merit(s_nd);
                memcpy(pool_hdr80_hex, new_hdr_hex, sizeof(pool_hdr80_hex));
                active_work.nonce = 0;
                if (!decode_hex80(new_hdr_hex, active_work.header_prefix) ||
                    gapcoin_gbt_work_hash(&active_work, initial_h256) != 0) {
                    fprintf(stderr, "[Main] Stopping: pool work is not a usable header\n");
                    miner_farm_stop(g_farm);
                } else {
                    if (!merit_threshold_overridden && new_share_merit > 0.0 &&
                        new_share_merit != merit_threshold) {
                        merit_threshold = new_share_merit;
                        miner_farm_set_merit_threshold(g_farm, merit_threshold);
                        printf("[Main] Pool moved the share target to %.4f merit\n",
                               merit_threshold);
                    }
                    work_difficulty = (uint32_t)merit_threshold;
                    printf("\n");
                    printf("★ POOL NEW WORK ★ share=%.4f merit | network=%.4f merit | shift=%u\n",
                           stratum_ndiff_to_merit(s_nd),
                           stratum_ndiff_to_merit(n_nd), user_shift);
                    printf("\n");
                    if (crt_mode) {
                        miner_farm_update_work_crt(g_farm, work_height, user_shift,
                                                   work_difficulty,
                                                   active_work.header_prefix,
                                                   active_work.nonce);
                    } else {
                        miner_farm_update_work(g_farm, work_height, user_shift,
                                               work_difficulty, initial_h256,
                                               active_work.nonce);
                    }
                    header_bases++;
                }
            }
        }

        if (!crt_mode && miner_farm_work_exhausted(g_farm, user_shift)) {
            uint8_t h256[32];
            if (gapcoin_gbt_work_next_hash(&active_work, h256) != 0) {
                fprintf(stderr, "[Main] Stopping: GBT header nonce range exhausted\n");
                miner_farm_stop(g_farm);
            } else {
                miner_farm_update_work(g_farm, work_height, user_shift,
                                       work_difficulty, h256, active_work.nonce);
                header_bases++;
                printf("[Main] Adder range exhausted; rotating GBT header nonce to %u\n",
                       active_work.nonce);
            }
        }

        /* Flush any BPSW-verified gaps queued for real submission. */
        if (enable_submission && (tmpl || stratum_mode)) {
            struct gap_queue_entry entry;
            while (worker_get_pending_gap(&entry)) {
                uint64_t current_generation = atomic_load_explicit(
                    &g_farm->work_generation, memory_order_acquire);
                char *nadd_dec = gap_entry_nadd_dec(&entry);
                if (entry.generation != current_generation) {
                    g_submit_stale++;
                    fprintf(stderr,
                            "[Main] Dropping stale gap submission: height=%u nAdd=%s "
                            "(header rotated since discovery)\n",
                            entry.height, nadd_dec ? nadd_dec : "?");
                    record_log_write_outcome_big(entry.height, entry.shift,
                                                 entry.header_nonce, nadd_dec,
                                                 entry.gap_length, entry.merit,
                                                 "stale");
                    free(nadd_dec);
                    continue;
                }

                /* nAdd as little-endian bytes (legacy uint64 or CRT big). */
                uint8_t local_nadd[8];
                size_t local_nadd_len = 0;
                const uint8_t *nadd_ptr = entry.nadd_len > 0 ?
                    entry.nadd_bytes : NULL;
                size_t nadd_sz = entry.nadd_len;
                if (entry.nadd_len == 0) {
                    uint64_t rem = entry.nadd;
                    if (rem == 0) {
                        local_nadd[0] = 0;
                        local_nadd_len = 1;
                    } else {
                        while (rem > 0) {
                            local_nadd[local_nadd_len++] =
                                (uint8_t)(rem & 0xffU);
                            rem >>= 8;
                        }
                    }
                    nadd_ptr = local_nadd;
                    nadd_sz = local_nadd_len;
                }

                /* Pool mode: send the PoW SOLUTION, not a block.  The payload
                   is hdr80 + nNonce + nShift + nAdd (>86 bytes) and the pool
                   holds the template (its merkle root is inside the header it
                   gave us), so there is nothing to assemble.  There is no
                   share/block flag either: the pool classifies the solution by
                   merit.  The verdict arrives asynchronously and is logged by
                   pool_verdict_cb(); the worker already logged "queued". */
                if (stratum_mode) {
                    struct stratum_share_meta meta;
                    memset(&meta, 0, sizeof(meta));
                    meta.gap_length = entry.gap_length;
                    meta.merit = entry.merit;
                    meta.height = 0;   /* the pool protocol carries no height */
                    g_submit_attempts++;
                    if (stratum_submit_share(g_stratum, active_work.header_prefix,
                                             entry.header_nonce,
                                             (uint16_t)entry.shift, nadd_ptr,
                                             nadd_sz, &meta)) {
                        g_pool_queued++;
                        printf("[Main] Share queued to pool: nAdd=%s gap=%u merit=%.2f "
                               "(share target %.4f, network %.4f)\n",
                               nadd_dec ? nadd_dec : "?", entry.gap_length,
                               entry.merit, stratum_share_merit(g_stratum),
                               stratum_network_merit(g_stratum));
                    } else {
                        int connected = stratum_is_connected(g_stratum);
                        g_pool_send_failed++;
                        fprintf(stderr,
                                "[Main] Pool share NOT sent (%s): nAdd=%s gap=%u "
                                "merit=%.2f\n",
                                connected ? "duplicate solution" : "pool disconnected",
                                nadd_dec ? nadd_dec : "?", entry.gap_length,
                                entry.merit);
                        record_log_write_outcome_big(entry.height, entry.shift,
                                                     entry.header_nonce, nadd_dec,
                                                     entry.gap_length, entry.merit,
                                                     connected ? "duplicate"
                                                               : "send-failed");
                    }
                    free(nadd_dec);
                    continue;
                }

                /* Size the assembly buffer from THIS template: a node whose
                   mempool holds large transactions hands out templates far
                   bigger than the 128 KiB floor, and a short buffer makes the
                   builder fail for every candidate with no RPC attempt (the
                   2026-09-18 incident: 470 KB of template txs, 6 gaps lost and
                   mis-counted as "stale"). */
                size_t hex_cap = gapcoin_gbt_submission_hex_need(tmpl, nadd_sz);
                if (hex_cap < GAPCOIN_SUBMIT_HEX_CAP) {
                    hex_cap = GAPCOIN_SUBMIT_HEX_CAP;
                }
                char *block_hex = malloc(hex_cap);
                if (!block_hex ||
                    gapcoin_gbt_work_build_submission_bytes(
                        active_work.header_prefix, entry.header_nonce, tmpl,
                        entry.shift, nadd_ptr, nadd_sz, block_hex,
                        hex_cap) != 0) {
                    fprintf(stderr,
                            "[Main] Failed to assemble submittable block: height=%u "
                            "nAdd=%s (buffer %zu hex chars = %zu B, template %zu txs; "
                            "the specific reason is on the [gapcoin_work] line)\n",
                            entry.height, nadd_dec ? nadd_dec : "?", hex_cap,
                            hex_cap / 2U, tmpl->transaction_count);
                    free(block_hex);
                    g_submit_assemble_failed++;
                    record_log_write_outcome_big(entry.height, entry.shift,
                                                 entry.header_nonce, nadd_dec,
                                                 entry.gap_length, entry.merit,
                                                 "assemble-failed");
                    free(nadd_dec);
                    continue;
                }

                g_submit_attempts++;
                /* Debug: print first 200 chars of block hex */
                size_t hex_len = strlen(block_hex);
                fprintf(stderr, "[Main] Submitting block (hex first 200 chars): %.200s...\n", block_hex);
                fprintf(stderr, "[Main] Block hex length: %zu bytes (%zu chars)\n", hex_len / 2, hex_len);
                if (gapcoin_rpc_submit_block(g_rpc, block_hex) == 0) {
                    g_submit_accepted++;
                    printf("[Main] Block submitted and ACCEPTED: height=%u nAdd=%s gap=%u merit=%.2f\n",
                           entry.height, nadd_dec ? nadd_dec : "?",
                           entry.gap_length, entry.merit);
                    record_log_write_outcome_big(entry.height, entry.shift,
                                                 entry.header_nonce, nadd_dec,
                                                 entry.gap_length, entry.merit,
                                                 "accepted");
                } else {
                    g_submit_rejected++;
                    fprintf(stderr,
                            "[Main] Block submission REJECTED: height=%u nAdd=%s gap=%u merit=%.2f\n",
                            entry.height, nadd_dec ? nadd_dec : "?",
                            entry.gap_length, entry.merit);
                    record_log_write_outcome_big(entry.height, entry.shift,
                                                 entry.header_nonce, nadd_dec,
                                                 entry.gap_length, entry.merit,
                                                 "rejected");
                }
                free(block_hex);
                free(nadd_dec);
            }
        }

        /* Short cycle: the header nonce must rotate promptly once the workers
           exhaust the current window range (at shift 26 that is only 256
           windows per header, so a 1-second cycle would hard-cap throughput
           at 256 windows/s).  10 ms gives ~100 header rotations/s headroom
           while keeping the idle loop from busy-spinning.  CRT mode scans a
           single small window per header, so a 2 ms cycle lets header
           rotation keep up with the GPU batch rate (~200+ rotations/s). */
        {
            const struct timespec cycle = {
                .tv_sec = 0,
                .tv_nsec = crt_mode ? 2000000L : 10000000L
            };
            nanosleep(&cycle, NULL);
        }
        
        /* Collect and display statistics every 30 seconds */
        time_t now = time(NULL);
        if (now - last_stats_time >= 30) {
            struct farm_stats stats;
            miner_farm_get_stats(g_farm, &stats);
            
            struct mining_info pool_info;
            int pool_info_local = 0;
            struct mining_info *info;
            if (stratum_mode) {
                /* Synthesise what the prints below expect, from the pool.  The
                   live number shown is the NETWORK difficulty; the pool's SHARE
                   target is already applied to the workers as the threshold. */
                memset(&pool_info, 0, sizeof(pool_info));
                pool_info.blocks = 0;
                pool_info.difficulty = stratum_network_merit(g_stratum);
                info = &pool_info;
                pool_info_local = 1;
            } else {
                info = gapcoin_rpc_get_mining_info(g_rpc);
            }
            if (info) {
                if (!pool_info_local && !merit_threshold_overridden &&
                    info->difficulty > 0.0 &&
                    info->difficulty != merit_threshold) {
                    merit_threshold = info->difficulty;
                    miner_farm_set_merit_threshold(g_farm, merit_threshold);
                    printf("[Main] Updated active merit threshold to %.2f from node difficulty\n",
                           merit_threshold);
                }

                /* Calculate throughput over last 30 seconds */
                double wall_interval_s = (double)(now - last_stats_time);
                uint64_t nonces_in_interval = stats.total_nonces - prev_total_nonces;
                double throughput = (double)nonces_in_interval / wall_interval_s;
                
                /* Calculate rolling average (EMA: 70% current, 30% previous) */
                if (stats_count == 0) {
                    prev_throughput = throughput;
                } else {
                    prev_throughput = 0.7 * throughput + 0.3 * prev_throughput;
                }
                stats_count++;
                
                /* Calculate uptime */
                time_t uptime = now - start_time;
                int uptime_hours = uptime / 3600;
                int uptime_mins = (uptime % 3600) / 60;
                int uptime_secs = uptime % 60;
                
                prev_total_nonces = stats.total_nonces;
                last_stats_time = now;
                
                /* Calculate time per sieve window. */
                double time_per_nonce = (throughput > 0) ? 1000.0 / throughput : 0.0;  /* ms per nonce */
                double stats_window = crt_mode ?
                    (2.0 * ceil(merit_threshold *
                                ((256.0 + (double)crt_rt.shift) * log(2.0)))) :
                    (double)owned_window_size;
                double miner_power_mhs =
                    throughput * stats_window / 1000000.0;

                /* GPU-accounted fraction (acc/wall): pure MR-kernel execution
                   time (CUDA-event measured, summed over all workers) divided
                   by the wall-clock stats interval (wall_interval_s, captured
                   before last_stats_time is advanced above).  Kernels from
                   different workers serialize on the shared GPU, so the sum
                   equals the GPU's total busy time:
                     acc/wall ≈ 1.0  → GPU ~fully busy, balanced
                     acc/wall > 1.0  → GPU saturated (workers demand more GPU
                                        time than wall clock; GPU is the
                                        bottleneck — the healthy mining case)
                     acc/wall < 1.0  → host-bound: launch/sync gaps or idle
                                        GPU (the cuda-sieve finding 53 signal)
                   acc/wall is only meaningful with GPU MR active; it stays
                   0.0 in pure-CPU runs. */
                double gpu_accounted_s =
                    (double)(stats.total_gpu_accounted_us -
                             prev_gpu_accounted_us) / 1000000.0;
                double acc_wall = (wall_interval_s > 0.0) ?
                    (gpu_accounted_s / wall_interval_s) : 0.0;
                prev_gpu_accounted_us = stats.total_gpu_accounted_us;
                
                /* Rolling stats format */
                printf("\n");
                printf("═══════════════════════════════════════════════════════════════\n");
                printf("  ROLLING STATS [%02dh:%02dm:%02ds]\n", uptime_hours, uptime_mins, uptime_secs);
                printf("═══════════════════════════════════════════════════════════════\n");
                  printf("  Network: Height=%u | Difficulty: %.2f | Active Merit Req: %.2f (%s)\n",
                      info->blocks, info->difficulty, merit_threshold,
                      merit_threshold_overridden ? "CLI override" : "live");
                  printf("  Network power: %.2fM mH/s | Miner power: %.3f mH/s (owned nAdd/s)\n",
                      info->networkminingpower / 1000000.0, miner_power_mhs);
                  if (throughput >= 1000000.0) {
                      printf("  Throughput: %.2fM windows/s (%.3f ms/window) | Rolling avg: %.2fM/s\n",
                          throughput / 1e6, time_per_nonce, prev_throughput / 1e6);
                  } else {
                      printf("  Throughput: %.0f windows/s (%.3f ms/window) | Rolling avg: %.0f/s\n",
                          throughput, time_per_nonce, prev_throughput);
                  }
                  printf("  Processed: %lu windows | Sieve survivors: %lu\n",
                       stats.total_nonces, stats.total_candidates);
                  if (stats.total_smart_tail_skipped > 0) {
                      printf("  Smart-scan: %lu tails skipped (uncovered region not needed)\n",
                          stats.total_smart_tail_skipped);
                  }
                  printf("  Euler passes: %lu | Euler pairs: %lu | Merit candidates: %lu\n",
                      stats.total_euler_passes, stats.total_euler_pairs,
                      stats.total_merit_candidates);
                  {
                      /* Expensive-test density: how many candidates each
                         window actually sends to the GPU MR kernel (the
                         full-scan path tests every sieve survivor; the
                         MINING_JUMP2 chain tests only the frontier) plus the
                         same ratio per 1000 survivors, which is independent
                         of the window geometry. */
                      double mr_per_window = stats.total_nonces ?
                          (double)stats.total_candidates_tested /
                          (double)stats.total_nonces : 0.0;
                      double mr_per_1000_surv = stats.total_candidates ?
                          1000.0 * (double)stats.total_candidates_tested /
                          (double)stats.total_candidates : 0.0;
                      printf("  GPU MR tests: %lu (%.1f/window, %.1f per 1000 survivors)\n",
                          stats.total_candidates_tested, mr_per_window,
                          mr_per_1000_surv);
                  }
                  if (stats.total_us_mark || stats.total_us_extract ||
                      stats.total_us_collect || stats.total_us_chain) {
                      /* FUSED_STAGE_TIMING=1 stage split, as a percentage of
                         total uptime.  "mark"/"extract"/"collect" are the
                         host-side stage calls of the fused pipeline; the
                         chain block is the whole MINING_JUMP2 flight with its
                         own gather/MR round trips. */
                      double up_s = (double)(uptime > 0 ? uptime : 1);
                      double pct = 100.0 / (up_s * 1e6);
                      printf("  Fused stage split: mark=%.1f%% extract=%.1f%% "
                             "collect=%.1f%% chain=%.1f%% (rounds=%lu, "
                             "gather=%.1f%%, mr=%.1f%%)\n",
                          (double)stats.total_us_mark * pct,
                          (double)stats.total_us_extract * pct,
                          (double)stats.total_us_collect * pct,
                          (double)stats.total_us_chain * pct,
                          (unsigned long)stats.total_chain_rounds,
                          (double)stats.total_us_chain_gather * pct,
                          (double)stats.total_us_chain_mr * pct);
                  }
                  if (stats.total_sieve_mark_us || stats.total_sieve_extract_us) {
                      /* GPU_SIEVE_TIMING=1: pure kernel time (CUDA events) of
                         the sieve stages, as a share of uptime.  Together
                         with GPU MR acc/wall this closes the GPU-busy
                         budget: MR + mark + extract + idle = wall. */
                      double up_s = (double)(uptime > 0 ? uptime : 1);
                      double pct = 100.0 / (up_s * 1e6);
                      printf("  GPU kernel split: MR=%.1f%% mark=%.1f%% "
                             "extract=%.1f%% of wall (mark=%.2f s, "
                             "extract=%.2f s)\n",
                          (double)stats.total_gpu_accounted_us * pct,
                          (double)stats.total_sieve_mark_us * pct,
                          (double)stats.total_sieve_extract_us * pct,
                          (double)stats.total_sieve_mark_us / 1e6,
                          (double)stats.total_sieve_extract_us / 1e6);
                  }
                  if (stats.total_gpu_euler_skipped > 0) {
                      printf("  GPU Fermat: %lu Euler calls skipped (composite pre-filter)\n",
                          stats.total_gpu_euler_skipped);
                  }
                  if (stats.total_gpu_sieve_calls > 0 ||
                      stats.total_gpu_sieve_windows > 0) {
                      if (stats.total_gpu_sieve_calls > 0) {
                          double windows_per_batch =
                              (double)stats.total_gpu_sieve_windows /
                              (double)stats.total_gpu_sieve_calls;
                          printf("  GPU sieve: %lu batches | %lu windows (avg %.2f windows/batch)\n",
                              stats.total_gpu_sieve_calls,
                              stats.total_gpu_sieve_windows,
                              windows_per_batch);
                      } else {
                          printf("  GPU sieve: %lu windows\n",
                              stats.total_gpu_sieve_windows);
                      }
                  }
                  printf("  GPU MR acc/wall: %.3f (%.3f s GPU-accounted of %.1f s wall) [<1 host-bound, >1 GPU-bound]\n",
                      acc_wall, gpu_accounted_s, wall_interval_s);
                  printf("  Header bases: %llu | Current header nonce: %u\n",
                      (unsigned long long)header_bases, active_work.nonce);
                  if (enable_submission) {
                      printf("  BPSW attempts: %lu | Passed: %lu | Submit: attempts=%llu accepted=%llu rejected=%llu stale=%llu asm_fail=%llu\n",
                          stats.total_bpsw_attempts, stats.total_gaps,
                          (unsigned long long)g_submit_attempts,
                          (unsigned long long)g_submit_accepted,
                          (unsigned long long)g_submit_rejected,
                          (unsigned long long)g_submit_stale,
                          (unsigned long long)g_submit_assemble_failed);
                  } else {
                      printf("  BPSW attempts: %lu | Passed: %lu | Submitted: %lu (dry-run)\n",
                          stats.total_bpsw_attempts, stats.total_gaps,
                          stats.total_submissions);
                  }
                  if (stratum_mode) {
                      /* Pool accounting: the verdicts arrive asynchronously, so
                         these are the authoritative share numbers in pool mode. */
                      uint64_t p_acc = 0, p_rej = 0, p_dup = 0, p_sf = 0;
                      uint64_t p_rec = 0, p_cf = 0, p_unres = 0;
                      stratum_get_stats(g_stratum, &p_acc, &p_rej, &p_dup, &p_sf,
                                        &p_rec, &p_cf, &p_unres);
                      printf("  Pool shares: queued=%llu accepted=%llu rejected=%llu "
                             "duplicate=%llu send-failed=%llu unresolved=%llu\n",
                             (unsigned long long)g_pool_queued,
                             (unsigned long long)p_acc, (unsigned long long)p_rej,
                             (unsigned long long)p_dup, (unsigned long long)p_sf,
                             (unsigned long long)p_unres);
                      printf("  Pool link: %s | reconnects=%llu connect-failures=%llu "
                             "| share target=%.4f network=%.4f merit\n",
                             stratum_is_connected(g_stratum) ? "connected" : "down",
                             (unsigned long long)p_rec, (unsigned long long)p_cf,
                             stratum_share_merit(g_stratum),
                             stratum_network_merit(g_stratum));
                  }
                  {
                      /* Yield instrumentation: expected blocks/h is
                         win/s x P(merit >= m), and P is MEASURED here
                         (merit candidates / windows) rather than fitted from a
                         sigma model.  Identity: candidates/uptime is exactly
                         win/s x P, so writing it this way cannot drift from
                         the throughput number.  A sigma model is deliberately
                         NOT used: it needs a second threshold point, and the
                         naive Cramer form is uncalibrated for the chain (it
                         predicts ~50x fewer qualifying windows than are
                         observed -- measured P 1.7e-7/window against a Cramer
                         estimate of 3.5e-9 at m=23.8 on the fleet, because the
                         covering, not Poisson hole statistics, produces them).
                         The per-million-window rate is the geometry
                         instrument: it divides out throughput, so two
                         covers/shifts can be compared directly.  Both figures
                         are cumulative and therefore only meaningful at a
                         FIXED threshold -- if the network difficulty moves
                         during the run the cumulative rate blends thresholds. */
                      double up_h = (uptime > 0) ? (double)uptime / 3600.0 : 0.0;
                      double cand = (double)stats.total_merit_candidates;
                      double win = (double)stats.total_nonces;
                      double per_win = (win > 0.0) ? cand / win : 0.0;
                      printf("  Yield: %.2f blocks/h expected | %.3f per Mwin",
                          (up_h > 0.0) ? cand / up_h : 0.0, per_win * 1e6);
                      if (per_win > 0.0) {
                          /* Same number inverted, scaled so low-threshold
                             runs (merit 10: ~1 candidate per 270 windows)
                             do not print a useless "1 in 0.00M windows". */
                          double inv = 1.0 / per_win;
                          if (inv >= 1e6) {
                              printf(" (1 in %.2fM windows)", inv / 1e6);
                          } else if (inv >= 1e3) {
                              printf(" (1 in %.0fk windows)", inv / 1e3);
                          } else {
                              printf(" (1 in %.0f windows)", inv);
                          }
                      }
                      printf(" | %.0f candidates @ merit>=%.2f (%s)",
                          cand, merit_threshold,
                          merit_threshold_overridden ? "CLI" : "live");
                      /* Poisson noise guard.  A rate rebuilt from n events
                         carries 1/sqrt(n) relative error, so a 30 s sample
                         with 2 events reads ~20x above the truth (measured
                         on the dev box: 240 b/h from n=2, while n=0 over the
                         next 3.1M windows bounded the same quantity at
                         <38 b/h).  Print the sample size next to the rate so
                         a short run cannot be mistaken for a measurement,
                         and print an upper bound rather than a fake 0.00 when
                         nothing has been seen yet. */
                      if (cand >= 1.0) {
                          printf(" | n=%.0f, 1-sigma %.0f%%",
                              cand, 100.0 / sqrt(cand));
                      } else if (up_h > 0.0) {
                          printf(" | n=0 -> 95%% upper %.1f b/h", 3.0 / up_h);
                      }
                      if (enable_submission && up_h > 0.0) {
                          /* In pool mode the pool's accepted/rejected verdicts
                             are the ground truth (they arrive asynchronously). */
                          uint64_t acc_c = g_submit_accepted;
                          uint64_t att_c = g_submit_attempts;
                          if (stratum_mode) {
                              uint64_t a = 0, r = 0, d = 0, s = 0, rc = 0, cf = 0, u = 0;
                              stratum_get_stats(g_stratum, &a, &r, &d, &s, &rc,
                                                &cf, &u);
                              acc_c = a;
                              att_c = g_pool_queued;
                          }
                          printf(" | accepted %.2f/h (%.0f%% of %llu attempts)",
                              (double)acc_c / up_h,
                              att_c ? 100.0 * (double)acc_c / (double)att_c : 0.0,
                              (unsigned long long)att_c);
                      }
                      printf("\n");
                  }
                  printf("  Max Euler pair: gap=%u | merit=%.2f\n",
                      stats.max_gap_length, stats.max_merit);
                  {
                      /* Gap-distribution health check vs Hardy-Littlewood.
                         CRT coverings modulate small-gap frequencies by up to
                         ~30% even in uncovered regions, so the warning
                         threshold is wider in CRT mode. */
                      double gap_dist_warn_pct = crt_mode
                          ? GAP_DIST_WARN_PCT_CRT : GAP_DIST_WARN_PCT;
                      struct gap_dist_health gdh;
                      gap_dist_health(&gdh);
                      if (!gap_dist_enabled()) {
                          printf("  Gap-dist health: disabled (HALF_CLASS mode)\n");
                      } else if (gdh.total_gaps > 0) {
                          if (gdh.enough_samples) {
                              printf("  Gap-dist health: %lu gaps (g2=%lu g4=%lu g6=%lu g8=%lu g10=%lu g12=%lu) | dev g=4:%+.1f%% g=6:%+.1f%% g=8:%+.1f%% g=10:%+.1f%% g=12:%+.1f%%",
                                     (unsigned long)gdh.total_gaps,
                                     (unsigned long)gdh.hist_small[0],
                                     (unsigned long)gdh.hist_small[1],
                                     (unsigned long)gdh.hist_small[2],
                                     (unsigned long)gdh.hist_small[3],
                                     (unsigned long)gdh.hist_small[4],
                                     (unsigned long)gdh.hist_small[5],
                                     gdh.dev_pct[0], gdh.dev_pct[1],
                                     gdh.dev_pct[2], gdh.dev_pct[3],
                                     gdh.dev_pct[4]);
                              if (gdh.max_abs_dev > gap_dist_warn_pct) {
                                  printf("\n  ⚠ GAP-DIST DEVIATION: g=%d deviates %+.1f%% from Hardy-Littlewood (|dev| > %.0f%%) — possible sieve/primality bug\n",
                                         gdh.worst_g, gdh.worst_dev,
                                         gap_dist_warn_pct);
                              } else {
                                  printf(" [OK]\n");
                              }
                          } else {
                              printf("  Gap-dist health: collecting (%lu gaps, g2=%lu/%llu)\n",
                                     (unsigned long)gdh.total_gaps,
                                     (unsigned long)gdh.hist_small[0],
                                     (unsigned long long)GAP_DIST_MIN_C2);
                          }
                      }
                  }
                  if (getenv("GAP_DIST_DUMP")) {
                      uint64_t snap[GAP_DIST_BUCKETS];
                      gap_dist_snapshot(snap);
                      printf("  Gap-dist dump (bucket i = gap 2i):\n");
                      for (int gi = 0; gi < 40; gi++) {
                          printf("    g=%d:%lu", gi * 2,
                                 (unsigned long)snap[gi]);
                          if ((gi & 3) == 3) printf("\n");
                      }
                  }
                    printf("---------------------------------------------------------------\n");
                printf("\n");
                
                if (!pool_info_local)
                    mining_info_free(info);
            }
        }
    }
    
    /* 6. Cleanup */
    if (stratum_mode) {
        printf("[Main] Disconnecting from the pool...\n");
        stratum_disconnect(g_stratum);
        g_stratum = NULL;
    } else {
        printf("[Main] Stopping RPC thread...\n");
        g_rpc_running = 0;
        pthread_join(g_rpc_thread, NULL);
    }
    
    printf("[Main] Joining workers...\n");
    miner_farm_join(g_farm);
    miner_farm_free(g_farm);
    if (crt_mode) {
        crt_runtime_free(&crt_rt);
    }
    
    if (tmpl) {
        block_template_free(tmpl);
    }
    
    if (g_rpc_template) {
        block_template_free(g_rpc_template);
    }
    
    /* gapcoin_rpc_free(NULL) is a no-op, so pool mode (g_rpc == NULL) is fine. */
    gapcoin_rpc_free(g_rpc);
    record_log_close();
    merit_records_free();
    
    printf("[Main] Shutdown complete\n");
    printf("================================================\n");
    
    return 0;
}
