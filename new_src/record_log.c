/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#define _POSIX_C_SOURCE 200809L

#include "record_log.h"
#include "merit_records.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

static FILE *g_record_log = NULL;
static pthread_mutex_t g_record_log_lock = PTHREAD_MUTEX_INITIALIZER;

int record_log_init(const char *path) {
    pthread_mutex_lock(&g_record_log_lock);
    if (g_record_log) fclose(g_record_log);
    g_record_log = fopen(path, "a");
    int ok = g_record_log != NULL;
    pthread_mutex_unlock(&g_record_log_lock);

    if (!ok) {
        fprintf(stderr, "[RecordLog] Could not open %s for writing\n", path);
        return -1;
    }
    printf("[RecordLog] Logging BPSW candidates to %s\n", path);
    return 0;
}

void record_log_write(uint32_t height, uint32_t shift, uint32_t header_nonce,
                      uint64_t nadd, const mpz_t start, uint32_t gap_length,
                      double merit, const char *status) {
    char nadd_dec[24];
    snprintf(nadd_dec, sizeof(nadd_dec), "%llu", (unsigned long long)nadd);
    record_log_write_big(height, shift, header_nonce, nadd_dec, start,
                         gap_length, merit, status);
}

void record_log_write_big(uint32_t height, uint32_t shift, uint32_t header_nonce,
                          const char *nadd_dec, const mpz_t start,
                          uint32_t gap_length, double merit, const char *status) {
    pthread_mutex_lock(&g_record_log_lock);
    if (!g_record_log) {
        pthread_mutex_unlock(&g_record_log_lock);
        return;
    }

    char *start_dec = mpz_get_str(NULL, 10, start);

    double best_merit = 0.0;
    int has_record = merit_records_lookup(gap_length, &best_merit);
    int is_new_record = has_record && merit > best_merit;
    char best_merit_str[32];
    if (has_record) {
        snprintf(best_merit_str, sizeof(best_merit_str), "%.4f", best_merit);
    } else {
        snprintf(best_merit_str, sizeof(best_merit_str), "unknown");
    }

    time_t now = time(NULL);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

    fprintf(g_record_log,
            "%s height=%u shift=%u header_nonce=%u nAdd=%s start=%s "
            "gap=%u merit=%.4f best_known_merit=%s new_record=%s "
            "claim=%s status=%s\n",
            timestamp, height, shift, header_nonce, nadd_dec ? nadd_dec : "?",
            start_dec ? start_dec : "?", gap_length, merit, best_merit_str,
            is_new_record ? "yes" : (has_record ? "no" : "unknown"),
            is_new_record ? "FIRST_KNOWN_OCCURRENCE" : "none", status);

    fflush(g_record_log);

    if (is_new_record) {
        fprintf(stderr,
                "[RecordLog] *** POSSIBLE FIRST_KNOWN_OCCURRENCE *** "
                "gap=%u merit=%.4f (best known: %.4f) height=%u nAdd=%s "
                "-- see the record log for the full number\n",
                gap_length, merit, best_merit, height, nadd_dec ? nadd_dec : "?");
    }

    if (start_dec) free(start_dec);
    pthread_mutex_unlock(&g_record_log_lock);
}

void record_log_close(void) {
    pthread_mutex_lock(&g_record_log_lock);
    if (g_record_log) {
        fclose(g_record_log);
        g_record_log = NULL;
    }
    pthread_mutex_unlock(&g_record_log_lock);
}

void record_log_write_outcome(uint32_t height, uint32_t shift, uint32_t header_nonce,
                              uint64_t nadd, uint32_t gap_length, double merit,
                              const char *status) {
    char nadd_dec[24];
    snprintf(nadd_dec, sizeof(nadd_dec), "%llu", (unsigned long long)nadd);
    record_log_write_outcome_big(height, shift, header_nonce, nadd_dec,
                                 gap_length, merit, status);
}

void record_log_write_outcome_big(uint32_t height, uint32_t shift, uint32_t header_nonce,
                                  const char *nadd_dec, uint32_t gap_length,
                                  double merit, const char *status) {
    pthread_mutex_lock(&g_record_log_lock);
    if (!g_record_log) {
        pthread_mutex_unlock(&g_record_log_lock);
        return;
    }

    time_t now = time(NULL);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

    fprintf(g_record_log,
            "%s height=%u shift=%u header_nonce=%u nAdd=%s gap=%u merit=%.4f status=%s\n",
            timestamp, height, shift, header_nonce, nadd_dec ? nadd_dec : "?",
            gap_length, merit, status);
    fflush(g_record_log);
    pthread_mutex_unlock(&g_record_log_lock);
}
