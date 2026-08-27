/* DEACTIVATED: unused module - removed from the build (not in Makefile SOURCES).
 * Kept for reference. To restore: re-add to Makefile SOURCES and delete this block.
 */
/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * RPC Interface Implementation (Phase 5 Stub)
 *
 * Wraps existing stratum.c functions from cpugapminer.
 * Full integration with pool communication in later phases.
 */

#include "rpc_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Global RPC state (stub) */
static struct {
    int connected;
    char *pool_url;
    char *worker_name;
} g_rpc_state = {0, NULL, NULL};

int rpc_connect(const char *pool_url, const char *worker_name) {
    if (!pool_url || !worker_name) return -1;
    
    g_rpc_state.pool_url = (char *)malloc(strlen(pool_url) + 1);
    g_rpc_state.worker_name = (char *)malloc(strlen(worker_name) + 1);
    
    if (!g_rpc_state.pool_url || !g_rpc_state.worker_name) return -1;
    
    strcpy(g_rpc_state.pool_url, pool_url);
    strcpy(g_rpc_state.worker_name, worker_name);
    g_rpc_state.connected = 1;
    
    printf("[RPC] Connected to %s as %s\n", pool_url, worker_name);
    
    return 0;
}

int rpc_get_work(struct rpc_work *work) {
    if (!work) return -1;
    
    /* Stub: return dummy work */
    work->height = 1000;
    work->shift = 512;
    work->pool_url = g_rpc_state.pool_url;
    work->worker_name = g_rpc_state.worker_name;
    
    /* TODO: Call existing stratum.c functions to fetch real work */
    
    return 0;
}

int rpc_submit_gap(struct rpc_result *result,
                  const struct rpc_work *work,
                  uint64_t offset_p1,
                  uint64_t offset_p2,
                  uint32_t gap_length) {
    if (!result || !work) return -1;
    
    result->accepted = 1;  /* Stub: always accept */
    result->message = (char *)malloc(256);
    snprintf(result->message, 256, "Share accepted: gap %u (offsets %lu, %lu)", 
             gap_length, offset_p1, offset_p2);
    result->difficulty = 1000;
    
    printf("[RPC] Submitted gap: length=%u, merit approx=%.2f\n", 
           gap_length, gap_length / log(10000000.0));
    
    return 0;
}

int rpc_work_updated(void) {
    /* TODO: Check for new work from pool */
    return 0;
}

void rpc_disconnect(void) {
    if (g_rpc_state.pool_url) free(g_rpc_state.pool_url);
    if (g_rpc_state.worker_name) free(g_rpc_state.worker_name);
    g_rpc_state.connected = 0;
    
    printf("[RPC] Disconnected\n");
}
