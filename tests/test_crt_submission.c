/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CRT submission test: verify that gapcoin_gbt_work_build_submission_bytes
 * correctly serializes an arbitrary-length (>64-bit) nAdd and that the nAdd
 * bytes round-trip through the assembled block hex.
 */

#include "../new_src/gapcoin_work.h"
#include "../new_src/gapcoin_rpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Decode a hex char pair. */
static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static size_t decode_hex_bytes(const char *hex, uint8_t *out, size_t cap) {
    size_t n = 0;
    for (size_t i = 0; hex[i] && hex[i + 1] && n < cap; i += 2) {
        int hi = hexval(hex[i]);
        int lo = hexval(hex[i + 1]);
        if (hi < 0 || lo < 0) break;
        out[n++] = (uint8_t)((hi << 4) | lo);
    }
    return n;
}

int main(void) {
    int failures = 0;

    struct block_template tmpl;
    memset(&tmpl, 0, sizeof(tmpl));
    tmpl.height = 2514084;
    tmpl.curtime = 1692000000;
    tmpl.transaction_count = 0;
    tmpl.transaction_data = NULL;
    tmpl.transaction_hashes = NULL;

    uint8_t header_prefix[80];
    for (int i = 0; i < 80; i++) header_prefix[i] = (uint8_t)(i * 7 + 3);

    uint32_t shift = 720;
    uint32_t header_nonce = 12345;

    /* 90-byte nAdd (720-bit CRT alignment offset). */
    uint8_t nadd[90];
    for (int i = 0; i < 90; i++) nadd[i] = (uint8_t)((i * 13 + 5) & 0xff);
    nadd[89] = 0x01;  /* most-significant byte nonzero */

    char hex[GAPCOIN_SUBMIT_HEX_CAP];
    int rc = gapcoin_gbt_work_build_submission_bytes(
        header_prefix, header_nonce, &tmpl, shift, nadd, 90, hex,
        sizeof(hex));
    if (rc != 0) {
        printf("FAIL: build_submission_bytes returned %d\n", rc);
        failures++;
    }

    /* Decode and locate the nAdd field: 80 (prefix) + 4 (nonce) + 2 (shift)
       + CompactSize. The nAdd bytes follow. */
    size_t hex_len = strlen(hex);
    uint8_t *block = (uint8_t *)malloc(hex_len / 2 + 1);
    size_t block_len = decode_hex_bytes(hex, block, hex_len / 2 + 1);
    if (block_len < 80 + 4 + 2 + 1) {
        printf("FAIL: block too short (%zu bytes)\n", block_len);
        failures++;
        free(block);
        return failures ? 1 : 0;
    }

    /* Verify prefix + nonce + shift. */
    if (memcmp(block, header_prefix, 80) != 0) {
        printf("FAIL: header prefix mismatch\n");
        failures++;
    }
    if (block[80] != (uint8_t)header_nonce || block[81] != (uint8_t)(header_nonce >> 8) ||
        block[82] != (uint8_t)(header_nonce >> 16) || block[83] != (uint8_t)(header_nonce >> 24)) {
        printf("FAIL: header nonce mismatch\n");
        failures++;
    }
    if (block[84] != (uint8_t)shift || block[85] != (uint8_t)(shift >> 8)) {
        printf("FAIL: shift mismatch\n");
        failures++;
    }

    /* Parse CompactSize at offset 86. */
    size_t pos = 86;
    size_t nadd_len = 0;
    if (block[pos] < 0xfd) {
        nadd_len = block[pos];
        pos += 1;
    } else if (block[pos] == 0xfd) {
        nadd_len = block[pos + 1] | (block[pos + 2] << 8);
        pos += 3;
    } else if (block[pos] == 0xfe) {
        nadd_len = (size_t)block[pos + 1] | ((size_t)block[pos + 2] << 8) |
                   ((size_t)block[pos + 3] << 16) | ((size_t)block[pos + 4] << 24);
        pos += 5;
    } else {
        printf("FAIL: unsupported CompactSize prefix 0x%02x\n", block[pos]);
        failures++;
        free(block);
        return 1;
    }

    if (nadd_len != 90) {
        printf("FAIL: nAdd length %zu != 90\n", nadd_len);
        failures++;
    }
    if (pos + nadd_len > block_len) {
        printf("FAIL: nAdd overruns block\n");
        failures++;
    } else if (memcmp(block + pos, nadd, 90) != 0) {
        printf("FAIL: nAdd bytes mismatch\n");
        failures++;
    }

    printf("nAdd round-trip: %zu bytes, block %zu bytes\n", nadd_len, block_len);
    if (failures == 0) {
        printf("PASS: big-nAdd block assembly\n");
    }
    free(block);
    return failures ? 1 : 0;
}
