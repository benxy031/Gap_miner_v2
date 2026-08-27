/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Real RPC Client Implementation
 *
 * Communicates with Gapcoin node via JSON-RPC 2.0 over HTTP.
 * Requires: libcurl, jansson
 */

#include "gapcoin_rpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <jansson.h>

/* Buffer for HTTP response */
struct http_response {
    char *data;
    size_t size;
};

/* Callback for curl to receive response */
static size_t curl_write_data_callback(void *contents, size_t size, 
                                       size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct http_response *mem = (struct http_response *)userp;
    
    char *ptr = realloc(mem->data, mem->size + realsize + 1);
    if (!ptr) {
        fprintf(stderr, "[RPC] Not enough memory for response\n");
        return 0;
    }
    
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    
    return realsize;
}

/* Execute JSON-RPC call.
 *
 * IMPORTANT: each call creates and destroys its OWN curl easy handle.
 * The RPC polling thread and the submission path run concurrently, and
 * libcurl easy handles are not thread-safe.  Sharing one handle across
 * threads produced CURLE_FAILED_INIT failures and crossed write
 * callbacks: an ACCEPTED submitblock could be reported as REJECTED
 * because its response landed in another thread's buffer.
 */
static json_t *gapcoin_rpc_call(struct gapcoin_rpc *rpc, 
                                const char *method,
                                json_t *params,
                                long timeout_sec) {
    if (!rpc || !method) return NULL;
    
    /* Build request */
    json_t *request = json_object();
    json_object_set_new(request, "jsonrpc", json_string("2.0"));
    json_object_set_new(request, "method", json_string(method));
    json_object_set_new(request, "id", json_integer(1));
    
    if (params) {
        json_object_set_new(request, "params", params);
    } else {
        json_object_set_new(request, "params", json_array());
    }
    
    char *request_str = json_dumps(request, 0);
    json_decref(request);
    
    /* Per-call handle (thread-safe: see note above). */
    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "[RPC] CURL init failed\n");
        free(request_str);
        return NULL;
    }
    struct http_response response = {NULL, 0};
    
    char auth_str[512];
    snprintf(auth_str, sizeof(auth_str), "%s:%s", rpc->username, rpc->password);
    
    char url[256];
    snprintf(url, sizeof(url), "http://%s:%u/", rpc->host, rpc->port);
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_USERPWD, auth_str);
    curl_easy_setopt(curl, CURLOPT_HTTPAUTH, (long)CURLAUTH_BASIC);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_str);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &curl_write_data_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    /* Execute */
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(request_str);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "[RPC] CURL error: %s\n", curl_easy_strerror(res));
        free(response.data);
        return NULL;
    }
    
    if (!response.data) {
        fprintf(stderr, "[RPC] Empty response from node\n");
        return NULL;
    }
    
    /* Parse response */
    json_error_t error;
    json_t *reply = json_loads(response.data, 0, &error);
    free(response.data);
    
    if (!reply) {
        fprintf(stderr, "[RPC] JSON parse error: %s\n", error.text);
        return NULL;
    }
    
    /* Check for error */
    json_t *err = json_object_get(reply, "error");
    if (err && !json_is_null(err)) {
        json_t *err_msg = json_object_get(err, "message");
        fprintf(stderr, "[RPC] RPC error: %s\n", 
                json_string_value(err_msg) ?: "unknown");
        json_decref(reply);
        return NULL;
    }
    
    /* Extract result */
    json_t *result = json_object_get(reply, "result");
    if (!result) {
        json_decref(reply);
        return NULL;
    }
    
    json_t *result_copy = json_deep_copy(result);
    json_decref(reply);
    
    return result_copy;
}

struct gapcoin_rpc *gapcoin_rpc_connect(const char *host, uint16_t port,
                                        const char *username, 
                                        const char *password) {
    if (!host || !username || !password) return NULL;
    
    struct gapcoin_rpc *rpc = 
        (struct gapcoin_rpc *)malloc(sizeof(struct gapcoin_rpc));
    if (!rpc) return NULL;
    
    rpc->host = (char *)malloc(strlen(host) + 1);
    rpc->username = (char *)malloc(strlen(username) + 1);
    rpc->password = (char *)malloc(strlen(password) + 1);
    
    if (!rpc->host || !rpc->username || !rpc->password) {
        free(rpc->host);
        free(rpc->username);
        free(rpc->password);
        free(rpc);
        return NULL;
    }
    
    strcpy(rpc->host, host);
    strcpy(rpc->username, username);
    strcpy(rpc->password, password);
    rpc->port = port;
    
    /* Process-wide libcurl init (idempotent).  Individual requests create
       their own easy handles, which is thread-safe. */
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    printf("[RPC] Connected to %s:%u as %s\n", host, port, username);
    
    return rpc;
}

uint32_t gapcoin_rpc_get_block_count(struct gapcoin_rpc *rpc) {
    if (!rpc) return 0;
    
    json_t *result = gapcoin_rpc_call(rpc, "getblockcount", NULL, 10L);
    if (!result) return 0;
    
    uint32_t count = json_integer_value(result);
    json_decref(result);
    
    return count;
}

struct mining_info *gapcoin_rpc_get_mining_info(struct gapcoin_rpc *rpc) {
    if (!rpc) return NULL;
    
    json_t *result = gapcoin_rpc_call(rpc, "getmininginfo", NULL, 10L);
    if (!result) return NULL;
    
    struct mining_info *info = 
        (struct mining_info *)malloc(sizeof(struct mining_info));
    if (!info) {
        json_decref(result);
        return NULL;
    }
    
    memset(info, 0, sizeof(*info));
    
    info->blocks = json_integer_value(json_object_get(result, "blocks"));
    info->difficulty = json_number_value(json_object_get(result, "difficulty"));
    info->networkminingpower = 
        json_number_value(json_object_get(result, "networkminingpower"));
    info->profitability = 
        json_number_value(json_object_get(result, "profitability"));
    
    /* Next block info */
    json_t *next_obj = json_object_get(result, "next");
    if (next_obj) {
        info->next.height = json_integer_value(json_object_get(next_obj, "height"));
        const char *bits_str = json_string_value(json_object_get(next_obj, "bits"));
        if (bits_str) {
            info->next.bits = (char *)malloc(strlen(bits_str) + 1);
            strcpy(info->next.bits, bits_str);
        }
        info->next.difficulty = 
            json_number_value(json_object_get(next_obj, "difficulty"));
    }
    
    json_decref(result);
    return info;
}

static int hex_nibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

int gapcoin_rpc_get_work(struct gapcoin_rpc *rpc, struct gapcoin_work *work) {
    if (!rpc || !work) return -1;

    json_t *result = gapcoin_rpc_call(rpc, "getwork", NULL, 10L);
    if (!result) return -1;

    json_t *data_value = json_object_get(result, "data");
    json_t *target_value = json_object_get(result, "difficulty");
    const char *data = json_string_value(data_value);
    const size_t required_hex = (GAPCOIN_WORK_HEADER_SIZE + 2U) * 2U;

    if (!data || !json_is_integer(target_value) ||
        strlen(data) < required_hex) {
        fprintf(stderr, "[RPC] getwork did not return a full Gapcoin header\n");
        json_decref(result);
        return -1;
    }

    memset(work, 0, sizeof(*work));
    for (size_t index = 0; index < GAPCOIN_WORK_HEADER_SIZE + 2U; index++) {
        int high = hex_nibble(data[index * 2U]);
        int low = hex_nibble(data[index * 2U + 1U]);
        if (high < 0 || low < 0) {
            fprintf(stderr, "[RPC] getwork returned malformed hex data\n");
            json_decref(result);
            return -1;
        }

        uint8_t value = (uint8_t)((high << 4) | low);
        if (index < GAPCOIN_WORK_HEADER_SIZE) {
            work->header[index] = value;
        } else if (index == GAPCOIN_WORK_HEADER_SIZE) {
            work->shift = value;
        } else {
            work->shift |= (uint16_t)value << 8;
        }
    }

    work->target = (uint64_t)json_integer_value(target_value);
    work->nonce = (uint32_t)work->header[80] |
                  ((uint32_t)work->header[81] << 8) |
                  ((uint32_t)work->header[82] << 16) |
                  ((uint32_t)work->header[83] << 24);
    json_decref(result);

    if (work->shift == 0 || work->target == 0) {
        fprintf(stderr, "[RPC] getwork returned an invalid shift or target\n");
        return -1;
    }

    return 0;
}

struct block_template *gapcoin_rpc_get_block_template(struct gapcoin_rpc *rpc) {
    if (!rpc) return NULL;
    
    json_t *params = json_array();
    json_t *rules = json_object();
    json_object_set_new(rules, "rules", json_array());
    json_array_append_new(params, rules);
    
    json_t *result = gapcoin_rpc_call(rpc, "getblocktemplate", params, 10L);
    if (!result) return NULL;
    
    struct block_template *tmpl = 
        (struct block_template *)malloc(sizeof(struct block_template));
    if (!tmpl) {
        json_decref(result);
        return NULL;
    }
    
    memset(tmpl, 0, sizeof(*tmpl));
    
    const char *prev_hash = 
        json_string_value(json_object_get(result, "previousblockhash"));
    if (prev_hash) {
        tmpl->previousblockhash = (char *)malloc(strlen(prev_hash) + 1);
        strcpy(tmpl->previousblockhash, prev_hash);
    }
    
    tmpl->height = json_integer_value(json_object_get(result, "height"));
    tmpl->curtime = json_integer_value(json_object_get(result, "curtime"));
    
    const char *difficulty_str =
        json_string_value(json_object_get(result, "difficulty"));
    const char *bits_str = json_string_value(json_object_get(result, "bits"));
    const char *encoded_difficulty = difficulty_str ? difficulty_str : bits_str;
    if (encoded_difficulty) {
        tmpl->bits = (char *)malloc(strlen(encoded_difficulty) + 1);
        if (!tmpl->bits) {
            block_template_free(tmpl);
            json_decref(result);
            return NULL;
        }
        strcpy(tmpl->bits, encoded_difficulty);
        tmpl->difficulty = strtoull(encoded_difficulty, NULL, 16);
    }
    
    tmpl->coinbasevalue = 
        json_integer_value(json_object_get(result, "coinbasevalue"));

    json_t *coinbaseaux = json_object_get(result, "coinbaseaux");
    const char *coinbase_flags = coinbaseaux ?
        json_string_value(json_object_get(coinbaseaux, "flags")) : NULL;
    if (coinbase_flags) {
        tmpl->coinbaseaux = (char *)malloc(strlen(coinbase_flags) + 1);
        if (!tmpl->coinbaseaux) {
            block_template_free(tmpl);
            json_decref(result);
            return NULL;
        }
        strcpy(tmpl->coinbaseaux, coinbase_flags);
    }

    json_t *transactions = json_object_get(result, "transactions");
    if (transactions && json_is_array(transactions)) {
        tmpl->transaction_count = json_array_size(transactions);
        if (tmpl->transaction_count > 0) {
            tmpl->transaction_hashes = calloc(
                tmpl->transaction_count, sizeof(*tmpl->transaction_hashes));
            tmpl->transaction_data = calloc(
                tmpl->transaction_count, sizeof(*tmpl->transaction_data));
            if (!tmpl->transaction_hashes || !tmpl->transaction_data) {
                block_template_free(tmpl);
                json_decref(result);
                return NULL;
            }

            for (size_t index = 0; index < tmpl->transaction_count; index++) {
                json_t *transaction = json_array_get(transactions, index);
                const char *txid = transaction ?
                    json_string_value(json_object_get(transaction, "txid")) : NULL;
                const char *data = transaction ?
                    json_string_value(json_object_get(transaction, "data")) : NULL;
                if (!txid || strlen(txid) != 64 || !data || strlen(data) == 0 ||
                    strlen(data) % 2 != 0) {
                    block_template_free(tmpl);
                    json_decref(result);
                    return NULL;
                }

                tmpl->transaction_hashes[index] = (char *)malloc(strlen(txid) + 1);
                tmpl->transaction_data[index] = (char *)malloc(strlen(data) + 1);
                if (!tmpl->transaction_hashes[index] || !tmpl->transaction_data[index]) {
                    block_template_free(tmpl);
                    json_decref(result);
                    return NULL;
                }
                strcpy(tmpl->transaction_hashes[index], txid);
                strcpy(tmpl->transaction_data[index], data);
            }
        }
    }
    
    const char *longpoll_id = 
        json_string_value(json_object_get(result, "longpollid"));
    if (longpoll_id) {
        tmpl->longpollid = (char *)malloc(strlen(longpoll_id) + 1);
        strcpy(tmpl->longpollid, longpoll_id);
    }
    
    tmpl->version = json_integer_value(json_object_get(result, "version"));
    
    json_decref(result);
    return tmpl;
}

struct block_template *gapcoin_rpc_get_block_template_longpoll(
    struct gapcoin_rpc *rpc,
    const char *longpoll_id) {
    if (!rpc) return NULL;
    
    json_t *params = json_array();
    json_t *rules = json_object();
    json_object_set_new(rules, "rules", json_array());
    json_array_append_new(params, rules);
    
    /* Add longpollid for blocking call */
    if (longpoll_id) {
        json_object_set_new(rules, "longpollid", json_string(longpoll_id));
    }
    
    /* Long timeout: the wallet can block up to 30 seconds. */
    json_t *result = gapcoin_rpc_call(rpc, "getblocktemplate", params, 60L);
    
    if (!result) return NULL;
    
    struct block_template *tmpl = 
        (struct block_template *)malloc(sizeof(struct block_template));
    if (!tmpl) {
        json_decref(result);
        return NULL;
    }
    
    memset(tmpl, 0, sizeof(*tmpl));
    
    const char *prev_hash = 
        json_string_value(json_object_get(result, "previousblockhash"));
    if (prev_hash) {
        tmpl->previousblockhash = (char *)malloc(strlen(prev_hash) + 1);
        strcpy(tmpl->previousblockhash, prev_hash);
    }
    
    tmpl->height = json_integer_value(json_object_get(result, "height"));
    tmpl->curtime = json_integer_value(json_object_get(result, "curtime"));
    
    const char *bits_str = json_string_value(json_object_get(result, "bits"));
    if (bits_str) {
        tmpl->bits = (char *)malloc(strlen(bits_str) + 1);
        strcpy(tmpl->bits, bits_str);
    }
    
    tmpl->coinbasevalue = 
        json_integer_value(json_object_get(result, "coinbasevalue"));
    
    const char *new_longpoll_id = 
        json_string_value(json_object_get(result, "longpollid"));
    if (new_longpoll_id) {
        tmpl->longpollid = (char *)malloc(strlen(new_longpoll_id) + 1);
        strcpy(tmpl->longpollid, new_longpoll_id);
    }
    
    tmpl->version = json_integer_value(json_object_get(result, "version"));
    
    json_decref(result);
    return tmpl;
}

int gapcoin_rpc_submit_gap(struct gapcoin_rpc *rpc, 
                          const struct gap_info *gap) {
    if (!rpc || !gap) return -1;
    
    /* Gapcoin expects gap submission with specific parameters */
    /* Format: submitgap height shift adder offset_p1 offset_p2 */
    
    json_t *params = json_array();
    json_array_append_new(params, json_integer(gap->height));
    json_array_append_new(params, json_integer(gap->shift));
    json_array_append_new(params, json_integer(gap->adder));
    json_array_append_new(params, json_integer(gap->offset_p1));
    json_array_append_new(params, json_integer(gap->offset_p2));
    
    json_t *result = gapcoin_rpc_call(rpc, "submitgap", params, 10L);
    
    if (!result) {
        printf("[RPC] submitgap failed (RPC error)\n");
        return -1;
    }
    
    /* Check result */
    int ret = 0;
    if (json_is_true(result)) {
        printf("[RPC] Gap submitted: height=%u shift=%u gap=%u merit=%.2f\n",
               gap->height, gap->shift, gap->gap_length, gap->merit);
        ret = 0;
    } else if (json_is_object(result)) {
        const char *error = json_string_value(json_object_get(result, "error"));
        if (error) {
            printf("[RPC] Gap submission error: %s\n", error);
        }
        ret = -1;
    } else {
        ret = -1;
    }
    
    json_decref(result);
    return ret;
}

int gapcoin_rpc_submit_block(struct gapcoin_rpc *rpc, 
                            const char *block_hex) {
    if (!rpc || !block_hex) return -1;
    
    json_t *params = json_array();
    json_array_append_new(params, json_string(block_hex));
    
    /* Generous timeout: the node verifies the gap against large-shift
       (e.g. 1254-bit) bignums, which can take much longer than 10 s. */
    json_t *result = gapcoin_rpc_call(rpc, "submitblock", params, 60L);
    
    if (!result) return -1;
    
    int ret = json_is_null(result) ? 0 : -1;
    if (ret != 0) {
        const char *reason = json_is_string(result)
                                 ? json_string_value(result)
                                 : NULL;
        fprintf(stderr, "[RPC] submitblock rejected: %s\n",
                reason ? reason : "unknown");
    }
    json_decref(result);
    
    return ret;
}

void gapcoin_rpc_free(struct gapcoin_rpc *rpc) {
    if (!rpc) return;
    
    free(rpc->host);
    free(rpc->username);
    free(rpc->password);
    free(rpc);
    
    printf("[RPC] Disconnected\n");
}

void block_template_free(struct block_template *tmpl) {
    if (!tmpl) return;
    
    free(tmpl->previousblockhash);
    free(tmpl->bits);
    free(tmpl->longpollid);
    free(tmpl->coinbaseaux);
    for (size_t index = 0; index < tmpl->transaction_count; index++) {
        if (tmpl->transaction_hashes) free(tmpl->transaction_hashes[index]);
        if (tmpl->transaction_data) free(tmpl->transaction_data[index]);
    }
    free(tmpl->transaction_hashes);
    free(tmpl->transaction_data);
    free(tmpl);
}

void mining_info_free(struct mining_info *info) {
    if (!info) return;
    
    free(info->next.bits);
    free(info);
}
