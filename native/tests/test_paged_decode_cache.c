#include "paged_kv.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Transaction ownership is tested independently of device command encoding. */
int main(void) {
    VxPagedKVOptions options = {.lanes = 3, .page_tokens = 2,
        .lane_token_capacity = 8, .max_pages = 6, .bytes_per_token = 16};
    VxPagedKVCache* cache = vx_paged_kv_create(&options);
    assert(cache);
    assert(vx_paged_kv_append(cache, 0, 4) == VX_PAGED_KV_OK);
    char key[513]; memset(key, 'p', sizeof(key) - 1); key[sizeof(key) - 1] = 0;
    assert(vx_paged_kv_publish_prefix(cache, key, 0, 4) == VX_PAGED_KV_OK);
    int tokens;
    assert(vx_paged_kv_acquire_prefix(cache, key, 1, &tokens) == VX_PAGED_KV_OK && tokens == 4);
    int shared = vx_paged_kv_physical_page(cache, 1, 1);
    VxPagedKVReservation transaction[2] = {0};
    assert(vx_paged_kv_reserve_write(cache, 1, 0, 3, &transaction[0]) == VX_PAGED_KV_OK);
    int copied = vx_paged_kv_physical_page(cache, 1, 1);
    assert(copied != shared && transaction[0].prior_pages[0] == shared);
    assert(vx_paged_kv_reserve_write(cache, 2, 3, 0, &transaction[1]) == VX_PAGED_KV_OK);
    assert(vx_paged_kv_lengths(cache)[2] == 0);
    assert(vx_paged_kv_rollback_batch(cache, transaction, 2) == VX_PAGED_KV_OK);
    assert(vx_paged_kv_physical_page(cache, 1, 1) == shared);
    assert(vx_paged_kv_physical_page(cache, 2, 0) == VX_PAGED_KV_UNMAPPED);
    VxPagedKVTelemetry telemetry;
    vx_paged_kv_telemetry(cache, &telemetry);
    assert(telemetry.copy_on_writes == 0 && telemetry.reserved_pages == 0);
    assert(telemetry.resident_pages == 2 && telemetry.free_pages == 4);
    assert(vx_paged_kv_reserve_write(cache, 1, 0, 3, &transaction[0]) == VX_PAGED_KV_OK);
    assert(vx_paged_kv_reserve_write(cache, 2, 3, 0, &transaction[1]) == VX_PAGED_KV_OK);
    assert(vx_paged_kv_commit_batch(cache, transaction, 2) == VX_PAGED_KV_OK);
    assert(vx_paged_kv_commit_batch(cache, transaction, 2) == VX_PAGED_KV_INVALID_ARGUMENT);
    assert(vx_paged_kv_lengths(cache)[2] == 3);
    assert(vx_paged_kv_physical_page(cache, 0, 1) == shared);
    assert(vx_paged_kv_physical_page(cache, 1, 1) == copied);
    vx_paged_kv_telemetry(cache, &telemetry);
    assert(telemetry.copy_on_writes == 1 && telemetry.reserved_pages == 0);
    assert(telemetry.resident_pages == 5 && telemetry.free_pages == 1);
    assert(vx_paged_kv_reserve_write(cache, 1, 4, 4, &transaction[0]) == VX_PAGED_KV_CAPACITY_EXHAUSTED);
    assert(vx_paged_kv_lengths(cache)[1] == 4);
    assert(vx_paged_kv_physical_page(cache, 1, 2) == VX_PAGED_KV_UNMAPPED);
    assert(vx_paged_kv_physical_page(cache, 1, 1) == copied);
    for (int lane = 0; lane < 3; lane++) assert(vx_paged_kv_release_lane(cache, lane) == VX_PAGED_KV_OK);
    assert(vx_paged_kv_evict(cache, 6) == 2);
    assert(!vx_paged_kv_has_prefix(cache, key));
    vx_paged_kv_telemetry(cache, &telemetry);
    assert(telemetry.resident_pages == 0 && telemetry.free_pages == 6);
    vx_paged_kv_destroy(cache);
    puts("paged COW reservation, commit, rollback, capacity and retirement: PASS");
    return 0;
}
