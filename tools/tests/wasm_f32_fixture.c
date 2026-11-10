/* Developer-only roots; none are exported by release profiles. */
#include "compiled_weight_resources.h"
#include "resource_sampling.h"
#include <stdlib.h>
#include <stdint.h>

static VxPreparedF32Weights* owners[2];
static int32_t pack_allocations_before_failure = -1;

void test_f32_fail_after(int32_t count) { pack_allocations_before_failure = count; }
void* test_f32_pack_malloc(size_t bytes) {
    if (pack_allocations_before_failure == 0) return NULL;
    if (pack_allocations_before_failure > 0) pack_allocations_before_failure--;
    return malloc(bytes);
}
void* test_f32_pack_calloc(size_t count, size_t bytes) {
    if (pack_allocations_before_failure == 0) return NULL;
    if (pack_allocations_before_failure > 0) pack_allocations_before_failure--;
    return calloc(count, bytes);
}
uint32_t test_f32_bound_add(uint32_t used, uint32_t k, uint32_t n) {
    return (uint32_t)vx_prepared_f32_weights_bound_add(used, k, n);
}

void* test_f32_alloc(uint32_t bytes) { return malloc(bytes); }
void test_f32_free(void* data) { free(data); }
const float* test_f32_prepare(uint32_t owner, const float* source,
        uint32_t k, uint32_t n, int out_in) {
    return owner < 2 ? vx_prepared_f32_weights_add(&owners[owner], source, k, n, out_in) : NULL;
}
const float* test_f32_find(uint32_t owner, const float* source,
        uint32_t k, uint32_t n, int out_in) {
    return owner < 2 ? vx_prepared_f32_weights_find(owners[owner], source, k, n, out_in) : NULL;
}
void test_f32_clear(uint32_t owner) {
    if (owner < 2) vx_prepared_f32_weights_clear(&owners[owner]);
}
uint32_t test_f32_bytes(uint32_t owner) {
    return owner < 2 && owners[owner] ? (uint32_t)owners[owner]->allocated_bytes : 0;
}
uint32_t test_f32_live_bytes(void) {
    VxWasmMemorySample sample;
    vx_wasm_memory_sample(&sample);
    return (uint32_t)sample.allocated;
}
