#ifndef VOLVOXAI_WEBGPU_DOMAIN_H
#define VOLVOXAI_WEBGPU_DOMAIN_H
#include <stdint.h>
#include <stddef.h>
typedef struct VxEngineState VxEngineState;
typedef struct {
    const char* name;
    uint32_t rank;
    int64_t maximum[8];
} VxWebGpuTensorBound;
/* The caller proves invariant broadcast/feature relationships across the
 * whole domain. This checks the monotonic maximum descriptor/device demand. */
int vx_webgpu_prove_node_domain(VxEngineState* state, const char* output_name,
    const VxWebGpuTensorBound* bounds, size_t count);
int vx_webgpu_decode_feedback(const char* token_name, const char* keep_name, const char* output_name, int position);
int vx_webgpu_transfer_rows(void* full, size_t full_bytes, size_t row_bytes,
    const int* indices, int count, void* packed, int scatter, int upload_packed);
void vx_webgpu_release_storage(void* host);
int vx_webgpu_node_row_compatible(const char* output_name, int row);
#if VOLVOXAI_ENABLE_TRAINING && VOLVOXAI_ENABLE_WEBGPU
/* Shared C RNG semantics used by both forward and backward plans. */
int vx_webgpu_training_rng(int node_index, int attention, uint32_t words[4]);
/* Private numerical command transport. Shader identities and binding layouts
 * come from generated WGSL metadata; the C training planner owns operands. */
int vx_webgpu_training_dispatch(const char* shader, const char* entry,
    void* const* hosts, const size_t* bytes, int count,
    const uint32_t groups[3], int validate);
void vx_webgpu_training_clear(void);
#endif
#endif
