/*
 * Minimal out-of-tree-style backend built from public headers only.
 *
 * It handles exact-shape F32 Add on the host.  A device backend can keep the
 * same registration and operand-discovery code, replace the loop with driver calls,
 * and add the device-resident coherence hooks described in
 * docs/backend-sdk.md.
 */

#include "host_backend.h"
#include "volvoxai_backend.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t handled_nodes;
} ExampleHostBackend;

static ExampleHostBackend g_example_host;

static int example_init(void* user_data) {
    ExampleHostBackend* backend = (ExampleHostBackend*)user_data;
    backend->handled_nodes = 0;
    return VX_INIT_READY;
}

static int example_supports(void* user_data, const VxNode* node) {
    (void)user_data;
    const VxTensor* left;
    const VxTensor* right;
    VxTensor* output;
    int32_t rank;
    if (!vx_node_op(node) || strcmp(vx_node_op(node), "Add") ||
        vx_node_attr_bool(node, "relu", 0) || vx_node_input_count(node) != 2 ||
        vx_node_output_count(node) != 1) return VX_DECLINED;
    left = vx_node_input_by_key(node, "a");
    right = vx_node_input_by_key(node, "b");
    output = vx_node_output_by_key(node, "out");
    if (!left || !right || !output || vx_tensor_dtype(left) != VX_DTYPE_F32 ||
        vx_tensor_dtype(right) != VX_DTYPE_F32 ||
        vx_tensor_dtype(output) != VX_DTYPE_F32) return VX_DECLINED;
    rank = vx_tensor_ndim(left);
    if (rank <= 0 || vx_tensor_ndim(right) != rank || vx_tensor_ndim(output) != rank ||
        vx_tensor_numel(left) <= 0 || vx_tensor_numel(right) != vx_tensor_numel(left) ||
        vx_tensor_numel(output) != vx_tensor_numel(left)) return VX_DECLINED;
    for (int32_t axis = 0; axis < rank; axis++) {
        if (!vx_tensor_shape(left) || !vx_tensor_shape(right) ||
            !vx_tensor_shape(output) || vx_tensor_shape(left)[axis] <= 0 ||
            vx_tensor_shape(right)[axis] != vx_tensor_shape(left)[axis] ||
            vx_tensor_shape(output)[axis] != vx_tensor_shape(left)[axis])
            return VX_DECLINED;
    }
    return VX_HANDLED;
}

static int example_run(void* user_data, const VxNode* node) {
    ExampleHostBackend* backend = (ExampleHostBackend*)user_data;
    const VxTensor* left = vx_node_input_by_key(node, "a");
    const VxTensor* right = vx_node_input_by_key(node, "b");
    VxTensor* output = vx_node_output_by_key(node, "out");
    int64_t elements;
    if (example_supports(user_data, node) != VX_HANDLED) return VX_DECLINED;
    elements = vx_tensor_numel(left);
    if (elements < 0 || (uint64_t)elements > SIZE_MAX / sizeof(float)) return VX_ERROR;
    if (vx_tensor_sync_host(left) != 0 || vx_tensor_sync_host(right) != 0 ||
        !vx_tensor_cdata(left) || !vx_tensor_cdata(right) || !vx_tensor_data(output))
        return VX_ERROR;
    const float* left_data = (const float*)vx_tensor_cdata(left);
    const float* right_data = (const float*)vx_tensor_cdata(right);
    float* output_data = (float*)vx_tensor_data(output);
    for (int64_t index = 0; index < elements; index++)
        output_data[index] = left_data[index] + right_data[index];
    backend->handled_nodes++;
    return VX_HANDLED;
}

static void example_teardown(void* user_data) { (void)user_data; }

int volvoxai_example_host_backend_register(void) {
    const VxBackendV1 backend = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "example-host",
        .user_data = &g_example_host,
        .init = example_init,
        .supports = example_supports,
        .run = example_run,
        .teardown = example_teardown,
    };
    return volvoxai_register_backend(&backend);
}

uint64_t volvoxai_example_host_backend_handled_nodes(void) {
    return g_example_host.handled_nodes;
}
