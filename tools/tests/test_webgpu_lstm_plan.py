"""LSTM optional inputs must not resize an existing WebGPU input span."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class WebGpuLstmPlanTests(unittest.TestCase):
    def test_optional_bindings_keep_distinct_span_extents(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

enum { T_F32 = 1 };
enum {
    VX_PORT_WEIGHT = 0,
    VX_PORT_RECURRENT_WEIGHT,
    VX_PORT_BIAS,
    VX_PORT_INITIAL_H,
    VX_PORT_INITIAL_C,
    VX_PORT_SCALE,
    VX_PORT_RECURRENT_SCALE,
    VX_PORT_COUNT
};
enum {
    VX_NODE_PARAM_DIRECTION = 1,
    VX_NODE_PARAM_GROUP_SIZE,
    VX_NODE_PARAM_WEIGHT_BITS,
    VX_SHADER_INFERENCE_LSTM
};

typedef struct { int ndim, shape[8], dtype; float* data; } T;
typedef struct {
    T* ports[16];
    int direction;
    int group_size;
    int weight_bits;
} Node;
typedef struct { const void* data; size_t bytes; } Binding;
typedef struct { unsigned shader_id, workgroup[3]; } Variant;
typedef struct {
    Binding bindings[8];
    unsigned count, variant_count;
    Variant variants[1];
    uint32_t params[8];
} VxWebGpuNodePlan;

static const Node* g_n = NULL;
static size_t g_node_capacity = 0;
static float** g_qwcache = NULL;

static int vx_packed_group_size_supported(int group_size) {
    return group_size == 32 || group_size == 64 || group_size == 128;
}
static T* vx_webgpu_port(const Node* node, int port) {
    assert(port >= 0 && port < 16);
    return node->ports[port];
}
static int vx_node_param_i32(const Node* node, int key, int fallback) {
    if (key == VX_NODE_PARAM_DIRECTION) return node->direction;
    if (key == VX_NODE_PARAM_GROUP_SIZE) return node->group_size;
    if (key == VX_NODE_PARAM_WEIGHT_BITS) return node->weight_bits;
    return fallback;
}
static int vx_webgpu_bind(VxWebGpuNodePlan* plan, const void* data, size_t bytes) {
    /* The bridge owns one buffer extent per host address. A second extent at
       that address would replace the first buffer and invalidate its binding. */
    for (unsigned i = 0; i < plan->count; i++)
        if (plan->bindings[i].data == data && plan->bindings[i].bytes != bytes) return 0;
    assert(plan->count < 8);
    plan->bindings[plan->count++] = (Binding){data, bytes};
    return 1;
}
int vx_webgpu_bind_scratch(VxWebGpuNodePlan* plan, const void* data, size_t words) {
    static float scratch[8];
    assert(words == 1 && plan->count < 8);
    memcpy(scratch + plan->count, data, 4);
    return vx_webgpu_bind(plan, scratch + plan->count, 4);
}
static int vx_webgpu_params(VxWebGpuNodePlan* plan, const uint32_t* words, size_t count) {
    assert(count == 8);
    memcpy(plan->params, words, 32);
    return 1;
}

#include "webgpu_lstm_plan.inc"

int main(void) {
    float x[30], w[140], r[196], bias[56], h[14], c[14], y[42];
    T input = {3, {3, 2, 5}, T_F32, x};
    T weight = {3, {1, 28, 5}, T_F32, w};
    T recurrent = {3, {1, 28, 7}, T_F32, r};
    T b = {2, {1, 56}, T_F32, bias};
    T initial_h = {3, {1, 2, 7}, T_F32, h};
    T initial_c = {3, {1, 2, 7}, T_F32, c};
    T output = {4, {3, 1, 2, 7}, T_F32, y};
    Node node = {0};
    node.ports[VX_PORT_WEIGHT] = &weight;
    node.ports[VX_PORT_RECURRENT_WEIGHT] = &recurrent;
    for (unsigned optional = 0; optional < 3; optional++) {
        VxWebGpuNodePlan plan = {0};
        node.ports[VX_PORT_BIAS] = optional ? &b : NULL;
        node.ports[VX_PORT_INITIAL_H] = optional == 2 ? &initial_h : NULL;
        node.ports[VX_PORT_INITIAL_C] = optional == 2 ? &initial_c : NULL;
        assert(vx_webgpu_plan_lstm(&node, &input, &output, &plan));
        assert(plan.count == 7 && plan.bindings[0].data == x && plan.bindings[0].bytes == sizeof(x));
        assert(plan.params[5] == (optional == 0 ? 0u : optional == 1 ? 1u : 7u));
        assert(plan.variants[0].workgroup[0] == 2 && plan.variant_count == 1);
        for (unsigned i = 3; i < 6; i++) assert(plan.bindings[i].data != x);
    }

    /* Test vx_webgpu_plan_packed_lstm with real execution, non-zero codes, and cache verification */
    uint32_t raw_w[28];
    uint32_t raw_r[28];
    for (int i = 0; i < 28; i++) {
        /* Nibbles: 9, 10, 7, 6, 11 -> codes produce 1.0f, 2.0f, -1.0f, -2.0f, 3.0f; tail=0x888 */
        raw_w[i] = 0x888b67a9u;
        /* Nibbles: 10, 9, 8, 7, 6, 11, 12 -> codes produce 4.0f, 2.0f, 0.0f, -2.0f, -4.0f, 6.0f, 8.0f; tail=0x8 */
        raw_r[i] = 0x8cb6789au;
    }
    float w_scales[28];
    float r_scales[28];
    for (int i = 0; i < 28; i++) {
        w_scales[i] = 1.0f;
        r_scales[i] = 2.0f;
    }
    T packed_w = {3, {1, 28, 5}, T_F32, (float*)raw_w};
    T packed_r = {3, {1, 28, 7}, T_F32, (float*)raw_r};
    T w_scale_t = {2, {28, 1}, T_F32, w_scales};
    T r_scale_t = {2, {28, 1}, T_F32, r_scales};
    Node packed_node = {0};
    packed_node.direction = 0;
    packed_node.group_size = 32;
    packed_node.weight_bits = 4;
    packed_node.ports[VX_PORT_WEIGHT] = &packed_w;
    packed_node.ports[VX_PORT_SCALE] = &w_scale_t;
    packed_node.ports[VX_PORT_RECURRENT_WEIGHT] = &packed_r;
    packed_node.ports[VX_PORT_RECURRENT_SCALE] = &r_scale_t;
    packed_node.ports[VX_PORT_BIAS] = &b;
    packed_node.ports[VX_PORT_INITIAL_H] = &initial_h;
    packed_node.ports[VX_PORT_INITIAL_C] = &initial_c;

    /* Set up 1-slot g_qwcache array to verify caching and avoid leaks */
    float* cache_slot = NULL;
    g_n = &packed_node;
    g_node_capacity = 1;
    g_qwcache = &cache_slot;

    VxWebGpuNodePlan packed_plan = {0};
    assert(vx_webgpu_plan_packed_lstm(&packed_node, &input, &output, &packed_plan));
    assert(packed_plan.count == 7);
    assert(packed_plan.bindings[0].data == x && packed_plan.bindings[0].bytes == sizeof(x));
    assert(packed_plan.bindings[1].data == cache_slot && packed_plan.bindings[1].bytes == 140 * sizeof(float));
    assert(packed_plan.bindings[2].data == (cache_slot + 140) && packed_plan.bindings[2].bytes == 196 * sizeof(float));
    assert(packed_plan.params[0] == 3u && packed_plan.params[1] == 2u);
    assert(packed_plan.params[2] == 5u && packed_plan.params[3] == 7u);
    assert(packed_plan.params[5] == 7u);
    assert(packed_plan.variant_count == 1);
    assert(packed_plan.variants[0].workgroup[0] == 2);

    /* Verify non-zero unpacked float values */
    const float* uw = (const float*)packed_plan.bindings[1].data;
    const float* ur = (const float*)packed_plan.bindings[2].data;
    for (int row = 0; row < 28; row++) {
        assert(uw[row * 5 + 0] == 1.0f);
        assert(uw[row * 5 + 1] == 2.0f);
        assert(uw[row * 5 + 2] == -1.0f);
        assert(uw[row * 5 + 3] == -2.0f);
        assert(uw[row * 5 + 4] == 3.0f);
        assert(ur[row * 7 + 0] == 4.0f);
        assert(ur[row * 7 + 1] == 2.0f);
        assert(ur[row * 7 + 2] == 0.0f);
        assert(ur[row * 7 + 3] == -2.0f);
        assert(ur[row * 7 + 4] == -4.0f);
        assert(ur[row * 7 + 5] == 6.0f);
        assert(ur[row * 7 + 6] == 8.0f);
    }

    /* Verify cache reuse on second planning call */
    VxWebGpuNodePlan plan_cached = {0};
    assert(vx_webgpu_plan_packed_lstm(&packed_node, &input, &output, &plan_cached));
    assert(plan_cached.bindings[1].data == cache_slot);
    assert(plan_cached.bindings[2].data == (cache_slot + 140));

    /* Clean up allocated cache memory to prevent leaks under ASan/LSan */
    free(cache_slot);
    cache_slot = NULL;
    g_n = NULL;
    g_node_capacity = 0;
    g_qwcache = NULL;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-webgpu-lstm-plan-") as directory:
            root = Path(directory)
            (root / "test.c").write_text(source)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "native/src/backends"), str(root / "test.c"), "-o", str(root / "test")], check=True)
            subprocess.run([str(root / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
