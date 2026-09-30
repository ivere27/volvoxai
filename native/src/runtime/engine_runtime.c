#include "engine_core.h"
#include "profiling.h"
#include "vx_platform.h"
#include "engine_internal.h"
#include "backend.h"
#include "backend_manager.h"
#include "w8a8_device_ops.h"
#include "attention_mask.h"
#include "sequence_runtime.h"
#if VOLVOXAI_ENABLE_WEBGPU
#include "webgpu_domain.h"
#endif
#if VOLVOXAI_ENABLE_TRAINING
#include "training/training_core.h"
#endif
#include "adapter_runtime_internal.h"
#include "cJSON.h"
#include "fusion_ops.h"
#include "inference_kernels.h"
#include "paged_attention.h"
#include "paged_binding.h"
#include "generated/kernel_registry.h"
#include "json_validation.h"
#include "quant_cpu_isa.h"
#include "safetensors.h"
#include "thread_pool.h"
#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#endif
#if VOLVOXAI_ENABLE_OPENGL
#include "opengl_engine.h"
#endif
#if VOLVOXAI_ENABLE_METAL
#include "metal_engine.h"
#endif
#if VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif
#include "conv_f32_isa.h"
#include "tensor_f32_isa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#if VOLVOXAI_ENABLE_VULKAN
extern int vk_matmul(const float*, const float*, const float*, float*, int, int, int);
#endif

double volvoxai_engine_now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

static T* nin(Node* n, VxPortKind key);
static int quantized_weight_i32(const T* wt, long idx);
static int widen_f16_tensor_to_f32(T* t);
static int linear_node_weight_layout(Node* node, const T* input, const T* output,
                                     const T* weight, int* out_in);
static int route_index_read(const T* indices, long index, int* value);
static int physical_shape_model_validate(Node* node, T* output);

/* These private fragments stay in one translation unit to preserve static
 * runtime state and the inference/full compilation boundary. */
#include "incremental_runtime.h"
#include "decode_projection.h"

#include "engine_runtime_model.inc"
#include "engine_runtime_f32_cpu.inc"
#include "engine_runtime_f32_gpu.inc"
#include "engine_runtime_w8a8.inc"
#include "engine_runtime_dispatch.inc"

/* Entered only from the instrumented outer route. Disabled/fused-away nodes
 * emit no fictitious duration. Recursive CPU dispatch remains uninstrumented. */
int vx_run_node_profiled(Node* node, int index, int last, int direct_cpu) {
    if (node->skip || node->disabled)
        return direct_cpu ? run_node_cpu_direct(node, index, last) : run_node(node, index, last);
#if VOLVOXAI_ENABLE_WEBGPU
    if (g_use_webgpu && validating) return run_node(node, index, last);
#endif
    VxTraceScope* trace = vx_engine_state_current()->profiling;
    VxTraceWork saved_work = trace->work;
    trace->work = (VxTraceWork){VX_TRACE_PHASE_FORWARD, index, node->fuse_relu6,
        vx_operator_kind_name(node->operator_kind), node->out, NULL};
    const char* name = vx_operator_kind_name(node->operator_kind);
    int device_token = -1;
    int external = vx_trace_external(trace), label_token = 0;
    if (external) {
        /* Vendor-tool ranges: NVTX on the host, a label in the device stream. */
        char label[160];
        vx_trace_external_push(trace, VX_TRACE_PHASE_FORWARD, index, name, node->out);
        vx_trace_external_label(label, sizeof(label), VX_TRACE_PHASE_FORWARD, index, name, node->out);
#if VOLVOXAI_ENABLE_VULKAN
        if (g_use_vulkan) {
            label_token = vk_annotate_begin(label);
            vx_trace_external_note(trace, VX_TRACE_EXTERNAL_VK_DEBUG_UTILS, label_token > 0 ? VX_OBSERVATION_AVAILABLE :
                label_token < 0 ? VX_OBSERVATION_FAILED : VX_OBSERVATION_UNSUPPORTED, label_token > 0);
        }
#endif
#if VOLVOXAI_ENABLE_OPENGL
        if (g_use_opengl) {
            label_token = opengl_annotate_begin(label);
            vx_trace_external_note(trace, VX_TRACE_EXTERNAL_KHR_DEBUG,
                label_token > 0 ? VX_OBSERVATION_AVAILABLE : VX_OBSERVATION_UNSUPPORTED, label_token > 0);
        }
#endif
#if VOLVOXAI_ENABLE_WEBGPU
        if (g_use_webgpu) {
            label_token = vx_webgpu_annotate_begin(label);
            vx_trace_external_note(trace, VX_TRACE_EXTERNAL_WEBGPU_DEBUG_GROUP,
                label_token > 0 ? VX_OBSERVATION_AVAILABLE : VX_OBSERVATION_UNSUPPORTED, label_token > 0);
        }
#endif
    }
#if VOLVOXAI_ENABLE_OPENGL
    if (g_use_opengl) device_token = opengl_trace_node_begin(index, name, node->out, node->fuse_relu6);
#endif
#if VOLVOXAI_ENABLE_VULKAN
    if (g_use_vulkan) device_token = vk_trace_node_begin(index, name, node->out, node->fuse_relu6);
#endif
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda) device_token = cuda_trace_node_begin(index, name, node->out, node->fuse_relu6);
#endif
#if VOLVOXAI_ENABLE_WEBGPU
    if (g_use_webgpu && !validating) vx_webgpu_trace_node_begin(index, name, node->out, node->fuse_relu6);
#endif
    uint64_t start = vx_trace_now_ns();
    int result = direct_cpu ? run_node_cpu_direct(node, index, last) : run_node(node, index, last);
#if VOLVOXAI_ENABLE_OPENGL
    if (g_use_opengl) opengl_trace_node_end(device_token);
#endif
#if VOLVOXAI_ENABLE_VULKAN
    if (g_use_vulkan) vk_trace_node_end(device_token);
#endif
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda) cuda_trace_node_end(device_token);
#endif
#if VOLVOXAI_ENABLE_WEBGPU
    if (g_use_webgpu && !validating) vx_webgpu_trace_node_end();
#endif
    if (external) {
#if VOLVOXAI_ENABLE_VULKAN
        if (g_use_vulkan) vk_annotate_end(label_token);
#endif
#if VOLVOXAI_ENABLE_OPENGL
        if (g_use_opengl) opengl_annotate_end(label_token);
#endif
#if VOLVOXAI_ENABLE_WEBGPU
        if (g_use_webgpu) vx_webgpu_annotate_end(label_token);
#endif
        vx_trace_external_pop(trace);
    }
    (void)device_token; (void)label_token;
    vx_trace_node(vx_engine_state_current()->profiling, start, index,
        vx_operator_kind_name(node->operator_kind), node->out, node->fuse_relu6);
    trace->work = saved_work;
    return result;
}
