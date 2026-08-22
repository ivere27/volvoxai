#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "metal_engine.h"
#include "runtime_state.h"
#include "shader_store.h"
#include "batch_matmul_f32_plan.h"
#include "expand_f32_plan.h"
#include "qbatch_matmul_plan.h"
#include "qlinear_multiplier.h"
#include "typed_control_plan.h"

#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define METAL_SHADER_DIR "metal"
#define METAL_GRAPH_MAX_TENSORS 4096
#define METAL_MAX_BINDINGS 8
/*
 * The offset alignment a row window must clear.
 *
 * Metal has no runtime query for this the way Vulkan and OpenGL do; it is a
 * documented rule instead. Every windowed binding here is a naga-emitted
 * `device` pointer, whose offset must be a multiple of four bytes. The two
 * `constant` bindings a kernel takes -- Params and the buffer-size block -- are
 * the ones that would owe 256 on Intel Macs, and both are always bound at
 * offset zero, so that rule never reaches a row.
 *
 * Stated as a constant rather than a literal 4 because a future `constant`
 * activation binding would have to change it, and this is where the reasoning
 * that permits 4 is written down.
 */
#define METAL_GRAPH_ALIGN 4u
#define METAL_GRAPH_MAX_RETAINED_BINDINGS (METAL_GRAPH_MAX_TENSORS * METAL_MAX_BINDINGS)
#if VOLVOXAI_ENABLE_TRAINING
#define METAL_TRAINING_MAX_BINDINGS 16
#define METAL_TRAINING_MAX_KERNELS 64
#define METAL_TRAINING_MAX_TRANSIENTS 4096
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(objc_arc)
#define VX_METAL_RELEASE(obj) do { (obj) = nil; } while (0)
#define VX_METAL_RETAIN_ASSIGN(dst, src) do { (dst) = (src); } while (0)
#else
#define VX_METAL_RELEASE(obj) do { if ((obj)) { [(obj) release]; (obj) = nil; } } while (0)
#define VX_METAL_RETAIN_ASSIGN(dst, src) do { \
    if ((dst) != (src)) { \
        if ((dst)) [(dst) release]; \
        (dst) = (src) ? [(src) retain] : nil; \
    } \
} while (0)
#endif

typedef struct {
    const char* name;
    const char* path;
    int binding_count;
    int uniform_binding;
    int wg_x;
    int wg_y;
    int wg_z;
} MetalKernel;

typedef struct {
    const void* host;
    size_t bytes;
    size_t cap;
    size_t domain_capacity;
    id<MTLBuffer> buffer;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    int host_dirty;
    int device_dirty;
    int is_weight;
    int is_alias;
    int domain_span;
} MetalTensorSlot;

/* `offset` is last so that every whole-tensor binding stays spelled
 * `{buffer, bytes}` and means offset zero, which is what all but the decoder
 * closure want. Only a row window sets it. */
typedef struct {
    id<MTLBuffer> buffer;
    size_t bytes;
    size_t offset;
} MetalBinding;

#if VOLVOXAI_ENABLE_TRAINING
typedef struct {
    const char* name;
    int binding_count;
    int wg_x;
    int wg_y;
    int wg_z;
    uint32_t read_write_mask;
    const char* entries[5];
} MetalTrainingShaderDesc;

typedef struct {
    char shader[64];
    char entry[64];
    id<MTLComputePipelineState> pipeline;
} MetalTrainingKernel;

typedef struct {
    void* host;
    size_t bytes;
    size_t cap;
    id<MTLBuffer> buffer;
    MetalTensorSlot* graph_slot;
    int device_dirty;
    int is_weight;
} MetalTrainingTensorSlot;
#endif

/* A QConv2D without I32 bias must bind durable zero storage sized for its
 * output channels. Blocks remain valid through graph residency and release at
 * backend cleanup. */
typedef struct QConvZeroBiasBacking {
    int32_t* values;
    size_t elements;
    struct QConvZeroBiasBacking* next;
} QConvZeroBiasBacking;

static const int32_t* qconv_zero_bias_get(uint32_t output_channels);
static void qconv_zero_bias_release_state(void* opaque_state);

/* qSDPAInt8 always needs a conventional I32 storage binding for its mask. */
static const int32_t qsdpa_dummy_mask[1] = {0};

#if VOLVOXAI_ENABLE_TRAINING
static const MetalTrainingShaderDesc training_shader_descs[] = {
    {"activationBackward", 5, 64, 1, 1, 0x008u, {"main", NULL}},
    {"basicBackward", 6, 64, 1, 1, 0x018u, {"a_main", "b_main", NULL}},
    {"batchNorm2DBackward", 9, 64, 1, 1, 0x0e0u, {"input_main", "param_main", NULL}},
    {"groupNormBackward", 7, 64, 1, 1, 0x038u, {"input_main", "param_main", NULL}},
    {"concatBackward", 3, 64, 1, 1, 0x002u, {"main", NULL}},
    {"conv2DBackward", 8, 64, 1, 1, 0x070u,
        {"input_main", "weight_main", "bias_main", NULL}},
    {"copyBackward", 3, 64, 1, 1, 0x002u, {"main", NULL}},
    {"dropoutBackward", 3, 64, 1, 1, 0x002u, {"main", NULL}},
    {"reduceBackward", 3, 64, 1, 1, 0x002u, {"main", NULL}},
    {"crossSdpaBackward", 9, 1, 1, 1, 0x0e0u, {"q_main", "k_main", "v_main", NULL}},
    {"embeddingBackward", 4, 64, 1, 1, 0x004u, {"main", NULL}},
    {"layerNormBackward", 7, 64, 1, 1, 0x038u, {"input_main", "param_main", NULL}},
    {"matMulBackward", 7, 8, 8, 1, 0x038u,
        {"input_main", "weight_main", "bias_main", NULL}},
    {"moeLinearBackward", 13, 64, 1, 1, 0x3c0u,
        {"input_main", "weight_main", "bias_main", "route_main", NULL}},
    {"moeRouterBackward", 11, 64, 1, 1, 0x3c0u,
        {"logit_main", "input_main", "weight_main", "bias_main", NULL}},
    {"poolingBackward", 4, 64, 1, 1, 0x004u,
        {"global_average_main", "max_pool_main", NULL}},
    {"preluBackward", 6, 64, 1, 1, 0x018u, {"input_main", "weight_main", NULL}},
    {"resizeBackward", 3, 64, 1, 1, 0x002u, {"main", NULL}},
    {"rmsNormBackward", 6, 64, 1, 1, 0x018u, {"input_main", "weight_main", NULL}},
    {"sdpaBackward", 5, 1, 1, 1, 0x008u, {"main", NULL}},
    {"softmaxBackward", 4, 64, 1, 1, 0x004u, {"main", NULL}},
    {"splitBackward", 3, 64, 1, 1, 0x002u, {"main", NULL}},
    {"transposeBackward", 3, 64, 1, 1, 0x002u, {"main", NULL}},
};
#endif

#define METAL_MAX_PIPELINES 128

typedef struct {
    const MetalKernel* descriptor;
    id<MTLComputePipelineState> pipeline;
    int failed;
} MetalPipelineCacheEntry;

/* The default Metal device, command queue, and model-independent inference
 * pipeline cache form one physical submission domain. References are held by
 * engine contexts; initialization, compilation, and final teardown are
 * serialized by the named mutex. */
typedef struct {
    pthread_mutex_t mutex;
    unsigned reference_count;
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    MetalPipelineCacheEntry pipelines[METAL_MAX_PIPELINES];
    size_t pipeline_count;
} MetalDeviceState;

static MetalDeviceState g_metal_device_state = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};

typedef struct {
    MetalTensorSlot graph_slot_storage[METAL_GRAPH_MAX_TENSORS];
    int graph_slots_count;
    /* Inference dispatches are encoded into one command buffer per graph
     * forward. Every bound resource remains owned by this context until that
     * command completes. */
    id<MTLCommandBuffer> graph_command_buffer;
    id<MTLBuffer>
        graph_retained_binding_storage[METAL_GRAPH_MAX_RETAINED_BINDINGS];
    int graph_retained_bindings_count;
    int graph_is_forward_active;
    int graph_forward_failed;
#ifdef VOLVOX_METAL_TESTING
    uint64_t graph_dispatch_count;
    uint64_t graph_commit_count;
    uint64_t graph_wait_count;
#endif
    id<MTLBuffer> qgroupnorm_scratch_buffer;
    size_t qgroupnorm_scratch_capacity;
    id<MTLBuffer> qlayernorm_scratch_buffer;
    size_t qlayernorm_scratch_capacity;
    QConvZeroBiasBacking* qconv_zero_bias_storage;
    char* shape_signature;
    uint64_t shape_generation;
    uint64_t capacity_generation;
    size_t domain_span_count;
    size_t domain_qgroupnorm_stats_bytes;
    size_t domain_qlayernorm_stats_bytes;
    int domain_enforced;
#ifdef VOLVOX_METAL_TESTING
    int test_domain_allocation_failure_after;
#endif
#if VOLVOXAI_ENABLE_TRAINING
    int training_is_active;
    MetalTrainingKernel training_kernel_storage[METAL_TRAINING_MAX_KERNELS];
    int training_kernels_count;
    MetalTrainingTensorSlot training_slot_storage[METAL_GRAPH_MAX_TENSORS];
    int training_slots_count;
    id<MTLCommandBuffer> training_command_buffer;
    id<MTLComputeCommandEncoder> training_command_encoder;
    id<MTLBuffer> training_transient_storage[METAL_TRAINING_MAX_TRANSIENTS];
    int training_transients_count;
#endif
    int device_acquired;
} MetalContextState;

static MetalContextState* metal_context_state_get(int create);
static void metal_context_state_destroy(void* opaque_state);

#define device (g_metal_device_state.device)
#define commandQueue (g_metal_device_state.command_queue)
#define graph_slots (metal_context_state_get(0)->graph_slot_storage)
#define graph_slot_count (metal_context_state_get(0)->graph_slots_count)
#define graph_command (metal_context_state_get(0)->graph_command_buffer)
#define graph_retained_bindings \
    (metal_context_state_get(0)->graph_retained_binding_storage)
#define graph_retained_binding_count \
    (metal_context_state_get(0)->graph_retained_bindings_count)
#define graph_forward_active \
    (metal_context_state_get(0)->graph_is_forward_active)
#define graph_forward_error (metal_context_state_get(0)->graph_forward_failed)
#ifdef VOLVOX_METAL_TESTING
#define graph_debug_dispatch_count \
    (metal_context_state_get(0)->graph_dispatch_count)
#define graph_debug_commit_count (metal_context_state_get(0)->graph_commit_count)
#define graph_debug_wait_count (metal_context_state_get(0)->graph_wait_count)
#endif
#define qgroupnorm_stats_buffer \
    (metal_context_state_get(0)->qgroupnorm_scratch_buffer)
#define qgroupnorm_stats_capacity \
    (metal_context_state_get(0)->qgroupnorm_scratch_capacity)
#define qlayernorm_stats_buffer \
    (metal_context_state_get(0)->qlayernorm_scratch_buffer)
#define qlayernorm_stats_capacity \
    (metal_context_state_get(0)->qlayernorm_scratch_capacity)
#define qconv_zero_bias_backings \
    (metal_context_state_get(0)->qconv_zero_bias_storage)
#if VOLVOXAI_ENABLE_TRAINING
#define training_active (metal_context_state_get(0)->training_is_active)
#define training_kernels \
    (metal_context_state_get(0)->training_kernel_storage)
#define training_kernel_count \
    (metal_context_state_get(0)->training_kernels_count)
#define training_slots (metal_context_state_get(0)->training_slot_storage)
#define training_slot_count (metal_context_state_get(0)->training_slots_count)
#define training_command \
    (metal_context_state_get(0)->training_command_buffer)
#define training_encoder \
    (metal_context_state_get(0)->training_command_encoder)
#define training_transients \
    (metal_context_state_get(0)->training_transient_storage)
#define training_transient_count \
    (metal_context_state_get(0)->training_transients_count)
#endif

static const MetalKernel k_mul = {"mul", METAL_SHADER_DIR "/mul.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_sub = {"sub", METAL_SHADER_DIR "/sub.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_div = {"div", METAL_SHADER_DIR "/div.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_broadcast_binary = {"broadcastBinaryNative", METAL_SHADER_DIR "/broadcastBinaryNative.metal", 4, -1, 64, 1, 1};
static const MetalKernel k_split = {"split", METAL_SHADER_DIR "/split.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_conv1d = {"conv1D", METAL_SHADER_DIR "/conv1D.metal", 5, 4, 64, 1, 1};
static const MetalKernel k_embedding = {"embedding", METAL_SHADER_DIR "/embedding.metal", 4, 3, 64, 1, 1};
/* Bindings 0-5 storage, 6 uniform params, 7 the resident-slot table. */
static const MetalKernel k_moe_linear = {"moeLinear", METAL_SHADER_DIR "/moeLinear.metal", 8, 6, 64, 1, 1};
/* Bindings 0-4 storage, 5 uniform params. */
static const MetalKernel k_moe_router = {"moeRouter", METAL_SHADER_DIR "/moeRouter.metal", 6, 5, 64, 1, 1};
static const MetalKernel k_sdpa = {"sDPA", METAL_SHADER_DIR "/sDPA.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_cross_sdpa = {"crossSDPA", METAL_SHADER_DIR "/crossSDPA.metal", 6, 5, 64, 1, 1};
#if VOLVOXAI_ENABLE_TRAINING
static const MetalKernel k_sdpa_training = {"sdpaTraining", METAL_SHADER_DIR "/sdpaTraining.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_cross_sdpa_training = {"crossSdpaTraining", METAL_SHADER_DIR "/crossSdpaTraining.metal", 6, 5, 64, 1, 1};
#endif
static const MetalKernel k_cross_attention = {"crossAttentionF32", METAL_SHADER_DIR "/crossAttentionF32.metal", 7, 6, 64, 1, 1};
static const MetalKernel k_quantize = {"quantizeLinear", METAL_SHADER_DIR "/quantizeLinear.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_dequantize = {"dequantizeLinear", METAL_SHADER_DIR "/dequantizeLinear.metal", 5, 4, 64, 1, 1};
static const MetalKernel k_qlinear_int8 = {"qLinearInt8", METAL_SHADER_DIR "/qLinearInt8.metal", 7, 6, 64, 1, 1};
static const MetalKernel k_qlinear_int8_tiled = {"qLinearInt8Tiled", METAL_SHADER_DIR "/qLinearInt8Tiled.metal", 7, 6, 8, 8, 1};
static const MetalKernel k_qembedding_int8 = {"qEmbeddingInt8", METAL_SHADER_DIR "/qEmbeddingInt8.metal", 6, 5, 64, 1, 1};
static const MetalKernel k_qconv2d_int8 = {"qConv2DInt8", METAL_SHADER_DIR "/qConv2DInt8.metal", 7, 6, 64, 1, 1};
static const MetalKernel k_quantize_typed_i8u8 = {"quantizeLinearTyped", METAL_SHADER_DIR "/quantizeLinearTyped.metal", 5, 4, 64, 1, 1};
static const MetalKernel k_dequantize_typed_i8u8 = {"dequantizeLinearTyped", METAL_SHADER_DIR "/dequantizeLinearTyped.metal", 5, 4, 64, 1, 1};
static const MetalKernel k_qadd_i8u8 = {"qAdd", METAL_SHADER_DIR "/qAdd.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_qbatch_matmul_i8u8 = {
    "qBatchMatMul", METAL_SHADER_DIR "/qBatchMatMul.metal", 5, 4, 64, 1, 1
};
static const MetalKernel k_qsilu_i8u8 = {"qSiLUInt8", METAL_SHADER_DIR "/qSiLUInt8.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_qgelu_i8u8 = {"qGELUInt8", METAL_SHADER_DIR "/qGELUInt8.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_qgroupnorm_stats = {"qGroupNormStats", METAL_SHADER_DIR "/qGroupNormStats.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_qgroupnorm_apply = {"qGroupNormApply", METAL_SHADER_DIR "/qGroupNormApply.metal", 6, 5, 64, 1, 1};
static const MetalKernel k_qlayernorm_stats = {"qLayerNormStats", METAL_SHADER_DIR "/qLayerNormStats.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_qlayernorm_apply = {"qLayerNormApply", METAL_SHADER_DIR "/qLayerNormApply.metal", 6, 5, 64, 1, 1};
static const MetalKernel k_qsdpa_int8 = {"qSDPAInt8", METAL_SHADER_DIR "/qSDPAInt8.metal", 6, 5, 32, 1, 1};
static const MetalKernel k_qargmax_int8 = {"qArgMaxInt8", METAL_SHADER_DIR "/qArgMaxInt8.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_qmaskedmean_int8 = {"qMaskedMeanInt8", METAL_SHADER_DIR "/qMaskedMeanInt8.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_requantize_linear_i8u8 = {"requantizeLinearTyped", METAL_SHADER_DIR "/requantizeLinearTyped.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_copy_typed_i8u8 = {"copyTyped", METAL_SHADER_DIR "/copyTyped.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_concat_typed_i8u8 = {"concatCopyTyped", METAL_SHADER_DIR "/concatCopyTyped.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_maxpool_typed_i8u8 = {"maxPool2DTyped", METAL_SHADER_DIR "/maxPool2DTyped.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_resize_nearest_typed_i8u8 = {"resizeNearestTyped", METAL_SHADER_DIR "/resizeNearestTyped.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_transpose_typed_i8u8 = {"transposeTyped", METAL_SHADER_DIR "/transposeTyped.metal", 3, -1, 64, 1, 1};
static const MetalKernel k_spatial_softargmax_y = {"spatialSoftargmaxY", METAL_SHADER_DIR "/spatialSoftargmaxY.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_profile_x = {"profileX", METAL_SHADER_DIR "/profileX.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_profile_y = {"profileY", METAL_SHADER_DIR "/profileY.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_mean_height = {"meanHeight", METAL_SHADER_DIR "/meanHeight.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_nms = {"nonMaxSuppression", METAL_SHADER_DIR "/nonMaxSuppression.metal", 4, 3, 1, 1, 1};
static const MetalKernel k_copy = {"copy", METAL_SHADER_DIR "/copy.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_add = {"add", METAL_SHADER_DIR "/add.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_sigmoid = {"sigmoid", METAL_SHADER_DIR "/sigmoid.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_gelu = {"gELU", METAL_SHADER_DIR "/gELU.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_silu = {"siLU", METAL_SHADER_DIR "/siLU.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_layernorm = {"layerNorm", METAL_SHADER_DIR "/layerNorm.metal", 5, 4, 64, 1, 1};
static const MetalKernel k_groupnorm = {"groupNorm", METAL_SHADER_DIR "/groupNorm.metal", 5, 4, 64, 1, 1};
#if VOLVOXAI_ENABLE_TRAINING
static const MetalKernel k_dropout = {"dropout", METAL_SHADER_DIR "/dropout.metal", 3, 2, 64, 1, 1};
#endif
static const MetalKernel k_softmax = {"softmax", METAL_SHADER_DIR "/softmax.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_reduce = {"reduce", METAL_SHADER_DIR "/reduce.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_transpose = {"generalTranspose", METAL_SHADER_DIR "/generalTranspose.metal", 3, -1, 64, 1, 1};
static const MetalKernel k_expand = {"expand", METAL_SHADER_DIR "/expand.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_batch_matmul = {"batchMatMul", METAL_SHADER_DIR "/batchMatMul.metal", 4, -1, 8, 8, 1};
static const MetalKernel k_gather = {"gather", METAL_SHADER_DIR "/gather.metal", 4, 3, 64, 1, 1};
static const MetalKernel k_slice = {"slice", METAL_SHADER_DIR "/slice.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_concat = {"concatCopy", METAL_SHADER_DIR "/concatCopy.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_typed_control_32 = {"typedControl32Native", METAL_SHADER_DIR "/typedControl32Native.metal", 4, -1, 64, 1, 1};
static const MetalKernel k_where_32 = {"where32Native", METAL_SHADER_DIR "/where32Native.metal", 5, 4, 64, 1, 1};
static const MetalKernel k_argmax_f32_i32 = {"argMaxF32I32Native", METAL_SHADER_DIR "/argMaxF32I32Native.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_concat_32 = {"concatCopy32Native", METAL_SHADER_DIR "/concatCopy32Native.metal", 3, 2, 64, 1, 1};
static const MetalKernel k_linear_in_out = {"linearF32RowMajor", METAL_SHADER_DIR "/linearF32RowMajor.metal", 5, 4, 64, 1, 1};
static const MetalKernel k_linear_in_out_tiled = {"linearF32RowMajorTiled", METAL_SHADER_DIR "/linearF32RowMajorTiled.metal", 5, 4, 8, 8, 1};
static const MetalKernel k_linear_out_in = {"linearF32", METAL_SHADER_DIR "/linearF32.metal", 6, 5, 64, 1, 1};
static const MetalKernel k_linear_out_in_tiled = {"linearF32Tiled", METAL_SHADER_DIR "/linearF32Tiled.metal", 6, 5, 8, 8, 1};
static const MetalKernel k_conv2d = {"conv2D", METAL_SHADER_DIR "/conv2D.metal", 5, 4, 8, 8, 1};

static MetalContextState* metal_context_state_get(int create) {
    VxEngineState* owner = vx_engine_state_current();
    if (!owner) return NULL;
    MetalContextState* state =
        (MetalContextState*)owner->metal_context_state;
    if (!state && create) {
        state = (MetalContextState*)calloc(1, sizeof(*state));
        if (!state) return NULL;
        state->shape_generation = 1;
        state->capacity_generation = 1;
#ifdef VOLVOX_METAL_TESTING
        state->test_domain_allocation_failure_after = -1;
#endif
        owner->metal_context_state = state;
        owner->metal_context_state_destroy = metal_context_state_destroy;
    }
    return state;
}

static void metal_device_lock(void) {
    pthread_mutex_lock(&g_metal_device_state.mutex);
}

static void metal_device_unlock(void) {
    pthread_mutex_unlock(&g_metal_device_state.mutex);
}

static int metal_ready(void) {
    MetalContextState* state = metal_context_state_get(0);
    return state && state->device_acquired && device != nil &&
        commandQueue != nil;
}

void metal_set_shader_root(const char* root) {
    if (volvoxai_shader_store_set_override_root(root) != VOLVOXAI_SHADER_STORE_OK) {
        fprintf(stderr, "[Metal] invalid configured shader root\n");
    }
}

static NSString* metal_shader_source(const char* path) {
    VolvoxAIShaderView view;
    if (volvoxai_shader_store_get(path, &view) !=
            VOLVOXAI_SHADER_STORE_OK ||
        view.size == 0) return nil;
    NSString* source = [[NSString alloc] initWithBytes:view.data
                                               length:view.size
                                             encoding:NSUTF8StringEncoding];
#if !__has_feature(objc_arc)
    return [source autorelease];
#else
    return source;
#endif
}

static void clear_slot(MetalTensorSlot* s) {
    if (!s) return;
    VX_METAL_RELEASE(s->buffer);
    memset(s, 0, sizeof(*s));
}

static MetalPipelineCacheEntry* metal_pipeline_cache_entry(
    const MetalKernel* descriptor, int create) {
    if (!descriptor) return NULL;
    for (size_t index = 0; index < g_metal_device_state.pipeline_count;
         index++) {
        MetalPipelineCacheEntry* entry =
            &g_metal_device_state.pipelines[index];
        if (entry->descriptor == descriptor) return entry;
    }
    if (!create ||
        g_metal_device_state.pipeline_count >= METAL_MAX_PIPELINES) return NULL;
    MetalPipelineCacheEntry* entry = &g_metal_device_state.pipelines[
        g_metal_device_state.pipeline_count++];
    memset(entry, 0, sizeof(*entry));
    entry->descriptor = descriptor;
    return entry;
}

static id<MTLComputePipelineState> compile_kernel(const MetalKernel* k) {
    if (!metal_ready() || !k) return nil;
    metal_device_lock();
    MetalPipelineCacheEntry* entry = metal_pipeline_cache_entry(k, 1);
    if (!entry || entry->failed) {
        metal_device_unlock();
        return nil;
    }
    if (entry->pipeline) {
        id<MTLComputePipelineState> cached = entry->pipeline;
        metal_device_unlock();
        return cached;
    }
    @autoreleasepool {
        NSError* error = nil;
        NSString* source = metal_shader_source(k->path);
        if (!source) {
            fprintf(stderr, "[Metal] failed to load generated shader %s\n", k->path);
            entry->failed = 1;
            metal_device_unlock();
            return nil;
        }

        if ([source rangeOfString:@"[[user(fake"].location != NSNotFound) {
            fprintf(stderr, "[Metal] shader %s has unresolved Naga bindings\n", k->name);
            entry->failed = 1;
            metal_device_unlock();
            return nil;
        }
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "[Metal] shader compile failed (%s): %s\n",
                    k->name, error ? [[error localizedDescription] UTF8String] : "unknown error");
            entry->failed = 1;
            metal_device_unlock();
            return nil;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"main_"];
        if (!fn) {
            fprintf(stderr, "[Metal] shader entry main_ not found (%s)\n", k->name);
            VX_METAL_RELEASE(library);
            entry->failed = 1;
            metal_device_unlock();
            return nil;
        }

        entry->pipeline =
            [device newComputePipelineStateWithFunction:fn error:&error];
        VX_METAL_RELEASE(fn);
        VX_METAL_RELEASE(library);
        if (!entry->pipeline) {
            fprintf(stderr, "[Metal] pipeline creation failed (%s): %s\n",
                    k->name, error ? [[error localizedDescription] UTF8String] : "unknown error");
            entry->failed = 1;
            metal_device_unlock();
            return nil;
        }
        id<MTLComputePipelineState> pipeline = entry->pipeline;
        metal_device_unlock();
        return pipeline;
    }
}

#if VOLVOXAI_ENABLE_TRAINING
static const MetalTrainingShaderDesc* training_shader_desc(const char* shader_name) {
    if (!shader_name || !shader_name[0]) return NULL;
    size_t name_len = strlen(shader_name);
    if (name_len > 5 && !strcmp(shader_name + name_len - 5, ".wgsl")) name_len -= 5;
    for (size_t i = 0; i < sizeof(training_shader_descs) / sizeof(training_shader_descs[0]); i++) {
        const char* candidate = training_shader_descs[i].name;
        if (strlen(candidate) == name_len && !strncmp(candidate, shader_name, name_len)) {
            return &training_shader_descs[i];
        }
    }
    return NULL;
}

static int training_entry_supported(const MetalTrainingShaderDesc* desc, const char* entry_point) {
    if (!desc || !entry_point || !entry_point[0]) return 0;
    for (size_t i = 0; i < sizeof(desc->entries) / sizeof(desc->entries[0]); i++) {
        if (!desc->entries[i]) break;
        if (!strcmp(desc->entries[i], entry_point)) return 1;
    }
    return 0;
}

static MetalTrainingKernel* training_find_kernel(const char* shader, const char* entry) {
    for (int i = 0; i < training_kernel_count; i++) {
        if (!strcmp(training_kernels[i].shader, shader) &&
            !strcmp(training_kernels[i].entry, entry)) return &training_kernels[i];
    }
    return NULL;
}

static MetalTrainingKernel* compile_training_kernel(const MetalTrainingShaderDesc* desc,
                                                     const char* entry_point) {
    if (!metal_ready() || !desc || !training_entry_supported(desc, entry_point)) return NULL;
    MetalTrainingKernel* cached = training_find_kernel(desc->name, entry_point);
    if (cached) return cached;
    if (training_kernel_count >= METAL_TRAINING_MAX_KERNELS) return NULL;

    @autoreleasepool {
        char relative_path[PATH_MAX];
        int path_length = snprintf(relative_path, sizeof(relative_path), "%s/%s.metal",
                                   METAL_SHADER_DIR, desc->name);
        if (path_length <= 0 || (size_t)path_length >= sizeof(relative_path)) return NULL;
        NSError* error = nil;
        NSString* source = metal_shader_source(relative_path);
        if (!source) {
            fprintf(stderr, "[Metal training] failed to load shader %s\n", desc->name);
            return NULL;
        }
        /* Multi-entry MSL is generated through Naga's resource-map API with
           logical binding numbers preserved. Reject output produced without
           that map instead of trying to repair generated source text. */
        if ([source rangeOfString:@"[[user(fake"].location != NSNotFound) {
            fprintf(stderr,
                    "[Metal training] shader %s has unresolved bindings; regenerate Metal training shaders\n",
                    desc->name);
            return NULL;
        }
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "[Metal training] shader compile failed (%s): %s\n",
                    desc->name, error ? [[error localizedDescription] UTF8String] : "unknown error");
            return NULL;
        }
        NSString* metal_entry = !strcmp(entry_point, "main")
            ? @"main_" : [NSString stringWithUTF8String:entry_point];
        id<MTLFunction> fn = [library newFunctionWithName:metal_entry];
        if (!fn) {
            fprintf(stderr, "[Metal training] entry %s not found in %s\n",
                    entry_point, desc->name);
            VX_METAL_RELEASE(library);
            return NULL;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:fn error:&error];
        VX_METAL_RELEASE(fn);
        VX_METAL_RELEASE(library);
        if (!pipeline) {
            fprintf(stderr, "[Metal training] pipeline creation failed (%s:%s): %s\n",
                    desc->name, entry_point,
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
            return NULL;
        }

        MetalTrainingKernel* out = &training_kernels[training_kernel_count++];
        out->shader[0] = 0;
        out->entry[0] = 0;
        out->pipeline = nil;
        strncpy(out->shader, desc->name, sizeof(out->shader) - 1);
        out->shader[sizeof(out->shader) - 1] = 0;
        strncpy(out->entry, entry_point, sizeof(out->entry) - 1);
        out->entry[sizeof(out->entry) - 1] = 0;
        out->pipeline = pipeline;
        return out;
    }
}
#endif

static uint64_t metal_generation_next(uint64_t generation) {
    generation++;
    return generation ? generation : 1u;
}

static size_t metal_max_buffer_length(void) {
    if (!device) return 0;
    if (@available(macOS 10.14, *)) return (size_t)[device maxBufferLength];
    return UINT32_MAX;
}

static int metal_buffer_size_valid(size_t bytes) {
    size_t limit = metal_max_buffer_length();
    return bytes > 0 && bytes <= UINT32_MAX && limit && bytes <= limit;
}

int metal_query_domain_limits(MetalDomainLimits* limits) {
    MetalContextState* state = metal_context_state_get(0);
    int ready = 0;
    if (!limits) return -1;
    memset(limits, 0, sizeof(*limits));
    if (!state || !state->device_acquired) return -1;
    metal_device_lock();
    if (device && commandQueue) {
        size_t buffer_limit = metal_max_buffer_length();
        MTLSize workgroup_limit = [device maxThreadsPerThreadgroup];
        uint64_t thread_product = (uint64_t)workgroup_limit.width;
        if (!workgroup_limit.height || thread_product >
                UINT64_MAX / (uint64_t)workgroup_limit.height) {
            thread_product = UINT64_MAX;
        } else {
            thread_product *= (uint64_t)workgroup_limit.height;
            if (!workgroup_limit.depth || thread_product >
                    UINT64_MAX / (uint64_t)workgroup_limit.depth)
                thread_product = UINT64_MAX;
            else
                thread_product *= (uint64_t)workgroup_limit.depth;
        }
        if (buffer_limit > UINT32_MAX) buffer_limit = UINT32_MAX;
        limits->maximum_buffer_bytes = (uint64_t)buffer_limit;
        for (size_t axis = 0; axis < 3u; axis++)
            limits->maximum_workgroups[axis] = UINT32_MAX;
        limits->maximum_workgroup_size[0] = workgroup_limit.width > UINT32_MAX
            ? UINT32_MAX : (uint32_t)workgroup_limit.width;
        limits->maximum_workgroup_size[1] = workgroup_limit.height > UINT32_MAX
            ? UINT32_MAX : (uint32_t)workgroup_limit.height;
        limits->maximum_workgroup_size[2] = workgroup_limit.depth > UINT32_MAX
            ? UINT32_MAX : (uint32_t)workgroup_limit.depth;
        limits->maximum_threads_per_workgroup = thread_product > UINT32_MAX
            ? UINT32_MAX : (uint32_t)thread_product;
        limits->maximum_tensor_slots = METAL_GRAPH_MAX_TENSORS;
        limits->maximum_bindings = METAL_MAX_BINDINGS - 1u;
        ready = limits->maximum_buffer_bytes > 0u &&
            limits->maximum_workgroup_size[0] > 0u &&
            limits->maximum_workgroup_size[1] > 0u &&
            limits->maximum_workgroup_size[2] > 0u &&
            limits->maximum_threads_per_workgroup > 0u;
    }
    metal_device_unlock();
    return ready ? 0 : -1;
}

static id<MTLBuffer> create_buffer(size_t bytes, const void* data) {
    if (!metal_ready() || !metal_buffer_size_valid(bytes)) return nil;
    id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) return nil;
    if (data) memcpy([b contents], data, bytes);
    return b;
}

static id<MTLBuffer> qgroupnorm_stats_ensure(size_t bytes) {
    MetalContextState* state = metal_context_state_get(0);
    id<MTLBuffer> next;
    if (!metal_ready() || !state || bytes == 0) return nil;
    if (qgroupnorm_stats_buffer && qgroupnorm_stats_capacity >= bytes)
        return qgroupnorm_stats_buffer;
    if (state->domain_enforced) return nil;
    next = create_buffer(bytes, NULL);
    if (!next) return nil;
    VX_METAL_RELEASE(qgroupnorm_stats_buffer);
    qgroupnorm_stats_buffer = next;
    qgroupnorm_stats_capacity = bytes;
    state->capacity_generation =
        metal_generation_next(state->capacity_generation);
    return qgroupnorm_stats_buffer;
}

static id<MTLBuffer> qlayernorm_stats_ensure(size_t bytes) {
    MetalContextState* state = metal_context_state_get(0);
    id<MTLBuffer> next;
    if (!metal_ready() || !state || bytes == 0) return nil;
    if (qlayernorm_stats_buffer && qlayernorm_stats_capacity >= bytes)
        return qlayernorm_stats_buffer;
    if (state->domain_enforced) return nil;
    next = create_buffer(bytes, NULL);
    if (!next) return nil;
    VX_METAL_RELEASE(qlayernorm_stats_buffer);
    qlayernorm_stats_buffer = next;
    qlayernorm_stats_capacity = bytes;
    state->capacity_generation =
        metal_generation_next(state->capacity_generation);
    return qlayernorm_stats_buffer;
}

static int graph_find_slot(const void* host);
static MetalTensorSlot* graph_ensure_device(const void* host, size_t bytes, int is_weight);

#if VOLVOXAI_ENABLE_TRAINING
static void training_clear_slot(MetalTrainingTensorSlot* slot) {
    if (!slot) return;
    VX_METAL_RELEASE(slot->buffer);
    slot->host = NULL;
    slot->bytes = 0;
    slot->cap = 0;
    slot->graph_slot = NULL;
    slot->device_dirty = 0;
    slot->is_weight = 0;
}

static int training_find_slot(void* host) {
    if (!host) return -1;
    for (int i = 0; i < training_slot_count; i++) {
        if (training_slots[i].host == host) return i;
    }
    return -1;
}

static MetalTrainingTensorSlot* training_ensure_slot(void* host, size_t bytes,
                                                     unsigned char access, int is_weight) {
    if (!training_active || !host || bytes == 0) return NULL;
    int index = training_find_slot(host);
    if (index < 0) {
        if (training_slot_count >= METAL_GRAPH_MAX_TENSORS) return NULL;
        index = training_slot_count++;
        training_clear_slot(&training_slots[index]);
        training_slots[index].host = host;
    }
    MetalTrainingTensorSlot* slot = &training_slots[index];
    if (!slot->buffer && (access & METAL_TRAINING_ACCESS_READ)) {
        int graph_index = graph_find_slot(host);
        if (graph_index >= 0) {
            MetalTensorSlot* graph = graph_ensure_device(host, bytes, is_weight);
            if (graph && graph->buffer && graph->cap >= bytes) {
                VX_METAL_RETAIN_ASSIGN(slot->buffer, graph->buffer);
                slot->cap = graph->cap;
                slot->device_dirty = graph->device_dirty;
                slot->graph_slot = graph;
            }
        }
    }
    if (!slot->buffer || slot->cap < bytes) {
        VX_METAL_RELEASE(slot->buffer);
        slot->graph_slot = NULL;
        slot->buffer = create_buffer(bytes,
            (access & METAL_TRAINING_ACCESS_READ) ? host : NULL);
        if (!slot->buffer) return NULL;
        if (!(access & METAL_TRAINING_ACCESS_READ)) memset([slot->buffer contents], 0, bytes);
        slot->cap = bytes;
        slot->device_dirty = 0;
    } else if ((access & METAL_TRAINING_ACCESS_READ) && !slot->device_dirty && slot->bytes == 0) {
        memcpy([slot->buffer contents], host, bytes);
    }
    slot->bytes = bytes;
    if (is_weight) slot->is_weight = 1;
    return slot;
}

static void training_release_transients(void) {
    for (int i = 0; i < training_transient_count; i++) {
        VX_METAL_RELEASE(training_transients[i]);
    }
    training_transient_count = 0;
}

static int training_flush_commands(void) {
    int rc = 0;
    @autoreleasepool {
        if (training_encoder) {
            [training_encoder endEncoding];
            VX_METAL_RELEASE(training_encoder);
        }
        if (training_command) {
            [training_command commit];
            [training_command waitUntilCompleted];
            if ([training_command status] == MTLCommandBufferStatusError) {
                NSError* error = [training_command error];
                fprintf(stderr, "[Metal training] command buffer failed: %s\n",
                        error ? [[error localizedDescription] UTF8String] : "unknown error");
                rc = -1;
            }
            VX_METAL_RELEASE(training_command);
        }
        training_release_transients();
    }
    return rc;
}

static int training_ensure_encoder(void) {
    if (!training_active || !metal_ready()) return -1;
    if (training_encoder) return 0;
    @autoreleasepool {
        if (!training_command) {
            id<MTLCommandBuffer> command = [commandQueue commandBuffer];
            if (!command) return -1;
            VX_METAL_RETAIN_ASSIGN(training_command, command);
        }
        id<MTLComputeCommandEncoder> encoder = [training_command computeCommandEncoder];
        if (!encoder) return -1;
        VX_METAL_RETAIN_ASSIGN(training_encoder, encoder);
    }
    return 0;
}
#endif

static int graph_find_slot(const void* host) {
    MetalContextState* state = metal_context_state_get(0);
    if (!host) return -1;
    for (int i = 0; i < graph_slot_count; i++) {
        if (graph_slots[i].host == host &&
            (graph_slots[i].is_weight ||
             (state && graph_slots[i].shape_generation ==
                           state->shape_generation))) return i;
    }
    return -1;
}

static int graph_find_reusable_slot(size_t bytes) {
    int best = -1;
    int empty = -1;
    int grow = -1;
    for (int index = 0; index < graph_slot_count; index++) {
        MetalTensorSlot* slot = &graph_slots[index];
        if (slot->host || slot->is_weight) continue;
        if (!slot->buffer || slot->is_alias) {
            if (empty < 0) empty = index;
            continue;
        }
        if (slot->cap >= bytes &&
            (best < 0 || slot->cap < graph_slots[best].cap)) best = index;
        else if (grow < 0 || slot->cap > graph_slots[grow].cap) grow = index;
    }
    return best >= 0 ? best : (empty >= 0 ? empty : grow);
}

static MetalTensorSlot* graph_get_slot(const void* host, size_t bytes, int is_weight) {
    MetalContextState* state = metal_context_state_get(0);
    if (!metal_ready() || !state || !host ||
        !metal_buffer_size_valid(bytes)) return NULL;
    int idx = graph_find_slot(host);
    if (state->domain_enforced) {
        MetalTensorSlot* slot = idx >= 0 ? &graph_slots[idx] : NULL;
        if (!slot || !slot->buffer || slot->is_alias || bytes > slot->cap)
            return NULL;
        if (slot->domain_span) {
            /* Public inputs retain activation lifetime even when a kernel
             * consumes one through a weight/affine argument. The binding
             * commit marks this reserved span host-dirty on each run. */
            if (bytes > slot->domain_capacity) return NULL;
            if (bytes > slot->bytes) {
                slot->bytes = bytes;
                slot->host_dirty = 1;
                slot->device_dirty = 0;
            }
            return slot;
        }
        if (!slot->is_weight || !slot->host || !slot->bytes ||
            bytes > slot->bytes) return NULL;
        return slot;
    }
    if (idx < 0) {
        idx = graph_find_reusable_slot(bytes);
        if (idx < 0) {
            if (graph_slot_count >= METAL_GRAPH_MAX_TENSORS) return NULL;
            idx = graph_slot_count++;
            memset(&graph_slots[idx], 0, sizeof(graph_slots[idx]));
        }
        MetalTensorSlot* s = &graph_slots[idx];
        s->host = host;
        s->bytes = 0;
        s->host_dirty = 1;
        s->device_dirty = 0;
        s->is_weight = is_weight;
        s->shape_generation = is_weight ? 0 : state->shape_generation;
    }
    MetalTensorSlot* s = &graph_slots[idx];
    /* Grow, never shrink -- see the Vulkan twin. A caller asking for fewer
     * bytes is asking about part of the tensor, not redefining it, and letting
     * it shrink turns the next whole-tensor read into a growth that re-uploads
     * a host mirror over rows the device just wrote. */
    if (bytes > s->bytes) s->bytes = bytes;
    if (is_weight) s->is_weight = 1;
    return s;
}

static int graph_slot_ensure_capacity(MetalTensorSlot* slot, size_t bytes,
                                      const void* data, size_t data_bytes,
                                      int clear_tail) {
    MetalContextState* state = metal_context_state_get(0);
    size_t capacity = bytes;
    id<MTLBuffer> candidate;
    if (!state || !slot || data_bytes > bytes ||
        !metal_buffer_size_valid(bytes)) return 0;
    if (state->domain_enforced &&
        (!slot->buffer || slot->is_alias || slot->cap < bytes)) return 0;
    if (!slot->is_alias && slot->cap && slot->cap <= SIZE_MAX / 2u &&
        slot->cap * 2u > capacity &&
        metal_buffer_size_valid(slot->cap * 2u)) capacity = slot->cap * 2u;
    candidate = create_buffer(capacity, NULL);
    if (!candidate) return 0;
    if (clear_tail) memset([candidate contents], 0, capacity);
    if (data && data_bytes) memcpy([candidate contents], data, data_bytes);
    VX_METAL_RELEASE(slot->buffer);
    slot->buffer = candidate;
    slot->cap = capacity;
    slot->is_alias = 0;
    state->capacity_generation =
        metal_generation_next(state->capacity_generation);
    slot->capacity_generation = state->capacity_generation;
    return 1;
}

static MetalTensorSlot* graph_ensure_device(const void* host, size_t bytes, int is_weight) {
    MetalTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return NULL;
    if (!s->buffer || s->cap < bytes || s->is_alias) {
        if (!graph_slot_ensure_capacity(s, bytes, host, bytes, 0)) return NULL;
        s->host_dirty = 0;
        s->device_dirty = 0;
        return s;
    }
    if (s->host_dirty) {
        memcpy([s->buffer contents], host, bytes);
        s->host_dirty = 0;
        s->device_dirty = 0;
    }
    return s;
}

static MetalTensorSlot* graph_output_slot(const void* host, size_t bytes);

/* qLinearInt8 addresses byte storage as packed u32 words. Preserve a
 * logical-byte host contract while retaining the padded device allocation for
 * a following W8A8 node. */
static int graph_packed_bytes(size_t logical_bytes, size_t* storage_bytes) {
    if (!storage_bytes || logical_bytes == 0 || logical_bytes > SIZE_MAX - 3u) return 0;
    *storage_bytes = (logical_bytes + 3u) & ~(size_t)3u;
    return 1;
}

static MetalTensorSlot* graph_ensure_packed_bytes(const void* host, size_t logical_bytes,
                                                   int is_weight) {
    size_t storage_bytes;
    if (!host || !graph_packed_bytes(logical_bytes, &storage_bytes)) return NULL;
    MetalTensorSlot* s = graph_get_slot(host, storage_bytes, is_weight);
    if (!s) return NULL;
    if (!s->buffer || s->cap < storage_bytes || s->is_alias) {
        /* Shared Metal storage can be cleared and populated directly. Avoid a
         * maximum-shape host calloc merely to materialize at most three tail
         * zero bytes for the shader's packed-u32 view. */
        if (!graph_slot_ensure_capacity(
                s, storage_bytes, host, logical_bytes, 1)) return NULL;
        s->host_dirty = 0;
        s->device_dirty = 0;
    } else if (s->host_dirty) {
        memset([s->buffer contents], 0, storage_bytes);
        memcpy([s->buffer contents], host, logical_bytes);
        s->host_dirty = 0;
        s->device_dirty = 0;
    }
    return s;
}

static MetalTensorSlot* graph_output_packed_bytes(const void* host, size_t logical_bytes) {
    size_t storage_bytes;
    if (!host || !graph_packed_bytes(logical_bytes, &storage_bytes)) return NULL;
    return graph_output_slot(host, storage_bytes);
}

static MetalTensorSlot* graph_output_slot(const void* host, size_t bytes) {
    MetalTensorSlot* s = graph_get_slot(host, bytes, 0);
    if (!s) return NULL;
    if (!s->buffer || s->cap < bytes || s->is_alias)
        if (!graph_slot_ensure_capacity(s, bytes, NULL, 0, 0)) return NULL;
    s->host_dirty = 0;
    s->device_dirty = 0;
    return s;
}

static void graph_mark_device(MetalTensorSlot* s) {
    if (!s) return;
    s->device_dirty = 1;
    s->host_dirty = 0;
}

/* A binding into part of a resident tensor -- the Metal spelling of Vulkan's
 * VkGraphWindow. Row execution asks for a *slice* of an activation, not for a
 * tensor, and the slot map is keyed by complete host-span bases, so an interior
 * pointer finds nothing on an exact lookup. Resolving it as an offset into the
 * containing slot is what lets a decode row stay on the device.
 *
 * `setBuffer:offset:` already takes an offset, so this costs no new machinery
 * at dispatch beyond carrying the field -- the same shape the Vulkan port had,
 * and unlike OpenGL, which had to gain a ranged bind. */
typedef struct {
    MetalTensorSlot* slot;
    size_t offset;
    size_t bytes;
} MetalGraphWindow;

/*
 * The resident slot whose span contains `[host, host + bytes)`.
 *
 * Exact base first, then the smallest containing span: a tensor and an arena
 * span that both cover the pointer are both legal answers, and the tighter one
 * is the tensor. Returns -1 when nothing contains it, which is the ordinary
 * "this is a new tensor" case and not an error.
 */
static int graph_find_containing_slot(const void* host, size_t bytes, size_t* inner) {
    MetalContextState* state = metal_context_state_get(0);
    uintptr_t start = (uintptr_t)host;
    uintptr_t end;
    int best = -1;
    int exact;
    if (!state || !host || !bytes || start > UINTPTR_MAX - bytes) return -1;
    end = start + bytes;
    exact = graph_find_slot(host);
    if (exact >= 0 && graph_slots[exact].bytes >= bytes) {
        if (inner) *inner = 0;
        return exact;
    }
    for (int index = 0; index < graph_slot_count; index++) {
        MetalTensorSlot* slot = &graph_slots[index];
        uintptr_t slot_start = (uintptr_t)slot->host;
        uintptr_t slot_end;
        if (!slot->host || !slot->bytes || !slot->buffer || slot->is_alias ||
            slot_start > UINTPTR_MAX - slot->bytes) continue;
        if (!slot->is_weight && slot->shape_generation != state->shape_generation)
            continue;
        slot_end = slot_start + slot->bytes;
        if (start < slot_start || end > slot_end) continue;
        if (best < 0 || slot->bytes < graph_slots[best].bytes) best = index;
    }
    if (best >= 0 && inner) {
        *inner = (size_t)((uintptr_t)host - (uintptr_t)graph_slots[best].host);
    }
    return best;
}

/*
 * Finish a window against a slot that is already resident.
 *
 * A row offset is `row * width * element size`, so a row is bindable exactly
 * when its *stride* clears METAL_GRAPH_ALIGN. Refusing here names the
 * constraint; the caller falls back to the host row path.
 */
static int graph_window_finish(MetalTensorSlot* slot, size_t inner, size_t bytes,
                               MetalGraphWindow* out) {
    if (!slot || !out || !bytes || !slot->buffer || slot->is_alias ||
        inner > slot->bytes || bytes > slot->bytes - inner) return 0;
    if (inner % METAL_GRAPH_ALIGN) return 0;
    out->slot = slot;
    out->offset = inner;
    out->bytes = bytes;
    return 1;
}

/*
 * Whether `host` names an interior slice, and of which base.
 *
 * Resolution goes through the ordinary `graph_ensure_*` on the *base* pointer
 * rather than around it, so a window inherits every domain, capacity and
 * dirty-state rule a whole tensor obeys. The only thing that differs is where
 * the binding starts.
 */
static int graph_window_base(const void* host, size_t bytes,
                             const void** base, size_t* base_bytes, size_t* inner) {
    int index = graph_find_containing_slot(host, bytes, inner);
    if (index < 0 || *inner == 0) return 0;
    *base = graph_slots[index].host;
    *base_bytes = graph_slots[index].bytes;
    return 1;
}

/* The packed spelling. A window's own bytes stay logical: only the containing
 * slot is rounded up to whole words, and the tail it pads belongs to the
 * tensor rather than to any one row. */
static int graph_window_packed_bytes(const void* host, size_t logical_bytes,
                                     int is_weight, MetalGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = logical_bytes;
    size_t inner = 0;
    size_t storage_bytes;
    MetalTensorSlot* slot;
    if (!host || !out || !graph_packed_bytes(logical_bytes, &storage_bytes)) return 0;
    if (graph_window_base(host, logical_bytes, &base, &base_bytes, &inner)) {
        slot = graph_ensure_packed_bytes(base, base_bytes, is_weight);
        return graph_window_finish(slot, inner, storage_bytes, out);
    }
    slot = graph_ensure_packed_bytes(host, logical_bytes, is_weight);
    if (!slot) return 0;
    out->slot = slot;
    out->offset = 0;
    out->bytes = storage_bytes;
    return 1;
}

/*
 * An output window.
 *
 * A row writes part of a tensor, so the rest has to already be on the device --
 * otherwise the next read of an untouched row sees whatever the buffer held.
 * When the containing slot is host-dirty this uploads it whole first and the
 * kernel then overwrites one row, which is the same order the host row path
 * achieves by syncing before it runs.
 */
static int graph_window_output_packed(const void* host, size_t logical_bytes,
                                      MetalGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = logical_bytes;
    size_t inner = 0;
    size_t storage_bytes;
    MetalTensorSlot* slot;
    if (!host || !out || !graph_packed_bytes(logical_bytes, &storage_bytes)) return 0;
    if (graph_window_base(host, logical_bytes, &base, &base_bytes, &inner)) {
        slot = graph_ensure_packed_bytes(base, base_bytes, 0);
        return graph_window_finish(slot, inner, storage_bytes, out);
    }
    slot = graph_output_packed_bytes(host, logical_bytes);
    if (!slot) return 0;
    out->slot = slot;
    out->offset = 0;
    out->bytes = storage_bytes;
    return 1;
}

static int graph_window_device(const void* host, size_t bytes, int is_weight,
                               MetalGraphWindow* out) {
    const void* base = host;
    size_t base_bytes = bytes;
    size_t inner = 0;
    MetalTensorSlot* slot;
    if (!host || !bytes || !out) return 0;
    if (!graph_window_base(host, bytes, &base, &base_bytes, &inner)) {
        base = host;
        base_bytes = bytes;
        inner = 0;
    }
    slot = graph_ensure_device(base, base_bytes, is_weight);
    if (!slot) return 0;
    if (inner == 0) {
        out->slot = slot;
        out->offset = 0;
        out->bytes = bytes;
        return 1;
    }
    return graph_window_finish(slot, inner, bytes, out);
}

/*
 * Where in the ids tensor a dispatch starts, as a token index.
 *
 * Every other activation binds as a window, because a row's byte offset is
 * `row * stride` and that clears the alignment whenever the stride does. A
 * token row is a single i32, so on a device whose alignment exceeds four bytes
 * no ids window past row zero would ever resolve. Binding the ids whole and
 * naming the row as a shader scalar keeps that answer the same on every
 * backend, which is the point -- see the Vulkan and OpenGL twins of this.
 */
static int graph_token_window(const int32_t* tokens, size_t token_bytes,
                              MetalTensorSlot** slot, uint32_t* token_offset) {
    const void* base = tokens;
    size_t base_bytes = token_bytes;
    size_t inner = 0;
    int index;
    if (!tokens || !token_bytes || !slot || !token_offset) return 0;
    *token_offset = 0;
    index = graph_find_containing_slot(tokens, token_bytes, &inner);
    if (index >= 0 && inner) {
        /* A misaligned interior pointer is not a token boundary, so it is not
         * a row of this tensor and nothing here can address it. */
        if (inner % sizeof(int32_t)) return 0;
        if (inner / sizeof(int32_t) > UINT32_MAX) return 0;
        base = graph_slots[index].host;
        base_bytes = graph_slots[index].bytes;
        *token_offset = (uint32_t)(inner / sizeof(int32_t));
    }
    *slot = graph_ensure_device(base, base_bytes, 0);
    return *slot != NULL;
}

static void graph_release_retained_bindings(void) {
    for (int index = 0; index < graph_retained_binding_count; index++) {
        VX_METAL_RELEASE(graph_retained_bindings[index]);
    }
    graph_retained_binding_count = 0;
}

static int graph_buffer_has_pending_use(id<MTLBuffer> buffer) {
    if (!buffer) return 0;
    for (int index = 0; index < graph_retained_binding_count; index++) {
        if (graph_retained_bindings[index] == buffer) return 1;
    }
    return 0;
}

static int graph_flush_commands(void) {
    int rc = 0;
    @autoreleasepool {
        if (graph_command) {
            [graph_command commit];
#ifdef VOLVOX_METAL_TESTING
            graph_debug_commit_count++;
#endif
            [graph_command waitUntilCompleted];
#ifdef VOLVOX_METAL_TESTING
            graph_debug_wait_count++;
#endif
            if ([graph_command status] == MTLCommandBufferStatusError) {
                NSError* error = [graph_command error];
                fprintf(stderr, "[Metal] graph command buffer failed: %s\n",
                        error ? [[error localizedDescription] UTF8String] : "unknown error");
                rc = -1;
            }
            VX_METAL_RELEASE(graph_command);
        }
        graph_release_retained_bindings();
    }
    return rc;
}

static int graph_ensure_command(void) {
    if (!metal_ready()) return -1;
    if (graph_command) return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = [commandQueue commandBuffer];
        if (!command) return -1;
        VX_METAL_RETAIN_ASSIGN(graph_command, command);
    }
    return 0;
}

static int dispatch_kernel(const MetalKernel* k, const MetalBinding* binds,
                           uint32_t gx, uint32_t gy, uint32_t gz) {
    id<MTLComputePipelineState> pipeline;
    if (!metal_ready() || !k || !binds || gx == 0 || gy == 0 || gz == 0)
        return 0;
    if (k->binding_count <= 0 || k->binding_count >= METAL_MAX_BINDINGS) return 0;
    uint64_t threads_per_group = (uint64_t)k->wg_x * (uint64_t)k->wg_y * (uint64_t)k->wg_z;
    MTLSize device_limit = [device maxThreadsPerThreadgroup];
    if (threads_per_group == 0 || k->wg_x <= 0 || k->wg_y <= 0 ||
        k->wg_z <= 0 || (NSUInteger)k->wg_x > device_limit.width ||
        (NSUInteger)k->wg_y > device_limit.height ||
        (NSUInteger)k->wg_z > device_limit.depth) return 0;
    uint32_t sizes[METAL_MAX_BINDINGS] = {0};
    for (int i = 0; i < k->binding_count; i++) {
        if (!binds[i].buffer || !metal_buffer_size_valid(binds[i].bytes))
            return 0;
        /* A window's length is what the shader bounds-checks against, and its
         * indices are relative to the bound offset -- so this stays the window's
         * own byte count, not the containing tensor's. */
        sizes[i] = (uint32_t)binds[i].bytes;
        if (binds[i].offset) {
            NSUInteger length = [binds[i].buffer length];
            if (binds[i].offset % METAL_GRAPH_ALIGN ||
                binds[i].offset > (size_t)length ||
                binds[i].bytes > (size_t)length - binds[i].offset) return 0;
        }
    }
    pipeline = compile_kernel(k);
    if (!pipeline || threads_per_group >
            (uint64_t)[pipeline maxTotalThreadsPerThreadgroup]) return 0;
    if (graph_forward_active && graph_forward_error) return 0;
    if (!graph_forward_active) graph_forward_error = 0;
    if (graph_retained_binding_count >
        METAL_GRAPH_MAX_RETAINED_BINDINGS - k->binding_count) {
        if (graph_flush_commands() != 0) {
            graph_forward_error = 1;
            return 0;
        }
    }
    if (graph_ensure_command() != 0) {
        return 0;
    }
    @autoreleasepool {
        /* Separate encoders are deliberate dependency boundaries. This keeps
         * producer/consumer and shared-scratch ordering explicit while still
         * amortizing command submission and completion across the forward. */
        id<MTLComputeCommandEncoder> enc = [graph_command computeCommandEncoder];
        if (!enc) {
            if (graph_flush_commands() != 0) graph_forward_error = 1;
            return 0;
        }
        [enc setComputePipelineState:pipeline];

        for (int i = 0; i < k->binding_count; i++) {
            [enc setBuffer:binds[i].buffer
                    offset:(NSUInteger)binds[i].offset
                   atIndex:(NSUInteger)i];
            int retained_index = graph_retained_binding_count++;
            VX_METAL_RETAIN_ASSIGN(
                graph_retained_bindings[retained_index], binds[i].buffer);
        }
        [enc setBytes:sizes
               length:(NSUInteger)(sizeof(uint32_t) * k->binding_count)
              atIndex:(NSUInteger)k->binding_count];

        MTLSize groups = MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);
        MTLSize threads = MTLSizeMake((NSUInteger)k->wg_x, (NSUInteger)k->wg_y, (NSUInteger)k->wg_z);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [enc endEncoding];
#ifdef VOLVOX_METAL_TESTING
        graph_debug_dispatch_count++;
#endif
    }
    if (!graph_forward_active && graph_flush_commands() != 0) {
        graph_forward_error = 1;
        return 0;
    }
    return 1;
}

int metal_init(void) {
    MetalContextState* state = metal_context_state_get(1);
    if (!state) return -1;
    metal_device_lock();
    if (state->device_acquired) {
        metal_device_unlock();
        return 0;
    }
    if (g_metal_device_state.reference_count != 0) {
        g_metal_device_state.reference_count++;
        state->device_acquired = 1;
        metal_device_unlock();
        return 0;
    }
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("[VolvoxAI GPU] Metal is not supported on this device.\n");
            metal_device_unlock();
            return -1;
        }
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            printf("[VolvoxAI GPU] Failed to create Metal command queue.\n");
            VX_METAL_RELEASE(device);
            metal_device_unlock();
            return -1;
        }
        const char* name = [[device name] UTF8String];
        printf("[VolvoxAI GPU] Metal initialized successfully on: %s\n", name ? name : "unknown");
    }
    g_metal_device_state.reference_count = 1;
    state->device_acquired = 1;
    metal_device_unlock();
    return 0;
}

#if VOLVOXAI_ENABLE_TRAINING
int metal_training_available(void) {
    return metal_ready() ? 1 : 0;
}

int metal_training_supports(const char* shader_name, const char* entry_point,
                            const size_t* bytes, int binding_count,
                            uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    const MetalTrainingShaderDesc* desc = training_shader_desc(shader_name);
    if (!metal_ready() || !training_entry_supported(desc, entry_point) || !bytes ||
        binding_count != desc->binding_count || groups_x == 0 || groups_y == 0 ||
        groups_z == 0) return 0;
    NSUInteger max_buffer_length = (NSUInteger)-1;
    if (@available(macOS 10.14, *)) {
        max_buffer_length = [device maxBufferLength];
    }
    for (int i = 0; i < binding_count; i++) {
        if (bytes[i] == 0 || bytes[i] > UINT32_MAX ||
            bytes[i] > (size_t)max_buffer_length) return 0;
    }
    MTLSize limit = [device maxThreadsPerThreadgroup];
    if ((NSUInteger)desc->wg_x > limit.width ||
        (NSUInteger)desc->wg_y > limit.height ||
        (NSUInteger)desc->wg_z > limit.depth) return 0;
    if ((uint64_t)desc->wg_x * desc->wg_y * desc->wg_z >
        (uint64_t)limit.width * limit.height * limit.depth) return 0;
    return 1;
}

int metal_training_begin(void) {
    if (!metal_ready() || training_active) return -1;
    /* Training owns a separate lazy command buffer. Never let it race a
     * partially encoded inference forward over shared graph-slot buffers. */
    if (graph_flush_commands() != 0) return -1;
    training_active = 1;
    return 0;
}

int metal_training_dispatch(const char* shader_name, const char* entry_point,
                            void* const* hosts, const size_t* bytes,
                            const unsigned char* access,
                            const unsigned char* is_weight,
                            int binding_count,
                            uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    if (!training_active || !hosts || !bytes || !access ||
        binding_count > METAL_TRAINING_MAX_BINDINGS ||
        !metal_training_supports(shader_name, entry_point, bytes, binding_count,
                                 groups_x, groups_y, groups_z)) {
        return -1;
    }
    const MetalTrainingShaderDesc* desc = training_shader_desc(shader_name);
    if (!desc || binding_count != desc->binding_count ||
        !training_entry_supported(desc, entry_point)) return -1;
    MetalTrainingKernel* kernel = compile_training_kernel(desc, entry_point);
    if (!kernel || !kernel->pipeline) return -1;
    uint64_t threads_per_group = (uint64_t)desc->wg_x * desc->wg_y * desc->wg_z;
    if (threads_per_group > (uint64_t)[kernel->pipeline maxTotalThreadsPerThreadgroup]) {
        return -1;
    }

    /* Params are immutable per dispatch. Keep a distinct copy alive until the
       batched command buffer completes so callers may reuse stack storage. */
    if (training_transient_count >= METAL_TRAINING_MAX_TRANSIENTS &&
        training_flush_commands() != 0) return -1;
    id<MTLBuffer> bound_buffers[METAL_TRAINING_MAX_BINDINGS] = {nil};
    uint32_t sizes[METAL_TRAINING_MAX_BINDINGS] = {0};
    for (int i = 0; i < binding_count; i++) {
        if (!hosts[i] || bytes[i] == 0 || bytes[i] > UINT32_MAX ||
            (is_weight && is_weight[i] > 1) ||
            access[i] < METAL_TRAINING_ACCESS_READ ||
            access[i] > METAL_TRAINING_ACCESS_READ_WRITE) return -1;
        unsigned char expected = desc->read_write_mask & (1u << i)
            ? METAL_TRAINING_ACCESS_READ_WRITE : METAL_TRAINING_ACCESS_READ;
        if (access[i] != expected) return -1;
        sizes[i] = (uint32_t)bytes[i];
        if (i == binding_count - 1) {
            if (access[i] != METAL_TRAINING_ACCESS_READ) return -1;
            id<MTLBuffer> params = create_buffer(bytes[i], hosts[i]);
            if (!params) return -1;
            training_transients[training_transient_count++] = params;
            bound_buffers[i] = params;
        } else {
            MetalTrainingTensorSlot* slot = training_ensure_slot(
                hosts[i], bytes[i], access[i], is_weight ? is_weight[i] : 0);
            if (!slot || !slot->buffer) return -1;
            bound_buffers[i] = slot->buffer;
        }
    }

    if (training_ensure_encoder() != 0) return -1;
    @autoreleasepool {
        [training_encoder setComputePipelineState:kernel->pipeline];
        for (int i = 0; i < binding_count; i++) {
            [training_encoder setBuffer:bound_buffers[i] offset:0 atIndex:(NSUInteger)i];
        }
        [training_encoder setBytes:sizes
                            length:(NSUInteger)(sizeof(uint32_t) * binding_count)
                           atIndex:(NSUInteger)binding_count];
        MTLSize groups = MTLSizeMake((NSUInteger)groups_x, (NSUInteger)groups_y,
                                     (NSUInteger)groups_z);
        MTLSize threads = MTLSizeMake((NSUInteger)desc->wg_x, (NSUInteger)desc->wg_y,
                                      (NSUInteger)desc->wg_z);
        [training_encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        /* Encoder boundaries are explicit Metal memory-dependency points.
           Keep one command buffer for the whole backward pass, but do not
           rely on implementation-specific overlap ordering between dispatches. */
        [training_encoder endEncoding];
        VX_METAL_RELEASE(training_encoder);
    }
    for (int i = 0; i < binding_count - 1; i++) {
        if (access[i] & METAL_TRAINING_ACCESS_WRITE) {
            int index = training_find_slot(hosts[i]);
            if (index >= 0) {
                training_slots[index].device_dirty = 1;
                if (training_slots[index].graph_slot) {
                    training_slots[index].graph_slot->device_dirty = 1;
                    training_slots[index].graph_slot->host_dirty = 0;
                }
            }
        }
    }
    return 0;
}

int metal_training_sync(void* host, size_t bytes) {
    if (!training_active || !host || bytes == 0) return -1;
    if (training_flush_commands() != 0) return -1;
    int index = training_find_slot(host);
    if (index < 0) return -1;
    MetalTrainingTensorSlot* slot = &training_slots[index];
    if (!slot->buffer || bytes > slot->bytes) return -1;
    if (slot->device_dirty) {
        memcpy(host, [slot->buffer contents], bytes);
        slot->device_dirty = 0;
        if (slot->graph_slot) {
            slot->graph_slot->device_dirty = 0;
            slot->graph_slot->host_dirty = 0;
        }
    }
    return 0;
}

void metal_training_end(void) {
    if (!training_active) return;
    (void)training_flush_commands();
    for (int i = 0; i < training_slot_count; i++) training_clear_slot(&training_slots[i]);
    training_slot_count = 0;
    training_active = 0;
}
#endif

void metal_graph_reset(void) {
    MetalContextState* state = metal_context_state_get(0);
    if (!state) return;
#if VOLVOXAI_ENABLE_TRAINING
    /* Training slots may retain aliases into graph_slots. End the lazy
       training session first so reset cannot invalidate a live alias. */
    if (training_active) metal_training_end();
#endif
    (void)graph_flush_commands();
    graph_forward_active = 0;
    graph_forward_error = 0;
    for (int i = 0; i < graph_slot_count; i++) clear_slot(&graph_slots[i]);
    graph_slot_count = 0;
    free(state->shape_signature);
    state->shape_signature = NULL;
    state->shape_generation = metal_generation_next(state->shape_generation);
    state->capacity_generation =
        metal_generation_next(state->capacity_generation);
    state->domain_span_count = 0;
    state->domain_qgroupnorm_stats_bytes = 0;
    state->domain_qlayernorm_stats_bytes = 0;
    state->domain_enforced = 0;
#ifdef VOLVOX_METAL_TESTING
    state->test_domain_allocation_failure_after = -1;
#endif
}

int metal_graph_bind_shape(const char* signature) {
    MetalContextState* state = metal_context_state_get(0);
    char* candidate;
    size_t length;
    uint64_t next_generation;
    if (!state || !metal_ready() || !signature || !signature[0] ||
        state->domain_enforced) return -1;
    if (state->shape_signature && !strcmp(state->shape_signature, signature))
        return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (training_active) return -1;
#endif
    length = strlen(signature);
    if (length > 1024u * 1024u) return -1;
    candidate = (char*)malloc(length + 1u);
    if (!candidate) return -1;
    memcpy(candidate, signature, length + 1u);
    if (graph_flush_commands() != 0) {
        free(candidate);
        return -1;
    }
    graph_forward_active = 0;
    graph_forward_error = 0;
    next_generation = metal_generation_next(state->shape_generation);
    for (int index = 0; index < graph_slot_count; index++) {
        MetalTensorSlot* slot = &graph_slots[index];
        if (slot->is_weight) continue;
        slot->host = NULL;
        slot->bytes = 0;
        slot->shape_generation = next_generation;
        slot->host_dirty = 0;
        slot->device_dirty = 0;
        if (slot->is_alias) {
            VX_METAL_RELEASE(slot->buffer);
            slot->cap = 0;
            slot->capacity_generation = 0;
            slot->is_alias = 0;
        }
    }
    free(state->shape_signature);
    state->shape_signature = candidate;
    state->shape_generation = next_generation;
    return 0;
}

static int metal_graph_domain_spans_match(
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count) {
    MetalContextState* state = metal_context_state_get(0);
    if (!state || !spans || span_count != state->domain_span_count) return 0;
    for (size_t index = 0; index < span_count; index++) {
        int slot_index = graph_find_slot(spans[index].host);
        MetalTensorSlot* slot;
        if (slot_index < 0) return 0;
        slot = &graph_slots[slot_index];
        if (!slot->domain_span || slot->is_weight || !slot->buffer ||
            slot->is_alias || slot->domain_capacity !=
                spans[index].capacity_bytes ||
            slot->bytes > slot->domain_capacity)
            return 0;
    }
    return 1;
}

static int metal_domain_candidate_allocation_allowed(
        MetalContextState* state) {
#ifdef VOLVOX_METAL_TESTING
    if (state->test_domain_allocation_failure_after == 0) {
        state->test_domain_allocation_failure_after = -1;
        return 0;
    }
    if (state->test_domain_allocation_failure_after > 0)
        state->test_domain_allocation_failure_after--;
#else
    (void)state;
#endif
    return 1;
}

int metal_graph_bind_shape_domain(
        const char* signature,
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count,
        size_t qgroupnorm_stats_bytes,
        size_t qlayernorm_stats_bytes) {
    MetalContextState* state = metal_context_state_get(0);
    char* candidate_signature = NULL;
    MetalTensorSlot* candidate_slots = NULL;
    id<MTLBuffer> candidate_qgroupnorm_stats_buffer = nil;
    id<MTLBuffer> candidate_qlayernorm_stats_buffer = nil;
    size_t signature_length;
    int weight_count = 0;
    int result = -1;
    uint64_t next_shape_generation;
    uint64_t next_capacity_generation;
    if (!state || !metal_ready() || !signature || !signature[0] ||
        !spans || !span_count || span_count > METAL_GRAPH_MAX_TENSORS ||
        graph_forward_active)
        return -1;
#if VOLVOXAI_ENABLE_TRAINING
    if (training_active) return -1;
#endif
    signature_length = strlen(signature);
    if (signature_length > 1024u * 1024u) return -1;
    if (qgroupnorm_stats_bytes > SIZE_MAX - qlayernorm_stats_bytes ||
        (qgroupnorm_stats_bytes &&
         !metal_buffer_size_valid(qgroupnorm_stats_bytes)) ||
        (qlayernorm_stats_bytes &&
         !metal_buffer_size_valid(qlayernorm_stats_bytes)))
        return -1;
    for (size_t index = 0; index < span_count; index++) {
        uintptr_t start = (uintptr_t)spans[index].host;
        size_t capacity = spans[index].capacity_bytes;
        if (!start || !capacity || start > UINTPTR_MAX - capacity ||
            !metal_buffer_size_valid(capacity) ||
            (index && (uintptr_t)spans[index - 1u].host +
                spans[index - 1u].capacity_bytes > start))
            return -1;
    }
    if (state->domain_enforced) {
        if (!metal_graph_domain_spans_match(spans, span_count) ||
            qgroupnorm_stats_bytes !=
                state->domain_qgroupnorm_stats_bytes ||
            qlayernorm_stats_bytes !=
                state->domain_qlayernorm_stats_bytes)
            return -1;
        if (state->shape_signature &&
            !strcmp(state->shape_signature, signature)) {
            for (int index = 0; index < graph_slot_count; index++) {
                MetalTensorSlot* slot = &graph_slots[index];
                if (!slot->domain_span) continue;
                slot->bytes = 0;
                slot->host_dirty = 0;
                slot->device_dirty = 0;
            }
            return 0;
        }
        candidate_signature = (char*)malloc(signature_length + 1u);
        if (!candidate_signature) return -2;
        memcpy(candidate_signature, signature, signature_length + 1u);
        if (graph_flush_commands() != 0) goto done;
        graph_forward_error = 0;
        next_shape_generation =
            metal_generation_next(state->shape_generation);
        for (int index = 0; index < graph_slot_count; index++) {
            MetalTensorSlot* slot = &graph_slots[index];
            if (!slot->domain_span) continue;
            slot->shape_generation = next_shape_generation;
            slot->bytes = 0;
            slot->host_dirty = 0;
            slot->device_dirty = 0;
        }
        free(state->shape_signature);
        state->shape_signature = candidate_signature;
        candidate_signature = NULL;
        state->shape_generation = next_shape_generation;
        result = 0;
        goto done;
    }

    for (int index = 0; index < graph_slot_count; index++) {
        MetalTensorSlot* slot = &graph_slots[index];
        if (!slot->is_weight) continue;
        uintptr_t weight_start = (uintptr_t)slot->host;
        if (!weight_start || !slot->buffer || slot->is_alias ||
            !slot->bytes || slot->bytes > slot->cap ||
            weight_start > UINTPTR_MAX - slot->bytes)
            return -1;
        weight_count++;
        for (size_t span_index = 0; span_index < span_count; span_index++) {
            uintptr_t span_start = (uintptr_t)spans[span_index].host;
            uintptr_t span_end = span_start + spans[span_index].capacity_bytes;
            uintptr_t weight_end = weight_start + slot->bytes;
            if (span_start < weight_end && weight_start < span_end)
                return -1;
        }
    }
    if (span_count > (size_t)(METAL_GRAPH_MAX_TENSORS - weight_count))
        return -1;
    candidate_signature = (char*)malloc(signature_length + 1u);
    candidate_slots =
        (MetalTensorSlot*)calloc(span_count, sizeof(*candidate_slots));
    if (!candidate_signature || !candidate_slots) {
        result = -2;
        goto done;
    }
    memcpy(candidate_signature, signature, signature_length + 1u);
    if (graph_flush_commands() != 0) goto done;
    graph_forward_error = 0;
    @autoreleasepool {
        for (size_t index = 0; index < span_count; index++) {
            if (!metal_domain_candidate_allocation_allowed(state)) {
                result = -2;
                break;
            }
            candidate_slots[index].buffer =
                create_buffer(spans[index].capacity_bytes, NULL);
            if (!candidate_slots[index].buffer) {
                result = -2;
                break;
            }
        }
        if (result != -2 && qgroupnorm_stats_bytes) {
            if (!metal_domain_candidate_allocation_allowed(state)) {
                result = -2;
            } else {
                candidate_qgroupnorm_stats_buffer =
                    create_buffer(qgroupnorm_stats_bytes, NULL);
                if (!candidate_qgroupnorm_stats_buffer) result = -2;
            }
        }
        if (result != -2 && qlayernorm_stats_bytes) {
            if (!metal_domain_candidate_allocation_allowed(state)) {
                result = -2;
            } else {
                candidate_qlayernorm_stats_buffer =
                    create_buffer(qlayernorm_stats_bytes, NULL);
                if (!candidate_qlayernorm_stats_buffer) result = -2;
            }
        }
    }
    if (result == -2) goto done;

    next_shape_generation =
        metal_generation_next(state->shape_generation);
    next_capacity_generation =
        metal_generation_next(state->capacity_generation);
    for (int index = 0; index < graph_slot_count; index++) {
        if (!graph_slots[index].is_weight) clear_slot(&graph_slots[index]);
    }
    VX_METAL_RELEASE(state->qgroupnorm_scratch_buffer);
    state->qgroupnorm_scratch_buffer = candidate_qgroupnorm_stats_buffer;
    state->qgroupnorm_scratch_capacity = qgroupnorm_stats_bytes;
    candidate_qgroupnorm_stats_buffer = nil;
    VX_METAL_RELEASE(state->qlayernorm_scratch_buffer);
    state->qlayernorm_scratch_buffer = candidate_qlayernorm_stats_buffer;
    state->qlayernorm_scratch_capacity = qlayernorm_stats_bytes;
    candidate_qlayernorm_stats_buffer = nil;
    {
        int kept = 0;
        int old_slot_count = graph_slot_count;
        for (int index = 0; index < old_slot_count; index++) {
            if (!graph_slots[index].is_weight) continue;
            if (kept != index) graph_slots[kept] = graph_slots[index];
            kept++;
        }
        if (kept < old_slot_count) {
            memset(&graph_slots[kept], 0,
                   (size_t)(old_slot_count - kept) * sizeof(graph_slots[0]));
        }
        graph_slot_count = kept;
    }
    for (size_t index = 0; index < span_count; index++) {
        MetalTensorSlot* slot = &graph_slots[graph_slot_count++];
        memset(slot, 0, sizeof(*slot));
        slot->host = spans[index].host;
        slot->cap = spans[index].capacity_bytes;
        slot->domain_capacity = spans[index].capacity_bytes;
        slot->buffer = candidate_slots[index].buffer;
        candidate_slots[index].buffer = nil;
        slot->shape_generation = next_shape_generation;
        slot->capacity_generation = next_capacity_generation;
        slot->domain_span = 1;
    }
    free(state->shape_signature);
    state->shape_signature = candidate_signature;
    candidate_signature = NULL;
    state->shape_generation = next_shape_generation;
    state->capacity_generation = next_capacity_generation;
    state->domain_span_count = span_count;
    state->domain_qgroupnorm_stats_bytes = qgroupnorm_stats_bytes;
    state->domain_qlayernorm_stats_bytes = qlayernorm_stats_bytes;
    state->domain_enforced = 1;
    result = 0;

done:
    if (candidate_slots) {
        for (size_t index = 0; index < span_count; index++)
            clear_slot(&candidate_slots[index]);
    }
    VX_METAL_RELEASE(candidate_qgroupnorm_stats_buffer);
    VX_METAL_RELEASE(candidate_qlayernorm_stats_buffer);
    free(candidate_slots);
    free(candidate_signature);
    return result;
}

void metal_graph_begin_forward(void) {
    graph_forward_error = graph_flush_commands() != 0;
    graph_forward_active = 1;
}

int metal_graph_end_forward(void) {
    int rc = graph_forward_error ? -1 : 0;
    if (graph_flush_commands() != 0) rc = -1;
    graph_forward_active = 0;
    graph_forward_error = 0;
    return rc;
}

void metal_graph_mark_host(const void* host, size_t bytes, int is_weight) {
    MetalTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return;
    /* A CPU fallback may overwrite an arena address whose Metal buffer is
     * still referenced by an earlier encoded dispatch. Complete that segment
     * before making the host copy authoritative; otherwise a later upload
     * could race the older command's access to the same shared buffer. */
    if (graph_buffer_has_pending_use(s->buffer) && graph_flush_commands() != 0) {
        if (graph_forward_active) graph_forward_error = 1;
        return;
    }
    s->host_dirty = 1;
    s->device_dirty = 0;
}

int metal_graph_sync_host(const void* host, size_t bytes, int is_weight) {
    (void)is_weight;
    if (!metal_ready() || !host || bytes == 0) return 0;
    int idx = graph_find_slot(host);
    /* No tracked slot means Metal has no newer copy; the host allocation is
       already authoritative and synchronization succeeds without a transfer. */
    if (idx < 0) return 1;
    MetalTensorSlot* s = &graph_slots[idx];
    if (!s->buffer || bytes > s->bytes) return 0;
    if (s->device_dirty) {
        if (graph_flush_commands() != 0) {
            graph_forward_error = graph_forward_active ? 1 : 0;
            return 0;
        }
        memcpy((void*)host, [s->buffer contents], bytes);
        s->device_dirty = 0;
        s->host_dirty = 0;
    }
    return 1;
}

void metal_graph_retain_weight(const void* host, size_t bytes) {
    MetalContextState* state = metal_context_state_get(0);
    if (!state || !host || !bytes) return;
    int index = graph_find_slot(host);
    if (index < 0) return;
    MetalTensorSlot* slot = &graph_slots[index];
    if (slot->domain_span || !slot->buffer || slot->is_alias ||
        !slot->bytes || bytes > slot->bytes || bytes > slot->cap)
        return;
    slot->is_weight = 1;
    slot->shape_generation = 0;
}

void metal_graph_demote_weight(const void* host, size_t bytes) {
    MetalContextState* state = metal_context_state_get(0);
    int index;
    MetalTensorSlot* slot;
    if (!state || !host || !bytes) return;
    index = graph_find_slot(host);
    if (index < 0) return;
    slot = &graph_slots[index];
    if (slot->domain_span || !slot->bytes || bytes > slot->bytes)
        return;
    slot->is_weight = 0;
    slot->shape_generation = state->shape_generation;
}

#ifdef VOLVOX_METAL_TESTING
void metal_graph_debug_reset_counters(void) {
    graph_debug_dispatch_count = 0;
    graph_debug_commit_count = 0;
    graph_debug_wait_count = 0;
}

void metal_graph_debug_counters(uint64_t* dispatches, uint64_t* commits,
                                uint64_t* waits) {
    if (dispatches) *dispatches = graph_debug_dispatch_count;
    if (commits) *commits = graph_debug_commit_count;
    if (waits) *waits = graph_debug_wait_count;
}

int metal_graph_debug_dynamic_state(MetalGraphDynamicStateProbe* probe) {
    MetalContextState* state = metal_context_state_get(0);
    if (!state || !probe) return -1;
    memset(probe, 0, sizeof(*probe));
    probe->shape_generation = state->shape_generation;
    probe->capacity_generation = state->capacity_generation;
    probe->domain_span_count = state->domain_span_count;
    probe->domain_scratch_capacity_bytes =
        state->qgroupnorm_scratch_capacity +
        state->qlayernorm_scratch_capacity;
    probe->domain_enforced = state->domain_enforced;
    probe->slot_count = state->graph_slots_count;
    for (int index = 0; index < state->graph_slots_count; index++) {
        MetalTensorSlot* slot = &state->graph_slot_storage[index];
        if (!slot->buffer || slot->is_alias) continue;
        if (slot->host) probe->active_capacity_bytes += slot->cap;
        else probe->pooled_capacity_bytes += slot->cap;
    }
    return 0;
}

int metal_test_fail_domain_allocation_after(size_t successful_allocations) {
    MetalContextState* state = metal_context_state_get(0);
    if (!state || successful_allocations > (size_t)INT_MAX) return -1;
    state->test_domain_allocation_failure_after =
        (int)successful_allocations;
    return 0;
}
#endif

static void metal_context_resources_release(MetalContextState* state) {
    if (!state) return;
    @autoreleasepool {
#if VOLVOXAI_ENABLE_TRAINING
        if (state->training_command_encoder) {
            [state->training_command_encoder endEncoding];
            VX_METAL_RELEASE(state->training_command_encoder);
        }
        if (state->training_command_buffer) {
            [state->training_command_buffer commit];
            [state->training_command_buffer waitUntilCompleted];
            VX_METAL_RELEASE(state->training_command_buffer);
        }
        for (int index = 0; index < state->training_transients_count; index++)
            VX_METAL_RELEASE(state->training_transient_storage[index]);
        state->training_transients_count = 0;
        for (int index = 0; index < state->training_slots_count; index++)
            training_clear_slot(&state->training_slot_storage[index]);
        state->training_slots_count = 0;
        state->training_is_active = 0;
        for (int index = 0; index < state->training_kernels_count; index++) {
            VX_METAL_RELEASE(state->training_kernel_storage[index].pipeline);
            state->training_kernel_storage[index].shader[0] = 0;
            state->training_kernel_storage[index].entry[0] = 0;
        }
        state->training_kernels_count = 0;
#endif
        if (state->graph_command_buffer) {
            [state->graph_command_buffer commit];
            [state->graph_command_buffer waitUntilCompleted];
            VX_METAL_RELEASE(state->graph_command_buffer);
        }
        for (int index = 0;
             index < state->graph_retained_bindings_count; index++)
            VX_METAL_RELEASE(state->graph_retained_binding_storage[index]);
        state->graph_retained_bindings_count = 0;
        state->graph_is_forward_active = 0;
        state->graph_forward_failed = 0;
        for (int index = 0; index < state->graph_slots_count; index++)
            clear_slot(&state->graph_slot_storage[index]);
        state->graph_slots_count = 0;
        VX_METAL_RELEASE(state->qgroupnorm_scratch_buffer);
        state->qgroupnorm_scratch_capacity = 0;
        VX_METAL_RELEASE(state->qlayernorm_scratch_buffer);
        state->qlayernorm_scratch_capacity = 0;
        qconv_zero_bias_release_state(state);
        free(state->shape_signature);
        state->shape_signature = NULL;
        state->domain_span_count = 0;
        state->domain_qgroupnorm_stats_bytes = 0;
        state->domain_qlayernorm_stats_bytes = 0;
        state->domain_enforced = 0;
    }
}

static void metal_device_shutdown_locked(void) {
    @autoreleasepool {
        for (size_t index = 0;
             index < g_metal_device_state.pipeline_count; index++) {
            VX_METAL_RELEASE(g_metal_device_state.pipelines[index].pipeline);
            g_metal_device_state.pipelines[index].descriptor = NULL;
            g_metal_device_state.pipelines[index].failed = 0;
        }
        g_metal_device_state.pipeline_count = 0;
        VX_METAL_RELEASE(commandQueue);
        VX_METAL_RELEASE(device);
    }
    g_metal_device_state.reference_count = 0;
}

static void metal_context_state_destroy(void* opaque_state) {
    MetalContextState* state = (MetalContextState*)opaque_state;
    if (!state) return;
    metal_context_resources_release(state);
    metal_device_lock();
    if (state->device_acquired &&
        g_metal_device_state.reference_count != 0) {
        state->device_acquired = 0;
        g_metal_device_state.reference_count--;
        if (g_metal_device_state.reference_count == 0)
            metal_device_shutdown_locked();
    }
    metal_device_unlock();
    free(state);
}

void metal_cleanup(void) {
    VxEngineState* owner = vx_engine_state_current();
    if (!owner || !owner->metal_context_state) return;
    MetalContextState* state =
        (MetalContextState*)owner->metal_context_state;
    owner->metal_context_state = NULL;
    owner->metal_context_state_destroy = NULL;
    metal_context_state_destroy(state);
}

/* See opengl_graph_alias_f32: device slots are keyed by host pointer and the
 * runtime pools many disjoint-lifetime tensors into one host span, so aliasing
 * across distinct host storage merges two spans on the device instead of
 * extending one tensor's lifetime. Distinct host storage must be a real copy. */
int metal_graph_alias_f32(const float* in, float* out, long n) {
    if (!in || !out || n <= 0 || (uint64_t)n > UINT32_MAX) return 0;
    if (in != out) return metal_graph_copy_f32(in, out, n);
    size_t bytes = (size_t)n * sizeof(float);
    MetalTensorSlot* slot = graph_ensure_device(in, bytes, 0);
    if (!slot || !slot->buffer) return 0;
    graph_mark_device(slot);
    return 1;
}

int metal_graph_copy_f32(const float* in, float* out, long n) {
    if (!in || !out || n <= 0 || (uint64_t)n > UINT32_MAX) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, bytes}, {dst->buffer, bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_copy, binds, ((uint32_t)n + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_add_f32(const float* a, const float* b, float* out, long n) {
    if (!a || !b || !out || n <= 0 || (uint64_t)n > UINT32_MAX) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    MetalTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    MetalTensorSlot* sb = graph_ensure_device(b, bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!sa || !sb || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {sa->buffer, bytes}, {sb->buffer, bytes}, {dst->buffer, bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_add, binds, ((uint32_t)n + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static int metal_graph_typed_control_32(
        const void* a, size_t a_bytes, const void* b, size_t b_bytes,
        void* output, size_t output_bytes,
        const VxTypedControlMetadata* metadata) {
    if (!metadata) return 0;
    MetalTensorSlot* a_slot = graph_ensure_device(a, a_bytes, 0);
    MetalTensorSlot* b_slot = graph_ensure_device(b, b_bytes, 0);
    MetalTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    id<MTLBuffer> metadata_buffer =
        create_buffer(sizeof(*metadata), metadata);
    if (!metadata_buffer) return 0;
    MetalBinding bindings[4] = {
        {a_slot->buffer, a_bytes},
        {b_slot->buffer, b_bytes},
        {output_slot->buffer, output_bytes},
        {metadata_buffer, sizeof(*metadata)},
    };
    uint32_t elements = metadata->values[0];
    int ok = dispatch_kernel(
        &k_typed_control_32, bindings,
        (elements + 63u) / 64u, 1u, 1u);
    VX_METAL_RELEASE(metadata_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_compare_i32(
        const int32_t* a, long a_elements,
        const int32_t* b, long b_elements,
        int32_t* output, long output_elements,
        const uint32_t* output_strides,
        const uint32_t* a_strides,
        const uint32_t* b_strides,
        int rank, int operation) {
    VxTypedControlMetadata metadata;
    size_t a_bytes;
    size_t b_bytes;
    size_t output_bytes;
    if (!vx_typed_control_compare_plan(
            a, a_elements, b, b_elements, output, output_elements,
            output_strides, a_strides, b_strides, rank, operation,
            &metadata, &a_bytes, &b_bytes, &output_bytes))
        return 0;
    return metal_graph_typed_control_32(
        a, a_bytes, b, b_bytes, output, output_bytes, &metadata);
}

static int metal_graph_unary_i32(
        const int32_t* input, int32_t* output, long elements,
        int operation, int32_t minimum, int32_t maximum) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_unary_plan(
            input, output, elements, operation, minimum, maximum,
            &metadata, &bytes))
        return 0;
    return metal_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int metal_graph_not_i32(
        const int32_t* input, int32_t* output, long elements) {
    return metal_graph_unary_i32(
        input, output, elements, VX_TYPED_CONTROL_NOT_I32, 0, 0);
}

int metal_graph_clip_i32(
        const int32_t* input, int32_t* output, long elements,
        int32_t minimum, int32_t maximum) {
    return metal_graph_unary_i32(
        input, output, elements, VX_TYPED_CONTROL_CLIP_I32,
        minimum, maximum);
}

int metal_graph_copy_32(
        const void* input, void* output, long elements) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_copy_plan(
            input, output, elements, &metadata, &bytes))
        return 0;
    return metal_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int metal_graph_cast_typed(
        const void* input, int input_dtype,
        void* output, int output_dtype, long elements) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_cast_plan(
            input, input_dtype, output, output_dtype, elements,
            &metadata, &bytes))
        return 0;
    return metal_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int metal_graph_where_32(
        const int32_t* condition, const void* a,
        const void* b, void* output, long elements) {
    uint32_t count;
    size_t bytes;
    if (!vx_typed_control_where_plan(
            condition, a, b, output, elements, &count, &bytes))
        return 0;
    MetalTensorSlot* condition_slot =
        graph_ensure_device(condition, bytes, 0);
    MetalTensorSlot* a_slot = graph_ensure_device(a, bytes, 0);
    MetalTensorSlot* b_slot = graph_ensure_device(b, bytes, 0);
    MetalTensorSlot* output_slot =
        graph_output_slot(output, bytes);
    if (!condition_slot || !a_slot || !b_slot || !output_slot) return 0;
    uint32_t params[4] = {count, 0u, 0u, 0u};
    id<MTLBuffer> params_buffer_handle =
        create_buffer(sizeof(params), params);
    if (!params_buffer_handle) return 0;
    MetalBinding bindings[5] = {
        {condition_slot->buffer, bytes},
        {a_slot->buffer, bytes},
        {b_slot->buffer, bytes},
        {output_slot->buffer, bytes},
        {params_buffer_handle, sizeof(params)},
    };
    int ok = dispatch_kernel(
        &k_where_32, bindings, (count + 63u) / 64u, 1u, 1u);
    VX_METAL_RELEASE(params_buffer_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_argmax_f32(
        const float* input, int32_t* output,
        uint32_t outer, uint32_t axis_size, uint32_t inner) {
    uint32_t input_elements;
    uint32_t output_elements;
    size_t input_bytes;
    size_t output_bytes;
    if (!vx_typed_control_argmax_plan(
            input, output, outer, axis_size, inner,
            &input_elements, &output_elements,
            &input_bytes, &output_bytes))
        return 0;
    MetalTensorSlot* input_slot =
        graph_ensure_device(input, input_bytes, 0);
    MetalTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    uint32_t params[4] = {outer, axis_size, inner, 0u};
    id<MTLBuffer> params_buffer_handle =
        create_buffer(sizeof(params), params);
    if (!params_buffer_handle) return 0;
    MetalBinding bindings[3] = {
        {input_slot->buffer, input_bytes},
        {output_slot->buffer, output_bytes},
        {params_buffer_handle, sizeof(params)},
    };
    int ok = dispatch_kernel(
        &k_argmax_f32_i32, bindings,
        (output_elements + 63u) / 64u, 1u, 1u);
    VX_METAL_RELEASE(params_buffer_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    (void)input_elements;
    return 1;
}

static uint32_t metal_groups_64(uint32_t elements) {
    return elements / 64u + (elements % 64u != 0u);
}

static int metal_f32_shape_size(const int* shape, int rank, int max_rank,
                                uint32_t* elements, size_t* bytes) {
    uint64_t product = 1u;
    uint64_t max_elements = (uint64_t)UINT32_MAX / sizeof(float);
    if (!shape || !elements || !bytes || rank <= 0 || rank > max_rank)
        return 0;
    if ((uint64_t)SIZE_MAX / sizeof(float) < max_elements)
        max_elements = (uint64_t)SIZE_MAX / sizeof(float);
    for (int dimension = 0; dimension < rank; dimension++) {
        uint32_t extent;
        if (shape[dimension] <= 0) return 0;
        extent = (uint32_t)shape[dimension];
        if (product > max_elements / extent) return 0;
        product *= extent;
    }
    *elements = (uint32_t)product;
    *bytes = (size_t)product * sizeof(float);
    return 1;
}

static int metal_graph_unary_f32(const MetalKernel* kernel, const float* in, float* out, long n) {
    if (!kernel || !in || !out || n <= 0 || (uint64_t)n > UINT32_MAX) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, bytes}, {dst->buffer, bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(kernel, binds, ((uint32_t)n + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_sigmoid_f32(const float* in, float* out, long n) {
    if (n <= 0 ||
        (uint64_t)n > (uint64_t)UINT32_MAX / sizeof(float) ||
        (uint64_t)n > (uint64_t)SIZE_MAX / sizeof(float))
        return 0;
    return metal_graph_unary_f32(&k_sigmoid, in, out, n);
}

int metal_graph_gelu_f32(const float* in, float* out, long n, int approximate_tanh) {
    if (!in || !out || n <= 0 || (uint64_t)n > UINT32_MAX ||
        (approximate_tanh != 0 && approximate_tanh != 1)) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)approximate_tanh, 0u, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, bytes}, {dst->buffer, bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_gelu, binds, ((uint32_t)n + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_silu_f32(const float* in, float* out, long n) {
    return metal_graph_unary_f32(&k_silu, in, out, n);
}

int metal_graph_layernorm_f32(const float* in, const float* weight, const float* bias,
                              float* out, int rows, int d_model, float eps) {
    if (!in || !weight || !bias || !out || rows <= 0 || d_model <= 0 || !(eps > 0.0f)) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t parameter_bytes = (size_t)d_model * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(weight, parameter_bytes, 1);
    MetalTensorSlot* sb = graph_ensure_device(bias, parameter_bytes, 1);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !sw || !sb || !dst) return 0;
    struct { uint32_t rows, d_model; float eps; uint32_t pad; } params = {
        (uint32_t)rows, (uint32_t)d_model, eps, 0u
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[5] = {
        {src->buffer, bytes}, {sw->buffer, parameter_bytes}, {sb->buffer, parameter_bytes},
        {dst->buffer, bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_layernorm, binds, ((uint32_t)rows + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_groupnorm_f32(const float* in, const float* weight, const float* bias,
                              float* out, int n, int h, int w, int c, int groups,
                              float eps) {
    if (!in || !weight || !bias || !out || n <= 0 || h <= 0 || w <= 0 || c <= 0 ||
        groups <= 0 || c % groups || !(eps > 0.0f)) return 0;
    size_t bytes = (size_t)n * h * w * c * sizeof(float);
    size_t parameter_bytes = (size_t)c * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(weight, parameter_bytes, 1);
    MetalTensorSlot* sb = graph_ensure_device(bias, parameter_bytes, 1);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !sw || !sb || !dst) return 0;
    struct {
        uint32_t n, h, w, c, groups, has_bias;
        float eps;
        uint32_t pad;
    } params = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                (uint32_t)groups, 1u, eps, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[5] = {
        {src->buffer, bytes}, {sw->buffer, parameter_bytes}, {sb->buffer, parameter_bytes},
        {dst->buffer, bytes}, {pb, sizeof(params)}
    };
    uint32_t total_groups = (uint32_t)n * (uint32_t)groups;
    int ok = dispatch_kernel(&k_groupnorm, binds, (total_groups + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int metal_graph_dropout_f32(const float* in, float* out, long n, uint32_t threshold,
                            uint32_t seed, uint32_t counter, float scale) {
    if (!in || !out || n <= 0 || (uint64_t)n > UINT32_MAX || !(scale >= 1.0f)) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct {
        uint32_t length, threshold, seed, counter;
        float scale;
        uint32_t pad[3];
    } params = {(uint32_t)n, threshold, seed, counter, scale, {0u, 0u, 0u}};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, bytes}, {dst->buffer, bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_dropout, binds, ((uint32_t)n + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int metal_graph_softmax_f32(const float* in, float* out, int rows, int d) {
    uint64_t elements;
    uint64_t max_elements = (uint64_t)UINT32_MAX / sizeof(float);
    if (!in || !out || rows <= 0 || d <= 0) return 0;
    if ((uint64_t)SIZE_MAX / sizeof(float) < max_elements)
        max_elements = (uint64_t)SIZE_MAX / sizeof(float);
    if ((uint64_t)rows > max_elements / (uint32_t)d) return 0;
    elements = (uint64_t)(uint32_t)rows * (uint32_t)d;
    size_t bytes = (size_t)elements * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[2] = {(uint32_t)rows, (uint32_t)d};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[3] = {
        {src->buffer, bytes}, {dst->buffer, bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_softmax, binds,
                             metal_groups_64((uint32_t)rows), 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_reduce_f32(const float* in, float* out, int rows, int width, float scale) {
    if (!in || !out || rows <= 0 || width <= 0 || !isfinite(scale)) return 0;
    size_t input_bytes = (size_t)rows * width * sizeof(float);
    size_t output_bytes = (size_t)rows * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, input_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, output_bytes);
    if (!src || !dst) return 0;
    struct { uint32_t rows, width; float scale; uint32_t pad; } params = {
        (uint32_t)rows, (uint32_t)width, scale, 0u
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {
        {src->buffer, input_bytes}, {dst->buffer, output_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_reduce, binds, ((uint32_t)rows + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_transpose_f32(const float* in, float* out, const int* in_shape,
                              const int* perm, int rank) {
    if (!in || !out || !in_shape || !perm || rank <= 0 || rank > 8) return 0;
    uint32_t in_stride[8] = {0}, out_shape[8] = {0}, out_stride[8] = {0};
    unsigned seen = 0;
    uint64_t total = 1;
    for (int i = 0; i < rank; i++) {
        if (in_shape[i] <= 0 || perm[i] < 0 || perm[i] >= rank ||
            (seen & (1u << perm[i]))) return 0;
        seen |= 1u << perm[i];
        out_shape[i] = (uint32_t)in_shape[perm[i]];
        total *= out_shape[i];
        if (total > UINT32_MAX) return 0;
    }
    in_stride[rank - 1] = out_stride[rank - 1] = 1u;
    for (int i = rank - 2; i >= 0; i--) {
        in_stride[i] = in_stride[i + 1] * (uint32_t)in_shape[i + 1];
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }
    size_t bytes = (size_t)total * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t metadata[18] = {0};
    metadata[0] = (uint32_t)total;
    metadata[1] = (uint32_t)rank;
    for (int dimension = 0; dimension < rank; dimension++) {
        metadata[2 + dimension] = out_stride[dimension];
        metadata[2 + rank + dimension] = in_stride[perm[dimension]];
    }
    size_t metadata_bytes = (size_t)(2 + 2 * rank) * sizeof(uint32_t);
    id<MTLBuffer> mb = create_buffer(metadata_bytes, metadata);
    if (!mb) return 0;
    MetalBinding binds[3] = {{src->buffer, bytes}, {dst->buffer, bytes}, {mb, metadata_bytes}};
    int ok = dispatch_kernel(&k_transpose, binds, ((uint32_t)total + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(mb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_expand_f32(const float* in, float* out, const int* in_shape,
                           int in_rank, const int* out_shape, int out_rank) {
    VxExpandF32Plan plan;
    if (!vx_expand_f32_plan(
            in, out, in_shape, in_rank, out_shape, out_rank, &plan))
        return 0;
    MetalTensorSlot* src = graph_ensure_device(in, plan.input_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, plan.output_bytes);
    if (!src || !dst) return 0;
    id<MTLBuffer> pb = create_buffer(sizeof(plan.params), plan.params);
    if (!pb) return 0;
    MetalBinding binds[3] = {
        {src->buffer, plan.input_bytes},
        {dst->buffer, plan.output_bytes},
        {pb, sizeof(plan.params)},
    };
    int ok = dispatch_kernel(&k_expand, binds,
                             metal_groups_64(plan.output_elements), 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_expand_32(const void* in, void* out, const int* in_shape,
                          int in_rank, const int* out_shape, int out_rank) {
    /* expand.wgsl uses u32 storage and is the shared F32/I32 word-copy route. */
    return metal_graph_expand_f32(
        (const float*)in, (float*)out,
        in_shape, in_rank, out_shape, out_rank);
}

int metal_graph_batch_matmul_f32(
        const float* a, const int* a_shape, int a_rank,
        const float* b, const int* b_shape, int b_rank,
        float* output, const int* output_shape, int output_rank) {
    VxBatchMatMulF32Plan plan;
    size_t metadata_bytes;
    if (!vx_batch_matmul_f32_plan(
            a, a_shape, a_rank, b, b_shape, b_rank,
            output, output_shape, output_rank, &plan))
        return 0;
    metadata_bytes = (size_t)plan.metadata_words * sizeof(uint32_t);
    MetalTensorSlot* a_slot = graph_ensure_device(a, plan.a_bytes, 0);
    MetalTensorSlot* b_slot = graph_ensure_device(b, plan.b_bytes, 0);
    MetalTensorSlot* output_slot =
        graph_output_slot(output, plan.output_bytes);
    id<MTLBuffer> metadata_buffer =
        create_buffer(metadata_bytes, plan.metadata);
    if (!a_slot || !b_slot || !output_slot || !metadata_buffer) {
        VX_METAL_RELEASE(metadata_buffer);
        return 0;
    }
    MetalBinding bindings[4] = {
        {a_slot->buffer, plan.a_bytes},
        {b_slot->buffer, plan.b_bytes},
        {output_slot->buffer, plan.output_bytes},
        {metadata_buffer, metadata_bytes},
    };
    int ok = dispatch_kernel(
        &k_batch_matmul, bindings,
        (plan.n + 7u) / 8u, (plan.m + 7u) / 8u,
        plan.output_batches);
    VX_METAL_RELEASE(metadata_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_gather_i32_f32(const float* input, const int32_t* indices,
                               float* output, int outer, int axis_size,
                               int inner, int indices_elements,
                               int output_elements) {
    uint64_t input_count;
    uint64_t expected_output;
    uint64_t max_f32_elements = (uint64_t)UINT32_MAX / sizeof(float);
    uint64_t max_i32_elements = (uint64_t)UINT32_MAX / sizeof(int32_t);
    if (!input || !indices || !output || input == output ||
        (const void*)indices == (const void*)output ||
        outer <= 0 || axis_size <= 0 || inner <= 0 ||
        indices_elements <= 0 || output_elements <= 0)
        return 0;
    if ((uint64_t)SIZE_MAX / sizeof(float) < max_f32_elements)
        max_f32_elements = (uint64_t)SIZE_MAX / sizeof(float);
    if ((uint64_t)SIZE_MAX / sizeof(int32_t) < max_i32_elements)
        max_i32_elements = (uint64_t)SIZE_MAX / sizeof(int32_t);
    if ((uint64_t)(uint32_t)outer >
        max_f32_elements / (uint32_t)axis_size)
        return 0;
    input_count = (uint64_t)(uint32_t)outer * (uint32_t)axis_size;
    if (input_count > max_f32_elements / (uint32_t)inner)
        return 0;
    input_count *= (uint32_t)inner;
    if ((uint64_t)(uint32_t)outer >
        max_f32_elements / (uint32_t)indices_elements)
        return 0;
    expected_output =
        (uint64_t)(uint32_t)outer * (uint32_t)indices_elements;
    if (expected_output > max_f32_elements / (uint32_t)inner)
        return 0;
    expected_output *= (uint32_t)inner;
    if (expected_output != (uint32_t)output_elements ||
        (uint32_t)indices_elements > max_i32_elements)
        return 0;
    size_t input_bytes = (size_t)input_count * sizeof(float);
    size_t index_bytes =
        (size_t)(uint32_t)indices_elements * sizeof(int32_t);
    size_t output_bytes = (size_t)expected_output * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(input, input_bytes, 0);
    MetalTensorSlot* idx = graph_ensure_device(indices, index_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(output, output_bytes);
    if (!src || !idx || !dst) return 0;
    uint32_t params[5] = {
        (uint32_t)outer, (uint32_t)axis_size, (uint32_t)inner,
        (uint32_t)indices_elements, (uint32_t)output_elements,
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {src->buffer, input_bytes},
        {idx->buffer, index_bytes},
        {dst->buffer, output_bytes},
        {pb, sizeof(params)},
    };
    int ok = dispatch_kernel(&k_gather, binds,
                             metal_groups_64((uint32_t)output_elements),
                             1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_slice4d_f32(const float* in, float* out, const int* in_shape,
                            int in_rank, const int* out_shape, int out_rank,
                            const int* starts, const int* steps) {
    uint32_t input_elements;
    uint32_t output_elements;
    size_t input_bytes;
    size_t output_bytes;
    uint32_t input4[4] = {1u, 1u, 1u, 1u};
    uint32_t output4[4] = {1u, 1u, 1u, 1u};
    if (!in || !out || in == out || !in_shape || !out_shape ||
        !starts || !steps || in_rank <= 0 || in_rank > 4 ||
        out_rank != in_rank ||
        !metal_f32_shape_size(in_shape, in_rank, 4,
                              &input_elements, &input_bytes) ||
        !metal_f32_shape_size(out_shape, out_rank, 4,
                              &output_elements, &output_bytes))
        return 0;
    int base = 4 - in_rank;
    for (int dimension = 0; dimension < in_rank; dimension++) {
        input4[base + dimension] = (uint32_t)in_shape[dimension];
        output4[base + dimension] = (uint32_t)out_shape[dimension];
    }
    for (int dimension = 0; dimension < 4; dimension++) {
        uint64_t start;
        uint64_t last;
        if (starts[dimension] < 0 || steps[dimension] <= 0)
            return 0;
        start = (uint32_t)starts[dimension];
        last = start +
            (uint64_t)(output4[dimension] - 1u) *
                (uint32_t)steps[dimension];
        if (start >= input4[dimension] || last >= input4[dimension])
            return 0;
    }
    MetalTensorSlot* src = graph_ensure_device(in, input_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, output_bytes);
    if (!src || !dst) return 0;
    uint32_t params[16] = {
        output4[0], output4[1], output4[2], output4[3],
        input4[1], input4[2], input4[3],
        (uint32_t)starts[0], (uint32_t)starts[1],
        (uint32_t)starts[2], (uint32_t)starts[3],
        (uint32_t)steps[0], (uint32_t)steps[1],
        (uint32_t)steps[2], (uint32_t)steps[3],
        output_elements,
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[3] = {
        {src->buffer, input_bytes},
        {dst->buffer, output_bytes},
        {pb, sizeof(params)},
    };
    int ok = dispatch_kernel(&k_slice, binds,
                             metal_groups_64(output_elements), 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    (void)input_elements;
    return 1;
}

int metal_graph_concat_f32(const float** inputs, const long* sizes, const int* input_axes,
                           int count, float* out, int output_axis, int inner) {
    if (!inputs || !sizes || !input_axes || !out || count <= 0 || output_axis <= 0 || inner <= 0) return 0;
    uint64_t total = 0;
    for (int input_index = 0; input_index < count; input_index++) {
        if (!inputs[input_index] || sizes[input_index] <= 0 || input_axes[input_index] <= 0) return 0;
        total += (uint64_t)sizes[input_index];
    }
    if (total > UINT32_MAX) return 0;
    size_t output_bytes = (size_t)total * sizeof(float);
    MetalTensorSlot* dst = graph_output_slot(out, output_bytes);
    if (!dst) return 0;
    uint32_t axis_offset = 0;
    for (int input_index = 0; input_index < count; input_index++) {
        size_t input_bytes = (size_t)sizes[input_index] * sizeof(float);
        MetalTensorSlot* src = graph_ensure_device(inputs[input_index], input_bytes, 0);
        if (!src) return 0;
        uint32_t params[8] = {(uint32_t)sizes[input_index], axis_offset,
                              (uint32_t)input_axes[input_index], (uint32_t)output_axis,
                              (uint32_t)inner, 0u, 0u, 0u};
        id<MTLBuffer> pb = create_buffer(sizeof(params), params);
        if (!pb) return 0;
        MetalBinding binds[3] = {{src->buffer, input_bytes}, {dst->buffer, output_bytes},
                                 {pb, sizeof(params)}};
        int ok = dispatch_kernel(&k_concat, binds,
                                 ((uint32_t)sizes[input_index] + 63u) / 64u, 1, 1);
        VX_METAL_RELEASE(pb);
        if (!ok) return 0;
        axis_offset += (uint32_t)input_axes[input_index];
    }
    if (axis_offset != (uint32_t)output_axis) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_concat_32(
        const void* const* inputs, const long* sizes,
        const int* input_axes, int count, void* output,
        int output_axis, int inner) {
    uint32_t output_elements;
    size_t output_bytes;
    if (!vx_typed_control_concat_plan(
            inputs, sizes, input_axes, count, output,
            output_axis, inner, &output_elements, &output_bytes))
        return 0;
    MetalTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!output_slot) return 0;
    uint32_t axis_offset = 0u;
    for (int index = 0; index < count; index++) {
        uint32_t input_elements = (uint32_t)sizes[index];
        size_t input_bytes =
            (size_t)input_elements * sizeof(uint32_t);
        MetalTensorSlot* input_slot =
            graph_ensure_device(inputs[index], input_bytes, 0);
        if (!input_slot) return 0;
        uint32_t params[8] = {
            input_elements, axis_offset,
            (uint32_t)input_axes[index], (uint32_t)output_axis,
            (uint32_t)inner, 0u, 0u, 0u,
        };
        id<MTLBuffer> params_buffer_handle =
            create_buffer(sizeof(params), params);
        if (!params_buffer_handle) return 0;
        MetalBinding bindings[3] = {
            {input_slot->buffer, input_bytes},
            {output_slot->buffer, output_bytes},
            {params_buffer_handle, sizeof(params)},
        };
        int ok = dispatch_kernel(
            &k_concat_32, bindings,
            (input_elements + 63u) / 64u, 1u, 1u);
        VX_METAL_RELEASE(params_buffer_handle);
        if (!ok) return 0;
        axis_offset += (uint32_t)input_axes[index];
    }
    graph_mark_device(output_slot);
    (void)output_elements;
    return 1;
}

int metal_graph_linear_f32(const float* in, const float* weight, const float* bias,
                           float* out, int rows, int d_in, int d_out, int out_in_layout) {
    if (!in || !weight || !out || rows <= 0 || d_in <= 0 || d_out <= 0) return 0;
    size_t input_bytes = (size_t)rows * d_in * sizeof(float);
    size_t weight_bytes = (size_t)d_in * d_out * sizeof(float);
    size_t bias_bytes = (size_t)d_out * sizeof(float);
    size_t output_bytes = (size_t)rows * d_out * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, input_bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(weight, weight_bytes, 1);
    MetalTensorSlot* dst = graph_output_slot(out, output_bytes);
    if (!src || !sw || !dst) return 0;
    id<MTLBuffer> bias_buffer = nil;
    int release_bias = 0;
    if (bias) {
        MetalTensorSlot* sb = graph_ensure_device(bias, bias_bytes, 1);
        if (!sb) return 0;
        bias_buffer = sb->buffer;
    } else {
        float* zero = (float*)calloc((size_t)d_out, sizeof(float));
        if (!zero) return 0;
        bias_buffer = create_buffer(bias_bytes, zero);
        free(zero);
        if (!bias_buffer) return 0;
        release_bias = 1;
    }
    uint32_t params[3] = {(uint32_t)rows, (uint32_t)d_in, (uint32_t)d_out};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) {
        if (release_bias) VX_METAL_RELEASE(bias_buffer);
        return 0;
    }
    int ok;
    int tiled = rows > 1 && d_in >= 16 && d_out >= 16;
    if (out_in_layout) {
        static const float dummy_scale[1] = {1.0f};
        id<MTLBuffer> scale_buffer = create_buffer(sizeof(dummy_scale), dummy_scale);
        if (!scale_buffer) {
            VX_METAL_RELEASE(pb);
            if (release_bias) VX_METAL_RELEASE(bias_buffer);
            return 0;
        }
        MetalBinding binds[6] = {{src->buffer, input_bytes}, {sw->buffer, weight_bytes},
                                 {scale_buffer, sizeof(dummy_scale)}, {bias_buffer, bias_bytes},
                                 {dst->buffer, output_bytes}, {pb, sizeof(params)}};
        const MetalKernel* kernel =
            tiled ? &k_linear_out_in_tiled : &k_linear_out_in;
        uint32_t groups_x = tiled ? ((uint32_t)d_out + 15u) / 16u
                                   : ((uint32_t)d_out + 63u) / 64u;
        uint32_t groups_y = tiled ? ((uint32_t)rows + 15u) / 16u : (uint32_t)rows;
        ok = dispatch_kernel(kernel, binds, groups_x, groups_y, 1);
        VX_METAL_RELEASE(scale_buffer);
    } else {
        MetalBinding binds[5] = {{src->buffer, input_bytes}, {sw->buffer, weight_bytes},
                                 {bias_buffer, bias_bytes}, {dst->buffer, output_bytes},
                                 {pb, sizeof(params)}};
        const MetalKernel* kernel =
            tiled ? &k_linear_in_out_tiled : &k_linear_in_out;
        uint32_t groups_x = tiled ? ((uint32_t)d_out + 15u) / 16u
                                   : ((uint32_t)d_out + 63u) / 64u;
        uint32_t groups_y = tiled ? ((uint32_t)rows + 15u) / 16u : (uint32_t)rows;
        ok = dispatch_kernel(kernel, binds, groups_x, groups_y, 1);
    }
    VX_METAL_RELEASE(pb);
    if (release_bias) VX_METAL_RELEASE(bias_buffer);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_conv2d_f32(const float* in, float* out, const float* weight,
                           const float* bias, int n, int h, int w, int c, int out_c,
                           int kh, int kw, int out_h, int out_w, int sy, int sx,
                           int pt, int pl, int groups, int relu, int dy, int dx) {
    if (!in || !out || !weight || n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_c <= 0 ||
        kh <= 0 || kw <= 0 || out_h <= 0 || out_w <= 0 || sy <= 0 || sx <= 0 ||
        groups <= 0 || c % groups || out_c % groups || dy <= 0 || dx <= 0) return 0;
    int weight_channels = groups == c ? c : c / groups;
    int multiplier = groups == c ? out_c / c : 1;
    size_t input_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t weight_bytes = groups == c
        ? (size_t)kh * kw * c * multiplier * sizeof(float)
        : (size_t)kh * kw * weight_channels * out_c * sizeof(float);
    size_t bias_bytes = (size_t)out_c * sizeof(float);
    size_t output_bytes = (size_t)n * out_h * out_w * out_c * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, input_bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(weight, weight_bytes, 1);
    MetalTensorSlot* dst = graph_output_slot(out, output_bytes);
    if (!src || !sw || !dst) return 0;
    id<MTLBuffer> bias_buffer = nil;
    int release_bias = 0;
    if (bias) {
        MetalTensorSlot* sb = graph_ensure_device(bias, bias_bytes, 1);
        if (!sb) return 0;
        bias_buffer = sb->buffer;
    } else {
        float* zero = (float*)calloc((size_t)out_c, sizeof(float));
        if (!zero) return 0;
        bias_buffer = create_buffer(bias_bytes, zero);
        free(zero);
        if (!bias_buffer) return 0;
        release_bias = 1;
    }
    uint32_t params[17] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                           (uint32_t)out_c, (uint32_t)out_h, (uint32_t)out_w,
                           (uint32_t)kh, (uint32_t)kw, (uint32_t)sy, (uint32_t)sx,
                           (uint32_t)pt, (uint32_t)pl, (uint32_t)groups, (uint32_t)relu,
                           (uint32_t)dy, (uint32_t)dx};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) {
        if (release_bias) VX_METAL_RELEASE(bias_buffer);
        return 0;
    }
    MetalBinding binds[5] = {{src->buffer, input_bytes}, {sw->buffer, weight_bytes},
                             {bias_buffer, bias_bytes}, {dst->buffer, output_bytes},
                             {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_conv2d, binds, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * out_c));
    VX_METAL_RELEASE(pb);
    if (release_bias) VX_METAL_RELEASE(bias_buffer);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
                           float* out, long out_numel, const uint32_t* output_strides,
                           const uint32_t* a_strides, const uint32_t* b_strides,
                           int rank, int op) {
    if (!a || !b || !out || !output_strides || !a_strides || !b_strides ||
        a_numel <= 0 || b_numel <= 0 || out_numel <= 0 ||
        (uint64_t)out_numel > UINT32_MAX || rank <= 0 || rank > 8 || op < 0 || op > 3)
        return 0;
    size_t a_bytes = (size_t)a_numel * sizeof(float);
    size_t b_bytes = (size_t)b_numel * sizeof(float);
    size_t out_bytes = (size_t)out_numel * sizeof(float);
    MetalTensorSlot* sa = graph_ensure_device(a, a_bytes, 0);
    MetalTensorSlot* sb = graph_ensure_device(b, b_bytes, 0);
    MetalTensorSlot* so = graph_output_slot(out, out_bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t metadata[28] = {(uint32_t)out_numel, (uint32_t)rank, (uint32_t)op, 0u};
    memcpy(metadata + 4, output_strides, 8 * sizeof(uint32_t));
    memcpy(metadata + 12, a_strides, 8 * sizeof(uint32_t));
    memcpy(metadata + 20, b_strides, 8 * sizeof(uint32_t));
    id<MTLBuffer> pb = create_buffer(sizeof(metadata), metadata);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {sa->buffer, a_bytes}, {sb->buffer, b_bytes}, {so->buffer, out_bytes}, {pb, sizeof(metadata)}
    };
    int ok = dispatch_kernel(&k_broadcast_binary, binds, ((uint32_t)out_numel + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int metal_graph_split_f32(const float* in, float* out, long input_numel, long output_numel,
                          int inner, int split_size, int axis_in, int offset) {
    if (!in || !out || input_numel <= 0 || output_numel <= 0 ||
        inner <= 0 || split_size <= 0 || axis_in <= 0 || offset < 0 || offset + split_size > axis_in) return 0;
    size_t in_bytes = (size_t)input_numel * sizeof(float);
    size_t out_bytes = (size_t)output_numel * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[5] = {(uint32_t)output_numel, (uint32_t)inner, (uint32_t)split_size,
                          (uint32_t)axis_in, (uint32_t)offset};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, in_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_split, binds, ((uint32_t)output_numel + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_conv1d_f32(const float* in, const float* weight, const float* bias, float* out,
                           int batch, int in_c, int in_l, int out_c, int out_l, int kernel,
                           int stride, int pad, int relu) {
    if (!in || !weight || !out || batch <= 0 || in_c <= 0 || in_l <= 0 || out_c <= 0 || out_l <= 0 ||
        kernel <= 0 || stride <= 0 || pad < 0 || relu < 0 || relu > 1) return 0;
    int64_t padded_l = (int64_t)in_l + 2LL * pad;
    if (padded_l < kernel || (padded_l - kernel) / stride + 1 != out_l) return 0;
    size_t in_bytes = (size_t)batch * (size_t)in_c * in_l * sizeof(float);
    size_t wbytes = (size_t)out_c * in_c * kernel * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)out_c * out_l * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sw || !dst) return 0;
    id<MTLBuffer> bias_buf = nil;
    int delete_bias = 0;
    if (bias) {
        MetalTensorSlot* sb = graph_ensure_device(bias, bbytes, 1);
        if (!sb) return 0;
        bias_buf = sb->buffer;
    } else {
        float* zeros = (float*)calloc((size_t)out_c, sizeof(float));
        if (!zeros) return 0;
        bias_buf = create_buffer(bbytes, zeros);
        free(zeros);
        if (!bias_buf) return 0;
        delete_bias = 1;
    }
    uint32_t params[12] = {(uint32_t)in_c, (uint32_t)in_l, (uint32_t)out_c,
                           (uint32_t)kernel, (uint32_t)stride, (uint32_t)pad,
                           (uint32_t)relu, (uint32_t)batch, 1u, (uint32_t)in_c,
                           (uint32_t)out_l, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) {
        if (delete_bias) VX_METAL_RELEASE(bias_buf);
        return 0;
    }
    MetalBinding binds[5] = {
        {src->buffer, in_bytes}, {sw->buffer, wbytes}, {bias_buf, bbytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_conv1d, binds, ((uint32_t)out_l + 63u) / 64u,
                             (uint32_t)out_c, (uint32_t)batch);
    VX_METAL_RELEASE(pb);
    if (delete_bias) VX_METAL_RELEASE(bias_buf);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

/* Routed expert linear. `experts` counts staged rows; route indices stay in
 * global slot space and are mapped through slot_rows when slot_domain != 0. */
int metal_graph_moe_linear_f32(const float* input, const float* expert_weight,
                               const float* expert_bias, const float* route_indices,
                               const float* route_weights, float* out, int rows,
                               int d_in, int d_out, int experts, int top_k,
                               const uint32_t* slot_rows, uint32_t slot_domain) {
    if (rows <= 0 || d_in <= 0 || d_out <= 0 || experts <= 0 || top_k <= 0 ||
        !input || !expert_weight || !route_indices || !route_weights || !out) return 0;
    if (slot_rows ? (slot_domain < (uint32_t)experts ||
                     (uint32_t)top_k > slot_domain)
                  : top_k > experts) return 0;
    size_t in_bytes = (size_t)rows * (size_t)d_in * sizeof(float);
    size_t w_bytes = (size_t)experts * (size_t)d_in * (size_t)d_out * sizeof(float);
    size_t bias_bytes = (size_t)experts * (size_t)d_out * sizeof(float);
    size_t route_bytes = (size_t)rows * (size_t)top_k * sizeof(float);
    size_t out_bytes = (size_t)rows * (size_t)d_out * sizeof(float);
    uint32_t slot_extent = slot_domain ? slot_domain : 1u;
    MetalTensorSlot* si = graph_ensure_device(input, in_bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(expert_weight, w_bytes, 1);
    MetalTensorSlot* sri = graph_ensure_device(route_indices, route_bytes, 0);
    MetalTensorSlot* srw = graph_ensure_device(route_weights, route_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    MetalTensorSlot* sb = expert_bias
        ? graph_ensure_device(expert_bias, bias_bytes, 1) : NULL;
    if (!si || !sw || !sri || !srw || !dst || (expert_bias && !sb)) return 0;
    /* The shader always declares a bias and a slot table; absent ones become
     * zero-filled transients so the binding set stays complete. */
    id<MTLBuffer> bias_dummy = expert_bias ? nil : create_buffer(bias_bytes, NULL);
    if (!expert_bias && !bias_dummy) return 0;
    if (!expert_bias) memset([bias_dummy contents], 0, bias_bytes);
    id<MTLBuffer> slot_buffer =
        create_buffer((size_t)slot_extent * sizeof(uint32_t), NULL);
    if (!slot_buffer) {
        VX_METAL_RELEASE(bias_dummy);
        return 0;
    }
    memset([slot_buffer contents], 0, (size_t)slot_extent * sizeof(uint32_t));
    if (slot_domain)
        memcpy([slot_buffer contents], slot_rows,
               (size_t)slot_domain * sizeof(uint32_t));
    uint32_t params[8] = {
        (uint32_t)rows, (uint32_t)d_in, (uint32_t)d_out, (uint32_t)experts,
        (uint32_t)top_k, expert_bias ? 1u : 0u, slot_domain, 0u,
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) {
        VX_METAL_RELEASE(bias_dummy);
        VX_METAL_RELEASE(slot_buffer);
        return 0;
    }
    MetalBinding binds[8] = {
        {si->buffer, in_bytes}, {sw->buffer, w_bytes},
        {expert_bias ? sb->buffer : bias_dummy, bias_bytes},
        {sri->buffer, route_bytes}, {srw->buffer, route_bytes},
        {dst->buffer, out_bytes}, {pb, sizeof(params)},
        {slot_buffer, (size_t)slot_extent * sizeof(uint32_t)},
    };
    int ok = dispatch_kernel(&k_moe_linear, binds,
                             ((uint32_t)d_out + 63u) / 64u, (uint32_t)rows, 1);
    VX_METAL_RELEASE(pb);
    VX_METAL_RELEASE(bias_dummy);
    VX_METAL_RELEASE(slot_buffer);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

/* Top-k expert routing. The router weight is [d_model, experts], so its expert
 * axis is 1 and it is never a slot-indexed bank. */
int metal_graph_moe_router_f32(const float* input, const float* weight,
                               const float* bias, float* route_indices,
                               float* route_weights, int rows, int d_model,
                               int experts, int top_k, float temperature,
                               int normalize) {
    if (rows <= 0 || d_model <= 0 || experts <= 0 || top_k <= 0 ||
        top_k > experts || top_k > 8 || !(temperature > 0.0f) ||
        !input || !weight || !route_indices || !route_weights) return 0;
    size_t in_bytes = (size_t)rows * (size_t)d_model * sizeof(float);
    size_t w_bytes = (size_t)d_model * (size_t)experts * sizeof(float);
    size_t bias_bytes = (size_t)experts * sizeof(float);
    size_t route_bytes = (size_t)rows * (size_t)top_k * sizeof(float);
    MetalTensorSlot* si = graph_ensure_device(input, in_bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(weight, w_bytes, 1);
    MetalTensorSlot* sidx = graph_output_slot(route_indices, route_bytes);
    MetalTensorSlot* srw = graph_output_slot(route_weights, route_bytes);
    MetalTensorSlot* sb = bias ? graph_ensure_device(bias, bias_bytes, 1) : NULL;
    if (!si || !sw || !sidx || !srw || (bias && !sb)) return 0;
    id<MTLBuffer> bias_dummy = bias ? nil : create_buffer(bias_bytes, NULL);
    if (!bias && !bias_dummy) return 0;
    if (!bias) memset([bias_dummy contents], 0, bias_bytes);
    uint32_t params[8];
    params[0] = (uint32_t)rows;
    params[1] = (uint32_t)d_model;
    params[2] = (uint32_t)experts;
    params[3] = (uint32_t)top_k;
    params[4] = normalize ? 1u : 0u;
    params[5] = bias ? 1u : 0u;
    memcpy(&params[6], &temperature, sizeof(float));
    params[7] = 0u;
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) { VX_METAL_RELEASE(bias_dummy); return 0; }
    MetalBinding binds[6] = {
        {si->buffer, in_bytes}, {sw->buffer, w_bytes},
        {bias ? sb->buffer : bias_dummy, bias_bytes},
        {sidx->buffer, route_bytes}, {srw->buffer, route_bytes},
        {pb, sizeof(params)},
    };
    int ok = dispatch_kernel(&k_moe_router, binds, ((uint32_t)rows + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    VX_METAL_RELEASE(bias_dummy);
    if (!ok) return 0;
    graph_mark_device(sidx);
    graph_mark_device(srw);
    return 1;
}

int metal_graph_embedding_f32(const int32_t* tokens, const float* weight, float* out,
                              int tokens_len, int d_model, int vocab_size) {
    if (!tokens || !weight || !out || tokens_len <= 0 || d_model <= 0 || vocab_size <= 0) return 0;
    size_t token_bytes = (size_t)tokens_len * sizeof(int32_t);
    size_t weight_bytes = (size_t)vocab_size * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)tokens_len * (size_t)d_model * sizeof(float);
    MetalTensorSlot* st = graph_ensure_device(tokens, token_bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(weight, weight_bytes, 1);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!st || !sw || !dst) return 0;
    uint32_t params[4] = {(uint32_t)tokens_len, (uint32_t)d_model,
                          (uint32_t)vocab_size, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {st->buffer, token_bytes}, {sw->buffer, weight_bytes},
        {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_embedding, binds, ((uint32_t)tokens_len + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static long attention_mask_numel(int mask_mode, int batch, int seq_q, int seq_kv) {
    if (mask_mode == 0) return 0;
    if (mask_mode == 1) return seq_kv;
    if (mask_mode == 2) return (long)batch * seq_kv;
    if (mask_mode == 3) return (long)seq_q * seq_kv;
    if (mask_mode == 4) return (long)batch * seq_q * seq_kv;
    return -1;
}

int metal_graph_sdpa_f32(const float* qkv, const int32_t* mask, long mask_numel,
                         float* out, int seq_len, int d_model, int num_heads,
                         int head_dim, int batch, float scale, int causal, int mask_mode) {
    if (!qkv || !out || seq_len <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        batch <= 0 || head_dim > 64 || d_model != num_heads * head_dim) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_len, seq_len);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t qkv_bytes = (size_t)batch * (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)seq_len * (size_t)d_model * sizeof(float);
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : qkv_bytes;
    MetalTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    MetalTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : src;
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sm || !dst) return 0;
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode;
    } params = {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {src->buffer, qkv_bytes}, {sm->buffer, mask_bytes},
        {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_sdpa, binds, ((uint32_t)seq_len + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int metal_graph_sdpa_training_f32(const float* qkv, const int32_t* mask, long mask_numel,
                                  float* out, int seq_len, int d_model, int num_heads,
                                  int head_dim, int batch, float scale, int causal, int mask_mode,
                                  uint32_t threshold, uint32_t seed, uint32_t counter,
                                  float dropout_scale) {
    if (!qkv || !out || seq_len <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        batch <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        !isfinite(dropout_scale) || dropout_scale < 1.0f) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_len, seq_len);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t qkv_bytes = (size_t)batch * (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)seq_len * (size_t)d_model * sizeof(float);
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : qkv_bytes;
    MetalTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    MetalTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : src;
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sm || !dst) return 0;
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
    } params = {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, (uint32_t)batch, scale, causal ? 1u : 0u,
                (uint32_t)mask_mode, threshold, seed, counter, dropout_scale};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {src->buffer, qkv_bytes}, {sm->buffer, mask_bytes},
        {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_sdpa_training, binds, ((uint32_t)seq_len + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int metal_graph_cross_sdpa_f32(const float* q, const float* k, const float* v,
                               const int32_t* mask, long mask_numel, float* out,
                               int seq_q, int seq_kv, int d_model, int num_heads,
                               int head_dim, int batch, float scale, int causal, int mask_mode) {
    if (!q || !k || !v || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || batch <= 0 || head_dim > 64 ||
        d_model != num_heads * head_dim) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_q, seq_kv);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : q_bytes;
    MetalTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    MetalTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    MetalTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    MetalTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : sq;
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !sm || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model,
                (uint32_t)num_heads, (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode, {0u, 0u, 0u}};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[6] = {
        {sq->buffer, q_bytes}, {sk->buffer, kv_bytes}, {sv->buffer, kv_bytes},
        {sm->buffer, mask_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_cross_sdpa, binds, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int metal_graph_cross_sdpa_training_f32(const float* q, const float* k, const float* v,
                                        const int32_t* mask, long mask_numel, float* out,
                                        int seq_q, int seq_kv, int d_model, int num_heads,
                                        int head_dim, int batch, float scale, int causal, int mask_mode,
                                        uint32_t threshold, uint32_t seed, uint32_t counter,
                                        float dropout_scale) {
    if (!q || !k || !v || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || batch <= 0 || head_dim > 64 ||
        d_model != num_heads * head_dim || !isfinite(dropout_scale) || dropout_scale < 1.0f) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_q, seq_kv);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : q_bytes;
    MetalTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    MetalTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    MetalTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    MetalTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : sq;
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !sm || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
        uint32_t pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model,
                (uint32_t)num_heads, (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode, threshold, seed, counter,
                dropout_scale, {0u, 0u, 0u}};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[6] = {
        {sq->buffer, q_bytes}, {sk->buffer, kv_bytes}, {sv->buffer, kv_bytes},
        {sm->buffer, mask_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_cross_sdpa_training, binds, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int metal_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                    const float* scale, const float* bias, float* out,
                                    int seq_q, int seq_kv, int d_model, int num_heads,
                                    int head_dim, int batch, int has_scale, int has_bias) {
    if (!q || !kv || !weight || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || batch <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        d_model > 64) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t wbytes = (size_t)3 * (size_t)d_model * (size_t)d_model * sizeof(float);
    size_t sb_bytes = (size_t)3 * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    static const float zero[1] = {0.0f};
    MetalTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    MetalTensorSlot* skv = graph_ensure_device(kv, kv_bytes, 0);
    MetalTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    MetalTensorSlot* ss = graph_ensure_device(has_scale && scale ? scale : zero, has_scale && scale ? sb_bytes : sizeof(float), 1);
    MetalTensorSlot* sb = graph_ensure_device(has_bias && bias ? bias : zero, has_bias && bias ? sb_bytes : sizeof(float), 1);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !skv || !sw || !ss || !sb || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim;
        float scale_factor;
        uint32_t has_scale, has_bias, batch, pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim),
                (uint32_t)(has_scale && scale), (uint32_t)(has_bias && bias), (uint32_t)batch, {0u, 0u, 0u}};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[7] = {
        {sq->buffer, q_bytes}, {skv->buffer, kv_bytes}, {sw->buffer, wbytes},
        {ss->buffer, ss->bytes}, {sb->buffer, sb->bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_cross_attention, binds, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_dequantize_linear_f32(const float* in, const float* scale, const float* zero_point,
                                      float* out, long n, int has_zero_point) {
    if (!in || !scale || !out || n <= 0) return 0;
    static const float zero[1] = {0.0f};
    size_t bytes = (size_t)n * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, bytes, 0);
    MetalTensorSlot* ss = graph_ensure_device(scale, sizeof(float), 1);
    MetalTensorSlot* sz = graph_ensure_device(has_zero_point && zero_point ? zero_point : zero, sizeof(float), 1);
    MetalTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !ss || !sz || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)(has_zero_point && zero_point), 0u, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[5] = {
        {src->buffer, bytes}, {ss->buffer, sizeof(float)}, {sz->buffer, sizeof(float)}, {dst->buffer, bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_dequantize, binds, ((uint32_t)n + 63u) / 64u, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

typedef struct {
    uint32_t rows;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t input_type;
    uint32_t weight_type;
    uint32_t output_type;
    uint32_t pad0;
    uint32_t pad1;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad2;
    int32_t pad3;
    float input_scale;
    float output_scale;
    float pad4;
    float pad5;
} MetalQLinearParams;

_Static_assert(sizeof(MetalQLinearParams) == 64, "qLinearInt8 uniform ABI");

static int qlinear_dtype_bounds(uint32_t dtype, int32_t zero_point,
                                int64_t* maximum_distance) {
    int32_t minimum;
    int32_t maximum;
    if (dtype == VX_DTYPE_I8) {
        minimum = -128;
        maximum = 127;
    } else if (dtype == VX_DTYPE_U8) {
        minimum = 0;
        maximum = 255;
    } else {
        return 0;
    }
    if (zero_point < minimum || zero_point > maximum) return 0;
    int64_t lower = (int64_t)minimum - zero_point;
    int64_t upper = (int64_t)maximum - zero_point;
    *maximum_distance = llabs(lower) > llabs(upper) ? llabs(lower) : llabs(upper);
    return 1;
}

static int qlinear_gpu_args_valid(const void* input, const void* weight,
                                  const float* weight_scales,
                                  const int32_t* weight_zero_points,
                                  const int32_t* bias, void* output,
                                  uint32_t rows, uint32_t d_in, uint32_t d_out,
                                  float input_scale, int32_t input_zero_point,
                                  float output_scale, int32_t output_zero_point,
                                  uint32_t input_dtype, uint32_t weight_dtype,
                                  uint32_t output_dtype,
                                  size_t* input_bytes, size_t* weight_bytes,
                                  size_t* output_bytes) {
    if (!input || !weight || !weight_scales || !weight_zero_points || !bias || !output ||
        rows == 0 || d_in == 0 || d_out == 0 || !isfinite(input_scale) ||
        !isfinite(output_scale) || input_scale <= 0.0f || output_scale <= 0.0f) return 0;
    int64_t input_distance = 0;
    int64_t output_distance = 0;
    if (!qlinear_dtype_bounds(input_dtype, input_zero_point, &input_distance) ||
        !qlinear_dtype_bounds(output_dtype, output_zero_point, &output_distance)) return 0;
    (void)output_distance;
    if (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) return 0;
    if (rows > UINT32_MAX / d_in || rows > UINT32_MAX / d_out ||
        d_out > UINT32_MAX / d_in) return 0;
    uint64_t input_elements = (uint64_t)rows * d_in;
    uint64_t weight_elements = (uint64_t)d_out * d_in;
    uint64_t output_elements = (uint64_t)rows * d_out;
    if (input_elements > SIZE_MAX || weight_elements > SIZE_MAX ||
        output_elements > SIZE_MAX ||
        (size_t)d_out > SIZE_MAX / sizeof(*weight_scales) ||
        (size_t)d_out > SIZE_MAX / sizeof(*weight_zero_points) ||
        (size_t)d_out > SIZE_MAX / sizeof(*bias)) return 0;
    for (uint32_t channel = 0; channel < d_out; channel++) {
        int64_t weight_distance = 0;
        if (!isfinite(weight_scales[channel]) || weight_scales[channel] <= 0.0f ||
            !qlinear_dtype_bounds(weight_dtype, weight_zero_points[channel],
                                  &weight_distance)) return 0;
        int64_t bias_distance = bias[channel] < 0 ? -(int64_t)bias[channel] : bias[channel];
        if (bias_distance > INT32_MAX) return 0;
        int64_t term_distance = input_distance * weight_distance;
        if (term_distance > 0 &&
            (uint64_t)d_in > (uint64_t)(INT32_MAX - bias_distance) /
                                  (uint64_t)term_distance) return 0;
    }
    *input_bytes = (size_t)input_elements;
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    return 1;
}

int metal_graph_qlinear_i8u8(const void* input, const void* weight,
                             const float* weight_scales, const int32_t* weight_zero_points,
                             const int32_t* bias, void* output,
                             uint32_t rows, uint32_t d_in, uint32_t d_out,
                             float input_scale, int32_t input_zero_point,
                             float output_scale, int32_t output_zero_point,
                             uint32_t input_dtype, uint32_t weight_dtype,
                             uint32_t output_dtype) {
    size_t input_bytes, weight_bytes, output_bytes;
    size_t multiplier_bytes;
    float* multipliers;
    if (!qlinear_gpu_args_valid(input, weight, weight_scales, weight_zero_points, bias, output,
                                rows, d_in, d_out, input_scale, input_zero_point,
                                output_scale, output_zero_point, input_dtype, weight_dtype,
                                output_dtype, &input_bytes, &weight_bytes, &output_bytes)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    multiplier_bytes = (size_t)d_out * sizeof(*multipliers);
    multipliers = (float*)malloc(multiplier_bytes);
    if (!multipliers ||
        !vx_qlinear_build_multipliers(
            input_scale, weight_scales, output_scale, d_out, multipliers)) {
        free(multipliers);
        return 0;
    }
    /* Activations may be a row window; weights never are. */
    MetalGraphWindow src, dst;
    int resolved = graph_window_packed_bytes(input, input_bytes, 0, &src) &&
        graph_window_output_packed(output, output_bytes, &dst);
    MetalTensorSlot* wt = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    MetalTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                        (size_t)d_out * sizeof(*weight_zero_points), 1);
    MetalTensorSlot* biases = graph_ensure_device(bias, (size_t)d_out * sizeof(*bias), 1);
    if (!resolved || !wt || !zero_points || !biases) {
        free(multipliers);
        return 0;
    }
    id<MTLBuffer> multiplier_buffer =
        create_buffer(multiplier_bytes, multipliers);
    free(multipliers);
    if (!multiplier_buffer) return 0;
    MetalQLinearParams params = {
        rows, d_in, d_out,
        input_dtype,
        weight_dtype,
        output_dtype, 0u, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) {
        VX_METAL_RELEASE(multiplier_buffer);
        return 0;
    }
    MetalBinding binds[7] = {
        {src.slot->buffer, src.bytes, src.offset},
        {wt->buffer, wt->bytes},
        {multiplier_buffer, multiplier_bytes},
        {zero_points->buffer, zero_points->bytes},
        {biases->buffer, biases->bytes},
        {dst.slot->buffer, dst.bytes, dst.offset},
        {pb, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int tiled = rows > 1u && d_in >= 16u && d_out >= 32u && (d_out & 3u) == 0u;
    uint32_t groups_x = tiled ? ((d_out / 4u + 7u) / 8u) : ((packed_words + 63u) / 64u);
    uint32_t groups_y = tiled ? ((rows + 7u) / 8u) : 1u;
    const MetalKernel* kernel =
        tiled ? &k_qlinear_int8_tiled : &k_qlinear_int8;
    int ok = dispatch_kernel(kernel, binds, groups_x, groups_y, 1u);
    VX_METAL_RELEASE(pb);
    VX_METAL_RELEASE(multiplier_buffer);
    if (!ok) return 0;
    graph_mark_device(dst.slot);
    return 1;
}

typedef struct {
    uint32_t tokens;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t weight_type;
    uint32_t output_type;
    int32_t output_zero_point;
    float output_scale;
    /* The first token this dispatch reads. See graph_token_window. */
    uint32_t token_offset;
} MetalQEmbeddingParams;

_Static_assert(sizeof(MetalQEmbeddingParams) == 32, "qEmbeddingInt8 uniform ABI");

static int qembedding_gpu_args_valid(const int32_t* tokens, const void* weight,
                                     const float* weight_scales,
                                     const int32_t* weight_zero_points, void* output,
                                     uint32_t token_count, uint32_t vocab, uint32_t hidden,
                                     float output_scale, int32_t output_zero_point,
                                     uint32_t weight_dtype, uint32_t output_dtype,
                                     size_t* token_bytes, size_t* weight_bytes,
                                     size_t* output_bytes) {
    int64_t distance = 0;
    uint64_t weight_elements;
    uint64_t output_elements;
    if (!tokens || !weight || !weight_scales || !weight_zero_points || !output ||
        !token_count || !vocab || !hidden || !isfinite(output_scale) ||
        output_scale <= 0.0f || !qlinear_dtype_bounds(output_dtype, output_zero_point,
                                                        &distance) ||
        (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) ||
        token_count > UINT32_MAX / hidden || vocab > UINT32_MAX / hidden) return 0;
    weight_elements = (uint64_t)vocab * hidden;
    output_elements = (uint64_t)token_count * hidden;
    if (weight_elements > SIZE_MAX || output_elements > SIZE_MAX ||
        (size_t)token_count > SIZE_MAX / sizeof(*tokens) ||
        (size_t)vocab > SIZE_MAX / sizeof(*weight_scales) ||
        (size_t)vocab > SIZE_MAX / sizeof(*weight_zero_points)) return 0;
    for (uint32_t row = 0; row < vocab; row++) {
        if (!isfinite(weight_scales[row]) || weight_scales[row] <= 0.0f ||
            !qlinear_dtype_bounds(weight_dtype, weight_zero_points[row], &distance)) return 0;
    }
    for (uint32_t token = 0; token < token_count; token++) {
        if (tokens[token] < 0 || (uint32_t)tokens[token] >= vocab) return 0;
    }
    *token_bytes = (size_t)token_count * sizeof(*tokens);
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    return 1;
}

int metal_graph_qembedding_i8u8(const int32_t* tokens, const void* weight,
                                const float* weight_scales,
                                const int32_t* weight_zero_points, void* output,
                                uint32_t token_count, uint32_t vocab, uint32_t hidden,
                                float output_scale, int32_t output_zero_point,
                                uint32_t weight_dtype, uint32_t output_dtype) {
    size_t token_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t packed_output_bytes;
    if (!qembedding_gpu_args_valid(tokens, weight, weight_scales, weight_zero_points,
                                   output, token_count, vocab, hidden, output_scale,
                                   output_zero_point, weight_dtype, output_dtype,
                                   &token_bytes, &weight_bytes, &output_bytes) ||
        !graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    MetalGraphWindow dst;
    MetalTensorSlot* ids = NULL;
    uint32_t token_offset = 0;
    int resolved = graph_token_window(tokens, token_bytes, &ids, &token_offset) &&
        graph_window_output_packed(output, output_bytes, &dst);
    MetalTensorSlot* table = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    MetalTensorSlot* scales = graph_ensure_device(weight_scales,
                                                   (size_t)vocab * sizeof(*weight_scales), 1);
    MetalTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                        (size_t)vocab * sizeof(*weight_zero_points), 1);
    if (!resolved || !table || !scales || !zero_points) return 0;
    MetalQEmbeddingParams params = {
        token_count, vocab, hidden,
        weight_dtype,
        output_dtype, output_zero_point,
        output_scale, token_offset
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[6] = {
        {ids->buffer, ids->bytes}, {table->buffer, table->bytes},
        {scales->buffer, scales->bytes}, {zero_points->buffer, zero_points->bytes},
        {dst.slot->buffer, dst.bytes, dst.offset}, {pb, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qembedding_int8, binds,
                             (packed_words + 63u) / 64u, 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst.slot);
    return 1;
}

typedef struct {
    uint32_t elements;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t output_type;
    int32_t a_zero_point;
    int32_t b_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    float a_scale;
    float b_scale;
    float output_scale;
    uint32_t relu;
} MetalQAddParams;

_Static_assert(sizeof(MetalQAddParams) == 48, "qAdd uniform ABI");

typedef struct {
    uint32_t batch_rank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t output_elements;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t output_type;
    int32_t a_zero_point;
    int32_t b_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    float a_scale;
    float b_scale;
    float output_scale;
    float pad1;
} MetalQBatchMatMulParams;

_Static_assert(sizeof(MetalQBatchMatMulParams) == 64,
               "qBatchMatMul uniform ABI");

typedef struct {
    uint32_t elements;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float input_scale;
    float output_scale;
    float pad3;
    float pad4;
} MetalQByteUnaryParams;

_Static_assert(sizeof(MetalQByteUnaryParams) == 48,
               "qSiLUInt8/qGELUInt8 uniform ABI");

typedef struct {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t groups;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float input_scale;
    float output_scale;
    float epsilon;
    float pad3;
} MetalQGroupNormParams;

_Static_assert(sizeof(MetalQGroupNormParams) == 64,
               "qGroupNormStats/qGroupNormApply uniform ABI");

typedef struct {
    uint32_t rows;
    uint32_t d_model;
    uint32_t input_type;
    uint32_t output_type;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    int32_t pad1;
    float input_scale;
    float output_scale;
    float epsilon;
    float pad2;
} MetalQLayerNormParams;

_Static_assert(sizeof(MetalQLayerNormParams) == 48,
               "qLayerNormStats/qLayerNormApply uniform ABI");

typedef struct {
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t d_model;
    uint32_t heads;
    uint32_t batch;
    uint32_t mask_mode;
    uint32_t causal;
    uint32_t dtypes;
    int32_t q_zero_point;
    int32_t k_zero_point;
    int32_t v_zero_point;
    int32_t output_zero_point;
    float q_scale;
    float k_scale;
    float v_scale;
    float output_scale;
    float attention_scale;
    float pad0;
    float pad1;
    float pad2;
} MetalQSDPAParams;

_Static_assert(sizeof(MetalQSDPAParams) == 80, "qSDPAInt8 uniform ABI");

typedef struct {
    uint32_t outer;
    uint32_t axis_size;
    uint32_t inner;
    uint32_t input_dtype;
} MetalQArgMaxParams;

_Static_assert(sizeof(MetalQArgMaxParams) == 16, "qArgMaxInt8 uniform ABI");

typedef struct {
    uint32_t batch;
    uint32_t sequence;
    uint32_t width;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    int32_t input_zero_point;
    int32_t output_zero_point;
    float input_scale;
    float output_scale;
} MetalQMaskedMeanParams;

_Static_assert(sizeof(MetalQMaskedMeanParams) == 48,
               "qMaskedMeanInt8 uniform ABI");

typedef struct {
    uint32_t elements;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float multiplier;
    float pad3;
    float pad4;
    float pad5;
} MetalRequantizeLinearParams;

_Static_assert(sizeof(MetalRequantizeLinearParams) == 48,
               "requantizeLinearTyped uniform ABI");

static int qbyte_dtype_zero_point_valid(uint32_t dtype, int32_t zero_point) {
    if (dtype == VX_DTYPE_I8) return zero_point >= -128 && zero_point <= 127;
    if (dtype == VX_DTYPE_U8) return zero_point >= 0 && zero_point <= 255;
    return 0;
}

static int qadd_gpu_args_valid(const void* a, uint32_t a_elements,
                               const void* b, uint32_t b_elements,
                               void* output, uint32_t output_elements,
                               float a_scale, int32_t a_zero_point,
                               float b_scale, int32_t b_zero_point,
                               float output_scale, int32_t output_zero_point,
                               uint32_t a_dtype, uint32_t b_dtype,
                               uint32_t output_dtype, uint32_t relu,
                               size_t* logical_bytes) {
    if (!a || !b || !output || output == a || output == b ||
        a_elements == 0 || a_elements != b_elements ||
        a_elements != output_elements || relu > 2u ||
        !isfinite(a_scale) || !isfinite(b_scale) || !isfinite(output_scale) ||
        a_scale <= 0.0f || b_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(a_dtype, a_zero_point) ||
        !qbyte_dtype_zero_point_valid(b_dtype, b_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point)) return 0;
    if ((uint64_t)a_elements > SIZE_MAX) return 0;
    *logical_bytes = (size_t)a_elements;
    return 1;
}

static int qbyte_unary_gpu_args_valid(const void* input, void* output, uint32_t elements,
                                      float input_scale, int32_t input_zero_point,
                                      float output_scale, int32_t output_zero_point,
                                      uint32_t input_dtype, uint32_t output_dtype,
                                      size_t* logical_bytes) {
    if (!input || !output || input == output || !elements ||
        !isfinite(input_scale) || !isfinite(output_scale) ||
        input_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        (uint64_t)elements > SIZE_MAX) return 0;
    *logical_bytes = (size_t)elements;
    return 1;
}

static float qnorm_f32_at(const float* values, uint32_t index) {
    float value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static int qgroupnorm_gpu_args_valid(const void* input, const float* weight,
                                     const float* bias, void* output,
                                     uint32_t batch, uint32_t height,
                                     uint32_t width, uint32_t channels,
                                     uint32_t groups, float input_scale,
                                     int32_t input_zero_point, float output_scale,
                                     int32_t output_zero_point, float epsilon,
                                     uint32_t input_dtype, uint32_t output_dtype,
                                     size_t* logical_bytes, size_t* affine_bytes,
                                     size_t* stats_bytes) {
    uint64_t elements;
    uint64_t group_count;
    if (!input || !weight || !bias || !output || output == input ||
        output == (const void*)weight || output == (const void*)bias ||
        !batch || !height || !width || !channels || !groups || channels % groups ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        batch > UINT32_MAX / groups) return 0;
    elements = batch;
    if (elements > UINT32_MAX / height) return 0;
    elements *= height;
    if (elements > UINT32_MAX / width) return 0;
    elements *= width;
    if (elements > UINT32_MAX / channels) return 0;
    elements *= channels;
    group_count = (uint64_t)batch * groups;
    if (elements == 0 || elements > SIZE_MAX || group_count == 0 ||
        group_count > SIZE_MAX / (2u * sizeof(float)) ||
        (size_t)channels > SIZE_MAX / sizeof(float)) return 0;
    for (uint32_t channel = 0; channel < channels; channel++) {
        if (!isfinite(qnorm_f32_at(weight, channel)) ||
            !isfinite(qnorm_f32_at(bias, channel))) return 0;
    }
    *logical_bytes = (size_t)elements;
    *affine_bytes = (size_t)channels * sizeof(float);
    *stats_bytes = (size_t)group_count * 2u * sizeof(float);
    return 1;
}

static int qlayernorm_gpu_args_valid(const void* input, const float* weight,
                                     const float* bias, void* output,
                                     uint32_t rows, uint32_t d_model,
                                     float input_scale, int32_t input_zero_point,
                                     float output_scale, int32_t output_zero_point,
                                     float epsilon, uint32_t input_dtype,
                                     uint32_t output_dtype, size_t* logical_bytes,
                                     size_t* affine_bytes, size_t* stats_bytes) {
    uint64_t elements;
    size_t row_count;
    if (!input || !weight || !bias || !output || output == input ||
        output == (const void*)weight || output == (const void*)bias ||
        !rows || !d_model || !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        rows > UINT32_MAX / d_model) return 0;
    elements = (uint64_t)rows * d_model;
    row_count = (size_t)rows;
    if (elements == 0 || elements > SIZE_MAX ||
        row_count > SIZE_MAX / (2u * sizeof(float)) ||
        (size_t)d_model > SIZE_MAX / sizeof(float)) return 0;
    for (uint32_t channel = 0; channel < d_model; channel++) {
        if (!isfinite(qnorm_f32_at(weight, channel)) ||
            !isfinite(qnorm_f32_at(bias, channel))) return 0;
    }
    *logical_bytes = (size_t)elements;
    *affine_bytes = (size_t)d_model * sizeof(float);
    *stats_bytes = row_count * 2u * sizeof(float);
    return 1;
}

static int qsdpa_ranges_overlap(const void* left, size_t left_bytes,
                                const void* right, size_t right_bytes) {
    uintptr_t left_start = (uintptr_t)left;
    uintptr_t right_start = (uintptr_t)right;
    if (left_start > UINTPTR_MAX - left_bytes ||
        right_start > UINTPTR_MAX - right_bytes) return 1;
    return left_start < right_start + right_bytes &&
        right_start < left_start + left_bytes;
}

static int qsdpa_centered_magnitude(uint32_t dtype, int32_t zero_point,
                                    uint64_t* magnitude_out) {
    int64_t low;
    int64_t high;
    uint64_t magnitude;
    if (!magnitude_out) return 0;
    if (dtype == VX_DTYPE_I8) {
        low = -128 - (int64_t)zero_point;
        high = 127 - (int64_t)zero_point;
    } else if (dtype == VX_DTYPE_U8) {
        low = -(int64_t)zero_point;
        high = 255 - (int64_t)zero_point;
    } else {
        return 0;
    }
    magnitude = (uint64_t)(low < 0 ? -low : low);
    if ((uint64_t)(high < 0 ? -high : high) > magnitude)
        magnitude = (uint64_t)(high < 0 ? -high : high);
    *magnitude_out = magnitude;
    return 1;
}

static int qsdpa_mask_bytes(uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
                            uint32_t mask_mode, size_t* mask_bytes) {
    uint64_t elements = seq_kv;
    if (!mask_bytes) return 0;
    if (mask_mode == 0u) {
        *mask_bytes = 0;
        return 1;
    }
    if (mask_mode == 2u) {
        if (elements > UINT32_MAX / batch) return 0;
        elements *= batch;
    } else if (mask_mode == 3u) {
        if (elements > UINT32_MAX / seq_q) return 0;
        elements *= seq_q;
    } else if (mask_mode == 4u) {
        if (elements > UINT32_MAX / seq_q) return 0;
        elements *= seq_q;
        if (elements > UINT32_MAX / batch) return 0;
        elements *= batch;
    } else if (mask_mode != 1u) {
        return 0;
    }
    if (elements == 0 || elements > UINT32_MAX ||
        elements > SIZE_MAX / sizeof(int32_t)) return 0;
    *mask_bytes = (size_t)elements * sizeof(int32_t);
    return 1;
}

static int qsdpa_gpu_args_valid(const void* q, const void* k, const void* v,
                                const int32_t* mask, void* output,
                                uint32_t batch, uint32_t seq_q,
                                uint32_t seq_kv, uint32_t d_model,
                                uint32_t heads, float q_scale,
                                int32_t q_zero_point, float k_scale,
                                int32_t k_zero_point, float v_scale,
                                int32_t v_zero_point, float output_scale,
                                int32_t output_zero_point, float attention_scale,
                                uint32_t q_dtype, uint32_t k_dtype,
                                uint32_t v_dtype, uint32_t output_dtype,
                                uint32_t causal, uint32_t mask_mode,
                                size_t* q_bytes, size_t* kv_bytes,
                                size_t* mask_bytes) {
    uint64_t q_elements = batch;
    uint64_t kv_elements = batch;
    uint64_t q_magnitude;
    uint64_t k_magnitude;
    uint64_t maximum_dot;
    uint32_t head_dim;
    float qk_scale;
    float score_scale;
    if (!q || !k || !v || !output || !batch || !seq_q || !seq_kv ||
        !d_model || !heads || causal > 1u || d_model % heads ||
        d_model % 4u || !isfinite(q_scale) || q_scale <= 0.0f ||
        !isfinite(k_scale) || k_scale <= 0.0f || !isfinite(v_scale) ||
        v_scale <= 0.0f || !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(attention_scale) || attention_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(q_dtype, q_zero_point) ||
        !qbyte_dtype_zero_point_valid(k_dtype, k_zero_point) ||
        !qbyte_dtype_zero_point_valid(v_dtype, v_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        !q_bytes || !kv_bytes || !mask_bytes) return 0;
    head_dim = d_model / heads;
    if (!head_dim || head_dim % 4u || head_dim > 64u ||
        q_elements > UINT32_MAX / seq_q) return 0;
    q_elements *= seq_q;
    if (q_elements > UINT32_MAX / d_model) return 0;
    q_elements *= d_model;
    if (kv_elements > UINT32_MAX / seq_kv) return 0;
    kv_elements *= seq_kv;
    if (kv_elements > UINT32_MAX / d_model) return 0;
    kv_elements *= d_model;
    if (q_elements == 0 || kv_elements == 0 || q_elements > SIZE_MAX ||
        kv_elements > SIZE_MAX || !qsdpa_mask_bytes(batch, seq_q, seq_kv,
                                                     mask_mode, mask_bytes) ||
        (mask_mode == 0u ? mask != NULL : mask == NULL)) return 0;
    qk_scale = q_scale * k_scale;
    score_scale = qk_scale * attention_scale;
    if (!isfinite(qk_scale) || qk_scale <= 0.0f ||
        !isfinite(score_scale) || score_scale <= 0.0f ||
        !qsdpa_centered_magnitude(q_dtype, q_zero_point, &q_magnitude) ||
        !qsdpa_centered_magnitude(k_dtype, k_zero_point, &k_magnitude)) return 0;
    maximum_dot = q_magnitude * k_magnitude * head_dim;
    if (!isfinite((float)maximum_dot * score_scale)) return 0;
    *q_bytes = (size_t)q_elements;
    *kv_bytes = (size_t)kv_elements;
    if (qsdpa_ranges_overlap(output, *q_bytes, q, *q_bytes) ||
        qsdpa_ranges_overlap(output, *q_bytes, k, *kv_bytes) ||
        qsdpa_ranges_overlap(output, *q_bytes, v, *kv_bytes) ||
        (*mask_bytes && qsdpa_ranges_overlap(output, *q_bytes, mask,
                                             *mask_bytes))) return 0;
    return 1;
}

static int qargmax_gpu_args_valid(const void* input, int32_t* output,
                                  uint32_t outer, uint32_t axis_size,
                                  uint32_t inner, uint32_t input_dtype,
                                  size_t* input_bytes, size_t* output_bytes,
                                  uint32_t* output_elements_out) {
    uint64_t input_elements = outer;
    uint64_t output_elements = outer;
    if (!input || !output || !outer || !axis_size || !inner ||
        axis_size > (uint32_t)INT32_MAX ||
        (input_dtype != VX_DTYPE_I8 && input_dtype != VX_DTYPE_U8) ||
        !input_bytes || !output_bytes || !output_elements_out ||
        input_elements > UINT32_MAX / axis_size) return 0;
    input_elements *= axis_size;
    if (input_elements > UINT32_MAX / inner ||
        output_elements > UINT32_MAX / inner) return 0;
    input_elements *= inner;
    output_elements *= inner;
    if (!input_elements || !output_elements || input_elements > SIZE_MAX ||
        output_elements > UINT32_MAX ||
        output_elements > SIZE_MAX / sizeof(*output)) return 0;
    *input_bytes = (size_t)input_elements;
    *output_bytes = (size_t)output_elements * sizeof(*output);
    *output_elements_out = (uint32_t)output_elements;
    return !qsdpa_ranges_overlap(output, *output_bytes, input, *input_bytes);
}

static int qmaskedmean_gpu_args_valid(const void* input, const int32_t* mask,
                                      void* output, uint32_t batch,
                                      uint32_t sequence, uint32_t width,
                                      float input_scale, int32_t input_zero_point,
                                      float output_scale, int32_t output_zero_point,
                                      uint32_t input_dtype, uint32_t output_dtype,
                                      size_t* input_bytes, size_t* mask_bytes,
                                      size_t* output_bytes) {
    uint64_t input_elements = batch;
    uint64_t mask_elements = batch;
    uint64_t output_elements = batch;
    int64_t low;
    int64_t high;
    uint64_t magnitude;
    float multiplier;
    if (!input || !mask || !output || !batch || !sequence || !width ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        !input_bytes || !mask_bytes || !output_bytes ||
        input_elements > UINT32_MAX / sequence) return 0;
    input_elements *= sequence;
    if (input_elements > UINT32_MAX / width ||
        mask_elements > UINT32_MAX / sequence ||
        output_elements > UINT32_MAX / width) return 0;
    input_elements *= width;
    mask_elements *= sequence;
    output_elements *= width;
    if (!input_elements || !mask_elements || !output_elements ||
        input_elements > SIZE_MAX || mask_elements > SIZE_MAX / sizeof(*mask) ||
        output_elements > SIZE_MAX) return 0;
    if (input_dtype == VX_DTYPE_I8) {
        low = -128 - (int64_t)input_zero_point;
        high = 127 - (int64_t)input_zero_point;
    } else {
        low = -(int64_t)input_zero_point;
        high = 255 - (int64_t)input_zero_point;
    }
    magnitude = (uint64_t)(low < 0 ? -low : low);
    if ((uint64_t)(high < 0 ? -high : high) > magnitude)
        magnitude = (uint64_t)(high < 0 ? -high : high);
    multiplier = input_scale / output_scale;
    if ((magnitude != 0 && (uint64_t)sequence > (uint64_t)INT32_MAX / magnitude) ||
        !isfinite(multiplier) || multiplier <= 0.0f) return 0;
    *input_bytes = (size_t)input_elements;
    *mask_bytes = (size_t)mask_elements * sizeof(*mask);
    *output_bytes = (size_t)output_elements;
    return !qsdpa_ranges_overlap(output, *output_bytes, input, *input_bytes) &&
        !qsdpa_ranges_overlap(output, *output_bytes, mask, *mask_bytes);
}

static int requantize_gpu_args_valid(const void* input, uint32_t input_elements,
                                     void* output, uint32_t output_elements,
                                     float input_scale, int32_t input_zero_point,
                                     float output_scale, int32_t output_zero_point,
                                     uint32_t input_dtype, uint32_t output_dtype,
                                     float* multiplier, size_t* logical_bytes) {
    if (!input || !output || input == output || input_elements == 0 ||
        input_elements != output_elements || !isfinite(input_scale) ||
        !isfinite(output_scale) || input_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point)) return 0;
    float ratio = input_scale / output_scale;
    if (!isfinite(ratio) || ratio <= 0.0f || (uint64_t)input_elements > SIZE_MAX) return 0;
    *multiplier = ratio;
    *logical_bytes = (size_t)input_elements;
    return 1;
}

int metal_graph_qbatch_matmul_i8u8(
        const void* a, const int* a_shape, int a_rank,
        float a_scale, int32_t a_zero_point, uint32_t a_dtype,
        const void* b, const int* b_shape, int b_rank,
        float b_scale, int32_t b_zero_point, uint32_t b_dtype,
        void* output, const int* output_shape, int output_rank,
        float output_scale, int32_t output_zero_point,
        uint32_t output_dtype) {
    VxQBatchMatMulDevicePlan plan;
    uint32_t metadata[24] = {0};
    uint32_t metadata_words;
    size_t metadata_bytes;
    size_t a_packed_bytes;
    size_t b_packed_bytes;
    size_t output_packed_bytes;
    if (!metal_ready() ||
        !vx_qbatch_matmul_device_plan(
            a, a_shape, a_rank, a_scale, a_zero_point, a_dtype,
            b, b_shape, b_rank, b_scale, b_zero_point, b_dtype,
            output, output_shape, output_rank, output_scale,
            output_zero_point, output_dtype, &plan) ||
        !graph_packed_bytes(plan.a_bytes, &a_packed_bytes) ||
        !graph_packed_bytes(plan.b_bytes, &b_packed_bytes) ||
        !graph_packed_bytes(plan.output_bytes, &output_packed_bytes))
        return 0;
    MetalTensorSlot* a_slot =
        graph_ensure_packed_bytes(a, plan.a_bytes, 0);
    MetalTensorSlot* b_slot =
        graph_ensure_packed_bytes(b, plan.b_bytes, 0);
    MetalTensorSlot* output_slot =
        graph_output_packed_bytes(output, plan.output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    memcpy(metadata, plan.output_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    memcpy(metadata + plan.batch_rank, plan.a_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    memcpy(metadata + 2u * plan.batch_rank, plan.b_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    metadata_words = plan.batch_rank ? 3u * plan.batch_rank : 1u;
    metadata_bytes = (size_t)metadata_words * sizeof(uint32_t);
    MetalQBatchMatMulParams params = {
        plan.batch_rank, plan.m, plan.k, plan.n,
        plan.output_elements, a_dtype, b_dtype, output_dtype,
        a_zero_point, b_zero_point, output_zero_point, 0,
        a_scale, b_scale, output_scale, 0.0f,
    };
    id<MTLBuffer> metadata_buffer =
        create_buffer(metadata_bytes, metadata);
    id<MTLBuffer> params_buffer_handle =
        create_buffer(sizeof(params), &params);
    if (!metadata_buffer || !params_buffer_handle) {
        VX_METAL_RELEASE(metadata_buffer);
        VX_METAL_RELEASE(params_buffer_handle);
        return 0;
    }
    MetalBinding binds[5] = {
        {a_slot->buffer, a_packed_bytes},
        {b_slot->buffer, b_packed_bytes},
        {output_slot->buffer, output_packed_bytes},
        {metadata_buffer, metadata_bytes},
        {params_buffer_handle, sizeof(params)},
    };
    uint32_t packed_words =
        (uint32_t)(output_packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(
        &k_qbatch_matmul_i8u8, binds,
        (packed_words + 63u) / 64u, 1u, 1u);
    VX_METAL_RELEASE(metadata_buffer);
    VX_METAL_RELEASE(params_buffer_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_qadd_i8u8(const void* a, uint32_t a_elements,
                          const void* b, uint32_t b_elements,
                          void* output, uint32_t output_elements,
                          float a_scale, int32_t a_zero_point,
                          float b_scale, int32_t b_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t a_dtype, uint32_t b_dtype,
                          uint32_t output_dtype, uint32_t relu) {
    if (!metal_ready()) return 0;
    size_t logical_bytes;
    if (!qadd_gpu_args_valid(a, a_elements, b, b_elements, output, output_elements,
                             a_scale, a_zero_point, b_scale, b_zero_point,
                             output_scale, output_zero_point, a_dtype, b_dtype,
                             output_dtype, relu, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    MetalGraphWindow a_slot, b_slot, output_slot;
    if (!graph_window_packed_bytes(a, logical_bytes, 0, &a_slot) ||
        !graph_window_packed_bytes(b, logical_bytes, 0, &b_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot)) return 0;
    MetalQAddParams params = {
        a_elements, a_dtype,
        b_dtype,
        output_dtype,
        a_zero_point, b_zero_point, output_zero_point, 0,
        a_scale, b_scale, output_scale, relu
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {a_slot.slot->buffer, a_slot.bytes, a_slot.offset},
        {b_slot.slot->buffer, b_slot.bytes, b_slot.offset},
        {output_slot.slot->buffer, output_slot.bytes, output_slot.offset},
        {pb, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qadd_i8u8, binds, (packed_words + 63u) / 64u,
                             1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

int metal_graph_qsilu_i8u8(const void* input, void* output, uint32_t elements,
                           float input_scale, int32_t input_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t input_dtype, uint32_t output_dtype) {
    if (!metal_ready()) return 0;
    size_t logical_bytes;
    if (!qbyte_unary_gpu_args_valid(input, output, elements, input_scale,
                                    input_zero_point, output_scale, output_zero_point,
                                    input_dtype, output_dtype, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    MetalGraphWindow input_slot, output_slot;
    if (!graph_window_packed_bytes(input, logical_bytes, 0, &input_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot)) return 0;
    MetalQByteUnaryParams params = {
        elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {
        {input_slot.slot->buffer, input_slot.bytes, input_slot.offset},
        {output_slot.slot->buffer, output_slot.bytes, output_slot.offset},
        {pb, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qsilu_i8u8, binds, (packed_words + 63u) / 64u,
                             1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

int metal_graph_qgelu_i8u8(const void* input, void* output, uint32_t elements,
                           float input_scale, int32_t input_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t input_dtype, uint32_t output_dtype) {
    if (!metal_ready()) return 0;
    size_t logical_bytes;
    if (!qbyte_unary_gpu_args_valid(input, output, elements, input_scale,
                                    input_zero_point, output_scale, output_zero_point,
                                    input_dtype, output_dtype, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    MetalGraphWindow input_slot, output_slot;
    if (!graph_window_packed_bytes(input, logical_bytes, 0, &input_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot)) return 0;
    MetalQByteUnaryParams params = {
        elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {
        {input_slot.slot->buffer, input_slot.bytes, input_slot.offset},
        {output_slot.slot->buffer, output_slot.bytes, output_slot.offset},
        {pb, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qgelu_i8u8, binds, (packed_words + 63u) / 64u,
                             1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

/* Stats and apply deliberately use separate command-buffer dispatches. The
 * reusable stats buffer remains private to this backend; packed activation
 * storage is the only graph-visible input/output representation. */
int metal_graph_qgroupnorm_i8u8(const void* input, const float* weight,
                                const float* bias, void* output, uint32_t batch,
                                uint32_t height, uint32_t width, uint32_t channels,
                                uint32_t groups, float input_scale,
                                int32_t input_zero_point, float output_scale,
                                int32_t output_zero_point, float epsilon,
                                uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    size_t affine_bytes;
    size_t stats_bytes;
    size_t packed_bytes;
    if (!metal_ready() ||
        !qgroupnorm_gpu_args_valid(input, weight, bias, output, batch, height,
                                   width, channels, groups, input_scale,
                                   input_zero_point, output_scale,
                                   output_zero_point, epsilon, input_dtype,
                                   output_dtype, &logical_bytes, &affine_bytes,
                                   &stats_bytes) ||
        !graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    MetalGraphWindow input_slot, output_slot;
    MetalTensorSlot* weight_slot = graph_ensure_device(weight, affine_bytes, 1);
    MetalTensorSlot* bias_slot = graph_ensure_device(bias, affine_bytes, 1);
    if (!graph_window_packed_bytes(input, logical_bytes, 0, &input_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot) ||
        !weight_slot || !bias_slot) return 0;
    id<MTLBuffer> stats_buffer = qgroupnorm_stats_ensure(stats_bytes);
    MetalQGroupNormParams params = {
        batch, height, width, channels, groups,
        input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, epsilon, 0.0f
    };
    id<MTLBuffer> params_buffer = create_buffer(sizeof(params), &params);
    if (!stats_buffer || !params_buffer) {
        VX_METAL_RELEASE(params_buffer);
        return 0;
    }
    MetalBinding stats_binds[3] = {
        {input_slot.slot->buffer, input_slot.bytes, input_slot.offset},
        {stats_buffer, stats_bytes}, {params_buffer, sizeof(params)}
    };
    MetalBinding apply_binds[6] = {
        {input_slot.slot->buffer, input_slot.bytes, input_slot.offset},
        {weight_slot->buffer, weight_slot->bytes}, {bias_slot->buffer, bias_slot->bytes},
        {stats_buffer, stats_bytes},
        {output_slot.slot->buffer, output_slot.bytes, output_slot.offset},
        {params_buffer, sizeof(params)}
    };
    uint32_t total_groups = batch * groups;
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    uint32_t apply_groups = packed_words / 64u + (packed_words % 64u != 0u);
    int ok = dispatch_kernel(&k_qgroupnorm_stats, stats_binds, total_groups, 1u, 1u) &&
             dispatch_kernel(&k_qgroupnorm_apply, apply_binds, apply_groups, 1u, 1u);
    VX_METAL_RELEASE(params_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

/* Metal dispatches the row-statistics and packed-word apply passes in order.
 * The private reusable stats buffer is the only F32 intermediate. */
int metal_graph_qlayernorm_i8u8(const void* input, const float* weight,
                                const float* bias, void* output, uint32_t rows,
                                uint32_t d_model, float input_scale,
                                int32_t input_zero_point, float output_scale,
                                int32_t output_zero_point, float epsilon,
                                uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    size_t affine_bytes;
    size_t stats_bytes;
    size_t packed_bytes;
    if (!metal_ready() ||
        !qlayernorm_gpu_args_valid(input, weight, bias, output, rows, d_model,
                                   input_scale, input_zero_point, output_scale,
                                   output_zero_point, epsilon, input_dtype,
                                   output_dtype, &logical_bytes, &affine_bytes,
                                   &stats_bytes) ||
        !graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    MetalGraphWindow input_slot, output_slot;
    MetalTensorSlot* weight_slot = graph_ensure_device(weight, affine_bytes, 1);
    MetalTensorSlot* bias_slot = graph_ensure_device(bias, affine_bytes, 1);
    if (!graph_window_packed_bytes(input, logical_bytes, 0, &input_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot) ||
        !weight_slot || !bias_slot) return 0;
    id<MTLBuffer> stats_buffer = qlayernorm_stats_ensure(stats_bytes);
    MetalQLayerNormParams params = {
        rows, d_model, input_dtype,
        output_dtype,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, epsilon, 0.0f
    };
    id<MTLBuffer> params_buffer = create_buffer(sizeof(params), &params);
    if (!stats_buffer || !params_buffer) {
        VX_METAL_RELEASE(params_buffer);
        return 0;
    }
    MetalBinding stats_binds[3] = {
        {input_slot.slot->buffer, input_slot.bytes, input_slot.offset},
        {stats_buffer, stats_bytes}, {params_buffer, sizeof(params)}
    };
    MetalBinding apply_binds[6] = {
        {input_slot.slot->buffer, input_slot.bytes, input_slot.offset},
        {weight_slot->buffer, weight_slot->bytes}, {bias_slot->buffer, bias_slot->bytes},
        {stats_buffer, stats_bytes},
        {output_slot.slot->buffer, output_slot.bytes, output_slot.offset},
        {params_buffer, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    uint32_t apply_groups = packed_words / 64u + (packed_words % 64u != 0u);
    int ok = dispatch_kernel(&k_qlayernorm_stats, stats_binds, rows, 1u, 1u) &&
             dispatch_kernel(&k_qlayernorm_apply, apply_binds, apply_groups, 1u, 1u);
    VX_METAL_RELEASE(params_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

int metal_graph_qsdpa_i8u8(const void* q, const void* k, const void* v,
                           const int32_t* mask, void* output, uint32_t batch,
                           uint32_t seq_q, uint32_t seq_kv, uint32_t d_model,
                           uint32_t heads, float q_scale, int32_t q_zero_point,
                           float k_scale, int32_t k_zero_point, float v_scale,
                           int32_t v_zero_point, float output_scale,
                           int32_t output_zero_point, float attention_scale,
                           uint32_t q_dtype, uint32_t k_dtype, uint32_t v_dtype,
                           uint32_t output_dtype, uint32_t causal,
                           uint32_t mask_mode) {
    size_t q_bytes;
    size_t kv_bytes;
    size_t mask_bytes;
    size_t packed_output_bytes;
    if (!metal_ready() ||
        !qsdpa_gpu_args_valid(q, k, v, mask, output, batch, seq_q, seq_kv,
                               d_model, heads, q_scale, q_zero_point, k_scale,
                               k_zero_point, v_scale, v_zero_point, output_scale,
                               output_zero_point, attention_scale, q_dtype, k_dtype,
                               v_dtype, output_dtype, causal, mask_mode, &q_bytes,
                               &kv_bytes, &mask_bytes) ||
        !graph_packed_bytes(q_bytes, &packed_output_bytes)) return 0;
    MetalGraphWindow q_slot, k_slot, v_slot, mask_slot, output_slot;
    if (!graph_window_packed_bytes(q, q_bytes, 0, &q_slot) ||
        !graph_window_packed_bytes(k, kv_bytes, 0, &k_slot) ||
        !graph_window_packed_bytes(v, kv_bytes, 0, &v_slot) ||
        !(mask_mode ? graph_window_device(mask, mask_bytes, 0, &mask_slot)
                    : graph_window_device(qsdpa_dummy_mask,
                                          sizeof(qsdpa_dummy_mask), 1, &mask_slot)) ||
        !graph_window_output_packed(output, q_bytes, &output_slot)) return 0;
    MetalQSDPAParams params = {
        seq_q, seq_kv, d_model, heads,
        batch, mask_mode, causal,
        q_dtype |
            (k_dtype << 8u) |
            (v_dtype << 16u) |
            (output_dtype << 24u),
        q_zero_point, k_zero_point, v_zero_point, output_zero_point,
        q_scale, k_scale, v_scale, output_scale,
        attention_scale, 0.0f, 0.0f, 0.0f
    };
    id<MTLBuffer> params_buffer = create_buffer(sizeof(params), &params);
    if (!params_buffer) return 0;
    MetalBinding binds[6] = {
        {q_slot.slot->buffer, q_slot.bytes, q_slot.offset},
        {k_slot.slot->buffer, k_slot.bytes, k_slot.offset},
        {v_slot.slot->buffer, v_slot.bytes, v_slot.offset},
        {mask_slot.slot->buffer, mask_slot.bytes, mask_slot.offset},
        {output_slot.slot->buffer, output_slot.bytes, output_slot.offset},
        {params_buffer, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_qsdpa_int8, binds, seq_q, heads, batch);
    VX_METAL_RELEASE(params_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

/* qArgMaxInt8 owns one output index per [outer,inner] coordinate. Input stays
 * byte-packed, output is conventional I32 storage, and the 16-byte uniform
 * exactly mirrors the WebGPU ABI. */
int metal_graph_qargmax_i8u8(const void* input, int32_t* output,
                             uint32_t outer, uint32_t axis_size,
                             uint32_t inner, uint32_t input_dtype) {
    size_t input_bytes;
    size_t output_bytes;
    uint32_t output_elements;
    if (!metal_ready() ||
        !qargmax_gpu_args_valid(input, output, outer, axis_size, inner,
                                input_dtype, &input_bytes, &output_bytes,
                                &output_elements)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    MetalTensorSlot* output_slot = graph_output_slot(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    MetalQArgMaxParams params = {
        outer, axis_size, inner, input_dtype
    };
    id<MTLBuffer> params_buffer = create_buffer(sizeof(params), &params);
    if (!params_buffer) return 0;
    MetalBinding binds[3] = {
        {input_slot->buffer, input_slot->bytes},
        {output_slot->buffer, output_slot->bytes},
        {params_buffer, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_qargmax_int8, binds,
                             output_elements / 64u + (output_elements % 64u != 0u),
                             1u, 1u);
    VX_METAL_RELEASE(params_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_qmaskedmean_i8u8(const void* input, const int32_t* mask,
                                 void* output, uint32_t batch,
                                 uint32_t sequence, uint32_t width,
                                 float input_scale, int32_t input_zero_point,
                                 float output_scale, int32_t output_zero_point,
                                 uint32_t input_dtype, uint32_t output_dtype) {
    size_t input_bytes;
    size_t mask_bytes;
    size_t output_bytes;
    size_t packed_output_bytes;
    if (!metal_ready() ||
        !qmaskedmean_gpu_args_valid(input, mask, output, batch, sequence, width,
                                    input_scale, input_zero_point, output_scale,
                                    output_zero_point, input_dtype, output_dtype,
                                    &input_bytes, &mask_bytes, &output_bytes) ||
        !graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    MetalTensorSlot* mask_slot = graph_ensure_device(mask, mask_bytes, 0);
    MetalTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !mask_slot || !output_slot) return 0;
    MetalQMaskedMeanParams params = {
        batch, sequence, width, input_dtype,
        output_dtype, 0u, 0u, 0u,
        input_zero_point, output_zero_point, input_scale, output_scale
    };
    id<MTLBuffer> params_buffer = create_buffer(sizeof(params), &params);
    if (!params_buffer) return 0;
    MetalBinding binds[4] = {
        {input_slot->buffer, input_slot->bytes},
        {mask_slot->buffer, mask_slot->bytes},
        {output_slot->buffer, packed_output_bytes},
        {params_buffer, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qmaskedmean_int8, binds,
                             (packed_words + 63u) / 64u, 1u, 1u);
    VX_METAL_RELEASE(params_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_requantize_linear_i8u8(const void* input, uint32_t input_elements,
                                       void* output, uint32_t output_elements,
                                       float input_scale, int32_t input_zero_point,
                                       float output_scale, int32_t output_zero_point,
                                       uint32_t input_dtype, uint32_t output_dtype) {
    if (!metal_ready()) return 0;
    float multiplier;
    size_t logical_bytes;
    if (!requantize_gpu_args_valid(input, input_elements, output, output_elements,
                                   input_scale, input_zero_point,
                                   output_scale, output_zero_point,
                                   input_dtype, output_dtype,
                                   &multiplier, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    MetalTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !output_slot) return 0;
    MetalRequantizeLinearParams params = {
        input_elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        multiplier, 0.0f, 0.0f, 0.0f
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {
        {input_slot->buffer, input_slot->bytes},
        {output_slot->buffer, packed_bytes}, {pb, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_requantize_linear_i8u8, binds,
                             (packed_words + 63u) / 64u, 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

static const int32_t* qconv_zero_bias_get(uint32_t output_channels) {
    if (output_channels == 0 ||
        (size_t)output_channels > SIZE_MAX / sizeof(int32_t)) return NULL;
    for (QConvZeroBiasBacking* block = qconv_zero_bias_backings; block;
         block = block->next) {
        /* Domain-enforced immutable slots retain an exact readable range.
         * A smaller node needs its own durable host key or it would narrow
         * the slot later reused by an earlier wider convolution. */
        if (block->elements == output_channels) return block->values;
    }
    QConvZeroBiasBacking* block = (QConvZeroBiasBacking*)calloc(1, sizeof(*block));
    if (!block) return NULL;
    block->values = (int32_t*)calloc((size_t)output_channels, sizeof(*block->values));
    if (!block->values) {
        free(block);
        return NULL;
    }
    block->elements = output_channels;
    block->next = qconv_zero_bias_backings;
    qconv_zero_bias_backings = block;
    return block->values;
}

static void qconv_zero_bias_release_state(void* opaque_state) {
    MetalContextState* state = (MetalContextState*)opaque_state;
    if (!state) return;
    while (state->qconv_zero_bias_storage) {
        QConvZeroBiasBacking* block = state->qconv_zero_bias_storage;
        state->qconv_zero_bias_storage = block->next;
        free(block->values);
        free(block);
    }
}

typedef struct {
    uint32_t batch;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t input_channels;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t output_channels;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t groups;
    uint32_t input_type;
    uint32_t weight_type;
    uint32_t output_type;
    uint32_t relu;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    int32_t pad1;
    float input_scale;
    float output_scale;
    float pad2;
    float pad3;
} MetalQConv2DParams;

_Static_assert(sizeof(MetalQConv2DParams) == 112, "qConv2DInt8 uniform ABI");

static int qconv_mul_u64(uint64_t left, uint64_t right, uint64_t* out) {
    if (!out || (left != 0 && right > UINT64_MAX / left)) return 0;
    *out = left * right;
    return 1;
}

static int32_t qconv_i32_at(const int32_t* values, uint32_t index) {
    int32_t value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static float qconv_f32_at(const float* values, uint32_t index) {
    float value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static int qconv_gpu_args_valid(const void* input, const void* weight,
                                const float* weight_scales,
                                const int32_t* weight_zero_points,
                                const int32_t* bias, void* output,
                                uint32_t batch, uint32_t input_height,
                                uint32_t input_width, uint32_t input_channels,
                                uint32_t output_height, uint32_t output_width,
                                uint32_t output_channels, uint32_t kernel_height,
                                uint32_t kernel_width, uint32_t input_per_group,
                                uint32_t stride_y, uint32_t stride_x,
                                uint32_t dilation_y, uint32_t dilation_x,
                                uint32_t padding_top, uint32_t padding_left,
                                uint32_t padding_bottom, uint32_t padding_right,
                                uint32_t groups, uint32_t relu,
                                float input_scale, int32_t input_zero_point,
                                float output_scale, int32_t output_zero_point,
                                uint32_t input_dtype, uint32_t weight_dtype,
                                uint32_t output_dtype,
                                size_t* input_bytes, size_t* weight_bytes,
                                size_t* output_bytes, size_t* metadata_bytes) {
    uint64_t input_elements = 1, weight_elements = 1, output_elements = 1;
    uint64_t terms = 1, padded_height, padded_width;
    uint64_t effective_height, effective_width;
    uint64_t expected_height, expected_width;
    uint64_t input_magnitude;
    if (!input || !weight || !weight_scales || !weight_zero_points || !output ||
        output == input || output == weight || !batch || !input_height ||
        !input_width || !input_channels || !output_height || !output_width ||
        !output_channels || !kernel_height || !kernel_width || !input_per_group ||
        !stride_y || !stride_x || !dilation_y || !dilation_x || !groups || relu > 2u ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) ||
        input_channels % groups || output_channels % groups ||
        (uint64_t)input_per_group * groups != input_channels) return 0;
    if (!qconv_mul_u64(input_elements, batch, &input_elements) ||
        !qconv_mul_u64(input_elements, input_height, &input_elements) ||
        !qconv_mul_u64(input_elements, input_width, &input_elements) ||
        !qconv_mul_u64(input_elements, input_channels, &input_elements) ||
        !qconv_mul_u64(weight_elements, output_channels, &weight_elements) ||
        !qconv_mul_u64(weight_elements, kernel_height, &weight_elements) ||
        !qconv_mul_u64(weight_elements, kernel_width, &weight_elements) ||
        !qconv_mul_u64(weight_elements, input_per_group, &weight_elements) ||
        !qconv_mul_u64(output_elements, batch, &output_elements) ||
        !qconv_mul_u64(output_elements, output_height, &output_elements) ||
        !qconv_mul_u64(output_elements, output_width, &output_elements) ||
        !qconv_mul_u64(output_elements, output_channels, &output_elements) ||
        !qconv_mul_u64(terms, kernel_height, &terms) ||
        !qconv_mul_u64(terms, kernel_width, &terms) ||
        !qconv_mul_u64(terms, input_per_group, &terms) ||
        input_elements > UINT32_MAX || weight_elements > UINT32_MAX ||
        output_elements > UINT32_MAX || input_elements > SIZE_MAX ||
        weight_elements > SIZE_MAX || output_elements > SIZE_MAX ||
        (size_t)output_channels > SIZE_MAX / sizeof(float) ||
        (size_t)output_channels > SIZE_MAX / sizeof(int32_t)) return 0;
    if ((uint64_t)(kernel_height - 1u) > (UINT64_MAX - 1u) / dilation_y ||
        (uint64_t)(kernel_width - 1u) > (UINT64_MAX - 1u) / dilation_x) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    effective_height = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_width = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return 0;
    expected_height = (padded_height - effective_height) / stride_y + 1u;
    expected_width = (padded_width - effective_width) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    {
        int64_t low = (int64_t)(input_dtype == VX_DTYPE_I8 ? -128 : 0) -
            input_zero_point;
        int64_t high = (int64_t)(input_dtype == VX_DTYPE_I8 ? 127 : 255) -
            input_zero_point;
        uint64_t low_magnitude = (uint64_t)(low < 0 ? -low : low);
        uint64_t high_magnitude = (uint64_t)(high < 0 ? -high : high);
        input_magnitude = low_magnitude > high_magnitude ? low_magnitude : high_magnitude;
    }
    for (uint32_t output_channel = 0; output_channel < output_channels;
         output_channel++) {
        const float weight_scale = qconv_f32_at(weight_scales, output_channel);
        const int32_t weight_zero_point = qconv_i32_at(weight_zero_points, output_channel);
        int64_t low, high;
        uint64_t weight_magnitude, accumulator_bound, bias_magnitude = 0;
        if (!isfinite(weight_scale) || weight_scale <= 0.0f ||
            !qbyte_dtype_zero_point_valid(weight_dtype, weight_zero_point)) return 0;
        low = (int64_t)(weight_dtype == VX_DTYPE_I8 ? -128 : 0) -
            weight_zero_point;
        high = (int64_t)(weight_dtype == VX_DTYPE_I8 ? 127 : 255) -
            weight_zero_point;
        weight_magnitude = (uint64_t)(low < 0 ? -low : low);
        {
            uint64_t high_magnitude = (uint64_t)(high < 0 ? -high : high);
            if (high_magnitude > weight_magnitude) weight_magnitude = high_magnitude;
        }
        if (input_magnitude && weight_magnitude &&
            terms > (uint64_t)INT32_MAX / input_magnitude / weight_magnitude) return 0;
        accumulator_bound = input_magnitude * weight_magnitude * terms;
        if (bias) {
            int64_t bias_value = qconv_i32_at(bias, output_channel);
            bias_magnitude = (uint64_t)(bias_value < 0 ? -bias_value : bias_value);
        }
        if (accumulator_bound > (uint64_t)INT32_MAX ||
            bias_magnitude > (uint64_t)INT32_MAX - accumulator_bound) return 0;
    }
    *input_bytes = (size_t)input_elements;
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    *metadata_bytes = (size_t)output_channels * sizeof(int32_t);
    return 1;
}

int metal_graph_qconv2d_i8u8(const void* input, const void* weight,
                              const float* weight_scales,
                              const int32_t* weight_zero_points,
                              const int32_t* bias, void* output,
                              uint32_t batch, uint32_t input_height,
                              uint32_t input_width, uint32_t input_channels,
                              uint32_t output_height, uint32_t output_width,
                              uint32_t output_channels, uint32_t kernel_height,
                              uint32_t kernel_width, uint32_t input_per_group,
                              uint32_t stride_y, uint32_t stride_x,
                              uint32_t dilation_y, uint32_t dilation_x,
                              uint32_t padding_top, uint32_t padding_left,
                              uint32_t padding_bottom, uint32_t padding_right,
                              uint32_t groups, uint32_t relu,
                              float input_scale, int32_t input_zero_point,
                              float output_scale, int32_t output_zero_point,
                              uint32_t input_dtype, uint32_t weight_dtype,
                              uint32_t output_dtype) {
    if (!metal_ready()) return 0;
    size_t input_bytes, weight_bytes, output_bytes, metadata_bytes;
    if (!qconv_gpu_args_valid(input, weight, weight_scales, weight_zero_points,
                              bias, output, batch, input_height, input_width,
                              input_channels, output_height, output_width,
                              output_channels, kernel_height, kernel_width,
                              input_per_group, stride_y, stride_x, dilation_y,
                              dilation_x, padding_top, padding_left,
                              padding_bottom, padding_right, groups, relu,
                              input_scale, input_zero_point, output_scale,
                              output_zero_point, input_dtype, weight_dtype,
                              output_dtype, &input_bytes, &weight_bytes,
                              &output_bytes, &metadata_bytes)) return 0;
    const int32_t* bound_bias = bias ? bias : qconv_zero_bias_get(output_channels);
    if (!bound_bias) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    MetalTensorSlot* weight_slot = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    MetalTensorSlot* scales_slot = graph_ensure_device(weight_scales,
                                                        (size_t)output_channels * sizeof(float), 1);
    MetalTensorSlot* zero_points_slot = graph_ensure_device(weight_zero_points, metadata_bytes, 1);
    MetalTensorSlot* bias_slot = graph_ensure_device(bound_bias, metadata_bytes, 1);
    MetalTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !weight_slot || !scales_slot || !zero_points_slot ||
        !bias_slot || !output_slot) return 0;
    MetalQConv2DParams params = {
        batch, input_height, input_width, input_channels,
        output_height, output_width, output_channels, kernel_height,
        kernel_width, stride_y, stride_x, dilation_y,
        dilation_x, padding_top, padding_left, groups,
        input_dtype,
        weight_dtype,
        output_dtype, relu,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[7] = {
        {input_slot->buffer, input_slot->bytes}, {weight_slot->buffer, weight_slot->bytes},
        {scales_slot->buffer, scales_slot->bytes}, {zero_points_slot->buffer, zero_points_slot->bytes},
        {bias_slot->buffer, bias_slot->bytes}, {output_slot->buffer, packed_output_bytes},
        {pb, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int ok = dispatch_kernel(&k_qconv2d_int8, binds,
                             (packed_words + 63u) / 64u, 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

typedef struct { uint32_t elements, pad0, pad1, pad2; } MetalTypedCopyParams;
typedef struct { uint32_t size, axis_offset, input_axis, output_axis, inner, pad0, pad1, pad2; } MetalTypedConcatParams;
typedef struct { uint32_t n, h, w, c, out_h, out_w, ky, kx, sy, sx, py, px, dtype, pad0, pad1, pad2; } MetalTypedMaxPoolParams;
typedef struct { uint32_t n, h, w, c, out_h, out_w, pad0, pad1; } MetalTypedResizeParams;
typedef struct { uint32_t elements, output_type, zero_point_type, has_zero_point; } MetalTypedQuantizeParams;
typedef struct { uint32_t size, input_type, scale_type, zero_point_type, output_type, has_zero_point, pad0, pad1; } MetalTypedDequantizeParams;

_Static_assert(sizeof(MetalTypedCopyParams) == 16, "copyTyped uniform ABI");
_Static_assert(sizeof(MetalTypedConcatParams) == 32, "concatCopyTyped uniform ABI");
_Static_assert(sizeof(MetalTypedMaxPoolParams) == 64, "maxPool2DTyped uniform ABI");
_Static_assert(sizeof(MetalTypedResizeParams) == 32, "resizeNearestTyped uniform ABI");
_Static_assert(sizeof(MetalTypedQuantizeParams) == 16, "quantizeLinearTyped uniform ABI");
_Static_assert(sizeof(MetalTypedDequantizeParams) == 32, "dequantizeLinearTyped uniform ABI");

static int typed_shape_qdesc_valid(float scale, int32_t zero_point, uint32_t dtype) {
    return isfinite(scale) && scale > 0.0f && qbyte_dtype_zero_point_valid(dtype, zero_point);
}

static int typed_shape_qdesc_same(float input_scale, int32_t input_zero_point,
                                  uint32_t input_dtype, float output_scale,
                                  int32_t output_zero_point, uint32_t output_dtype) {
    return typed_shape_qdesc_valid(input_scale, input_zero_point, input_dtype) &&
        typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype) &&
        input_dtype == output_dtype && input_zero_point == output_zero_point && input_scale == output_scale;
}

static int typed_shape_nhwc_elements(uint32_t n, uint32_t h, uint32_t w, uint32_t c,
                                     uint32_t* elements) {
    uint64_t value = n;
    if (!n || !h || !w || !c || !elements || !qconv_mul_u64(value, h, &value) ||
        !qconv_mul_u64(value, w, &value) || !qconv_mul_u64(value, c, &value) ||
        value > UINT32_MAX) return 0;
    *elements = (uint32_t)value;
    return 1;
}

static uint32_t typed_shape_groups(uint32_t elements) { return elements / 64u + (elements % 64u != 0u); }
static uint32_t typed_shape_packed_groups(size_t bytes) { return typed_shape_groups((uint32_t)(bytes / sizeof(uint32_t))); }
static uint32_t typed_shape_zero_word(int32_t zero_point, uint32_t dtype) {
    return dtype == VX_DTYPE_I8 ? (uint32_t)(uint8_t)(int8_t)zero_point :
        (uint32_t)(uint8_t)zero_point;
}

static int graph_zero_packed_output(MetalTensorSlot* slot, size_t bytes) {
    if (!slot || !slot->buffer || bytes == 0) return 0;
    memset([slot->buffer contents], 0, bytes);
    slot->host_dirty = 0;
    slot->device_dirty = 0;
    return 1;
}

int metal_graph_quantize_typed_f32_i8u8(const float* input, uint32_t elements,
                                        void* output, float output_scale,
                                        int32_t output_zero_point, uint32_t output_dtype) {
    if (!metal_ready() || !input || !output || input == output || !elements ||
        !typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype) ||
        (uint64_t)elements > SIZE_MAX / sizeof(float)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(elements, &packed_output_bytes)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_device(input, (size_t)elements * sizeof(float), 0);
    MetalTensorSlot* output_slot = graph_output_packed_bytes(output, elements);
    if (!input_slot || !output_slot) return 0;
    uint32_t zero_word = typed_shape_zero_word(output_zero_point, output_dtype);
    MetalTypedQuantizeParams params = {
        elements, output_dtype, output_dtype, 1u
    };
    id<MTLBuffer> scale_buffer = create_buffer(sizeof(output_scale), &output_scale);
    id<MTLBuffer> zero_buffer = create_buffer(sizeof(zero_word), &zero_word);
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!scale_buffer || !zero_buffer || !pb) {
        VX_METAL_RELEASE(scale_buffer); VX_METAL_RELEASE(zero_buffer); VX_METAL_RELEASE(pb);
        return 0;
    }
    MetalBinding binds[5] = {{input_slot->buffer, input_slot->bytes}, {scale_buffer, sizeof(output_scale)},
                             {zero_buffer, sizeof(zero_word)}, {output_slot->buffer, packed_output_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_quantize_typed_i8u8, binds, typed_shape_packed_groups(packed_output_bytes), 1u, 1u);
    VX_METAL_RELEASE(scale_buffer); VX_METAL_RELEASE(zero_buffer); VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_dequantize_typed_i8u8_f32(const void* input, uint32_t elements,
                                          float input_scale, int32_t input_zero_point,
                                          uint32_t input_dtype, float* output) {
    if (!metal_ready() || !input || !output || input == output || !elements ||
        !typed_shape_qdesc_valid(input_scale, input_zero_point, input_dtype) ||
        (uint64_t)elements > SIZE_MAX / sizeof(float)) return 0;
    size_t packed_input_bytes;
    if (!graph_packed_bytes(elements, &packed_input_bytes)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_packed_bytes(input, elements, 0);
    MetalTensorSlot* output_slot = graph_output_slot(output, (size_t)elements * sizeof(float));
    if (!input_slot || !output_slot) return 0;
    uint32_t zero_word = typed_shape_zero_word(input_zero_point, input_dtype);
    MetalTypedDequantizeParams params = {
        elements, input_dtype, VX_DTYPE_F32, input_dtype,
        VX_DTYPE_F32, 1u, 0u, 0u
    };
    id<MTLBuffer> scale_buffer = create_buffer(sizeof(input_scale), &input_scale);
    id<MTLBuffer> zero_buffer = create_buffer(sizeof(zero_word), &zero_word);
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!scale_buffer || !zero_buffer || !pb) {
        VX_METAL_RELEASE(scale_buffer); VX_METAL_RELEASE(zero_buffer); VX_METAL_RELEASE(pb);
        return 0;
    }
    MetalBinding binds[5] = {{input_slot->buffer, packed_input_bytes}, {scale_buffer, sizeof(input_scale)},
                             {zero_buffer, sizeof(zero_word)}, {output_slot->buffer, output_slot->bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_dequantize_typed_i8u8, binds, typed_shape_groups(elements), 1u, 1u);
    VX_METAL_RELEASE(scale_buffer); VX_METAL_RELEASE(zero_buffer); VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_copy_i8u8(const void* input, uint32_t input_elements, void* output,
                          uint32_t output_elements, float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point, uint32_t input_dtype,
                          uint32_t output_dtype) {
    if (!metal_ready() || !input || !output || input == output || !input_elements || input_elements != output_elements ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(input_elements, &packed_bytes)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    MetalTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    MetalTypedCopyParams params = {input_elements, 0u, 0u, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{input_slot->buffer, input_slot->bytes}, {output_slot->buffer, packed_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_copy_typed_i8u8, binds, typed_shape_packed_groups(packed_bytes), 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_transpose_i8u8(
        const void* input, void* output, const uint32_t* input_shape,
        const uint32_t* permutation, uint32_t rank, uint32_t elements,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    uint32_t input_strides[8] = {0};
    uint32_t output_shape[8] = {0};
    uint32_t output_strides[8] = {0};
    uint32_t metadata[18] = {0};
    uint32_t seen = 0;
    uint64_t product = 1;
    uint64_t stride = 1;
    size_t packed_bytes;
    if (!metal_ready() || !input || !output || !input_shape || !permutation ||
        rank == 0 || rank > 8 || elements == 0 ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype,
                                output_scale, output_zero_point, output_dtype) ||
        qsdpa_ranges_overlap(output, elements, input, elements)) return 0;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        if (!input_shape[reverse] || stride > UINT32_MAX) return 0;
        input_strides[reverse] = (uint32_t)stride;
        stride *= input_shape[reverse];
        if (stride > UINT32_MAX) return 0;
    }
    if (stride != elements) return 0;
    for (uint32_t dimension = 0; dimension < rank; dimension++) {
        uint32_t source = permutation[dimension];
        if (source >= rank || (seen & (1u << source))) return 0;
        seen |= 1u << source;
        output_shape[dimension] = input_shape[source];
        if (product > UINT32_MAX / output_shape[dimension]) return 0;
        product *= output_shape[dimension];
    }
    if (product != elements) return 0;
    stride = 1;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        output_strides[reverse] = (uint32_t)stride;
        stride *= output_shape[reverse];
    }
    if (!graph_packed_bytes(elements, &packed_bytes)) return 0;
    MetalTensorSlot* input_slot =
        graph_ensure_packed_bytes(input, elements, 0);
    MetalTensorSlot* output_slot =
        graph_output_packed_bytes(output, elements);
    if (!input_slot || !output_slot) return 0;
    metadata[0] = elements;
    metadata[1] = rank;
    for (uint32_t dimension = 0; dimension < rank; dimension++) {
        metadata[2 + dimension] = output_strides[dimension];
        metadata[2 + rank + dimension] =
            input_strides[permutation[dimension]];
    }
    size_t metadata_bytes = (size_t)(2 + 2 * rank) * sizeof(uint32_t);
    id<MTLBuffer> metadata_buffer =
        create_buffer(metadata_bytes, metadata);
    if (!metadata_buffer) return 0;
    MetalBinding binds[3] = {
        {input_slot->buffer, packed_bytes},
        {output_slot->buffer, packed_bytes},
        {metadata_buffer, metadata_bytes},
    };
    int ok = dispatch_kernel(
        &k_transpose_typed_i8u8, binds,
        typed_shape_packed_groups(packed_bytes), 1u, 1u);
    VX_METAL_RELEASE(metadata_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_concat_i8u8(const void* const* inputs, const uint32_t* input_elements,
                            const uint32_t* input_axes, const float* input_scales,
                            const int32_t* input_zero_points, const uint32_t* input_dtypes,
                            uint32_t input_count, void* output, uint32_t output_elements,
                            uint32_t output_axis, uint32_t inner, float output_scale,
                            int32_t output_zero_point, uint32_t output_dtype) {
    if (!metal_ready() || !inputs || !input_elements || !input_axes || !input_scales || !input_zero_points ||
        !input_dtypes || !output || !input_count || !output_elements || !output_axis || !inner ||
        !typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype)) return 0;
    uint64_t output_width = (uint64_t)output_axis * inner;
    if (!output_width || output_width > output_elements || output_elements % output_width) return 0;
    uint64_t outer = output_elements / output_width, axis_sum = 0;
    for (uint32_t index = 0; index < input_count; index++) {
        uint64_t expected = 0;
        if (!qconv_mul_u64(outer, input_axes[index], &expected) ||
            !qconv_mul_u64(expected, inner, &expected)) {
            return 0;
        }
        if (!inputs[index] || inputs[index] == output || !input_axes[index] || expected != input_elements[index] ||
            !typed_shape_qdesc_same(input_scales[index], input_zero_points[index], input_dtypes[index], output_scale, output_zero_point, output_dtype) ||
            axis_sum > UINT32_MAX - input_axes[index]) return 0;
        axis_sum += input_axes[index];
    }
    if (axis_sum != output_axis) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    MetalTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!output_slot || !graph_zero_packed_output(output_slot, packed_output_bytes)) return 0;
    uint32_t axis_offset = 0;
    for (uint32_t index = 0; index < input_count; index++) {
        MetalTensorSlot* input_slot = graph_ensure_packed_bytes(inputs[index], input_elements[index], 0);
        if (!input_slot) return 0;
        MetalTypedConcatParams params = {input_elements[index], axis_offset, input_axes[index], output_axis, inner, 0u, 0u, 0u};
        id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
        if (!pb) return 0;
        MetalBinding binds[3] = {{input_slot->buffer, input_slot->bytes}, {output_slot->buffer, packed_output_bytes}, {pb, sizeof(params)}};
        int ok = dispatch_kernel(&k_concat_typed_i8u8, binds, typed_shape_groups(input_elements[index]), 1u, 1u);
        VX_METAL_RELEASE(pb);
        if (!ok) return 0;
        axis_offset += input_axes[index];
    }
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_maxpool2d_i8u8(const void* input, void* output, uint32_t batch,
                               uint32_t input_height, uint32_t input_width, uint32_t channels,
                               uint32_t output_height, uint32_t output_width, uint32_t kernel_y,
                               uint32_t kernel_x, uint32_t stride_y, uint32_t stride_x,
                               uint32_t padding_top, uint32_t padding_left, uint32_t padding_bottom,
                               uint32_t padding_right, float input_scale, int32_t input_zero_point,
                               float output_scale, int32_t output_zero_point, uint32_t input_dtype,
                               uint32_t output_dtype) {
    uint32_t input_elements, output_elements;
    uint64_t padded_height, padded_width, expected_height, expected_width;
    if (!metal_ready() || !input || !output || input == output || !kernel_y || !kernel_x || !stride_y || !stride_x ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype) ||
        !typed_shape_nhwc_elements(batch, input_height, input_width, channels, &input_elements) ||
        !typed_shape_nhwc_elements(batch, output_height, output_width, channels, &output_elements)) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    if (padded_height < kernel_y || padded_width < kernel_x) return 0;
    expected_height = (padded_height - kernel_y) / stride_y + 1u;
    expected_width = (padded_width - kernel_x) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    MetalTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    MetalTypedMaxPoolParams params = {
        batch, input_height, input_width, channels, output_height, output_width,
        kernel_y, kernel_x, stride_y, stride_x, padding_top, padding_left,
        input_dtype, 0u, 0u, 0u
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{input_slot->buffer, input_slot->bytes}, {output_slot->buffer, packed_output_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_maxpool_typed_i8u8, binds, typed_shape_packed_groups(packed_output_bytes), 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_resize_nearest_i8u8(const void* input, void* output, uint32_t batch,
                                    uint32_t input_height, uint32_t input_width, uint32_t channels,
                                    uint32_t output_height, uint32_t output_width, float input_scale,
                                    int32_t input_zero_point, float output_scale, int32_t output_zero_point,
                                    uint32_t input_dtype, uint32_t output_dtype) {
    uint32_t input_elements, output_elements;
    if (!metal_ready() || !input || !output || input == output ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype) ||
        !typed_shape_nhwc_elements(batch, input_height, input_width, channels, &input_elements) ||
        !typed_shape_nhwc_elements(batch, output_height, output_width, channels, &output_elements)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    MetalTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    MetalTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    MetalTypedResizeParams params = {batch, input_height, input_width, channels, output_height, output_width, 0u, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{input_slot->buffer, input_slot->bytes}, {output_slot->buffer, packed_output_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_resize_nearest_typed_i8u8, binds, typed_shape_packed_groups(packed_output_bytes), 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int metal_graph_quantize_linear_i8(const float* in, signed char* out, long n,
                                   float input_scale, int input_zp,
                                   float output_scale, int output_zp) {
    if (!in || !out || n <= 0 || output_scale <= 0.0f) return 0;
    size_t in_bytes = (size_t)n * sizeof(float);
    size_t packed_words = ((size_t)n + 3u) / 4u;
    size_t out_bytes = packed_words * sizeof(uint32_t);
    MetalTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct {
        uint32_t size;
        int32_t input_zp;
        int32_t output_zp;
        uint32_t has_input_scale;
        float input_scale;
        float output_scale;
        uint32_t pad0;
        uint32_t pad1;
    } params = {
        (uint32_t)n, (int32_t)input_zp, (int32_t)output_zp,
        input_scale > 0.0f ? 1u : 0u, input_scale, output_scale, 0u, 0u
    };
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, in_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_quantize, binds, (uint32_t)((packed_words + 63u) / 64u), 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return metal_graph_sync_host(out, (size_t)n, 0);
}

static int metal_graph_profile_common(const MetalKernel* kernel, const float* in, float* out,
                                      int n, int h, int w, int c, long out_elems_per_batch,
                                      uint32_t gx, uint32_t gy) {
    if (!kernel || !in || !out || n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_elems_per_batch <= 0) return 0;
    size_t in_bytes = (size_t)n * (size_t)h * (size_t)w * (size_t)c * sizeof(float);
    size_t out_bytes = (size_t)n * (size_t)out_elems_per_batch * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)h, (uint32_t)w, (uint32_t)c, (uint32_t)n};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, in_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(kernel, binds, gx, gy, (uint32_t)n);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_spatial_softargmax_y_f32(const float* in, float* out, int n, int h, int w, int c) {
    return metal_graph_profile_common(&k_spatial_softargmax_y, in, out, n, h, w, c,
                                      (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int metal_graph_profile_x_f32(const float* in, float* out, int n, int h, int w, int c) {
    return metal_graph_profile_common(&k_profile_x, in, out, n, h, w, c,
                                      (long)2 * c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int metal_graph_profile_y_f32(const float* in, float* out, int n, int h, int w, int c) {
    return metal_graph_profile_common(&k_profile_y, in, out, n, h, w, c,
                                      (long)2 * c * h, ((uint32_t)h + 63u) / 64u, (uint32_t)c);
}

int metal_graph_mean_height_f32(const float* in, float* out, int n, int h, int w, int c) {
    return metal_graph_profile_common(&k_mean_height, in, out, n, h, w, c,
                                      (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int metal_graph_nms_f32(const float* boxes, const float* scores, float* out,
                        int batches, int spatial, int classes, int max_output,
                        int output_rows, float iou_threshold, float score_threshold) {
    if (!boxes || !scores || !out || batches <= 0 || spatial <= 0 || classes <= 0 ||
        max_output <= 0 || output_rows <= 0) return 0;
    size_t boxes_bytes = (size_t)batches * (size_t)spatial * 4u * sizeof(float);
    size_t scores_bytes = (size_t)batches * (size_t)classes * (size_t)spatial * sizeof(float);
    size_t out_bytes = (size_t)output_rows * 3u * sizeof(float);
    MetalTensorSlot* sb = graph_ensure_device(boxes, boxes_bytes, 0);
    MetalTensorSlot* ss = graph_ensure_device(scores, scores_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sb || !ss || !dst) return 0;
    struct {
        uint32_t batches, spatial, classes, max_output, output_rows;
        float iou_threshold, score_threshold;
        uint32_t pad;
    } params = {(uint32_t)batches, (uint32_t)spatial, (uint32_t)classes, (uint32_t)max_output,
                (uint32_t)output_rows, iou_threshold, score_threshold, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {sb->buffer, boxes_bytes}, {ss->buffer, scores_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_nms, binds, 1, 1, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
