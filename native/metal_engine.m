#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "metal_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define METAL_SHADER_DIR "native/shaders/metal"
#define METAL_GRAPH_MAX_TENSORS 4096
#define METAL_MAX_BINDINGS 8

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(objc_arc)
#define VX_METAL_RELEASE(obj) do { (obj) = nil; } while (0)
#else
#define VX_METAL_RELEASE(obj) do { if ((obj)) { [(obj) release]; (obj) = nil; } } while (0)
#endif

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;

typedef struct {
    const char* name;
    const char* path;
    int binding_count;
    int uniform_binding;
    int wg_x;
    int wg_y;
    int wg_z;
    id<MTLComputePipelineState> pipeline;
    int ready;
    int failed;
} MetalKernel;

typedef struct {
    const void* host;
    size_t bytes;
    size_t cap;
    id<MTLBuffer> buffer;
    int host_dirty;
    int device_dirty;
    int is_weight;
} MetalTensorSlot;

typedef struct {
    id<MTLBuffer> buffer;
    size_t bytes;
} MetalBinding;

static MetalTensorSlot graph_slots[METAL_GRAPH_MAX_TENSORS];
static int graph_slot_count = 0;

static MetalKernel k_mul = {"mul", METAL_SHADER_DIR "/mul.metal", 4, 3, 64, 1, 1, nil, 0, 0};
static MetalKernel k_sub = {"sub", METAL_SHADER_DIR "/sub.metal", 4, 3, 64, 1, 1, nil, 0, 0};
static MetalKernel k_div = {"div", METAL_SHADER_DIR "/div.metal", 4, 3, 64, 1, 1, nil, 0, 0};
static MetalKernel k_split = {"split", METAL_SHADER_DIR "/split.metal", 3, 2, 64, 1, 1, nil, 0, 0};
static MetalKernel k_conv1d = {"conv1D", METAL_SHADER_DIR "/conv1D.metal", 5, 4, 64, 1, 1, nil, 0, 0};
static MetalKernel k_sdpa = {"sDPA", METAL_SHADER_DIR "/sDPA.metal", 3, 2, 64, 1, 1, nil, 0, 0};
static MetalKernel k_cross_sdpa = {"crossSDPA", METAL_SHADER_DIR "/crossSDPA.metal", 5, 4, 64, 1, 1, nil, 0, 0};
static MetalKernel k_cross_attention = {"crossAttentionF32", METAL_SHADER_DIR "/crossAttentionF32.metal", 7, 6, 64, 1, 1, nil, 0, 0};
static MetalKernel k_quantize = {"quantizeLinear", METAL_SHADER_DIR "/quantizeLinear.metal", 3, 2, 64, 1, 1, nil, 0, 0};
static MetalKernel k_dequantize = {"dequantizeLinear", METAL_SHADER_DIR "/dequantizeLinear.metal", 5, 4, 64, 1, 1, nil, 0, 0};
static MetalKernel k_spatial_softargmax_y = {"spatialSoftargmaxY", METAL_SHADER_DIR "/spatialSoftargmaxY.metal", 3, 2, 64, 1, 1, nil, 0, 0};
static MetalKernel k_profile_x = {"profileX", METAL_SHADER_DIR "/profileX.metal", 3, 2, 64, 1, 1, nil, 0, 0};
static MetalKernel k_profile_y = {"profileY", METAL_SHADER_DIR "/profileY.metal", 3, 2, 64, 1, 1, nil, 0, 0};
static MetalKernel k_mean_height = {"meanHeight", METAL_SHADER_DIR "/meanHeight.metal", 3, 2, 64, 1, 1, nil, 0, 0};
static MetalKernel k_nms = {"nonMaxSuppression", METAL_SHADER_DIR "/nonMaxSuppression.metal", 4, 3, 1, 1, 1, nil, 0, 0};

static MetalKernel* all_kernels[] = {
    &k_mul, &k_sub, &k_div, &k_split, &k_conv1d, &k_sdpa, &k_cross_sdpa,
    &k_cross_attention, &k_quantize, &k_dequantize, &k_spatial_softargmax_y,
    &k_profile_x, &k_profile_y, &k_mean_height, &k_nms
};

static int metal_ready(void) {
    return device != nil && commandQueue != nil;
}

static void clear_slot(MetalTensorSlot* s) {
    if (!s) return;
    VX_METAL_RELEASE(s->buffer);
    s->host = NULL;
    s->bytes = 0;
    s->cap = 0;
    s->host_dirty = 0;
    s->device_dirty = 0;
    s->is_weight = 0;
}

static NSString* patched_msl_source(NSString* source) {
    NSMutableString* out = [source mutableCopy];
    NSUInteger idx = 0;
    while (1) {
        NSRange range = [out rangeOfString:@"[[user(fake"];
        if (range.location == NSNotFound) break;
        NSRange tail = NSMakeRange(range.location, [out length] - range.location);
        NSRange end = [out rangeOfString:@")]]" options:0 range:tail];
        if (end.location == NSNotFound) break;
        range.length = end.location + end.length - range.location;
        NSString* repl = [NSString stringWithFormat:@"[[buffer(%lu)]]", (unsigned long)idx++];
        [out replaceCharactersInRange:range withString:repl];
    }
    return out;
}

static int compile_kernel(MetalKernel* k) {
    if (!metal_ready() || !k) return 0;
    if (k->ready) return k->pipeline != nil;
    if (k->failed) return 0;
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:k->path];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            fprintf(stderr, "[Metal] failed to read generated shader %s: %s\n",
                    k->path, error ? [[error localizedDescription] UTF8String] : "unknown error");
            k->failed = 1;
            return 0;
        }

        NSString* patched = patched_msl_source(source);
        id<MTLLibrary> library = [device newLibraryWithSource:patched options:nil error:&error];
        if (!library) {
            fprintf(stderr, "[Metal] shader compile failed (%s): %s\n",
                    k->name, error ? [[error localizedDescription] UTF8String] : "unknown error");
            VX_METAL_RELEASE(patched);
            k->failed = 1;
            return 0;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"main_"];
        if (!fn) {
            fprintf(stderr, "[Metal] shader entry main_ not found (%s)\n", k->name);
            VX_METAL_RELEASE(library);
            VX_METAL_RELEASE(patched);
            k->failed = 1;
            return 0;
        }

        k->pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
        VX_METAL_RELEASE(fn);
        VX_METAL_RELEASE(library);
        VX_METAL_RELEASE(patched);
        if (!k->pipeline) {
            fprintf(stderr, "[Metal] pipeline creation failed (%s): %s\n",
                    k->name, error ? [[error localizedDescription] UTF8String] : "unknown error");
            k->failed = 1;
            return 0;
        }
        k->ready = 1;
        return 1;
    }
}

static id<MTLBuffer> create_buffer(size_t bytes, const void* data) {
    if (!metal_ready() || bytes == 0) return nil;
    id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) return nil;
    if (data) memcpy([b contents], data, bytes);
    return b;
}

static int graph_find_slot(const void* host) {
    if (!host) return -1;
    for (int i = 0; i < graph_slot_count; i++) {
        if (graph_slots[i].host == host) return i;
    }
    return -1;
}

static MetalTensorSlot* graph_get_slot(const void* host, size_t bytes, int is_weight) {
    if (!metal_ready() || !host || bytes == 0) return NULL;
    int idx = graph_find_slot(host);
    if (idx < 0) {
        if (graph_slot_count >= METAL_GRAPH_MAX_TENSORS) return NULL;
        idx = graph_slot_count++;
        MetalTensorSlot* s = &graph_slots[idx];
        s->host = host;
        s->bytes = 0;
        s->cap = 0;
        s->buffer = nil;
        s->host_dirty = 1;
        s->device_dirty = 0;
        s->is_weight = is_weight;
    }
    MetalTensorSlot* s = &graph_slots[idx];
    s->bytes = bytes;
    if (is_weight) s->is_weight = 1;
    return s;
}

static MetalTensorSlot* graph_ensure_device(const void* host, size_t bytes, int is_weight) {
    MetalTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return NULL;
    if (!s->buffer || s->cap < bytes) {
        VX_METAL_RELEASE(s->buffer);
        s->buffer = create_buffer(bytes, host);
        if (!s->buffer) return NULL;
        s->cap = bytes;
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

static MetalTensorSlot* graph_output_slot(const void* host, size_t bytes) {
    MetalTensorSlot* s = graph_get_slot(host, bytes, 0);
    if (!s) return NULL;
    if (!s->buffer || s->cap < bytes) {
        VX_METAL_RELEASE(s->buffer);
        s->buffer = create_buffer(bytes, NULL);
        if (!s->buffer) return NULL;
        s->cap = bytes;
    }
    s->host_dirty = 0;
    s->device_dirty = 0;
    return s;
}

static void graph_mark_device(MetalTensorSlot* s) {
    if (!s) return;
    s->device_dirty = 1;
    s->host_dirty = 0;
}

static uint32_t binding_size_u32(size_t bytes) {
    return bytes > UINT32_MAX ? UINT32_MAX : (uint32_t)bytes;
}

static int dispatch_kernel(MetalKernel* k, const MetalBinding* binds,
                           uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!compile_kernel(k) || !binds || gx == 0 || gy == 0 || gz == 0) return 0;
    if (k->binding_count <= 0 || k->binding_count >= METAL_MAX_BINDINGS) return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return 0;
        [enc setComputePipelineState:k->pipeline];

        uint32_t sizes[METAL_MAX_BINDINGS] = {0};
        for (int i = 0; i < k->binding_count; i++) {
            if (!binds[i].buffer || binds[i].bytes == 0) {
                [enc endEncoding];
                return 0;
            }
            [enc setBuffer:binds[i].buffer offset:0 atIndex:(NSUInteger)i];
            sizes[i] = binding_size_u32(binds[i].bytes);
        }
        [enc setBytes:sizes
               length:(NSUInteger)(sizeof(uint32_t) * k->binding_count)
              atIndex:(NSUInteger)k->binding_count];

        MTLSize groups = MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);
        MTLSize threads = MTLSizeMake((NSUInteger)k->wg_x, (NSUInteger)k->wg_y, (NSUInteger)k->wg_z);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] == MTLCommandBufferStatusError) {
            NSError* error = [cb error];
            fprintf(stderr, "[Metal] dispatch failed (%s): %s\n",
                    k->name, error ? [[error localizedDescription] UTF8String] : "unknown error");
            return 0;
        }
        return 1;
    }
}

int metal_init(void) {
    @autoreleasepool {
        if (metal_ready()) return 0;
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("[VolvoxAI GPU] Metal is not supported on this device.\n");
            return -1;
        }
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            printf("[VolvoxAI GPU] Failed to create Metal command queue.\n");
            VX_METAL_RELEASE(device);
            return -1;
        }
        const char* name = [[device name] UTF8String];
        printf("[VolvoxAI GPU] Metal initialized successfully on: %s\n", name ? name : "unknown");
    }
    return 0;
}

void metal_graph_reset(void) {
    for (int i = 0; i < graph_slot_count; i++) clear_slot(&graph_slots[i]);
    graph_slot_count = 0;
}

void metal_graph_begin_forward(void) {
}

int metal_graph_end_forward(void) {
    return 0;
}

void metal_graph_mark_host(const void* host, size_t bytes, int is_weight) {
    MetalTensorSlot* s = graph_get_slot(host, bytes, is_weight);
    if (!s) return;
    s->host_dirty = 1;
    s->device_dirty = 0;
}

int metal_graph_sync_host(const void* host, size_t bytes, int is_weight) {
    (void)is_weight;
    if (!metal_ready() || !host || bytes == 0) return 0;
    int idx = graph_find_slot(host);
    if (idx < 0) return 0;
    MetalTensorSlot* s = &graph_slots[idx];
    if (!s->buffer || bytes > s->bytes) return 0;
    if (s->device_dirty) {
        memcpy((void*)host, [s->buffer contents], bytes);
        s->device_dirty = 0;
        s->host_dirty = 0;
    }
    return 1;
}

void metal_cleanup(void) {
    @autoreleasepool {
        metal_graph_reset();
        for (size_t i = 0; i < sizeof(all_kernels) / sizeof(all_kernels[0]); i++) {
            VX_METAL_RELEASE(all_kernels[i]->pipeline);
            all_kernels[i]->ready = 0;
            all_kernels[i]->failed = 0;
        }
        VX_METAL_RELEASE(commandQueue);
        VX_METAL_RELEASE(device);
    }
}

int metal_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
                           float* out, long out_numel, int op) {
    if (!a || !b || !out || a_numel <= 0 || b_numel <= 0 || out_numel <= 0) return 0;
    if (a_numel > out_numel || b_numel > out_numel) return 0;
    MetalKernel* kernel = op == 1 ? &k_sub : (op == 2 ? &k_div : &k_mul);
    size_t a_bytes = (size_t)a_numel * sizeof(float);
    size_t b_bytes = (size_t)b_numel * sizeof(float);
    size_t out_bytes = (size_t)out_numel * sizeof(float);
    MetalTensorSlot* sa = graph_ensure_device(a, a_bytes, 0);
    MetalTensorSlot* sb = graph_ensure_device(b, b_bytes, 0);
    MetalTensorSlot* so = graph_output_slot(out, out_bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t params[5] = {(uint32_t)out_numel, b_numel == 1 ? 1u : 0u, (uint32_t)b_numel,
                          (uint32_t)a_numel, a_numel == 1 ? 1u : 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[4] = {
        {sa->buffer, a_bytes}, {sb->buffer, b_bytes}, {so->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(kernel, binds, ((uint32_t)out_numel + 63u) / 64u, 1, 1);
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
                           int in_c, int in_l, int out_c, int out_l, int kernel,
                           int stride, int pad, int relu) {
    if (!in || !weight || !out || in_c <= 0 || in_l <= 0 || out_c <= 0 || out_l <= 0 ||
        kernel <= 0 || stride <= 0 || pad < 0 || relu < 0 || relu > 1) return 0;
    size_t in_bytes = (size_t)in_c * in_l * sizeof(float);
    size_t wbytes = (size_t)out_c * in_c * kernel * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)out_c * out_l * sizeof(float);
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
    uint32_t params[8] = {(uint32_t)in_c, (uint32_t)in_l, (uint32_t)out_c, (uint32_t)kernel,
                          (uint32_t)stride, (uint32_t)pad, (uint32_t)relu, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) {
        if (delete_bias) VX_METAL_RELEASE(bias_buf);
        return 0;
    }
    MetalBinding binds[5] = {
        {src->buffer, in_bytes}, {sw->buffer, wbytes}, {bias_buf, bbytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_conv1d, binds, ((uint32_t)out_l + 63u) / 64u, (uint32_t)out_c, 1);
    VX_METAL_RELEASE(pb);
    if (delete_bias) VX_METAL_RELEASE(bias_buf);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_sdpa_f32(const float* qkv, float* out, int seq_len, int d_model,
                         int num_heads, int head_dim, float scale) {
    if (!qkv || !out || seq_len <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        head_dim > 64 || d_model != num_heads * head_dim) return 0;
    size_t qkv_bytes = (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)seq_len * (size_t)d_model * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct { uint32_t seq_len, d_model, num_heads, head_dim; float scale; uint32_t pad[3]; } params =
        {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads, (uint32_t)head_dim, scale, {0, 0, 0}};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, qkv_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(&k_sdpa, binds, ((uint32_t)seq_len + 63u) / 64u, (uint32_t)num_heads, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_cross_sdpa_f32(const float* q, const float* k, const float* v, float* out,
                               int seq_q, int seq_kv, int d_model, int num_heads,
                               int head_dim, float scale) {
    if (!q || !k || !v || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim) return 0;
    size_t q_bytes = (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    MetalTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    MetalTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    MetalTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !dst) return 0;
    struct { uint32_t seq_q, seq_kv, d_model, num_heads, head_dim; float scale; uint32_t pad[2]; } params =
        {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads, (uint32_t)head_dim, scale, {0, 0}};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[5] = {
        {sq->buffer, q_bytes}, {sk->buffer, kv_bytes}, {sv->buffer, kv_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_cross_sdpa, binds, ((uint32_t)seq_q + 63u) / 64u, (uint32_t)num_heads, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                    const float* scale, const float* bias, float* out,
                                    int seq_q, int seq_kv, int d_model, int num_heads,
                                    int head_dim, int has_scale, int has_bias) {
    if (!q || !kv || !weight || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        d_model > 64) return 0;
    size_t q_bytes = (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)seq_kv * (size_t)d_model * sizeof(float);
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
        uint32_t has_scale, has_bias;
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim),
                (uint32_t)(has_scale && scale), (uint32_t)(has_bias && bias)};
    id<MTLBuffer> pb = create_buffer(sizeof(params), &params);
    if (!pb) return 0;
    MetalBinding binds[7] = {
        {sq->buffer, q_bytes}, {skv->buffer, kv_bytes}, {sw->buffer, wbytes},
        {ss->buffer, ss->bytes}, {sb->buffer, sb->bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}
    };
    int ok = dispatch_kernel(&k_cross_attention, binds, ((uint32_t)seq_q + 63u) / 64u, (uint32_t)num_heads, 1);
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

static int metal_graph_profile_common(MetalKernel* kernel, const float* in, float* out,
                                      int h, int w, int c, long out_elems,
                                      uint32_t gx, uint32_t gy) {
    if (!kernel || !in || !out || h <= 0 || w <= 0 || c <= 0 || out_elems <= 0) return 0;
    size_t in_bytes = (size_t)h * (size_t)w * (size_t)c * sizeof(float);
    size_t out_bytes = (size_t)out_elems * sizeof(float);
    MetalTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    MetalTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)h, (uint32_t)w, (uint32_t)c, 0u};
    id<MTLBuffer> pb = create_buffer(sizeof(params), params);
    if (!pb) return 0;
    MetalBinding binds[3] = {{src->buffer, in_bytes}, {dst->buffer, out_bytes}, {pb, sizeof(params)}};
    int ok = dispatch_kernel(kernel, binds, gx, gy, 1);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int metal_graph_spatial_softargmax_y_f32(const float* in, float* out, int h, int w, int c) {
    return metal_graph_profile_common(&k_spatial_softargmax_y, in, out, h, w, c,
                                      (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int metal_graph_profile_x_f32(const float* in, float* out, int h, int w, int c) {
    return metal_graph_profile_common(&k_profile_x, in, out, h, w, c,
                                      (long)2 * c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}

int metal_graph_profile_y_f32(const float* in, float* out, int h, int w, int c) {
    return metal_graph_profile_common(&k_profile_y, in, out, h, w, c,
                                      (long)2 * c * h, ((uint32_t)h + 63u) / 64u, (uint32_t)c);
}

int metal_graph_mean_height_f32(const float* in, float* out, int h, int w, int c) {
    return metal_graph_profile_common(&k_mean_height, in, out, h, w, c,
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
