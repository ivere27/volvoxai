#include "engine_internal.h"
#include "cJSON.h"
#include "fusion_ops.h"
#include "quant_cpu_opt.h"
#include "vulkan_engine.h"
#include "opengl_engine.h"
#ifdef __APPLE__
#include "metal_engine.h"
#endif
#include "conv_f32_opt.h"
#include "tensor_f32_opt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

extern int g_use_vulkan;
extern int g_use_nnapi;
extern int g_use_opengl;
extern int g_use_metal;
extern int g_debug;

#ifdef USE_NNAPI
extern void nnapi_matmul(const float*, const float*, const float*, float*, int, int, int);
#endif

extern int g_last_token;
extern void vk_matmul(const float*, const float*, const float*, float*, int, int, int);
extern void vk_free_weight_cache(void);
extern void matmul_f32(const float*, const float*, const float*, float*, int, int, int);
extern void add_f32(const float*, const float*, float*, int);
extern void add_broadcast_f32(const float*, const float*, float*, int, int, int);
extern void mul_f32(const float*, const float*, float*, int);
extern void mul_broadcast_f32(const float*, const float*, float*, int, int, int);
extern void batch_norm2d_f32(const float*, const float*, const float*, const float*, const float*, float*,
                             int, int, int, int, float);
extern void relu_f32(const float*, float*, int);
extern void sigmoid_f32(const float*, float*, int);
extern void gelu_f32(const float*, float*, int);
extern void silu_f32(const float*, float*, int);
extern void hardswish_f32(const float*, float*, int);
extern void hardsigmoid_f32(const float*, float*, int);
extern void tanh_f32(const float*, float*, int);
extern void clip_f32(const float*, float*, int, float, float);
extern void maxpool2d_f32(const float*, float*, int, int, int, int, int, int, int, int, int, int, int);
extern void global_average_pool_f32(const float*, float*, int, int, int, int);
extern void copy_f32(const float*, float*, int);
extern void layernorm_f32(const float*, const float*, const float*, float*, int, int, float);
extern void rmsnorm_f32(const float*, const float*, float*, int, int, float);
extern void softmax_f32(const float*, float*, int, int);
extern void sdpa_f32(const float*, float*, int, int, int, int, float);
extern void cross_sdpa_f32(const float*, const float*, const float*, float*, int, int, int, int, int, float);
extern void embedding_f32(uintptr_t, uintptr_t, uintptr_t, int, int);

double engine_now_ms(void) {
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

static T* nin(Node* n, const char* key);
static inline float relu6_apply(float x, int relu);
static int quantized_value_i32(const T* t, long idx);
static int quantized_weight_i32(const T* wt, long idx);
static int widen_f16_tensor_to_f32(T* t);
static int p_int(cJSON* p, const char* k, int def);
static float p_flt(cJSON* p, const char* k, float def);
static int layout_is(const char* layout, const char* want);

T* t_find(const char* name) {
    for (int i = 0; i < g_nt; i++) if (strcmp(g_t[i].name, name) == 0) return &g_t[i];
    return NULL;
}
static int t_index(const T* t) {
    if (!t) return -1;
    ptrdiff_t idx = t - g_t;
    return (idx >= 0 && idx < g_nt) ? (int)idx : -1;
}
static long numel_of(const int* shape, int ndim) { long n = 1; for (int i = 0; i < ndim; i++) n *= shape[i]; return n; }

static inline signed char clamp_i8_i32(int v) {
    if (v < -128) return -128;
    if (v > 127) return 127;
    return (signed char)v;
}

static inline signed char quantize_scalar_i8(float x, float scale, int zp) {
    return clamp_i8_i32((int)lrintf(x / scale + (float)zp));
}

static int qt_alloc_for(T* t, float scale, int zp) {
    int ti = t_index(t);
    if (ti < 0) return -1;
    QTensor* q = &g_qt[ti];
    if (q->cap < t->numel) {
        signed char* p = (signed char*)realloc(q->data, (size_t)t->numel);
        if (!p) return -1;
        q->data = p;
        q->cap = t->numel;
    }
    q->scale = scale;
    q->zp = zp;
    q->has_params = 1;
    return 0;
}

static QTensor* qt_get(T* t) {
    int ti = t_index(t);
    return ti >= 0 ? &g_qt[ti] : NULL;
}

static int qt_mark_params(T* t, float scale, int zp) {
    if (!t || scale <= 0.0f) return 0;
    QTensor* q = qt_get(t);
    if (!q) return -1;
    if (q->has_params) return 0;
    q->scale = scale;
    q->zp = zp;
    q->has_params = 1;
    return 0;
}

void qt_invalidate_all(void) {
    for (int i = 0; i < g_nt; i++) g_qt[i].valid = 0;
}

static int qt_quantize_from_f32(T* t, float scale, int zp) {
    if (!t || t->dtype != T_F32 || scale <= 0.0f) return 0;
    if (qt_alloc_for(t, scale, zp) != 0) return -1;
    QTensor* q = qt_get(t);
    for (long i = 0; i < t->numel; i++) q->data[i] = quantize_scalar_i8(t->data[i], scale, zp);
    q->valid = 1;
    return 1;
}

static QTensor* qt_ensure_quantized(T* t) {
    QTensor* q = qt_get(t);
    if (!q || !q->has_params) return NULL;
    if (q->valid) return q;
    if (qt_quantize_from_f32(t, q->scale, q->zp) < 0) return NULL;
    return qt_get(t);
}

static int qt_materialize_f32(T* t) {
    QTensor* q = qt_get(t);
    if (!t || !q || !q->valid || !q->has_params || t->dtype != T_F32) return 0;
    float scale = q->scale;
    int zp = q->zp;
    for (long i = 0; i < t->numel; i++) t->data[i] = ((int)q->data[i] - zp) * scale;
    return 1;
}

void materialize_tensor_f32(T* t) {
    qt_materialize_f32(t);
    widen_f16_tensor_to_f32(t);
}

static int q_add_quantized(Node* n, T* out) {
    QTensor* qo = qt_get(out);
    if (!qo || !qo->has_params) return 0;
    T* a = nin(n, "a");
    T* b = nin(n, "b");
    if (!a || !b || a->numel != b->numel || a->numel != out->numel) return 0;
    QTensor* qa = qt_ensure_quantized(a);
    QTensor* qb = qt_ensure_quantized(b);
    if (!qa || !qb) return 0;
    if (qt_alloc_for(out, qo->scale, qo->zp) != 0) return -1;
    qo = qt_get(out);
    vx_qadd_i8(qa->data, qb->data, qo->data, out->numel,
                    qa->scale, qa->zp, qb->scale, qb->zp, qo->scale, qo->zp);
    qo->valid = 1;
    return 1;
}

static int q_resize_nearest2d_quantized(T* in, T* out) {
    QTensor* qi = qt_ensure_quantized(in);
    QTensor* qo = qt_get(out);
    if (!qi || !qo || !qo->has_params || in->ndim != 4 || out->ndim != 4 ||
        in->shape[0] != 1 || out->shape[0] != 1) return 0;
    if (in->shape[3] != out->shape[3]) return 0;
    if (qt_alloc_for(out, qo->scale, qo->zp) != 0) return -1;
    qo = qt_get(out);
    int c = in->shape[3];
    int h = in->shape[1];
    int w = in->shape[2];
    int oh = out->shape[1];
    int ow = out->shape[2];
    int same_q = fabsf(qi->scale - qo->scale) <= 1e-8f && qi->zp == qo->zp;
    for (int y = 0; y < oh; y++) {
        int sy = (int)((long)y * h / oh);
        for (int x = 0; x < ow; x++) {
            int sx = (int)((long)x * w / ow);
            const signed char* src = qi->data + ((long)sy * w + sx) * c;
            signed char* dst = qo->data + ((long)y * ow + x) * c;
            if (same_q) {
                memcpy(dst, src, (size_t)c);
            } else {
                for (int ch = 0; ch < c; ch++) {
                    float v = ((int)src[ch] - qi->zp) * qi->scale;
                    dst[ch] = quantize_scalar_i8(v, qo->scale, qo->zp);
                }
            }
        }
    }
    qo->valid = 1;
    return 1;
}

static int q_maxpool2d_quantized(T* in, T* out, int ky, int kx, int sy, int sx, int py, int px) {
    QTensor* qi = qt_ensure_quantized(in);
    QTensor* qo = qt_get(out);
    if (!qi || !qo || !qo->has_params || in->ndim != 4 || out->ndim != 4 ||
        in->shape[0] != 1 || out->shape[0] != 1) return 0;
    if (in->shape[3] != out->shape[3]) return 0;
    if (qt_alloc_for(out, qo->scale, qo->zp) != 0) return -1;
    qo = qt_get(out);
    int c = in->shape[3];
    int h = in->shape[1];
    int w = in->shape[2];
    int oh = out->shape[1];
    int ow = out->shape[2];
    int same_q = fabsf(qi->scale - qo->scale) <= 1e-8f && qi->zp == qo->zp;
    if (same_q && vx_qmaxpool_i8_sameq(qi->data, qo->data, c, h, w, oh, ow,
                                            ky, kx, sy, sx, py, px, qi->zp)) {
        qo->valid = 1;
        return 1;
    }
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            for (int ch = 0; ch < c; ch++) {
                int best = -129;
                for (int yy = 0; yy < ky; yy++) {
                    int iy = oy * sy + yy - py;
                    if ((unsigned)iy >= (unsigned)h) continue;
                    for (int xx = 0; xx < kx; xx++) {
                        int ix = ox * sx + xx - px;
                        if ((unsigned)ix >= (unsigned)w) continue;
                        int qv = qi->data[((long)iy * w + ix) * c + ch];
                        if (qv > best) best = qv;
                    }
                }
                if (best < -128) best = qi->zp;
                if (same_q) qo->data[((long)oy * ow + ox) * c + ch] = (signed char)best;
                else {
                    float v = (best - qi->zp) * qi->scale;
                    qo->data[((long)oy * ow + ox) * c + ch] = quantize_scalar_i8(v, qo->scale, qo->zp);
                }
            }
        }
    }
    qo->valid = 1;
    return 1;
}

static int q_quantize_linear(Node* n, T* in, T* out) {
    if (!n || !in || !out || in->dtype != T_F32 || out->dtype != T_F32) return 0;
    float input_scale = p_flt(n->params, "input_scale", 0.0f);
    int input_zp = p_int(n->params, "input_zero_point", 0);
    float output_scale = p_flt(n->params, "output_scale", 0.0f);
    int output_zp = p_int(n->params, "output_zero_point", 0);
    if (output_scale <= 0.0f || in->numel != out->numel) return 0;
    if (qt_alloc_for(out, output_scale, output_zp) != 0) return -1;
    QTensor* qo = qt_get(out);
    if (!qo) return -1;
    vx_quantize_f32_to_i8(in->data, qo->data, in->numel, input_scale, input_zp, output_scale, output_zp);
    qo->valid = 1;
    return 1;
}

static size_t dtype_size(int dtype) {
    switch (dtype) {
        case T_I8: return 1;
        case T_U8: return 1;
        case T_F16: return 2;
        case T_I32: return 4;
        case T_F32:
        default: return 4;
    }
}

static int dtype_from_safetensors(const char* dtype) {
    if (!dtype) return T_F32;
    if (!strcmp(dtype, "F16")) return T_F16;
    if (!strcmp(dtype, "I8")) return T_I8;
    if (!strcmp(dtype, "U8")) return T_U8;
    if (!strcmp(dtype, "I32")) return T_I32;
    return T_F32;
}

static float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t f16_load_bits(const void* data, long idx) {
    const unsigned char* p = (const unsigned char*)data + idx * 2;
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float tensor_scalar_f32(const T* t, long idx) {
    if (!t) return 0.0f;
    if (t->dtype == T_F16) return f16_to_f32_scalar(f16_load_bits(t->data, idx));
    if (t->dtype == T_F32) return t->data[idx];
    return 0.0f;
}

static float tensor_scalar_any(const T* t, long idx) {
    if (!t || idx < 0 || idx >= t->numel) return 0.0f;
    if (t->dtype == T_F32 || t->dtype == T_F16) return tensor_scalar_f32(t, idx);
    if (t->dtype == T_I8) return (float)((const int8_t*)t->data)[idx];
    if (t->dtype == T_U8) return (float)((const uint8_t*)t->data)[idx];
    if (t->dtype == T_I32) return (float)((const int32_t*)t->data)[idx];
    return 0.0f;
}

static int widen_f16_tensor_to_f32(T* t) {
    if (!t || t->dtype != T_F16) return 0;
    float* dst = (float*)malloc((size_t)t->numel * sizeof(float));
    if (!dst) return -1;
    for (long i = 0; i < t->numel; i++) dst[i] = f16_to_f32_scalar(f16_load_bits(t->data, i));
    if (t->owns) free(t->data);
    t->data = dst;
    t->dtype = T_F32;
    t->elem_size = sizeof(float);
    t->owns = 1;
    return 1;
}

static T* t_add_ex(const char* name, const int* shape, int ndim, void* data, int dtype) {
    T* t = &g_t[g_nt++];
    strncpy(t->name, name, 127); t->name[127] = 0;
    t->ndim = ndim; for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    t->numel = numel_of(shape, ndim);
    t->dtype = dtype;
    t->elem_size = dtype_size(dtype);
    t->owns = data ? 0 : 1;
    t->data = data ? (float*)data : (float*)calloc(t->numel > 0 ? t->numel : 1, t->elem_size);
    return t;
}

static T* t_add(const char* name, const int* shape, int ndim, float* data) {
    return t_add_ex(name, shape, ndim, data, T_F32);
}

// Row window [r0, r0+rc) over a tensor whose leading (batch-folded) dim is the sequence.
static void seq_range(long total_rows, int* r0, int* rc) {
    if (g_dec_pos >= 0) { *r0 = g_dec_pos; *rc = 1; }
    else { *r0 = 0; *rc = (g_prefill_len > 0 && g_prefill_len < total_rows) ? g_prefill_len : (int)total_rows; }
}

// Incremental attention: one query position attends over cached K/V [0..len).
static void sdpa_decode(const float* q, const float* kc, const float* vc, int len,
                        float* out, int d_model, int heads, int head_dim, float scale) {
    static float scores[MAX_SEQ];
    for (int h = 0; h < heads; h++) {
        const float* qh = q + h * head_dim;
        float maxv = -1e38f;
        for (int j = 0; j < len; j++) {
            const float* kh = kc + (long)j * d_model + h * head_dim;
            float s = 0; for (int d = 0; d < head_dim; d++) s += qh[d] * kh[d];
            s *= scale; scores[j] = s; if (s > maxv) maxv = s;
        }
        float sum = 0; for (int j = 0; j < len; j++) { scores[j] = expf(scores[j] - maxv); sum += scores[j]; }
        for (int d = 0; d < head_dim; d++) {
            float acc = 0;
            for (int j = 0; j < len; j++) acc += scores[j] * vc[(long)j * d_model + h * head_dim + d];
            out[h * head_dim + d] = acc / sum;
        }
    }
}

char* read_file(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[engine] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    buf[sz] = 0; fclose(f); if (out_size) *out_size = sz; return buf;
}

// ---- safetensors: load weights; F16 tensors remain half-precision in the blob ----
int load_weights(const char* path) {
    double t0 = g_debug ? engine_now_ms() : 0.0;
    long sz; g_blob = read_file(path, &sz);
    if (!g_blob) return -1;
    uint64_t header_size; memcpy(&header_size, g_blob, 8);
    char* json = (char*)malloc(header_size + 1);
    memcpy(json, g_blob + 8, header_size); json[header_size] = 0;
    cJSON* root = cJSON_Parse(json); free(json);
    if (!root) { fprintf(stderr, "[engine] bad safetensors header\n"); return -1; }
    long data_base = 8 + (long)header_size;
    int weight_count = 0;
    int f16_loaded = 0;
    for (cJSON* it = root->child; it; it = it->next) {
        if (strcmp(it->string, "__metadata__") == 0) continue;
        cJSON* dtype_node = cJSON_GetObjectItem(it, "dtype");
        cJSON* shape = cJSON_GetObjectItem(it, "shape");
        cJSON* off = cJSON_GetObjectItem(it, "data_offsets");
        int ndim = cJSON_GetArraySize(shape);
        int sh[8]; for (int i = 0; i < ndim && i < 8; i++) sh[i] = cJSON_GetArrayItem(shape, i)->valueint;
        long start = (long)cJSON_GetArrayItem(off, 0)->valuedouble;
        int dtype = dtype_from_safetensors(dtype_node ? dtype_node->valuestring : "F32");
        if (dtype == T_F16) f16_loaded++;
        t_add_ex(it->string, sh, ndim, g_blob + data_base + start, dtype);
        weight_count++;
    }
    cJSON_Delete(root);
    if (g_debug) {
        fprintf(stderr, "[debug] load_weights tensors=%d f16_loaded=%d bytes=%ld %.3f ms\n",
                weight_count, f16_loaded, sz, engine_now_ms() - t0);
    }
    return 0;
}

// ---- param helpers --------------------------------------------------------
static void p_pair(cJSON* p, const char* k, int def, int* y, int* x) {
    cJSON* v = p ? cJSON_GetObjectItem(p, k) : NULL;
    if (!v) { *y = def; *x = def; return; }
    if (cJSON_IsArray(v)) { *y = cJSON_GetArrayItem(v, 0)->valueint; cJSON* x1 = cJSON_GetArrayItem(v, 1); *x = x1 ? x1->valueint : *y; }
    else { *y = v->valueint; *x = v->valueint; }
}
static int   p_int(cJSON* p, const char* k, int def)   { cJSON* v = p ? cJSON_GetObjectItem(p, k) : NULL; return v ? v->valueint : def; }
static float p_flt(cJSON* p, const char* k, float def) { cJSON* v = p ? cJSON_GetObjectItem(p, k) : NULL; return v ? (float)v->valuedouble : def; }
static const char* p_str(cJSON* p, const char* k, const char* def) {
    cJSON* v = p ? cJSON_GetObjectItem(p, k) : NULL;
    return (v && cJSON_IsString(v) && v->valuestring) ? v->valuestring : def;
}
static T* nin(Node* n, const char* key) {
    for (int i = 0; i < n->nin; i++) if (strcmp(n->ins[i].key, key) == 0) return t_find(n->ins[i].name);
    return NULL;
}

static void p_iarr(cJSON* p, const char* k, int* out, int maxn, int def) {
    for (int i = 0; i < maxn; i++) out[i] = def;
    cJSON* v = p ? cJSON_GetObjectItem(p, k) : NULL;
    if (!v) return;
    if (cJSON_IsArray(v)) {
        int n = cJSON_GetArraySize(v);
        if (n > maxn) n = maxn;
        for (int i = 0; i < n; i++) out[i] = cJSON_GetArrayItem(v, i)->valueint;
    } else {
        out[0] = v->valueint;
    }
}

static size_t tensor_bytes(const T* t) {
    return t ? (size_t)t->numel * t->elem_size : 0;
}

static void vk_mark_host_tensor(T* t) {
    if (g_use_vulkan && t && t->dtype == T_F32) {
        vk_graph_mark_host(t->data, tensor_bytes(t), !t->owns);
    }
    if (g_use_opengl && t && t->dtype == T_F32) {
        opengl_graph_mark_host(t->data, tensor_bytes(t), !t->owns);
    }
#ifdef __APPLE__
    if (g_use_metal && t && t->dtype == T_F32) {
        metal_graph_mark_host(t->data, tensor_bytes(t), !t->owns);
    }
#endif
}

void vk_sync_host_tensor(T* t) {
    if (g_use_vulkan && t && t->dtype == T_F32) {
        vk_graph_sync_host(t->data, tensor_bytes(t), !t->owns);
    }
    if (g_use_opengl && t && t->dtype == T_F32) {
        opengl_graph_sync_host(t->data, tensor_bytes(t), !t->owns);
    }
#ifdef __APPLE__
    if (g_use_metal && t && t->dtype == T_F32) {
        metal_graph_sync_host(t->data, tensor_bytes(t), !t->owns);
    }
#endif
}

static void vk_sync_node_inputs(Node* n) {
    if ((!g_use_vulkan && !g_use_opengl && !g_use_metal) || !n) return;
    for (int i = 0; i < n->nin; i++) {
        T* t = t_find(n->ins[i].name);
        vk_sync_host_tensor(t);
    }
}

static void materialize_node_inputs(Node* n) {
    if (!n) return;
    for (int i = 0; i < n->nin; i++) {
        T* t = t_find(n->ins[i].name);
        materialize_tensor_f32(t);
    }
}

void vk_mark_owned_tensors_host_dirty(void) {
    if (!g_use_vulkan && !g_use_opengl && !g_use_metal) return;
    if (g_use_vulkan) vk_graph_begin_forward();
    if (g_use_opengl) opengl_graph_begin_forward();
#ifdef __APPLE__
    if (g_use_metal) metal_graph_begin_forward();
#endif
    for (int i = 0; i < g_nt; i++) {
        if (g_t[i].owns && g_t[i].dtype == T_F32) {
            if (g_use_vulkan) vk_graph_mark_host(g_t[i].data, tensor_bytes(&g_t[i]), 0);
            if (g_use_opengl) opengl_graph_mark_host(g_t[i].data, tensor_bytes(&g_t[i]), 0);
#ifdef __APPLE__
            if (g_use_metal) metal_graph_mark_host(g_t[i].data, tensor_bytes(&g_t[i]), 0);
#endif
        }
    }
}

// ---- build graph from config.json -----------------------------------------
// Fuse ReLU6 (Clip min=0,max=6) into the conv that produces its input. When the conv's
// output feeds only the Clip, we redirect the conv to write the Clip's output tensor with a
// clamped epilogue and skip the Clip node — removing a full read+write pass over the tensor.
static const char* node_input_name(Node* n) {
    for (int i = 0; i < n->nin; i++) if (!strcmp(n->ins[i].key, "input")) return n->ins[i].name;
    return n->nin > 0 ? n->ins[0].name : NULL;
}
static int tensor_use_count(const char* name) {
    int uses = 0;
    for (int i = 0; i < g_nn; i++) for (int j = 0; j < g_n[i].nin; j++)
        if (!strcmp(g_n[i].ins[j].name, name)) uses++;
    return uses;
}
static int tensor_same_shape(const T* a, const T* b) {
    if (!a || !b || a->ndim != b->ndim || a->numel != b->numel) return 0;
    for (int i = 0; i < a->ndim; i++) if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}
static int producer_index_for(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_nn; i++) if (!g_n[i].skip && !strcmp(g_n[i].out, name)) return i;
    return -1;
}
static int producer_index_any(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_nn; i++) if (!strcmp(g_n[i].out, name)) return i;
    return -1;
}
#define VOLVOX_ENGINE_PRIVATE_INCLUDE 1
#include "graph_opt_fusion.c"
#undef VOLVOX_ENGINE_PRIVATE_INCLUDE
static int build_quant_params_pass(void) {
    if (g_use_vulkan || g_use_opengl || g_use_metal) return 0;
    for (int i = 0; i < g_nn; i++) {
        Node* n = &g_n[i];
        if (strcmp(n->op, "QConv2D")) continue;
        T* in = nin(n, "input");
        float scale = p_flt(n->params, "input_scale", 0.0f);
        int zp = p_int(n->params, "input_zero_point", 0);
        if (qt_mark_params(in, scale, zp) != 0) return -1;
    }
    for (int iter = 0; iter < 4; iter++) {
        for (int i = 0; i < g_nn; i++) {
            Node* n = &g_n[i];
            T* out = t_find(n->out);
            QTensor* qo = qt_get(out);
            if (!qo || !qo->has_params) continue;
            if (!strcmp(n->op, "Add")) {
                T* a = nin(n, "a");
                T* b = nin(n, "b");
                if (qt_mark_params(a, qo->scale, qo->zp) != 0) return -1;
                if (qt_mark_params(b, qo->scale, qo->zp) != 0) return -1;
            } else if (!strcmp(n->op, "ResizeNearest2D") || !strcmp(n->op, "UpsampleNearest2D") ||
                       !strcmp(n->op, "MaxPool2D") || !strcmp(n->op, "Reshape") ||
                       !strcmp(n->op, "Flatten") || !strcmp(n->op, "Squeeze") ||
                       !strcmp(n->op, "Unsqueeze") || !strcmp(n->op, "Dropout") ||
                       !strcmp(n->op, "Identity")) {
                T* in = nin(n, "input");
                if (qt_mark_params(in, qo->scale, qo->zp) != 0) return -1;
            }
        }
    }
    return 0;
}

// ---- Activation arena: reuse physical buffers across non-overlapping lifetimes ----
// Similar to planned tensor arenas in optimized inference runtimes. Only transient F32
// activations are pooled: owned (calloc'd by us -> NOT weights, which point into the blob),
// with a non-skipped producer node (excludes graph inputs/constants) and at least one
// consumer (excludes graph outputs and dead tensors), and not aliased by another tensor
// (excludes Reshape/Flatten passthroughs).
// last-use scans ALL nodes INCLUDING skipped ones, so conv+add residuals and other
// fused-away consumers keep their buffer live conservatively. Cuts peak activation memory
// (~88 -> ~10 MB here) which removes most cold-start first-touch page faults.
static float** g_arena_bufs = NULL;
static int g_arena_nbufs = 0;

void engine_free_arena(void) {
    for (int i = 0; i < g_arena_nbufs; i++) free(g_arena_bufs[i]);
    free(g_arena_bufs);
    g_arena_bufs = NULL;
    g_arena_nbufs = 0;
}

static int t_index_by_name(const char* name) {
    T* t = t_find(name);
    return t ? (int)(t - g_t) : -1;
}

static void plan_memory_arena(void) {
    const char* en = getenv("VOLVOX_ARENA");
    if (en && en[0] && !strcmp(en, "0")) return;   // default-on; VOLVOX_ARENA=0 disables
    if (g_use_vulkan || g_use_opengl || g_use_metal || g_use_nnapi) return;
    int nt = g_nt, nn = g_nn;
    if (nt <= 0 || nn <= 0) return;

    int* prod = (int*)malloc(sizeof(int) * nt);
    int* last = (int*)malloc(sizeof(int) * nt);
    int* buf_of = (int*)malloc(sizeof(int) * nt);
    char* poolable = (char*)calloc(nt, 1);
    long* cap = (long*)malloc(sizeof(long) * nt);
    char* busy = (char*)calloc(nt, 1);
    float** bufs = (float**)malloc(sizeof(float*) * nt);
    if (!prod || !last || !buf_of || !poolable || !cap || !busy || !bufs) {
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }
    for (int i = 0; i < nt; i++) { prod[i] = -1; last[i] = -1; buf_of[i] = -1; }

    for (int j = 0; j < nn; j++) {
        if (g_n[j].skip) continue;
        int ti = t_index_by_name(g_n[j].out);
        if (ti >= 0) prod[ti] = j;
    }
    for (int j = 0; j < nn; j++) {
        for (int k = 0; k < g_n[j].nin; k++) {
            int ti = t_index_by_name(g_n[j].ins[k].name);
            if (ti >= 0 && j > last[ti]) last[ti] = j;
        }
    }
    for (int i = 0; i < nt; i++) {
        T* t = &g_t[i];
        if (!(t->owns && t->dtype == T_F32 && t->numel > 0)) continue;
        if (prod[i] < 0 || last[i] < 0) continue;          // graph input/const or output/dead
        int aliased = 0;
        for (int u = 0; u < nt; u++) if (u != i && g_t[u].data == t->data) { aliased = 1; break; }
        if (aliased) continue;
        poolable[i] = 1;
    }

    // Phase 1: assign each poolable tensor a buffer INDEX, growing only integer capacities.
    // (Buffers are allocated once after sizing so pointers never move — a tensor is written
    // every forward within its lifetime, so its data pointer must stay valid for good.)
    int nbufs = 0;
    long orig_bytes = 0;
    for (int i = 0; i < nn; i++) {
        for (int t = 0; t < nt; t++) {                     // tensors produced at node i
            if (!poolable[t] || prod[t] != i) continue;
            long need = g_t[t].numel;
            orig_bytes += need * (long)g_t[t].elem_size;
            int best = -1;                                  // best-fit among free buffers
            for (int b = 0; b < nbufs; b++)
                if (!busy[b] && cap[b] >= need && (best < 0 || cap[b] < cap[best])) best = b;
            if (best < 0) {                                 // none fit: grow largest free, else new
                for (int b = 0; b < nbufs; b++)
                    if (!busy[b] && (best < 0 || cap[b] > cap[best])) best = b;
                if (best < 0) { best = nbufs++; cap[best] = 0; }
                if (cap[best] < need) cap[best] = need;
            }
            busy[best] = 1;
            buf_of[t] = best;
        }
        for (int t = 0; t < nt; t++)                        // free tensors last-used at node i
            if (poolable[t] && last[t] == i && buf_of[t] >= 0) busy[buf_of[t]] = 0;
    }

    // Phase 2: allocate the arena buffers once at their final sizes.
    int ok = 1;
    for (int b = 0; b < nbufs; b++) {
        bufs[b] = (float*)malloc((size_t)(cap[b] > 0 ? cap[b] : 1) * sizeof(float));
        if (!bufs[b]) { ok = 0; break; }
    }
    if (!ok) {                                              // OOM: undo, keep original calloc buffers
        for (int b = 0; b < nbufs; b++) free(bufs[b]);
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }

    // Phase 3: point each pooled tensor at its buffer, dropping its own calloc buffer.
    for (int t = 0; t < nt; t++) {
        if (!poolable[t] || buf_of[t] < 0) continue;
        free(g_t[t].data);
        g_t[t].data = bufs[buf_of[t]];
        g_t[t].owns = 0;
    }

    long arena_bytes = 0;
    for (int b = 0; b < nbufs; b++) arena_bytes += cap[b] * (long)sizeof(float);
    g_arena_bufs = bufs;
    g_arena_nbufs = nbufs;
    if (g_debug)
        fprintf(stderr, "[debug] arena_plan pooled=%.1f MB -> arena=%.1f MB (%d buffers)\n",
                orig_bytes / 1048576.0, arena_bytes / 1048576.0, nbufs);

    free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy);
}

int build_graph(const char* config_path) {
    double t0 = g_debug ? engine_now_ms() : 0.0;
    graph_opt_reset_state();
    long sz; char* cfg = read_file(config_path, &sz);
    if (!cfg) return -1;
    g_cfg_root = cJSON_Parse(cfg); free(cfg);
    cJSON* root = g_cfg_root;
    if (!root) { fprintf(stderr, "[engine] bad config.json\n"); return -1; }

    // Graph inputs (allocated, zero-filled; the caller pokes them before forward()).
    cJSON* inputs = cJSON_GetObjectItem(root, "inputs");
    if (inputs) for (cJSON* it = inputs->child; it; it = it->next) {
        cJSON* shape = cJSON_GetObjectItem(it, "shape");
        int ndim = cJSON_GetArraySize(shape); int sh[8];
        for (int i = 0; i < ndim && i < 8; i++) sh[i] = cJSON_GetArrayItem(shape, i)->valueint;
        t_add(it->string, sh, ndim, NULL);
        if (!g_first_input[0]) { strncpy(g_first_input, it->string, 127); g_first_input[127] = 0; }
    }
    // A "positions" input defaults to 0..seq-1.
    T* pt = t_find("positions");
    if (pt) for (long j = 0; j < pt->numel; j++) pt->data[j] = (float)j;

    // Nodes.
    cJSON* nodes = cJSON_GetObjectItem(root, "nodes");
    for (cJSON* nd = nodes ? nodes->child : NULL; nd; nd = nd->next) {
        Node* n = &g_n[g_nn++];
        cJSON* op = cJSON_GetObjectItem(nd, "opType"); if (!op) op = cJSON_GetObjectItem(nd, "op");
        strncpy(n->op, op->valuestring, 39); n->op[39] = 0;
        n->params = cJSON_GetObjectItem(nd, "params");  // borrowed (g_cfg_root kept alive)
        n->nin = 0; n->nout = 0; n->fuse_relu6 = 0; n->skip = 0;
        cJSON* nins = cJSON_GetObjectItem(nd, "inputs");
        if (nins) for (cJSON* it = nins->child; it; it = it->next) {
            if (n->nin >= MAXIN) break;
            strncpy(n->ins[n->nin].key, it->string, 23);
            strncpy(n->ins[n->nin].name, it->valuestring, 127);
            n->nin++;
        }
        // Create output tensor(s) from outputs_shape; keep the first as the node output.
        cJSON* outs = cJSON_GetObjectItem(nd, "outputs");
        cJSON* oshape = cJSON_GetObjectItem(nd, "outputs_shape");
        n->out[0] = 0;
        if (outs) for (cJSON* it = outs->child; it; it = it->next) {
            const char* tname = it->valuestring;
            cJSON* sh = oshape ? cJSON_GetObjectItem(oshape, it->string) : NULL;
            int ndim = sh ? cJSON_GetArraySize(sh) : 0; int shp[8];
            for (int i = 0; i < ndim && i < 8; i++) shp[i] = cJSON_GetArrayItem(sh, i)->valueint;
            if (!t_find(tname)) t_add(tname, shp, ndim, NULL);
            if (n->nout < MAXIN) {
                strncpy(n->outs[n->nout].key, it->string, 23);
                n->outs[n->nout].key[23] = 0;
                strncpy(n->outs[n->nout].name, tname, 127);
                n->outs[n->nout].name[127] = 0;
                n->nout++;
            }
            if (!n->out[0]) { strncpy(n->out, tname, 127); n->out[127] = 0; }
        }
    }
    GraphOptStats opt = graph_optimize_operator_fusion();
    if (build_quant_params_pass() != 0) return -1;
    if (g_debug) {
        fprintf(stderr, "[debug] build_graph tensors=%d nodes=%d bytes=%ld %.3f ms\n",
                g_nt, g_nn, sz, engine_now_ms() - t0);
        fprintf(stderr, "[debug] graph_opt operator_fusion relu6=%d conv_add=%d depthwise_pointwise=%d chained_add=%d concat_sigmoid=%d alias=%d skipped=%d\n",
                opt.relu6, opt.conv_add, opt.depthwise_pointwise, opt.chained_add,
                opt.concat_sigmoid, opt.alias, opt.skipped);
    }
    plan_memory_arena();
    return 0;
}

// ---- Add/Mul via the shared kernels (row-windowed; per-channel/scalar full) ----
static void binop(Node* n, int is_mul) {
    T* a = nin(n, "a"); T* b = nin(n, "b"); T* o = t_find(n->out);
    if (!a) a = nin(n, "input");
    int D = o->shape[o->ndim - 1]; long total = o->numel / D;
    int r0, rc; seq_range(total, &r0, &rc);
    long off = (long)r0 * D; int cnt = rc * D;
    if (b->numel == a->numel) {
        if (is_mul) mul_f32(a->data + off, b->data + off, o->data + off, cnt);
        else        vx_fused_add_relu_f32(a->data + off, b->data + off, o->data + off, cnt, p_int(n->params, "relu", 0));
    } else if (a->ndim >= 2 && b->numel == a->shape[a->ndim - 1]) {
        int C = a->shape[a->ndim - 1], sp = (int)(o->numel / C);
        if (is_mul) mul_broadcast_f32(a->data, b->data, o->data, C, sp, 1);
        else        add_broadcast_f32(a->data, b->data, o->data, C, sp, 1);
    } else {                                                        // scalar broadcast
        if (is_mul) mul_broadcast_f32(a->data, b->data, o->data, 1, (int)o->numel, 1);
        else        add_broadcast_f32(a->data, b->data, o->data, 1, (int)o->numel, 1);
    }
    int relu = p_int(n->params, "relu", 0);
    if (!is_mul && relu && b->numel != a->numel) {
        for (long i = off; i < off + cnt; i++) {
            float x = o->data[i];
            if (x < 0.0f) x = 0.0f;
            if (relu >= 2 && x > 6.0f) x = 6.0f;
            o->data[i] = x;
        }
    }
}

static void transpose_tensor(const T* in, T* out, const int* perm) {
    long in_stride[8] = {0}, out_stride[8] = {0};
    int ndim = in->ndim;
    in_stride[ndim - 1] = 1;
    out_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        in_stride[i] = in_stride[i + 1] * in->shape[i + 1];
        out_stride[i] = out_stride[i + 1] * out->shape[i + 1];
    }
    for (long idx = 0; idx < out->numel; idx++) {
        long rem = idx, src = 0;
        for (int od = 0; od < ndim; od++) {
            int coord = (int)(rem / out_stride[od]);
            rem %= out_stride[od];
            src += (long)coord * in_stride[perm[od]];
        }
        out->data[idx] = in->data[src];
    }
}

static void concat_tensor(Node* n, T* out, int axis) {
    if (axis < 0) axis += out->ndim;
    long inner = 1;
    for (int i = axis + 1; i < out->ndim; i++) inner *= out->shape[i];
    long outer = out->numel / ((long)out->shape[axis] * inner);
    for (long oidx = 0; oidx < outer; oidx++) {
        long dst_axis = 0;
        for (int i = 0; i < n->nin; i++) {
            T* src = t_find(n->ins[i].name);
            if (!src) continue;
            int axis_len = src->shape[axis];
            long count = (long)axis_len * inner;
            memcpy(out->data + oidx * out->shape[axis] * inner + dst_axis * inner,
                   src->data + oidx * count, (size_t)count * sizeof(float));
            dst_axis += axis_len;
        }
    }
}

static void concat_tensor_sigmoid(Node* n, T* out, int axis) {
    if (axis < 0) axis += out->ndim;
    long inner = 1;
    for (int i = axis + 1; i < out->ndim; i++) inner *= out->shape[i];
    long outer = out->numel / ((long)out->shape[axis] * inner);
    const float* srcs[MAXIN];
    int axis_lens[MAXIN];
    int nsrc = 0;
    for (int i = 0; i < n->nin && nsrc < MAXIN; i++) {
        T* src = t_find(n->ins[i].name);
        if (!src) continue;
        srcs[nsrc] = src->data;
        axis_lens[nsrc] = src->shape[axis];
        nsrc++;
    }
    vx_fused_concat_sigmoid_f32(srcs, axis_lens, nsrc, out->data, out->shape[axis], outer, inner);
}

static inline float relu6_apply(float x, int relu);

static void resize_nearest2d_image(const T* in, T* out) {
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int oh = out->shape[1], ow = out->shape[2];
    for (int b = 0; b < n; b++) {
        const float* src_b = in->data + (long)b * h * w * c;
        float* dst_b = out->data + (long)b * oh * ow * c;
        for (int y = 0; y < oh; y++) {
            int iy = (int)floorf((float)y * (float)h / (float)oh);
            if (iy >= h) iy = h - 1;
            for (int x = 0; x < ow; x++) {
                int ix = (int)floorf((float)x * (float)w / (float)ow);
                if (ix >= w) ix = w - 1;
                memcpy(dst_b + ((long)y * ow + x) * c,
                       src_b + ((long)iy * w + ix) * c,
                       (size_t)c * sizeof(float));
            }
        }
    }
}

static void maxpool2d_f32_image(const T* in, T* out, int ky, int kx, int sy, int sx, int py, int px) {
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int oh = out->shape[1], ow = out->shape[2];
    vx_maxpool2d_f32(in->data, out->data, n, h, w, c, oh, ow, ky, kx, sy, sx, py, px);
}

static void conv2d_pointwise_f32_image(int node_idx, const T* in, T* out, const T* wt, const T* bias,
                                      const T* add, int relu) {
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int out_c = out->shape[3];
    long pixels = (long)n * h * w;
    vx_conv2d_pointwise_f32(node_idx, in->data, out->data, wt->data,
                                 bias ? bias->data : NULL,
                                 add ? add->data : NULL,
                                 pixels, c, out_c, relu);
}

static int conv2d_depthwise_pointwise_f32_image(int dw_node_idx, int pw_node_idx,
                                               const T* in, T* out,
                                               const T* dw_wt, const T* dw_bias,
                                               const T* pw_wt, const T* pw_bias,
                                               const T* add,
                                               int sy, int sx, const int* pads,
                                               int dw_relu, int pw_relu) {
    if (!in || !out || !dw_wt || !pw_wt || in->ndim != 4 || out->ndim != 4 ||
        dw_wt->ndim < 4 || pw_wt->ndim < 4) return 0;
    return vx_conv2d_depthwise_pointwise_f32(dw_node_idx, pw_node_idx,
                                                  in->data, out->data,
                                                  dw_wt->data, dw_bias ? dw_bias->data : NULL,
                                                  pw_wt->data, pw_bias ? pw_bias->data : NULL,
                                                  add ? add->data : NULL,
                                                  in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                                  out->shape[1], out->shape[2], out->shape[3],
                                                  dw_wt->shape[0], dw_wt->shape[1], dw_wt->shape[2], dw_wt->shape[3],
                                                  pw_wt->shape[0], pw_wt->shape[1], pw_wt->shape[2],
                                                  sy, sx, pads, dw_relu, pw_relu);
}

static void conv2d_f32_image(int node_idx, const T* in, T* out, const T* wt, const T* bias, const T* add,
                            int sy, int sx, const int* pads, int groups, int dy, int dx, int relu) {
    if (!in || !out || !wt || in->ndim != 4 || out->ndim != 4 || wt->ndim < 4) return;
    if (dy <= 0) dy = 1;
    if (dx <= 0) dx = 1;
    int kh = wt->shape[0], kw = wt->shape[1];
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int oh = out->shape[1], ow = out->shape[2], out_c = out->shape[3];
    if (groups == c && wt->shape[2] == c && out_c == c * wt->shape[3]) {
        vx_conv2d_depthwise_f32(in->data, out->data, wt->data, bias ? bias->data : NULL,
                                     n, h, w, c, oh, ow, out_c,
                                     kh, kw, wt->shape[2], wt->shape[3],
                                     sy, sx, pads, dy, dx, relu);
    } else if (groups == 1 && kh == 1 && kw == 1 && sy == 1 && sx == 1 &&
               pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
        conv2d_pointwise_f32_image(node_idx, in, out, wt, bias, add, relu);
    } else {
        vx_conv2d_generic_f32(node_idx, in->data, out->data, wt->data, bias ? bias->data : NULL,
                                  n, h, w, c, oh, ow, out_c, kh, kw, wt->shape[2],
                                  groups, sy, sx, pads, dy, dx, relu);
    }
}

static int quantized_value_i32(const T* t, long idx) {
    if (!t) return 0;
    if (t->dtype == T_I8) return ((const signed char*)t->data)[idx];
    if (t->dtype == T_U8) return ((const unsigned char*)t->data)[idx];
    if (t->dtype == T_I32) return ((const int*)t->data)[idx];
    return (int)lrintf(t->data[idx]);
}

static int quantized_weight_i32(const T* wt, long idx) {
    if (wt->dtype == T_U8) return ((const unsigned char*)wt->data)[idx];
    return ((const signed char*)wt->data)[idx];
}

static int16_t* qconv_i16_weight_cache(int node_idx, const T* wt, const T* wzp, const char* layout) {
    if (g_qiwcache[node_idx]) return g_qiwcache[node_idx];
    if ((wt->dtype != T_I8 && wt->dtype != T_U8) || wt->ndim != 4) return NULL;
    int16_t* out = (int16_t*)malloc((size_t)wt->numel * sizeof(int16_t));
    if (!out) return NULL;
    int ohwi = layout && !strcmp(layout, "OHWI");
    int hwo = layout && (!strcmp(layout, "1HWO") || !strcmp(layout, "1HWM"));
    if (ohwi) {
        int out_c = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], in_per_group = wt->shape[3];
        for (int oc = 0; oc < out_c; oc++) {
            int wz = wzp ? quantized_value_i32(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            for (int ci = 0; ci < in_per_group; ci++) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        long src = (((long)oc * kh + ky) * kw + kx) * in_per_group + ci;
                        long dst = (((long)oc * in_per_group + ci) * kh + ky) * kw + kx;
                        out[dst] = (int16_t)(quantized_weight_i32(wt, src) - wz);
                    }
                }
            }
        }
    } else if (hwo) {
        int one = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], out_c = wt->shape[3];
        if (one != 1) { free(out); return NULL; }
        for (int oc = 0; oc < out_c; oc++) {
            int wz = wzp ? quantized_value_i32(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    long src = (((long)0 * kh + ky) * kw + kx) * out_c + oc;
                    long dst = (((long)oc * 1 + 0) * kh + ky) * kw + kx;
                    out[dst] = (int16_t)(quantized_weight_i32(wt, src) - wz);
                }
            }
        }
    } else {
        int out_c = wt->shape[0];
        long per_oc = wt->numel / out_c;
        for (int oc = 0; oc < out_c; oc++) {
            int wz = wzp ? quantized_value_i32(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            long base = (long)oc * per_oc;
            for (long i = 0; i < per_oc; i++) {
                long idx = base + i;
                out[idx] = (int16_t)(quantized_weight_i32(wt, idx) - wz);
            }
        }
    }
    g_qiwcache[node_idx] = out;
    return out;
}

static int16_t* qconv_depthwise_tap_cache(int node_idx, const T* wt, const T* wzp, const char* layout) {
    if (g_qdwcache[node_idx]) return g_qdwcache[node_idx];
    if ((wt->dtype != T_I8 && wt->dtype != T_U8) || wt->ndim != 4) return NULL;
    int hwo = layout && (!strcmp(layout, "1HWO") || !strcmp(layout, "1HWM"));
    int c = hwo ? wt->shape[3] : wt->shape[0];
    int kh = hwo ? wt->shape[1] : wt->shape[2];
    int kw = hwo ? wt->shape[2] : wt->shape[3];
    if (!hwo && wt->shape[1] != 1) return NULL;
    if (hwo && wt->shape[0] != 1) return NULL;
    int taps = kh * kw;
    int16_t* out = (int16_t*)malloc((size_t)c * taps * sizeof(int16_t));
    if (!out) return NULL;
    for (int ch = 0; ch < c; ch++) {
        int wz = wzp ? quantized_value_i32(wzp, wzp->numel == 1 ? 0 : ch) : 0;
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                int t = ky * kw + kx;
                long src = hwo
                    ? (((long)0 * kh + ky) * kw + kx) * c + ch
                    : ((long)ch * taps + t);
                out[(long)t * c + ch] = (int16_t)(quantized_weight_i32(wt, src) - wz);
            }
        }
    }
    g_qdwcache[node_idx] = out;
    return out;
}

static int16_t* qconv_pointwise_koc_cache(int node_idx, const T* wt, const int16_t* wi) {
    if (g_qpwcache[node_idx]) return g_qpwcache[node_idx];
    if (!wi || wt->ndim != 4) return NULL;
    int out_c = wt->shape[0], k_total = wt->shape[1] * wt->shape[2] * wt->shape[3];
    int16_t* out = (int16_t*)malloc((size_t)k_total * out_c * sizeof(int16_t));
    if (!out) return NULL;
    for (int k = 0; k < k_total; k++) {
        for (int oc = 0; oc < out_c; oc++) {
            out[(long)k * out_c + oc] = wi[(long)oc * k_total + k];
        }
    }
    g_qpwcache[node_idx] = out;
    return out;
}

// Pre-interleave weights so the pointwise SIMD pair loop reads one contiguous 16xi16
// (matching the SIMD madd lane order) instead of re-interleaving every iteration.
// Layout per (oc block of 8, k pair): lane 2*j = w[oc+j][2*kp], lane 2*j+1 = w[oc+j][2*kp+1].
static int16_t* qconv_pointwise_il_cache(int node_idx, const T* wt, const int16_t* wi) {
    if (g_qpwilcache[node_idx]) return g_qpwilcache[node_idx];
    if (!wi || wt->ndim != 4) return NULL;
    int out_c = wt->shape[0], k_total = wt->shape[1] * wt->shape[2] * wt->shape[3];
    int oc_blocks = out_c / 8, n_kpair = k_total / 2;
    size_t n = (size_t)oc_blocks * n_kpair * 16;
    int16_t* out = (int16_t*)malloc((n ? n : 1) * sizeof(int16_t));
    if (!out) return NULL;
    for (int ob = 0; ob < oc_blocks; ob++) {
        int oc = ob * 8;
        for (int kp = 0; kp < n_kpair; kp++) {
            int16_t* dst = out + ((long)ob * n_kpair + kp) * 16;
            int k0 = 2 * kp, k1 = 2 * kp + 1;
            for (int j = 0; j < 8; j++) {
                dst[2 * j + 0] = wi[(long)(oc + j) * k_total + k0];
                dst[2 * j + 1] = wi[(long)(oc + j) * k_total + k1];
            }
        }
    }
    g_qpwilcache[node_idx] = out;
    return out;
}

// Pack each pair as [w0,0,w1,0] per output channel so AVX2 vpmaddubsw sees
// only one real product per int16 lane and cannot saturate the pair sum.
static signed char* qconv_pointwise_u8s8z_cache(int node_idx, const T* wt, const int16_t* wi) {
    if (g_qpw8zcache[node_idx]) return g_qpw8zcache[node_idx];
    if (!wi || wt->ndim != 4) return NULL;
    int out_c = wt->shape[0], k_total = wt->shape[1] * wt->shape[2] * wt->shape[3];
    int oc_blocks = out_c / 8, n_kpair = k_total / 2;
    if (oc_blocks <= 0 || n_kpair <= 0) return NULL;
    for (long i = 0; i < (long)out_c * k_total; i++) {
        if (wi[i] < -128 || wi[i] > 127) return NULL;
    }
    size_t n = (size_t)oc_blocks * n_kpair * 32;
    signed char* out = (signed char*)malloc(n ? n : 1);
    if (!out) return NULL;
    for (int ob = 0; ob < oc_blocks; ob++) {
        int oc = ob * 8;
        for (int kp = 0; kp < n_kpair; kp++) {
            signed char* dst = out + ((long)ob * n_kpair + kp) * 32;
            int k0 = 2 * kp, k1 = 2 * kp + 1;
            for (int j = 0; j < 8; j++) {
                dst[4 * j + 0] = (signed char)wi[(long)(oc + j) * k_total + k0];
                dst[4 * j + 1] = 0;
                dst[4 * j + 2] = (signed char)wi[(long)(oc + j) * k_total + k1];
                dst[4 * j + 3] = 0;
            }
        }
    }
    g_qpw8zcache[node_idx] = out;
    return out;
}

static float* qconv_dequant_weight_cache(int node_idx, const T* wt, const T* wscale, const T* wzp,
                                         const char* layout) {
    if (g_qwcache[node_idx]) return g_qwcache[node_idx];
    float* out = (float*)malloc((size_t)wt->numel * sizeof(float));
    if (!out) return NULL;
    int ohwi = layout && !strcmp(layout, "OHWI");
    int hwo = layout && (!strcmp(layout, "1HWO") || !strcmp(layout, "1HWM"));
    if (ohwi) {
        int out_c = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], in_c = wt->shape[3];
        for (int oc = 0; oc < out_c; oc++) {
            float ws = wscale->numel == 1 ? wscale->data[0] : wscale->data[oc];
            int wz = wzp ? quantized_value_i32(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    for (int ic = 0; ic < in_c; ic++) {
                        long idx = (((long)oc * kh + ky) * kw + kx) * in_c + ic;
                        int wv = quantized_weight_i32(wt, idx) - wz;
                        out[idx] = (float)wv * ws;
                    }
                }
            }
        }
    } else if (hwo) {
        int one = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], out_c = wt->shape[3];
        if (one != 1) { free(out); return NULL; }
        for (int oc = 0; oc < out_c; oc++) {
            float ws = wscale->numel == 1 ? wscale->data[0] : wscale->data[oc];
            int wz = wzp ? quantized_value_i32(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    long idx = (((long)0 * kh + ky) * kw + kx) * out_c + oc;
                    int wv = quantized_weight_i32(wt, idx) - wz;
                    out[idx] = (float)wv * ws;
                }
            }
        }
    } else {
        int out_c = wt->shape[0];
        long per_oc = wt->numel / out_c;
        for (int oc = 0; oc < out_c; oc++) {
            float ws = wscale->numel == 1 ? wscale->data[0] : wscale->data[oc];
            int wz = wzp ? quantized_value_i32(wzp, wzp->numel == 1 ? 0 : oc) : 0;
            long base = (long)oc * per_oc;
            for (long i = 0; i < per_oc; i++) {
                long idx = base + i;
                int wv = quantized_weight_i32(wt, idx) - wz;
                out[idx] = (float)wv * ws;
            }
        }
    }
    g_qwcache[node_idx] = out;
    return out;
}

static float* f16_tensor_f32_cache(float** cache, int node_idx, const T* t) {
    if (!t || t->dtype != T_F16 || node_idx < 0 || node_idx >= MAXN) return NULL;
    if (cache[node_idx]) return cache[node_idx];
    float* out = (float*)malloc((size_t)t->numel * sizeof(float));
    if (!out) return NULL;
    for (long i = 0; i < t->numel; i++) out[i] = tensor_scalar_f32(t, i);
    cache[node_idx] = out;
    return out;
}

static int layout_is(const char* layout, const char* want) {
    return layout && want && !strcmp(layout, want);
}

static T* prepare_conv_weight_image(int node_idx, T* wt, cJSON* params, int in_c, int out_c, int groups, T* tmp) {
    if (!wt || wt->ndim != 4 || node_idx < 0 || node_idx >= MAXN) return NULL;
    const char* layout = p_str(params, "weight_layout", groups == in_c ? "1HWO" : "OHWI");

    if (layout_is(layout, "HWIO") || layout_is(layout, "HWCM")) {
        if (wt->dtype == T_F32) return wt;
        if (wt->dtype != T_F16) return NULL;
        float* wf = f16_tensor_f32_cache(g_conv_wcache, node_idx, wt);
        if (!wf) return NULL;
        *tmp = *wt;
        tmp->data = wf;
        tmp->dtype = T_F32;
        tmp->elem_size = sizeof(float);
        return tmp;
    }

    int conv_ohwi = layout_is(layout, "OHWI");
    int dw_1hwo = layout_is(layout, "1HWO") || layout_is(layout, "1HWM");
    if (!conv_ohwi && !dw_1hwo) return NULL;

    if (!g_conv_wcache[node_idx]) {
        float* out = (float*)malloc((size_t)wt->numel * sizeof(float));
        if (!out) return NULL;
        if (conv_ohwi) {
            int oc_n = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], ic_n = wt->shape[3];
            if (groups != 1 || oc_n != out_c || ic_n != in_c) { free(out); return NULL; }
            for (int oc = 0; oc < oc_n; oc++) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        for (int ic = 0; ic < ic_n; ic++) {
                            long src = (((long)oc * kh + ky) * kw + kx) * ic_n + ic;
                            long dst = (((long)ky * kw + kx) * ic_n + ic) * oc_n + oc;
                            out[dst] = tensor_scalar_f32(wt, src);
                        }
                    }
                }
            }
        } else {
            int one = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], oc_n = wt->shape[3];
            if (one != 1 || groups != in_c || oc_n != out_c || in_c <= 0 || out_c % in_c) {
                free(out);
                return NULL;
            }
            int mult = out_c / in_c;
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    for (int ch = 0; ch < in_c; ch++) {
                        for (int m = 0; m < mult; m++) {
                            int oc = ch * mult + m;
                            long src = (((long)0 * kh + ky) * kw + kx) * out_c + oc;
                            long dst = (((long)ky * kw + kx) * in_c + ch) * mult + m;
                            out[dst] = tensor_scalar_f32(wt, src);
                        }
                    }
                }
            }
        }
        g_conv_wcache[node_idx] = out;
    }

    *tmp = *wt;
    tmp->data = g_conv_wcache[node_idx];
    tmp->dtype = T_F32;
    tmp->elem_size = sizeof(float);
    if (conv_ohwi) {
        tmp->shape[0] = wt->shape[1];
        tmp->shape[1] = wt->shape[2];
        tmp->shape[2] = wt->shape[3];
        tmp->shape[3] = wt->shape[0];
    } else {
        tmp->shape[0] = wt->shape[1];
        tmp->shape[1] = wt->shape[2];
        tmp->shape[2] = in_c;
        tmp->shape[3] = out_c / in_c;
    }
    return tmp;
}

int prepack_conv_weights(void) {
    double t0 = g_debug ? engine_now_ms() : 0.0;
    int conv_weights = 0, biases = 0, pointwise_packs = 0, igemm_indirs = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* n = &g_n[i];
        if (strcmp(n->op, "Conv2D")) continue;
        T* w = nin(n, "weight");
        T* b = nin(n, "bias");
        T* in = nin(n, "input");
        T* out = t_find(n->out);
        T tmp;
        int groups = p_int(n->params, "groups", 1);
        T* wp = (w && in && out) ? prepare_conv_weight_image(i, w, n->params, in->shape[3], out->shape[3], groups, &tmp) : NULL;
        if (wp && wp != w) {
            if (!g_conv_wcache[i]) return -1;
            conv_weights++;
        }
        if (wp && in && out && groups == 1 && wp->ndim >= 4 &&
            wp->shape[0] == 1 && wp->shape[1] == 1 &&
            vx_pwf32_pack_cache(i, wp->data, in->shape[3], out->shape[3])) {
            pointwise_packs++;
        } else if (wp && in && out && groups == 1 && wp->ndim >= 4 &&
                   wp->shape[0] * wp->shape[1] > 1 && out->shape[3] >= 16) {
            int py = p_int(n->params, "padding_y", p_int(n->params, "padding", 0));
            int px = p_int(n->params, "padding_x", p_int(n->params, "padding", 0));
            int stride[2] = {1, 1}, dilation[2] = {1, 1}, pads[4] = {py, px, py, px};
            p_iarr(n->params, "stride", stride, 2, 1);
            p_iarr(n->params, "dilation", dilation, 2, 1);
            if (cJSON_GetObjectItem(n->params, "pads")) p_iarr(n->params, "pads", pads, 4, 0);
            if (vx_f32_igemm_indirection_cache(i, in->data,
                                               in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                               out->shape[1], out->shape[2], wp->shape[0], wp->shape[1],
                                               stride[0], stride[1], pads, dilation[0], dilation[1])) {
                igemm_indirs++;
            }
        }
        if (b && b->dtype == T_F16) {
            if (!f16_tensor_f32_cache(g_f16bcache, i, b)) return -1;
            biases++;
        }
    }
    if (g_debug && (conv_weights || biases || pointwise_packs || igemm_indirs)) {
        fprintf(stderr, "[debug] prepack_conv weights=%d biases=%d pointwise_packs=%d igemm_indirs=%d %.3f ms\n",
                conv_weights, biases, pointwise_packs, igemm_indirs, engine_now_ms() - t0);
    }
    return 0;
}

static int qconv2d_quantized(int node_idx, const T* in, T* out, const T* wt, const T* wscale,
                              const T* wzp, const T* bias, int sy, int sx,
                              const int* pads, int groups, int dy, int dx, int relu,
                              float input_scale, int input_zp,
                              const char* weight_layout, const char** backend) {
    (void)node_idx;
    const char* disable_qconv = getenv("VOLVOX_DISABLE_QCONV");
    if (disable_qconv && disable_qconv[0] && strcmp(disable_qconv, "0")) return 0;
    if (!in || !out || !wt || !wscale || in->ndim != 4 || out->ndim != 4 || wt->ndim != 4) return 0;
    if (input_scale <= 0.0f || (wt->dtype != T_I8 && wt->dtype != T_U8) || dy != 1 || dx != 1) return 0;
    if (in->shape[0] != 1 || out->shape[0] != 1) return 0;
    QTensor* qi = qt_get((T*)in);
    if (!qi || !qi->valid || fabsf(qi->scale - input_scale) > 1e-8f || qi->zp != input_zp) {
        if (qt_quantize_from_f32((T*)in, input_scale, input_zp) < 0) return -1;
        qi = qt_get((T*)in);
    }
    QTensor* qo = qt_get(out);
    int output_quant = qo && qo->has_params;
    signed char* yq = NULL;
    float out_scale = 1.0f;
    int out_zp = 0;
    if (output_quant) {
        if (qt_alloc_for(out, qo->scale, qo->zp) != 0) return -1;
        qo = qt_get(out);
        yq = qo->data;
        out_scale = qo->scale;
        out_zp = qo->zp;
    }
    int c = in->shape[3];
    int h = in->shape[1];
    int w = in->shape[2];
    int oh = out->shape[1];
    int ow = out->shape[2];
    int out_c = wt->shape[0], in_per_group = wt->shape[1], kh = wt->shape[2], kw = wt->shape[3];
    if (layout_is(weight_layout, "OHWI")) {
        out_c = wt->shape[0];
        kh = wt->shape[1];
        kw = wt->shape[2];
        in_per_group = wt->shape[3];
    } else if (layout_is(weight_layout, "1HWO") || layout_is(weight_layout, "1HWM")) {
        out_c = wt->shape[3];
        kh = wt->shape[1];
        kw = wt->shape[2];
        in_per_group = 1;
    }
    int pt = pads[0], pl = pads[1];
    float* yf = output_quant ? NULL : out->data;
    int16_t* wi = qconv_i16_weight_cache(node_idx, wt, wzp, weight_layout);
    if (!wi) return -1;
    int16_t* wpack = NULL;
    int16_t* stem_il = NULL;
    if (c == 3 && in_per_group == 3 && kh == 3 && kw == 3 && sy == 2 && sx == 2 && groups == 1 &&
        (wpack = qconv_pointwise_koc_cache(node_idx, wt, wi)) &&
        (stem_il = qconv_pointwise_il_cache(node_idx, wt, wi)) &&
        vx_qconv2d_stem3s2(qi->data, yq, yf, wi, wpack, stem_il,
                                wscale->data, wscale->numel, bias ? bias->data : NULL,
                                h, w, out_c, oh, ow, pt, pl, input_scale, input_zp, out_scale, out_zp, relu)) {
        // done
    } else if (kh == 1 && kw == 1 && sy == 1 && sx == 1 && groups == 1 && pt == 0 && pl == 0 && pads[2] == 0 && pads[3] == 0) {
        wpack = qconv_pointwise_koc_cache(node_idx, wt, wi);
        int16_t* wpack_il = qconv_pointwise_il_cache(node_idx, wt, wi);
        signed char* wpack8z = input_zp == -128 ? qconv_pointwise_u8s8z_cache(node_idx, wt, wi) : NULL;
        if (!wpack || !wpack_il) return -1;
        vx_qconv2d_pointwise(qi->data, yq, yf, wi, wpack, wpack_il, wpack8z,
                                  wscale->data, wscale->numel, bias ? bias->data : NULL,
                                  c, h, w, out_c, input_scale, input_zp, out_scale, out_zp, relu);
    } else if (groups == c && out_c == c && in_per_group == 1) {
        int16_t* wtap = qconv_depthwise_tap_cache(node_idx, wt, wzp, weight_layout);
        if (!wtap) return -1;
        vx_qconv2d_depthwise(qi->data, yq, yf, wtap,
                                  wscale->data, wscale->numel, bias ? bias->data : NULL,
                                  c, h, w, oh, ow, kh, kw, sy, sx, pt, pl,
                                  input_scale, input_zp, out_scale, out_zp, relu);
    } else {
        vx_qconv2d_generic(qi->data, yq, yf, wi, T_F16,
                                wscale->data, wscale->numel,
                                NULL, T_I32, 0,
                                bias ? bias->data : NULL,
                                c, h, w, out_c, in_per_group, oh, ow, kh, kw, sy, sx, pt, pl, groups,
                                input_scale, input_zp, out_scale, out_zp, relu);
    }
    if (output_quant) qo->valid = 1;
    if (backend) *backend = output_quant ? "cpu-qconv" : "cpu-qconv-f32";
    return 1;
}

static int qconv2d_cached_f32(int node_idx, const T* in, T* out, const T* wt, const T* wscale,
                              const T* wzp, const T* bias, int sy, int sx,
                              const int* pads, int groups, int dy, int dx, int relu,
                              cJSON* params, const char* weight_layout) {
    float* w_deq = qconv_dequant_weight_cache(node_idx, wt, wscale, wzp, weight_layout);
    if (!w_deq) return -1;
    T wraw = *wt;
    wraw.data = w_deq;
    wraw.dtype = T_F32;
    wraw.elem_size = sizeof(float);
    T wtmp;
    T* wp = prepare_conv_weight_image(node_idx, &wraw, params, in->shape[3], out->shape[3], groups, &wtmp);
    if (!wp) return -1;
    conv2d_f32_image(node_idx, in, out, wp, bias, NULL, sy, sx, pads, groups, dy, dx, relu);
    return 0;
}

int prepack_qconv_weights(void) {
    double t0 = g_debug ? engine_now_ms() : 0.0;
    if (g_debug) fprintf(stderr, "[debug] prepack_qconv weights=0 %.3f ms\n", engine_now_ms() - t0);
    return 0;
}

enum { GPU_GRAPH_VULKAN = 1, GPU_GRAPH_OPENGL = 2, GPU_GRAPH_METAL = 3 };

static int output_ref_cmp(const Ref* a, const Ref* b) {
    return strcmp(a->key, b->key);
}

static T* node_data_input(Node* n, T* input) {
    if (input) return input;
    T* x = nin(n, "data");
    if (!x) x = nin(n, "x");
    if (!x && n->nin > 0) x = t_find(n->ins[0].name);
    return x;
}

static int normalize_axis(int axis, int rank) {
    if (axis < 0) axis += rank;
    return axis;
}

static int tensor_inner_after_axis(const T* t, int axis) {
    int inner = 1;
    if (!t || axis < 0 || axis >= t->ndim) return 0;
    for (int i = axis + 1; i < t->ndim; i++) inner *= t->shape[i];
    return inner;
}

static int tensor_seq_len_from_last_dim(const T* t, int d_model) {
    if (!t || d_model <= 0 || t->numel % d_model) return 0;
    return (int)(t->numel / d_model);
}

static int batch_is_one_or_flat(const T* t) {
    return t && (t->ndim < 3 || t->shape[0] == 1);
}

static int flat_broadcast_ok(long n, long out_n) {
    return n > 0 && out_n > 0 && n <= out_n && (n == 1 || out_n % n == 0);
}

static int try_gpu_graph_binary(Node* n, T* in, T* o, int backend) {
    T* a = nin(n, "a");
    T* b = nin(n, "b");
    if (!a) a = node_data_input(n, in);
    if (!a || !b || !o) return 0;
    materialize_tensor_f32(a);
    materialize_tensor_f32(b);
    if (a->dtype != T_F32 || b->dtype != T_F32 || !flat_broadcast_ok(a->numel, o->numel) ||
        !flat_broadcast_ok(b->numel, o->numel)) return 0;
    int op = !strcmp(n->op, "Sub") ? 1 : (!strcmp(n->op, "Div") ? 2 : 0);
    if (backend == GPU_GRAPH_VULKAN)
        return vk_graph_binary_f32(a->data, a->numel, b->data, b->numel, o->data, o->numel, op);
#ifdef __APPLE__
    if (backend == GPU_GRAPH_METAL)
        return metal_graph_binary_f32(a->data, a->numel, b->data, b->numel, o->data, o->numel, op);
#endif
    return opengl_graph_binary_f32(a->data, a->numel, b->data, b->numel, o->data, o->numel, op);
}

static int try_gpu_graph_conv1d(Node* n, T* in, T* o, int backend) {
    T* x = node_data_input(n, in);
    T* w = nin(n, "weight");
    T* b = nin(n, "bias");
    if (!x || !w || !o || w->ndim != 3) return 0;
    if (!((x->ndim == 3 && x->shape[0] == 1) || x->ndim == 2)) return 0;
    if (!((o->ndim == 3 && o->shape[0] == 1) || o->ndim == 2)) return 0;
    materialize_tensor_f32(x);
    materialize_tensor_f32(w);
    if (b) materialize_tensor_f32(b);
    if (x->dtype != T_F32 || w->dtype != T_F32 || (b && b->dtype != T_F32)) return 0;
    int in_c = x->ndim == 3 ? x->shape[1] : x->shape[0];
    int in_l = x->ndim == 3 ? x->shape[2] : x->shape[1];
    int out_c = o->ndim == 3 ? o->shape[1] : o->shape[0];
    int out_l = o->ndim == 3 ? o->shape[2] : o->shape[1];
    if (w->shape[0] != out_c || w->shape[1] != in_c || (b && b->numel < out_c)) return 0;
    int sy, sx, py, px;
    p_pair(n->params, "stride", 1, &sy, &sx);
    p_pair(n->params, "padding", 0, &py, &px);
    int relu = p_int(n->params, "relu", 0) ? 1 : 0;
    if (backend == GPU_GRAPH_VULKAN)
        return vk_graph_conv1d_f32(x->data, w->data, b ? b->data : NULL, o->data,
                                   in_c, in_l, out_c, out_l, w->shape[2], sy, py, relu);
#ifdef __APPLE__
    if (backend == GPU_GRAPH_METAL)
        return metal_graph_conv1d_f32(x->data, w->data, b ? b->data : NULL, o->data,
                                      in_c, in_l, out_c, out_l, w->shape[2], sy, py, relu);
#endif
    return opengl_graph_conv1d_f32(x->data, w->data, b ? b->data : NULL, o->data,
                                   in_c, in_l, out_c, out_l, w->shape[2], sy, py, relu);
}

static int try_gpu_graph_sdpa(Node* n, T* in, T* o, int backend) {
    T* qkv = nin(n, "qkv");
    if (!qkv) qkv = node_data_input(n, in);
    if (!qkv || !o || !batch_is_one_or_flat(qkv) || !batch_is_one_or_flat(o) || o->ndim <= 0 || qkv->ndim <= 0) return 0;
    materialize_tensor_f32(qkv);
    if (qkv->dtype != T_F32 || qkv->shape[qkv->ndim - 1] != o->shape[o->ndim - 1] * 3) return 0;
    int d_model = o->shape[o->ndim - 1];
    int seq_len = tensor_seq_len_from_last_dim(o, d_model);
    int heads = p_int(n->params, "heads", 8);
    if (heads <= 0 || d_model % heads) return 0;
    int head_dim = d_model / heads;
    if (qkv->numel != (long)seq_len * 3 * d_model || head_dim > 64) return 0;
    float scale = p_flt(n->params, "scale", 1.0f / sqrtf((float)head_dim));
    if (backend == GPU_GRAPH_VULKAN)
        return vk_graph_sdpa_f32(qkv->data, o->data, seq_len, d_model, heads, head_dim, scale);
#ifdef __APPLE__
    if (backend == GPU_GRAPH_METAL)
        return metal_graph_sdpa_f32(qkv->data, o->data, seq_len, d_model, heads, head_dim, scale);
#endif
    return opengl_graph_sdpa_f32(qkv->data, o->data, seq_len, d_model, heads, head_dim, scale);
}

static int try_gpu_graph_cross_sdpa(Node* n, T* o, int backend) {
    T* q = nin(n, "q");
    T* k = nin(n, "k");
    T* v = nin(n, "v");
    if (!q || !k || !v || !o || !batch_is_one_or_flat(q) || !batch_is_one_or_flat(k) ||
        !batch_is_one_or_flat(v) || o->ndim <= 0 || q->ndim <= 0 || k->ndim <= 0 || v->ndim <= 0) return 0;
    materialize_tensor_f32(q);
    materialize_tensor_f32(k);
    materialize_tensor_f32(v);
    int d_model = o->shape[o->ndim - 1];
    if (q->dtype != T_F32 || k->dtype != T_F32 || v->dtype != T_F32 ||
        q->shape[q->ndim - 1] != d_model || k->shape[k->ndim - 1] != d_model ||
        v->shape[v->ndim - 1] != d_model) return 0;
    int seq_q = tensor_seq_len_from_last_dim(q, d_model);
    int seq_kv = tensor_seq_len_from_last_dim(k, d_model);
    if (seq_q <= 0 || seq_kv <= 0 || v->numel != (long)seq_kv * d_model || o->numel != (long)seq_q * d_model) return 0;
    int heads = p_int(n->params, "heads", 8);
    if (heads <= 0 || d_model % heads) return 0;
    int head_dim = d_model / heads;
    if (head_dim > 64) return 0;
    float scale = p_flt(n->params, "scale", 1.0f / sqrtf((float)head_dim));
    if (backend == GPU_GRAPH_VULKAN)
        return vk_graph_cross_sdpa_f32(q->data, k->data, v->data, o->data, seq_q, seq_kv, d_model, heads, head_dim, scale);
#ifdef __APPLE__
    if (backend == GPU_GRAPH_METAL)
        return metal_graph_cross_sdpa_f32(q->data, k->data, v->data, o->data, seq_q, seq_kv, d_model, heads, head_dim, scale);
#endif
    return opengl_graph_cross_sdpa_f32(q->data, k->data, v->data, o->data, seq_q, seq_kv, d_model, heads, head_dim, scale);
}

static int try_gpu_graph_cross_attention(Node* n, T* o, int backend) {
    T* q = nin(n, "q");
    T* kv = nin(n, "kv");
    T* w = nin(n, "weight");
    T* scale = nin(n, "scale");
    T* bias = nin(n, "bias");
    if (!q || !kv || !w || !o || !batch_is_one_or_flat(q) || !batch_is_one_or_flat(kv) ||
        q->ndim <= 0 || kv->ndim <= 0 || o->ndim <= 0) return 0;
    materialize_tensor_f32(q);
    materialize_tensor_f32(kv);
    materialize_tensor_f32(w);
    if (scale) materialize_tensor_f32(scale);
    if (bias) materialize_tensor_f32(bias);
    int d_model = o->shape[o->ndim - 1];
    if (q->dtype != T_F32 || kv->dtype != T_F32 || w->dtype != T_F32 ||
        (scale && scale->dtype != T_F32) || (bias && bias->dtype != T_F32) ||
        q->shape[q->ndim - 1] != d_model || kv->shape[kv->ndim - 1] != d_model ||
        d_model > 64 || w->numel < (long)3 * d_model * d_model) return 0;
    if (scale && scale->numel < 3 * d_model) return 0;
    if (bias && bias->numel < 3 * d_model) return 0;
    int seq_q = tensor_seq_len_from_last_dim(q, d_model);
    int seq_kv = tensor_seq_len_from_last_dim(kv, d_model);
    int heads = p_int(n->params, "heads", 8);
    if (seq_q <= 0 || seq_kv <= 0 || o->numel != (long)seq_q * d_model || heads <= 0 || d_model % heads) return 0;
    int head_dim = d_model / heads;
    if (head_dim > 64) return 0;
    if (backend == GPU_GRAPH_VULKAN)
        return vk_graph_cross_attention_f32(q->data, kv->data, w->data, scale ? scale->data : NULL,
                                            bias ? bias->data : NULL, o->data, seq_q, seq_kv,
                                            d_model, heads, head_dim, scale != NULL, bias != NULL);
#ifdef __APPLE__
    if (backend == GPU_GRAPH_METAL)
        return metal_graph_cross_attention_f32(q->data, kv->data, w->data, scale ? scale->data : NULL,
                                               bias ? bias->data : NULL, o->data, seq_q, seq_kv,
                                               d_model, heads, head_dim, scale != NULL, bias != NULL);
#endif
    return opengl_graph_cross_attention_f32(q->data, kv->data, w->data, scale ? scale->data : NULL,
                                            bias ? bias->data : NULL, o->data, seq_q, seq_kv,
                                            d_model, heads, head_dim, scale != NULL, bias != NULL);
}

static int try_gpu_graph_dequantize_linear(Node* n, T* in, T* o, int backend) {
    T* x = node_data_input(n, in);
    T* scale = nin(n, "scale");
    T* zp = nin(n, "zero_point");
    if (!x || !scale || !o || x->numel != o->numel || scale->numel <= 0) return 0;
    QTensor* q = qt_get(x);
    if (q && q->valid && q->has_params) return 0;
    materialize_tensor_f32(x);
    materialize_tensor_f32(scale);
    if (zp) materialize_tensor_f32(zp);
    if (x->dtype != T_F32 || scale->dtype != T_F32 || (zp && zp->dtype != T_F32)) return 0;
    if (backend == GPU_GRAPH_VULKAN)
        return vk_graph_dequantize_linear_f32(x->data, scale->data, zp ? zp->data : NULL, o->data, o->numel, zp != NULL);
#ifdef __APPLE__
    if (backend == GPU_GRAPH_METAL)
        return metal_graph_dequantize_linear_f32(x->data, scale->data, zp ? zp->data : NULL, o->data, o->numel, zp != NULL);
#endif
    return opengl_graph_dequantize_linear_f32(x->data, scale->data, zp ? zp->data : NULL, o->data, o->numel, zp != NULL);
}

static int try_gpu_graph_quantize_linear(Node* n, T* in, T* o, int backend) {
    T* x = node_data_input(n, in);
    if (!n || !x || !o || x->dtype != T_F32 || o->dtype != T_F32 || x->numel != o->numel) return 0;
    float input_scale = p_flt(n->params, "input_scale", 0.0f);
    int input_zp = p_int(n->params, "input_zero_point", 0);
    float output_scale = p_flt(n->params, "output_scale", 0.0f);
    int output_zp = p_int(n->params, "output_zero_point", 0);
    if (output_scale <= 0.0f) return 0;

    QTensor* qi = qt_get(x);
    int materialized_qinput = qi && qi->valid && qi->has_params;
    materialize_tensor_f32(x);
    if (materialized_qinput) vk_mark_host_tensor(x);
    if (qt_alloc_for(o, output_scale, output_zp) != 0) return 0;
    QTensor* qo = qt_get(o);
    if (!qo || !qo->data) return 0;
    qo->valid = 0;

    int ok = 0;
    if (backend == GPU_GRAPH_VULKAN) {
        ok = vk_graph_quantize_linear_i8(x->data, qo->data, x->numel,
                                         input_scale, input_zp, output_scale, output_zp);
    } else if (backend == GPU_GRAPH_OPENGL) {
        ok = opengl_graph_quantize_linear_i8(x->data, qo->data, x->numel,
                                             input_scale, input_zp, output_scale, output_zp);
    }
#ifdef __APPLE__
    else if (backend == GPU_GRAPH_METAL) {
        ok = metal_graph_quantize_linear_i8(x->data, qo->data, x->numel,
                                            input_scale, input_zp, output_scale, output_zp);
    }
#endif
    if (!ok) return 0;
    qo->valid = 1;
    return 1;
}

static int try_gpu_graph_split(Node* n, T* in, int backend) {
    T* x = node_data_input(n, in);
    if (!x || n->nout <= 0) return 0;
    int axis = normalize_axis(p_int(n->params, "axis", 0), x->ndim);
    if (axis < 0 || axis >= x->ndim) return 0;
    int order[MAXIN];
    for (int i = 0; i < n->nout; i++) order[i] = i;
    for (int i = 0; i < n->nout; i++) {
        for (int j = i + 1; j < n->nout; j++) {
            if (output_ref_cmp(&n->outs[order[j]], &n->outs[order[i]]) < 0) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }
    materialize_tensor_f32(x);
    if (x->dtype != T_F32) return 0;
    int inner = tensor_inner_after_axis(x, axis);
    int offset = 0;
    for (int oi = 0; oi < n->nout; oi++) {
        T* dst = t_find(n->outs[order[oi]].name);
        if (!dst || dst->dtype != T_F32 || dst->ndim != x->ndim) return 0;
        for (int d = 0; d < x->ndim; d++) {
            if (d != axis && dst->shape[d] != x->shape[d]) return 0;
        }
        offset += dst->shape[axis];
    }
    if (offset != x->shape[axis]) return 0;
    offset = 0;
    for (int oi = 0; oi < n->nout; oi++) {
        T* dst = t_find(n->outs[order[oi]].name);
        int split_size = dst->shape[axis];
        int ok = 0;
        if (backend == GPU_GRAPH_VULKAN) {
            ok = vk_graph_split_f32(x->data, dst->data, x->numel, dst->numel, inner, split_size, x->shape[axis], offset);
#ifdef __APPLE__
        } else if (backend == GPU_GRAPH_METAL) {
            ok = metal_graph_split_f32(x->data, dst->data, x->numel, dst->numel, inner, split_size, x->shape[axis], offset);
#endif
        } else {
            ok = opengl_graph_split_f32(x->data, dst->data, x->numel, dst->numel, inner, split_size, x->shape[axis], offset);
        }
        if (!ok) return 0;
        offset += split_size;
    }
    return 1;
}

static int try_gpu_graph_profile_op(Node* n, T* in, T* o, int backend) {
    T* x = node_data_input(n, in);
    if (!x || !o || x->ndim != 4 || x->shape[0] != 1) return 0;
    materialize_tensor_f32(x);
    if (x->dtype != T_F32) return 0;
    int h = x->shape[1], w = x->shape[2], c = x->shape[3];
    int ok = 0;
    if (!strcmp(n->op, "SpatialSoftargmaxY")) {
        if (o->numel != (long)c * w) return 0;
        if (backend == GPU_GRAPH_VULKAN) ok = vk_graph_spatial_softargmax_y_f32(x->data, o->data, h, w, c);
#ifdef __APPLE__
        else if (backend == GPU_GRAPH_METAL) ok = metal_graph_spatial_softargmax_y_f32(x->data, o->data, h, w, c);
#endif
        else ok = opengl_graph_spatial_softargmax_y_f32(x->data, o->data, h, w, c);
    } else if (!strcmp(n->op, "ProfileX")) {
        if (o->numel != (long)2 * c * w) return 0;
        if (backend == GPU_GRAPH_VULKAN) ok = vk_graph_profile_x_f32(x->data, o->data, h, w, c);
#ifdef __APPLE__
        else if (backend == GPU_GRAPH_METAL) ok = metal_graph_profile_x_f32(x->data, o->data, h, w, c);
#endif
        else ok = opengl_graph_profile_x_f32(x->data, o->data, h, w, c);
    } else if (!strcmp(n->op, "ProfileY")) {
        if (o->numel != (long)2 * c * h) return 0;
        if (backend == GPU_GRAPH_VULKAN) ok = vk_graph_profile_y_f32(x->data, o->data, h, w, c);
#ifdef __APPLE__
        else if (backend == GPU_GRAPH_METAL) ok = metal_graph_profile_y_f32(x->data, o->data, h, w, c);
#endif
        else ok = opengl_graph_profile_y_f32(x->data, o->data, h, w, c);
    } else if (!strcmp(n->op, "MeanHeight")) {
        if (o->numel != (long)c * w) return 0;
        if (backend == GPU_GRAPH_VULKAN) ok = vk_graph_mean_height_f32(x->data, o->data, h, w, c);
#ifdef __APPLE__
        else if (backend == GPU_GRAPH_METAL) ok = metal_graph_mean_height_f32(x->data, o->data, h, w, c);
#endif
        else ok = opengl_graph_mean_height_f32(x->data, o->data, h, w, c);
    }
    return ok;
}

static int try_gpu_graph_nms(Node* n, T* o, int backend) {
    T* boxes = nin(n, "boxes");
    T* scores = nin(n, "scores");
    if (!boxes || !scores || !o || boxes->ndim != 3 || scores->ndim != 3 || boxes->shape[2] != 4 ||
        scores->shape[0] != boxes->shape[0] || scores->shape[2] != boxes->shape[1] || o->numel % 3) return 0;
    materialize_tensor_f32(boxes);
    materialize_tensor_f32(scores);
    if (boxes->dtype != T_F32 || scores->dtype != T_F32) return 0;
    T* max_t = nin(n, "max_output_boxes_per_class");
    T* iou_t = nin(n, "iou_threshold");
    T* score_t = nin(n, "score_threshold");
    int max_output = p_int(n->params, "max_output_boxes_per_class", 0);
    float iou = p_flt(n->params, "iou_threshold", 0.5f);
    float score = p_flt(n->params, "score_threshold", 0.0f);
    if (max_t) vk_sync_host_tensor(max_t);
    if (iou_t) vk_sync_host_tensor(iou_t);
    if (score_t) vk_sync_host_tensor(score_t);
    if (max_t && max_t->numel > 0) max_output = (int)tensor_scalar_any(max_t, 0);
    if (iou_t && iou_t->numel > 0) iou = tensor_scalar_any(iou_t, 0);
    if (score_t && score_t->numel > 0) score = tensor_scalar_any(score_t, 0);
    int output_rows = (int)(o->numel / 3);
    if (max_output <= 0) max_output = output_rows;
    if (backend == GPU_GRAPH_VULKAN)
        return vk_graph_nms_f32(boxes->data, scores->data, o->data, boxes->shape[0], boxes->shape[1],
                                scores->shape[1], max_output, output_rows, iou, score);
#ifdef __APPLE__
    if (backend == GPU_GRAPH_METAL)
        return metal_graph_nms_f32(boxes->data, scores->data, o->data, boxes->shape[0], boxes->shape[1],
                                   scores->shape[1], max_output, output_rows, iou, score);
#endif
    return opengl_graph_nms_f32(boxes->data, scores->data, o->data, boxes->shape[0], boxes->shape[1],
                                scores->shape[1], max_output, output_rows, iou, score);
}

static int try_vulkan_graph_node(Node* n, int idx, const char** node_backend) {
    if (!g_use_vulkan || !n || g_dec_pos >= 0 || g_prefill_len > 0) return 0;
    T* o = t_find(n->out);
    T* in = nin(n, "input");
    cJSON* p = n->params;
    const char* op = n->op;
    if (!o || o->dtype != T_F32) return 0;

    if (!strcmp(op, "QConv2D")) {
        return 0;
    }

    if (!strcmp(op, "Conv2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || in->dtype != T_F32 || (w->dtype != T_F32 && w->dtype != T_F16) ||
            o->ndim != 4 || in->ndim != 4 || w->ndim != 4) return 0;
        if (in->shape[0] != 1 || o->shape[0] != 1) return 0;
        int sy, sx, py, px, dy, dx; p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px); p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px}; p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1), relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);
        if (pads[0] < 0 || pads[1] < 0 || relu > 2) return 0;
        T wtmp, btmp;
        T* wp = prepare_conv_weight_image(idx, w, p, in->shape[3], o->shape[3], groups, &wtmp);
        if (!wp) return 0;
        T* bp = b;
        if (b && b->dtype == T_F16) {
            float* bf = f16_tensor_f32_cache(g_f16bcache, idx, b);
            if (!bf) return 0;
            btmp = *b; btmp.data = bf; btmp.dtype = T_F32; btmp.elem_size = sizeof(float);
            bp = &btmp;
        }
        if (b && bp->dtype != T_F32) return 0;
        if (vk_graph_conv2d_f32(in->data, o->data, wp->data, bp ? bp->data : NULL,
                                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                o->shape[3], wp->shape[0], wp->shape[1], o->shape[1], o->shape[2],
                                sy, sx, pads[0], pads[1], groups, relu, dy, dx)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Add")) {
        T* a = nin(n, "a"); T* b = nin(n, "b");
        if (!a) a = in;
        if (a && b && a->dtype == T_F32 && b->dtype == T_F32 &&
            a->numel == b->numel && a->numel == o->numel &&
            vk_graph_add_relu_f32(a->data, b->data, o->data, o->numel, p_int(p, "relu", 0))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Mul") || !strcmp(op, "Sub") || !strcmp(op, "Div")) {
        if (try_gpu_graph_binary(n, in, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Sigmoid")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            vk_graph_sigmoid_f32(in->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Clip")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            vk_graph_clip_f32(in->data, o->data, o->numel,
                              p_flt(p, "min", -1e30f), p_flt(p, "max", 1e30f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReLU") || !strcmp(op, "GELU") || !strcmp(op, "SiLU") || !strcmp(op, "Swish") ||
        !strcmp(op, "Tanh") || !strcmp(op, "HardSwish") || !strcmp(op, "HardSigmoid")) {
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype != T_F32 || in->numel != o->numel) return 0;
        int ok = 0;
        if (!strcmp(op, "ReLU")) ok = vk_graph_relu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "GELU")) ok = vk_graph_gelu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "SiLU") || !strcmp(op, "Swish")) ok = vk_graph_silu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "Tanh")) ok = vk_graph_tanh_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "HardSwish")) ok = vk_graph_hardswish_f32(in->data, o->data, o->numel);
        else ok = vk_graph_hardsigmoid_f32(in->data, o->data, o->numel);
        if (ok) { *node_backend = "vulkan-graph"; return 1; }
        return 0;
    }

    if (!strcmp(op, "LeakyReLU")) {
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 && in->numel == o->numel &&
            vk_graph_leaky_relu_f32(in->data, o->data, o->numel, p_flt(p, "alpha", 0.01f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "PReLU")) {
        T* w = nin(n, "weight");
        if (!in || !w) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int c = in->ndim > 0 ? in->shape[in->ndim - 1] : 1;
        if (in->dtype == T_F32 && w->dtype == T_F32 && in->numel == o->numel && w->numel >= c &&
            vk_graph_prelu_f32(in->data, w->data, o->data, o->numel, c)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "LayerNorm")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || !b) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); materialize_tensor_f32(b);
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        int rows = d_model > 0 ? (int)(in->numel / d_model) : 0;
        if (in->dtype == T_F32 && w->dtype == T_F32 && b->dtype == T_F32 && o->numel == in->numel &&
            w->numel >= d_model && b->numel >= d_model &&
            vk_graph_layernorm_f32(in->data, w->data, b->data, o->data, rows, d_model)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "RMSNorm")) {
        T* w = nin(n, "weight");
        if (!in || !w) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        int rows = d_model > 0 ? (int)(in->numel / d_model) : 0;
        if (in->dtype == T_F32 && w->dtype == T_F32 && o->numel == in->numel && w->numel >= d_model &&
            vk_graph_rmsnorm_f32(in->data, w->data, o->data, rows, d_model, p_flt(p, "eps", 1e-6f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Softmax") || !strcmp(op, "LogSoftmax")) {
        if (!in || in->ndim <= 0) return 0;
        int axis = p_int(p, "axis", in->ndim - 1);
        if (axis < 0) axis += in->ndim;
        if (axis != in->ndim - 1) return 0;
        materialize_tensor_f32(in);
        int d = in->shape[in->ndim - 1];
        int rows = d > 0 ? (int)(in->numel / d) : 0;
        if (in->dtype != T_F32 || o->numel != in->numel) return 0;
        int ok = (!strcmp(op, "Softmax"))
            ? vk_graph_softmax_f32(in->data, o->data, rows, d)
            : vk_graph_logsoftmax_f32(in->data, o->data, rows, d);
        if (ok) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReduceSum") || !strcmp(op, "ReduceMean")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        int rows = 1, d = (int)x->numel;
        if (x->ndim == 2) { rows = x->shape[0]; d = x->shape[1]; }
        if (x->dtype == T_F32 && o->numel == rows) {
            float inv = !strcmp(op, "ReduceMean") ? 1.0f / (float)d : 1.0f;
            if (vk_graph_reduce_f32(x->data, o->data, rows, d, inv)) {
                *node_backend = "vulkan-graph";
                return 1;
            }
        }
        return 0;
    }

    if (!strcmp(op, "BatchNorm2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias"); T* rm = nin(n, "running_mean"); T* rv = nin(n, "running_var");
        if (!in || !w || !b || !rm || !rv || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); materialize_tensor_f32(b); materialize_tensor_f32(rm); materialize_tensor_f32(rv);
        int c = in->shape[3];
        if (in->dtype == T_F32 && w->dtype == T_F32 && b->dtype == T_F32 && rm->dtype == T_F32 && rv->dtype == T_F32 &&
            w->numel >= c && b->numel >= c && rm->numel >= c && rv->numel >= c &&
            vk_graph_batchnorm2d_f32(in->data, w->data, b->data, rm->data, rv->data, o->data,
                                     in->shape[0], in->shape[1], in->shape[2], c, p_flt(p, "eps", 1e-5f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GlobalAveragePool")) {
        if (!in || in->ndim != 4) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 && o->numel == (long)in->shape[0] * in->shape[3] &&
            vk_graph_global_average_pool_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3])) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Conv1D")) {
        if (try_gpu_graph_conv1d(n, in, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "AveragePool") || !strcmp(op, "AveragePool2D")) {
        if (!in || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in);
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px);
        if (in->dtype == T_F32 &&
            vk_graph_average_pool2d_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                        o->shape[1], o->shape[2], ky, kx, sy, sx, py, px)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Embedding")) {
        T* w = nin(n, "weight");
        if (!in || !w || w->ndim <= 0) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int d_model = w->shape[w->ndim - 1];
        int vocab = d_model > 0 ? (int)(w->numel / d_model) : 0;
        if (in->dtype == T_F32 && w->dtype == T_F32 && o->numel == in->numel * d_model &&
            vk_graph_embedding_f32(in->data, w->data, o->data, (int)in->numel, d_model, vocab)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Resize") || !strcmp(op, "ResizeNearest2D") || !strcmp(op, "UpsampleNearest2D")) {
        if (in && in->dtype == T_F32 && in->ndim == 4 && o->ndim == 4 && in->shape[0] == 1 && o->shape[0] == 1) {
            int mode = (!strcmp(op, "Resize") && strcmp(p_str(p, "mode", "bilinear"), "nearest")) ? 1 : 0;
            if (o->shape[3] == in->shape[3] && vk_graph_resize_f32(in->data, o->data,
                    in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                    o->shape[1], o->shape[2], mode)) {
                *node_backend = "vulkan-graph";
                return 1;
            }
        }
        return 0;
    }

    if (!strcmp(op, "MaxPool2D")) {
        if (!in || in->dtype != T_F32 || in->ndim != 4 || o->ndim != 4 || in->shape[0] != 1 || o->shape[0] != 1) return 0;
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx);
        p_pair(p, "stride", 1, &sy, &sx);
        p_pair(p, "padding", 0, &py, &px);
        if (vk_graph_maxpool2d_f32(in->data, o->data, in->shape[1], in->shape[2], in->shape[3],
                                   o->shape[1], o->shape[2], ky, kx, sy, sx, py, px)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Transpose")) {
        if (!in || in->ndim <= 0 || in->ndim > 8) return 0;
        materialize_tensor_f32(in);
        int perm[8]; for (int i = 0; i < in->ndim; i++) perm[i] = i;
        p_iarr(p, "perm", perm, in->ndim, 0);
        if (in->dtype == T_F32 && o->numel == in->numel &&
            vk_graph_transpose_f32(in->data, o->data, in->shape, perm, in->ndim)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Cast")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        if (x->dtype == T_F32 && x->numel == o->numel && vk_graph_cast_copy_f32(x->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "QuantizeLinear")) {
        if (try_gpu_graph_quantize_linear(n, in, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "DequantizeLinear")) {
        if (try_gpu_graph_dequantize_linear(n, in, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Where") || !strcmp(op, "Mask")) {
        T* cond = nin(n, "cond"); if (!cond) cond = nin(n, "condition");
        T* a = nin(n, "x"); if (!a) a = nin(n, "a");
        T* b = nin(n, "y"); if (!b) b = nin(n, "b");
        if (!cond || !a || !b) return 0;
        materialize_tensor_f32(cond); materialize_tensor_f32(a); materialize_tensor_f32(b);
        if (cond->dtype == T_F32 && a->dtype == T_F32 && b->dtype == T_F32 &&
            cond->numel == o->numel && a->numel == o->numel && b->numel == o->numel &&
            vk_graph_where_f32(cond->data, a->data, b->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Expand") || !strcmp(op, "Broadcast")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        if (x->dtype == T_F32 && vk_graph_expand_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Gather")) {
        T* idx = nin(n, "indices");
        if (!in || !idx || in->ndim <= 0) return 0;
        int axis = p_int(p, "axis", 0); if (axis < 0) axis += in->ndim;
        if (axis != 0) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(idx);
        int row_size = 1; for (int i = 1; i < in->ndim; i++) row_size *= in->shape[i];
        int num_idx = row_size > 0 ? (int)(o->numel / row_size) : 0;
        if (in->dtype == T_F32 && idx->dtype == T_F32 && num_idx > 0 &&
            vk_graph_gather_axis0_f32(in->data, idx->data, o->data, row_size, in->shape[0], num_idx)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Pad")) {
        T* x = in ? in : nin(n, "data");
        if (!x || x->ndim > 4 || o->ndim > 4) return 0;
        materialize_tensor_f32(x);
        int pads[8] = {0}; p_iarr(p, "pads", pads, 8, 0);
        cJSON* pads_node = p ? cJSON_GetObjectItem(p, "pads") : NULL;
        int pads_len = pads_node && cJSON_IsArray(pads_node) ? cJSON_GetArraySize(pads_node) : 0;
        int pt = pads_len == 8 ? pads[1] : pads[0];
        int pl = pads_len == 8 ? pads[2] : pads[1];
        if (pt < 0 || pl < 0) return 0;
        if (x->dtype == T_F32 && vk_graph_pad4d_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim,
                                                    pt, pl, p_flt(p, "value", 0.0f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Slice")) {
        T* x = in ? in : nin(n, "data");
        if (!x || x->ndim > 4 || o->ndim > 4) return 0;
        materialize_tensor_f32(x);
        int starts4[4] = {0, 0, 0, 0}, steps4[4] = {1, 1, 1, 1};
        int starts[8] = {0}, steps[8] = {1,1,1,1,1,1,1,1}, axes[8] = {0,1,2,3,4,5,6,7};
        p_iarr(p, "starts", starts, 8, 0); p_iarr(p, "steps", steps, 8, 1); p_iarr(p, "axes", axes, 8, 0);
        cJSON* axes_node = p ? cJSON_GetObjectItem(p, "axes") : NULL;
        int n_axes = axes_node && cJSON_IsArray(axes_node) ? cJSON_GetArraySize(axes_node) : x->ndim;
        int base = 4 - x->ndim;
        for (int i = 0; i < n_axes && i < 8; i++) {
            int ax = axes_node ? axes[i] : i;
            if (ax < 0) ax += x->ndim;
            if (ax < 0 || ax >= x->ndim || steps[i] <= 0) return 0;
            int padded = base + ax;
            int st = starts[i] < 0 ? starts[i] + x->shape[ax] : starts[i];
            if (st < 0) return 0;
            starts4[padded] = st;
            steps4[padded] = steps[i];
        }
        if (x->dtype == T_F32 && vk_graph_slice4d_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim, starts4, steps4)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ConvTranspose2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); if (b) materialize_tensor_f32(b);
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px);
        if (in->dtype == T_F32 && w->dtype == T_F32 && (!b || b->dtype == T_F32) &&
            vk_graph_conv_transpose2d_f32(in->data, w->data, b ? b->data : NULL, o->data,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                o->shape[1], o->shape[2], o->shape[3], ky, kx, sy, sx, py, px)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SDPA")) {
        if (try_gpu_graph_sdpa(n, in, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossSDPA")) {
        if (try_gpu_graph_cross_sdpa(n, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossAttention")) {
        if (try_gpu_graph_cross_attention(n, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Split")) {
        if (try_gpu_graph_split(n, in, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SpatialSoftargmaxY") || !strcmp(op, "ProfileX") ||
        !strcmp(op, "ProfileY") || !strcmp(op, "MeanHeight")) {
        if (try_gpu_graph_profile_op(n, in, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "NonMaxSuppression")) {
        if (try_gpu_graph_nms(n, o, GPU_GRAPH_VULKAN)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "InterpLinear1D") || !strcmp(op, "Interp1D")) {
        if (!in || in->ndim != 2 || o->ndim != 2) return 0;
        materialize_tensor_f32(in);
        int channels = in->shape[in->ndim - 2];
        int in_l = in->shape[in->ndim - 1];
        int out_l = o->shape[o->ndim - 1];
        if (in->dtype == T_F32 && vk_graph_interp1d_f32(in->data, o->data, channels, in_l, out_l)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Reshape") || !strcmp(op, "Flatten") || !strcmp(op, "Squeeze")
            || !strcmp(op, "Unsqueeze") || !strcmp(op, "Dropout") || !strcmp(op, "Identity")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            vk_graph_alias_f32(in->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph-alias";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Concat")) {
        int axis = p_int(p, "axis", 1);
        if (axis < 0) axis += o->ndim;
        if (!((axis == 0) || (axis == 1 && o->ndim >= 2 && o->shape[0] == 1))) return 0;
        const float* ptrs[MAXIN];
        long sizes[MAXIN];
        long total = 0;
        for (int i = 0; i < n->nin; i++) {
            T* src = t_find(n->ins[i].name);
            if (!src || src->dtype != T_F32) return 0;
            ptrs[i] = src->data;
            sizes[i] = src->numel;
            total += sizes[i];
        }
        int ok = p_int(p, "sigmoid", 0)
            ? vk_graph_concat_sigmoid_flat_f32(ptrs, sizes, n->nin, o->data)
            : vk_graph_concat_flat_f32(ptrs, sizes, n->nin, o->data);
        if (total == o->numel && ok) {
            *node_backend = p_int(p, "sigmoid", 0) ? "vulkan-graph+sigmoid" : "vulkan-graph";
            return 1;
        }
        return 0;
    }

    return 0;
}

static int try_opengl_graph_node(Node* n, int idx, const char** node_backend) {
    if (!g_use_opengl || !n || g_dec_pos >= 0 || g_prefill_len > 0) return 0;
    T* o = t_find(n->out);
    T* in = nin(n, "input");
    cJSON* p = n->params;
    const char* op = n->op;
    if (!o || o->dtype != T_F32) return 0;

    if (!strcmp(op, "QConv2D")) {
        return 0;
    }

    if (!strcmp(op, "Conv2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || in->dtype != T_F32 || (w->dtype != T_F32 && w->dtype != T_F16) ||
            o->ndim != 4 || in->ndim != 4 || w->ndim != 4) return 0;
        int sy, sx, py, px, dy, dx; p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px); p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px}; p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1), relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);
        if (in->shape[0] != 1 || o->shape[0] != 1) return 0;
        if (pads[0] < 0 || pads[1] < 0) return 0;
        T wtmp, btmp;
        T* wp = prepare_conv_weight_image(idx, w, p, in->shape[3], o->shape[3], groups, &wtmp);
        if (!wp) return 0;
        T* bp = b;
        if (b && b->dtype == T_F16) {
            float* bf = f16_tensor_f32_cache(g_f16bcache, idx, b);
            if (!bf) return 0;
            btmp = *b; btmp.data = bf; btmp.dtype = T_F32; btmp.elem_size = sizeof(float);
            bp = &btmp;
        }
        if (b && bp->dtype != T_F32) return 0;
        int kh = wp->shape[0], kw = wp->shape[1];
        if (opengl_graph_conv2d_f32(in->data, o->data, wp->data, bp ? bp->data : NULL,
                                    in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                    o->shape[3], kh, kw, o->shape[1], o->shape[2],
                                    sy, sx, pads[0], pads[1], groups, relu, dy, dx)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Add")) {
        T* a = nin(n, "a"); T* b = nin(n, "b");
        if (!a) a = in;
        const GraphNodeFusion* add3_fusion = graph_opt_node_fusion(idx, GRAPH_FUSION_CHAINED_ELEMENTWISE);
        if (add3_fusion) {
            Node* peer = (add3_fusion->peer_idx >= 0 && add3_fusion->peer_idx < g_nn) ? &g_n[add3_fusion->peer_idx] : NULL;
            T* c = t_find(add3_fusion->aux_tensor);
            if (peer && a && b && c &&
                a->dtype == T_F32 && b->dtype == T_F32 && c->dtype == T_F32 &&
                a->numel == b->numel && a->numel == c->numel && a->numel == o->numel &&
                opengl_graph_add3_relu_f32(a->data, b->data, c->data, o->data, o->numel,
                                           p_int(peer->params, "relu", 0))) {
                *node_backend = "opengl-graph+add3";
                return 1;
            }
            return 0;
        }
        if (a && b && a->dtype == T_F32 && b->dtype == T_F32 &&
            a->numel == b->numel && a->numel == o->numel &&
            opengl_graph_add_relu_f32(a->data, b->data, o->data, o->numel, p_int(p, "relu", 0))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Mul") || !strcmp(op, "Sub") || !strcmp(op, "Div")) {
        if (try_gpu_graph_binary(n, in, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Sigmoid")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            opengl_graph_sigmoid_f32(in->data, o->data, o->numel)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Clip")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            opengl_graph_clip_f32(in->data, o->data, o->numel,
                                  p_flt(p, "min", -1e30f), p_flt(p, "max", 1e30f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReLU") || !strcmp(op, "GELU") || !strcmp(op, "SiLU") || !strcmp(op, "Swish") ||
        !strcmp(op, "Tanh") || !strcmp(op, "HardSwish") || !strcmp(op, "HardSigmoid")) {
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype != T_F32 || in->numel != o->numel) return 0;
        int ok = 0;
        if (!strcmp(op, "ReLU")) ok = opengl_graph_relu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "GELU")) ok = opengl_graph_gelu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "SiLU") || !strcmp(op, "Swish")) ok = opengl_graph_silu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "Tanh")) ok = opengl_graph_tanh_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "HardSwish")) ok = opengl_graph_hardswish_f32(in->data, o->data, o->numel);
        else ok = opengl_graph_hardsigmoid_f32(in->data, o->data, o->numel);
        if (ok) { *node_backend = "opengl-graph"; return 1; }
        return 0;
    }

    if (!strcmp(op, "LeakyReLU")) {
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 && in->numel == o->numel &&
            opengl_graph_leaky_relu_f32(in->data, o->data, o->numel, p_flt(p, "alpha", 0.01f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "PReLU")) {
        T* w = nin(n, "weight");
        if (!in || !w) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int c = in->ndim > 0 ? in->shape[in->ndim - 1] : 1;
        if (in->dtype == T_F32 && w->dtype == T_F32 && in->numel == o->numel && w->numel >= c &&
            opengl_graph_prelu_f32(in->data, w->data, o->data, o->numel, c)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "LayerNorm")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || !b) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); materialize_tensor_f32(b);
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        int rows = d_model > 0 ? (int)(in->numel / d_model) : 0;
        if (in->dtype == T_F32 && w->dtype == T_F32 && b->dtype == T_F32 && o->numel == in->numel &&
            w->numel >= d_model && b->numel >= d_model &&
            opengl_graph_layernorm_f32(in->data, w->data, b->data, o->data, rows, d_model)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "RMSNorm")) {
        T* w = nin(n, "weight");
        if (!in || !w) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        int rows = d_model > 0 ? (int)(in->numel / d_model) : 0;
        if (in->dtype == T_F32 && w->dtype == T_F32 && o->numel == in->numel && w->numel >= d_model &&
            opengl_graph_rmsnorm_f32(in->data, w->data, o->data, rows, d_model, p_flt(p, "eps", 1e-6f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Softmax") || !strcmp(op, "LogSoftmax")) {
        if (!in || in->ndim <= 0) return 0;
        int axis = p_int(p, "axis", in->ndim - 1);
        if (axis < 0) axis += in->ndim;
        if (axis != in->ndim - 1) return 0;
        materialize_tensor_f32(in);
        int d = in->shape[in->ndim - 1];
        int rows = d > 0 ? (int)(in->numel / d) : 0;
        if (in->dtype != T_F32 || o->numel != in->numel) return 0;
        int ok = (!strcmp(op, "Softmax"))
            ? opengl_graph_softmax_f32(in->data, o->data, rows, d)
            : opengl_graph_logsoftmax_f32(in->data, o->data, rows, d);
        if (ok) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReduceSum") || !strcmp(op, "ReduceMean")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        int rows = 1, d = (int)x->numel;
        if (x->ndim == 2) { rows = x->shape[0]; d = x->shape[1]; }
        if (x->dtype == T_F32 && o->numel == rows) {
            float inv = !strcmp(op, "ReduceMean") ? 1.0f / (float)d : 1.0f;
            if (opengl_graph_reduce_f32(x->data, o->data, rows, d, inv)) {
                *node_backend = "opengl-graph";
                return 1;
            }
        }
        return 0;
    }

    if (!strcmp(op, "BatchNorm2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias"); T* rm = nin(n, "running_mean"); T* rv = nin(n, "running_var");
        if (!in || !w || !b || !rm || !rv || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); materialize_tensor_f32(b); materialize_tensor_f32(rm); materialize_tensor_f32(rv);
        int c = in->shape[3];
        if (in->dtype == T_F32 && w->dtype == T_F32 && b->dtype == T_F32 && rm->dtype == T_F32 && rv->dtype == T_F32 &&
            w->numel >= c && b->numel >= c && rm->numel >= c && rv->numel >= c &&
            opengl_graph_batchnorm2d_f32(in->data, w->data, b->data, rm->data, rv->data, o->data,
                                         in->shape[0], in->shape[1], in->shape[2], c, p_flt(p, "eps", 1e-5f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GlobalAveragePool")) {
        if (!in || in->ndim != 4) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 && o->numel == (long)in->shape[0] * in->shape[3] &&
            opengl_graph_global_average_pool_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3])) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Conv1D")) {
        if (try_gpu_graph_conv1d(n, in, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "AveragePool") || !strcmp(op, "AveragePool2D")) {
        if (!in || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in);
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px);
        if (in->dtype == T_F32 &&
            opengl_graph_average_pool2d_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                            o->shape[1], o->shape[2], ky, kx, sy, sx, py, px)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Embedding")) {
        T* w = nin(n, "weight");
        if (!in || !w || w->ndim <= 0) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int d_model = w->shape[w->ndim - 1];
        int vocab = d_model > 0 ? (int)(w->numel / d_model) : 0;
        if (in->dtype == T_F32 && w->dtype == T_F32 && o->numel == in->numel * d_model &&
            opengl_graph_embedding_f32(in->data, w->data, o->data, (int)in->numel, d_model, vocab)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Resize") || !strcmp(op, "ResizeNearest2D") || !strcmp(op, "UpsampleNearest2D")) {
        if (in && in->dtype == T_F32 && in->ndim == 4 && o->ndim == 4 && in->shape[0] == 1 && o->shape[0] == 1) {
            int mode = (!strcmp(op, "Resize") && strcmp(p_str(p, "mode", "bilinear"), "nearest")) ? 1 : 0;
            if (o->shape[3] == in->shape[3] && opengl_graph_resize_f32(in->data, o->data,
                    in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                    o->shape[1], o->shape[2], mode)) {
                *node_backend = "opengl-graph";
                return 1;
            }
        }
        return 0;
    }

    if (!strcmp(op, "MaxPool2D")) {
        if (!in || in->dtype != T_F32 || in->ndim != 4 || o->ndim != 4 || in->shape[0] != 1 || o->shape[0] != 1) return 0;
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx);
        p_pair(p, "stride", 1, &sy, &sx);
        p_pair(p, "padding", 0, &py, &px);
        if (opengl_graph_maxpool2d_f32(in->data, o->data, in->shape[1], in->shape[2], in->shape[3],
                                       o->shape[1], o->shape[2], ky, kx, sy, sx, py, px)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Transpose")) {
        if (!in || in->ndim <= 0 || in->ndim > 8) return 0;
        materialize_tensor_f32(in);
        int perm[8]; for (int i = 0; i < in->ndim; i++) perm[i] = i;
        p_iarr(p, "perm", perm, in->ndim, 0);
        if (in->dtype == T_F32 && o->numel == in->numel &&
            opengl_graph_transpose_f32(in->data, o->data, in->shape, perm, in->ndim)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Cast")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        if (x->dtype == T_F32 && x->numel == o->numel && opengl_graph_cast_copy_f32(x->data, o->data, o->numel)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "QuantizeLinear")) {
        if (try_gpu_graph_quantize_linear(n, in, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "DequantizeLinear")) {
        if (try_gpu_graph_dequantize_linear(n, in, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Where") || !strcmp(op, "Mask")) {
        T* cond = nin(n, "cond"); if (!cond) cond = nin(n, "condition");
        T* a = nin(n, "x"); if (!a) a = nin(n, "a");
        T* b = nin(n, "y"); if (!b) b = nin(n, "b");
        if (!cond || !a || !b) return 0;
        materialize_tensor_f32(cond); materialize_tensor_f32(a); materialize_tensor_f32(b);
        if (cond->dtype == T_F32 && a->dtype == T_F32 && b->dtype == T_F32 &&
            cond->numel == o->numel && a->numel == o->numel && b->numel == o->numel &&
            opengl_graph_where_f32(cond->data, a->data, b->data, o->data, o->numel)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Expand") || !strcmp(op, "Broadcast")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        if (x->dtype == T_F32 && opengl_graph_expand_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Gather")) {
        T* idx = nin(n, "indices");
        if (!in || !idx || in->ndim <= 0) return 0;
        int axis = p_int(p, "axis", 0); if (axis < 0) axis += in->ndim;
        if (axis != 0) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(idx);
        int row_size = 1; for (int i = 1; i < in->ndim; i++) row_size *= in->shape[i];
        int num_idx = row_size > 0 ? (int)(o->numel / row_size) : 0;
        if (in->dtype == T_F32 && idx->dtype == T_F32 && num_idx > 0 &&
            opengl_graph_gather_axis0_f32(in->data, idx->data, o->data, row_size, in->shape[0], num_idx)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Pad")) {
        T* x = in ? in : nin(n, "data");
        if (!x || x->ndim > 4 || o->ndim > 4) return 0;
        materialize_tensor_f32(x);
        int pads[8] = {0}; p_iarr(p, "pads", pads, 8, 0);
        cJSON* pads_node = p ? cJSON_GetObjectItem(p, "pads") : NULL;
        int pads_len = pads_node && cJSON_IsArray(pads_node) ? cJSON_GetArraySize(pads_node) : 0;
        int pt = pads_len == 8 ? pads[1] : pads[0];
        int pl = pads_len == 8 ? pads[2] : pads[1];
        if (pt < 0 || pl < 0) return 0;
        if (x->dtype == T_F32 && opengl_graph_pad4d_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim,
                                                        pt, pl, p_flt(p, "value", 0.0f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Slice")) {
        T* x = in ? in : nin(n, "data");
        if (!x || x->ndim > 4 || o->ndim > 4) return 0;
        materialize_tensor_f32(x);
        int starts4[4] = {0, 0, 0, 0}, steps4[4] = {1, 1, 1, 1};
        int starts[8] = {0}, steps[8] = {1,1,1,1,1,1,1,1}, axes[8] = {0,1,2,3,4,5,6,7};
        p_iarr(p, "starts", starts, 8, 0); p_iarr(p, "steps", steps, 8, 1); p_iarr(p, "axes", axes, 8, 0);
        cJSON* axes_node = p ? cJSON_GetObjectItem(p, "axes") : NULL;
        int n_axes = axes_node && cJSON_IsArray(axes_node) ? cJSON_GetArraySize(axes_node) : x->ndim;
        int base = 4 - x->ndim;
        for (int i = 0; i < n_axes && i < 8; i++) {
            int ax = axes_node ? axes[i] : i;
            if (ax < 0) ax += x->ndim;
            if (ax < 0 || ax >= x->ndim || steps[i] <= 0) return 0;
            int padded = base + ax;
            int st = starts[i] < 0 ? starts[i] + x->shape[ax] : starts[i];
            if (st < 0) return 0;
            starts4[padded] = st;
            steps4[padded] = steps[i];
        }
        if (x->dtype == T_F32 && opengl_graph_slice4d_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim, starts4, steps4)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ConvTranspose2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); if (b) materialize_tensor_f32(b);
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px);
        if (in->dtype == T_F32 && w->dtype == T_F32 && (!b || b->dtype == T_F32) &&
            opengl_graph_conv_transpose2d_f32(in->data, w->data, b ? b->data : NULL, o->data,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                o->shape[1], o->shape[2], o->shape[3], ky, kx, sy, sx, py, px)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SDPA")) {
        if (try_gpu_graph_sdpa(n, in, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossSDPA")) {
        if (try_gpu_graph_cross_sdpa(n, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossAttention")) {
        if (try_gpu_graph_cross_attention(n, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Split")) {
        if (try_gpu_graph_split(n, in, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SpatialSoftargmaxY") || !strcmp(op, "ProfileX") ||
        !strcmp(op, "ProfileY") || !strcmp(op, "MeanHeight")) {
        if (try_gpu_graph_profile_op(n, in, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "NonMaxSuppression")) {
        if (try_gpu_graph_nms(n, o, GPU_GRAPH_OPENGL)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "InterpLinear1D") || !strcmp(op, "Interp1D")) {
        if (!in || in->ndim != 2 || o->ndim != 2) return 0;
        materialize_tensor_f32(in);
        int channels = in->shape[in->ndim - 2];
        int in_l = in->shape[in->ndim - 1];
        int out_l = o->shape[o->ndim - 1];
        if (in->dtype == T_F32 && opengl_graph_interp1d_f32(in->data, o->data, channels, in_l, out_l)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Reshape") || !strcmp(op, "Flatten") || !strcmp(op, "Squeeze")
            || !strcmp(op, "Unsqueeze") || !strcmp(op, "Dropout") || !strcmp(op, "Identity")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            opengl_graph_alias_f32(in->data, o->data, o->numel)) {
            *node_backend = "opengl-graph-alias";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Concat")) {
        int axis = p_int(p, "axis", 1);
        if (axis < 0) axis += o->ndim;
        if (!((axis == 0) || (axis == 1 && o->ndim >= 2 && o->shape[0] == 1))) return 0;
        const float* ptrs[MAXIN];
        long sizes[MAXIN];
        long total = 0;
        for (int i = 0; i < n->nin; i++) {
            T* src = t_find(n->ins[i].name);
            if (!src || src->dtype != T_F32) return 0;
            ptrs[i] = src->data;
            sizes[i] = src->numel;
            total += sizes[i];
        }
        int ok = p_int(p, "sigmoid", 0)
            ? opengl_graph_concat_sigmoid_flat_f32(ptrs, sizes, n->nin, o->data)
            : opengl_graph_concat_flat_f32(ptrs, sizes, n->nin, o->data);
        if (total == o->numel && ok) {
            *node_backend = p_int(p, "sigmoid", 0) ? "opengl-graph+sigmoid" : "opengl-graph";
            return 1;
        }
        return 0;
    }

    return 0;
}

#ifdef __APPLE__
static int try_metal_graph_node(Node* n, int idx, const char** node_backend) {
    (void)idx;
    if (!g_use_metal || !n || g_dec_pos >= 0 || g_prefill_len > 0) return 0;
    T* o = t_find(n->out);
    T* in = nin(n, "input");
    const char* op = n->op;
    if (!o || o->dtype != T_F32) return 0;

    if (!strcmp(op, "Mul") || !strcmp(op, "Sub") || !strcmp(op, "Div")) {
        if (try_gpu_graph_binary(n, in, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Conv1D")) {
        if (try_gpu_graph_conv1d(n, in, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "QuantizeLinear")) {
        if (try_gpu_graph_quantize_linear(n, in, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "DequantizeLinear")) {
        if (try_gpu_graph_dequantize_linear(n, in, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SDPA")) {
        if (try_gpu_graph_sdpa(n, in, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossSDPA")) {
        if (try_gpu_graph_cross_sdpa(n, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossAttention")) {
        if (try_gpu_graph_cross_attention(n, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Split")) {
        if (try_gpu_graph_split(n, in, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SpatialSoftargmaxY") || !strcmp(op, "ProfileX") ||
        !strcmp(op, "ProfileY") || !strcmp(op, "MeanHeight")) {
        if (try_gpu_graph_profile_op(n, in, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "NonMaxSuppression")) {
        if (try_gpu_graph_nms(n, o, GPU_GRAPH_METAL)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    return 0;
}
#endif

// relu: 0 none, 1 ReLU, 2 ReLU6 (clamp to [0,6]).
static inline float relu6_apply(float x, int relu) {
    if (!relu) return x;
    if (x < 0.0f) x = 0.0f;
    if (relu >= 2 && x > 6.0f) x = 6.0f;
    return x;
}

// ---- per-op timing aggregation (--debug) ----------------------------------
typedef struct { char op[40]; double ms; int count; } ProfEntry;
static ProfEntry g_prof[128];
static int g_nprof = 0;
void prof_reset(void) { g_nprof = 0; }
static void prof_add(const char* op, double ms) {
    for (int i = 0; i < g_nprof; i++) if (!strcmp(g_prof[i].op, op)) { g_prof[i].ms += ms; g_prof[i].count++; return; }
    if (g_nprof < (int)(sizeof(g_prof) / sizeof(g_prof[0]))) {
        strncpy(g_prof[g_nprof].op, op, 39); g_prof[g_nprof].op[39] = 0;
        g_prof[g_nprof].ms = ms; g_prof[g_nprof].count = 1; g_nprof++;
    }
}
void prof_add_entry(const char* op, double ms) { prof_add(op, ms); }
void prof_report(void) {
    double total = 0.0;
    for (int i = 0; i < g_nprof; i++) {  // selection sort by total time, descending
        int mx = i;
        for (int j = i + 1; j < g_nprof; j++) if (g_prof[j].ms > g_prof[mx].ms) mx = j;
        if (mx != i) { ProfEntry t = g_prof[i]; g_prof[i] = g_prof[mx]; g_prof[mx] = t; }
        total += g_prof[i].ms;
    }
    fprintf(stderr, "[debug] --- op time summary (by total) ---\n");
    for (int i = 0; i < g_nprof; i++)
        fprintf(stderr, "[debug]   %-16s %8.2f ms  (n=%d)\n", g_prof[i].op, g_prof[i].ms, g_prof[i].count);
    fprintf(stderr, "[debug]   %-16s %8.2f ms\n", "TOTAL", total);
}

// ---- run one node ---------------------------------------------------------
int run_node(Node* n, int idx, int is_last) {
    if (n->skip) return 0;  // node fused/aliased by graph_optimize_operator_fusion()
    cJSON* p = n->params;
    T* o = t_find(n->out);
    T* in = nin(n, "input");
    const char* op = n->op;
    int log_node = g_debug;
    double node_t0 = log_node ? engine_now_ms() : 0.0;
    const char* node_backend = "cpu";

    // Row window for elementwise/shape-preserving ops (in and out share layout).
    int e_D = o ? (o->ndim ? o->shape[o->ndim - 1] : (int)o->numel) : 1;
    long e_total = (o && e_D) ? o->numel / e_D : 1;
    int e_r0, e_rc; seq_range(e_total, &e_r0, &e_rc);
    long e_off = (long)e_r0 * e_D; int e_cnt = e_rc * e_D;

    int ran_gpu_graph = 0;
    if (try_vulkan_graph_node(n, idx, &node_backend)) {
        ran_gpu_graph = 1;
        goto done;
    }
    if (try_opengl_graph_node(n, idx, &node_backend)) {
        ran_gpu_graph = 1;
        goto done;
    }
#ifdef __APPLE__
    if (try_metal_graph_node(n, idx, &node_backend)) {
        ran_gpu_graph = 1;
        goto done;
    }
#endif
    vk_sync_node_inputs(n);

    if (!strcmp(op, "QuantizeLinear")) {
        int qdone = q_quantize_linear(n, in, o);
        if (qdone < 0) return -1;
        if (!qdone) return -1;
        node_backend = "cpu-quant";
    } else if (!strcmp(op, "DequantizeLinear")) {
        materialize_tensor_f32(in);
        if (in != o && in && o && in->numel == o->numel) memcpy(o->data, in->data, (size_t)o->numel * sizeof(float));
    } else if (!strcmp(op, "QConv2D")) {
        T* w = nin(n, "weight"); T* ws = nin(n, "weight_scale"); T* wzp = nin(n, "weight_zero_point"); T* b = nin(n, "bias");
        int sy, sx, py, px, dy, dx; p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px); p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px}; p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1);
        int relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);   // 2 = ReLU6, fused from a following Clip(0,6)
        int qdone = qconv2d_quantized(idx, in, o, w, ws, wzp, b, sy, sx, pads, groups, dy, dx, relu,
                                       p_flt(p, "input_scale", 1.0f), p_int(p, "input_zero_point", 0),
                                       p_str(p, "weight_layout", groups == 1 ? "OIHW" : "OIHW"), &node_backend);
        if (qdone < 0) return -1;
        if (!qdone) {
            materialize_tensor_f32(in);
            if (qconv2d_cached_f32(idx, in, o, w, ws, wzp, b, sy, sx, pads, groups, dy, dx, relu,
                                   p, p_str(p, "weight_layout", groups == 1 ? "OIHW" : "OIHW")) != 0) return -1;
            node_backend = "cpu-f32cache";
        }
    } else if (!strcmp(op, "Conv2D")) {
        materialize_tensor_f32(in);
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        int sy, sx, py, px, dy, dx; p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px); p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px}; p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1), relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);
        if (!w) return -1;
            T* bp = b;
            T* add = NULL;
            const GraphNodeFusion* conv_add_fusion = graph_opt_node_fusion(idx, GRAPH_FUSION_CONV_ADD);
            if (conv_add_fusion) {
                add = t_find(conv_add_fusion->aux_tensor);
                if (!add || add->dtype != T_F32 || add->numel != o->numel || add->data == o->data) return -1;
                materialize_tensor_f32(add);
            }
            T wtmp, btmp;
            T* wp = prepare_conv_weight_image(idx, w, p, in->shape[3], o->shape[3], groups, &wtmp);
            if (!wp) return -1;
            if (b && b->dtype == T_F16) {
                float* bf = f16_tensor_f32_cache(g_f16bcache, idx, b);
                if (!bf) return -1;
                btmp = *b; btmp.data = bf; btmp.dtype = T_F32; btmp.elem_size = sizeof(float);
                bp = &btmp;
            }
            const GraphNodeFusion* dw_pw_fusion = graph_opt_node_fusion(idx, GRAPH_FUSION_DEPTHWISE_POINTWISE);
            int pw_idx = dw_pw_fusion ? dw_pw_fusion->peer_idx : -1;
            if (pw_idx >= 0) {
                Node* pw = &g_n[pw_idx];
                T* po = t_find(pw->out);
                T* pw_w = nin(pw, "weight");
                T* pw_b = nin(pw, "bias");
                if (!po || !pw_w) return -1;
                T pw_wtmp, pw_btmp;
                T* pw_wp = prepare_conv_weight_image(pw_idx, pw_w, pw->params, o->shape[3], po->shape[3], 1, &pw_wtmp);
                if (!pw_wp) return -1;
                T* pw_bp = pw_b;
                if (pw_b && pw_b->dtype == T_F16) {
                    float* pbf = f16_tensor_f32_cache(g_f16bcache, pw_idx, pw_b);
                    if (!pbf) return -1;
                    pw_btmp = *pw_b; pw_btmp.data = pbf; pw_btmp.dtype = T_F32; pw_btmp.elem_size = sizeof(float);
                    pw_bp = &pw_btmp;
                }
                T* pw_add = NULL;
                const GraphNodeFusion* pw_add_fusion = graph_opt_node_fusion(pw_idx, GRAPH_FUSION_CONV_ADD);
                if (pw_add_fusion) {
                    pw_add = t_find(pw_add_fusion->aux_tensor);
                    if (!pw_add || pw_add->dtype != T_F32 || pw_add->numel != po->numel || pw_add->data == po->data) return -1;
                    materialize_tensor_f32(pw_add);
                }
                int pw_relu = pw->fuse_relu6 ? 2 : p_int(pw->params, "relu", 0);
                if (!conv2d_depthwise_pointwise_f32_image(idx, pw_idx, in, po, wp, bp, pw_wp, pw_bp, pw_add,
                                                          sy, sx, pads, relu, pw_relu)) return -1;
                o = po;
                node_backend = pw_add ? "cpu-conv+dw-pw+add" : "cpu-conv+dw-pw";
                goto done;
            }
            if (wp != w) {
                node_backend = add ? (w->dtype == T_F16 ? "cpu-f16w-pack+add" : "cpu-pack+add")
                                   : (w->dtype == T_F16 ? "cpu-f16w-pack" : "cpu-pack");
            } else {
                node_backend = add ? "cpu-conv+add" : "cpu-conv";
            }
            conv2d_f32_image(idx, in, o, wp, bp, add, sy, sx, pads, groups, dy, dx, relu);
    } else if (!strcmp(op, "MatMul") || !strcmp(op, "Gemm") || !strcmp(op, "Linear")) {
        materialize_tensor_f32(in);
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        int d_in = in->shape[in->ndim - 1], d_out = o->shape[o->ndim - 1];
        long total = in->numel / d_in;
        int r0, rc; seq_range(total, &r0, &rc);
        // In prefill only the last position's logits are read, so the final projection
        // (the wide lm_head) needs just that one row.
        if (is_last && g_dec_pos < 0 && g_last_token >= 0 && total > 1) { r0 = g_last_token; rc = 1; }
        const float* in_p = in->data + (long)r0 * d_in;
        float* out_p = o->data + (long)r0 * d_out;
        // GPU only pays off for large matmuls: a tiny 1-row decode matmul costs more in
        // dispatch/sync than it saves, so offload only when the work clears a threshold
        // (e.g. the wide lm_head). This cuts GPU round-trips from ~33/token to ~1/token.
        long cost = (long)rc * d_in * d_out;
#ifdef USE_NNAPI
        if (g_use_nnapi && cost >= (1L << 20)) {
            node_backend = "nnapi";
            nnapi_matmul(in_p, w->data, b ? b->data : NULL, out_p, rc, d_in, d_out);
        } else
#endif
        if (g_use_vulkan && cost >= (1L << 20)) {
            node_backend = "vulkan";
            vk_matmul(in_p, w->data, b ? b->data : NULL, out_p, rc, d_in, d_out);
        } else if (g_use_opengl && cost >= (1L << 20)) {
            node_backend = "opengl";
            opengl_matmul(in_p, w->data, b ? b->data : NULL, out_p, rc, d_in, d_out);
        } else {
            matmul_f32(in_p, w->data, b ? b->data : NULL, out_p, rc, d_in, d_out);
        }
    } else if (!strcmp(op, "BatchNorm2D")) {
        materialize_tensor_f32(in);
        T* w = nin(n, "weight"); T* b = nin(n, "bias"); T* rm = nin(n, "running_mean"); T* rv = nin(n, "running_var");
        batch_norm2d_f32(in->data, w->data, b->data, rm->data, rv->data, o->data,
                         in->shape[0], in->shape[1], in->shape[2], in->shape[3], p_flt(p, "eps", 1e-5f));
    } else if (!strcmp(op, "MaxPool2D")) {
        int ky, kx, sy, sx, py, px; p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px);
        int qdone = q_maxpool2d_quantized(in, o, ky, kx, sy, sx, py, px);
        if (qdone < 0) return -1;
        if (qdone) node_backend = "cpu-qmaxpool";
        else {
            materialize_tensor_f32(in);
            maxpool2d_f32_image(in, o, ky, kx, sy, sx, py, px);
            node_backend = "cpu-maxpool";
        }
    } else if (!strcmp(op, "GlobalAveragePool")) {
        materialize_tensor_f32(in);
        global_average_pool_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3]);
    } else if (!strcmp(op, "Add"))  {
        int qdone = q_add_quantized(n, o);
        if (qdone < 0) return -1;
        if (qdone) node_backend = "cpu-qadd";
        else { materialize_node_inputs(n); binop(n, 0); }
    } else if (!strcmp(op, "Mul"))  { materialize_node_inputs(n); binop(n, 1);
    } else if (!strcmp(op, "ReLU")) { materialize_tensor_f32(in); relu_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "Sigmoid"))     { materialize_tensor_f32(in); sigmoid_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "GELU"))        { materialize_tensor_f32(in); gelu_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "SiLU") || !strcmp(op, "Swish")) { materialize_tensor_f32(in); silu_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "HardSwish"))   { materialize_tensor_f32(in); hardswish_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "HardSigmoid")) { materialize_tensor_f32(in); hardsigmoid_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "Clip"))        { materialize_tensor_f32(in); clip_f32(in->data + e_off, o->data + e_off, e_cnt, p_flt(p, "min", -1e30f), p_flt(p, "max", 1e30f));
    } else if (!strcmp(op, "Transpose")) {
        materialize_tensor_f32(in);
        int perm[8]; for (int i = 0; i < in->ndim; i++) perm[i] = i;
        p_iarr(p, "perm", perm, in->ndim, 0);
        transpose_tensor(in, o, perm);
    } else if (!strcmp(op, "Concat")) {
        materialize_node_inputs(n);
        int fused_sigmoid = idx >= 0 && idx < MAXN ? g_concat_sigmoid_fuse[idx] : -1;
        if (fused_sigmoid >= 0) {
            T* so = t_find(g_n[fused_sigmoid].out);
            if (!so) return -1;
            concat_tensor_sigmoid(n, so, p_int(p, "axis", 1));
            node_backend = "cpu+sigmoid";
        } else if (p_int(p, "sigmoid", 0)) {
            concat_tensor_sigmoid(n, o, p_int(p, "axis", 1));
            node_backend = "cpu+sigmoid";
        } else {
            concat_tensor(n, o, p_int(p, "axis", 1));
        }
    } else if (!strcmp(op, "ResizeNearest2D") || !strcmp(op, "UpsampleNearest2D")) {
        int qdone = q_resize_nearest2d_quantized(in, o);
        if (qdone < 0) return -1;
        if (qdone) node_backend = "cpu-qresize";
        else {
            materialize_tensor_f32(in);
            resize_nearest2d_image(in, o);
            node_backend = "cpu-resize";
        }
    } else if (!strcmp(op, "Reshape") || !strcmp(op, "Flatten") || !strcmp(op, "Squeeze")
            || !strcmp(op, "Unsqueeze") || !strcmp(op, "Dropout") || !strcmp(op, "Identity")) {
        materialize_tensor_f32(in);
        if (e_off == 0 && e_cnt == o->numel) memcpy(o->data, in->data, (size_t)o->numel * sizeof(float));
        else copy_f32(in->data + e_off, o->data + e_off, e_cnt);
    } else if (!strcmp(op, "LayerNorm")) {
        materialize_tensor_f32(in);
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        long total = in->numel / d_model; int r0, rc; seq_range(total, &r0, &rc);
        layernorm_f32(in->data + (long)r0 * d_model, w->data, b ? b->data : NULL, o->data + (long)r0 * d_model, rc, d_model, p_flt(p, "eps", 1e-6f));
    } else if (!strcmp(op, "RMSNorm")) {
        materialize_tensor_f32(in);
        T* w = nin(n, "weight");
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        long total = in->numel / d_model; int r0, rc; seq_range(total, &r0, &rc);
        rmsnorm_f32(in->data + (long)r0 * d_model, w->data, o->data + (long)r0 * d_model, rc, d_model, p_flt(p, "eps", 1e-6f));
    } else if (!strcmp(op, "Softmax")) {
        materialize_tensor_f32(in);
        int d = in->shape[in->ndim - 1];
        int b = (int)(in->numel / d);
        softmax_f32(in->data, o->data, b, d);
    } else if (!strcmp(op, "Embedding")) {
        materialize_tensor_f32(in);
        T* w = nin(n, "weight");
        int d_model = w->shape[w->ndim - 1];
        long total = in->numel; int r0, rc; seq_range(total, &r0, &rc);
        embedding_f32((uintptr_t)(in->data + r0), (uintptr_t)w->data, (uintptr_t)(o->data + (long)r0 * d_model), rc, d_model);
    } else if (!strcmp(op, "SDPA")) {
        materialize_tensor_f32(in);
        T* qkv = nin(n, "qkv");
        int num_heads = p_int(p, "heads", 8);
        int d_model = qkv->shape[qkv->ndim - 1] / 3;
        int max_seq = (int)(qkv->numel / (3 * d_model));
        int head_dim = d_model / num_heads;
        float scale = p_flt(p, "scale", 1.0f / sqrtf((float)head_dim));
        if (!g_kcache[idx]) { g_kcache[idx] = (float*)malloc((size_t)max_seq * d_model * 4); g_vcache[idx] = (float*)malloc((size_t)max_seq * d_model * 4); }
        if (g_dec_pos < 0) {
            int r0, rc; seq_range(max_seq, &r0, &rc);  // r0==0
            sdpa_f32(qkv->data, o->data, rc, d_model, num_heads, head_dim, scale);
            for (int pos = 0; pos < rc; pos++) {       // fill cache [0..rc)
                const float* src = qkv->data + (long)pos * 3 * d_model;
                memcpy(g_kcache[idx] + (long)pos * d_model, src + d_model, d_model * 4);
                memcpy(g_vcache[idx] + (long)pos * d_model, src + 2 * d_model, d_model * 4);
            }
        } else {
            int pos = g_dec_pos;
            const float* src = qkv->data + (long)pos * 3 * d_model;   // qkv row of the new token
            memcpy(g_kcache[idx] + (long)pos * d_model, src + d_model, d_model * 4);
            memcpy(g_vcache[idx] + (long)pos * d_model, src + 2 * d_model, d_model * 4);
            sdpa_decode(src, g_kcache[idx], g_vcache[idx], pos + 1, o->data + (long)pos * d_model, d_model, num_heads, head_dim, scale);
        }
    } else if (!strcmp(op, "CrossSDPA")) {
        materialize_tensor_f32(in);
        T* q = nin(n, "q"); T* k = nin(n, "k"); T* v = nin(n, "v");
        int num_heads = p_int(p, "heads", 8);
        int d_model = q->shape[q->ndim - 1];
        int seq_q = (int)(q->numel / d_model);
        int seq_kv = (int)(k->numel / d_model);
        int head_dim = d_model / num_heads;
        float scale = p_flt(p, "scale", 1.0f / sqrtf((float)head_dim));
        cross_sdpa_f32(q->data, k->data, v->data, o->data, seq_q, seq_kv, d_model, num_heads, head_dim, scale);
    } else {
        fprintf(stderr, "[engine] unsupported op '%s' — node skipped\n", op);
        return -1;
    }
done:
    if (!ran_gpu_graph) vk_mark_host_tensor(o);
    if (log_node) {
        double el = engine_now_ms() - node_t0;
        fprintf(stderr, "[debug] node=%03d op=%s backend=%s out=%s numel=%ld %.3f ms\n",
                idx, op, node_backend, n->out, o ? o->numel : 0L, el);
        prof_add(op, el);
    }
    return 0;
}
