#include "engine.h"
#include "engine_internal.h"
#include "conv_f32_opt.h"
#include "vulkan_engine.h"
#include "opengl_engine.h"
#ifdef __APPLE__
#include "metal_engine.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

T g_t[MAXT];
int g_nt = 0;
Node g_n[MAXN];
int g_nn = 0;
QTensor g_qt[MAXT];
char* g_blob = NULL;
cJSON* g_cfg_root = NULL;
char g_first_input[128] = {0};
int g_loaded = 0;
int g_dec_pos = -1;
int g_prefill_len = 0;
float* g_kcache[MAXN];
float* g_vcache[MAXN];
float* g_qwcache[MAXN];
float* g_f16wcache[MAXN];
float* g_f16bcache[MAXN];
float* g_conv_wcache[MAXN];
int16_t* g_qiwcache[MAXN];
int16_t* g_qpwcache[MAXN];
int16_t* g_qpwilcache[MAXN];
signed char* g_qpw8zcache[MAXN];
int16_t* g_qdwcache[MAXN];
int g_concat_sigmoid_fuse[MAXN];

int engine_init(const char* config_path, const char* weights_path) {
    if (g_loaded) engine_free_ctx();
    g_nt = 0;
    g_nn = 0;
    g_first_input[0] = 0;
    g_dec_pos = -1;
    g_prefill_len = 0;
    if (g_use_vulkan) vk_graph_reset();
    if (g_use_opengl) opengl_graph_reset();
#ifdef __APPLE__
    if (g_use_metal) metal_graph_reset();
#endif
    if (load_weights(weights_path) != 0) return -1;
    if (build_graph(config_path) != 0) return -1;
    if (prepack_qconv_weights() != 0) return -1;
    if (prepack_conv_weights() != 0) return -1;
    g_loaded = 1;
    return 0;
}

float* engine_input_ptr(const char* name, long* numel) {
    T* t = t_find(name && name[0] ? name : g_first_input);
    if (!t) return NULL;
    if (numel) *numel = t->numel;
    return t->data;
}

float* engine_tensor_ptr(const char* name, long* numel) {
    T* t = t_find(name && name[0] ? name : g_first_input);
    if (!t) return NULL;
    vk_sync_host_tensor(t);
    materialize_tensor_f32(t);
    if (numel) *numel = t->numel;
    return t->data;
}

int engine_tensor_info(const char* name, long* numel, int* shape, int* ndim) {
    T* t = t_find(name && name[0] ? name : g_first_input);
    if (!t) return -1;
    if (numel) *numel = t->numel;
    if (ndim) *ndim = t->ndim;
    if (shape) {
        for (int i = 0; i < t->ndim; i++) shape[i] = t->shape[i];
    }
    return 0;
}

int engine_forward(void) {
    if (!g_loaded || g_nn == 0) return -1;
    g_dec_pos = -1;
    g_prefill_len = 0;
    qt_invalidate_all();
    vk_mark_owned_tensors_host_dirty();
    if (g_debug) prof_reset();
    double t0 = g_debug ? engine_now_ms() : 0.0;
    for (int i = 0; i < g_nn; i++) {
        if (run_node(&g_n[i], i, i == g_nn - 1) != 0) return -1;
    }
    if (g_use_vulkan) {
        double wait_t0 = g_debug ? engine_now_ms() : 0.0;
        if (vk_graph_end_forward() != 0) return -1;
        if (g_debug) prof_add_entry("GPUWait", engine_now_ms() - wait_t0);
    }
    if (g_use_opengl) {
        double wait_t0 = g_debug ? engine_now_ms() : 0.0;
        if (opengl_graph_end_forward() != 0) return -1;
        if (g_debug) prof_add_entry("GPUWait", engine_now_ms() - wait_t0);
    }
#ifdef __APPLE__
    if (g_use_metal) {
        double wait_t0 = g_debug ? engine_now_ms() : 0.0;
        if (metal_graph_end_forward() != 0) return -1;
        if (g_debug) prof_add_entry("GPUWait", engine_now_ms() - wait_t0);
    }
#endif
    if (g_debug) {
        fprintf(stderr, "[debug] engine_forward nodes=%d %.3f ms\n", g_nn, engine_now_ms() - t0);
        prof_report();
    }
    return 0;
}

void engine_profile_reset(void) {
    prof_reset();
}

void engine_profile_report(void) {
    prof_report();
}

int engine_prefill(int n_tokens) {
    if (!g_loaded || g_nn == 0) return -1;
    g_dec_pos = -1;
    g_prefill_len = n_tokens;
    g_last_token = n_tokens - 1;
    qt_invalidate_all();
    double t0 = g_debug ? engine_now_ms() : 0.0;
    for (int i = 0; i < g_nn; i++) {
        if (run_node(&g_n[i], i, i == g_nn - 1) != 0) return -1;
    }
    g_prefill_len = 0;
    if (g_debug) {
        fprintf(stderr, "[debug] engine_prefill tokens=%d nodes=%d %.3f ms\n",
                n_tokens, g_nn, engine_now_ms() - t0);
    }
    return 0;
}

int engine_decode(int pos) {
    if (!g_loaded || g_nn == 0) return -1;
    g_dec_pos = pos;
    g_last_token = pos;
    qt_invalidate_all();
    double t0 = g_debug ? engine_now_ms() : 0.0;
    for (int i = 0; i < g_nn; i++) {
        if (run_node(&g_n[i], i, i == g_nn - 1) != 0) return -1;
    }
    g_dec_pos = -1;
    if (g_debug) {
        fprintf(stderr, "[debug] engine_decode pos=%d nodes=%d %.3f ms\n",
                pos, g_nn, engine_now_ms() - t0);
    }
    return 0;
}

const float* engine_last_logits(int* count) {
    if (g_nn == 0) return NULL;
    T* out = t_find(g_n[g_nn - 1].out);
    if (!out) return NULL;
    vk_sync_host_tensor(out);
    materialize_tensor_f32(out);
    int d = out->shape[out->ndim - 1];
    if (g_last_token >= 0) {
        if (count) *count = d;
        return out->data + (long)g_last_token * d;
    }
    if (count) *count = (int)out->numel;
    return out->data;
}

void engine_free_ctx(void) {
    vk_free_weight_cache();
    vk_graph_reset();
    opengl_free_weight_cache();
    opengl_graph_reset();
#ifdef __APPLE__
    metal_graph_reset();
#endif
    vx_conv_f32_opt_free_all();
    engine_free_arena();
    for (int i = 0; i < g_nn; i++) {
        free(g_kcache[i]);
        free(g_vcache[i]);
        free(g_qwcache[i]);
        free(g_f16wcache[i]);
        free(g_f16bcache[i]);
        free(g_conv_wcache[i]);
        free(g_qiwcache[i]);
        free(g_qpwcache[i]);
        free(g_qpwilcache[i]);
        free(g_qpw8zcache[i]);
        free(g_qdwcache[i]);
        g_kcache[i] = NULL;
        g_vcache[i] = NULL;
        g_qwcache[i] = NULL;
        g_f16wcache[i] = NULL;
        g_f16bcache[i] = NULL;
        g_conv_wcache[i] = NULL;
        g_qiwcache[i] = NULL;
        g_qpwcache[i] = NULL;
        g_qpwilcache[i] = NULL;
        g_qpw8zcache[i] = NULL;
        g_qdwcache[i] = NULL;
    }
    for (int i = 0; i < g_nt; i++) {
        free(g_qt[i].data);
        memset(&g_qt[i], 0, sizeof(g_qt[i]));
    }
    for (int i = 0; i < g_nt; i++) {
        if (g_t[i].owns && g_t[i].data) {
            free(g_t[i].data);
            g_t[i].data = NULL;
        }
    }
    if (g_cfg_root) {
        cJSON_Delete(g_cfg_root);
        g_cfg_root = NULL;
    }
    if (g_blob) {
        free(g_blob);
        g_blob = NULL;
    }
    g_nt = 0;
    g_nn = 0;
    g_loaded = 0;
    g_first_input[0] = 0;
    g_dec_pos = -1;
    g_prefill_len = 0;
}

int engine_run(const char* config_path, const char* weights_path,
               const char* input_path, const char* output_path) {
    if (engine_init(config_path, weights_path) != 0) return -1;

    if (input_path) {
        long n;
        float* dst = engine_input_ptr(g_first_input, &n);
        long isz;
        char* idata = read_file(input_path, &isz);
        if (dst && idata) {
            long c = isz / 4;
            if (c > n) c = n;
            memcpy(dst, idata, (size_t)c * 4);
        }
        free(idata);
    }
    engine_forward();

    const char* vd = getenv("VDUMP");
    if (vd) {
        int idx = atoi(vd);
        if (idx >= 0 && idx < g_nn) {
            T* t = t_find(g_n[idx].out);
            int d = t->shape[t->ndim - 1];
            fprintf(stderr, "[dump] node %d %s row5:", idx, g_n[idx].op);
            for (int i = 0; i < 5; i++) fprintf(stderr, " %.4f", t->data[5 * d + i]);
            fprintf(stderr, "\n");
        }
    }

    if (output_path) {
        int count;
        const float* logits = engine_last_logits(&count);
        FILE* f = fopen(output_path, "wb");
        if (f && logits) {
            fwrite(logits, sizeof(float), (size_t)count, f);
            fclose(f);
        }
    }
    engine_free_ctx();
    return 0;
}
