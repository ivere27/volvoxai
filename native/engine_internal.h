#ifndef VOLVOX_ENGINE_INTERNAL_H
#define VOLVOX_ENGINE_INTERNAL_H

#include "cJSON.h"
#include <stddef.h>
#include <stdint.h>

#define MAXT 1024
#define MAXN 1024
#define MAXIN 12
#define MAX_SEQ 4096

enum { T_F32 = 0, T_I8 = 1, T_U8 = 2, T_I32 = 3, T_F16 = 4 };

typedef struct { char name[128]; int shape[8]; int ndim; float* data; long numel; int owns; int dtype; size_t elem_size; } T;
typedef struct { char key[24]; char name[128]; } Ref;
typedef struct { char op[40]; Ref ins[MAXIN]; int nin; Ref outs[MAXIN]; int nout; char out[128]; cJSON* params;
                 int fuse_relu6; int skip; } Node;
typedef struct {
    signed char* data;
    long cap;
    float scale;
    int zp;
    int has_params;
    int valid;
} QTensor;

extern T g_t[MAXT];
extern int g_nt;
extern Node g_n[MAXN];
extern int g_nn;
extern QTensor g_qt[MAXT];
extern char* g_blob;
extern cJSON* g_cfg_root;
extern char g_first_input[128];
extern int g_loaded;
extern int g_dec_pos;
extern int g_prefill_len;
extern float* g_kcache[MAXN];
extern float* g_vcache[MAXN];
extern float* g_qwcache[MAXN];
extern float* g_f16wcache[MAXN];
extern float* g_f16bcache[MAXN];
extern float* g_conv_wcache[MAXN];
extern int16_t* g_qiwcache[MAXN];
extern int16_t* g_qpwcache[MAXN];
extern int16_t* g_qpwilcache[MAXN];
extern signed char* g_qpw8zcache[MAXN];
extern int16_t* g_qdwcache[MAXN];
extern int g_concat_sigmoid_fuse[MAXN];

extern int g_use_vulkan;
extern int g_use_nnapi;
extern int g_use_opengl;
extern int g_use_metal;
extern int g_debug;
extern int g_last_token;

double engine_now_ms(void);
T* t_find(const char* name);
void materialize_tensor_f32(T* t);
void qt_invalidate_all(void);
void vk_sync_host_tensor(T* t);
void vk_mark_owned_tensors_host_dirty(void);
char* read_file(const char* path, long* out_size);
int load_weights(const char* path);
int build_graph(const char* config_path);
void engine_free_arena(void);
int prepack_qconv_weights(void);
int prepack_conv_weights(void);
int run_node(Node* n, int idx, int is_last);
void prof_reset(void);
void prof_add_entry(const char* op, double ms);
void prof_report(void);

#endif
