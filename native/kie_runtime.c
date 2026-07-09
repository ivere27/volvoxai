#include "kie_runtime.h"
#include "cJSON.h"
#include "image_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define KIE_MAX_TENSORS 512
#define KIE_MAX_VOCAB 2048
#define KIE_MAX_TEXT 8192

typedef struct {
    char name[128];
    int shape[8];
    int ndim;
    long numel;
    float* data;
} KieTensor;

typedef struct {
    int vocab_size;
    int d_model;
    int heads;
    int enc_layers;
    int dec_layers;
    int ff_mult;
    int max_q_len;
    int max_out_len;
    int img_tokens;
    int adapter_families;
    int adapter_bottleneck;
    int use_router;
    int pad;
    int bos;
    int eos;
    int unk;
    char* vocab[KIE_MAX_VOCAB];
    char* blob;
    KieTensor tensors[KIE_MAX_TENSORS];
    int tensor_count;
} KieModel;

typedef struct {
    float* data;
    int c;
    int h;
    int w;
} KieImage;

static double kie_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static char* read_file_text(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[sz] = 0;
    fclose(f);
    if (out_size) *out_size = sz;
    return buf;
}

static long numel_of(const int* shape, int ndim) {
    long n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

static cJSON* read_json(const char* path) {
    long sz = 0;
    char* text = read_file_text(path, &sz);
    if (!text) return NULL;
    cJSON* root = cJSON_Parse(text);
    free(text);
    return root;
}

int kie_config_is_tiny_receipt(const char* config_path) {
    cJSON* root = read_json(config_path);
    if (!root) return 0;
    cJSON* iface = cJSON_GetObjectItem(root, "interface");
    cJSON* type = iface ? cJSON_GetObjectItem(iface, "type") : NULL;
    int ok = cJSON_IsString(type) && strcmp(type->valuestring, "tiny_receipt_kie") == 0;
    cJSON_Delete(root);
    return ok;
}

static int cfg_int(cJSON* obj, const char* key, int def) {
    cJSON* v = obj ? cJSON_GetObjectItem(obj, key) : NULL;
    return cJSON_IsNumber(v) ? v->valueint : def;
}

static int cfg_bool(cJSON* obj, const char* key, int def) {
    cJSON* v = obj ? cJSON_GetObjectItem(obj, key) : NULL;
    if (cJSON_IsBool(v)) return cJSON_IsTrue(v) ? 1 : 0;
    if (cJSON_IsNumber(v)) return v->valueint != 0;
    return def;
}

static int vocab_id(KieModel* m, const char* token) {
    for (int i = 0; i < m->vocab_size; i++) {
        if (m->vocab[i] && strcmp(m->vocab[i], token) == 0) return i;
    }
    return -1;
}

static int load_config(KieModel* m, const char* config_path) {
    cJSON* root = read_json(config_path);
    if (!root) { fprintf(stderr, "[kie] cannot parse config.json\n"); return -1; }
    cJSON* cfg = cJSON_GetObjectItem(root, "model_config");
    if (!cfg) cfg = cJSON_GetObjectItem(root, "config");
    m->vocab_size = cfg_int(cfg, "vocab_size", 0);
    m->d_model = cfg_int(cfg, "d_model", 320);
    m->heads = cfg_int(cfg, "heads", 8);
    m->enc_layers = cfg_int(cfg, "enc_layers", 6);
    m->dec_layers = cfg_int(cfg, "dec_layers", 4);
    m->ff_mult = cfg_int(cfg, "ff_mult", 4);
    m->max_q_len = cfg_int(cfg, "max_q_len", 192);
    m->max_out_len = cfg_int(cfg, "max_out_len", 192);
    m->img_tokens = cfg_int(cfg, "img_tokens", 210);
    m->adapter_families = cfg_int(cfg, "adapter_families", 8);
    m->adapter_bottleneck = cfg_int(cfg, "adapter_bottleneck", 64);
    m->use_router = cfg_bool(cfg, "use_router", 0);

    cJSON* vocab = cJSON_GetObjectItem(root, "vocab");
    cJSON* itos = vocab ? cJSON_GetObjectItem(vocab, "itos") : NULL;
    if (!cJSON_IsArray(itos)) { cJSON_Delete(root); fprintf(stderr, "[kie] missing vocab.itos\n"); return -1; }
    int n = cJSON_GetArraySize(itos);
    if (n > KIE_MAX_VOCAB) n = KIE_MAX_VOCAB;
    m->vocab_size = n;
    for (int i = 0; i < n; i++) {
        cJSON* item = cJSON_GetArrayItem(itos, i);
        const char* s = cJSON_IsString(item) ? item->valuestring : "";
        m->vocab[i] = (char*)malloc(strlen(s) + 1);
        if (!m->vocab[i]) { cJSON_Delete(root); return -1; }
        strcpy(m->vocab[i], s);
    }
    m->pad = vocab_id(m, "<pad>");
    m->bos = vocab_id(m, "<bos>");
    m->eos = vocab_id(m, "<eos>");
    m->unk = vocab_id(m, "<unk>");
    if (m->pad < 0) m->pad = 0;
    if (m->bos < 0) m->bos = 1;
    if (m->eos < 0) m->eos = 2;
    if (m->unk < 0) m->unk = 3;
    cJSON_Delete(root);
    return 0;
}

static int load_weights(KieModel* m, const char* path) {
    long sz = 0;
    m->blob = read_file_text(path, &sz);
    if (!m->blob) { fprintf(stderr, "[kie] cannot open weights: %s\n", path); return -1; }
    if (sz < 8) return -1;
    uint64_t header_size = 0;
    memcpy(&header_size, m->blob, 8);
    char* json = (char*)malloc((size_t)header_size + 1);
    if (!json) return -1;
    memcpy(json, m->blob + 8, (size_t)header_size);
    json[header_size] = 0;
    cJSON* root = cJSON_Parse(json);
    free(json);
    if (!root) { fprintf(stderr, "[kie] bad safetensors header\n"); return -1; }
    long data_base = 8 + (long)header_size;
    for (cJSON* it = root->child; it; it = it->next) {
        if (strcmp(it->string, "__metadata__") == 0) continue;
        if (m->tensor_count >= KIE_MAX_TENSORS) break;
        cJSON* dtype = cJSON_GetObjectItem(it, "dtype");
        if (!cJSON_IsString(dtype) || strcmp(dtype->valuestring, "F32") != 0) continue;
        cJSON* shape = cJSON_GetObjectItem(it, "shape");
        cJSON* off = cJSON_GetObjectItem(it, "data_offsets");
        if (!cJSON_IsArray(shape) || !cJSON_IsArray(off)) continue;
        KieTensor* t = &m->tensors[m->tensor_count++];
        strncpy(t->name, it->string, sizeof(t->name) - 1);
        t->name[sizeof(t->name) - 1] = 0;
        t->ndim = cJSON_GetArraySize(shape);
        if (t->ndim > 8) t->ndim = 8;
        for (int i = 0; i < t->ndim; i++) t->shape[i] = cJSON_GetArrayItem(shape, i)->valueint;
        t->numel = numel_of(t->shape, t->ndim);
        long start = (long)cJSON_GetArrayItem(off, 0)->valuedouble;
        t->data = (float*)(m->blob + data_base + start);
    }
    cJSON_Delete(root);
    return 0;
}

static void free_model(KieModel* m) {
    for (int i = 0; i < m->vocab_size; i++) free(m->vocab[i]);
    free(m->blob);
    memset(m, 0, sizeof(*m));
}

static KieTensor* tensor(KieModel* m, const char* name) {
    for (int i = 0; i < m->tensor_count; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    }
    fprintf(stderr, "[kie] missing tensor: %s\n", name);
    return NULL;
}

static int has_tensor(KieModel* m, const char* name) {
    for (int i = 0; i < m->tensor_count; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) return 1;
    }
    return 0;
}

static float* tdata(KieModel* m, const char* name) {
    KieTensor* t = tensor(m, name);
    return t ? t->data : NULL;
}

static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static void clean_text(const char* in, char* out, size_t out_size) {
    size_t o = 0;
    int pending_space = 0;
    while (*in && o + 1 < out_size) {
        unsigned char c = (unsigned char)*in;
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            pending_space = o > 0;
            in++;
            continue;
        }
        if (pending_space && o + 1 < out_size) out[o++] = ' ';
        pending_space = 0;
        int n = utf8_len(c);
        if (o + (size_t)n >= out_size) break;
        for (int i = 0; i < n && *in; i++) out[o++] = *in++;
    }
    out[o] = 0;
}

static int family_id_from_name(const char* family) {
    if (family && family[0] && strcmp(family, "auto") != 0) {
        const char* names[] = {"phone", "address", "store", "item_row", "item_math", "item_lookup", "math", "other"};
        for (int i = 0; i < 8; i++) if (strcmp(family, names[i]) == 0) return i;
    }
    return -1;
}

static int encode_question(KieModel* m, const char* text, int* ids) {
    char clean[KIE_MAX_TEXT];
    clean_text(text ? text : "", clean, sizeof(clean));
    int n = 0;
    const char* p = clean;
    while (*p && n < m->max_q_len) {
        int len = utf8_len((unsigned char)*p);
        char ch[8] = {0};
        for (int i = 0; i < len && p[i]; i++) ch[i] = p[i];
        int id = vocab_id(m, ch);
        ids[n++] = id >= 0 ? id : m->unk;
        p += len;
    }
    if (n < m->max_q_len) ids[n++] = m->eos;
    else ids[n - 1] = m->eos;
    return n;
}

static void decode_ids(KieModel* m, const int* ids, int n, char* out, size_t out_size) {
    size_t o = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id == m->eos) break;
        if (id == m->bos || id == m->pad) continue;
        const char* s = (id >= 0 && id < m->vocab_size && m->vocab[id]) ? m->vocab[id] : "";
        size_t len = strlen(s);
        if (o + len >= out_size) break;
        memcpy(out + o, s, len);
        o += len;
    }
    out[o] = 0;
}

static void linear_rows(const float* input, int rows, int in_dim, const float* weight,
                        const float* bias, int start_out, int out_dim, float* output) {
    for (int r = 0; r < rows; r++) {
        const float* in = input + (long)r * in_dim;
        float* out = output + (long)r * out_dim;
        for (int o = 0; o < out_dim; o++) {
            int row = start_out + o;
            const float* w = weight + (long)row * in_dim;
            float sum = bias ? bias[row] : 0.0f;
            for (int k = 0; k < in_dim; k++) sum += in[k] * w[k];
            out[o] = sum;
        }
    }
}

static void linear_vec(const float* input, int in_dim, const float* weight, const float* bias,
                       int start_out, int out_dim, float* output) {
    for (int o = 0; o < out_dim; o++) {
        int row = start_out + o;
        const float* w = weight + (long)row * in_dim;
        float sum = bias ? bias[row] : 0.0f;
        for (int k = 0; k < in_dim; k++) sum += input[k] * w[k];
        output[o] = sum;
    }
}

static void add_inplace(float* dst, const float* src, long n) {
    for (long i = 0; i < n; i++) dst[i] += src[i];
}

static float gelu_value(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

static void gelu_inplace(float* x, long n) {
    for (long i = 0; i < n; i++) x[i] = gelu_value(x[i]);
}

static int route_family_from_model(KieModel* m, const int* q_ids, int q_len, int debug) {
    if (!m->use_router ||
        !has_tensor(m, "router.net.0.weight") || !has_tensor(m, "router.net.0.bias") ||
        !has_tensor(m, "router.net.2.weight") || !has_tensor(m, "router.net.2.bias")) {
        return -1;
    }
    int d = m->d_model;
    int hidden = d / 2;
    int families = m->adapter_families;
    float* pooled = (float*)calloc((size_t)d, sizeof(float));
    float* h = (float*)malloc((size_t)hidden * sizeof(float));
    float* logits = (float*)malloc((size_t)families * sizeof(float));
    if (!pooled || !h || !logits) {
        free(pooled); free(h); free(logits);
        return -1;
    }

    float* tok = tdata(m, "tok.weight");
    float* q_pos = tdata(m, "q_pos");
    float* type_q = tdata(m, "type_q");
    for (int t = 0; t < q_len; t++) {
        int id = q_ids[t];
        if (id < 0 || id >= m->vocab_size) id = m->unk;
        const float* emb = tok + (long)id * d;
        for (int i = 0; i < d; i++) {
            pooled[i] += emb[i] + q_pos[(long)t * d + i] + type_q[i];
        }
    }
    float inv = q_len > 0 ? 1.0f / (float)q_len : 0.0f;
    for (int i = 0; i < d; i++) pooled[i] *= inv;

    linear_vec(pooled, d, tdata(m, "router.net.0.weight"), tdata(m, "router.net.0.bias"), 0, hidden, h);
    gelu_inplace(h, hidden);
    linear_vec(h, hidden, tdata(m, "router.net.2.weight"), tdata(m, "router.net.2.bias"), 0, families, logits);
    int best = 0;
    for (int i = 1; i < families; i++) if (logits[i] > logits[best]) best = i;
    if (debug) fprintf(stderr, "[debug] kie router family=%d logit=%.5f\n", best, logits[best]);

    free(pooled); free(h); free(logits);
    return best;
}

static void layernorm_rows(const float* input, int rows, int d, const float* weight,
                           const float* bias, float* output) {
    for (int r = 0; r < rows; r++) {
        const float* in = input + (long)r * d;
        float* out = output + (long)r * d;
        float mean = 0.0f;
        for (int i = 0; i < d; i++) mean += in[i];
        mean /= (float)d;
        float var = 0.0f;
        for (int i = 0; i < d; i++) { float z = in[i] - mean; var += z * z; }
        float inv = 1.0f / sqrtf(var / (float)d + 1e-5f);
        for (int i = 0; i < d; i++) out[i] = (in[i] - mean) * inv * weight[i] + bias[i];
    }
}

static void layernorm_vec(const float* input, int d, const float* weight, const float* bias, float* output) {
    layernorm_rows(input, 1, d, weight, bias, output);
}

static void attention_rows(const float* q, const float* k, const float* v, int q_len, int k_len,
                           int d, int heads, int causal, float* out) {
    int hd = d / heads;
    float scale = 1.0f / sqrtf((float)hd);
    float* scores = (float*)malloc((size_t)k_len * sizeof(float));
    for (int qi = 0; qi < q_len; qi++) {
        int limit = causal ? qi + 1 : k_len;
        for (int h = 0; h < heads; h++) {
            const float* qh = q + (long)qi * d + h * hd;
            float maxv = -1e38f;
            for (int kj = 0; kj < limit; kj++) {
                const float* kh = k + (long)kj * d + h * hd;
                float s = 0.0f;
                for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
                s *= scale;
                scores[kj] = s;
                if (s > maxv) maxv = s;
            }
            float denom = 0.0f;
            for (int kj = 0; kj < limit; kj++) {
                scores[kj] = expf(scores[kj] - maxv);
                denom += scores[kj];
            }
            float* oh = out + (long)qi * d + h * hd;
            for (int i = 0; i < hd; i++) {
                float acc = 0.0f;
                for (int kj = 0; kj < limit; kj++) acc += (scores[kj] / denom) * v[(long)kj * d + h * hd + i];
                oh[i] = acc;
            }
        }
    }
    free(scores);
}

static void attention_vec(const float* q, const float* k, const float* v, int k_len,
                          int d, int heads, float* out) {
    int hd = d / heads;
    float scale = 1.0f / sqrtf((float)hd);
    float* scores = (float*)malloc((size_t)k_len * sizeof(float));
    for (int h = 0; h < heads; h++) {
        const float* qh = q + h * hd;
        float maxv = -1e38f;
        for (int kj = 0; kj < k_len; kj++) {
            const float* kh = k + (long)kj * d + h * hd;
            float s = 0.0f;
            for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
            s *= scale;
            scores[kj] = s;
            if (s > maxv) maxv = s;
        }
        float denom = 0.0f;
        for (int kj = 0; kj < k_len; kj++) {
            scores[kj] = expf(scores[kj] - maxv);
            denom += scores[kj];
        }
        float* oh = out + h * hd;
        for (int i = 0; i < hd; i++) {
            float acc = 0.0f;
            for (int kj = 0; kj < k_len; kj++) acc += (scores[kj] / denom) * v[(long)kj * d + h * hd + i];
            oh[i] = acc;
        }
    }
    free(scores);
}

static int norm_groups(int c) {
    int groups[] = {32, 16, 8, 4, 2};
    for (int i = 0; i < 5; i++) if (c % groups[i] == 0) return groups[i];
    return 1;
}

static void groupnorm_silu(KieImage* img, const float* gamma, const float* beta, int activate) {
    int groups = norm_groups(img->c);
    int gc = img->c / groups;
    int area = img->h * img->w;
    for (int g = 0; g < groups; g++) {
        int c0 = g * gc, c1 = c0 + gc;
        int count = gc * area;
        float mean = 0.0f;
        for (int c = c0; c < c1; c++) {
            const float* src = img->data + (long)c * area;
            for (int i = 0; i < area; i++) mean += src[i];
        }
        mean /= (float)count;
        float var = 0.0f;
        for (int c = c0; c < c1; c++) {
            const float* src = img->data + (long)c * area;
            for (int i = 0; i < area; i++) { float z = src[i] - mean; var += z * z; }
        }
        float inv = 1.0f / sqrtf(var / (float)count + 1e-5f);
        for (int c = c0; c < c1; c++) {
            float* dst = img->data + (long)c * area;
            for (int i = 0; i < area; i++) {
                float v = (dst[i] - mean) * inv * gamma[c] + beta[c];
                dst[i] = activate ? v / (1.0f + expf(-v)) : v;
            }
        }
    }
}

static KieImage conv3x3(const KieImage* in, const float* w, int out_c, int stride) {
    KieImage out;
    out.c = out_c;
    out.h = (in->h + 2 - 3) / stride + 1;
    out.w = (in->w + 2 - 3) / stride + 1;
    long n = (long)out.c * out.h * out.w;
    out.data = (float*)calloc((size_t)n, sizeof(float));
    int in_area = in->h * in->w;
    int out_area = out.h * out.w;
    for (int oc = 0; oc < out_c; oc++) {
        float* dst = out.data + (long)oc * out_area;
        const float* woc = w + (long)oc * in->c * 9;
        for (int oy = 0; oy < out.h; oy++) {
            int iy0 = oy * stride - 1;
            for (int ox = 0; ox < out.w; ox++) {
                int ix0 = ox * stride - 1;
                float sum = 0.0f;
                for (int ic = 0; ic < in->c; ic++) {
                    const float* src = in->data + (long)ic * in_area;
                    const float* wc = woc + ic * 9;
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = iy0 + ky;
                        if (iy < 0 || iy >= in->h) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = ix0 + kx;
                            if (ix < 0 || ix >= in->w) continue;
                            sum += src[iy * in->w + ix] * wc[ky * 3 + kx];
                        }
                    }
                }
                dst[oy * out.w + ox] = sum;
            }
        }
    }
    return out;
}

static KieImage conv_block(KieModel* m, const KieImage* in, const char* prefix, int stride) {
    char name[128];
    snprintf(name, sizeof(name), "%s.0.weight", prefix);
    KieTensor* wt = tensor(m, name);
    KieImage out = conv3x3(in, wt->data, wt->shape[0], stride);
    snprintf(name, sizeof(name), "%s.1.weight", prefix);
    float* gamma = tdata(m, name);
    snprintf(name, sizeof(name), "%s.1.bias", prefix);
    float* beta = tdata(m, name);
    groupnorm_silu(&out, gamma, beta, 1);
    return out;
}

static KieImage res_block(KieModel* m, const KieImage* in, const char* prefix) {
    char name[128];
    snprintf(name, sizeof(name), "%s.0.weight", prefix);
    KieTensor* wt = tensor(m, name);
    KieImage y = conv3x3(in, wt->data, wt->shape[0], 1);
    snprintf(name, sizeof(name), "%s.1.weight", prefix);
    float* gamma = tdata(m, name);
    snprintf(name, sizeof(name), "%s.1.bias", prefix);
    float* beta = tdata(m, name);
    groupnorm_silu(&y, gamma, beta, 1);

    snprintf(name, sizeof(name), "%s.3.weight", prefix);
    wt = tensor(m, name);
    KieImage z = conv3x3(&y, wt->data, wt->shape[0], 1);
    free(y.data);
    snprintf(name, sizeof(name), "%s.4.weight", prefix);
    gamma = tdata(m, name);
    snprintf(name, sizeof(name), "%s.4.bias", prefix);
    beta = tdata(m, name);
    groupnorm_silu(&z, gamma, beta, 0);
    long n = (long)z.c * z.h * z.w;
    for (long i = 0; i < n; i++) {
        float v = z.data[i] + in->data[i];
        z.data[i] = v / (1.0f + expf(-v));
    }
    return z;
}

static KieImage run_stem(KieModel* m, float* image) {
    KieImage x = {image, 1, 320, 672};
    KieImage y;
    y = conv_block(m, &x, "stem.0.net", 2);
    x = conv_block(m, &y, "stem.1.net", 2); free(y.data);
    y = res_block(m, &x, "stem.2.net"); free(x.data); x = y;
    y = conv_block(m, &x, "stem.3.net", 2); free(x.data); x = y;
    y = res_block(m, &x, "stem.4.net"); free(x.data); x = y;
    y = conv_block(m, &x, "stem.5.net", 2); free(x.data); x = y;
    y = res_block(m, &x, "stem.6.net"); free(x.data); x = y;
    y = conv_block(m, &x, "stem.7.net", 2); free(x.data); x = y;
    y = res_block(m, &x, "stem.8.net"); free(x.data);
    return y;
}

static float* build_memory(KieModel* m, const KieImage* stem, const int* q_ids, int q_len, int* rows_out) {
    int d = m->d_model;
    int img_rows = stem->h * stem->w;
    int rows = img_rows + q_len;
    float* mem = (float*)malloc((size_t)rows * d * sizeof(float));
    float* img_pos = tdata(m, "img_pos");
    float* q_pos = tdata(m, "q_pos");
    float* type_img = tdata(m, "type_img");
    float* type_q = tdata(m, "type_q");
    float* tok = tdata(m, "tok.weight");
    int area = stem->h * stem->w;
    for (int y = 0; y < stem->h; y++) {
        for (int x = 0; x < stem->w; x++) {
            int token = y * stem->w + x;
            float* dst = mem + (long)token * d;
            int spatial = y * stem->w + x;
            for (int c = 0; c < d; c++) dst[c] = stem->data[(long)c * area + spatial] + img_pos[(long)token * d + c] + type_img[c];
        }
    }
    for (int t = 0; t < q_len; t++) {
        int id = q_ids[t];
        if (id < 0 || id >= m->vocab_size) id = m->unk;
        float* dst = mem + (long)(img_rows + t) * d;
        const float* emb = tok + (long)id * d;
        for (int i = 0; i < d; i++) dst[i] = emb[i] + q_pos[(long)t * d + i] + type_q[i];
    }
    *rows_out = rows;
    return mem;
}

static void transformer_encoder(KieModel* m, float* x, int rows) {
    int d = m->d_model;
    int ff = d * m->ff_mult;
    float* n1 = (float*)malloc((size_t)rows * d * sizeof(float));
    float* q = (float*)malloc((size_t)rows * d * sizeof(float));
    float* k = (float*)malloc((size_t)rows * d * sizeof(float));
    float* v = (float*)malloc((size_t)rows * d * sizeof(float));
    float* attn = (float*)malloc((size_t)rows * d * sizeof(float));
    float* proj = (float*)malloc((size_t)rows * d * sizeof(float));
    float* ff1 = (float*)malloc((size_t)rows * ff * sizeof(float));
    float* ff2 = (float*)malloc((size_t)rows * d * sizeof(float));
    char name[128];
    for (int layer = 0; layer < m->enc_layers; layer++) {
        snprintf(name, sizeof(name), "encoder.layers.%d.norm1.weight", layer); float* lnw = tdata(m, name);
        snprintf(name, sizeof(name), "encoder.layers.%d.norm1.bias", layer); float* lnb = tdata(m, name);
        layernorm_rows(x, rows, d, lnw, lnb, n1);
        snprintf(name, sizeof(name), "encoder.layers.%d.self_attn.in_proj_weight", layer); float* iw = tdata(m, name);
        snprintf(name, sizeof(name), "encoder.layers.%d.self_attn.in_proj_bias", layer); float* ib = tdata(m, name);
        linear_rows(n1, rows, d, iw, ib, 0, d, q);
        linear_rows(n1, rows, d, iw, ib, d, d, k);
        linear_rows(n1, rows, d, iw, ib, d * 2, d, v);
        attention_rows(q, k, v, rows, rows, d, m->heads, 0, attn);
        snprintf(name, sizeof(name), "encoder.layers.%d.self_attn.out_proj.weight", layer); float* ow = tdata(m, name);
        snprintf(name, sizeof(name), "encoder.layers.%d.self_attn.out_proj.bias", layer); float* ob = tdata(m, name);
        linear_rows(attn, rows, d, ow, ob, 0, d, proj);
        add_inplace(x, proj, (long)rows * d);

        snprintf(name, sizeof(name), "encoder.layers.%d.norm2.weight", layer); lnw = tdata(m, name);
        snprintf(name, sizeof(name), "encoder.layers.%d.norm2.bias", layer); lnb = tdata(m, name);
        layernorm_rows(x, rows, d, lnw, lnb, n1);
        snprintf(name, sizeof(name), "encoder.layers.%d.linear1.weight", layer); float* w1 = tdata(m, name);
        snprintf(name, sizeof(name), "encoder.layers.%d.linear1.bias", layer); float* b1 = tdata(m, name);
        linear_rows(n1, rows, d, w1, b1, 0, ff, ff1);
        gelu_inplace(ff1, (long)rows * ff);
        snprintf(name, sizeof(name), "encoder.layers.%d.linear2.weight", layer); float* w2 = tdata(m, name);
        snprintf(name, sizeof(name), "encoder.layers.%d.linear2.bias", layer); float* b2 = tdata(m, name);
        linear_rows(ff1, rows, ff, w2, b2, 0, d, ff2);
        add_inplace(x, ff2, (long)rows * d);
    }
    free(n1); free(q); free(k); free(v); free(attn); free(proj); free(ff1); free(ff2);
}

static void apply_adapter_rows(KieModel* m, float* x, int rows, int family, const char* prefix) {
    int d = m->d_model, h = m->adapter_bottleneck;
    char name[128];
    snprintf(name, sizeof(name), "%s.%d.down.weight", prefix, family); float* dw = tdata(m, name);
    snprintf(name, sizeof(name), "%s.%d.down.bias", prefix, family); float* db = tdata(m, name);
    snprintf(name, sizeof(name), "%s.%d.up.weight", prefix, family); float* uw = tdata(m, name);
    snprintf(name, sizeof(name), "%s.%d.up.bias", prefix, family); float* ub = tdata(m, name);
    float* down = (float*)malloc((size_t)rows * h * sizeof(float));
    float* up = (float*)malloc((size_t)rows * d * sizeof(float));
    linear_rows(x, rows, d, dw, db, 0, h, down);
    gelu_inplace(down, (long)rows * h);
    linear_rows(down, rows, h, uw, ub, 0, d, up);
    add_inplace(x, up, (long)rows * d);
    free(down); free(up);
}

static void apply_adapter_vec(KieModel* m, float* x, int family, const char* prefix) {
    apply_adapter_rows(m, x, 1, family, prefix);
}

static int argmax(const float* x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

static int generate(KieModel* m, float* memory, int mem_rows, int family, int max_new, int debug, int* out_ids) {
    int d = m->d_model;
    int vocab = m->vocab_size;
    if (max_new <= 0 || max_new > m->max_out_len) max_new = m->max_out_len;
    int ids_len = 1;
    out_ids[0] = m->bos;

    float** cross_k = (float**)calloc((size_t)m->dec_layers, sizeof(float*));
    float** cross_v = (float**)calloc((size_t)m->dec_layers, sizeof(float*));
    float** self_k = (float**)calloc((size_t)m->dec_layers, sizeof(float*));
    float** self_v = (float**)calloc((size_t)m->dec_layers, sizeof(float*));
    int* self_rows = (int*)calloc((size_t)m->dec_layers, sizeof(int));
    char name[128];
    for (int layer = 0; layer < m->dec_layers; layer++) {
        cross_k[layer] = (float*)malloc((size_t)mem_rows * d * sizeof(float));
        cross_v[layer] = (float*)malloc((size_t)mem_rows * d * sizeof(float));
        self_k[layer] = (float*)calloc((size_t)m->max_out_len * d, sizeof(float));
        self_v[layer] = (float*)calloc((size_t)m->max_out_len * d, sizeof(float));
        snprintf(name, sizeof(name), "decoder.layers.%d.multihead_attn.in_proj_weight", layer); float* w = tdata(m, name);
        snprintf(name, sizeof(name), "decoder.layers.%d.multihead_attn.in_proj_bias", layer); float* b = tdata(m, name);
        linear_rows(memory, mem_rows, d, w, b, d, d, cross_k[layer]);
        linear_rows(memory, mem_rows, d, w, b, d * 2, d, cross_v[layer]);
    }

    float* x = (float*)malloc((size_t)d * sizeof(float));
    float* n = (float*)malloc((size_t)d * sizeof(float));
    float* q = (float*)malloc((size_t)d * sizeof(float));
    float* k = (float*)malloc((size_t)d * sizeof(float));
    float* v = (float*)malloc((size_t)d * sizeof(float));
    float* attn = (float*)malloc((size_t)d * sizeof(float));
    float* proj = (float*)malloc((size_t)d * sizeof(float));
    float* ff1 = (float*)malloc((size_t)d * m->ff_mult * sizeof(float));
    float* ff2 = (float*)malloc((size_t)d * sizeof(float));
    float* logits = (float*)malloc((size_t)vocab * sizeof(float));
    float* tok = tdata(m, "tok.weight");
    float* y_pos = tdata(m, "y_pos");

    for (int pos = 0; pos < max_new - 1; pos++) {
        int token = out_ids[ids_len - 1];
        if (token < 0 || token >= vocab) token = m->unk;
        for (int i = 0; i < d; i++) x[i] = tok[(long)token * d + i] + y_pos[(long)pos * d + i];
        for (int layer = 0; layer < m->dec_layers; layer++) {
            snprintf(name, sizeof(name), "decoder.layers.%d.norm1.weight", layer); float* lnw = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.norm1.bias", layer); float* lnb = tdata(m, name);
            layernorm_vec(x, d, lnw, lnb, n);
            snprintf(name, sizeof(name), "decoder.layers.%d.self_attn.in_proj_weight", layer); float* sw = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.self_attn.in_proj_bias", layer); float* sb = tdata(m, name);
            linear_vec(n, d, sw, sb, 0, d, q);
            linear_vec(n, d, sw, sb, d, d, k);
            linear_vec(n, d, sw, sb, d * 2, d, v);
            memcpy(self_k[layer] + (long)self_rows[layer] * d, k, (size_t)d * sizeof(float));
            memcpy(self_v[layer] + (long)self_rows[layer] * d, v, (size_t)d * sizeof(float));
            self_rows[layer]++;
            attention_vec(q, self_k[layer], self_v[layer], self_rows[layer], d, m->heads, attn);
            snprintf(name, sizeof(name), "decoder.layers.%d.self_attn.out_proj.weight", layer); float* ow = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.self_attn.out_proj.bias", layer); float* ob = tdata(m, name);
            linear_vec(attn, d, ow, ob, 0, d, proj);
            add_inplace(x, proj, d);

            snprintf(name, sizeof(name), "decoder.layers.%d.norm2.weight", layer); lnw = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.norm2.bias", layer); lnb = tdata(m, name);
            layernorm_vec(x, d, lnw, lnb, n);
            snprintf(name, sizeof(name), "decoder.layers.%d.multihead_attn.in_proj_weight", layer); float* cw = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.multihead_attn.in_proj_bias", layer); float* cb = tdata(m, name);
            linear_vec(n, d, cw, cb, 0, d, q);
            attention_vec(q, cross_k[layer], cross_v[layer], mem_rows, d, m->heads, attn);
            snprintf(name, sizeof(name), "decoder.layers.%d.multihead_attn.out_proj.weight", layer); ow = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.multihead_attn.out_proj.bias", layer); ob = tdata(m, name);
            linear_vec(attn, d, ow, ob, 0, d, proj);
            add_inplace(x, proj, d);

            snprintf(name, sizeof(name), "decoder.layers.%d.norm3.weight", layer); lnw = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.norm3.bias", layer); lnb = tdata(m, name);
            layernorm_vec(x, d, lnw, lnb, n);
            snprintf(name, sizeof(name), "decoder.layers.%d.linear1.weight", layer); float* w1 = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.linear1.bias", layer); float* b1 = tdata(m, name);
            linear_vec(n, d, w1, b1, 0, d * m->ff_mult, ff1);
            gelu_inplace(ff1, d * m->ff_mult);
            snprintf(name, sizeof(name), "decoder.layers.%d.linear2.weight", layer); float* w2 = tdata(m, name);
            snprintf(name, sizeof(name), "decoder.layers.%d.linear2.bias", layer); float* b2 = tdata(m, name);
            linear_vec(ff1, d * m->ff_mult, w2, b2, 0, d, ff2);
            add_inplace(x, ff2, d);
        }
        apply_adapter_vec(m, x, family, "decoder_adapters");
        float* nw = tdata(m, "norm.weight");
        float* nb = tdata(m, "norm.bias");
        layernorm_vec(x, d, nw, nb, n);
        float* head = tdata(m, "head.weight");
        float* out_bias = tdata(m, "out_bias");
        linear_vec(n, d, head, NULL, 0, vocab, logits);
        for (int i = 0; i < vocab; i++) logits[i] += out_bias[i];
        int next = argmax(logits, vocab);
        out_ids[ids_len++] = next;
        if (debug) fprintf(stderr, "[debug] kie decode pos=%d token=%d\n", pos, next);
        if (next == m->eos) break;
    }

    for (int i = 0; i < m->dec_layers; i++) { free(cross_k[i]); free(cross_v[i]); free(self_k[i]); free(self_v[i]); }
    free(cross_k); free(cross_v); free(self_k); free(self_v); free(self_rows);
    free(x); free(n); free(q); free(k); free(v); free(attn); free(proj); free(ff1); free(ff2); free(logits);
    return ids_len;
}

int kie_chat(const char* config_path, const char* weights_path, const char* image_path,
             const char* prompt, const char* family, int max_new, int debug) {
    KieModel m;
    memset(&m, 0, sizeof(m));
    double t0 = kie_now_ms();
    if (load_config(&m, config_path) != 0) return 1;
    if (load_weights(&m, weights_path) != 0) { free_model(&m); return 1; }
    if (debug) fprintf(stderr, "[debug] kie load tensors=%d %.3f ms\n", m.tensor_count, kie_now_ms() - t0);

    float* image = (float*)calloc((size_t)320 * 672, sizeof(float));
    int shape[4] = {1, 1, 320, 672};
    char err[256] = {0};
    t0 = kie_now_ms();
    if (volvox_load_image_to_tensor(image_path, image, shape, 4, VOLVOX_IMAGE_MINUS_ONE_ONE, err, sizeof(err)) != 0) {
        fprintf(stderr, "[kie] image load failed: %s\n", err[0] ? err : "unknown error");
        free(image); free_model(&m); return 1;
    }
    if (debug) fprintf(stderr, "[debug] kie image %.3f ms\n", kie_now_ms() - t0);

    char clean[KIE_MAX_TEXT];
    clean_text(prompt ? prompt : "", clean, sizeof(clean));
    int q_ids[512];
    int q_len = encode_question(&m, clean, q_ids);
    int fam = family_id_from_name(family);
    const char* family_source = "manual";
    if (fam < 0) {
        if (family && family[0] && strcmp(family, "auto") != 0) {
            fprintf(stderr, "[kie] unknown family: %s\n", family);
            free(image);
            free_model(&m);
            return 2;
        }
        fam = route_family_from_model(&m, q_ids, q_len, debug);
        family_source = "router";
    }
    if (fam < 0) {
        fprintf(stderr, "[kie] model router unavailable; pass --family <name> for manual routing.\n");
        free(image);
        free_model(&m);
        return 2;
    }
    if (fam >= m.adapter_families) {
        fprintf(stderr, "[kie] selected family id %d exceeds adapter_families=%d.\n", fam, m.adapter_families);
        free(image);
        free_model(&m);
        return 2;
    }
    if (debug) fprintf(stderr, "[debug] kie q_len=%d family=%d source=%s\n", q_len, fam, family_source);

    t0 = kie_now_ms();
    KieImage stem = run_stem(&m, image);
    if (debug) fprintf(stderr, "[debug] kie stem shape=%dx%dx%d %.3f ms\n", stem.c, stem.h, stem.w, kie_now_ms() - t0);
    int mem_rows = 0;
    float* memory = build_memory(&m, &stem, q_ids, q_len, &mem_rows);
    free(stem.data);

    t0 = kie_now_ms();
    transformer_encoder(&m, memory, mem_rows);
    apply_adapter_rows(&m, memory, mem_rows, fam, "memory_adapters");
    if (debug) fprintf(stderr, "[debug] kie encoder rows=%d %.3f ms\n", mem_rows, kie_now_ms() - t0);

    int ids[512];
    t0 = kie_now_ms();
    int n_ids = generate(&m, memory, mem_rows, fam, max_new, debug, ids);
    if (debug) fprintf(stderr, "[debug] kie decoder ids=%d %.3f ms\n", n_ids, kie_now_ms() - t0);

    char out[KIE_MAX_TEXT];
    decode_ids(&m, ids, n_ids, out, sizeof(out));
    printf("%s\n", out);

    free(memory);
    free(image);
    free_model(&m);
    return 0;
}
