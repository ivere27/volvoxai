/* MAXT is a power of two and the table is kept at or below 50% occupancy.
 * Open addressing avoids per-model heap allocation and pointer lifetime
 * concerns while reducing a successful lookup from O(g_nt) strcmp calls to
 * expected O(1).  Entries store tensor_index + 1 so zero remains the empty
 * sentinel. */
#define TENSOR_NAME_INDEX_CAPACITY (MAXT * 2)
_Static_assert((TENSOR_NAME_INDEX_CAPACITY & (TENSOR_NAME_INDEX_CAPACITY - 1)) == 0,
               "tensor name index capacity must be a power of two");

static int g_tensor_name_index[TENSOR_NAME_INDEX_CAPACITY];
static int g_tensor_name_index_count;

static uint32_t tensor_name_hash(const char* name) {
    uint32_t hash = UINT32_C(2166136261);
    for (const unsigned char* byte = (const unsigned char*)name; *byte; byte++) {
        hash ^= *byte;
        hash *= UINT32_C(16777619);
    }
    return hash;
}

static int tensor_name_index_insert(int tensor_index) {
    if (tensor_index < 0 || tensor_index >= g_nt || !g_t[tensor_index].name[0]) return -1;
    uint32_t slot = tensor_name_hash(g_t[tensor_index].name) &
                    (TENSOR_NAME_INDEX_CAPACITY - 1u);
    for (int probe = 0; probe < TENSOR_NAME_INDEX_CAPACITY; probe++) {
        int encoded = g_tensor_name_index[slot];
        if (!encoded) {
            g_tensor_name_index[slot] = tensor_index + 1;
            return 0;
        }
        /* Preserve the former linear lookup's first-entry behavior if an
         * invalid internal caller constructs duplicate names. */
        int existing = encoded - 1;
        if (existing >= 0 && existing < g_nt &&
            !strcmp(g_t[existing].name, g_t[tensor_index].name)) return 0;
        slot = (slot + 1u) & (TENSOR_NAME_INDEX_CAPACITY - 1u);
    }
    return -1;
}

void volvoxai_engine_tensor_name_index_rebuild(void) {
    if (g_tensor_name_index_count == g_nt) return;
    memset(g_tensor_name_index, 0, sizeof(g_tensor_name_index));
    for (int index = 0; index < g_nt; index++) {
        if (tensor_name_index_insert(index) != 0) {
            /* Valid runtime tables cannot reach this at <= 50% occupancy.
             * Leave the count stale so a later lookup retries after the
             * caller repairs an invalid table. */
            g_tensor_name_index_count = -1;
            return;
        }
    }
    g_tensor_name_index_count = g_nt;
}

void volvoxai_engine_tensor_name_index_invalidate(void) {
    g_tensor_name_index_count = -1;
}

void volvoxai_engine_tensor_name_index_add(int tensor_index) {
    /* Appending is the common load/build path.  Keep it O(1) when the old
     * table was current; otherwise defer one full rebuild until lookup. */
    if (tensor_index != g_nt - 1 || g_tensor_name_index_count != tensor_index ||
        tensor_name_index_insert(tensor_index) != 0) {
        volvoxai_engine_tensor_name_index_invalidate();
        return;
    }
    g_tensor_name_index_count = g_nt;
}

T* t_find(const char* name) {
    if (!name || !name[0]) return NULL;
    if (g_tensor_name_index_count == g_nt) {
        uint32_t slot = tensor_name_hash(name) & (TENSOR_NAME_INDEX_CAPACITY - 1u);
        for (int probe = 0; probe < TENSOR_NAME_INDEX_CAPACITY; probe++) {
            int encoded = g_tensor_name_index[slot];
            if (!encoded) return NULL;
            int index = encoded - 1;
            if (index < 0 || index >= g_nt) break;
            if (!strcmp(g_t[index].name, name)) return &g_t[index];
            slot = (slot + 1u) & (TENSOR_NAME_INDEX_CAPACITY - 1u);
        }
    }

    /* Structural writers invalidate the index while they rename or compact
     * tensors. Lookups can run under either the model or metadata mutex, so a
     * reader must never lazily rebuild shared state under only one of them.
     * The mutation boundary publishes a rebuilt index; until then this
     * read-only fallback preserves the original lookup semantics. */
    for (int index = 0; index < g_nt; index++)
        if (!strcmp(g_t[index].name, name)) return &g_t[index];
    return NULL;
}
static int removed_graph_output_index(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_removed_graph_output_count; i++) {
        if (!strcmp(g_removed_graph_outputs[i], name)) return i;
    }
    return -1;
}

static void removed_graph_output_add(const char* name) {
    if (!name || !name[0] || removed_graph_output_index(name) >= 0 ||
        g_removed_graph_output_count >= MAXT) return;
    strncpy(g_removed_graph_outputs[g_removed_graph_output_count], name, 127);
    g_removed_graph_outputs[g_removed_graph_output_count][127] = 0;
    g_removed_graph_output_count++;
}

static void removed_graph_output_remove(const char* name) {
    int index = removed_graph_output_index(name);
    if (index < 0) return;
    memmove(&g_removed_graph_outputs[index], &g_removed_graph_outputs[index + 1],
            (size_t)(g_removed_graph_output_count - index - 1) * sizeof(g_removed_graph_outputs[0]));
    g_removed_graph_output_count--;
    g_removed_graph_outputs[g_removed_graph_output_count][0] = 0;
}

int volvoxai_engine_tensor_is_removed_output(const char* name) {
    return removed_graph_output_index(name) >= 0;
}
static int t_index(const T* t) {
    if (!t) return -1;
    ptrdiff_t idx = t - g_t;
    return (idx >= 0 && idx < g_nt) ? (int)idx : -1;
}
static long numel_of(const int* shape, int ndim) {
    if (ndim < 0 || ndim > 8 || (ndim > 0 && !shape)) return -1;
    long n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0 || n > LONG_MAX / shape[i]) return -1;
        n *= shape[i];
    }
    return n;
}

static inline signed char clamp_i8_i32(int v) {
    if (v < -128) return -128;
    if (v > 127) return 127;
    return (signed char)v;
}

static inline signed char quantize_scalar_i8(float x, float scale, int zp) {
    float quantized = x / scale + (float)zp;
    if (isnan(quantized)) return clamp_i8_i32(zp);
    if (quantized <= -128.0f) return -128;
    if (quantized >= 127.0f) return 127;
    return (signed char)lrintf(quantized);
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

void volvoxai_engine_clear_qlinear_metadata(void) {
    for (int i = 0; i < MAXN; i++) {
        free(g_qlinear_meta[i].weight_scales);
        free(g_qlinear_meta[i].weight_zero_points);
        free(g_qlinear_meta[i].bias);
        free(g_qlinear_meta[i].packed_weight);
        memset(&g_qlinear_meta[i], 0, sizeof(g_qlinear_meta[i]));
    }
}

void volvoxai_engine_clear_qconv_metadata(void) {
    for (int i = 0; i < MAXN; i++) {
        free(g_qconv_meta[i].weight_scales);
        free(g_qconv_meta[i].weight_zero_points);
        free(g_qconv_meta[i].bias);
        memset(&g_qconv_meta[i], 0, sizeof(g_qconv_meta[i]));
    }
}

void volvoxai_engine_clear_qembedding_metadata(void) {
    for (int i = 0; i < MAXN; i++) {
        free(g_qembedding_meta[i].weight_scales);
        free(g_qembedding_meta[i].weight_zero_points);
        memset(&g_qembedding_meta[i], 0, sizeof(g_qembedding_meta[i]));
    }
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

static int qt_mark_params_strict(T* t, float scale, int zp) {
    if (!t || !isfinite(scale) || scale <= 0.0f || zp < -128 || zp > 127) return -1;
    QTensor* q = qt_get(t);
    if (!q) return -1;
    if (q->has_params) {
        return fabsf(q->scale - scale) <= 1.0e-8f && q->zp == zp ? 0 : -1;
    }
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

/* A W8A8 consumer must never reinterpret a valid sidecar at a different
 * scale. Materialize first, then quantize from the authoritative F32 values. */
static QTensor* qt_ensure_quantized_params(T* t, float scale, int zp) {
    if (!t || t->dtype != T_F32 || !isfinite(scale) || scale <= 0.0f || zp < -128 || zp > 127)
        return NULL;
    QTensor* q = qt_get(t);
    if (!q) return NULL;
    if (q->valid && q->has_params && fabsf(q->scale - scale) <= 1.0e-8f && q->zp == zp)
        return q;
    if (q->valid) {
        qt_materialize_f32(t);
        q->valid = 0;
    }
    if (qt_quantize_from_f32(t, scale, zp) < 0) return NULL;
    return qt_get(t);
}

/* Native GPU graph paths read F32 backing buffers directly. Keep every node
 * that participates in a QTensor island on CPU so a valid sidecar is never
 * mistaken for stale F32 storage by a later GPU dispatch. */
static int node_uses_qtensor(const Node* n, T* output) {
    QTensor* q = qt_get(output);
    if (q && q->has_params) return 1;
    if (!n) return 0;
    for (int i = 0; i < n->nin; i++) {
        T* input = t_find(n->ins[i].name);
        q = qt_get(input);
        if (q && q->has_params) return 1;
    }
    return 0;
}

void materialize_tensor_f32(T* t) {
    qt_materialize_f32(t);
    widen_f16_tensor_to_f32(t);
}

static int q_resize_nearest2d_quantized(T* in, T* out) {
    QTensor* qi = qt_ensure_quantized(in);
    QTensor* qo = qt_get(out);
    if (!qi || !qo || !qo->has_params || in->ndim != 4 || out->ndim != 4 ||
        in->shape[0] <= 0 || in->shape[0] != out->shape[0]) return 0;
    if (in->shape[3] != out->shape[3]) return 0;
    if (qt_alloc_for(out, qo->scale, qo->zp) != 0) return -1;
    qo = qt_get(out);
    int c = in->shape[3];
    int h = in->shape[1];
    int w = in->shape[2];
    int oh = out->shape[1];
    int ow = out->shape[2];
    if (c <= 0 || h <= 0 || w <= 0 || oh <= 0 || ow <= 0) return 0;
    int same_q = fabsf(qi->scale - qo->scale) <= 1e-8f && qi->zp == qo->zp;
    long in_batch_stride = (long)h * w * c;
    long out_batch_stride = (long)oh * ow * c;
    for (int batch = 0; batch < in->shape[0]; batch++) {
        const signed char* in_batch = qi->data + (long)batch * in_batch_stride;
        signed char* out_batch = qo->data + (long)batch * out_batch_stride;
        for (int y = 0; y < oh; y++) {
            int sy = (int)((long)y * h / oh);
            for (int x = 0; x < ow; x++) {
                int sx = (int)((long)x * w / ow);
                const signed char* src = in_batch + ((long)sy * w + sx) * c;
                signed char* dst = out_batch + ((long)y * ow + x) * c;
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
    }
    qo->valid = 1;
    return 1;
}

static int q_maxpool2d_quantized(T* in, T* out, int ky, int kx, int sy, int sx, int py, int px) {
    QTensor* qi = qt_ensure_quantized(in);
    QTensor* qo = qt_get(out);
    if (!qi || !qo || !qo->has_params || in->ndim != 4 || out->ndim != 4 ||
        in->shape[0] <= 0 || in->shape[0] != out->shape[0]) return 0;
    if (in->shape[3] != out->shape[3]) return 0;
    if (qt_alloc_for(out, qo->scale, qo->zp) != 0) return -1;
    qo = qt_get(out);
    int c = in->shape[3];
    int h = in->shape[1];
    int w = in->shape[2];
    int oh = out->shape[1];
    int ow = out->shape[2];
    if (c <= 0 || h <= 0 || w <= 0 || oh <= 0 || ow <= 0 ||
        ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0) return 0;
    int same_q = fabsf(qi->scale - qo->scale) <= 1e-8f && qi->zp == qo->zp;
    long in_batch_stride = (long)h * w * c;
    long out_batch_stride = (long)oh * ow * c;
    for (int batch = 0; batch < in->shape[0]; batch++) {
        const signed char* in_batch = qi->data + (long)batch * in_batch_stride;
        signed char* out_batch = qo->data + (long)batch * out_batch_stride;
        if (same_q && vx_qmaxpool_i8_sameq(in_batch, out_batch, c, h, w, oh, ow,
                                           ky, kx, sy, sx, py, px, qi->zp)) {
            continue;
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
                            int qv = in_batch[((long)iy * w + ix) * c + ch];
                            if (qv > best) best = qv;
                        }
                    }
                    if (best < -128) best = qi->zp;
                    if (same_q) out_batch[((long)oy * ow + ox) * c + ch] = (signed char)best;
                    else {
                        float v = (best - qi->zp) * qi->scale;
                        out_batch[((long)oy * ow + ox) * c + ch] = quantize_scalar_i8(v, qo->scale, qo->zp);
                    }
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
    QTensor* qi = qt_get(in);
    int source_is_qtensor = qi && qi->valid && qi->has_params;
    if (source_is_qtensor) {
        /* The F32 backing is stale after an integer producer. Materializing
           yields real values, so do not apply the source quantization a
           second time when requantizing to `out`. */
        materialize_tensor_f32(in);
    }
    if (qt_alloc_for(out, output_scale, output_zp) != 0) return -1;
    QTensor* qo = qt_get(out);
    if (!qo) return -1;
    vx_quantize_f32_to_i8(in->data, qo->data, in->numel,
                          source_is_qtensor ? 0.0f : input_scale,
                          source_is_qtensor ? 0 : input_zp,
                          output_scale, output_zp);
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

static int tensor_dtype_from_safetensor(SafetensorsDType dtype) {
    switch (dtype) {
        case SAFETENSORS_DTYPE_F32: return T_F32;
        case SAFETENSORS_DTYPE_F16: return T_F16;
        case SAFETENSORS_DTYPE_I8: return T_I8;
        case SAFETENSORS_DTYPE_U8: return T_U8;
        case SAFETENSORS_DTYPE_I32: return T_I32;
        case SAFETENSORS_DTYPE_BOOL:
        case SAFETENSORS_DTYPE_F4:
        case SAFETENSORS_DTYPE_F6_E2M3:
        case SAFETENSORS_DTYPE_F6_E3M2:
        case SAFETENSORS_DTYPE_F8_E5M2:
        case SAFETENSORS_DTYPE_F8_E4M3:
        case SAFETENSORS_DTYPE_F8_E8M0:
        case SAFETENSORS_DTYPE_F8_E4M3FNUZ:
        case SAFETENSORS_DTYPE_F8_E5M2FNUZ:
        case SAFETENSORS_DTYPE_I16:
        case SAFETENSORS_DTYPE_U16:
        case SAFETENSORS_DTYPE_BF16:
        case SAFETENSORS_DTYPE_U32:
        case SAFETENSORS_DTYPE_C64:
        case SAFETENSORS_DTYPE_F64:
        case SAFETENSORS_DTYPE_I64:
        case SAFETENSORS_DTYPE_U64:
        case SAFETENSORS_DTYPE_UNKNOWN:
        default: return -1;
    }
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
    if (t->dtype == T_F32) {
        float value;
        memcpy(&value, (const unsigned char*)t->data + (size_t)idx * sizeof(float), sizeof(value));
        return value;
    }
    return 0.0f;
}

static float tensor_scalar_any(const T* t, long idx) {
    if (!t || idx < 0 || idx >= t->numel) return 0.0f;
    if (t->dtype == T_F32 || t->dtype == T_F16) return tensor_scalar_f32(t, idx);
    if (t->dtype == T_I8) return (float)((const int8_t*)t->data)[idx];
    if (t->dtype == T_U8) return (float)((const uint8_t*)t->data)[idx];
    if (t->dtype == T_I32) {
        int32_t value;
        memcpy(&value, (const unsigned char*)t->data + (size_t)idx * sizeof(value), sizeof(value));
        return (float)value;
    }
    return 0.0f;
}

/* CPU fallback may widen an owned F16 activation into a new F32 allocation.
 * Public backends are allowed to retain the old host pointer as an opaque
 * device-storage key until reset/teardown, so defer freeing replaced storage
 * until that lifetime boundary.  A tensor widens at most once. */
static void* g_retired_f16_storage[MAXT];
static int g_retired_f16_storage_count;

static void release_retired_f16_storage(void) {
    for (int index = 0; index < g_retired_f16_storage_count; index++)
        free(g_retired_f16_storage[index]);
    memset(g_retired_f16_storage, 0, sizeof(g_retired_f16_storage));
    g_retired_f16_storage_count = 0;
}

static int widen_f16_tensor_to_f32(T* t) {
    if (!t || t->dtype != T_F16) return 0;
    float* dst = (float*)malloc((size_t)t->numel * sizeof(float));
    if (!dst) return -1;
    for (long i = 0; i < t->numel; i++) dst[i] = f16_to_f32_scalar(f16_load_bits(t->data, i));
    if (t->owns) {
        if (g_retired_f16_storage_count >= MAXT) {
            free(dst);
            return -1;
        }
        g_retired_f16_storage[g_retired_f16_storage_count++] = t->data;
    }
    t->data = dst;
    t->dtype = T_F32;
    t->elem_size = sizeof(float);
    t->owns = 1;
    return 1;
}

static T* t_add_ex(const char* name, const int* shape, int ndim, void* data, int dtype) {
    long numel = numel_of(shape, ndim);
    size_t elem_size = dtype_size(dtype);
    if (!name || !name[0] || strlen(name) >= sizeof(g_t[0].name) ||
        g_nt >= MAXT || numel < 0 || !elem_size ||
        (size_t)numel > SIZE_MAX / elem_size) return NULL;
    void* storage = data;
    if (!storage) {
        storage = calloc((size_t)(numel > 0 ? numel : 1), elem_size);
        if (!storage) return NULL;
    }
    T* t = &g_t[g_nt];
    memset(t, 0, sizeof(*t));
    memcpy(t->name, name, strlen(name) + 1);
    t->ndim = ndim; for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    t->numel = numel;
    t->dtype = dtype;
    t->elem_size = elem_size;
    t->owns = data ? 0 : 1;
    t->data = (float*)storage;
    g_nt++;
    volvoxai_engine_tensor_name_index_add(g_nt - 1);
    return t;
}

static T* t_add(const char* name, const int* shape, int ndim, float* data) {
    return t_add_ex(name, shape, ndim, data, T_F32);
}

// Row window [r0, r0+rc) over a tensor whose leading (batch-folded) dim is the sequence.
static void seq_range(long total_rows, int* r0, int* rc) {
    if (g_active_row >= 0) { *r0 = g_active_row; *rc = 1; }
    else { *r0 = 0; *rc = (g_prefix_rows > 0 && g_prefix_rows < total_rows) ? g_prefix_rows : (int)total_rows; }
}

// Incremental attention: one query row attends over cached K/V [0..len).
static int sdpa_cached_row(const float* q, const float* kc, const float* vc, int len,
                       float* out, int d_model, int heads, int head_dim, float scale) {
    if (!q || !kc || !vc || !out || len <= 0 || d_model <= 0 || heads <= 0 ||
        head_dim <= 0 || (size_t)len > SIZE_MAX / sizeof(float)) return -1;
    float* scores = (float*)malloc((size_t)len * sizeof(float));
    if (!scores) return -1;
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
    free(scores);
    return 0;
}

char* read_file(const char* path, long* out_size) {
    if (out_size) *out_size = 0;
    if (!path || !path[0]) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[engine] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    buf[sz] = 0; fclose(f); if (out_size) *out_size = sz; return buf;
}

// ---- safetensors: load weights; F16 tensors remain half-precision in the blob ----
int volvoxai_engine_load_weight_files(const char* const* paths, int count) {
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    if (count < 0 || count > MAX_WEIGHT_FILES || (count > 0 && !paths)) return -1;
    for (int i = 0; i < g_weight_file_count; i++) {
        safetensors_free(&g_weight_files[i]);
        g_weight_paths[i][0] = 0;
    }
    g_weight_file_count = 0;
    g_blob = NULL;

    if (count == 0) {
        if (g_debug) {
            fprintf(stderr, "[debug] load_weights files=0 tensors=0 bytes=0 %.3f ms\n",
                    volvoxai_engine_now_ms() - t0);
        }
        return 0;
    }

    int weight_count = 0;
    int override_count = 0;
    int f16_loaded = 0;
    long total_size = 0;
    for (int f = 0; f < count; f++) {
        if (!paths[f] || !paths[f][0]) return -1;
        SafetensorsLoadOptions options = { SAFETENSORS_OPEN_READ_WRITE };
        SafetensorsFile weights;
        if (safetensors_load_with_options(paths[f], &options, &weights) != 0) return -1;
        for (int i = 0; i < weights.tensor_count; i++) {
            if (g_nt >= MAXT) {
                fprintf(stderr, "[engine] too many tensors in safetensors files\n");
                safetensors_free(&weights);
                return -1;
            }
            const SafetensorsTensor* tensor = &weights.tensors[i];
            int dtype = tensor_dtype_from_safetensor(tensor->dtype);
            if (dtype < 0) {
                fprintf(stderr, "[engine] unsupported safetensors dtype %s for tensor %s\n",
                        safetensors_dtype_name(tensor->dtype), tensor->name);
                safetensors_free(&weights);
                return -1;
            }
            if (dtype == T_F16) f16_loaded++;
            T* existing = t_find(tensor->name);
            if (existing) {
                if (existing->owns && existing->data) free(existing->data);
                existing->ndim = tensor->ndim;
                for (int d = 0; d < tensor->ndim && d < 8; d++) existing->shape[d] = tensor->shape[d];
                existing->numel = numel_of(existing->shape, existing->ndim);
                existing->dtype = dtype;
                existing->elem_size = dtype_size(dtype);
                existing->owns = 0;
                existing->data = (float*)tensor->data;
                override_count++;
            } else {
                if (!t_add_ex(tensor->name, tensor->shape, tensor->ndim, tensor->data, dtype)) {
                    safetensors_free(&weights);
                    return -1;
                }
            }
            weight_count++;
        }
        total_size += weights.size;
        g_weight_files[g_weight_file_count] = weights;
        strncpy(g_weight_paths[g_weight_file_count], paths[f], sizeof(g_weight_paths[g_weight_file_count]) - 1);
        g_weight_paths[g_weight_file_count][sizeof(g_weight_paths[g_weight_file_count]) - 1] = 0;
        if (!g_blob) g_blob = weights.blob;
        g_weight_file_count++;
    }
    if (g_debug) {
        fprintf(stderr, "[debug] load_weights files=%d tensors=%d overrides=%d f16_loaded=%d bytes=%ld %.3f ms\n",
                count, weight_count, override_count, f16_loaded, total_size, volvoxai_engine_now_ms() - t0);
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

typedef struct {
    float input_scale;
    int input_zp;
    float output_scale;
    int output_zp;
} QLinearParams;

/* Existing I8-weight Linear nodes are W8A32 unless all four activation/output
 * parameters are present. Requiring the complete set keeps this native-only
 * W8A8 extension explicit instead of guessing graph semantics. */
static int qlinear_params(const Node* n, QLinearParams* out) {
    if (!n || !out) return -1;
    cJSON* p = n->params;
    cJSON* input_scale = p ? cJSON_GetObjectItem(p, "input_scale") : NULL;
    cJSON* input_zp = p ? cJSON_GetObjectItem(p, "input_zero_point") : NULL;
    cJSON* output_scale = p ? cJSON_GetObjectItem(p, "output_scale") : NULL;
    cJSON* output_zp = p ? cJSON_GetObjectItem(p, "output_zero_point") : NULL;
    int present = !!input_scale + !!input_zp + !!output_scale + !!output_zp;
    if (!present) return 0;
    if (present != 4 || !cJSON_IsNumber(input_scale) || !cJSON_IsNumber(input_zp) ||
        !cJSON_IsNumber(output_scale) || !cJSON_IsNumber(output_zp)) return -1;
    if (!isfinite(input_scale->valuedouble) || !isfinite(output_scale->valuedouble) ||
        input_scale->valuedouble <= 0.0 || output_scale->valuedouble <= 0.0 ||
        input_zp->valuedouble != trunc(input_zp->valuedouble) ||
        output_zp->valuedouble != trunc(output_zp->valuedouble) ||
        input_zp->valuedouble < -128.0 || input_zp->valuedouble > 127.0 ||
        output_zp->valuedouble < -128.0 || output_zp->valuedouble > 127.0) return -1;
    out->input_scale = (float)input_scale->valuedouble;
    out->input_zp = input_zp->valueint;
    out->output_scale = (float)output_scale->valuedouble;
    out->output_zp = output_zp->valueint;
    return isfinite(out->input_scale) && out->input_scale > 0.0f &&
           isfinite(out->output_scale) && out->output_scale > 0.0f ? 1 : -1;
}

static int gelu_approximate_tanh(cJSON* params, int* out) {
    if (!out) return -1;
    cJSON* value = params ? cJSON_GetObjectItem(params, "approximate") : NULL;
    if (!value) { *out = 0; return 0; }
    if (!cJSON_IsString(value) || !value->valuestring) return -1;
    if (!strcmp(value->valuestring, "none")) { *out = 0; return 0; }
    if (!strcmp(value->valuestring, "tanh")) { *out = 1; return 0; }
    return -1;
}
static T* nin(Node* n, const char* key) {
    for (int i = 0; i < n->nin; i++) if (strcmp(n->ins[i].key, key) == 0) return t_find(n->ins[i].name);
    return NULL;
}
static T* nout(Node* n, const char* key) {
    for (int i = 0; i < n->nout; i++) if (strcmp(n->outs[i].key, key) == 0) return t_find(n->outs[i].name);
    return NULL;
}

static int native_qlinear_op(const char* op) {
    return op && (!strcmp(op, "QLinear") || !strcmp(op, "QMatMul") || !strcmp(op, "QGemm"));
}

static int native_qconv2d_op(const char* op) {
    return op && !strcmp(op, "QConv2D");
}

static int native_qembedding_op(const char* op) {
    return op && !strcmp(op, "QEmbedding");
}

static uint32_t physical_byte_portable_dtype(int dtype) {
    if (dtype == T_I8) return 2u;
    if (dtype == T_U8) return 3u;
    return 0u;
}

static T* qlinear_node_input(Node* n) {
    T* input = nin(n, "input");
    if (!input) input = nin(n, "x");
    if (!input) input = nin(n, "a");
    return input;
}

static T* qconv2d_node_input(Node* n) {
    T* input = nin(n, "input");
    if (!input) input = nin(n, "x");
    return input;
}

static T* qembedding_node_input(Node* n) {
    return nin(n, "input");
}

/* The config representation mirrors ModelBuilder serialization. Activations
 * intentionally support only per-tensor metadata for this first physical-byte
 * CPU path. */
static int parse_per_tensor_quantization(cJSON* descriptor, int dtype,
                                         TensorQuantization* out) {
    TensorQuantization parsed = {0};
    cJSON* scheme;
    cJSON* scale;
    cJSON* zero_point;
    double zero_value = 0.0;
    int minimum, maximum;
    if (!out) return -1;
    if (!descriptor) {
        *out = parsed;
        return 0;
    }
    if (!cJSON_IsObject(descriptor) || (dtype != T_I8 && dtype != T_U8)) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    if (scheme && (!cJSON_IsString(scheme) || !scheme->valuestring ||
                   strcmp(scheme->valuestring, "per_tensor"))) return -1;
    scale = cJSON_GetObjectItem(descriptor, "scale");
    zero_point = cJSON_GetObjectItem(descriptor, "zero_point");
    if (!cJSON_IsNumber(scale) || (zero_point && !cJSON_IsNumber(zero_point)) ||
        !isfinite(scale->valuedouble) || scale->valuedouble <= 0.0) return -1;
    if (zero_point) zero_value = zero_point->valuedouble;
    minimum = dtype == T_I8 ? -128 : 0;
    maximum = dtype == T_I8 ? 127 : 255;
    if (!isfinite(zero_value) || zero_value != trunc(zero_value) ||
        zero_value < (double)minimum || zero_value > (double)maximum) return -1;
    parsed.scale = (float)scale->valuedouble;
    if (!isfinite(parsed.scale) || parsed.scale <= 0.0f) return -1;
    parsed.zero_point = (int)zero_value;
    parsed.valid = 1;
    *out = parsed;
    return 0;
}

typedef struct {
    int companion_backed;
    const T* companion;
    int count;
    cJSON* inline_scales;
} PerAxisScaleSource;

static int json_object_exact_keys(cJSON* object, const char* first,
                                  const char* second) {
    int saw_first = 0;
    int saw_second = second ? 0 : 1;
    if (!cJSON_IsObject(object) || !first) return 0;
    for (cJSON* item = object->child; item; item = item->next) {
        if (!item->string) return 0;
        if (!strcmp(item->string, first) && !saw_first) saw_first = 1;
        else if (second && !strcmp(item->string, second) && !saw_second) saw_second = 1;
        else return 0;
    }
    return saw_first && saw_second;
}

/* Authority keys are exact and case-sensitive. Recursive config validation
 * has already rejected duplicate exact keys; case variants are unrelated
 * extension fields and must not shadow an authority key through cJSON's
 * default case-insensitive lookup. */
static int json_object_authority_item(cJSON* object, const char* name,
                                      cJSON** out) {
    if (!out || !name || !cJSON_IsObject(object)) return -1;
    *out = cJSON_GetObjectItemCaseSensitive(object, name);
    return 0;
}

/* Companion storage is an all-or-nothing graph contract. The format marker is
 * intentionally the only property so future representations require a new
 * version instead of silently changing this one's interpretation. */
static int weight_scale_companion_mode(cJSON* root) {
    cJSON* storage = NULL;
    cJSON* format;
    if (!root) return 0;
    if (json_object_authority_item(
            root, "weights_quantization_storage", &storage) != 0) return -1;
    if (!storage) return 0;
    if (!json_object_exact_keys(storage, "format", NULL)) return -1;
    format = cJSON_GetObjectItemCaseSensitive(storage, "format");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvoxai-f32-companion-scales-v1")) return -1;
    return 1;
}

static float safetensors_f32_scalar(const SafetensorsTensor* tensor, int index) {
    float value = 0.0f;
    if (!tensor || !tensor->data || index < 0) return value;
    memcpy(&value,
           (const unsigned char*)tensor->data + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

/* Companion-mode graph configs deliberately carry only the storage marker.
 * The authoritative descriptor table lives in each safetensors shard's
 * string-only __metadata__ object.  Merge those local manifests into the
 * in-memory config so the established operator metadata path has one source
 * of truth without repeating the table in every graph config. */
static int install_weight_scale_companion_manifest(cJSON* root) {
    cJSON* merged = NULL;
    cJSON* inline_manifest = NULL;
    int descriptor_count = 0;
    int mode = weight_scale_companion_mode(root);
    if (mode <= 0) return mode;
    if (json_object_authority_item(
            root, "weights_quantization", &inline_manifest) != 0 || inline_manifest) return -1;
    merged = cJSON_CreateObject();
    if (!merged) return -1;

    for (int file_index = 0; file_index < g_weight_file_count; file_index++) {
        const SafetensorsFile* file = &g_weight_files[file_index];
        cJSON* metadata = NULL;
        cJSON* storage = NULL;
        cJSON* manifest_text = NULL;
        cJSON* manifest = NULL;
        for (int tensor_index = 0; tensor_index < file->tensor_count;
             tensor_index++) {
            for (int previous = 0; previous < file_index; previous++) {
                if (safetensors_find_tensor(
                        &g_weight_files[previous], file->tensors[tensor_index].name))
                    goto fail;
            }
        }
        if (file->metadata_json) metadata = cJSON_Parse(file->metadata_json);
        if (file->has_metadata && !metadata) goto fail;
        if (metadata && !cJSON_IsObject(metadata)) {
            cJSON_Delete(metadata);
            goto fail;
        }
        if (metadata &&
            (json_object_authority_item(
                 metadata, "weights_quantization_storage", &storage) != 0 ||
             json_object_authority_item(
                 metadata, "weights_quantization", &manifest_text) != 0)) {
            cJSON_Delete(metadata);
            goto fail;
        }
        if (!storage && !manifest_text) {
            cJSON_Delete(metadata);
            continue;
        }
        if (!cJSON_IsString(storage) || !storage->valuestring ||
            strcmp(storage->valuestring, "volvoxai-f32-companion-scales-v1") ||
            !cJSON_IsString(manifest_text) || !manifest_text->valuestring) {
            cJSON_Delete(metadata);
            goto fail;
        }
        manifest = cJSON_Parse(manifest_text->valuestring);
        if (!cJSON_IsObject(manifest) ||
            !vx_json_object_keys_unique_recursive(manifest)) {
            cJSON_Delete(manifest);
            cJSON_Delete(metadata);
            goto fail;
        }

        for (cJSON* descriptor = manifest->child; descriptor;
             descriptor = descriptor->next) {
            const SafetensorsTensor* weight;
            const SafetensorsTensor* companion;
            cJSON* scheme;
            int expected_count;
            int weight_occurrences = 0;
            int companion_occurrences = 0;
            char companion_name[sizeof(((SafetensorsTensor*)0)->name)];
            int companion_length;
            if (!descriptor->string || !descriptor->string[0] ||
                !cJSON_IsObject(descriptor) ||
                cJSON_GetObjectItemCaseSensitive(merged, descriptor->string)) {
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }
            weight = safetensors_find_tensor(file, descriptor->string);
            companion_length = snprintf(companion_name, sizeof(companion_name),
                                        "%s_scale", descriptor->string);
            if (!weight || weight->dtype != SAFETENSORS_DTYPE_I8 ||
                companion_length <= 0 ||
                (size_t)companion_length >= sizeof(companion_name)) {
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }
            companion = safetensors_find_tensor(file, companion_name);
            if (!companion || companion->dtype != SAFETENSORS_DTYPE_F32 ||
                companion->ndim != 1 || !companion->data) {
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }
            for (int other = 0; other < g_weight_file_count; other++) {
                if (safetensors_find_tensor(&g_weight_files[other], descriptor->string))
                    weight_occurrences++;
                if (safetensors_find_tensor(&g_weight_files[other], companion_name))
                    companion_occurrences++;
            }
            if (weight_occurrences != 1 || companion_occurrences != 1) {
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }

            scheme = cJSON_GetObjectItemCaseSensitive(descriptor, "scheme");
            if (!cJSON_IsString(scheme) || !scheme->valuestring) {
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }
            if (!strcmp(scheme->valuestring, "per_axis")) {
                cJSON* axis = cJSON_GetObjectItemCaseSensitive(descriptor, "axis");
                if (!json_object_exact_keys(descriptor, "scheme", "axis") ||
                    !cJSON_IsNumber(axis) || axis->valuedouble != 0.0 ||
                    weight->ndim <= 0 || weight->shape[0] <= 0) {
                    cJSON_Delete(manifest);
                    cJSON_Delete(metadata);
                    goto fail;
                }
                expected_count = weight->shape[0];
            } else if (!strcmp(scheme->valuestring, "per_tensor")) {
                if (!json_object_exact_keys(descriptor, "scheme", NULL)) {
                    cJSON_Delete(manifest);
                    cJSON_Delete(metadata);
                    goto fail;
                }
                expected_count = 1;
            } else {
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }
            if (companion->shape[0] != expected_count) {
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }
            for (int index = 0; index < expected_count; index++) {
                float scale = safetensors_f32_scalar(companion, index);
                if (!isfinite(scale) || scale <= 0.0f) {
                    cJSON_Delete(manifest);
                    cJSON_Delete(metadata);
                    goto fail;
                }
            }
            cJSON* copy = cJSON_Duplicate(descriptor, 1);
            if (!copy) {
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }
            if (!cJSON_AddItemToObject(merged, descriptor->string, copy)) {
                cJSON_Delete(copy);
                cJSON_Delete(manifest);
                cJSON_Delete(metadata);
                goto fail;
            }
            descriptor_count++;
        }
        cJSON_Delete(manifest);
        cJSON_Delete(metadata);
    }
    if (descriptor_count == 0) goto fail;
    if (!cJSON_AddItemToObject(root, "weights_quantization", merged)) goto fail;
    merged = NULL;
    return 0;

fail:
    cJSON_Delete(merged);
    return -1;
}

static int weight_scale_companion(const T* weight, int expected_count,
                                  const T** out) {
    char name[sizeof(((T*)0)->name)];
    int length;
    int occurrences = 0;
    T* companion;
    if (!weight || !out || expected_count <= 0) return -1;
    *out = NULL;
    length = snprintf(name, sizeof(name), "%s_scale", weight->name);
    if (length <= 0 || (size_t)length >= sizeof(name)) return -1;
    for (int file = 0; file < g_weight_file_count; file++) {
        if (safetensors_find_tensor(&g_weight_files[file], name)) occurrences++;
    }
    if (occurrences != 1) return -1;
    companion = t_find(name);
    if (!companion || !companion->data || companion->dtype != T_F32 ||
        companion->elem_size != sizeof(float) || companion->ndim != 1 ||
        companion->shape[0] != expected_count || companion->numel != expected_count ||
        !volvoxai_engine_tensor_is_model_weight_locked(companion->name)) return -1;
    for (int index = 0; index < expected_count; index++) {
        float scale = tensor_scalar_f32(companion, index);
        if (!isfinite(scale) || scale <= 0.0f) return -1;
    }
    *out = companion;
    return 0;
}

int volvoxai_engine_tensor_is_internal_companion_locked(const char* name) {
    cJSON* descriptors;
    if (!name || weight_scale_companion_mode(g_cfg_root) != 1) return 0;
    descriptors = cJSON_GetObjectItemCaseSensitive(g_cfg_root, "weights_quantization");
    for (cJSON* descriptor = cJSON_IsObject(descriptors) ? descriptors->child : NULL;
         descriptor; descriptor = descriptor->next) {
        char companion[sizeof(((T*)0)->name)];
        int length;
        if (!descriptor->string) continue;
        length = snprintf(companion, sizeof(companion), "%s_scale", descriptor->string);
        if (length > 0 && (size_t)length < sizeof(companion) &&
            !strcmp(name, companion)) return 1;
    }
    return 0;
}

static int validate_companions_are_internal(cJSON* root) {
    cJSON* nodes = root ? cJSON_GetObjectItem(root, "nodes") : NULL;
    cJSON* outputs = root ? cJSON_GetObjectItem(root, "outputs") : NULL;
    for (cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        cJSON* inputs = cJSON_GetObjectItem(node, "inputs");
        for (cJSON* input = cJSON_IsObject(inputs) ? inputs->child : NULL;
             input; input = input->next) {
            if (cJSON_IsString(input) && input->valuestring &&
                volvoxai_engine_tensor_is_internal_companion_locked(
                    input->valuestring)) return -1;
        }
    }
    for (cJSON* output = (cJSON_IsArray(outputs) || cJSON_IsObject(outputs))
             ? outputs->child : NULL;
         output; output = output->next) {
        if (cJSON_IsString(output) && output->valuestring &&
            volvoxai_engine_tensor_is_internal_companion_locked(
                output->valuestring)) return -1;
    }
    return 0;
}

static int per_axis_scale_source(cJSON* descriptor, const T* weight,
                                 int expected_count,
                                 PerAxisScaleSource* out) {
    int companion_mode;
    if (!descriptor || !weight || !out || expected_count <= 0) return -1;
    memset(out, 0, sizeof(*out));
    companion_mode = weight_scale_companion_mode(g_cfg_root);
    if (companion_mode < 0) return -1;
    if (companion_mode) {
        if (!json_object_exact_keys(descriptor, "scheme", "axis") ||
            weight_scale_companion(weight, expected_count, &out->companion) != 0)
            return -1;
        out->companion_backed = 1;
        out->count = expected_count;
        return 0;
    }
    out->inline_scales = cJSON_GetObjectItem(descriptor, "scales");
    if (!cJSON_IsArray(out->inline_scales) ||
        cJSON_GetArraySize(out->inline_scales) != expected_count) return -1;
    out->count = expected_count;
    return 0;
}

static int per_axis_scale_at(const PerAxisScaleSource* source, int index,
                             float* out) {
    float value;
    if (!source || !out || index < 0 || index >= source->count) return -1;
    if (source->companion_backed) {
        value = tensor_scalar_f32(source->companion, index);
    } else {
        cJSON* item = cJSON_GetArrayItem(source->inline_scales, index);
        if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
            item->valuedouble <= 0.0) return -1;
        value = (float)item->valuedouble;
    }
    if (!isfinite(value) || value <= 0.0f) return -1;
    *out = value;
    return 0;
}

/* `weights_quantization` normally carries per-axis descriptors for QLinear,
 * QEmbedding, and QConv2D, which each parse their own channel metadata below.
 * It can also describe a byte-typed constant used as an activation operand
 * (for example a quantized positional embedding feeding QAdd).  Unlike an
 * operator-specific weight, that constant needs its immutable per-tensor
 * mapping installed on the loaded tensor itself.  Keep the two cases
 * distinct: validate every declared descriptor here, but only attach the
 * per-tensor form that a generic typed operator can consume. */
static int validate_per_axis_weight_quantization(cJSON* descriptor, const T* tensor) {
    cJSON* scheme;
    cJSON* axis_json;
    PerAxisScaleSource scales;
    cJSON* zero_points;
    int axis;
    int minimum;
    int maximum;
    if (!descriptor || !tensor || !cJSON_IsObject(descriptor) ||
        (tensor->dtype != T_I8 && tensor->dtype != T_U8) || tensor->ndim <= 0) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    axis_json = cJSON_GetObjectItem(descriptor, "axis");
    zero_points = cJSON_GetObjectItem(descriptor, "zero_points");
    if (!cJSON_IsString(scheme) || !scheme->valuestring || strcmp(scheme->valuestring, "per_axis") ||
        !cJSON_IsNumber(axis_json) || axis_json->valuedouble != trunc(axis_json->valuedouble) ||
        axis_json->valuedouble < (double)INT_MIN ||
        axis_json->valuedouble > (double)INT_MAX) return -1;
    axis = axis_json->valueint;
    if (axis < 0) axis += tensor->ndim;
    if (axis < 0 || axis >= tensor->ndim || tensor->shape[axis] <= 0 ||
        per_axis_scale_source(descriptor, tensor, tensor->shape[axis], &scales) != 0 ||
        (zero_points && (!cJSON_IsArray(zero_points) ||
                         cJSON_GetArraySize(zero_points) != tensor->shape[axis]))) return -1;
    minimum = tensor->dtype == T_I8 ? -128 : 0;
    maximum = tensor->dtype == T_I8 ? 127 : 255;
    for (int index = 0; index < tensor->shape[axis]; index++) {
        cJSON* zero = zero_points ? cJSON_GetArrayItem(zero_points, index) : NULL;
        double zero_value = zero ? zero->valuedouble : 0.0;
        float scale_value;
        if ((zero && !cJSON_IsNumber(zero)) ||
            !isfinite(zero_value) || zero_value != trunc(zero_value) ||
            zero_value < (double)minimum || zero_value > (double)maximum) return -1;
        if (per_axis_scale_at(&scales, index, &scale_value) != 0) return -1;
    }
    return 0;
}

static int apply_weight_quantization(cJSON* root) {
    cJSON* descriptors = NULL;
    int companion_mode = weight_scale_companion_mode(root);
    if (companion_mode < 0) return -1;
    if (root && json_object_authority_item(
            root, "weights_quantization", &descriptors) != 0) return -1;
    if (!descriptors) return companion_mode ? -1 : 0;
    if (!cJSON_IsObject(descriptors)) return -1;
    if (companion_mode && validate_companions_are_internal(root) != 0)
        return -1;
    for (cJSON* item = descriptors->child; item; item = item->next) {
        T* tensor;
        cJSON* scheme;
        if (!item->string || !item->string[0] || !cJSON_IsObject(item)) return -1;
        tensor = t_find(item->string);
        if (!tensor) return -1;
        scheme = cJSON_GetObjectItem(item, "scheme");
        if (!scheme || !cJSON_IsString(scheme) || !scheme->valuestring) return -1;
        if (!strcmp(scheme->valuestring, "per_tensor")) {
            if (companion_mode) {
                const T* companion;
                float scale;
                if ((tensor->dtype != T_I8 && tensor->dtype != T_U8) ||
                    !json_object_exact_keys(item, "scheme", NULL) ||
                    weight_scale_companion(tensor, 1, &companion) != 0) return -1;
                scale = tensor_scalar_f32(companion, 0);
                tensor->quantization.scale = scale;
                tensor->quantization.zero_point = 0;
                tensor->quantization.valid = 1;
            } else if (parse_per_tensor_quantization(
                           item, tensor->dtype, &tensor->quantization) != 0) return -1;
        } else if (!strcmp(scheme->valuestring, "per_axis")) {
            if (validate_per_axis_weight_quantization(item, tensor) != 0) return -1;
        } else {
            return -1;
        }
    }
    return 0;
}

static int qlinear_metadata_build(int node_index, Node* n) {
    QLinearMetadata parsed = {0};
    T* input;
    T* weight;
    T* bias;
    T* output;
    cJSON* all_weights;
    cJSON* descriptor;
    cJSON* scheme;
    cJSON* axis;
    PerAxisScaleSource scales;
    cJSON* zero_points;
    int d_in, d_out;
    long rows;
    if (node_index < 0 || node_index >= MAXN || !n || !native_qlinear_op(n->op)) return -1;
    input = qlinear_node_input(n);
    weight = nin(n, "weight");
    bias = nin(n, "bias");
    output = t_find(n->out);
    if (!input || !weight || !bias || !output || input->ndim <= 0 ||
        output->ndim != input->ndim || weight->ndim != 2 || bias->ndim != 1 ||
        (input->dtype != T_I8 && input->dtype != T_U8) ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8) || bias->dtype != T_I32 ||
        !input->quantization.valid || !output->quantization.valid) return -1;
    d_in = input->shape[input->ndim - 1];
    d_out = output->shape[output->ndim - 1];
    if (d_in <= 0 || d_out <= 0 || weight->shape[0] != d_out || weight->shape[1] != d_in ||
        weight->numel != (long)d_out * d_in || bias->shape[0] != d_out ||
        bias->numel != d_out) return -1;
    for (int dimension = 0; dimension < input->ndim - 1; dimension++) {
        if (input->shape[dimension] != output->shape[dimension]) return -1;
    }
    rows = input->numel / d_in;
    if (rows <= 0 || input->numel != rows * d_in || output->numel != rows * d_out ||
        (unsigned long)rows > UINT32_MAX || (unsigned)d_in > UINT32_MAX ||
        (unsigned)d_out > UINT32_MAX) return -1;
    all_weights = g_cfg_root
        ? cJSON_GetObjectItemCaseSensitive(g_cfg_root, "weights_quantization") : NULL;
    descriptor = all_weights && cJSON_IsObject(all_weights)
        ? cJSON_GetObjectItemCaseSensitive(all_weights, weight->name) : NULL;
    if (!descriptor || !cJSON_IsObject(descriptor)) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    axis = cJSON_GetObjectItem(descriptor, "axis");
    zero_points = cJSON_GetObjectItem(descriptor, "zero_points");
    if (!cJSON_IsString(scheme) || !scheme->valuestring || strcmp(scheme->valuestring, "per_axis") ||
        !cJSON_IsNumber(axis) || axis->valuedouble != 0.0 ||
        per_axis_scale_source(descriptor, weight, d_out, &scales) != 0 ||
        (zero_points && (!cJSON_IsArray(zero_points) || cJSON_GetArraySize(zero_points) != d_out))) return -1;
    if ((size_t)d_out > SIZE_MAX / sizeof(*parsed.weight_scales) ||
        (size_t)d_out > SIZE_MAX / sizeof(*parsed.weight_zero_points) ||
        (size_t)d_out > SIZE_MAX / sizeof(*parsed.bias)) return -1;
    parsed.weight_scales = (float*)malloc((size_t)d_out * sizeof(*parsed.weight_scales));
    parsed.weight_zero_points = (int32_t*)malloc((size_t)d_out * sizeof(*parsed.weight_zero_points));
    parsed.bias = (int32_t*)malloc((size_t)d_out * sizeof(*parsed.bias));
    if (!parsed.weight_scales || !parsed.weight_zero_points || !parsed.bias) goto fail;
    for (int channel = 0; channel < d_out; channel++) {
        cJSON* zero = zero_points ? cJSON_GetArrayItem(zero_points, channel) : NULL;
        double zero_value = zero ? zero->valuedouble : 0.0;
        int minimum = weight->dtype == T_I8 ? -128 : 0;
        int maximum = weight->dtype == T_I8 ? 127 : 255;
        if ((zero && !cJSON_IsNumber(zero)) ||
            !isfinite(zero_value) || zero_value != trunc(zero_value) ||
            zero_value < (double)minimum || zero_value > (double)maximum) goto fail;
        if (per_axis_scale_at(&scales, channel, &parsed.weight_scales[channel]) != 0)
            goto fail;
        parsed.weight_zero_points[channel] = (int32_t)zero_value;
        memcpy(&parsed.bias[channel], (const unsigned char*)bias->data +
               (size_t)channel * sizeof(parsed.bias[channel]), sizeof(parsed.bias[channel]));
    }
    parsed.valid = 1;
    parsed.d_in = d_in;
    parsed.d_out = d_out;
    parsed.input_dtype = input->dtype;
    parsed.weight_dtype = weight->dtype;
    parsed.output_dtype = output->dtype;
    parsed.packed_weight_bytes = vx_packed_q8_weight_size((uint32_t)d_in,
        (uint32_t)d_out);
    if (parsed.packed_weight_bytes &&
        volvoxai_engine_tensor_is_model_weight_locked(weight->name)) {
        parsed.packed_weight = malloc(parsed.packed_weight_bytes);
        if (!parsed.packed_weight || !vx_pack_q8_weight(parsed.packed_weight,
                parsed.packed_weight_bytes, weight->data, (uint32_t)d_in,
                (uint32_t)d_out, physical_byte_portable_dtype(weight->dtype), 1u)) {
            free(parsed.packed_weight);
            parsed.packed_weight = NULL;
            parsed.packed_weight_bytes = 0;
        }
    }
    g_qlinear_meta[node_index] = parsed;
    return 0;
fail:
    free(parsed.weight_scales);
    free(parsed.weight_zero_points);
    free(parsed.bias);
    free(parsed.packed_weight);
    return -1;
}

/* QEmbedding has no activation quantization: IDs are ordinary I32 values and
 * each table row has its own I8/U8 descriptor. Keep this strict parser apart
 * from QLinear because the per-axis dimension is vocabulary, not output
 * channels with an accumulator bias. */
static int qembedding_metadata_build(int node_index, Node* n) {
    QEmbeddingMetadata parsed = {0};
    T* input;
    T* weight;
    T* output;
    cJSON* all_weights;
    cJSON* descriptor;
    cJSON* scheme;
    cJSON* axis;
    PerAxisScaleSource scales;
    cJSON* zero_points;
    uint64_t weight_elements;
    uint64_t output_elements;
    int input_refs = 0;
    int weight_refs = 0;
    if (node_index < 0 || node_index >= MAXN || !n || !native_qembedding_op(n->op) ||
        n->nin != 2 || n->nout != 1 || strcmp(n->outs[0].key, "out")) return -1;
    for (int input_index = 0; input_index < n->nin; input_index++) {
        if (!strcmp(n->ins[input_index].key, "input")) input_refs++;
        else if (!strcmp(n->ins[input_index].key, "weight")) weight_refs++;
        else return -1;
    }
    if (input_refs != 1 || weight_refs != 1) return -1;
    input = qembedding_node_input(n);
    weight = nin(n, "weight");
    output = t_find(n->out);
    if (!input || !weight || !output || !input->is_graph_input || input->dtype != T_I32 || input->ndim <= 0 ||
        input->numel <= 0 || input->numel > (long)UINT32_MAX ||
        input->elem_size != sizeof(int32_t) || weight->ndim != 2 ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) || weight->elem_size != 1 ||
        output->ndim != input->ndim + 1 ||
        (output->dtype != T_I8 && output->dtype != T_U8) || output->elem_size != 1 ||
        !output->quantization.valid || !isfinite(output->quantization.scale) ||
        output->quantization.scale <= 0.0f ||
        output->quantization.zero_point < (output->dtype == T_I8 ? -128 : 0) ||
        output->quantization.zero_point > (output->dtype == T_I8 ? 127 : 255) ||
        weight->shape[0] <= 0 || weight->shape[1] <= 0) return -1;
    for (int dimension = 0; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0 || output->shape[dimension] != input->shape[dimension])
            return -1;
    }
    if (output->shape[input->ndim] != weight->shape[1]) return -1;
    weight_elements = (uint64_t)(uint32_t)weight->shape[0] * (uint32_t)weight->shape[1];
    output_elements = (uint64_t)input->numel * (uint32_t)weight->shape[1];
    if (weight_elements > (uint64_t)LONG_MAX || output_elements > (uint64_t)LONG_MAX ||
        weight->numel != (long)weight_elements || output->numel != (long)output_elements)
        return -1;
    all_weights = g_cfg_root
        ? cJSON_GetObjectItemCaseSensitive(g_cfg_root, "weights_quantization") : NULL;
    descriptor = all_weights && cJSON_IsObject(all_weights)
        ? cJSON_GetObjectItemCaseSensitive(all_weights, weight->name) : NULL;
    if (!descriptor || !cJSON_IsObject(descriptor)) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    axis = cJSON_GetObjectItem(descriptor, "axis");
    zero_points = cJSON_GetObjectItem(descriptor, "zero_points");
    if (!cJSON_IsString(scheme) || !scheme->valuestring || strcmp(scheme->valuestring, "per_axis") ||
        !cJSON_IsNumber(axis) || axis->valuedouble != 0.0 ||
        per_axis_scale_source(descriptor, weight, weight->shape[0], &scales) != 0 ||
        (zero_points && (!cJSON_IsArray(zero_points) ||
                         cJSON_GetArraySize(zero_points) != weight->shape[0]))) return -1;
    if ((size_t)weight->shape[0] > SIZE_MAX / sizeof(*parsed.weight_scales) ||
        (size_t)weight->shape[0] > SIZE_MAX / sizeof(*parsed.weight_zero_points)) return -1;
    parsed.weight_scales = (float*)malloc((size_t)weight->shape[0] * sizeof(*parsed.weight_scales));
    parsed.weight_zero_points = (int32_t*)malloc((size_t)weight->shape[0] *
                                                  sizeof(*parsed.weight_zero_points));
    if (!parsed.weight_scales || !parsed.weight_zero_points) goto fail;
    for (int row = 0; row < weight->shape[0]; row++) {
        cJSON* zero = zero_points ? cJSON_GetArrayItem(zero_points, row) : NULL;
        double zero_value = zero ? zero->valuedouble : 0.0;
        int minimum = weight->dtype == T_I8 ? -128 : 0;
        int maximum = weight->dtype == T_I8 ? 127 : 255;
        if ((zero && !cJSON_IsNumber(zero)) ||
            !isfinite(zero_value) || zero_value != trunc(zero_value) ||
            zero_value < (double)minimum || zero_value > (double)maximum) goto fail;
        if (per_axis_scale_at(&scales, row, &parsed.weight_scales[row]) != 0)
            goto fail;
        parsed.weight_zero_points[row] = (int32_t)zero_value;
    }
    parsed.valid = 1;
    parsed.vocab = (uint32_t)weight->shape[0];
    parsed.hidden = (uint32_t)weight->shape[1];
    parsed.weight_dtype = weight->dtype;
    parsed.output_dtype = output->dtype;
    g_qembedding_meta[node_index] = parsed;
    return 0;
fail:
    free(parsed.weight_scales);
    free(parsed.weight_zero_points);
    return -1;
}

static int qconv2d_parse_u32(cJSON* value, uint32_t minimum, uint32_t* out) {
    double number;
    if (!value || !out || !cJSON_IsNumber(value)) return -1;
    number = value->valuedouble;
    if (!isfinite(number) || number != trunc(number) || number < (double)minimum ||
        number > (double)UINT32_MAX) return -1;
    *out = (uint32_t)number;
    return 0;
}

static int qconv2d_pair_parameter(cJSON* params, const char* name,
                                  uint32_t fallback, uint32_t minimum,
                                  uint32_t values[2]) {
    cJSON* value = params ? cJSON_GetObjectItem(params, name) : NULL;
    if (!value || cJSON_IsNull(value)) {
        values[0] = fallback;
        values[1] = fallback;
        return 0;
    }
    if (cJSON_IsArray(value)) {
        int count = cJSON_GetArraySize(value);
        if (count < 1 || count > 2 ||
            qconv2d_parse_u32(cJSON_GetArrayItem(value, 0), minimum, &values[0]) != 0) return -1;
        if (count == 1) values[1] = values[0];
        else if (qconv2d_parse_u32(cJSON_GetArrayItem(value, 1), minimum, &values[1]) != 0) return -1;
        return 0;
    }
    if (qconv2d_parse_u32(value, minimum, &values[0]) != 0) return -1;
    values[1] = values[0];
    return 0;
}

static int qconv2d_pads_parameter(cJSON* params, const uint32_t padding[2],
                                  uint32_t pads[4]) {
    cJSON* value = params ? cJSON_GetObjectItem(params, "pads") : NULL;
    if (!value || cJSON_IsNull(value)) {
        pads[0] = padding[0];
        pads[1] = padding[1];
        pads[2] = padding[0];
        pads[3] = padding[1];
        return 0;
    }
    if (!cJSON_IsArray(value) || cJSON_GetArraySize(value) != 4) return -1;
    for (int i = 0; i < 4; i++) {
        if (qconv2d_parse_u32(cJSON_GetArrayItem(value, i), 0, &pads[i]) != 0) return -1;
    }
    return 0;
}

static int qconv2d_layout_parameter(cJSON* params, const char* name, const char* expected) {
    cJSON* value = params ? cJSON_GetObjectItem(params, name) : NULL;
    if (!value || cJSON_IsNull(value)) return 0;
    if (!cJSON_IsString(value) || !value->valuestring) return -1;
    return value->valuestring[0] && strcmp(value->valuestring, expected) ? -1 : 0;
}

static int qconv2d_product4(const uint32_t dimensions[4], uint64_t* product) {
    uint64_t value = 1;
    if (!dimensions || !product) return -1;
    for (int i = 0; i < 4; i++) {
        if (!dimensions[i] || value > UINT64_MAX / dimensions[i]) return -1;
        value *= dimensions[i];
    }
    *product = value;
    return 0;
}

static int qconv2d_shape4(const T* tensor, uint32_t dimensions[4]) {
    uint64_t elements;
    if (!tensor || !dimensions || tensor->ndim != 4 || tensor->numel <= 0) return -1;
    for (int i = 0; i < 4; i++) {
        if (tensor->shape[i] <= 0) return -1;
        dimensions[i] = (uint32_t)tensor->shape[i];
    }
    if (qconv2d_product4(dimensions, &elements) != 0 || elements > (uint64_t)LONG_MAX ||
        tensor->numel != (long)elements) return -1;
    return 0;
}

static int qconv2d_weight_only_parameter(const Node* node, int* enabled) {
    cJSON* value;
    if (!node || !enabled) return -1;
    value = node->params ? cJSON_GetObjectItem(node->params, "weight_only") : NULL;
    if (!value || cJSON_IsNull(value)) {
        *enabled = 0;
        return 0;
    }
    if (!cJSON_IsBool(value)) return -1;
    *enabled = cJSON_IsTrue(value) ? 1 : 0;
    return 0;
}

/* The weight-only spelling owns a persistent dequantized-weight cache, so all
 * quantization operands must be immutable model tensors and the complete
 * NHWC/OHWI convolution contract must be known before the first forward. */
static int qconv2d_weight_only_validate(Node* node) {
    T* input;
    T* weight;
    T* weight_scale;
    T* weight_zero_point;
    T* bias;
    T* output;
    T* declared_output;
    uint32_t input_shape[4], weight_shape[4], output_shape[4];
    uint32_t stride[2], dilation[2], padding[2], pads[4], groups, relu;
    uint64_t padded_height, padded_width, effective_height, effective_width;
    uint64_t expected_height, expected_width;
    cJSON* data_layout;
    cJSON* weight_layout;
    int input_refs = 0, weight_refs = 0, scale_refs = 0;
    int zero_point_refs = 0, bias_refs = 0;
    if (!node || strcmp(node->op, "QConv2D") || node->nout != 1 ||
        strcmp(node->outs[0].key, "out")) return -1;
    for (int index = 0; index < node->nin; index++) {
        const char* key = node->ins[index].key;
        if (!strcmp(key, "input")) input_refs++;
        else if (!strcmp(key, "weight")) weight_refs++;
        else if (!strcmp(key, "weight_scale")) scale_refs++;
        else if (!strcmp(key, "weight_zero_point")) zero_point_refs++;
        else if (!strcmp(key, "bias")) bias_refs++;
        else return -1;
    }
    if (input_refs != 1 || weight_refs != 1 || scale_refs != 1 ||
        zero_point_refs > 1 || bias_refs > 1) return -1;
    input = nin(node, "input");
    weight = nin(node, "weight");
    weight_scale = nin(node, "weight_scale");
    weight_zero_point = nin(node, "weight_zero_point");
    bias = nin(node, "bias");
    output = t_find(node->out);
    declared_output = nout(node, "out");
    if (!input || !weight || !weight_scale || !output || !declared_output ||
        !input->data || !weight->data || !weight_scale->data || !output->data ||
        !declared_output->data || input->dtype != T_F32 || output->dtype != T_F32 ||
        declared_output->dtype != T_F32 || input->elem_size != sizeof(float) ||
        output->elem_size != sizeof(float) || declared_output->elem_size != sizeof(float) ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) || weight->elem_size != 1 ||
        weight_scale->dtype != T_F32 || weight_scale->elem_size != sizeof(float) ||
        weight_scale->ndim != 1 || input->quantization.valid ||
        output->quantization.valid || declared_output->quantization.valid ||
        qconv2d_shape4(input, input_shape) != 0 ||
        qconv2d_shape4(weight, weight_shape) != 0 ||
        qconv2d_shape4(output, output_shape) != 0 ||
        !volvoxai_engine_tensor_is_model_weight_locked(weight->name) ||
        !volvoxai_engine_tensor_is_model_weight_locked(weight_scale->name)) return -1;
    if (output != declared_output) {
        uint32_t declared_shape[4];
        if (qconv2d_shape4(declared_output, declared_shape) != 0) return -1;
        for (int dimension = 0; dimension < 4; dimension++)
            if (declared_shape[dimension] != output_shape[dimension]) return -1;
    }
    if (weight_scale->numel != 1 && weight_scale->numel != weight_shape[0]) return -1;
    for (long index = 0; index < weight_scale->numel; index++) {
        float value = tensor_scalar_f32(weight_scale, index);
        if (!isfinite(value) || value <= 0.0f) return -1;
    }
    if (weight_zero_point) {
        int minimum = weight->dtype == T_I8 ? -128 : 0;
        int maximum = weight->dtype == T_I8 ? 127 : 255;
        if (!weight_zero_point->data || weight_zero_point->ndim != 1 ||
            (weight_zero_point->dtype != T_I8 && weight_zero_point->dtype != T_U8 &&
             weight_zero_point->dtype != T_I32) ||
            (weight_zero_point->numel != 1 &&
             weight_zero_point->numel != weight_shape[0]) ||
            !volvoxai_engine_tensor_is_model_weight_locked(weight_zero_point->name)) return -1;
        for (long index = 0; index < weight_zero_point->numel; index++) {
            float value = tensor_scalar_any(weight_zero_point, index);
            if (!isfinite(value) || value != truncf(value) ||
                value < (float)minimum || value > (float)maximum) return -1;
        }
    }
    if (bias) {
        if (!bias->data || bias->dtype != T_F32 || bias->elem_size != sizeof(float) ||
            bias->ndim != 1 || bias->numel != weight_shape[0] ||
            !volvoxai_engine_tensor_is_model_weight_locked(bias->name)) return -1;
        for (long index = 0; index < bias->numel; index++)
            if (!isfinite(tensor_scalar_f32(bias, index))) return -1;
    }
    data_layout = node->params ? cJSON_GetObjectItem(node->params, "data_layout") : NULL;
    weight_layout = node->params ? cJSON_GetObjectItem(node->params, "weight_layout") : NULL;
    if ((data_layout && !cJSON_IsNull(data_layout) &&
         (!cJSON_IsString(data_layout) || !data_layout->valuestring ||
          strcmp(data_layout->valuestring, "NHWC"))) ||
        (weight_layout && !cJSON_IsNull(weight_layout) &&
         (!cJSON_IsString(weight_layout) || !weight_layout->valuestring ||
          strcmp(weight_layout->valuestring, "OHWI")))) return -1;
    if (qconv2d_layout_parameter(node->params, "data_layout", "NHWC") != 0 ||
        qconv2d_layout_parameter(node->params, "weight_layout", "OHWI") != 0 ||
        qconv2d_pair_parameter(node->params, "stride", 1, 1, stride) != 0 ||
        qconv2d_pair_parameter(node->params, "dilation", 1, 1, dilation) != 0 ||
        qconv2d_pair_parameter(node->params, "padding", 0, 0, padding) != 0 ||
        qconv2d_pads_parameter(node->params, padding, pads) != 0) return -1;
    {
        cJSON* groups_value = node->params ? cJSON_GetObjectItem(node->params, "groups") : NULL;
        cJSON* relu_value = node->params ? cJSON_GetObjectItem(node->params, "relu") : NULL;
        if (!groups_value || cJSON_IsNull(groups_value)) groups = 1;
        else if (qconv2d_parse_u32(groups_value, 1, &groups) != 0) return -1;
        if (!relu_value || cJSON_IsNull(relu_value)) relu = 0;
        else if (qconv2d_parse_u32(relu_value, 0, &relu) != 0 || relu > 2) return -1;
    }
    if (groups > INT_MAX || stride[0] > INT_MAX || stride[1] > INT_MAX ||
        dilation[0] > INT_MAX || dilation[1] > INT_MAX ||
        pads[0] > INT_MAX || pads[1] > INT_MAX || pads[2] > INT_MAX ||
        pads[3] > INT_MAX || output_shape[0] != input_shape[0] ||
        weight_shape[0] != output_shape[3] || input_shape[3] % groups ||
        output_shape[3] % groups ||
        (uint64_t)weight_shape[3] * groups != input_shape[3]) return -1;
    padded_height = (uint64_t)input_shape[1] + pads[0] + pads[2];
    padded_width = (uint64_t)input_shape[2] + pads[1] + pads[3];
    effective_height = (uint64_t)(weight_shape[1] - 1u) * dilation[0] + 1u;
    effective_width = (uint64_t)(weight_shape[2] - 1u) * dilation[1] + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return -1;
    expected_height = (padded_height - effective_height) / stride[0] + 1u;
    expected_width = (padded_width - effective_width) / stride[1] + 1u;
    return expected_height == output_shape[1] && expected_width == output_shape[2] ? 0 : -1;
}

static int qconv2d_weight_quantization_build(const T* weight, uint32_t output_channels,
                                             float** scales_out, int32_t** zero_points_out) {
    cJSON* all_weights;
    cJSON* descriptor;
    cJSON* scheme;
    cJSON* axis;
    PerAxisScaleSource scales;
    cJSON* zero_points;
    float* parsed_scales = NULL;
    int32_t* parsed_zero_points = NULL;
    if (!weight || !scales_out || !zero_points_out || !output_channels ||
        output_channels > (uint32_t)INT_MAX) return -1;
    *scales_out = NULL;
    *zero_points_out = NULL;
    all_weights = g_cfg_root
        ? cJSON_GetObjectItemCaseSensitive(g_cfg_root, "weights_quantization") : NULL;
    descriptor = all_weights && cJSON_IsObject(all_weights)
        ? cJSON_GetObjectItemCaseSensitive(all_weights, weight->name) : NULL;
    if (!descriptor || !cJSON_IsObject(descriptor)) return -1;
    scheme = cJSON_GetObjectItem(descriptor, "scheme");
    axis = cJSON_GetObjectItem(descriptor, "axis");
    zero_points = cJSON_GetObjectItem(descriptor, "zero_points");
    if (!cJSON_IsString(scheme) || !scheme->valuestring || strcmp(scheme->valuestring, "per_axis") ||
        !cJSON_IsNumber(axis) || axis->valuedouble != 0.0 ||
        per_axis_scale_source(descriptor, weight, (int)output_channels, &scales) != 0 ||
        (zero_points && (!cJSON_IsArray(zero_points) ||
                         cJSON_GetArraySize(zero_points) != (int)output_channels))) return -1;
    if ((size_t)output_channels > SIZE_MAX / sizeof(*parsed_scales) ||
        (size_t)output_channels > SIZE_MAX / sizeof(*parsed_zero_points)) return -1;
    parsed_scales = (float*)malloc((size_t)output_channels * sizeof(*parsed_scales));
    parsed_zero_points = (int32_t*)malloc((size_t)output_channels * sizeof(*parsed_zero_points));
    if (!parsed_scales || !parsed_zero_points) goto fail;
    for (uint32_t channel = 0; channel < output_channels; channel++) {
        cJSON* zero = zero_points ? cJSON_GetArrayItem(zero_points, (int)channel) : NULL;
        double zero_value = zero ? zero->valuedouble : 0.0;
        int minimum = weight->dtype == T_I8 ? -128 : 0;
        int maximum = weight->dtype == T_I8 ? 127 : 255;
        if ((zero && !cJSON_IsNumber(zero)) ||
            !isfinite(zero_value) || zero_value != trunc(zero_value) ||
            zero_value < (double)minimum || zero_value > (double)maximum) goto fail;
        if (per_axis_scale_at(&scales, (int)channel, &parsed_scales[channel]) != 0)
            goto fail;
        parsed_zero_points[channel] = (int32_t)zero_value;
    }
    *scales_out = parsed_scales;
    *zero_points_out = parsed_zero_points;
    return 0;
fail:
    free(parsed_scales);
    free(parsed_zero_points);
    return -1;
}

/* Returns 1 for a complete physical-byte descriptor, 0 when no physical-byte
 * descriptor is present, and -1 for a malformed typed QConv2D. Callers may
 * separately admit explicit weight-only nodes, but must reject every other
 * zero result rather than inventing activation quantizers. */
static int qconv2d_metadata_build(int node_index, Node* n) {
    QConv2DMetadata parsed = {0};
    T* input;
    T* weight;
    T* bias;
    T* output;
    uint32_t input_shape[4], weight_shape[4], output_shape[4];
    uint32_t stride[2], dilation[2], padding[2], pads[4];
    uint64_t expected_weight_elements;
    uint64_t padded_height, padded_width, effective_height, effective_width;
    uint64_t expected_height, expected_width;
    int physical_hint;
    if (node_index < 0 || node_index >= MAXN || !n || !native_qconv2d_op(n->op)) return -1;
    input = qconv2d_node_input(n);
    weight = nin(n, "weight");
    bias = nin(n, "bias");
    output = t_find(n->out);
    physical_hint = (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                               input->quantization.valid)) ||
                    (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                                output->quantization.valid));
    if (!physical_hint) return 0;
    if (!input || !weight || !output || !input->data || !weight->data || !output->data ||
        (input->dtype != T_I8 && input->dtype != T_U8) ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) ||
        (output->dtype != T_I8 && output->dtype != T_U8) ||
        input->elem_size != 1 || weight->elem_size != 1 || output->elem_size != 1 ||
        !input->quantization.valid || !output->quantization.valid ||
        qconv2d_shape4(input, input_shape) != 0 ||
        qconv2d_shape4(weight, weight_shape) != 0 ||
        qconv2d_shape4(output, output_shape) != 0) goto fail;
    {
        cJSON* groups = n->params ? cJSON_GetObjectItem(n->params, "groups") : NULL;
        if (!groups || cJSON_IsNull(groups)) parsed.groups = 1;
        else if (qconv2d_parse_u32(groups, 1, &parsed.groups) != 0) goto fail;
    }
    if (qconv2d_layout_parameter(n->params, "data_layout", "NHWC") != 0 ||
        qconv2d_layout_parameter(n->params, "weight_layout", "OHWI") != 0 ||
        qconv2d_pair_parameter(n->params, "stride", 1, 1, stride) != 0 ||
        qconv2d_pair_parameter(n->params, "dilation", 1, 1, dilation) != 0 ||
        qconv2d_pair_parameter(n->params, "padding", 0, 0, padding) != 0 ||
        qconv2d_pads_parameter(n->params, padding, pads) != 0) goto fail;
    {
        cJSON* relu = n->params ? cJSON_GetObjectItem(n->params, "relu") : NULL;
        if (!relu || cJSON_IsNull(relu)) parsed.relu = 0;
        else if (qconv2d_parse_u32(relu, 0, &parsed.relu) != 0 || parsed.relu > 2) goto fail;
    }
    if (n->fuse_relu6) parsed.relu = 2;
    parsed.batch = input_shape[0];
    parsed.input_height = input_shape[1];
    parsed.input_width = input_shape[2];
    parsed.input_channels = input_shape[3];
    parsed.output_height = output_shape[1];
    parsed.output_width = output_shape[2];
    parsed.output_channels = output_shape[3];
    parsed.kernel_height = weight_shape[1];
    parsed.kernel_width = weight_shape[2];
    parsed.input_per_group = weight_shape[3];
    parsed.stride_y = stride[0];
    parsed.stride_x = stride[1];
    parsed.dilation_y = dilation[0];
    parsed.dilation_x = dilation[1];
    parsed.padding_top = pads[0];
    parsed.padding_left = pads[1];
    parsed.padding_bottom = pads[2];
    parsed.padding_right = pads[3];
    if (output_shape[0] != parsed.batch || weight_shape[0] != parsed.output_channels ||
        !parsed.groups || parsed.input_channels % parsed.groups ||
        parsed.output_channels % parsed.groups ||
        (uint64_t)parsed.input_per_group * parsed.groups != parsed.input_channels ||
        qconv2d_product4(weight_shape, &expected_weight_elements) != 0 ||
        expected_weight_elements > (uint64_t)LONG_MAX ||
        weight->numel != (long)expected_weight_elements) goto fail;
    padded_height = (uint64_t)parsed.input_height + parsed.padding_top + parsed.padding_bottom;
    padded_width = (uint64_t)parsed.input_width + parsed.padding_left + parsed.padding_right;
    effective_height = (uint64_t)(parsed.kernel_height - 1u) * parsed.dilation_y + 1u;
    effective_width = (uint64_t)(parsed.kernel_width - 1u) * parsed.dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) goto fail;
    expected_height = (padded_height - effective_height) / parsed.stride_y + 1u;
    expected_width = (padded_width - effective_width) / parsed.stride_x + 1u;
    if (expected_height != parsed.output_height || expected_width != parsed.output_width) goto fail;
    if (bias && (!bias->data || bias->dtype != T_I32 || bias->ndim != 1 ||
                 bias->shape[0] != (int)parsed.output_channels ||
                 bias->numel != (long)parsed.output_channels)) goto fail;
    if (qconv2d_weight_quantization_build(weight, parsed.output_channels,
                                          &parsed.weight_scales,
                                          &parsed.weight_zero_points) != 0) goto fail;
    if (bias) {
        if ((size_t)parsed.output_channels > SIZE_MAX / sizeof(*parsed.bias)) goto fail;
        parsed.bias = (int32_t*)malloc((size_t)parsed.output_channels * sizeof(*parsed.bias));
        if (!parsed.bias) goto fail;
        for (uint32_t channel = 0; channel < parsed.output_channels; channel++) {
            memcpy(&parsed.bias[channel], (const unsigned char*)bias->data +
                   (size_t)channel * sizeof(parsed.bias[channel]), sizeof(parsed.bias[channel]));
        }
    }
    parsed.input_dtype = input->dtype;
    parsed.weight_dtype = weight->dtype;
    parsed.output_dtype = output->dtype;
    parsed.valid = 1;
    g_qconv_meta[node_index] = parsed;
    return 1;
fail:
    free(parsed.weight_scales);
    free(parsed.weight_zero_points);
    free(parsed.bias);
    return -1;
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

static int gpu_graph_storage_tensor(const T* t) {
    const VxBackend* sdk_backend;
    if (!t) return 0;
    if (t->dtype == T_F32 || t->dtype == T_I32 ||
        ((t->dtype == T_I8 || t->dtype == T_U8) && t->quantization.valid)) return 1;

    /* Built-in graph backends intentionally retain their historic storage
     * filter.  A public backend, however, may advertise any stable SDK dtype;
     * keep F16 and unquantized byte outputs coherent when a later node falls
     * back to CPU.  Selecting an SDK backend deactivates the built-in device,
     * so this does not create unsupported Vulkan/OpenGL/Metal graph slots. */
    sdk_backend = vx_sdk_selected_backend();
    return sdk_backend && sdk_backend->mark_host && sdk_backend->sync_host &&
           t->dtype >= T_F32 && t->dtype <= T_F16;
}

static void vk_mark_host_tensor(T* t) {
    if (gpu_graph_storage_tensor(t))
        vx_runtime_backend_mark_host(
            t->data, tensor_bytes(t), volvoxai_engine_tensor_is_model_weight_locked(t->name));
}

static void vk_mark_node_outputs_host(Node* node, T* actual_output) {
    T* marked[MAXIN + 1];
    int marked_count = 0;
    if (!node) return;
    /* A pure CPU route has no device ownership state to invalidate.  Avoid
     * resolving every declared output and probing model-weight ownership on
     * each incremental token.  CPU fallbacks inside either a built-in graph
     * or public SDK backend still take the coherence path below. */
    if (!vx_runtime_backend_has_graph() && !vx_sdk_selected_backend()) return;
    if (actual_output) {
        vk_mark_host_tensor(actual_output);
        marked[marked_count++] = actual_output;
    }
    T* primary = t_find(node->out);
    /* A private fusion can write a skipped peer or redirected output while
     * the declared V1 outputs still describe the original graph node.  Mark
     * only the actual write in that case.  Ordinary multi-output nodes retain
     * Node.out == outs[0] and need every declared output invalidated. */
    if (actual_output != primary || node->nout <= 0 ||
        strcmp(node->out, node->outs[0].name)) return;
    for (int output = 0; output < node->nout; output++) {
        T* tensor = t_find(node->outs[output].name);
        int duplicate = 0;
        if (!tensor) continue;
        for (int prior = 0; prior < marked_count; prior++) {
            if (marked[prior] == tensor || marked[prior]->data == tensor->data) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            vk_mark_host_tensor(tensor);
            marked[marked_count++] = tensor;
        }
    }
}

int vk_sync_host_tensor(T* t) {
    return !gpu_graph_storage_tensor(t) ||
           vx_runtime_backend_sync_host(
               t->data, tensor_bytes(t),
               volvoxai_engine_tensor_is_model_weight_locked(t->name)) == 0
        ? 0 : -1;
}

static int vk_sync_node_inputs(Node* n) {
    if (!vx_runtime_backend_has_graph() || !n) return 0;
    for (int i = 0; i < n->nin; i++) {
        T* t = t_find(n->ins[i].name);
        if (vk_sync_host_tensor(t) != 0) return -1;
    }
    return 0;
}

static void materialize_node_inputs(Node* n) {
    if (!n) return;
    for (int i = 0; i < n->nin; i++) {
        T* t = t_find(n->ins[i].name);
        materialize_tensor_f32(t);
    }
}

void vk_mark_owned_tensors_host_dirty(void) {
    if (!vx_runtime_backend_has_graph()) return;
    for (int i = 0; i < g_nt; i++) {
        if (g_t[i].owns && gpu_graph_storage_tensor(&g_t[i])) {
            vx_runtime_backend_mark_host(g_t[i].data, tensor_bytes(&g_t[i]), 0);
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
#include "graph_opt_fusion.inc"
#undef VOLVOX_ENGINE_PRIVATE_INCLUDE
static int build_quant_params_pass(void) {
    volvoxai_engine_clear_qlinear_metadata();
    volvoxai_engine_clear_qconv_metadata();
    volvoxai_engine_clear_qembedding_metadata();
    for (int i = 0; i < g_nn; i++) {
        Node* n = &g_n[i];
        if (!strcmp(n->op, "Add")) {
            T* a = nin(n, "a");
            T* b = nin(n, "b");
            T* output = t_find(n->out);
            int physical_bytes =
                (a && (a->dtype == T_I8 || a->dtype == T_U8 || a->quantization.valid)) ||
                (b && (b->dtype == T_I8 || b->dtype == T_U8 || b->quantization.valid)) ||
                (output && (output->dtype == T_I8 || output->dtype == T_U8 ||
                            output->quantization.valid));
            if (physical_bytes) {
                fprintf(stderr,
                        "[engine] generic Add cannot consume or produce I8/U8 tensors; "
                        "use QAdd with explicit quantization metadata\n");
                goto fail;
            }
        }
        if (native_qlinear_op(n->op)) {
            if (qlinear_metadata_build(i, n) != 0) goto fail;
            continue;
        }
        if (native_qembedding_op(n->op)) {
            if (qembedding_metadata_build(i, n) != 0) goto fail;
            continue;
        }
        if (!strcmp(n->op, "QConv2D")) {
            int weight_only;
            if (qconv2d_weight_only_parameter(n, &weight_only) != 0) goto fail;
            if (weight_only) {
                if (qconv2d_weight_only_validate(n) != 0) goto fail;
                continue;
            }
            int qconv = qconv2d_metadata_build(i, n);
            if (qconv < 0) goto fail;
            if (!qconv) {
                fprintf(stderr,
                        "[engine] QConv2D requires explicit I8/U8 input and output "
                        "quantization metadata or params.weight_only=true\n");
                goto fail;
            }
            continue;
        }
        if (!strcmp(n->op, "MatMul") || !strcmp(n->op, "Gemm") || !strcmp(n->op, "Linear")) {
            QLinearParams params;
            int qlinear = qlinear_params(n, &params);
            if (qlinear < 0) goto fail;
            if (!qlinear) continue;
            T* in = nin(n, "input");
            if (!in) in = nin(n, "x");
            if (!in) in = nin(n, "a");
            T* out = t_find(n->out);
            T* weight = nin(n, "weight");
            if (!in || !out || !weight || in->dtype != T_F32 || out->dtype != T_F32 ||
                (weight->dtype != T_I8 && weight->dtype != T_U8) ||
                qt_mark_params_strict(in, params.input_scale, params.input_zp) != 0 ||
                qt_mark_params_strict(out, params.output_scale, params.output_zp) != 0) goto fail;
        }
    }
    for (int iter = 0; iter < 4; iter++) {
        for (int i = 0; i < g_nn; i++) {
            Node* n = &g_n[i];
            T* out = t_find(n->out);
            QTensor* qo = qt_get(out);
            if (!qo || !qo->has_params) continue;
            if (!strcmp(n->op, "Add")) {
                fprintf(stderr,
                        "[engine] generic Add cannot produce an implicitly quantized "
                        "activation; use QAdd with explicit I8/U8 quantization metadata\n");
                goto fail;
            } else if (!strcmp(n->op, "ResizeNearest2D") || !strcmp(n->op, "UpsampleNearest2D") ||
                       !strcmp(n->op, "MaxPool2D") || !strcmp(n->op, "Reshape") ||
                       !strcmp(n->op, "Flatten") || !strcmp(n->op, "Squeeze") ||
                       !strcmp(n->op, "Unsqueeze") || !strcmp(n->op, "Dropout") ||
                       !strcmp(n->op, "Identity")) {
                T* in = nin(n, "input");
                if (qt_mark_params(in, qo->scale, qo->zp) != 0) goto fail;
            }
        }
    }
    return 0;
fail:
    volvoxai_engine_clear_qlinear_metadata();
    volvoxai_engine_clear_qconv_metadata();
    volvoxai_engine_clear_qembedding_metadata();
    return -1;
}

static int graph_declared_dtype(cJSON* item) {
    if (!item) return T_F32;
    if (!cJSON_IsString(item) || !item->valuestring) return -1;
    const char* value = item->valuestring;
    if (!strcmp(value, "float32") || !strcmp(value, "F32") || !strcmp(value, "f32")) return T_F32;
    if (!strcmp(value, "int32") || !strcmp(value, "I32") || !strcmp(value, "i32")) return T_I32;
    if (!strcmp(value, "int8") || !strcmp(value, "I8") || !strcmp(value, "i8")) return T_I8;
    if (!strcmp(value, "uint8") || !strcmp(value, "U8") || !strcmp(value, "u8")) return T_U8;
    if (!strcmp(value, "float16") || !strcmp(value, "F16") || !strcmp(value, "f16")) return T_F16;
    return -1;
}

/* Older native blueprints used I8/U8/F16 input labels while still supplying
 * F32 storage. Only a quantization descriptor opts an I8/U8 graph input into
 * the new physical-byte contract. */
static int graph_input_dtype(cJSON* descriptor) {
    int dtype = graph_declared_dtype(descriptor ? cJSON_GetObjectItem(descriptor, "dtype") : NULL);
    if (dtype == T_I8 || dtype == T_U8) {
        return cJSON_GetObjectItem(descriptor, "quantization") ? dtype : T_F32;
    }
    return dtype == T_F16 ? T_F32 : dtype;
}

static int graph_parse_shape(cJSON* item, int required, int* shape, int* ndim, long* numel) {
    if (!item) {
        if (required) return -1;
        if (ndim) *ndim = 0;
        if (numel) *numel = 1;
        return 0;
    }
    if (!cJSON_IsArray(item)) return -1;
    int rank = cJSON_GetArraySize(item);
    if (rank < 0 || rank > 8) return -1;
    int parsed[8] = {0};
    for (int index = 0; index < rank; index++) {
        cJSON* dimension = cJSON_GetArrayItem(item, index);
        if (!cJSON_IsNumber(dimension) || !isfinite(dimension->valuedouble) ||
            dimension->valuedouble != (double)dimension->valueint ||
            dimension->valueint <= 0) return -1;
        parsed[index] = dimension->valueint;
    }
    long count = numel_of(parsed, rank);
    if (count < 0) return -1;
    if (shape) for (int index = 0; index < rank; index++) shape[index] = parsed[index];
    if (ndim) *ndim = rank;
    if (numel) *numel = count;
    return 0;
}

static int graph_json_name_valid(const char* name, size_t capacity) {
    return name && name[0] && strlen(name) < capacity;
}

static int graph_object_has_prior_key(const cJSON* object, const cJSON* stop, const char* key) {
    if (!object || !key) return 0;
    for (const cJSON* item = object->child; item && item != stop; item = item->next) {
        if (item->string && !strcmp(item->string, key)) return 1;
    }
    return 0;
}

static int graph_inputs_declare(const cJSON* inputs, const char* name) {
    if (!inputs || !name) return 0;
    for (const cJSON* item = inputs->child; item; item = item->next) {
        if (item->string && !strcmp(item->string, name)) return 1;
    }
    return 0;
}

static int graph_nodes_declare_before(const cJSON* nodes, const cJSON* stop,
                                      const char* name) {
    if (!nodes || !name) return 0;
    for (const cJSON* node = nodes->child; node && node != stop; node = node->next) {
        cJSON* outputs = cJSON_GetObjectItem(node, "outputs");
        for (cJSON* output = outputs ? outputs->child : NULL; output; output = output->next) {
            if (cJSON_IsString(output) && output->valuestring &&
                !strcmp(output->valuestring, name)) return 1;
        }
    }
    return 0;
}

static int graph_tensor_declared_before(const cJSON* inputs, const cJSON* nodes,
                                        const cJSON* node, const char* name) {
    return t_find(name) || graph_inputs_declare(inputs, name) ||
           graph_nodes_declare_before(nodes, node, name);
}

static int graph_validate_output_metadata(const cJSON* outputs, const cJSON* metadata) {
    if (!metadata) return 0;
    if (!cJSON_IsObject(metadata)) return -1;
    for (const cJSON* item = metadata->child; item; item = item->next) {
        if (!graph_json_name_valid(item->string, sizeof(((Ref*)0)->key)) ||
            !outputs || !cJSON_GetObjectItem(outputs, item->string)) return -1;
    }
    return 0;
}

static int graph_validate_config(cJSON* root) {
    if (!cJSON_IsObject(root)) return -1;
    cJSON* inputs = cJSON_GetObjectItem(root, "inputs");
    cJSON* nodes = cJSON_GetObjectItem(root, "nodes");
    cJSON* graph_outputs = cJSON_GetObjectItem(root, "outputs");
    if ((inputs && !cJSON_IsObject(inputs)) || (nodes && !cJSON_IsArray(nodes)) ||
        (graph_outputs && !cJSON_IsObject(graph_outputs) && !cJSON_IsArray(graph_outputs))) {
        return -1;
    }

    int projected_tensors = g_nt;
    for (cJSON* input = inputs ? inputs->child : NULL; input; input = input->next) {
        int shape[8];
        int ndim;
        long numel;
        int dtype = graph_input_dtype(input);
        TensorQuantization quantization = {0};
        if (!graph_json_name_valid(input->string, sizeof(g_t[0].name)) ||
            !cJSON_IsObject(input) || graph_object_has_prior_key(inputs, input, input->string) ||
            t_find(input->string) ||
            graph_parse_shape(cJSON_GetObjectItem(input, "shape"), 1,
                              shape, &ndim, &numel) != 0 ||
            dtype < 0 || !dtype_size(dtype) ||
            (size_t)numel > SIZE_MAX / dtype_size(dtype) ||
            parse_per_tensor_quantization(cJSON_GetObjectItem(input, "quantization"),
                                          dtype, &quantization) != 0 ||
            ++projected_tensors > MAXT) return -1;
    }

    int node_count = nodes ? cJSON_GetArraySize(nodes) : 0;
    if (node_count < 0 || node_count > MAXN) return -1;
    for (cJSON* node = nodes ? nodes->child : NULL; node; node = node->next) {
        if (!cJSON_IsObject(node)) return -1;
        cJSON* op = cJSON_GetObjectItem(node, "opType");
        if (!op) op = cJSON_GetObjectItem(node, "op");
        cJSON* node_inputs = cJSON_GetObjectItem(node, "inputs");
        cJSON* outputs = cJSON_GetObjectItem(node, "outputs");
        cJSON* output_shapes = cJSON_GetObjectItem(node, "outputs_shape");
        cJSON* output_dtypes = cJSON_GetObjectItem(node, "outputs_dtype");
        cJSON* output_quantization = cJSON_GetObjectItem(node, "outputs_quantization");
        cJSON* params = cJSON_GetObjectItem(node, "params");
        if (!cJSON_IsString(op) || !graph_json_name_valid(op->valuestring, sizeof(g_n[0].op)) ||
            (node_inputs && !cJSON_IsObject(node_inputs)) ||
            !cJSON_IsObject(outputs) || !outputs->child ||
            !cJSON_IsObject(output_shapes) ||
            (params && !cJSON_IsObject(params)) ||
            graph_validate_output_metadata(outputs, output_shapes) != 0 ||
            graph_validate_output_metadata(outputs, output_dtypes) != 0 ||
            graph_validate_output_metadata(outputs, output_quantization) != 0 ||
            cJSON_GetArraySize(node_inputs) > MAXIN || cJSON_GetArraySize(outputs) > MAXIN) {
            return -1;
        }

        for (cJSON* input = node_inputs ? node_inputs->child : NULL; input; input = input->next) {
            if (!graph_json_name_valid(input->string, sizeof(g_n[0].ins[0].key)) ||
                graph_object_has_prior_key(node_inputs, input, input->string) ||
                !cJSON_IsString(input) ||
                !graph_json_name_valid(input->valuestring, sizeof(g_t[0].name)) ||
                !graph_tensor_declared_before(inputs, nodes, node, input->valuestring)) {
                return -1;
            }
        }

        for (cJSON* output = outputs->child; output; output = output->next) {
            int shape[8];
            int ndim;
            long numel;
            if (!graph_json_name_valid(output->string, sizeof(g_n[0].outs[0].key)) ||
                graph_object_has_prior_key(outputs, output, output->string) ||
                !cJSON_IsString(output) ||
                !graph_json_name_valid(output->valuestring, sizeof(g_t[0].name)) ||
                graph_tensor_declared_before(inputs, nodes, node, output->valuestring)) return -1;
            for (cJSON* prior = outputs->child; prior && prior != output; prior = prior->next) {
                if (cJSON_IsString(prior) && prior->valuestring &&
                    !strcmp(prior->valuestring, output->valuestring)) return -1;
            }
            cJSON* shape_item = output_shapes
                ? cJSON_GetObjectItem(output_shapes, output->string) : NULL;
            cJSON* dtype_item = output_dtypes
                ? cJSON_GetObjectItem(output_dtypes, output->string) : NULL;
            cJSON* quantization_item = output_quantization
                ? cJSON_GetObjectItem(output_quantization, output->string) : NULL;
            int dtype = graph_declared_dtype(dtype_item);
            TensorQuantization quantization = {0};
            if ((output_dtypes && !dtype_item) ||
                (output_quantization && (!quantization_item || !output_dtypes)) ||
                graph_parse_shape(shape_item, 1, shape, &ndim, &numel) != 0 ||
                dtype < 0 || !dtype_size(dtype) ||
                (size_t)numel > SIZE_MAX / dtype_size(dtype) ||
                parse_per_tensor_quantization(quantization_item, dtype, &quantization) != 0 ||
                ++projected_tensors > MAXT) return -1;
        }
    }

    if (cJSON_IsArray(graph_outputs)) {
        for (cJSON* output = graph_outputs->child; output; output = output->next) {
            if (!cJSON_IsString(output) ||
                !graph_json_name_valid(output->valuestring, sizeof(g_t[0].name)) ||
                !graph_tensor_declared_before(inputs, nodes, NULL, output->valuestring)) return -1;
            for (cJSON* prior = graph_outputs->child; prior && prior != output;
                 prior = prior->next) {
                if (cJSON_IsString(prior) && prior->valuestring &&
                    !strcmp(prior->valuestring, output->valuestring)) return -1;
            }
        }
    } else if (cJSON_IsObject(graph_outputs)) {
        for (cJSON* output = graph_outputs->child; output; output = output->next) {
            if (!graph_json_name_valid(output->string, sizeof(g_t[0].name)) ||
                graph_object_has_prior_key(graph_outputs, output, output->string) ||
                !cJSON_IsString(output) ||
                !graph_json_name_valid(output->valuestring, sizeof(g_t[0].name)) ||
                !graph_tensor_declared_before(inputs, nodes, NULL, output->valuestring)) return -1;
            for (cJSON* prior = graph_outputs->child; prior && prior != output;
                 prior = prior->next) {
                if (cJSON_IsString(prior) && prior->valuestring &&
                    !strcmp(prior->valuestring, output->valuestring)) return -1;
            }
        }
    }
    return 0;
}

static void graph_build_rollback(int tensor_count) {
    for (int index = 0; index < g_nn; index++) {
        if (g_n[index].owns_params && g_n[index].params) cJSON_Delete(g_n[index].params);
        memset(&g_n[index], 0, sizeof(g_n[index]));
    }
    for (int index = 0; index < g_nt; index++) {
        free(g_qt[index].data);
        memset(&g_qt[index], 0, sizeof(g_qt[index]));
        if (index >= tensor_count) {
            if (g_t[index].owns) free(g_t[index].data);
            memset(&g_t[index], 0, sizeof(g_t[index]));
        }
    }
    volvoxai_engine_clear_qlinear_metadata();
    volvoxai_engine_clear_qconv_metadata();
    volvoxai_engine_clear_qembedding_metadata();
    g_nt = tensor_count;
    volvoxai_engine_tensor_name_index_invalidate();
    g_nn = 0;
    g_first_input[0] = 0;
    if (g_cfg_root) {
        cJSON_Delete(g_cfg_root);
        g_cfg_root = NULL;
    }
}

int build_graph(const char* config_path) {
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    g_removed_graph_output_count = 0;
    graph_opt_reset_state();
    if (g_cfg_root) {
        cJSON_Delete(g_cfg_root);
        g_cfg_root = NULL;
    }
    long sz; char* cfg = read_file(config_path, &sz);
    if (!cfg) return -1;
    cJSON* root = cJSON_Parse(cfg); free(cfg);
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        graph_validate_config(root) != 0) {
        fprintf(stderr, "[engine] invalid config.json graph contract\n");
        cJSON_Delete(root);
        return -1;
    }
    int initial_tensor_count = g_nt;
    g_cfg_root = root;
    if (install_weight_scale_companion_manifest(root) != 0) {
        fprintf(stderr, "[engine] invalid safetensors companion-scale manifest\n");
        goto fail;
    }
    if (apply_weight_quantization(root) != 0) {
        fprintf(stderr, "[engine] invalid weights_quantization metadata\n");
        goto fail;
    }

    // Graph inputs (allocated, zero-filled; the caller pokes them before forward()).
    cJSON* inputs = cJSON_GetObjectItem(root, "inputs");
    if (inputs) for (cJSON* it = inputs->child; it; it = it->next) {
        cJSON* shape = cJSON_GetObjectItem(it, "shape");
        int ndim; int sh[8]; long numel;
        int dtype = graph_input_dtype(it);
        if (graph_parse_shape(shape, 1, sh, &ndim, &numel) != 0) goto fail;
        T* input = t_add_ex(it->string, sh, ndim, NULL, dtype);
        if (!input || !input->data) goto fail;
        if (parse_per_tensor_quantization(cJSON_GetObjectItem(it, "quantization"), dtype,
                                          &input->quantization) != 0) goto fail;
        input->is_graph_input = 1;
        if (!g_first_input[0]) { strncpy(g_first_input, it->string, 127); g_first_input[127] = 0; }
    }

    // Nodes.
    cJSON* nodes = cJSON_GetObjectItem(root, "nodes");
    for (cJSON* nd = nodes ? nodes->child : NULL; nd; nd = nd->next) {
        Node* n = &g_n[g_nn];
        memset(n, 0, sizeof(*n));
        cJSON* op = cJSON_GetObjectItem(nd, "opType"); if (!op) op = cJSON_GetObjectItem(nd, "op");
        memcpy(n->op, op->valuestring, strlen(op->valuestring) + 1);
        n->params = cJSON_GetObjectItem(nd, "params");  // borrowed (g_cfg_root kept alive)
        n->nin = 0; n->nout = 0; n->owns_params = 0; n->disabled = 0; n->fuse_relu6 = 0; n->skip = 0;
        cJSON* nins = cJSON_GetObjectItem(nd, "inputs");
        if (nins) for (cJSON* it = nins->child; it; it = it->next) {
            memcpy(n->ins[n->nin].key, it->string, strlen(it->string) + 1);
            memcpy(n->ins[n->nin].name, it->valuestring, strlen(it->valuestring) + 1);
            n->nin++;
        }
        // Create output tensor(s) from outputs_shape; keep the first as the node output.
        cJSON* outs = cJSON_GetObjectItem(nd, "outputs");
        cJSON* oshape = cJSON_GetObjectItem(nd, "outputs_shape");
        cJSON* odtype = cJSON_GetObjectItem(nd, "outputs_dtype");
        cJSON* oquant = cJSON_GetObjectItem(nd, "outputs_quantization");
        n->out[0] = 0;
        if (outs) for (cJSON* it = outs->child; it; it = it->next) {
            const char* tname = it->valuestring;
            cJSON* sh = oshape ? cJSON_GetObjectItem(oshape, it->string) : NULL;
            cJSON* dtype_item = odtype ? cJSON_GetObjectItem(odtype, it->string) : NULL;
            cJSON* quantization = oquant ? cJSON_GetObjectItem(oquant, it->string) : NULL;
            int ndim; int shp[8]; long numel;
            int dtype = graph_declared_dtype(dtype_item);
            if (graph_parse_shape(sh, 1, shp, &ndim, &numel) != 0) goto fail;
            T* output = t_add_ex(tname, shp, ndim, NULL, dtype);
            if (!output || parse_per_tensor_quantization(quantization, dtype,
                                                          &output->quantization) != 0) goto fail;
            memcpy(n->outs[n->nout].key, it->string, strlen(it->string) + 1);
            memcpy(n->outs[n->nout].name, tname, strlen(tname) + 1);
            n->nout++;
            if (!n->out[0]) { strncpy(n->out, tname, 127); n->out[127] = 0; }
        }
        g_nn++;
    }
    GraphOptStats opt = graph_optimize_operator_fusion();
    if (build_quant_params_pass() != 0) goto fail;
    if (g_debug) {
        fprintf(stderr, "[debug] build_graph tensors=%d nodes=%d bytes=%ld %.3f ms\n",
                g_nt, g_nn, sz, volvoxai_engine_now_ms() - t0);
        fprintf(stderr, "[debug] graph_opt operator_fusion relu6=%d conv_add=%d depthwise_pointwise=%d chained_add=%d concat_sigmoid=%d alias=%d skipped=%d\n",
                opt.relu6, opt.conv_add, opt.depthwise_pointwise, opt.chained_add,
                opt.concat_sigmoid, opt.alias, opt.skipped);
    }
    volvoxai_engine_finish_tensor_table_mutation();
    return 0;
fail:
    graph_build_rollback(initial_tensor_count);
    return -1;
}

static const char* volvoxai_engine_dtype_name(int dtype) {
    switch (dtype) {
        case T_F32: return "F32";
        case T_I8: return "I8";
        case T_U8: return "U8";
        case T_I32: return "I32";
        case T_F16: return "F16";
        default: return "UNKNOWN";
    }
}

static cJSON* shape_to_json(const int* shape, int ndim) {
    cJSON* arr = cJSON_CreateArray();
    if (!arr) return NULL;
    for (int i = 0; i < ndim; i++) cJSON_AddItemToArray(arr, cJSON_CreateNumber(shape[i]));
    return arr;
}

static cJSON* tensor_to_json(const T* t) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    cJSON_AddStringToObject(obj, "name", t->name);
    cJSON_AddStringToObject(obj, "dtype", volvoxai_engine_dtype_name(t->dtype));
    cJSON_AddNumberToObject(obj, "size_bytes", (double)(t->numel * (long)t->elem_size));
    cJSON_AddBoolToObject(obj, "owns", t->owns ? 1 : 0);
    cJSON* shape = shape_to_json(t->shape, t->ndim);
    if (shape) cJSON_AddItemToObject(obj, "shape", shape);
    return obj;
}

static cJSON* refs_to_json(const Ref* refs, int count) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    for (int i = 0; i < count; i++) cJSON_AddStringToObject(obj, refs[i].key, refs[i].name);
    return obj;
}

static cJSON* node_output_shapes_json(const Node* n) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    for (int i = 0; i < n->nout; i++) {
        T* t = t_find(n->outs[i].name);
        if (!t) continue;
        cJSON* shape = shape_to_json(t->shape, t->ndim);
        if (shape) cJSON_AddItemToObject(obj, n->outs[i].key, shape);
    }
    return obj;
}

static const char* config_dtype_name(int dtype) {
    switch (dtype) {
        case T_F32: return "float32";
        case T_I8: return "int8";
        case T_U8: return "uint8";
        case T_I32: return "int32";
        case T_F16: return "float16";
        default: return NULL;
    }
}

static cJSON* node_output_dtypes_json(const Node* n) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    for (int i = 0; i < n->nout; i++) {
        T* tensor = t_find(n->outs[i].name);
        const char* dtype = tensor ? config_dtype_name(tensor->dtype) : NULL;
        if (!dtype) {
            cJSON_Delete(obj);
            return NULL;
        }
        cJSON_AddStringToObject(obj, n->outs[i].key, dtype);
    }
    return obj;
}

static cJSON* node_output_quantization_json(const Node* n) {
    cJSON* obj = cJSON_CreateObject();
    int count = 0;
    if (!obj) return NULL;
    for (int i = 0; i < n->nout; i++) {
        T* tensor = t_find(n->outs[i].name);
        if (!tensor || !tensor->quantization.valid) continue;
        cJSON* descriptor = cJSON_CreateObject();
        if (!descriptor) {
            cJSON_Delete(obj);
            return NULL;
        }
        cJSON_AddStringToObject(descriptor, "scheme", "per_tensor");
        cJSON_AddNumberToObject(descriptor, "scale", tensor->quantization.scale);
        cJSON_AddNumberToObject(descriptor, "zero_point", tensor->quantization.zero_point);
        cJSON_AddItemToObject(obj, n->outs[i].key, descriptor);
        count++;
    }
    if (!count) {
        cJSON_Delete(obj);
        return NULL;
    }
    return obj;
}

static cJSON* node_to_json(const Node* n, int index, int include_params) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    cJSON_AddNumberToObject(obj, "index", index);
    cJSON_AddStringToObject(obj, "id", "");
    cJSON_AddStringToObject(obj, "op", n->op);
    cJSON_AddBoolToObject(obj, "disabled", n->disabled ? 1 : 0);
    cJSON* inputs = refs_to_json(n->ins, n->nin);
    cJSON* outputs = refs_to_json(n->outs, n->nout);
    cJSON* output_shapes = node_output_shapes_json(n);
    if (inputs) cJSON_AddItemToObject(obj, "inputs", inputs);
    if (outputs) cJSON_AddItemToObject(obj, "outputs", outputs);
    if (output_shapes) cJSON_AddItemToObject(obj, "output_shapes", output_shapes);
    if (include_params && n->params) {
        cJSON* params = cJSON_Duplicate(n->params, 1);
        if (params) cJSON_AddItemToObject(obj, "params", params);
    }
    return obj;
}

static cJSON* safetensors_file_to_json(int index, const SafetensorsFile* file) {
    cJSON* obj = cJSON_CreateObject();
    cJSON* tensors = cJSON_CreateArray();
    if (!obj || !tensors) {
        cJSON_Delete(obj);
        cJSON_Delete(tensors);
        return NULL;
    }
    cJSON_AddNumberToObject(obj, "index", index);
    cJSON_AddStringToObject(obj, "path", g_weight_paths[index]);
    cJSON_AddNumberToObject(obj, "size_bytes", (double)file->size);
    if (file->has_metadata && file->metadata_json) {
        cJSON* md = cJSON_Parse(file->metadata_json);
        if (md) cJSON_AddItemToObject(obj, "metadata", md);
    }
    for (int i = 0; i < file->tensor_count; i++) {
        const SafetensorsTensor* t = &file->tensors[i];
        cJSON* tj = cJSON_CreateObject();
        if (!tj) continue;
        cJSON_AddStringToObject(tj, "name", t->name);
        cJSON_AddStringToObject(tj, "dtype", safetensors_dtype_name(t->dtype));
        cJSON_AddNumberToObject(tj, "data_start", (double)t->data_start);
        cJSON_AddNumberToObject(tj, "data_end", (double)t->data_end);
        cJSON_AddNumberToObject(tj, "access_flags", (double)t->flags);
        cJSON* shape = shape_to_json(t->shape, t->ndim);
        if (shape) cJSON_AddItemToObject(tj, "shape", shape);
        cJSON_AddItemToArray(tensors, tj);
    }
    cJSON_AddItemToObject(obj, "tensors", tensors);
    return obj;
}

char* volvoxai_engine_inspect_model_json(int include_graph, int include_tensors, int include_weight_files, int include_params) {
    cJSON* root = cJSON_CreateObject();
    if (!root) return NULL;
    int visible_tensor_count = 0;
    for (int i = 0; i < g_nt; i++) {
        if (removed_graph_output_index(g_t[i].name) < 0 &&
            !volvoxai_engine_tensor_is_internal_companion_locked(g_t[i].name)) {
            visible_tensor_count++;
        }
    }
    cJSON_AddNumberToObject(root, "num_ops", g_nn);
    cJSON_AddNumberToObject(root, "num_tensors", visible_tensor_count);
    if (include_tensors) {
        cJSON* tensors = cJSON_CreateArray();
        if (tensors) {
            for (int i = 0; i < g_nt; i++) {
                if (removed_graph_output_index(g_t[i].name) >= 0 ||
                    volvoxai_engine_tensor_is_internal_companion_locked(g_t[i].name)) continue;
                cJSON* tj = tensor_to_json(&g_t[i]);
                if (tj) cJSON_AddItemToArray(tensors, tj);
            }
            cJSON_AddItemToObject(root, "tensors", tensors);
        }
    }
    if (include_graph) {
        cJSON* nodes = cJSON_CreateArray();
        if (nodes) {
            for (int i = 0; i < g_nn; i++) {
                cJSON* nj = node_to_json(&g_n[i], i, include_params);
                if (nj) cJSON_AddItemToArray(nodes, nj);
            }
            cJSON_AddItemToObject(root, "nodes", nodes);
        }
    }
    if (include_weight_files) {
        cJSON* weights = cJSON_CreateArray();
        if (weights) {
            for (int i = 0; i < g_weight_file_count; i++) {
                cJSON* wj = safetensors_file_to_json(i, &g_weight_files[i]);
                if (wj) cJSON_AddItemToArray(weights, wj);
            }
            cJSON_AddItemToObject(root, "weights", weights);
        }
    }
    char* out = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return out;
}

static int free_node_caches(void) {
    if (volvoxai_engine_prepare_tensor_table_mutation() != 0) return -1;
    vx_conv_f32_opt_free_all();
    vx_gemm_f32_cache_free_all();
#if VOLVOXAI_ENABLE_VULKAN
    vk_free_weight_cache();
#endif
#if VOLVOXAI_ENABLE_OPENGL
    opengl_free_weight_cache();
#endif
    vx_runtime_backend_reset();
    for (int i = 0; i < MAXN; i++) {
        free(g_kcache[i]); g_kcache[i] = NULL;
        free(g_vcache[i]); g_vcache[i] = NULL;
        free(g_qwcache[i]); g_qwcache[i] = NULL;
        free(g_f16wcache[i]); g_f16wcache[i] = NULL;
        free(g_f16bcache[i]); g_f16bcache[i] = NULL;
        free(g_conv_wcache[i]); g_conv_wcache[i] = NULL;
        free(g_q8wcache[i]); g_q8wcache[i] = NULL; g_q8wcache_bytes[i] = 0;
    }
    for (int i = 0; i < g_nt; i++) {
        free(g_qt[i].data);
        memset(&g_qt[i], 0, sizeof(g_qt[i]));
    }
    return 0;
}

int volvoxai_engine_refresh_weight_caches(void) {
    vx_conv_f32_opt_free_all();
    vx_gemm_f32_cache_free_all();
#if VOLVOXAI_ENABLE_VULKAN
    vk_free_weight_cache();
#endif
#if VOLVOXAI_ENABLE_OPENGL
    opengl_free_weight_cache();
#endif
    vx_runtime_backend_reset();
    for (int i = 0; i < MAXN; i++) {
        free(g_qwcache[i]); g_qwcache[i] = NULL;
        free(g_f16wcache[i]); g_f16wcache[i] = NULL;
        free(g_f16bcache[i]); g_f16bcache[i] = NULL;
        free(g_conv_wcache[i]); g_conv_wcache[i] = NULL;
        free(g_q8wcache[i]); g_q8wcache[i] = NULL; g_q8wcache_bytes[i] = 0;
    }
    for (int i = 0; i < g_nt; i++) {
        free(g_qt[i].data);
        memset(&g_qt[i], 0, sizeof(g_qt[i]));
    }
    if (build_quant_params_pass() != 0) return -1;
    if (prepack_conv_weights() != 0) return -1;
    g_weight_caches_dirty = 0;
    return 0;
}

static int rebuild_after_graph_patch_ex(int reoptimize) {
    if (free_node_caches() != 0) return -1;
    graph_opt_reset_state();
    for (int i = 0; i < g_nn; i++) {
        g_n[i].skip = g_n[i].disabled ? 1 : 0;
        g_n[i].fuse_relu6 = 0;
    }
    if (reoptimize) graph_optimize_operator_fusion();
    if (build_quant_params_pass() != 0) return -1;
    volvoxai_engine_finish_tensor_table_mutation();
    if (prepack_conv_weights() != 0) return -1;
    g_weight_caches_dirty = 0;
    return 0;
}

static void clear_owned_params(Node* n) {
    if (n->owns_params && n->params) cJSON_Delete(n->params);
    n->params = NULL;
    n->owns_params = 0;
}

static int shape_from_json(cJSON* arr, int* shape, int* ndim) {
    if (!cJSON_IsArray(arr)) return -1;
    int n = cJSON_GetArraySize(arr);
    if (n < 0 || n > 8) return -1;
    for (int i = 0; i < n; i++) {
        cJSON* d = cJSON_GetArrayItem(arr, i);
        if (!cJSON_IsNumber(d) || d->valueint < 0) return -1;
        shape[i] = d->valueint;
    }
    *ndim = n;
    return 0;
}

static void graph_patch_reuse_begin(void) {
    g_graph_patch_reused_count = 0;
}

static void graph_patch_reuse_rollback(void) {
    for (int i = g_graph_patch_reused_count - 1; i >= 0; i--) {
        int index = g_graph_patch_reused_indices[i];
        if (index < 0 || index >= g_nt) continue;
        if (g_t[index].owns && g_t[index].data &&
            g_t[index].data != g_graph_patch_reused_old_tensors[i].data) {
            free(g_t[index].data);
        }
        g_t[index] = g_graph_patch_reused_old_tensors[i];
    }
    if (g_graph_patch_reused_count > 0)
        volvoxai_engine_tensor_name_index_invalidate();
    g_graph_patch_reused_count = 0;
}

static void graph_patch_reuse_commit(void) {
    for (int i = 0; i < g_graph_patch_reused_count; i++) {
        T* old = &g_graph_patch_reused_old_tensors[i];
        int index = g_graph_patch_reused_indices[i];
        if (old->owns && old->data && index >= 0 && index < g_nt && old->data != g_t[index].data) {
            free(old->data);
        }
    }
    g_graph_patch_reused_count = 0;
}

static int graph_patch_tensor_was_reused(int index) {
    for (int i = 0; i < g_graph_patch_reused_count; i++) {
        if (g_graph_patch_reused_indices[i] == index) return 1;
    }
    return 0;
}

static T* graph_patch_add_output_tensor(const char* name, const int* shape, int ndim) {
    for (int removed = 0; removed < g_removed_graph_output_count; removed++) {
        T* candidate = t_find(g_removed_graph_outputs[removed]);
        if (!candidate || candidate->is_graph_input ||
            volvoxai_engine_tensor_weight_file_index(candidate->name) >= 0) continue;
        int referenced = 0;
        for (int node = 0; node < g_nn && !referenced; node++) {
            for (int input = 0; input < g_n[node].nin; input++) {
                if (!strcmp(g_n[node].ins[input].name, candidate->name)) { referenced = 1; break; }
            }
            for (int output = 0; output < g_n[node].nout && !referenced; output++) {
                if (!strcmp(g_n[node].outs[output].name, candidate->name)) { referenced = 1; break; }
            }
            if (!referenced && !strcmp(g_n[node].out, candidate->name)) referenced = 1;
        }
        if (referenced || g_graph_patch_reused_count >= MAXT) continue;
        long numel = 1;
        for (int d = 0; d < ndim; d++) numel *= shape[d];
        float* data = (float*)calloc(numel > 0 ? (size_t)numel : 1u, sizeof(float));
        if (!data) return NULL;
        int index = (int)(candidate - g_t);
        g_graph_patch_reused_indices[g_graph_patch_reused_count] = index;
        g_graph_patch_reused_old_tensors[g_graph_patch_reused_count] = *candidate;
        g_graph_patch_reused_count++;
        char old_name[128];
        strncpy(old_name, candidate->name, sizeof(old_name) - 1);
        old_name[sizeof(old_name) - 1] = 0;
        memset(candidate, 0, sizeof(*candidate));
        strncpy(candidate->name, name, sizeof(candidate->name) - 1);
        candidate->ndim = ndim;
        for (int d = 0; d < ndim; d++) candidate->shape[d] = shape[d];
        candidate->numel = numel;
        candidate->data = data;
        candidate->owns = 1;
        candidate->dtype = T_F32;
        candidate->elem_size = sizeof(float);
        volvoxai_engine_tensor_name_index_invalidate();
        removed_graph_output_remove(old_name);
        return candidate;
    }
    return g_nt < MAXT ? t_add(name, shape, ndim, NULL) : NULL;
}

static int set_ref(Ref* refs, int* count, const char* key, const char* name) {
    if (!key || !name) return -1;
    for (int i = 0; i < *count; i++) {
        if (strcmp(refs[i].key, key) == 0) {
            strncpy(refs[i].name, name, sizeof(refs[i].name) - 1);
            refs[i].name[sizeof(refs[i].name) - 1] = 0;
            return 0;
        }
    }
    if (*count >= MAXIN) return -1;
    strncpy(refs[*count].key, key, sizeof(refs[*count].key) - 1);
    refs[*count].key[sizeof(refs[*count].key) - 1] = 0;
    strncpy(refs[*count].name, name, sizeof(refs[*count].name) - 1);
    refs[*count].name[sizeof(refs[*count].name) - 1] = 0;
    (*count)++;
    return 0;
}

static int apply_node_patch(Node* n, cJSON* patch, int replace) {
    if (replace) {
        clear_owned_params(n);
        memset(n, 0, sizeof(*n));
    }
    cJSON* op = cJSON_GetObjectItem(patch, "op");
    if (!op) op = cJSON_GetObjectItem(patch, "opType");
    if (op) {
        if (!cJSON_IsString(op)) return -1;
        strncpy(n->op, op->valuestring, sizeof(n->op) - 1);
        n->op[sizeof(n->op) - 1] = 0;
    } else if (replace || !n->op[0]) {
        return -1;
    }

    cJSON* disabled = cJSON_GetObjectItem(patch, "disabled");
    if (cJSON_IsBool(disabled)) n->disabled = cJSON_IsTrue(disabled) ? 1 : 0;

    cJSON* inputs = cJSON_GetObjectItem(patch, "inputs");
    if (inputs) {
        if (!cJSON_IsObject(inputs)) return -1;
        if (replace) n->nin = 0;
        for (cJSON* it = inputs->child; it; it = it->next) {
            if (!cJSON_IsString(it) || !t_find(it->valuestring) ||
                volvoxai_engine_tensor_is_internal_companion_locked(
                    it->valuestring)) return -1;
            if (set_ref(n->ins, &n->nin, it->string, it->valuestring) != 0) return -1;
        }
    }

    cJSON* output_shapes = cJSON_GetObjectItem(patch, "output_shapes");
    cJSON* outputs = cJSON_GetObjectItem(patch, "outputs");
    if (outputs) {
        if (!cJSON_IsObject(outputs)) return -1;
        if (replace) {
            n->nout = 0;
            n->out[0] = 0;
        }
        for (cJSON* it = outputs->child; it; it = it->next) {
            if (!cJSON_IsString(it)) return -1;
            T* existing = t_find(it->valuestring);
            if (existing && volvoxai_engine_tensor_is_internal_companion_locked(
                                it->valuestring))
                return -1;
            if (!existing) {
                cJSON* sh = output_shapes ? cJSON_GetObjectItem(output_shapes, it->string) : NULL;
                int shape[8], ndim = 0;
                if (shape_from_json(sh, shape, &ndim) != 0) return -1;
                existing = graph_patch_add_output_tensor(it->valuestring, shape, ndim);
                if (!existing) return -1;
            }
            if (set_ref(n->outs, &n->nout, it->string, it->valuestring) != 0) return -1;
            if (!n->out[0]) {
                strncpy(n->out, it->valuestring, sizeof(n->out) - 1);
                n->out[sizeof(n->out) - 1] = 0;
            }
        }
    }

    cJSON* params = cJSON_GetObjectItem(patch, "params");
    cJSON* params_json = cJSON_GetObjectItem(patch, "params_json");
    if (params || params_json) {
        cJSON* next = NULL;
        if (params) {
            if (!cJSON_IsObject(params)) return -1;
            next = cJSON_Duplicate(params, 1);
        } else if (cJSON_IsString(params_json)) {
            next = cJSON_Parse(params_json->valuestring);
        }
        if (!next) return -1;
        if (!replace && n->params && params) {
            for (cJSON* it = params->child; it; it = it->next) {
                cJSON_DeleteItemFromObject(n->params, it->string);
                cJSON_AddItemToObject(n->params, it->string, cJSON_Duplicate(it, 1));
            }
            cJSON_Delete(next);
        } else {
            clear_owned_params(n);
            n->params = next;
            n->owns_params = 1;
        }
    } else if (!n->params) {
        n->params = cJSON_CreateObject();
        n->owns_params = n->params ? 1 : 0;
    }
    return 0;
}

static int volvoxai_engine_patch_node_json_impl(int node_index, const char* patch_json,
                                                VolvoxAIEngineNodePatchMode mode,
                                                int rebuild_execution_plan) {
    if (!patch_json) return -1;
    cJSON* patch = cJSON_Parse(patch_json);
    if (!patch) return -1;
    int rc = -1;
    if (mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_DELETE) {
        if (node_index >= 0 && node_index < g_nn) {
            clear_owned_params(&g_n[node_index]);
            memmove(&g_n[node_index], &g_n[node_index + 1],
                    (size_t)(g_nn - node_index - 1) * sizeof(g_n[0]));
            memset(&g_n[g_nn - 1], 0, sizeof(g_n[0]));
            g_nn--;
            rc = 0;
        }
    } else if (mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_BEFORE ||
               mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER) {
        /* -1 is the builder-friendly append sentinel, including an empty graph. */
        if (((node_index == -1) || (node_index >= 0 && node_index < g_nn)) && g_nn < MAXN) {
            int pos = node_index == -1
                          ? g_nn
                          : mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_BEFORE
                                ? node_index
                                : node_index + 1;
            memmove(&g_n[pos + 1], &g_n[pos], (size_t)(g_nn - pos) * sizeof(g_n[0]));
            memset(&g_n[pos], 0, sizeof(g_n[pos]));
            g_nn++;
            rc = apply_node_patch(&g_n[pos], patch, 1);
            if (rc != 0) {
                clear_owned_params(&g_n[pos]);
                memmove(&g_n[pos], &g_n[pos + 1], (size_t)(g_nn - pos - 1) * sizeof(g_n[0]));
                memset(&g_n[g_nn - 1], 0, sizeof(g_n[0]));
                g_nn--;
            }
        }
    } else if (mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_MERGE ||
               mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_REPLACE) {
        if (node_index >= 0 && node_index < g_nn) {
            rc = apply_node_patch(&g_n[node_index], patch,
                                  mode == VOLVOXAI_ENGINE_NODE_PATCH_MODE_REPLACE);
        }
    }
    cJSON_Delete(patch);
    (void)rebuild_execution_plan;
    return rc;
}

typedef struct {
    Node* nodes;
    int node_count;
    int tensor_count;
    cJSON* config_root;
    GraphNodeFusion* fusions;
    GraphOptStats graph_opt_stats;
    int* concat_sigmoid_fuse;
    char (*removed_outputs)[128];
    int removed_output_count;
} GraphPatchSnapshot;

static void graph_patch_snapshot_free(GraphPatchSnapshot* snapshot);

static int graph_node_output_count(const Node* node) {
    if (!node) return 0;
    return node->nout > 0 ? node->nout : (node->out[0] ? 1 : 0);
}

static const char* graph_node_output_name(const Node* node, int index) {
    if (!node || index < 0) return NULL;
    if (node->nout > 0) return index < node->nout ? node->outs[index].name : NULL;
    return index == 0 && node->out[0] ? node->out : NULL;
}

static int graph_nodes_producer(const Node* nodes, int count, const char* name) {
    int producer = -1;
    if (!nodes || !name || !name[0]) return -1;
    for (int i = 0; i < count; i++) {
        if (nodes[i].disabled) continue;
        int outputs = graph_node_output_count(&nodes[i]);
        for (int j = 0; j < outputs; j++) {
            const char* output = graph_node_output_name(&nodes[i], j);
            if (output && !strcmp(output, name)) {
                if (producer >= 0 && producer != i) return -2;
                producer = i;
            }
        }
    }
    return producer;
}

static int graph_patch_snapshot_take(GraphPatchSnapshot* snapshot) {
    if (!snapshot) return -1;
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->node_count = g_nn;
    snapshot->tensor_count = g_nt;
    snapshot->nodes = g_nn > 0 ? (Node*)calloc((size_t)g_nn, sizeof(Node)) : NULL;
    snapshot->fusions = (GraphNodeFusion*)malloc(sizeof(g_node_fusion));
    snapshot->concat_sigmoid_fuse = (int*)malloc(sizeof(g_concat_sigmoid_fuse));
    snapshot->removed_outputs = (char (*)[128])malloc(sizeof(g_removed_graph_outputs));
    if ((g_nn > 0 && !snapshot->nodes) || !snapshot->fusions ||
        !snapshot->concat_sigmoid_fuse || !snapshot->removed_outputs) goto fail;
    for (int i = 0; i < g_nn; i++) {
        snapshot->nodes[i] = g_n[i];
        snapshot->nodes[i].params = g_n[i].params ? cJSON_Duplicate(g_n[i].params, 1) : NULL;
        snapshot->nodes[i].owns_params = snapshot->nodes[i].params ? 1 : 0;
        if (g_n[i].params && !snapshot->nodes[i].params) goto fail;
    }
    snapshot->config_root = g_cfg_root ? cJSON_Duplicate(g_cfg_root, 1) : NULL;
    if (g_cfg_root && !snapshot->config_root) goto fail;
    memcpy(snapshot->fusions, g_node_fusion, sizeof(g_node_fusion));
    snapshot->graph_opt_stats = g_graph_opt_stats;
    memcpy(snapshot->concat_sigmoid_fuse, g_concat_sigmoid_fuse, sizeof(g_concat_sigmoid_fuse));
    memcpy(snapshot->removed_outputs, g_removed_graph_outputs, sizeof(g_removed_graph_outputs));
    snapshot->removed_output_count = g_removed_graph_output_count;
    return 0;
fail:
    graph_patch_snapshot_free(snapshot);
    return -1;
}

static void graph_patch_snapshot_free(GraphPatchSnapshot* snapshot) {
    if (!snapshot) return;
    if (snapshot->nodes) {
        for (int i = 0; i < snapshot->node_count; i++) {
            if (snapshot->nodes[i].owns_params && snapshot->nodes[i].params) {
                cJSON_Delete(snapshot->nodes[i].params);
            }
        }
    }
    free(snapshot->nodes);
    cJSON_Delete(snapshot->config_root);
    free(snapshot->fusions);
    free(snapshot->concat_sigmoid_fuse);
    free(snapshot->removed_outputs);
    memset(snapshot, 0, sizeof(*snapshot));
}

static int graph_patch_snapshot_restore(GraphPatchSnapshot* snapshot) {
    if (!snapshot || !snapshot->fusions || !snapshot->concat_sigmoid_fuse ||
        !snapshot->removed_outputs) return -1;
    int rc = free_node_caches();
    graph_patch_reuse_rollback();
    for (int i = 0; i < g_nn; i++) clear_owned_params(&g_n[i]);
    for (int i = snapshot->tensor_count; i < g_nt; i++) {
        if (g_t[i].owns) free(g_t[i].data);
        free(g_qt[i].data);
        memset(&g_t[i], 0, sizeof(g_t[i]));
        memset(&g_qt[i], 0, sizeof(g_qt[i]));
    }
    g_nt = snapshot->tensor_count;
    volvoxai_engine_tensor_name_index_invalidate();
    if (g_cfg_root) cJSON_Delete(g_cfg_root);
    g_cfg_root = snapshot->config_root;
    snapshot->config_root = NULL;
    memset(g_n, 0, sizeof(g_n));
    g_nn = snapshot->node_count;
    for (int i = 0; i < g_nn; i++) {
        g_n[i] = snapshot->nodes[i];
        snapshot->nodes[i].params = NULL;
        snapshot->nodes[i].owns_params = 0;
    }
    memcpy(g_node_fusion, snapshot->fusions, sizeof(g_node_fusion));
    g_graph_opt_stats = snapshot->graph_opt_stats;
    memcpy(g_concat_sigmoid_fuse, snapshot->concat_sigmoid_fuse, sizeof(g_concat_sigmoid_fuse));
    memcpy(g_removed_graph_outputs, snapshot->removed_outputs, sizeof(g_removed_graph_outputs));
    g_removed_graph_output_count = snapshot->removed_output_count;
    if (build_quant_params_pass() != 0) rc = -1;
    volvoxai_engine_finish_tensor_table_mutation();
    if (prepack_conv_weights() != 0) rc = -1;
    g_weight_caches_dirty = rc == 0 ? 0 : 1;
    graph_patch_snapshot_free(snapshot);
    return rc;
}

static void graph_patch_restore_canonical_nodes(void) {
    for (int producer = 0; producer < g_nn; producer++) {
        const GraphNodeFusion* fusion = &g_node_fusion[producer];
        int peer = fusion->peer_idx;
        if (peer < 0 || peer >= g_nn) continue;
        if (fusion->id == GRAPH_FUSION_CONV_ADD || fusion->id == GRAPH_FUSION_CHAINED_ELEMENTWISE) {
            Node* consumer = &g_n[peer];
            for (int k = 0; k < consumer->nin; k++) {
                if (strcmp(consumer->ins[k].name, fusion->aux_tensor)) {
                    strncpy(g_n[producer].out, consumer->ins[k].name, sizeof(g_n[producer].out) - 1);
                    g_n[producer].out[sizeof(g_n[producer].out) - 1] = 0;
                    break;
                }
            }
        }
    }
    for (int producer = 0; producer < g_nn; producer++) {
        if (!g_n[producer].fuse_relu6) continue;
        for (int consumer = 0; consumer < g_nn; consumer++) {
            if (!g_n[consumer].skip || strcmp(g_n[consumer].op, "Clip") ||
                strcmp(g_n[consumer].out, g_n[producer].out)) continue;
            const char* original = node_input_name(&g_n[consumer]);
            if (original) {
                strncpy(g_n[producer].out, original, sizeof(g_n[producer].out) - 1);
                g_n[producer].out[sizeof(g_n[producer].out) - 1] = 0;
            }
            break;
        }
    }
    graph_opt_reset_state();
    for (int i = 0; i < g_nn; i++) {
        g_n[i].skip = g_n[i].disabled ? 1 : 0;
        g_n[i].fuse_relu6 = 0;
    }
}

static int graph_patch_snapshot_has_output(const GraphPatchSnapshot* snapshot, const char* name) {
    return snapshot && graph_nodes_producer(snapshot->nodes, snapshot->node_count, name) != -1;
}

static int graph_patch_validate(const GraphPatchSnapshot* snapshot) {
    for (int i = 0; i < g_nn; i++) {
        Node* node = &g_n[i];
        if (node->disabled) continue;
        if (!node->op[0] || graph_node_output_count(node) <= 0) return -1;
        int outputs = graph_node_output_count(node);
        for (int oi = 0; oi < outputs; oi++) {
            const char* output = graph_node_output_name(node, oi);
            T* tensor = t_find(output);
            if (!output || !output[0] || !tensor || graph_nodes_producer(g_n, g_nn, output) == -2) return -1;
            for (int previous = 0; previous < oi; previous++) {
                if (!strcmp(graph_node_output_name(node, previous), output)) return -1;
            }
            int existed = tensor >= g_t && tensor < g_t + snapshot->tensor_count;
            if (existed && !graph_patch_snapshot_has_output(snapshot, output) &&
                removed_graph_output_index(output) < 0 &&
                !graph_patch_tensor_was_reused((int)(tensor - g_t))) return -1;
        }
        for (int ii = 0; ii < node->nin; ii++) {
            const char* input = node->ins[ii].name;
            if (!input[0] || !t_find(input)) return -1;
            int producer = graph_nodes_producer(g_n, g_nn, input);
            if (producer == -2 || producer >= i) return -1;
            if (producer < 0 && (graph_patch_snapshot_has_output(snapshot, input) ||
                                 removed_graph_output_index(input) >= 0)) return -1;
        }
    }
    return 0;
}

static void graph_patch_update_removed_outputs(const GraphPatchSnapshot* snapshot) {
    for (int i = 0; i < snapshot->node_count; i++) {
        int outputs = graph_node_output_count(&snapshot->nodes[i]);
        for (int oi = 0; oi < outputs; oi++) {
            const char* output = graph_node_output_name(&snapshot->nodes[i], oi);
            if (!output || graph_nodes_producer(g_n, g_nn, output) >= 0) continue;
            removed_graph_output_add(output);
            T* tensor = t_find(output);
            if (!tensor || !tensor->owns || !tensor->data) continue;
            int aliased = 0;
            for (int t = 0; t < g_nt; t++) {
                if (&g_t[t] != tensor && g_t[t].data == tensor->data) { aliased = 1; break; }
            }
            if (!aliased) memset(tensor->data, 0, (size_t)tensor->numel * tensor->elem_size);
        }
    }
    for (int i = 0; i < g_nn; i++) {
        int outputs = graph_node_output_count(&g_n[i]);
        for (int oi = 0; oi < outputs; oi++) {
            removed_graph_output_remove(graph_node_output_name(&g_n[i], oi));
        }
    }
}

static int graph_patch_transaction_begin(GraphPatchSnapshot* snapshot) {
    graph_patch_reuse_begin();
    if (graph_patch_snapshot_take(snapshot) != 0) {
        graph_patch_snapshot_free(snapshot);
        return -1;
    }
    if (free_node_caches() != 0) {
        graph_patch_snapshot_restore(snapshot);
        return -1;
    }
    graph_patch_restore_canonical_nodes();
    return 0;
}

static int graph_patch_transaction_finish(GraphPatchSnapshot* snapshot, int mutation_rc,
                                          int reoptimize, const char* persist_config_path) {
    int rc = mutation_rc;
    if (rc == 0) rc = graph_patch_validate(snapshot);
    if (rc == 0) {
        graph_patch_update_removed_outputs(snapshot);
        rc = rebuild_after_graph_patch_ex(reoptimize ? 1 : 0);
    }
    if (rc == 0 && persist_config_path &&
        (!persist_config_path[0] || volvoxai_engine_save_config(persist_config_path) != 0)) rc = -1;
    if (rc != 0) {
        graph_patch_snapshot_restore(snapshot);
        return -1;
    }
    graph_patch_reuse_commit();
    graph_patch_snapshot_free(snapshot);
    return 0;
}

static int volvoxai_engine_patch_node_json_transaction(int node_index, const char* patch_json,
                                                       VolvoxAIEngineNodePatchMode mode,
                                                       int reoptimize) {
    GraphPatchSnapshot snapshot;
    if (graph_patch_transaction_begin(&snapshot) != 0) return -1;
    int rc = volvoxai_engine_patch_node_json_impl(node_index, patch_json, mode, 0);
    return graph_patch_transaction_finish(&snapshot, rc, reoptimize, NULL);
}

static int graph_patch_set_declared_outputs(cJSON* declared_outputs) {
    if (!declared_outputs) return 0;
    if (!cJSON_IsArray(declared_outputs)) return -1;
    cJSON* outputs = cJSON_CreateObject();
    if (!outputs) return -1;
    int count = cJSON_GetArraySize(declared_outputs);
    for (int i = 0; i < count; i++) {
        cJSON* name_json = cJSON_GetArrayItem(declared_outputs, i);
        if (!cJSON_IsString(name_json) || !name_json->valuestring || !name_json->valuestring[0] ||
            !t_find(name_json->valuestring) ||
            graph_nodes_producer(g_n, g_nn, name_json->valuestring) < 0 ||
            removed_graph_output_index(name_json->valuestring) >= 0) {
            cJSON_Delete(outputs);
            return -1;
        }
        for (int j = 0; j < i; j++) {
            cJSON* previous = cJSON_GetArrayItem(declared_outputs, j);
            if (cJSON_IsString(previous) && !strcmp(previous->valuestring, name_json->valuestring)) {
                cJSON_Delete(outputs);
                return -1;
            }
        }
        cJSON_AddStringToObject(outputs, name_json->valuestring, name_json->valuestring);
    }
    if (!g_cfg_root) g_cfg_root = cJSON_CreateObject();
    if (!g_cfg_root) { cJSON_Delete(outputs); return -1; }
    cJSON_DeleteItemFromObject(g_cfg_root, "outputs");
    cJSON_AddItemToObject(g_cfg_root, "outputs", outputs);
    return 0;
}

int volvoxai_engine_patch_node_json(int node_index, const char* patch_json,
                                    VolvoxAIEngineNodePatchMode mode,
                                    int rebuild_execution_plan) {
    if (volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    volvoxai_engine_metadata_lock();
    /* Structural metadata must never be left stale; rebuilding is mandatory. */
    (void)rebuild_execution_plan;
    int rc = volvoxai_engine_patch_node_json_transaction(node_index, patch_json, mode, 1);
    if (rc == 0) volvoxai_engine_model_generation_advance_locked();
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_patch_graph_json(const char* patches_json, int rebuild_execution_plan,
                                     int reoptimize) {
    if (!patches_json || volvoxai_engine_model_route_lease_active()) return -1;
    cJSON* envelope = cJSON_Parse(patches_json);
    cJSON* patches = cJSON_IsArray(envelope) ? envelope : cJSON_GetObjectItem(envelope, "patches");
    cJSON* declared_outputs = cJSON_IsObject(envelope)
                                  ? cJSON_GetObjectItem(envelope, "declared_outputs") : NULL;
    cJSON* persist_path_json = cJSON_IsObject(envelope)
                                   ? cJSON_GetObjectItem(envelope, "persist_config_path") : NULL;
    const char* persist_path = cJSON_IsString(persist_path_json)
                                   ? persist_path_json->valuestring : NULL;
    if (persist_path_json && !persist_path) {
        cJSON_Delete(envelope);
        return -1;
    }
    if (!cJSON_IsArray(patches) ||
        (cJSON_GetArraySize(patches) <= 0 && !declared_outputs)) {
        cJSON_Delete(envelope);
        return -1;
    }
    volvoxai_engine_model_lock();
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        cJSON_Delete(envelope);
        return -1;
    }
    volvoxai_engine_metadata_lock();
    GraphPatchSnapshot snapshot;
    int rc = graph_patch_transaction_begin(&snapshot);
    if (rc == 0) {
        int count = cJSON_GetArraySize(patches);
        for (int i = 0; i < count; i++) {
            cJSON* item = cJSON_GetArrayItem(patches, i);
            cJSON* node_index_json = cJSON_GetObjectItem(item, "node_index");
            cJSON* mode_json = cJSON_GetObjectItem(item, "mode");
            cJSON* patch_json = cJSON_GetObjectItem(item, "patch");
            if (!cJSON_IsObject(item) || !cJSON_IsNumber(node_index_json) ||
                !cJSON_IsNumber(mode_json)) { rc = -1; break; }
            char* patch_text = NULL;
            if (cJSON_IsString(patch_json) || !patch_json) {
                const char* source = cJSON_IsString(patch_json) ? patch_json->valuestring : "{}";
                size_t length = strlen(source);
                patch_text = (char*)malloc(length + 1);
                if (patch_text) memcpy(patch_text, source, length + 1);
            } else {
                patch_text = cJSON_PrintUnformatted(patch_json);
            }
            if (!patch_text) { rc = -1; break; }
            rc = volvoxai_engine_patch_node_json_impl(node_index_json->valueint, patch_text,
                                                       (VolvoxAIEngineNodePatchMode)mode_json->valueint, 0);
            free(patch_text);
            if (rc != 0) break;
        }
        if (rc == 0) rc = graph_patch_set_declared_outputs(declared_outputs);
        rc = graph_patch_transaction_finish(&snapshot, rc, reoptimize, persist_path);
    }
    if (rc == 0) volvoxai_engine_model_generation_advance_locked();
    (void)rebuild_execution_plan;
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    cJSON_Delete(envelope);
    return rc;
}

static cJSON* node_to_config_json(const Node* n) {
    cJSON* obj = cJSON_CreateObject();
    if (!obj) return NULL;
    cJSON_AddStringToObject(obj, "opType", n->op);
    cJSON* inputs = refs_to_json(n->ins, n->nin);
    cJSON* outputs = refs_to_json(n->outs, n->nout);
    cJSON* output_shapes = node_output_shapes_json(n);
    cJSON* output_dtypes = node_output_dtypes_json(n);
    cJSON* output_quantization = node_output_quantization_json(n);
    if (inputs) cJSON_AddItemToObject(obj, "inputs", inputs);
    if (outputs) cJSON_AddItemToObject(obj, "outputs", outputs);
    if (output_shapes) cJSON_AddItemToObject(obj, "outputs_shape", output_shapes);
    if (output_dtypes) cJSON_AddItemToObject(obj, "outputs_dtype", output_dtypes);
    if (output_quantization) cJSON_AddItemToObject(obj, "outputs_quantization", output_quantization);
    if (n->params) {
        cJSON* params = cJSON_Duplicate(n->params, 1);
        if (params) cJSON_AddItemToObject(obj, "params", params);
    }
    return obj;
}

static int write_text_file_atomic(const char* path, const char* text) {
    if (!path || !path[0] || !text) return -1;
    char tmp_path[4096];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld", path, (long)time(NULL));
    FILE* f = fopen(tmp_path, "wb");
    if (!f) return -1;
    size_t n = strlen(text);
    if (n > 0 && fwrite(text, 1, n, f) != n) {
        fclose(f);
        remove(tmp_path);
        return -1;
    }
    if (fclose(f) != 0) {
        remove(tmp_path);
        return -1;
    }
    if (rename(tmp_path, path) != 0) {
        remove(tmp_path);
        return -1;
    }
    return 0;
}

int volvoxai_engine_save_config(const char* path) {
    if (!g_cfg_root || !path || !path[0]) return -1;
    cJSON* root = cJSON_Duplicate(g_cfg_root, 1);
    if (!root) return -1;
    cJSON* nodes = cJSON_CreateArray();
    if (!nodes) {
        cJSON_Delete(root);
        return -1;
    }
    for (int i = 0; i < g_nn; i++) {
        if (g_n[i].disabled) continue;
        cJSON* node = node_to_config_json(&g_n[i]);
        if (!node) {
            cJSON_Delete(nodes);
            cJSON_Delete(root);
            return -1;
        }
        cJSON_AddItemToArray(nodes, node);
    }
    cJSON_DeleteItemFromObject(root, "nodes");
    cJSON_AddItemToObject(root, "nodes", nodes);
    char* text = cJSON_Print(root);
    cJSON_Delete(root);
    if (!text) return -1;
    int rc = write_text_file_atomic(path, text);
    free(text);
    return rc;
}

/*
 * Forward helpers shared by the inference runtime and the guarded training
 * implementation. Backward orchestration itself lives under src/training.
 */
