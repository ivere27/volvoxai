static int tensor_contiguous_strides(const T* tensor, long strides[8]) {
    if (!tensor || !strides || tensor->ndim <= 0 || tensor->ndim > 8) return -1;
    long stride = 1;
    for (int dimension = tensor->ndim - 1; dimension >= 0; dimension--) {
        if (tensor->shape[dimension] <= 0 || stride > LONG_MAX / tensor->shape[dimension])
            return -1;
        strides[dimension] = stride;
        stride *= tensor->shape[dimension];
    }
    return stride == tensor->numel ? 0 : -1;
}

static int tensor_broadcast_strides(const T* input, const T* output, long strides[8]) {
    if (!input || !output || !strides || output->ndim <= 0 || output->ndim > 8 ||
        input->ndim < 0 || input->ndim > output->ndim) return -1;
    long contiguous[8] = {0};
    if (input->ndim > 0 && tensor_contiguous_strides(input, contiguous) != 0) return -1;
    for (int dimension = 0; dimension < 8; dimension++) strides[dimension] = 0;
    int offset = output->ndim - input->ndim;
    for (int dimension = 0; dimension < output->ndim; dimension++) {
        int source = dimension - offset;
        int input_size = source < 0 ? 1 : input->shape[source];
        int output_size = output->shape[dimension];
        if (output_size <= 0 || (input_size != 1 && input_size != output_size)) return -1;
        if (source >= 0 && input_size != 1) strides[dimension] = contiguous[source];
    }
    return 0;
}

static long tensor_broadcast_index(long output_index, int rank,
                                   const long output_strides[8],
                                   const long input_strides[8]) {
    long remainder = output_index;
    long input_index = 0;
    for (int dimension = 0; dimension < rank; dimension++) {
        long coordinate = remainder / output_strides[dimension];
        remainder %= output_strides[dimension];
        input_index += coordinate * input_strides[dimension];
    }
    return input_index;
}

static T* node_input_default(Node* n) {
    T* in = nin(n, "input");
    if (!in) in = nin(n, "x");
    if (!in) in = nin(n, "data");
    if (!in && n->nin > 0) in = t_find(n->ins[0].name);
    return in;
}

static int groupnorm_dimensions(const T* in, const T* out, const T* weight,
                                const T* bias, int groups, int* batch, int* height,
                                int* width, int* channels) {
    if (!in || !out || !weight || in->dtype != T_F32 || out->dtype != T_F32 ||
        weight->dtype != T_F32 || in->ndim != 4 || out->ndim != 4 ||
        in->numel != out->numel) return -1;
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || groups <= 0 || c % groups ||
        out->shape[0] != n || out->shape[1] != h || out->shape[2] != w ||
        out->shape[3] != c || weight->numel != c ||
        (bias && (bias->dtype != T_F32 || bias->numel != c))) return -1;
    if (batch) *batch = n;
    if (height) *height = h;
    if (width) *width = w;
    if (channels) *channels = c;
    return 0;
}

static int groupnorm_forward_f32(const T* in, const T* weight, const T* bias,
                                 T* out, int groups, float epsilon) {
    int batch, height, width, channels;
    if (!(epsilon > 0.0f) || !isfinite(epsilon) ||
        groupnorm_dimensions(in, out, weight, bias, groups, &batch, &height,
                             &width, &channels) != 0) return -1;
    return groupnorm_f32(in->data, weight->data, bias ? bias->data : NULL,
                         out->data, (uint32_t)batch, (uint32_t)height,
                         (uint32_t)width, (uint32_t)channels,
                         (uint32_t)groups, (double)epsilon) ? 0 : -1;
}

static int last_axis_reduction_dimensions(Node* n, T* out, T** input,
                                          long* rows, int* width, float* scale) {
    T* in = node_input_default(n);
    if (!in) in = nin(n, "data");
    if (!n || !in || !out || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim <= 0) return -1;
    int axis = p_int(n->params, "axis", in->ndim - 1);
    if (axis < 0) axis += in->ndim;
    int d = in->shape[in->ndim - 1];
    long r = d > 0 ? in->numel / d : 0;
    if (axis != in->ndim - 1 || d <= 0 || r <= 0 || in->numel != r * d ||
        out->numel != r) return -1;
    if (input) *input = in;
    if (rows) *rows = r;
    if (width) *width = d;
    if (scale) *scale = !strcmp(n->op, "ReduceMean") ? 1.0f / (float)d : 1.0f;
    return 0;
}

/* ONNX-style F32 ArgMax with an ordinary I32 index output.  The decoder-row
 * case is intentionally narrow: incremental inference currently exposes one
 * active sequence row for B=1 rank-3 logits. */
static int argmax_forward_f32(Node* node, T* output) {
    T* input = node_input_default(node);
    long outer = 1;
    long inner = 1;
    int axis;
    int keepdims;
    int select_last_index;
    if (!node || !input || !output || !input->data || !output->data ||
        input->dtype != T_F32 || output->dtype != T_I32 ||
        output->elem_size != sizeof(int32_t) || input->ndim <= 0 ||
        input->ndim > 8 || input->data == output->data) return -1;
    axis = p_int(node->params, "axis", 0);
    if (axis < 0) axis += input->ndim;
    keepdims = p_int(node->params, "keepdims", 1);
    select_last_index = p_int(node->params, "select_last_index", 0);
    if (axis < 0 || axis >= input->ndim || (keepdims != 0 && keepdims != 1) ||
        select_last_index != 0 ||
        input->shape[axis] <= 0) return -1;
    for (int dimension = 0; dimension < axis; dimension++) {
        if (input->shape[dimension] <= 0 ||
            outer > LONG_MAX / input->shape[dimension]) return -1;
        outer *= input->shape[dimension];
    }
    for (int dimension = axis + 1; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0 ||
            inner > LONG_MAX / input->shape[dimension]) return -1;
        inner *= input->shape[dimension];
    }
    if (outer <= 0 || inner <= 0 || outer > LONG_MAX / inner ||
        output->numel != outer * inner) return -1;
    if (keepdims) {
        if (output->ndim != input->ndim) return -1;
        for (int dimension = 0; dimension < input->ndim; dimension++) {
            int expected = dimension == axis ? 1 : input->shape[dimension];
            if (output->shape[dimension] != expected) return -1;
        }
    } else {
        if (output->ndim != input->ndim - 1) return -1;
        for (int dimension = 0; dimension < input->ndim; dimension++) {
            if (dimension == axis) continue;
            int output_dimension = dimension < axis ? dimension : dimension - 1;
            if (output->shape[output_dimension] != input->shape[dimension]) return -1;
        }
    }

    long first_outer = 0;
    long outer_count = outer;
    if (g_active_row >= 0) {
        if (input->ndim != 3 || input->shape[0] != 1 || axis != 2 ||
            inner != 1 || output->ndim != (keepdims ? 3 : 2) ||
            g_active_row >= input->shape[1]) return -1;
        first_outer = g_active_row;
        outer_count = 1;
    }
    const int axis_size = input->shape[axis];
    int32_t* indices = (int32_t*)output->data;
    for (long outer_index = first_outer;
         outer_index < first_outer + outer_count; outer_index++) {
        for (long inner_index = 0; inner_index < inner; inner_index++) {
            long base = (outer_index * axis_size * inner) + inner_index;
            float best = input->data[base];
            int best_index = 0;
            for (int axis_index = 1; axis_index < axis_size; axis_index++) {
                float value = input->data[base + (long)axis_index * inner];
                if (value > best) {
                    best = value;
                    best_index = axis_index;
                }
            }
            indices[outer_index * inner + inner_index] = best_index;
        }
    }
    return 0;
}

static int gpu_broadcast_strides(const T* input, const T* output, uint32_t strides[8]) {
    if (!input || !output || output->ndim < 0 || output->ndim > 8 ||
        input->ndim < 0 || input->ndim > output->ndim) return -1;
    for (int i = 0; i < 8; i++) strides[i] = 0;
    long contiguous[8] = {0};
    long stride = 1;
    for (int i = input->ndim - 1; i >= 0; i--) {
        if (input->shape[i] <= 0 || stride > UINT32_MAX) return -1;
        contiguous[i] = stride;
        stride *= input->shape[i];
    }
    int offset = output->ndim - input->ndim;
    for (int dimension = 0; dimension < output->ndim; dimension++) {
        int source = dimension - offset;
        int input_dim = source < 0 ? 1 : input->shape[source];
        int output_dim = output->shape[dimension];
        if (output_dim <= 0 || (input_dim != 1 && input_dim != output_dim)) return -1;
        if (source >= 0 && input_dim != 1) strides[dimension] = (uint32_t)contiguous[source];
    }
    return 0;
}

#if VOLVOXAI_ENABLE_TRAINING
#include "../training/training_forward_dropout.inc"
#include "../training/training_runtime.inc"
#endif

// ---- Shape-aware Add/Mul (row-windowed) -----------------------------------
static int binop(Node* n, int is_mul) {
    T* a = nin(n, "a"); T* b = nin(n, "b"); T* o = t_find(n->out);
    if (!a) a = nin(n, "input");
    if (!a || !b || !o || a->dtype != T_F32 || b->dtype != T_F32 || o->dtype != T_F32 ||
        o->ndim <= 0 || o->ndim > 8) return -1;
    long output_strides[8] = {0}, a_strides[8] = {0}, b_strides[8] = {0};
    if (tensor_contiguous_strides(o, output_strides) != 0 ||
        tensor_broadcast_strides(a, o, a_strides) != 0 ||
        tensor_broadcast_strides(b, o, b_strides) != 0) return -1;
    int D = o->shape[o->ndim - 1]; long total = o->numel / D;
    int r0, rc; seq_range(total, &r0, &rc);
    long off = (long)r0 * D; int cnt = rc * D;
    int relu = p_int(n->params, "relu", 0);
    if (a->numel == o->numel && b->numel == o->numel) {
        if (is_mul) mul_f32(a->data + off, b->data + off, o->data + off, cnt);
        else        vx_fused_add_relu_f32(a->data + off, b->data + off, o->data + off, cnt, relu);
    } else if (a->ndim > 0 && b->ndim > 0 && off == 0 && cnt == o->numel &&
               o->numel <= UINT32_MAX) {
        uint32_t a_shape[8], b_shape[8], output_shape[8];
        for (int dimension = 0; dimension < a->ndim; dimension++)
            a_shape[dimension] = (uint32_t)a->shape[dimension];
        for (int dimension = 0; dimension < b->ndim; dimension++)
            b_shape[dimension] = (uint32_t)b->shape[dimension];
        for (int dimension = 0; dimension < o->ndim; dimension++)
            output_shape[dimension] = (uint32_t)o->shape[dimension];
        if (!binary_broadcast_f32(
                a->data, b->data, o->data, a_shape, b_shape, output_shape,
                (uint32_t)a->ndim, (uint32_t)b->ndim, (uint32_t)o->ndim,
                (uint32_t)o->numel,
                is_mul ? VX_PORTABLE_BINARY_MUL : VX_PORTABLE_BINARY_ADD))
            return -1;
        if (!is_mul && relu == 1)
            relu_f32(o->data, o->data, (int)o->numel);
        else if (!is_mul && relu >= 2)
            clip_f32(o->data, o->data, (int)o->numel, 0.0f, 6.0f);
    } else {
        for (long output_index = off; output_index < off + cnt; output_index++) {
            long a_index = tensor_broadcast_index(output_index, o->ndim,
                                                  output_strides, a_strides);
            long b_index = tensor_broadcast_index(output_index, o->ndim,
                                                  output_strides, b_strides);
            float value = is_mul ? a->data[a_index] * b->data[b_index]
                                 : a->data[a_index] + b->data[b_index];
            if (!is_mul && relu) {
                if (value < 0.0f) value = 0.0f;
                if (relu >= 2 && value > 6.0f) value = 6.0f;
            }
            o->data[output_index] = value;
        }
    }
    return 0;
}

static int transpose_tensor(const T* in, T* out, const int* perm) {
    uint32_t input_shape[8], permutation[8];
    if (!in || !out || !perm || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim <= 0 || in->ndim > 8 || out->ndim != in->ndim ||
        in->numel != out->numel || in->numel <= 0 || in->numel > UINT32_MAX)
        return -1;
    for (int axis = 0; axis < in->ndim; axis++) {
        if (in->shape[axis] <= 0 || perm[axis] < 0 || perm[axis] >= in->ndim ||
            out->shape[axis] != in->shape[perm[axis]])
            return -1;
        input_shape[axis] = (uint32_t)in->shape[axis];
        permutation[axis] = (uint32_t)perm[axis];
    }
    return transpose_nd_f32(in->data, out->data, input_shape, permutation,
                            (uint32_t)in->ndim,
                            (uint32_t)in->numel) ? 0 : -1;
}

static int concat_tensor(Node* n, T* out, int axis) {
    if (!n || !out || out->dtype != T_F32 || out->ndim <= 0 || out->ndim > 8)
        return -1;
    if (axis < 0) axis += out->ndim;
    if (axis < 0 || axis >= out->ndim || out->shape[axis] <= 0) return -1;
    long inner = 1;
    for (int i = axis + 1; i < out->ndim; i++) {
        if (out->shape[i] <= 0 || inner > LONG_MAX / out->shape[i]) return -1;
        inner *= out->shape[i];
    }
    long output_block = (long)out->shape[axis] * inner;
    long outer = output_block > 0 ? out->numel / output_block : 0;
    if (inner <= 0 || outer <= 0 || inner > UINT32_MAX || outer > UINT32_MAX)
        return -1;
    if (out->numel != outer * output_block) return -1;
    int axis_offset = 0;
    for (int input_index = 0; input_index < n->nin; input_index++) {
        T* source = t_find(n->ins[input_index].name);
        if (!source) continue;
        if (source->dtype != T_F32 || source->ndim != out->ndim ||
            source->shape[axis] <= 0 ||
            axis_offset > out->shape[axis] - source->shape[axis])
            return -1;
        for (int dimension = 0; dimension < out->ndim; dimension++) {
            if (dimension != axis && source->shape[dimension] != out->shape[dimension])
                return -1;
        }
        if (source->numel != outer * (long)source->shape[axis] * inner)
            return -1;
        if (!concat_slice_f32(source->data, out->data, (uint32_t)outer,
                              (uint32_t)source->shape[axis],
                              (uint32_t)out->shape[axis], (uint32_t)inner,
                              (uint32_t)axis_offset, 0u))
            return -1;
        axis_offset += source->shape[axis];
    }
    return axis_offset == out->shape[axis] ? 0 : -1;
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

static int resize_nearest2d_image(const T* in, T* out) {
    if (!in || !out || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim != 4 || out->ndim != 4 || in->shape[0] != out->shape[0] ||
        in->shape[3] != out->shape[3])
        return -1;
    int n = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int oh = out->shape[1], ow = out->shape[2];
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || oh <= 0 || ow <= 0)
        return -1;
    return resize_nearest2d_f32(
        in->data, out->data, (uint32_t)n, (uint32_t)h, (uint32_t)w,
        (uint32_t)c, (uint32_t)oh, (uint32_t)ow) ? 0 : -1;
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

static int quantized_weight_i32(const T* wt, long idx) {
    if (wt->dtype == T_U8) return ((const unsigned char*)wt->data)[idx];
    return ((const signed char*)wt->data)[idx];
}

static float* qconv_weight_only_dequant_cache(int node_idx, const T* wt, const T* wscale,
                                              const T* wzp, const char* layout) {
    if (g_qwcache[node_idx]) return g_qwcache[node_idx];
    float* out = (float*)malloc((size_t)wt->numel * sizeof(float));
    if (!out) return NULL;
    int ohwi = layout && !strcmp(layout, "OHWI");
    int hwo = layout && (!strcmp(layout, "1HWO") || !strcmp(layout, "1HWM"));
    if (ohwi) {
        int out_c = wt->shape[0], kh = wt->shape[1], kw = wt->shape[2], in_c = wt->shape[3];
        for (int oc = 0; oc < out_c; oc++) {
            float ws = tensor_scalar_f32(wscale, wscale->numel == 1 ? 0 : oc);
            int wz = wzp ? (int)tensor_scalar_any(wzp, wzp->numel == 1 ? 0 : oc) : 0;
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
            float ws = tensor_scalar_f32(wscale, wscale->numel == 1 ? 0 : oc);
            int wz = wzp ? (int)tensor_scalar_any(wzp, wzp->numel == 1 ? 0 : oc) : 0;
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
            float ws = tensor_scalar_f32(wscale, wscale->numel == 1 ? 0 : oc);
            int wz = wzp ? (int)tensor_scalar_any(wzp, wzp->numel == 1 ? 0 : oc) : 0;
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
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
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
                conv_weights, biases, pointwise_packs, igemm_indirs, volvoxai_engine_now_ms() - t0);
    }
    return 0;
}

static int qconv2d_weight_only_f32(int node_idx, const T* in, T* out, const T* wt,
                                   const T* wscale, const T* wzp, const T* bias,
                                   int sy, int sx, const int* pads, int groups,
                                   int dy, int dx, int relu, cJSON* params,
                                   const char* weight_layout) {
    float* w_deq = qconv_weight_only_dequant_cache(
        node_idx, wt, wscale, wzp, weight_layout);
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
