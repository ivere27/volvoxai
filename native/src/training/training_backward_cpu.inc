/*
 * CPU reference backward kernels for native training.
 *
 * Fragment of the training runtime translation unit: included only from
 * training_runtime.inc, never compiled or included on its own. Defines the
 * gradient helpers and the per-operator backward passes (backward_matmul,
 * backward_conv2d, ...) that the accelerated kernels are validated against.
 */

static float* grad_for(float** grads, T* t) {
    if (!t || t->dtype != T_F32 || t->numel <= 0) return NULL;
    int idx = t_index(t);
    if (idx < 0) return NULL;
    if (!grads[idx]) grads[idx] = (float*)calloc((size_t)t->numel, sizeof(float));
    return grads[idx];
}

static float* grad_existing(float** grads, T* t) {
    int idx = t_index(t);
    return idx >= 0 ? grads[idx] : NULL;
}

static void grad_add_full(float* dst, const float* src, long n, float scale) {
    if (!dst || !src) return;
    for (long i = 0; i < n; i++) dst[i] += src[i] * scale;
}

static int training_node_pair(const Node* node, VxNodeParamKey key,
                              VxNodeParamArraySlot slot, int default_value,
                              int minimum, int required, int output[2]) {
    const VxCachedNodeParam* cached = vx_node_param(node, key);
    int count = 0;
    const int32_t* values;
    if (!node || !cached || !output) return -1;
    if (cached->kind == VX_NODE_PARAM_ABSENT) {
        if (required) return -1;
        output[0] = default_value;
        output[1] = default_value;
        return 0;
    }
    values = vx_node_param_array(node, key, slot, &count);
    if (!values || count != 2 || values[0] < minimum || values[1] < minimum)
        return -1;
    output[0] = values[0];
    output[1] = values[1];
    return 0;
}

static int training_gelu_approximate_tanh(const Node* node, int* output) {
    const VxCachedNodeParam* cached = vx_node_param(
        node, VX_NODE_PARAM_APPROXIMATE);
    VxNodeParamSymbol approximate;
    if (!cached || !output) return -1;
    if (cached->kind == VX_NODE_PARAM_ABSENT) {
        *output = 0;
        return 0;
    }
    if (cached->kind != VX_NODE_PARAM_SYMBOL) return -1;
    approximate = (VxNodeParamSymbol)cached->symbol;
    if (approximate == VX_NODE_SYMBOL_NONE) {
        *output = 0;
        return 0;
    }
    if (approximate == VX_NODE_SYMBOL_TANH) {
        *output = 1;
        return 0;
    }
    return -1;
}

static int grad_add_broadcast_tensor(float* destination, const T* input,
                                     const float* source, const T* output) {
    long output_strides[8] = {0}, input_strides[8] = {0};
    if (!destination || !source || tensor_contiguous_strides(output, output_strides) != 0 ||
        tensor_broadcast_strides(input, output, input_strides) != 0) return -1;
    for (long output_index = 0; output_index < output->numel; output_index++) {
        long input_index = tensor_broadcast_index(output_index, output->ndim,
                                                  output_strides, input_strides);
        destination[input_index] += source[output_index];
    }
    return 0;
}

static int transpose_backward(float** grads, Node* n, T* in, T* out, const float* go) {
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 || in->numel != out->numel) return -1;
    float* gi = grad_for(grads, in);
    if (!gi) return -1;
    int rank = in->ndim;
    if (rank <= 0 || rank > 8) return -1;
    int perm[8];
    int perm_count = 0;
    const int32_t* cached_perm = vx_node_param_array(
        n, VX_NODE_PARAM_PERM, VX_NODE_ARRAY_PERM, &perm_count);
    if (cached_perm) {
        if (perm_count != rank) return -1;
        for (int i = 0; i < rank; i++) perm[i] = cached_perm[i];
    } else {
        if (vx_node_param(n, VX_NODE_PARAM_PERM)->kind !=
            VX_NODE_PARAM_ABSENT) return -1;
        for (int i = 0; i < rank; i++) perm[i] = rank - 1 - i;
    }
    long in_strides[8], out_strides[8];
    long s = 1;
    for (int i = rank - 1; i >= 0; i--) { in_strides[i] = s; s *= in->shape[i]; }
    s = 1;
    for (int i = rank - 1; i >= 0; i--) { out_strides[i] = s; s *= out->shape[i]; }
    for (long oi = 0; oi < out->numel; oi++) {
        long tmp = oi;
        long ii = 0;
        for (int d = 0; d < rank; d++) {
            long coord = out_strides[d] ? tmp / out_strides[d] : 0;
            tmp = out_strides[d] ? tmp % out_strides[d] : 0;
            if (perm[d] < 0 || perm[d] >= rank) return -1;
            ii += coord * in_strides[perm[d]];
        }
        gi[ii] += go[oi];
    }
    return 0;
}

static int backward_matmul(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    T* w = nin(n, VX_PORT_WEIGHT);
    T* b = nin(n, VX_PORT_BIAS);
    if (!in || !w || !out || !go || in->dtype != T_F32 ||
        (w->dtype != T_F32 && w->dtype != T_F16) || out->dtype != T_F32) return -1;
    materialize_tensor_f32(in);
    int k = in->shape[in->ndim - 1];
    int d_out = out->shape[out->ndim - 1];
    long rows = out->numel / d_out;
    if (in->numel != rows * (long)k || w->numel != (long)k * d_out) return -1;
    int out_in = 0;
    if (linear_node_weight_layout(n, in, out, w, &out_in) != 0) return -1;
    if (b && ((b->dtype != T_F32 && b->dtype != T_F16) || b->numel != d_out)) return -1;
    float* gi = grad_for(grads, in);
    float* gw = w->dtype == T_F32 ? grad_for(grads, w) : NULL;
    float* gb = b && b->dtype == T_F32 ? grad_for(grads, b) : NULL;
    if (!gi || (w->dtype == T_F32 && !gw) || (b && b->dtype == T_F32 && !gb)) return -1;
    if (w->dtype == T_F32 && rows > 0 && (unsigned long)rows <= UINT32_MAX) {
        uint32_t packed_elements = vx_gemm_f32_packed_elements(
            (uint32_t)d_out, (uint32_t)k);
        float* packed = packed_elements &&
            (size_t)packed_elements <= SIZE_MAX / sizeof(float)
            ? (float*)malloc((size_t)packed_elements * sizeof(float)) : NULL;
        /* dx = dY * W^T.  Relative to that GEMM, an IN_OUT forward weight is
         * OUT_IN storage and vice versa.  dW/db retain their additive scalar
         * reductions so each destination has one unambiguous owner. */
        if (packed && vx_gemm_f32_pack_b(w->data, packed,
                (uint32_t)d_out, (uint32_t)k, out_in ? 0 : 1) &&
            vx_gemm_f32_run_packed_add(go, packed, NULL, gi,
                (uint32_t)rows, (uint32_t)d_out, (uint32_t)k)) {
            free(packed);
            for (long r = 0; r < rows; r++) {
                const float* x = in->data + r * k;
                const float* dy = go + r * d_out;
                for (int j = 0; j < d_out; j++) {
                    float g = dy[j];
                    if (gb) gb[j] += g;
                    for (int kk = 0; gw && kk < k; kk++) {
                        long wi = out_in ? (long)j * k + kk : (long)kk * d_out + j;
                        gw[wi] += x[kk] * g;
                    }
                }
            }
            return 0;
        }
        free(packed);
    }
    for (long r = 0; r < rows; r++) {
        const float* x = in->data + r * k;
        const float* dy = go + r * d_out;
        for (int j = 0; j < d_out; j++) {
            float g = dy[j];
            if (gb) gb[j] += g;
            for (int kk = 0; kk < k; kk++) {
                long wi = out_in ? (long)j * k + kk : (long)kk * d_out + j;
                gi[r * k + kk] += g * tensor_scalar_f32(w, wi);
                if (gw) gw[wi] += x[kk] * g;
            }
        }
    }
    return 0;
}

static int backward_mul(float** grads, Node* n, T* out, const float* go) {
    T* a = nin(n, VX_PORT_A);
    if (!a) a = node_input_default(n);
    T* b = nin(n, VX_PORT_B);
    if (!a || !b || !out || !go || a->dtype != T_F32 || b->dtype != T_F32) return -1;
    float* ga = grad_for(grads, a);
    float* gb = grad_for(grads, b);
    long output_strides[8] = {0}, a_strides[8] = {0}, b_strides[8] = {0};
    if (!ga || !gb || tensor_contiguous_strides(out, output_strides) != 0 ||
        tensor_broadcast_strides(a, out, a_strides) != 0 ||
        tensor_broadcast_strides(b, out, b_strides) != 0) return -1;
    for (long output_index = 0; output_index < out->numel; output_index++) {
        long a_index = tensor_broadcast_index(output_index, out->ndim,
                                              output_strides, a_strides);
        long b_index = tensor_broadcast_index(output_index, out->ndim,
                                              output_strides, b_strides);
        ga[a_index] += go[output_index] * b->data[b_index];
        gb[b_index] += go[output_index] * a->data[a_index];
    }
    return 0;
}

static int backward_embedding(float** grads, Node* n, T* out, const float* go) {
    T* tokens = node_input_default(n);
    T* w = nin(n, VX_PORT_WEIGHT);
    if (!tokens || !w || !out || !go || tokens->dtype != T_I32 ||
        w->dtype != T_F32 || out->dtype != T_F32 || w->ndim <= 0) return -1;
    int d_model = w->shape[w->ndim - 1];
    if (d_model <= 0 || w->numel % d_model ||
        out->numel != tokens->numel * (long)d_model) return -1;
    long vocab = w->numel / d_model;
    float* gw = grad_for(grads, w);
    if (!gw) return -1;
    const int32_t* ids = (const int32_t*)tokens->data;
    for (long row = 0; row < tokens->numel; row++) {
        int id = ids[row];
        if (id < 0 || id >= vocab) return -1;
        grad_add_full(gw + (long)id * d_model, go + row * d_model, d_model, 1.0f);
    }
    return 0;
}

static int backward_groupnorm(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    if (!weight) weight = nin(n, VX_PORT_SCALE);
    T* bias = nin(n, VX_PORT_BIAS);
    int groups = vx_node_param_i32(n, VX_NODE_PARAM_NUM_GROUPS,
        vx_node_param_i32(n, VX_NODE_PARAM_GROUPS, 1));
    float epsilon = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-5f);
    int batch, height, width, channels;
    if (!go || !(epsilon > 0.0f) || !isfinite(epsilon) ||
        groupnorm_dimensions(in, out, weight, bias, groups, &batch, &height,
                             &width, &channels) != 0) return -1;
    float* grad_input = grad_for(grads, in);
    float* grad_weight = grad_for(grads, weight);
    float* grad_bias = bias ? grad_for(grads, bias) : NULL;
    if (!grad_input || !grad_weight || (bias && !grad_bias)) return -1;
    int channels_per_group = channels / groups;
    long area = (long)height * width;
    long count = area * channels_per_group;
    for (int sample = 0; sample < batch; sample++) {
        for (int group = 0; group < groups; group++) {
            int channel_start = group * channels_per_group;
            double sum = 0.0, square_sum = 0.0;
            for (long spatial = 0; spatial < area; spatial++) {
                long base = ((long)sample * area + spatial) * channels + channel_start;
                for (int local = 0; local < channels_per_group; local++) {
                    double value = in->data[base + local];
                    sum += value;
                    square_sum += value * value;
                }
            }
            float mean = (float)(sum / count);
            float variance = (float)(square_sum / count - (double)mean * mean);
            if (variance < 0.0f) variance = 0.0f;
            float inverse_stddev = 1.0f / sqrtf(variance + epsilon);
            double sum_scaled_grad = 0.0, sum_scaled_grad_xhat = 0.0;
            for (long spatial = 0; spatial < area; spatial++) {
                long base = ((long)sample * area + spatial) * channels + channel_start;
                for (int local = 0; local < channels_per_group; local++) {
                    int channel = channel_start + local;
                    long index = base + local;
                    float xhat = (in->data[index] - mean) * inverse_stddev;
                    float dy = go[index];
                    float scaled_grad = dy * weight->data[channel];
                    sum_scaled_grad += scaled_grad;
                    sum_scaled_grad_xhat += (double)scaled_grad * xhat;
                    grad_weight[channel] += dy * xhat;
                    if (grad_bias) grad_bias[channel] += dy;
                }
            }
            for (long spatial = 0; spatial < area; spatial++) {
                long base = ((long)sample * area + spatial) * channels + channel_start;
                for (int local = 0; local < channels_per_group; local++) {
                    int channel = channel_start + local;
                    long index = base + local;
                    float xhat = (in->data[index] - mean) * inverse_stddev;
                    float scaled_grad = go[index] * weight->data[channel];
                    grad_input[index] += inverse_stddev / (float)count *
                        ((float)count * scaled_grad - (float)sum_scaled_grad -
                         xhat * (float)sum_scaled_grad_xhat);
                }
            }
        }
    }
    return 0;
}

static int backward_dropout(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    int node_index = (int)(n - g_n);
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->numel != out->numel || in->numel > UINT32_MAX || node_index < 0 ||
        node_index >= g_nn) return -1;
    NativeDropoutParams params;
    float* grad_input = grad_for(grads, in);
    if (!grad_input || native_dropout_params(n, node_index, &params) != 0) return -1;
    for (long index = 0; index < in->numel; index++) {
        if (dropout_random_bits(params.seed, params.counter, (uint32_t)index) >= params.threshold) {
            grad_input[index] += go[index] * params.scale;
        }
    }
    return 0;
}

static int backward_last_axis_reduction(float** grads, Node* n, T* out, const float* go) {
    T* in = NULL;
    long rows = 0;
    int width = 0;
    float scale = 1.0f;
    if (!go || last_axis_reduction_dimensions(n, out, &in, &rows, &width, &scale) != 0)
        return -1;
    float* grad_input = grad_for(grads, in);
    if (!grad_input) return -1;
    for (long row = 0; row < rows; row++) {
        float value = go[row] * scale;
        for (int column = 0; column < width; column++)
            grad_input[row * width + column] += value;
    }
    return 0;
}

static int backward_layernorm(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    T* w = nin(n, VX_PORT_WEIGHT);
    T* b = nin(n, VX_PORT_BIAS);
    if (!in || !w || !out || !go || in->dtype != T_F32 ||
        w->dtype != T_F32 || out->dtype != T_F32 || in->ndim <= 0) return -1;
    int d = vx_node_param_i32(
        n, VX_NODE_PARAM_D_MODEL, in->shape[in->ndim - 1]);
    if (d <= 0 || in->numel != out->numel || in->numel % d || w->numel < d ||
        (b && (b->dtype != T_F32 || b->numel < d))) return -1;
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, w);
    float* gb = b ? grad_for(grads, b) : NULL;
    if (!gi || !gw || (b && !gb)) return -1;
    long rows = in->numel / d;
    float eps = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-6f);
    for (long r = 0; r < rows; r++) {
        const float* x = in->data + r * d;
        const float* dy = go + r * d;
        float* dx = gi + r * d;
        double mean_acc = 0.0;
        for (int j = 0; j < d; j++) mean_acc += x[j];
        float mean = (float)(mean_acc / d);
        double var_acc = 0.0;
        for (int j = 0; j < d; j++) {
            double centered = (double)x[j] - mean;
            var_acc += centered * centered;
        }
        float inv = 1.0f / sqrtf((float)(var_acc / d) + eps);
        double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
        for (int j = 0; j < d; j++) {
            float xhat = (x[j] - mean) * inv;
            float dxhat = dy[j] * w->data[j];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += (double)dxhat * xhat;
            gw[j] += dy[j] * xhat;
            if (gb) gb[j] += dy[j];
        }
        for (int j = 0; j < d; j++) {
            float xhat = (x[j] - mean) * inv;
            float dxhat = dy[j] * w->data[j];
            dx[j] += inv * (dxhat - (float)(sum_dxhat / d) -
                            xhat * (float)(sum_dxhat_xhat / d));
        }
    }
    return 0;
}

static int backward_rmsnorm(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    T* w = nin(n, VX_PORT_WEIGHT);
    if (!in || !w || !out || !go || in->dtype != T_F32 ||
        w->dtype != T_F32 || out->dtype != T_F32 || in->ndim <= 0) return -1;
    int d = vx_node_param_i32(
        n, VX_NODE_PARAM_D_MODEL, in->shape[in->ndim - 1]);
    if (d <= 0 || in->numel != out->numel || in->numel % d || w->numel < d) return -1;
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, w);
    if (!gi || !gw) return -1;
    long rows = in->numel / d;
    float eps = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-6f);
    for (long r = 0; r < rows; r++) {
        const float* x = in->data + r * d;
        const float* dy = go + r * d;
        float* dx = gi + r * d;
        double ss = 0.0;
        for (int j = 0; j < d; j++) ss += (double)x[j] * x[j];
        float inv = 1.0f / sqrtf((float)(ss / d) + eps);
        double dot = 0.0;
        for (int j = 0; j < d; j++) {
            float u = dy[j] * w->data[j];
            dot += (double)u * x[j];
            gw[j] += dy[j] * x[j] * inv;
        }
        float correction = inv * inv * inv * (float)(dot / d);
        for (int j = 0; j < d; j++) {
            dx[j] += inv * dy[j] * w->data[j] - x[j] * correction;
        }
    }
    return 0;
}

static int backward_activation(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->numel != out->numel) return -1;
    float* gi = grad_for(grads, in);
    if (!gi) return -1;
    const float gelu_c = 0.7978845608028654f;
    int gelu_tanh = 0;
    if ((n->operator_kind == VX_OP_GELU) &&
        training_gelu_approximate_tanh(n, &gelu_tanh) != 0)
        return -1;
    for (long i = 0; i < out->numel; i++) {
        float x = in->data[i];
        float derivative;
        if (n->operator_kind == VX_OP_RELU) {
            derivative = x > 0.0f ? 1.0f : 0.0f;
        } else if (n->operator_kind == VX_OP_GELU) {
            if (gelu_tanh) {
                float x2 = x * x;
                float u = gelu_c * (x + 0.044715f * x * x2);
                float t = tanhf(u);
                derivative = 0.5f * (1.0f + t) +
                             0.5f * x * (1.0f - t * t) * gelu_c * (1.0f + 0.134145f * x2);
            } else {
                derivative = 0.5f * (1.0f + erff(x * 0.7071067811865475f)) +
                             x * 0.3989422804014327f * expf(-0.5f * x * x);
            }
        } else if (n->operator_kind == VX_OP_SILU) {
            float s = 1.0f / (1.0f + expf(-x));
            derivative = s * (1.0f + x * (1.0f - s));
        } else if (n->operator_kind == VX_OP_SIGMOID) {
            float y = out->data[i];
            derivative = y * (1.0f - y);
        } else if (n->operator_kind == VX_OP_TANH) {
            float y = out->data[i];
            derivative = 1.0f - y * y;
        } else if (n->operator_kind == VX_OP_HARD_SIGMOID) {
            derivative = (x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f;
        } else if (n->operator_kind == VX_OP_HARD_SWISH) {
            derivative = x <= -3.0f ? 0.0f : x >= 3.0f ? 1.0f : x / 3.0f + 0.5f;
        } else if (n->operator_kind == VX_OP_LEAKY_RELU) {
            derivative = x > 0.0f ? 1.0f : vx_node_param_f32(
                n, VX_NODE_PARAM_ALPHA, 0.01f);
        } else if (n->operator_kind == VX_OP_CLIP) {
            T* lower = nin(n, VX_PORT_MIN);
            T* upper = nin(n, VX_PORT_MAX);
            float lo = lower ? lower->data[0] : n->parsed_params.min_val;
            float hi = upper ? upper->data[0] : n->parsed_params.max_val;
            derivative = (x > lo && x < hi) ? 1.0f : 0.0f;
        } else {
            return -1;
        }
        gi[i] += go[i] * derivative;
    }
    return 0;
}

static int backward_prelu(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    if (!weight) weight = nin(n, VX_PORT_SLOPE);
    if (!in || !weight || !out || !go || in->dtype != T_F32 ||
        weight->dtype != T_F32 || out->dtype != T_F32 || in->ndim <= 0 ||
        in->numel != out->numel) return -1;
    int channels = in->shape[in->ndim - 1];
    if (channels <= 0 || (weight->numel != 1 && weight->numel != channels)) return -1;
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, weight);
    if (!gi || !gw) return -1;
    for (long i = 0; i < in->numel; i++) {
        long wi = weight->numel == 1 ? 0 : i % channels;
        float x = in->data[i];
        if (x > 0.0f) {
            gi[i] += go[i];
        } else {
            gi[i] += go[i] * weight->data[wi];
            gw[wi] += go[i] * x;
        }
    }
    return 0;
}

static int backward_softmax(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->numel != out->numel || in->ndim <= 0) return -1;
    int axis = vx_node_param_i32(n, VX_NODE_PARAM_AXIS, in->ndim - 1);
    if (axis < 0) axis += in->ndim;
    if (axis != in->ndim - 1) return -1;
    int d = in->shape[in->ndim - 1];
    if (d <= 0 || in->numel % d) return -1;
    float* gi = grad_for(grads, in);
    if (!gi) return -1;
    long rows = in->numel / d;
    for (long r = 0; r < rows; r++) {
        const float* y = out->data + r * d;
        const float* dy = go + r * d;
        if (n->operator_kind == VX_OP_LOG_SOFTMAX) {
            double sum = 0.0;
            for (int j = 0; j < d; j++) sum += dy[j];
            for (int j = 0; j < d; j++) {
                gi[r * d + j] += dy[j] - expf(y[j]) * (float)sum;
            }
        } else {
            double dot = 0.0;
            for (int j = 0; j < d; j++) dot += (double)dy[j] * y[j];
            for (int j = 0; j < d; j++) gi[r * d + j] += y[j] * (dy[j] - (float)dot);
        }
    }
    return 0;
}

static int backward_sdpa(float** grads, Node* n, T* out, const float* go) {
    T* qkv = nin(n, VX_PORT_QKV);
    if (!qkv) qkv = node_input_default(n);
    T* mask = nin(n, VX_PORT_MASK);
    if (!qkv || !out || !go || qkv->dtype != T_F32 || out->dtype != T_F32 ||
        qkv->ndim <= 0 || out->ndim <= 0) return -1;
    if ((qkv->ndim != 2 && qkv->ndim != 3) || out->ndim != qkv->ndim) return -1;
    int batch = qkv->ndim == 3 ? qkv->shape[0] : 1;
    int out_batch = out->ndim == 3 ? out->shape[0] : 1;
    int seq = qkv->shape[qkv->ndim - 2];
    int d_model = out->shape[out->ndim - 1];
    int heads = vx_node_param_i32(n, VX_NODE_PARAM_HEADS, 8);
    if (d_model <= 0 || heads <= 0 || d_model % heads ||
        batch <= 0 || batch != out_batch || seq <= 0 || out->shape[out->ndim - 2] != seq ||
        qkv->shape[qkv->ndim - 1] != 3 * d_model ||
        qkv->numel != (long)batch * seq * 3 * d_model) return -1;
    int head_dim = d_model / heads;
    if (out->numel != (long)batch * seq * d_model) return -1;
    int mask_mode = attention_mask_mode(mask, batch, seq, seq);
    if (mask_mode < 0) return -1;
    int causal = vx_node_param_i32(n, VX_NODE_PARAM_CAUSAL, 1) ? 1 : 0;
    float scale = vx_node_param_f32(
        n, VX_NODE_PARAM_SCALE, 1.0f / sqrtf((float)head_dim));
    NativeAttentionDropoutParams dropout;
    if (native_attention_dropout_params(n, (int)(n - g_n), &dropout) != 0) return -1;
    float* gqkv = grad_for(grads, qkv);
    float* probs = (float*)malloc((size_t)seq * sizeof(float));
    float* dprob = (float*)malloc((size_t)seq * sizeof(float));
    if (!gqkv || !probs || !dprob) { free(probs); free(dprob); return -1; }
    for (int batch_index = 0; batch_index < batch; batch_index++) {
      long qkv_base = (long)batch_index * seq * 3 * d_model;
      long output_base = (long)batch_index * seq * d_model;
      for (int qi = 0; qi < seq; qi++) {
        for (int h = 0; h < heads; h++) {
            const float* q = qkv->data + qkv_base + (long)qi * 3 * d_model + h * head_dim;
            const float* dy = go + output_base + (long)qi * d_model + h * head_dim;
            float max_score = -INFINITY;
            int valid_keys = 0;
            memset(probs, 0, (size_t)seq * sizeof(float));
            for (int ki = 0; ki < seq; ki++) {
                if (!attention_mask_allowed(mask, mask_mode, batch_index, qi, ki,
                                            seq, seq, causal)) continue;
                const float* k = qkv->data + qkv_base + (long)ki * 3 * d_model + d_model + h * head_dim;
                double dot = 0.0;
                for (int d = 0; d < head_dim; d++) dot += (double)q[d] * k[d];
                probs[ki] = (float)dot * scale;
                if (probs[ki] > max_score) max_score = probs[ki];
                valid_keys++;
            }
            if (!valid_keys) continue;
            double sum = 0.0;
            for (int ki = 0; ki < seq; ki++) {
                if (!attention_mask_allowed(mask, mask_mode, batch_index, qi, ki,
                                            seq, seq, causal)) continue;
                probs[ki] = expf(probs[ki] - max_score); sum += probs[ki];
            }
            for (int ki = 0; ki < seq; ki++) if (probs[ki] != 0.0f) probs[ki] /= (float)sum;
            double weighted_dprob = 0.0;
            for (int ki = 0; ki < seq; ki++) {
                if (!attention_mask_allowed(mask, mask_mode, batch_index, qi, ki,
                                            seq, seq, causal)) continue;
                const float* v = qkv->data + qkv_base + (long)ki * 3 * d_model + 2 * d_model + h * head_dim;
                double dp = 0.0;
                for (int d = 0; d < head_dim; d++) dp += (double)dy[d] * v[d];
                float multiplier = attention_dropout_multiplier(&dropout, batch_index,
                    h, qi, ki, heads, seq, seq);
                dprob[ki] = (float)dp * multiplier;
                weighted_dprob += (double)probs[ki] * dprob[ki];
                float* gv = gqkv + qkv_base + (long)ki * 3 * d_model + 2 * d_model + h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    gv[d] += probs[ki] * multiplier * dy[d];
            }
            float* gq = gqkv + qkv_base + (long)qi * 3 * d_model + h * head_dim;
            for (int ki = 0; ki < seq; ki++) {
                if (!attention_mask_allowed(mask, mask_mode, batch_index, qi, ki,
                                            seq, seq, causal)) continue;
                const float* k = qkv->data + qkv_base + (long)ki * 3 * d_model + d_model + h * head_dim;
                float* gk = gqkv + qkv_base + (long)ki * 3 * d_model + d_model + h * head_dim;
                float ds = probs[ki] * (dprob[ki] - (float)weighted_dprob);
                for (int d = 0; d < head_dim; d++) {
                    gq[d] += ds * scale * k[d];
                    gk[d] += ds * scale * q[d];
                }
            }
        }
    }
    }
    free(probs);
    free(dprob);
    return 0;
}

static int backward_cross_sdpa(float** grads, Node* n, T* out, const float* go) {
    T* q = nin(n, VX_PORT_Q);
    T* k = nin(n, VX_PORT_K);
    T* v = nin(n, VX_PORT_V);
    T* mask = nin(n, VX_PORT_MASK);
    if (!q || !k || !v || !out || !go || q->dtype != T_F32 || k->dtype != T_F32 ||
        v->dtype != T_F32 || out->dtype != T_F32 || q->ndim <= 0 || k->ndim <= 0 || v->ndim <= 0) return -1;
    if ((q->ndim != 2 && q->ndim != 3) || k->ndim != q->ndim || v->ndim != q->ndim ||
        out->ndim != q->ndim) return -1;
    int batch = q->ndim == 3 ? q->shape[0] : 1;
    int k_batch = k->ndim == 3 ? k->shape[0] : 1;
    int v_batch = v->ndim == 3 ? v->shape[0] : 1;
    int out_batch = out->ndim == 3 ? out->shape[0] : 1;
    int d_model = q->shape[q->ndim - 1];
    int heads = vx_node_param_i32(n, VX_NODE_PARAM_HEADS, 8);
    if (d_model <= 0 || heads <= 0 || d_model % heads ||
        batch <= 0 || batch != k_batch || batch != v_batch || batch != out_batch ||
        k->shape[k->ndim - 1] != d_model || v->shape[v->ndim - 1] != d_model) return -1;
    int seq_q = q->shape[q->ndim - 2];
    int seq_kv = k->shape[k->ndim - 2];
    int head_dim = d_model / heads;
    if (seq_q <= 0 || seq_kv <= 0 || v->shape[v->ndim - 2] != seq_kv ||
        out->shape[out->ndim - 2] != seq_q ||
        q->numel != (long)batch * seq_q * d_model ||
        k->numel != (long)batch * seq_kv * d_model || k->numel != v->numel ||
        out->numel != q->numel) return -1;
    int mask_mode = attention_mask_mode(mask, batch, seq_q, seq_kv);
    if (mask_mode < 0) return -1;
    int causal = vx_node_param_i32(n, VX_NODE_PARAM_CAUSAL, 0) ? 1 : 0;
    float scale = vx_node_param_f32(
        n, VX_NODE_PARAM_SCALE, 1.0f / sqrtf((float)head_dim));
    NativeAttentionDropoutParams dropout;
    if (native_attention_dropout_params(n, (int)(n - g_n), &dropout) != 0) return -1;
    float* gq = grad_for(grads, q);
    float* gk = grad_for(grads, k);
    float* gv = grad_for(grads, v);
    float* probs = (float*)malloc((size_t)seq_kv * sizeof(float));
    float* dprob = (float*)malloc((size_t)seq_kv * sizeof(float));
    if (!gq || !gk || !gv || !probs || !dprob) { free(probs); free(dprob); return -1; }
    for (int batch_index = 0; batch_index < batch; batch_index++) {
      long q_base = (long)batch_index * seq_q * d_model;
      long kv_base = (long)batch_index * seq_kv * d_model;
      for (int qi = 0; qi < seq_q; qi++) {
        for (int h = 0; h < heads; h++) {
            const float* qv = q->data + q_base + (long)qi * d_model + h * head_dim;
            const float* dy = go + q_base + (long)qi * d_model + h * head_dim;
            float max_score = -INFINITY;
            int valid_keys = 0;
            memset(probs, 0, (size_t)seq_kv * sizeof(float));
            for (int ki = 0; ki < seq_kv; ki++) {
                if (!attention_mask_allowed(mask, mask_mode, batch_index, qi, ki,
                                            seq_q, seq_kv, causal)) continue;
                const float* kv = k->data + kv_base + (long)ki * d_model + h * head_dim;
                double dot = 0.0;
                for (int d = 0; d < head_dim; d++) dot += (double)qv[d] * kv[d];
                probs[ki] = (float)dot * scale;
                if (probs[ki] > max_score) max_score = probs[ki];
                valid_keys++;
            }
            if (!valid_keys) continue;
            double sum = 0.0;
            for (int ki = 0; ki < seq_kv; ki++) {
                if (!attention_mask_allowed(mask, mask_mode, batch_index, qi, ki,
                                            seq_q, seq_kv, causal)) continue;
                probs[ki] = expf(probs[ki] - max_score); sum += probs[ki];
            }
            for (int ki = 0; ki < seq_kv; ki++) if (probs[ki] != 0.0f) probs[ki] /= (float)sum;
            double weighted_dprob = 0.0;
            for (int ki = 0; ki < seq_kv; ki++) {
                if (!attention_mask_allowed(mask, mask_mode, batch_index, qi, ki,
                                            seq_q, seq_kv, causal)) continue;
                const float* vv = v->data + kv_base + (long)ki * d_model + h * head_dim;
                double dp = 0.0;
                for (int d = 0; d < head_dim; d++) dp += (double)dy[d] * vv[d];
                float multiplier = attention_dropout_multiplier(&dropout, batch_index,
                    h, qi, ki, heads, seq_q, seq_kv);
                dprob[ki] = (float)dp * multiplier;
                weighted_dprob += (double)probs[ki] * dprob[ki];
                float* gvv = gv + kv_base + (long)ki * d_model + h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    gvv[d] += probs[ki] * multiplier * dy[d];
            }
            float* gqv = gq + q_base + (long)qi * d_model + h * head_dim;
            for (int ki = 0; ki < seq_kv; ki++) {
                if (!attention_mask_allowed(mask, mask_mode, batch_index, qi, ki,
                                            seq_q, seq_kv, causal)) continue;
                const float* kv = k->data + kv_base + (long)ki * d_model + h * head_dim;
                float* gkv = gk + kv_base + (long)ki * d_model + h * head_dim;
                float ds = probs[ki] * (dprob[ki] - (float)weighted_dprob);
                for (int d = 0; d < head_dim; d++) {
                    gqv[d] += ds * scale * kv[d];
                    gkv[d] += ds * scale * qv[d];
                }
            }
        }
    }
    }
    free(probs);
    free(dprob);
    return 0;
}

static int backward_conv2d(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    T* w = nin(n, VX_PORT_WEIGHT);
    T* b = nin(n, VX_PORT_BIAS);
    if (!in || !w || !out || !go || in->dtype != T_F32 ||
        (w->dtype != T_F32 && w->dtype != T_F16) ||
        out->dtype != T_F32 || in->ndim != 4 || out->ndim != 4 || w->ndim != 4) return -1;
    int n_batch = in->shape[0], in_h = in->shape[1], in_w = in->shape[2], in_c = in->shape[3];
    int out_h = out->shape[1], out_w = out->shape[2], out_c = out->shape[3];
    if (out->shape[0] != n_batch || n_batch <= 0 || in_h <= 0 || in_w <= 0 || in_c <= 0 ||
        out_h <= 0 || out_w <= 0 || out_c <= 0) return -1;
    int stride[2], padding[2], dilation[2];
    if (training_node_pair(n, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE,
                           1, INT_MIN, 0, stride) != 0 ||
        training_node_pair(n, VX_NODE_PARAM_PADDING, VX_NODE_ARRAY_PADDING,
                           0, INT_MIN, 0, padding) != 0 ||
        training_node_pair(n, VX_NODE_PARAM_DILATION, VX_NODE_ARRAY_DILATION,
                           1, INT_MIN, 0, dilation) != 0) return -1;
    int sy = stride[0], sx = stride[1];
    int py = padding[0], px = padding[1];
    int dy = dilation[0], dx = dilation[1];
    int pads[4] = {py, px, py, px};
    int pads_count = 0;
    const int32_t* cached_pads = vx_node_param_array(
        n, VX_NODE_PARAM_PADS, VX_NODE_ARRAY_PADS, &pads_count);
    if (cached_pads) {
        if (pads_count != 4) return -1;
        for (int index = 0; index < 4; index++) pads[index] = cached_pads[index];
    } else if (vx_node_param(n, VX_NODE_PARAM_PADS)->kind !=
               VX_NODE_PARAM_ABSENT) {
        return -1;
    }
    int groups = vx_node_param_i32(n, VX_NODE_PARAM_GROUPS, 1);
    if (sy <= 0 || sx <= 0 || dy <= 0 || dx <= 0 || groups <= 0 ||
        in_c % groups || out_c % groups) return -1;
    VxNodeParamSymbol layout = vx_node_param_symbol(
        n, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_HWIO);
    int depthwise = groups == in_c;
    int kh, kw, in_per_group, mult = 1;
    if (depthwise && (layout == VX_NODE_SYMBOL_HWCM ||
                      layout == VX_NODE_SYMBOL_1HWO ||
                      layout == VX_NODE_SYMBOL_1HWM)) {
        if (out_c % in_c) return -1;
        mult = out_c / in_c;
        if (layout == VX_NODE_SYMBOL_HWCM) {
            kh = w->shape[0]; kw = w->shape[1];
            if (w->shape[2] != in_c || w->shape[3] != mult) return -1;
        } else {
            if (w->shape[0] != 1 || w->shape[3] != out_c) return -1;
            kh = w->shape[1]; kw = w->shape[2];
        }
        in_per_group = 1;
    } else if (layout == VX_NODE_SYMBOL_HWIO) {
        kh = w->shape[0]; kw = w->shape[1]; in_per_group = w->shape[2];
        if (w->shape[3] != out_c || in_per_group != in_c / groups) return -1;
        depthwise = 0;
    } else if (layout == VX_NODE_SYMBOL_OHWI) {
        if (w->shape[0] != out_c) return -1;
        kh = w->shape[1]; kw = w->shape[2]; in_per_group = w->shape[3];
        if (in_per_group != in_c / groups) return -1;
        depthwise = 0;
    } else {
        return -1;
    }
    if (kh <= 0 || kw <= 0 || (b && ((b->dtype != T_F32 && b->dtype != T_F16) || b->numel != out_c))) return -1;
    float* gi = grad_for(grads, in);
    float* gw = w->dtype == T_F32 ? grad_for(grads, w) : NULL;
    float* gb = b && b->dtype == T_F32 ? grad_for(grads, b) : NULL;
    if (!gi || (w->dtype == T_F32 && !gw) || (b && b->dtype == T_F32 && !gb)) return -1;
    int relu = n->fuse_relu6 ? 2 :
        vx_node_param_i32(n, VX_NODE_PARAM_RELU, 0);
    int group_out = out_c / groups;
    for (int bn = 0; bn < n_batch; bn++) {
        for (int oy = 0; oy < out_h; oy++) {
            int iy0 = oy * sy - pads[0];
            for (int ox = 0; ox < out_w; ox++) {
                int ix0 = ox * sx - pads[1];
                for (int oc = 0; oc < out_c; oc++) {
                    long oi = (((long)bn * out_h + oy) * out_w + ox) * out_c + oc;
                    float g = go[oi];
                    if ((relu == 1 && out->data[oi] <= 0.0f) ||
                        (relu == 2 && (out->data[oi] <= 0.0f || out->data[oi] >= 6.0f))) g = 0.0f;
                    if (gb) gb[oc] += g;
                    if (depthwise) {
                        int ic = oc / mult;
                        int m = oc % mult;
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = iy0 + ky * dy;
                            if ((unsigned)iy >= (unsigned)in_h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = ix0 + kx * dx;
                                if ((unsigned)ix >= (unsigned)in_w) continue;
                                long ii = (((long)bn * in_h + iy) * in_w + ix) * in_c + ic;
                                long wi = layout == VX_NODE_SYMBOL_HWCM
                                    ? (((long)ky * kw + kx) * in_c + ic) * mult + m
                                    : (((long)ky * kw + kx) * out_c + oc);
                                gi[ii] += g * tensor_scalar_f32(w, wi);
                                if (gw) gw[wi] += g * in->data[ii];
                            }
                        }
                    } else {
                        int group = oc / group_out;
                        int in_start = group * in_per_group;
                        for (int icl = 0; icl < in_per_group; icl++) {
                            int ic = in_start + icl;
                            for (int ky = 0; ky < kh; ky++) {
                                int iy = iy0 + ky * dy;
                                if ((unsigned)iy >= (unsigned)in_h) continue;
                                for (int kx = 0; kx < kw; kx++) {
                                    int ix = ix0 + kx * dx;
                                    if ((unsigned)ix >= (unsigned)in_w) continue;
                                    long ii = (((long)bn * in_h + iy) * in_w + ix) * in_c + ic;
                                    long wi = layout == VX_NODE_SYMBOL_HWIO
                                        ? (((long)ky * kw + kx) * in_per_group + icl) * out_c + oc
                                        : (((long)oc * kh + ky) * kw + kx) * in_per_group + icl;
                                    gi[ii] += g * tensor_scalar_f32(w, wi);
                                    if (gw) gw[wi] += g * in->data[ii];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

static int backward_batchnorm2d(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    T* w = nin(n, VX_PORT_WEIGHT);
    T* b = nin(n, VX_PORT_BIAS);
    T* rm = nin(n, VX_PORT_RUNNING_MEAN);
    T* rv = nin(n, VX_PORT_RUNNING_VAR);
    if (!in || !w || !b || !rm || !rv || !out || !go || in->dtype != T_F32 ||
        w->dtype != T_F32 || b->dtype != T_F32 || rm->dtype != T_F32 || rv->dtype != T_F32 ||
        out->dtype != T_F32 || in->ndim != 4 || in->numel != out->numel) return -1;
    int c = in->shape[3];
    if (c <= 0 || w->numel < c || b->numel < c || rm->numel < c || rv->numel < c) return -1;
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, w);
    float* gb = grad_for(grads, b);
    if (!gi || !gw || !gb) return -1;
    float eps = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-5f);
    for (long i = 0; i < in->numel; i++) {
        int ch = (int)(i % c);
        float inv = 1.0f / sqrtf(rv->data[ch] + eps);
        gi[i] += go[i] * w->data[ch] * inv;
        gw[ch] += go[i] * (in->data[i] - rm->data[ch]) * inv;
        gb[ch] += go[i];
    }
    return 0;
}

static int backward_global_average_pool(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 || in->ndim != 4) return -1;
    int batch = in->shape[0], h = in->shape[1], w = in->shape[2], c = in->shape[3];
    int spatial = h * w;
    if (batch <= 0 || spatial <= 0 || c <= 0 || out->numel != (long)batch * c) return -1;
    float* gi = grad_for(grads, in);
    if (!gi) return -1;
    for (int bn = 0; bn < batch; bn++) {
        for (int pos = 0; pos < spatial; pos++) {
            for (int ch = 0; ch < c; ch++) {
                gi[((long)bn * spatial + pos) * c + ch] += go[(long)bn * c + ch] / spatial;
            }
        }
    }
    return 0;
}

typedef struct {
    int batch;
    int input_height;
    int input_width;
    int channels;
    int output_height;
    int output_width;
    int kernel_y;
    int kernel_x;
    int stride_y;
    int stride_x;
    int pads[4];
} TrainingMaxPool2DGeometry;

static int training_maxpool_pads(const Node* node, int pads[4]) {
    int padding[2];
    int pad_count = 0;
    const VxCachedNodeParam* padding_value = vx_node_param(
        node, VX_NODE_PARAM_PADDING);
    const VxCachedNodeParam* pads_value = vx_node_param(
        node, VX_NODE_PARAM_PADS);
    const int32_t* cached_pads;
    if (!padding_value || !pads_value || !pads) return -1;
    if (training_node_pair(node, VX_NODE_PARAM_PADDING,
                           VX_NODE_ARRAY_PADDING, 0, 0, 0, padding) != 0)
        return -1;
    pads[0] = padding[0];
    pads[1] = padding[1];
    pads[2] = padding[0];
    pads[3] = padding[1];
    if (pads_value->kind == VX_NODE_PARAM_ABSENT) return 0;
    cached_pads = vx_node_param_array(
        node, VX_NODE_PARAM_PADS, VX_NODE_ARRAY_PADS, &pad_count);
    if (!cached_pads || pad_count != 4) return -1;
    for (int index = 0; index < 4; index++) {
        if (cached_pads[index] < 0) return -1;
        pads[index] = cached_pads[index];
    }
    if (padding_value->kind != VX_NODE_PARAM_ABSENT &&
        (pads[0] != padding[0] || pads[1] != padding[1] ||
         pads[2] != padding[0] || pads[3] != padding[1]))
        return -1;
    return 0;
}

static int training_maxpool_numel(const int shape[4], int64_t* output) {
    int64_t value = 1;
    if (!shape || !output) return -1;
    for (int axis = 0; axis < 4; axis++) {
        if (shape[axis] <= 0 || value > INT64_MAX / shape[axis]) return -1;
        value *= shape[axis];
    }
    *output = value;
    return 0;
}

/* CPU and GPU training share this descriptor-only check.  In particular,
 * bottom/right padding participates in the exact output shape even though the
 * kernels need only the top/left origin offsets.  No destination-gradient
 * allocation or write is allowed until this complete geometry is accepted. */
static int training_maxpool2d_geometry(Node* n, T* in, T* out,
                                       TrainingMaxPool2DGeometry* geometry) {
    int kernel[2], stride[2], dilation[2], pads[4];
    int64_t input_numel, output_numel;
    int64_t padded_height, padded_width;
    int64_t output_height, output_width;
    const VxCachedNodeParam* ceil_mode;
    const VxCachedNodeParam* data_layout;
    if (!n || !in || !out || !geometry || in->dtype != T_F32 ||
        out->dtype != T_F32 || in->ndim != 4 || out->ndim != 4)
        return -1;
    for (int axis = 0; axis < 4; axis++)
        if (in->shape[axis] <= 0 || out->shape[axis] <= 0) return -1;
    if (training_node_pair(n, VX_NODE_PARAM_KERNEL, VX_NODE_ARRAY_KERNEL,
                           1, 1, 1, kernel) != 0 ||
        training_node_pair(n, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE,
                           1, 1, 0, stride) != 0 ||
        training_node_pair(n, VX_NODE_PARAM_DILATION,
                           VX_NODE_ARRAY_DILATION, 1, 1, 0, dilation) != 0 ||
        training_maxpool_pads(n, pads) != 0 ||
        dilation[0] != 1 || dilation[1] != 1)
        return -1;
    ceil_mode = vx_node_param(n, VX_NODE_PARAM_CEIL_MODE);
    if (!ceil_mode ||
        (ceil_mode->kind != VX_NODE_PARAM_ABSENT &&
         !((ceil_mode->kind == VX_NODE_PARAM_BOOL ||
            ceil_mode->kind == VX_NODE_PARAM_I32) &&
           ceil_mode->value.i32 == 0)))
        return -1;
    data_layout = vx_node_param(n, VX_NODE_PARAM_DATA_LAYOUT);
    if (!data_layout ||
        (data_layout->kind != VX_NODE_PARAM_ABSENT &&
         !(data_layout->kind == VX_NODE_PARAM_SYMBOL &&
           data_layout->symbol == VX_NODE_SYMBOL_NHWC)))
        return -1;

    if (training_maxpool_numel(in->shape, &input_numel) != 0 ||
        training_maxpool_numel(out->shape, &output_numel) != 0)
        return -1;
    padded_height = (int64_t)in->shape[1] + pads[0] + pads[2];
    padded_width = (int64_t)in->shape[2] + pads[1] + pads[3];
    if (input_numel != in->numel || padded_height < kernel[0] ||
        padded_width < kernel[1])
        return -1;
    output_height = (padded_height - kernel[0]) / stride[0] + 1;
    output_width = (padded_width - kernel[1]) / stride[1] + 1;
    if (output_height <= 0 || output_height > INT_MAX || output_width <= 0 ||
        output_width > INT_MAX || out->shape[0] != in->shape[0] ||
        out->shape[1] != output_height || out->shape[2] != output_width ||
        out->shape[3] != in->shape[3])
        return -1;
    if (output_numel != out->numel) return -1;

    geometry->batch = in->shape[0];
    geometry->input_height = in->shape[1];
    geometry->input_width = in->shape[2];
    geometry->channels = in->shape[3];
    geometry->output_height = (int)output_height;
    geometry->output_width = (int)output_width;
    geometry->kernel_y = kernel[0];
    geometry->kernel_x = kernel[1];
    geometry->stride_y = stride[0];
    geometry->stride_x = stride[1];
    memcpy(geometry->pads, pads, sizeof(geometry->pads));
    return 0;
}

static int backward_maxpool2d(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    TrainingMaxPool2DGeometry geometry;
    if (!go || training_maxpool2d_geometry(n, in, out, &geometry) != 0)
        return -1;
    float* gi = grad_for(grads, in);
    if (!gi) return -1;
    for (int bn = 0; bn < geometry.batch; bn++) {
        for (int oy = 0; oy < geometry.output_height; oy++) {
            int iy0 = oy * geometry.stride_y - geometry.pads[0];
            for (int ox = 0; ox < geometry.output_width; ox++) {
                int ix0 = ox * geometry.stride_x - geometry.pads[1];
                for (int ch = 0; ch < geometry.channels; ch++) {
                    long best_i = -1;
                    float best = -INFINITY;
                    for (int yy = 0; yy < geometry.kernel_y; yy++) {
                        int iy = iy0 + yy;
                        if ((unsigned)iy >=
                            (unsigned)geometry.input_height) continue;
                        for (int xx = 0; xx < geometry.kernel_x; xx++) {
                            int ix = ix0 + xx;
                            if ((unsigned)ix >=
                                (unsigned)geometry.input_width) continue;
                            long ii = (((long)bn * geometry.input_height + iy) *
                                geometry.input_width + ix) *
                                geometry.channels + ch;
                            if (in->data[ii] > best) { best = in->data[ii]; best_i = ii; }
                        }
                    }
                    if (best_i >= 0) {
                        long oi = (((long)bn * geometry.output_height + oy) *
                            geometry.output_width + ox) *
                            geometry.channels + ch;
                        gi[best_i] += go[oi];
                    }
                }
            }
        }
    }
    return 0;
}

static int backward_resize2d(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    int nearest = 1;
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim != 4 || out->ndim != 4 || in->shape[0] != out->shape[0] ||
        in->shape[3] != out->shape[3] ||
        ((n->operator_kind == VX_OP_RESIZE) && resize_f32_parameters(n, &nearest))) return -1;
    for (int i = 0; i < 4; i++) if (in->shape[i] <= 0 || out->shape[i] <= 0) return -1;
    float* gi = grad_for(grads, in);
    if (!gi) return -1;
    volvoxai_training_resize2d_backward_f32(go, gi, in->shape[0], in->shape[1],
        in->shape[2], in->shape[3], out->shape[1], out->shape[2], nearest);
    return 0;
}

static int backward_concat(float** grads, Node* n, T* out, const float* go) {
    if (!n || !out || !go || out->dtype != T_F32 || out->ndim <= 0) return -1;
    int sigmoid = vx_node_param_i32(n, VX_NODE_PARAM_SIGMOID, 0);
    int axis = vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0);
    if (axis < 0) axis += out->ndim;
    if (axis < 0 || axis >= out->ndim) return -1;
    long inner = 1;
    for (int d = axis + 1; d < out->ndim; d++) inner *= out->shape[d];
    long outer = out->numel / ((long)out->shape[axis] * inner);
    int axis_offset = 0;
    for (int ri = 0; ri < n->nin; ri++) {
        T* src = t_find(n->ins[ri].name);
        if (!src || src->dtype != T_F32 || src->ndim != out->ndim) return -1;
        for (int d = 0; d < out->ndim; d++) {
            if (d != axis && src->shape[d] != out->shape[d]) return -1;
        }
        float* gs = grad_for(grads, src);
        if (!gs) return -1;
        int axis_len = src->shape[axis];
        long block = (long)axis_len * inner;
        for (long oi = 0; oi < outer; oi++) {
            long offset = (oi * out->shape[axis] + axis_offset) * inner;
            const float* from = go + offset;
            if (!sigmoid) grad_add_full(gs + oi * block, from, block, 1.0f);
            else for (long i = 0; i < block; ++i) {
                float value = out->data[offset + i];
                gs[oi * block + i] += from[i] * value * (1.0f - value);
            }
        }
        axis_offset += axis_len;
    }
    return axis_offset == out->shape[axis] ? 0 : -1;
}

static int backward_split(float** grads, Node* n) {
    T* in = node_input_default(n);
    if (!n || !in || in->dtype != T_F32 || in->ndim <= 0 || n->nout <= 0) return -1;
    int axis = vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0);
    if (axis < 0) axis += in->ndim;
    if (axis < 0 || axis >= in->ndim || in->shape[axis] <= 0) return -1;
    long inner = 1;
    for (int d = axis + 1; d < in->ndim; d++) {
        if (in->shape[d] <= 0) return -1;
        inner *= in->shape[d];
    }
    long outer = in->numel / ((long)in->shape[axis] * inner);
    if (inner <= 0 || outer <= 0 || in->numel != outer * in->shape[axis] * inner) return -1;

    float* gi = grad_for(grads, in);
    if (!gi) return -1;
    int axis_offset = 0;
    for (int oi = 0; oi < n->nout; oi++) {
        T* out = t_find(n->outs[oi].name);
        if (!out || out->dtype != T_F32 || out->ndim != in->ndim || out->shape[axis] <= 0) return -1;
        for (int d = 0; d < in->ndim; d++) {
            if (d != axis && out->shape[d] != in->shape[d]) return -1;
        }
        long block = (long)out->shape[axis] * inner;
        if (out->numel != outer * block || axis_offset > in->shape[axis] - out->shape[axis]) return -1;
        const float* go = grad_existing(grads, out);
        if (go) {
            for (long row = 0; row < outer; row++) {
                long input_offset = (row * in->shape[axis] + axis_offset) * inner;
                grad_add_full(gi + input_offset, go + row * block, block, 1.0f);
            }
        }
        axis_offset += out->shape[axis];
    }
    return axis_offset == in->shape[axis] ? 0 : -1;
}

static int backward_moe_linear(float** grads, Node* n, T* out, const float* go) {
    T* in = node_input_default(n);
    T* ew = nin(n, VX_PORT_EXPERT_WEIGHT);
    T* eb = nin(n, VX_PORT_EXPERT_BIAS);
    T* indices = nin(n, VX_PORT_ROUTE_INDICES);
    T* weights = nin(n, VX_PORT_ROUTE_WEIGHTS);
    if (!indices) indices = nin(n, VX_PORT_INDICES);
    if (!weights) weights = nin(n, VX_PORT_WEIGHTS);
    if (!in || !ew || !indices || !weights || !out || !go || in->dtype != T_F32 ||
        ew->dtype != T_F32 || weights->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim <= 0 || ew->ndim != 3 || out->ndim <= 0) return -1;
    int num_experts = ew->shape[0], d_in = ew->shape[1], d_out = ew->shape[2];
    if (num_experts <= 0 || d_in <= 0 || d_out <= 0 || in->shape[in->ndim - 1] != d_in ||
        out->shape[out->ndim - 1] != d_out || (eb && (eb->dtype != T_F32 ||
        eb->numel != (long)num_experts * d_out))) return -1;
    long tokens = in->numel / d_in;
    if (tokens <= 0 || out->numel != tokens * d_out || indices->numel != weights->numel ||
        indices->numel % tokens) return -1;
    int top_k = (int)(indices->numel / tokens);
    float* gi = grad_for(grads, in);
    float* gew = grad_for(grads, ew);
    float* geb = eb ? grad_for(grads, eb) : NULL;
    float* grw = grad_for(grads, weights);
    if (!gi || !gew || !grw || (eb && !geb)) return -1;
    for (long token = 0; token < tokens; token++) {
        const float* x = in->data + token * d_in;
        const float* dy = go + token * d_out;
        float* dx = gi + token * d_in;
        for (int slot = 0; slot < top_k; slot++) {
            long ri = token * top_k + slot;
            int expert;
            if (route_index_read(indices, ri, &expert) != 0 || expert < 0 || expert >= num_experts) return -1;
            float rw = weights->data[ri];
            const float* we = ew->data + (long)expert * d_in * d_out;
            float* gwe = gew + (long)expert * d_in * d_out;
            const float* be = eb ? eb->data + (long)expert * d_out : NULL;
            float* gbe = geb ? geb + (long)expert * d_out : NULL;
            double route_grad = 0.0;
            for (int j = 0; j < d_out; j++) {
                double value = be ? be[j] : 0.0;
                for (int d = 0; d < d_in; d++) value += (double)x[d] * we[(long)d * d_out + j];
                route_grad += (double)dy[j] * value;
                if (gbe) gbe[j] += rw * dy[j];
                for (int d = 0; d < d_in; d++) {
                    dx[d] += rw * dy[j] * we[(long)d * d_out + j];
                    gwe[(long)d * d_out + j] += rw * x[d] * dy[j];
                }
            }
            grw[ri] += (float)route_grad;
        }
    }
    return 0;
}

static int backward_moe_router(float** grads, Node* n, T* route_weights, const float* go) {
    T* in = node_input_default(n);
    T* w = nin(n, VX_PORT_WEIGHT);
    T* b = nin(n, VX_PORT_BIAS);
    T* indices = nout(n, VX_PORT_INDICES);
    if (!indices) indices = nout(n, VX_PORT_ROUTE_INDICES);
    if (!in || !w || !indices || !route_weights || !go || in->dtype != T_F32 ||
        w->dtype != T_F32 || route_weights->dtype != T_F32 || in->ndim <= 0 || w->ndim != 2) return -1;
    int d_model = in->shape[in->ndim - 1];
    int num_experts = vx_node_param_i32(
        n, VX_NODE_PARAM_NUM_EXPERTS, w->shape[1]);
    int top_k = vx_node_param_i32(n, VX_NODE_PARAM_TOP_K, 2);
    float temperature = vx_node_param_f32(
        n, VX_NODE_PARAM_TEMPERATURE, 1.0f);
    if (d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts ||
        !isfinite(temperature) || temperature <= 0.0f ||
        w->shape[0] != d_model || w->shape[1] != num_experts ||
        (b && (b->dtype != T_F32 || b->numel != num_experts))) return -1;
    long tokens = in->numel / d_model;
    if (indices->numel != tokens * top_k || route_weights->numel != tokens * top_k) return -1;
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, w);
    float* gb = b ? grad_for(grads, b) : NULL;
    float* logits = (float*)malloc((size_t)num_experts * sizeof(float));
    float* probs = (float*)malloc((size_t)num_experts * sizeof(float));
    float* dz = (float*)malloc((size_t)num_experts * sizeof(float));
    int* selected = (int*)malloc((size_t)top_k * sizeof(int));
    if (!gi || !gw || (b && !gb) || !logits || !probs || !dz || !selected) {
        free(logits); free(probs); free(dz); free(selected); return -1;
    }
    int normalize = vx_node_param_i32(n, VX_NODE_PARAM_NORMALIZE, 1);
    for (long token = 0; token < tokens; token++) {
        const float* x = in->data + token * d_model;
        float max_logit = -INFINITY;
        for (int e = 0; e < num_experts; e++) {
            double sum = b ? b->data[e] : 0.0;
            for (int d = 0; d < d_model; d++) sum += (double)x[d] * w->data[(long)d * num_experts + e];
            logits[e] = (float)sum / temperature;
            if (logits[e] > max_logit) max_logit = logits[e];
            dz[e] = 0.0f;
        }
        for (int slot = 0; slot < top_k; slot++) {
            long ri = token * top_k + slot;
            if (route_index_read(indices, ri, &selected[slot]) != 0 ||
                selected[slot] < 0 || selected[slot] >= num_experts) {
                free(logits); free(probs); free(dz); free(selected); return -1;
            }
        }
        if (normalize) {
            double dot = 0.0;
            for (int slot = 0; slot < top_k; slot++) {
                long ri = token * top_k + slot;
                dot += (double)go[ri] * route_weights->data[ri];
            }
            for (int slot = 0; slot < top_k; slot++) {
                long ri = token * top_k + slot;
                int e = selected[slot];
                dz[e] += route_weights->data[ri] * (go[ri] - (float)dot);
            }
        } else {
            double denom = 0.0;
            for (int e = 0; e < num_experts; e++) { probs[e] = expf(logits[e] - max_logit); denom += probs[e]; }
            for (int e = 0; e < num_experts; e++) probs[e] /= (float)denom;
            double dot = 0.0;
            for (int slot = 0; slot < top_k; slot++) dot += (double)go[token * top_k + slot] * probs[selected[slot]];
            for (int e = 0; e < num_experts; e++) dz[e] = -probs[e] * (float)dot;
            for (int slot = 0; slot < top_k; slot++) {
                int e = selected[slot];
                dz[e] += probs[e] * go[token * top_k + slot];
            }
        }
        for (int e = 0; e < num_experts; e++) {
            float raw_dz = dz[e] / temperature;
            if (gb) gb[e] += raw_dz;
            for (int d = 0; d < d_model; d++) {
                gi[token * d_model + d] += raw_dz * w->data[(long)d * num_experts + e];
                gw[(long)d * num_experts + e] += x[d] * raw_dz;
            }
        }
    }
    free(logits); free(probs); free(dz); free(selected);
    return 0;
}

static int seed_cross_entropy_grad(float** grads, T* logits, const int* targets, int target_count,
                                   int ignore_index, int row_index, float weight,
                                   float normalizer, float* out_loss, int* out_correct,
                                   int* out_examples, float* out_normalizer) {
    if (!logits || logits->dtype != T_F32 || logits->ndim <= 0 || !targets || target_count <= 0) return -1;
    if (!isfinite(weight) || weight < 0.0f || !isfinite(normalizer) || normalizer < 0.0f) return -1;
    materialize_tensor_f32(logits);
    int classes = logits->shape[logits->ndim - 1];
    if (classes <= 0 || logits->numel % classes != 0) return -1;
    int rows = (int)(logits->numel / classes);
    int count = target_count == rows ? rows : target_count;
    if (count <= 0 || count > rows) return -1;
    int row0 = target_count == rows ? 0 : rows - count;
    int batch = logits->ndim >= 3 ? logits->shape[0] : 1;
    int rows_per_batch = batch > 0 && rows % batch == 0 ? rows / batch : 0;
    int per_batch_target = batch > 1 && target_count == batch && rows_per_batch > 0;
    if (row_index >= 0 && ((per_batch_target && row_index >= rows_per_batch) ||
        (!per_batch_target && target_count == 1 && row_index >= rows))) return -1;
    if (target_count == 1 && row_index >= 0 && row_index < rows) row0 = row_index;
    int active_count = 0;
    for (int r = 0; r < count; r++) {
        int target = targets[r];
        if (target == ignore_index) continue;
        if (target < 0 || target >= classes) return -1;
        active_count++;
    }
    if (active_count <= 0) {
        if (out_loss) *out_loss = 0.0f;
        if (out_correct) *out_correct = 0;
        if (out_examples) *out_examples = 0;
        if (out_normalizer) *out_normalizer = normalizer;
        return 1; /* valid no-op batch */
    }
    float denominator = normalizer > 0.0f ? normalizer : (float)active_count;
    if (!isfinite(denominator) || denominator <= 0.0f) return -1;
    if (weight == 0.0f) {
        if (out_loss) *out_loss = 0.0f;
        if (out_correct) *out_correct = 0;
        if (out_examples) *out_examples = active_count;
        if (out_normalizer) *out_normalizer = denominator;
        return 0;
    }
    float* gl = grad_for(grads, logits);
    if (!gl) return -1;
    float loss = 0.0f;
    int correct = 0;
    for (int r = 0; r < count; r++) {
        int row = per_batch_target
            ? r * rows_per_batch + ((row_index >= 0 && row_index < rows_per_batch)
                                    ? row_index : rows_per_batch - 1)
            : row0 + r;
        int target = targets[r];
        if (target == ignore_index) continue;
        const float* z = logits->data + (long)row * classes;
        float* gz = gl + (long)row * classes;
        float maxv = z[0];
        int argmax = 0;
        for (int j = 1; j < classes; j++) {
            if (z[j] > maxv) { maxv = z[j]; argmax = j; }
        }
        double sum = 0.0;
        for (int j = 0; j < classes; j++) sum += exp((double)z[j] - (double)maxv);
        double logsum = log(sum);
        loss += (float)((double)maxv + logsum - (double)z[target]);
        if (argmax == target) correct++;
        for (int j = 0; j < classes; j++) {
            float p = (float)(exp((double)z[j] - (double)maxv) / sum);
            gz[j] += (p - (j == target ? 1.0f : 0.0f)) * weight / denominator;
        }
    }
    if (out_loss) *out_loss = weight * loss / denominator;
    if (out_correct) *out_correct = correct;
    if (out_examples) *out_examples = active_count;
    if (out_normalizer) *out_normalizer = denominator;
    return 0;
}
