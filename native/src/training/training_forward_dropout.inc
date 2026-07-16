/*
 * Training-only forward Dropout and attention-dropout support.
 *
 * This fragment is included exactly once by engine_runtime_f32_cpu.inc behind
 * VOLVOXAI_ENABLE_TRAINING. It intentionally shares the runtime translation
 * unit so TrainStep can switch forward execution into training mode without
 * exposing private graph state through an internal ABI.
 */

/* Set only around the forward/backward portion of TrainStep. */
static int g_native_training_mode;
static uint32_t g_native_training_counter;

typedef struct {
    uint32_t threshold;
    uint32_t seed;
    uint32_t counter;
    float scale;
} NativeDropoutParams;

typedef struct {
    uint32_t threshold;
    uint32_t seed;
    uint32_t counter;
    float scale;
    int enabled;
} NativeAttentionDropoutParams;

static uint32_t dropout_random_bits(uint32_t seed, uint32_t counter, uint32_t index) {
    uint32_t value = seed ^ index ^ counter * 0x9e3779b9u;
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

static int native_attention_dropout_params(Node* n, int node_index,
                                           NativeAttentionDropoutParams* out) {
    if (!n || !out || node_index < 0 || node_index >= g_nn) return -1;
    float probability = p_flt(n->params, "dropout",
        p_flt(n->params, "attention_dropout", 0.0f));
    if (!isfinite(probability) || probability < 0.0f || probability >= 1.0f) return -1;
    int configured_seed = p_int(n->params, "dropout_seed",
        p_int(n->params, "training_seed", p_int(n->params, "seed", 0)));
    if (configured_seed < 0) return -1;
    double threshold = floor((double)probability * 4294967296.0);
    if (threshold < 0.0 || threshold > 4294967295.0) return -1;
    out->threshold = (uint32_t)threshold;
    out->seed = (uint32_t)configured_seed ^
        ((uint32_t)(node_index + 1) * 0x9e3779b9u);
    out->counter = g_native_training_counter;
    out->scale = 1.0f / (1.0f - probability);
    out->enabled = probability > 0.0f;
    return isfinite(out->scale) ? 0 : -1;
}

static uint32_t attention_probability_index(int batch, int head, int query, int key,
                                            int heads, int queries, int keys) {
    uint32_t index = (uint32_t)batch * (uint32_t)heads + (uint32_t)head;
    index = index * (uint32_t)queries + (uint32_t)query;
    return index * (uint32_t)keys + (uint32_t)key;
}

static float attention_dropout_multiplier(const NativeAttentionDropoutParams* dropout,
                                           int batch, int head, int query, int key,
                                           int heads, int queries, int keys) {
    if (!dropout || !dropout->enabled) return 1.0f;
    uint32_t index = attention_probability_index(batch, head, query, key,
                                                 heads, queries, keys);
    return dropout_random_bits(dropout->seed, dropout->counter, index) >=
        dropout->threshold ? dropout->scale : 0.0f;
}

static int native_dropout_params(Node* n, int node_index, NativeDropoutParams* out) {
    if (!n || !out || node_index < 0) return -1;
    float probability = p_flt(n->params, "p",
        p_flt(n->params, "ratio", p_flt(n->params, "rate", 0.5f)));
    if (!isfinite(probability) || probability < 0.0f || probability >= 1.0f) return -1;
    double threshold = floor((double)probability * 4294967296.0);
    if (threshold < 0.0 || threshold > 4294967295.0) return -1;
    uint32_t configured_seed = (uint32_t)p_int(n->params, "seed", 0);
    uint32_t counter_offset = (uint32_t)p_int(n->params, "counter", 0);
    out->threshold = (uint32_t)threshold;
    out->seed = configured_seed ^ ((uint32_t)(node_index + 1) * 0x9e3779b9u);
    out->counter = g_native_training_counter + counter_offset;
    out->scale = 1.0f / (1.0f - probability);
    return isfinite(out->scale) ? 0 : -1;
}

static int dropout_forward_f32(Node* n, int node_index, const T* in, T* out,
                               long offset, long count) {
    if (!n || !in || !out || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->numel != out->numel || offset < 0 || count < 0 ||
        offset > out->numel || count > out->numel - offset) return -1;
    NativeDropoutParams params;
    if (native_dropout_params(n, node_index, &params) != 0) return -1;
    for (long local = 0; local < count; local++) {
        long index = offset + local;
        if ((uint64_t)index > UINT32_MAX) return -1;
        out->data[index] = dropout_random_bits(params.seed, params.counter, (uint32_t)index) >=
            params.threshold ? in->data[index] * params.scale : 0.0f;
    }
    return 0;
}

static int sdpa_attention_dropout_forward(const T* qkv, const T* mask, T* out,
                                          int batch, int seq, int d_model,
                                          int heads, int head_dim, float scale,
                                          int causal, int mask_mode,
                                          const NativeAttentionDropoutParams* dropout) {
    float* probabilities = (float*)malloc((size_t)seq * sizeof(float));
    if (!qkv || !out || !dropout || !probabilities) {
        free(probabilities);
        return -1;
    }
    for (int batch_index = 0; batch_index < batch; batch_index++) {
        long qkv_base = (long)batch_index * seq * 3 * d_model;
        long output_base = (long)batch_index * seq * d_model;
        for (int head = 0; head < heads; head++) {
            for (int query = 0; query < seq; query++) {
                const float* q = qkv->data + qkv_base +
                    (long)query * 3 * d_model + head * head_dim;
                float maximum = -INFINITY;
                int valid_keys = 0;
                memset(probabilities, 0, (size_t)seq * sizeof(float));
                for (int key = 0; key < seq; key++) {
                    if (!attention_mask_allowed(mask, mask_mode, batch_index, query,
                                                key, seq, seq, causal)) continue;
                    const float* k = qkv->data + qkv_base +
                        (long)key * 3 * d_model + d_model + head * head_dim;
                    double score = 0.0;
                    for (int dimension = 0; dimension < head_dim; dimension++)
                        score += (double)q[dimension] * k[dimension];
                    probabilities[key] = (float)score * scale;
                    if (probabilities[key] > maximum) maximum = probabilities[key];
                    valid_keys++;
                }
                float* output = out->data + output_base +
                    (long)query * d_model + head * head_dim;
                if (!valid_keys) {
                    memset(output, 0, (size_t)head_dim * sizeof(float));
                    continue;
                }
                double denominator = 0.0;
                for (int key = 0; key < seq; key++) {
                    if (!attention_mask_allowed(mask, mask_mode, batch_index, query,
                                                key, seq, seq, causal)) continue;
                    probabilities[key] = expf(probabilities[key] - maximum);
                    denominator += probabilities[key];
                }
                for (int dimension = 0; dimension < head_dim; dimension++) {
                    double value = 0.0;
                    for (int key = 0; key < seq; key++) {
                        if (!attention_mask_allowed(mask, mask_mode, batch_index, query,
                                                    key, seq, seq, causal)) continue;
                        const float* v = qkv->data + qkv_base +
                            (long)key * 3 * d_model + 2 * d_model + head * head_dim;
                        float multiplier = attention_dropout_multiplier(dropout, batch_index,
                            head, query, key, heads, seq, seq);
                        value += ((double)probabilities[key] / denominator) *
                            multiplier * v[dimension];
                    }
                    output[dimension] = (float)value;
                }
            }
        }
    }
    free(probabilities);
    return 0;
}

static int cross_sdpa_attention_dropout_forward(const T* q, const T* k, const T* v,
                                                const T* mask, T* out, int batch,
                                                int seq_q, int seq_kv, int d_model,
                                                int heads, int head_dim, float scale,
                                                int causal, int mask_mode,
                                                const NativeAttentionDropoutParams* dropout) {
    float* probabilities = (float*)malloc((size_t)seq_kv * sizeof(float));
    if (!q || !k || !v || !out || !dropout || !probabilities) {
        free(probabilities);
        return -1;
    }
    for (int batch_index = 0; batch_index < batch; batch_index++) {
        long q_base = (long)batch_index * seq_q * d_model;
        long kv_base = (long)batch_index * seq_kv * d_model;
        for (int head = 0; head < heads; head++) {
            for (int query = 0; query < seq_q; query++) {
                const float* q_value = q->data + q_base +
                    (long)query * d_model + head * head_dim;
                float maximum = -INFINITY;
                int valid_keys = 0;
                memset(probabilities, 0, (size_t)seq_kv * sizeof(float));
                for (int key = 0; key < seq_kv; key++) {
                    if (!attention_mask_allowed(mask, mask_mode, batch_index, query,
                                                key, seq_q, seq_kv, causal)) continue;
                    const float* k_value = k->data + kv_base +
                        (long)key * d_model + head * head_dim;
                    double score = 0.0;
                    for (int dimension = 0; dimension < head_dim; dimension++)
                        score += (double)q_value[dimension] * k_value[dimension];
                    probabilities[key] = (float)score * scale;
                    if (probabilities[key] > maximum) maximum = probabilities[key];
                    valid_keys++;
                }
                float* output = out->data + q_base +
                    (long)query * d_model + head * head_dim;
                if (!valid_keys) {
                    memset(output, 0, (size_t)head_dim * sizeof(float));
                    continue;
                }
                double denominator = 0.0;
                for (int key = 0; key < seq_kv; key++) {
                    if (!attention_mask_allowed(mask, mask_mode, batch_index, query,
                                                key, seq_q, seq_kv, causal)) continue;
                    probabilities[key] = expf(probabilities[key] - maximum);
                    denominator += probabilities[key];
                }
                for (int dimension = 0; dimension < head_dim; dimension++) {
                    double value = 0.0;
                    for (int key = 0; key < seq_kv; key++) {
                        if (!attention_mask_allowed(mask, mask_mode, batch_index, query,
                                                    key, seq_q, seq_kv, causal)) continue;
                        const float* v_value = v->data + kv_base +
                            (long)key * d_model + head * head_dim;
                        float multiplier = attention_dropout_multiplier(dropout, batch_index,
                            head, query, key, heads, seq_q, seq_kv);
                        value += ((double)probabilities[key] / denominator) *
                            multiplier * v_value[dimension];
                    }
                    output[dimension] = (float)value;
                }
            }
        }
    }
    free(probabilities);
    return 0;
}
