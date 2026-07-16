// --- Embedding (row gather) ---
// Token ids are conventional int32 values. Weight is float32 [vocab, d_model]
// and output is float32 [seq_len, d_model]. Bounds are validated by the graph
// runtime before entering this hot loop.
void embedding_f32(uintptr_t tok_p, uintptr_t w_p, uintptr_t out_p, int seq_len, int d_model) {
    const float* w = (const float*)w_p;
    const int32_t* tok = (const int32_t*)tok_p;
    float* out = (float*)out_p;
    for (int i = 0; i < seq_len; i++) {
        int id = tok[i];

        const float* src = w + id * d_model;
        float* dst = out + i * d_model;
        for (int j = 0; j < d_model; j++) dst[j] = src[j];
    }
}
