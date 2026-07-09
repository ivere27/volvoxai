// --- Embedding (row gather) ---
// Token ids arrive in the heap as float32 (inputs are written through a
// Float32Array view), so read them as float and truncate. Weight is float32
// [vocab, d_model]; output [seq_len, d_model]. Mirrors CPUEngine._cpuEmbedding.
void embedding_f32(uintptr_t tok_p, uintptr_t w_p, uintptr_t out_p, int seq_len, int d_model) {
    const float* w = (const float*)w_p;
    const float* tok = (const float*)tok_p;
    float* out = (float*)out_p;
    for (int i = 0; i < seq_len; i++) {
        int id = (int)tok[i];

        const float* src = w + id * d_model;
        float* dst = out + i * d_model;
        for (int j = 0; j < d_model; j++) dst[j] = src[j];
    }
}
