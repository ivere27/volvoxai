// --- 1. INT8 MatMul ---
WASM_EXPORT("linear_f32")
void linear_f32(const float* input, const float* weight, const float* bias, float* output, int seq_len, int d_in, int d_out) {
    for (int row = 0; row < seq_len; row++) {
        const float* in_row = input + row * d_in;
        float* out_row = output + row * d_out;
        for (int col = 0; col < d_out; col++) {
            float sum = bias ? bias[col] : 0.0f;
            const float* w_col = weight + col * d_in;
            for (int k = 0; k < d_in; k++) {
                sum += in_row[k] * w_col[k];
            }
            out_row[col] = sum;
        }
    }
}
void matmul_int8_f32(const float* input, const uint32_t* weight_packed, const float* scales, const float* bias, float* output, int seq_len, int d_in, int d_out) {
    int d_in_4 = d_in / 4;
    for (int row = 0; row < seq_len; row++) {
        const float* in_row = input + row * d_in;
        float* out_row = output + row * d_out;
        for (int col = 0; col < d_out; col++) {
            const uint32_t* w_col = weight_packed + col * d_in_4;
            float sum = 0.0f;
            for (int i = 0; i < d_in_4; i++) {
                uint32_t w_pack = w_col[i];
                sum += in_row[i*4+0] * (float)((int8_t)((w_pack >> 0) & 0xFF));
                sum += in_row[i*4+1] * (float)((int8_t)((w_pack >> 8) & 0xFF));
                sum += in_row[i*4+2] * (float)((int8_t)((w_pack >> 16) & 0xFF));
                sum += in_row[i*4+3] * (float)((int8_t)((w_pack >> 24) & 0xFF));
            }
            out_row[col] = (sum * scales[col]) + bias[col];
        }
    }
}
