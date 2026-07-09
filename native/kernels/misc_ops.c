// --- Missing Final Primitives (Batch 3) ---
void where_f32(const float* cond, const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = cond[i] != 0.0f ? a[i] : b[i];
}

void dequantize_linear_f32(const float* input, const float* scale, const float* zero_point, float* output, int n) {
    float s = scale[0]; float zp = zero_point ? zero_point[0] : 0.0f;
    for (int i = 0; i < n; i++) output[i] = (input[i] - zp) * s;
}

void quantize_linear_f32(const float* input, const float* scale, const float* zero_point, float* output, int n) {
    float s = scale[0]; float zp = zero_point ? zero_point[0] : 0.0f;
    for (int i = 0; i < n; i++) {
        float val = input[i] / s + zp;
        int quantized = (int)(val + (val >= 0.0f ? 0.5f : -0.5f));
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        output[i] = (float)quantized;
    }
}

void conv_transpose2d_f32(const float* input, const float* weight, const float* bias, float* output, int b, int in_h, int in_w, int in_c, int out_h, int out_w, int out_c, int kh, int kw, int sh, int sw, int ph, int pw) {
    for (int i = 0; i < b * out_c * out_h * out_w; i++) output[i] = bias ? bias[i % out_c] : 0.0f;
    
    for (int i_b = 0; i_b < b; i_b++) {
        for (int i_ic = 0; i_ic < in_c; i_ic++) {
            for (int iy = 0; iy < in_h; iy++) {
                for (int ix = 0; ix < in_w; ix++) {
                    float in_val = input[((long)i_b * in_h * in_w + (long)iy * in_w + ix) * in_c + i_ic];
                    for (int oc = 0; oc < out_c; oc++) {
                        for (int ky = 0; ky < kh; ky++) {
                            for (int kx = 0; kx < kw; kx++) {
                                int oy = iy * sh - ph + ky;
                                int ox = ix * sw - pw + kx;
                                if (oy >= 0 && oy < out_h && ox >= 0 && ox < out_w) {
                                    float w_val = weight[i_ic * (out_c * kh * kw) + oc * (kh * kw) + ky * kw + kx];
                                    output[((long)i_b * out_h * out_w + (long)oy * out_w + ox) * out_c + oc] += in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
