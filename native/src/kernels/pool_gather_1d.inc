/* AveragePool2D and 1-D Gather. */

void averagepool2d_f32(const float* input, float* output, int b, int h, int w, int c, int kh, int kw, int sh, int sw, int ph, int pw, int out_h, int out_w) {
    for (int i_b = 0; i_b < b; i_b++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                for (int i_c = 0; i_c < c; i_c++) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int in_y = y * sh - ph + ky;
                            int in_x = x * sw - pw + kx;
                            if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
                                sum += input[((long)i_b * h * w + (long)in_y * w + in_x) * c + i_c];
                                count++;
                            }
                        }
                    }
                    output[((long)i_b * out_h * out_w + (long)y * out_w + x) * c + i_c] = sum / (float)(count > 0 ? count : 1);
                }
            }
        }
    }
}
void gather_1d_f32(const float* input, const float* indices, float* output, int d, int num_indices) {
    for (int i = 0; i < num_indices; i++) {
        int idx = (int)indices[i];
        if (idx >= 0 && idx < d) {
            output[i] = input[idx];
        } else {
            output[i] = 0.0f;
        }
    }
}
