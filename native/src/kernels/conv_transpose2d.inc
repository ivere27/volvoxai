/* ConvTranspose2D F32 reference kernel. */

void conv_transpose2d_f32(const float* input, const float* weight, const float* bias, float* output, int b, int in_h, int in_w, int in_c, int out_h, int out_w, int out_c, int kh, int kw, int sh, int sw, int ph, int pw) {
    if (sh <= 0 || sw <= 0) return;
    for (int i_b = 0; i_b < b; i_b++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float* out_row = output +
                    ((long)i_b * out_h * out_w + (long)oy * out_w + ox) * out_c;
                for (int oc = 0; oc < out_c; oc++)
                    out_row[oc] = bias ? bias[oc] : 0.0f;
                for (int ky = 0; ky < kh; ky++) {
                    int shifted_y = oy + ph - ky;
                    int iy = shifted_y / sh;
                    if (shifted_y < 0 || shifted_y % sh || iy >= in_h) continue;
                    for (int kx = 0; kx < kw; kx++) {
                        int shifted_x = ox + pw - kx;
                        int ix = shifted_x / sw;
                        if (shifted_x < 0 || shifted_x % sw || ix >= in_w) continue;
                        const float* in_row = input +
                            ((long)i_b * in_h * in_w + (long)iy * in_w + ix) * in_c;
                        const float* w_tap = weight +
                            ((long)ky * kw + kx) * in_c * out_c;
                        for (int ic = 0; ic < in_c; ic++) {
                            float value = in_row[ic];
                            const float* w_row = w_tap + (long)ic * out_c;
                            for (int oc = 0; oc < out_c; oc++)
                                out_row[oc] += value * w_row[oc];
                        }
                    }
                }
            }
        }
    }
}
