// --- Missing Deep Math & Shape Primitives (Batch 4) ---
void shape_f32(float* output, int n0, int n1, int n2, int n3, int ndims) {
    if (ndims > 0) output[0] = (float)n0;
    if (ndims > 1) output[1] = (float)n1;
    if (ndims > 2) output[2] = (float)n2;
    if (ndims > 3) output[3] = (float)n3;
}

void expand_4d_f32(const float* input, float* output, 
                   int in_b, int in_h, int in_w, int in_c,
                   int out_b, int out_h, int out_w, int out_c) {
    for (int ob = 0; ob < out_b; ob++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int oc = 0; oc < out_c; oc++) {
                    int ib = ob % in_b;
                    int ih = oh % in_h;
                    int iw = ow % in_w;
                    int ic = oc % in_c;
                    output[((long)ob * out_h * out_w + (long)oh * out_w + ow) * out_c + oc] =
                        input[((long)ib * in_h * in_w + (long)ih * in_w + iw) * in_c + ic];
                }
            }
        }
    }
}

void pad_2d_f32(const float* input, float* output, float pad_val,
                int b, int in_h, int in_w, int c,
                int pt, int pb, int pl, int pr) {
    int out_h = in_h + pt + pb;
    int out_w = in_w + pl + pr;
    for (int i_b = 0; i_b < b; i_b++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                for (int i_c = 0; i_c < c; i_c++) {
                    float val = pad_val;
                    if (y >= pt && y < pt + in_h && x >= pl && x < pl + in_w) {
                        val = input[((long)i_b * in_h * in_w + (long)(y - pt) * in_w + (x - pl)) * c + i_c];
                    }
                    output[((long)i_b * out_h * out_w + (long)y * out_w + x) * c + i_c] = val;
                }
            }
        }
    }
}
