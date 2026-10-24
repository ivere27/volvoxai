#include "../src/kernels/inference_kernels.h"
#include "../src/kernels/w8a8_affine.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c)                                                                                   \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c);                                \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

typedef struct Geometry {
    uint32_t h, w, ic, oc, kh, kw, groups, sy, sx, dy, dx, pt, pl, pb, pr;
} Geometry;

static void test_case(Geometry g, uint32_t in_dtype, uint32_t out_dtype, int32_t iz, int32_t oz,
                      uint32_t relu, int with_bias, uint32_t bits) {
    const uint32_t batch = 2;
    uint32_t oh = (g.h + g.pt + g.pb - (g.kh - 1) * g.dy - 1) / g.sy + 1;
    uint32_t ow = (g.w + g.pl + g.pr - (g.kw - 1) * g.dx - 1) / g.sx + 1;
    size_t k = (size_t)g.kh * g.kw * (g.ic / g.groups), row = (k * bits + 7) / 8;
    size_t ni = (size_t)batch * g.h * g.w * g.ic, no = (size_t)batch * oh * ow * g.oc;
    uint8_t *input = malloc(ni), *packed = malloc(row * g.oc), *output = malloc(no),
            *reference = malloc(no);
    int8_t *logical = malloc(k * g.oc);
    float *scales = malloc(g.oc * sizeof(float));
    int32_t *bias = malloc(g.oc * sizeof(int32_t));
    size_t workspace_size = qconv2d_packed_a8_workspace_bytes();
    void *workspace = malloc(workspace_size);
    CHECK(input && packed && output && reference && logical && scales && bias && workspace);
    CHECK(workspace_size > 5120 && workspace_size < 2u * 1024u * 1024u);
    for (size_t i = 0; i < ni; i++)
        input[i] = (uint8_t)((i * 73 + 19) % 256);
    memset(packed, bits == 3 ? 0 : bits == 2 ? 0xaa : 0x88, row * g.oc);
    for (uint32_t c = 0; c < g.oc; c++) {
        scales[c] = 0.002f * (float)(1 + c % 7);
        bias[c] = (int32_t)(c * 13) - 127;
        for (size_t j = 0; j < k; j++) {
            int q = (int)((j * 5 + c * 11) % (1u << bits)) - (1 << (bits - 1));
            logical[c * k + j] = (int8_t)q;
            unsigned code = (unsigned)(q + (1 << (bits - 1)));
            for (unsigned b = 0; b < bits; b++) {
                size_t at = c * row + (j * bits + b) / 8;
                unsigned shift = (unsigned)(j * bits + b) % 8;
                packed[at] = (uint8_t)((packed[at] & ~(1u << shift)) | (((code >> b) & 1u) << shift));
            }
        }
    }
    float is = 0.015f, os = 0.03f;
#define RUN(cap)                                                                                   \
    qconv2d_packed_a8(bits, input, packed, with_bias ? bias : NULL, scales, output, batch, g.h, g.w, g.ic,    \
                 oh, ow, g.oc, g.kh, g.kw, g.ic / g.groups, g.sy, g.sx, g.dy, g.dx, g.pt, g.pl,    \
                 g.pb, g.pr, g.groups, relu, is, iz, os, oz, in_dtype, out_dtype, workspace,       \
                 (cap))
    CHECK(RUN(workspace_size) == 1);
    for (uint32_t b = 0; b < batch; b++)
        for (uint32_t y = 0; y < oh; y++)
            for (uint32_t x = 0; x < ow; x++)
                for (uint32_t c = 0; c < g.oc; c++) {
                    int64_t sum = with_bias ? bias[c] : 0;
                    uint32_t group = c / (g.oc / g.groups), icg = g.ic / g.groups;
                    for (uint32_t ky = 0; ky < g.kh; ky++)
                        for (uint32_t kx = 0; kx < g.kw; kx++) {
                            int64_t iy = (int64_t)y * g.sy + (int64_t)ky * g.dy - g.pt;
                            int64_t ix = (int64_t)x * g.sx + (int64_t)kx * g.dx - g.pl;
                            if (iy < 0 || iy >= g.h || ix < 0 || ix >= g.w)
                                continue;
                            for (uint32_t ci = 0; ci < icg; ci++) {
                                size_t at =
                                    (((size_t)b * g.h + (size_t)iy) * g.w + (size_t)ix) * g.ic +
                                    group * icg + ci;
                                int32_t a = in_dtype == VX_DTYPE_I8 ? (int32_t)(int8_t)input[at]
                                                                    : input[at];
                                sum += (int64_t)(a - iz) *
                                       logical[c * k + ((size_t)ky * g.kw + kx) * icg + ci];
                            }
                        }
                    float product = is * scales[c], multiplier = product / os;
                    float scaled = (float)sum * multiplier, transformed = scaled + (float)oz;
                    int32_t lo = out_dtype == VX_DTYPE_I8 ? -128 : 0,
                            hi = out_dtype == VX_DTYPE_I8 ? 127 : 255;
                    int32_t q = vx_w8a8_requantize(transformed, lo, hi, oz);
                    if (relu && q < oz)
                        q = oz;
                    int32_t upper = vx_w8a8_requantize(6.0f / os + (float)oz, lo, hi, 0);
                    if (relu == 2 && q > upper)
                        q = upper;
                    reference[(((size_t)b * oh + y) * ow + x) * g.oc + c] = (uint8_t)q;
                }
    CHECK(memcmp(output, reference, no) == 0);
    memset(output, 0x5a, no);
    CHECK(RUN(workspace_size - 1) == 0);
    void *saved_workspace = workspace;
    workspace = (uint8_t *)saved_workspace + 1;
    CHECK(RUN(workspace_size) == 0);
    workspace = malloc(1);
    CHECK(workspace);
    CHECK(RUN(1) == 0);
    free(workspace);
    workspace = saved_workspace;
    for (size_t i = 0; i < no; i++)
        CHECK(output[i] == 0x5a);
    float old = scales[0];
    scales[0] = 0;
    CHECK(RUN(workspace_size) == 0);
    scales[0] = old;
    float old_is = is, old_os = os;
    is = FLT_MAX;
    scales[0] = 2.0f;
    CHECK(RUN(workspace_size) == 0);
    is = FLT_MIN;
    scales[0] = FLT_MIN;
    CHECK(RUN(workspace_size) == 0);
    is = old_is;
    scales[0] = old;
    os = FLT_MIN;
    scales[0] = FLT_MAX;
    CHECK(RUN(workspace_size) == 0);
    os = FLT_MAX;
    scales[0] = FLT_MIN;
    CHECK(RUN(workspace_size) == 0);
    is = old_is;
    os = old_os;
    scales[0] = old;
    if (with_bias) {
        int32_t oldbias = bias[0];
        bias[0] = INT32_MIN;
        CHECK(RUN(workspace_size) == 0);
        bias[0] = oldbias;
    }
    if (k * bits % 8) {
        uint8_t oldtail = packed[row - 1];
        packed[row - 1] ^= (uint8_t)(1u << (k * bits % 8));
        CHECK(RUN(workspace_size) == 0);
        packed[row - 1] = oldtail;
    }
    for (size_t i = 0; i < no; i++)
        CHECK(output[i] == 0x5a);
#undef RUN
    free(input);
    free(packed);
    free(output);
    free(reference);
    free(logical);
    free(scales);
    free(bias);
    free(workspace);
}

int main(void) {
    static const Geometry shapes[] = {
        {2, 3, 1, 5, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {2, 3, 2, 5, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {2, 3, 5, 5, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {5, 7, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {7, 8, 3, 19, 3, 2, 1, 2, 1, 1, 2, 1, 2, 0, 1},
        {5, 9, 18, 34, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1},
        {9, 7, 3, 3, 3, 3, 3, 2, 1, 2, 1, 2, 1, 2, 1},
        {2, 9, 65, 17, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {4, 5, 129, 33, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {2, 2, 4096, 17, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {2, 2, 4097, 17, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {3, 4, 513, 17, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    };
    unsigned cases = 0;
    for (uint32_t bits = 2; bits <= 4; bits++)
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
        for (unsigned a = 0; a < 2; a++)
            for (unsigned o = 0; o < 2; o++)
                for (unsigned relu = 0; relu < 3; relu++)
                    for (unsigned bias = 0; bias < 2; bias++) {
                        test_case(shapes[s], a ? VX_DTYPE_U8 : VX_DTYPE_I8,
                                  o ? VX_DTYPE_U8 : VX_DTYPE_I8,
                                  a ? (s % 2 ? 255 : 0) : (s % 2 ? -128 : 127), o ? 113 : -17, relu,
                                  bias, bits);
                        cases++;
                    }
    printf("packed Conv2/3/4: %u exact integer-reference cases and malformed-input checks passed\n",
           cases);
    return 0;
}
