/*
 * The exported surface of conv_f32_isa.c, as data.
 *
 * conv_f32_isa.c is compiled twice — once at the build's baseline ISA and once
 * with -mavx2 -mfma — so the AVX2 conv bodies exist in every binary regardless
 * of the build's CPU target, and the choice moves to runtime.  See
 * conv_f32_isa.h for why that is a list instead of per-function declarations.
 *
 * Each entry names the return type, the symbol, its parameter list, and the
 * argument list needed to forward a call.  The includer defines
 * VX_CONV_F32_ENTRY and VX_CONV_F32_ENTRY_VOID; both are undefined on exit so
 * the file can be included repeatedly.  The split exists because C forbids
 * `return expr;` in a function returning void.
 */

#if !defined(VX_CONV_F32_ENTRY) || !defined(VX_CONV_F32_ENTRY_VOID)
#error "define VX_CONV_F32_ENTRY and VX_CONV_F32_ENTRY_VOID before including"
#endif

VX_CONV_F32_ENTRY(const float*, vx_pwf32_pack_cache,
    (int node_idx, const float* wgt, int c, int out_c),
    (node_idx, wgt, c, out_c))

VX_CONV_F32_ENTRY_VOID(void, vx_conv2d_pointwise_f32,
    (int node_idx, const float* input, float* output,
     const float* weights, const float* bias, const float* add,
     long pixels, int channels, int out_channels, int relu),
    (node_idx, input, output, weights, bias, add,
     pixels, channels, out_channels, relu))

VX_CONV_F32_ENTRY_VOID(void, vx_conv2d_dw3x3s1_f32,
    (const float* input, float* output,
     const float* weights, const float* bias,
     int n, int h, int w, int c, int oh, int ow,
     const int* pads, int relu),
    (input, output, weights, bias, n, h, w, c, oh, ow, pads, relu))

VX_CONV_F32_ENTRY_VOID(void, vx_conv2d_dw5x5s1_f32,
    (const float* input, float* output,
     const float* weights, const float* bias,
     int n, int h, int w, int c, int oh, int ow,
     const int* pads, int relu),
    (input, output, weights, bias, n, h, w, c, oh, ow, pads, relu))

VX_CONV_F32_ENTRY_VOID(void, vx_conv2d_depthwise_f32,
    (const float* input, float* output,
     const float* weights, const float* bias,
     int n, int h, int w, int c, int oh, int ow, int out_channels,
     int kh, int kw, int weight_channels, int multiplier,
     int sy, int sx, const int* pads, int dy, int dx, int relu),
    (input, output, weights, bias, n, h, w, c, oh, ow, out_channels,
     kh, kw, weight_channels, multiplier, sy, sx, pads, dy, dx, relu))

VX_CONV_F32_ENTRY_VOID(void, vx_conv2d_generic_f32,
    (int node_idx, const float* input, float* output,
     const float* weights, const float* bias,
     int n, int h, int w, int c, int oh, int ow, int out_channels,
     int kh, int kw, int in_per_group, int groups, int sy, int sx,
     const int* pads, int dy, int dx, int relu),
    (node_idx, input, output, weights, bias, n, h, w, c, oh, ow, out_channels,
     kh, kw, in_per_group, groups, sy, sx, pads, dy, dx, relu))

VX_CONV_F32_ENTRY(int, vx_conv2d_depthwise_pointwise_f32,
    (int dw_node_idx, int pw_node_idx,
     const float* input, float* output,
     const float* dw_weights, const float* dw_bias,
     const float* pw_weights, const float* pw_bias, const float* add,
     int n, int h, int w, int c, int oh, int ow, int out_channels,
     int kh, int kw, int weight_channels, int multiplier,
     int pw_kh, int pw_kw, int pw_in_channels,
     int sy, int sx, const int* pads, int dw_relu, int pw_relu),
    (dw_node_idx, pw_node_idx, input, output, dw_weights, dw_bias,
     pw_weights, pw_bias, add, n, h, w, c, oh, ow, out_channels,
     kh, kw, weight_channels, multiplier, pw_kh, pw_kw, pw_in_channels,
     sy, sx, pads, dw_relu, pw_relu))

VX_CONV_F32_ENTRY(const float**, vx_f32_igemm_indirection_cache,
    (int node_idx, const float* input,
     int n, int h, int w, int c, int oh, int ow, int kh, int kw,
     int sy, int sx, const int* pads, int dy, int dx),
    (node_idx, input, n, h, w, c, oh, ow, kh, kw, sy, sx, pads, dy, dx))

VX_CONV_F32_ENTRY(const float*, vx_f32_igemm_pack_cache,
    (int node_idx, const float* weights, int c, int out_channels, int ks),
    (node_idx, weights, c, out_channels, ks))

VX_CONV_F32_ENTRY(int, vx_conv2d_spatial_igemm_f32,
    (int node_idx, const float* input, float* output,
     const float* weights, const float* bias,
     int n, int h, int w, int c, int oh, int ow, int out_channels,
     int kh, int kw, int sy, int sx, const int* pads,
     int dy, int dx, int relu),
    (node_idx, input, output, weights, bias, n, h, w, c, oh, ow, out_channels,
     kh, kw, sy, sx, pads, dy, dx, relu))

VX_CONV_F32_ENTRY_VOID(void, vx_conv_f32_isa_free_all, (void), ())

#undef VX_CONV_F32_ENTRY
#undef VX_CONV_F32_ENTRY_VOID
