/* OpenGL ES F32 graph paths: copy, comparison, unary and elementwise,
 * normalization, pooling, embedding, resize, gather, split and attention. */


int opengl_graph_copy_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_copy, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_add_f32(const float* a, const float* bptr, float* out, long n) {
    return opengl_graph_add_relu_f32(a, bptr, out, n, 0);
}

int opengl_graph_add_relu_f32(const float* a, const float* bptr, float* out, long n, int relu) {
    if (n <= 0 || !a || !bptr || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(bptr, bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t params[2] = {(uint32_t)n, (uint32_t)relu};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {sa->buffer, sb->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_add, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_add3_relu_f32(const float* a, const float* bptr, const float* c, float* out, long n, int relu) {
    if (n <= 0 || !a || !bptr || !c || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(bptr, bytes, 0);
    OglTensorSlot* sc = graph_ensure_device(c, bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, bytes);
    if (!sa || !sb || !sc || !so) return 0;
    uint32_t params[2] = {(uint32_t)n, (uint32_t)relu};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {sa->buffer, sb->buffer, sc->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_add3, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_clip_f32(const float* in, float* out, long n, float min_v, float max_v) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct { uint32_t size; float min_v; float max_v; } params = {(uint32_t)n, min_v, max_v};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_clip, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static int opengl_graph_typed_control_32(
        const void* a, size_t a_bytes, const void* b, size_t b_bytes,
        void* output, size_t output_bytes,
        const VxTypedControlMetadata* metadata) {
    if (!metadata) return 0;
    OglTensorSlot* a_slot = graph_ensure_device(a, a_bytes, 0);
    OglTensorSlot* b_slot = graph_ensure_device(b, b_bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    GLuint metadata_buffer =
        create_buffer(sizeof(*metadata), metadata);
    if (!metadata_buffer) return 0;
    GLuint buffers[4] = {
        a_slot->buffer, b_slot->buffer,
        output_slot->buffer, metadata_buffer,
    };
    uint32_t elements = metadata->values[0];
    int ok = dispatch_kernel(
        &k_typed_control_32, buffers,
        (elements + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &metadata_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_compare_i32(
        const int32_t* a, long a_elements,
        const int32_t* b, long b_elements,
        int32_t* output, long output_elements,
        const uint32_t* output_strides,
        const uint32_t* a_strides,
        const uint32_t* b_strides,
        int rank, int operation) {
    VxTypedControlMetadata metadata;
    size_t a_bytes;
    size_t b_bytes;
    size_t output_bytes;
    if (!vx_typed_control_compare_plan(
            a, a_elements, b, b_elements, output, output_elements,
            output_strides, a_strides, b_strides, rank, operation,
            &metadata, &a_bytes, &b_bytes, &output_bytes))
        return 0;
    return opengl_graph_typed_control_32(
        a, a_bytes, b, b_bytes, output, output_bytes, &metadata);
}

static int opengl_graph_unary_i32(
        const int32_t* input, int32_t* output, long elements,
        int operation, int32_t minimum, int32_t maximum) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_unary_plan(
            input, output, elements, operation, minimum, maximum,
            &metadata, &bytes))
        return 0;
    return opengl_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int opengl_graph_not_i32(
        const int32_t* input, int32_t* output, long elements) {
    return opengl_graph_unary_i32(
        input, output, elements, VX_TYPED_CONTROL_NOT_I32, 0, 0);
}

int opengl_graph_clip_i32(
        const int32_t* input, int32_t* output, long elements,
        int32_t minimum, int32_t maximum) {
    return opengl_graph_unary_i32(
        input, output, elements, VX_TYPED_CONTROL_CLIP_I32,
        minimum, maximum);
}

int opengl_graph_copy_32(
        const void* input, void* output, long elements) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_copy_plan(
            input, output, elements, &metadata, &bytes))
        return 0;
    return opengl_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int opengl_graph_cast_typed(
        const void* input, int input_dtype,
        void* output, int output_dtype, long elements) {
    VxTypedControlMetadata metadata;
    size_t bytes;
    if (!vx_typed_control_cast_plan(
            input, input_dtype, output, output_dtype, elements,
            &metadata, &bytes))
        return 0;
    return opengl_graph_typed_control_32(
        input, bytes, input, bytes, output, bytes, &metadata);
}

int opengl_graph_sigmoid_f32(const float* in, float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_sigmoid, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static int opengl_graph_unary_size_f32(const OglKernel* k, const float* in,
                                       float* out, long n) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[1] = {(uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(k, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_relu_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_relu, in, out, n); }
int opengl_graph_gelu_f32(const float* in, float* out, long n, int approximate_tanh) {
    if (n <= 0 || (uint64_t)n > UINT32_MAX || !in || !out ||
        (approximate_tanh != 0 && approximate_tanh != 1)) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)approximate_tanh, 0u, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_gelu, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
int opengl_graph_silu_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_silu, in, out, n); }
int opengl_graph_tanh_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_tanh, in, out, n); }
int opengl_graph_hardswish_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_hardswish, in, out, n); }
int opengl_graph_hardsigmoid_f32(const float* in, float* out, long n) { return opengl_graph_unary_size_f32(&k_hardsigmoid, in, out, n); }
int opengl_graph_cast_copy_f32(const float* in, float* out, long n) { return opengl_graph_copy_f32(in, out, n); }

int opengl_graph_leaky_relu_f32(const float* in, float* out, long n, float alpha) {
    if (n <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct { uint32_t size; float alpha; } params = {(uint32_t)n, alpha};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_leaky_relu, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_prelu_f32(const float* in, const float* weight, float* out, long n, int channels) {
    if (n <= 0 || channels <= 0 || !in || !weight || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    size_t wbytes = (size_t)channels * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)channels, (uint32_t)channels, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {src->buffer, w->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_prelu, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_layernorm_f32(const float* in, const float* weight, const float* bias,
                               float* out, int rows, int d_model, float eps) {
    if (rows <= 0 || d_model <= 0 || !in || !weight || !bias || !out ||
        !(eps > 0.0f) || !isfinite(eps)) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t wbytes = (size_t)d_model * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* b = graph_ensure_device(bias, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !b || !dst) return 0;
    struct {
        uint32_t rows;
        uint32_t d_model;
        float eps;
        uint32_t pad;
    } params = {(uint32_t)rows, (uint32_t)d_model, eps, 0u};
    _Static_assert(sizeof(params) == 16, "LayerNorm uniform ABI");
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[5] = {src->buffer, w->buffer, b->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_layernorm, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_rmsnorm_f32(const float* in, const float* weight, float* out,
                             int rows, int d_model, float eps) {
    if (rows <= 0 || d_model <= 0 || !in || !weight || !out) return 0;
    size_t bytes = (size_t)rows * d_model * sizeof(float);
    size_t wbytes = (size_t)d_model * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* w = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !w || !dst) return 0;
    struct { uint32_t rows; uint32_t d_model; float eps; } params = {(uint32_t)rows, (uint32_t)d_model, eps};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[4] = {src->buffer, w->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_rmsnorm, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static int opengl_graph_softmax_like_f32(const OglKernel* k, const float* in,
                                         float* out, int rows, int d) {
    if (rows <= 0 || d <= 0 || !in || !out) return 0;
    size_t bytes = (size_t)rows * d * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t params[2] = {(uint32_t)rows, (uint32_t)d};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(k, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_softmax_f32(const float* in, float* out, int rows, int d) {
    return opengl_graph_softmax_like_f32(&k_softmax, in, out, rows, d);
}

int opengl_graph_logsoftmax_f32(const float* in, float* out, int rows, int d) {
    return opengl_graph_softmax_like_f32(&k_logsoftmax, in, out, rows, d);
}

int opengl_graph_reduce_f32(const float* in, float* out, int rows, int d, float inv) {
    if (rows <= 0 || d <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)rows * d * sizeof(float);
    size_t out_bytes = (size_t)rows * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct { uint32_t rows; uint32_t d; float inv; } params = {(uint32_t)rows, (uint32_t)d, inv};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_reduce, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_global_average_pool_f32(const float* in, float* out, int n, int h, int w, int c) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_globalavg, bufs, ((uint32_t)c + 63u) / 64u, (uint32_t)n, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_average_pool2d_f32(const float* in, float* out, int n, int h, int w, int c,
                                    int out_h, int out_w, int ky, int kx, int sy, int sx,
                                    int py, int px) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_h <= 0 || out_w <= 0 ||
        ky <= 0 || kx <= 0 || sy <= 0 || sx <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[12] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                           (uint32_t)out_h, (uint32_t)out_w, (uint32_t)ky, (uint32_t)kx,
                           (uint32_t)sy, (uint32_t)sx, (uint32_t)py, (uint32_t)px};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_avgpool, bufs, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * c));
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_batchnorm2d_f32(const float* in, const float* weight, const float* bias,
                                 const float* mean, const float* var, float* out,
                                 int n, int h, int w, int c, float eps) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || !in || !weight || !bias || !mean || !var || !out) return 0;
    size_t bytes = (size_t)n * h * w * c * sizeof(float);
    size_t cbytes = (size_t)c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, cbytes, 1);
    OglTensorSlot* sb = graph_ensure_device(bias, cbytes, 1);
    OglTensorSlot* sm = graph_ensure_device(mean, cbytes, 1);
    OglTensorSlot* sv = graph_ensure_device(var, cbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !sw || !sb || !sm || !sv || !dst) return 0;
    struct { uint32_t n; uint32_t c; uint32_t h; uint32_t w; float eps; uint32_t pad[3]; } params =
        {(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w, eps, {0, 0, 0}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[7] = {src->buffer, sw->buffer, sb->buffer, sm->buffer, sv->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_batchnorm, bufs, ((uint32_t)(n * h * w * c) + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_groupnorm_f32(const float* in, const float* weight, const float* bias,
                               float* out, int n, int h, int w, int c, int groups,
                               float eps) {
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || groups <= 0 || c % groups ||
        !in || !weight || !bias || !out || !(eps > 0.0f)) return 0;
    size_t bytes = (size_t)n * h * w * c * sizeof(float);
    size_t cbytes = (size_t)c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, cbytes, 1);
    OglTensorSlot* sb = graph_ensure_device(bias, cbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !sw || !sb || !dst) return 0;
    struct {
        uint32_t n, h, w, c, groups, has_bias;
        float eps;
        uint32_t pad;
    } params = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                (uint32_t)groups, 1u, eps, 0u};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[5] = {src->buffer, sw->buffer, sb->buffer, dst->buffer, pb};
    uint32_t total_groups = (uint32_t)n * (uint32_t)groups;
    int ok = dispatch_kernel(&k_groupnorm, bufs, (total_groups + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int opengl_graph_dropout_f32(const float* in, float* out, long n, uint32_t threshold,
                             uint32_t seed, uint32_t counter, float scale) {
    if (n <= 0 || (uint64_t)n > UINT32_MAX || !in || !out || !(scale >= 1.0f)) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    struct {
        uint32_t length, threshold, seed, counter;
        float scale;
        uint32_t pad[3];
    } params = {(uint32_t)n, threshold, seed, counter, scale, {0u, 0u, 0u}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_dropout, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

/* Routed expert linear. `experts` counts staged rows; route indices stay in
 * global slot space and are mapped through slot_rows when slot_domain != 0. */
int opengl_graph_moe_linear_f32(const float* input, const float* expert_weight,
                                const float* expert_bias, const float* route_indices,
                                const float* route_weights, float* out, int rows,
                                int d_in, int d_out, int experts, int top_k,
                                const uint32_t* slot_rows, uint32_t slot_domain) {
    if (rows <= 0 || d_in <= 0 || d_out <= 0 || experts <= 0 || top_k <= 0 ||
        !input || !expert_weight || !route_indices || !route_weights || !out) return 0;
    if (slot_rows ? (slot_domain < (uint32_t)experts ||
                     (uint32_t)top_k > slot_domain)
                  : top_k > experts) return 0;
    size_t in_bytes = (size_t)rows * (size_t)d_in * sizeof(float);
    size_t w_bytes = (size_t)experts * (size_t)d_in * (size_t)d_out * sizeof(float);
    size_t bias_bytes = (size_t)experts * (size_t)d_out * sizeof(float);
    size_t route_bytes = (size_t)rows * (size_t)top_k * sizeof(float);
    size_t out_bytes = (size_t)rows * (size_t)d_out * sizeof(float);
    uint32_t slot_extent = slot_domain ? slot_domain : 1u;
    OglTensorSlot* si = graph_ensure_device(input, in_bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(expert_weight, w_bytes, 1);
    OglTensorSlot* sri = graph_ensure_device(route_indices, route_bytes, 0);
    OglTensorSlot* srw = graph_ensure_device(route_weights, route_bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, out_bytes);
    OglTensorSlot* sb = expert_bias
        ? graph_ensure_device(expert_bias, bias_bytes, 1) : NULL;
    if (!si || !sw || !sri || !srw || !so || (expert_bias && !sb)) return 0;
    /* The shader always declares a bias and a slot table; absent ones become
     * zero-filled transients so the binding set stays complete. */
    GLuint bias_dummy = 0;
    if (!expert_bias) {
        float* zeros = (float*)calloc((size_t)experts * (size_t)d_out, sizeof(float));
        if (!zeros) return 0;
        bias_dummy = create_buffer_target(GL_SHADER_STORAGE_BUFFER, bias_bytes, zeros);
        free(zeros);
        if (!bias_dummy) return 0;
    }
    uint32_t* table = (uint32_t*)calloc(slot_extent, sizeof(uint32_t));
    if (!table) {
        if (bias_dummy) p_glDeleteBuffers(1, &bias_dummy);
        return 0;
    }
    if (slot_domain) memcpy(table, slot_rows, (size_t)slot_domain * sizeof(uint32_t));
    GLuint slot_buffer = create_buffer_target(
        GL_SHADER_STORAGE_BUFFER, (size_t)slot_extent * sizeof(uint32_t), table);
    free(table);
    if (!slot_buffer) {
        if (bias_dummy) p_glDeleteBuffers(1, &bias_dummy);
        return 0;
    }
    uint32_t params[8] = {
        (uint32_t)rows, (uint32_t)d_in, (uint32_t)d_out, (uint32_t)experts,
        (uint32_t)top_k, expert_bias ? 1u : 0u, slot_domain, 0u,
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[8] = {
        si->buffer, sw->buffer, expert_bias ? sb->buffer : bias_dummy,
        sri->buffer, srw->buffer, so->buffer, pb, slot_buffer,
    };
    int ok = dispatch_kernel(&k_moe_linear, bufs,
                             ((uint32_t)d_out + 63u) / 64u, (uint32_t)rows, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (bias_dummy) p_glDeleteBuffers(1, &bias_dummy);
    p_glDeleteBuffers(1, &slot_buffer);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

/* Top-k expert routing. The router weight is [d_model, experts], so its expert
 * axis is 1 and it is never a slot-indexed bank. */
int opengl_graph_moe_router_f32(const float* input, const float* weight,
                                const float* bias, float* route_indices,
                                float* route_weights, int rows, int d_model,
                                int experts, int top_k, float temperature,
                                int normalize) {
    if (rows <= 0 || d_model <= 0 || experts <= 0 || top_k <= 0 ||
        top_k > experts || top_k > 8 || !(temperature > 0.0f) ||
        !input || !weight || !route_indices || !route_weights) return 0;
    size_t in_bytes = (size_t)rows * (size_t)d_model * sizeof(float);
    size_t w_bytes = (size_t)d_model * (size_t)experts * sizeof(float);
    size_t bias_bytes = (size_t)experts * sizeof(float);
    size_t route_bytes = (size_t)rows * (size_t)top_k * sizeof(float);
    OglTensorSlot* si = graph_ensure_device(input, in_bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, w_bytes, 1);
    OglTensorSlot* sidx = graph_output_slot(route_indices, route_bytes);
    OglTensorSlot* srw = graph_output_slot(route_weights, route_bytes);
    OglTensorSlot* sb = bias ? graph_ensure_device(bias, bias_bytes, 1) : NULL;
    if (!si || !sw || !sidx || !srw || (bias && !sb)) return 0;
    GLuint bias_dummy = 0;
    if (!bias) {
        float* zeros = (float*)calloc((size_t)experts, sizeof(float));
        if (!zeros) return 0;
        bias_dummy = create_buffer_target(GL_SHADER_STORAGE_BUFFER, bias_bytes, zeros);
        free(zeros);
        if (!bias_dummy) return 0;
    }
    uint32_t params[8];
    params[0] = (uint32_t)rows;
    params[1] = (uint32_t)d_model;
    params[2] = (uint32_t)experts;
    params[3] = (uint32_t)top_k;
    params[4] = normalize ? 1u : 0u;
    params[5] = bias ? 1u : 0u;
    memcpy(&params[6], &temperature, sizeof(float));
    params[7] = 0u;
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[6] = {
        si->buffer, sw->buffer, bias ? sb->buffer : bias_dummy,
        sidx->buffer, srw->buffer, pb,
    };
    int ok = dispatch_kernel(&k_moe_router, bufs, ((uint32_t)rows + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (bias_dummy) p_glDeleteBuffers(1, &bias_dummy);
    if (!ok) return 0;
    graph_mark_device(sidx);
    graph_mark_device(srw);
    return 1;
}

int opengl_graph_embedding_f32(const int32_t* tokens, const float* weight, float* out,
                               int tokens_len, int d_model, int vocab_size) {
    if (tokens_len <= 0 || d_model <= 0 || vocab_size <= 0 || !tokens || !weight || !out) return 0;
    size_t tbytes = (size_t)tokens_len * sizeof(int32_t);
    size_t wbytes = (size_t)vocab_size * d_model * sizeof(float);
    size_t obytes = (size_t)tokens_len * d_model * sizeof(float);
    OglTensorSlot* st = graph_ensure_device(tokens, tbytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* so = graph_output_slot(out, obytes);
    if (!st || !sw || !so) return 0;
    uint32_t params[4] = {(uint32_t)tokens_len, (uint32_t)d_model,
                          (uint32_t)vocab_size, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {st->buffer, sw->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_embedding, bufs, ((uint32_t)tokens_len + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_transpose_f32(const float* in, float* out, const int* in_shape,
                               const int* perm, int rank) {
    if (rank <= 0 || rank > 8 || !in || !out || !in_shape || !perm) return 0;
    uint32_t in_stride[8] = {0}, out_shape[8] = {0}, out_stride[8] = {0};
    long total = 1;
    for (int i = 0; i < rank; i++) {
        if (in_shape[i] <= 0 || perm[i] < 0 || perm[i] >= rank) return 0;
        out_shape[i] = (uint32_t)in_shape[perm[i]];
        total *= out_shape[i];
    }
    in_stride[rank - 1] = 1;
    out_stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        in_stride[i] = in_stride[i + 1] * (uint32_t)in_shape[i + 1];
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }
    size_t bytes = (size_t)total * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !dst) return 0;
    uint32_t meta[2 + 16] = {0};
    meta[0] = (uint32_t)total;
    meta[1] = (uint32_t)rank;
    for (int d = 0; d < rank; d++) {
        meta[2 + d] = out_stride[d];
        meta[2 + rank + d] = in_stride[perm[d]];
    }
    size_t mbytes = (size_t)(2 + 2 * rank) * sizeof(uint32_t);
    GLuint mb = create_buffer(mbytes, meta);
    GLuint bufs[3] = {src->buffer, dst->buffer, mb};
    int ok = dispatch_kernel(&k_transpose, bufs, ((uint32_t)total + 63u) / 64u, 1, 1);
    if (mb) p_glDeleteBuffers(1, &mb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_where_f32(const float* cond, const float* a, const float* b, float* out, long n) {
    if (n <= 0 || !cond || !a || !b || !out) return 0;
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* sc = graph_ensure_device(cond, bytes, 0);
    OglTensorSlot* sa = graph_ensure_device(a, bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(b, bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, bytes);
    if (!sc || !sa || !sb || !so) return 0;
    uint32_t params[1] = {(uint32_t)n};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {sc->buffer, sa->buffer, sb->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_where, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_where_32(
        const int32_t* condition, const void* a,
        const void* b, void* output, long elements) {
    uint32_t count;
    size_t bytes;
    if (!vx_typed_control_where_plan(
            condition, a, b, output, elements, &count, &bytes))
        return 0;
    OglTensorSlot* condition_slot =
        graph_ensure_device(condition, bytes, 0);
    OglTensorSlot* a_slot = graph_ensure_device(a, bytes, 0);
    OglTensorSlot* b_slot = graph_ensure_device(b, bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_slot(output, bytes);
    if (!condition_slot || !a_slot || !b_slot || !output_slot) return 0;
    uint32_t params[4] = {count, 0u, 0u, 0u};
    GLuint params_handle = params_buffer(params, sizeof(params));
    if (!params_handle) return 0;
    GLuint buffers[5] = {
        condition_slot->buffer, a_slot->buffer, b_slot->buffer,
        output_slot->buffer, params_handle,
    };
    int ok = dispatch_kernel(
        &k_where_32, buffers, (count + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &params_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_argmax_f32(
        const float* input, int32_t* output,
        uint32_t outer, uint32_t axis_size, uint32_t inner) {
    uint32_t input_elements;
    uint32_t output_elements;
    size_t input_bytes;
    size_t output_bytes;
    if (!vx_typed_control_argmax_plan(
            input, output, outer, axis_size, inner,
            &input_elements, &output_elements,
            &input_bytes, &output_bytes))
        return 0;
    OglTensorSlot* input_slot =
        graph_ensure_device(input, input_bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_slot(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    uint32_t params[4] = {outer, axis_size, inner, 0u};
    GLuint params_handle = params_buffer(params, sizeof(params));
    if (!params_handle) return 0;
    GLuint buffers[3] = {
        input_slot->buffer, output_slot->buffer, params_handle,
    };
    int ok = dispatch_kernel(
        &k_argmax_f32_i32, buffers,
        (output_elements + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &params_handle);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    (void)input_elements;
    return 1;
}

int opengl_graph_upsample2x_f32(const float* in, float* out, int n, int h, int w, int c) {
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * h * 2 * w * 2 * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_upsample, bufs, ((uint32_t)(w * 2) + 7u) / 8u,
                             ((uint32_t)(h * 2) + 7u) / 8u, (uint32_t)(n * c));
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_resize_nearest_f32(const float* in, float* out, int n, int h, int w, int c,
                                    int out_h, int out_w) {
    return opengl_graph_resize_f32(in, out, n, h, w, c, out_h, out_w, 0);
}

int opengl_graph_resize_f32(const float* in, float* out, int n, int h, int w, int c,
                            int out_h, int out_w, int mode) {
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || out_h <= 0 || out_w <= 0 || !in || !out ||
        (uint64_t)(uint32_t)n * (uint32_t)c > UINT32_MAX) return 0;
    size_t in_bytes = (size_t)n * h * w * c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[8] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)c,
                          (uint32_t)out_h, (uint32_t)out_w, (uint32_t)mode, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_resize, bufs, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * c));
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static void pad4_shape(const int* shape, int rank, uint32_t out[4]) {
    for (int i = 0; i < 4; i++) out[i] = 1u;
    if (!shape || rank <= 0) return;
    int base = 4 - rank;
    if (base < 0) base = 0;
    for (int i = 0; i < rank && i < 4; i++) out[base + i] = (uint32_t)shape[i];
}

int opengl_graph_expand_f32(const float* in, float* out, const int* in_shape, int in_rank,
                            const int* out_shape, int out_rank) {
    VxExpandF32Plan plan;
    if (!vx_expand_f32_plan(
            in, out, in_shape, in_rank, out_shape, out_rank, &plan))
        return 0;
    OglTensorSlot* src = graph_ensure_device(in, plan.input_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, plan.output_bytes);
    if (!src || !dst) return 0;
    GLuint pb = params_buffer(plan.params, sizeof(plan.params));
    if (!pb) return 0;
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(
        &k_expand, bufs, (plan.output_elements + 63u) / 64u, 1u, 1u);
    p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_expand_32(const void* in, void* out, const int* in_shape,
                           int in_rank, const int* out_shape, int out_rank) {
    /* expand.wgsl is word-typed and therefore preserves every I32 bit. */
    return opengl_graph_expand_f32(
        (const float*)in, (float*)out,
        in_shape, in_rank, out_shape, out_rank);
}

int opengl_graph_batch_matmul_f32(
        const float* a, const int* a_shape, int a_rank,
        const float* b, const int* b_shape, int b_rank,
        float* output, const int* output_shape, int output_rank) {
    VxBatchMatMulF32Plan plan;
    size_t metadata_bytes;
    if (!vx_batch_matmul_f32_plan(
            a, a_shape, a_rank, b, b_shape, b_rank,
            output, output_shape, output_rank, &plan))
        return 0;
    metadata_bytes = (size_t)plan.metadata_words * sizeof(uint32_t);
    OglTensorSlot* a_slot = graph_ensure_device(a, plan.a_bytes, 0);
    OglTensorSlot* b_slot = graph_ensure_device(b, plan.b_bytes, 0);
    OglTensorSlot* output_slot =
        graph_output_slot(output, plan.output_bytes);
    GLuint metadata_buffer = create_buffer(metadata_bytes, plan.metadata);
    if (!a_slot || !b_slot || !output_slot || !metadata_buffer) {
        if (metadata_buffer) p_glDeleteBuffers(1, &metadata_buffer);
        return 0;
    }
    GLuint buffers[4] = {
        a_slot->buffer, b_slot->buffer, output_slot->buffer, metadata_buffer,
    };
    int ok = dispatch_kernel(
        &k_batch_matmul, buffers,
        (plan.n + 7u) / 8u, (plan.m + 7u) / 8u,
        plan.output_batches);
    p_glDeleteBuffers(1, &metadata_buffer);
    if (!ok) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int opengl_graph_gather_i32_f32(const float* input, const int32_t* indices,
                                float* output, int outer, int axis_size,
                                int inner, int indices_elements,
                                int output_elements) {
    uint64_t input_count;
    uint64_t expected_output;
    size_t input_bytes;
    size_t index_bytes;
    size_t output_bytes;
    if (!input || !indices || !output || outer <= 0 || axis_size <= 0 ||
        inner <= 0 || indices_elements <= 0 || output_elements <= 0) return 0;
    input_count = (uint64_t)(uint32_t)outer * (uint32_t)axis_size *
        (uint32_t)inner;
    expected_output = (uint64_t)(uint32_t)outer * (uint32_t)indices_elements *
        (uint32_t)inner;
    if (input_count > SIZE_MAX / sizeof(float) ||
        expected_output != (uint32_t)output_elements ||
        expected_output > SIZE_MAX / sizeof(float)) return 0;
    input_bytes = (size_t)input_count * sizeof(float);
    index_bytes = (size_t)(uint32_t)indices_elements * sizeof(int32_t);
    if (index_bytes / sizeof(int32_t) !=
        (size_t)(uint32_t)indices_elements) return 0;
    output_bytes = (size_t)expected_output * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(input, input_bytes, 0);
    OglTensorSlot* idx = graph_ensure_device(indices, index_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(output, output_bytes);
    if (!src || !idx || !dst) return 0;
    uint32_t params[5] = {
        (uint32_t)outer, (uint32_t)axis_size, (uint32_t)inner,
        (uint32_t)indices_elements, (uint32_t)output_elements,
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[4] = {src->buffer, idx->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_gather, bufs,
                             ((uint32_t)output_elements + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_pad4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                           const int* out_shape, int out_rank, int pad_top, int pad_left, float value) {
    if (!in || !out || !in_shape || !out_shape || in_rank <= 0 || out_rank <= 0 || in_rank > 4 || out_rank > 4) return 0;
    uint32_t is[4], os[4];
    pad4_shape(in_shape, in_rank, is);
    pad4_shape(out_shape, out_rank, os);
    size_t in_elems = (size_t)is[0] * is[1] * is[2] * is[3];
    size_t out_elems = (size_t)os[0] * os[1] * os[2] * os[3];
    OglTensorSlot* src = graph_ensure_device(in, in_elems * sizeof(float), 0);
    OglTensorSlot* dst = graph_output_slot(out, out_elems * sizeof(float));
    if (!src || !dst) return 0;
    struct {
        uint32_t b, in_h, in_w, c, out_h, out_w, pt, pl;
        float val;
        uint32_t pad[3];
    } params = {is[0], is[1], is[2], is[3], os[1], os[2], (uint32_t)pad_top, (uint32_t)pad_left, value, {0, 0, 0}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_pad, bufs, ((uint32_t)out_elems + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_slice4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                             const int* out_shape, int out_rank, const int* starts,
                             const int* steps) {
    if (!in || !out || !in_shape || !out_shape || !starts || !steps ||
        in_rank <= 0 || out_rank <= 0 || in_rank > 4 || out_rank > 4) return 0;
    uint32_t is[4], os[4];
    pad4_shape(in_shape, in_rank, is);
    pad4_shape(out_shape, out_rank, os);
    size_t in_elems = (size_t)is[0] * is[1] * is[2] * is[3];
    size_t out_elems = (size_t)os[0] * os[1] * os[2] * os[3];
    OglTensorSlot* src = graph_ensure_device(in, in_elems * sizeof(float), 0);
    OglTensorSlot* dst = graph_output_slot(out, out_elems * sizeof(float));
    if (!src || !dst) return 0;
    uint32_t params[16] = {
        os[0], os[1], os[2], os[3], is[1], is[2], is[3],
        (uint32_t)starts[0], (uint32_t)starts[1], (uint32_t)starts[2], (uint32_t)starts[3],
        (uint32_t)steps[0], (uint32_t)steps[1], (uint32_t)steps[2], (uint32_t)steps[3],
        (uint32_t)out_elems
    };
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_slice, bufs, ((uint32_t)out_elems + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_conv_transpose2d_f32(const float* in, const float* weight, const float* bias,
                                      float* out, int n, int in_h, int in_w, int in_c,
                                      int out_h, int out_w, int out_c, int kh, int kw,
                                      int sh, int sw, int ph, int pw) {
    if (n <= 0 || in_h <= 0 || in_w <= 0 || in_c <= 0 || out_h <= 0 || out_w <= 0 ||
        out_c <= 0 || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || !in || !weight || !out) return 0;
    size_t in_bytes = (size_t)n * in_h * in_w * in_c * sizeof(float);
    size_t wbytes = (size_t)in_c * out_c * kh * kw * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)n * out_h * out_w * out_c * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* swt = graph_ensure_device(weight, wbytes, 1);
    static const float zero_bias[1] = {0.0f};
    OglTensorSlot* sb = graph_ensure_device(bias ? bias : zero_bias, bias ? bbytes : sizeof(float), 1);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !swt || !sb || !dst) return 0;
    uint32_t params[16] = {(uint32_t)n, (uint32_t)in_h, (uint32_t)in_w, (uint32_t)in_c,
                           (uint32_t)out_h, (uint32_t)out_w, (uint32_t)out_c,
                           (uint32_t)kh, (uint32_t)kw, (uint32_t)sh, (uint32_t)sw,
                           (uint32_t)ph, (uint32_t)pw, bias ? 1u : 0u, 0u, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, swt->buffer, sb->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_convtranspose, bufs, ((uint32_t)out_w + 7u) / 8u,
                             ((uint32_t)out_h + 7u) / 8u, (uint32_t)(n * out_c));
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_interp1d_f32(const float* in, float* out, int channels, int in_l, int out_l) {
    if (channels <= 0 || in_l <= 0 || out_l <= 0 || !in || !out) return 0;
    size_t in_bytes = (size_t)channels * in_l * sizeof(float);
    size_t out_bytes = (size_t)channels * out_l * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[3] = {(uint32_t)channels, (uint32_t)in_l, (uint32_t)out_l};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_interp1d, bufs, ((uint32_t)out_l + 63u) / 64u, (uint32_t)channels, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
                            float* out, long out_numel, const uint32_t* output_strides,
                            const uint32_t* a_strides, const uint32_t* b_strides,
                            int rank, int op) {
    if (!a || !b || !out || !output_strides || !a_strides || !b_strides ||
        a_numel <= 0 || b_numel <= 0 || out_numel <= 0 ||
        (uint64_t)out_numel > UINT32_MAX || rank <= 0 || rank > 8 || op < 0 || op > 3)
        return 0;
    size_t a_bytes = (size_t)a_numel * sizeof(float);
    size_t b_bytes = (size_t)b_numel * sizeof(float);
    size_t out_bytes = (size_t)out_numel * sizeof(float);
    OglTensorSlot* sa = graph_ensure_device(a, a_bytes, 0);
    OglTensorSlot* sb = graph_ensure_device(b, b_bytes, 0);
    OglTensorSlot* so = graph_output_slot(out, out_bytes);
    if (!sa || !sb || !so) return 0;
    uint32_t metadata[28] = {(uint32_t)out_numel, (uint32_t)rank, (uint32_t)op, 0u};
    memcpy(metadata + 4, output_strides, 8 * sizeof(uint32_t));
    memcpy(metadata + 12, a_strides, 8 * sizeof(uint32_t));
    memcpy(metadata + 20, b_strides, 8 * sizeof(uint32_t));
    GLuint pb = params_buffer(metadata, sizeof(metadata));
    GLuint bufs[4] = {sa->buffer, sb->buffer, so->buffer, pb};
    int ok = dispatch_kernel(&k_broadcast_binary, bufs, ((uint32_t)out_numel + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(so);
    return 1;
}

int opengl_graph_split_f32(const float* in, float* out, long input_numel, long output_numel,
                           int inner, int split_size, int axis_in, int offset) {
    if (!in || !out || input_numel <= 0 || output_numel <= 0 ||
        inner <= 0 || split_size <= 0 || axis_in <= 0 || offset < 0 || offset + split_size > axis_in) return 0;
    size_t in_bytes = (size_t)input_numel * sizeof(float);
    size_t out_bytes = (size_t)output_numel * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[5] = {(uint32_t)output_numel, (uint32_t)inner, (uint32_t)split_size,
                          (uint32_t)axis_in, (uint32_t)offset};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[3] = {src->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_split, bufs, ((uint32_t)output_numel + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_conv1d_f32(const float* in, const float* weight, const float* bias, float* out,
                            int batch, int in_c, int in_l, int out_c, int out_l, int kernel,
                            int stride, int pad, int relu) {
    if (!in || !weight || !out || batch <= 0 || in_c <= 0 || in_l <= 0 || out_c <= 0 || out_l <= 0 ||
        kernel <= 0 || stride <= 0 || pad < 0 || relu < 0 || relu > 1) return 0;
    int64_t padded_l = (int64_t)in_l + 2LL * pad;
    if (padded_l < kernel || (padded_l - kernel) / stride + 1 != out_l) return 0;
    size_t in_bytes = (size_t)batch * (size_t)in_c * in_l * sizeof(float);
    size_t wbytes = (size_t)out_c * in_c * kernel * sizeof(float);
    size_t bbytes = (size_t)out_c * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)out_c * out_l * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sw || !dst) return 0;
    GLuint bias_buf = 0;
    int delete_bias = 0;
    if (bias) {
        OglTensorSlot* sb = graph_ensure_device(bias, bbytes, 1);
        if (!sb) return 0;
        bias_buf = sb->buffer;
    } else {
        float* zeros = (float*)calloc((size_t)out_c, sizeof(float));
        if (!zeros) return 0;
        bias_buf = create_buffer(bbytes, zeros);
        free(zeros);
        delete_bias = 1;
        if (!bias_buf) return 0;
    }
    uint32_t params[12] = {(uint32_t)in_c, (uint32_t)in_l, (uint32_t)out_c,
                           (uint32_t)kernel, (uint32_t)stride, (uint32_t)pad,
                           (uint32_t)relu, (uint32_t)batch, 1u, (uint32_t)in_c,
                           (uint32_t)out_l, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, sw->buffer, bias_buf, dst->buffer, pb};
    int ok = dispatch_kernel(&k_conv1d, bufs, ((uint32_t)out_l + 63u) / 64u,
                             (uint32_t)out_c, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (delete_bias && bias_buf) p_glDeleteBuffers(1, &bias_buf);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

static long attention_mask_numel(int mask_mode, int batch, int seq_q, int seq_kv) {
    if (mask_mode == 0) return 0;
    if (mask_mode == 1) return seq_kv;
    if (mask_mode == 2) return (long)batch * seq_kv;
    if (mask_mode == 3) return (long)seq_q * seq_kv;
    if (mask_mode == 4) return (long)batch * seq_q * seq_kv;
    return -1;
}

int opengl_graph_sdpa_f32(const float* qkv, const int32_t* mask, long mask_numel,
                          float* out, int seq_len, int d_model, int num_heads,
                          int head_dim, int batch, float scale, int causal, int mask_mode) {
    if (!ogl_ready() || !qkv || !out || batch <= 0 || seq_len <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 ||
        head_dim > 64 || d_model != num_heads * head_dim) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_len, seq_len);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t qkv_bytes = (size_t)batch * (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)seq_len * (size_t)d_model * sizeof(float);
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : qkv_bytes;
    OglTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    OglTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : src;
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sm || !dst) return 0;
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode;
    } params = {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[4] = {src->buffer, sm->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_sdpa, bufs, ((uint32_t)seq_len + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int opengl_graph_sdpa_training_f32(const float* qkv, const int32_t* mask, long mask_numel,
                                   float* out, int seq_len, int d_model, int num_heads,
                                   int head_dim, int batch, float scale, int causal, int mask_mode,
                                   uint32_t threshold, uint32_t seed, uint32_t counter,
                                   float dropout_scale) {
    if (!ogl_ready() || !qkv || !out || batch <= 0 || seq_len <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        !isfinite(dropout_scale) || dropout_scale < 1.0f) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_len, seq_len);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t qkv_bytes = (size_t)batch * (size_t)seq_len * 3u * (size_t)d_model * sizeof(float);
    size_t out_bytes = (size_t)batch * (size_t)seq_len * (size_t)d_model * sizeof(float);
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : qkv_bytes;
    OglTensorSlot* src = graph_ensure_device(qkv, qkv_bytes, 0);
    OglTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : src;
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !sm || !dst) return 0;
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
    } params = {(uint32_t)seq_len, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, (uint32_t)batch, scale, causal ? 1u : 0u,
                (uint32_t)mask_mode, threshold, seed, counter, dropout_scale};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[4] = {src->buffer, sm->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_sdpa_training, bufs, ((uint32_t)seq_len + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int opengl_graph_cross_sdpa_f32(const float* q, const float* k, const float* v,
                                const int32_t* mask, long mask_numel, float* out,
                                int seq_q, int seq_kv, int d_model, int num_heads,
                                int head_dim, int batch, float scale, int causal, int mask_mode) {
    if (!ogl_ready() || !q || !k || !v || !out || batch <= 0 || seq_q <= 0 ||
        seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 64 || d_model != num_heads * head_dim) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_q, seq_kv);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : q_bytes;
    OglTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    OglTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    OglTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    OglTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : sq;
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !sm || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model,
                (uint32_t)num_heads, (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode, {0u, 0u, 0u}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[6] = {sq->buffer, sk->buffer, sv->buffer, sm->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_cross_sdpa, bufs, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
int opengl_graph_cross_sdpa_training_f32(const float* q, const float* k, const float* v,
                                         const int32_t* mask, long mask_numel, float* out,
                                         int seq_q, int seq_kv, int d_model, int num_heads,
                                         int head_dim, int batch, float scale, int causal, int mask_mode,
                                         uint32_t threshold, uint32_t seed, uint32_t counter,
                                         float dropout_scale) {
    if (!ogl_ready() || !q || !k || !v || !out || batch <= 0 || seq_q <= 0 ||
        seq_kv <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        head_dim > 64 || d_model != num_heads * head_dim ||
        !isfinite(dropout_scale) || dropout_scale < 1.0f) return 0;
    long expected_mask = attention_mask_numel(mask_mode, batch, seq_q, seq_kv);
    if (expected_mask < 0 || (mask_mode != 0 && (!mask || mask_numel != expected_mask))) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t mask_bytes = mask_mode ? (size_t)expected_mask * sizeof(int32_t) : q_bytes;
    OglTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    OglTensorSlot* sk = graph_ensure_device(k, kv_bytes, 0);
    OglTensorSlot* sv = graph_ensure_device(v, kv_bytes, 0);
    OglTensorSlot* sm = mask_mode ? graph_ensure_device(mask, mask_bytes, 0) : sq;
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !sk || !sv || !sm || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
        uint32_t pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model,
                (uint32_t)num_heads, (uint32_t)head_dim, (uint32_t)batch, scale,
                causal ? 1u : 0u, (uint32_t)mask_mode, threshold, seed, counter,
                dropout_scale, {0u, 0u, 0u}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[6] = {sq->buffer, sk->buffer, sv->buffer, sm->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_cross_sdpa_training, bufs, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
#endif

int opengl_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                     const float* scale, const float* bias, float* out,
                                     int seq_q, int seq_kv, int d_model, int num_heads,
                                     int head_dim, int batch, int has_scale, int has_bias) {
    if (!q || !kv || !weight || !out || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 ||
        num_heads <= 0 || head_dim <= 0 || batch <= 0 || head_dim > 64 || d_model != num_heads * head_dim ||
        d_model > 64) return 0;
    size_t q_bytes = (size_t)batch * (size_t)seq_q * (size_t)d_model * sizeof(float);
    size_t kv_bytes = (size_t)batch * (size_t)seq_kv * (size_t)d_model * sizeof(float);
    size_t wbytes = (size_t)3 * (size_t)d_model * (size_t)d_model * sizeof(float);
    size_t sb_bytes = (size_t)3 * (size_t)d_model * sizeof(float);
    size_t out_bytes = q_bytes;
    static const float zero[1] = {0.0f};
    OglTensorSlot* sq = graph_ensure_device(q, q_bytes, 0);
    OglTensorSlot* skv = graph_ensure_device(kv, kv_bytes, 0);
    OglTensorSlot* sw = graph_ensure_device(weight, wbytes, 1);
    OglTensorSlot* ss = graph_ensure_device(has_scale && scale ? scale : zero, has_scale && scale ? sb_bytes : sizeof(float), 1);
    OglTensorSlot* sb = graph_ensure_device(has_bias && bias ? bias : zero, has_bias && bias ? sb_bytes : sizeof(float), 1);
    OglTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!sq || !skv || !sw || !ss || !sb || !dst) return 0;
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim;
        float scale_factor;
        uint32_t has_scale, has_bias, batch, pad[3];
    } params = {(uint32_t)seq_q, (uint32_t)seq_kv, (uint32_t)d_model, (uint32_t)num_heads,
                (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim),
                (uint32_t)(has_scale && scale), (uint32_t)(has_bias && bias), (uint32_t)batch, {0u, 0u, 0u}};
    GLuint pb = params_buffer(&params, sizeof(params));
    GLuint bufs[7] = {sq->buffer, skv->buffer, sw->buffer, ss->buffer, sb->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_cross_attention, bufs, ((uint32_t)seq_q + 63u) / 64u,
                             (uint32_t)num_heads, (uint32_t)batch);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

int opengl_graph_dequantize_linear_f32(const float* in, const float* scale, const float* zero_point,
                                       float* out, long n, int has_zero_point) {
    if (!in || !scale || !out || n <= 0) return 0;
    static const float zero[1] = {0.0f};
    size_t bytes = (size_t)n * sizeof(float);
    OglTensorSlot* src = graph_ensure_device(in, bytes, 0);
    OglTensorSlot* ss = graph_ensure_device(scale, sizeof(float), 1);
    OglTensorSlot* sz = graph_ensure_device(has_zero_point && zero_point ? zero_point : zero, sizeof(float), 1);
    OglTensorSlot* dst = graph_output_slot(out, bytes);
    if (!src || !ss || !sz || !dst) return 0;
    uint32_t params[4] = {(uint32_t)n, (uint32_t)(has_zero_point && zero_point), 0u, 0u};
    GLuint pb = params_buffer(params, sizeof(params));
    GLuint bufs[5] = {src->buffer, ss->buffer, sz->buffer, dst->buffer, pb};
    int ok = dispatch_kernel(&k_dequantize, bufs, ((uint32_t)n + 63u) / 64u, 1, 1);
    if (pb) p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}

typedef struct {
    uint32_t rows;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t input_type;
    uint32_t weight_type;
    uint32_t output_type;
    uint32_t pad0;
    uint32_t pad1;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad2;
    int32_t pad3;
    float input_scale;
    float output_scale;
    float pad4;
    float pad5;
} OglQLinearParams;

_Static_assert(sizeof(OglQLinearParams) == 64, "qLinearInt8 uniform ABI");

static int qlinear_dtype_bounds(uint32_t dtype, int32_t zero_point,
                                int64_t* maximum_distance) {
    int32_t minimum;
    int32_t maximum;
    if (dtype == VX_DTYPE_I8) {
        minimum = -128;
        maximum = 127;
    } else if (dtype == VX_DTYPE_U8) {
        minimum = 0;
        maximum = 255;
    } else {
        return 0;
    }
    if (zero_point < minimum || zero_point > maximum) return 0;
    int64_t lower = (int64_t)minimum - zero_point;
    int64_t upper = (int64_t)maximum - zero_point;
    *maximum_distance = llabs(lower) > llabs(upper) ? llabs(lower) : llabs(upper);
    return 1;
}

static int qlinear_gpu_args_valid(const void* input, const void* weight,
                                  const float* weight_scales,
                                  const int32_t* weight_zero_points,
                                  const int32_t* bias, void* output,
                                  uint32_t rows, uint32_t d_in, uint32_t d_out,
                                  float input_scale, int32_t input_zero_point,
                                  float output_scale, int32_t output_zero_point,
                                  uint32_t input_dtype, uint32_t weight_dtype,
                                  uint32_t output_dtype,
                                  size_t* input_bytes, size_t* weight_bytes,
                                  size_t* output_bytes) {
    if (!input || !weight || !weight_scales || !weight_zero_points || !bias || !output ||
        rows == 0 || d_in == 0 || d_out == 0 || !isfinite(input_scale) ||
        !isfinite(output_scale) || input_scale <= 0.0f || output_scale <= 0.0f) return 0;
    int64_t input_distance = 0;
    int64_t output_distance = 0;
    if (!qlinear_dtype_bounds(input_dtype, input_zero_point, &input_distance) ||
        !qlinear_dtype_bounds(output_dtype, output_zero_point, &output_distance)) return 0;
    (void)output_distance;
    if (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) return 0;
    if (rows > UINT32_MAX / d_in || rows > UINT32_MAX / d_out ||
        d_out > UINT32_MAX / d_in) return 0;
    uint64_t input_elements = (uint64_t)rows * d_in;
    uint64_t weight_elements = (uint64_t)d_out * d_in;
    uint64_t output_elements = (uint64_t)rows * d_out;
    if (input_elements > SIZE_MAX || weight_elements > SIZE_MAX ||
        output_elements > SIZE_MAX ||
        (size_t)d_out > SIZE_MAX / sizeof(*weight_scales) ||
        (size_t)d_out > SIZE_MAX / sizeof(*weight_zero_points) ||
        (size_t)d_out > SIZE_MAX / sizeof(*bias)) return 0;
    for (uint32_t channel = 0; channel < d_out; channel++) {
        int64_t weight_distance = 0;
        if (!isfinite(weight_scales[channel]) || weight_scales[channel] <= 0.0f ||
            !qlinear_dtype_bounds(weight_dtype, weight_zero_points[channel],
                                  &weight_distance)) return 0;
        int64_t bias_distance = bias[channel] < 0 ? -(int64_t)bias[channel] : bias[channel];
        if (bias_distance > INT32_MAX) return 0;
        int64_t term_distance = input_distance * weight_distance;
        if (term_distance > 0 &&
            (uint64_t)d_in > (uint64_t)(INT32_MAX - bias_distance) /
                                  (uint64_t)term_distance) return 0;
    }
    *input_bytes = (size_t)input_elements;
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    return 1;
}

int opengl_graph_qlinear_i8u8(const void* input, const void* weight,
                              const float* weight_scales, const int32_t* weight_zero_points,
                              const int32_t* bias, void* output,
                              uint32_t rows, uint32_t d_in, uint32_t d_out,
                              float input_scale, int32_t input_zero_point,
                              float output_scale, int32_t output_zero_point,
                              uint32_t input_dtype, uint32_t weight_dtype,
                              uint32_t output_dtype) {
    size_t input_bytes, weight_bytes, output_bytes;
    size_t multiplier_bytes;
    float* multipliers;
    if (max_ssbo_bindings < 6 || max_uniform_bindings < 1) return 0;
    if (!qlinear_gpu_args_valid(input, weight, weight_scales, weight_zero_points, bias, output,
                                rows, d_in, d_out, input_scale, input_zero_point,
                                output_scale, output_zero_point, input_dtype, weight_dtype,
                                output_dtype, &input_bytes, &weight_bytes, &output_bytes)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    multiplier_bytes = (size_t)d_out * sizeof(*multipliers);
    multipliers = (float*)malloc(multiplier_bytes);
    if (!multipliers ||
        !vx_qlinear_build_multipliers(
            input_scale, weight_scales, output_scale, d_out, multipliers)) {
        free(multipliers);
        return 0;
    }
    /* Activations may be a row window; weights never are. */
    OglGraphWindow src, dst;
    int resolved = graph_window_packed_bytes(input, input_bytes, 0, &src) &&
        graph_window_output_packed(output, output_bytes, &dst);
    OglTensorSlot* wt = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    OglTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                      (size_t)d_out * sizeof(*weight_zero_points), 1);
    OglTensorSlot* biases = graph_ensure_device(bias, (size_t)d_out * sizeof(*bias), 1);
    if (!resolved || !wt || !zero_points || !biases) {
        free(multipliers);
        return 0;
    }
    GLuint multiplier_buffer = create_buffer(multiplier_bytes, multipliers);
    free(multipliers);
    if (!multiplier_buffer) return 0;
    OglQLinearParams params = {
        rows, d_in, d_out,
        input_dtype,
        weight_dtype,
        output_dtype, 0u, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    GLuint pb = params_buffer(&params, sizeof(params));
    if (!pb) {
        p_glDeleteBuffers(1, &multiplier_buffer);
        return 0;
    }
    OglBufferRange bufs[7] = {
        {src.buffer, src.offset, src.bytes}, {wt->buffer, 0, 0},
        {multiplier_buffer, 0, 0}, {zero_points->buffer, 0, 0},
        {biases->buffer, 0, 0}, {dst.buffer, dst.offset, dst.bytes},
        {pb, 0, 0}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int tiled = rows > 1u && d_in >= 16u && d_out >= 32u && (d_out & 3u) == 0u &&
                opengl_workgroup_supported(8u, 8u, 1u);
    uint32_t groups_x = tiled ? ((d_out / 4u + 7u) / 8u) : ((packed_words + 63u) / 64u);
    uint32_t groups_y = tiled ? ((rows + 7u) / 8u) : 1u;
    const OglKernel* kernel = tiled ? &k_qlinear_int8_tiled : &k_qlinear_int8;
    int ok = dispatch_kernel_ranges(kernel, bufs, groups_x, groups_y, 1u);
    p_glDeleteBuffers(1, &pb);
    p_glDeleteBuffers(1, &multiplier_buffer);
    if (!ok) return 0;
    graph_mark_device(dst.slot);
    return 1;
}

typedef struct {
    uint32_t tokens;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t weight_type;
    uint32_t output_type;
    int32_t output_zero_point;
    float output_scale;
    /* The first token this dispatch reads. See graph_token_window. */
    uint32_t token_offset;
} OglQEmbeddingParams;

_Static_assert(sizeof(OglQEmbeddingParams) == 32, "qEmbeddingInt8 uniform ABI");
