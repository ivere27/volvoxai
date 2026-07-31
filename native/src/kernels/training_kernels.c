#include "training_kernels.h"
#include "gemm_f32.h"
#include "mathcompat.h"

#include <stddef.h>

#if defined(__wasm__)
#define VX_TRAINING_EXPORT(name) __attribute__((export_name(name)))
#else
#define VX_TRAINING_EXPORT(name)
#endif

static uint32_t vx_finite_f32(float value) {
    union { float f; uint32_t u; } bits = { value };
    return (bits.u & 0x7f800000u) != 0x7f800000u;
}

static uint32_t vx_product_fits(uint32_t a, uint32_t b, size_t *result) {
    size_t maximum = (size_t)-1;
    if (a != 0 && (size_t)b > maximum / (size_t)a) return 0;
    *result = (size_t)a * (size_t)b;
    return 1;
}

static float vx_erf_approx(float value) {
    const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
    const float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
    float magnitude = value < 0.0f ? -value : value;
    float t = 1.0f / (1.0f + p * magnitude);
    float polynomial = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    float result = 1.0f - polynomial * expf(-magnitude * magnitude);
    return value < 0.0f ? -result : result;
}

static float vx_tanh_f32(float value) {
    if (value > 10.0f) return 1.0f;
    if (value < -10.0f) return -1.0f;
    float exponential = expf(2.0f * value);
    return (exponential - 1.0f) / (exponential + 1.0f);
}

static float vx_sigmoid_f32(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    float exponential = expf(value);
    return exponential / (1.0f + exponential);
}

/* Matches the integer mixer in ts/ops/dropout.ts and dropout.wgsl. */
static uint32_t vx_dropout_bits(uint32_t index, uint32_t seed, uint32_t counter) {
    uint32_t value = index ^ seed ^ (counter * 0x9e3779b9u);
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

VX_TRAINING_EXPORT("volvoxai_training_abi_version")
uint32_t volvoxai_training_abi_version(void) {
    return VOLVOXAI_TRAINING_ABI_VERSION;
}

VX_TRAINING_EXPORT("volvoxai_training_capabilities")
uint32_t volvoxai_training_capabilities(void) {
    return VOLVOXAI_TRAINING_CAP_ALL;
}

VX_TRAINING_EXPORT("volvoxai_training_zero_f32")
void volvoxai_training_zero_f32(float *dst, uint32_t n) {
    if (!dst) return;
    for (uint32_t i = 0; i < n; i++) dst[i] = 0.0f;
}

VX_TRAINING_EXPORT("volvoxai_training_add_f32")
void volvoxai_training_add_f32(float *dst, const float *src, uint32_t n) {
    if (!dst || !src) return;
    for (uint32_t i = 0; i < n; i++) dst[i] += src[i];
}

VX_TRAINING_EXPORT("volvoxai_training_all_finite_f32")
uint32_t volvoxai_training_all_finite_f32(const float *values, uint32_t n) {
    if (!values && n) return 0;
    for (uint32_t i = 0; i < n; i++) if (!vx_finite_f32(values[i])) return 0;
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_sum_squares_f32")
float volvoxai_training_sum_squares_f32(const float *values, uint32_t n) {
    if (!values && n) return 0.0f;
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += (double)values[i] * values[i];
    return (float)sum;
}

VX_TRAINING_EXPORT("volvoxai_training_scale_f32")
void volvoxai_training_scale_f32(float *values, uint32_t n, float scale) {
    if (!values) return;
    for (uint32_t i = 0; i < n; i++) values[i] *= scale;
}

VX_TRAINING_EXPORT("volvoxai_training_cross_entropy_f32")
uint32_t volvoxai_training_cross_entropy_f32(
        const float *logits, const int32_t *rows, const int32_t *targets, float *grad,
        uint32_t examples, uint32_t classes, float gradient_scale,
        float *loss_sum, uint32_t *correct) {
    if (examples == 0) return 1;
    if (!logits || !rows || !targets || !grad || classes == 0 ||
        !vx_finite_f32(gradient_scale)) return 0;
    for (uint32_t e = 0; e < examples; e++) {
        if (rows[e] < 0 || targets[e] < 0 || (uint32_t)targets[e] >= classes) return 0;
        size_t base = (size_t)(uint32_t)rows[e] * classes;
        for (uint32_t c = 0; c < classes; c++)
            if (!vx_finite_f32(logits[base + c])) return 0;
    }
    float local_loss = 0.0f;
    uint32_t local_correct = 0;
    for (uint32_t e = 0; e < examples; e++) {
        size_t base = (size_t)(uint32_t)rows[e] * classes;
        float maximum = logits[base];
        uint32_t prediction = 0;
        for (uint32_t c = 1; c < classes; c++) {
            if (logits[base + c] > maximum) {
                maximum = logits[base + c];
                prediction = c;
            }
        }
        double denominator = 0.0;
        for (uint32_t c = 0; c < classes; c++)
            denominator += expf(logits[base + c] - maximum);
        if (!(denominator > 0.0)) return 0;
        uint32_t target = (uint32_t)targets[e];
        local_loss += maximum + logf((float)denominator) - logits[base + target];
        if (prediction == target) local_correct++;
        for (uint32_t c = 0; c < classes; c++) {
            float probability = (float)(expf(logits[base + c] - maximum) / denominator);
            grad[base + c] += gradient_scale * (probability - (c == target ? 1.0f : 0.0f));
        }
    }
    if (loss_sum) *loss_sum += local_loss;
    if (correct) *correct += local_correct;
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_linear_backward_f32")
void volvoxai_training_linear_backward_f32(
        const float *x, const float *w, const float *dy,
        float *dx, float *dw, float *db,
        uint32_t m, uint32_t k, uint32_t n, uint32_t layout) {
    size_t mk, kn, mn;
    if (!x || !w || !dy || (!dx && !dw && !db) ||
        layout > VOLVOXAI_TRAINING_LINEAR_OUT_IN ||
        !vx_product_fits(m, k, &mk) || !vx_product_fits(k, n, &kn) ||
        !vx_product_fits(m, n, &mn)) return;
    (void)mk; (void)kn; (void)mn;
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t col = 0; col < n; col++) {
            float gradient = dy[(size_t)row * n + col];
            if (db) db[col] += gradient;
            for (uint32_t inner = 0; inner < k; inner++) {
                size_t x_index = (size_t)row * k + inner;
                size_t w_index = layout == VOLVOXAI_TRAINING_LINEAR_OUT_IN
                    ? (size_t)col * k + inner : (size_t)inner * n + col;
                if (dx) dx[x_index] += gradient * w[w_index];
                if (dw) dw[w_index] += x[x_index] * gradient;
            }
        }
    }
}

VX_TRAINING_EXPORT("volvoxai_training_linear_backward_packed_f32")
uint32_t volvoxai_training_linear_backward_packed_f32(
        const float *x, const float *w, const float *dy,
        float *dx, float *dw, float *db, float *packed_weight,
        uint32_t packed_elements,
        uint32_t m, uint32_t k, uint32_t n, uint32_t layout) {
    size_t mk, kn, mn;
    uint32_t required = dx ? vx_gemm_f32_packed_elements(n, k) : 0u;
    if (!x || !w || !dy || (!dx && !dw && !db) ||
        layout > VOLVOXAI_TRAINING_LINEAR_OUT_IN ||
        !vx_product_fits(m, k, &mk) || !vx_product_fits(k, n, &kn) ||
        !vx_product_fits(m, n, &mn) ||
        (dx && (!required || !packed_weight || packed_elements < required))) return 0;
    (void)mk; (void)kn; (void)mn;
    if (dx) {
        /* dx = dY * W^T; the physical layout is reversed relative to the
         * forward KxN interpretation. */
        int dx_out_in = layout == VOLVOXAI_TRAINING_LINEAR_IN_OUT ? 1 : 0;
        if (!vx_gemm_f32_pack_b(w, packed_weight, n, k, dx_out_in) ||
            !vx_gemm_f32_run_packed_add(dy, packed_weight, NULL, dx, m, n, k))
            return 0;
    }
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t col = 0; col < n; col++) {
            float gradient = dy[(size_t)row * n + col];
            if (db) db[col] += gradient;
            for (uint32_t inner = 0; dw && inner < k; inner++) {
                size_t x_index = (size_t)row * k + inner;
                size_t w_index = layout == VOLVOXAI_TRAINING_LINEAR_OUT_IN
                    ? (size_t)col * k + inner : (size_t)inner * n + col;
                dw[w_index] += x[x_index] * gradient;
            }
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_embedding_backward_f32")
uint32_t volvoxai_training_embedding_backward_f32(
        const int32_t *ids, const float *dy, float *dw,
        uint32_t tokens, uint32_t vocab, uint32_t dim) {
    if (tokens == 0) return 1;
    if (!ids || !dy || !dw || vocab == 0 || dim == 0) return 0;
    for (uint32_t token = 0; token < tokens; token++)
        if (ids[token] < 0 || (uint32_t)ids[token] >= vocab) return 0;
    for (uint32_t token = 0; token < tokens; token++) {
        size_t source = (size_t)token * dim;
        size_t destination = (size_t)(uint32_t)ids[token] * dim;
        for (uint32_t column = 0; column < dim; column++)
            dw[destination + column] += dy[source + column];
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_layernorm_backward_f32")
void volvoxai_training_layernorm_backward_f32(
        const float *x, const float *weight, const float *dy,
        float *dx, float *dweight, float *dbias,
        uint32_t rows, uint32_t width, float epsilon) {
    if (!x || !weight || !dy || (!dx && !dweight && !dbias) || width == 0 ||
        !(epsilon > 0.0f) || !vx_finite_f32(epsilon)) return;
    for (uint32_t row = 0; row < rows; row++) {
        size_t offset = (size_t)row * width;
        double mean = 0.0;
        for (uint32_t column = 0; column < width; column++) mean += x[offset + column];
        mean /= width;
        double variance = 0.0;
        for (uint32_t column = 0; column < width; column++) {
            double centered = x[offset + column] - mean;
            variance += centered * centered;
        }
        float inverse = 1.0f / __builtin_sqrtf((float)(variance / width) + epsilon);
        double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
        for (uint32_t column = 0; column < width; column++) {
            float xhat = (x[offset + column] - (float)mean) * inverse;
            float dxhat = dy[offset + column] * weight[column];
            if (dweight) dweight[column] += dy[offset + column] * xhat;
            if (dbias) dbias[column] += dy[offset + column];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += (double)dxhat * xhat;
        }
        if (dx) for (uint32_t column = 0; column < width; column++) {
            float xhat = (x[offset + column] - (float)mean) * inverse;
            float dxhat = dy[offset + column] * weight[column];
            dx[offset + column] += inverse *
                (dxhat - (float)(sum_dxhat / width) - xhat * (float)(sum_dxhat_xhat / width));
        }
    }
}

VX_TRAINING_EXPORT("volvoxai_training_rmsnorm_backward_f32")
void volvoxai_training_rmsnorm_backward_f32(
        const float *x, const float *weight, const float *dy,
        float *dx, float *dweight,
        uint32_t rows, uint32_t width, float epsilon) {
    if (!x || !weight || !dy || (!dx && !dweight) || width == 0 ||
        !(epsilon > 0.0f) || !vx_finite_f32(epsilon)) return;
    for (uint32_t row = 0; row < rows; row++) {
        size_t offset = (size_t)row * width;
        double mean_square = 0.0;
        for (uint32_t column = 0; column < width; column++)
            mean_square += (double)x[offset + column] * x[offset + column];
        float inverse = 1.0f / __builtin_sqrtf((float)(mean_square / width) + epsilon);
        double projection = 0.0;
        for (uint32_t column = 0; column < width; column++) {
            float value = x[offset + column];
            float scaled_gradient = dy[offset + column] * weight[column];
            projection += (double)scaled_gradient * value;
            if (dweight) dweight[column] += dy[offset + column] * value * inverse;
        }
        if (dx) {
            float correction = (float)(projection / width) * inverse * inverse * inverse;
            for (uint32_t column = 0; column < width; column++)
                dx[offset + column] += dy[offset + column] * weight[column] * inverse -
                                       x[offset + column] * correction;
        }
    }
}

VX_TRAINING_EXPORT("volvoxai_training_groupnorm_f32")
uint32_t volvoxai_training_groupnorm_f32(
        const float *input, const float *weight, const float *bias, float *output,
        uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
        uint32_t groups, float epsilon) {
    if (!input || !weight || !bias || !output || !batch || !height || !width || !channels ||
        !groups || channels % groups || !(epsilon > 0.0f) || !vx_finite_f32(epsilon)) return 0;
    uint32_t spatial = height * width, per_group_channels = channels / groups;
    uint32_t values = spatial * per_group_channels, sample_stride = spatial * channels;
    for (uint32_t sample = 0; sample < batch; sample++) for (uint32_t group = 0; group < groups; group++) {
        uint32_t first_channel = group * per_group_channels;
        size_t sample_offset = (size_t)sample * sample_stride;
        double mean = 0.0, variance = 0.0;
        for (uint32_t point = 0; point < spatial; point++) for (uint32_t local = 0; local < per_group_channels; local++)
            mean += input[sample_offset + (size_t)point * channels + first_channel + local];
        mean /= values;
        for (uint32_t point = 0; point < spatial; point++) for (uint32_t local = 0; local < per_group_channels; local++) {
            float centered = input[sample_offset + (size_t)point * channels + first_channel + local] - (float)mean;
            variance += (double)centered * centered;
        }
        float inverse = 1.0f / __builtin_sqrtf((float)(variance / values) + epsilon);
        for (uint32_t point = 0; point < spatial; point++) for (uint32_t local = 0; local < per_group_channels; local++) {
            uint32_t channel = first_channel + local;
            size_t index = sample_offset + (size_t)point * channels + channel;
            output[index] = (input[index] - (float)mean) * inverse * weight[channel] + bias[channel];
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_groupnorm_backward_f32")
uint32_t volvoxai_training_groupnorm_backward_f32(
        const float *input, const float *weight, const float *dy,
        float *dx, float *dweight, float *dbias,
        uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
        uint32_t groups, float epsilon) {
    if (!input || !weight || !dy || !dx || !dweight || !dbias || !batch || !height || !width || !channels ||
        !groups || channels % groups || !(epsilon > 0.0f) || !vx_finite_f32(epsilon)) return 0;
    uint32_t spatial = height * width, per_group_channels = channels / groups;
    uint32_t values = spatial * per_group_channels, sample_stride = spatial * channels;
    for (uint32_t sample = 0; sample < batch; sample++) for (uint32_t group = 0; group < groups; group++) {
        uint32_t first_channel = group * per_group_channels;
        size_t sample_offset = (size_t)sample * sample_stride;
        double mean = 0.0, variance = 0.0, sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
        for (uint32_t point = 0; point < spatial; point++) for (uint32_t local = 0; local < per_group_channels; local++)
            mean += input[sample_offset + (size_t)point * channels + first_channel + local];
        mean /= values;
        for (uint32_t point = 0; point < spatial; point++) for (uint32_t local = 0; local < per_group_channels; local++) {
            float centered = input[sample_offset + (size_t)point * channels + first_channel + local] - (float)mean;
            variance += (double)centered * centered;
        }
        float inverse = 1.0f / __builtin_sqrtf((float)(variance / values) + epsilon);
        for (uint32_t point = 0; point < spatial; point++) for (uint32_t local = 0; local < per_group_channels; local++) {
            uint32_t channel = first_channel + local;
            size_t index = sample_offset + (size_t)point * channels + channel;
            float xhat = (input[index] - (float)mean) * inverse;
            float dxhat = dy[index] * weight[channel];
            dweight[channel] += dy[index] * xhat; dbias[channel] += dy[index];
            sum_dxhat += dxhat; sum_dxhat_xhat += (double)dxhat * xhat;
        }
        for (uint32_t point = 0; point < spatial; point++) for (uint32_t local = 0; local < per_group_channels; local++) {
            uint32_t channel = first_channel + local;
            size_t index = sample_offset + (size_t)point * channels + channel;
            float xhat = (input[index] - (float)mean) * inverse;
            float dxhat = dy[index] * weight[channel];
            dx[index] += inverse * (dxhat - (float)(sum_dxhat / values) - xhat * (float)(sum_dxhat_xhat / values));
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_softmax_f32")
uint32_t volvoxai_training_softmax_f32(
        const float *input, float *output, uint32_t rows, uint32_t width, uint32_t log_output) {
    if (!input || !output || !width || log_output > 1) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        size_t offset = (size_t)row * width;
        float maximum = input[offset];
        for (uint32_t column = 1; column < width; column++) if (input[offset + column] > maximum) maximum = input[offset + column];
        double sum = 0.0;
        for (uint32_t column = 0; column < width; column++) sum += expf(input[offset + column] - maximum);
        if (!(sum > 0.0)) return 0;
        float log_sum = logf((float)sum);
        for (uint32_t column = 0; column < width; column++)
            output[offset + column] = log_output ? input[offset + column] - maximum - log_sum :
                expf(input[offset + column] - maximum) / (float)sum;
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_softmax_backward_f32")
uint32_t volvoxai_training_softmax_backward_f32(
        const float *output, const float *dy, float *dx,
        uint32_t rows, uint32_t width, uint32_t log_output) {
    if (!output || !dy || !dx || !width || log_output > 1) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        size_t offset = (size_t)row * width;
        float sum = 0.0f;
        if (log_output) for (uint32_t column = 0; column < width; column++) sum += dy[offset + column];
        else for (uint32_t column = 0; column < width; column++) sum += dy[offset + column] * output[offset + column];
        for (uint32_t column = 0; column < width; column++) {
            float probability = log_output ? expf(output[offset + column]) : output[offset + column];
            dx[offset + column] += log_output ? dy[offset + column] - probability * sum : probability * (dy[offset + column] - sum);
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_concat_f32")
void volvoxai_training_concat_f32(const float *input, float *output,
        uint32_t outer, uint32_t input_axis, uint32_t output_axis,
        uint32_t inner, uint32_t axis_offset) {
    if (!input || !output) return;
    uint32_t block = input_axis * inner;
    for (uint32_t group = 0; group < outer; group++) {
        size_t source = (size_t)group * block;
        size_t destination = ((size_t)group * output_axis + axis_offset) * inner;
        for (uint32_t i = 0; i < block; i++) output[destination + i] = input[source + i];
    }
}

VX_TRAINING_EXPORT("volvoxai_training_concat_backward_f32")
void volvoxai_training_concat_backward_f32(const float *dy, float *dx,
        uint32_t outer, uint32_t input_axis, uint32_t output_axis,
        uint32_t inner, uint32_t axis_offset) {
    if (!dy || !dx) return;
    uint32_t block = input_axis * inner;
    for (uint32_t group = 0; group < outer; group++) {
        size_t source = ((size_t)group * output_axis + axis_offset) * inner;
        size_t destination = (size_t)group * block;
        for (uint32_t i = 0; i < block; i++) dx[destination + i] += dy[source + i];
    }
}

VX_TRAINING_EXPORT("volvoxai_training_conv2d_backward_f32")
uint32_t volvoxai_training_conv2d_backward_f32(const float *input, const float *weight,
        const float *output, const float *dy, float *dx, float *dw, float *db,
        uint32_t batch, uint32_t in_h, uint32_t in_w, uint32_t in_c, uint32_t kh,
        uint32_t kw, uint32_t group_in, uint32_t out_c, uint32_t out_h, uint32_t out_w,
        uint32_t stride_y, uint32_t stride_x, uint32_t pad_top, uint32_t pad_left,
        uint32_t groups, uint32_t relu, uint32_t dilation_y, uint32_t dilation_x) {
    if (!input || !weight || !output || !dy || !dx || !dw || !batch || !in_h || !in_w || !in_c ||
        !kh || !kw || !group_in || !out_c || !out_h || !out_w || !stride_y || !stride_x ||
        !groups || in_c % groups || out_c % groups) return 0;
    uint32_t group_out = out_c / groups;
    for (uint32_t n=0;n<batch;n++) for(uint32_t oh=0;oh<out_h;oh++) for(uint32_t ow=0;ow<out_w;ow++) for(uint32_t oc=0;oc<out_c;oc++) {
        size_t oi=(((size_t)n*out_h+oh)*out_w+ow)*out_c+oc; float g=dy[oi];
        if ((relu==1 && output[oi]<=0) || (relu>=2 && (output[oi]<=0 || output[oi]>=6))) g=0;
        if(db) db[oc]+=g;
        uint32_t depthwise = groups == in_c;
        uint32_t multiplier = depthwise ? out_c / in_c : 0;
        uint32_t group=oc/group_out, start=depthwise ? oc / multiplier : group*group_in;
        uint32_t inputs = depthwise ? 1 : group_in;
        for(uint32_t local=0;local<inputs;local++) for(uint32_t y=0;y<kh;y++) for(uint32_t x=0;x<kw;x++) {
            int ih=(int)(oh*stride_y+y*dilation_y)-(int)pad_top, iw=(int)(ow*stride_x+x*dilation_x)-(int)pad_left;
            if(ih<0||iw<0||ih>=(int)in_h||iw>=(int)in_w) continue;
            uint32_t ic=start+local; size_t ii=(((size_t)n*in_h+(uint32_t)ih)*in_w+(uint32_t)iw)*in_c+ic;
            size_t wi=depthwise ? (((size_t)y*kw+x)*in_c+ic)*multiplier+(oc%multiplier) : (((size_t)y*kw+x)*group_in+local)*out_c+oc;
            dx[ii]+=g*weight[wi]; dw[wi]+=g*input[ii];
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_prelu_f32")
uint32_t volvoxai_training_prelu_f32(const float *input, const float *slope, float *output,
        uint32_t elements, uint32_t slope_elements, uint32_t channels) {
    if (!input || !slope || !output || !slope_elements || !channels) return 0;
    for (uint32_t i=0;i<elements;i++) { float x=input[i], a=slope[(slope_elements==channels ? i%channels : i%slope_elements)]; output[i]=x<0.0f?x*a:x; }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_prelu_backward_f32")
uint32_t volvoxai_training_prelu_backward_f32(const float *input, const float *slope,
        const float *dy, float *dx, float *dslope, uint32_t elements,
        uint32_t slope_elements, uint32_t channels) {
    if (!input || !slope || !dy || !dx || !dslope || !slope_elements || !channels) return 0;
    for (uint32_t i=0;i<elements;i++) { uint32_t si=slope_elements==channels?i%channels:i%slope_elements; if(input[i]<0.0f){dx[i]+=dy[i]*slope[si];dslope[si]+=dy[i]*input[i];}else dx[i]+=dy[i]; }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_global_average_pool_backward_f32")
void volvoxai_training_global_average_pool_backward_f32(const float *dy, float *dx,
        uint32_t batch, uint32_t height, uint32_t width, uint32_t channels) {
    if (!dy || !dx || !batch || !height || !width || !channels) return;
    float scale = 1.0f / (float)(height * width);
    for (uint32_t n=0;n<batch;n++) for(uint32_t y=0;y<height;y++) for(uint32_t x=0;x<width;x++) for(uint32_t c=0;c<channels;c++)
        dx[(((size_t)n*height+y)*width+x)*channels+c] += dy[n*channels+c] * scale;
}

VX_TRAINING_EXPORT("volvoxai_training_batchnorm2d_backward_f32")
void volvoxai_training_batchnorm2d_backward_f32(const float *input, const float *weight,
        const float *running_mean, const float *running_var, const float *dy, float *dx,
        float *dweight, float *dbias, uint32_t batch, uint32_t height, uint32_t width,
        uint32_t channels, float epsilon) {
    if (!input || !weight || !running_mean || !running_var || !dy || !dx || !dweight || !dbias || !channels || !(epsilon > 0.0f)) return;
    uint32_t count=batch*height*width;
    for(uint32_t c=0;c<channels;c++) { float inv=1.0f/__builtin_sqrtf(running_var[c]+epsilon); for(uint32_t n=0;n<count;n++){size_t i=(size_t)n*channels+c;dx[i]+=dy[i]*weight[c]*inv;dweight[c]+=dy[i]*(input[i]-running_mean[c])*inv;dbias[c]+=dy[i];}}
}

VX_TRAINING_EXPORT("volvoxai_training_maxpool2d_f32")
void volvoxai_training_maxpool2d_f32(const float *input, float *output,
        uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
        uint32_t out_height, uint32_t out_width, uint32_t kernel_y, uint32_t kernel_x,
        uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x) {
    if (!input || !output || !batch || !height || !width || !channels || !out_height || !out_width || !kernel_y || !kernel_x || !stride_y || !stride_x) return;
    for (uint32_t n=0;n<batch;n++) for(uint32_t oy=0;oy<out_height;oy++) for(uint32_t ox=0;ox<out_width;ox++) for(uint32_t c=0;c<channels;c++) {
        float best=-3.4028234663852886e38f;
        for(uint32_t ky=0;ky<kernel_y;ky++) for(uint32_t kx=0;kx<kernel_x;kx++) {
            int iy=(int)(oy*stride_y+ky)-(int)pad_y, ix=(int)(ox*stride_x+kx)-(int)pad_x;
            if(iy>=0 && ix>=0 && iy<(int)height && ix<(int)width) { float value=input[(((size_t)n*height+(uint32_t)iy)*width+(uint32_t)ix)*channels+c]; if(value>best) best=value; }
        }
        output[(((size_t)n*out_height+oy)*out_width+ox)*channels+c]=best;
    }
}

VX_TRAINING_EXPORT("volvoxai_training_maxpool2d_backward_f32")
void volvoxai_training_maxpool2d_backward_f32(const float *input, const float *dy, float *dx,
        uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
        uint32_t out_height, uint32_t out_width, uint32_t kernel_y, uint32_t kernel_x,
        uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x) {
    if (!input || !dy || !dx || !batch || !height || !width || !channels || !out_height || !out_width || !kernel_y || !kernel_x || !stride_y || !stride_x) return;
    for (uint32_t n=0;n<batch;n++) for(uint32_t oy=0;oy<out_height;oy++) for(uint32_t ox=0;ox<out_width;ox++) for(uint32_t c=0;c<channels;c++) {
        float best=-3.4028234663852886e38f; size_t best_index=0;
        for(uint32_t ky=0;ky<kernel_y;ky++) for(uint32_t kx=0;kx<kernel_x;kx++) { int iy=(int)(oy*stride_y+ky)-(int)pad_y, ix=(int)(ox*stride_x+kx)-(int)pad_x; if(iy>=0 && ix>=0 && iy<(int)height && ix<(int)width) { size_t index=(((size_t)n*height+(uint32_t)iy)*width+(uint32_t)ix)*channels+c; if(input[index]>best){best=input[index];best_index=index;} } }
        dx[best_index]+=dy[(((size_t)n*out_height+oy)*out_width+ox)*channels+c];
    }
}

VX_TRAINING_EXPORT("volvoxai_training_averagepool2d_backward_f32")
void volvoxai_training_averagepool2d_backward_f32(const float *dy, float *dx,
        uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
        uint32_t out_height, uint32_t out_width, uint32_t kernel_y, uint32_t kernel_x,
        uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x) {
    if (!dy || !dx || !batch || !height || !width || !channels || !out_height || !out_width || !kernel_y || !kernel_x || !stride_y || !stride_x) return;
    for (uint32_t n=0;n<batch;n++) for(uint32_t oy=0;oy<out_height;oy++) for(uint32_t ox=0;ox<out_width;ox++) {
        uint32_t count=0; for(uint32_t ky=0;ky<kernel_y;ky++) for(uint32_t kx=0;kx<kernel_x;kx++){int iy=(int)(oy*stride_y+ky)-(int)pad_y, ix=(int)(ox*stride_x+kx)-(int)pad_x; if(iy>=0&&ix>=0&&iy<(int)height&&ix<(int)width) count++;}
        if(!count) continue;
        for(uint32_t ky=0;ky<kernel_y;ky++) for(uint32_t kx=0;kx<kernel_x;kx++){int iy=(int)(oy*stride_y+ky)-(int)pad_y, ix=(int)(ox*stride_x+kx)-(int)pad_x; if(iy>=0&&ix>=0&&iy<(int)height&&ix<(int)width) for(uint32_t c=0;c<channels;c++) dx[(((size_t)n*height+(uint32_t)iy)*width+(uint32_t)ix)*channels+c]+=dy[(((size_t)n*out_height+oy)*out_width+ox)*channels+c]/(float)count;}
    }
}

VX_TRAINING_EXPORT("volvoxai_training_resize2d_f32")
void volvoxai_training_resize2d_f32(const float *input, float *output, uint32_t batch,
        uint32_t in_height, uint32_t in_width, uint32_t channels, uint32_t out_height,
        uint32_t out_width, uint32_t nearest) {
    if (!input || !output || !batch || !in_height || !in_width || !channels || !out_height || !out_width) return;
    for(uint32_t n=0;n<batch;n++) for(uint32_t oy=0;oy<out_height;oy++) for(uint32_t ox=0;ox<out_width;ox++) {
        float fy=((float)oy+.5f)*(float)in_height/(float)out_height-.5f, fx=((float)ox+.5f)*(float)in_width/(float)out_width-.5f;
        if(nearest) { uint32_t iy=(uint32_t)((uint64_t)oy*in_height/out_height), ix=(uint32_t)((uint64_t)ox*in_width/out_width); for(uint32_t c=0;c<channels;c++) output[(((size_t)n*out_height+oy)*out_width+ox)*channels+c]=input[(((size_t)n*in_height+iy)*in_width+ix)*channels+c]; continue; }
        if(fy<0)fy=0; if(fx<0)fx=0; uint32_t y0=(uint32_t)fy, x0=(uint32_t)fx, y1=y0+1<in_height?y0+1:y0, x1=x0+1<in_width?x0+1:x0; float wy=fy-y0, wx=fx-x0;
        for(uint32_t c=0;c<channels;c++){float v00=input[(((size_t)n*in_height+y0)*in_width+x0)*channels+c],v01=input[(((size_t)n*in_height+y0)*in_width+x1)*channels+c],v10=input[(((size_t)n*in_height+y1)*in_width+x0)*channels+c],v11=input[(((size_t)n*in_height+y1)*in_width+x1)*channels+c];output[(((size_t)n*out_height+oy)*out_width+ox)*channels+c]=v00*(1-wy)*(1-wx)+v01*(1-wy)*wx+v10*wy*(1-wx)+v11*wy*wx;}
    }
}

VX_TRAINING_EXPORT("volvoxai_training_resize2d_backward_f32")
void volvoxai_training_resize2d_backward_f32(const float *dy, float *dx, uint32_t batch,
        uint32_t in_height, uint32_t in_width, uint32_t channels, uint32_t out_height,
        uint32_t out_width, uint32_t nearest) {
    if (!dy || !dx || !batch || !in_height || !in_width || !channels || !out_height || !out_width) return;
    for(uint32_t n=0;n<batch;n++) for(uint32_t oy=0;oy<out_height;oy++) for(uint32_t ox=0;ox<out_width;ox++) {
        float fy=((float)oy+.5f)*(float)in_height/(float)out_height-.5f, fx=((float)ox+.5f)*(float)in_width/(float)out_width-.5f;
        if(nearest) { uint32_t iy=(uint32_t)((uint64_t)oy*in_height/out_height), ix=(uint32_t)((uint64_t)ox*in_width/out_width); for(uint32_t c=0;c<channels;c++)dx[(((size_t)n*in_height+iy)*in_width+ix)*channels+c]+=dy[(((size_t)n*out_height+oy)*out_width+ox)*channels+c]; continue; }
        if(fy<0)fy=0; if(fx<0)fx=0; uint32_t y0=(uint32_t)fy,x0=(uint32_t)fx,y1=y0+1<in_height?y0+1:y0,x1=x0+1<in_width?x0+1:x0;float wy=fy-y0,wx=fx-x0;
        for(uint32_t c=0;c<channels;c++){float g=dy[(((size_t)n*out_height+oy)*out_width+ox)*channels+c];dx[(((size_t)n*in_height+y0)*in_width+x0)*channels+c]+=g*(1-wy)*(1-wx);dx[(((size_t)n*in_height+y0)*in_width+x1)*channels+c]+=g*(1-wy)*wx;dx[(((size_t)n*in_height+y1)*in_width+x0)*channels+c]+=g*wy*(1-wx);dx[(((size_t)n*in_height+y1)*in_width+x1)*channels+c]+=g*wy*wx;}
    }
}

VX_TRAINING_EXPORT("volvoxai_training_split_backward_f32")
void volvoxai_training_split_backward_f32(const float *dy, float *dx, uint32_t outer,
        uint32_t input_axis, uint32_t output_axis, uint32_t inner, uint32_t axis_offset) {
    if (!dy || !dx) return;
    uint32_t block=output_axis*inner;
    for(uint32_t group=0;group<outer;group++) {
        size_t source=(size_t)group*block, destination=((size_t)group*input_axis+axis_offset)*inner;
        for(uint32_t i=0;i<block;i++) dx[destination+i]+=dy[source+i];
    }
}

VX_TRAINING_EXPORT("volvoxai_training_clip_backward_f32")
void volvoxai_training_clip_backward_f32(const float *input, const float *dy, float *dx,
        uint32_t elements, float minimum, float maximum) {
    if (!input || !dy || !dx || minimum > maximum) return;
    for(uint32_t i=0;i<elements;i++) if(input[i]>minimum && input[i]<maximum) dx[i]+=dy[i];
}

VX_TRAINING_EXPORT("volvoxai_training_activation_backward_f32")
uint32_t volvoxai_training_activation_backward_f32(
        uint32_t kind, const float *x, const float *y, const float *dy,
        float *dx, uint32_t n, float alpha) {
    if (n == 0) return 1;
    if (!x || !dy || !dx || kind > VOLVOXAI_TRAINING_ACTIVATION_HARD_SWISH ||
        !vx_finite_f32(alpha)) return 0;
    for (uint32_t i = 0; i < n; i++) {
        float value = x[i], derivative;
        if (kind == VOLVOXAI_TRAINING_ACTIVATION_RELU) {
            derivative = value > 0.0f ? 1.0f : 0.0f;
        } else if (kind == VOLVOXAI_TRAINING_ACTIVATION_GELU_ERF) {
            derivative = 0.5f * (1.0f + vx_erf_approx(value * 0.7071067811865475f)) +
                         value * 0.3989422804014327f * expf(-0.5f * value * value);
        } else if (kind == VOLVOXAI_TRAINING_ACTIVATION_GELU_TANH) {
            float square = value * value;
            float t = vx_tanh_f32(0.7978845608028654f *
                                  (value + 0.044715f * value * square));
            derivative = 0.5f * (1.0f + t) + 0.5f * value * (1.0f - t * t) *
                         0.7978845608028654f * (1.0f + 0.134145f * square);
        } else if (kind == VOLVOXAI_TRAINING_ACTIVATION_SILU) {
            float sigmoid = vx_sigmoid_f32(value);
            derivative = sigmoid * (1.0f + value * (1.0f - sigmoid));
        } else if (kind == VOLVOXAI_TRAINING_ACTIVATION_SIGMOID) {
            float output = y ? y[i] : vx_sigmoid_f32(value);
            derivative = output * (1.0f - output);
        } else if (kind == VOLVOXAI_TRAINING_ACTIVATION_TANH) {
            float output = y ? y[i] : vx_tanh_f32(value);
            derivative = 1.0f - output * output;
        } else if (kind == VOLVOXAI_TRAINING_ACTIVATION_LEAKY_RELU) {
            derivative = value > 0.0f ? 1.0f : alpha;
        } else if (kind == VOLVOXAI_TRAINING_ACTIVATION_HARD_SIGMOID) {
            derivative = value > -3.0f && value < 3.0f ? 1.0f / 6.0f : 0.0f;
        } else if (kind == VOLVOXAI_TRAINING_ACTIVATION_HARD_SWISH) {
            derivative = value <= -3.0f ? 0.0f : (value >= 3.0f ? 1.0f : value / 3.0f + 0.5f);
        } else {
            derivative = value <= -3.0f ? 0.0f : (value >= 3.0f ? 1.0f : value / 3.0f + 0.5f);
        }
        dx[i] += dy[i] * derivative;
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_add_backward_f32")
void volvoxai_training_add_backward_f32(
        const float *dy, float *da, float *db, uint32_t n) {
    if (!dy || (!da && !db)) return;
    for (uint32_t i = 0; i < n; i++) {
        float gradient = dy[i];
        if (da) da[i] += gradient;
        if (db) db[i] += gradient;
    }
}

VX_TRAINING_EXPORT("volvoxai_training_mul_backward_f32")
void volvoxai_training_mul_backward_f32(
        const float *a, const float *b, const float *dy,
        float *da, float *db, uint32_t n) {
    if (!a || !b || !dy || (!da && !db)) return;
    for (uint32_t i = 0; i < n; i++) {
        float left = a[i], right = b[i], gradient = dy[i];
        if (da) da[i] += gradient * right;
        if (db) db[i] += gradient * left;
    }
}

VX_TRAINING_EXPORT("volvoxai_training_dropout_f32")
void volvoxai_training_dropout_f32(
        const float *input, float *output, uint32_t n,
        uint32_t threshold, uint32_t seed, uint32_t counter, float scale) {
    if (!input || !output || !vx_finite_f32(scale) || scale < 0.0f) return;
    for (uint32_t i = 0; i < n; i++) {
        output[i] = vx_dropout_bits(i, seed, counter) >= threshold ? input[i] * scale : 0.0f;
    }
}

VX_TRAINING_EXPORT("volvoxai_training_dropout_backward_f32")
void volvoxai_training_dropout_backward_f32(
        const float *dy, float *dx, uint32_t n,
        uint32_t threshold, uint32_t seed, uint32_t counter, float scale) {
    if (!dy || !dx || !vx_finite_f32(scale) || scale < 0.0f) return;
    for (uint32_t i = 0; i < n; i++) {
        if (vx_dropout_bits(i, seed, counter) >= threshold) dx[i] += dy[i] * scale;
    }
}

typedef struct {
    const float *qkv;
    const float *q;
    const float *k;
    const float *v;
    const int32_t *mask;
    uint32_t batch;
    uint32_t queries;
    uint32_t keys;
    uint32_t d_model;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
    uint32_t causal;
    uint32_t mask_mode;
    uint32_t dropout_threshold;
    uint32_t dropout_seed;
    uint32_t dropout_counter;
    float dropout_scale;
    uint32_t packed_qkv;
} VXAttentionParams;

static uint32_t vx_attention_size_mul(size_t *value, uint32_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

static uint32_t vx_attention_common_valid(
        const int32_t *mask, uint32_t batch, uint32_t queries, uint32_t keys,
        uint32_t d_model, uint32_t heads, float scale, uint32_t causal,
        uint32_t mask_mode, float dropout_scale) {
    if (!batch || !queries || !keys || !d_model || !heads || d_model % heads ||
        causal > 1 || mask_mode > 4 || (mask_mode && !mask) ||
        !vx_finite_f32(scale) || !vx_finite_f32(dropout_scale) || dropout_scale < 0.0f) {
        return 0;
    }
    return 1;
}

static uint32_t vx_attention_storage_fits(
        uint32_t batch, uint32_t queries, uint32_t keys, uint32_t d_model,
        uint32_t packed_qkv) {
    size_t count = 1;
    if (!vx_attention_size_mul(&count, batch) ||
        !vx_attention_size_mul(&count, queries) ||
        !vx_attention_size_mul(&count, d_model)) return 0;
    if (packed_qkv) return vx_attention_size_mul(&count, 3);
    count = 1;
    return vx_attention_size_mul(&count, batch) &&
        vx_attention_size_mul(&count, keys) &&
        vx_attention_size_mul(&count, d_model);
}

static uint32_t vx_attention_key_allowed(
        const VXAttentionParams *params, uint32_t batch, uint32_t query, uint32_t key) {
    size_t index;
    if (params->causal && key > query) return 0;
    if (!params->mask || params->mask_mode == 0) return 1;
    if (params->mask_mode == 1) index = key;
    else if (params->mask_mode == 2) index = (size_t)batch * params->keys + key;
    else if (params->mask_mode == 3) index = (size_t)query * params->keys + key;
    else index = ((size_t)batch * params->queries + query) * params->keys + key;
    return params->mask[index] != 0;
}

static size_t vx_attention_q_index(
        const VXAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t head, uint32_t dimension) {
    size_t offset = ((size_t)batch * params->queries + query) * params->d_model +
        (size_t)head * params->head_dim + dimension;
    return params->packed_qkv ?
        ((size_t)batch * params->queries + query) * (3u * (size_t)params->d_model) +
            (size_t)head * params->head_dim + dimension : offset;
}

static size_t vx_attention_k_index(
        const VXAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t head, uint32_t dimension) {
    size_t offset = ((size_t)batch * params->keys + key) * params->d_model +
        (size_t)head * params->head_dim + dimension;
    return params->packed_qkv ?
        ((size_t)batch * params->queries + key) * (3u * (size_t)params->d_model) +
            params->d_model + (size_t)head * params->head_dim + dimension : offset;
}

static size_t vx_attention_v_index(
        const VXAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t head, uint32_t dimension) {
    size_t offset = ((size_t)batch * params->keys + key) * params->d_model +
        (size_t)head * params->head_dim + dimension;
    return params->packed_qkv ?
        ((size_t)batch * params->queries + key) * (3u * (size_t)params->d_model) +
            2u * (size_t)params->d_model + (size_t)head * params->head_dim + dimension : offset;
}

static float vx_attention_q_value(
        const VXAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t head, uint32_t dimension) {
    size_t index = vx_attention_q_index(params, batch, query, head, dimension);
    return params->packed_qkv ? params->qkv[index] : params->q[index];
}

static float vx_attention_k_value(
        const VXAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t head, uint32_t dimension) {
    size_t index = vx_attention_k_index(params, batch, key, head, dimension);
    return params->packed_qkv ? params->qkv[index] : params->k[index];
}

static float vx_attention_v_value(
        const VXAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t head, uint32_t dimension) {
    size_t index = vx_attention_v_index(params, batch, key, head, dimension);
    return params->packed_qkv ? params->qkv[index] : params->v[index];
}

static double vx_attention_score(
        const VXAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t key, uint32_t head) {
    double score = 0.0;
    for (uint32_t dimension = 0; dimension < params->head_dim; dimension++) {
        score += (double)vx_attention_q_value(params, batch, query, head, dimension) *
            vx_attention_k_value(params, batch, key, head, dimension);
    }
    return score * (double)params->scale;
}

static uint32_t vx_attention_statistics(
        const VXAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t head, double *maximum, double *denominator) {
    uint32_t visible = 0;
    double local_maximum = 0.0;
    for (uint32_t key = 0; key < params->keys; key++) {
        if (!vx_attention_key_allowed(params, batch, query, key)) continue;
        double score = vx_attention_score(params, batch, query, key, head);
        if (!visible || score > local_maximum) local_maximum = score;
        visible++;
    }
    if (!visible) return 0;
    double local_denominator = 0.0;
    for (uint32_t key = 0; key < params->keys; key++) {
        if (!vx_attention_key_allowed(params, batch, query, key)) continue;
        local_denominator += expf((float)(vx_attention_score(params, batch, query, key, head) -
            local_maximum));
    }
    *maximum = local_maximum;
    *denominator = local_denominator;
    return 1;
}

static double vx_attention_probability(
        const VXAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t key, uint32_t head, double maximum, double denominator) {
    return expf((float)(vx_attention_score(params, batch, query, key, head) - maximum)) /
        denominator;
}

static float vx_attention_dropout_multiplier(
        const VXAttentionParams *params, uint32_t batch, uint32_t head,
        uint32_t query, uint32_t key) {
    uint32_t index = (uint32_t)((((uint64_t)batch * params->heads + head) *
        params->queries + query) * params->keys + key);
    return vx_dropout_bits(index, params->dropout_seed, params->dropout_counter) >=
        params->dropout_threshold ? params->dropout_scale : 0.0f;
}

static void vx_attention_forward(const VXAttentionParams *params, float *output) {
    for (uint32_t batch = 0; batch < params->batch; batch++) {
        size_t output_base = (size_t)batch * params->queries * params->d_model;
        for (uint32_t head = 0; head < params->heads; head++) {
            for (uint32_t query = 0; query < params->queries; query++) {
                size_t output_offset = output_base + (size_t)query * params->d_model +
                    (size_t)head * params->head_dim;
                double maximum, denominator;
                if (!vx_attention_statistics(params, batch, query, head, &maximum, &denominator)) {
                    for (uint32_t dimension = 0; dimension < params->head_dim; dimension++)
                        output[output_offset + dimension] = 0.0f;
                    continue;
                }
                for (uint32_t dimension = 0; dimension < params->head_dim; dimension++) {
                    double value = 0.0;
                    for (uint32_t key = 0; key < params->keys; key++) {
                        if (!vx_attention_key_allowed(params, batch, query, key)) continue;
                        double probability = vx_attention_probability(
                            params, batch, query, key, head, maximum, denominator);
                        value += probability * vx_attention_dropout_multiplier(
                            params, batch, head, query, key) *
                            vx_attention_v_value(params, batch, key, head, dimension);
                    }
                    output[output_offset + dimension] = (float)value;
                }
            }
        }
    }
}

static void vx_attention_backward(
        const VXAttentionParams *params, const float *dy,
        float *dq, float *dk, float *dv) {
    for (uint32_t batch = 0; batch < params->batch; batch++) {
        size_t output_base = (size_t)batch * params->queries * params->d_model;
        for (uint32_t head = 0; head < params->heads; head++) {
            for (uint32_t query = 0; query < params->queries; query++) {
                double maximum, denominator;
                if (!vx_attention_statistics(params, batch, query, head, &maximum, &denominator)) continue;
                size_t output_offset = output_base + (size_t)query * params->d_model +
                    (size_t)head * params->head_dim;
                double weighted_probability_gradient = 0.0;
                for (uint32_t key = 0; key < params->keys; key++) {
                    if (!vx_attention_key_allowed(params, batch, query, key)) continue;
                    double probability = vx_attention_probability(
                        params, batch, query, key, head, maximum, denominator);
                    float multiplier = vx_attention_dropout_multiplier(
                        params, batch, head, query, key);
                    double probability_gradient = 0.0;
                    for (uint32_t dimension = 0; dimension < params->head_dim; dimension++) {
                        float output_gradient = dy[output_offset + dimension];
                        probability_gradient += (double)output_gradient *
                            vx_attention_v_value(params, batch, key, head, dimension);
                        dv[vx_attention_v_index(params, batch, key, head, dimension)] +=
                            (float)(probability * multiplier * output_gradient);
                    }
                    weighted_probability_gradient += probability * probability_gradient * multiplier;
                }
                for (uint32_t key = 0; key < params->keys; key++) {
                    if (!vx_attention_key_allowed(params, batch, query, key)) continue;
                    double probability = vx_attention_probability(
                        params, batch, query, key, head, maximum, denominator);
                    float multiplier = vx_attention_dropout_multiplier(
                        params, batch, head, query, key);
                    double probability_gradient = 0.0;
                    for (uint32_t dimension = 0; dimension < params->head_dim; dimension++) {
                        probability_gradient += (double)dy[output_offset + dimension] *
                            vx_attention_v_value(params, batch, key, head, dimension);
                    }
                    double score_gradient = probability *
                        (probability_gradient * multiplier - weighted_probability_gradient);
                    for (uint32_t dimension = 0; dimension < params->head_dim; dimension++) {
                        size_t q_index = vx_attention_q_index(params, batch, query, head, dimension);
                        size_t k_index = vx_attention_k_index(params, batch, key, head, dimension);
                        dq[q_index] += (float)(score_gradient * params->scale *
                            vx_attention_k_value(params, batch, key, head, dimension));
                        dk[k_index] += (float)(score_gradient * params->scale *
                            vx_attention_q_value(params, batch, query, head, dimension));
                    }
                }
            }
        }
    }
}

VX_TRAINING_EXPORT("volvoxai_training_sdpa_f32")
uint32_t volvoxai_training_sdpa_f32(
        const float *qkv, const int32_t *mask, float *output,
        uint32_t batch, uint32_t seq, uint32_t d_model, uint32_t heads,
        float scale, uint32_t causal, uint32_t mask_mode,
        uint32_t dropout_threshold, uint32_t dropout_seed,
        uint32_t dropout_counter, float dropout_scale) {
    if (!qkv || !output || !vx_attention_common_valid(mask, batch, seq, seq, d_model,
        heads, scale, causal, mask_mode, dropout_scale) ||
        !vx_attention_storage_fits(batch, seq, seq, d_model, 1)) return 0;
    VXAttentionParams params = {0};
    params.qkv = qkv;
    params.mask = mask;
    params.batch = batch;
    params.queries = seq;
    params.keys = seq;
    params.d_model = d_model;
    params.heads = heads;
    params.head_dim = d_model / heads;
    params.scale = scale;
    params.causal = causal;
    params.mask_mode = mask_mode;
    params.dropout_threshold = dropout_threshold;
    params.dropout_seed = dropout_seed;
    params.dropout_counter = dropout_counter;
    params.dropout_scale = dropout_scale;
    params.packed_qkv = 1;
    vx_attention_forward(&params, output);
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_sdpa_backward_f32")
uint32_t volvoxai_training_sdpa_backward_f32(
        const float *qkv, const int32_t *mask, const float *dy, float *dqkv,
        uint32_t batch, uint32_t seq, uint32_t d_model, uint32_t heads,
        float scale, uint32_t causal, uint32_t mask_mode,
        uint32_t dropout_threshold, uint32_t dropout_seed,
        uint32_t dropout_counter, float dropout_scale) {
    if (!qkv || !dy || !dqkv || !vx_attention_common_valid(mask, batch, seq, seq,
        d_model, heads, scale, causal, mask_mode, dropout_scale) ||
        !vx_attention_storage_fits(batch, seq, seq, d_model, 1)) return 0;
    VXAttentionParams params = {0};
    params.qkv = qkv;
    params.mask = mask;
    params.batch = batch;
    params.queries = seq;
    params.keys = seq;
    params.d_model = d_model;
    params.heads = heads;
    params.head_dim = d_model / heads;
    params.scale = scale;
    params.causal = causal;
    params.mask_mode = mask_mode;
    params.dropout_threshold = dropout_threshold;
    params.dropout_seed = dropout_seed;
    params.dropout_counter = dropout_counter;
    params.dropout_scale = dropout_scale;
    params.packed_qkv = 1;
    vx_attention_backward(&params, dy, dqkv, dqkv, dqkv);
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_cross_sdpa_f32")
uint32_t volvoxai_training_cross_sdpa_f32(
        const float *q, const float *k, const float *v, const int32_t *mask,
        float *output, uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        uint32_t d_model, uint32_t heads, float scale, uint32_t causal,
        uint32_t mask_mode, uint32_t dropout_threshold, uint32_t dropout_seed,
        uint32_t dropout_counter, float dropout_scale) {
    if (!q || !k || !v || !output || !vx_attention_common_valid(mask, batch, seq_q,
        seq_kv, d_model, heads, scale, causal, mask_mode, dropout_scale) ||
        !vx_attention_storage_fits(batch, seq_q, seq_kv, d_model, 0)) return 0;
    VXAttentionParams params = {0};
    params.q = q;
    params.k = k;
    params.v = v;
    params.mask = mask;
    params.batch = batch;
    params.queries = seq_q;
    params.keys = seq_kv;
    params.d_model = d_model;
    params.heads = heads;
    params.head_dim = d_model / heads;
    params.scale = scale;
    params.causal = causal;
    params.mask_mode = mask_mode;
    params.dropout_threshold = dropout_threshold;
    params.dropout_seed = dropout_seed;
    params.dropout_counter = dropout_counter;
    params.dropout_scale = dropout_scale;
    vx_attention_forward(&params, output);
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_cross_sdpa_backward_f32")
uint32_t volvoxai_training_cross_sdpa_backward_f32(
        const float *q, const float *k, const float *v, const int32_t *mask,
        const float *dy, float *dq, float *dk, float *dv,
        uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        uint32_t d_model, uint32_t heads, float scale, uint32_t causal,
        uint32_t mask_mode, uint32_t dropout_threshold, uint32_t dropout_seed,
        uint32_t dropout_counter, float dropout_scale) {
    if (!q || !k || !v || !dy || !dq || !dk || !dv ||
        !vx_attention_common_valid(mask, batch, seq_q, seq_kv, d_model, heads,
            scale, causal, mask_mode, dropout_scale) ||
        !vx_attention_storage_fits(batch, seq_q, seq_kv, d_model, 0)) return 0;
    VXAttentionParams params = {0};
    params.q = q;
    params.k = k;
    params.v = v;
    params.mask = mask;
    params.batch = batch;
    params.queries = seq_q;
    params.keys = seq_kv;
    params.d_model = d_model;
    params.heads = heads;
    params.head_dim = d_model / heads;
    params.scale = scale;
    params.causal = causal;
    params.mask_mode = mask_mode;
    params.dropout_threshold = dropout_threshold;
    params.dropout_seed = dropout_seed;
    params.dropout_counter = dropout_counter;
    params.dropout_scale = dropout_scale;
    vx_attention_backward(&params, dy, dq, dk, dv);
    return 1;
}

static uint32_t vx_moe_size_mul(size_t *value, uint32_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

static uint32_t vx_moe_router_dimensions_valid(
        uint32_t rows, uint32_t d_model, uint32_t experts, uint32_t top_k,
        float temperature, uint32_t normalize) {
    size_t product = 1;
    if (!rows || !d_model || !experts || !top_k || top_k > experts || normalize > 1 ||
        !vx_finite_f32(temperature) || !(temperature > 0.0f)) return 0;
    if (!vx_moe_size_mul(&product, rows) || !vx_moe_size_mul(&product, d_model)) return 0;
    product = 1;
    if (!vx_moe_size_mul(&product, d_model) || !vx_moe_size_mul(&product, experts)) return 0;
    product = 1;
    return vx_moe_size_mul(&product, rows) && vx_moe_size_mul(&product, top_k);
}

static uint32_t vx_moe_linear_dimensions_valid(
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        uint32_t experts, uint32_t top_k) {
    size_t product = 1;
    if (!rows || !d_in || !d_out || !experts || !top_k || top_k > experts) return 0;
    if (!vx_moe_size_mul(&product, rows) || !vx_moe_size_mul(&product, d_in)) return 0;
    product = 1;
    if (!vx_moe_size_mul(&product, experts) || !vx_moe_size_mul(&product, d_in) ||
        !vx_moe_size_mul(&product, d_out)) return 0;
    product = 1;
    if (!vx_moe_size_mul(&product, rows) || !vx_moe_size_mul(&product, d_out)) return 0;
    product = 1;
    return vx_moe_size_mul(&product, rows) && vx_moe_size_mul(&product, top_k);
}

static double vx_moe_router_logit_f64(
        const float *input, const float *weight, const float *bias,
        uint32_t row, uint32_t expert, uint32_t d_model, uint32_t experts,
        float temperature) {
    double value = bias ? bias[expert] : 0.0;
    size_t input_offset = (size_t)row * d_model;
    for (uint32_t dimension = 0; dimension < d_model; dimension++) {
        value += (double)input[input_offset + dimension] *
            weight[(size_t)dimension * experts + expert];
    }
    return value / temperature;
}

static float vx_moe_router_logit(
        const float *input, const float *weight, const float *bias,
        uint32_t row, uint32_t expert, uint32_t d_model, uint32_t experts,
        float temperature) {
    return (float)vx_moe_router_logit_f64(
        input, weight, bias, row, expert, d_model, experts, temperature);
}

static uint32_t vx_moe_route_index(
        const float *indices, uint32_t offset, uint32_t experts, uint32_t *expert) {
    float raw = indices[offset];
    if (!vx_finite_f32(raw) || raw < 0.0f || raw >= (float)experts) return 0;
    uint32_t value = (uint32_t)raw;
    if ((float)value != raw) return 0;
    *expert = value;
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_moe_router_f32")
uint32_t volvoxai_training_moe_router_f32(
        const float *input, const float *weight, const float *bias,
        float *indices, float *route_weights,
        uint32_t rows, uint32_t d_model, uint32_t experts, uint32_t top_k,
        float temperature, uint32_t normalize) {
    if (!input || !weight || !indices || !route_weights ||
        !vx_moe_router_dimensions_valid(rows, d_model, experts, top_k, temperature, normalize)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        float maximum = 0.0f;
        uint32_t have_maximum = 0;
        for (uint32_t expert = 0; expert < experts; expert++) {
            float logit = vx_moe_router_logit(
                input, weight, bias, row, expert, d_model, experts, temperature);
            if (!vx_finite_f32(logit)) return 0;
            if (!have_maximum || logit > maximum) {
                maximum = logit;
                have_maximum = 1;
            }
        }
        for (uint32_t slot = 0; slot < top_k; slot++) {
            uint32_t best = 0;
            float best_value = 0.0f;
            uint32_t have_best = 0;
            for (uint32_t expert = 0; expert < experts; expert++) {
                uint32_t selected = 0;
                for (uint32_t prior = 0; prior < slot; prior++) {
                    if ((uint32_t)indices[(size_t)row * top_k + prior] == expert) {
                        selected = 1;
                        break;
                    }
                }
                if (selected) continue;
                float logit = vx_moe_router_logit(
                    input, weight, bias, row, expert, d_model, experts, temperature);
                if (!have_best || logit > best_value ||
                    (logit == best_value && expert < best)) {
                    best = expert;
                    best_value = logit;
                    have_best = 1;
                }
            }
            if (!have_best) return 0;
            indices[(size_t)row * top_k + slot] = (float)best;
        }
        double denominator = 0.0;
        if (normalize) {
            for (uint32_t slot = 0; slot < top_k; slot++) {
                uint32_t expert = (uint32_t)indices[(size_t)row * top_k + slot];
                denominator += expf(vx_moe_router_logit(
                    input, weight, bias, row, expert, d_model, experts, temperature) - maximum);
            }
        } else {
            for (uint32_t expert = 0; expert < experts; expert++) {
                denominator += expf(vx_moe_router_logit(
                    input, weight, bias, row, expert, d_model, experts, temperature) - maximum);
            }
        }
        if (!(denominator > 0.0)) return 0;
        for (uint32_t slot = 0; slot < top_k; slot++) {
            uint32_t expert = (uint32_t)indices[(size_t)row * top_k + slot];
            route_weights[(size_t)row * top_k + slot] = (float)(expf(vx_moe_router_logit(
                input, weight, bias, row, expert, d_model, experts, temperature) - maximum) /
                denominator);
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_moe_router_backward_f32")
uint32_t volvoxai_training_moe_router_backward_f32(
        const float *input, const float *weight, const float *bias,
        const float *indices, const float *route_weights,
        const float *grad_route_weights,
        float *dinput, float *dweight, float *dbias,
        uint32_t rows, uint32_t d_model, uint32_t experts, uint32_t top_k,
        float temperature, uint32_t normalize) {
    if (!input || !weight || !indices || !route_weights || !grad_route_weights ||
        !dinput || !dweight || (bias && !dbias) ||
        !vx_moe_router_dimensions_valid(rows, d_model, experts, top_k, temperature, normalize)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        if (normalize) {
            double gate_dot = 0.0;
            for (uint32_t slot = 0; slot < top_k; slot++) {
                uint32_t route_offset = row * top_k + slot;
                uint32_t expert;
                if (!vx_moe_route_index(indices, route_offset, experts, &expert) ||
                    !vx_finite_f32(route_weights[route_offset]) ||
                    !vx_finite_f32(grad_route_weights[route_offset])) return 0;
                gate_dot += (double)grad_route_weights[route_offset] * route_weights[route_offset];
            }
            for (uint32_t slot = 0; slot < top_k; slot++) {
                uint32_t route_offset = row * top_k + slot;
                uint32_t expert;
                (void)vx_moe_route_index(indices, route_offset, experts, &expert);
                double d_logit = route_weights[route_offset] *
                    (grad_route_weights[route_offset] - gate_dot) / temperature;
                if (dbias) dbias[expert] += (float)d_logit;
                for (uint32_t dimension = 0; dimension < d_model; dimension++) {
                    size_t input_offset = (size_t)row * d_model + dimension;
                    size_t weight_offset = (size_t)dimension * experts + expert;
                    dinput[input_offset] += (float)(d_logit * weight[weight_offset]);
                    dweight[weight_offset] += (float)(input[input_offset] * d_logit);
                }
            }
            continue;
        }

        double maximum = 0.0;
        for (uint32_t expert = 0; expert < experts; expert++) {
            double logit = vx_moe_router_logit_f64(
                input, weight, bias, row, expert, d_model, experts, temperature);
            if (!vx_finite_f32((float)logit)) return 0;
            if (expert == 0 || logit > maximum) maximum = logit;
        }
        double denominator = 0.0;
        for (uint32_t expert = 0; expert < experts; expert++) {
            denominator += expf((float)(vx_moe_router_logit_f64(
                input, weight, bias, row, expert, d_model, experts, temperature) - maximum));
        }
        if (!(denominator > 0.0)) return 0;
        double probability_dot = 0.0;
        for (uint32_t slot = 0; slot < top_k; slot++) {
            uint32_t route_offset = row * top_k + slot;
            uint32_t expert;
            if (!vx_moe_route_index(indices, route_offset, experts, &expert) ||
                !vx_finite_f32(grad_route_weights[route_offset])) return 0;
            double probability = expf((float)(vx_moe_router_logit_f64(
                input, weight, bias, row, expert, d_model, experts, temperature) - maximum)) /
                denominator;
            probability_dot += probability * grad_route_weights[route_offset];
        }
        for (uint32_t expert = 0; expert < experts; expert++) {
            double probability = expf((float)(vx_moe_router_logit_f64(
                input, weight, bias, row, expert, d_model, experts, temperature) - maximum)) /
                denominator;
            double upstream = 0.0;
            for (uint32_t slot = 0; slot < top_k; slot++) {
                uint32_t route_offset = row * top_k + slot;
                uint32_t selected;
                (void)vx_moe_route_index(indices, route_offset, experts, &selected);
                if (selected == expert) upstream += grad_route_weights[route_offset];
            }
            double d_logit = probability * (upstream - probability_dot) / temperature;
            if (dbias) dbias[expert] += (float)d_logit;
            for (uint32_t dimension = 0; dimension < d_model; dimension++) {
                size_t input_offset = (size_t)row * d_model + dimension;
                size_t weight_offset = (size_t)dimension * experts + expert;
                dinput[input_offset] += (float)(d_logit * weight[weight_offset]);
                dweight[weight_offset] += (float)(input[input_offset] * d_logit);
            }
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_moe_linear_f32")
uint32_t volvoxai_training_moe_linear_f32(
        const float *input, const float *expert_weight, const float *expert_bias,
        const float *route_indices, const float *route_weights, float *output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        uint32_t experts, uint32_t top_k) {
    if (!input || !expert_weight || !route_indices || !route_weights || !output ||
        !vx_moe_linear_dimensions_valid(rows, d_in, d_out, experts, top_k)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < d_out; column++) {
            double sum = 0.0;
            for (uint32_t slot = 0; slot < top_k; slot++) {
                uint32_t route_offset = row * top_k + slot;
                uint32_t expert;
                float gate = route_weights[route_offset];
                if (!vx_moe_route_index(route_indices, route_offset, experts, &expert) ||
                    !vx_finite_f32(gate)) return 0;
                size_t expert_offset = (size_t)expert * d_in * d_out;
                double value = expert_bias ? expert_bias[(size_t)expert * d_out + column] : 0.0;
                for (uint32_t dimension = 0; dimension < d_in; dimension++) {
                    value += (double)input[(size_t)row * d_in + dimension] *
                        expert_weight[expert_offset + (size_t)dimension * d_out + column];
                }
                sum += gate * value;
            }
            output[(size_t)row * d_out + column] = (float)sum;
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_moe_linear_backward_f32")
uint32_t volvoxai_training_moe_linear_backward_f32(
        const float *input, const float *expert_weight, const float *expert_bias,
        const float *route_indices, const float *route_weights, const float *dy,
        float *dinput, float *dweight, float *dbias, float *droute_weights,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        uint32_t experts, uint32_t top_k) {
    if (!input || !expert_weight || !route_indices || !route_weights || !dy ||
        !dinput || !dweight || !droute_weights || (expert_bias && !dbias) ||
        !vx_moe_linear_dimensions_valid(rows, d_in, d_out, experts, top_k)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t slot = 0; slot < top_k; slot++) {
            uint32_t route_offset = row * top_k + slot;
            uint32_t expert;
            float gate = route_weights[route_offset];
            if (!vx_moe_route_index(route_indices, route_offset, experts, &expert) ||
                !vx_finite_f32(gate)) return 0;
            size_t expert_offset = (size_t)expert * d_in * d_out;
            double route_gradient = 0.0;
            for (uint32_t column = 0; column < d_out; column++) {
                float output_gradient = dy[(size_t)row * d_out + column];
                double value = expert_bias ? expert_bias[(size_t)expert * d_out + column] : 0.0;
                for (uint32_t dimension = 0; dimension < d_in; dimension++) {
                    size_t input_offset = (size_t)row * d_in + dimension;
                    size_t weight_offset = expert_offset + (size_t)dimension * d_out + column;
                    value += (double)input[input_offset] * expert_weight[weight_offset];
                    dinput[input_offset] += (float)(output_gradient * gate * expert_weight[weight_offset]);
                    dweight[weight_offset] += (float)(output_gradient * gate * input[input_offset]);
                }
                route_gradient += (double)output_gradient * value;
                if (dbias) dbias[(size_t)expert * d_out + column] += output_gradient * gate;
            }
            droute_weights[route_offset] += (float)route_gradient;
        }
    }
    return 1;
}

typedef struct {
    const float *q;
    const float *kv;
    const float *weight;
    const float *scale;
    const float *bias;
    uint32_t batch;
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t d_model;
    uint32_t heads;
    uint32_t head_dim;
    double attention_scale;
} VXCrossAttentionParams;

static uint32_t vx_cross_attention_size_mul(size_t *value, uint32_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

static uint32_t vx_cross_attention_dimensions_valid(
        uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        uint32_t d_model, uint32_t heads) {
    size_t product = 1;
    if (!batch || !seq_q || !seq_kv || !d_model || !heads || d_model % heads) return 0;
    if (!vx_cross_attention_size_mul(&product, batch) ||
        !vx_cross_attention_size_mul(&product, seq_q) ||
        !vx_cross_attention_size_mul(&product, d_model)) return 0;
    product = 1;
    if (!vx_cross_attention_size_mul(&product, batch) ||
        !vx_cross_attention_size_mul(&product, seq_kv) ||
        !vx_cross_attention_size_mul(&product, d_model)) return 0;
    product = 1;
    return vx_cross_attention_size_mul(&product, 3) &&
        vx_cross_attention_size_mul(&product, d_model) &&
        vx_cross_attention_size_mul(&product, d_model);
}

static double vx_cross_attention_q_raw(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t column) {
    size_t source = ((size_t)batch * params->seq_q + query) * params->d_model;
    size_t weight = (size_t)column * params->d_model;
    double value = 0.0;
    for (uint32_t dimension = 0; dimension < params->d_model; dimension++)
        value += (double)params->q[source + dimension] * params->weight[weight + dimension];
    return value;
}

static double vx_cross_attention_kv_raw(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t column) {
    size_t source = ((size_t)batch * params->seq_kv + key) * params->d_model;
    size_t weight = (size_t)column * params->d_model;
    double value = 0.0;
    for (uint32_t dimension = 0; dimension < params->d_model; dimension++)
        value += (double)params->kv[source + dimension] * params->weight[weight + dimension];
    return value;
}

static double vx_cross_attention_affine(
        const VXCrossAttentionParams *params, uint32_t column, double raw) {
    if (params->scale) raw *= params->scale[column];
    if (params->bias) raw += params->bias[column];
    return raw;
}

static float vx_cross_attention_forward_q_projected(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t head, uint32_t dimension) {
    uint32_t column = head * params->head_dim + dimension;
    return (float)vx_cross_attention_affine(
        params, column, vx_cross_attention_q_raw(params, batch, query, column));
}

static double vx_cross_attention_forward_k_projected(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t head, uint32_t dimension) {
    uint32_t column = params->d_model + head * params->head_dim + dimension;
    return vx_cross_attention_affine(
        params, column, vx_cross_attention_kv_raw(params, batch, key, column));
}

static double vx_cross_attention_forward_v_projected(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t head, uint32_t dimension) {
    uint32_t column = 2u * params->d_model + head * params->head_dim + dimension;
    return vx_cross_attention_affine(
        params, column, vx_cross_attention_kv_raw(params, batch, key, column));
}

static double vx_cross_attention_forward_score(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t key, uint32_t head) {
    double score = 0.0;
    for (uint32_t dimension = 0; dimension < params->head_dim; dimension++) {
        score += (double)vx_cross_attention_forward_q_projected(
            params, batch, query, head, dimension) *
            vx_cross_attention_forward_k_projected(params, batch, key, head, dimension);
    }
    return score * params->attention_scale;
}

static void vx_cross_attention_forward_statistics(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t head, double *maximum, double *denominator) {
    double local_maximum = 0.0;
    for (uint32_t key = 0; key < params->seq_kv; key++) {
        double score = vx_cross_attention_forward_score(params, batch, query, key, head);
        if (key == 0 || score > local_maximum) local_maximum = score;
    }
    double local_denominator = 0.0;
    for (uint32_t key = 0; key < params->seq_kv; key++) {
        float score = (float)vx_cross_attention_forward_score(params, batch, query, key, head);
        local_denominator += expf((float)((double)score - local_maximum));
    }
    *maximum = local_maximum;
    *denominator = local_denominator;
}

static double vx_cross_attention_forward_probability(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t key, uint32_t head, double maximum, double denominator) {
    float score = (float)vx_cross_attention_forward_score(params, batch, query, key, head);
    return expf((float)((double)score - maximum)) / denominator;
}

static double vx_cross_attention_backward_q_projected(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t head, uint32_t dimension) {
    uint32_t column = head * params->head_dim + dimension;
    return vx_cross_attention_affine(
        params, column, vx_cross_attention_q_raw(params, batch, query, column));
}

static double vx_cross_attention_backward_k_projected(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t head, uint32_t dimension) {
    uint32_t column = params->d_model + head * params->head_dim + dimension;
    return vx_cross_attention_affine(
        params, column, vx_cross_attention_kv_raw(params, batch, key, column));
}

static double vx_cross_attention_backward_v_projected(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t key,
        uint32_t head, uint32_t dimension) {
    uint32_t column = 2u * params->d_model + head * params->head_dim + dimension;
    return vx_cross_attention_affine(
        params, column, vx_cross_attention_kv_raw(params, batch, key, column));
}

static double vx_cross_attention_backward_score(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t key, uint32_t head) {
    double score = 0.0;
    for (uint32_t dimension = 0; dimension < params->head_dim; dimension++) {
        score += vx_cross_attention_backward_q_projected(params, batch, query, head, dimension) *
            vx_cross_attention_backward_k_projected(params, batch, key, head, dimension);
    }
    return score * params->attention_scale;
}

static void vx_cross_attention_backward_statistics(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t head, double *maximum, double *denominator) {
    double local_maximum = 0.0;
    for (uint32_t key = 0; key < params->seq_kv; key++) {
        double score = vx_cross_attention_backward_score(params, batch, query, key, head);
        if (key == 0 || score > local_maximum) local_maximum = score;
    }
    double local_denominator = 0.0;
    for (uint32_t key = 0; key < params->seq_kv; key++) {
        local_denominator += expf((float)(vx_cross_attention_backward_score(
            params, batch, query, key, head) - local_maximum));
    }
    *maximum = local_maximum;
    *denominator = local_denominator;
}

static double vx_cross_attention_backward_probability(
        const VXCrossAttentionParams *params, uint32_t batch, uint32_t query,
        uint32_t key, uint32_t head, double maximum, double denominator) {
    return expf((float)(vx_cross_attention_backward_score(
        params, batch, query, key, head) - maximum)) / denominator;
}

static double vx_cross_attention_probability_gradient(
        const VXCrossAttentionParams *params, const float *dy,
        uint32_t batch, uint32_t query, uint32_t key, uint32_t head) {
    size_t output = ((size_t)batch * params->seq_q + query) * params->d_model +
        (size_t)head * params->head_dim;
    double value = 0.0;
    for (uint32_t dimension = 0; dimension < params->head_dim; dimension++) {
        value += (double)dy[output + dimension] *
            vx_cross_attention_backward_v_projected(params, batch, key, head, dimension);
    }
    return value;
}

static void vx_cross_attention_params_init(
        VXCrossAttentionParams *params, const float *q, const float *kv,
        const float *weight, const float *scale, const float *bias,
        uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        uint32_t d_model, uint32_t heads) {
    params->q = q;
    params->kv = kv;
    params->weight = weight;
    params->scale = scale;
    params->bias = bias;
    params->batch = batch;
    params->seq_q = seq_q;
    params->seq_kv = seq_kv;
    params->d_model = d_model;
    params->heads = heads;
    params->head_dim = d_model / heads;
    params->attention_scale = 1.0 / sqrtf((float)params->head_dim);
}

VX_TRAINING_EXPORT("volvoxai_training_cross_attention_f32")
uint32_t volvoxai_training_cross_attention_f32(
        const float *q, const float *kv, const float *weight,
        const float *scale, const float *bias, float *output,
        uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        uint32_t d_model, uint32_t heads) {
    if (!q || !kv || !weight || !output ||
        !vx_cross_attention_dimensions_valid(batch, seq_q, seq_kv, d_model, heads)) return 0;
    VXCrossAttentionParams params;
    vx_cross_attention_params_init(&params, q, kv, weight, scale, bias,
        batch, seq_q, seq_kv, d_model, heads);
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        for (uint32_t head = 0; head < params.heads; head++) {
            for (uint32_t query = 0; query < params.seq_q; query++) {
                double maximum, denominator;
                vx_cross_attention_forward_statistics(
                    &params, batch_index, query, head, &maximum, &denominator);
                if (!(denominator > 0.0)) return 0;
                size_t output_offset = ((size_t)batch_index * params.seq_q + query) * params.d_model +
                    (size_t)head * params.head_dim;
                for (uint32_t dimension = 0; dimension < params.head_dim; dimension++) {
                    double value = 0.0;
                    for (uint32_t key = 0; key < params.seq_kv; key++) {
                        value += vx_cross_attention_forward_probability(
                            &params, batch_index, query, key, head, maximum, denominator) *
                            vx_cross_attention_forward_v_projected(
                                &params, batch_index, key, head, dimension);
                    }
                    output[output_offset + dimension] = (float)value;
                }
            }
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_cross_attention_backward_f32")
uint32_t volvoxai_training_cross_attention_backward_f32(
        const float *q, const float *kv, const float *weight,
        const float *scale, const float *bias, const float *dy,
        float *dq, float *dkv, float *dweight, float *dscale, float *dbias,
        uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        uint32_t d_model, uint32_t heads) {
    if (!q || !kv || !weight || !dy || !dq || !dkv || !dweight ||
        (scale && !dscale) || (bias && !dbias) ||
        !vx_cross_attention_dimensions_valid(batch, seq_q, seq_kv, d_model, heads)) return 0;
    VXCrossAttentionParams params;
    vx_cross_attention_params_init(&params, q, kv, weight, scale, bias,
        batch, seq_q, seq_kv, d_model, heads);
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        for (uint32_t head = 0; head < params.heads; head++) {
            for (uint32_t query = 0; query < params.seq_q; query++) {
                double maximum, denominator;
                vx_cross_attention_backward_statistics(
                    &params, batch_index, query, head, &maximum, &denominator);
                if (!(denominator > 0.0)) return 0;
                double probability_dot = 0.0;
                for (uint32_t key = 0; key < params.seq_kv; key++) {
                    probability_dot += vx_cross_attention_backward_probability(
                        &params, batch_index, query, key, head, maximum, denominator) *
                        vx_cross_attention_probability_gradient(
                            &params, dy, batch_index, query, key, head);
                }

                for (uint32_t dimension = 0; dimension < params.head_dim; dimension++) {
                    double projected_gradient = 0.0;
                    for (uint32_t key = 0; key < params.seq_kv; key++) {
                        double probability = vx_cross_attention_backward_probability(
                            &params, batch_index, query, key, head, maximum, denominator);
                        double score_gradient = probability *
                            (vx_cross_attention_probability_gradient(
                                &params, dy, batch_index, query, key, head) - probability_dot);
                        projected_gradient += score_gradient * params.attention_scale *
                            vx_cross_attention_backward_k_projected(
                                &params, batch_index, key, head, dimension);
                    }
                    uint32_t column = head * params.head_dim + dimension;
                    double raw = vx_cross_attention_q_raw(&params, batch_index, query, column);
                    double raw_gradient = projected_gradient *
                        (params.scale ? params.scale[column] : 1.0);
                    if (dscale) dscale[column] += (float)(projected_gradient * raw);
                    if (dbias) dbias[column] += (float)projected_gradient;
                    for (uint32_t input_dimension = 0; input_dimension < params.d_model; input_dimension++) {
                        size_t source = ((size_t)batch_index * params.seq_q + query) * params.d_model +
                            input_dimension;
                        size_t weight_index = (size_t)column * params.d_model + input_dimension;
                        dq[source] += (float)(raw_gradient * params.weight[weight_index]);
                        dweight[weight_index] += (float)(raw_gradient * params.q[source]);
                    }
                }

                for (uint32_t key = 0; key < params.seq_kv; key++) {
                    double probability = vx_cross_attention_backward_probability(
                        &params, batch_index, query, key, head, maximum, denominator);
                    double score_gradient = probability *
                        (vx_cross_attention_probability_gradient(
                            &params, dy, batch_index, query, key, head) - probability_dot);
                    for (uint32_t dimension = 0; dimension < params.head_dim; dimension++) {
                        uint32_t k_column = params.d_model + head * params.head_dim + dimension;
                        uint32_t v_column = 2u * params.d_model + head * params.head_dim + dimension;
                        size_t output = ((size_t)batch_index * params.seq_q + query) * params.d_model +
                            (size_t)head * params.head_dim + dimension;
                        double k_projected_gradient = score_gradient * params.attention_scale *
                            vx_cross_attention_backward_q_projected(
                                &params, batch_index, query, head, dimension);
                        double v_projected_gradient = probability * dy[output];
                        double k_raw = vx_cross_attention_kv_raw(&params, batch_index, key, k_column);
                        double v_raw = vx_cross_attention_kv_raw(&params, batch_index, key, v_column);
                        double k_raw_gradient = k_projected_gradient *
                            (params.scale ? params.scale[k_column] : 1.0);
                        double v_raw_gradient = v_projected_gradient *
                            (params.scale ? params.scale[v_column] : 1.0);
                        if (dscale) {
                            dscale[k_column] += (float)(k_projected_gradient * k_raw);
                            dscale[v_column] += (float)(v_projected_gradient * v_raw);
                        }
                        if (dbias) {
                            dbias[k_column] += (float)k_projected_gradient;
                            dbias[v_column] += (float)v_projected_gradient;
                        }
                        for (uint32_t input_dimension = 0; input_dimension < params.d_model; input_dimension++) {
                            size_t source = ((size_t)batch_index * params.seq_kv + key) * params.d_model +
                                input_dimension;
                            size_t k_weight = (size_t)k_column * params.d_model + input_dimension;
                            size_t v_weight = (size_t)v_column * params.d_model + input_dimension;
                            float source_value = params.kv[source];
                            dkv[source] += (float)(k_raw_gradient * params.weight[k_weight] +
                                v_raw_gradient * params.weight[v_weight]);
                            dweight[k_weight] += (float)(k_raw_gradient * source_value);
                            dweight[v_weight] += (float)(v_raw_gradient * source_value);
                        }
                    }
                }
            }
        }
    }
    return 1;
}

static uint32_t vx_training_typed_valid(uint32_t type) {
    return type == VX_DTYPE_F32 || type == VX_DTYPE_I32 ||
        type == VX_DTYPE_I8 || type == VX_DTYPE_U8;
}

static double vx_training_typed_number(const void *values, uint32_t type,
        uint32_t index) {
    if (type == VOLVOXAI_TRAINING_TYPED_F32) return (double)((const float *)values)[index];
    if (type == VOLVOXAI_TRAINING_TYPED_I32) return (double)((const int32_t *)values)[index];
    if (type == VOLVOXAI_TRAINING_TYPED_I8) return (double)((const int8_t *)values)[index];
    return (double)((const uint8_t *)values)[index];
}

static int32_t vx_training_integer_value(const void *values, uint32_t type,
        uint32_t index) {
    if (type == VOLVOXAI_TRAINING_TYPED_I32) return ((const int32_t *)values)[index];
    if (type == VOLVOXAI_TRAINING_TYPED_I8) return (int32_t)((const int8_t *)values)[index];
    return (int32_t)((const uint8_t *)values)[index];
}

/* ECMAScript TypedArray integer stores use ToInt32: truncate a finite value
   toward zero, reduce modulo 2^32, then reinterpret as signed. Derive that
   directly from the F32 bit pattern so conversion never invokes undefined
   out-of-range C float-to-int casts. */
static uint32_t vx_training_f32_to_u32_mod(float value) {
    union { float f; uint32_t u; } bits = { value };
    uint32_t exponent = (bits.u >> 23u) & 0xffu;
    if (exponent == 0xffu || exponent < 127u) return 0;
    uint32_t significand = (bits.u & 0x7fffffu) | 0x800000u;
    int32_t shift = (int32_t)exponent - 127 - 23;
    uint32_t magnitude;
    if (shift >= 32) magnitude = 0;
    else if (shift >= 0) magnitude = significand << (uint32_t)shift;
    else magnitude = significand >> (uint32_t)(-shift);
    return (bits.u & 0x80000000u) ? 0u - magnitude : magnitude;
}

static int32_t vx_training_i32_from_bits(uint32_t bits) {
    if (bits <= 0x7fffffffu) return (int32_t)bits;
    return (int32_t)(bits - 0x80000000u) - 2147483647 - 1;
}

static int32_t vx_training_f32_to_i32(float value) {
    return vx_training_i32_from_bits(vx_training_f32_to_u32_mod(value));
}

static int8_t vx_training_i8_from_u8(uint8_t value) {
    if (value <= 127u) return (int8_t)value;
    return (int8_t)(value - 128u) - 128;
}

static void vx_training_store_integer(void *values, uint32_t type,
        uint32_t index, int32_t integer) {
    if (type == VOLVOXAI_TRAINING_TYPED_I32) ((int32_t *)values)[index] = integer;
    else if (type == VOLVOXAI_TRAINING_TYPED_I8) {
        ((int8_t *)values)[index] = vx_training_i8_from_u8((uint8_t)integer);
    }
    else ((uint8_t *)values)[index] = (uint8_t)integer;
}

VX_TRAINING_EXPORT("volvoxai_training_cast_typed")
uint32_t volvoxai_training_cast_typed(
        const void *input, uint32_t input_type, void *output,
        uint32_t output_type, uint32_t elements) {
    if ((!input || !output) && elements) return 0;
    if (!vx_training_typed_valid(input_type) || !vx_training_typed_valid(output_type)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        if (input_type == VOLVOXAI_TRAINING_TYPED_F32) {
            float value = ((const float *)input)[index];
            if (output_type == VOLVOXAI_TRAINING_TYPED_F32) ((float *)output)[index] = value;
            else vx_training_store_integer(output, output_type, index,
                vx_training_f32_to_i32(value));
        } else {
            int32_t value = vx_training_integer_value(input, input_type, index);
            if (output_type == VOLVOXAI_TRAINING_TYPED_F32) ((float *)output)[index] = (float)value;
            else vx_training_store_integer(output, output_type, index, value);
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_cast_backward_f32")
uint32_t volvoxai_training_cast_backward_f32(
        const float *dy, float *dx, uint32_t input_type,
        uint32_t output_type, uint32_t elements) {
    if (!dy && elements) return 0;
    if (!vx_training_typed_valid(input_type) || !vx_training_typed_valid(output_type)) return 0;
    if (input_type != VOLVOXAI_TRAINING_TYPED_F32 ||
        output_type != VOLVOXAI_TRAINING_TYPED_F32) return 1;
    if (!dx && elements) return 0;
    for (uint32_t index = 0; index < elements; index++) dx[index] += dy[index];
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_dequantize_linear_typed")
uint32_t volvoxai_training_dequantize_linear_typed(
        const void *input, uint32_t input_type, const float *scale,
        const void *zero_point, uint32_t zero_point_type, float *output,
        uint32_t elements) {
    if ((!input || !scale || !output) && elements) return 0;
    if (!vx_training_typed_valid(input_type) ||
        (zero_point && !vx_training_typed_valid(zero_point_type))) return 0;
    double zero = zero_point
        ? vx_training_typed_number(zero_point, zero_point_type, 0) : 0.0;
    for (uint32_t index = 0; index < elements; index++) {
        output[index] = (float)((vx_training_typed_number(input, input_type, index) - zero) *
            (double)scale[0]);
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_dequantize_linear_backward_f32")
uint32_t volvoxai_training_dequantize_linear_backward_f32(
        const void *input, uint32_t input_type, const float *scale,
        const void *zero_point, uint32_t zero_point_type, const float *dy,
        float *dx, float *dscale, uint32_t elements) {
    if ((!input || !scale || !dy) && elements) return 0;
    if (!vx_training_typed_valid(input_type) ||
        (zero_point && !vx_training_typed_valid(zero_point_type))) return 0;
    if (!dx && !dscale && elements) return 0;
    double zero = zero_point
        ? vx_training_typed_number(zero_point, zero_point_type, 0) : 0.0;
    if (input_type == VOLVOXAI_TRAINING_TYPED_F32 && dx) {
        for (uint32_t index = 0; index < elements; index++) {
            dx[index] += (double)dy[index] * (double)scale[0];
        }
    }
    if (dscale) {
        for (uint32_t index = 0; index < elements; index++) {
            dscale[0] += (double)dy[index] *
                (vx_training_typed_number(input, input_type, index) - zero);
        }
    }
    return 1;
}

static uint32_t vx_binary_index(
        uint32_t output_index, const uint32_t *shape, uint32_t rank,
        const uint32_t *out_shape, uint32_t out_rank, const size_t *strides) {
    uint32_t result = 0;
    uint32_t remaining = output_index;
    uint32_t offset = out_rank - rank;
    for (uint32_t reverse = out_rank; reverse-- > 0;) {
        uint32_t coordinate = remaining % out_shape[reverse];
        remaining /= out_shape[reverse];
        if (reverse >= offset && shape[reverse - offset] != 1)
            result += coordinate * (uint32_t)strides[reverse - offset];
    }
    return result;
}

VX_TRAINING_EXPORT("volvoxai_training_binary_broadcast_backward_f32")
uint32_t volvoxai_training_binary_broadcast_backward_f32(
        const float *a, const float *b, const float *dy, float *da, float *db,
        const uint32_t *a_shape, const uint32_t *b_shape, const uint32_t *out_shape,
        uint32_t a_rank, uint32_t b_rank, uint32_t out_rank, uint32_t elements,
        uint32_t kind) {
    if (!a || !b || !dy || !da || !db || !a_shape || !b_shape || !out_shape ||
        a_rank == 0 || b_rank == 0 || out_rank == 0 || a_rank > out_rank ||
        b_rank > out_rank || out_rank > 8 || kind > 3) return 0;
    size_t a_strides[8], b_strides[8], product = 1;
    for (uint32_t reverse = a_rank; reverse-- > 0;) {
        if (a_shape[reverse] == 0) return 0;
        a_strides[reverse] = product; product *= a_shape[reverse];
    }
    product = 1;
    for (uint32_t reverse = b_rank; reverse-- > 0;) {
        if (b_shape[reverse] == 0) return 0;
        b_strides[reverse] = product; product *= b_shape[reverse];
    }
    product = 1;
    for (uint32_t dim = 0; dim < out_rank; dim++) {
        if (out_shape[dim] == 0) return 0;
        uint32_t a_dim = dim < out_rank - a_rank ? 1 : a_shape[dim - (out_rank - a_rank)];
        uint32_t b_dim = dim < out_rank - b_rank ? 1 : b_shape[dim - (out_rank - b_rank)];
        if ((a_dim != 1 && a_dim != out_shape[dim]) || (b_dim != 1 && b_dim != out_shape[dim])) return 0;
        product *= out_shape[dim];
    }
    if (product != elements) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        uint32_t ai = vx_binary_index(index, a_shape, a_rank, out_shape, out_rank, a_strides);
        uint32_t bi = vx_binary_index(index, b_shape, b_rank, out_shape, out_rank, b_strides);
        if (kind == 0) { da[ai] += dy[index]; db[bi] += dy[index]; }
        else if (kind == 1) { da[ai] += dy[index] * b[bi]; db[bi] += dy[index] * a[ai]; }
        else if (kind == 2) { da[ai] += dy[index]; db[bi] -= dy[index]; }
        else { da[ai] += dy[index] / b[bi]; db[bi] -= dy[index] * a[ai] / (b[bi] * b[bi]); }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_binary_broadcast_f32")
uint32_t volvoxai_training_binary_broadcast_f32(
        const float *a, const float *b, float *output,
        const uint32_t *a_shape, const uint32_t *b_shape, const uint32_t *out_shape,
        uint32_t a_rank, uint32_t b_rank, uint32_t out_rank, uint32_t elements,
        uint32_t kind) {
    if (!a || !b || !output || !a_shape || !b_shape || !out_shape ||
        a_rank == 0 || b_rank == 0 || out_rank == 0 || a_rank > out_rank ||
        b_rank > out_rank || out_rank > 8 || kind > 3) return 0;
    size_t a_strides[8], b_strides[8], product = 1;
    for (uint32_t reverse = a_rank; reverse-- > 0;) {
        if (a_shape[reverse] == 0) return 0;
        a_strides[reverse] = product; product *= a_shape[reverse];
    }
    product = 1;
    for (uint32_t reverse = b_rank; reverse-- > 0;) {
        if (b_shape[reverse] == 0) return 0;
        b_strides[reverse] = product; product *= b_shape[reverse];
    }
    product = 1;
    for (uint32_t dim = 0; dim < out_rank; dim++) {
        if (out_shape[dim] == 0) return 0;
        uint32_t a_dim = dim < out_rank - a_rank ? 1 : a_shape[dim - (out_rank - a_rank)];
        uint32_t b_dim = dim < out_rank - b_rank ? 1 : b_shape[dim - (out_rank - b_rank)];
        if ((a_dim != 1 && a_dim != out_shape[dim]) || (b_dim != 1 && b_dim != out_shape[dim])) return 0;
        product *= out_shape[dim];
    }
    if (product != elements) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        uint32_t ai = vx_binary_index(index, a_shape, a_rank, out_shape, out_rank, a_strides);
        uint32_t bi = vx_binary_index(index, b_shape, b_rank, out_shape, out_rank, b_strides);
        if (kind == 0) output[index] = a[ai] + b[bi];
        else if (kind == 1) output[index] = a[ai] * b[bi];
        else if (kind == 2) output[index] = a[ai] - b[bi];
        else output[index] = a[ai] / b[bi];
    }
    return 1;
}

static uint32_t vx_where_condition(const void *condition, uint32_t condition_type,
        uint32_t index) {
    if (condition_type == VX_DTYPE_F32) {
        return ((const float *)condition)[index] != 0.0f;
    }
    return ((const int32_t *)condition)[index] != 0;
}

VX_TRAINING_EXPORT("volvoxai_training_where_f32")
uint32_t volvoxai_training_where_f32(
        const void *condition, const float *a, const float *b, float *output,
        uint32_t condition_type, uint32_t elements) {
    if (!condition || !a || !b || !output ||
        (condition_type != VX_DTYPE_F32 && condition_type != VX_DTYPE_I32)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        output[index] = vx_where_condition(condition, condition_type, index)
            ? a[index] : b[index];
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_where_backward_f32")
uint32_t volvoxai_training_where_backward_f32(
        const void *condition, const float *dy, float *da, float *db,
        uint32_t condition_type, uint32_t elements) {
    if (!condition || !dy || !da || !db ||
        (condition_type != VX_DTYPE_F32 && condition_type != VX_DTYPE_I32)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        if (vx_where_condition(condition, condition_type, index)) da[index] += dy[index];
        else db[index] += dy[index];
    }
    return 1;
}

static uint32_t vx_slice_validate(const uint32_t *input_shape,
        const uint32_t *output_shape, const uint32_t *starts, const uint32_t *steps,
        uint32_t rank, uint32_t output_elements, size_t *input_strides) {
    if (!input_shape || !output_shape || !starts || !steps || rank == 0 || rank > 8) return 0;
    size_t stride = 1, output_product = 1;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        if (input_shape[reverse] == 0 || output_shape[reverse] == 0 || steps[reverse] == 0 ||
            starts[reverse] >= input_shape[reverse]) return 0;
        if ((size_t)(output_shape[reverse] - 1u) * steps[reverse] + starts[reverse] >= input_shape[reverse]) return 0;
        input_strides[reverse] = stride;
        stride *= input_shape[reverse];
        output_product *= output_shape[reverse];
    }
    return output_product == output_elements;
}

static uint32_t vx_slice_input_index(uint32_t output_index, const uint32_t *output_shape,
        const uint32_t *starts, const uint32_t *steps, uint32_t rank,
        const size_t *input_strides) {
    uint32_t result = 0;
    uint32_t remaining = output_index;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        uint32_t coordinate = remaining % output_shape[reverse];
        remaining /= output_shape[reverse];
        result += (starts[reverse] + coordinate * steps[reverse]) * (uint32_t)input_strides[reverse];
    }
    return result;
}

VX_TRAINING_EXPORT("volvoxai_training_slice_f32")
uint32_t volvoxai_training_slice_f32(
        const float *input, float *output, const uint32_t *input_shape,
        const uint32_t *output_shape, const uint32_t *starts, const uint32_t *steps,
        uint32_t rank, uint32_t output_elements) {
    size_t input_strides[8];
    if (!input || !output || !vx_slice_validate(input_shape, output_shape, starts, steps,
        rank, output_elements, input_strides)) return 0;
    for (uint32_t index = 0; index < output_elements; index++) {
        output[index] = input[vx_slice_input_index(index, output_shape, starts, steps, rank, input_strides)];
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_slice_backward_f32")
uint32_t volvoxai_training_slice_backward_f32(
        const float *dy, float *dx, const uint32_t *input_shape,
        const uint32_t *output_shape, const uint32_t *starts, const uint32_t *steps,
        uint32_t rank, uint32_t output_elements) {
    size_t input_strides[8];
    if (!dy || !dx || !vx_slice_validate(input_shape, output_shape, starts, steps,
        rank, output_elements, input_strides)) return 0;
    for (uint32_t index = 0; index < output_elements; index++) {
        dx[vx_slice_input_index(index, output_shape, starts, steps, rank, input_strides)] += dy[index];
    }
    return 1;
}

static uint32_t vx_gather_validate(uint32_t outer, uint32_t axis_size, uint32_t inner,
        uint32_t indices_elements, uint32_t output_elements) {
    size_t selected, total;
    if (!outer || !axis_size || !inner || !indices_elements ||
        !vx_product_fits(outer, indices_elements, &selected) ||
        selected > (size_t)-1 / inner) return 0;
    total = selected * inner;
    return total == output_elements;
}

static uint32_t vx_gather_normalize_index(
        int32_t index, uint32_t axis_size, uint32_t *selected) {
    int64_t normalized = index;
    if (!selected) return 0;
    if (normalized < 0) normalized += (int64_t)axis_size;
    if (normalized < 0 || normalized >= (int64_t)axis_size) return 0;
    *selected = (uint32_t)normalized;
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_gather_f32")
uint32_t volvoxai_training_gather_f32(
        const float *input, const int32_t *indices, float *output,
        uint32_t outer, uint32_t axis_size, uint32_t inner,
        uint32_t indices_elements, uint32_t output_elements) {
    if (!input || !indices || !output || !vx_gather_validate(outer, axis_size, inner,
        indices_elements, output_elements)) return 0;
    for (uint32_t outer_index = 0; outer_index < outer; outer_index++) {
        for (uint32_t index_position = 0; index_position < indices_elements; index_position++) {
            uint32_t selected;
            if (!vx_gather_normalize_index(
                    indices[index_position], axis_size, &selected)) return 0;
            for (uint32_t inner_index = 0; inner_index < inner; inner_index++) {
                size_t output_index = ((size_t)outer_index * indices_elements + index_position) * inner + inner_index;
                size_t input_index = ((size_t)outer_index * axis_size + selected) * inner + inner_index;
                output[output_index] = input[input_index];
            }
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_gather_backward_f32")
uint32_t volvoxai_training_gather_backward_f32(
        const int32_t *indices, const float *dy, float *dx,
        uint32_t outer, uint32_t axis_size, uint32_t inner,
        uint32_t indices_elements, uint32_t output_elements) {
    if (!indices || !dy || !dx || !vx_gather_validate(outer, axis_size, inner,
        indices_elements, output_elements)) return 0;
    for (uint32_t outer_index = 0; outer_index < outer; outer_index++) {
        for (uint32_t index_position = 0; index_position < indices_elements; index_position++) {
            uint32_t selected;
            if (!vx_gather_normalize_index(
                    indices[index_position], axis_size, &selected)) return 0;
            for (uint32_t inner_index = 0; inner_index < inner; inner_index++) {
                size_t output_index = ((size_t)outer_index * indices_elements + index_position) * inner + inner_index;
                size_t input_index = ((size_t)outer_index * axis_size + selected) * inner + inner_index;
                dx[input_index] += dy[output_index];
            }
        }
    }
    return 1;
}

static uint32_t vx_gather_elements_validate(const uint32_t *input_shape,
        const uint32_t *indices_shape, uint32_t rank, uint32_t axis,
        uint32_t elements, size_t *input_strides) {
    size_t stride = 1, output_product = 1;
    if (!input_shape || !indices_shape || rank == 0 || rank > 8 || axis >= rank) return 0;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        if (input_shape[reverse] == 0 || indices_shape[reverse] == 0 ||
            (reverse != axis && indices_shape[reverse] > input_shape[reverse])) return 0;
        input_strides[reverse] = stride;
        stride *= input_shape[reverse];
        output_product *= indices_shape[reverse];
    }
    return output_product == elements;
}

static uint32_t vx_gather_elements_input_index(const int32_t *indices, uint32_t output_index,
        const uint32_t *input_shape, const uint32_t *indices_shape, uint32_t rank,
        uint32_t axis, const size_t *input_strides, size_t *input_index) {
    uint32_t remaining = output_index;
    size_t result = 0;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        uint32_t coordinate = remaining % indices_shape[reverse];
        remaining /= indices_shape[reverse];
        if (reverse == axis) {
            int32_t selected = indices[output_index];
            if (selected < 0) selected += (int32_t)input_shape[reverse];
            if (selected < 0 || (uint32_t)selected >= input_shape[reverse]) return 0;
            result += (size_t)(uint32_t)selected * input_strides[reverse];
        } else {
            result += (size_t)coordinate * input_strides[reverse];
        }
    }
    *input_index = result;
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_gather_elements_f32")
uint32_t volvoxai_training_gather_elements_f32(
        const float *input, const int32_t *indices, float *output,
        const uint32_t *input_shape, const uint32_t *indices_shape,
        uint32_t rank, uint32_t axis, uint32_t elements) {
    size_t input_strides[8];
    if (!input || !indices || !output || !vx_gather_elements_validate(input_shape,
        indices_shape, rank, axis, elements, input_strides)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        size_t input_index;
        if (!vx_gather_elements_input_index(indices, index, input_shape, indices_shape,
            rank, axis, input_strides, &input_index)) return 0;
        output[index] = input[input_index];
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_gather_elements_backward_f32")
uint32_t volvoxai_training_gather_elements_backward_f32(
        const int32_t *indices, const float *dy, float *dx,
        const uint32_t *input_shape, const uint32_t *indices_shape,
        uint32_t rank, uint32_t axis, uint32_t elements) {
    size_t input_strides[8];
    if (!indices || !dy || !dx || !vx_gather_elements_validate(input_shape,
        indices_shape, rank, axis, elements, input_strides)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        size_t input_index;
        if (!vx_gather_elements_input_index(indices, index, input_shape, indices_shape,
            rank, axis, input_strides, &input_index)) return 0;
        dx[input_index] += dy[index];
    }
    return 1;
}

static uint32_t vx_nhwc_profile_valid(uint32_t batch, uint32_t height,
        uint32_t width, uint32_t channels) {
    return batch != 0 && height != 0 && width != 0 && channels != 0;
}

VX_TRAINING_EXPORT("volvoxai_training_mean_height_f32")
uint32_t volvoxai_training_mean_height_f32(
        const float *input, float *output, uint32_t batch, uint32_t height,
        uint32_t width, uint32_t channels) {
    if (!input || !output || !vx_nhwc_profile_valid(batch, height, width, channels)) return 0;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t channel = 0; channel < channels; channel++) {
        for (uint32_t x = 0; x < width; x++) {
            float sum = 0.0f;
            for (uint32_t y = 0; y < height; y++) {
                size_t input_index = (((size_t)b * height + y) * width + x) * channels + channel;
                sum += input[input_index];
            }
            output[((size_t)b * channels + channel) * width + x] = sum / (float)height;
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_mean_height_backward_f32")
uint32_t volvoxai_training_mean_height_backward_f32(
        const float *dy, float *dx, uint32_t batch, uint32_t height,
        uint32_t width, uint32_t channels) {
    if (!dy || !dx || !vx_nhwc_profile_valid(batch, height, width, channels)) return 0;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t channel = 0; channel < channels; channel++) {
        for (uint32_t x = 0; x < width; x++) {
            float gradient = dy[((size_t)b * channels + channel) * width + x] / (float)height;
            for (uint32_t y = 0; y < height; y++) {
                size_t input_index = (((size_t)b * height + y) * width + x) * channels + channel;
                dx[input_index] += gradient;
            }
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_profile_x_f32")
uint32_t volvoxai_training_profile_x_f32(
        const float *input, float *output, uint32_t batch, uint32_t height,
        uint32_t width, uint32_t channels) {
    if (!input || !output || !vx_nhwc_profile_valid(batch, height, width, channels)) return 0;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t channel = 0; channel < channels; channel++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t first_index = (((size_t)b * height) * width + x) * channels + channel;
            float maximum = input[first_index], sum = maximum;
            for (uint32_t y = 1; y < height; y++) {
                float value = input[(((size_t)b * height + y) * width + x) * channels + channel];
                if (value > maximum) maximum = value;
                sum += value;
            }
            size_t output_base = (size_t)b * 2u * channels * width;
            output[output_base + (size_t)channel * width + x] = maximum;
            output[output_base + (size_t)(channels + channel) * width + x] = sum / (float)height;
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_profile_x_backward_f32")
uint32_t volvoxai_training_profile_x_backward_f32(
        const float *input, const float *dy, float *dx, uint32_t batch,
        uint32_t height, uint32_t width, uint32_t channels) {
    if (!input || !dy || !dx || !vx_nhwc_profile_valid(batch, height, width, channels)) return 0;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t channel = 0; channel < channels; channel++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t max_y = 0;
            float maximum = input[(((size_t)b * height) * width + x) * channels + channel];
            for (uint32_t y = 1; y < height; y++) {
                float value = input[(((size_t)b * height + y) * width + x) * channels + channel];
                if (value > maximum) { maximum = value; max_y = y; }
            }
            size_t output_base = (size_t)b * 2u * channels * width;
            float max_gradient = dy[output_base + (size_t)channel * width + x];
            float mean_gradient = dy[output_base + (size_t)(channels + channel) * width + x] / (float)height;
            for (uint32_t y = 0; y < height; y++) {
                size_t input_index = (((size_t)b * height + y) * width + x) * channels + channel;
                dx[input_index] += mean_gradient;
            }
            dx[(((size_t)b * height + max_y) * width + x) * channels + channel] += max_gradient;
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_profile_y_f32")
uint32_t volvoxai_training_profile_y_f32(
        const float *input, float *output, uint32_t batch, uint32_t height,
        uint32_t width, uint32_t channels) {
    if (!input || !output || !vx_nhwc_profile_valid(batch, height, width, channels)) return 0;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t channel = 0; channel < channels; channel++) {
        for (uint32_t y = 0; y < height; y++) {
            size_t first_index = (((size_t)b * height + y) * width) * channels + channel;
            float maximum = input[first_index], sum = maximum;
            for (uint32_t x = 1; x < width; x++) {
                float value = input[(((size_t)b * height + y) * width + x) * channels + channel];
                if (value > maximum) maximum = value;
                sum += value;
            }
            size_t output_base = (size_t)b * 2u * channels * height;
            output[output_base + (size_t)channel * height + y] = maximum;
            output[output_base + (size_t)(channels + channel) * height + y] = sum / (float)width;
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_profile_y_backward_f32")
uint32_t volvoxai_training_profile_y_backward_f32(
        const float *input, const float *dy, float *dx, uint32_t batch,
        uint32_t height, uint32_t width, uint32_t channels) {
    if (!input || !dy || !dx || !vx_nhwc_profile_valid(batch, height, width, channels)) return 0;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t channel = 0; channel < channels; channel++) {
        for (uint32_t y = 0; y < height; y++) {
            uint32_t max_x = 0;
            float maximum = input[(((size_t)b * height + y) * width) * channels + channel];
            for (uint32_t x = 1; x < width; x++) {
                float value = input[(((size_t)b * height + y) * width + x) * channels + channel];
                if (value > maximum) { maximum = value; max_x = x; }
            }
            size_t output_base = (size_t)b * 2u * channels * height;
            float max_gradient = dy[output_base + (size_t)channel * height + y];
            float mean_gradient = dy[output_base + (size_t)(channels + channel) * height + y] / (float)width;
            for (uint32_t x = 0; x < width; x++) {
                size_t input_index = (((size_t)b * height + y) * width + x) * channels + channel;
                dx[input_index] += mean_gradient;
            }
            dx[(((size_t)b * height + y) * width + max_x) * channels + channel] += max_gradient;
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_spatial_softargmax_y_f32")
uint32_t volvoxai_training_spatial_softargmax_y_f32(
        const float *input, float *output, uint32_t batch, uint32_t height,
        uint32_t width, uint32_t channels) {
    if (!input || !output || !vx_nhwc_profile_valid(batch, height, width, channels)) return 0;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t channel = 0; channel < channels; channel++) {
        for (uint32_t x = 0; x < width; x++) {
            float maximum = input[(((size_t)b * height) * width + x) * channels + channel];
            for (uint32_t y = 1; y < height; y++) {
                float value = input[(((size_t)b * height + y) * width + x) * channels + channel];
                if (value > maximum) maximum = value;
            }
            float denominator = 0.0f, weighted = 0.0f;
            for (uint32_t y = 0; y < height; y++) {
                float probability = expf(input[(((size_t)b * height + y) * width + x) * channels + channel] - maximum);
                denominator += probability;
                weighted += probability * ((float)y + 0.5f) / (float)height;
            }
            output[((size_t)b * channels + channel) * width + x] = denominator > 0.0f ? weighted / denominator : 0.0f;
        }
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_spatial_softargmax_y_backward_f32")
uint32_t volvoxai_training_spatial_softargmax_y_backward_f32(
        const float *input, const float *output, const float *dy, float *dx,
        uint32_t batch, uint32_t height, uint32_t width, uint32_t channels) {
    if (!input || !output || !dy || !dx || !vx_nhwc_profile_valid(batch, height, width, channels)) return 0;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t channel = 0; channel < channels; channel++) {
        for (uint32_t x = 0; x < width; x++) {
            float maximum = input[(((size_t)b * height) * width + x) * channels + channel];
            for (uint32_t y = 1; y < height; y++) {
                float value = input[(((size_t)b * height + y) * width + x) * channels + channel];
                if (value > maximum) maximum = value;
            }
            float denominator = 0.0f;
            for (uint32_t y = 0; y < height; y++) {
                denominator += expf(input[(((size_t)b * height + y) * width + x) * channels + channel] - maximum);
            }
            if (denominator <= 0.0f) continue;
            size_t output_index = ((size_t)b * channels + channel) * width + x;
            float gradient = dy[output_index];
            float expectation = output[output_index];
            for (uint32_t y = 0; y < height; y++) {
                size_t input_index = (((size_t)b * height + y) * width + x) * channels + channel;
                float probability = expf(input[input_index] - maximum) / denominator;
                float coordinate = ((float)y + 0.5f) / (float)height;
                dx[input_index] += gradient * probability * (coordinate - expectation);
            }
        }
    }
    return 1;
}

static uint32_t vx_expand_validate(const uint32_t *input_shape, const uint32_t *output_shape,
        uint32_t input_rank, uint32_t output_rank, uint32_t output_elements, size_t *input_strides) {
    if (!input_shape || !output_shape || input_rank == 0 || output_rank == 0 ||
        input_rank > output_rank || output_rank > 8) return 0;
    size_t stride = 1, output_product = 1;
    for (uint32_t reverse = input_rank; reverse-- > 0;) {
        if (input_shape[reverse] == 0) return 0;
        input_strides[reverse] = stride; stride *= input_shape[reverse];
    }
    for (uint32_t dimension = 0; dimension < output_rank; dimension++) {
        if (output_shape[dimension] == 0) return 0;
        uint32_t input_dimension = dimension < output_rank - input_rank ? 1 : input_shape[dimension - (output_rank - input_rank)];
        if (input_dimension != 1 && input_dimension != output_shape[dimension]) return 0;
        output_product *= output_shape[dimension];
    }
    return output_product == output_elements;
}

VX_TRAINING_EXPORT("volvoxai_training_expand_f32")
uint32_t volvoxai_training_expand_f32(const float *input, float *output, const uint32_t *input_shape,
        const uint32_t *output_shape, uint32_t input_rank, uint32_t output_rank, uint32_t output_elements) {
    size_t input_strides[8];
    if (!input || !output || !vx_expand_validate(input_shape, output_shape, input_rank, output_rank, output_elements, input_strides)) return 0;
    for (uint32_t index = 0; index < output_elements; index++) {
        output[index] = input[vx_binary_index(index, input_shape, input_rank, output_shape, output_rank, input_strides)];
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_expand_backward_f32")
uint32_t volvoxai_training_expand_backward_f32(const float *dy, float *dx, const uint32_t *input_shape,
        const uint32_t *output_shape, uint32_t input_rank, uint32_t output_rank, uint32_t output_elements) {
    size_t input_strides[8];
    if (!dy || !dx || !vx_expand_validate(input_shape, output_shape, input_rank, output_rank, output_elements, input_strides)) return 0;
    for (uint32_t index = 0; index < output_elements; index++) {
        dx[vx_binary_index(index, input_shape, input_rank, output_shape, output_rank, input_strides)] += dy[index];
    }
    return 1;
}

static uint32_t vx_conv1d_valid(uint32_t batch, uint32_t in_channels, uint32_t input_length,
        uint32_t out_channels, uint32_t input_per_group, uint32_t kernel,
        uint32_t output_length, uint32_t stride, uint32_t padding, uint32_t groups) {
    if (!batch || !in_channels || !input_length || !out_channels || !input_per_group || !kernel ||
        !output_length || !stride || !groups || in_channels % groups || out_channels % groups ||
        input_per_group != in_channels / groups || kernel > input_length + 2u * padding) return 0;
    return ((input_length + 2u * padding - kernel) / stride + 1u) == output_length;
}

VX_TRAINING_EXPORT("volvoxai_training_conv1d_f32")
uint32_t volvoxai_training_conv1d_f32(const float *input, const float *weight, const float *bias, float *output,
        uint32_t batch, uint32_t in_channels, uint32_t input_length, uint32_t out_channels,
        uint32_t input_per_group, uint32_t kernel, uint32_t output_length, uint32_t stride,
        uint32_t padding, uint32_t groups, uint32_t relu) {
    if (!input || !weight || !output || !vx_conv1d_valid(batch, in_channels, input_length, out_channels,
        input_per_group, kernel, output_length, stride, padding, groups)) return 0;
    uint32_t group_out = out_channels / groups;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t oc = 0; oc < out_channels; oc++)
      for (uint32_t ox = 0; ox < output_length; ox++) {
        float sum = bias ? bias[oc] : 0.0f; uint32_t group = oc / group_out;
        for (uint32_t local_ic = 0; local_ic < input_per_group; local_ic++) for (uint32_t kk = 0; kk < kernel; kk++) {
          int32_t ix = (int32_t)(ox * stride + kk) - (int32_t)padding;
          if (ix >= 0 && (uint32_t)ix < input_length) {
            uint32_t ic = group * input_per_group + local_ic;
            sum += input[((b * in_channels + ic) * input_length) + (uint32_t)ix] *
                   weight[((oc * input_per_group + local_ic) * kernel) + kk];
          }
        }
        output[((b * out_channels + oc) * output_length) + ox] = relu && sum < 0.0f ? 0.0f : sum;
      }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_conv1d_backward_f32")
uint32_t volvoxai_training_conv1d_backward_f32(const float *input, const float *weight, const float *output,
        const float *dy, float *dx, float *dw, float *db, uint32_t batch, uint32_t in_channels,
        uint32_t input_length, uint32_t out_channels, uint32_t input_per_group, uint32_t kernel,
        uint32_t output_length, uint32_t stride, uint32_t padding, uint32_t groups, uint32_t relu) {
    if (!input || !weight || !output || !dy || !dx || !dw || !vx_conv1d_valid(batch, in_channels, input_length,
        out_channels, input_per_group, kernel, output_length, stride, padding, groups)) return 0;
    uint32_t group_out = out_channels / groups;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t oc = 0; oc < out_channels; oc++)
      for (uint32_t ox = 0; ox < output_length; ox++) {
        uint32_t out_index = ((b * out_channels + oc) * output_length) + ox;
        float gradient = relu && output[out_index] <= 0.0f ? 0.0f : dy[out_index];
        if (db) db[oc] += gradient;
        uint32_t group = oc / group_out;
        for (uint32_t local_ic = 0; local_ic < input_per_group; local_ic++) for (uint32_t kk = 0; kk < kernel; kk++) {
          int32_t ix = (int32_t)(ox * stride + kk) - (int32_t)padding;
          if (ix >= 0 && (uint32_t)ix < input_length) {
            uint32_t ic = group * input_per_group + local_ic;
            uint32_t input_index = ((b * in_channels + ic) * input_length) + (uint32_t)ix;
            uint32_t weight_index = ((oc * input_per_group + local_ic) * kernel) + kk;
            dx[input_index] += gradient * weight[weight_index];
            dw[weight_index] += gradient * input[input_index];
          }
        }
      }
    return 1;
}

static uint32_t vx_conv_transpose2d_valid(uint32_t batch, uint32_t in_height, uint32_t in_width, uint32_t in_channels,
    uint32_t out_height, uint32_t out_width, uint32_t out_channels, uint32_t kernel_y, uint32_t kernel_x,
    uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x) {
  return batch && in_height && in_width && in_channels && out_height && out_width && out_channels && kernel_y && kernel_x && stride_y && stride_x &&
    out_height == (in_height - 1u) * stride_y + kernel_y - 2u * pad_y && out_width == (in_width - 1u) * stride_x + kernel_x - 2u * pad_x;
}

VX_TRAINING_EXPORT("volvoxai_training_conv_transpose2d_f32")
uint32_t volvoxai_training_conv_transpose2d_f32(const float *input, const float *weight, const float *bias, float *output,
    uint32_t batch, uint32_t in_height, uint32_t in_width, uint32_t in_channels, uint32_t out_height, uint32_t out_width,
    uint32_t out_channels, uint32_t kernel_y, uint32_t kernel_x, uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x) {
  if (!input || !weight || !output || !vx_conv_transpose2d_valid(batch,in_height,in_width,in_channels,out_height,out_width,out_channels,kernel_y,kernel_x,stride_y,stride_x,pad_y,pad_x)) return 0;
  uint32_t count=batch*out_height*out_width*out_channels; for(uint32_t i=0;i<count;i++) output[i]=bias?bias[i%out_channels]:0.0f;
  for(uint32_t b=0;b<batch;b++) for(uint32_t iy=0;iy<in_height;iy++) for(uint32_t ix=0;ix<in_width;ix++) for(uint32_t ic=0;ic<in_channels;ic++) for(uint32_t oc=0;oc<out_channels;oc++) for(uint32_t ky=0;ky<kernel_y;ky++) for(uint32_t kx=0;kx<kernel_x;kx++) {
    int32_t oy=(int32_t)(iy*stride_y+ky)-(int32_t)pad_y, ox=(int32_t)(ix*stride_x+kx)-(int32_t)pad_x;
    if(oy>=0 && ox>=0 && (uint32_t)oy<out_height && (uint32_t)ox<out_width) output[((b*out_height+(uint32_t)oy)*out_width+(uint32_t)ox)*out_channels+oc]+=input[((b*in_height+iy)*in_width+ix)*in_channels+ic]*weight[((ic*out_channels+oc)*kernel_y+ky)*kernel_x+kx];
  }
  return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_conv_transpose2d_backward_f32")
uint32_t volvoxai_training_conv_transpose2d_backward_f32(const float *input, const float *weight, const float *dy, float *dx, float *dw, float *db,
    uint32_t batch, uint32_t in_height, uint32_t in_width, uint32_t in_channels, uint32_t out_height, uint32_t out_width,
    uint32_t out_channels, uint32_t kernel_y, uint32_t kernel_x, uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x) {
  if(!input||!weight||!dy||!dx||!dw||!vx_conv_transpose2d_valid(batch,in_height,in_width,in_channels,out_height,out_width,out_channels,kernel_y,kernel_x,stride_y,stride_x,pad_y,pad_x)) return 0;
  for(uint32_t b=0;b<batch;b++) for(uint32_t iy=0;iy<in_height;iy++) for(uint32_t ix=0;ix<in_width;ix++) for(uint32_t ic=0;ic<in_channels;ic++) for(uint32_t oc=0;oc<out_channels;oc++) for(uint32_t ky=0;ky<kernel_y;ky++) for(uint32_t kx=0;kx<kernel_x;kx++) {
    int32_t oy=(int32_t)(iy*stride_y+ky)-(int32_t)pad_y, ox=(int32_t)(ix*stride_x+kx)-(int32_t)pad_x; if(oy<0||ox<0||(uint32_t)oy>=out_height||(uint32_t)ox>=out_width) continue;
    uint32_t ii=((b*in_height+iy)*in_width+ix)*in_channels+ic, wi=((ic*out_channels+oc)*kernel_y+ky)*kernel_x+kx, oi=((b*out_height+(uint32_t)oy)*out_width+(uint32_t)ox)*out_channels+oc; dx[ii]+=dy[oi]*weight[wi]; dw[wi]+=dy[oi]*input[ii]; if(db) db[oc]+=dy[oi];
  }
  return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_pad2d_f32")
uint32_t volvoxai_training_pad2d_f32(const float *input, float *output,
        uint32_t batch, uint32_t in_height, uint32_t in_width, uint32_t channels,
        uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
        uint32_t pad_right, float value) {
    if (!input || !output || !batch || !in_height || !in_width || !channels) return 0;
    uint32_t out_height = in_height + pad_top + pad_bottom;
    uint32_t out_width = in_width + pad_left + pad_right;
    size_t count = (size_t)batch * out_height * out_width * channels;
    for (size_t index = 0; index < count; index++) output[index] = value;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t y = 0; y < in_height; y++)
      for (uint32_t x = 0; x < in_width; x++) for (uint32_t c = 0; c < channels; c++) {
        output[((size_t)(b * out_height + y + pad_top) * out_width + x + pad_left) * channels + c] =
          input[((size_t)(b * in_height + y) * in_width + x) * channels + c];
      }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_pad2d_backward_f32")
uint32_t volvoxai_training_pad2d_backward_f32(const float *dy, float *dx,
        uint32_t batch, uint32_t in_height, uint32_t in_width, uint32_t channels,
        uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
    if (!dy || !dx || !batch || !in_height || !in_width || !channels) return 0;
    uint32_t out_height = in_height + pad_top + pad_bottom;
    uint32_t out_width = in_width + pad_left + pad_right;
    for (uint32_t b = 0; b < batch; b++) for (uint32_t y = 0; y < in_height; y++)
      for (uint32_t x = 0; x < in_width; x++) for (uint32_t c = 0; c < channels; c++) {
        dx[((size_t)(b * in_height + y) * in_width + x) * channels + c] +=
          dy[((size_t)(b * out_height + y + pad_top) * out_width + x + pad_left) * channels + c];
      }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_interp1d_f32")
uint32_t volvoxai_training_interp1d_f32(const float *input, float *output,
        uint32_t batch, uint32_t channels, uint32_t input_length, uint32_t output_length) {
    if (!input || !output || !batch || !channels || !input_length || !output_length) return 0;
    float scale = (float)input_length / (float)output_length;
    for (uint32_t b=0;b<batch;b++) for(uint32_t c=0;c<channels;c++) for(uint32_t x=0;x<output_length;x++) {
      float position=((float)x+0.5f)*scale-0.5f;if(position<0.0f)position=0.0f;if(position>(float)(input_length-1))position=(float)(input_length-1);
      uint32_t x0=(uint32_t)position,x1=x0+1<input_length?x0+1:x0;float fraction=position-(float)x0;size_t base=((size_t)b*channels+c)*input_length;
      output[((size_t)b*channels+c)*output_length+x]=input[base+x0]+fraction*(input[base+x1]-input[base+x0]);
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_interp1d_backward_f32")
uint32_t volvoxai_training_interp1d_backward_f32(const float *dy, float *dx,
        uint32_t batch, uint32_t channels, uint32_t input_length, uint32_t output_length) {
    if(!dy||!dx||!batch||!channels||!input_length||!output_length)return 0;float scale=(float)input_length/(float)output_length;
    for(uint32_t b=0;b<batch;b++)for(uint32_t c=0;c<channels;c++)for(uint32_t x=0;x<output_length;x++){float position=((float)x+0.5f)*scale-0.5f;if(position<0.0f)position=0.0f;if(position>(float)(input_length-1))position=(float)(input_length-1);uint32_t x0=(uint32_t)position,x1=x0+1<input_length?x0+1:x0;float fraction=position-(float)x0,g=dy[((size_t)b*channels+c)*output_length+x];size_t base=((size_t)b*channels+c)*input_length;dx[base+x0]+=g*(1.0f-fraction);dx[base+x1]+=g*fraction;}return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_reduce_backward_f32")
void volvoxai_training_reduce_backward_f32(
        const float *dy, float *dx, uint32_t rows, uint32_t width, float scale) {
    if (!dy || !dx) return;
    for (uint32_t row = 0; row < rows; row++) {
        float gradient = dy[row] * scale;
        size_t offset = (size_t)row * width;
        for (uint32_t column = 0; column < width; column++) dx[offset + column] += gradient;
    }
}

VX_TRAINING_EXPORT("volvoxai_training_copy_backward_f32")
void volvoxai_training_copy_backward_f32(const float *dy, float *dx, uint32_t n) {
    if (!dy || !dx) return;
    for (uint32_t i = 0; i < n; i++) dx[i] += dy[i];
}

VX_TRAINING_EXPORT("volvoxai_training_transpose_backward_f32")
uint32_t volvoxai_training_transpose_backward_f32(
        const float *dy, float *dx, const uint32_t *shape, const uint32_t *perm,
        uint32_t rank, uint32_t elements) {
    if (!dy || !dx || !shape || !perm || rank == 0 || rank > 8) return 0;
    size_t input_strides[8], output_strides[8], product = 1;
    uint32_t seen = 0;
    for (uint32_t axis = 0; axis < rank; axis++) {
        if (shape[axis] == 0 || perm[axis] >= rank || (seen & (1u << perm[axis]))) return 0;
        seen |= 1u << perm[axis];
        if (product > (size_t)-1 / shape[axis]) return 0;
        product *= shape[axis];
    }
    if (product != elements) return 0;
    size_t stride = 1;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        input_strides[reverse] = stride;
        stride *= shape[reverse];
    }
    stride = 1;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        output_strides[reverse] = stride;
        stride *= shape[perm[reverse]];
    }
    for (uint32_t output_index = 0; output_index < elements; output_index++) {
        size_t remaining = output_index, input_index = 0;
        for (uint32_t axis = 0; axis < rank; axis++) {
            size_t coordinate = remaining / output_strides[axis];
            remaining %= output_strides[axis];
            input_index += coordinate * input_strides[perm[axis]];
        }
        dx[input_index] += dy[output_index];
    }
    return 1;
}

VX_TRAINING_EXPORT("volvoxai_training_sgd_update_f32")
void volvoxai_training_sgd_update_f32(
        float *weights, const float *gradient, uint32_t n,
        float learning_rate, float weight_decay) {
    if (!weights || !gradient || !vx_finite_f32(learning_rate) || learning_rate < 0.0f ||
        !vx_finite_f32(weight_decay) || weight_decay < 0.0f) return;
    for (uint32_t i = 0; i < n; i++) {
        if (!vx_finite_f32(weights[i]) || !vx_finite_f32(gradient[i])) return;
        float updated = weights[i] - learning_rate * (gradient[i] + weight_decay * weights[i]);
        if (!vx_finite_f32(updated)) return;
    }
    for (uint32_t i = 0; i < n; i++)
        weights[i] -= learning_rate * (gradient[i] + weight_decay * weights[i]);
}

VX_TRAINING_EXPORT("volvoxai_training_adamw_update_f32")
uint32_t volvoxai_training_adamw_update_f32(
        float *weights, const float *gradient,
        float *first_moment, float *second_moment, uint32_t n,
        float learning_rate, float beta1, float beta2, float epsilon,
        float weight_decay, uint32_t step) {
    if ((!weights || !gradient || !first_moment || !second_moment) && n) return 0;
    if (!vx_finite_f32(learning_rate) || learning_rate < 0.0f ||
        !vx_finite_f32(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !vx_finite_f32(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !vx_finite_f32(epsilon) || !(epsilon > 0.0f) ||
        !vx_finite_f32(weight_decay) || weight_decay < 0.0f || step == 0) return 0;
    float correction1 = 1.0f - powf(beta1, (float)step);
    float correction2 = 1.0f - powf(beta2, (float)step);
    if (!(correction1 > 0.0f) || !(correction2 > 0.0f)) return 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!vx_finite_f32(weights[i]) || !vx_finite_f32(gradient[i]) ||
            !vx_finite_f32(first_moment[i]) || !vx_finite_f32(second_moment[i])) return 0;
        float next_first = beta1 * first_moment[i] + (1.0f - beta1) * gradient[i];
        float next_second = beta2 * second_moment[i] +
                            (1.0f - beta2) * gradient[i] * gradient[i];
        float update = (next_first / correction1) /
                       (__builtin_sqrtf(next_second / correction2) + epsilon);
        float next_weight = weights[i] - learning_rate *
                            (weight_decay * weights[i] + update);
        if (!vx_finite_f32(next_first) || !vx_finite_f32(next_second) ||
            !vx_finite_f32(next_weight)) return 0;
    }
    for (uint32_t i = 0; i < n; i++) {
        first_moment[i] = beta1 * first_moment[i] + (1.0f - beta1) * gradient[i];
        second_moment[i] = beta2 * second_moment[i] +
                           (1.0f - beta2) * gradient[i] * gradient[i];
        float update = (first_moment[i] / correction1) /
                       (__builtin_sqrtf(second_moment[i] / correction2) + epsilon);
        weights[i] -= learning_rate * (weight_decay * weights[i] + update);
    }
    return 1;
}
