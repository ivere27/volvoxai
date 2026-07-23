/*
 * Manual CUDA Driver API/PTX kernels for shape, resize, and pooling backward
 * contracts. This file is included by cuda_training_kernels.cu after the
 * freestanding CUDA macros, vx_u32, and vx_training_global_x are defined.
 *
 * Every Params argument is the device-resident u32 buffer emitted by its
 * native GPU planner. Required launch is block=(64,1,1), with one x group per
 * 64 destination elements. Each lane exclusively owns one destination
 * gradient, allowing deterministic += accumulation without float atomics.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_SHAPE_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_SHAPE_KERNELS_INC

/* Params (34 u32):
 * [rank, total, input_shape[8], perm[8], input_strides[8],
 *  output_strides[8]]. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_transpose_backward_f32(
        const float* grad_output, float* grad_input,
        const vx_u32* params) {
    vx_u32 output_index = vx_training_global_x();
    vx_u32 rank = params[0];
    vx_u32 remainder;
    vx_u32 input_index = 0u;

    if (output_index >= params[1] || rank > 8u) return;
    remainder = output_index;
    for (vx_u32 dimension = 0u; dimension < rank; ++dimension) {
        vx_u32 output_stride = params[26u + dimension];
        vx_u32 coordinate = remainder / output_stride;
        vx_u32 input_dimension = params[10u + dimension];
        remainder %= output_stride;
        input_index += coordinate * params[18u + input_dimension];
    }
    grad_input[input_index] += grad_output[output_index];
}

/* Params (8 u32):
 * [length, axis_offset, input_axis, output_axis, inner, pad x3]. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_concat_backward_f32(
        const float* grad_output, float* grad_input,
        const vx_u32* params) {
    vx_u32 index = vx_training_global_x();
    vx_u32 inner;
    vx_u32 outer_axis;
    vx_u32 axis_index;
    vx_u32 outer_index;
    vx_u32 output_index;

    if (index >= params[0]) return;
    inner = params[4];
    outer_axis = index / inner;
    axis_index = outer_axis % params[2];
    outer_index = outer_axis / params[2];
    output_index = ((outer_index * params[3] + params[1] + axis_index) *
        inner) + index % inner;
    grad_input[index] += grad_output[output_index];
}

/* Params (5 u32):
 * [total, inner, split_size, input_axis, axis_offset]. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_split_backward_f32(
        const float* grad_output, float* grad_input,
        const vx_u32* params) {
    vx_u32 index = vx_training_global_x();
    vx_u32 block;
    vx_u32 outer;
    vx_u32 within;
    vx_u32 input_index;

    if (index >= params[0]) return;
    block = params[2] * params[1];
    outer = index / block;
    within = index - outer * block;
    input_index = (outer * params[3] + params[4]) * params[1] + within;
    grad_input[input_index] += grad_output[index];
}

/* Params (8 u32): [n, h, w, c, out_h, out_w, mode, pad].
 * mode 0 is nearest-neighbor and every other value follows the WGSL
 * half-pixel bilinear branch. One lane owns one input pixel/channel and scans
 * all output locations that may contribute to it. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_resize_backward_f32(
        const float* grad_output, float* grad_input,
        const vx_u32* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 count = params[0] * params[1] * params[2] * params[3];
    vx_u32 temporary;
    vx_u32 channel;
    vx_u32 input_x;
    vx_u32 input_y;
    vx_u32 batch;
    float sum = 0.0f;

    if (flat >= count) return;
    temporary = flat;
    channel = temporary % params[3];
    temporary /= params[3];
    input_x = temporary % params[2];
    temporary /= params[2];
    input_y = temporary % params[1];
    batch = temporary / params[1];

    for (vx_u32 output_y = 0u; output_y < params[4]; ++output_y) {
        for (vx_u32 output_x = 0u; output_x < params[5]; ++output_x) {
            float coefficient = 0.0f;
            if (params[6] == 0u) {
                vx_u32 source_y = output_y * params[1] / params[4];
                vx_u32 source_x = output_x * params[2] / params[5];
                if (source_y == input_y && source_x == input_x)
                    coefficient = 1.0f;
            } else {
                float scale_y = (float)params[1] / (float)params[4];
                float scale_x = (float)params[2] / (float)params[5];
                float fy = ((float)output_y + 0.5f) * scale_y - 0.5f;
                float fx = ((float)output_x + 0.5f) * scale_x - 0.5f;
                vx_u32 y0;
                vx_u32 x0;
                vx_u32 y1;
                vx_u32 x1;
                float delta_y;
                float delta_x;

                if (fy < 0.0f) fy = 0.0f;
                if (fx < 0.0f) fx = 0.0f;
                y0 = (vx_u32)fy;
                x0 = (vx_u32)fx;
                y1 = y0 + 1u < params[1] ? y0 + 1u : params[1] - 1u;
                x1 = x0 + 1u < params[2] ? x0 + 1u : params[2] - 1u;
                delta_y = fy - (float)y0;
                delta_x = fx - (float)x0;
                if (input_y == y0 && input_x == x0)
                    coefficient += (1.0f - delta_y) * (1.0f - delta_x);
                if (input_y == y0 && input_x == x1)
                    coefficient += (1.0f - delta_y) * delta_x;
                if (input_y == y1 && input_x == x0)
                    coefficient += delta_y * (1.0f - delta_x);
                if (input_y == y1 && input_x == x1)
                    coefficient += delta_y * delta_x;
            }
            vx_u32 output_index =
                ((batch * params[4] + output_y) * params[5] + output_x) *
                    params[3] + channel;
            sum += grad_output[output_index] * coefficient;
        }
    }
    grad_input[flat] += sum;
}

/* Pooling Params (12 u32):
 * [n, h, w, c, out_h, out_w, ky, kx, sy, sx, py, px]. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_pooling_backward_global_average_f32(
        const float* input, const float* grad_output, float* grad_input,
        const vx_u32* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 spatial = params[1] * params[2];
    vx_u32 batch_stride = spatial * params[3];
    vx_u32 count = params[0] * batch_stride;

    (void)input;
    if (flat >= count) return;
    vx_u32 channel = flat % params[3];
    vx_u32 batch = flat / batch_stride;
    grad_input[flat] += grad_output[batch * params[3] + channel] /
        (float)spatial;
}

/* Each lane owns one input gradient and recomputes every output window in
 * oy/ox/yy/xx order. A candidate replaces the winner only on strict greater-
 * than, so equal maxima deterministically route to the first valid element,
 * exactly matching poolingBackward.wgsl. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_pooling_backward_max_pool_f32(
        const float* input, const float* grad_output, float* grad_input,
        const vx_u32* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 spatial = params[1] * params[2];
    vx_u32 batch_stride = spatial * params[3];
    vx_u32 count = params[0] * batch_stride;
    vx_u32 batch;
    vx_u32 local;
    vx_u32 input_base;
    vx_u32 output_base;
    vx_u32 channel;
    float sum = 0.0f;

    if (flat >= count) return;
    batch = flat / batch_stride;
    local = flat % batch_stride;
    input_base = batch * batch_stride;
    output_base = batch * params[4] * params[5] * params[3];
    channel = local % params[3];

    for (vx_u32 output_y = 0u; output_y < params[4]; ++output_y) {
        for (vx_u32 output_x = 0u; output_x < params[5]; ++output_x) {
            float best = -3.402823466e38f;
            vx_u32 best_index = 0xffffffffu;
            for (vx_u32 kernel_y = 0u; kernel_y < params[6]; ++kernel_y) {
                int input_y = (int)(output_y * params[8] + kernel_y) -
                    (int)params[10];
                if (input_y < 0 || input_y >= (int)params[1]) continue;
                for (vx_u32 kernel_x = 0u; kernel_x < params[7]; ++kernel_x) {
                    int input_x = (int)(output_x * params[9] + kernel_x) -
                        (int)params[11];
                    vx_u32 candidate;
                    float value;
                    if (input_x < 0 || input_x >= (int)params[2]) continue;
                    candidate = input_base +
                        ((vx_u32)input_y * params[2] + (vx_u32)input_x) *
                            params[3] + channel;
                    value = input[candidate];
                    if (best_index == 0xffffffffu || value > best) {
                        best = value;
                        best_index = candidate;
                    }
                }
            }
            if (best_index == flat) {
                sum += grad_output[output_base +
                    (output_y * params[5] + output_x) * params[3] + channel];
            }
        }
    }
    grad_input[flat] += sum;
}

#endif
