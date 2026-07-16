#include "mathcompat.h"
#include <stdint.h>
// --- Missing CV & NLP Primitives (Batch 2) ---
void prelu_f32(const float* input, const float* weight, float* output, int b, int h, int w, int c) {
    int spatial = h * w;
    for (int i_b = 0; i_b < b; i_b++) {
        for (int i = 0; i < spatial; i++) {
            for (int i_c = 0; i_c < c; i_c++) {
                float alpha = weight[i_c];
                long idx = ((long)i_b * spatial + i) * c + i_c;
                float v = input[idx];
                output[idx] = v > 0.0f ? v : v * alpha;
            }
        }
    }
}
void logsoftmax_f32(const float* input, float* output, int b, int d) {
    if (!input || !output || b <= 0 || d <= 0) return;
    for (int i = 0; i < b; i++) {
        float max_val = input[i*d];
        for (int j = 1; j < d; j++) {
            if (input[i*d + j] > max_val) max_val = input[i*d + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += accurate_expf(input[i*d + j] - max_val);
        }
        float log_sum = max_val + logf(sum);
        for (int j = 0; j < d; j++) output[i*d + j] = input[i*d + j] - log_sum;
    }
}

void reduce_mean_f32(const float* input, float* output, int b, int d) {
    for (int i = 0; i < b; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += input[i*d + j];
        output[i] = sum / (float)d;
    }
}
void reduce_sum_f32(const float* input, float* output, int b, int d) {
    for (int i = 0; i < b; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += input[i*d + j];
        output[i] = sum;
    }
}

/* Keep these private to the freestanding kernel bundle.  The values match the
 * dtype codes selected by WasmEngine; they are not part of a native public API. */
enum {
    VX_KERNEL_DTYPE_F32 = 0,
    VX_KERNEL_DTYPE_I32 = 1,
    VX_KERNEL_DTYPE_I8 = 2,
    VX_KERNEL_DTYPE_U8 = 3,
};

static double vx_kernel_value_at(const void* values, int dtype, long index) {
    switch (dtype) {
        case VX_KERNEL_DTYPE_F32: return ((const float*)values)[index];
        case VX_KERNEL_DTYPE_I32: return ((const int32_t*)values)[index];
        case VX_KERNEL_DTYPE_I8: return ((const int8_t*)values)[index];
        case VX_KERNEL_DTYPE_U8: return ((const uint8_t*)values)[index];
        default: return 0.0;
    }
}

static void vx_kernel_store_index(void* values, int dtype, long index, int value) {
    switch (dtype) {
        case VX_KERNEL_DTYPE_F32:
            ((float*)values)[index] = (float)value;
            break;
        case VX_KERNEL_DTYPE_I32:
            ((int32_t*)values)[index] = (int32_t)value;
            break;
        case VX_KERNEL_DTYPE_I8:
        case VX_KERNEL_DTYPE_U8:
            /* Store the byte representation directly so Int8Array and
             * Uint8Array retain JavaScript TypedArray wrapping semantics. */
            ((uint8_t*)values)[index] = (uint8_t)value;
            break;
    }
}

/* General ONNX-style ArgMax indexing: [outer, axis, inner].  Values are read
 * and indices are written with the tensor's actual WASM storage dtype. */
void argmax_axis_typed(const void* input, void* output, int outer, int axis_size,
                       int inner, int input_dtype, int output_dtype) {
    if (!input || !output || outer <= 0 || axis_size <= 0 || inner <= 0 ||
        input_dtype < VX_KERNEL_DTYPE_F32 || input_dtype > VX_KERNEL_DTYPE_U8 ||
        output_dtype < VX_KERNEL_DTYPE_F32 || output_dtype > VX_KERNEL_DTYPE_U8) return;
    for (int outer_index = 0; outer_index < outer; outer_index++) {
        for (int inner_index = 0; inner_index < inner; inner_index++) {
            long base = ((long)outer_index * axis_size * inner) + inner_index;
            double best = vx_kernel_value_at(input, input_dtype, base);
            int best_index = 0;
            for (int axis_index = 1; axis_index < axis_size; axis_index++) {
                double value = vx_kernel_value_at(input, input_dtype,
                    base + (long)axis_index * inner);
                /* Strict comparison deliberately preserves the first index on
                 * ties and follows JavaScript's NaN comparison behavior. */
                if (value > best) {
                    best = value;
                    best_index = axis_index;
                }
            }
            vx_kernel_store_index(output, output_dtype,
                (long)outer_index * inner + inner_index, best_index);
        }
    }
}

void argmax_f32(const float* input, float* output, int b, int d) {
    argmax_axis_typed(input, output, b, d, 1,
        VX_KERNEL_DTYPE_F32, VX_KERNEL_DTYPE_F32);
}

static double vx_kernel_max(double left, double right) {
    /* Math.max returns NaN when either input is NaN. */
    if (left != left || right != right) return left + right;
    return left > right ? left : right;
}

static double vx_kernel_min(double left, double right) {
    /* Math.min returns NaN when either input is NaN. */
    if (left != left || right != right) return left + right;
    return left < right ? left : right;
}

static double vx_kernel_non_max_iou(const void* boxes, int boxes_dtype,
                                    int spatial, int batch, int first, int second) {
    long first_base = ((long)batch * spatial + first) * 4;
    long second_base = ((long)batch * spatial + second) * 4;
    double first_y1 = vx_kernel_value_at(boxes, boxes_dtype, first_base + 0);
    double first_x1 = vx_kernel_value_at(boxes, boxes_dtype, first_base + 1);
    double first_y2 = vx_kernel_value_at(boxes, boxes_dtype, first_base + 2);
    double first_x2 = vx_kernel_value_at(boxes, boxes_dtype, first_base + 3);
    double second_y1 = vx_kernel_value_at(boxes, boxes_dtype, second_base + 0);
    double second_x1 = vx_kernel_value_at(boxes, boxes_dtype, second_base + 1);
    double second_y2 = vx_kernel_value_at(boxes, boxes_dtype, second_base + 2);
    double second_x2 = vx_kernel_value_at(boxes, boxes_dtype, second_base + 3);
    double xx1 = vx_kernel_max(first_x1, second_x1);
    double yy1 = vx_kernel_max(first_y1, second_y1);
    double xx2 = vx_kernel_min(first_x2, second_x2);
    double yy2 = vx_kernel_min(first_y2, second_y2);
    double width = vx_kernel_max(0.0, xx2 - xx1);
    double height = vx_kernel_max(0.0, yy2 - yy1);
    double intersection = width * height;
    double first_area = (first_x2 - first_x1) * (first_y2 - first_y1);
    double second_area = (second_x2 - second_x1) * (second_y2 - second_y1);
    return intersection / (first_area + second_area - intersection);
}

/* Allocation-free greedy NMS.  Selected indices are retained in the output
 * rows while a class is being processed, avoiding a runtime heap dependency. */
void non_max_suppression_typed(const void* boxes, const void* scores, void* output,
                               int batches, int spatial, int classes, int output_records,
                               int boxes_dtype, int scores_dtype, int output_dtype,
                               double max_output_boxes_per_class, double iou_threshold,
                               double score_threshold) {
    if (!boxes || !scores || !output || batches < 0 || spatial < 0 || classes < 0 ||
        output_records < 0 || boxes_dtype < VX_KERNEL_DTYPE_F32 ||
        boxes_dtype > VX_KERNEL_DTYPE_U8 || scores_dtype < VX_KERNEL_DTYPE_F32 ||
        scores_dtype > VX_KERNEL_DTYPE_U8 || output_dtype < VX_KERNEL_DTYPE_F32 ||
        output_dtype > VX_KERNEL_DTYPE_U8) return;

    int output_index = 0;
    for (int batch = 0; batch < batches && output_index < output_records; batch++) {
        for (int class_index = 0; class_index < classes && output_index < output_records;
             class_index++) {
            int class_start = output_index;
            for (int selected = 0;
                 (double)selected < max_output_boxes_per_class && output_index < output_records;
                 selected++) {
                int best = -1;
                double best_score = 0.0;
                for (int candidate = 0; candidate < spatial; candidate++) {
                    long score_index = ((long)batch * classes + class_index) * spatial + candidate;
                    double score = vx_kernel_value_at(scores, scores_dtype, score_index);
                    if (!(score >= score_threshold)) continue;
                    int suppressed = 0;
                    for (int prior = 0; prior < selected; prior++) {
                        int previous = (int)vx_kernel_value_at(output, output_dtype,
                            ((long)(class_start + prior) * 3) + 2);
                        if (vx_kernel_non_max_iou(boxes, boxes_dtype, spatial, batch,
                                candidate, previous) > iou_threshold) {
                            suppressed = 1;
                            break;
                        }
                    }
                    /* The source candidate order is ascending spatial index;
                     * not replacing equal scores reproduces stable JS sort. */
                    if (!suppressed && (best < 0 || score > best_score)) {
                        best = candidate;
                        best_score = score;
                    }
                }
                if (best < 0) break;
                vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 0, batch);
                vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 1, class_index);
                vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 2, best);
                output_index++;
            }
        }
    }
    while (output_index < output_records) {
        vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 0, -1);
        vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 1, -1);
        vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 2, -1);
        output_index++;
    }
}
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
