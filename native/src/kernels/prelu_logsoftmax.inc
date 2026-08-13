#include "mathcompat.h"
#include "thread_pool.h"
#include "../../include/volvoxai_enums.h"
#include <stddef.h>
#include <stdint.h>
/* PReLU and LogSoftmax. */
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
typedef struct VxLogSoftmaxF32RowsContext {
    const float* input;
    float* output;
    int width;
} VxLogSoftmaxF32RowsContext;

static void vx_logsoftmax_f32_rows(void* opaque, int begin, int end) {
    VxLogSoftmaxF32RowsContext* context =
        (VxLogSoftmaxF32RowsContext*)opaque;
    const int d = context->width;
    for (int i = begin; i < end; i++) {
        const size_t offset = (size_t)i * (size_t)d;
        const float* input = context->input + offset;
        float* output = context->output + offset;
        float max_val = input[0];
        for (int j = 1; j < d; j++) {
            if (input[j] > max_val) max_val = input[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += accurate_expf(input[j] - max_val);
        }
        float log_sum = max_val + logf(sum);
        for (int j = 0; j < d; j++) output[j] = input[j] - log_sum;
    }
}

void logsoftmax_f32(const float* input, float* output, int b, int d) {
    VxLogSoftmaxF32RowsContext context;
    int threads;
    int grain;
    if (!input || !output || b <= 0 || d <= 0) return;
    context.input = input;
    context.output = output;
    context.width = d;
    threads = vx_kernels_thread_count();
    if (threads <= 1 || b <= 1 ||
        (uint64_t)(uint32_t)b * (uint64_t)(uint32_t)d < 65536u) {
        vx_logsoftmax_f32_rows(&context, 0, b);
        return;
    }
    {
        const uint64_t target_chunks = (uint64_t)(uint32_t)threads * 4u;
        grain = (int)((uint64_t)(uint32_t)b / target_chunks +
                      ((uint64_t)(uint32_t)b % target_chunks != 0u));
    }
    if (grain < 1) grain = 1;
    vx_kernels_parallel_for(b, grain, vx_logsoftmax_f32_rows, &context);
}

