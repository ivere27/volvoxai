extern "C" VX_CUDA_GLOBAL void vx_cuda_spatial_softargmax_y_f32(
        const float* input, float* output, vx_u32 batch, vx_u32 height,
        vx_u32 width, vx_u32 channels) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * channels * width;
    if (index >= elements) return;
    vx_u32 x = index % width;
    vx_u32 row = index / width;
    vx_u32 channel = row % channels;
    vx_u32 batch_index = row / channels;
    vx_u32 input_base = batch_index * height * width * channels;
    float maximum = input[input_base + x * channels + channel];
    for (vx_u32 y = 1u; y < height; y++) {
        float value = input[input_base + (y * width + x) * channels + channel];
        if (value > maximum) maximum = value;
    }
    float denominator = 0.0f;
    float weighted = 0.0f;
    for (vx_u32 y = 0u; y < height; y++) {
        float value = input[input_base + (y * width + x) * channels + channel];
        float exponential = vx_exp(value - maximum);
        denominator += exponential;
        weighted += exponential * ((float)y + 0.5f) / (float)height;
    }
    output[index] = denominator > 0.0f ? weighted / denominator : 0.0f;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_profile_x_f32(
        const float* input, float* output, vx_u32 batch, vx_u32 height,
        vx_u32 width, vx_u32 channels) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * channels * width;
    if (index >= elements) return;
    vx_u32 x = index % width;
    vx_u32 row = index / width;
    vx_u32 channel = row % channels;
    vx_u32 batch_index = row / channels;
    vx_u32 input_base = batch_index * height * width * channels;
    float first = input[input_base + x * channels + channel];
    float maximum = first;
    float sum = first;
    for (vx_u32 y = 1u; y < height; y++) {
        float value = input[input_base + (y * width + x) * channels + channel];
        if (value > maximum) maximum = value;
        sum += value;
    }
    vx_u32 output_base = batch_index * 2u * channels * width;
    vx_u32 output_index = channel * width + x;
    output[output_base + output_index] = maximum;
    output[output_base + channels * width + output_index] =
        sum / (float)height;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_profile_y_f32(
        const float* input, float* output, vx_u32 batch, vx_u32 height,
        vx_u32 width, vx_u32 channels) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * channels * height;
    if (index >= elements) return;
    vx_u32 y = index % height;
    vx_u32 row = index / height;
    vx_u32 channel = row % channels;
    vx_u32 batch_index = row / channels;
    vx_u32 input_base = batch_index * height * width * channels;
    float first = input[input_base + y * width * channels + channel];
    float maximum = first;
    float sum = first;
    for (vx_u32 x = 1u; x < width; x++) {
        float value = input[input_base + (y * width + x) * channels + channel];
        if (value > maximum) maximum = value;
        sum += value;
    }
    vx_u32 output_base = batch_index * 2u * channels * height;
    vx_u32 output_index = channel * height + y;
    output[output_base + output_index] = maximum;
    output[output_base + channels * height + output_index] =
        sum / (float)width;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_mean_height_f32(
        const float* input, float* output, vx_u32 batch, vx_u32 height,
        vx_u32 width, vx_u32 channels) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * channels * width;
    if (index >= elements) return;
    vx_u32 x = index % width;
    vx_u32 row = index / width;
    vx_u32 channel = row % channels;
    vx_u32 batch_index = row / channels;
    vx_u32 input_base = batch_index * height * width * channels;
    float sum = 0.0f;
    for (vx_u32 y = 0u; y < height; y++)
        sum += input[input_base + (y * width + x) * channels + channel];
    output[index] = sum / (float)height;
}

static VX_CUDA_DEVICE float vx_nms_iou(
        const float* boxes, vx_u32 spatial, vx_u32 batch_index,
        vx_u32 first, vx_u32 second) {
    vx_u32 first_base = (batch_index * spatial + first) * 4u;
    vx_u32 second_base = (batch_index * spatial + second) * 4u;
    float first_y1 = boxes[first_base];
    float first_x1 = boxes[first_base + 1u];
    float first_y2 = boxes[first_base + 2u];
    float first_x2 = boxes[first_base + 3u];
    float second_y1 = boxes[second_base];
    float second_x1 = boxes[second_base + 1u];
    float second_y2 = boxes[second_base + 2u];
    float second_x2 = boxes[second_base + 3u];
    float intersection_width = vx_max(0.0f,
        vx_min(first_x2, second_x2) - vx_max(first_x1, second_x1));
    float intersection_height = vx_max(0.0f,
        vx_min(first_y2, second_y2) - vx_max(first_y1, second_y1));
    float intersection = intersection_width * intersection_height;
    float first_area = vx_max(0.0f, first_x2 - first_x1) *
        vx_max(0.0f, first_y2 - first_y1);
    float second_area = vx_max(0.0f, second_x2 - second_x1) *
        vx_max(0.0f, second_y2 - second_y1);
    float denominator = first_area + second_area - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_nms_f32(
        const float* boxes, const float* scores, float* output,
        vx_u32 batches, vx_u32 spatial, vx_u32 classes,
        vx_u32 max_output, vx_u32 output_rows,
        float iou_threshold, float score_threshold) {
    if (vx_global_x() != 0u) return;
    for (vx_u32 row = 0u; row < output_rows; row++) {
        output[row * 3u] = -1.0f;
        output[row * 3u + 1u] = -1.0f;
        output[row * 3u + 2u] = -1.0f;
    }
    vx_u32 output_row = 0u;
    for (vx_u32 batch_index = 0u; batch_index < batches; batch_index++) {
        for (vx_u32 class_index = 0u; class_index < classes; class_index++) {
            vx_u32 class_start = output_row;
            vx_u32 selected_count = 0u;
            while (selected_count < max_output && output_row < output_rows) {
                float best_score = score_threshold;
                vx_u32 best_spatial = spatial;
                for (vx_u32 candidate = 0u; candidate < spatial; candidate++) {
                    float score = scores[(batch_index * classes + class_index) *
                        spatial + candidate];
                    if (score < best_score) continue;
                    vx_u32 suppressed = 0u;
                    for (vx_u32 previous = 0u; previous < selected_count;
                         previous++) {
                        vx_u32 previous_row = class_start + previous;
                        if (previous_row >= output_rows) break;
                        vx_u32 previous_batch =
                            (vx_u32)output[previous_row * 3u];
                        vx_u32 previous_class =
                            (vx_u32)output[previous_row * 3u + 1u];
                        vx_u32 previous_spatial =
                            (vx_u32)output[previous_row * 3u + 2u];
                        if (previous_batch == batch_index &&
                            previous_class == class_index &&
                            vx_nms_iou(boxes, spatial, batch_index,
                                       candidate, previous_spatial) >
                                iou_threshold) {
                            suppressed = 1u;
                            break;
                        }
                    }
                    if (suppressed != 0u) continue;
                    if (best_spatial == spatial || score > best_score) {
                        best_score = score;
                        best_spatial = candidate;
                    }
                }
                if (best_spatial == spatial) break;
                output[output_row * 3u] = (float)batch_index;
                output[output_row * 3u + 1u] = (float)class_index;
                output[output_row * 3u + 2u] = (float)best_spatial;
                output_row++;
                selected_count++;
            }
        }
    }
}
