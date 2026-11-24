/* Immutable 16-bit weights are decoded into bounded working storage. They
 * never enter the persistent F32 conversion or GEMM packing caches. */
typedef struct {
    const float* input;
    const T* weight;
    const float* bias;
    float* output;
    int rows, d_in, d_out, out_in;
} VxCompactLinearTask;

static void vx_compact_linear_task(void* opaque, int begin, int end) {
    const VxCompactLinearTask* task = opaque;
    float panel[32 * 8];
    for (int tile = begin; tile < end; ++tile) {
        int column = tile * 8;
        int columns = task->d_out - column;
        if (columns > 8) columns = 8;
        for (int row = 0; row < task->rows; ++row)
            for (int j = 0; j < columns; ++j)
                task->output[(size_t)row * task->d_out + column + j] =
                    task->bias ? task->bias[column + j] : 0.0f;
        for (int first = 0; first < task->d_in;) {
            int width = task->d_in - first;
            if (width > 32) width = 32;
            for (int k = 0; k < width; ++k)
                for (int j = 0; j < columns; ++j) {
                    size_t index = task->out_in
                        ? (size_t)(column + j) * (size_t)task->d_in + (size_t)(first + k)
                        : (size_t)(first + k) * (size_t)task->d_out + (size_t)(column + j);
                    panel[k * 8 + j] = tensor_scalar_f32(task->weight, index);
                }
            for (int row = 0; row < task->rows; ++row) {
                float* output = task->output + (size_t)row * task->d_out + column;
                const float* input = task->input + (size_t)row * task->d_in + first;
                for (int k = 0; k < width; ++k)
                    for (int j = 0; j < columns; ++j)
                        output[j] += input[k] * panel[k * 8 + j];
            }
            first += width;
        }
    }
}

static int vx_compact_linear_f32(const float* input, const T* weight,
        const float* bias, float* output, int rows, int d_in, int d_out, int out_in) {
    if (!input || !output || !tensor_float_lookup_storage(weight) || rows < 0 ||
        d_in <= 0 || d_out <= 0 || (out_in != 0 && out_in != 1) ||
        (uint64_t)d_in * d_out != (uint64_t)weight->numel) return -1;
    VxCompactLinearTask task = {input, weight, bias, output, rows, d_in, d_out, out_in};
    int tiles = d_out / 8 + (d_out % 8 != 0);
    if (vx_kernels_thread_count() > 1 && tiles > 1 &&
        (uint64_t)rows * d_in * d_out >= 65536u)
        vx_kernels_parallel_for(tiles, 1, vx_compact_linear_task, &task);
    else vx_compact_linear_task(&task, 0, tiles);
    return 0;
}

static int vx_compact_moe_expert(const T* indices, size_t index, uint32_t experts,
        const uint32_t* slots, uint32_t domain, uint32_t* expert) {
    uint32_t extent = slots ? domain : experts;
    if (!indices || indices->numel < 0 || index >= (size_t)indices->numel || !extent) return 0;
    if (indices->dtype == T_I32) {
        int32_t value = ((const int32_t*)indices->data)[index];
        if (value < 0 || (uint32_t)value >= extent) return 0;
        *expert = (uint32_t)value;
    } else if (indices->dtype == T_F32) {
        float value = indices->data[index];
        if (!isfinite(value) || value < 0 || (double)value >= extent ||
            floorf(value) != value) return 0;
        *expert = (uint32_t)value;
    } else return 0;
    if (slots) *expert = slots[*expert];
    return *expert < experts;
}

static int vx_compact_moe_linear_f32(const T* input, const T* weight,
        const T* bias, const T* indices, const T* routes, T* output,
        uint32_t rows, uint32_t d_in, uint32_t d_out, uint32_t experts,
        uint32_t top_k, const uint32_t* slots, uint32_t domain) {
    if (indices->numel < 0) return -1;
    for (size_t route = 0; route < (size_t)indices->numel; ++route) {
        uint32_t expert;
        if (!vx_compact_moe_expert(indices, route, experts, slots, domain, &expert) ||
            !isfinite(routes->data[route])) return -1;
    }
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < d_out; ++column) {
            double sum = 0.0;
            for (uint32_t slot = 0; slot < top_k; ++slot) {
                size_t route = (size_t)row * top_k + slot;
                uint32_t expert;
                if (!vx_compact_moe_expert(indices, route, experts, slots, domain, &expert)) return -1;
                double value = bias ? bias->data[(size_t)expert * d_out + column] : 0.0;
                for (uint32_t k = 0; k < d_in; ++k)
                    value += (double)input->data[(size_t)row * d_in + k] *
                        tensor_scalar_f32(weight, ((size_t)expert * d_in + k) * d_out + column);
                sum += routes->data[route] * value;
            }
            output->data[(size_t)row * d_out + column] = (float)sum;
        }
    return 0;
}

static int vx_compact_moe_gated_f32(const T* input, const T* gate_weight,
        const T* up_weight, const T* down_weight, const T* indices,
        const T* routes, T* output, uint32_t rows, uint32_t d_in,
        uint32_t hidden, uint32_t d_out, uint32_t experts, uint32_t top_k,
        uint32_t input_scale, float limit) {
    if (!top_k || top_k > experts || input_scale > 1 || !isfinite(limit) ||
        limit < 0 || indices->numel < 0 || (uint64_t)hidden * sizeof(float) > SIZE_MAX) return -1;
    for (size_t route = 0; route < (size_t)indices->numel; ++route) {
        uint32_t expert;
        if (!vx_compact_moe_expert(indices, route, experts, NULL, 0, &expert) ||
            !isfinite(routes->data[route])) return -1;
    }
    float* intermediate = malloc((size_t)hidden * sizeof(float));
    if (!intermediate) return -1;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t o = 0; o < d_out; ++o) output->data[(size_t)row * d_out + o] = 0;
        for (uint32_t k = 0; k < top_k; ++k) {
            size_t route_index = (size_t)row * top_k + k;
            uint32_t expert;
            if (!vx_compact_moe_expert(indices, route_index, experts, NULL, 0, &expert)) {
                free(intermediate);
                return -1;
            }
            float route = routes->data[route_index];
            for (uint32_t h = 0; h < hidden; ++h) {
                double gate = 0.0, up = 0.0;
                for (uint32_t d = 0; d < d_in; ++d) {
                    float x = input->data[(size_t)row * d_in + d] * (input_scale ? route : 1.0f);
                    size_t offset = ((size_t)expert * d_in + d) * hidden + h;
                    gate += (double)x * tensor_scalar_f32(gate_weight, offset);
                    up += (double)x * tensor_scalar_f32(up_weight, offset);
                }
                if (limit > 0) {
                    if (gate > limit) gate = limit;
                    if (up > limit) up = limit;
                    if (up < -limit) up = -limit;
                }
                intermediate[h] = (float)((gate / (1.0 + expf((float)-gate))) * up);
            }
            for (uint32_t o = 0; o < d_out; ++o) {
                double value = 0.0;
                for (uint32_t h = 0; h < hidden; ++h)
                    value += (double)intermediate[h] * tensor_scalar_f32(down_weight,
                        ((size_t)expert * hidden + h) * d_out + o);
                output->data[(size_t)row * d_out + o] += (float)((input_scale ? 1.0 : route) * value);
            }
        }
    }
    free(intermediate);
    return 0;
}
