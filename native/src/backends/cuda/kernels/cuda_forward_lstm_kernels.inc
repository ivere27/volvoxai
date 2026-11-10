extern "C" VX_CUDA_GLOBAL void vx_cuda_lstm_f32(const float* input,
        const float* weight, const float* recurrent, const float* bias,
        const float* initial_h, const float* initial_c, float* output,
        vx_u32 time, vx_u32 batch, vx_u32 width, vx_u32 hidden,
        vx_u32 direction, vx_u32 flags) {
    VX_CUDA_SHARED float h[1024];
    VX_CUDA_SHARED float c[1024];
    VX_CUDA_SHARED float next_h[1024];
    vx_u32 directions = direction == 2u ? 2u : 1u;
    vx_u32 d = blockIdx.x / batch, b = blockIdx.x % batch;
    vx_u32 state_base = (d * batch + b) * hidden;
    for (vx_u32 j = threadIdx.x; j < hidden; j += 256u) {
        h[j] = flags & 2u ? initial_h[state_base + j] : 0.0f;
        c[j] = flags & 4u ? initial_c[state_base + j] : 0.0f;
    }
    vx_sync_threads();
    for (vx_u32 step = 0; step < time; ++step) {
        vx_u32 t = direction == 1u || d == 1u ? time - 1u - step : step;
        for (vx_u32 j = threadIdx.x; j < hidden; j += 256u) {
            float gates[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (vx_u32 gate = 0; gate < 4u; ++gate) {
                float sum = 0.0f, correction = 0.0f;
                vx_u32 row = d * 4u * hidden + gate * hidden + j;
                for (vx_u32 k = 0; k < width; ++k) {
                    float product = vx_mul_rn(input[(t * batch + b) * width + k], weight[row * width + k]);
                    float adjusted = vx_add_rn(product, -correction);
                    float total = vx_add_rn(sum, adjusted);
                    correction = vx_add_rn(vx_add_rn(total, -sum), -adjusted);
                    sum = total;
                }
                for (vx_u32 k = 0; k < hidden; ++k) {
                    float product = vx_mul_rn(h[k], recurrent[row * hidden + k]);
                    float adjusted = vx_add_rn(product, -correction);
                    float total = vx_add_rn(sum, adjusted);
                    correction = vx_add_rn(vx_add_rn(total, -sum), -adjusted);
                    sum = total;
                }
                if (flags & 1u) sum += bias[d * 8u * hidden + gate * hidden + j] +
                    bias[d * 8u * hidden + (gate + 4u) * hidden + j];
                gates[gate] = gate == 3u ? vx_tanh(sum) : 1.0f / (1.0f + vx_exp(-sum));
            }
            c[j] = vx_add_rn(vx_mul_rn(gates[2], c[j]), vx_mul_rn(gates[0], gates[3]));
            next_h[j] = vx_mul_rn(gates[1], vx_tanh(c[j]));
        }
        vx_sync_threads();
        for (vx_u32 j = threadIdx.x; j < hidden; j += 256u) {
            h[j] = next_h[j];
            output[((t * directions + d) * batch + b) * hidden + j] = next_h[j];
        }
        vx_sync_threads();
    }
}

/* Transposition is arithmetic-free and performed on the live weights for each
 * request. Consecutive gate rows then read consecutive addresses at each k. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_lstm_transpose_f32(const float* input,
        float* output, vx_u32 rows, vx_u32 columns) {
    VX_CUDA_SHARED float tile[32][33];
    vx_u32 x = blockIdx.x * 32u + threadIdx.x;
    vx_u32 y = blockIdx.y * 32u + threadIdx.y;
    vx_u32 base = blockIdx.z * rows * columns;
    for (vx_u32 offset = 0; offset < 32u; offset += 8u)
        if (x < columns && y + offset < rows)
            tile[threadIdx.y + offset][threadIdx.x] = input[base + (y + offset) * columns + x];
    vx_sync_threads();
    x = blockIdx.y * 32u + threadIdx.x;
    y = blockIdx.x * 32u + threadIdx.y;
    for (vx_u32 offset = 0; offset < 32u; offset += 8u)
        if (x < rows && y + offset < columns)
            output[base + (y + offset) * rows + x] = tile[threadIdx.x][threadIdx.y + offset];
}

/* Preserve both Kahan values at the input/recurrent boundary: storing only
 * sum loses the final input correction and changes the recurrent reduction. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_lstm_project_f32(const float* input,
        const float* weight, float* projection, vx_u32 time, vx_u32 batch,
        vx_u32 width, vx_u32 hidden, vx_u32 directions) {
    vx_u32 index = vx_global_x();
    vx_u32 gate_width = 4u * hidden;
    vx_u32 count = time * directions * batch * gate_width;
    if (index >= count) return;
    vx_u32 row = index % gate_width;
    vx_u32 b = index / gate_width % batch;
    vx_u32 d = index / gate_width / batch % directions;
    vx_u32 t = index / gate_width / batch / directions;
    float sum = 0.0f, correction = 0.0f;
    for (vx_u32 k = 0; k < width; ++k) {
        float product = vx_mul_rn(input[(t * batch + b) * width + k],
            weight[(d * width + k) * gate_width + row]);
        float adjusted = vx_add_rn(product, -correction);
        float total = vx_add_rn(sum, adjusted);
        correction = vx_add_rn(vx_add_rn(total, -sum), -adjusted);
        sum = total;
    }
    projection[2u * index] = sum;
    projection[2u * index + 1u] = correction;
}

/* Each timestep has independent gate rows. Separate launches provide the
 * cross-block dependency between recurrence steps without a host readback. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_lstm_recurrent_f32(const float* projection,
        const float* recurrent, const float* bias, const float* initial_h,
        const float* output, float* gates, vx_u32 time, vx_u32 batch,
        vx_u32 hidden, vx_u32 direction, vx_u32 flags, vx_u32 step) {
    vx_u32 index = vx_global_x();
    vx_u32 directions = direction == 2u ? 2u : 1u;
    vx_u32 gate_width = 4u * hidden;
    vx_u32 count = directions * batch * gate_width;
    if (index >= count) return;
    vx_u32 row = index % gate_width;
    vx_u32 b = index / gate_width % batch;
    vx_u32 d = index / gate_width / batch;
    vx_u32 reverse = direction == 1u || d == 1u;
    vx_u32 t = reverse ? time - 1u - step : step;
    vx_u32 projected = ((t * directions + d) * batch + b) * gate_width + row;
    float sum = projection[2u * projected];
    float correction = projection[2u * projected + 1u];
    for (vx_u32 k = 0; k < hidden; ++k) {
        float h = 0.0f;
        if (step) {
            vx_u32 previous = reverse ? t + 1u : t - 1u;
            h = output[((previous * directions + d) * batch + b) * hidden + k];
        } else if (flags & 2u) h = initial_h[(d * batch + b) * hidden + k];
        float product = vx_mul_rn(h, recurrent[(d * hidden + k) * gate_width + row]);
        float adjusted = vx_add_rn(product, -correction);
        float total = vx_add_rn(sum, adjusted);
        correction = vx_add_rn(vx_add_rn(total, -sum), -adjusted);
        sum = total;
    }
    if (flags & 1u) sum += bias[d * 8u * hidden + row] + bias[d * 8u * hidden + 4u * hidden + row];
    gates[index] = row / hidden == 3u ? vx_tanh(sum) : 1.0f / (1.0f + vx_exp(-sum));
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_lstm_update_f32(const float* gates,
        const float* initial_c, float* cells, float* output, vx_u32 time,
        vx_u32 batch, vx_u32 hidden, vx_u32 direction, vx_u32 flags, vx_u32 step) {
    vx_u32 index = vx_global_x();
    vx_u32 directions = direction == 2u ? 2u : 1u;
    if (index >= directions * batch * hidden) return;
    vx_u32 j = index % hidden;
    vx_u32 b = index / hidden % batch;
    vx_u32 d = index / hidden / batch;
    vx_u32 base = (d * batch + b) * 4u * hidden + j;
    float cell = step ? cells[index] : flags & 4u ? initial_c[index] : 0.0f;
    cell = vx_add_rn(vx_mul_rn(gates[base + 2u * hidden], cell),
                     vx_mul_rn(gates[base], gates[base + 3u * hidden]));
    cells[index] = cell;
    vx_u32 t = direction == 1u || d == 1u ? time - 1u - step : step;
    output[((t * directions + d) * batch + b) * hidden + j] =
        vx_mul_rn(gates[base + hidden], vx_tanh(cell));
}
