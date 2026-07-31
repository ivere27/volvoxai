static int byte_dtype_valid(uint32_t dtype) {
    return dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

static uint64_t byte_centered_magnitude(uint32_t dtype, int32_t zero_point) {
    int64_t low =
        (int64_t)(dtype == VX_DTYPE_I8 ? -128 : 0) - zero_point;
    int64_t high =
        (int64_t)(dtype == VX_DTYPE_I8 ? 127 : 255) - zero_point;
    uint64_t low_magnitude = (uint64_t)(low < 0 ? -low : low);
    uint64_t high_magnitude = (uint64_t)(high < 0 ? -high : high);
    return low_magnitude > high_magnitude ? low_magnitude : high_magnitude;
}

static int byte_zero_point_valid(uint32_t dtype, int32_t zero_point) {
    return dtype == VX_DTYPE_I8
        ? zero_point >= -128 && zero_point <= 127
        : dtype == VX_DTYPE_U8 && zero_point >= 0 && zero_point <= 255;
}

static int byte_qdesc_valid(float scale, int32_t zero_point,
                            uint32_t dtype) {
    return byte_dtype_valid(dtype) && byte_zero_point_valid(dtype, zero_point) &&
        isfinite(scale) && scale > 0.0f;
}

static int byte_qdesc_same(float input_scale, int32_t input_zero_point,
                           uint32_t input_dtype, float output_scale,
                           int32_t output_zero_point, uint32_t output_dtype) {
    return byte_qdesc_valid(input_scale, input_zero_point, input_dtype) &&
        byte_qdesc_valid(output_scale, output_zero_point, output_dtype) &&
        input_scale == output_scale && input_zero_point == output_zero_point &&
        input_dtype == output_dtype;
}

static uint32_t qact_float_bits(float value) {
    union { float f; uint32_t u; } bits = {value};
    return bits.u;
}

static uint32_t qact_lut_hash(uint32_t input_scale_bits,
                              int32_t input_zero_point,
                              uint32_t output_scale_bits,
                              int32_t output_zero_point,
                              uint32_t input_dtype, uint32_t output_dtype,
                              int operation) {
    uint32_t hash = input_scale_bits * 0x9e3779b1u;
    hash ^= output_scale_bits + 0x85ebca6bu + (hash << 6u) + (hash >> 2u);
    hash ^= (uint32_t)input_zero_point * 0xc2b2ae35u;
    hash ^= (uint32_t)output_zero_point * 0x27d4eb2fu;
    hash ^= input_dtype * 0x165667b1u;
    hash ^= output_dtype * 0xd3a2646cu;
    hash ^= (uint32_t)operation * 0xa24baed5u;
    hash ^= hash >> 16u;
    return hash;
}

static int qact_lut_matches(const CudaQactLutEntry* entry,
                            uint32_t input_scale_bits,
                            int32_t input_zero_point,
                            uint32_t output_scale_bits,
                            int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t output_dtype,
                            int operation) {
    return entry->valid && entry->input_scale_bits == input_scale_bits &&
        entry->output_scale_bits == output_scale_bits &&
        entry->input_zero_point == input_zero_point &&
        entry->output_zero_point == output_zero_point &&
        entry->input_dtype == input_dtype &&
        entry->output_dtype == output_dtype &&
        entry->operation == operation;
}

static int32_t qact_raw_value(uint8_t raw, uint32_t dtype) {
    return dtype == VX_DTYPE_I8 && raw >= 128u
        ? (int32_t)raw - 256 : (int32_t)raw;
}

static int32_t qact_round_ties_even(float value) {
    int32_t lower = (int32_t)floorf(value);
    float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return lower % 2 == 0 ? lower : lower + 1;
}

static uint8_t qact_quantize(float transformed, int32_t output_zero_point,
                             uint32_t output_dtype) {
    int32_t minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    int32_t maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    int32_t quantized;
    if (transformed != transformed) quantized = output_zero_point;
    else if (transformed <= (float)minimum) quantized = minimum;
    else if (transformed >= (float)maximum) quantized = maximum;
    else quantized = qact_round_ties_even(transformed);
    return (uint8_t)quantized;
}

static float qact_qgelu_erf(float value) {
    const float sign = value >= 0.0f ? 1.0f : -1.0f;
    const float magnitude = fabsf(value);
    const float denominator = 1.0f + 0.3275911f * magnitude;
    const float t = 1.0f / denominator;
    float polynomial = 1.061405429f * t;
    polynomial = polynomial - 1.453152027f;
    polynomial = polynomial * t;
    polynomial = polynomial + 1.421413741f;
    polynomial = polynomial * t;
    polynomial = polynomial - 0.284496736f;
    polynomial = polynomial * t;
    polynomial = polynomial + 0.254829592f;
    polynomial = polynomial * t;
    {
        const float exponential = expf(-(magnitude * magnitude));
        return sign * (1.0f - polynomial * exponential);
    }
}

static uint8_t qact_value(uint8_t raw, float input_scale,
                          int32_t input_zero_point, float output_scale,
                          int32_t output_zero_point, uint32_t input_dtype,
                          uint32_t output_dtype, int operation) {
    const float input_value =
        (float)(qact_raw_value(raw, input_dtype) - input_zero_point) *
        input_scale;
    float activated;
    if (operation == 0) {
        activated = input_value / (1.0f + expf(-input_value));
    } else {
        const float erf_input = input_value * 0.7071067811865476f;
        const float cdf = 0.5f * (1.0f + qact_qgelu_erf(erf_input));
        activated = input_value * cdf;
    }
    {
        const float scaled = activated / output_scale;
        const float transformed = scaled + (float)output_zero_point;
        return qact_quantize(transformed, output_zero_point, output_dtype);
    }
}

static CudaQactLutEntry* qact_lut_acquire(
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype, int operation, int* rebuilt) {
    uint32_t input_scale_bits = qact_float_bits(input_scale);
    uint32_t output_scale_bits = qact_float_bits(output_scale);
    uint32_t set_index = qact_lut_hash(input_scale_bits, input_zero_point,
        output_scale_bits, output_zero_point, input_dtype, output_dtype,
        operation) & (CUDA_QACT_LUT_SETS - 1u);
    CudaQactLutEntry* set = qact_lut_cache.entries[set_index];
    uint32_t way;
    if (!rebuilt) return NULL;
    *rebuilt = 0;
    for (way = 0; way < CUDA_QACT_LUT_WAYS; way++) {
        if (qact_lut_matches(&set[way], input_scale_bits, input_zero_point,
                output_scale_bits, output_zero_point, input_dtype,
                output_dtype, operation)) return &set[way];
    }
    for (way = 0; way < CUDA_QACT_LUT_WAYS && set[way].valid; way++) {}
    if (way == CUDA_QACT_LUT_WAYS) {
        way = qact_lut_cache.next_victim[set_index];
        qact_lut_cache.next_victim[set_index] =
            (uint8_t)((way + 1u) % CUDA_QACT_LUT_WAYS);
    }
    set[way].valid = 0;
    set[way].input_scale_bits = input_scale_bits;
    set[way].output_scale_bits = output_scale_bits;
    set[way].input_zero_point = input_zero_point;
    set[way].output_zero_point = output_zero_point;
    set[way].input_dtype = (uint8_t)input_dtype;
    set[way].output_dtype = (uint8_t)output_dtype;
    set[way].operation = (uint8_t)operation;
    for (uint32_t raw = 0; raw < CUDA_QACT_LUT_VALUES; raw++) {
        set[way].values[raw] = qact_value((uint8_t)raw, input_scale,
            input_zero_point, output_scale, output_zero_point, input_dtype,
            output_dtype, operation);
    }
    set[way].valid = 1;
    *rebuilt = 1;
    return &set[way];
}
