/* Full-profile deterministic initializers. The generated request is the only
 * public configuration; RNG and every numerical operation stay in C. */
#include <float.h>
#include <math.h>
#include <stdlib.h>

static double vx_initializer_random(uint32_t* state) {
    uint32_t value = (*state += UINT32_C(0x6d2b79f5));
    value = (value ^ (value >> 15u)) * (value | 1u);
    value ^= value + (value ^ (value >> 7u)) * (value | 61u);
    return (double)(value ^ (value >> 14u)) / 4294967296.0;
}

static int vx_initializer_seed(const VolvoxaiV1TensorInitializer* spec, uint32_t* seed) {
    *seed = spec && spec->which_seed_source == 6 ? spec->field_seed : 0;
    if (!spec || !spec->which_seed_source || spec->which_seed_source == 6) return 1;
    if (spec->which_seed_source != 7) return 0;
    const uint8_t* text = spec->field_seed_text.data;
    size_t size = spec->field_seed_text.len;
    uint32_t hash = UINT32_C(2166136261);
    for (size_t i = 0; i < size;) {
        uint32_t c = text[i++], minimum = 0;
        unsigned tail = 0;
        if (c >= 0xc2u && c <= 0xdfu) { c &= 31u; tail = 1; minimum = 0x80u; }
        else if (c >= 0xe0u && c <= 0xefu) { c &= 15u; tail = 2; minimum = 0x800u; }
        else if (c >= 0xf0u && c <= 0xf4u) { c &= 7u; tail = 3; minimum = 0x10000u; }
        else if (c >= 0x80u) return 0;
        if (tail > size - i) return 0;
        for (unsigned j = 0; j < tail; ++j) {
            uint32_t next = text[i++];
            if ((next & 0xc0u) != 0x80u) return 0;
            c = (c << 6u) | (next & 63u);
        }
        if (c < minimum || c > 0x10ffffu || (c >= 0xd800u && c <= 0xdfffu)) return 0;
        if (c >= 0x10000u) {
            c -= 0x10000u;
            hash = (hash ^ (0xd800u + (c >> 10u))) * UINT32_C(16777619);
            c = 0xdc00u + (c & 1023u);
        }
        hash = (hash ^ c) * UINT32_C(16777619);
    }
    *seed = hash;
    return 1;
}

VxStatus vx_api_initialize_f32(const int64_t* shape, size_t rank,
        const VolvoxaiV1TensorInitializer* spec, float** output, size_t* count) {
    uint64_t elements = 1;
    uint32_t seed;
    double mean = 0, deviation = 0.02, gain = 1;
    int kind = spec ? spec->which_distribution : 0;
    *output = NULL; *count = 0;
    if (!rank || !shape || !vx_initializer_seed(spec, &seed)) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t d = 0; d < rank; ++d) {
        if (shape[d] <= 0 || (uint64_t)shape[d] > UINT64_C(9007199254740991) / elements)
            return VX_STATUS_INVALID_ARGUMENT;
        elements *= (uint64_t)shape[d];
    }
    if (elements > SIZE_MAX / sizeof(float)) return VX_STATUS_OUT_OF_MEMORY;
    if (kind == 3) {
        const VolvoxaiV1NormalInitializer* n = spec->field_normal;
        if (!n) return VX_STATUS_INVALID_ARGUMENT;
        mean = n->has_mean ? n->field_mean : 0;
        deviation = n->has_stddev ? n->field_stddev : 0.02;
        if (!isfinite(mean) || !isfinite(deviation) || deviation < 0) return VX_STATUS_INVALID_ARGUMENT;
    } else if (kind == 4 || kind == 5) {
        const VolvoxaiV1XavierInitializer* x = kind == 4 ? spec->field_xavier_uniform : spec->field_xavier_normal;
        if (!x) return VX_STATUS_INVALID_ARGUMENT;
        gain = x->has_gain ? x->field_gain : 1;
        if (!isfinite(gain) || gain < 0) return VX_STATUS_INVALID_ARGUMENT;
        double receptive = 1;
        for (size_t d = 0; d + 2 < rank; ++d) receptive *= (double)shape[d];
        double fan_sum = rank == 1 ? 2.0 * (double)shape[0] :
            ((double)shape[rank - 2] + (double)shape[rank - 1]) * receptive;
        deviation = gain * sqrt((kind == 4 ? 6.0 : 2.0) / fan_sum);
    } else if (kind < 0 || kind > 2 || (kind == 1 && !spec->field_zeros) ||
               (kind == 2 && !spec->field_ones)) return VX_STATUS_INVALID_ARGUMENT;
    float* values = calloc((size_t)elements, sizeof(float));
    if (!values) return VX_STATUS_OUT_OF_MEMORY;
    if (kind == 2) for (size_t i = 0; i < elements; ++i) values[i] = 1;
    else if (kind == 4) for (size_t i = 0; i < elements; ++i)
        values[i] = (float)((vx_initializer_random(&seed) * 2.0 - 1.0) * deviation);
    else if (kind == 3 || kind == 5) for (size_t i = 0; i < elements; i += 2) {
        double u1 = vx_initializer_random(&seed);
        double u2 = vx_initializer_random(&seed);
        if (u1 < DBL_EPSILON) u1 = DBL_EPSILON;
        double radius = sqrt(-2.0 * log(u1));
        double angle = 6.283185307179586476925286766559 * u2;
        values[i] = (float)(mean + deviation * radius * cos(angle));
        if (i + 1 < elements) values[i + 1] = (float)(mean + deviation * radius * sin(angle));
    }
    *output = values; *count = (size_t)elements;
    return VX_STATUS_OK;
}

static int vx_api_initialize_tensor(const VolvoxaiV1InitializeTensorRequest* request,
        VolvoxaiV1InitializedTensor* response, void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* a = response->_allocator;
    const VolvoxaiV1PlanningWeight* definition = request->field_tensor;
    float* values = NULL;
    size_t count = 0;
    VxStatus status = !definition || definition->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32
        ? VX_STATUS_INVALID_ARGUMENT : vx_api_initialize_f32(definition->field_shape.data,
            definition->field_shape.len, request->field_initializer, &values, &count);
    if (status != VX_STATUS_OK) return vx_api_report_fail(a, &response->field_report, status,
        VX_STAGE_TENSOR_INITIALIZE, status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT,
        "initializer requires a positive F32 shape and valid distribution parameters") ? 0 : -1;
    response->field_tensor = a->allocate(a->context, sizeof(*response->field_tensor));
    if (!response->field_tensor) { free(values); return -1; }
    VolvoxaiV1Tensor* tensor = response->field_tensor;
    volvoxai_v1_tensor_init_with_allocator(tensor, a);
    tensor->field_dtype = VOLVOXAI_V1_DATA_TYPE_F32;
    tensor->which_payload = 5;
    int ok = synurang_lite_bytes_assign(a, &tensor->field_name,
                 definition->field_name.data, definition->field_name.len) == SYNURANG_LITE_OK &&
        synurang_lite_bytes_assign(a, &tensor->field_inline, values, count * sizeof(float)) == SYNURANG_LITE_OK;
    free(values);
    if (!ok) return -1;
    for (size_t i = 0; i < definition->field_shape.len; ++i) {
        int64_t* axis = volvoxai_v1_tensor_add_shape(tensor);
        if (!axis) return -1;
        *axis = definition->field_shape.data[i];
    }
    return vx_api_report_ok(a, &response->field_report, VX_STAGE_TENSOR_INITIALIZE) ? 0 : -1;
}
