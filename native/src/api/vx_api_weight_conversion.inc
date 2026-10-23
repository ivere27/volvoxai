/* Tensor conversion and model authoring share the same storage kernels. */
#include "training_kernels.h"
#include "packed_linear.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>

static VxStatus vx_weight_tensor(VxApiScratch* scratch, const VolvoxaiV1Tensor* source,
        VxDataType dtype, VxTensorBinding* binding) {
    VxStatus status = vx_api_binding_from_tensor(scratch, binding, source);
    if (status != VX_STATUS_OK) return vx_api_scratch_failed(scratch) ? VX_STATUS_OUT_OF_MEMORY : status;
    if (binding->location != VX_MEMORY_HOST) return VX_STATUS_TRANSPORT_UNSUPPORTED;
    if (binding->dtype != dtype || binding->rank < 1 || binding->rank > 2)
        return VX_STATUS_INVALID_ARGUMENT;
    uint64_t elements = 1;
    for (uint32_t i = 0; i < binding->rank; ++i) {
        if (binding->shape[i] <= 0 || binding->shape[i] > INT_MAX) return VX_STATUS_INVALID_ARGUMENT;
        elements *= (uint64_t)binding->shape[i];
    }
    size_t unit = dtype == VX_DTYPE_I8 ? 1u : 4u;
    if (elements > UINT32_MAX || elements > SIZE_MAX / unit || !binding->data ||
        binding->byte_size != elements * unit) return VX_STATUS_INVALID_ARGUMENT;
    /* Inline/retained host storage stays alive for this call. Copy only when
     * a borrowed byte offset cannot be read through the typed kernel pointer. */
    if ((uintptr_t)binding->data % unit) {
        void* aligned = vx_api_scratch_alloc(scratch, binding->byte_size);
        if (!aligned) return VX_STATUS_OUT_OF_MEMORY;
        memcpy(aligned, binding->data, binding->byte_size);
        binding->data = aligned;
    }
    return VX_STATUS_OK;
}

static int vx_weight_output(const SynurangLiteAllocator* a, VolvoxaiV1Tensor** slot,
        const VolvoxaiV1Tensor* source, VolvoxaiV1DataType dtype, const int64_t* shape,
        size_t rank, const void* data, size_t size) {
    *slot = a->allocate(a->context, sizeof(**slot));
    if (!*slot) return 0;
    volvoxai_v1_tensor_init_with_allocator(*slot, a);
    VolvoxaiV1Tensor* output = *slot;
    output->field_dtype = dtype;
    output->which_payload = 5;
    if ((source && synurang_lite_bytes_assign(a, &output->field_name,
            source->field_name.data, source->field_name.len)) ||
        synurang_lite_bytes_assign(a, &output->field_inline, data, size)) return 0;
    for (size_t i = 0; i < rank; ++i) {
        int64_t* axis = volvoxai_v1_tensor_add_shape(output);
        if (!axis) return 0;
        *axis = shape[i];
    }
    return 1;
}

static int vx_weight_fail(const SynurangLiteAllocator* a, VolvoxaiV1OperationReport** slot,
                          VxStatus status, VxStage stage) {
    return vx_api_report_fail(a, slot, status, stage,
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
        status == VX_STATUS_TRANSPORT_UNSUPPORTED ? VX_CODE_TRANSPORT_UNSUPPORTED : VX_CODE_INVALID_ARGUMENT,
        "weight conversion requires a supported format, exact host storage and positive finite scales") ? 0 : -1;
}

static int vx_weight_format(const VolvoxaiV1WeightFormat* format) {
    return format && ((format->field_bits == 8 && format->field_group_size == 0) ||
        ((format->field_bits == 2 || format->field_bits == 3 || format->field_bits == 4 || format->field_bits == 8) &&
         vx_packed_group_size_supported((int)format->field_group_size)));
}

static int vx_api_quantize_weight(const VolvoxaiV1QuantizeWeightRequest* request,
        VolvoxaiV1QuantizeWeightResponse* response, void* user_data) {
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxTensorBinding source;
    void* codes = NULL;
    float* scales = NULL;
    int result = 0;
    VxStatus status = vx_weight_tensor(&scratch, request->field_weight, VX_DTYPE_F32, &source);
    if (status != VX_STATUS_OK) goto done;
    const VolvoxaiV1WeightQuantization* policy = request->field_quantization;
    if (source.rank != 2 || !policy || !vx_weight_format(policy->field_format)) goto invalid;
    const VolvoxaiV1WeightFormat* format = policy->field_format;
    uint32_t group_size = format->field_group_size;
    int grouped = group_size != 0;
    if (policy->field_algorithm != VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN &&
        !(grouped && policy->field_algorithm == VOLVOXAI_V1_WEIGHT_ALGORITHM_MSE_CLIP)) goto invalid;
    uint32_t rows = (uint32_t)source.shape[request->field_transpose_source ? 1 : 0];
    uint32_t columns = (uint32_t)source.shape[request->field_transpose_source ? 0 : 1];
    uint32_t words = (uint32_t)(((uint64_t)columns * format->field_bits + 31u) / 32u);
    uint32_t groups = grouped ? columns / group_size + (columns % group_size != 0) : 1u;
    size_t scale_count = (size_t)rows * groups;
    size_t code_bytes = (size_t)rows * (grouped ? (size_t)words * 4u : columns);
    if (scale_count > SIZE_MAX / sizeof(float) ||
        (request->field_preserve_scales.len &&
         (request->field_preserve_scales.len != scale_count ||
          policy->field_algorithm != VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN))) goto invalid;
    codes = malloc(code_bytes);
    scales = malloc(scale_count * sizeof(float));
    if (!codes || !scales) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    uint64_t saturation = 0;
    const float* preserve = request->field_preserve_scales.len ? request->field_preserve_scales.data : NULL;
    if (grouped) {
        if (!volvoxai_training_quantize_grouped_weight(source.data, rows, columns,
                !!request->field_transpose_source, format->field_bits, group_size, policy->field_algorithm,
                NULL, preserve, codes, scales, &saturation)) goto invalid;
    } else {
        if (preserve) memcpy(scales, preserve, scale_count * sizeof(float));
        if (!volvoxai_training_quantize_weight_f32_to_i8(source.data, codes, scales, rows, columns,
                !!request->field_transpose_source, preserve ? VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE
                : VOLVOXAI_TRAINING_WEIGHT_SCALES_RECOMPUTE, &saturation)) goto invalid;
    }
    response->field_weight = a->allocate(a->context, sizeof(*response->field_weight));
    if (!response->field_weight) { result = -1; goto done; }
    VolvoxaiV1QuantizedWeight* weight = response->field_weight;
    volvoxai_v1_quantized_weight_init_with_allocator(weight, a);
    weight->field_format = a->allocate(a->context, sizeof(*weight->field_format));
    if (!weight->field_format) { result = -1; goto done; }
    volvoxai_v1_weight_format_init_with_allocator(weight->field_format, a);
    weight->field_format->field_bits = format->field_bits;
    weight->field_format->field_group_size = format->field_group_size;
    int64_t code_shape[2] = {rows, grouped ? words : columns};
    int64_t scale_shape[2] = {rows, groups};
    if (!vx_weight_output(a, &weight->field_codes, request->field_weight,
            grouped ? VOLVOXAI_V1_DATA_TYPE_I32 : VOLVOXAI_V1_DATA_TYPE_I8,
            code_shape, 2, codes, code_bytes) ||
        !vx_weight_output(a, &weight->field_scales, NULL, VOLVOXAI_V1_DATA_TYPE_F32,
            scale_shape, grouped ? 2 : 1, scales, scale_count * sizeof(float))) { result = -1; goto done; }
    int64_t* axis = volvoxai_v1_quantized_weight_add_logical_shape(weight);
    if (!axis) { result = -1; goto done; }
    *axis = rows;
    axis = volvoxai_v1_quantized_weight_add_logical_shape(weight);
    if (!axis) { result = -1; goto done; }
    *axis = columns;
    response->field_saturation_count = saturation;
    goto done;
invalid:
    status = VX_STATUS_INVALID_ARGUMENT;
done:
    free(codes); free(scales); vx_api_scratch_release(&scratch);
    if (result) return result;
    if (status != VX_STATUS_OK) return vx_weight_fail(a, &response->field_report, status, VX_STAGE_WEIGHT_QUANTIZE);
    return vx_api_report_ok(a, &response->field_report, VX_STAGE_WEIGHT_QUANTIZE) ? 0 : -1;
}

static int vx_api_dequantize_weight(const VolvoxaiV1DequantizeWeightRequest* request,
        VolvoxaiV1InitializedTensor* response, void* user_data) {
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    float* data = NULL;
    int result = 0;
    const VolvoxaiV1QuantizedWeight* weight = request->field_weight;
    if (!weight || !vx_weight_format(weight->field_format) || weight->field_logical_shape.len != 2) goto done;
    int64_t nr = weight->field_logical_shape.data[0], nk = weight->field_logical_shape.data[1];
    if (nr <= 0 || nk <= 0 || nr > INT_MAX || nk > INT_MAX ||
        (uint64_t)nr * (uint64_t)nk > UINT32_MAX || (uint64_t)nr * (uint64_t)nk > SIZE_MAX / sizeof(float)) goto done;
    uint32_t rows = (uint32_t)nr, columns = (uint32_t)nk;
    uint32_t bits = weight->field_format->field_bits, lanes = 32u / bits;
    uint32_t group_size = weight->field_format->field_group_size;
    int grouped = group_size != 0;
    uint32_t words = (uint32_t)(((uint64_t)columns * bits + 31u) / 32u);
    uint32_t groups = grouped ? columns / group_size + (columns % group_size != 0) : 1u;
    VxTensorBinding codes, scales;
    status = vx_weight_tensor(&scratch, weight->field_codes, grouped ? VX_DTYPE_I32 : VX_DTYPE_I8, &codes);
    if (status != VX_STATUS_OK) goto done;
    status = vx_weight_tensor(&scratch, weight->field_scales, VX_DTYPE_F32, &scales);
    if (status != VX_STATUS_OK) goto done;
    status = VX_STATUS_INVALID_ARGUMENT;
    if (codes.rank != 2 || codes.shape[0] != rows || codes.shape[1] != (grouped ? words : columns) ||
        scales.rank != (uint32_t)(grouped ? 2 : 1) || scales.shape[0] != rows ||
        (grouped && scales.shape[1] != groups)) goto done;
    const float* scale_data = scales.data;
    for (size_t i = 0; i < (size_t)rows * groups; ++i)
        if (!(scale_data[i] > 0.0f) || !isfinite(scale_data[i])) goto done;
    data = malloc((size_t)rows * columns * sizeof(float));
    if (!data) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    const uint32_t* packed = codes.data;
    const int8_t* bytes = codes.data;
    uint32_t mask = (1u << bits) - 1u, offset = 1u << (bits - 1u);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < columns; ++k) {
            int q;
            if (grouped) {
                uint64_t position = (uint64_t)k * bits;
                uint32_t shift = (uint32_t)(position % 32u);
                size_t word = (size_t)r * words + (size_t)(position / 32u);
                uint32_t code = packed[word] >> shift;
                if (shift + bits > 32u) code |= packed[word + 1u] << (32u - shift);
                q = (int)(code & mask) - (int)offset;
            }
            else { q = bytes[(size_t)r * columns + k]; if (q == -128) goto done; }
            float value = (float)q * scale_data[(size_t)r * groups + (grouped ? k / group_size : 0u)];
            data[request->field_transpose_destination ? (size_t)k * rows + r : (size_t)r * columns + k] = value;
        }
        if (grouped && bits == 3) {
            uint32_t used = (uint32_t)(((uint64_t)columns * bits) % 32u);
            if (used && (packed[(size_t)r * words + words - 1u] >> used)) goto done;
        } else if (grouped && columns % lanes) for (uint32_t j = columns % lanes; j < lanes; ++j)
            if (((packed[(size_t)r * words + words - 1] >> (j * bits)) & mask) != offset) goto done;
    }
    int64_t shape[2] = {request->field_transpose_destination ? columns : rows,
                        request->field_transpose_destination ? rows : columns};
    if (!vx_weight_output(a, &response->field_tensor, weight->field_codes, VOLVOXAI_V1_DATA_TYPE_F32,
            shape, 2, data, (size_t)rows * columns * sizeof(float))) result = -1;
    status = VX_STATUS_OK;
done:
    free(data); vx_api_scratch_release(&scratch);
    if (result) return result;
    if (status != VX_STATUS_OK) return vx_weight_fail(a, &response->field_report, status, VX_STAGE_WEIGHT_DEQUANTIZE);
    return vx_api_report_ok(a, &response->field_report, VX_STAGE_WEIGHT_DEQUANTIZE) ? 0 : -1;
}
