/* The former master/quantized-weight synchronization kernels, reached only
 * through generated messages. Conversion never changes a retained Model. */
#include "training_kernels.h"
#include <limits.h>
#include <stdlib.h>

static VxStatus vx_weight_matrix(VxApiScratch* scratch, const VolvoxaiV1Tensor* source,
        VxDataType dtype, VxTensorBinding* binding, size_t* count) {
    VxStatus status = vx_api_binding_from_tensor(scratch, binding, source);
    if (status != VX_STATUS_OK) return vx_api_scratch_failed(scratch) ? VX_STATUS_OUT_OF_MEMORY : status;
    if (binding->location != VX_MEMORY_HOST) return VX_STATUS_TRANSPORT_UNSUPPORTED;
    if (binding->dtype != dtype || binding->rank != 2 || binding->shape[0] <= 0 || binding->shape[1] <= 0 ||
        binding->shape[0] > INT_MAX || binding->shape[1] > INT_MAX) return VX_STATUS_INVALID_ARGUMENT;
    uint64_t elements = (uint64_t)binding->shape[0] * (uint64_t)binding->shape[1];
    if (elements > UINT32_MAX || elements > SIZE_MAX / sizeof(float)) return VX_STATUS_INVALID_ARGUMENT;
    size_t bytes = (size_t)elements * (dtype == VX_DTYPE_F32 ? sizeof(float) : 1u);
    if (!binding->data || binding->byte_size != bytes) return VX_STATUS_INVALID_ARGUMENT;
    /* Native BufferViews may have an unaligned byte offset. The numerical
     * kernel receives aligned call-owned storage on every transport. */
    void* aligned = vx_api_scratch_alloc(scratch, bytes);
    if (!aligned) return VX_STATUS_OUT_OF_MEMORY;
    memcpy(aligned, binding->data, bytes);
    binding->data = aligned;
    *count = (size_t)elements;
    return VX_STATUS_OK;
}

static int vx_weight_output(const SynurangLiteAllocator* a, VolvoxaiV1Tensor** slot,
        const VolvoxaiV1Tensor* source, VolvoxaiV1DataType dtype, uint32_t rows,
        uint32_t columns, const void* data, size_t size) {
    *slot = a->allocate(a->context, sizeof(**slot));
    if (!*slot) return 0;
    volvoxai_v1_tensor_init_with_allocator(*slot, a);
    VolvoxaiV1Tensor* output = *slot;
    output->field_dtype = dtype; output->which_payload = 5;
    if (synurang_lite_bytes_assign(a, &output->field_name, source->field_name.data, source->field_name.len) ||
        synurang_lite_bytes_assign(a, &output->field_inline, data, size)) return 0;
    int64_t* axis = volvoxai_v1_tensor_add_shape(output);
    if (!axis) return 0;
    *axis = rows;
    axis = volvoxai_v1_tensor_add_shape(output);
    if (!axis) return 0;
    *axis = columns;
    return 1;
}

static int vx_weight_fail(const SynurangLiteAllocator* a, VolvoxaiV1OperationReport** slot,
                            VxStatus status, VxStage stage) {
    return vx_api_report_fail(a, slot, status, stage,
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
        status == VX_STATUS_TRANSPORT_UNSUPPORTED ? VX_CODE_TRANSPORT_UNSUPPORTED : VX_CODE_INVALID_ARGUMENT,
        "weight conversion requires an exact host matrix and one finite positive scale per output row") ? 0 : -1;
}

static int vx_api_quantize_weight(const VolvoxaiV1QuantizeWeightRequest* request,
        VolvoxaiV1QuantizedWeight* response, void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxTensorBinding source;
    size_t count = 0;
    VxStatus status = vx_weight_matrix(&scratch, request->field_weight, VX_DTYPE_F32, &source, &count);
    int result = 0;
    int8_t* data = NULL;
    float* scales = NULL;
    uint32_t rows = 0, columns = 0, policy = VOLVOXAI_TRAINING_WEIGHT_SCALES_RECOMPUTE;
    uint64_t saturation = 0;
    if (status != VX_STATUS_OK) goto done;
    rows = (uint32_t)source.shape[request->field_transpose_source ? 1 : 0];
    columns = (uint32_t)source.shape[request->field_transpose_source ? 0 : 1];
    if (request->which_scale_policy == 4) {
        if (!request->field_preserve || request->field_preserve->field_scales.len != rows) goto invalid;
        policy = VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE;
    } else if (request->which_scale_policy &&
               (request->which_scale_policy != 3 || !request->field_recompute)) goto invalid;
    data = malloc(count); scales = calloc(rows, sizeof(float));
    if (!data || !scales) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    if (policy == VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE)
        memcpy(scales, request->field_preserve->field_scales.data, (size_t)rows * sizeof(float));
    if (!volvoxai_training_quantize_weight_f32_to_i8(source.data, data, scales, rows, columns,
        !!request->field_transpose_source, policy, &saturation)) goto invalid;
    if (!vx_weight_output(a, &response->field_weight, request->field_weight,
                           VOLVOXAI_V1_DATA_TYPE_I8, rows, columns, data, count)) { result = -1; goto done; }
    for (uint32_t i = 0; i < rows; ++i) {
        float* scale = volvoxai_v1_quantized_weight_add_scales(response);
        if (!scale) { result = -1; goto done; }
        *scale = scales[i];
    }
    response->field_saturation_count = saturation;
    goto done;
invalid:
    status = VX_STATUS_INVALID_ARGUMENT;
done:
    free(data); free(scales); vx_api_scratch_release(&scratch);
    if (result) return result;
    if (status != VX_STATUS_OK) return vx_weight_fail(a, &response->field_report, status, VX_STAGE_WEIGHT_QUANTIZE);
    return vx_api_report_ok(a, &response->field_report, VX_STAGE_WEIGHT_QUANTIZE) ? 0 : -1;
}

static int vx_api_dequantize_weight(const VolvoxaiV1DequantizeWeightRequest* request,
        VolvoxaiV1InitializedTensor* response, void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxTensorBinding source;
    size_t count = 0;
    float* data = NULL;
    int result = 0;
    VxStatus status = vx_weight_matrix(&scratch, request->field_weight, VX_DTYPE_I8, &source, &count);
    if (status != VX_STATUS_OK) goto done;
    if (request->field_scales.len != (size_t)source.shape[0]) { status = VX_STATUS_INVALID_ARGUMENT; goto done; }
    data = malloc(count * sizeof(float));
    if (!data) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    if (!volvoxai_training_dequantize_weight_i8_to_f32(source.data, request->field_scales.data, data,
        (uint32_t)source.shape[0], (uint32_t)source.shape[1], !!request->field_transpose_destination)) {
        status = VX_STATUS_INVALID_ARGUMENT; goto done;
    }
    if (!vx_weight_output(a, &response->field_tensor, request->field_weight, VOLVOXAI_V1_DATA_TYPE_F32,
        (uint32_t)source.shape[request->field_transpose_destination ? 1 : 0],
        (uint32_t)source.shape[request->field_transpose_destination ? 0 : 1], data, count * sizeof(float))) result = -1;
done:
    free(data); vx_api_scratch_release(&scratch);
    if (result) return result;
    if (status != VX_STATUS_OK) return vx_weight_fail(a, &response->field_report, status, VX_STAGE_WEIGHT_DEQUANTIZE);
    return vx_api_report_ok(a, &response->field_report, VX_STAGE_WEIGHT_DEQUANTIZE) ? 0 : -1;
}
