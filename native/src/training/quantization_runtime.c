#include "training_core.h"
#include "engine_internal.h"
#if VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int volvoxai_engine_ptq_observe_tensor(const char* tensor_name,
                                       volvoxai_ptq_observer_t* observer) {
    if (!tensor_name || !tensor_name[0] || !observer ||
        volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    T* tensor = t_find(tensor_name);
    int result = -1;
    float* values = NULL;
    if (!tensor || tensor->dtype != T_F32 || tensor->elem_size != sizeof(float) ||
        !tensor->data || tensor->numel < 0 ||
        (uint64_t)tensor->numel > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        vk_sync_host_tensor(tensor) != 0) goto done;
    materialize_tensor_f32(tensor);
    values = tensor->numel > 0
        ? (float*)malloc((size_t)tensor->numel * sizeof(*values)) : NULL;
    if (tensor->numel > 0 && !values) goto done;
    if (tensor->numel > 0)
        memcpy(values, tensor->data, (size_t)tensor->numel * sizeof(*values));
    result = volvoxai_ptq_observer_observe_f32(observer, values, tensor->numel);
done:
    free(values);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return result;
}

int volvoxai_engine_ptq_materialize_weight_i8(const char* source_name,
                                              const char* output_name,
                                              const char* scale_name,
                                              const char* zero_point_name,
                                              int32_t axis,
                                              int32_t* out_scale_count) {
    if (!source_name || !source_name[0] || !output_name || !output_name[0] ||
        !scale_name || !scale_name[0] || !zero_point_name ||
        !zero_point_name[0] || !strcmp(source_name, output_name) ||
        !strcmp(source_name, scale_name) || !strcmp(output_name, scale_name) ||
        !strcmp(source_name, zero_point_name) ||
        !strcmp(output_name, zero_point_name) ||
        !strcmp(scale_name, zero_point_name) ||
        volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    int result = -1;
    T* tensor = t_find(source_name);
    int shape[8] = {0};
    int32_t packed_shape[8] = {0};
    if (volvoxai_engine_training_accumulation_pending() || !g_loaded ||
        volvoxai_engine_adapter_effect_active_locked() ||
        t_find(output_name) || t_find(scale_name) || t_find(zero_point_name) ||
        !tensor ||
        !volvoxai_engine_tensor_is_model_weight_locked(source_name) ||
        tensor->dtype != T_F32 || tensor->elem_size != sizeof(float) ||
        !tensor->data || tensor->ndim <= 0 || tensor->ndim > 8 ||
        tensor->numel <= 0 ||
        (uint64_t)tensor->numel > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        vk_sync_host_tensor(tensor) != 0) goto done;
    materialize_tensor_f32(tensor);
    long count = tensor->numel;
    int ndim = tensor->ndim;
    for (int index = 0; index < ndim; index++) shape[index] = tensor->shape[index];
    int32_t resolved_axis = axis < 0 ? axis + ndim : axis;
    if (resolved_axis < 0 || resolved_axis >= ndim || shape[resolved_axis] <= 0)
        goto done;
    for (int index = 0; index < ndim; index++) {
        if (shape[index] <= 0 || (uint64_t)shape[index] > INT32_MAX) goto done;
        packed_shape[index] = (int32_t)shape[index];
    }
    int32_t scales_count = shape[resolved_axis];
    float* source = (float*)malloc((size_t)count * sizeof(*source));
    int8_t* packed = (int8_t*)malloc((size_t)count);
    float* scales = (float*)malloc((size_t)scales_count * sizeof(*scales));
    int8_t* encoded_zero_points = (int8_t*)calloc(
        (size_t)scales_count, sizeof(*encoded_zero_points));
    int32_t* cuda_zero_points = NULL;
    if (!source || !packed || !scales || !encoded_zero_points) {
        free(source);
        free(packed);
        free(scales);
        free(encoded_zero_points);
        goto done;
    }
    memcpy(source, tensor->data, (size_t)count * sizeof(*source));
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda && resolved_axis == 0 && (uint64_t)count <= UINT32_MAX &&
        (uint64_t)(count / scales_count) <= UINT32_MAX) {
        cuda_zero_points = (int32_t*)malloc(
            (size_t)scales_count * sizeof(*cuda_zero_points));
        result = cuda_zero_points && cuda_training_quantize_w8_available() &&
            cuda_training_quantize_w8_f32(
                source, packed, scales, cuda_zero_points,
                (uint32_t)scales_count, (uint32_t)(count / scales_count),
                0u, 0u, 0u, NULL, 0.0f, NULL, NULL, NULL)
            ? 0 : -1;
        for (int32_t index = 0; result == 0 && index < scales_count; index++) {
            if (cuda_zero_points[index] != 0) result = -1;
        }
    } else
#endif
    {
        result = volvoxai_ptq_pack_weight_i8(
            source, packed_shape, ndim, axis, packed, scales, scales_count,
            NULL);
    }
    if (result == 0) {
        result = volvoxai_engine_add_model_tensor_raw_locked(
            output_name, shape, ndim, VOLVOXAI_DTYPE_I8,
            packed, (size_t)count);
    }
    int packed_added = result == 0;
    int scale_added = 0;
    if (result == 0) {
        int scale_shape[1] = { scales_count };
        result = volvoxai_engine_add_model_tensor_raw_locked(
            scale_name, scale_shape, 1, VOLVOXAI_DTYPE_F32, scales,
            (size_t)scales_count * sizeof(*scales));
        scale_added = result == 0;
    }
    if (result == 0) {
        int zero_shape[1] = { scales_count };
        result = volvoxai_engine_add_model_tensor_raw_locked(
            zero_point_name, zero_shape, 1, VOLVOXAI_DTYPE_I8,
            encoded_zero_points, (size_t)scales_count);
    }
    if (result != 0) {
        if (scale_added)
            (void)volvoxai_engine_remove_model_tensor_locked(scale_name);
        if (packed_added)
            (void)volvoxai_engine_remove_model_tensor_locked(output_name);
    }
    if (result == 0 && out_scale_count) *out_scale_count = scales_count;
    free(source);
    free(packed);
    free(scales);
    free(encoded_zero_points);
    free(cuda_zero_points);
done:
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return result;
}
