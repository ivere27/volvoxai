#include "compiled_weight_resources.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int vx_name_is_excluded(const char* name,
                               const char* const* excluded,
                               size_t excluded_count) {
    if (!name) return 1;
    for (size_t index = 0; index < excluded_count; index++)
        if (excluded[index] && !strcmp(name, excluded[index])) return 1;
    return 0;
}

static float vx_compiled_f16_to_f32(uint16_t half) {
    uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)half >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)half & 0x03ffu;
    uint32_t bits;
    float result;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            int normalized_exponent = -14;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1;
                normalized_exponent--;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                ((uint32_t)(normalized_exponent + 127) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    memcpy(&result, &bits, sizeof(result));
    return result;
}

typedef struct VxCompiledWeightCandidate {
    const SafetensorsTensor* tensor;
    size_t precedence;
} VxCompiledWeightCandidate;

static int vx_compiled_candidate_compare(const void* left,
                                         const void* right) {
    const VxCompiledWeightCandidate* a =
        (const VxCompiledWeightCandidate*)left;
    const VxCompiledWeightCandidate* b =
        (const VxCompiledWeightCandidate*)right;
    int order = strcmp(a->tensor->name, b->tensor->name);
    if (order != 0) return order;
    /* Later files/tensors override earlier ones. Put the winner first so each
     * unique name is converted exactly once after sorting. */
    if (a->precedence > b->precedence) return -1;
    if (a->precedence < b->precedence) return 1;
    return 0;
}

static int vx_compiled_weight_materialize(
        VxCompiledCpuWeight* destination,
        const SafetensorsTensor* source,
        uint64_t* allocated_bytes) {
    size_t element_count;
    size_t output_bytes;
    float* output;
    if (!destination || !source || !source->data || !allocated_bytes ||
        source->dtype != SAFETENSORS_DTYPE_F16 ||
        source->nbytes == 0u || (source->nbytes & 1u) != 0u ||
        strlen(source->name) >= sizeof(destination->name))
        return -1;
    element_count = source->nbytes / sizeof(uint16_t);
    if (element_count > SIZE_MAX / sizeof(float)) return -1;
    output_bytes = element_count * sizeof(float);
    if ((uint64_t)output_bytes > UINT64_MAX - *allocated_bytes) return -1;
    output = (float*)malloc(output_bytes);
    if (!output) return -1;
    for (size_t index = 0; index < element_count; index++) {
        const unsigned char* bytes =
            (const unsigned char*)source->data + index * sizeof(uint16_t);
        uint16_t half = (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8u);
        output[index] = vx_compiled_f16_to_f32(half);
    }
    free((void*)destination->data);
    memset(destination, 0, sizeof(*destination));
    memcpy(destination->name, source->name, strlen(source->name) + 1u);
    destination->data = output;
    destination->byte_size = output_bytes;
    destination->source_byte_size = source->nbytes;
    *allocated_bytes += (uint64_t)output_bytes;
    return 0;
}

int vx_compiled_cpu_weight_store_create(
        const SafetensorsFile* files,
        size_t file_count,
        const char* const* excluded_weight_names,
        size_t excluded_weight_count,
        VxCompiledCpuWeightStore** out_store) {
    VxCompiledCpuWeightStore* store;
    VxCompiledWeightCandidate* candidates = NULL;
    size_t candidate_count = 0u;
    size_t candidate_index = 0u;
    size_t unique_count = 0u;
    size_t precedence = 0u;
    if (!out_store || (file_count && !files) ||
        (excluded_weight_count && !excluded_weight_names))
        return -1;
    *out_store = NULL;
    for (size_t file_index = 0; file_index < file_count; file_index++) {
        if (files[file_index].tensor_count < 0 ||
            (files[file_index].tensor_count && !files[file_index].tensors))
            return -1;
        for (int tensor_index = 0;
             tensor_index < files[file_index].tensor_count; tensor_index++) {
            const SafetensorsTensor* tensor =
                &files[file_index].tensors[tensor_index];
            if (precedence == SIZE_MAX) return -1;
            precedence++;
            if (vx_name_is_excluded(tensor->name, excluded_weight_names,
                                    excluded_weight_count))
                continue;
            if (candidate_count == SIZE_MAX) return -1;
            candidate_count++;
        }
    }
    if (candidate_count > SIZE_MAX / sizeof(*candidates)) return -1;
    if (candidate_count) {
        candidates = (VxCompiledWeightCandidate*)malloc(
            candidate_count * sizeof(*candidates));
        if (!candidates) return -1;
    }
    precedence = 0u;
    for (size_t file_index = 0; file_index < file_count; file_index++) {
        const SafetensorsFile* file = &files[file_index];
        for (int tensor_index = 0; tensor_index < file->tensor_count;
             tensor_index++) {
            const SafetensorsTensor* tensor = &file->tensors[tensor_index];
            size_t current_precedence = precedence++;
            if (vx_name_is_excluded(tensor->name, excluded_weight_names,
                                    excluded_weight_count))
                continue;
            candidates[candidate_index++] = (VxCompiledWeightCandidate){
                .tensor = tensor,
                .precedence = current_precedence,
            };
        }
    }
    if (candidate_count > 1u)
        qsort(candidates, candidate_count, sizeof(*candidates),
              vx_compiled_candidate_compare);
    for (size_t index = 0; index < candidate_count; index++)
        if (index == 0u ||
            strcmp(candidates[index - 1u].tensor->name,
                   candidates[index].tensor->name)) {
            if (candidates[index].tensor->dtype != SAFETENSORS_DTYPE_F16)
                continue;
            unique_count++;
        }
    if (unique_count > SIZE_MAX / sizeof(VxCompiledCpuWeight)) {
        free(candidates);
        return -1;
    }
    store = (VxCompiledCpuWeightStore*)calloc(1, sizeof(*store));
    if (!store) {
        free(candidates);
        return -1;
    }
    store->allocated_bytes = sizeof(*store);
    if (unique_count) {
        store->weights = (VxCompiledCpuWeight*)calloc(
            unique_count, sizeof(*store->weights));
        if (!store->weights) {
            free(candidates);
            free(store);
            return -1;
        }
        if ((uint64_t)(unique_count * sizeof(*store->weights)) >
                UINT64_MAX - store->allocated_bytes) {
            free(candidates);
            free(store->weights);
            free(store);
            return -1;
        }
        store->allocated_bytes +=
            (uint64_t)(unique_count * sizeof(*store->weights));
    }
    for (size_t index = 0; index < candidate_count; index++) {
        const SafetensorsTensor* tensor = candidates[index].tensor;
        if (index > 0u &&
            !strcmp(candidates[index - 1u].tensor->name, tensor->name))
            continue;
        if (tensor->dtype != SAFETENSORS_DTYPE_F16) continue;
        if (vx_compiled_weight_materialize(
                &store->weights[store->count++], tensor,
                &store->allocated_bytes) != 0)
            goto fail;
        if ((uint64_t)store->weights[store->count - 1u].byte_size >
                UINT64_MAX - store->materialized_bytes)
            goto fail;
        store->materialized_bytes +=
            (uint64_t)store->weights[store->count - 1u].byte_size;
        store->materialization_count++;
    }
    free(candidates);
    *out_store = store;
    return 0;
fail:
    free(candidates);
    vx_compiled_cpu_weight_store_destroy(store);
    return -1;
}

const VxCompiledCpuWeight* vx_compiled_cpu_weight_store_find(
        const VxCompiledCpuWeightStore* store,
        const char* name) {
    size_t low = 0u;
    size_t high;
    if (!store || !name) return NULL;
    high = store->count;
    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        int order = strcmp(name, store->weights[middle].name);
        if (order == 0) return &store->weights[middle];
        if (order < 0) high = middle;
        else low = middle + 1u;
    }
    return NULL;
}

void vx_compiled_cpu_weight_store_destroy(VxCompiledCpuWeightStore* store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++)
        free((void*)store->weights[index].data);
    free(store->weights);
    free(store);
}
