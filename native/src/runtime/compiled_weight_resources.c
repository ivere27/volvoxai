#include "compiled_weight_resources.h"
#include "cJSON.h"
#include "generated/operator_vocabulary.h"
#include "../kernels/gemm_f32.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

uint64_t vx_prepared_f32_weights_bound_add(uint64_t used, uint32_t k, uint32_t n) {
#if !defined(__wasm__)
    (void)k; (void)n;
    return used;
#else
    const uint32_t elements = vx_gemm_f32_packed_elements(k, n);
    const uint64_t bytes = (uint64_t)elements * sizeof(float) + sizeof(VxPreparedF32Weight);
    const uint64_t limit = VX_PREPARED_F32_WEIGHT_LIMIT_BYTES;
    if (!elements || bytes > limit - sizeof(VxPreparedF32Weights)) return used;
    if (!used) used = sizeof(VxPreparedF32Weights);
    return used >= limit - bytes ? limit : used + bytes;
#endif
}

const float* vx_prepared_f32_weights_find(const VxPreparedF32Weights* store,
        const float* source, uint32_t k, uint32_t n, int out_in) {
    for (const VxPreparedF32Weight* entry = store ? store->first : NULL;
         entry; entry = entry->next)
        if (entry->source == source && entry->k == k && entry->n == n &&
            entry->out_in == out_in) return entry->packed;
    return NULL;
}

const float* vx_prepared_f32_weights_add(VxPreparedF32Weights** owner,
        const float* source, uint32_t k, uint32_t n, int out_in) {
#if !defined(__wasm__)
    (void)owner; (void)source; (void)k; (void)n; (void)out_in;
    return NULL;
#else
    if (!owner || !source || (out_in != 0 && out_in != 1)) return NULL;
    const float* existing = vx_prepared_f32_weights_find(*owner, source, k, n, out_in);
    if (existing) return existing;
    const uint32_t elements = vx_gemm_f32_packed_elements(k, n);
    const uint64_t bytes = (uint64_t)elements * sizeof(float) + sizeof(VxPreparedF32Weight);
    const uint64_t used = *owner ? (*owner)->allocated_bytes : sizeof(VxPreparedF32Weights);
    if (!elements || bytes > VX_PREPARED_F32_WEIGHT_LIMIT_BYTES ||
        used > VX_PREPARED_F32_WEIGHT_LIMIT_BYTES - bytes) return NULL;
    VxPreparedF32Weights* store = *owner;
    if (!store) {
        store = (VxPreparedF32Weights*)calloc(1, sizeof(*store));
        if (!store) return NULL;
        store->allocated_bytes = sizeof(*store);
    }
    VxPreparedF32Weight* entry = (VxPreparedF32Weight*)calloc(1, sizeof(*entry));
    float* packed = (float*)malloc((size_t)elements * sizeof(float));
    if (!entry || !packed || !vx_gemm_f32_pack_b(source, packed, k, n, out_in)) {
        free(entry);
        free(packed);
        if (!*owner) free(store);
        return NULL;
    }
    entry->source = source;
    entry->packed = packed;
    entry->k = k;
    entry->n = n;
    entry->out_in = out_in;
    entry->next = store->first;
    store->first = entry;
    store->allocated_bytes += bytes;
    *owner = store;
    return packed;
#endif
}

void vx_prepared_f32_weights_clear(VxPreparedF32Weights** owner) {
    if (!owner || !*owner) return;
    VxPreparedF32Weight* entry = (*owner)->first;
    while (entry) {
        VxPreparedF32Weight* next = entry->next;
        free(entry->packed);
        free(entry);
        entry = next;
    }
    free(*owner);
    *owner = NULL;
}

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
    int preserve_storage;
} VxCompiledWeightCandidate;

static int vx_weight_name_occurs(const cJSON* value, const char* name) {
    if (!value) return 0;
    if ((value->string && !strcmp(value->string, name)) ||
        (cJSON_IsString(value) && value->valuestring &&
         !strcmp(value->valuestring, name))) return 1;
    for (const cJSON* child = value->child; child; child = child->next)
        if (vx_weight_name_occurs(child, name)) return 1;
    return 0;
}

/* A shared weight can stay compact only when every reference has a direct
 * storage kernel. Exclude public outputs, bindings, mutation/state declarations and
 * opaque parameter references, even if one ordinary consumer is eligible. */
static int vx_weight_compact_consumers(const cJSON* graph, const char* name) {
    const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(graph, "nodes");
    const cJSON* banks = cJSON_GetObjectItemCaseSensitive(graph, "banks");
    int used = 0;
    if (!cJSON_IsObject(graph) || !cJSON_IsArray(nodes)) return 0;
    /* A bank declaration only names its leading dimension. Selected bank
     * overlays are already excluded by the conversion-store caller. */
    if (banks) {
        if (!cJSON_IsObject(banks)) return 0;
        for (const cJSON* bank = banks->child; bank; bank = bank->next)
            if (!bank->string || !bank->string[0] || !cJSON_IsString(bank) ||
                !bank->valuestring || !bank->valuestring[0]) return 0;
    }
    for (const cJSON* field = graph->child; field; field = field->next)
        if (field != nodes && field != banks &&
            vx_weight_name_occurs(field, name)) return 0;
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(node, "id");
        const cJSON* source = NULL;
        if (!cJSON_IsObject(inputs) || !cJSON_IsString(op)) return 0;
        VxOperatorKind kind = vx_operator_kind_from_name(op->valuestring);
        if (kind == VX_OP_EMBEDDING || kind == VX_OP_LINEAR ||
            kind == VX_OP_MATMUL || kind == VX_OP_GEMM)
            source = cJSON_GetObjectItemCaseSensitive(inputs, "weight");
        else if (kind == VX_OP_MOE_LINEAR)
            source = cJSON_GetObjectItemCaseSensitive(inputs, "expert_weight");
        else if (kind == VX_OP_GATHER) {
            source = cJSON_GetObjectItemCaseSensitive(inputs, "input");
            if (!source) source = cJSON_GetObjectItemCaseSensitive(inputs, "x");
            if (!source) source = cJSON_GetObjectItemCaseSensitive(inputs, "data");
        }
        for (const cJSON* input = inputs->child; input; input = input->next) {
            if (!cJSON_IsString(input) || !input->valuestring) return 0;
            if (strcmp(input->valuestring, name)) continue;
            if (input != source && !(kind == VX_OP_MOE_GATED_MLP && input->string &&
                (!strcmp(input->string, "gate_weight") ||
                 !strcmp(input->string, "up_weight") ||
                 !strcmp(input->string, "down_weight")))) return 0;
            used = 1;
        }
        /* Node identity and operator kind are metadata, even when their
         * strings happen to equal a tensor name. Outputs and parameters can
         * reference or replace tensors and must still exclude preservation. */
        for (const cJSON* field = node->child; field; field = field->next)
            if (field != inputs && field != id && field != op &&
                vx_weight_name_occurs(field, name)) return 0;
    }
    return used;
}

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
        VxConvertedWeight* destination,
        const SafetensorsTensor* source,
        uint64_t* allocated_bytes) {
    size_t element_count;
    size_t output_bytes;
    float* output;
    if (!destination || !source || !source->data || !allocated_bytes ||
        (source->dtype != SAFETENSORS_DTYPE_F16 &&
         source->dtype != SAFETENSORS_DTYPE_BF16) ||
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
        if (source->dtype == SAFETENSORS_DTYPE_BF16) {
            uint32_t bits = (uint32_t)half << 16u;
            memcpy(&output[index], &bits, sizeof(bits));
        } else {
            output[index] = vx_compiled_f16_to_f32(half);
        }
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

int vx_compiled_weight_conversions_create(
        const SafetensorsFile* files,
        size_t file_count,
        const char* const* excluded_weight_names,
        size_t excluded_weight_count,
        VxBackendKind backend,
        const cJSON* logical_graph,
        VxCompiledWeightConversions** out_store) {
    VxCompiledWeightConversions* store;
    VxCompiledWeightCandidate* candidates = NULL;
    size_t candidate_count = 0u;
    size_t candidate_index = 0u;
    size_t unique_count = 0u;
    size_t precedence = 0u;
    int include_f16 = backend == VX_PORTABLE_BACKEND_KIND || backend == VX_BACKEND_KIND_CUDA;
    int direct_storage = include_f16;
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
            VxDataType dtype = candidates[index].tensor->dtype;
            if (dtype != SAFETENSORS_DTYPE_BF16 &&
                !(include_f16 && dtype == SAFETENSORS_DTYPE_F16))
                continue;
            if (direct_storage && vx_weight_compact_consumers(
                    logical_graph, candidates[index].tensor->name)) {
                candidates[index].preserve_storage = 1;
                continue;
            }
            unique_count++;
        }
    if (unique_count > SIZE_MAX / sizeof(VxConvertedWeight)) {
        free(candidates);
        return -1;
    }
    store = (VxCompiledWeightConversions*)calloc(1, sizeof(*store));
    if (!store) {
        free(candidates);
        return -1;
    }
    store->allocated_bytes = sizeof(*store);
    if (unique_count) {
        store->weights = (VxConvertedWeight*)calloc(
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
        if (candidates[index].preserve_storage) continue;
        if (tensor->dtype != SAFETENSORS_DTYPE_BF16 &&
            !(include_f16 && tensor->dtype == SAFETENSORS_DTYPE_F16)) continue;
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
    vx_compiled_weight_conversions_destroy(store);
    return -1;
}

const VxConvertedWeight* vx_compiled_weight_conversions_find(
        const VxCompiledWeightConversions* store,
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

void vx_compiled_weight_conversions_destroy(VxCompiledWeightConversions* store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++)
        free((void*)store->weights[index].data);
    free(store->weights);
    free(store);
}
