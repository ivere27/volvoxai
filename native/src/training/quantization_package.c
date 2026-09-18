#include "generated/operator_param_registry.h"
#include "training_core.h"

#include "cJSON.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "safetensors.h"
#if VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(__wasm__)
#include <sys/stat.h>
#endif

#if defined(_WIN32)
#include <process.h>
#include <windows.h>
#define VX_PTQ_PROCESS_ID() ((long)_getpid())
#elif defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
#include <unistd.h>
#define VX_PTQ_PROCESS_ID() ((long)getpid())
#else
#define VX_PTQ_PROCESS_ID() 0L
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define VX_PTQ_NAME_CAPACITY 128
#define VX_PTQ_GRAPH_FORMAT "volvox-graph/v1"
#define VX_PTQ_QUANTIZATION_FORMAT "volvox-affine-safetensors/v1"

typedef struct {
    char name[VX_PTQ_NAME_CAPACITY];
    /* The byte tensor the template renamed this value to. Empty when the
     * template kept the name, in which case the writer names the affine. */
    char quantized_name[VX_PTQ_NAME_CAPACITY];
    int32_t dtype;
    int32_t scheme;
    volvoxai_ptq_observer_t observer;
} VxPTQPlanTensor;

typedef struct {
    int32_t kind;
    int32_t node_index;
    int32_t weight_axis;
    /* Locates the node inside the template, where positions have shifted. */
    char node_id[VX_PTQ_NAME_CAPACITY];
    char input_name[VX_PTQ_NAME_CAPACITY];
    char output_name[VX_PTQ_NAME_CAPACITY];
    char source_weight[VX_PTQ_NAME_CAPACITY];
    char packed_weight[VX_PTQ_NAME_CAPACITY];
    char source_bias[VX_PTQ_NAME_CAPACITY];
    char packed_bias[VX_PTQ_NAME_CAPACITY];
    int has_bias;
    int has_source_bias;
} VxPTQPlanLayer;

typedef char VxPTQSampleName[VX_PTQ_NAME_CAPACITY];

struct VolvoxAIPTQPlan {
    VxPTQPlanTensor* tensors;
    int32_t tensor_count;
    VxPTQPlanLayer* layers;
    int32_t layer_count;
    VxPTQSampleName* sample_names;
    int32_t sample_name_count;
    uint64_t calibration_samples;
    uint64_t model_generation;
};

static atomic_uint_fast64_t g_ptq_package_sequence = 1;

static int ptq_valid_name(const char* name) {
    return name && name[0] && strlen(name) < VX_PTQ_NAME_CAPACITY;
}

static int ptq_copy_name(char output[VX_PTQ_NAME_CAPACITY], const char* name) {
    if (!output || !ptq_valid_name(name)) return -1;
    size_t length = strlen(name);
    memcpy(output, name, length + 1);
    return 0;
}

static int ptq_supported_activation(int32_t dtype, int32_t scheme) {
    return (dtype == VOLVOXAI_DTYPE_I8 || dtype == VOLVOXAI_DTYPE_U8) &&
           (scheme == VX_PTQ_SCHEME_SYMMETRIC ||
            scheme == VX_PTQ_SCHEME_ASYMMETRIC);
}

static VxPTQPlanTensor* ptq_find_tensor(VolvoxAIPTQPlan* plan,
                                        const char* name) {
    if (!plan || !name) return NULL;
    for (int32_t index = 0; index < plan->tensor_count; index++) {
        if (!strcmp(plan->tensors[index].name, name)) return &plan->tensors[index];
    }
    return NULL;
}

static const VxPTQPlanTensor* ptq_find_tensor_const(
    const VolvoxAIPTQPlan* plan, const char* name) {
    return ptq_find_tensor((VolvoxAIPTQPlan*)plan, name);
}

static int ptq_plan_is_current_locked(const VolvoxAIPTQPlan* plan) {
    return plan && g_loaded && plan->model_generation &&
           plan->model_generation == volvoxai_engine_model_generation_locked();
}

static int ptq_validate_loaded_layer_locked(const VxPTQPlanLayer* layer);

VolvoxAIPTQPlan* volvoxai_ptq_plan_create(void) {
    if (volvoxai_engine_model_route_lease_active()) return NULL;
    volvoxai_engine_model_lock();
    VolvoxAIPTQPlan* plan = g_loaded && !volvoxai_engine_adapter_effect_active_locked()
        ? (VolvoxAIPTQPlan*)calloc(1, sizeof(VolvoxAIPTQPlan)) : NULL;
    if (plan) plan->model_generation = volvoxai_engine_model_generation_locked();
    volvoxai_engine_model_unlock();
    return plan;
}

void volvoxai_ptq_plan_destroy(VolvoxAIPTQPlan* plan) {
    if (!plan) return;
    free(plan->tensors);
    free(plan->layers);
    free(plan->sample_names);
    free(plan);
}

int volvoxai_ptq_plan_add_tensor(VolvoxAIPTQPlan* plan,
                                 const volvoxai_ptq_tensor_spec_t* spec) {
    if (!plan || !spec || spec->struct_size != sizeof(*spec) ||
        !ptq_valid_name(spec->tensor_name) ||
        !ptq_supported_activation(spec->dtype, spec->scheme) ||
        volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    int result = -1;
    T* source = t_find(spec->tensor_name);
    if (!ptq_plan_is_current_locked(plan) || plan->calibration_samples ||
        !source || source->dtype != T_F32 || !source->data || source->numel <= 0 ||
        ptq_find_tensor(plan, spec->tensor_name) ||
        plan->tensor_count == INT32_MAX ||
        (size_t)(plan->tensor_count + 1) > SIZE_MAX / sizeof(VxPTQPlanTensor))
        goto done;
    VxPTQPlanTensor* next = (VxPTQPlanTensor*)realloc(
        plan->tensors, (size_t)(plan->tensor_count + 1) * sizeof(*next));
    if (!next) goto done;
    plan->tensors = next;
    VxPTQPlanTensor* tensor = &plan->tensors[plan->tensor_count];
    memset(tensor, 0, sizeof(*tensor));
    if (ptq_copy_name(tensor->name, spec->tensor_name) != 0) goto done;
    if (spec->quantized_tensor_name && spec->quantized_tensor_name[0] &&
        ptq_copy_name(tensor->quantized_name,
                      spec->quantized_tensor_name) != 0) goto done;
    tensor->dtype = spec->dtype;
    tensor->scheme = spec->scheme;
    volvoxai_ptq_observer_reset(&tensor->observer);
    plan->tensor_count++;
    result = 0;
done:
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return result;
}

static int ptq_layer_target_name_used(const VolvoxAIPTQPlan* plan,
                                      const char* name) {
    for (int32_t index = 0; index < plan->layer_count; index++) {
        const VxPTQPlanLayer* layer = &plan->layers[index];
        if (!strcmp(layer->packed_weight, name) ||
            (layer->has_bias && !strcmp(layer->packed_bias, name))) return 1;
    }
    return 0;
}

int volvoxai_ptq_plan_add_layer(VolvoxAIPTQPlan* plan,
                                const volvoxai_ptq_layer_spec_t* spec) {
    if (!plan || !spec || spec->struct_size != sizeof(*spec) ||
        spec->node_index < 0 ||
        spec->weight_axis != 0 ||
        (spec->kind != VX_PTQ_LAYER_QLINEAR &&
         spec->kind != VX_PTQ_LAYER_QCONV2D) ||
        !ptq_valid_name(spec->input_tensor_name) ||
        !ptq_valid_name(spec->output_tensor_name) ||
        !ptq_valid_name(spec->source_weight_name) ||
        !ptq_valid_name(spec->packed_weight_name) ||
        !strcmp(spec->source_weight_name, spec->packed_weight_name) ||
        !strcmp(spec->input_tensor_name, spec->output_tensor_name) ||
        volvoxai_engine_model_route_lease_active())
        return -1;
    int has_source_bias = spec->source_bias_name && spec->source_bias_name[0];
    int has_packed_bias = spec->packed_bias_name && spec->packed_bias_name[0];
    if ((has_source_bias && !has_packed_bias) ||
        (spec->kind == VX_PTQ_LAYER_QLINEAR && !has_packed_bias)) return -1;
    if (has_source_bias &&
        (!ptq_valid_name(spec->source_bias_name) ||
         !strcmp(spec->source_bias_name, spec->packed_bias_name))) return -1;
    if (has_packed_bias &&
        (!ptq_valid_name(spec->packed_bias_name) ||
         !strcmp(spec->packed_weight_name, spec->packed_bias_name))) return -1;
    VxPTQPlanLayer candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.kind = spec->kind;
    candidate.node_index = spec->node_index;
    candidate.weight_axis = spec->weight_axis;
    candidate.has_bias = has_packed_bias;
    candidate.has_source_bias = has_source_bias;
    if ((spec->node_id && spec->node_id[0] &&
         ptq_copy_name(candidate.node_id, spec->node_id) != 0) ||
        ptq_copy_name(candidate.input_name, spec->input_tensor_name) != 0 ||
        ptq_copy_name(candidate.output_name, spec->output_tensor_name) != 0 ||
        ptq_copy_name(candidate.source_weight, spec->source_weight_name) != 0 ||
        ptq_copy_name(candidate.packed_weight, spec->packed_weight_name) != 0 ||
        (has_source_bias &&
         ptq_copy_name(candidate.source_bias, spec->source_bias_name) != 0) ||
        (has_packed_bias &&
         ptq_copy_name(candidate.packed_bias, spec->packed_bias_name) != 0)) return -1;

    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    int result = -1;
    if (!ptq_plan_is_current_locked(plan) || plan->calibration_samples ||
        !ptq_find_tensor(plan, spec->input_tensor_name) ||
        !ptq_find_tensor(plan, spec->output_tensor_name) ||
        plan->layer_count == INT32_MAX ||
        (size_t)(plan->layer_count + 1) > SIZE_MAX / sizeof(VxPTQPlanLayer) ||
        ptq_layer_target_name_used(plan, spec->packed_weight_name) ||
        (has_packed_bias && ptq_layer_target_name_used(plan, spec->packed_bias_name)) ||
        ptq_validate_loaded_layer_locked(&candidate) != 0) goto done;
    for (int32_t index = 0; index < plan->layer_count; index++) {
        const VxPTQPlanLayer* layer = &plan->layers[index];
        if (layer->node_index == spec->node_index ||
            !strcmp(layer->output_name, spec->output_tensor_name)) goto done;
    }
    VxPTQPlanLayer* next = (VxPTQPlanLayer*)realloc(
        plan->layers, (size_t)(plan->layer_count + 1) * sizeof(*next));
    if (!next) goto done;
    plan->layers = next;
    plan->layers[plan->layer_count++] = candidate;
    result = 0;
done:
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return result;
}

static int ptq_binding_index(const volvoxai_ptq_input_binding_t* bindings,
                             int32_t count, const char* name) {
    int found = -1;
    for (int32_t index = 0; index < count; index++) {
        if (bindings[index].tensor_name &&
            !strcmp(bindings[index].tensor_name, name)) {
            if (found >= 0) return -2;
            found = index;
        }
    }
    return found;
}

static int ptq_tensor_expected_bytes_locked(const char* name, int32_t* dtype,
                                            size_t* nbytes) {
    T* tensor = name ? t_find(name) : NULL;
    if (!tensor || !dtype || !nbytes || tensor->numel < 0 ||
        !tensor->elem_size ||
        (tensor->numel > 0 &&
         tensor->elem_size > SIZE_MAX / (size_t)tensor->numel)) return -1;
    *dtype = tensor->dtype;
    *nbytes = (size_t)tensor->numel * tensor->elem_size;
    return 0;
}

static int ptq_observer_merge(const volvoxai_ptq_observer_t* current,
                              const volvoxai_ptq_observer_t* sample,
                              volvoxai_ptq_observer_t* output) {
    if (!current || !sample || !output || !sample->sample_count ||
        !isfinite(sample->minimum) || !isfinite(sample->maximum) ||
        sample->minimum > sample->maximum ||
        sample->sample_count > UINT64_MAX - current->sample_count) return -1;
    if (!current->sample_count) {
        *output = *sample;
        return 0;
    }
    if (!isfinite(current->minimum) || !isfinite(current->maximum) ||
        current->minimum > current->maximum) return -1;
    *output = *current;
    if (sample->minimum < output->minimum) output->minimum = sample->minimum;
    if (sample->maximum > output->maximum) output->maximum = sample->maximum;
    output->sample_count += sample->sample_count;
    return 0;
}

static int ptq_plan_calibrate_sample(
    VolvoxAIPTQPlan* plan, const char* requested_sample_name,
    const volvoxai_ptq_input_binding_t* bindings, int32_t binding_count) {
    if (!plan || !bindings || binding_count <= 0 ||
        (requested_sample_name && !ptq_valid_name(requested_sample_name)) ||
        volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    int result = -1;
    int row_forced = 0;
    int saved_execution_row = -1;
    cJSON* graph_inputs = g_graph_root
        ? cJSON_GetObjectItemCaseSensitive(g_graph_root, "inputs") : NULL;
    int graph_input_count = cJSON_IsObject(graph_inputs)
        ? cJSON_GetArraySize(graph_inputs) : 0;
    if (!ptq_plan_is_current_locked(plan) ||
        volvoxai_engine_adapter_effect_active_locked() || !plan->tensor_count ||
        graph_input_count != binding_count ||
        plan->sample_name_count == INT32_MAX ||
        plan->calibration_samples != (uint64_t)plan->sample_name_count ||
        (size_t)(plan->sample_name_count + 1) >
            SIZE_MAX / sizeof(VxPTQSampleName)) goto done;

    char generated_name[VX_PTQ_NAME_CAPACITY];
    const char* sample_name = requested_sample_name;
    if (!sample_name) {
        int written = snprintf(generated_name, sizeof(generated_name),
                               "sample-%llu",
                               (unsigned long long)plan->calibration_samples);
        if (written <= 0 || (size_t)written >= sizeof(generated_name)) goto done;
        sample_name = generated_name;
    }
    for (int32_t index = 0; index < plan->sample_name_count; index++) {
        if (!strcmp(plan->sample_names[index], sample_name)) goto done;
    }

    for (int32_t index = 0; index < binding_count; index++) {
        const volvoxai_ptq_input_binding_t* binding = &bindings[index];
        int32_t dtype = -1;
        size_t expected = 0;
        if (binding->struct_size != sizeof(*binding) ||
            !ptq_valid_name(binding->tensor_name) ||
            binding->nbytes > SIZE_MAX ||
            (binding->nbytes && !binding->data) ||
            !t_find(binding->tensor_name) ||
            !t_find(binding->tensor_name)->is_graph_input ||
            ptq_binding_index(bindings, binding_count, binding->tensor_name) != index ||
            ptq_tensor_expected_bytes_locked(binding->tensor_name, &dtype,
                                             &expected) != 0 ||
            dtype != binding->dtype || expected != (size_t)binding->nbytes) goto done;
    }
    for (int index = 0; index < graph_input_count; index++) {
        cJSON* descriptor = cJSON_GetArrayItem(graph_inputs, index);
        const char* name = descriptor ? descriptor->string : NULL;
        if (!name || ptq_binding_index(bindings, binding_count, name) < 0) goto done;
    }
    for (int32_t index = 0; index < plan->tensor_count; index++) {
        int32_t dtype = -1;
        size_t nbytes = 0;
        if (ptq_tensor_expected_bytes_locked(plan->tensors[index].name,
                                             &dtype, &nbytes) != 0 ||
            dtype != VOLVOXAI_DTYPE_F32 || !nbytes || nbytes % sizeof(float)) goto done;
    }

    VxPTQSampleName* next_names = (VxPTQSampleName*)realloc(
        plan->sample_names,
        (size_t)(plan->sample_name_count + 1) * sizeof(*next_names));
    if (!next_names) goto done;
    plan->sample_names = next_names;

    volvoxai_ptq_observer_t* staged = (volvoxai_ptq_observer_t*)calloc(
        (size_t)plan->tensor_count, sizeof(*staged));
    volvoxai_ptq_observer_t* merged = (volvoxai_ptq_observer_t*)calloc(
        (size_t)plan->tensor_count, sizeof(*merged));
    if (!staged || !merged) {
        free(staged);
        free(merged);
        goto done;
    }

    for (int32_t index = 0; index < binding_count; index++) {
        const volvoxai_ptq_input_binding_t* binding = &bindings[index];
        T* tensor = t_find(binding->tensor_name);
        size_t nbytes = (size_t)binding->nbytes;
        if (!tensor || !tensor->is_graph_input || tensor->dtype != binding->dtype ||
            !tensor->data) goto calibration_done;
        if (nbytes) memcpy(tensor->data, binding->data, nbytes);
        vx_incremental_mark_tensor_locked(tensor);
        if (vx_runtime_backend_has_graph() && nbytes)
            vx_runtime_backend_mark_host(tensor->data, nbytes, 0);
    }
    saved_execution_row = g_execution_row;
    g_execution_row = -1;
    row_forced = 1;
    if (volvoxai_engine_forward_locked() != 0) goto calibration_done;
    for (int32_t index = 0; index < plan->tensor_count; index++) {
        T* tensor = t_find(plan->tensors[index].name);
        volvoxai_ptq_observer_reset(&staged[index]);
        if (!tensor || tensor->dtype != T_F32 || tensor->numel <= 0 ||
            !tensor->data || vk_sync_host_tensor(tensor) != 0) goto calibration_done;
        materialize_tensor_f32(tensor);
        if (volvoxai_ptq_observer_observe_f32(&staged[index], tensor->data,
                                              tensor->numel) != 0 ||
            ptq_observer_merge(&plan->tensors[index].observer, &staged[index],
                               &merged[index]) != 0) goto calibration_done;
    }
    if (ptq_copy_name(plan->sample_names[plan->sample_name_count],
                      sample_name) != 0) goto calibration_done;
    for (int32_t index = 0; index < plan->tensor_count; index++)
        plan->tensors[index].observer = merged[index];
    plan->sample_name_count++;
    plan->calibration_samples++;
    result = 0;
calibration_done:
    if (row_forced) g_execution_row = saved_execution_row;
    free(staged);
    free(merged);
done:
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return result;
}

int volvoxai_engine_ptq_plan_calibrate(
    VolvoxAIPTQPlan* plan, const volvoxai_ptq_input_binding_t* bindings,
    int32_t binding_count) {
    return ptq_plan_calibrate_sample(plan, NULL, bindings, binding_count);
}

int volvoxai_engine_ptq_plan_calibrate_sample(
    VolvoxAIPTQPlan* plan, const char* sample_name,
    const volvoxai_ptq_input_binding_t* bindings, int32_t binding_count) {
    if (!ptq_valid_name(sample_name)) return -1;
    return ptq_plan_calibrate_sample(plan, sample_name, bindings, binding_count);
}

int volvoxai_ptq_plan_tensor_params(const VolvoxAIPTQPlan* plan,
                                    const char* tensor_name,
                                    volvoxai_ptq_params_t* out_params) {
    if (!plan || volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    const VxPTQPlanTensor* tensor = ptq_find_tensor_const(plan, tensor_name);
    int result = !ptq_plan_is_current_locked(plan) || !tensor || !out_params
        ? -1 : volvoxai_ptq_calculate_params(&tensor->observer, tensor->dtype,
                                             tensor->scheme, out_params);
    volvoxai_engine_model_unlock();
    return result;
}

uint64_t volvoxai_ptq_plan_calibration_samples(const VolvoxAIPTQPlan* plan) {
    if (!plan || volvoxai_engine_model_route_lease_active()) return 0;
    volvoxai_engine_model_lock();
    uint64_t result = ptq_plan_is_current_locked(plan)
        ? plan->calibration_samples : 0;
    volvoxai_engine_model_unlock();
    return result;
}

static char* ptq_read_text_file(const char* path) {
    if (!path || !path[0]) return NULL;
    FILE* file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0 ||
        (unsigned long)length >= SIZE_MAX) {
        fclose(file);
        return NULL;
    }
    char* text = (char*)malloc((size_t)length + 1);
    if (!text) {
        fclose(file);
        return NULL;
    }
    if (length && fread(text, 1, (size_t)length, file) != (size_t)length) {
        free(text);
        fclose(file);
        return NULL;
    }
    text[length] = 0;
    if (fclose(file) != 0) {
        free(text);
        return NULL;
    }
    return text;
}

static const char* ptq_dtype_name(int32_t dtype) {
    if (dtype == VOLVOXAI_DTYPE_I8) return "int8";
    if (dtype == VOLVOXAI_DTYPE_U8) return "uint8";
    return NULL;
}

static int ptq_json_string_is(cJSON* object, const char* key,
                              const char* expected) {
    cJSON* item = object ? cJSON_GetObjectItemCaseSensitive(object, key) : NULL;
    return cJSON_IsString(item) && item->valuestring &&
           !strcmp(item->valuestring, expected);
}

static int ptq_json_add_owned(cJSON* object, const char* key, cJSON* item) {
    if (!object || !key || !item || !cJSON_AddItemToObject(object, key, item)) {
        cJSON_Delete(item);
        return -1;
    }
    return 0;
}

static int ptq_json_exact_object(cJSON* object,
                                 const char* const* keys, int key_count) {
    if (!cJSON_IsObject(object) || cJSON_GetArraySize(object) != key_count)
        return 0;
    for (int index = 0; index < key_count; index++) {
        if (!cJSON_GetObjectItemCaseSensitive(object, keys[index])) return 0;
    }
    return 1;
}

/* Dynamic-v1 node outputs are descriptor envelopes.  Keep every PTQ graph
 * rewrite on that sole schema; legacy string-valued outputs and parallel
 * outputs_shape/outputs_dtype maps are intentionally not accepted. */
static cJSON* ptq_node_output_descriptor(cJSON* output) {
    static const char* const keys[] = {"tensor", "dtype", "shape"};
    cJSON* tensor = cJSON_IsObject(output)
        ? cJSON_GetObjectItemCaseSensitive(output, "tensor") : NULL;
    cJSON* dtype = cJSON_IsObject(output)
        ? cJSON_GetObjectItemCaseSensitive(output, "dtype") : NULL;
    cJSON* shape = cJSON_IsObject(output)
        ? cJSON_GetObjectItemCaseSensitive(output, "shape") : NULL;
    return ptq_json_exact_object(output, keys, 3) &&
           cJSON_IsString(tensor) && tensor->valuestring && tensor->valuestring[0] &&
           cJSON_IsString(dtype) && dtype->valuestring && dtype->valuestring[0] &&
           cJSON_IsArray(shape) ? output : NULL;
}

static const char* ptq_node_output_tensor(cJSON* output) {
    cJSON* descriptor = ptq_node_output_descriptor(output);
    cJSON* tensor = descriptor
        ? cJSON_GetObjectItemCaseSensitive(descriptor, "tensor") : NULL;
    return tensor ? tensor->valuestring : NULL;
}

static int ptq_graph_declares_tensor(cJSON* root, const char* name) {
    cJSON* inputs = root ? cJSON_GetObjectItemCaseSensitive(root, "inputs") : NULL;
    cJSON* nodes = root ? cJSON_GetObjectItemCaseSensitive(root, "nodes") : NULL;
    cJSON* outputs = root ? cJSON_GetObjectItemCaseSensitive(root, "outputs") : NULL;
    if (!name || !name[0]) return 0;
    if (cJSON_IsObject(inputs) &&
        cJSON_GetObjectItemCaseSensitive(inputs, name)) return 1;
    for (cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        cJSON* node_inputs = cJSON_IsObject(node)
            ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
        cJSON* node_outputs = cJSON_IsObject(node)
            ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
        for (cJSON* input = cJSON_IsObject(node_inputs)
                 ? node_inputs->child : NULL;
             input; input = input->next) {
            if (cJSON_IsString(input) && input->valuestring &&
                !strcmp(input->valuestring, name)) return 1;
        }
        for (cJSON* output = cJSON_IsObject(node_outputs)
                 ? node_outputs->child : NULL;
             output; output = output->next) {
            const char* tensor_name = ptq_node_output_tensor(output);
            if (tensor_name && !strcmp(tensor_name, name)) return 1;
        }
    }
    for (cJSON* output = cJSON_IsArray(outputs) ? outputs->child : NULL;
         output; output = output->next) {
        if (cJSON_IsString(output) && output->valuestring &&
            !strcmp(output->valuestring, name)) return 1;
    }
    return 0;
}

static uint64_t ptq_parameter_hash(const char* target, const char* kind) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const unsigned char* cursor = (const unsigned char*)target;
    while (cursor && *cursor) {
        hash ^= *cursor++;
        hash *= UINT64_C(1099511628211);
    }
    cursor = (const unsigned char*)kind;
    while (cursor && *cursor) {
        hash ^= *cursor++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int ptq_parameter_name(cJSON* root, const SafetensorsFile* weights,
                              const char* target, const char* kind,
                              char output[VX_PTQ_NAME_CAPACITY]) {
    uint64_t seed;
    if (!root || !weights || !ptq_valid_name(target) || !kind || !kind[0] ||
        !output) return -1;
    seed = ptq_parameter_hash(target, kind);
    for (uint64_t attempt = 0; attempt < UINT64_C(1024); attempt++) {
        uint64_t value = seed + attempt * UINT64_C(0x9e3779b97f4a7c15);
        int length = snprintf(output, VX_PTQ_NAME_CAPACITY,
                              "__quant__.%016llx.%s",
                              (unsigned long long)value, kind);
        if (length <= 0 || length >= VX_PTQ_NAME_CAPACITY) return -1;
        if (!safetensors_find_tensor(weights, output) &&
            !ptq_graph_declares_tensor(root, output)) return 0;
    }
    return -1;
}

static cJSON* ptq_quant_reference_descriptor(
        int per_axis, int axis, const char* scale_name,
        const char* zero_point_name) {
    if (!scale_name || !zero_point_name || !strcmp(scale_name, zero_point_name))
        return NULL;
    cJSON* descriptor = cJSON_CreateObject();
    if (!descriptor ||
        ptq_json_add_owned(descriptor, "scheme",
                           cJSON_CreateString(per_axis
                               ? "per_axis" : "per_tensor")) != 0 ||
        (per_axis && ptq_json_add_owned(
            descriptor, "axis", cJSON_CreateNumber(axis)) != 0) ||
        ptq_json_add_owned(descriptor, "scale_tensor",
                           cJSON_CreateString(scale_name)) != 0 ||
        ptq_json_add_owned(descriptor, "zero_point_tensor",
                           cJSON_CreateString(zero_point_name)) != 0) {
        cJSON_Delete(descriptor);
        return NULL;
    }
    return descriptor;
}

/* The affine the template already declared for `target`, if it declared one
 * and it agrees with what the plan is about to write. A declared entry that
 * disagrees about per-tensor versus per-axis, or about the axis, is a
 * template describing a different quantization than the plan performs — a
 * silent accuracy bug if honoured, so it is refused. */
static const cJSON* ptq_adopted_descriptor(cJSON* descriptors,
                                           const char* target, int per_axis,
                                           int axis) {
    cJSON* declared = cJSON_GetObjectItemCaseSensitive(descriptors, target);
    cJSON* declared_axis;
    if (!cJSON_IsObject(declared)) return NULL;
    if (!ptq_json_string_is(declared, "scheme",
                            per_axis ? "per_axis" : "per_tensor")) {
        return NULL;
    }
    declared_axis = cJSON_GetObjectItemCaseSensitive(declared, "axis");
    if (per_axis) {
        if (!cJSON_IsNumber(declared_axis) || declared_axis->valueint != axis) {
            return NULL;
        }
    } else if (declared_axis) {
        return NULL;
    }
    if (!cJSON_IsString(cJSON_GetObjectItemCaseSensitive(
            declared, "scale_tensor")) ||
        !cJSON_IsString(cJSON_GetObjectItemCaseSensitive(
            declared, "zero_point_tensor"))) {
        return NULL;
    }
    return declared;
}

static int ptq_add_quantization_entry(
        cJSON* root, cJSON* descriptors, SafetensorsFile* weights,
        const char* target, int32_t dtype, int per_axis, int axis,
        const float* scales, const int32_t* zero_points, int32_t count) {
    char scale_name[VX_PTQ_NAME_CAPACITY];
    char zero_name[VX_PTQ_NAME_CAPACITY];
    unsigned char* encoded_zero_points = NULL;
    cJSON* descriptor = NULL;
    const cJSON* adopted;
    int shape[1];
    int minimum;
    int maximum;
    int scale_added = 0;
    int zero_added = 0;
    if (!root || !cJSON_IsObject(descriptors) || !weights ||
        !ptq_valid_name(target) || !scales || count <= 0 ||
        (dtype != VOLVOXAI_DTYPE_I8 && dtype != VOLVOXAI_DTYPE_U8)) return -1;
    adopted = ptq_adopted_descriptor(descriptors, target, per_axis, axis);
    /* An entry the writer neither authored nor can adopt would be overwritten
     * silently, so refuse instead. */
    if (!adopted && cJSON_GetObjectItemCaseSensitive(descriptors, target)) {
        return -1;
    }
    minimum = dtype == VOLVOXAI_DTYPE_I8 ? -128 : 0;
    maximum = dtype == VOLVOXAI_DTYPE_I8 ? 127 : 255;
    encoded_zero_points = (unsigned char*)malloc((size_t)count);
    if (!encoded_zero_points) return -1;
    for (int32_t index = 0; index < count; index++) {
        int32_t zero_point = zero_points ? zero_points[index] : 0;
        if (!isfinite(scales[index]) || scales[index] <= 0.0f ||
            zero_point < minimum || zero_point > maximum) goto fail;
        if (dtype == VOLVOXAI_DTYPE_I8)
            ((int8_t*)encoded_zero_points)[index] = (int8_t)zero_point;
        else
            ((uint8_t*)encoded_zero_points)[index] = (uint8_t)zero_point;
    }
    if (adopted) {
        const char* declared_scale = cJSON_GetObjectItemCaseSensitive(
            (cJSON*)adopted, "scale_tensor")->valuestring;
        const char* declared_zero = cJSON_GetObjectItemCaseSensitive(
            (cJSON*)adopted, "zero_point_tensor")->valuestring;
        if (!ptq_valid_name(declared_scale) ||
            !ptq_valid_name(declared_zero) ||
            !strcmp(declared_scale, declared_zero) ||
            strlen(declared_scale) >= VX_PTQ_NAME_CAPACITY ||
            strlen(declared_zero) >= VX_PTQ_NAME_CAPACITY) goto fail;
        snprintf(scale_name, sizeof(scale_name), "%s", declared_scale);
        snprintf(zero_name, sizeof(zero_name), "%s", declared_zero);
    } else if (ptq_parameter_name(root, weights, target, "scale",
                                  scale_name) != 0 ||
               ptq_parameter_name(root, weights, target, "zero_point",
                                  zero_name) != 0) {
        goto fail;
    }
    if (!strcmp(target, scale_name) || !strcmp(target, zero_name)) goto fail;
    shape[0] = count;
    if (safetensors_add_tensor(
            weights, scale_name, SAFETENSORS_DTYPE_F32, shape, 1, scales,
            (size_t)count * sizeof(*scales)) != 0) goto fail;
    scale_added = 1;
    if (safetensors_add_tensor(
            weights, zero_name,
            dtype == VOLVOXAI_DTYPE_I8
                ? SAFETENSORS_DTYPE_I8 : SAFETENSORS_DTYPE_U8,
            shape, 1, encoded_zero_points, (size_t)count) != 0) goto fail;
    zero_added = 1;
    if (!adopted) {
        descriptor = ptq_quant_reference_descriptor(
            per_axis, axis, scale_name, zero_name);
        if (!descriptor ||
            ptq_json_add_owned(descriptors, target, descriptor) != 0) {
            descriptor = NULL; /* ptq_json_add_owned owns deletion on failure. */
            goto fail;
        }
    }
    free(encoded_zero_points);
    return 0;
fail:
    cJSON_Delete(descriptor);
    if (zero_added) (void)safetensors_remove_tensor(weights, zero_name);
    if (scale_added) (void)safetensors_remove_tensor(weights, scale_name);
    free(encoded_zero_points);
    return -1;
}

static int ptq_json_output_key(cJSON* node, const char* tensor_name,
                               const char** output_key) {
    cJSON* outputs = node ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
    if (!cJSON_IsObject(outputs) || !outputs->child || outputs->child->next ||
        !outputs->child->string ||
        !ptq_node_output_descriptor(outputs->child) ||
        strcmp(ptq_node_output_tensor(outputs->child), tensor_name)) return -1;
    *output_key = outputs->child->string;
    return 0;
}

static int ptq_json_activation_input(cJSON* node, int32_t kind,
                                     const char* expected) {
    cJSON* inputs = node ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const char* keys[] = {"input", "x", "a"};
    int key_count = kind == VX_PTQ_LAYER_QLINEAR ? 3 : 2;
    int found = 0;
    if (!cJSON_IsObject(inputs)) return -1;
    for (int index = 0; index < key_count; index++) {
        cJSON* item = cJSON_GetObjectItemCaseSensitive(inputs, keys[index]);
        if (item) {
            if (!cJSON_IsString(item) || !item->valuestring ||
                strcmp(item->valuestring, expected)) return -1;
            found++;
        }
    }
    return found == 1 ? 0 : -1;
}

static cJSON* ptq_json_tensor_shape(cJSON* root, const char* tensor_name) {
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* input = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, tensor_name) : NULL;
    cJSON* shape = input ? cJSON_GetObjectItemCaseSensitive(input, "shape") : NULL;
    if (cJSON_IsArray(shape)) return shape;
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if (!cJSON_IsArray(nodes)) return NULL;
    int count = cJSON_GetArraySize(nodes);
    for (int index = 0; index < count; index++) {
        cJSON* node = cJSON_GetArrayItem(nodes, index);
        const char* output_key = NULL;
        if (ptq_json_output_key(node, tensor_name, &output_key) != 0) continue;
        cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
        cJSON* descriptor = cJSON_IsObject(outputs)
            ? cJSON_GetObjectItemCaseSensitive(outputs, output_key) : NULL;
        shape = ptq_node_output_descriptor(descriptor)
            ? cJSON_GetObjectItemCaseSensitive(descriptor, "shape") : NULL;
        return cJSON_IsArray(shape) ? shape : NULL;
    }
    return NULL;
}

static int ptq_json_shape_dim(cJSON* shape, int index, int* output) {
    int count = cJSON_IsArray(shape) ? cJSON_GetArraySize(shape) : -1;
    if (index < 0) index += count;
    cJSON* value = index >= 0 && index < count ? cJSON_GetArrayItem(shape, index) : NULL;
    if (!cJSON_IsNumber(value) || !isfinite(value->valuedouble) ||
        value->valuedouble != (double)value->valueint || value->valueint <= 0)
        return -1;
    *output = value->valueint;
    return 0;
}

static int ptq_json_shape_axis_equal(cJSON* left, cJSON* right, int index) {
    cJSON* left_value;
    cJSON* right_value;
    int left_count = cJSON_IsArray(left) ? cJSON_GetArraySize(left) : -1;
    int right_count = cJSON_IsArray(right) ? cJSON_GetArraySize(right) : -1;
    if (left_count < 0 || right_count != left_count ||
        index < 0 || index >= left_count) return 0;
    left_value = cJSON_GetArrayItem(left, index);
    right_value = cJSON_GetArrayItem(right, index);
    if (cJSON_IsString(left_value) && cJSON_IsString(right_value))
        return left_value->valuestring && right_value->valuestring &&
               !strcmp(left_value->valuestring, right_value->valuestring);
    return cJSON_IsNumber(left_value) && cJSON_IsNumber(right_value) &&
           isfinite(left_value->valuedouble) &&
           isfinite(right_value->valuedouble) &&
           left_value->valuedouble == (double)left_value->valueint &&
           right_value->valuedouble == (double)right_value->valueint &&
           left_value->valueint > 0 &&
           left_value->valueint == right_value->valueint;
}

static int ptq_layer_produces_before(const VolvoxAIPTQPlan* plan,
                                     const char* tensor_name,
                                     int32_t node_index) {
    for (int32_t index = 0; index < plan->layer_count; index++) {
        if (plan->layers[index].node_index < node_index &&
            !strcmp(plan->layers[index].output_name, tensor_name)) return 1;
    }
    return 0;
}

static int ptq_template_node_references(cJSON* root, const char* tensor_name) {
    /* A named bank is also a retained reference to immutable source storage,
     * even if every compute consumer was replaced by a packed successor. */
    cJSON* banks = cJSON_GetObjectItemCaseSensitive(root, "banks");
    if (cJSON_IsObject(banks) &&
        cJSON_GetObjectItemCaseSensitive(banks, tensor_name)) return 1;
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if (!cJSON_IsArray(nodes)) return 0;
    int node_count = cJSON_GetArraySize(nodes);
    for (int node_index = 0; node_index < node_count; node_index++) {
        cJSON* node = cJSON_GetArrayItem(nodes, node_index);
        cJSON* inputs = cJSON_IsObject(node)
            ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
        if (!cJSON_IsObject(inputs)) continue;
        for (cJSON* input = inputs->child; input; input = input->next) {
            if (cJSON_IsString(input) && input->valuestring &&
                !strcmp(input->valuestring, tensor_name)) return 1;
        }
    }
    return 0;
}

static int ptq_plan_packed_name(const VolvoxAIPTQPlan* plan,
                                const char* tensor_name) {
    for (int32_t index = 0; index < plan->layer_count; index++) {
        const VxPTQPlanLayer* layer = &plan->layers[index];
        if (!strcmp(layer->packed_weight, tensor_name) ||
            (layer->has_bias && !strcmp(layer->packed_bias, tensor_name))) return 1;
    }
    return 0;
}

static int ptq_template_producer(cJSON* nodes, const char* tensor_name) {
    int producer = -1;
    int count = cJSON_IsArray(nodes) ? cJSON_GetArraySize(nodes) : 0;
    for (int index = 0; index < count; index++) {
        cJSON* node = cJSON_GetArrayItem(nodes, index);
        cJSON* outputs = cJSON_IsObject(node)
            ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
        if (!cJSON_IsObject(outputs)) return -2;
        for (cJSON* output = outputs->child; output; output = output->next) {
            const char* produced = ptq_node_output_tensor(output);
            if (!produced) return -2;
            if (!strcmp(produced, tensor_name)) {
                if (producer >= 0) return -2;
                producer = index;
            }
        }
    }
    return producer;
}

static int ptq_template_declares_graph_io(cJSON* root, const char* name) {
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    if (cJSON_IsObject(inputs) && cJSON_GetObjectItemCaseSensitive(inputs, name))
        return 1;
    cJSON* outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
    if (cJSON_IsArray(outputs)) {
        for (cJSON* item = outputs->child; item; item = item->next) {
            if (cJSON_IsString(item) && item->valuestring &&
                !strcmp(item->valuestring, name)) return 1;
        }
    }
    return 0;
}

static VxDataType ptq_safetensors_dtype(int dtype) {
    switch (dtype) {
        case VX_DTYPE_F32:
        case VX_DTYPE_F16:
        case VX_DTYPE_I8:
        case VX_DTYPE_U8:
        case VX_DTYPE_I32:
            return dtype;
        default: return SAFETENSORS_DTYPE_UNKNOWN;
    }
}

static int ptq_loaded_weight_matches(const SafetensorsFile* source_file,
                                     const char* name) {
    T* tensor = name ? t_find(name) : NULL;
    const SafetensorsTensor* stored = source_file
        ? safetensors_find_tensor(source_file, name) : NULL;
    if (!tensor || !stored || !tensor->data || tensor->numel <= 0 ||
        tensor->ndim < 0 || tensor->ndim > 8 || !tensor->elem_size ||
        !volvoxai_engine_tensor_is_model_weight_locked(name) ||
        stored->dtype != ptq_safetensors_dtype(tensor->dtype) ||
        stored->ndim != tensor->ndim ||
        (uint64_t)tensor->numel > (uint64_t)(SIZE_MAX / tensor->elem_size) ||
        stored->nbytes != (size_t)tensor->numel * tensor->elem_size ||
        vk_sync_host_tensor(tensor) != 0) return 0;
    if (tensor->dtype == T_F32) materialize_tensor_f32(tensor);
    for (int index = 0; index < tensor->ndim; index++) {
        if (stored->shape[index] != tensor->shape[index]) return 0;
    }
    return !memcmp(stored->data, tensor->data, stored->nbytes);
}

/* A template may name its own affines.
 *
 * AuthorPtqTemplate decides what every introduced tensor is called, including
 * the scale and zero point of each quantized value, and writes that into a
 * `quantization` block. The writer then fills in payloads for names that
 * already exist rather than inventing a second set — one naming authority
 * instead of two, and a template that reads the same whichever
 * implementation produced it.
 *
 * A template without the block is still valid: a caller that hand-authored
 * one, or an older one on disk, leaves the naming to the writer. What is not
 * valid is a block that names an affine for a tensor the plan never
 * quantizes, which ptq_adopted_descriptor catches at use. */
static cJSON* ptq_template_declared_affines(cJSON* root) {
    cJSON* quantization =
        root ? cJSON_GetObjectItemCaseSensitive(root, "quantization") : NULL;
    cJSON* tensors;
    static const char* const keys[] = {"format", "tensors"};
    if (!quantization) return NULL;
    if (!ptq_json_exact_object(quantization, keys, 2) ||
        !ptq_json_string_is(quantization, "format",
                            VX_PTQ_QUANTIZATION_FORMAT)) {
        return NULL;
    }
    tensors = cJSON_GetObjectItemCaseSensitive(quantization, "tensors");
    return cJSON_IsObject(tensors) ? tensors : NULL;
}

static int ptq_template_is_closed_v1(cJSON* root) {
    const char* root_keys[7] = {
        "format", "dimensions", "inputs", "outputs", "nodes",
    };
    static const char* const input_keys[] = {"shape", "dtype"};
    static const char* const node_keys[] = {
        "id", "opType", "inputs", "outputs", "params",
    };
    static const char* const dimension_keys[] = {"min", "max"};
    static const char* const divisible_dimension_keys[] = {
        "min", "max", "multiple_of",
    };
    int declares_affines =
        cJSON_GetObjectItemCaseSensitive(root, "quantization") != NULL;
    cJSON* banks = cJSON_GetObjectItemCaseSensitive(root, "banks");
    int root_key_count = 5;
    if (declares_affines) root_keys[root_key_count++] = "quantization";
    if (banks) root_keys[root_key_count++] = "banks";
    if (!ptq_json_exact_object(root, root_keys, root_key_count) ||
        !ptq_json_string_is(root, "format", VX_PTQ_GRAPH_FORMAT))
        return 0;
    if (banks) {
        if (!cJSON_IsObject(banks)) return 0;
        for (cJSON* bank = banks->child; bank; bank = bank->next) {
            if (!ptq_valid_name(bank->string) || !cJSON_IsString(bank) ||
                !ptq_valid_name(bank->valuestring) ||
                !cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(
                    cJSON_GetObjectItemCaseSensitive(root, "dimensions"),
                    bank->valuestring))) return 0;
            for (cJSON* prior = banks->child; prior != bank; prior = prior->next)
                if (!strcmp(prior->string, bank->string)) return 0;
        }
    }
    if (declares_affines && !ptq_template_declared_affines(root)) return 0;
    cJSON* dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if (!cJSON_IsObject(dimensions) || !cJSON_IsObject(inputs) ||
        !cJSON_IsArray(outputs) || !outputs->child || !cJSON_IsArray(nodes))
        return 0;
    for (cJSON* dimension = dimensions->child; dimension;
         dimension = dimension->next) {
        cJSON* minimum = cJSON_GetObjectItemCaseSensitive(dimension, "min");
        cJSON* maximum = cJSON_GetObjectItemCaseSensitive(dimension, "max");
        cJSON* multiple = cJSON_GetObjectItemCaseSensitive(
            dimension, "multiple_of");
        const char* const* keys = multiple
            ? divisible_dimension_keys : dimension_keys;
        int count = multiple ? 3 : 2;
        if (!dimension->string || !dimension->string[0] ||
            !ptq_json_exact_object(dimension, keys, count) ||
            !cJSON_IsNumber(minimum) || !cJSON_IsNumber(maximum) ||
            minimum->valuedouble != (double)minimum->valueint ||
            maximum->valuedouble != (double)maximum->valueint ||
            minimum->valueint <= 0 || maximum->valueint < minimum->valueint ||
            (multiple && (!cJSON_IsNumber(multiple) ||
                multiple->valuedouble != (double)multiple->valueint ||
                multiple->valueint <= 0))) return 0;
    }
    for (cJSON* input = inputs->child; input; input = input->next) {
        cJSON* shape = cJSON_GetObjectItemCaseSensitive(input, "shape");
        cJSON* dtype = cJSON_GetObjectItemCaseSensitive(input, "dtype");
        if (!input->string || !input->string[0] ||
            !ptq_json_exact_object(input, input_keys, 2) ||
            !cJSON_IsArray(shape) || !shape->child ||
            !cJSON_IsString(dtype) || !dtype->valuestring ||
            !dtype->valuestring[0]) return 0;
    }
    for (cJSON* output = outputs->child; output; output = output->next) {
        if (!cJSON_IsString(output) || !output->valuestring ||
            !output->valuestring[0]) return 0;
    }
    for (cJSON* node = nodes->child; node; node = node->next) {
        cJSON* id = cJSON_GetObjectItemCaseSensitive(node, "id");
        cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        cJSON* node_inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        cJSON* node_outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
        cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
        if (!ptq_json_exact_object(node, node_keys, 5) ||
            !cJSON_IsString(id) || !id->valuestring || !id->valuestring[0] ||
            !cJSON_IsString(op) || !op->valuestring || !op->valuestring[0] ||
            !cJSON_IsObject(node_inputs) || !cJSON_IsObject(node_outputs) ||
            !node_outputs->child || !cJSON_IsObject(params)) return 0;
        for (cJSON* input = node_inputs->child; input; input = input->next) {
            if (!input->string || !input->string[0] ||
                !cJSON_IsString(input) || !input->valuestring ||
                !input->valuestring[0]) return 0;
        }
        for (cJSON* output = node_outputs->child; output;
             output = output->next) {
            if (!output->string || !output->string[0] ||
                !ptq_node_output_descriptor(output)) return 0;
        }
    }
    return 1;
}

/* True when `name` is a scale or zero point the template's affine table
 * declares.
 *
 * Such a tensor does not exist yet: authoring named it, and the writer creates
 * it once calibration has decided its value. The dependency preflight runs
 * before that, so without this a template would be rejected for referring to
 * exactly the tensors the next step is about to add. */
static int ptq_template_declares_affine_payload(cJSON* root, const char* name) {
    cJSON* tensors = ptq_template_declared_affines(root);
    cJSON* entry;
    if (!tensors || !name || !name[0]) return 0;
    cJSON_ArrayForEach(entry, tensors) {
        if (ptq_json_string_is(entry, "scale_tensor", name) ||
            ptq_json_string_is(entry, "zero_point_tensor", name)) {
            return 1;
        }
    }
    return 0;
}

static int ptq_template_dependency_preflight(const VolvoxAIPTQPlan* plan,
                                             cJSON* root,
                                             const SafetensorsFile* weights) {
    cJSON* graph_inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if (!cJSON_IsObject(graph_inputs) || !cJSON_IsArray(nodes)) return -1;
    cJSON* banks = cJSON_GetObjectItemCaseSensitive(root, "banks");
    if (cJSON_IsObject(banks)) {
        for (cJSON* bank = banks->child; bank; bank = bank->next) {
            if (!ptq_loaded_weight_matches(weights, bank->string)) return -1;
            const SafetensorsTensor* stored = safetensors_find_tensor(weights, bank->string);
            cJSON* dimension = cJSON_GetObjectItemCaseSensitive(
                cJSON_GetObjectItemCaseSensitive(root, "dimensions"), bank->valuestring);
            cJSON* minimum = cJSON_GetObjectItemCaseSensitive(dimension, "min");
            cJSON* maximum = cJSON_GetObjectItemCaseSensitive(dimension, "max");
            cJSON* multiple = cJSON_GetObjectItemCaseSensitive(dimension, "multiple_of");
            if (!stored || stored->ndim < 2 ||
                stored->shape[0] < minimum->valueint || stored->shape[0] > maximum->valueint ||
                stored->shape[0] % (multiple ? multiple->valueint : 1)) return -1;
        }
    }
    for (int32_t index = 0; index < plan->layer_count; index++) {
        const VxPTQPlanLayer* layer = &plan->layers[index];
        const char* names[2] = {layer->packed_weight,
                                layer->has_bias ? layer->packed_bias : NULL};
        for (int name_index = 0; name_index < 2; name_index++) {
            const char* name = names[name_index];
            if (!name) continue;
            if (ptq_template_declares_graph_io(root, name) ||
                ptq_template_producer(nodes, name) >= 0) return -1;
        }
    }
    int node_count = cJSON_GetArraySize(nodes);
    for (int node_index = 0; node_index < node_count; node_index++) {
        cJSON* node = cJSON_GetArrayItem(nodes, node_index);
        cJSON* inputs = cJSON_IsObject(node)
            ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
        if (!cJSON_IsObject(inputs)) return -1;
        for (cJSON* input = inputs->child; input; input = input->next) {
            if (!cJSON_IsString(input) || !input->valuestring ||
                !input->valuestring[0]) return -1;
            const char* name = input->valuestring;
            if (cJSON_GetObjectItemCaseSensitive(graph_inputs, name)) continue;
            int producer = ptq_template_producer(nodes, name);
            if (producer == -2 || producer >= node_index) return -1;
            if (producer >= 0 || ptq_plan_packed_name(plan, name)) continue;
            if (ptq_template_declares_affine_payload(root, name)) continue;
            if (ptq_loaded_weight_matches(weights, name)) continue;
            /* The package emits one weight file. A dependency available only
               through another loaded shard would otherwise author a package
               that cannot be reloaded from the advertised pair. */
            return -1;
        }
    }
    return 0;
}

static int ptq_template_prepare_quantization(cJSON* root,
                                             const SafetensorsFile* weights,
                                             cJSON** descriptors_out) {
    cJSON* format;
    cJSON* inputs;
    cJSON* nodes;
    cJSON* quantization = NULL;
    cJSON* descriptors = NULL;
    cJSON* storage_format = NULL;
    cJSON* metadata = NULL;
    if (!ptq_template_is_closed_v1(root) || !weights || !descriptors_out)
        return -1;
    *descriptors_out = NULL;
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, VX_PTQ_GRAPH_FORMAT) ||
        !cJSON_IsObject(inputs) || !cJSON_IsArray(nodes) ||
        cJSON_GetObjectItemCaseSensitive(root, "weights_quantization") ||
        cJSON_GetObjectItemCaseSensitive(root, "weights_quantization_storage"))
        return -1;
    for (cJSON* input = inputs->child; input; input = input->next) {
        if (!cJSON_IsObject(input) ||
            cJSON_GetObjectItemCaseSensitive(input, "quantization")) return -1;
    }
    for (cJSON* node = nodes->child; node; node = node->next) {
        if (!cJSON_IsObject(node) ||
            cJSON_GetObjectItemCaseSensitive(node, "outputs_quantization"))
            return -1;
    }
    if (weights->metadata_json) {
        metadata = cJSON_Parse(weights->metadata_json);
        if (!cJSON_IsObject(metadata) ||
            cJSON_GetObjectItemCaseSensitive(
                metadata, "weights_quantization") ||
            cJSON_GetObjectItemCaseSensitive(
                metadata, "weights_quantization_storage")) {
            cJSON_Delete(metadata);
            return -1;
        }
        cJSON_Delete(metadata);
    }
    /* Authored by the template, so the writer fills in payloads for names it
     * did not choose. Everything downstream is identical either way; only the
     * question of who named the tensors changes. */
    descriptors = ptq_template_declared_affines(root);
    if (descriptors) {
        *descriptors_out = descriptors;
        return 0;
    }
    quantization = cJSON_CreateObject();
    descriptors = cJSON_CreateObject();
    storage_format = cJSON_CreateString(VX_PTQ_QUANTIZATION_FORMAT);
    if (!quantization || !descriptors || !storage_format ||
        !cJSON_AddItemToObject(quantization, "format", storage_format)) {
        cJSON_Delete(storage_format);
        cJSON_Delete(descriptors);
        cJSON_Delete(quantization);
        return -1;
    }
    storage_format = NULL;
    if (!cJSON_AddItemToObject(quantization, "tensors", descriptors)) {
        cJSON_Delete(descriptors);
        cJSON_Delete(quantization);
        return -1;
    }
    if (!cJSON_AddItemToObject(root, "quantization", quantization)) {
        cJSON_Delete(quantization);
        return -1;
    }
    *descriptors_out = descriptors;
    return 0;
}

static int ptq_template_set_graph_input(cJSON* root, cJSON* descriptors,
                                        SafetensorsFile* weights,
                                        const VxPTQPlanTensor* tensor) {
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* descriptor = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, tensor->name) : NULL;
    const char* dtype_name = ptq_dtype_name(tensor->dtype);
    volvoxai_ptq_params_t params;
    int32_t zero_point;
    if (!cJSON_IsObject(descriptor) || !dtype_name ||
        !ptq_json_string_is(descriptor, "dtype", dtype_name) ||
        cJSON_GetObjectItemCaseSensitive(descriptor, "quantization") ||
        volvoxai_ptq_calculate_params(&tensor->observer, tensor->dtype,
                                      tensor->scheme, &params) != 0)
        return -1;
    zero_point = params.zero_point;
    return ptq_add_quantization_entry(
        root, descriptors, weights, tensor->name, tensor->dtype,
        0, 0, &params.scale, &zero_point, 1);
}

static int ptq_template_set_output(cJSON* root, cJSON* node,
                                   const char* output_key,
                                   cJSON* descriptors,
                                   SafetensorsFile* weights,
                                   const VxPTQPlanTensor* tensor) {
    cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
    cJSON* output = cJSON_IsObject(outputs)
        ? cJSON_GetObjectItemCaseSensitive(outputs, output_key) : NULL;
    const char* dtype_name = ptq_dtype_name(tensor->dtype);
    volvoxai_ptq_params_t params;
    int32_t zero_point;
    if (!ptq_node_output_descriptor(output) || !dtype_name ||
        !ptq_json_string_is(output, "dtype", dtype_name) ||
        cJSON_GetObjectItemCaseSensitive(node, "outputs_quantization") ||
        volvoxai_ptq_calculate_params(&tensor->observer, tensor->dtype,
                                      tensor->scheme, &params) != 0)
        return -1;
    zero_point = params.zero_point;
    return ptq_add_quantization_entry(
        root, descriptors, weights, tensor->name, tensor->dtype,
        0, 0, &params.scale, &zero_point, 1);
}

static int ptq_set_authoring_metadata(
        SafetensorsFile* weights,
        const volvoxai_ptq_package_options_t* options) {
    cJSON* metadata = NULL;
    cJSON* coverage = NULL;
    char* encoded = NULL;
    int result = -1;
    if (!weights || !options) return -1;
    if (!options->logical_fingerprint && !options->profile_coverage_json)
        return 0;
    if (!options->logical_fingerprint || !options->logical_fingerprint[0] ||
        !options->profile_coverage_json || !options->profile_coverage_json[0])
        return -1;
    coverage = cJSON_Parse(options->profile_coverage_json);
    if (!cJSON_IsObject(coverage) ||
        !ptq_json_string_is(coverage, "format", "volvox.ptq-coverage/v1") ||
        !ptq_json_string_is(coverage, "logicalFingerprint",
                            options->logical_fingerprint) ||
        !cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(coverage, "complete")))
        goto done;
    metadata = weights->metadata_json
        ? cJSON_Parse(weights->metadata_json) : cJSON_CreateObject();
    if (!cJSON_IsObject(metadata) ||
        cJSON_GetObjectItemCaseSensitive(metadata, "format") ||
        cJSON_GetObjectItemCaseSensitive(metadata, "logical_fingerprint") ||
        cJSON_GetObjectItemCaseSensitive(metadata, "profile_coverage") ||
        !cJSON_AddStringToObject(metadata, "format", "volvox.ptq.v1") ||
        !cJSON_AddStringToObject(metadata, "logical_fingerprint",
                                 options->logical_fingerprint) ||
        !cJSON_AddStringToObject(metadata, "profile_coverage",
                                 options->profile_coverage_json))
        goto done;
    encoded = cJSON_PrintUnformatted(metadata);
    if (!encoded || safetensors_set_metadata_json(weights, encoded) != 0)
        goto done;
    result = 0;
done:
    free(encoded);
    cJSON_Delete(metadata);
    cJSON_Delete(coverage);
    return result;
}

/* The activation name the template gave `source`, or `source` itself when the
 * template kept it. */
static const char* ptq_template_activation(const VolvoxAIPTQPlan* plan,
                                           const char* source) {
    const VxPTQPlanTensor* tensor = ptq_find_tensor_const(plan, source);
    if (tensor && tensor->quantized_name[0]) return tensor->quantized_name;
    return source;
}

/* The node in the template that corresponds to one planned layer.
 *
 * Located by id, not by index: authoring inserts the Quantize and Dequantize
 * boundaries, so a template's node array is longer than the source graph's
 * and the positions no longer line up. The ids do — every quantized node
 * keeps the id it had — and matching on them is what lets one plan describe
 * both graphs. */
static cJSON* ptq_template_node_by_id(cJSON* root, const char* node_id) {
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    cJSON* node;
    if (!cJSON_IsArray(nodes) || !node_id || !node_id[0]) return NULL;
    cJSON_ArrayForEach(node, nodes) {
        if (ptq_json_string_is(node, "id", node_id)) return node;
    }
    return NULL;
}

static int ptq_template_validate_layer(const VolvoxAIPTQPlan* plan,
                                       const VxPTQPlanLayer* layer,
                                       cJSON* root, cJSON** node_out,
                                       const char** output_key_out) {
    cJSON* node = ptq_template_node_by_id(root, layer->node_id);
    cJSON* inputs = node ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const char* op = layer->kind == VX_PTQ_LAYER_QLINEAR
        ? "QLinear" : "QConv2D";
    int expected_inputs = layer->has_bias ? 3 : 2;
    const char* output_key = NULL;
    cJSON* params = node ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    cJSON* layout_json = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "weight_layout") : NULL;
    int params_valid = layer->kind == VX_PTQ_LAYER_QLINEAR
        ? cJSON_IsObject(params) && cJSON_GetArraySize(params) == 0
        : cJSON_IsString(layout_json) && layout_json->valuestring &&
          (vx_generated_node_param_symbol_from_json(layout_json->valuestring) == VX_NODE_SYMBOL_OHWI);
    if (!cJSON_IsObject(node) || !ptq_json_string_is(node, "opType", op) ||
        !cJSON_IsObject(inputs) || cJSON_GetArraySize(inputs) != expected_inputs ||
        !params_valid ||
        ptq_json_activation_input(
            node, layer->kind,
            ptq_template_activation(plan, layer->input_name)) != 0 ||
        !ptq_json_string_is(inputs, "weight", layer->packed_weight) ||
        (layer->has_bias && !ptq_json_string_is(inputs, "bias", layer->packed_bias)) ||
        (!layer->has_bias && cJSON_GetObjectItemCaseSensitive(inputs, "bias")) ||
        ptq_json_output_key(
            node, ptq_template_activation(plan, layer->output_name),
            &output_key) != 0)
        return -1;
    *node_out = node;
    *output_key_out = output_key;
    return 0;
}

static int ptq_engine_source_f32(const SafetensorsFile* source_file,
                                 const char* name, int expected_ndim,
                                 float** values_out, int shape[8],
                                 int* ndim_out, int64_t* count_out) {
    T* tensor = name ? t_find(name) : NULL;
    if (!source_file || !ptq_valid_name(name) || !values_out || !shape ||
        !ndim_out || !count_out || !tensor ||
        !volvoxai_engine_tensor_is_model_weight_locked(name) ||
        tensor->dtype != T_F32 || tensor->elem_size != sizeof(float) ||
        !tensor->data || tensor->numel <= 0 || tensor->ndim != expected_ndim ||
        tensor->ndim <= 0 || tensor->ndim > 8 ||
        (uint64_t)tensor->numel > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        vk_sync_host_tensor(tensor) != 0) return -1;
    materialize_tensor_f32(tensor);
    for (int index = 0; index < tensor->ndim; index++) shape[index] = tensor->shape[index];
    const SafetensorsTensor* stored = safetensors_find_tensor(source_file, name);
    if (!stored || stored->dtype != SAFETENSORS_DTYPE_F32 ||
        stored->ndim != tensor->ndim ||
        stored->nbytes != (size_t)tensor->numel * sizeof(float))
        return -1;
    for (int index = 0; index < tensor->ndim; index++) {
        if (stored->shape[index] != shape[index]) return -1;
    }
    size_t nbytes = (size_t)tensor->numel * sizeof(float);
    if (memcmp(tensor->data, stored->data, nbytes)) return -1;
    float* values = (float*)malloc(nbytes);
    if (!values) {
        free(values);
        return -1;
    }
    memcpy(values, tensor->data, nbytes);
    *values_out = values;
    *ndim_out = tensor->ndim;
    *count_out = tensor->numel;
    return 0;
}

static int ptq_output_path_available(const char* path) {
    if (!path || !path[0] || strlen(path) >= PATH_MAX - 64) return 0;
#if defined(__wasm__)
    FILE* file = fopen(path, "rb");
    if (!file) return 1;
    fclose(file);
    return 0;
#else
    struct stat info;
    if (stat(path, &info) == 0) return 0;
    return errno == ENOENT;
#endif
}

static int ptq_publish_no_replace(const char* staged, const char* output) {
    if (!staged || !output) return -1;
#if defined(_WIN32)
    if (!MoveFileExA(staged, output, MOVEFILE_WRITE_THROUGH)) return -1;
    return 0;
#elif defined(__wasm__)
    /* Calls into one module are serialized. The VFS cannot change between
     * the no-replace check and publication. */
    return ptq_output_path_available(output) ? rename(staged, output) : -1;
#elif defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
    if (link(staged, output) != 0) return -1;
    if (unlink(staged) != 0) {
        /* The destination is already committed. Best-effort cleanup is the
           only safe action; never roll back a successfully linked output. */
        (void)remove(staged);
    }
    return 0;
#else
    (void)staged;
    (void)output;
    return -1;
#endif
}

static int ptq_staging_path(const char* output, const char* suffix,
                            char path[PATH_MAX]) {
    uint64_t sequence = atomic_fetch_add_explicit(
        &g_ptq_package_sequence, 1, memory_order_relaxed);
    int written = snprintf(path, PATH_MAX, "%s.ptqtmp.%ld.%llu.%s", output,
                           VX_PTQ_PROCESS_ID(),
                           (unsigned long long)sequence, suffix);
    return written > 0 && written < PATH_MAX ? 0 : -1;
}

static int ptq_write_text(const char* path, const char* text) {
    if (!path || !text) return -1;
    size_t length = strlen(text);
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    int result = length && fwrite(text, 1, length, file) != length ? -1 : 0;
    if (fflush(file) != 0) result = -1;
    if (fclose(file) != 0) result = -1;
    if (result != 0) remove(path);
    return result;
}

static int ptq_optional_layout_is(cJSON* params, VxNodeParamKey key,
                                  VxNodeParamSymbol expected) {
    cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, vx_generated_node_param_name(key)) : NULL;
    return !value || (cJSON_IsString(value) && value->valuestring &&
        vx_generated_node_param_symbol_from_json(value->valuestring) == expected);
}

static int ptq_json_dimension(cJSON* value, int minimum, int* output) {
    if (!value || !output || !cJSON_IsNumber(value) ||
        !isfinite(value->valuedouble) ||
        value->valuedouble != (double)value->valueint ||
        value->valueint < minimum) return -1;
    *output = value->valueint;
    return 0;
}

static int ptq_json_pair(cJSON* params, const char* key, int fallback,
                         int minimum, int output[2]) {
    cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, key) : NULL;
    if (!value || cJSON_IsNull(value)) {
        output[0] = fallback;
        output[1] = fallback;
        return 0;
    }
    if (!cJSON_IsArray(value)) {
        if (ptq_json_dimension(value, minimum, &output[0]) != 0) return -1;
        output[1] = output[0];
        return 0;
    }
    int count = cJSON_GetArraySize(value);
    if (count < 1 || count > 2 ||
        ptq_json_dimension(cJSON_GetArrayItem(value, 0), minimum,
                           &output[0]) != 0) return -1;
    if (count == 1) output[1] = output[0];
    else if (ptq_json_dimension(cJSON_GetArrayItem(value, 1), minimum,
                                &output[1]) != 0) return -1;
    return 0;
}

static int ptq_json_pads(cJSON* params, const int padding[2], int pads[4]) {
    cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "pads") : NULL;
    if (!value || cJSON_IsNull(value)) {
        pads[0] = padding[0];
        pads[1] = padding[1];
        pads[2] = padding[0];
        pads[3] = padding[1];
        return 0;
    }
    if (!cJSON_IsArray(value) || cJSON_GetArraySize(value) != 4) return -1;
    for (int index = 0; index < 4; index++) {
        if (ptq_json_dimension(cJSON_GetArrayItem(value, index), 0,
                               &pads[index]) != 0) return -1;
    }
    return 0;
}

static int ptq_node_ref_is_once(const Ref* refs, int count, VxPortKind port,
                                const char* name) {
    int matches = 0;
    for (int index = 0; index < count; index++) {
        if (refs[index].port == port) {
            if (strcmp(refs[index].name, name)) return 0;
            matches++;
        }
    }
    return matches == 1;
}

static int ptq_node_activation_ref(const Node* node, int32_t kind,
                                   const char* name) {
    static const VxPortKind keys[] = {VX_PORT_INPUT, VX_PORT_X, VX_PORT_A};
    int key_count = kind == VX_PTQ_LAYER_QLINEAR ? 3 : 2;
    int matches = 0;
    for (int key = 0; key < key_count; key++) {
        for (int index = 0; index < node->nin; index++) {
            if (node->ins[index].port == keys[key]) {
                if (strcmp(node->ins[index].name, name)) return -1;
                matches++;
            }
        }
    }
    return matches == 1 ? 0 : -1;
}

static int ptq_explicit_layout(cJSON* params, VxNodeParamKey key,
                               VxNodeParamSymbol expected) {
    cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, vx_generated_node_param_name(key)) : NULL;
    return cJSON_IsString(value) && value->valuestring &&
        vx_generated_node_param_symbol_from_json(value->valuestring) == expected;
}

static int ptq_tensor_prefix_matches(const T* left, const T* right) {
    if (!left || !right || left->ndim <= 0 || left->ndim != right->ndim)
        return 0;
    for (int index = 0; index < left->ndim - 1; index++) {
        if (left->shape[index] != right->shape[index]) return 0;
    }
    return 1;
}

static int ptq_linear_weight_shape(const int* source, cJSON* params,
                                    int packed[2]) {
    int transposed = ptq_explicit_layout(
        params, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_DIN_DOUT);
    if (!transposed && !ptq_explicit_layout(
            params, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_DOUT_DIN))
        return -1;
    if (source[0] <= 0 || source[1] <= 0) return -1;
    packed[0] = source[transposed ? 1 : 0];
    packed[1] = source[transposed ? 0 : 1];
    return 0;
}

/* Linear calibration reads the source matrix in its declared layout. Pack a
 * private [dout, din] copy before computing per-output-channel scales. */
static int ptq_linear_pack_layout(float** values, int shape[8], int64_t count,
                                   cJSON* params) {
    int packed[2];
    if (ptq_linear_weight_shape(shape, params, packed) != 0 ||
        count <= 0 || (uint64_t)count > SIZE_MAX / sizeof(float) ||
        count != (int64_t)packed[0] * packed[1]) return -1;
    if (ptq_explicit_layout(
            params, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_DIN_DOUT)) {
        float* transposed = (float*)malloc((size_t)count * sizeof(*transposed));
        if (!transposed) return -1;
        for (size_t output = 0; output < (size_t)packed[0]; output++) {
            for (size_t input = 0; input < (size_t)packed[1]; input++) {
                transposed[output * (size_t)packed[1] + input] =
                    (*values)[input * (size_t)packed[0] + output];
            }
        }
        free(*values);
        *values = transposed;
    }
    memcpy(shape, packed, sizeof(packed));
    return 0;
}

/* The source keeps its declared layout throughout calibration. Only the
 * private packing copy moves output channels to the leading OHWI axis. */
static int ptq_conv_weight_shape(const int* source, cJSON* params,
                                  int packed[4]) {
    int hwio = ptq_explicit_layout(
        params, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_HWIO);
    if (!hwio && !ptq_explicit_layout(
            params, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_OHWI)) return -1;
    for (int index = 0; index < 4; index++) {
        int axis = hwio ? (index + 3) % 4 : index;
        if (source[axis] <= 0) return -1;
        packed[index] = source[axis];
    }
    return 0;
}

static int ptq_conv_pack_layout(float** values, int shape[8], int64_t count,
                                 cJSON* params) {
    int packed[4];
    if (ptq_conv_weight_shape(shape, params, packed) != 0) return -1;
    if (ptq_explicit_layout(params, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_HWIO)) {
        if (count <= 0 || (uint64_t)count > SIZE_MAX / sizeof(float) ||
            count % packed[0]) return -1;
        float* transposed = (float*)malloc((size_t)count * sizeof(*transposed));
        if (!transposed) return -1;
        size_t channels = (size_t)packed[0];
        size_t row = (size_t)count / channels;
        for (size_t channel = 0; channel < channels; channel++) {
            for (size_t index = 0; index < row; index++) {
                transposed[channel * row + index] = (*values)[index * channels + channel];
            }
        }
        free(*values);
        *values = transposed;
    }
    memcpy(shape, packed, sizeof(packed));
    return 0;
}

static int ptq_conv_geometry_loaded(const T* input, const T* output,
                                    const T* weight, cJSON* params) {
    int weight_shape[4];
    int stride[2] = {0};
    int dilation[2] = {0};
    int padding[2] = {0};
    int pads[4] = {0};
    int groups = 1;
    cJSON* groups_json = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "groups") : NULL;
    if (!input || !output || !weight || input->ndim != 4 || output->ndim != 4 ||
        weight->ndim != 4 || !ptq_explicit_layout(params, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC) ||
        ptq_conv_weight_shape(weight->shape, params, weight_shape) != 0 ||
        ptq_json_pair(params, "stride", 1, 1, stride) != 0 ||
        ptq_json_pair(params, "dilation", 1, 1, dilation) != 0 ||
        ptq_json_pair(params, "padding", 0, 0, padding) != 0 ||
        ptq_json_pads(params, padding, pads) != 0) return -1;
    if (groups_json && ptq_json_dimension(groups_json, 1, &groups) != 0) return -1;
    if (input->shape[0] != output->shape[0] ||
        weight_shape[3] > INT_MAX / groups ||
        input->shape[3] != weight_shape[3] * groups ||
        output->shape[3] != weight_shape[0] ||
        input->shape[3] % groups || output->shape[3] % groups) return -1;
    uint64_t padded_height = (uint64_t)input->shape[1] +
        (uint64_t)pads[0] + (uint64_t)pads[2];
    uint64_t padded_width = (uint64_t)input->shape[2] +
        (uint64_t)pads[1] + (uint64_t)pads[3];
    uint64_t effective_height =
        (uint64_t)(weight_shape[1] - 1) * (uint64_t)dilation[0] + 1u;
    uint64_t effective_width =
        (uint64_t)(weight_shape[2] - 1) * (uint64_t)dilation[1] + 1u;
    if (padded_height < effective_height || padded_width < effective_width)
        return -1;
    uint64_t expected_height =
        (padded_height - effective_height) / (uint64_t)stride[0] + 1u;
    uint64_t expected_width =
        (padded_width - effective_width) / (uint64_t)stride[1] + 1u;
    return expected_height == (uint64_t)output->shape[1] &&
           expected_width == (uint64_t)output->shape[2] ? 0 : -1;
}

static int ptq_conv_params_equal(cJSON* source, cJSON* target) {
    int source_stride[2], target_stride[2];
    int source_dilation[2], target_dilation[2];
    int source_padding[2], target_padding[2];
    int source_pads[4], target_pads[4];
    int source_groups = 1, target_groups = 1;
    int source_relu = 0, target_relu = 0;
    cJSON* source_groups_json = cJSON_IsObject(source)
        ? cJSON_GetObjectItemCaseSensitive(source, "groups") : NULL;
    cJSON* target_groups_json = cJSON_IsObject(target)
        ? cJSON_GetObjectItemCaseSensitive(target, "groups") : NULL;
    cJSON* source_relu_json = cJSON_IsObject(source)
        ? cJSON_GetObjectItemCaseSensitive(source, "relu") : NULL;
    cJSON* target_relu_json = cJSON_IsObject(target)
        ? cJSON_GetObjectItemCaseSensitive(target, "relu") : NULL;
    if (!ptq_explicit_layout(source, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC) ||
        (!ptq_explicit_layout(source, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_OHWI) &&
         !ptq_explicit_layout(source, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_HWIO)) ||
        !ptq_explicit_layout(target, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC) ||
        !ptq_explicit_layout(target, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_OHWI) ||
        ptq_json_pair(source, "stride", 1, 1, source_stride) != 0 ||
        ptq_json_pair(target, "stride", 1, 1, target_stride) != 0 ||
        ptq_json_pair(source, "dilation", 1, 1, source_dilation) != 0 ||
        ptq_json_pair(target, "dilation", 1, 1, target_dilation) != 0 ||
        ptq_json_pair(source, "padding", 0, 0, source_padding) != 0 ||
        ptq_json_pair(target, "padding", 0, 0, target_padding) != 0 ||
        ptq_json_pads(source, source_padding, source_pads) != 0 ||
        ptq_json_pads(target, target_padding, target_pads) != 0 ||
        (source_groups_json &&
         ptq_json_dimension(source_groups_json, 1, &source_groups) != 0) ||
        (target_groups_json &&
         ptq_json_dimension(target_groups_json, 1, &target_groups) != 0) ||
        (source_relu_json && !cJSON_IsNull(source_relu_json) &&
         ptq_json_dimension(source_relu_json, 0, &source_relu) != 0) ||
        (target_relu_json && !cJSON_IsNull(target_relu_json) &&
         ptq_json_dimension(target_relu_json, 0, &target_relu) != 0)) return 0;
    return !memcmp(source_stride, target_stride, sizeof(source_stride)) &&
           !memcmp(source_dilation, target_dilation, sizeof(source_dilation)) &&
           !memcmp(source_pads, target_pads, sizeof(source_pads)) &&
           source_groups == target_groups && source_relu == target_relu;
}

static int ptq_validate_loaded_layer_locked(const VxPTQPlanLayer* layer) {
    if (!layer || layer->node_index < 0 || layer->node_index >= g_nn) return -1;
    const Node* node = &g_n[layer->node_index];
    VxOperatorKind expected_op = layer->kind == VX_PTQ_LAYER_QLINEAR
        ? VX_OP_LINEAR : VX_OP_CONV_2D;
    int expected_inputs = layer->has_source_bias ? 3 : 2;
    T* input = t_find(layer->input_name);
    T* output = t_find(layer->output_name);
    T* weight = t_find(layer->source_weight);
    T* bias = layer->has_source_bias ? t_find(layer->source_bias) : NULL;
    if (node->operator_kind != expected_op || node->disabled ||
        node->nin != expected_inputs || node->nout != 1 ||
        ptq_node_activation_ref(node, layer->kind, layer->input_name) != 0 ||
        !ptq_node_ref_is_once(node->ins, node->nin, VX_PORT_WEIGHT,
                              layer->source_weight) ||
        (layer->has_source_bias &&
         !ptq_node_ref_is_once(node->ins, node->nin, VX_PORT_BIAS,
                               layer->source_bias)) ||
        (!layer->has_source_bias &&
         ptq_node_ref_is_once(node->ins, node->nin, VX_PORT_BIAS, "")) ||
        strcmp(node->outs[0].name, layer->output_name) ||
        strcmp(node->out, layer->output_name) || !input || !output || !weight ||
        input->dtype != T_F32 || output->dtype != T_F32 ||
        weight->dtype != T_F32 || !weight->data ||
        !volvoxai_engine_tensor_is_model_weight_locked(layer->source_weight) ||
        (layer->has_source_bias &&
         (!bias || bias->dtype != T_F32 || !bias->data ||
          !volvoxai_engine_tensor_is_model_weight_locked(layer->source_bias))))
        return -1;
    if (layer->kind == VX_PTQ_LAYER_QLINEAR) {
        int weight_shape[2];
        if (weight->ndim != 2 ||
            ptq_linear_weight_shape(weight->shape, node->params, weight_shape) != 0 ||
            input->ndim <= 0 ||
            !ptq_tensor_prefix_matches(input, output) ||
            input->shape[input->ndim - 1] != weight_shape[1] ||
            output->shape[output->ndim - 1] != weight_shape[0] ||
            !layer->has_bias || (layer->has_source_bias &&
            (bias->ndim != 1 || bias->shape[0] != weight_shape[0]))) return -1;
        return 0;
    }
    int weight_shape[4];
    if (weight->ndim != 4 ||
        ptq_conv_weight_shape(weight->shape, node->params, weight_shape) != 0 ||
        (layer->has_source_bias &&
         (bias->ndim != 1 || bias->shape[0] != weight_shape[0]))) return -1;
    return ptq_conv_geometry_loaded(input, output, weight, node->params);
}

/* Shapes are read out of the template, so the names must be the template's:
   authoring renamed every activation on its way into the byte domain, and the
   source names it started from are no longer declared anywhere in it. */
static int ptq_layer_basic_shapes(const VolvoxAIPTQPlan* plan, cJSON* root,
                                  cJSON* node, const VxPTQPlanLayer* layer,
                                  const int weight_shape[8], int weight_ndim) {
    cJSON* input_shape = ptq_json_tensor_shape(
        root, ptq_template_activation(plan, layer->input_name));
    cJSON* output_shape = ptq_json_tensor_shape(
        root, ptq_template_activation(plan, layer->output_name));
    int input_rank = cJSON_IsArray(input_shape) ? cJSON_GetArraySize(input_shape) : -1;
    int output_rank = cJSON_IsArray(output_shape) ? cJSON_GetArraySize(output_shape) : -1;
    int input_channels = 0;
    int output_channels = 0;
    if (!input_shape || !output_shape ||
        ptq_json_shape_dim(input_shape, -1, &input_channels) != 0 ||
        ptq_json_shape_dim(output_shape, -1, &output_channels) != 0 ||
        output_channels != weight_shape[0]) return -1;
    if (layer->kind == VX_PTQ_LAYER_QLINEAR) {
        if (weight_ndim != 2 || input_rank <= 0 || output_rank != input_rank ||
            input_channels != weight_shape[1]) return -1;
        for (int index = 0; index < input_rank - 1; index++) {
            if (!ptq_json_shape_axis_equal(input_shape, output_shape, index))
                return -1;
        }
        return 0;
    }
    cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    cJSON* groups_json = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "groups") : NULL;
    int groups = 1;
    if (groups_json) {
        if (!cJSON_IsNumber(groups_json) || !isfinite(groups_json->valuedouble) ||
            groups_json->valuedouble != (double)groups_json->valueint ||
            groups_json->valueint <= 0) return -1;
        groups = groups_json->valueint;
    }
    if (weight_ndim != 4 || input_rank != 4 || output_rank != 4 ||
        !ptq_optional_layout_is(params, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC) ||
        !ptq_optional_layout_is(params, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_OHWI) ||
        weight_shape[3] > INT_MAX / groups ||
        input_channels != weight_shape[3] * groups ||
        input_channels % groups || output_channels % groups) return -1;
    int input_dimensions[4] = {0};
    int output_dimensions[4] = {0};
    if (!ptq_json_shape_axis_equal(input_shape, output_shape, 0)) return -1;
    input_dimensions[0] = output_dimensions[0] = 1;
    for (int index = 1; index < 4; index++) {
        if (ptq_json_shape_dim(input_shape, index, &input_dimensions[index]) != 0 ||
            ptq_json_shape_dim(output_shape, index, &output_dimensions[index]) != 0)
            return -1;
    }
    int stride[2] = {0};
    int dilation[2] = {0};
    int padding[2] = {0};
    int pads[4] = {0};
    if (ptq_json_pair(params, "stride", 1, 1, stride) != 0 ||
        ptq_json_pair(params, "dilation", 1, 1, dilation) != 0 ||
        ptq_json_pair(params, "padding", 0, 0, padding) != 0 ||
        ptq_json_pads(params, padding, pads) != 0 ||
        output_dimensions[0] != input_dimensions[0]) return -1;
    cJSON* relu = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "relu") : NULL;
    int relu_value = 0;
    if (relu && !cJSON_IsNull(relu) &&
        (ptq_json_dimension(relu, 0, &relu_value) != 0 || relu_value > 2))
        return -1;
    uint64_t padded_height = (uint64_t)input_dimensions[1] +
        (uint64_t)pads[0] + (uint64_t)pads[2];
    uint64_t padded_width = (uint64_t)input_dimensions[2] +
        (uint64_t)pads[1] + (uint64_t)pads[3];
    uint64_t effective_height =
        (uint64_t)(weight_shape[1] - 1) * (uint64_t)dilation[0] + 1u;
    uint64_t effective_width =
        (uint64_t)(weight_shape[2] - 1) * (uint64_t)dilation[1] + 1u;
    if (padded_height < effective_height || padded_width < effective_width)
        return -1;
    uint64_t expected_height =
        (padded_height - effective_height) / (uint64_t)stride[0] + 1u;
    uint64_t expected_width =
        (padded_width - effective_width) / (uint64_t)stride[1] + 1u;
    return expected_height == (uint64_t)output_dimensions[1] &&
           expected_width == (uint64_t)output_dimensions[2]
        ? 0 : -1;
}

static int ptq_plan_write_package_locked(
    const VolvoxAIPTQPlan* plan,
    const volvoxai_ptq_package_options_t* options) {
    int byte_export = options && options->graph_bytes && options->graph_size &&
        options->weights_bytes && options->weights_size &&
        !(options->output_graph_path && options->output_graph_path[0]) &&
        !(options->output_weights_path && options->output_weights_path[0]);
    if (!plan || !options || options->struct_size != sizeof(*options) ||
        !ptq_plan_is_current_locked(plan) ||
        volvoxai_engine_adapter_effect_active_locked() ||
        !plan->calibration_samples || !plan->tensor_count || !plan->layer_count ||
        !options->template_graph_path || !options->source_weights_path ||
        (!byte_export && (!options->output_graph_path || !options->output_weights_path ||
        !strcmp(options->template_graph_path, options->output_graph_path) ||
        !strcmp(options->source_weights_path, options->output_weights_path) ||
        !strcmp(options->output_graph_path, options->output_weights_path) ||
        !ptq_output_path_available(options->output_graph_path) ||
        !ptq_output_path_available(options->output_weights_path)))) return -1;
    for (int32_t index = 0; index < plan->layer_count; index++) {
        if (ptq_validate_loaded_layer_locked(&plan->layers[index]) != 0) return -1;
    }
    for (int32_t index = 0; index < plan->tensor_count; index++) {
        volvoxai_ptq_params_t params;
        if (volvoxai_ptq_calculate_params(&plan->tensors[index].observer,
                                          plan->tensors[index].dtype,
                                          plan->tensors[index].scheme,
                                          &params) != 0) return -1;
    }

    int result = -1;
    char* template_text = ptq_read_text_file(options->template_graph_path);
    cJSON* root = template_text ? cJSON_Parse(template_text) : NULL;
    free(template_text);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return -1;
    }
    SafetensorsFile weights;
    SafetensorsLoadOptions load_options = {SAFETENSORS_OPEN_READ_WRITE};
    if (safetensors_load_with_options(options->source_weights_path,
                                      &load_options, &weights) != 0) {
        cJSON_Delete(root);
        return -1;
    }
    cJSON* quantization_tensors = NULL;
    if (ptq_template_prepare_quantization(
            root, &weights, &quantization_tensors) != 0) goto done;
    if (ptq_template_dependency_preflight(plan, root, &weights) != 0) goto done;

    /* Every observed activation gets its calibrated affine written under the
       name the template gave it.

       This used to be where the writer decided the public ABI, converting a
       graph input to byte storage and declaring an affine on it. Authoring now
       owns that: it places a QuantizeLinear after each float input and a
       DequantizeLinear before each float output, so the package's boundary is
       the same as the FP32 one it replaces and callers do not change. What is
       left here is measurement — turning observed ranges into scale and zero
       point, and storing them. */
    cJSON* graph_inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    if (!cJSON_IsObject(graph_inputs)) goto done;
    for (int32_t tensor_index = 0; tensor_index < plan->tensor_count; tensor_index++) {
        const VxPTQPlanTensor* tensor = &plan->tensors[tensor_index];
        volvoxai_ptq_params_t params;
        int32_t zero_point;
        if (!tensor->quantized_name[0]) {
            /* A template that renamed nothing has no byte tensor to carry the
               affine, so there is nothing to write. */
            continue;
        }
        if (volvoxai_ptq_calculate_params(&tensor->observer, tensor->dtype,
                                          tensor->scheme, &params) != 0) {
            goto done;
        }
        zero_point = params.zero_point;
        if (ptq_add_quantization_entry(root, quantization_tensors, &weights,
                                       tensor->quantized_name, tensor->dtype,
                                       0, 0, &params.scale, &zero_point,
                                       1) != 0) {
            goto done;
        }
    }

    for (int32_t layer_index = 0; layer_index < plan->layer_count; layer_index++) {
        const VxPTQPlanLayer* layer = &plan->layers[layer_index];
        const VxPTQPlanTensor* input_tensor =
            ptq_find_tensor_const(plan, layer->input_name);
        const VxPTQPlanTensor* output_tensor =
            ptq_find_tensor_const(plan, layer->output_name);
        cJSON* node = NULL;
        const char* output_key = NULL;
        if (!input_tensor || !output_tensor ||
            ptq_template_validate_layer(plan, layer, root, &node,
                                        &output_key) != 0 ||
            (layer->kind == VX_PTQ_LAYER_QCONV2D &&
             !ptq_conv_params_equal(g_n[layer->node_index].params,
                                    cJSON_GetObjectItemCaseSensitive(node,
                                                                     "params"))) ||
            /* A declared affine for the packed weight is expected, not a
               collision: authoring named it. ptq_add_quantization_entry
               adopts it when it agrees with the plan and refuses it when it
               does not, so the decision belongs there. */
            safetensors_find_tensor(&weights, layer->packed_weight) ||
            (layer->has_bias &&
             safetensors_find_tensor(&weights, layer->packed_bias))) goto done;

        int expected_ndim = layer->kind == VX_PTQ_LAYER_QLINEAR ? 2 : 4;
        float* source_weight = NULL;
        int weight_shape[8] = {0};
        int weight_ndim = 0;
        int64_t weight_count = 0;
        if (ptq_engine_source_f32(&weights, layer->source_weight, expected_ndim,
                                  &source_weight, weight_shape, &weight_ndim,
                                  &weight_count) != 0 ||
            (layer->kind == VX_PTQ_LAYER_QLINEAR &&
             ptq_linear_pack_layout(&source_weight, weight_shape, weight_count,
                                    g_n[layer->node_index].params) != 0) ||
            (layer->kind == VX_PTQ_LAYER_QCONV2D &&
             ptq_conv_pack_layout(&source_weight, weight_shape, weight_count,
                                  g_n[layer->node_index].params) != 0) ||
            ptq_layer_basic_shapes(plan, root, node, layer, weight_shape,
                                   weight_ndim) != 0 ||
            weight_shape[0] <= 0 || (uint64_t)weight_count > (uint64_t)SIZE_MAX) {
            free(source_weight);
            goto done;
        }
        int32_t packed_shape[8] = {0};
        for (int index = 0; index < weight_ndim; index++) {
            if (weight_shape[index] <= 0) {
                free(source_weight);
                goto done;
            }
            packed_shape[index] = weight_shape[index];
        }
        int32_t channel_count = weight_shape[0];
        if ((size_t)channel_count > SIZE_MAX / sizeof(float) ||
            (size_t)channel_count > SIZE_MAX / sizeof(int32_t)) {
            free(source_weight);
            goto done;
        }
        int8_t* packed_weight = (int8_t*)malloc((size_t)weight_count);
        float* scales = (float*)malloc((size_t)channel_count * sizeof(*scales));
        float* source_bias = NULL;
        int32_t* packed_bias = NULL;
        int32_t* zero_points = NULL;
        int bias_shape[8] = {0};
        int bias_ndim = 0;
        int64_t bias_count = 0;
        volvoxai_ptq_params_t input_params;
        memset(&input_params, 0, sizeof(input_params));
        if (layer->has_bias) {
            packed_bias = (int32_t*)calloc((size_t)channel_count, sizeof(*packed_bias));
            bias_shape[0] = channel_count;
            bias_ndim = 1;
            if (!packed_bias ||
                volvoxai_ptq_calculate_params(&input_tensor->observer,
                                              input_tensor->dtype,
                                              input_tensor->scheme,
                                              &input_params) != 0 ||
                (layer->has_source_bias &&
                 (ptq_engine_source_f32(&weights, layer->source_bias, 1,
                                      &source_bias, bias_shape, &bias_ndim,
                                      &bias_count) != 0 ||
                bias_ndim != 1 || bias_count != channel_count ||
                bias_shape[0] != channel_count))) {
                free(source_weight);
                free(packed_weight);
                free(scales);
                free(source_bias);
                free(packed_bias);
                goto done;
            }
        }
        int packed_ok = 0;
#if VOLVOXAI_ENABLE_CUDA
        if (g_use_cuda && (uint64_t)weight_count <= UINT32_MAX &&
            weight_count % channel_count == 0 &&
            (uint64_t)(weight_count / channel_count) <= UINT32_MAX) {
            zero_points = (int32_t*)malloc(
                (size_t)channel_count * sizeof(*zero_points));
            packed_ok = zero_points && cuda_training_quantize_w8_available() &&
                cuda_training_quantize_w8_f32(
                    source_weight, packed_weight, scales, zero_points,
                    (uint32_t)channel_count,
                    (uint32_t)(weight_count / channel_count), 0u, 0u, 0u,
                    source_bias, input_params.scale, packed_bias,
                    NULL, NULL);
            for (int32_t channel = 0; packed_ok && channel < channel_count;
                 channel++) {
                if (zero_points[channel] != 0) packed_ok = 0;
            }
        } else
#endif
        {
            packed_ok = packed_weight && scales &&
                volvoxai_ptq_pack_weight_i8(
                    source_weight, packed_shape, weight_ndim, 0,
                    packed_weight, scales, channel_count, NULL) == 0 &&
                (!layer->has_source_bias ||
                 volvoxai_ptq_pack_bias_i32(
                     source_bias, channel_count, input_params.scale, scales,
                     channel_count, packed_bias) == 0);
        }
        if (!packed_ok ||
            safetensors_add_tensor(&weights, layer->packed_weight,
                                   SAFETENSORS_DTYPE_I8, weight_shape,
                                   weight_ndim, packed_weight,
                                   (size_t)weight_count) != 0 ||
            (layer->has_bias &&
             safetensors_add_tensor(
                 &weights, layer->packed_bias, SAFETENSORS_DTYPE_I32,
                 bias_shape, 1, packed_bias,
                 (size_t)channel_count * sizeof(*packed_bias)) != 0)) {
            free(source_weight);
            free(packed_weight);
            free(scales);
            free(source_bias);
            free(packed_bias);
            free(zero_points);
            goto done;
        }
        free(source_weight);
        free(packed_weight);
        free(source_bias);
        free(packed_bias);
        int metadata_ok = ptq_add_quantization_entry(
            root, quantization_tensors, &weights, layer->packed_weight,
            VOLVOXAI_DTYPE_I8, 1, 0, scales, zero_points, channel_count);
        free(scales);
        free(zero_points);
        /* The output descriptor is authoring's, not the writer's: the
           template already declares the byte storage and the affine, and the
           activation loop above filled in that affine's payload. Rewriting it
           here would make the writer a second authority on the same fact. */
        (void)output_key;
        (void)output_tensor;
        if (metadata_ok != 0) goto done;
    }
    /* Keep pass-through tensors, but do not make a quantized package carry the
       selected FP32 source payload when the explicit target graph no longer
       references it. Shared sources still used by another target node remain. */
    for (int32_t layer_index = 0; layer_index < plan->layer_count; layer_index++) {
        const VxPTQPlanLayer* layer = &plan->layers[layer_index];
        if (!ptq_template_node_references(root, layer->source_weight) &&
            safetensors_find_tensor(&weights, layer->source_weight) &&
            safetensors_remove_tensor(&weights, layer->source_weight) != 0) goto done;
        if (layer->has_source_bias &&
            !ptq_template_node_references(root, layer->source_bias) &&
            safetensors_find_tensor(&weights, layer->source_bias) &&
            safetensors_remove_tensor(&weights, layer->source_bias) != 0) goto done;
    }
    if (ptq_set_authoring_metadata(&weights, options) != 0) goto done;
    char* graph_text = cJSON_PrintUnformatted(root);
    if (byte_export) {
        unsigned char* weights_bytes = NULL;
        size_t weights_size = 0;
        if (!graph_text ||
            safetensors_serialize(&weights, &weights_bytes, &weights_size) != 0) {
            free(graph_text);
            goto done;
        }
        *options->graph_bytes = (unsigned char*)graph_text;
        *options->graph_size = strlen(graph_text);
        *options->weights_bytes = weights_bytes;
        *options->weights_size = weights_size;
        result = 0;
        goto done;
    }
    char staged_graph[PATH_MAX];
    char staged_weights[PATH_MAX];
    if (!graph_text ||
        ptq_staging_path(options->output_graph_path, "graph",
                         staged_graph) != 0 ||
        ptq_staging_path(options->output_weights_path, "weights",
                         staged_weights) != 0) {
        free(graph_text);
        goto done;
    }
    remove(staged_graph);
    remove(staged_weights);
    if (safetensors_save(staged_weights, &weights) != 0 ||
        ptq_write_text(staged_graph, graph_text) != 0) {
        free(graph_text);
        remove(staged_graph);
        remove(staged_weights);
        goto done;
    }
    free(graph_text);
    if (ptq_publish_no_replace(staged_weights,
                               options->output_weights_path) != 0) {
        remove(staged_graph);
        remove(staged_weights);
        goto done;
    }
    if (ptq_publish_no_replace(staged_graph,
                               options->output_graph_path) != 0) {
        remove(staged_graph);
        remove(options->output_weights_path);
        goto done;
    }
    result = 0;
done:
    safetensors_free(&weights);
    cJSON_Delete(root);
    return result;
}

int volvoxai_ptq_plan_write_package(
    const VolvoxAIPTQPlan* plan,
    const volvoxai_ptq_package_options_t* options) {
    if (!plan || volvoxai_engine_model_route_lease_active()) return -1;
    /* One model lease covers every borrowed Node/T pointer, all source-byte
       checks, and both publications. Reload, shutdown, graph patches, and
       weight updates therefore either happen before this call or after it. */
    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    int result = ptq_plan_write_package_locked(plan, options);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return result;
}
