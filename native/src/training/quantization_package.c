#include "volvoxai_training.h"

#include "cJSON.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "safetensors.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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

typedef struct {
    char name[VX_PTQ_NAME_CAPACITY];
    int32_t dtype;
    int32_t scheme;
    volvoxai_ptq_observer_t observer;
} VxPTQPlanTensor;

typedef struct {
    int32_t kind;
    int32_t node_index;
    int32_t weight_axis;
    char input_name[VX_PTQ_NAME_CAPACITY];
    char output_name[VX_PTQ_NAME_CAPACITY];
    char source_weight[VX_PTQ_NAME_CAPACITY];
    char packed_weight[VX_PTQ_NAME_CAPACITY];
    char source_bias[VX_PTQ_NAME_CAPACITY];
    char packed_bias[VX_PTQ_NAME_CAPACITY];
    int has_bias;
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
           (scheme == VOLVOXAI_PTQ_SYMMETRIC ||
            scheme == VOLVOXAI_PTQ_ASYMMETRIC);
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
    if (!plan || !spec || spec->struct_size < sizeof(*spec) ||
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
    if (!plan || !spec || spec->struct_size < sizeof(*spec) ||
        spec->node_index < 0 ||
        spec->weight_axis != 0 ||
        (spec->kind != VOLVOXAI_PTQ_LAYER_QLINEAR &&
         spec->kind != VOLVOXAI_PTQ_LAYER_QCONV2D) ||
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
    if (has_source_bias != has_packed_bias ||
        (spec->kind == VOLVOXAI_PTQ_LAYER_QLINEAR && !has_source_bias)) return -1;
    if (has_source_bias &&
        (!ptq_valid_name(spec->source_bias_name) ||
         !ptq_valid_name(spec->packed_bias_name) ||
         !strcmp(spec->source_bias_name, spec->packed_bias_name) ||
         !strcmp(spec->packed_weight_name, spec->packed_bias_name))) return -1;
    VxPTQPlanLayer candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.kind = spec->kind;
    candidate.node_index = spec->node_index;
    candidate.weight_axis = spec->weight_axis;
    candidate.has_bias = has_source_bias;
    if (ptq_copy_name(candidate.input_name, spec->input_tensor_name) != 0 ||
        ptq_copy_name(candidate.output_name, spec->output_tensor_name) != 0 ||
        ptq_copy_name(candidate.source_weight, spec->source_weight_name) != 0 ||
        ptq_copy_name(candidate.packed_weight, spec->packed_weight_name) != 0 ||
        (has_source_bias &&
         (ptq_copy_name(candidate.source_bias, spec->source_bias_name) != 0 ||
          ptq_copy_name(candidate.packed_bias, spec->packed_bias_name) != 0))) return -1;

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
    cJSON* graph_inputs = g_cfg_root
        ? cJSON_GetObjectItemCaseSensitive(g_cfg_root, "inputs") : NULL;
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
        if (binding->struct_size < sizeof(*binding) ||
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

static cJSON* ptq_quant_descriptor(const volvoxai_ptq_params_t* params) {
    if (!params || !isfinite(params->scale) || params->scale <= 0.0f) return NULL;
    cJSON* descriptor = cJSON_CreateObject();
    if (!descriptor ||
        ptq_json_add_owned(descriptor, "scheme",
                           cJSON_CreateString("per_tensor")) != 0 ||
        ptq_json_add_owned(descriptor, "scale",
                           cJSON_CreateNumber(params->scale)) != 0 ||
        ptq_json_add_owned(descriptor, "zero_point",
                           cJSON_CreateNumber(params->zero_point)) != 0) {
        cJSON_Delete(descriptor);
        return NULL;
    }
    return descriptor;
}

static cJSON* ptq_weight_descriptor(const float* scales, int32_t count) {
    if (!scales || count <= 0) return NULL;
    cJSON* descriptor = cJSON_CreateObject();
    cJSON* scale_array = cJSON_CreateArray();
    cJSON* zero_array = cJSON_CreateArray();
    if (!descriptor || !scale_array || !zero_array) goto fail;
    for (int32_t index = 0; index < count; index++) {
        if (!isfinite(scales[index]) || scales[index] <= 0.0f ||
            !cJSON_AddItemToArray(scale_array, cJSON_CreateNumber(scales[index])) ||
            !cJSON_AddItemToArray(zero_array, cJSON_CreateNumber(0))) goto fail;
    }
    if (ptq_json_add_owned(descriptor, "scheme",
                           cJSON_CreateString("per_axis")) != 0 ||
        ptq_json_add_owned(descriptor, "axis", cJSON_CreateNumber(0)) != 0)
        goto fail;
    if (ptq_json_add_owned(descriptor, "scales", scale_array) != 0) {
        scale_array = NULL; /* ptq_json_add_owned deleted it. */
        goto fail;
    }
    scale_array = NULL;
    if (ptq_json_add_owned(descriptor, "zero_points", zero_array) != 0) {
        zero_array = NULL; /* ptq_json_add_owned deleted it. */
        goto fail;
    }
    zero_array = NULL;
    return descriptor;
fail:
    cJSON_Delete(scale_array);
    cJSON_Delete(zero_array);
    cJSON_Delete(descriptor);
    return NULL;
}

static int ptq_json_object_ensure(cJSON* parent, const char* key,
                                  cJSON** output) {
    cJSON* object = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (!object) {
        object = cJSON_CreateObject();
        if (ptq_json_add_owned(parent, key, object) != 0) return -1;
    }
    if (!cJSON_IsObject(object)) return -1;
    *output = object;
    return 0;
}

static int ptq_json_output_key(cJSON* node, const char* tensor_name,
                               const char** output_key) {
    cJSON* outputs = node ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
    if (!cJSON_IsObject(outputs) || !outputs->child || outputs->child->next ||
        !outputs->child->string || !cJSON_IsString(outputs->child) ||
        !outputs->child->valuestring ||
        strcmp(outputs->child->valuestring, tensor_name)) return -1;
    *output_key = outputs->child->string;
    return 0;
}

static int ptq_json_activation_input(cJSON* node, int32_t kind,
                                     const char* expected) {
    cJSON* inputs = node ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const char* keys[] = {"input", "x", "a"};
    int key_count = kind == VOLVOXAI_PTQ_LAYER_QLINEAR ? 3 : 2;
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
        cJSON* shapes = cJSON_GetObjectItemCaseSensitive(node, "outputs_shape");
        shape = cJSON_IsObject(shapes)
            ? cJSON_GetObjectItemCaseSensitive(shapes, output_key) : NULL;
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
            if (!cJSON_IsString(output) || !output->valuestring) return -2;
            if (!strcmp(output->valuestring, tensor_name)) {
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
    if (cJSON_IsArray(outputs) || cJSON_IsObject(outputs)) {
        for (cJSON* item = outputs->child; item; item = item->next) {
            if ((item->string && !strcmp(item->string, name)) ||
                (cJSON_IsString(item) && item->valuestring &&
                 !strcmp(item->valuestring, name))) return 1;
        }
    }
    return 0;
}

static SafetensorsDType ptq_safetensors_dtype(int dtype) {
    switch (dtype) {
        case T_F32: return SAFETENSORS_DTYPE_F32;
        case T_F16: return SAFETENSORS_DTYPE_F16;
        case T_I8: return SAFETENSORS_DTYPE_I8;
        case T_U8: return SAFETENSORS_DTYPE_U8;
        case T_I32: return SAFETENSORS_DTYPE_I32;
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

static int ptq_template_dependency_preflight(const VolvoxAIPTQPlan* plan,
                                             cJSON* root,
                                             const SafetensorsFile* weights) {
    cJSON* graph_inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if (!cJSON_IsObject(graph_inputs) || !cJSON_IsArray(nodes)) return -1;
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
            if (ptq_loaded_weight_matches(weights, name)) continue;
            /* The package emits one weight file. A dependency available only
               through another loaded shard would otherwise author a package
               that cannot be reloaded from the advertised pair. */
            return -1;
        }
    }
    return 0;
}

static int ptq_template_set_graph_input(cJSON* root,
                                        const VxPTQPlanTensor* tensor) {
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    cJSON* descriptor = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, tensor->name) : NULL;
    const char* dtype_name = ptq_dtype_name(tensor->dtype);
    volvoxai_ptq_params_t params;
    if (!cJSON_IsObject(descriptor) || !dtype_name ||
        !ptq_json_string_is(descriptor, "dtype", dtype_name) ||
        cJSON_GetObjectItemCaseSensitive(descriptor, "quantization") ||
        volvoxai_ptq_calculate_params(&tensor->observer, tensor->dtype,
                                      tensor->scheme, &params) != 0)
        return -1;
    cJSON* quantization = ptq_quant_descriptor(&params);
    return quantization
        ? ptq_json_add_owned(descriptor, "quantization", quantization) : -1;
}

static int ptq_template_set_output(cJSON* node, const char* output_key,
                                   const VxPTQPlanTensor* tensor) {
    cJSON* dtypes = cJSON_GetObjectItemCaseSensitive(node, "outputs_dtype");
    cJSON* quantizations = NULL;
    const char* dtype_name = ptq_dtype_name(tensor->dtype);
    volvoxai_ptq_params_t params;
    if (!cJSON_IsObject(dtypes) || !dtype_name ||
        !ptq_json_string_is(dtypes, output_key, dtype_name) ||
        ptq_json_object_ensure(node, "outputs_quantization", &quantizations) != 0 ||
        cJSON_GetObjectItemCaseSensitive(quantizations, output_key) ||
        volvoxai_ptq_calculate_params(&tensor->observer, tensor->dtype,
                                      tensor->scheme, &params) != 0)
        return -1;
    cJSON* descriptor = ptq_quant_descriptor(&params);
    return descriptor
        ? ptq_json_add_owned(quantizations, output_key, descriptor) : -1;
}

static int ptq_template_validate_layer(const VolvoxAIPTQPlan* plan,
                                       const VxPTQPlanLayer* layer,
                                       cJSON* root, cJSON** node_out,
                                       const char** output_key_out) {
    cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    cJSON* node = cJSON_IsArray(nodes)
        ? cJSON_GetArrayItem(nodes, layer->node_index) : NULL;
    cJSON* inputs = node ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const char* op = layer->kind == VOLVOXAI_PTQ_LAYER_QLINEAR
        ? "QLinear" : "QConv2D";
    int expected_inputs = layer->has_bias ? 3 : 2;
    const char* output_key = NULL;
    cJSON* params = node ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    const char* layout = layer->kind == VOLVOXAI_PTQ_LAYER_QLINEAR
        ? "OUT_IN" : "OHWI";
    cJSON* layout_json = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "weight_layout") : NULL;
    if (!cJSON_IsObject(node) || !ptq_json_string_is(node, "opType", op) ||
        !cJSON_IsObject(inputs) || cJSON_GetArraySize(inputs) != expected_inputs ||
        !cJSON_IsString(layout_json) || !layout_json->valuestring ||
        strcmp(layout_json->valuestring, layout) ||
        ptq_json_activation_input(node, layer->kind, layer->input_name) != 0 ||
        !ptq_json_string_is(inputs, "weight", layer->packed_weight) ||
        (layer->has_bias && !ptq_json_string_is(inputs, "bias", layer->packed_bias)) ||
        (!layer->has_bias && cJSON_GetObjectItemCaseSensitive(inputs, "bias")) ||
        ptq_json_output_key(node, layer->output_name, &output_key) != 0)
        return -1;
    cJSON* graph_inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    if (!cJSON_IsObject(graph_inputs)) return -1;
    if (!cJSON_GetObjectItemCaseSensitive(graph_inputs, layer->input_name) &&
        !ptq_layer_produces_before(plan, layer->input_name, layer->node_index))
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

static int ptq_add_authoring_metadata(cJSON* root,
                                      const VolvoxAIPTQPlan* plan) {
    if (cJSON_GetObjectItemCaseSensitive(root, "ptq_authoring")) return -1;
    cJSON* metadata = cJSON_CreateObject();
    cJSON* tensors = cJSON_CreateObject();
    cJSON* samples = cJSON_CreateArray();
    if (!metadata || !tensors || !samples ||
        ptq_json_add_owned(metadata, "format",
                           cJSON_CreateString("volvoxai-native-ptq-plan-v1")) != 0 ||
        ptq_json_add_owned(metadata, "calibration_samples",
                           cJSON_CreateNumber((double)plan->calibration_samples)) != 0)
        goto fail;
    for (int32_t index = 0; index < plan->sample_name_count; index++) {
        cJSON* name = cJSON_CreateString(plan->sample_names[index]);
        if (!name || !cJSON_AddItemToArray(samples, name)) {
            cJSON_Delete(name);
            goto fail;
        }
    }
    if (ptq_json_add_owned(metadata, "samples", samples) != 0) {
        samples = NULL; /* ptq_json_add_owned deleted it. */
        goto fail;
    }
    samples = NULL;
    for (int32_t index = 0; index < plan->tensor_count; index++) {
        const VxPTQPlanTensor* tensor = &plan->tensors[index];
        cJSON* entry = cJSON_CreateObject();
        if (!entry ||
            ptq_json_add_owned(entry, "observed_min",
                               cJSON_CreateNumber(tensor->observer.minimum)) != 0 ||
            ptq_json_add_owned(entry, "observed_max",
                               cJSON_CreateNumber(tensor->observer.maximum)) != 0 ||
            ptq_json_add_owned(entry, "sample_count",
                               cJSON_CreateNumber((double)tensor->observer.sample_count)) != 0) {
            cJSON_Delete(entry);
            goto fail;
        }
        if (ptq_json_add_owned(tensors, tensor->name, entry) != 0) goto fail;
    }
    if (ptq_json_add_owned(metadata, "tensors", tensors) != 0) {
        tensors = NULL; /* ptq_json_add_owned deleted it. */
        goto fail_no_tensors;
    }
    tensors = NULL;
    if (ptq_json_add_owned(root, "ptq_authoring", metadata) != 0) return -1;
    return 0;
fail:
    cJSON_Delete(samples);
    cJSON_Delete(tensors);
fail_no_tensors:
    cJSON_Delete(metadata);
    return -1;
}

static int ptq_output_path_available(const char* path) {
    if (!path || !path[0] || strlen(path) >= PATH_MAX - 64) return 0;
    struct stat info;
    if (stat(path, &info) == 0) return 0;
    return errno == ENOENT;
}

static int ptq_publish_no_replace(const char* staged, const char* output) {
    if (!staged || !output) return -1;
#if defined(_WIN32)
    if (!MoveFileExA(staged, output, MOVEFILE_WRITE_THROUGH)) return -1;
    return 0;
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

static int ptq_optional_layout_is(cJSON* params, const char* key,
                                  const char* expected) {
    cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, key) : NULL;
    return !value || (cJSON_IsString(value) && value->valuestring &&
                      !strcmp(value->valuestring, expected));
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

static int ptq_node_ref_is_once(const Ref* refs, int count, const char* key,
                                const char* name) {
    int matches = 0;
    for (int index = 0; index < count; index++) {
        if (!strcmp(refs[index].key, key)) {
            if (strcmp(refs[index].name, name)) return 0;
            matches++;
        }
    }
    return matches == 1;
}

static int ptq_node_activation_ref(const Node* node, int32_t kind,
                                   const char* name) {
    static const char* keys[] = {"input", "x", "a"};
    int key_count = kind == VOLVOXAI_PTQ_LAYER_QLINEAR ? 3 : 2;
    int matches = 0;
    for (int key = 0; key < key_count; key++) {
        for (int index = 0; index < node->nin; index++) {
            if (!strcmp(node->ins[index].key, keys[key])) {
                if (strcmp(node->ins[index].name, name)) return -1;
                matches++;
            }
        }
    }
    return matches == 1 ? 0 : -1;
}

static int ptq_explicit_layout(cJSON* params, const char* key,
                               const char* expected) {
    cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, key) : NULL;
    return cJSON_IsString(value) && value->valuestring &&
           !strcmp(value->valuestring, expected);
}

static int ptq_tensor_prefix_matches(const T* left, const T* right) {
    if (!left || !right || left->ndim <= 0 || left->ndim != right->ndim)
        return 0;
    for (int index = 0; index < left->ndim - 1; index++) {
        if (left->shape[index] != right->shape[index]) return 0;
    }
    return 1;
}

static int ptq_conv_geometry_loaded(const T* input, const T* output,
                                    const T* weight, cJSON* params) {
    int stride[2] = {0};
    int dilation[2] = {0};
    int padding[2] = {0};
    int pads[4] = {0};
    int groups = 1;
    cJSON* groups_json = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "groups") : NULL;
    if (!input || !output || !weight || input->ndim != 4 || output->ndim != 4 ||
        weight->ndim != 4 || !ptq_explicit_layout(params, "data_layout", "NHWC") ||
        !ptq_explicit_layout(params, "weight_layout", "OHWI") ||
        ptq_json_pair(params, "stride", 1, 1, stride) != 0 ||
        ptq_json_pair(params, "dilation", 1, 1, dilation) != 0 ||
        ptq_json_pair(params, "padding", 0, 0, padding) != 0 ||
        ptq_json_pads(params, padding, pads) != 0) return -1;
    if (groups_json && ptq_json_dimension(groups_json, 1, &groups) != 0) return -1;
    if (input->shape[0] != output->shape[0] ||
        weight->shape[3] > INT_MAX / groups ||
        input->shape[3] != weight->shape[3] * groups ||
        output->shape[3] != weight->shape[0] ||
        input->shape[3] % groups || output->shape[3] % groups) return -1;
    uint64_t padded_height = (uint64_t)input->shape[1] +
        (uint64_t)pads[0] + (uint64_t)pads[2];
    uint64_t padded_width = (uint64_t)input->shape[2] +
        (uint64_t)pads[1] + (uint64_t)pads[3];
    uint64_t effective_height =
        (uint64_t)(weight->shape[1] - 1) * (uint64_t)dilation[0] + 1u;
    uint64_t effective_width =
        (uint64_t)(weight->shape[2] - 1) * (uint64_t)dilation[1] + 1u;
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
    if (!ptq_explicit_layout(source, "data_layout", "NHWC") ||
        !ptq_explicit_layout(source, "weight_layout", "OHWI") ||
        !ptq_explicit_layout(target, "data_layout", "NHWC") ||
        !ptq_explicit_layout(target, "weight_layout", "OHWI") ||
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
    const char* expected_op = layer->kind == VOLVOXAI_PTQ_LAYER_QLINEAR
        ? "Linear" : "Conv2D";
    int expected_inputs = layer->has_bias ? 3 : 2;
    T* input = t_find(layer->input_name);
    T* output = t_find(layer->output_name);
    T* weight = t_find(layer->source_weight);
    T* bias = layer->has_bias ? t_find(layer->source_bias) : NULL;
    if (strcmp(node->op, expected_op) || node->disabled ||
        node->nin != expected_inputs || node->nout != 1 ||
        ptq_node_activation_ref(node, layer->kind, layer->input_name) != 0 ||
        !ptq_node_ref_is_once(node->ins, node->nin, "weight",
                              layer->source_weight) ||
        (layer->has_bias &&
         !ptq_node_ref_is_once(node->ins, node->nin, "bias",
                               layer->source_bias)) ||
        (!layer->has_bias &&
         ptq_node_ref_is_once(node->ins, node->nin, "bias", "")) ||
        strcmp(node->outs[0].name, layer->output_name) ||
        strcmp(node->out, layer->output_name) || !input || !output || !weight ||
        input->dtype != T_F32 || output->dtype != T_F32 ||
        weight->dtype != T_F32 || !weight->data ||
        !volvoxai_engine_tensor_is_model_weight_locked(layer->source_weight) ||
        (layer->has_bias &&
         (!bias || bias->dtype != T_F32 || !bias->data ||
          !volvoxai_engine_tensor_is_model_weight_locked(layer->source_bias))))
        return -1;
    if (layer->kind == VOLVOXAI_PTQ_LAYER_QLINEAR) {
        if (!ptq_explicit_layout(node->params, "weight_layout", "OUT_IN") ||
            weight->ndim != 2 || input->ndim <= 0 ||
            !ptq_tensor_prefix_matches(input, output) ||
            input->shape[input->ndim - 1] != weight->shape[1] ||
            output->shape[output->ndim - 1] != weight->shape[0] ||
            !layer->has_bias || bias->ndim != 1 ||
            bias->shape[0] != weight->shape[0]) return -1;
        return 0;
    }
    if (layer->has_bias &&
        (bias->ndim != 1 || bias->shape[0] != weight->shape[0])) return -1;
    return ptq_conv_geometry_loaded(input, output, weight, node->params);
}

static int ptq_layer_basic_shapes(cJSON* root, cJSON* node,
                                  const VxPTQPlanLayer* layer,
                                  const int weight_shape[8], int weight_ndim) {
    cJSON* input_shape = ptq_json_tensor_shape(root, layer->input_name);
    cJSON* output_shape = ptq_json_tensor_shape(root, layer->output_name);
    int input_rank = cJSON_IsArray(input_shape) ? cJSON_GetArraySize(input_shape) : -1;
    int output_rank = cJSON_IsArray(output_shape) ? cJSON_GetArraySize(output_shape) : -1;
    int input_channels = 0;
    int output_channels = 0;
    if (!input_shape || !output_shape ||
        ptq_json_shape_dim(input_shape, -1, &input_channels) != 0 ||
        ptq_json_shape_dim(output_shape, -1, &output_channels) != 0 ||
        output_channels != weight_shape[0]) return -1;
    if (layer->kind == VOLVOXAI_PTQ_LAYER_QLINEAR) {
        if (weight_ndim != 2 || input_rank <= 0 || output_rank != input_rank ||
            input_channels != weight_shape[1]) return -1;
        for (int index = 0; index < input_rank - 1; index++) {
            int input_dim = 0;
            int output_dim = 0;
            if (ptq_json_shape_dim(input_shape, index, &input_dim) != 0 ||
                ptq_json_shape_dim(output_shape, index, &output_dim) != 0 ||
                input_dim != output_dim) return -1;
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
        !ptq_optional_layout_is(params, "data_layout", "NHWC") ||
        !ptq_optional_layout_is(params, "weight_layout", "OHWI") ||
        weight_shape[3] > INT_MAX / groups ||
        input_channels != weight_shape[3] * groups ||
        input_channels % groups || output_channels % groups) return -1;
    int input_dimensions[4] = {0};
    int output_dimensions[4] = {0};
    for (int index = 0; index < 4; index++) {
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
    if (!plan || !options || options->struct_size < sizeof(*options) ||
        !ptq_plan_is_current_locked(plan) ||
        volvoxai_engine_adapter_effect_active_locked() ||
        !plan->calibration_samples || !plan->tensor_count || !plan->layer_count ||
        !options->template_config_path || !options->source_weights_path ||
        !options->output_config_path || !options->output_weights_path ||
        !strcmp(options->template_config_path, options->output_config_path) ||
        !strcmp(options->source_weights_path, options->output_weights_path) ||
        !strcmp(options->output_config_path, options->output_weights_path) ||
        !ptq_output_path_available(options->output_config_path) ||
        !ptq_output_path_available(options->output_weights_path)) return -1;
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
    char* template_text = ptq_read_text_file(options->template_config_path);
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
    if (ptq_template_dependency_preflight(plan, root, &weights) != 0) goto done;

    /* Quantize every graph-input activation once, even when it fans out to
       multiple explicitly selected layers. Internal inputs are metadata owned
       by the preceding planned layer. */
    cJSON* graph_inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    if (!cJSON_IsObject(graph_inputs)) goto done;
    for (int32_t tensor_index = 0; tensor_index < plan->tensor_count; tensor_index++) {
        const VxPTQPlanTensor* tensor = &plan->tensors[tensor_index];
        int used_as_input = 0;
        for (int32_t layer_index = 0; layer_index < plan->layer_count; layer_index++) {
            if (!strcmp(plan->layers[layer_index].input_name, tensor->name)) {
                used_as_input = 1;
                break;
            }
        }
        if (used_as_input &&
            cJSON_GetObjectItemCaseSensitive(graph_inputs, tensor->name) &&
            ptq_template_set_graph_input(root, tensor) != 0) goto done;
    }

    cJSON* weights_quantization = NULL;
    if (ptq_json_object_ensure(root, "weights_quantization",
                               &weights_quantization) != 0) goto done;
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
            (layer->kind == VOLVOXAI_PTQ_LAYER_QCONV2D &&
             !ptq_conv_params_equal(g_n[layer->node_index].params,
                                    cJSON_GetObjectItemCaseSensitive(node,
                                                                     "params"))) ||
            cJSON_GetObjectItemCaseSensitive(weights_quantization,
                                             layer->packed_weight) ||
            (layer->has_bias &&
             cJSON_GetObjectItemCaseSensitive(weights_quantization,
                                              layer->packed_bias)) ||
            safetensors_find_tensor(&weights, layer->packed_weight) ||
            (layer->has_bias &&
             safetensors_find_tensor(&weights, layer->packed_bias))) goto done;

        int expected_ndim = layer->kind == VOLVOXAI_PTQ_LAYER_QLINEAR ? 2 : 4;
        float* source_weight = NULL;
        int weight_shape[8] = {0};
        int weight_ndim = 0;
        int64_t weight_count = 0;
        if (ptq_engine_source_f32(&weights, layer->source_weight, expected_ndim,
                                  &source_weight, weight_shape, &weight_ndim,
                                  &weight_count) != 0 ||
            ptq_layer_basic_shapes(root, node, layer, weight_shape,
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
        if (!packed_weight || !scales ||
            volvoxai_ptq_pack_weight_i8(source_weight, packed_shape, weight_ndim,
                                        0, packed_weight, scales,
                                        channel_count, NULL) != 0 ||
            safetensors_add_tensor(&weights, layer->packed_weight,
                                   SAFETENSORS_DTYPE_I8, weight_shape,
                                   weight_ndim, packed_weight,
                                   (size_t)weight_count) != 0) {
            free(source_weight);
            free(packed_weight);
            free(scales);
            goto done;
        }
        free(source_weight);
        free(packed_weight);

        if (layer->has_bias) {
            float* source_bias = NULL;
            int bias_shape[8] = {0};
            int bias_ndim = 0;
            int64_t bias_count = 0;
            volvoxai_ptq_params_t input_params;
            int32_t* packed_bias = (int32_t*)malloc(
                (size_t)channel_count * sizeof(*packed_bias));
            if (!packed_bias ||
                volvoxai_ptq_calculate_params(&input_tensor->observer,
                                              input_tensor->dtype,
                                              input_tensor->scheme,
                                              &input_params) != 0 ||
                ptq_engine_source_f32(&weights, layer->source_bias, 1,
                                      &source_bias, bias_shape, &bias_ndim,
                                      &bias_count) != 0 ||
                bias_count != channel_count || bias_shape[0] != channel_count ||
                volvoxai_ptq_pack_bias_i32(source_bias, channel_count,
                                           input_params.scale, scales,
                                           channel_count, packed_bias) != 0 ||
                safetensors_add_tensor(&weights, layer->packed_bias,
                                       SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                       packed_bias,
                                       (size_t)channel_count * sizeof(*packed_bias)) != 0) {
                free(source_bias);
                free(packed_bias);
                free(scales);
                goto done;
            }
            free(source_bias);
            free(packed_bias);
        }

        cJSON* weight_descriptor = ptq_weight_descriptor(scales, channel_count);
        free(scales);
        if (!weight_descriptor ||
            ptq_json_add_owned(weights_quantization, layer->packed_weight,
                               weight_descriptor) != 0 ||
            ptq_template_set_output(node, output_key, output_tensor) != 0)
            goto done;
    }
    /* Keep pass-through tensors, but do not make a quantized package carry the
       selected FP32 source payload when the explicit target graph no longer
       references it. Shared sources still used by another target node remain. */
    for (int32_t layer_index = 0; layer_index < plan->layer_count; layer_index++) {
        const VxPTQPlanLayer* layer = &plan->layers[layer_index];
        if (!ptq_template_node_references(root, layer->source_weight) &&
            safetensors_find_tensor(&weights, layer->source_weight) &&
            safetensors_remove_tensor(&weights, layer->source_weight) != 0) goto done;
        if (!safetensors_find_tensor(&weights, layer->source_weight))
            cJSON_DeleteItemFromObjectCaseSensitive(weights_quantization,
                                                    layer->source_weight);
        if (layer->has_bias &&
            !ptq_template_node_references(root, layer->source_bias) &&
            safetensors_find_tensor(&weights, layer->source_bias) &&
            safetensors_remove_tensor(&weights, layer->source_bias) != 0) goto done;
        if (layer->has_bias &&
            !safetensors_find_tensor(&weights, layer->source_bias))
            cJSON_DeleteItemFromObjectCaseSensitive(weights_quantization,
                                                    layer->source_bias);
    }
    if (ptq_add_authoring_metadata(root, plan) != 0) goto done;

    char* config_text = cJSON_PrintUnformatted(root);
    char staged_config[PATH_MAX];
    char staged_weights[PATH_MAX];
    if (!config_text ||
        ptq_staging_path(options->output_config_path, "config",
                         staged_config) != 0 ||
        ptq_staging_path(options->output_weights_path, "weights",
                         staged_weights) != 0) {
        free(config_text);
        goto done;
    }
    remove(staged_config);
    remove(staged_weights);
    if (safetensors_save(staged_weights, &weights) != 0 ||
        ptq_write_text(staged_config, config_text) != 0) {
        free(config_text);
        remove(staged_config);
        remove(staged_weights);
        goto done;
    }
    free(config_text);
    if (ptq_publish_no_replace(staged_weights,
                               options->output_weights_path) != 0) {
        remove(staged_config);
        remove(staged_weights);
        goto done;
    }
    if (ptq_publish_no_replace(staged_config,
                               options->output_config_path) != 0) {
        remove(staged_config);
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
