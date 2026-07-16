/* Public opaque backend ABI and its bridge into the private runtime registry. */

#include "volvoxai_backend.h"

#include "backend_sdk.h"
#include "engine_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(int) == sizeof(int32_t), "backend ABI requires 32-bit int");
_Static_assert(T_F32 == VX_DTYPE_F32 && T_I8 == VX_DTYPE_I8 &&
               T_U8 == VX_DTYPE_U8 && T_I32 == VX_DTYPE_I32 &&
               T_F16 == VX_DTYPE_F16, "public and runtime dtype codes differ");

static const T* sdk_tensor(const VxTensor* tensor) { return (const T*)tensor; }
static T* sdk_mutable_tensor(VxTensor* tensor) { return (T*)tensor; }
static const Node* sdk_node(const VxNode* node) { return (const Node*)node; }

const char* vx_tensor_name(const VxTensor* tensor) {
    return tensor ? sdk_tensor(tensor)->name : NULL;
}

VxDtype vx_tensor_dtype(const VxTensor* tensor) {
    const T* value = sdk_tensor(tensor);
    return value && value->dtype >= T_F32 && value->dtype <= T_F16
        ? (VxDtype)value->dtype : (VxDtype)-1;
}

int32_t vx_tensor_ndim(const VxTensor* tensor) {
    return tensor ? (int32_t)sdk_tensor(tensor)->ndim : 0;
}

const int32_t* vx_tensor_shape(const VxTensor* tensor) {
    return tensor ? (const int32_t*)sdk_tensor(tensor)->shape : NULL;
}

int64_t vx_tensor_numel(const VxTensor* tensor) {
    return tensor ? (int64_t)sdk_tensor(tensor)->numel : 0;
}

size_t vx_tensor_element_size(const VxTensor* tensor) {
    return tensor ? sdk_tensor(tensor)->elem_size : 0;
}

int vx_tensor_is_graph_input(const VxTensor* tensor) {
    return tensor && sdk_tensor(tensor)->is_graph_input ? 1 : 0;
}

int vx_tensor_is_weight(const VxTensor* tensor) {
    const T* value = sdk_tensor(tensor);
    return value && volvoxai_engine_tensor_is_model_weight_locked(value->name) ? 1 : 0;
}

void* vx_tensor_data(VxTensor* tensor) {
    return tensor ? (void*)sdk_mutable_tensor(tensor)->data : NULL;
}

const void* vx_tensor_cdata(const VxTensor* tensor) {
    return tensor ? (const void*)sdk_tensor(tensor)->data : NULL;
}

int vx_tensor_sync_host(const VxTensor* tensor) {
    const T* value = sdk_tensor(tensor);
    size_t bytes;
    if (!value || !value->data || value->numel < 0 ||
        (value->numel > 0 && value->elem_size > SIZE_MAX / (size_t)value->numel)) return -1;
    bytes = (size_t)value->numel * value->elem_size;
    return vx_runtime_backend_sync_host(
        value->data, bytes, volvoxai_engine_tensor_is_model_weight_locked(value->name)) == 0
        ? 0 : -1;
}

static cJSON* tensor_quantization_descriptor(const T* tensor) {
    cJSON* descriptors;
    if (!tensor || !g_cfg_root) return NULL;
    descriptors = cJSON_GetObjectItemCaseSensitive(
        g_cfg_root, "weights_quantization");
    return cJSON_IsObject(descriptors)
        ? cJSON_GetObjectItemCaseSensitive(descriptors, tensor->name) : NULL;
}

static int sdk_json_object_exact_keys(cJSON* object, const char* first,
                                      const char* second) {
    int saw_first = 0;
    int saw_second = second ? 0 : 1;
    if (!cJSON_IsObject(object) || !first) return 0;
    for (cJSON* item = object->child; item; item = item->next) {
        if (!item->string) return 0;
        if (!strcmp(item->string, first) && !saw_first) saw_first = 1;
        else if (second && !strcmp(item->string, second) && !saw_second) saw_second = 1;
        else return 0;
    }
    return saw_first && saw_second;
}

/* Return 0 for legacy inline descriptors, 1 for strict companion tensors,
 * and -1 for a malformed or unsupported storage marker. */
static int tensor_quantization_companion_mode(void) {
    cJSON* storage = g_cfg_root
        ? cJSON_GetObjectItemCaseSensitive(
            g_cfg_root, "weights_quantization_storage") : NULL;
    cJSON* format;
    if (!storage) return 0;
    if (!sdk_json_object_exact_keys(storage, "format", NULL)) return -1;
    format = cJSON_GetObjectItemCaseSensitive(storage, "format");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvoxai-f32-companion-scales-v1")) return -1;
    return 1;
}

static int tensor_quantization_companion(const T* weight, int32_t expected_count,
                                         const T** out) {
    char name[sizeof(((T*)0)->name)];
    int length;
    int occurrences = 0;
    const T* companion;
    if (!weight || !out || expected_count <= 0) return 0;
    length = snprintf(name, sizeof(name), "%s_scale", weight->name);
    if (length <= 0 || (size_t)length >= sizeof(name)) return 0;
    for (int file = 0; file < g_weight_file_count; file++) {
        if (safetensors_find_tensor(&g_weight_files[file], name)) occurrences++;
    }
    if (occurrences != 1) return 0;
    companion = t_find(name);
    if (!companion || !companion->data || companion->dtype != T_F32 ||
        companion->elem_size != sizeof(float) || companion->ndim != 1 ||
        companion->shape[0] != expected_count || companion->numel != expected_count ||
        !volvoxai_engine_tensor_is_model_weight_locked(companion->name)) return 0;
    *out = companion;
    return 1;
}

VxQuantizationKind vx_tensor_quantization_kind(const VxTensor* tensor) {
    const T* value = sdk_tensor(tensor);
    cJSON* descriptor;
    cJSON* scheme;
    if (!value) return VX_QUANT_NONE;
    if (value->quantization.valid) return VX_QUANT_PER_TENSOR;
    descriptor = tensor_quantization_descriptor(value);
    scheme = descriptor ? cJSON_GetObjectItem(descriptor, "scheme") : NULL;
    return cJSON_IsString(scheme) && scheme->valuestring &&
           !strcmp(scheme->valuestring, "per_axis")
        ? VX_QUANT_PER_AXIS : VX_QUANT_NONE;
}

int vx_tensor_quant(const VxTensor* tensor, float* scale, int32_t* zero_point) {
    const T* value = sdk_tensor(tensor);
    if (!value || !value->quantization.valid) return 0;
    if (scale) *scale = value->quantization.scale;
    if (zero_point) *zero_point = (int32_t)value->quantization.zero_point;
    return 1;
}

int vx_tensor_quant_axis(const VxTensor* tensor, int32_t* axis, int32_t* count) {
    const T* value = sdk_tensor(tensor);
    cJSON* descriptor = tensor_quantization_descriptor(value);
    cJSON* scheme = descriptor ? cJSON_GetObjectItem(descriptor, "scheme") : NULL;
    cJSON* axis_json = descriptor ? cJSON_GetObjectItem(descriptor, "axis") : NULL;
    cJSON* scales = descriptor ? cJSON_GetObjectItem(descriptor, "scales") : NULL;
    const T* companion;
    int companion_mode;
    int normalized;
    int length;
    if (!value || !cJSON_IsString(scheme) || !scheme->valuestring ||
        strcmp(scheme->valuestring, "per_axis") || !cJSON_IsNumber(axis_json) ||
        axis_json->valuedouble != trunc(axis_json->valuedouble) ||
        axis_json->valuedouble < INT32_MIN || axis_json->valuedouble > INT32_MAX)
        return 0;
    normalized = axis_json->valueint;
    if (normalized < 0) normalized += value->ndim;
    if (normalized < 0 || normalized >= value->ndim || value->shape[normalized] <= 0)
        return 0;
    companion_mode = tensor_quantization_companion_mode();
    if (companion_mode < 0) return 0;
    if (companion_mode) {
        length = value->shape[normalized];
        if (!sdk_json_object_exact_keys(descriptor, "scheme", "axis") ||
            !tensor_quantization_companion(value, (int32_t)length, &companion))
            return 0;
    } else {
        if (!cJSON_IsArray(scales)) return 0;
        length = cJSON_GetArraySize(scales);
        if (length <= 0 || length > INT32_MAX || length != value->shape[normalized])
            return 0;
    }
    if (axis) *axis = (int32_t)normalized;
    if (count) *count = (int32_t)length;
    return 1;
}

int vx_tensor_quant_axis_value(const VxTensor* tensor, int32_t index,
                               float* scale, int32_t* zero_point) {
    const T* value = sdk_tensor(tensor);
    cJSON* descriptor = tensor_quantization_descriptor(value);
    cJSON* scales = descriptor ? cJSON_GetObjectItem(descriptor, "scales") : NULL;
    cJSON* zero_points = descriptor ? cJSON_GetObjectItem(descriptor, "zero_points") : NULL;
    cJSON* scale_json;
    cJSON* zero_json;
    const T* companion;
    int companion_mode;
    float scale_value;
    double zero_value = 0.0;
    int32_t axis;
    int32_t count;
    if (!vx_tensor_quant_axis(tensor, &axis, &count) || index < 0 || index >= count)
        return 0;
    (void)axis;
    companion_mode = tensor_quantization_companion_mode();
    if (companion_mode < 0) return 0;
    if (companion_mode) {
        if (!sdk_json_object_exact_keys(descriptor, "scheme", "axis") ||
            !tensor_quantization_companion(value, count, &companion)) return 0;
        memcpy(&scale_value,
               (const unsigned char*)companion->data +
                   (size_t)index * sizeof(scale_value),
               sizeof(scale_value));
        zero_json = NULL;
    } else {
        scale_json = cJSON_GetArrayItem(scales, index);
        zero_json = cJSON_IsArray(zero_points) ? cJSON_GetArrayItem(zero_points, index) : NULL;
        if (!cJSON_IsNumber(scale_json) || !isfinite(scale_json->valuedouble) ||
            scale_json->valuedouble <= 0.0 ||
            (zero_json && (!cJSON_IsNumber(zero_json) ||
                           zero_json->valuedouble != trunc(zero_json->valuedouble) ||
                           zero_json->valuedouble < INT32_MIN ||
                           zero_json->valuedouble > INT32_MAX))) return 0;
        scale_value = (float)scale_json->valuedouble;
    }
    if (!isfinite(scale_value) || scale_value <= 0.0f) return 0;
    if (zero_json) zero_value = zero_json->valuedouble;
    if (scale) *scale = scale_value;
    if (zero_point) *zero_point = (int32_t)zero_value;
    return 1;
}

const char* vx_node_op(const VxNode* node) {
    return node ? sdk_node(node)->op : NULL;
}

int32_t vx_node_input_count(const VxNode* node) {
    return node ? (int32_t)sdk_node(node)->nin : 0;
}

const char* vx_node_input_key(const VxNode* node, int32_t index) {
    const Node* value = sdk_node(node);
    return value && index >= 0 && index < value->nin ? value->ins[index].key : NULL;
}

const VxTensor* vx_node_input(const VxNode* node, int32_t index) {
    const Node* value = sdk_node(node);
    return value && index >= 0 && index < value->nin
        ? (const VxTensor*)t_find(value->ins[index].name) : NULL;
}

const VxTensor* vx_node_input_by_key(const VxNode* node, const char* key) {
    const Node* value = sdk_node(node);
    if (!value || !key) return NULL;
    for (int index = 0; index < value->nin; index++) {
        if (!strcmp(value->ins[index].key, key))
            return (const VxTensor*)t_find(value->ins[index].name);
    }
    return NULL;
}

int32_t vx_node_output_count(const VxNode* node) {
    return node ? (int32_t)sdk_node(node)->nout : 0;
}

const char* vx_node_output_key(const VxNode* node, int32_t index) {
    const Node* value = sdk_node(node);
    return value && index >= 0 && index < value->nout ? value->outs[index].key : NULL;
}

VxTensor* vx_node_output(const VxNode* node, int32_t index) {
    const Node* value = sdk_node(node);
    return value && index >= 0 && index < value->nout
        ? (VxTensor*)t_find(value->outs[index].name) : NULL;
}

VxTensor* vx_node_output_by_key(const VxNode* node, const char* key) {
    const Node* value = sdk_node(node);
    if (!value || !key) return NULL;
    for (int index = 0; index < value->nout; index++) {
        if (!strcmp(value->outs[index].key, key))
            return (VxTensor*)t_find(value->outs[index].name);
    }
    return NULL;
}

static cJSON* node_attr(const VxNode* node, const char* key) {
    const Node* value = sdk_node(node);
    return value && value->params && key
        ? cJSON_GetObjectItem(value->params, key) : NULL;
}

int32_t vx_node_attr_int(const VxNode* node, const char* key, int32_t fallback) {
    cJSON* value = node_attr(node, key);
    return cJSON_IsNumber(value) && isfinite(value->valuedouble) &&
           value->valuedouble == trunc(value->valuedouble) &&
           value->valuedouble >= INT32_MIN && value->valuedouble <= INT32_MAX
        ? (int32_t)value->valuedouble : fallback;
}

float vx_node_attr_float(const VxNode* node, const char* key, float fallback) {
    cJSON* value = node_attr(node, key);
    float result;
    if (!cJSON_IsNumber(value) || !isfinite(value->valuedouble)) return fallback;
    result = (float)value->valuedouble;
    return isfinite(result) ? result : fallback;
}

int vx_node_attr_bool(const VxNode* node, const char* key, int fallback) {
    cJSON* value = node_attr(node, key);
    if (cJSON_IsBool(value)) return cJSON_IsTrue(value) ? 1 : 0;
    if (cJSON_IsNumber(value) && isfinite(value->valuedouble))
        return value->valuedouble != 0.0;
    return fallback;
}

const char* vx_node_attr_string(const VxNode* node, const char* key,
                                const char* fallback) {
    cJSON* value = node_attr(node, key);
    return cJSON_IsString(value) && value->valuestring ? value->valuestring : fallback;
}

int32_t vx_node_attr_ints(const VxNode* node, const char* key,
                          int32_t* out, int32_t capacity) {
    cJSON* value = node_attr(node, key);
    int count;
    int index = 0;
    if (!cJSON_IsArray(value)) return 0;
    count = cJSON_GetArraySize(value);
    if (count < 0 || (out == NULL && capacity != 0)) return -1;
    for (cJSON* item = value->child; item; item = item->next) {
        if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
            item->valuedouble != trunc(item->valuedouble) ||
            item->valuedouble < INT32_MIN || item->valuedouble > INT32_MAX)
            return -1;
    }
    if (!out) return (int32_t)count;
    if (capacity < count) return -1;
    for (cJSON* item = value->child; item; item = item->next)
        out[index++] = (int32_t)item->valuedouble;
    return (int32_t)count;
}

int32_t vx_node_attr_floats(const VxNode* node, const char* key,
                            float* out, int32_t capacity) {
    cJSON* value = node_attr(node, key);
    int count;
    int index = 0;
    if (!cJSON_IsArray(value)) return 0;
    count = cJSON_GetArraySize(value);
    if (count < 0 || (out == NULL && capacity != 0)) return -1;
    for (cJSON* item = value->child; item; item = item->next) {
        float result;
        if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble)) return -1;
        result = (float)item->valuedouble;
        if (!isfinite(result)) return -1;
    }
    if (!out) return (int32_t)count;
    if (capacity < count) return -1;
    for (cJSON* item = value->child; item; item = item->next)
        out[index++] = (float)item->valuedouble;
    return (int32_t)count;
}

enum { VX_SDK_SLOTS = 8, VX_SDK_NAME_CAPACITY = 64 };

typedef struct {
    VxBackendV1 public_backend;
    VxBackend bridge;
    char name[VX_SDK_NAME_CAPACITY];
} VxSdkSlot;

static VxSdkSlot g_slots[VX_SDK_SLOTS];
static size_t g_slot_count;
static VxSdkSlot* g_selected;

static int sdk_init(void* user_data) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    int status = slot->public_backend.init(slot->public_backend.user_data);
    return status == VX_INIT_READY ? VX_BACKEND_INIT_READY :
           status == VX_INIT_UNAVAILABLE ? VX_BACKEND_INIT_UNAVAILABLE :
           VX_BACKEND_INIT_ERROR;
}

static int sdk_supports(void* user_data, const Node* node) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    if (!vx_runtime_backend_sdk_node_eligible(node)) return VX_BACKEND_DECLINED;
    int status = slot->public_backend.supports(slot->public_backend.user_data,
                                               (const VxNode*)node);
    return status == VX_HANDLED ? VX_BACKEND_HANDLED :
           status == VX_DECLINED ? VX_BACKEND_DECLINED : VX_BACKEND_ERROR;
}

static int sdk_run(void* user_data, const Node* node, T* input, T* output) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    int status;
    (void)input;
    (void)output;
    status = slot->public_backend.run(slot->public_backend.user_data,
                                      (const VxNode*)node);
    return status == VX_HANDLED ? VX_BACKEND_HANDLED :
           status == VX_DECLINED ? VX_BACKEND_DECLINED : VX_BACKEND_ERROR;
}

static void sdk_teardown(void* user_data) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    slot->public_backend.teardown(slot->public_backend.user_data);
}

static void sdk_reset(void* user_data) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    slot->public_backend.reset(slot->public_backend.user_data);
}

static void sdk_begin(void* user_data) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    if (vx_runtime_backend_sdk_execution_eligible())
        slot->public_backend.begin_forward(slot->public_backend.user_data);
}

static int sdk_end(void* user_data) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    return vx_runtime_backend_sdk_execution_eligible()
        ? slot->public_backend.end_forward(slot->public_backend.user_data) : 0;
}

static void sdk_mark(void* user_data, const void* host, size_t bytes, int is_weight) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    slot->public_backend.mark_host(slot->public_backend.user_data, host, bytes, is_weight);
}

static int sdk_sync(void* user_data, const void* host, size_t bytes, int is_weight) {
    VxSdkSlot* slot = (VxSdkSlot*)user_data;
    /* Public callbacks use conventional zero-success; the private graph
     * registry retains the existing truthy-success device-hook convention. */
    return slot->public_backend.sync_host(slot->public_backend.user_data,
                                           (void*)host, bytes, is_weight) == 0 ? 1 : 0;
}

static int reserved_backend_name(const char* name) {
    static const char* const names[] = { "cpu", "vulkan", "opengl", "metal", "nnapi" };
    for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); index++) {
        if (!strcmp(name, names[index])) return 1;
    }
    return 0;
}

static int backend_descriptor_valid(const VxBackendV1* backend) {
    const uint32_t known_flags = VX_BACKEND_FLAG_DEVICE_RESIDENT_OUTPUTS;
    if (!backend || backend->struct_size < sizeof(VxBackendV1) ||
        backend->abi_version != VX_BACKEND_ABI_V1 || !backend->name ||
        !backend->name[0] || strlen(backend->name) >= VX_SDK_NAME_CAPACITY ||
        reserved_backend_name(backend->name) || (backend->flags & ~known_flags) ||
        !backend->init || !backend->supports || !backend->run || !backend->teardown)
        return 0;
    if ((backend->begin_forward == NULL) != (backend->end_forward == NULL) ||
        (backend->mark_host == NULL) != (backend->sync_host == NULL)) return 0;
    if ((backend->flags & VX_BACKEND_FLAG_DEVICE_RESIDENT_OUTPUTS) &&
        (!backend->mark_host || !backend->sync_host)) return 0;
    return 1;
}

int volvoxai_register_backend(const VxBackendV1* backend) {
    int result = -1;
    volvoxai_engine_model_lock();
    if (g_loaded || !backend_descriptor_valid(backend) || g_slot_count >= VX_SDK_SLOTS)
        goto done;
    for (size_t index = 0; index < g_slot_count; index++) {
        if (!strcmp(g_slots[index].name, backend->name)) goto done;
    }
    VxSdkSlot* slot = &g_slots[g_slot_count];
    memset(slot, 0, sizeof(*slot));
    memcpy(&slot->public_backend, backend, sizeof(slot->public_backend));
    memcpy(slot->name, backend->name, strlen(backend->name) + 1u);
    slot->public_backend.name = slot->name;
    slot->bridge = (VxBackend){
        .name = slot->name,
        .user_data = slot,
        .required = 1,
        .init = sdk_init,
        .supports = sdk_supports,
        .run = sdk_run,
        .outputs_device_resident =
            !!(backend->flags & VX_BACKEND_FLAG_DEVICE_RESIDENT_OUTPUTS),
        .reset = backend->reset ? sdk_reset : NULL,
        .begin_forward = backend->begin_forward ? sdk_begin : NULL,
        .end_forward = backend->end_forward ? sdk_end : NULL,
        .mark_host = backend->mark_host ? sdk_mark : NULL,
        .sync_host = backend->sync_host ? sdk_sync : NULL,
        .teardown = sdk_teardown,
    };
    g_slot_count++;
    result = 0;
done:
    volvoxai_engine_model_unlock();
    return result;
}

int vx_sdk_select_backend(const char* name) {
    const VxBackend* backend = vx_sdk_find_backend(name);
    if (!backend) return -1;
    for (size_t index = 0; index < g_slot_count; index++) {
        if (&g_slots[index].bridge == backend) {
            g_selected = &g_slots[index];
            return 0;
        }
    }
    return -1;
}

void vx_sdk_clear_backend(void) { g_selected = NULL; }

const char* vx_sdk_selected_name(void) {
    return g_selected ? g_selected->name : NULL;
}

const VxBackend* vx_sdk_selected_backend(void) {
    return g_selected ? &g_selected->bridge : NULL;
}

const VxBackend* vx_sdk_find_backend(const char* name) {
    if (!name || !name[0]) return NULL;
    for (size_t index = 0; index < g_slot_count; index++) {
        if (!strcmp(g_slots[index].name, name)) return &g_slots[index].bridge;
    }
    return NULL;
}
