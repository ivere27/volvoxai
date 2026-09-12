#include "adapter_runtime_internal.h"
#include "cJSON.h"
#include "lora_linear.h"
#include "runtime_state.h"
#include "safetensors.h"

#include <math.h>
#include <limits.h>
#include "vx_thread.h"
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[128];
    float* data;
    size_t count;
    int rows;
    int cols;
} AdapterTensor;

typedef struct {
    char weight_name[128];
    VxAdapterKind kind;
    int d_in;
    int d_out;
    int rank;
    float alpha;
    float scale;
    AdapterTensor a;
    AdapterTensor b;
} AdapterTarget;

typedef struct AdapterVersion {
    char adapter_id[128];
    char version_id[128];
    AdapterTarget* targets;
    int target_count;
    unsigned refs;
    int removed;
    char* manifest_json;
    struct AdapterVersion* next;
} AdapterVersion;

typedef struct {
    AdapterVersion** request_versions;
    float* request_scales;
    AdapterVersion* request_inline_version;
    float request_inline_scale;
    int request_count;
    int request_set;
    AdapterVersion** run_versions;
    float* run_scales;
    AdapterVersion* run_inline_version;
    float run_inline_scale;
    int run_count;
    int run_refs_owned;
    int run_depth;
    float* rank_workspace;
    int rank_workspace_count;
} AdapterTls;

typedef struct AdapterTombstone {
    char version_id[128];
    struct AdapterTombstone* next;
} AdapterTombstone;

typedef struct AdapterRegistryState {
    VxMutex mutex;
    AdapterVersion* versions;
    AdapterVersion* active;
    _Atomic int active_present;
    _Atomic unsigned long run_registry_lock_count;
    AdapterTombstone* tombstones;
    AdapterTls tls;
} AdapterRegistryState;

static VxMutex g_adapter_state_init_mutex = VX_MUTEX_INITIALIZER;
static void adapter_registry_state_destroy(void* opaque);

static AdapterRegistryState* adapter_registry_state(void) {
    VxEngineState* owner = vx_engine_state_current();
    AdapterRegistryState* state;
    if (!owner) return NULL;
    state = (AdapterRegistryState*)owner->adapter_registry_state;
    if (state) return state;
    vx_mutex_lock(&g_adapter_state_init_mutex);
    state = (AdapterRegistryState*)owner->adapter_registry_state;
    if (!state) {
        state = (AdapterRegistryState*)calloc(1, sizeof(*state));
        if (state && vx_mutex_init(&state->mutex) != 0) {
            free(state);
            state = NULL;
        }
        if (state) {
            atomic_init(&state->active_present, 0);
            atomic_init(&state->run_registry_lock_count, 0);
            owner->adapter_registry_state = state;
            owner->adapter_registry_state_destroy = adapter_registry_state_destroy;
        }
    }
    vx_mutex_unlock(&g_adapter_state_init_mutex);
    return state;
}

#define g_registry_mutex (adapter_registry_state()->mutex)
#define g_versions (adapter_registry_state()->versions)
#define g_active (adapter_registry_state()->active)
#define g_active_present (adapter_registry_state()->active_present)
#define g_run_registry_lock_count \
    (adapter_registry_state()->run_registry_lock_count)
#define g_tombstones (adapter_registry_state()->tombstones)
#define g_tls (adapter_registry_state()->tls)

static int copy_name(char out[128], const char* src) {
    if (!src || !src[0] || strlen(src) >= 128) return -1;
    memcpy(out, src, strlen(src) + 1);
    return 0;
}

static const char* kind_name(VxAdapterKind kind) {
    return kind == VX_ADAPTER_LORA ? "lora" : "unknown";
}

static int kind_from_name(const char* name, VxAdapterKind* out) {
    if (!name || !out) return -1;
    if (strcmp(name, "lora")) return -1;
    *out = VX_ADAPTER_LORA;
    return 0;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -14;
            while (!(mant & 0x0400u)) { mant <<= 1; e--; }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t load_u16_le(const void* data, size_t index) {
    const unsigned char* bytes = (const unsigned char*)data + index * 2u;
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static float load_f32_unaligned(const void* data, size_t index) {
    float value;
    memcpy(&value, (const unsigned char*)data + index * sizeof(float), sizeof(value));
    return value;
}

static float base_value(const void* data, int dtype, size_t index) {
    if (dtype == VX_ADAPTER_DTYPE_F32) return load_f32_unaligned(data, index);
    if (dtype == VX_ADAPTER_DTYPE_F16) return f16_to_f32(load_u16_le(data, index));
    return 0.0f;
}

static int tensor_from_spec(AdapterTensor* out, const char* default_name,
                            const VxAdapterTensorSpec* spec, size_t expected) {
    memset(out, 0, sizeof(*out));
    if (!spec || !spec->data || expected == 0 || spec->rows <= 0 || spec->cols <= 0) return -1;
    size_t elem_size = spec->dtype == VX_ADAPTER_DTYPE_F32 ? 4u :
                       spec->dtype == VX_ADAPTER_DTYPE_F16 ? 2u : 0u;
    if (!elem_size || expected > SIZE_MAX / elem_size || spec->nbytes != expected * elem_size) return -1;
    if (copy_name(out->name, default_name) != 0) return -1;
    out->data = (float*)malloc(expected * sizeof(float));
    if (!out->data) return -1;
    if (spec->dtype == VX_ADAPTER_DTYPE_F32) {
        memcpy(out->data, spec->data, expected * sizeof(float));
    } else {
        for (size_t i = 0; i < expected; i++) out->data[i] = f16_to_f32(load_u16_le(spec->data, i));
    }
    for (size_t i = 0; i < expected; i++) {
        if (!isfinite(out->data[i])) { free(out->data); memset(out, 0, sizeof(*out)); return -1; }
    }
    out->count = expected;
    out->rows = spec->rows;
    out->cols = spec->cols;
    return 0;
}

static void tensor_free(AdapterTensor* tensor) {
    if (!tensor) return;
    free(tensor->data);
    memset(tensor, 0, sizeof(*tensor));
}

static void target_free(AdapterTarget* target) {
    if (!target) return;
    tensor_free(&target->a);
    tensor_free(&target->b);
    memset(target, 0, sizeof(*target));
}

static void version_free(AdapterVersion* version) {
    if (!version) return;
    for (int i = 0; i < version->target_count; i++) target_free(&version->targets[i]);
    free(version->targets);
    free(version->manifest_json);
    free(version);
}

static void adapter_registry_state_destroy(void* opaque) {
    AdapterRegistryState* state = (AdapterRegistryState*)opaque;
    AdapterVersion* version;
    AdapterTombstone* tombstone;
    if (!state) return;
    version = state->versions;
    while (version) {
        AdapterVersion* next = version->next;
        version_free(version);
        version = next;
    }
    tombstone = state->tombstones;
    while (tombstone) {
        AdapterTombstone* next = tombstone->next;
        free(tombstone);
        tombstone = next;
    }
    if (state->tls.request_count > 1) {
        free(state->tls.request_versions);
        free(state->tls.request_scales);
    }
    free(state->tls.rank_workspace);
    vx_mutex_destroy(&state->mutex);
    free(state);
}

static AdapterVersion* find_version_locked(const char* version_id) {
    if (!version_id || !version_id[0]) return NULL;
    for (AdapterVersion* v = g_versions; v; v = v->next) {
        if (!v->removed && !strcmp(v->version_id, version_id)) return v;
    }
    return NULL;
}

static int version_id_seen_locked(const char* version_id) {
    for (AdapterTombstone* item = g_tombstones; item; item = item->next) {
        if (!strcmp(item->version_id, version_id)) return 1;
    }
    return 0;
}

static void release_version_locked(AdapterVersion* version) {
    if (!version) return;
    if (version->refs > 0) version->refs--;
    if (version->removed && version->refs == 0) version_free(version);
}

static AdapterVersion* acquire_version(const char* version_id) {
    vx_mutex_lock(&g_registry_mutex);
    AdapterVersion* version = find_version_locked(version_id);
    if (version) version->refs++;
    vx_mutex_unlock(&g_registry_mutex);
    return version;
}

static void release_version(AdapterVersion* version) {
    vx_mutex_lock(&g_registry_mutex);
    release_version_locked(version);
    vx_mutex_unlock(&g_registry_mutex);
}

static AdapterTarget* target_find(AdapterVersion* version, const char* weight_name) {
    if (!version || !weight_name) return NULL;
    for (int i = 0; i < version->target_count; i++) {
        if (!strcmp(version->targets[i].weight_name, weight_name)) return &version->targets[i];
    }
    return NULL;
}

static int generated_tensor_name(char out[128], const char* weight, const char* role) {
    int n = snprintf(out, 128, "%s.%s", weight, role);
    return n > 0 && n < 128 ? 0 : -1;
}

static int json_object_has_only_keys(const cJSON* object, const char* const* allowed,
                                     size_t allowed_count) {
    if (!cJSON_IsObject(object)) return 0;
    for (const cJSON* item = object->child; item; item = item->next) {
        if (!item->string) return 0;
        int found = 0;
        for (size_t i = 0; i < allowed_count; i++) {
            if (!strcmp(item->string, allowed[i])) { found = 1; break; }
        }
        if (!found) return 0;
        for (const cJSON* previous = object->child; previous != item; previous = previous->next) {
            if (previous->string && !strcmp(previous->string, item->string)) return 0;
        }
    }
    return 1;
}

static int json_string_map_valid(const cJSON* object) {
    if (!cJSON_IsObject(object)) return 0;
    for (const cJSON* item = object->child; item; item = item->next) {
        if (!item->string || !item->string[0] || !cJSON_IsString(item) || !item->valuestring) return 0;
    }
    return 1;
}

static int adapter_manifest_schema_valid(const cJSON* manifest, int target_count) {
    static const char* const root_keys[] = {
        "format", "name", "adapter_id", "version_id", "kind", "targets", "metadata"
    };
    static const char* const target_keys[] = {
        "id", "op", "weight", "kind", "a", "b", "layout",
        "rank", "alpha", "scale", "tensors"
    };
    static const char* const tensor_keys[] = { "role", "name", "shape", "dtype" };
    if (!json_object_has_only_keys(manifest, root_keys, sizeof(root_keys) / sizeof(root_keys[0]))) return 0;
    cJSON* format = cJSON_GetObjectItemCaseSensitive(manifest, "format");
    cJSON* adapter_id = cJSON_GetObjectItemCaseSensitive(manifest, "adapter_id");
    cJSON* version_id = cJSON_GetObjectItemCaseSensitive(manifest, "version_id");
    cJSON* kind = cJSON_GetObjectItemCaseSensitive(manifest, "kind");
    cJSON* targets = cJSON_GetObjectItemCaseSensitive(manifest, "targets");
    cJSON* metadata = cJSON_GetObjectItemCaseSensitive(manifest, "metadata");
    if (!cJSON_IsString(format) || strcmp(format->valuestring, "volvox.adapter.v1") ||
        !cJSON_IsString(adapter_id) || !adapter_id->valuestring[0] ||
        !cJSON_IsString(version_id) || !version_id->valuestring[0] ||
        !cJSON_IsString(kind) || strcmp(kind->valuestring, "lora") ||
        !cJSON_IsArray(targets) || cJSON_GetArraySize(targets) != target_count ||
        (metadata && !json_string_map_valid(metadata))) return 0;
    for (int i = 0; i < target_count; i++) {
        cJSON* target = cJSON_GetArrayItem(targets, i);
        if (!json_object_has_only_keys(target, target_keys, sizeof(target_keys) / sizeof(target_keys[0]))) return 0;
        cJSON* target_kind = cJSON_GetObjectItemCaseSensitive(target, "kind");
        cJSON* alpha = cJSON_GetObjectItemCaseSensitive(target, "alpha");
        cJSON* tensors = cJSON_GetObjectItemCaseSensitive(target, "tensors");
        if ((target_kind && (!cJSON_IsString(target_kind) || strcmp(target_kind->valuestring, "lora"))) ||
            !cJSON_IsNumber(alpha) || !isfinite(alpha->valuedouble) || alpha->valuedouble <= 0.0 ||
            (tensors && !cJSON_IsArray(tensors))) return 0;
        for (cJSON* tensor = tensors ? tensors->child : NULL; tensor; tensor = tensor->next) {
            if (!json_object_has_only_keys(tensor, tensor_keys, sizeof(tensor_keys) / sizeof(tensor_keys[0]))) return 0;
            cJSON* role = cJSON_GetObjectItemCaseSensitive(tensor, "role");
            cJSON* name = cJSON_GetObjectItemCaseSensitive(tensor, "name");
            cJSON* shape = cJSON_GetObjectItemCaseSensitive(tensor, "shape");
            cJSON* dtype = cJSON_GetObjectItemCaseSensitive(tensor, "dtype");
            if (!cJSON_IsString(role) || (strcmp(role->valuestring, "a") && strcmp(role->valuestring, "b")) ||
                !cJSON_IsString(name) || !name->valuestring[0] || !cJSON_IsArray(shape) ||
                cJSON_GetArraySize(shape) != 2 || !cJSON_IsString(dtype) ||
                (strcmp(dtype->valuestring, "F32") && strcmp(dtype->valuestring, "F16"))) return 0;
            for (int d = 0; d < 2; d++) {
                cJSON* dim = cJSON_GetArrayItem(shape, d);
                if (!cJSON_IsNumber(dim) || !isfinite(dim->valuedouble) || dim->valuedouble < 1.0 ||
                    dim->valuedouble > INT_MAX || dim->valuedouble != floor(dim->valuedouble)) return 0;
            }
        }
    }
    return 1;
}

static int target_from_spec(AdapterTarget* out, const VxAdapterTargetSpec* spec) {
    memset(out, 0, sizeof(*out));
    if (!spec || copy_name(out->weight_name, spec->weight_name) != 0 ||
        spec->d_in <= 0 || spec->d_out <= 0 || spec->rank <= 0 ||
        spec->kind != VX_ADAPTER_LORA || !isfinite(spec->alpha) || spec->alpha <= 0.0f) return -1;
    if ((size_t)spec->d_in > SIZE_MAX / (size_t)spec->rank ||
        (size_t)spec->rank > SIZE_MAX / (size_t)spec->d_out) return -1;
    out->kind = spec->kind;
    out->d_in = spec->d_in;
    out->d_out = spec->d_out;
    out->rank = spec->rank;
    out->alpha = spec->alpha;
    if (isfinite(spec->scale)) out->scale = spec->scale;
    else out->scale = out->alpha / (float)out->rank;

    char a_name[128], b_name[128];
    if (spec->a_name) { if (copy_name(a_name, spec->a_name) != 0) goto fail; }
    else if (generated_tensor_name(a_name, out->weight_name, "lora_A") != 0) goto fail;
    if (spec->b_name) { if (copy_name(b_name, spec->b_name) != 0) goto fail; }
    else if (generated_tensor_name(b_name, out->weight_name, "lora_B") != 0) goto fail;
    if (spec->a.rows != out->d_in || spec->a.cols != out->rank ||
        spec->b.rows != out->rank || spec->b.cols != out->d_out) goto fail;
    if (tensor_from_spec(&out->a, a_name, &spec->a, (size_t)out->d_in * out->rank) != 0 ||
        tensor_from_spec(&out->b, b_name, &spec->b, (size_t)out->rank * out->d_out) != 0) goto fail;

    return 0;

fail:
    target_free(out);
    return -1;
}

static AdapterVersion* version_from_spec(const VxAdapterVersionSpec* spec) {
    if (!spec || !spec->version_id || !spec->version_id[0] || spec->target_count <= 0 || !spec->targets) return NULL;
    AdapterVersion* version = (AdapterVersion*)calloc(1, sizeof(*version));
    if (!version || copy_name(version->version_id, spec->version_id) != 0 ||
        copy_name(version->adapter_id, spec->adapter_id && spec->adapter_id[0] ? spec->adapter_id : spec->version_id) != 0) {
        version_free(version);
        return NULL;
    }
    version->targets = (AdapterTarget*)calloc((size_t)spec->target_count, sizeof(*version->targets));
    if (!version->targets) { version_free(version); return NULL; }
    version->target_count = spec->target_count;
    for (int i = 0; i < spec->target_count; i++) {
        if (target_from_spec(&version->targets[i], &spec->targets[i]) != 0) { version_free(version); return NULL; }
        for (int j = 0; j < i; j++) {
            if (!strcmp(version->targets[i].weight_name, version->targets[j].weight_name)) { version_free(version); return NULL; }
        }
    }
    for (int i = 0; i < version->target_count; i++) {
        AdapterTensor* left[] = { &version->targets[i].a, &version->targets[i].b };
        for (size_t li = 0; li < sizeof(left) / sizeof(left[0]); li++) {
            if (!left[li]->data) continue;
            for (int j = 0; j <= i; j++) {
                AdapterTensor* right[] = { &version->targets[j].a, &version->targets[j].b };
                size_t rlimit = j == i ? li : sizeof(right) / sizeof(right[0]);
                for (size_t ri = 0; ri < rlimit; ri++) {
                    if (!right[ri]->data || strcmp(left[li]->name, right[ri]->name)) continue;
                    version_free(version);
                    return NULL;
                }
            }
        }
    }
    if (spec->manifest_json && spec->manifest_json[0]) {
        cJSON* parsed = cJSON_Parse(spec->manifest_json);
        if (!parsed || !adapter_manifest_schema_valid(parsed, spec->target_count)) {
            cJSON_Delete(parsed); version_free(version); return NULL;
        }
        cJSON* parsed_adapter = cJSON_GetObjectItemCaseSensitive(parsed, "adapter_id");
        cJSON* parsed_version = cJSON_GetObjectItemCaseSensitive(parsed, "version_id");
        if (strcmp(parsed_adapter->valuestring, version->adapter_id) ||
            strcmp(parsed_version->valuestring, version->version_id)) {
            cJSON_Delete(parsed); version_free(version); return NULL;
        }
        version->manifest_json = cJSON_PrintUnformatted(parsed);
        cJSON_Delete(parsed);
        if (!version->manifest_json) { version_free(version); return NULL; }
    }
    return version;
}

static int publish_version(AdapterVersion* version) {
    if (!version) return -1;
    vx_mutex_lock(&g_registry_mutex);
    if (find_version_locked(version->version_id) || version_id_seen_locked(version->version_id)) {
        vx_mutex_unlock(&g_registry_mutex);
        return -1;
    }
    AdapterTombstone* tombstone = (AdapterTombstone*)calloc(1, sizeof(*tombstone));
    if (!tombstone || copy_name(tombstone->version_id, version->version_id) != 0) {
        free(tombstone);
        vx_mutex_unlock(&g_registry_mutex);
        return -1;
    }
    tombstone->next = g_tombstones;
    g_tombstones = tombstone;
    version->next = g_versions;
    g_versions = version;
    vx_mutex_unlock(&g_registry_mutex);
    return 0;
}

int vx_adapter_stage(const VxAdapterVersionSpec* spec) {
    AdapterVersion* version = version_from_spec(spec);
    if (!version) return -1;
    if (publish_version(version) != 0) { version_free(version); return -1; }
    return 0;
}

static int tensor_clone(AdapterTensor* out, const AdapterTensor* src) {
    *out = *src;
    out->data = (float*)malloc(src->count * sizeof(float));
    if (!out->data) { memset(out, 0, sizeof(*out)); return -1; }
    memcpy(out->data, src->data, src->count * sizeof(float));
    return 0;
}

static AdapterVersion* version_clone(const AdapterVersion* src, const char* adapter_id, const char* version_id) {
    AdapterVersion* out = (AdapterVersion*)calloc(1, sizeof(*out));
    if (!out || copy_name(out->adapter_id, adapter_id && adapter_id[0] ? adapter_id : src->adapter_id) != 0 ||
        copy_name(out->version_id, version_id) != 0) { version_free(out); return NULL; }
    out->target_count = src->target_count;
    out->targets = (AdapterTarget*)calloc((size_t)out->target_count, sizeof(*out->targets));
    if (!out->targets) { version_free(out); return NULL; }
    for (int i = 0; i < out->target_count; i++) {
        AdapterTarget* d = &out->targets[i];
        const AdapterTarget* s = &src->targets[i];
        memcpy(d->weight_name, s->weight_name, sizeof(d->weight_name));
        d->kind = s->kind; d->d_in = s->d_in; d->d_out = s->d_out; d->rank = s->rank;
        d->alpha = s->alpha; d->scale = s->scale;
        if (tensor_clone(&d->a, &s->a) != 0 || tensor_clone(&d->b, &s->b) != 0) {
            version_free(out); return NULL;
        }
    }
    if (src->manifest_json) {
        cJSON* manifest = cJSON_Parse(src->manifest_json);
        if (!manifest ||
            !cJSON_ReplaceItemInObjectCaseSensitive(manifest, "adapter_id", cJSON_CreateString(out->adapter_id)) ||
            !cJSON_ReplaceItemInObjectCaseSensitive(manifest, "version_id", cJSON_CreateString(out->version_id))) {
            cJSON_Delete(manifest);
            version_free(out);
            return NULL;
        }
        cJSON* metadata = cJSON_GetObjectItem(manifest, "metadata");
        if (!metadata) {
            metadata = cJSON_CreateObject();
            if (!metadata) { cJSON_Delete(manifest); version_free(out); return NULL; }
            cJSON_AddItemToObject(manifest, "metadata", metadata);
        }
        cJSON_DeleteItemFromObjectCaseSensitive(metadata, "volvox.parent_version_id");
        cJSON_AddStringToObject(metadata, "volvox.parent_version_id", src->version_id);
        out->manifest_json = cJSON_PrintUnformatted(manifest);
        cJSON_Delete(manifest);
        if (!out->manifest_json) { version_free(out); return NULL; }
    }
    return out;
}

static int version_merge_metadata(AdapterVersion* version, const char* metadata_json) {
    if (!metadata_json || !metadata_json[0]) return 0;
    if (!version || !version->manifest_json) return -1;
    cJSON* additions = cJSON_Parse(metadata_json);
    cJSON* manifest = cJSON_Parse(version->manifest_json);
    if (!cJSON_IsObject(additions) || !cJSON_IsObject(manifest)) { cJSON_Delete(additions); cJSON_Delete(manifest); return -1; }
    cJSON* metadata = cJSON_GetObjectItem(manifest, "metadata");
    if (!metadata) {
        metadata = cJSON_CreateObject();
        if (metadata) cJSON_AddItemToObject(manifest, "metadata", metadata);
    }
    if (!cJSON_IsObject(metadata)) { cJSON_Delete(additions); cJSON_Delete(manifest); return -1; }
    for (cJSON* item = additions->child; item; item = item->next) {
        if (!item->string || !cJSON_IsString(item) || !item->valuestring) {
            cJSON_Delete(additions); cJSON_Delete(manifest); return -1;
        }
        cJSON_DeleteItemFromObjectCaseSensitive(metadata, item->string);
        cJSON_AddStringToObject(metadata, item->string, item->valuestring);
    }
    char* encoded = cJSON_PrintUnformatted(manifest);
    cJSON_Delete(additions); cJSON_Delete(manifest);
    if (!encoded) return -1;
    free(version->manifest_json);
    version->manifest_json = encoded;
    return 0;
}

static int apply_tensor_update(AdapterTensor* tensor, const VxAdapterTensorUpdate* update);

static int version_tensor_update_all(AdapterVersion* version, const VxAdapterTensorUpdate* update) {
    if (!version || !update || !update->tensor_name) return -1;
    int matches = 0;
    for (int i = 0; i < version->target_count; i++) {
        AdapterTarget* t = &version->targets[i];
        AdapterTensor* candidates[] = { &t->a, &t->b };
        for (size_t j = 0; j < sizeof(candidates) / sizeof(candidates[0]); j++) {
            if (candidates[j]->data && !strcmp(candidates[j]->name, update->tensor_name)) {
                if (apply_tensor_update(candidates[j], update) != 0) return -1;
                matches++;
            }
        }
    }
    return matches > 0 ? 0 : -1;
}

static int apply_tensor_update(AdapterTensor* tensor, const VxAdapterTensorUpdate* update) {
    if (!tensor || !update || !update->data) return -1;
    size_t elem_size = update->dtype == VX_ADAPTER_DTYPE_F32 ? 4u :
                       update->dtype == VX_ADAPTER_DTYPE_F16 ? 2u : 0u;
    if (!elem_size || update->nbytes != tensor->count * elem_size ||
        (update->mode != VX_ADAPTER_UPDATE_ASSIGN && update->mode != VX_ADAPTER_UPDATE_ADD)) return -1;
    for (size_t i = 0; i < tensor->count; i++) {
        float value = update->dtype == VX_ADAPTER_DTYPE_F32 ? load_f32_unaligned(update->data, i) :
                      f16_to_f32(load_u16_le(update->data, i));
        if (!isfinite(value)) return -1;
        if (update->mode == VX_ADAPTER_UPDATE_ASSIGN) tensor->data[i] = value;
        else tensor->data[i] += value;
        if (!isfinite(tensor->data[i])) return -1;
    }
    return 0;
}

int vx_adapter_clone_update_with_metadata(const char* source_version, const char* new_adapter_id,
                                          const char* new_version_id, const VxAdapterTensorUpdate* updates,
                                          int update_count, const char* metadata_json) {
    if (!new_version_id || !new_version_id[0] || !updates || update_count <= 0) return -1;
    AdapterVersion* source = acquire_version(source_version);
    if (!source) return -1;
    AdapterVersion* clone = version_clone(source, new_adapter_id, new_version_id);
    release_version(source);
    if (!clone) return -1;
    if (version_merge_metadata(clone, metadata_json) != 0) { version_free(clone); return -1; }
    for (int i = 0; i < update_count; i++) {
        if (!updates[i].tensor_name || !updates[i].tensor_name[0] || !updates[i].data) {
            version_free(clone); return -1;
        }
        for (int j = 0; j < i; j++) {
            if (!strcmp(updates[i].tensor_name, updates[j].tensor_name)) { version_free(clone); return -1; }
        }
        if (version_tensor_update_all(clone, &updates[i]) != 0) { version_free(clone); return -1; }
    }
    if (publish_version(clone) != 0) { version_free(clone); return -1; }
    return 0;
}

int vx_adapter_clone_update(const char* source_version, const char* new_adapter_id,
                            const char* new_version_id, const VxAdapterTensorUpdate* updates,
                            int update_count) {
    return vx_adapter_clone_update_with_metadata(source_version, new_adapter_id, new_version_id,
                                                 updates, update_count, NULL);
}

int vx_adapter_activate(const char* version_id) {
    vx_mutex_lock(&g_registry_mutex);
    AdapterVersion* next = NULL;
    if (version_id && version_id[0]) {
        next = find_version_locked(version_id);
        if (!next) { vx_mutex_unlock(&g_registry_mutex); return -1; }
    }
    g_active = next;
    atomic_store_explicit(&g_active_present, next ? 1 : 0, memory_order_release);
    vx_mutex_unlock(&g_registry_mutex);
    return 0;
}

int vx_adapter_remove(const char* version_id) {
    if (!version_id || !version_id[0]) return -1;
    vx_mutex_lock(&g_registry_mutex);
    AdapterVersion** link = &g_versions;
    while (*link && strcmp((*link)->version_id, version_id)) link = &(*link)->next;
    if (!*link) { vx_mutex_unlock(&g_registry_mutex); return -1; }
    AdapterVersion* version = *link;
    *link = version->next;
    version->next = NULL;
    version->removed = 1;
    if (g_active == version) {
        g_active = NULL;
        atomic_store_explicit(&g_active_present, 0, memory_order_release);
    }
    if (version->refs == 0) version_free(version);
    vx_mutex_unlock(&g_registry_mutex);
    return 0;
}

void vx_adapter_reset(void) {
    vx_adapter_request_end();
    if (g_tls.run_depth == 0) {
        free(g_tls.rank_workspace);
        g_tls.rank_workspace = NULL;
        g_tls.rank_workspace_count = 0;
    }
    vx_mutex_lock(&g_registry_mutex);
    AdapterVersion* version = g_versions;
    g_versions = NULL;
    g_active = NULL;
    atomic_store_explicit(&g_active_present, 0, memory_order_release);
    while (version) {
        AdapterVersion* next = version->next;
        version->next = NULL;
        version->removed = 1;
        if (version->refs == 0) version_free(version);
        version = next;
    }
    AdapterTombstone* tombstone = g_tombstones;
    g_tombstones = NULL;
    while (tombstone) {
        AdapterTombstone* next = tombstone->next;
        free(tombstone);
        tombstone = next;
    }
    vx_mutex_unlock(&g_registry_mutex);
}

int vx_adapter_request_begin_many(const char* const* version_ids, const float* scales, int count) {
    if (count < 0 || (count > 0 && !version_ids) || g_tls.request_set || g_tls.run_depth) return -1;
    int n = count > 0 ? count : 1;
    AdapterVersion** versions;
    float* route_scales;
    if (n == 1) {
        g_tls.request_inline_version = NULL;
        g_tls.request_inline_scale = 1.0f;
        versions = &g_tls.request_inline_version;
        route_scales = &g_tls.request_inline_scale;
    } else {
        versions = (AdapterVersion**)calloc((size_t)n, sizeof(*versions));
        route_scales = (float*)malloc((size_t)n * sizeof(*route_scales));
        if (!versions || !route_scales) { free(versions); free(route_scales); return -1; }
    }
    int needs_registry = 0;
    for (int i = 0; i < n; i++) {
        float scale = count > 0 && scales ? scales[i] : 1.0f;
        if (!isfinite(scale)) goto fail;
        route_scales[i] = scale;
        if (count > 0 && version_ids[i] && version_ids[i][0]) needs_registry = 1;
    }
    if (needs_registry) {
        vx_mutex_lock(&g_registry_mutex);
        for (int i = 0; i < n; i++) {
            if (count == 0 || !version_ids[i] || !version_ids[i][0]) continue;
            versions[i] = find_version_locked(version_ids[i]);
            if (!versions[i]) goto fail_locked;
            versions[i]->refs++;
        }
        vx_mutex_unlock(&g_registry_mutex);
    }
    g_tls.request_versions = versions;
    g_tls.request_scales = route_scales;
    g_tls.request_count = n;
    g_tls.request_set = 1;
    return 0;

fail_locked:
    for (int i = 0; i < n; i++) if (versions[i]) release_version_locked(versions[i]);
    vx_mutex_unlock(&g_registry_mutex);
fail:
    if (n > 1) {
        free(versions);
        free(route_scales);
    } else {
        g_tls.request_inline_version = NULL;
        g_tls.request_inline_scale = 1.0f;
    }
    return -1;
}

int vx_adapter_request_begin(const char* version_id) {
    if (g_tls.request_set || g_tls.run_depth) return -1;
    g_tls.request_inline_version = NULL;
    g_tls.request_inline_scale = 1.0f;
    int needs_registry = (version_id && version_id[0]) ||
                         (!version_id && atomic_load_explicit(&g_active_present, memory_order_acquire));
    if (needs_registry) {
        vx_mutex_lock(&g_registry_mutex);
        if (!version_id) g_tls.request_inline_version = g_active;
        else g_tls.request_inline_version = find_version_locked(version_id);
        if (version_id && !g_tls.request_inline_version) {
            vx_mutex_unlock(&g_registry_mutex);
            return -1;
        }
        if (g_tls.request_inline_version) g_tls.request_inline_version->refs++;
        vx_mutex_unlock(&g_registry_mutex);
    }
    g_tls.request_versions = &g_tls.request_inline_version;
    g_tls.request_scales = &g_tls.request_inline_scale;
    g_tls.request_count = 1;
    g_tls.request_set = 1;
    return 0;
}

void vx_adapter_request_end(void) {
    if (!g_tls.request_set || g_tls.run_depth) return;
    int has_versions = 0;
    for (int i = 0; i < g_tls.request_count; i++) if (g_tls.request_versions[i]) { has_versions = 1; break; }
    if (has_versions) {
        vx_mutex_lock(&g_registry_mutex);
        for (int i = 0; i < g_tls.request_count; i++) release_version_locked(g_tls.request_versions[i]);
        vx_mutex_unlock(&g_registry_mutex);
    }
    if (g_tls.request_count > 1) {
        free(g_tls.request_versions);
        free(g_tls.request_scales);
    }
    g_tls.request_versions = NULL;
    g_tls.request_scales = NULL;
    g_tls.request_inline_version = NULL;
    g_tls.request_inline_scale = 1.0f;
    g_tls.request_count = 0;
    g_tls.request_set = 0;
}

int vx_adapter_run_begin(void) {
    if (g_tls.run_depth++ > 0) return 0;
    if (g_tls.request_set) {
        int has_versions = 0;
        for (int i = 0; i < g_tls.request_count; i++) if (g_tls.request_versions[i]) { has_versions = 1; break; }
        if (!has_versions) {
            g_tls.run_versions = NULL;
            g_tls.run_scales = NULL;
            g_tls.run_count = 0;
            g_tls.run_refs_owned = 0;
            return 0;
        }
        g_tls.run_versions = g_tls.request_versions;
        g_tls.run_scales = g_tls.request_scales;
        g_tls.run_count = g_tls.request_count;
        g_tls.run_refs_owned = 0;
        return 0;
    }
    g_tls.run_inline_version = NULL;
    g_tls.run_inline_scale = 1.0f;
    if (!atomic_load_explicit(&g_active_present, memory_order_acquire)) {
        g_tls.run_versions = NULL;
        g_tls.run_scales = NULL;
        g_tls.run_count = 0;
        g_tls.run_refs_owned = 0;
        return 0;
    }
    atomic_fetch_add_explicit(&g_run_registry_lock_count, 1, memory_order_relaxed);
    vx_mutex_lock(&g_registry_mutex);
    g_tls.run_inline_version = g_active;
    if (g_active) g_active->refs++;
    vx_mutex_unlock(&g_registry_mutex);
    g_tls.run_versions = &g_tls.run_inline_version;
    g_tls.run_scales = &g_tls.run_inline_scale;
    g_tls.run_count = 1;
    g_tls.run_refs_owned = 1;
    return 0;
}

void vx_adapter_run_end(void) {
    if (g_tls.run_depth <= 0) return;
    if (--g_tls.run_depth > 0) return;
    if (g_tls.run_refs_owned && g_tls.run_count > 0) {
        vx_mutex_lock(&g_registry_mutex);
        for (int i = 0; i < g_tls.run_count; i++) release_version_locked(g_tls.run_versions[i]);
        vx_mutex_unlock(&g_registry_mutex);
    }
    g_tls.run_versions = NULL;
    g_tls.run_scales = NULL;
    g_tls.run_inline_version = NULL;
    g_tls.run_inline_scale = 1.0f;
    g_tls.run_count = 0;
    g_tls.run_refs_owned = 0;
}

int vx_adapter_run_has_target(const char* weight_name) {
    if (!weight_name || g_tls.run_depth <= 0) return 0;
    for (int i = 0; i < g_tls.run_count; i++) {
        if (target_find(g_tls.run_versions[i], weight_name)) return 1;
    }
    return 0;
}

unsigned long vx_adapter_debug_run_registry_lock_count(void) {
    return atomic_load_explicit(&g_run_registry_lock_count, memory_order_relaxed);
}

void vx_adapter_debug_reset_run_registry_lock_count(void) {
    atomic_store_explicit(&g_run_registry_lock_count, 0, memory_order_relaxed);
}

static float* grow_workspace(float** workspace, int* capacity, int needed) {
    if (needed <= 0) return NULL;
    if (*capacity < needed) {
        float* next = (float*)realloc(*workspace, (size_t)needed * sizeof(float));
        if (!next) return NULL;
        *workspace = next;
        *capacity = needed;
    }
    return *workspace;
}

static AdapterVersion* route_for_row(long global_row, long total_rows, float* scale) {
    if (scale) *scale = 1.0f;
    if (g_tls.run_depth <= 0 || g_tls.run_count <= 0 || !g_tls.run_versions) return NULL;
    int route = 0;
    if (g_tls.run_count > 1) {
        if (total_rows <= 0 || total_rows % g_tls.run_count != 0 || global_row < 0 || global_row >= total_rows) return NULL;
        long rows_per_route = total_rows / g_tls.run_count;
        route = (int)(global_row / rows_per_route);
    }
    if (scale && g_tls.run_scales) *scale = g_tls.run_scales[route];
    return g_tls.run_versions[route];
}

int vx_adapter_run_linear_segments(
        const char* weight_name, int batch_size, long row_offset, int rows,
        long total_rows, int d_in, int d_out,
        VxAdapterLinearSegment* segments, int capacity) {
    int count = 0;
    AdapterTarget* previous_target = NULL;
    float previous_scale = 0.0f;
    if (!weight_name || !weight_name[0] || batch_size <= 0 ||
        row_offset < 0 || rows < 0 || total_rows <= 0 ||
        row_offset > total_rows || rows > total_rows - row_offset ||
        total_rows % batch_size != 0 || d_in <= 0 || d_out <= 0 ||
        g_tls.run_depth <= 0 ||
        (g_tls.run_count != 1 && g_tls.run_count != batch_size) ||
        capacity < 0 || (!segments && capacity != 0)) return -1;
    for (int row = 0; row < rows; row++) {
        float route_scale = 1.0f;
        AdapterVersion* version = route_for_row(
            row_offset + row, total_rows, &route_scale);
        AdapterTarget* target = target_find(version, weight_name);
        float combined_scale = 0.0f;
        if (target) {
            if (target->kind != VX_ADAPTER_LORA || target->d_in != d_in ||
                target->d_out != d_out || target->rank <= 0 ||
                !target->a.data || !target->b.data ||
                target->a.rows != d_in || target->a.cols != target->rank ||
                target->b.rows != target->rank || target->b.cols != d_out ||
                !isfinite(target->scale) || !isfinite(route_scale) ||
                !isfinite(target->scale * route_scale)) return -1;
            combined_scale = target->scale * route_scale;
        }
        if (target && previous_target == target &&
            previous_scale == combined_scale) {
            if (segments && count > 0) segments[count - 1].row_count++;
            continue;
        }
        previous_target = target;
        previous_scale = combined_scale;
        if (!target) continue;
        if (segments) {
            if (count >= capacity) return -1;
            segments[count].a = target->a.data;
            segments[count].b = target->b.data;
            segments[count].row_start = row;
            segments[count].row_count = 1;
            segments[count].rank = target->rank;
            segments[count].scale = combined_scale;
        }
        count++;
    }
    return count;
}

int vx_adapter_apply_linear(const char* weight_name, const float* input,
                            const void* base_weight, VxDataType base_dtype, const float* bias,
                            float* output, int rows, int d_in, int d_out,
                            int base_out_in, int batch_size, long row_offset, long total_rows) {
    (void)base_weight;
    (void)base_dtype;
    (void)base_out_in;
    if (!weight_name || !input || !output || rows < 0 || d_in <= 0 || d_out <= 0 || g_tls.run_depth <= 0) return -1;
    if (batch_size <= 0 || total_rows <= 0 || row_offset < 0 || row_offset > total_rows ||
        rows > total_rows - row_offset || total_rows % batch_size != 0 ||
        (g_tls.run_count != 1 && g_tls.run_count != batch_size)) return -1;
    int max_rank = 0;
    for (int i = 0; i < g_tls.run_count; i++) {
        AdapterTarget* target = target_find(g_tls.run_versions[i], weight_name);
        if (target && target->rank > max_rank) max_rank = target->rank;
    }
    float* rank_tmp = grow_workspace(&g_tls.rank_workspace, &g_tls.rank_workspace_count, max_rank);
    if (max_rank && !rank_tmp) return -1;
    int applied = 0;
    AdapterVersion* cached_version = NULL;
    AdapterTarget* cached_target = NULL;
    int target_cached = 0;
    for (int row = 0; row < rows; row++) {
        float route_scale = 1.0f;
        AdapterVersion* version = route_for_row(row_offset + row, total_rows, &route_scale);
        if (!target_cached || version != cached_version) {
            cached_version = version;
            cached_target = target_find(version, weight_name);
            target_cached = 1;
        }
        AdapterTarget* target = cached_target;
        float* y = output + (size_t)row * d_out;
        const float* x = input + (size_t)row * d_in;
        if (!target) {
            if (!vx_lora_apply_row_f32(x, NULL, NULL, bias, y, NULL, d_in, 0,
                                       d_out, 1.0f, 1.0f))
                return -1;
            continue;
        }
        if (target->d_in != d_in || target->d_out != d_out) return -1;
        if (!vx_lora_apply_row_f32(x, target->a.data, target->b.data, bias, y,
                                   rank_tmp, d_in, target->rank, d_out,
                                   target->scale, route_scale))
            return -1;
        applied = 1;
    }
    return applied;
}

int vx_adapter_materialize_weight(const char* version_id, const char* weight_name,
                                  const void* base_weight, VxDataType base_dtype,
                                  int d_in, int d_out, int base_out_in, float* out_weight) {
    if (!base_weight || !out_weight || (base_dtype != VX_ADAPTER_DTYPE_F32 && base_dtype != VX_ADAPTER_DTYPE_F16)) return -1;
    AdapterVersion* version = acquire_version(version_id);
    if (!version) return -1;
    AdapterTarget* target = target_find(version, weight_name);
    if (!target || target->d_in != d_in || target->d_out != d_out) { release_version(version); return -1; }
    int ok = vx_lora_materialize_weight_f32(
        base_weight, base_dtype, base_out_in, target->a.data, target->b.data,
        target->scale, out_weight, d_in, target->rank, d_out);
    release_version(version);
    return ok ? 0 : -1;
}

int vx_adapter_target_count(const char* version_id) {
    AdapterVersion* version = acquire_version(version_id);
    if (!version) return -1;
    int count = version->target_count;
    release_version(version);
    return count;
}

int vx_adapter_target_info(const char* version_id, int index, VxAdapterTargetInfo* out) {
    if (!out) return -1;
    AdapterVersion* version = acquire_version(version_id);
    if (!version || index < 0 || index >= version->target_count) { release_version(version); return -1; }
    AdapterTarget* target = &version->targets[index];
    memset(out, 0, sizeof(*out));
    memcpy(out->weight_name, target->weight_name, sizeof(out->weight_name));
    out->kind = target->kind; out->d_in = target->d_in; out->d_out = target->d_out;
    out->rank = target->rank; out->alpha = target->alpha; out->scale = target->scale;
    release_version(version);
    return 0;
}

int vx_adapter_get_active(char* out, size_t out_size) {
    if (!out || out_size == 0) return -1;
    vx_mutex_lock(&g_registry_mutex);
    const char* name = g_active ? g_active->version_id : "";
    size_t n = strlen(name);
    if (n >= out_size) { vx_mutex_unlock(&g_registry_mutex); return -1; }
    memcpy(out, name, n + 1);
    vx_mutex_unlock(&g_registry_mutex);
    return 0;
}

static int add_tensor_to_file(SafetensorsFile* file, const AdapterTensor* tensor) {
    if (!tensor || !tensor->data) return 0;
    int shape[2] = { tensor->rows, tensor->cols };
    const SafetensorsTensor* existing = safetensors_find_tensor(file, tensor->name);
    if (existing) {
        if (existing->dtype != SAFETENSORS_DTYPE_F32 || existing->ndim != 2 ||
            existing->nbytes != tensor->count * sizeof(float) ||
            memcmp(existing->data, tensor->data, existing->nbytes)) return -1;
        for (int i = 0; i < 2; i++) if (existing->shape[i] != shape[i]) return -1;
        return 0;
    }
    return safetensors_add_tensor(file, tensor->name, SAFETENSORS_DTYPE_F32,
                                  shape, 2, tensor->data, tensor->count * sizeof(float));
}

static void json_set_string(cJSON* object, const char* name, const char* value) {
    cJSON_DeleteItemFromObjectCaseSensitive(object, name);
    cJSON_AddStringToObject(object, name, value);
}

static void json_set_number(cJSON* object, const char* name, double value) {
    cJSON_DeleteItemFromObjectCaseSensitive(object, name);
    cJSON_AddNumberToObject(object, name, value);
}

static cJSON* manifest_tensor_entry(const char* role, const AdapterTensor* tensor) {
    cJSON* entry = cJSON_CreateObject();
    cJSON* shape = cJSON_CreateArray();
    if (!entry || !shape) { cJSON_Delete(entry); cJSON_Delete(shape); return NULL; }
    cJSON_AddStringToObject(entry, "role", role);
    cJSON_AddStringToObject(entry, "name", tensor->name);
    cJSON_AddItemToArray(shape, cJSON_CreateNumber(tensor->rows));
    cJSON_AddItemToArray(shape, cJSON_CreateNumber(tensor->cols));
    cJSON_AddItemToObject(entry, "shape", shape);
    cJSON_AddStringToObject(entry, "dtype", "F32");
    return entry;
}

static int canonicalize_saved_manifest(cJSON* manifest, const AdapterVersion* version) {
    if (!cJSON_IsObject(manifest) || !version) return -1;
    json_set_string(manifest, "format", "volvox.adapter.v1");
    json_set_string(manifest, "adapter_id", version->adapter_id);
    json_set_string(manifest, "version_id", version->version_id);
    json_set_string(manifest, "kind", kind_name(version->targets[0].kind));
    cJSON* targets = cJSON_GetObjectItem(manifest, "targets");
    if (!cJSON_IsArray(targets) || cJSON_GetArraySize(targets) != version->target_count) return -1;
    for (int i = 0; i < version->target_count; i++) {
        cJSON* original = cJSON_GetArrayItem(targets, i);
        const AdapterTarget* target = &version->targets[i];
        if (!cJSON_IsObject(original)) return -1;
        cJSON* item = cJSON_CreateObject();
        if (!item) return -1;
        const char* preserved_keys[] = { "id", "op" };
        for (size_t p = 0; p < sizeof(preserved_keys) / sizeof(preserved_keys[0]); p++) {
            cJSON* value = cJSON_GetObjectItemCaseSensitive(original, preserved_keys[p]);
            cJSON* copy = value ? cJSON_Duplicate(value, 1) : NULL;
            if (value && (!copy || !cJSON_AddItemToObject(item, preserved_keys[p], copy))) {
                cJSON_Delete(copy);
                cJSON_Delete(item);
                return -1;
            }
        }
        json_set_string(item, "weight", target->weight_name);
        json_set_string(item, "kind", kind_name(target->kind));
        json_set_string(item, "a", target->a.name);
        json_set_string(item, "b", target->b.name);
        json_set_string(item, "layout", "din_r_r_dout");
        json_set_number(item, "rank", target->rank);
        json_set_number(item, "alpha", target->alpha);
        json_set_number(item, "scale", target->scale);
        cJSON* tensor_specs = cJSON_CreateArray();
        cJSON* a_spec = manifest_tensor_entry("a", &target->a);
        cJSON* b_spec = manifest_tensor_entry("b", &target->b);
        if (!tensor_specs || !a_spec || !b_spec) {
            cJSON_Delete(tensor_specs); cJSON_Delete(a_spec); cJSON_Delete(b_spec); return -1;
        }
        cJSON_AddItemToArray(tensor_specs, a_spec);
        cJSON_AddItemToArray(tensor_specs, b_spec);
        cJSON_AddItemToObject(item, "tensors", tensor_specs);
        if (!cJSON_ReplaceItemInArray(targets, i, item)) {
            cJSON_Delete(item);
            return -1;
        }
    }
    return 0;
}

int vx_adapter_save_safetensors(const char* version_id, const char* path) {
    if (!path || !path[0]) return -1;
    AdapterVersion* version = acquire_version(version_id);
    if (!version) return -1;
    SafetensorsFile file;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) { release_version(version); return -1; }
    cJSON* manifest = cJSON_CreateObject();
    cJSON* targets = cJSON_CreateArray();
    cJSON* metadata = cJSON_CreateObject();
    if (!manifest || !targets || !metadata) goto fail;
    cJSON_AddStringToObject(manifest, "format", "volvox.adapter.v1");
    cJSON_AddStringToObject(manifest, "adapter_id", version->adapter_id);
    cJSON_AddStringToObject(manifest, "version_id", version->version_id);
    cJSON_AddStringToObject(manifest, "kind", version->target_count > 0 ? kind_name(version->targets[0].kind) : "lora");
    for (int i = 0; i < version->target_count; i++) {
        AdapterTarget* target = &version->targets[i];
        cJSON* item = cJSON_CreateObject();
        if (!item || add_tensor_to_file(&file, &target->a) != 0 ||
            add_tensor_to_file(&file, &target->b) != 0) { cJSON_Delete(item); goto fail; }
        cJSON_AddStringToObject(item, "weight", target->weight_name);
        cJSON_AddStringToObject(item, "kind", kind_name(target->kind));
        cJSON_AddStringToObject(item, "a", target->a.name);
        cJSON_AddStringToObject(item, "b", target->b.name);
        cJSON_AddStringToObject(item, "layout", "din_r_r_dout");
        cJSON_AddNumberToObject(item, "rank", target->rank);
        cJSON_AddNumberToObject(item, "alpha", target->alpha);
        cJSON_AddNumberToObject(item, "scale", target->scale);
        cJSON* tensor_specs = cJSON_CreateArray();
        cJSON* a_spec = manifest_tensor_entry("a", &target->a);
        cJSON* b_spec = manifest_tensor_entry("b", &target->b);
        if (!tensor_specs || !a_spec || !b_spec) {
            cJSON_Delete(tensor_specs); cJSON_Delete(a_spec); cJSON_Delete(b_spec);
            cJSON_Delete(item); goto fail;
        }
        cJSON_AddItemToArray(tensor_specs, a_spec);
        cJSON_AddItemToArray(tensor_specs, b_spec);
        cJSON_AddItemToObject(item, "tensors", tensor_specs);
        cJSON_AddItemToArray(targets, item);
    }
    cJSON_AddItemToObject(manifest, "targets", targets); targets = NULL;
    if (version->manifest_json) {
        cJSON* preserved = cJSON_Parse(version->manifest_json);
        if (!preserved || canonicalize_saved_manifest(preserved, version) != 0) { cJSON_Delete(preserved); goto fail; }
        cJSON_Delete(manifest);
        manifest = preserved;
    }
    char* manifest_json = cJSON_PrintUnformatted(manifest);
    if (!manifest_json) goto fail;
    cJSON_AddStringToObject(metadata, "volvox_adapter_manifest", manifest_json);
    free(manifest_json);
    char* metadata_json = cJSON_PrintUnformatted(metadata);
    if (!metadata_json || safetensors_set_metadata_json(&file, metadata_json) != 0) { free(metadata_json); goto fail; }
    free(metadata_json);
    cJSON_Delete(manifest); cJSON_Delete(metadata);
    int rc = safetensors_save(path, &file);
    safetensors_free(&file);
    release_version(version);
    return rc;

fail:
    cJSON_Delete(manifest); cJSON_Delete(targets); cJSON_Delete(metadata);
    safetensors_free(&file);
    release_version(version);
    return -1;
}

static cJSON* json_string_item(cJSON* object, const char* name) {
    cJSON* item = object ? cJSON_GetObjectItem(object, name) : NULL;
    return cJSON_IsString(item) && item->valuestring ? item : NULL;
}

static int load_tensor_spec(const SafetensorsFile* file, const char* name,
                            VxAdapterTensorSpec* out, int transpose) {
    const SafetensorsTensor* tensor = safetensors_find_tensor(file, name);
    if (!tensor || (tensor->dtype != SAFETENSORS_DTYPE_F32 && tensor->dtype != SAFETENSORS_DTYPE_F16) ||
        tensor->ndim != 2) return -1;
    memset(out, 0, sizeof(*out));
    out->dtype = tensor->dtype == SAFETENSORS_DTYPE_F32 ? VX_ADAPTER_DTYPE_F32 : VX_ADAPTER_DTYPE_F16;
    out->data = tensor->data;
    out->nbytes = tensor->nbytes;
    out->rows = tensor->shape[0];
    out->cols = tensor->shape[1];
    if (!transpose) return 0;
    size_t count = (size_t)out->rows * out->cols;
    float* values = (float*)malloc(count * sizeof(float));
    if (!values) return -1;
    for (int r = 0; r < out->rows; r++) {
        for (int c = 0; c < out->cols; c++) {
            size_t src = (size_t)r * out->cols + c;
            values[(size_t)c * out->rows + r] = base_value(out->data, out->dtype, src);
        }
    }
    int old_rows = out->rows;
    out->rows = out->cols;
    out->cols = old_rows;
    out->data = values;
    out->nbytes = count * sizeof(float);
    out->dtype = VX_ADAPTER_DTYPE_F32;
    return 1;
}

static int checkpoint_tensor_specs_match(cJSON* target, const SafetensorsFile* file,
                                         const char* a_name, const char* b_name) {
    cJSON* specs = cJSON_GetObjectItemCaseSensitive(target, "tensors");
    if (!specs) return 1;
    if (!cJSON_IsArray(specs) || cJSON_GetArraySize(specs) != 2) return 0;
    int seen_a = 0, seen_b = 0;
    for (int i = 0; i < 2; i++) {
        cJSON* spec = cJSON_GetArrayItem(specs, i);
        cJSON* role = cJSON_GetObjectItemCaseSensitive(spec, "role");
        cJSON* name = cJSON_GetObjectItemCaseSensitive(spec, "name");
        cJSON* shape = cJSON_GetObjectItemCaseSensitive(spec, "shape");
        cJSON* dtype = cJSON_GetObjectItemCaseSensitive(spec, "dtype");
        if (!cJSON_IsString(role) || !cJSON_IsString(name) || !cJSON_IsArray(shape) ||
            cJSON_GetArraySize(shape) != 2 || !cJSON_IsString(dtype)) return 0;
        int is_a = !strcmp(role->valuestring, "a");
        if ((!is_a && strcmp(role->valuestring, "b")) || (is_a ? seen_a++ : seen_b++)) return 0;
        const char* expected_name = is_a ? a_name : b_name;
        const SafetensorsTensor* tensor = safetensors_find_tensor(file, expected_name);
        const char* expected_dtype = tensor && tensor->dtype == SAFETENSORS_DTYPE_F32 ? "F32" :
                                     tensor && tensor->dtype == SAFETENSORS_DTYPE_F16 ? "F16" : "";
        cJSON* row = cJSON_GetArrayItem(shape, 0);
        cJSON* col = cJSON_GetArrayItem(shape, 1);
        if (!tensor || tensor->ndim != 2 || strcmp(name->valuestring, expected_name) ||
            !cJSON_IsNumber(row) || !cJSON_IsNumber(col) ||
            row->valuedouble != tensor->shape[0] || col->valuedouble != tensor->shape[1] ||
            strcmp(dtype->valuestring, expected_dtype)) return 0;
    }
    return seen_a == 1 && seen_b == 1;
}

int vx_adapter_load_safetensors(const char* path, const char* version_override,
                                char* out_version_id, size_t out_version_id_size) {
    SafetensorsFile file;
    if (!path || safetensors_load(path, &file) != 0) return -1;
    cJSON* metadata = file.metadata_json ? cJSON_Parse(file.metadata_json) : NULL;
    cJSON* manifest_string = json_string_item(metadata, "volvox_adapter_manifest");
    cJSON* manifest = manifest_string ? cJSON_Parse(manifest_string->valuestring) : NULL;
    cJSON* format = json_string_item(manifest, "format");
    cJSON* adapter_id = json_string_item(manifest, "adapter_id");
    cJSON* version_id = json_string_item(manifest, "version_id");
    cJSON* root_kind = json_string_item(manifest, "kind");
    cJSON* targets_json = manifest ? cJSON_GetObjectItem(manifest, "targets") : NULL;
    VxAdapterKind manifest_kind;
    if (!format || strcmp(format->valuestring, "volvox.adapter.v1") || !adapter_id || !root_kind ||
        kind_from_name(root_kind->valuestring, &manifest_kind) != 0 ||
        (!version_id && !(version_override && version_override[0])) || !cJSON_IsArray(targets_json)) goto fail;
    int count = cJSON_GetArraySize(targets_json);
    if (count <= 0 || count > INT_MAX / 2 || file.tensor_count != count * 2) goto fail;
    VxAdapterTargetSpec* targets = (VxAdapterTargetSpec*)calloc((size_t)count, sizeof(*targets));
    int* free_a = (int*)calloc((size_t)count, sizeof(int));
    int* free_b = (int*)calloc((size_t)count, sizeof(int));
    if (!targets || !free_a || !free_b) { free(targets); free(free_a); free(free_b); goto fail; }
    int ok = 1;
    for (int i = 0; i < count && ok; i++) {
        cJSON* item = cJSON_GetArrayItem(targets_json, i);
        cJSON* weight = json_string_item(item, "weight");
        cJSON* a = json_string_item(item, "a");
        cJSON* b = json_string_item(item, "b");
        cJSON* layout = json_string_item(item, "layout");
        cJSON* kind = json_string_item(item, "kind");
        cJSON* rank = item ? cJSON_GetObjectItem(item, "rank") : NULL;
        cJSON* alpha = item ? cJSON_GetObjectItem(item, "alpha") : NULL;
        cJSON* scale = item ? cJSON_GetObjectItem(item, "scale") : NULL;
        VxAdapterKind target_kind = manifest_kind;
        if (kind && (kind_from_name(kind->valuestring, &target_kind) != 0 || target_kind != manifest_kind)) { ok = 0; break; }
        double rank_value = cJSON_IsNumber(rank) ? rank->valuedouble : 0.0;
        if (!weight || !a || !b || !cJSON_IsNumber(rank) || !isfinite(rank_value) ||
            rank_value < 1.0 || rank_value > INT_MAX || rank_value != floor(rank_value) ||
            !cJSON_IsNumber(alpha) || !isfinite(alpha->valuedouble) || alpha->valuedouble <= 0.0 ||
            (scale && (!cJSON_IsNumber(scale) || !isfinite(scale->valuedouble)))) { ok = 0; break; }
        targets[i].kind = manifest_kind;
        int peft = layout && !strcmp(layout->valuestring, "peft");
        if (!layout || (!peft && strcmp(layout->valuestring, "din_r_r_dout"))) { ok = 0; break; }
        targets[i].weight_name = weight->valuestring;
        targets[i].rank = (int)rank_value;
        targets[i].alpha = (float)alpha->valuedouble;
        targets[i].scale = cJSON_IsNumber(scale) ? (float)scale->valuedouble : NAN;
        targets[i].a_name = a->valuestring; targets[i].b_name = b->valuestring;
        if (!checkpoint_tensor_specs_match(item, &file, a->valuestring, b->valuestring)) { ok = 0; break; }
        free_a[i] = load_tensor_spec(&file, a->valuestring, &targets[i].a, peft);
        free_b[i] = load_tensor_spec(&file, b->valuestring, &targets[i].b, peft);
        if (free_a[i] < 0 || free_b[i] < 0) { ok = 0; break; }
        targets[i].d_in = targets[i].a.rows;
        targets[i].d_out = targets[i].b.cols;
        if (targets[i].a.cols != targets[i].rank || targets[i].b.rows != targets[i].rank) { ok = 0; break; }
    }
    if (version_override && version_override[0] &&
        !cJSON_ReplaceItemInObjectCaseSensitive(manifest, "version_id", cJSON_CreateString(version_override))) ok = 0;
    char* stored_manifest = cJSON_PrintUnformatted(manifest);
    VxAdapterVersionSpec spec = { adapter_id->valuestring,
        version_override && version_override[0] ? version_override : version_id->valuestring,
        targets, count, stored_manifest };
    int rc = ok && stored_manifest ? vx_adapter_stage(&spec) : -1;
    if (rc == 0 && out_version_id) {
        size_t n = strlen(spec.version_id);
        if (out_version_id_size == 0 || n >= out_version_id_size) {
            vx_adapter_remove(spec.version_id);
            rc = -1;
        } else {
            memcpy(out_version_id, spec.version_id, n + 1);
        }
    }
    free(stored_manifest);
    for (int i = 0; i < count; i++) {
        if (free_a[i] > 0) free((void*)targets[i].a.data);
        if (free_b[i] > 0) free((void*)targets[i].b.data);
    }
    free(targets); free(free_a); free(free_b);
    cJSON_Delete(manifest); cJSON_Delete(metadata); safetensors_free(&file);
    return rc;

fail:
    cJSON_Delete(manifest); cJSON_Delete(metadata); safetensors_free(&file);
    return -1;
}

char* vx_adapter_list_json(void) {
    cJSON* array = cJSON_CreateArray();
    if (!array) return NULL;
    vx_mutex_lock(&g_registry_mutex);
    for (AdapterVersion* version = g_versions; version; version = version->next) {
        cJSON* item = cJSON_CreateObject();
        cJSON* targets = cJSON_CreateArray();
        if (!item || !targets) { cJSON_Delete(item); cJSON_Delete(targets); continue; }
        cJSON_AddStringToObject(item, "adapter_id", version->adapter_id);
        cJSON_AddStringToObject(item, "version_id", version->version_id);
        cJSON_AddBoolToObject(item, "active", version == g_active);
        for (int i = 0; i < version->target_count; i++) {
            AdapterTarget* target = &version->targets[i];
            cJSON* tj = cJSON_CreateObject();
            if (!tj) continue;
            cJSON_AddStringToObject(tj, "weight", target->weight_name);
            cJSON_AddStringToObject(tj, "kind", kind_name(target->kind));
            cJSON_AddNumberToObject(tj, "rank", target->rank);
            cJSON_AddNumberToObject(tj, "alpha", target->alpha);
            cJSON_AddNumberToObject(tj, "scale", target->scale);
            cJSON_AddItemToArray(targets, tj);
        }
        cJSON_AddItemToObject(item, "targets", targets);
        cJSON_AddItemToArray(array, item);
    }
    vx_mutex_unlock(&g_registry_mutex);
    char* json = cJSON_PrintUnformatted(array);
    cJSON_Delete(array);
    return json;
}
