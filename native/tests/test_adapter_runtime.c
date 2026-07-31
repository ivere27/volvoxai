#include "adapter_runtime_internal.h"
#include "cJSON.h"
#include "engine_core.h"
#include "runtime_state.h"
#include "training/training_core.h"
#include "engine_internal.h"
#include "safetensors.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(VX_ADAPTER_DTYPE_F32 == VX_DTYPE_F32 &&
               VX_ADAPTER_DTYPE_F16 == VX_DTYPE_F16,
               "adapter uses protobuf dtype contract");

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return -1; } } while (0)

#ifdef VX_ADAPTER_ALLOC_FAILURE_TEST
static atomic_int g_reject_heap_allocations;

void* __real_malloc(size_t size);
void* __real_calloc(size_t count, size_t size);
void* __real_realloc(void* ptr, size_t size);

void* __wrap_malloc(size_t size) {
    if (atomic_load_explicit(&g_reject_heap_allocations, memory_order_relaxed)) return NULL;
    return __real_malloc(size);
}

void* __wrap_calloc(size_t count, size_t size) {
    if (atomic_load_explicit(&g_reject_heap_allocations, memory_order_relaxed)) return NULL;
    return __real_calloc(count, size);
}

void* __wrap_realloc(void* ptr, size_t size) {
    if (atomic_load_explicit(&g_reject_heap_allocations, memory_order_relaxed)) return NULL;
    return __real_realloc(ptr, size);
}

static void reject_heap_allocations(int reject) {
    atomic_store_explicit(&g_reject_heap_allocations, reject, memory_order_relaxed);
}
#else
static void reject_heap_allocations(int reject) { (void)reject; }
#endif

static int closef(float a, float b) { return fabsf(a - b) <= 1.0e-5f * (1.0f + fabsf(a) + fabsf(b)); }
static int close_tol(float a, float b, float tolerance) { return fabsf(a - b) <= tolerance; }

static VxAdapterTensorSpec matrix(const float* data, int rows, int cols) {
    VxAdapterTensorSpec spec = { data, (size_t)rows * cols * sizeof(float), VX_ADAPTER_DTYPE_F32, rows, cols };
    return spec;
}

static int stage_one(const char* id, const char* weight, int d_in, int d_out, int rank,
                     const float* a, const float* b, float alpha, float scale) {
    VxAdapterTargetSpec target;
    memset(&target, 0, sizeof(target));
    target.weight_name = weight; target.kind = VX_ADAPTER_LORA; target.d_in = d_in; target.d_out = d_out;
    target.rank = rank; target.alpha = alpha; target.scale = scale;
    target.a = matrix(a, d_in, rank); target.b = matrix(b, rank, d_out);
    VxAdapterVersionSpec version = { id, id, &target, 1, NULL };
    return vx_adapter_stage(&version);
}

static void base_matmul(const float* x, const float* w, float* y, int rows, int d_in, int d_out) {
    for (int r = 0; r < rows; r++) for (int j = 0; j < d_out; j++) {
        float sum = 0.0f;
        for (int k = 0; k < d_in; k++) sum += x[(size_t)r * d_in + k] * w[(size_t)k * d_out + j];
        y[(size_t)r * d_out + j] = sum;
    }
}

static int apply_named(const char* name, float route_scale, const char* weight,
                       const float* x, const float* w, const float* bias, float* y,
                       int rows, int d_in, int d_out, int batch) {
    const char* names[1] = { name };
    CHECK(vx_adapter_request_begin_many(names, &route_scale, 1) == 0);
    CHECK(vx_adapter_run_begin() == 0);
    base_matmul(x, w, y, rows, d_in, d_out);
    int rc = vx_adapter_apply_linear(weight, x, w, VX_ADAPTER_DTYPE_F32, bias, y,
                                     rows, d_in, d_out, 0, batch, 0, rows);
    vx_adapter_run_end();
    vx_adapter_request_end();
    return rc;
}

static int test_lora_routes(void) {
    const float w[4] = {1, 0, 0, 2};
    const float x[2] = {2, 3};
    const float bias[2] = {0.25f, -0.5f};
    const float a[2] = {1, 1};
    const float b[2] = {1, 1};
    float y[2];
    CHECK(stage_one("lora", "w", 2, 2, 1, a, b, 1, 1) == 0);
    CHECK(apply_named("lora", 1, "w", x, w, bias, y, 1, 2, 2, 1) == 1);
    CHECK(closef(y[0], 7.25f) && closef(y[1], 10.5f));
    CHECK(apply_named("lora", 0, "w", x, w, bias, y, 1, 2, 2, 1) == 1);
    CHECK(closef(y[0], 2.25f) && closef(y[1], 5.5f));

    vx_adapter_reset();
    return 0;
}

static int test_inline_tls_routes(void) {
    const float a[2] = {1, 1};
    const float b[2] = {1, 0};
    const char* single_route[1] = {"tls-inline"};
    const float single_scale[1] = {0.5f};
    const char* multi_routes[2] = {"tls-inline", "tls-inline"};
    const float multi_scales[2] = {1.0f, 0.5f};

    vx_adapter_reset();
    vx_adapter_debug_reset_run_registry_lock_count();
    reject_heap_allocations(1);
    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 0);
    vx_adapter_run_end();
    CHECK(vx_adapter_debug_run_registry_lock_count() == 0);
    reject_heap_allocations(0);

    CHECK(stage_one("tls-inline", "w", 2, 2, 1, a, b, 1, 1) == 0);
    CHECK(vx_adapter_activate("tls-inline") == 0);
    vx_adapter_debug_reset_run_registry_lock_count();

    /* These base/single-route lifecycles must not consult the heap. */
    reject_heap_allocations(1);

    CHECK(vx_adapter_request_begin("") == 0);
    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 0);
    vx_adapter_run_end();
    vx_adapter_request_end();
    CHECK(vx_adapter_debug_run_registry_lock_count() == 0);

    CHECK(vx_adapter_request_begin(NULL) == 0);
    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 1);
    vx_adapter_run_end();
    vx_adapter_request_end();

    CHECK(vx_adapter_request_begin("tls-inline") == 0);
    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 1);
    vx_adapter_run_end();
    vx_adapter_request_end();

    CHECK(vx_adapter_request_begin_many(NULL, NULL, 0) == 0);
    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 0);
    vx_adapter_run_end();
    vx_adapter_request_end();

    CHECK(vx_adapter_request_begin_many(single_route, single_scale, 1) == 0);
    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 1);
    vx_adapter_run_end();
    vx_adapter_request_end();

    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 1);
    CHECK(vx_adapter_run_begin() == 0);
    vx_adapter_run_end();
    vx_adapter_run_end();
    CHECK(vx_adapter_debug_run_registry_lock_count() == 1);

    CHECK(vx_adapter_activate("") == 0);
    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 0);
    vx_adapter_run_end();
    CHECK(vx_adapter_debug_run_registry_lock_count() == 1);
    CHECK(vx_adapter_activate("tls-inline") == 0);

#ifdef VX_ADAPTER_ALLOC_FAILURE_TEST
    CHECK(vx_adapter_request_begin_many(multi_routes, multi_scales, 2) == -1);
#endif
    reject_heap_allocations(0);

    CHECK(vx_adapter_request_begin_many(multi_routes, multi_scales, 2) == 0);
    CHECK(vx_adapter_run_begin() == 0);
    CHECK(vx_adapter_run_has_target("w") == 1);
    vx_adapter_run_end();
    vx_adapter_request_end();

    vx_adapter_reset();
    return 0;
}

static int test_scaling_and_clone(void) {
    const float w[4] = {1, 0, 0, 1};
    const float identity[4] = {1, 0, 0, 1};
    CHECK(stage_one("scaled", "w", 2, 2, 2, identity, identity, 4, NAN) == 0);
    VxAdapterTargetInfo info;
    CHECK(vx_adapter_target_info("scaled", 0, &info) == 0);
    CHECK(closef(info.scale, 2.0f));
    CHECK(stage_one("bad-alpha", "w2", 2, 2, 2, identity, identity, 0, NAN) == -1);
    vx_adapter_reset();

    const float a[2] = {1, 0};
    const float x[2] = {2, 5};
    float y[2];
    const float old_b[2] = {1, 0};
    const float new_b[2] = {0, 2};
    CHECK(stage_one("clone-old", "w", 2, 2, 1, a, old_b, 1, 1) == 0);
    VxAdapterTensorUpdate update = { "w.lora_B", new_b, sizeof(new_b),
                                     VX_ADAPTER_DTYPE_F32, VX_ADAPTER_UPDATE_ASSIGN };
    CHECK(vx_adapter_clone_update("clone-old", "clone", "clone-new", &update, 1) == 0);
    CHECK(apply_named("clone-old", 1, "w", x, w, NULL, y, 1, 2, 2, 1) == 1);
    CHECK(closef(y[0], 4) && closef(y[1], 5));
    CHECK(apply_named("clone-new", 1, "w", x, w, NULL, y, 1, 2, 2, 1) == 1);
    CHECK(closef(y[0], 2) && closef(y[1], 9));
    CHECK(vx_adapter_remove("clone-old") == 0);
    CHECK(stage_one("clone-old", "w", 2, 2, 1, a, old_b, 1, 1) == -1);
    vx_adapter_reset();
    CHECK(stage_one("clone-old", "w", 2, 2, 1, a, old_b, 1, 1) == 0);
    vx_adapter_reset();
    return 0;
}

static int test_misaligned_f16_payload(void) {
    unsigned char a_storage[5] = {0, 0x00, 0x3c, 0x00, 0x3c};
    unsigned char b_storage[5] = {0, 0x00, 0x3c, 0x00, 0x40};
    VxAdapterTargetSpec target;
    memset(&target, 0, sizeof(target));
    target.weight_name = "w"; target.kind = VX_ADAPTER_LORA;
    target.d_in = 2; target.d_out = 2; target.rank = 1; target.alpha = 1; target.scale = 1;
    target.a = (VxAdapterTensorSpec){a_storage + 1, 4, VX_ADAPTER_DTYPE_F16, 2, 1};
    target.b = (VxAdapterTensorSpec){b_storage + 1, 4, VX_ADAPTER_DTYPE_F16, 1, 2};
    VxAdapterVersionSpec version = {"misaligned", "misaligned", &target, 1, NULL};
    CHECK(vx_adapter_stage(&version) == 0);
    const float w[4] = {1, 0, 0, 1};
    const float x[2] = {2, 3};
    float y[2];
    CHECK(apply_named("misaligned", 1, "w", x, w, NULL, y, 1, 2, 2, 1) == 1);
    CHECK(closef(y[0], 7) && closef(y[1], 13));
    vx_adapter_reset();
    return 0;
}

typedef struct {
    VxEngineState* engine_state;
    const char* version;
} ActivateThreadArgs;

static void* activate_thread(void* opaque) {
    ActivateThreadArgs* args = (ActivateThreadArgs*)opaque;
    VxEngineStateScope scope = vx_engine_state_scope_enter(args->engine_state);
    int result = vx_adapter_activate(args->version);
    vx_engine_state_scope_leave(scope);
    return (void*)(intptr_t)result;
}

static int test_hot_swap_and_batch(void) {
    const float w[4] = {1, 0, 0, 1};
    const float a[2] = {1, 1};
    const float b1[2] = {1, 0};
    const float b2[2] = {0, 2};
    const float x[4] = {1, 1, 2, 1};
    float y[4];
    CHECK(stage_one("old", "w", 2, 2, 1, a, b1, 1, 1) == 0);
    CHECK(stage_one("new", "w", 2, 2, 1, a, b2, 1, 1) == 0);
    CHECK(vx_adapter_activate("old") == 0);
    CHECK(vx_adapter_request_begin(NULL) == 0);
    ActivateThreadArgs activate = {vx_engine_state_current(), "new"};
    pthread_t thread;
    CHECK(pthread_create(&thread, NULL, activate_thread, &activate) == 0);
    void* result = NULL;
    CHECK(pthread_join(thread, &result) == 0 && (intptr_t)result == 0);
    CHECK(vx_adapter_remove("old") == 0);
    CHECK(vx_adapter_run_begin() == 0);
    base_matmul(x, w, y, 1, 2, 2);
    CHECK(vx_adapter_apply_linear("w", x, w, VX_ADAPTER_DTYPE_F32, NULL, y,
                                  1, 2, 2, 0, 1, 0, 1) == 1);
    CHECK(closef(y[0], 3) && closef(y[1], 1));
    vx_adapter_run_end();
    vx_adapter_request_end();

    CHECK(stage_one("old2", "w", 2, 2, 1, a, b1, 1, 1) == 0);
    const char* routes[2] = {"old2", "new"};
    const float scales[2] = {1, 0.5f};
    CHECK(vx_adapter_request_begin_many(routes, scales, 2) == 0);
    CHECK(vx_adapter_run_begin() == 0);
    base_matmul(x, w, y, 2, 2, 2);
    CHECK(vx_adapter_apply_linear("w", x, w, VX_ADAPTER_DTYPE_F32, NULL, y,
                                  2, 2, 2, 0, 2, 0, 2) == 1);
    CHECK(closef(y[0], 3) && closef(y[1], 1));
    CHECK(closef(y[2], 2) && closef(y[3], 4));
    vx_adapter_run_end(); vx_adapter_request_end();

    const float x_flat[8] = {1, 1, 2, 1, 1, 2, 2, 2};
    float y_flat[8];
    CHECK(vx_adapter_request_begin_many(routes, scales, 2) == 0);
    CHECK(vx_adapter_run_begin() == 0);
    base_matmul(x_flat, w, y_flat, 4, 2, 2);
    CHECK(vx_adapter_apply_linear("w", x_flat, w, VX_ADAPTER_DTYPE_F32, NULL, y_flat,
                                  4, 2, 2, 0, 2, 0, 4) == 1);
    CHECK(closef(y_flat[0], 3) && closef(y_flat[1], 1) && closef(y_flat[2], 5) && closef(y_flat[3], 1));
    CHECK(closef(y_flat[4], 1) && closef(y_flat[5], 5) && closef(y_flat[6], 2) && closef(y_flat[7], 6));
    vx_adapter_run_end(); vx_adapter_request_end();
    vx_adapter_reset();
    return 0;
}

typedef struct {
    const char* version;
    atomic_int done;
    int rc;
    VxEngineState* engine_state;
} MergeThreadArgs;

static void* merge_thread(void* opaque) {
    MergeThreadArgs* args = (MergeThreadArgs*)opaque;
    VxEngineStateScope scope = vx_engine_state_scope_enter(args->engine_state);
    args->rc = volvoxai_engine_adapter_merge(args->version);
    vx_engine_state_scope_leave(scope);
    atomic_store_explicit(&args->done, 1, memory_order_release);
    return NULL;
}

static int test_manifest_metadata_roundtrip(void) {
    const float a[2] = {1, 0};
    const float b[2] = {1, 1};
    VxAdapterTargetSpec target;
    memset(&target, 0, sizeof(target));
    target.weight_name = "w"; target.kind = VX_ADAPTER_LORA; target.d_in = 2; target.d_out = 2;
    target.rank = 1; target.alpha = 1; target.scale = 1;
    target.a_name = "a"; target.b_name = "b";
    target.a = matrix(a, 2, 1); target.b = matrix(b, 1, 2);
    const char* manifest =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"meta\",\"version_id\":\"meta\","
        "\"kind\":\"lora\",\"metadata\":{\"owner\":\"native-test\"},\"targets\":[{"
        "\"id\":\"target-1\",\"op\":\"Linear\",\"weight\":\"w\",\"a\":\"a\",\"b\":\"b\","
        "\"layout\":\"peft\",\"rank\":1,\"alpha\":1,\"scale\":1}]}";
    VxAdapterVersionSpec version = {"meta", "meta", &target, 1, manifest};
    CHECK(vx_adapter_stage(&version) == 0);
    const float child_b[2] = {2, 3};
    VxAdapterTensorUpdate child_update = {"b", child_b, sizeof(child_b), VX_ADAPTER_DTYPE_F32,
                                           VX_ADAPTER_UPDATE_ASSIGN};
    CHECK(vx_adapter_clone_update_with_metadata("meta", "meta", "meta-child", &child_update, 1,
                                                "{\"volvox.optimizer_step\":\"7\"}") == 0);
    const char* path = "/tmp/volvox-adapter-metadata.safetensors";
    CHECK(vx_adapter_save_safetensors("meta-child", path) == 0);
    SafetensorsFile file;
    CHECK(safetensors_load(path, &file) == 0);
    cJSON* metadata = cJSON_Parse(file.metadata_json);
    cJSON* encoded = cJSON_GetObjectItem(metadata, "volvox_adapter_manifest");
    CHECK(cJSON_IsString(encoded));
    cJSON* saved = cJSON_Parse(encoded->valuestring);
    cJSON* custom = cJSON_GetObjectItem(cJSON_GetObjectItem(saved, "metadata"), "owner");
    cJSON* parent = cJSON_GetObjectItem(cJSON_GetObjectItem(saved, "metadata"), "volvox.parent_version_id");
    cJSON* step = cJSON_GetObjectItem(cJSON_GetObjectItem(saved, "metadata"), "volvox.optimizer_step");
    cJSON* saved_target = cJSON_GetArrayItem(cJSON_GetObjectItem(saved, "targets"), 0);
    cJSON* id = cJSON_GetObjectItem(saved_target, "id");
    cJSON* layout = cJSON_GetObjectItem(saved_target, "layout");
    cJSON* tensor_spec = cJSON_GetArrayItem(cJSON_GetObjectItem(saved_target, "tensors"), 0);
    cJSON* tensor_dtype = cJSON_GetObjectItem(tensor_spec, "dtype");
    cJSON* tensor_shape = cJSON_GetObjectItem(tensor_spec, "shape");
    cJSON* tensor_quant = cJSON_GetObjectItem(tensor_spec, "quant");
    CHECK(cJSON_IsString(custom) && !strcmp(custom->valuestring, "native-test"));
    CHECK(cJSON_IsString(parent) && !strcmp(parent->valuestring, "meta"));
    CHECK(cJSON_IsString(step) && !strcmp(step->valuestring, "7"));
    CHECK(cJSON_IsString(id) && !strcmp(id->valuestring, "target-1"));
    CHECK(cJSON_IsString(layout) && !strcmp(layout->valuestring, "din_r_r_dout"));
    CHECK(cJSON_IsString(tensor_dtype) && !strcmp(tensor_dtype->valuestring, "F32"));
    CHECK(tensor_quant == NULL);
    CHECK(cJSON_GetArraySize(tensor_shape) == 2 && cJSON_GetArrayItem(tensor_shape, 0)->valueint == 2 &&
          cJSON_GetArrayItem(tensor_shape, 1)->valueint == 1);
    cJSON_Delete(saved); cJSON_Delete(metadata); safetensors_free(&file);
    vx_adapter_reset();
    char loaded[128];
    CHECK(vx_adapter_load_safetensors(path, "meta-loaded", loaded, sizeof(loaded)) == 0);
    CHECK(!strcmp(loaded, "meta-loaded"));
    vx_adapter_reset(); remove(path);
    return 0;
}

static int test_engine_merge_exact(void) {
    static float weight[4] = {1, 2, 3, 4};
    static float input[2];
    static float output[2];
    const float backup[4] = {1, 2, 3, 4};
    const float a[2] = {1, 1};
    const float b[2] = {1, -1};
    memset(g_t, 0, sizeof(g_t)); memset(g_n, 0, sizeof(g_n));
    g_nt = 3; g_nn = 1;
    strcpy(g_t[0].name, "w"); g_t[0].shape[0] = 2; g_t[0].shape[1] = 2; g_t[0].ndim = 2;
    g_t[0].data = weight; g_t[0].numel = 4; g_t[0].dtype = T_F32; g_t[0].elem_size = 4;
    strcpy(g_t[1].name, "x"); g_t[1].shape[0] = 1; g_t[1].shape[1] = 2; g_t[1].ndim = 2;
    g_t[1].data = input; g_t[1].numel = 2; g_t[1].dtype = T_F32; g_t[1].elem_size = 4;
    strcpy(g_t[2].name, "y"); g_t[2].shape[0] = 1; g_t[2].shape[1] = 2; g_t[2].ndim = 2;
    g_t[2].data = output; g_t[2].numel = 2; g_t[2].dtype = T_F32; g_t[2].elem_size = 4;
    strcpy(g_n[0].op, "Linear"); strcpy(g_n[0].out, "y"); g_n[0].nin = 2;
    strcpy(g_n[0].ins[0].key, "input"); strcpy(g_n[0].ins[0].name, "x");
    strcpy(g_n[0].ins[1].key, "weight"); strcpy(g_n[0].ins[1].name, "w");
    volvoxai_engine_tensor_name_index_invalidate();
    input[0] = 2.0f; input[1] = -1.0f; g_loaded = 1;
    CHECK(stage_one("merge", "w", 2, 2, 1, a, b, 1, 1) == 0);
    CHECK(volvoxai_engine_adapter_activate("merge") == 0);
    CHECK(volvoxai_engine_adapter_remove("merge") == -1);
    CHECK(volvoxai_engine_forward() == 0);
    float routed_output[2]; memcpy(routed_output, output, sizeof(routed_output));
    MergeThreadArgs args = {"merge", 0, -1, vx_engine_state_current()};
    atomic_init(&args.done, 0);
    pthread_t thread;
    CHECK(volvoxai_engine_adapter_route_begin("merge") == 0);
    CHECK(pthread_create(&thread, NULL, merge_thread, &args) == 0);
    struct timespec delay = {0, 50000000};
    nanosleep(&delay, NULL);
    CHECK(atomic_load_explicit(&args.done, memory_order_acquire) == 0);
    volvoxai_engine_adapter_route_end();
    CHECK(pthread_join(thread, NULL) == 0 && args.rc == 0);
    CHECK(memcmp(weight, backup, sizeof(weight)) != 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(closef(output[0], routed_output[0]) && closef(output[1], routed_output[1]));
    float merged[4]; memcpy(merged, weight, sizeof(merged));
    CHECK(volvoxai_engine_set_tensor_f32("w", backup, 4) == -1);
    CHECK(volvoxai_engine_adapter_remove("merge") == -1);
    CHECK(volvoxai_engine_adapter_activate("") == -1);
    weight[0] += 1.0f;
    CHECK(volvoxai_engine_adapter_unmerge() == -1);
    memcpy(weight, merged, sizeof(merged));
    CHECK(volvoxai_engine_adapter_unmerge() == 0);
    CHECK(memcmp(weight, backup, sizeof(weight)) == 0);
    char active[128]; CHECK(vx_adapter_get_active(active, sizeof(active)) == 0 && !strcmp(active, "merge"));
    CHECK(volvoxai_engine_adapter_remove("merge") == -1);
    CHECK(volvoxai_engine_adapter_activate("") == 0);
    CHECK(volvoxai_engine_adapter_remove("merge") == 0);
    vx_adapter_reset(); g_nt = 0; g_nn = 0; g_loaded = 0;
    return 0;
}

static int test_out_in_lora_merge_exact(void) {
    static float weight[6] = {1, 2, 3, 4, 5, 6};
    static float input[2];
    static float output[3];
    const float backup[6] = {1, 2, 3, 4, 5, 6};
    const float a[2] = {1, 1};
    const float b[3] = {1, -1, 2};
    memset(g_t, 0, sizeof(g_t)); memset(g_n, 0, sizeof(g_n));
    g_nt = 3; g_nn = 1;
    strcpy(g_t[0].name, "w-out-in"); g_t[0].shape[0] = 3; g_t[0].shape[1] = 2; g_t[0].ndim = 2;
    g_t[0].data = weight; g_t[0].numel = 6; g_t[0].dtype = T_F32; g_t[0].elem_size = 4;
    strcpy(g_t[1].name, "x"); g_t[1].shape[0] = 1; g_t[1].shape[1] = 2; g_t[1].ndim = 2;
    g_t[1].data = input; g_t[1].numel = 2; g_t[1].dtype = T_F32; g_t[1].elem_size = 4;
    strcpy(g_t[2].name, "y"); g_t[2].shape[0] = 1; g_t[2].shape[1] = 3; g_t[2].ndim = 2;
    g_t[2].data = output; g_t[2].numel = 3; g_t[2].dtype = T_F32; g_t[2].elem_size = 4;
    strcpy(g_n[0].op, "Linear"); strcpy(g_n[0].out, "y"); g_n[0].nin = 2;
    strcpy(g_n[0].ins[0].key, "input"); strcpy(g_n[0].ins[0].name, "x");
    strcpy(g_n[0].ins[1].key, "weight"); strcpy(g_n[0].ins[1].name, "w-out-in");
    volvoxai_engine_tensor_name_index_invalidate();
    g_n[0].params = cJSON_Parse("{\"weight_layout\":\"OUT_IN\"}");
    CHECK(g_n[0].params != NULL);
    input[0] = 1.5f; input[1] = -0.5f; g_loaded = 1;
    CHECK(stage_one("lora-out-in", "w-out-in", 2, 3, 1, a, b, 1, 1) == 0);
    CHECK(volvoxai_engine_adapter_activate("lora-out-in") == 0);
    CHECK(volvoxai_engine_forward() == 0);
    float routed_output[3]; memcpy(routed_output, output, sizeof(routed_output));
    CHECK(volvoxai_engine_adapter_merge("lora-out-in") == 0);
    CHECK(memcmp(weight, backup, sizeof(weight)) != 0);
    CHECK(volvoxai_engine_forward() == 0);
    for (int i = 0; i < 3; i++) CHECK(closef(output[i], routed_output[i]));
    CHECK(volvoxai_engine_adapter_unmerge() == 0);
    CHECK(memcmp(weight, backup, sizeof(weight)) == 0);
    cJSON_Delete(g_n[0].params); g_n[0].params = NULL;
    vx_adapter_reset(); g_nt = 0; g_nn = 0; g_loaded = 0;
    return 0;
}

static int test_flush_and_patch_lifetimes(void) {
    const char* graph_path = "/tmp/volvox-adapter-graph.json";
    const char* weights_path = "/tmp/volvox-adapter-weights.safetensors";
    const char* flushed_path = "/tmp/volvox-adapter-weights-flushed.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,2],\"dtype\":\"float32\"}},\"nodes\":["
        "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},"
        "\"outputs\":{\"output\":\"h\"},\"outputs_shape\":{\"output\":[2,2]}},"
        "{\"opType\":\"GELU\",\"inputs\":{\"input\":\"h\"},"
        "\"outputs\":{\"output\":\"y\"},\"outputs_shape\":{\"output\":[2,2]}}],\"outputs\":[\"y\"]}";
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL);
    CHECK(fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    const uint16_t weight[4] = {0x2e66, 0x3266, 0x34cd, 0x3666};
    const int shape[2] = {2, 2};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_F16, shape, 2, weight, sizeof(weight)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_is_graph_input("x") == 1 && volvoxai_engine_is_graph_input(NULL) == 1);
    CHECK(volvoxai_engine_is_graph_input("w") == 0 && volvoxai_engine_is_graph_input("y") == 0);
    long rejected_numel = -1;
    CHECK(volvoxai_engine_input_ptr("w", &rejected_numel) == NULL && rejected_numel == 0);
    CHECK(volvoxai_engine_input_ptr("y", &rejected_numel) == NULL && rejected_numel == 0);
    T* graph_input = t_find("x");
    CHECK(graph_input != NULL && graph_input->is_graph_input);
    int input_dtype = graph_input->dtype;
    size_t input_elem_size = graph_input->elem_size;
    graph_input->dtype = T_F16; graph_input->elem_size = 2;
    CHECK(volvoxai_engine_is_graph_input("x") == 1);
    CHECK(volvoxai_engine_input_ptr("x", &rejected_numel) == NULL && rejected_numel == 0);
    graph_input->dtype = input_dtype; graph_input->elem_size = input_elem_size;
    long weight_numel = 0;
    float weight_values[4];
    CHECK(volvoxai_engine_copy_tensor_f32("w", weight_values, 4) == 0);
    int weight_shape[8], weight_ndim = 0, weight_dtype = -1; size_t weight_elem = 0;
    CHECK(volvoxai_engine_tensor_info_ex("w", &weight_numel, weight_shape, &weight_ndim, &weight_dtype, &weight_elem) == 0 &&
          weight_dtype == T_F16 && weight_elem == 2);
    const float adapter_a[2] = {1, 0};
    const float adapter_b[2] = {0.01f, 0.02f};
    const char* adapter_names[2] = {"fa", "fb"};
    const void* adapter_data[2] = {adapter_a, adapter_b};
    const int adapter_dtypes[2] = {T_F32, T_F32};
    const size_t adapter_bytes[2] = {sizeof(adapter_a), sizeof(adapter_b)};
    const char* adapter_manifest =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"f16\",\"version_id\":\"f16\","
        "\"kind\":\"lora\",\"targets\":[{\"weight\":\"w\",\"a\":\"fa\",\"b\":\"fb\","
        "\"layout\":\"din_r_r_dout\",\"rank\":1,\"alpha\":1,\"scale\":1}]}";
    const char* bad_field_manifest =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"bad-field\",\"version_id\":\"bad-field\","
        "\"kind\":\"lora\",\"targets\":[{\"weight\":\"w\",\"a\":\"fa\",\"b\":\"fb\","
        "\"layout\":\"din_r_r_dout\",\"rank\":1,\"alpha\":1,\"metadata\":{\"x\":\"y\"}}]}";
    const char* bad_spec_manifest =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"bad-spec\",\"version_id\":\"bad-spec\","
        "\"kind\":\"lora\",\"targets\":[{\"weight\":\"w\",\"a\":\"fa\",\"b\":\"fb\","
        "\"layout\":\"din_r_r_dout\",\"rank\":1,\"alpha\":1,\"tensors\":["
        "{\"role\":\"a\",\"name\":\"fa\",\"shape\":[1,2],\"dtype\":\"F32\"},"
        "{\"role\":\"b\",\"name\":\"fb\",\"shape\":[1,2],\"dtype\":\"F32\"}]}]}";
    const char* bad_layout_manifest =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"bad-layout\",\"version_id\":\"bad-layout\","
        "\"kind\":\"lora\",\"targets\":[{\"weight\":\"w\",\"a\":\"fa\",\"b\":\"fb\","
        "\"layout\":\"IN_OUT\",\"rank\":1,\"alpha\":1}]}";
    const char* bad_alpha_manifest =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"bad-alpha\",\"version_id\":\"bad-alpha\","
        "\"kind\":\"lora\",\"targets\":[{\"weight\":\"w\",\"a\":\"fa\",\"b\":\"fb\","
        "\"layout\":\"din_r_r_dout\",\"rank\":1,\"alpha\":0}]}";
    const char* missing_alpha_manifest =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"missing-alpha\",\"version_id\":\"missing-alpha\","
        "\"kind\":\"lora\",\"targets\":[{\"weight\":\"w\",\"a\":\"fa\",\"b\":\"fb\","
        "\"layout\":\"din_r_r_dout\",\"rank\":1}]}";
    CHECK(volvoxai_engine_adapter_stage_json(bad_field_manifest, adapter_names, adapter_data,
                                    adapter_dtypes, adapter_bytes, 2) == -1);
    CHECK(volvoxai_engine_adapter_stage_json(bad_spec_manifest, adapter_names, adapter_data,
                                    adapter_dtypes, adapter_bytes, 2) == -1);
    CHECK(volvoxai_engine_adapter_stage_json(bad_layout_manifest, adapter_names, adapter_data,
                                    adapter_dtypes, adapter_bytes, 2) == -1);
    CHECK(volvoxai_engine_adapter_stage_json(bad_alpha_manifest, adapter_names, adapter_data,
                                    adapter_dtypes, adapter_bytes, 2) == -1);
    CHECK(volvoxai_engine_adapter_stage_json(missing_alpha_manifest, adapter_names, adapter_data,
                                    adapter_dtypes, adapter_bytes, 2) == -1);
    CHECK(volvoxai_engine_adapter_stage_json(adapter_manifest, adapter_names, adapter_data,
                                    adapter_dtypes, adapter_bytes, 2) == 0);
    CHECK(volvoxai_engine_adapter_activate("f16") == 0);
    long n = 0;
    float* input = volvoxai_engine_input_ptr("x", &n);
    CHECK(input && n == 4);
    for (long i = 0; i < n; i++) input[i] = 0.1f * (float)(i + 1);
    CHECK(volvoxai_engine_forward() == 0);
    float routed_output[4];
    CHECK(volvoxai_engine_copy_tensor_f32("y", routed_output, 4) == 0);
    CHECK(volvoxai_engine_save_weight_file(0, flushed_path) == 0);
    uint16_t weight_backup[4];
    CHECK(volvoxai_engine_copy_tensor_raw("w", weight_backup, sizeof(weight_backup)) == 0);
    CHECK(volvoxai_engine_adapter_merge("f16") == 0);
    CHECK(volvoxai_engine_save_weight_file(0, flushed_path) == -1);
    CHECK(volvoxai_engine_forward() == 0);
    float merged_output[4];
    CHECK(volvoxai_engine_copy_tensor_f32("y", merged_output, 4) == 0);
    for (int i = 0; i < 4; i++) CHECK(close_tol(merged_output[i], routed_output[i], 1.0e-3f));
    uint16_t weight_merged[4];
    CHECK(volvoxai_engine_copy_tensor_raw("w", weight_merged, sizeof(weight_merged)) == 0 &&
          memcmp(weight_merged, weight_backup, sizeof(weight_backup)) != 0);
    CHECK(volvoxai_engine_adapter_unmerge() == 0);
    uint16_t weight_restored[4];
    CHECK(volvoxai_engine_copy_tensor_raw("w", weight_restored, sizeof(weight_restored)) == 0 &&
          memcmp(weight_restored, weight_backup, sizeof(weight_backup)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_patch_node_json(1, "{\"params\":{\"approximate\":\"none\"}}",
                                          VOLVOXAI_ENGINE_NODE_PATCH_MODE_MERGE, 1) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    volvoxai_engine_shutdown();
    remove(graph_path); remove(weights_path); remove(flushed_path);
    return 0;
}

static int test_quantized_base_rejected(void) {
    const char* graph_path = "/tmp/volvox-quantized-graph.json";
    const char* weights_path = "/tmp/volvox-quantized-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wq\","
        "\"weight_scale\":\"wq.scale\"},\"outputs\":{\"output\":\"y\"},"
        "\"outputs_shape\":{\"output\":[1,2]}}],\"outputs\":[\"y\"]}";
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    const int8_t weight[4] = {1, 2, 3, 4};
    const float weight_scale[2] = {0.5f, 0.25f};
    const int matrix_shape[2] = {2, 2};
    const int vector_shape[1] = {2};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "wq", SAFETENSORS_DTYPE_I8, matrix_shape, 2, weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "wq.scale", SAFETENSORS_DTYPE_F32, vector_shape, 1,
                                 weight_scale, sizeof(weight_scale)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    int8_t raw_weight[4] = {0};
    CHECK(volvoxai_engine_copy_tensor_raw("wq", raw_weight, sizeof(raw_weight)) == 0 &&
          memcmp(raw_weight, weight, sizeof(weight)) == 0);
    const float a[2] = {1, 1};
    const float b[2] = {1, 2};
    const char* names[2] = {"qa", "qb"};
    const void* data[2] = {a, b};
    const int dtypes[2] = {T_F32, T_F32};
    const size_t nbytes[2] = {sizeof(a), sizeof(b)};
    const char* manifest =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"quantized\",\"version_id\":\"quantized\","
        "\"kind\":\"lora\",\"targets\":[{\"weight\":\"wq\",\"a\":\"qa\",\"b\":\"qb\","
        "\"layout\":\"din_r_r_dout\",\"rank\":1,\"alpha\":1,\"scale\":1}]}";
    CHECK(volvoxai_engine_adapter_stage_json(manifest, names, data, dtypes, nbytes, 2) == -1);
    volvoxai_engine_shutdown(); remove(graph_path); remove(weights_path);
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    CHECK(test_lora_routes() == 0);
    CHECK(test_inline_tls_routes() == 0);
    CHECK(test_scaling_and_clone() == 0);
    CHECK(test_misaligned_f16_payload() == 0);
    CHECK(test_hot_swap_and_batch() == 0);
    CHECK(test_manifest_metadata_roundtrip() == 0);
    CHECK(test_engine_merge_exact() == 0);
    CHECK(test_out_in_lora_merge_exact() == 0);
    CHECK(test_flush_and_patch_lifetimes() == 0);
    CHECK(test_quantized_base_rejected() == 0);
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("adapter runtime tests passed");
    return 0;
}
