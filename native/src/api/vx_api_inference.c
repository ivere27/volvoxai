/* VxInferenceService — the Runtime/Model/CompiledModel/Context/Result cycle.
 *
 * Handlers return 0 and carry domain outcomes inside response.report. A
 * non-zero return is reserved for failures with no response to carry, which
 * the generated dispatch turns into a transport error.
 */
#include "vx_api_convert.h"
#include "vx_api_handles.h"
#include "public_api_internal.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"
#include "vx_api_buffer.h"

#include <stdlib.h>
#include <string.h>

/* The browser path transport must validate handle state and request metadata
 * before it fetches caller-named files. portable_control_wasm.c brackets one
 * synchronous generated-dispatch call with this mode. It is intentionally
 * unavailable to threaded/native compositions. */
#if defined(__wasm__)
static int vx_api_wasm_path_preflight_active;

static int vx_api_wasm_path_preflight_begin(void) {
    if (vx_api_wasm_path_preflight_active) return 0;
    vx_api_wasm_path_preflight_active = 1;
    return 1;
}

static void vx_api_wasm_path_preflight_end(void) {
    vx_api_wasm_path_preflight_active = 0;
}

static int vx_api_wasm_path_preflight_is_active(void) {
    return vx_api_wasm_path_preflight_active;
}
#endif

/* ---------------------------------------------------------------------------
 * Handle plumbing
 * ------------------------------------------------------------------------ */

static void vx_api_retain_runtime(void* p) { vx_runtime_retain((VxRuntime*)p); }
static void vx_api_release_runtime(void* p) { vx_runtime_release((VxRuntime*)p); }
static void vx_api_retain_model(void* p) { vx_model_retain((VxModel*)p); }
static void vx_api_release_model(void* p) { vx_model_release((VxModel*)p); }
static void vx_api_retain_compiled(void* p) {
    vx_compiled_model_retain((VxCompiledModel*)p);
}
static void vx_api_release_compiled(void* p) {
    vx_compiled_model_release((VxCompiledModel*)p);
}
static void vx_api_retain_context(void* p) {
    vx_execution_context_retain((VxExecutionContext*)p);
}
static void vx_api_release_context(void* p) {
    vx_execution_context_release((VxExecutionContext*)p);
}
static void vx_api_retain_result(void* p) { vx_result_retain((VxResult*)p); }
static void vx_api_release_result(void* p) { vx_result_release((VxResult*)p); }

static void vx_api_runtime_destroyed(uint64_t runtime_id, void* context) {
    (void)context;
    vx_api_capture_forget(runtime_id);
}

static void vx_api_apply_public_lineage(
    VolvoxaiV1OperationReport* report,
    const VxApiHandleLineage* lineage) {
    if (!lineage) return;
    vx_api_report_set_public_lineage(
        report, lineage->runtime_id, lineage->model_id,
        lineage->compiled_model_id, lineage->context_id);
}

#if defined(__wasm__)
/* A fetch-ready preflight response is intentionally synthetic: projecting the
 * native VxReport would attach memory-capture evidence and consume a capture
 * sequence for a response the host discards before the real operation. */
static int vx_api_wasm_path_preflight_ok(
    const SynurangLiteAllocator* allocator,
    VolvoxaiV1OperationReport** slot,
    VxStage stage,
    const char* message,
    const VxApiHandleLineage* public_lineage) {
    VolvoxaiV1Lineage* lineage;

    if (!vx_api_report_fail(allocator, slot, VX_STATUS_OK, stage,
                            VX_CODE_NONE, message))
        return 0;
    lineage = (VolvoxaiV1Lineage*)allocator->allocate(
        allocator->context, sizeof(*lineage));
    if (!lineage) return 0;
    volvoxai_v1_lineage_init_with_allocator(lineage, allocator);
    (*slot)->field_lineage = lineage;
    vx_api_apply_public_lineage(*slot, public_lineage);
    return 1;
}
#endif

static void vx_api_set_lineage_handle(VxApiHandleLineage* lineage,
                                      VxApiHandleKind kind,
                                      int64_t id) {
    if (!lineage || id <= 0) return;
    switch (kind) {
        case VX_API_HANDLE_RUNTIME:
            lineage->runtime_id = (uint64_t)id;
            break;
        case VX_API_HANDLE_MODEL:
            lineage->model_id = (uint64_t)id;
            break;
        case VX_API_HANDLE_COMPILED_MODEL:
            lineage->compiled_model_id = (uint64_t)id;
            break;
        case VX_API_HANDLE_CONTEXT:
            lineage->context_id = (uint64_t)id;
            break;
        default:
            break;
    }
}

/* Publishes a scheduler-produced result into the same registry the inference
 * service uses, so a caller releases it through ReleaseResult. */
int64_t vx_api_publish_scheduler_result(VxApiRegistry* user_data,
    VxResult* result,
    const VxApiHandleLineage* lineage);

int64_t vx_api_publish_scheduler_result(VxApiRegistry* user_data,
    VxResult* result,
    const VxApiHandleLineage* lineage) {
    int64_t id = vx_api_handle_insert_with_lineage(user_data,
        VX_API_HANDLE_RESULT, result, vx_api_retain_result,
        vx_api_release_result, lineage);
    if (!id) vx_result_release(result);
    return id;
}

/* Publishes a freshly created engine handle under a new id, releasing the
 * engine handle when the registry cannot record it. */
static int64_t vx_api_publish(VxApiRegistry* user_data, VxApiHandleKind kind,
                              void* pointer,
                              void (*retain)(void*),
                              void (*release)(void*),
                              const VxApiHandleLineage* lineage) {
    if (kind == VX_API_HANDLE_RUNTIME &&
        !vx_api_registry_track_runtime(user_data, pointer)) {
        release(pointer);
        return 0;
    }
    int64_t id = vx_api_handle_insert_with_lineage(user_data,
        kind, pointer, retain, release, lineage);
    if (!id && pointer && release) release(pointer);
    return id;
}

/* ---------------------------------------------------------------------------
 * Input binding
 * ------------------------------------------------------------------------ */

/* Materializes one engine binding array from a repeated Tensor field. */
static VxStatus vx_api_bindings(VxApiScratch* scratch,
                                const VolvoxaiV1Tensor* tensors,
                                size_t count,
                                VxTensorBinding** out, size_t* failed_index) {
    VxTensorBinding* bindings;
    size_t index;

    *out = NULL;
    *failed_index = 0;
    if (count == 0u) return VX_STATUS_OK;
    bindings = (VxTensorBinding*)vx_api_scratch_alloc(scratch,
                                                      sizeof(*bindings) * count);
    if (!bindings) return VX_STATUS_OUT_OF_MEMORY;
    for (index = 0; index < count; index++) {
        VxStatus status = vx_api_binding_from_tensor(scratch, &bindings[index],
                                                     &tensors[index]);
        if (status != VX_STATUS_OK) {
            *failed_index = index;
            return status;
        }
    }
    *out = bindings;
    return VX_STATUS_OK;
}

/* ---------------------------------------------------------------------------
 * Runtime
 * ------------------------------------------------------------------------ */

static int vx_api_create_runtime(const VolvoxaiV1CreateRuntimeRequest* request,
                                 VolvoxaiV1RuntimeHandle* response,
                                 void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxRuntimeOptions options = VX_RUNTIME_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxStatus status;
    VxApiHandleLineage lineage = VX_API_HANDLE_LINEAGE_INIT;

    options.debug = request->field_debug;
    options.cpu_threads = request->field_cpu_threads;
    if (!request->has_execution_mode ||
        request->field_execution_mode == VOLVOXAI_V1_EXECUTION_MODE_DIRECT) {
        options.execution_mode = VX_EXECUTION_MODE_DIRECT;
    } else if (request->field_execution_mode ==
               VOLVOXAI_V1_EXECUTION_MODE_SCHEDULED) {
        options.execution_mode = VX_EXECUTION_MODE_SCHEDULED;
    } else {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_INVALID_ARGUMENT,
                                  VX_STAGE_RUNTIME_CREATE,
                                  VX_CODE_INVALID_ARGUMENT,
                                  "execution_mode is invalid")
                   ? 0 : -1;
    }
    if (request->field_budget) {
        const VolvoxaiV1RuntimeBudget* budget = request->field_budget;
        /* Absent fields keep the engine default rather than becoming zero. */
        if (budget->has_max_scheduled_requests) {
            options.max_scheduled_requests = (size_t)budget->field_max_scheduled_requests;
        }
        if (budget->has_max_scheduled_input_bytes) {
            options.max_scheduled_input_bytes = (size_t)budget->field_max_scheduled_input_bytes;
        }
        if (budget->has_max_batch_delay_milliseconds) {
            options.max_batch_delay_milliseconds = budget->field_max_batch_delay_milliseconds;
        }
        if (budget->has_max_unconsumed_results) {
            options.max_unconsumed_results = (size_t)budget->field_max_unconsumed_results;
        }
        if (budget->has_max_unconsumed_result_bytes) {
            options.max_unconsumed_result_bytes = (size_t)budget->field_max_unconsumed_result_bytes;
        }
    }

    if (request->field_memory_capture) {
        const char* rejection = vx_api_capture_validate(request->field_memory_capture);
        if (rejection) {
            return vx_api_report_fail(allocator, &response->field_report,
                                      VX_STATUS_INVALID_ARGUMENT,
                                      VX_STAGE_RUNTIME_CREATE,
                                      VX_CODE_INVALID_ARGUMENT, rejection)
                       ? 0 : -1;
        }
    }

    status = vx_runtime_create(&options, &runtime, &report);
    if (status == VX_STATUS_OK && request->field_memory_capture &&
        !vx_api_capture_register(report.runtime_id, request->field_memory_capture)) {
        vx_runtime_release(runtime);
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_OUT_OF_MEMORY, VX_STAGE_RUNTIME_CREATE,
                                  VX_CODE_OUT_OF_MEMORY, "memory capture policy allocation failed")
                   ? 0 : -1;
    }
    if (status == VX_STATUS_OK && request->field_memory_capture) {
        vx_runtime_set_destroy_callback(runtime, vx_api_runtime_destroyed, NULL);
    }
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        if (runtime) vx_runtime_release(runtime);
        return -1;
    }
    if (status != VX_STATUS_OK) return 0;

    response->field_runtime_id =
        vx_api_publish(user_data, VX_API_HANDLE_RUNTIME, runtime,
                       vx_api_retain_runtime, vx_api_release_runtime, NULL);
    if (!response->field_runtime_id) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_OUT_OF_MEMORY, VX_STAGE_RUNTIME_CREATE,
                                  VX_CODE_OUT_OF_MEMORY, "handle registry allocation failed")
                   ? 0 : -1;
    }
    vx_api_set_lineage_handle(&lineage, VX_API_HANDLE_RUNTIME,
                              response->field_runtime_id);
    vx_api_apply_public_lineage(response->field_report, &lineage);
    return 0;
}

static int vx_api_release_runtime_handler(const VolvoxaiV1RuntimeRef* request,
                                          VolvoxaiV1OperationReport* response,
                                          void* user_data) {
    (void)user_data;
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_RUNTIME, request->field_runtime_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}

static int vx_api_list_backends(const VolvoxaiV1RuntimeRef* request,
                                VolvoxaiV1BackendList* response,
                                void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    char (*names)[VX_REPORT_BACKEND_CAPACITY] = NULL;
    SynurangLiteBytes* entries;
    size_t count = 0u;
    size_t index;
    int result = 0;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_RUNTIME,
                               request->field_runtime_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_NONE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released runtime")
                   ? 0 : -1;
    }

    status = vx_runtime_internal_backend_names(
        (VxRuntime*)lease.pointer, &names, &count, &report);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        result = -1;
        goto done;
    }
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    if (status != VX_STATUS_OK) goto done;
    entries = (SynurangLiteBytes*)allocator->allocate(allocator->context,
                                                      sizeof(*entries) * count);
    if (!entries) {
        result = -1;
        goto done;
    }
    memset(entries, 0, sizeof(*entries) * count);
    response->field_backends.data = entries;
    response->field_backends.len = count;
    response->field_backends.cap = count;
    for (index = 0; index < count; index++) {
        if (synurang_lite_bytes_assign(allocator, &entries[index], names[index],
                                       strlen(names[index])) != SYNURANG_LITE_OK) {
            result = -1;
            goto done;
        }
    }
done:
    free(names);
    vx_api_handle_lease_release(&lease);
    return result;
}

/* ---------------------------------------------------------------------------
 * Model
 * ------------------------------------------------------------------------ */

static int vx_api_load_model(const VolvoxaiV1LoadModelRequest* request,
                             VolvoxaiV1ModelHandle* response,
                             void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxModelPackageSource package = {0};
    VxSourceBytes* shards = NULL;
    VxBankResidency* residency = NULL;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime;
    VxModel* model = NULL;
    VxStatus status;
    size_t index;
    int result = 0;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_RUNTIME,
                               request->field_runtime_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_MODEL_LOAD,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released runtime")
                   ? 0 : -1;
    }
    runtime = (VxRuntime*)lease.pointer;

    if (request->field_package &&
        (request->field_graph_path.len || request->field_weight_paths.len)) {
        result = vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
            VX_CODE_INVALID_MODEL_SOURCE, "ModelPackage cannot be combined with paths") ? 0 : -1;
        goto done;
    }
    source.graph_path = vx_api_scratch_cstr(&scratch, &request->field_graph_path);
    source.weight_paths = vx_api_scratch_cstr_array(&scratch,
                                                    request->field_weight_paths.data,
                                                    sizeof(SynurangLiteBytes),
                                                    request->field_weight_paths.len);
    source.weight_path_count = request->field_weight_paths.len;

    if (request->field_package) {
        const VolvoxaiV1ModelPackage* input = request->field_package;
        package.graph.data = input->field_graph_document.data;
        package.graph.size = input->field_graph_document.len;
        package.weight_count = input->field_weight_shards.len;
        if (package.weight_count) {
            shards = (VxSourceBytes*)vx_api_scratch_alloc(
                &scratch, package.weight_count * sizeof(*shards));
            if (shards)
                for (index = 0; index < package.weight_count; index++) {
                    shards[index].data = input->field_weight_shards.data[index].data;
                    shards[index].size = input->field_weight_shards.data[index].len;
                }
        }
        package.weights = shards;
    }

    if (request->field_bank_residency.len) {
        residency = (VxBankResidency*)vx_api_scratch_alloc(
            &scratch, sizeof(*residency) * request->field_bank_residency.len);
        if (residency) {
            for (index = 0; index < request->field_bank_residency.len; index++) {
                const VolvoxaiV1BankResidency* entry =
                    &request->field_bank_residency.data[index];
                residency[index] = (VxBankResidency)VX_BANK_RESIDENCY_INIT;
                residency[index].bank = vx_api_scratch_cstr(&scratch, &entry->field_bank);
                residency[index].slots = entry->field_slots.data;
                residency[index].slot_count = entry->field_slots.len;
            }
            source.bank_residency = residency;
            source.bank_residency_count = request->field_bank_residency.len;
        }
    }

    if (vx_api_scratch_failed(&scratch)) {
        result = vx_api_report_fail(allocator, &response->field_report,
                                    VX_STATUS_OUT_OF_MEMORY, VX_STAGE_MODEL_LOAD,
                                    VX_CODE_OUT_OF_MEMORY,
                                    "model source allocation failed")
                     ? 0 : -1;
        goto done;
    }

    if (request->field_package) {
        int preflight_only = 0;
#if defined(__wasm__)
        preflight_only = vx_api_wasm_path_preflight_is_active();
#endif
        status = vx_runtime_internal_load_model_package(
            runtime, &source, &package, &model, &report, preflight_only);
    } else {
#if defined(__wasm__)
        if (vx_api_wasm_path_preflight_is_active())
            status = vx_runtime_internal_preflight_load_model(
                runtime, &source, &report);
        else
#endif
            status = vx_runtime_load_model(runtime, &source, &model, &report);
    }
#if defined(__wasm__)
    if (vx_api_wasm_path_preflight_is_active() && status == VX_STATUS_OK) {
        result = vx_api_wasm_path_preflight_ok(
                     allocator, &response->field_report,
                     VX_STAGE_MODEL_LOAD, "model source preflight accepted",
                     &lease.lineage)
                     ? 0 : -1;
        goto done;
    }
#endif
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        result = -1;
        goto done;
    }
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    if (status != VX_STATUS_OK) goto done;

    response->field_model_id =
        vx_api_publish(user_data, VX_API_HANDLE_MODEL, model,
                       vx_api_retain_model, vx_api_release_model,
                       &lease.lineage);
    model = NULL; /* publish transferred or released the created reference */
    if (!response->field_model_id) {
        result = vx_api_report_fail(allocator, &response->field_report,
                                    VX_STATUS_OUT_OF_MEMORY, VX_STAGE_MODEL_LOAD,
                                     VX_CODE_OUT_OF_MEMORY, "handle registry allocation failed")
                     ? 0 : -1;
    } else {
        vx_api_set_lineage_handle(&lease.lineage, VX_API_HANDLE_MODEL,
                                  response->field_model_id);
        vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    }
done:
    vx_api_scratch_release(&scratch);
    if (model) vx_model_release(model);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_get_model_info(const VolvoxaiV1ModelRef* request,
                                  VolvoxaiV1ModelInfo* response,
                                  void* user_data) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxRevisionInfo revision = VX_REVISION_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model;
    int result = -1;
    (void)user_data;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_MODEL, request->field_model_id, &lease))
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_HANDLE_DISPOSED, VX_STAGE_MODEL_LOAD,
            VX_CODE_HANDLE_DISPOSED, "unknown or released model") ? 0 : -1;
    model = (VxModel*)lease.pointer;
    for (int output = 0; output < 2; output++) {
        size_t count = output ? vx_model_internal_output_count(model)
                              : vx_model_internal_input_count(model);
        VolvoxaiV1TensorSpec* specs = count ? (VolvoxaiV1TensorSpec*)allocator->allocate(
            allocator->context, count * sizeof(*specs)) : NULL;
        if (count && !specs) goto done;
        for (size_t index = 0; index < count; index++)
            volvoxai_v1_tensor_spec_init_with_allocator(&specs[index], allocator);
        if (output) {
            response->field_outputs.data = specs;
            response->field_outputs.len = response->field_outputs.cap = count;
        } else {
            response->field_inputs.data = specs;
            response->field_inputs.len = response->field_inputs.cap = count;
        }
        for (size_t index = 0; index < count; index++) {
            VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
            VxStatus status = output ? vx_model_internal_output_spec(model, index, &spec)
                                     : vx_model_internal_input_spec(model, index, &spec);
            if (status != VX_STATUS_OK ||
                !vx_api_tensor_spec_from_native(allocator, &specs[index], &spec)) goto done;
        }
    }
    response->field_model_id = request->field_model_id;
    if (vx_model_revision_info(model, &revision, &report) != VX_STATUS_OK ||
        !vx_api_report_attach(allocator, &response->field_report, &report)) goto done;
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    result = 0;
done:
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_get_model_revision(const VolvoxaiV1ModelRef* request,
                                     VolvoxaiV1RevisionInfo* response,
                                     void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxRevisionInfo info = VX_REVISION_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxModel* model;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_MODEL,
                               request->field_model_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_NONE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released model")
                   ? 0 : -1;
    }
    model = (VxModel*)lease.pointer;
    if (vx_model_revision_info(model, &info, &report) == VX_STATUS_OK) {
        response->field_graph_id = info.graph_id;
        response->field_graph_revision = info.graph_revision;
        response->field_weight_id = info.weight_id;
        response->field_weight_revision = info.weight_revision;
        response->field_adapter_id = info.adapter_id;
        response->field_adapter_revision = info.adapter_revision;
    }
    result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
    if (result == 0) {
        vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    }
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_publish_adapter(const VolvoxaiV1PublishAdapterRequest* request,
                                  VolvoxaiV1AdapterRevision* response,
                                  void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxAdapterSource source = VX_ADAPTER_SOURCE_INIT;
    VxAdapterRevision published = VX_ADAPTER_REVISION_INIT;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxModel* model;
    VxStatus status;
    int result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_MODEL,
                               request->field_model_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_ADAPTER,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released model")
                   ? 0 : -1;
    }
    model = (VxModel*)lease.pointer;
    source.adapter_name = vx_api_scratch_cstr(&scratch, &request->field_adapter_name);
    source.package_path = vx_api_scratch_cstr(&scratch, &request->field_package_path);
    source.version_name = vx_api_scratch_cstr(&scratch, &request->field_version_name);

    if (vx_api_scratch_failed(&scratch)) {
        result = vx_api_report_fail(
                     allocator, &response->field_report,
                     VX_STATUS_OUT_OF_MEMORY, VX_STAGE_ADAPTER,
                     VX_CODE_OUT_OF_MEMORY, "adapter source allocation failed")
                     ? 0 : -1;
        vx_api_apply_public_lineage(response->field_report, &lease.lineage);
        goto done;
    }
#if defined(__wasm__)
    if (vx_api_wasm_path_preflight_is_active())
        status = vx_model_internal_preflight_publish_adapter(
            model, &source, &report);
    else
#endif
        status = vx_model_publish_adapter(model, &source, &published, &report);
    if (status == VX_STATUS_OK
#if defined(__wasm__)
        && !vx_api_wasm_path_preflight_is_active()
#endif
    ) {
        response->field_adapter_id = published.adapter_id;
        response->field_adapter_revision = published.adapter_revision;
    }
#if defined(__wasm__)
    if (vx_api_wasm_path_preflight_is_active() && status == VX_STATUS_OK) {
        result = vx_api_wasm_path_preflight_ok(
                     allocator, &response->field_report,
                     VX_STAGE_ADAPTER, "adapter source preflight accepted",
                     &lease.lineage)
                     ? 0 : -1;
        goto done;
    }
#endif
    result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
    if (result == 0) {
        vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    }
done:
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_release_model_handler(const VolvoxaiV1ModelRef* request,
                                        VolvoxaiV1OperationReport* response,
                                        void* user_data) {
    (void)user_data;
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_MODEL, request->field_model_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}

/* ---------------------------------------------------------------------------
 * Compilation
 * ------------------------------------------------------------------------ */

static VxBackendPolicy vx_api_compile_policy(
    const VolvoxaiV1CompileModelRequest* request,
    VxApiScratch* scratch) {
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    if (request->field_policy) {
        const VolvoxaiV1BackendPolicy* source = request->field_policy;
        policy.mode = (VxBackendPolicyMode)source->field_mode;
        policy.operator_fallback = (VxOperatorFallback)source->field_operator_fallback;
        policy.backends = vx_api_scratch_cstr_array(scratch,
                                                    source->field_backends.data,
                                                    sizeof(SynurangLiteBytes),
                                                    source->field_backends.len);
        policy.backend_count = source->field_backends.len;
    }
    return policy;
}

static int vx_api_compile_model(const VolvoxaiV1CompileModelRequest* request,
                                VolvoxaiV1CompiledModelHandle* response,
                                void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxReport report = VX_REPORT_INIT;
    VxCompiledModel* compiled = NULL;
    VxStatus status;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxModel* model;
    VxCompiledDomainAttestationView domain;
    int result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_MODEL,
                               request->field_model_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_COMPILE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released model")
                   ? 0 : -1;
    }
    model = (VxModel*)lease.pointer;
    policy = vx_api_compile_policy(request, &scratch);

    status = vx_model_compile(model, &policy, &compiled, &report);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        result = -1;
        goto done;
    }
    if (status == VX_STATUS_OK &&
        vx_compiled_model_internal_domain_attestation(compiled, &domain) ==
            VX_STATUS_OK &&
        !vx_api_capture_attach_compiled_domain(response->field_report,
                                               &report, &domain)) {
        result = -1;
        goto done;
    }
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    if (status != VX_STATUS_OK) goto done;

    response->field_compiled_model_id =
        vx_api_publish(user_data, VX_API_HANDLE_COMPILED_MODEL, compiled,
                       vx_api_retain_compiled, vx_api_release_compiled,
                       &lease.lineage);
    compiled = NULL; /* publish transferred or released the created reference */
    if (!response->field_compiled_model_id) {
        result = vx_api_report_fail(allocator, &response->field_report,
                                    VX_STATUS_OUT_OF_MEMORY, VX_STAGE_COMPILE,
                                     VX_CODE_OUT_OF_MEMORY, "handle registry allocation failed")
                     ? 0 : -1;
    } else {
        vx_api_set_lineage_handle(&lease.lineage,
                                  VX_API_HANDLE_COMPILED_MODEL,
                                  response->field_compiled_model_id);
        vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    }
done:
    vx_api_scratch_release(&scratch);
    if (compiled) vx_compiled_model_release(compiled);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_release_compiled_handler(const VolvoxaiV1CompiledModelRef* request,
                                           VolvoxaiV1OperationReport* response,
                                           void* user_data) {
    (void)user_data;
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_COMPILED_MODEL,
                               request->field_compiled_model_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}

/* ---------------------------------------------------------------------------
 * Execution
 * ------------------------------------------------------------------------ */

/* Publishes a produced VxResult under a new id and fills the handle message. */
static int vx_api_publish_result(VxApiRegistry* user_data, const SynurangLiteAllocator* allocator,
                                 VolvoxaiV1ExecutionResultHandle* response,
                                 VxResult* result,
                                 const VxReport* report,
                                 VxStatus status,
                                 const VxApiHandleLineage* lineage) {
    if (!vx_api_report_attach(allocator, &response->field_report, report)) {
        if (result) vx_result_release(result);
        return -1;
    }
    vx_api_apply_public_lineage(response->field_report, lineage);
    if (status != VX_STATUS_OK || !result) {
        if (result) vx_result_release(result);
        return 0;
    }

    response->field_execution_id = vx_result_execution_id(result);
    response->field_state = (int)vx_result_state(result);
    response->field_result_id =
        vx_api_publish(user_data, VX_API_HANDLE_RESULT, result,
                       vx_api_retain_result, vx_api_release_result, lineage);
    if (!response->field_result_id) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_OUT_OF_MEMORY, VX_STAGE_EXECUTE,
                                  VX_CODE_OUT_OF_MEMORY, "handle registry allocation failed")
                   ? 0 : -1;
    }
    return 0;
}

static int vx_api_run(const VolvoxaiV1RunRequest* request,
                      VolvoxaiV1ExecutionResultHandle* response,
                      void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxTensorBinding* bindings = NULL;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status;
    VxRuntime* runtime;
    VxCompiledModel* compiled;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    int api_result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_COMPILED_MODEL,
                               request->field_compiled_model_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_EXECUTE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released compiled model")
                   ? 0 : -1;
    }
    compiled = (VxCompiledModel*)lease.pointer;
    runtime = vx_compiled_model_runtime(compiled);
    if (!runtime) {
        vx_api_handle_lease_release(&lease);
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_EXECUTE,
                                  VX_CODE_HANDLE_DISPOSED, "compiled model has no runtime")
                   ? 0 : -1;
    }

    size_t failed_index = 0;
    status = vx_api_bindings(&scratch, request->field_inputs.data,
                             request->field_inputs.len, &bindings, &failed_index);
    if (status != VX_STATUS_OK) {
        vx_api_scratch_release(&scratch);
        api_result = vx_api_report_binding_fail(
                         allocator, &response->field_report, status,
                         VX_STAGE_INPUT, "input tensor could not be bound",
                         request->field_inputs.len ? &request->field_inputs.data[failed_index] : NULL, failed_index)
                         ? 0 : -1;
        vx_api_handle_lease_release(&lease);
        return api_result;
    }

    status = vx_runtime_run(runtime, compiled, bindings, request->field_inputs.len,
                            &result, &report);
    vx_api_scratch_release(&scratch);
    api_result = vx_api_publish_result(user_data, allocator, response, result, &report, status,
                                       &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return api_result;
}

static int vx_api_create_execution_context(
    const VolvoxaiV1CreateExecutionContextRequest* request,
    VolvoxaiV1ExecutionContextHandle* response,
    void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxReport report = VX_REPORT_INIT;
    VxExecutionContext* context = NULL;
    VxStatus status;
    size_t count;
    size_t index;
    VolvoxaiV1TensorSpec* specs;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxCompiledModel* compiled;
    int result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_COMPILED_MODEL,
                               request->field_compiled_model_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_CONTEXT_CREATE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released compiled model")
                   ? 0 : -1;
    }
    compiled = (VxCompiledModel*)lease.pointer;
    options.decode_row_mode = (VxDecodeRowMode)request->field_decode_row_mode;
    options.require_incremental = request->field_require_incremental;
    options.decode_lanes = request->has_decode_lanes ? request->field_decode_lanes : 1;

    options.decode_input_count = request->field_decode_inputs.len;
    options.decode_inputs = vx_api_scratch_cstr_array(&scratch, request->field_decode_inputs.data,
        sizeof(*request->field_decode_inputs.data), request->field_decode_inputs.len);
    if (vx_api_scratch_failed(&scratch)) { result = -1; goto done; }
    status = vx_compiled_model_create_context(compiled, &options, &context, &report);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        result = -1;
        goto done;
    }
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    if (status != VX_STATUS_OK) goto done;

    /* The declared inputs travel with the handle so a caller can bind without
     * a second round trip. */
    count = vx_execution_context_input_count(context);
    if (count != 0u) {
        specs = (VolvoxaiV1TensorSpec*)allocator->allocate(allocator->context,
                                                           sizeof(*specs) * count);
        if (!specs) {
            result = -1;
            goto done;
        }
        response->field_inputs.data = specs;
        response->field_inputs.len = count;
        response->field_inputs.cap = count;
        for (index = 0; index < count; index++) {
            VxTensorSpec native = VX_TENSOR_SPEC_INIT;
            VxReport ignored = VX_REPORT_INIT;
            volvoxai_v1_tensor_spec_init_with_allocator(&specs[index], allocator);
            if (vx_execution_context_input_spec(context, index, &native, &ignored) !=
                VX_STATUS_OK) {
                continue;
            }
            if (!vx_api_tensor_spec_from_native(allocator, &specs[index], &native)) {
                result = -1;
                goto done;
            }
        }
    }

    response->field_context_id =
        vx_api_publish(user_data, VX_API_HANDLE_CONTEXT, context,
                       vx_api_retain_context, vx_api_release_context,
                       &lease.lineage);
    context = NULL; /* publish transferred or released the created reference */
    if (!response->field_context_id) {
        result = vx_api_report_fail(allocator, &response->field_report,
                                    VX_STATUS_OUT_OF_MEMORY, VX_STAGE_CONTEXT_CREATE,
                                     VX_CODE_OUT_OF_MEMORY, "handle registry allocation failed")
                     ? 0 : -1;
    } else {
        vx_api_set_lineage_handle(&lease.lineage, VX_API_HANDLE_CONTEXT,
                                  response->field_context_id);
        vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    }
done:
    vx_api_scratch_release(&scratch);
    if (context) vx_execution_context_release(context);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_release_context_handler(const VolvoxaiV1ExecutionContextRef* request,
                                          VolvoxaiV1OperationReport* response,
                                          void* user_data) {
    (void)user_data;
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_CONTEXT, request->field_context_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}

static int vx_api_get_input_affine_quantization(
    const VolvoxaiV1GetInputAffineQuantizationRequest* request,
    VolvoxaiV1AffineQuantization* response,
    void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxAffineQuantization quantization = VX_AFFINE_QUANTIZATION_INIT;
    VxReport report = VX_REPORT_INIT;
    const char* name;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxExecutionContext* context;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT,
                               request->field_context_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_NONE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released context")
                   ? 0 : -1;
    }
    context = (VxExecutionContext*)lease.pointer;
    name = vx_api_scratch_cstr(&scratch, &request->field_name);
    if (vx_execution_context_input_affine_quantization(context, name, &quantization,
                                                       &report) == VX_STATUS_OK) {
        response->field_defined = quantization.defined != 0;
        response->field_scale = quantization.scale;
        response->field_zero_point = quantization.zero_point;
    }
    vx_api_scratch_release(&scratch);
    result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
    if (result == 0) {
        vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    }
    vx_api_handle_lease_release(&lease);
    return result;
}

/* Shared body for the four context-scoped execution entry points. */
static int vx_api_context_execute(VxApiRegistry* user_data, const SynurangLiteAllocator* allocator,
                                  VolvoxaiV1ExecutionResultHandle* response,
                                  int64_t context_id,
                                  const VolvoxaiV1Tensor* tensors,
                                  size_t tensor_count,
                                  int mode,
                                  int32_t position,
                                  const int32_t* positions, size_t lane_count) {
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxTensorBinding* bindings = NULL;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxExecutionContext* context;
    int api_result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT, context_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_EXECUTE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released context")
                   ? 0 : -1;
    }
    context = (VxExecutionContext*)lease.pointer;

    size_t failed_index = 0;
    status = vx_api_bindings(&scratch, tensors, tensor_count, &bindings, &failed_index);
    if (status != VX_STATUS_OK) {
        vx_api_scratch_release(&scratch);
        api_result = vx_api_report_binding_fail(
                         allocator, &response->field_report, status,
                         VX_STAGE_INPUT, "input tensor could not be bound",
                         tensor_count ? &tensors[failed_index] : NULL, failed_index)
                         ? 0 : -1;
        vx_api_handle_lease_release(&lease);
        return api_result;
    }

    switch (mode) {
        case 1:
            status = vx_execution_context_execute_prefix(context, position, bindings,
                                                         tensor_count, &result, &report);
            break;
        case 2:
            status = vx_execution_context_decode_lanes(
                context, 1, position, positions, lane_count, bindings, tensor_count, &result, &report);
            break;
        case 3:
            status = vx_execution_context_decode_lanes(context, 0, position, positions, lane_count, bindings,
                                                      tensor_count, &result, &report);
            break;
        case 4:
            status = vx_execution_context_decode_step(context, -1, bindings,
                                                      tensor_count, &result, &report);
            break;
        default:
            status = vx_execution_context_execute(context, bindings, tensor_count,
                                                  &result, &report);
            break;
    }
    vx_api_scratch_release(&scratch);
    api_result = vx_api_publish_result(user_data, allocator, response, result, &report, status,
                                       &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return api_result;
}

static int vx_api_execute(const VolvoxaiV1ExecuteRequest* request,
                          VolvoxaiV1ExecutionResultHandle* response,
                          void* user_data) {
    (void)user_data;
    return vx_api_context_execute(user_data, response->_allocator, response,
                                  request->field_context_id,
                                  request->field_inputs.data,
                                  request->field_inputs.len, 0, 0, NULL, 0);
}

static int vx_api_execute_prefix(const VolvoxaiV1ExecutePrefixRequest* request,
                                 VolvoxaiV1ExecutionResultHandle* response,
                                 void* user_data) {
    (void)user_data;
    return vx_api_context_execute(user_data, response->_allocator, response,
                                  request->field_context_id,
                                  request->field_inputs.data,
                                  request->field_inputs.len, 1,
                                  request->field_row_count, NULL, 0);
}

static int vx_api_decode_cursor_fail(const SynurangLiteAllocator* allocator,
    VolvoxaiV1ExecutionResultHandle* response) {
    return vx_api_report_fail(allocator, &response->field_report, VX_STATUS_INVALID_ARGUMENT,
        VX_STAGE_DECODE, VX_CODE_INVALID_DECODE_POSITION, "decode cursor must contain a valid scalar or one action per lane") ? 0 : -1;
}

static int vx_api_decode_prefill(const VolvoxaiV1DecodePrefillRequest* request,
                                 VolvoxaiV1ExecutionResultHandle* response,
                                 void* user_data) {
    (void)user_data;
    const int32_t* positions = NULL;
    size_t count = 0;
    int32_t position = -1;
    if (request->which_cursor == 3) {
        position = request->field_position;
        if (position < 0) return vx_api_decode_cursor_fail(response->_allocator, response);
    } else if (request->which_cursor == 4) {
        if (!request->field_lane_positions || !request->field_lane_positions->field_positions.len)
            return vx_api_decode_cursor_fail(response->_allocator, response);
        positions = request->field_lane_positions->field_positions.data;
        count = request->field_lane_positions->field_positions.len;
    }
    return vx_api_context_execute(user_data, response->_allocator, response, request->field_context_id,
        request->field_inputs.data, request->field_inputs.len, 2, position, positions, count);
}

static int vx_api_decode_step(const VolvoxaiV1DecodeStepRequest* request,
                              VolvoxaiV1ExecutionResultHandle* response,
                              void* user_data) {
    (void)user_data;
    int32_t position = -1;
    int32_t* positions = NULL;
    size_t count = 0;
    if (request->which_cursor == 2) {
        position = request->field_position;
        if (position < 0) return vx_api_decode_cursor_fail(response->_allocator, response);
    } else if (request->which_cursor == 4) {
        if (!request->field_lane_actions || !request->field_lane_actions->field_lanes.len ||
            request->field_lane_actions->field_lanes.len > SIZE_MAX / sizeof(int32_t))
            return vx_api_decode_cursor_fail(response->_allocator, response);
        count = request->field_lane_actions->field_lanes.len;
        positions = malloc(count * sizeof(int32_t));
        if (!positions) return -1;
        for (size_t lane = 0; lane < count; lane++) {
            const VolvoxaiV1DecodeLaneAction* action = &request->field_lane_actions->field_lanes.data[lane];
            if (action->which_action == 1 && action->field_position >= 0) positions[lane] = action->field_position;
            else if (action->which_action == 2 && action->field_idle) positions[lane] = -2;
            else if (action->which_action == 3 && action->field_parked) positions[lane] = -1;
            else { free(positions); return vx_api_decode_cursor_fail(response->_allocator, response); }
        }
    } else if (request->which_cursor == 5 && !request->field_dependency_update) {
        return vx_api_decode_cursor_fail(response->_allocator, response);
    }
    int result = vx_api_context_execute(user_data, response->_allocator, response, request->field_context_id,
        request->field_inputs.data, request->field_inputs.len,
        request->which_cursor == 5 ? 4 : 3, position, positions, count);
    free(positions);
    return result;
}

static int vx_api_decode_generate(const VolvoxaiV1DecodeGenerateRequest* request,
    VolvoxaiV1ExecutionResultHandle* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    const SynurangLiteAllocator* allocator = response->_allocator;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT, request->field_context_id, &lease))
        return vx_api_report_fail(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED,
            VX_STAGE_DECODE, VX_CODE_HANDLE_DISPOSED, "unknown or released context") ? 0 : -1;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxDecodeFeedbackOptions options = {
        .token_input = vx_api_scratch_cstr(&scratch, &request->field_token_input),
        .keep_input = vx_api_scratch_cstr(&scratch, &request->field_keep_input),
        .token_output = vx_api_scratch_cstr(&scratch, &request->field_token_output),
        .token_count = request->field_token_count,
        .has_cache_generation = request->has_cache_generation,
        .cache_generation = request->field_cache_generation,
    };
    VxResult* result = NULL;
    VxReport report = VX_REPORT_INIT;
    VxStatus status = vx_execution_context_decode_generate(lease.pointer, &options, &result, &report);
    int api_result = vx_api_publish_result(user_data, allocator, response, result, &report, status, &lease.lineage);
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return api_result;
}

static int vx_api_reset_decode(const VolvoxaiV1ExecutionContextRef* request,
                               VolvoxaiV1OperationReport* response,
                               void* user_data) {
    (void)user_data;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxExecutionContext* context;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT,
                               request->field_context_id, &lease)) {
        response->field_status = VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED;
        response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_DECODE;
        return 0;
    }
    context = (VxExecutionContext*)lease.pointer;
    (void)vx_execution_context_decode_reset(context, &report);
    result = vx_api_report_from_native(response, &report) ? 0 : -1;
    if (result == 0) vx_api_apply_public_lineage(response, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return result;
}

static VxStatus vx_api_write_decode_state(const VxDecodeStateView* state, void* user) {
    VolvoxaiV1DecodeContextState* response = user;
    const SynurangLiteAllocator* allocator = response->_allocator;
    response->field_active_lengths.data = allocator->allocate(allocator->context,
        (size_t)state->lanes * sizeof(uint32_t));
    if (!response->field_active_lengths.data) return VX_STATUS_OUT_OF_MEMORY;
    response->field_active_lengths.len = response->field_active_lengths.cap = state->lanes;
    response->field_parked.data = allocator->allocate(allocator->context,
        (size_t)state->lanes * sizeof(int));
    if (!response->field_parked.data) return VX_STATUS_OUT_OF_MEMORY;
    response->field_parked.len = response->field_parked.cap = state->lanes;
    for (uint32_t lane = 0; lane < state->lanes; lane++) {
        response->field_active_lengths.data[lane] = state->active_lengths[lane];
        response->field_parked.data[lane] = state->parked[lane] != 0;
    }
    response->field_lanes = state->lanes;
    response->field_prefilled = state->prefilled;
    response->field_mode = state->mode;
    response->field_cache_generation = state->cache_generation;
    return VX_STATUS_OK;
}

static int vx_api_get_decode_state(const VolvoxaiV1ExecutionContextRef* request,
    VolvoxaiV1DecodeContextState* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxReport report = VX_REPORT_INIT;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT, request->field_context_id, &lease))
        return vx_api_report_fail(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED,
            VX_STAGE_DECODE, VX_CODE_HANDLE_DISPOSED, "unknown or released context") ? 0 : -1;
    VxStatus status = vx_execution_context_inspect_decode(lease.pointer, vx_api_write_decode_state, response, &report);
    if (status != VX_STATUS_OK) volvoxai_v1_decode_context_state_free(response);
    int result = vx_api_report_attach(response->_allocator, &response->field_report, &report) ? 0 : -1;
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return result;
}

#include "vx_api_decode_cache.inc"

static int vx_api_select_adapter(const VolvoxaiV1SelectAdapterRequest* request,
                                 VolvoxaiV1OperationReport* response,
                                 void* user_data) {
    (void)user_data;
    VxAdapterRevision revision = VX_ADAPTER_REVISION_INIT;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxExecutionContext* context;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT,
                               request->field_context_id, &lease)) {
        response->field_status = VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED;
        response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_ADAPTER;
        return 0;
    }
    context = (VxExecutionContext*)lease.pointer;
    if (request->field_revision) {
        revision.adapter_id = request->field_revision->field_adapter_id;
        revision.adapter_revision =
            request->field_revision->field_adapter_revision;
    }
    (void)vx_execution_context_select_adapter(
        context, request->field_revision ? &revision : NULL, &report);
    result = vx_api_report_from_native(response, &report) ? 0 : -1;
    if (result == 0) vx_api_apply_public_lineage(response, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_rebind_adapter(const VolvoxaiV1ExecutionContextRef* request,
                                 VolvoxaiV1OperationReport* response,
                                 void* user_data) {
    (void)user_data;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxExecutionContext* context;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_CONTEXT,
                               request->field_context_id, &lease)) {
        response->field_status = VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED;
        response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_ADAPTER;
        return 0;
    }
    context = (VxExecutionContext*)lease.pointer;
    (void)vx_execution_context_rebind_adapter(context, &report);
    result = vx_api_report_from_native(response, &report) ? 0 : -1;
    if (result == 0) vx_api_apply_public_lineage(response, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return result;
}

/* ---------------------------------------------------------------------------
 * Results
 * ------------------------------------------------------------------------ */

static int vx_api_get_result(const VolvoxaiV1ResultRef* request,
                             VolvoxaiV1ResultInfo* response,
                             void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VolvoxaiV1TensorInfo* outputs;
    size_t count;
    size_t index;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxResult* result;
    int api_result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_RESULT,
                               request->field_result_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_RESULT_DISPOSED, VX_STAGE_READBACK,
                                  VX_CODE_RESULT_DISPOSED, "unknown or released result")
                   ? 0 : -1;
    }
    result = (VxResult*)lease.pointer;
    response->field_result_id = request->field_result_id;
    response->field_execution_id = vx_result_execution_id(result);
    VxReport report = VX_REPORT_INIT;
    (void)vx_result_poll(result, &report);
    response->field_state = (int)vx_result_state(result);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        api_result = -1;
        goto done;
    }

    count = vx_result_output_count(result);
    if (count == 0u) goto done;
    outputs = (VolvoxaiV1TensorInfo*)allocator->allocate(allocator->context,
                                                         sizeof(*outputs) * count);
    if (!outputs) {
        api_result = -1;
        goto done;
    }
    response->field_outputs.data = outputs;
    response->field_outputs.len = count;
    response->field_outputs.cap = count;
    for (index = 0; index < count; index++) {
        VxTensorInfo native = VX_TENSOR_INFO_INIT;
        VxReport ignored = VX_REPORT_INIT;
        volvoxai_v1_tensor_info_init_with_allocator(&outputs[index], allocator);
        if (vx_result_output_info(result, index, &native, &ignored) != VX_STATUS_OK) {
            continue;
        }
        if (!vx_api_tensor_info_from_native(allocator, &outputs[index], &native)) {
            api_result = -1;
            goto done;
        }
    }
done:
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return api_result;
}

static int vx_api_read_output(const VolvoxaiV1ReadOutputRequest* request,
                              VolvoxaiV1ReadOutputResponse* response,
                              void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxReport report = VX_REPORT_INIT;
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    VolvoxaiV1Tensor* tensor;
    const char* name;
    size_t required = 0u;
    size_t count;
    size_t index;
    int located = 0;
    VxStatus status;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxResult* result;
    int api_result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_RESULT,
                               request->field_result_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_RESULT_DISPOSED, VX_STAGE_READBACK,
                                  VX_CODE_RESULT_DISPOSED, "unknown or released result")
                   ? 0 : -1;
    }
    result = (VxResult*)lease.pointer;
    name = vx_api_scratch_cstr(&scratch, &request->field_name);
    if (!name) {
        api_result = vx_api_report_fail(allocator, &response->field_report,
                                        VX_STATUS_INVALID_ARGUMENT, VX_STAGE_READBACK,
                                        VX_CODE_INVALID_ARGUMENT, "output name is required")
                         ? 0 : -1;
        goto done;
    }

    /* Query the byte count first so a caller-supplied destination can be
     * validated before any copy begins. */
    status = vx_result_read(result, name, NULL, 0u, &required, &report);
    if (status != VX_STATUS_OK) {
        api_result = vx_api_report_attach(allocator, &response->field_report, &report)
                         ? 0 : -1;
        goto done;
    }
    response->field_required_bytes = required;

    /* Recover the descriptor so the response carries shape and dtype. */
    count = vx_result_output_count(result);
    for (index = 0; index < count; index++) {
        VxReport ignored = VX_REPORT_INIT;
        VxTensorInfo candidate = VX_TENSOR_INFO_INIT;
        if (vx_result_output_info(result, index, &candidate, &ignored) != VX_STATUS_OK) {
            continue;
        }
        if (candidate.name && !strcmp(candidate.name, name)) {
            info = candidate;
            located = 1;
            break;
        }
    }

    tensor = (VolvoxaiV1Tensor*)allocator->allocate(allocator->context, sizeof(*tensor));
    if (!tensor) {
        api_result = -1;
        goto done;
    }
    volvoxai_v1_tensor_init_with_allocator(tensor, allocator);
    response->field_tensor = tensor;
    if (located) {
        size_t axis;
        int64_t* shape;
        if (synurang_lite_bytes_assign(allocator, &tensor->field_name, name,
                                       strlen(name)) != SYNURANG_LITE_OK) {
            api_result = -1;
            goto done;
        }
        tensor->field_dtype = (VolvoxaiV1DataType)info.dtype;
        if (info.rank) {
            shape = (int64_t*)allocator->allocate(allocator->context,
                                                  sizeof(*shape) * info.rank);
            if (!shape) {
                api_result = -1;
                goto done;
            }
            for (axis = 0; axis < info.rank; axis++) shape[axis] = info.shape[axis];
            tensor->field_shape.data = shape;
            tensor->field_shape.len = info.rank;
            tensor->field_shape.cap = info.rank;
        }
    }

    if (request->field_into) {
        /* Copy straight into caller memory and echo the written range, so a
         * decode loop never pays a protobuf copy of the payload. */
        const VolvoxaiV1BorrowedBuffer* into = request->field_into;
        VolvoxaiV1BorrowedBuffer* view;
        void* destination = NULL;
        size_t destination_bytes = 0u;

        status = vx_api_borrowed_resolve(into, &destination,
                                            &destination_bytes);
        if (status != VX_STATUS_OK) {
            api_result = vx_api_report_binding_fail(
                             allocator, &response->field_report, status,
                             VX_STAGE_READBACK,
                             "destination view is malformed", NULL, 0)
                             ? 0 : -1;
            goto done;
        }
        if (destination_bytes < required) {
            api_result = vx_api_report_fail(
                allocator, &response->field_report,
                VX_STATUS_BUFFER_TOO_SMALL, VX_STAGE_READBACK,
                VX_CODE_BUFFER_TOO_SMALL, "destination view is smaller than the snapshot")
                             ? 0 : -1;
            goto done;
        }
        status = vx_result_read(result, name, destination, required, &required, &report);
        if (status == VX_STATUS_OK) {
            view = (VolvoxaiV1BorrowedBuffer*)allocator->allocate(allocator->context,
                                                              sizeof(*view));
            if (!view) {
                api_result = -1;
                goto done;
            }
            volvoxai_v1_borrowed_buffer_init_with_allocator(view, allocator);
            VxNativeBuffer memory;
            status = vx_api_borrowed_native(into, &memory);
            memory.length = required;
            synurang_lite_release(allocator, view);
            if (status != VX_STATUS_OK || !vx_api_borrowed_descriptor(allocator, &tensor->field_borrowed, &memory)) {
                api_result = -1; goto done;
            }
            tensor->which_payload = 7;
        }
    } else if (required) {
        void* buffer = vx_api_scratch_alloc(&scratch, required);
        if (!buffer) {
            api_result = -1;
            goto done;
        }
        status = vx_result_read(result, name, buffer, required, &required, &report);
        if (status == VX_STATUS_OK) {
            if (synurang_lite_bytes_assign(allocator, &tensor->field_inline, buffer,
                                           required) != SYNURANG_LITE_OK) {
                api_result = -1;
                goto done;
            }
            tensor->which_payload = 5; /* Tensor.inline */
        }
    }

    api_result = vx_api_report_attach(allocator, &response->field_report, &report)
                     ? 0 : -1;
done:
    vx_api_scratch_release(&scratch);
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return api_result;
}

static int vx_api_release_result_handler(const VolvoxaiV1ResultRef* request,
                                         VolvoxaiV1OperationReport* response,
                                         void* user_data) {
    (void)user_data;
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_RESULT, request->field_result_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}

/* ------------------------------------------------------------------------ */

#include "vx_api_tensor_interop.inc"

VX_API_UNARY(vx_api_get_tensor_interop_info, VolvoxaiV1ExecutionContextRef, VolvoxaiV1TensorInteropInfo,
    volvoxai_v1_tensor_interop_info, vx_inference_get_tensor_interop_info_respond)
VX_API_UNARY(vx_api_execute_tensors, VolvoxaiV1ExecuteTensorsRequest, VolvoxaiV1TensorBatch,
    volvoxai_v1_tensor_batch, vx_inference_execute_tensors_respond)
VX_API_UNARY(vx_api_create_runtime, VolvoxaiV1CreateRuntimeRequest, VolvoxaiV1RuntimeHandle,
    volvoxai_v1_runtime_handle, vx_inference_create_runtime_respond)
VX_API_UNARY(vx_api_release_runtime_handler, VolvoxaiV1RuntimeRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_release_runtime_respond)
VX_API_UNARY(vx_api_list_backends, VolvoxaiV1RuntimeRef, VolvoxaiV1BackendList,
    volvoxai_v1_backend_list, vx_inference_list_backends_respond)
VX_API_UNARY(vx_api_load_model, VolvoxaiV1LoadModelRequest, VolvoxaiV1ModelHandle,
    volvoxai_v1_model_handle, vx_inference_load_model_respond)
VX_API_UNARY(vx_api_get_model_revision, VolvoxaiV1ModelRef, VolvoxaiV1RevisionInfo,
    volvoxai_v1_revision_info, vx_inference_get_model_revision_respond)
VX_API_UNARY(vx_api_get_model_info, VolvoxaiV1ModelRef, VolvoxaiV1ModelInfo,
    volvoxai_v1_model_info, vx_inference_get_model_info_respond)
VX_API_UNARY(vx_api_publish_adapter, VolvoxaiV1PublishAdapterRequest, VolvoxaiV1AdapterRevision,
    volvoxai_v1_adapter_revision, vx_inference_publish_adapter_respond)
VX_API_UNARY(vx_api_release_model_handler, VolvoxaiV1ModelRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_release_model_respond)
VX_API_UNARY(vx_api_compile_model, VolvoxaiV1CompileModelRequest, VolvoxaiV1CompiledModelHandle,
    volvoxai_v1_compiled_model_handle, vx_inference_compile_model_respond)
VX_API_UNARY(vx_api_release_compiled_handler, VolvoxaiV1CompiledModelRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_release_compiled_model_respond)
VX_API_UNARY(vx_api_run, VolvoxaiV1RunRequest, VolvoxaiV1ExecutionResultHandle,
    volvoxai_v1_execution_result_handle, vx_inference_run_respond)
VX_API_UNARY(vx_api_create_execution_context, VolvoxaiV1CreateExecutionContextRequest, VolvoxaiV1ExecutionContextHandle,
    volvoxai_v1_execution_context_handle, vx_inference_create_execution_context_respond)
VX_API_UNARY(vx_api_release_context_handler, VolvoxaiV1ExecutionContextRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_release_execution_context_respond)
VX_API_UNARY(vx_api_get_input_affine_quantization, VolvoxaiV1GetInputAffineQuantizationRequest, VolvoxaiV1AffineQuantization,
    volvoxai_v1_affine_quantization, vx_inference_get_input_affine_quantization_respond)
VX_API_UNARY(vx_api_execute, VolvoxaiV1ExecuteRequest, VolvoxaiV1ExecutionResultHandle,
    volvoxai_v1_execution_result_handle, vx_inference_execute_respond)
VX_API_UNARY(vx_api_execute_prefix, VolvoxaiV1ExecutePrefixRequest, VolvoxaiV1ExecutionResultHandle,
    volvoxai_v1_execution_result_handle, vx_inference_execute_prefix_respond)
VX_API_UNARY(vx_api_decode_prefill, VolvoxaiV1DecodePrefillRequest, VolvoxaiV1ExecutionResultHandle,
    volvoxai_v1_execution_result_handle, vx_inference_decode_prefill_respond)
VX_API_UNARY(vx_api_decode_step, VolvoxaiV1DecodeStepRequest, VolvoxaiV1ExecutionResultHandle,
    volvoxai_v1_execution_result_handle, vx_inference_decode_step_respond)
VX_API_UNARY(vx_api_decode_generate, VolvoxaiV1DecodeGenerateRequest, VolvoxaiV1ExecutionResultHandle,
    volvoxai_v1_execution_result_handle, vx_inference_decode_generate_respond)
VX_API_UNARY(vx_api_get_decode_state, VolvoxaiV1ExecutionContextRef, VolvoxaiV1DecodeContextState,
    volvoxai_v1_decode_context_state, vx_inference_get_decode_state_respond)
VX_API_UNARY(vx_api_configure_decode_cache, VolvoxaiV1ConfigureDecodeCacheRequest, VolvoxaiV1DecodeCacheState,
    volvoxai_v1_decode_cache_state, vx_inference_configure_decode_cache_respond)
VX_API_UNARY(vx_api_get_decode_cache, VolvoxaiV1ExecutionContextRef, VolvoxaiV1DecodeCacheState,
    volvoxai_v1_decode_cache_state, vx_inference_get_decode_cache_respond)
VX_API_UNARY(vx_api_publish_decode_prefix, VolvoxaiV1PublishDecodePrefixRequest, VolvoxaiV1DecodeCacheState,
    volvoxai_v1_decode_cache_state, vx_inference_publish_decode_prefix_respond)
VX_API_UNARY(vx_api_reuse_decode_prefix, VolvoxaiV1ReuseDecodePrefixRequest, VolvoxaiV1DecodeCacheState,
    volvoxai_v1_decode_cache_state, vx_inference_reuse_decode_prefix_respond)
VX_API_UNARY(vx_api_release_decode_lane, VolvoxaiV1DecodeLaneRef, VolvoxaiV1DecodeCacheState,
    volvoxai_v1_decode_cache_state, vx_inference_release_decode_lane_respond)
VX_API_UNARY(vx_api_evict_decode_prefixes, VolvoxaiV1EvictDecodePrefixesRequest, VolvoxaiV1DecodeCacheState,
    volvoxai_v1_decode_cache_state, vx_inference_evict_decode_prefixes_respond)
VX_API_UNARY(vx_api_reset_decode, VolvoxaiV1ExecutionContextRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_reset_decode_respond)
VX_API_UNARY(vx_api_select_adapter, VolvoxaiV1SelectAdapterRequest, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_select_adapter_respond)
VX_API_UNARY(vx_api_rebind_adapter, VolvoxaiV1ExecutionContextRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_rebind_adapter_respond)
VX_API_UNARY(vx_api_get_result, VolvoxaiV1ResultRef, VolvoxaiV1ResultInfo,
    volvoxai_v1_result_info, vx_inference_get_result_respond)
VX_API_UNARY(vx_api_read_output, VolvoxaiV1ReadOutputRequest, VolvoxaiV1ReadOutputResponse,
    volvoxai_v1_read_output_response, vx_inference_read_output_respond)
VX_API_UNARY(vx_api_release_result_handler, VolvoxaiV1ResultRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_release_result_respond)

int vx_api_install_inference_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxInferenceServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));

    handlers.create_runtime.message = vx_api_create_runtime_call;
    handlers.release_runtime.message = vx_api_release_runtime_handler_call;
    handlers.list_backends.message = vx_api_list_backends_call;

    handlers.load_model.message = vx_api_load_model_call;
    handlers.get_model_revision.message = vx_api_get_model_revision_call;
    handlers.get_model_info.message = vx_api_get_model_info_call;
    handlers.publish_adapter.message = vx_api_publish_adapter_call;
    handlers.release_model.message = vx_api_release_model_handler_call;

    handlers.compile_model.message = vx_api_compile_model_call;
    handlers.release_compiled_model.message = vx_api_release_compiled_handler_call;

    handlers.run.message = vx_api_run_call;

    handlers.create_execution_context.message = vx_api_create_execution_context_call;
    handlers.release_execution_context.message = vx_api_release_context_handler_call;
    handlers.get_input_affine_quantization.message = vx_api_get_input_affine_quantization_call;

    handlers.execute.message = vx_api_execute_call;
    handlers.get_tensor_interop_info.message = vx_api_get_tensor_interop_info_call;
    handlers.execute_tensors.message = vx_api_execute_tensors_call;
    handlers.execute_prefix.message = vx_api_execute_prefix_call;
    handlers.decode_prefill.message = vx_api_decode_prefill_call;
    handlers.decode_step.message = vx_api_decode_step_call;
    handlers.decode_generate.message = vx_api_decode_generate_call;
    handlers.get_decode_state.message = vx_api_get_decode_state_call;
    handlers.configure_decode_cache.message = vx_api_configure_decode_cache_call;
    handlers.get_decode_cache.message = vx_api_get_decode_cache_call;
    handlers.publish_decode_prefix.message = vx_api_publish_decode_prefix_call;
    handlers.reuse_decode_prefix.message = vx_api_reuse_decode_prefix_call;
    handlers.release_decode_lane.message = vx_api_release_decode_lane_call;
    handlers.evict_decode_prefixes.message = vx_api_evict_decode_prefixes_call;
    handlers.reset_decode.message = vx_api_reset_decode_call;

    handlers.select_adapter.message = vx_api_select_adapter_call;
    handlers.rebind_adapter.message = vx_api_rebind_adapter_call;

    handlers.get_result.message = vx_api_get_result_call;
    handlers.read_output.message = vx_api_read_output_call;
    handlers.release_result.message = vx_api_release_result_handler_call;

    return vx_inference_register(instance, &handlers, registry);
}
