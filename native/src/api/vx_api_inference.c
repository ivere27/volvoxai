/* VxInferenceService — the Runtime/Model/CompiledModel/Context execution cycle.
 *
 * Handlers return 0 and carry domain outcomes inside response.report. A
 * non-zero return is reserved for failures with no response to carry, which
 * the generated dispatch turns into a transport error.
 */
#include "vx_api_convert.h"
#include "vx_api_handles.h"
#include "public_api_internal.h"
#include "profiling.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"
#include "vx_api_buffer.h"
#include "vx_api_execution.h"

#include <stdlib.h>
#include <string.h>

static VxStatus vx_api_execution_failure(VxReport*, VxStatus, const char*);

#include "vx_api_execution_contract.inc"

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
static void vx_api_release_result(void* p) { vx_result_release((VxResult*)p); }

static void vx_api_apply_public_lineage(
    VolvoxaiV1OperationReport* report,
    const VxApiHandleLineage* lineage) {
    if (!lineage) return;
    vx_api_report_set_public_lineage(
        report, lineage->runtime_id, lineage->model_id,
        lineage->compiled_model_id, lineage->context_id);
}

#if defined(__wasm__)
/* The host discards this fetch-ready response before the real operation. */
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
    if (id) vx_profiling_set_public_id(pointer, kind, (uint64_t)id);
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
        if (budget->has_max_batch_delay_ns) {
            uint64_t ticks = vx_api_ns_ticks(budget->field_max_batch_delay_ns, 1000000);
            if (ticks > UINT32_MAX)
                return vx_api_report_fail(allocator, &response->field_report,
                    VX_STATUS_INVALID_ARGUMENT, VX_STAGE_RUNTIME_CREATE,
                    VX_CODE_INVALID_ARGUMENT, "max_batch_delay_ns exceeds the scheduler limit") ? 0 : -1;
            options.max_batch_delay_milliseconds = (uint32_t)ticks;
        }
        if (budget->has_max_unconsumed_results) {
            options.max_unconsumed_results = (size_t)budget->field_max_unconsumed_results;
        }
        if (budget->has_max_unconsumed_result_bytes) {
            options.max_unconsumed_result_bytes = (size_t)budget->field_max_unconsumed_result_bytes;
        }
    }

    status = vx_runtime_create(&options, &runtime, &report);
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
    VxSourceBytes graph_document = { request->field_graph_document.data,
                                    request->field_graph_document.len };
    VxModelSourceOptions options = {0};
    VxModelSourceIntegrity graph_integrity = {0};
    VxModelSourceIntegrity* weight_integrity = NULL;
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

    if ((!!request->field_package + !!request->field_graph_path.len +
         !!request->field_graph_document.len) != 1 ||
        (request->field_package &&
         (request->field_weight_paths.len || request->field_source_options))) {
        result = vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
            VX_CODE_INVALID_MODEL_SOURCE,
            "provide one graph source; ModelPackage cannot accompany paths or source_options") ? 0 : -1;
        goto done;
    }
    source.graph_path = vx_api_scratch_cstr(&scratch, &request->field_graph_path);
    source.weight_paths = vx_api_scratch_cstr_array(&scratch,
                                                    request->field_weight_paths.data,
                                                    sizeof(SynurangLiteBytes),
                                                    request->field_weight_paths.len);
    source.weight_path_count = request->field_weight_paths.len;

    if (request->field_source_options) {
        const VolvoxaiV1ModelSourceOptions* input=request->field_source_options;
        options.max_total_bytes=input->field_max_total_bytes;
        options.max_shard_bytes=input->field_max_shard_bytes;
        if (input->field_graph) {
            graph_integrity.byte_length=input->field_graph->field_byte_length;
            graph_integrity.sha256=input->field_graph->field_sha256.data;
            graph_integrity.sha256_size=input->field_graph->field_sha256.len;
            options.graph=&graph_integrity;
        }
        options.weight_count=input->field_weight_shards.len;
        if (options.weight_count>SIZE_MAX/sizeof(*weight_integrity)) {
            result=vx_api_report_fail(allocator,&response->field_report,
                VX_STATUS_INVALID_ARGUMENT,VX_STAGE_MODEL_LOAD,VX_CODE_INVALID_MODEL_SOURCE,
                "source integrity manifest is too large")?0:-1;
            goto done;
        }
        if (options.weight_count) {
            weight_integrity=(VxModelSourceIntegrity*)vx_api_scratch_alloc(
                &scratch,options.weight_count*sizeof(*weight_integrity));
            if (weight_integrity) for (index=0;index<options.weight_count;index++) {
                const VolvoxaiV1ModelSourceIntegrity* entry=&input->field_weight_shards.data[index];
                weight_integrity[index].byte_length=entry->field_byte_length;
                weight_integrity[index].sha256=entry->field_sha256.data;
                weight_integrity[index].sha256_size=entry->field_sha256.len;
            }
        }
        options.weights=weight_integrity;
    }

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

    if (request->field_source_options || graph_document.size) {
        int preflight_only=0;
#if defined(__wasm__)
        preflight_only=vx_api_wasm_path_preflight_is_active();
#endif
        status=vx_runtime_internal_load_model_sources(runtime,&source,
            graph_document.size?&graph_document:NULL,
            request->field_source_options?&options:NULL,&model,&report,preflight_only);
    } else if (request->field_package) {
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
    if (!vx_api_execution_contract_from_json(allocator, &response->field_execution,
        vx_model_execution_contract_document(model))) goto done;
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

    VxAdapterRevision revision = VX_ADAPTER_REVISION_INIT;
    if (request->field_adapter_revision) {
        revision.adapter_id = request->field_adapter_revision->field_adapter_id;
        revision.adapter_revision = request->field_adapter_revision->field_adapter_revision;
    }
    status = vx_model_compile_options_revision(model, &policy,
        request->field_preserve_node_boundaries,
        request->field_adapter_revision ? &revision : NULL, &compiled, &report);
    response->field_preserves_node_boundaries = request->field_preserve_node_boundaries;
    if (!vx_api_compilation_metrics(response, &report) ||
        !vx_api_report_attach(allocator, &response->field_report, &report)) {
        result = -1;
        goto done;
    }
    if (status == VX_STATUS_OK &&
        vx_compiled_model_internal_domain_attestation(compiled, &domain) ==
            VX_STATUS_OK &&
        !vx_api_compiled_memory_bounds(response, &domain)) {
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

static int vx_api_create_execution_context(
    const VolvoxaiV1CreateExecutionContextRequest* request,
    VolvoxaiV1ExecutionContextHandle* response,
    void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxExecutionConfig options = {.context = VX_CONTEXT_OPTIONS_INIT};
    VxDecodeCacheCommand cache = {.action = VX_DECODE_CACHE_CONFIGURE};
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
    if (request->which_execution == 2 && request->field_graph) {
        const VolvoxaiV1GraphExecution* graph = request->field_graph;
        options.context.decode_row_mode = (VxDecodeRowMode)graph->field_row_mode;
        options.context.require_incremental = graph->field_require_incremental;
        options.context.decode_slots = graph->has_slots ? graph->field_slots : 1;
    } else if (request->which_execution == 3 && request->field_autoregressive) {
        const VolvoxaiV1AutoregressiveExecution* generation = request->field_autoregressive;
        options.autoregressive = 1;
        options.context.decode_row_mode = VX_DECODE_ROW_REQUIRED;
        options.context.require_incremental = 1;
        options.context.decode_slots = 1;
        options.max_steps = generation->field_max_steps;
        options.sampling = generation->field_sampling;
        options.has_seed = generation->has_seed;
        options.seed = generation->field_seed;
        options.has_temperature = generation->has_temperature;
        options.temperature = generation->field_temperature;
        options.has_top_k = generation->has_top_k;
        options.top_k = generation->field_top_k;
        options.has_top_p = generation->has_top_p;
        options.top_p = generation->field_top_p;
        options.stop_ids = generation->field_stop_ids.data;
        options.stop_id_count = generation->field_stop_ids.len;
        options.eos_ids = generation->field_eos_ids.data;
        options.eos_id_count = generation->field_eos_ids.len;
        options.stop_sequence_count = generation->field_stop_sequences.len;
        if (options.stop_sequence_count > 128) {
            result = vx_api_report_fail(allocator, &response->field_report,
                VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CONTEXT_CREATE,
                VX_CODE_INVALID_CONTEXT_OPTIONS, "too many generation stop sequences") ? 0 : -1;
            goto done;
        }
        VxGenerationIdSequence* sequences = options.stop_sequence_count ?
            vx_api_scratch_alloc(&scratch, options.stop_sequence_count * sizeof(*sequences)) : NULL;
        if (options.stop_sequence_count && !sequences) { result = -1; goto done; }
        for (size_t i = 0; i < options.stop_sequence_count; i++) {
            sequences[i].ids = generation->field_stop_sequences.data[i].field_ids.data;
            sequences[i].count = generation->field_stop_sequences.data[i].field_ids.len;
        }
        options.stop_sequences = sequences;
    } else {
        result = vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CONTEXT_CREATE,
            VX_CODE_INVALID_ARGUMENT, "one execution mode is required") ? 0 : -1;
        goto done;
    }
    if (request->field_limits) {
        const VolvoxaiV1ExecutionLimits* limits = request->field_limits;
        if (limits->has_max_memory_bytes && !limits->field_max_memory_bytes) {
            result = vx_api_report_fail(allocator, &response->field_report,
                VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CONTEXT_CREATE,
                VX_CODE_INVALID_ARGUMENT, "explicit memory limit must be positive") ? 0 : -1;
            goto done;
        }
        options.max_memory_bytes = limits->has_max_memory_bytes ? limits->field_max_memory_bytes : 0;
        if (limits->field_cache) {
            const VolvoxaiV1ContextCacheOptions* source = limits->field_cache;
            cache.tensor_count = source->field_tensors.len;
            cache.tensors = vx_api_scratch_cstr_array(&scratch, source->field_tensors.data,
                sizeof(SynurangLiteBytes), cache.tensor_count);
            cache.page_tokens = source->field_page_tokens;
            cache.max_pages = source->field_max_pages;
            cache.has_max_pages = source->has_max_pages;
            cache.policy = source->field_policy;
            cache.clear_on_recycle = source->field_clear_on_recycle;
            options.cache = &cache;
        }
    }
    if (vx_api_scratch_failed(&scratch)) { result = -1; goto done; }
    status = vx_compiled_model_create_execution_context(compiled, &options, &context, &report);
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

#include "vx_api_context_cache.inc"
#include "vx_api_tensor_interop.inc"
#include "vx_api_execution.inc"

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
VX_API_UNARY(vx_api_create_execution_context, VolvoxaiV1CreateExecutionContextRequest, VolvoxaiV1ExecutionContextHandle,
    volvoxai_v1_execution_context_handle, vx_inference_create_execution_context_respond)
VX_API_UNARY(vx_api_release_context_handler, VolvoxaiV1ExecutionContextRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_release_execution_context_respond)
VX_API_UNARY(vx_api_get_execution_context_info, VolvoxaiV1ExecutionContextRef, VolvoxaiV1ExecutionContextInfo,
    volvoxai_v1_execution_context_info, vx_inference_get_execution_context_info_respond)
VX_API_UNARY(vx_api_cancel_execution_context, VolvoxaiV1ExecutionContextRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_inference_cancel_execution_context_respond)
VX_API_UNARY(vx_api_get_input_affine_quantization, VolvoxaiV1GetInputAffineQuantizationRequest, VolvoxaiV1AffineQuantization,
    volvoxai_v1_affine_quantization, vx_inference_get_input_affine_quantization_respond)
VX_API_UNARY(vx_api_get_tensor_interop_info, VolvoxaiV1ExecutionContextRef, VolvoxaiV1TensorInteropInfo,
    volvoxai_v1_tensor_interop_info, vx_inference_get_tensor_interop_info_respond)
VX_API_UNARY(vx_api_publish_context_prefix, VolvoxaiV1PublishContextPrefixRequest, VolvoxaiV1ContextCacheState,
    volvoxai_v1_context_cache_state, vx_inference_publish_context_prefix_respond)
VX_API_UNARY(vx_api_reuse_context_prefix, VolvoxaiV1ReuseContextPrefixRequest, VolvoxaiV1ContextCacheState,
    volvoxai_v1_context_cache_state, vx_inference_reuse_context_prefix_respond)
VX_API_UNARY(vx_api_release_context_slot, VolvoxaiV1ContextSlotRef, VolvoxaiV1ContextCacheState,
    volvoxai_v1_context_cache_state, vx_inference_release_context_slot_respond)
VX_API_UNARY(vx_api_evict_context_prefixes, VolvoxaiV1EvictContextPrefixesRequest, VolvoxaiV1ContextCacheState,
    volvoxai_v1_context_cache_state, vx_inference_evict_context_prefixes_respond)

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
    handlers.create_execution_context.message = vx_api_create_execution_context_call;
    handlers.release_execution_context.message = vx_api_release_context_handler_call;
    handlers.get_execution_context_info.message = vx_api_get_execution_context_info_call;
    handlers.cancel_execution_context.message = vx_api_cancel_execution_context_call;
    handlers.get_input_affine_quantization.message = vx_api_get_input_affine_quantization_call;
    handlers.get_tensor_interop_info.message = vx_api_get_tensor_interop_info_call;
    handlers.publish_context_prefix.message = vx_api_publish_context_prefix_call;
    handlers.reuse_context_prefix.message = vx_api_reuse_context_prefix_call;
    handlers.release_context_slot.message = vx_api_release_context_slot_call;
    handlers.evict_context_prefixes.message = vx_api_evict_context_prefixes_call;
    handlers.execute.open = vx_api_execution_open;
    handlers.execute.message = vx_api_execute_call;
    handlers.execute.cancel = vx_api_execution_cancel;
    handlers.execute.destroy = vx_api_execution_destroy;
    handlers.generate.open = vx_api_execution_open;
    handlers.generate.message = vx_api_generate_call;
    handlers.generate.cancel = vx_api_execution_cancel;
    handlers.generate.destroy = vx_api_execution_destroy;
    return vx_inference_register(instance, &handlers, registry);
}
