/* VxSchedulerService — cross-request admission, deadlines and coalescing.
 *
 * This is the serving composition above the policy-neutral engine submission
 * seam. A Runtime created with EXECUTION_MODE_DIRECT rejects submission, and
 * that rejection is reported by the engine rather than pre-empted here.
 */
#include "vx_api_convert.h"
#include "vx_api_buffer.h"
#include "vx_api_handles.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"

#include <stdlib.h>
#include <string.h>

/* Results produced by the scheduler enter the same registry the inference
 * service uses, so a caller releases them through ReleaseResult. */
int64_t vx_api_publish_scheduler_result(VxApiRegistry* user_data,
    VxResult* result,
    const VxApiHandleLineage* lineage);

static void vx_api_scheduler_apply_lineage(
    VolvoxaiV1OperationReport* report,
    const VxApiHandleLineage* lineage) {
    if (!lineage) return;
    vx_api_report_set_public_lineage(
        report, lineage->runtime_id, lineage->model_id,
        lineage->compiled_model_id, lineage->context_id);
}

static void vx_api_retain_request(void* p) { vx_request_retain((VxRequest*)p); }
static void vx_api_release_request(void* p) { vx_request_release((VxRequest*)p); }

static int vx_api_submit(const VolvoxaiV1SubmitRequest* request,
                         VolvoxaiV1RequestHandle* response,
                         void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxRuntimeSubmitOptions options = VX_RUNTIME_SUBMIT_OPTIONS_INIT;
    VxTensorBinding* bindings = NULL;
    VxReport report = VX_REPORT_INIT;
    VxRequest* handle = NULL;
    VxStatus status;
    size_t index;
    VxRuntime* runtime;
    VxCompiledModel* compiled;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    int api_result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_COMPILED_MODEL,
                               request->field_compiled_model_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_SUBMIT,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released compiled model")
                   ? 0 : -1;
    }
    compiled = (VxCompiledModel*)lease.pointer;
    runtime = vx_compiled_model_runtime(compiled);
    if (!runtime) {
        vx_api_handle_lease_release(&lease);
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_SUBMIT,
                                  VX_CODE_HANDLE_DISPOSED, "compiled model has no runtime")
                   ? 0 : -1;
    }

    if (request->field_options) {
        const VolvoxaiV1SubmitOptions* source = request->field_options;
        /* Absent fields keep the engine default rather than becoming zero. */
        if (source->has_priority) options.priority = source->field_priority;
        if (source->has_deadline_monotonic_ns) {
            options.deadline_monotonic_micros = vx_api_ns_ticks(source->field_deadline_monotonic_ns, 1000);
        }
        if (source->has_freshness) {
            options.freshness = (VxRequestFreshness)source->field_freshness;
        }
        if (source->has_stream_key) options.stream_key = source->field_stream_key;
    }

    if (request->field_inputs.len) {
        bindings = (VxTensorBinding*)vx_api_scratch_alloc(
            &scratch, sizeof(*bindings) * request->field_inputs.len);
        if (!bindings) {
            vx_api_scratch_release(&scratch);
            vx_api_handle_lease_release(&lease);
            return -1;
        }
        for (index = 0; index < request->field_inputs.len; index++) {
            status = vx_api_binding_from_tensor(&scratch, &bindings[index],
                                                &request->field_inputs.data[index]);
            if (status != VX_STATUS_OK) {
                vx_api_scratch_release(&scratch);
                api_result = vx_api_report_binding_fail(
                    allocator, &response->field_report, status,
                    VX_STAGE_INPUT,
                    "input tensor could not be bound", &request->field_inputs.data[index], index)
                                 ? 0 : -1;
                vx_api_handle_lease_release(&lease);
                return api_result;
            }
        }
    }

    status = vx_runtime_submit(runtime, compiled, bindings,
                               request->field_inputs.len, &options, &handle, &report);
    vx_api_scratch_release(&scratch);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        if (handle) vx_request_release(handle);
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    vx_api_scheduler_apply_lineage(response->field_report, &lease.lineage);
    if (status != VX_STATUS_OK || !handle) {
        if (handle) vx_request_release(handle);
        vx_api_handle_lease_release(&lease);
        return 0;
    }

    response->field_request_id = vx_api_handle_insert_with_lineage(user_data,
        VX_API_HANDLE_REQUEST, handle, vx_api_retain_request,
        vx_api_release_request, &lease.lineage);
    if (!response->field_request_id) {
        vx_request_release(handle);
        api_result = vx_api_report_fail(allocator, &response->field_report,
                                        VX_STATUS_OUT_OF_MEMORY, VX_STAGE_SUBMIT,
                                        VX_CODE_OUT_OF_MEMORY,
                                        "handle registry allocation failed")
                         ? 0 : -1;
        vx_api_handle_lease_release(&lease);
        return api_result;
    }
    vx_api_handle_lease_release(&lease);
    return 0;
}

/* Shared projection of VxRequestInfo onto the response message. */
static int vx_api_request_info(const SynurangLiteAllocator* allocator,
                               VolvoxaiV1RequestInfo* response,
                               int64_t request_id,
                               const VxRequestInfo* info,
                               const VxReport* report) {
    response->field_request_id = request_id;
    response->field_state = (VolvoxaiV1RequestState)info->state;
    response->field_status = (VolvoxaiV1NativeStatus)info->status;
    response->field_owned_input_bytes = info->owned_input_bytes;
    response->field_deadline_missed = info->deadline_missed != 0;
    return vx_api_report_attach(allocator, &response->field_report, report) ? 0 : -1;
}

static int vx_api_poll_request(const VolvoxaiV1RequestRef* request,
                               VolvoxaiV1RequestInfo* response,
                               void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxRequestInfo info = VX_REQUEST_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxRequest* handle;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_REQUEST,
                               request->field_request_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_NONE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released request")
                   ? 0 : -1;
    }
    handle = (VxRequest*)lease.pointer;
    (void)vx_request_poll(handle, &info, &report);
    result = vx_api_request_info(allocator, response, request->field_request_id,
                                 &info, &report);
    if (result == 0) {
        vx_api_scheduler_apply_lineage(response->field_report, &lease.lineage);
    }
    vx_api_handle_lease_release(&lease);
    return result;
}

/* WaitRequest retains the request, never its borrowed protobuf input. A
 * transport timeout/cancel retires this observer without cancelling engine
 * work. Native worker notifications and browser GPU notifications only wake
 * the host; completion encoding runs during the module's bounded poll. */
typedef struct VxApiWaitRequest {
    VxApiPending pending;
    VxApiRegistry* registry;
    SynurangStream* stream;
    VxApiHandleLease lease;
    VxRequestWatch watch;
    int64_t request_id;
} VxApiWaitRequest;

static void vx_api_wait_request_detach(VxApiWaitRequest* call) {
    if (!call || !call->stream) return;
    vx_request_unwatch(call->lease.pointer, &call->watch);
    vx_api_pending_remove(&call->pending);
    vx_api_handle_lease_release(&call->lease);
    SynurangStream* stream = call->stream;
    call->stream = NULL;
    synurang_stream_release(stream);
}

static void vx_api_wait_request_poll(VxApiPending* pending) {
    VxApiWaitRequest* call = (VxApiWaitRequest*)pending;
    VxRequestInfo info = VX_REQUEST_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    int more = vx_request_progress(call->lease.pointer);
    (void)vx_request_poll(call->lease.pointer, &info, &report);
    if (info.state == VX_REQUEST_STATE_QUEUED || info.state == VX_REQUEST_STATE_RUNNING) {
        if (more) vx_api_pending_notify(pending);
        return;
    }
    VolvoxaiV1RequestInfo response;
    volvoxai_v1_request_info_init(&response);
    int result = vx_api_request_info(response._allocator, &response,
                                     call->request_id, &info, &report);
    if (result == 0) vx_api_scheduler_apply_lineage(response.field_report, &call->lease.lineage);
    SynurangStatus status = result == 0
        ? vx_scheduler_wait_request_respond(call->stream, &response) : SYNURANG_INTERNAL;
    volvoxai_v1_request_info_free(&response);
    if (status == SYNURANG_OK) (void)synurang_stream_finish(call->stream);
    else (void)synurang_stream_fail_error(call->stream, status, 13,
                                         "Response construction failed");
    vx_api_wait_request_detach(call);
}

static void* vx_api_wait_request_open(SynurangStream* stream,
                                      const SynurangCallOptions* options, void* registry) {
    (void)options;
    VxApiWaitRequest* call = calloc(1, sizeof(*call));
    if (!call) {
        (void)synurang_stream_fail_error(stream, SYNURANG_OUT_OF_MEMORY, 8,
                                        "Wait allocation failed");
        return NULL;
    }
    call->registry = registry;
    call->pending.poll = vx_api_wait_request_poll;
    call->watch.notify = vx_api_pending_notify;
    call->watch.user_data = &call->pending;
    return call;
}

static void vx_api_wait_request_call(SynurangStream* stream,
                                     const VolvoxaiV1RequestRef* request, void* user_data) {
    VxApiWaitRequest* call = user_data;
    if (!call) return;
    call->request_id = request->field_request_id;
    if (!vx_api_handle_acquire(call->registry, VX_API_HANDLE_REQUEST,
                               call->request_id, &call->lease)) {
        VolvoxaiV1RequestInfo response;
        volvoxai_v1_request_info_init(&response);
        int result = vx_api_poll_request(request, &response, call->registry);
        SynurangStatus status = result == 0
            ? vx_scheduler_wait_request_respond(stream, &response) : SYNURANG_INTERNAL;
        volvoxai_v1_request_info_free(&response);
        if (status == SYNURANG_OK) (void)synurang_stream_finish(stream);
        else (void)synurang_stream_fail_error(stream, status, 13,
                                             "Response construction failed");
        return;
    }
    call->stream = synurang_stream_retain(stream);
    vx_api_pending_add(call->registry, &call->pending);
    vx_request_watch(call->lease.pointer, &call->watch);
}

static void vx_api_wait_request_cancel(SynurangStream* stream, void* user_data) {
    (void)stream;
    vx_api_wait_request_detach(user_data);
}

static void vx_api_wait_request_destroy(void* user_data) {
    vx_api_wait_request_detach(user_data);
    free(user_data);
}

static int vx_api_cancel_request(const VolvoxaiV1RequestRef* request,
                                 VolvoxaiV1OperationReport* response,
                                 void* user_data) {
    (void)user_data;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxRequest* handle;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_REQUEST,
                               request->field_request_id, &lease)) {
        response->field_status = VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED;
        return 0;
    }
    handle = (VxRequest*)lease.pointer;
    (void)vx_request_cancel(handle, &report);
    result = vx_api_report_from_native(response, &report) ? 0 : -1;
    if (result == 0) vx_api_scheduler_apply_lineage(response, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_take_request_result(const VolvoxaiV1RequestRef* request,
                                      VolvoxaiV1ExecutionResultHandle* response,
                                      void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxRequest* handle;
    int api_result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_REQUEST,
                               request->field_request_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_EXECUTE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released request")
                   ? 0 : -1;
    }
    handle = (VxRequest*)lease.pointer;
    status = vx_request_result(handle, &result, &report);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        if (result) vx_result_release(result);
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    vx_api_scheduler_apply_lineage(response->field_report, &lease.lineage);
    if (status != VX_STATUS_OK || !result) {
        if (result) vx_result_release(result);
        vx_api_handle_lease_release(&lease);
        return 0;
    }

    if (!vx_api_execution_result_fields(response, result, &report)) {
        vx_result_release(result);
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    response->field_result_id =
        vx_api_publish_scheduler_result(user_data, result, &lease.lineage);
    if (!response->field_result_id) {
        api_result = vx_api_report_fail(allocator, &response->field_report,
                                        VX_STATUS_OUT_OF_MEMORY, VX_STAGE_EXECUTE,
                                        VX_CODE_OUT_OF_MEMORY,
                                        "handle registry allocation failed")
                         ? 0 : -1;
        vx_api_handle_lease_release(&lease);
        return api_result;
    }
    vx_api_handle_lease_release(&lease);
    return 0;
}

static int vx_api_release_request_handler(const VolvoxaiV1RequestRef* request,
                                          VolvoxaiV1OperationReport* response,
                                          void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (vx_api_handle_acquire(user_data, VX_API_HANDLE_REQUEST, request->field_request_id, &lease)) {
        VxReport ignored = VX_REPORT_INIT;
        (void)vx_request_cancel((VxRequest*)lease.pointer, &ignored);
        vx_api_handle_lease_release(&lease);
    }
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_REQUEST, request->field_request_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}

#include "vx_api_batch_queue.inc"

VX_API_UNARY(vx_api_submit, VolvoxaiV1SubmitRequest, VolvoxaiV1RequestHandle,
    volvoxai_v1_request_handle, vx_scheduler_submit_respond)
VX_API_UNARY(vx_api_poll_request, VolvoxaiV1RequestRef, VolvoxaiV1RequestInfo,
    volvoxai_v1_request_info, vx_scheduler_poll_request_respond)
VX_API_UNARY(vx_api_cancel_request, VolvoxaiV1RequestRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_scheduler_cancel_request_respond)
VX_API_UNARY(vx_api_take_request_result, VolvoxaiV1RequestRef, VolvoxaiV1ExecutionResultHandle,
    volvoxai_v1_execution_result_handle, vx_scheduler_take_request_result_respond)
VX_API_UNARY(vx_api_release_request_handler, VolvoxaiV1RequestRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_scheduler_release_request_respond)
VX_API_UNARY(vx_api_create_batch_queue, VolvoxaiV1CreateBatchQueueRequest, VolvoxaiV1BatchQueueHandle,
    volvoxai_v1_batch_queue_handle, vx_scheduler_create_batch_queue_respond)
VX_API_UNARY(vx_api_submit_batch_work, VolvoxaiV1SubmitBatchWorkRequest, VolvoxaiV1BatchWorkHandle,
    volvoxai_v1_batch_work_handle, vx_scheduler_submit_batch_work_respond)
VX_API_UNARY(vx_api_next_batch_dispatch, VolvoxaiV1BatchQueueRef, VolvoxaiV1BatchDispatch,
    volvoxai_v1_batch_dispatch, vx_scheduler_next_batch_dispatch_respond)
VX_API_UNARY(vx_api_complete_batch_dispatch, VolvoxaiV1CompleteBatchDispatchRequest, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_scheduler_complete_batch_dispatch_respond)
VX_API_UNARY(vx_api_get_batch_work, VolvoxaiV1BatchWorkRef, VolvoxaiV1BatchWorkInfo,
    volvoxai_v1_batch_work_info, vx_scheduler_get_batch_work_respond)
VX_API_UNARY(vx_api_take_batch_work, VolvoxaiV1BatchWorkRef, VolvoxaiV1BatchWorkInfo,
    volvoxai_v1_batch_work_info, vx_scheduler_take_batch_work_respond)
VX_API_UNARY(vx_api_get_batch_queue, VolvoxaiV1BatchQueueRef, VolvoxaiV1BatchQueueInfo,
    volvoxai_v1_batch_queue_info, vx_scheduler_get_batch_queue_respond)
VX_API_UNARY(vx_api_cancel_batch_work, VolvoxaiV1BatchWorkRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_scheduler_cancel_batch_work_respond)
VX_API_UNARY(vx_api_close_batch_queue, VolvoxaiV1CloseBatchQueueRequest, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_scheduler_close_batch_queue_respond)
VX_API_UNARY(vx_api_release_batch_queue, VolvoxaiV1BatchQueueRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_scheduler_release_batch_queue_respond)

int vx_api_install_scheduler_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxSchedulerServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.submit.message = vx_api_submit_call;
    handlers.poll_request.message = vx_api_poll_request_call;
    handlers.wait_request.open = vx_api_wait_request_open;
    handlers.wait_request.message = vx_api_wait_request_call;
    handlers.wait_request.cancel = vx_api_wait_request_cancel;
    handlers.wait_request.destroy = vx_api_wait_request_destroy;
    handlers.cancel_request.message = vx_api_cancel_request_call;
    handlers.take_request_result.message = vx_api_take_request_result_call;
    handlers.release_request.message = vx_api_release_request_handler_call;
    handlers.create_batch_queue.message = vx_api_create_batch_queue_call;
    handlers.submit_batch_work.message = vx_api_submit_batch_work_call;
    handlers.next_batch_dispatch.message = vx_api_next_batch_dispatch_call;
    handlers.complete_batch_dispatch.message = vx_api_complete_batch_dispatch_call;
    handlers.get_batch_work.message = vx_api_get_batch_work_call;
    handlers.take_batch_work.message = vx_api_take_batch_work_call;
    handlers.get_batch_queue.message = vx_api_get_batch_queue_call;
    handlers.cancel_batch_work.message = vx_api_cancel_batch_work_call;
    handlers.close_batch_queue.message = vx_api_close_batch_queue_call;
    handlers.release_batch_queue.message = vx_api_release_batch_queue_call;
    return vx_scheduler_register(instance, &handlers, registry);
}
