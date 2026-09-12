/* Exercise the production threadless execution-context queue in its real
 * WebAssembly composition.  Including the amalgamation keeps the internal
 * queue helpers private while making them visible to this one test function. */
#include "../../native/src/runtime/portable_control_wasm.c"

static int close_callback_count;

static VxStatus contract_context_close(void* context_instance,
                                       VxReport* report) {
    (void)context_instance;
    (void)report;
    close_callback_count++;
    return VX_STATUS_OK;
}

int volvoxai_test_threadless_context_reentry(void) {
    VxBackendProvider provider;
    VxModel model;
    VxCompiledModel compiled;
    VxExecutionContext context;
    VxContextOperation outer;
    VxContextOperation rejected;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;

    memset(&provider, 0, sizeof(provider));
    memset(&model, 0, sizeof(model));
    memset(&compiled, 0, sizeof(compiled));
    memset(&context, 0, sizeof(context));
    provider.context_close = contract_context_close;
    model.input_count = 1u;
    compiled.model = &model;
    compiled.provider = &provider;
    snprintf(compiled.backend, sizeof(compiled.backend), "%s", "contract");
    context.compiled = &compiled;

    status = vx_context_operation_begin(
        &context, &outer, VX_STAGE_EXECUTE, &report);
    if (status != VX_STATUS_OK || !outer.accepted || outer.ticket != 0u ||
        context.next_ticket != 1u || context.serving_ticket != 0u)
        return 1;

    report = (VxReport)VX_REPORT_INIT;
    status = vx_context_operation_begin(
        &context, &rejected, VX_STAGE_INPUT, &report);
    if (status != VX_STATUS_BUSY || rejected.accepted ||
        context.next_ticket != 1u || context.serving_ticket != 0u ||
        context.closing || context.closed)
        return 2;

    report = (VxReport)VX_REPORT_INIT;
    status = vx_execution_context_close(&context, &report);
    if (status != VX_STATUS_BUSY || context.next_ticket != 1u ||
        context.serving_ticket != 0u || context.closing || context.closed ||
        close_callback_count != 0)
        return 3;

    vx_context_operation_end(&context, &outer);
    if (context.next_ticket != 1u || context.serving_ticket != 1u)
        return 4;

    if (vx_execution_context_input_count(&context) != 1u ||
        context.next_ticket != 2u || context.serving_ticket != 2u)
        return 5;

    report = (VxReport)VX_REPORT_INIT;
    status = vx_execution_context_close(&context, &report);
    if (status != VX_STATUS_OK || !context.closing || !context.closed ||
        context.next_ticket != 3u || context.serving_ticket != 3u ||
        close_callback_count != 1)
        return 6;

    return 0;
}

int volvoxai_test_threadless_request_wait(void) {
    VxCompiledModel compiled;
    VxRequest request;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;

    memset(&compiled, 0, sizeof(compiled));
    memset(&request, 0, sizeof(request));
    snprintf(compiled.backend, sizeof(compiled.backend), "%s", "contract");
    request.compiled = &compiled;
    request.state = VX_REQUEST_STATE_QUEUED;
    request.terminal_status = VX_STATUS_BUSY;
    request.report = (VxReport)VX_REPORT_INIT;
    if (pthread_mutex_init(&request.mutex, NULL) != 0) return 10;

    /* A finite wait must not enter the non-preemptible cooperative step.  A
     * deliberately invalid non-NULL pointer turns an accidental pump into a
     * deterministic trap instead of letting this regression pass by luck. */
    request.coordinator = (VxRuntimeCoordinator*)(uintptr_t)1u;
    status = vx_request_wait(&request, 0u, &report);
    if (status != VX_STATUS_BUSY ||
        request.state != VX_REQUEST_STATE_QUEUED ||
        report.status != VX_STATUS_BUSY || report.code != VX_CODE_BUSY)
        return 11;

    report = (VxReport)VX_REPORT_INIT;
    status = vx_request_wait(&request, 1u, &report);
    if (status != VX_STATUS_BUSY ||
        request.state != VX_REQUEST_STATE_QUEUED ||
        report.status != VX_STATUS_BUSY || report.code != VX_CODE_BUSY)
        return 12;

    /* Infinite normally pumps, but a detached request has no coordinator.
     * It stays live and reports BUSY rather than reading address zero. */
    request.coordinator = NULL;
    report = (VxReport)VX_REPORT_INIT;
    status = vx_request_wait(
        &request, VX_REQUEST_WAIT_INFINITE, &report);
    if (status != VX_STATUS_BUSY ||
        request.state != VX_REQUEST_STATE_QUEUED ||
        report.status != VX_STATUS_BUSY || report.code != VX_CODE_BUSY)
        return 13;

    /* A terminal snapshot is returned independently of timeout/coordinator. */
    request.state = VX_REQUEST_STATE_SUCCEEDED;
    request.terminal_status = VX_STATUS_OK;
    request.report = (VxReport)VX_REPORT_INIT;
    report = (VxReport)VX_REPORT_INIT;
    status = vx_request_wait(&request, 0u, &report);
    pthread_mutex_destroy(&request.mutex);
    if (status != VX_STATUS_OK || report.status != VX_STATUS_OK)
        return 14;

    return 0;
}
