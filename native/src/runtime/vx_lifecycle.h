/* Engine lifecycle — internal implementation surface.
 *
 * These functions are NOT the public API. proto/volvoxai.proto is, and the
 * generated dispatch in runtime/generated/c/volvoxai_ffi.c reaches
 * them through the handler tables in native/src/api/.
 *
 * This header deliberately lives outside native/include/ so that code outside
 * the repository cannot reach the lifecycle directly. An in-tree file that
 * includes it is either the engine itself, the API handler tables, or a
 * consumer still queued for migration to the protobuf API.
 */
#ifndef VOLVOXAI_LIFECYCLE_H
#define VOLVOXAI_LIFECYCLE_H

#include "vx_lifecycle_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VxRuntime VxRuntime;
typedef struct VxModel VxModel;
typedef struct VxCompiledModel VxCompiledModel;
typedef struct VxExecutionContext VxExecutionContext;
typedef struct VxResult VxResult;
typedef struct VxRequest VxRequest;
/* Private module root accounting; callers hold the last root reference while
 * checking, so a true result cannot race with an unowned object's destruction. */
int vx_runtime_is_unique(const VxRuntime* runtime);
typedef struct VxBackendProvider VxBackendProvider;

VX_API const char* vx_status_string(VxStatus status);
/* Current CLOCK_MONOTONIC time in microseconds, or zero when unavailable.
 * Submit deadlines use this clock domain. */
VX_API uint64_t vx_runtime_monotonic_time_micros(void);

/* Returns one when the exact v1 descriptor was accepted, even if no platform
 * signal was available, and zero for NULL, struct-size, or ABI-version
 * negotiation failure. The function has no runtime handle and cannot change
 * inference state. */
VX_API int vx_process_memory_sample_v1(VxProcessMemorySampleV1* sample);

VX_API VxStatus vx_runtime_create(const VxRuntimeOptions* options,
                           VxRuntime** out_runtime,
                           VxReport* report);
VX_API void vx_runtime_retain(VxRuntime* runtime);
VX_API void vx_runtime_release(VxRuntime* runtime);
typedef void (*VxRuntimeDestroyCallback)(uint64_t runtime_id, void* context);
/* Internal ownership hook used by protocol-layer state keyed to Runtime
 * lineage. The callback runs exactly once, immediately before final free. */
VX_API void vx_runtime_set_destroy_callback(
    VxRuntime* runtime,
    VxRuntimeDestroyCallback callback,
    void* context);
/* Internal per-Runtime seam. Public deployment composition registers a
 * descriptor with vx_backend_register_provider() before generated
 * CreateRuntime; runtime creation reaches this function on its behalf. */
VX_API VxStatus vx_runtime_register_provider(
    VxRuntime* runtime,
    const VxBackendProvider* provider,
    VxReport* report);
/* Logical close is idempotent. It rejects provider registration, model load,
 * and compilation of already loaded models. Compiled models, their contexts,
 * and stable results retain the pinned runtime state and remain usable. */
VX_API VxStatus vx_runtime_close(VxRuntime* runtime, VxReport* report);

VX_API VxStatus vx_runtime_load_model(VxRuntime* runtime,
                               const VxModelSource* source,
                               VxModel** out_model,
                               VxReport* report);
VX_API void vx_model_retain(VxModel* model);
VX_API void vx_model_release(VxModel* model);
VX_API VxStatus vx_model_revision_info(VxModel* model,
                                VxRevisionInfo* info,
                                VxReport* report);
VX_API VxStatus vx_model_publish_adapter(VxModel* model,
                                  const VxAdapterSource* source,
                                  VxAdapterRevision* published,
                                  VxReport* report);

VX_API VxStatus vx_model_compile(VxModel* model,
                          const VxBackendPolicy* policy,
                          VxCompiledModel** out_compiled,
                          VxReport* report);
VX_API void vx_compiled_model_retain(VxCompiledModel* compiled);
VX_API void vx_compiled_model_release(VxCompiledModel* compiled);
VX_API VxStatus vx_compiled_model_report(const VxCompiledModel* compiled,
                                  VxReport* report);
/* Borrowed through the compiled model's retained Model. */
VX_API VxRuntime* vx_compiled_model_runtime(VxCompiledModel* compiled);

/* Direct synchronous execution of one logical request, including a
 * caller-authored bulk B=N binding. The Runtime borrows inputs only until this
 * call returns, reuses the exact compiled route's retained mutable context,
 * and allocates neither a request handle nor the Runtime coordinator. A direct
 * scope retains Runtime through the complete provider invocation and retires
 * its active-call count before dropping that reference. It still reserves one
 * bounded result ticket before entering the provider. A busy route returns
 * VX_STATUS_BUSY and never falls back to scheduling. */
VX_API VxStatus vx_runtime_run(
    VxRuntime* runtime,
    VxCompiledModel* compiled,
    const VxTensorBinding* inputs,
    size_t input_count,
    VxResult** out_result,
    VxReport* report);

/* Scheduled stateless Runtime execution. A Runtime created in DIRECT mode
 * rejects submission. SCHEDULED validates normalized metadata, reserves the
 * Runtime request/input and result budgets, then copies HOST payload bytes.
 * Normalized names borrow the retained model's immutable storage; caller
 * descriptors, names, and payloads may be released once submit returns. */
VX_API VxStatus vx_runtime_submit(
    VxRuntime* runtime,
    VxCompiledModel* compiled,
    const VxTensorBinding* inputs,
    size_t input_count,
    const VxRuntimeSubmitOptions* options,
    VxRequest** out_request,
    VxReport* report);
VX_API VxStatus vx_request_poll(const VxRequest* request,
                                VxRequestInfo* info,
                                VxReport* report);
/* Internal event-loop integration. Notifications run under the request lock
 * and may only schedule later work. Unwatch before freeing the watch storage;
 * unwatch synchronizes with every in-flight notification. Keep a request lease
 * for the entire watch lifetime. Multiple calls may watch the same request. */
typedef struct VxRequestWatch {
    void (*notify)(void* user_data);
    void* user_data;
    struct VxRequestWatch* next;
} VxRequestWatch;
void vx_request_watch(VxRequest* request, VxRequestWatch* watch);
void vx_request_unwatch(VxRequest* request, VxRequestWatch* watch);
/* Advance at most one cooperative scheduler dispatch. Native workers remain
 * responsible for native execution. Returns true if queued work can progress
 * on a later turn; a pending GPU request waits for a device notification. */
int vx_request_progress(VxRequest* request);
/* Zero is a non-blocking wait. VX_REQUEST_WAIT_INFINITE (UINT64_MAX) waits
 * without a timeout. Threaded runtimes wait for at most a finite timeout. On a
 * threadless cooperative runtime every finite timeout is non-blocking because
 * one execution step cannot be preempted; only INFINITE pumps queued work.
 * Every form returns BUSY when it returns while the request remains live. */
VX_API VxStatus vx_request_wait(VxRequest* request,
                                uint64_t timeout_milliseconds,
                                VxReport* report);
/* Queued work is removed immediately. Running device/CPU work is logically
 * cancelled and its result suppressed after the physical execution returns. */
VX_API VxStatus vx_request_cancel(VxRequest* request, VxReport* report);
/* Returns a retained immutable result on success. The caller releases it with
 * vx_result_release(). */
VX_API VxStatus vx_request_result(VxRequest* request,
                                  VxResult** out_result,
                                  VxReport* report);
VX_API void vx_request_retain(VxRequest* request);
VX_API void vx_request_release(VxRequest* request);

VX_API VxStatus vx_compiled_model_create_context(
    VxCompiledModel* compiled,
    const VxContextOptions* options,
    VxExecutionContext** out_context,
    VxReport* report);
VX_API void vx_execution_context_retain(VxExecutionContext* context);
VX_API void vx_execution_context_release(VxExecutionContext* context);
/* Rejects newly submitted operations, drains already accepted operations, and
 * releases runtime-owned execution storage. Safe to call repeatedly. */
VX_API VxStatus vx_execution_context_close(VxExecutionContext* context,
                                    VxReport* report);

VX_API size_t vx_execution_context_input_count(VxExecutionContext* context);
VX_API VxStatus vx_execution_context_input_spec(VxExecutionContext* context,
                                         size_t index,
                                         VxTensorSpec* spec,
                                         VxReport* report);
VX_API VxStatus vx_execution_context_input_affine_quantization(
    VxExecutionContext* context,
    const char* name,
    VxAffineQuantization* quantization,
    VxReport* report);
VX_API VxStatus vx_execution_context_execute(VxExecutionContext* context,
                                      const VxTensorBinding* inputs,
                                      size_t input_count,
                                      VxResult** out_result,
                                      VxReport* report);
/* Recompute only the leading row_count rows of a fixed-shape sequence graph.
 * This is ordinary execution without retained decode/KV state. Built-in
 * backends support it when the graph's row-aware operators accept the prefix;
 * external providers return BACKEND_UNSUPPORTED. */
VX_API VxStatus vx_execution_context_execute_prefix(
                                      VxExecutionContext* context,
                                      int32_t row_count,
                                      const VxTensorBinding* inputs,
                                      size_t input_count,
                                      VxResult** out_result,
                                      VxReport* report);
/* Decode contexts are enabled through VxContextOptions. Prefill executes the
 * complete graph, step executes the negotiated dependency/row path, and reset
 * clears the context-local decode/KV state. Prefill and step return ordinary
 * immutable result snapshots. */
VX_API VxStatus vx_execution_context_decode_prefill(VxExecutionContext* context,
                                          int32_t position,
                                          const VxTensorBinding* inputs,
                                          size_t input_count,
                                          VxResult** out_result,
                                          VxReport* report);
VX_API VxStatus vx_execution_context_decode_step(VxExecutionContext* context,
                                          int32_t position,
                                          const VxTensorBinding* inputs,
                                          size_t input_count,
                                          VxResult** out_result,
                                          VxReport* report);
VX_API VxStatus vx_execution_context_decode_reset(VxExecutionContext* context,
                                           VxReport* report);
/* Adapter changes are FIFO context operations. select pins an exact published
 * revision, or selects the immutable base adapter when revision is NULL.
 * rebind explicitly adopts the model's currently published adapter; no
 * context changes revisions implicitly. */
VX_API VxStatus vx_execution_context_select_adapter(
    VxExecutionContext* context,
    const VxAdapterRevision* revision,
    VxReport* report);
VX_API VxStatus vx_execution_context_rebind_adapter(
    VxExecutionContext* context,
    VxReport* report);

VX_API void vx_result_retain(VxResult* result);
VX_API void vx_result_release(VxResult* result);
VX_API uint64_t vx_result_execution_id(const VxResult* result);
VX_API VxResultState vx_result_state(const VxResult* result);
VX_API VxStatus vx_result_poll(VxResult* result, VxReport* report);
VX_API size_t vx_result_output_count(const VxResult* result);
VX_API VxStatus vx_result_output_info(const VxResult* result,
                               size_t index,
                               VxTensorInfo* info,
                               VxReport* report);
/* Copies a named immutable snapshot into caller-owned storage. Passing NULL
 * with capacity zero queries the required byte count. */
VX_API VxStatus vx_result_read(const VxResult* result,
                        const char* name,
                        void* destination,
                        size_t capacity,
                        size_t* required,
                        VxReport* report);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_LIFECYCLE_H */
