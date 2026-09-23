/* Conversions between the engine's C lifecycle types and the protobuf API.
 *
 * The engine still records routing, fallback and decode evidence as packed
 * "key=value;" strings inside VxReport. The public contract in
 * proto/volvoxai.proto is typed, so this layer parses those strings once, at
 * the boundary, and no consumer ever sees the packed spelling.
 *
 * That parsing is deliberate, bounded technical debt: the packed format is an
 * engine implementation detail scheduled for replacement by typed VxReport
 * fields. Until then every accepted key is listed here, and an unrecognized
 * key is ignored rather than guessed at.
 */
#ifndef VOLVOXAI_API_CONVERT_H
#define VOLVOXAI_API_CONVERT_H

#include "vx_lifecycle.h"
#include "volvoxai_lite.h"
#include "vx_api_handles.h"

/* Private scheduler clocks retain their existing ticks; the wire uses ns. */
uint64_t vx_api_duration_ns(double milliseconds);
uint64_t vx_api_ns_ticks(uint64_t nanoseconds, uint64_t tick_ns);

typedef struct VxCompiledDomainAttestationView
    VxCompiledDomainAttestationView;

/* Call-scoped storage for values the engine needs but protobuf does not
 * provide directly.
 *
 * Generated lite strings are length-delimited and deliberately not
 * NUL-terminated, while the engine's C lifecycle takes `const char*`. Every
 * name, path and backend spelling therefore needs a terminated copy that
 * lives exactly as long as the call. The scratch owns those copies and frees
 * them in one step, so no handler has to unwind partial allocations. */
typedef struct VxApiScratchBlock VxApiScratchBlock;
typedef struct VxApiScratchCleanup VxApiScratchCleanup;

typedef struct VxApiScratch {
    VxApiScratchBlock* head;
    int failed;
    VxApiRegistry* registry;
    VxApiScratchCleanup* cleanups;
} VxApiScratch;

#define VX_API_SCRATCH_INIT { NULL, 0, NULL, NULL }
#define VX_API_SCRATCH_OWNER(owner) { NULL, 0, (owner), NULL }
int vx_api_scratch_cleanup(VxApiScratch* scratch, void (*release)(void*), void* pointer);

/* Returns a NUL-terminated copy of bytes, or NULL when bytes is empty. On
 * allocation failure the scratch is marked failed and NULL is returned; a
 * caller may defer checking until vx_api_scratch_failed(). */
const char* vx_api_scratch_cstr(VxApiScratch* scratch,
                                const SynurangLiteBytes* bytes);

/* Returns an array of count NUL-terminated copies, or NULL when count is 0. */
const char* const* vx_api_scratch_cstr_array(VxApiScratch* scratch,
                                             const SynurangLiteBytes* items,
                                             size_t stride,
                                             size_t count);

/* Reserves zeroed, suitably aligned call-scoped storage. */
void* vx_api_scratch_alloc(VxApiScratch* scratch, size_t size);

int vx_api_scratch_failed(const VxApiScratch* scratch);

void vx_api_scratch_release(VxApiScratch* scratch);

/* Fills report from the engine record. Returns 0 on allocation failure. */
int vx_api_report_from_native(VolvoxaiV1OperationReport* report,
                              const VxReport* native);

/* Convenience: initializes report inside owner and fills it. `owner_alloc` is
 * the allocator of the message that owns report. Returns 0 on failure. */
int vx_api_report_attach(const SynurangLiteAllocator* owner_alloc,
                         VolvoxaiV1OperationReport** slot,
                         const VxReport* native);

/* Replaces core-private object identities with the public protobuf handle
 * ancestry retained by vx_api_handles. Execution and immutable revision
 * identities remain native because they are not registry handles. */
void vx_api_report_set_public_lineage(VolvoxaiV1OperationReport* report,
                                      uint64_t runtime_id,
                                      uint64_t model_id,
                                      uint64_t compiled_model_id,
                                      uint64_t context_id);

/* Adds the retained planning handle without changing any other ancestry. */
void vx_api_report_set_graph_plan_lineage(VolvoxaiV1OperationReport* report,
                                          uint64_t graph_plan_id);

/* Writes an OK report for an operation the engine never had to run, such as a
 * pure query answered from retained state. Returns 0 on allocation failure. */
int vx_api_report_ok(const SynurangLiteAllocator* owner_alloc,
                     VolvoxaiV1OperationReport** slot,
                     VxStage stage);

/* Writes a report carrying only a status/stage/message, for failures the
 * engine never saw (bad handle, malformed request). Returns 0 on failure. */
int vx_api_report_fail(const SynurangLiteAllocator* owner_alloc,
                       VolvoxaiV1OperationReport** slot,
                       VxStatus status,
                       VxStage stage,
                       VxOperationCode code,
                       const char* message);

/* Projects one engine tensor descriptor onto the protobuf spec message. */
int vx_api_tensor_spec_from_native(const SynurangLiteAllocator* allocator,
                                   VolvoxaiV1TensorSpec* spec,
                                   const VxTensorSpec* native);

int vx_api_tensor_info_from_native(const SynurangLiteAllocator* allocator,
                                   VolvoxaiV1TensorInfo* info,
                                   const VxTensorInfo* native);

/* Resolves one caller-owned BorrowedBuffer only when this dispatch shares the
 * caller's address space. Browser wasm dispatch is REMOTE by contract, so it
 * returns TRANSPORT_UNSUPPORTED before performing address arithmetic. */
VxStatus vx_api_borrowed_resolve(const VolvoxaiV1BorrowedBuffer* view,
                                    void** data,
                                    size_t* byte_size);

/* Converts a tensor-binding failure into the typed data-plane report promised
 * by proto/volvoxai.proto. */
int vx_api_report_binding_fail(const SynurangLiteAllocator* owner_alloc,
                               VolvoxaiV1OperationReport** slot,
                               VxStatus status,
                               VxStage stage,
                               const char* invalid_message,
                               const VolvoxaiV1Tensor* tensor, size_t input_index);

/* Borrowed view of one protobuf tensor as an engine binding.
 *
 * The binding points into the request message, so it is valid only while that
 * message is alive. `inline` payloads borrow the protobuf bytes directly and
 * `borrowed` payloads borrow local memory; `buffer` payloads retain storage, so no
 * data copy happens on this path. Returns a VxStatus. */
VxStatus vx_api_binding_from_tensor(VxApiScratch* scratch,
                                    VxTensorBinding* binding,
                                    const VolvoxaiV1Tensor* tensor);

int vx_api_compiled_memory_bounds(VolvoxaiV1CompiledModelHandle* response,
    const VxCompiledDomainAttestationView* domain);
int vx_api_execution_metrics(const SynurangLiteAllocator* allocator,
    VolvoxaiV1ExecutionMetrics** output, const VxReport* report);
int vx_api_execution_result_fields(VolvoxaiV1ExecutionResultHandle* output,
    const VxResult* result, const VxReport* report);
int vx_api_memory_counter(VolvoxaiV1MemorySnapshot* snapshot, const char* name,
    uint64_t value, int metric);
int vx_api_memory_snapshot(VolvoxaiV1MemorySnapshot* snapshot,
    int owner_kind, uint64_t owner_id, const VxProcessMemorySampleV1* sample,
    uint64_t start, uint64_t end);

#endif /* VOLVOXAI_API_CONVERT_H */
