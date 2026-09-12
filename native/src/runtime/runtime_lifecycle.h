#ifndef VOLVOXAI_RUNTIME_LIFECYCLE_H
#define VOLVOXAI_RUNTIME_LIFECYCLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_RUNTIME_LIFECYCLE_CORE_API
#define VX_RUNTIME_LIFECYCLE_CORE_API
#endif

/*
 * Portable inference-owner lifecycle state.
 *
 * The host owns and synchronizes every record.  The core contains no pointer,
 * allocator, lock, clock, thread, callback, handle table, or process-global
 * state.  A browser adapter may keep the records in WASM linear memory while
 * native owners embed them directly beside their pthread synchronization.
 *
 * Runtime and CompiledModel admissions may execute concurrently.  An
 * ExecutionContext instead admits operations into one strict FIFO.  Starting
 * close rejects every later admission but preserves all earlier tickets.  A
 * close reaches its physical-host boundary only after accepted operations and
 * retained children have drained.
 *
 * Operation tickets and owner states are canonical mutable capabilities.  A
 * host adapter may marshal their bytes into a call, but it must retain exactly
 * one authoritative live copy and atomically publish the returned phase.  The
 * phase and integrity tag reject accidental corruption and ordinary
 * duplicate/stale completion.  The core deliberately does not create a hidden
 * registry to recognize hostile copies of an otherwise valid concurrent-
 * admission ticket.
 */
#define VX_RUNTIME_LIFECYCLE_ABI_VERSION UINT32_C(1)
#define VX_RUNTIME_LIFECYCLE_DOMAIN_MAGIC UINT32_C(0x31445256) /* "VRD1" */
#define VX_RUNTIME_LIFECYCLE_STATE_MAGIC UINT32_C(0x314C5256) /* "VRL1" */
#define VX_RUNTIME_LIFECYCLE_TICKET_MAGIC UINT32_C(0x31545256) /* "VRT1" */

typedef int32_t VxRuntimeLifecycleStatusV1;
enum {
    VX_RUNTIME_LIFECYCLE_OK = 0,
    VX_RUNTIME_LIFECYCLE_WAIT = 1,
    VX_RUNTIME_LIFECYCLE_ALREADY_CLOSING = 2,
    VX_RUNTIME_LIFECYCLE_ALREADY_CLOSED = 3,

    VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT = -1,
    VX_RUNTIME_LIFECYCLE_INVALID_STATE = -2,
    VX_RUNTIME_LIFECYCLE_HANDLE_DISPOSED = -3,
    VX_RUNTIME_LIFECYCLE_OVERFLOW = -4,
    VX_RUNTIME_LIFECYCLE_STALE = -5,
    VX_RUNTIME_LIFECYCLE_DUPLICATE = -6,
    VX_RUNTIME_LIFECYCLE_PARENT_MISMATCH = -7
};

typedef uint32_t VxRuntimeLifecycleOwnerKindV1;
enum {
    VX_RUNTIME_LIFECYCLE_OWNER_NONE = 0u,
    VX_RUNTIME_LIFECYCLE_OWNER_RUNTIME = 1u,
    VX_RUNTIME_LIFECYCLE_OWNER_COMPILED_MODEL = 2u,
    VX_RUNTIME_LIFECYCLE_OWNER_EXECUTION_CONTEXT = 3u
};

typedef uint32_t VxRuntimeLifecycleAdmissionModeV1;
enum {
    VX_RUNTIME_LIFECYCLE_ADMISSION_NONE = 0u,
    VX_RUNTIME_LIFECYCLE_ADMISSION_PARALLEL = 1u,
    VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO = 2u
};

typedef uint32_t VxRuntimeLifecyclePhaseV1;
enum {
    VX_RUNTIME_LIFECYCLE_PHASE_NONE = 0u,
    VX_RUNTIME_LIFECYCLE_PHASE_OPEN = 1u,
    VX_RUNTIME_LIFECYCLE_PHASE_CLOSING = 2u,
    VX_RUNTIME_LIFECYCLE_PHASE_CLOSED = 3u
};

typedef uint32_t VxRuntimeLifecycleTicketPhaseV1;
enum {
    VX_RUNTIME_LIFECYCLE_TICKET_EMPTY = 0u,
    VX_RUNTIME_LIFECYCLE_TICKET_ADMITTED = 1u,
    VX_RUNTIME_LIFECYCLE_TICKET_ACTIVE = 2u,
    VX_RUNTIME_LIFECYCLE_TICKET_COMPLETED = 3u
};

/* Operation values are host-visible diagnostics.  The lifecycle formulas are
 * deliberately independent of the numerical/backend operation kind. */
typedef uint32_t VxRuntimeLifecycleOperationV1;
enum {
    VX_RUNTIME_LIFECYCLE_OPERATION_NONE = 0u,
    VX_RUNTIME_LIFECYCLE_OPERATION_WORK = 1u,
    VX_RUNTIME_LIFECYCLE_OPERATION_CHILD_CONSTRUCTION = 2u,
    VX_RUNTIME_LIFECYCLE_OPERATION_ROUTE = 3u,
    VX_RUNTIME_LIFECYCLE_OPERATION_EXECUTE = 4u,
    VX_RUNTIME_LIFECYCLE_OPERATION_DECODE_PREFILL = 5u,
    VX_RUNTIME_LIFECYCLE_OPERATION_DECODE_STEP = 6u,
    VX_RUNTIME_LIFECYCLE_OPERATION_DECODE_RESET = 7u,
    VX_RUNTIME_LIFECYCLE_OPERATION_SELECT_ADAPTER = 8u
};
#define VX_RUNTIME_LIFECYCLE_OPERATION_CLOSE UINT32_MAX

typedef uint32_t VxRuntimeLifecycleCloseStateV1;
enum {
    VX_RUNTIME_LIFECYCLE_CLOSE_NONE = 0u,
    VX_RUNTIME_LIFECYCLE_CLOSE_ADMITTED = 1u,
    VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE = 2u,
    VX_RUNTIME_LIFECYCLE_CLOSE_COMPLETED = 3u
};

typedef uint32_t VxRuntimeLifecycleCloseResultV1;
enum {
    VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_NONE = 0u,
    VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_SUCCEEDED = 1u,
    VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_FAILED = 2u
};

typedef uint32_t VxRuntimeLifecycleCloseReadinessV1;
enum {
    VX_RUNTIME_LIFECYCLE_CLOSE_NOT_REQUESTED = 0u,
    VX_RUNTIME_LIFECYCLE_CLOSE_DRAINING_WORK = 1u,
    VX_RUNTIME_LIFECYCLE_CLOSE_DRAINING_CHILDREN = 2u,
    VX_RUNTIME_LIFECYCLE_CLOSE_READY = 3u,
    VX_RUNTIME_LIFECYCLE_CLOSE_HOST_ACTIVE = 4u,
    VX_RUNTIME_LIFECYCLE_CLOSE_FINISHED = 5u
};

/*
 * Exactly one authoritative, resetless identity-domain record belongs to one
 * host lifecycle-control scope.  It may create several Runtime roots and all
 * of their descendants, but it and its State/Ticket records must never be
 * copied into, reset as, or mixed with another scope.  A browser may satisfy
 * this with one private WASM control instance; native adapters must keep one
 * persistent domain and serialize domain -> parent -> child mutation in that
 * order.  The core, rather than either host adapter, owns the checked monotonic
 * identity sequence.  UINT64_MAX is a valid final identity;
 * identity_exhausted distinguishes "next is UINT64_MAX" from "UINT64_MAX was
 * already issued".  Existing owners never consult the domain while settling,
 * so exhaustion can reject creation without wedging accepted work or close.
 */
typedef struct VxRuntimeLifecycleDomainV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t identity_exhausted;
    uint32_t reserved;
    uint64_t next_owner_identity;
    uint64_t issued_owner_identities;
} VxRuntimeLifecycleDomainV1;

typedef struct VxRuntimeLifecycleStateV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t owner_kind;
    uint32_t admission_mode;
    uint32_t phase;
    uint32_t close_state;
    uint32_t close_result;
    uint32_t ticket_exhausted;
    uint64_t identity;
    uint64_t parent_identity;
    /* Saturating diagnostic serial; never an admission/settlement gate. */
    uint64_t transition_id;
    uint64_t next_ticket;
    uint64_t serving_ticket;
    uint64_t active_ticket;
    uint64_t close_ticket;
    uint64_t admitted_operations;
    uint64_t active_operations;
    uint64_t child_count;
    uint32_t parent_pin_held;
    uint32_t reserved;
} VxRuntimeLifecycleStateV1;

typedef struct VxRuntimeLifecycleTicketV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t operation;
    uint32_t phase;
    uint64_t owner_identity;
    uint64_t ticket;
    uint64_t integrity_tag;
} VxRuntimeLifecycleTicketV1;

#if defined(__cplusplus)
#define VX_RUNTIME_LIFECYCLE_DOMAIN_V1_INIT {}
#define VX_RUNTIME_LIFECYCLE_STATE_V1_INIT {}
#define VX_RUNTIME_LIFECYCLE_TICKET_V1_INIT {}
#else
#define VX_RUNTIME_LIFECYCLE_DOMAIN_V1_INIT {0}
#define VX_RUNTIME_LIFECYCLE_STATE_V1_INIT {0}
#define VX_RUNTIME_LIFECYCLE_TICKET_V1_INIT {0}
#endif

#if defined(__cplusplus)
static_assert(sizeof(VxRuntimeLifecycleDomainV1) == 32,
              "portable runtime-lifecycle domain ABI drift");
static_assert(sizeof(VxRuntimeLifecycleStateV1) == 120,
              "portable runtime-lifecycle state ABI drift");
static_assert(sizeof(VxRuntimeLifecycleTicketV1) == 40,
              "portable runtime-lifecycle ticket ABI drift");
#else
_Static_assert(sizeof(VxRuntimeLifecycleDomainV1) == 32,
               "portable runtime-lifecycle domain ABI drift");
_Static_assert(sizeof(VxRuntimeLifecycleStateV1) == 120,
               "portable runtime-lifecycle state ABI drift");
_Static_assert(sizeof(VxRuntimeLifecycleTicketV1) == 40,
               "portable runtime-lifecycle ticket ABI drift");
#endif

VX_RUNTIME_LIFECYCLE_CORE_API uint32_t
vx_runtime_lifecycle_core_abi_version(void);

/* Domain initialization is zero-initialized-once and intentionally has no
 * reset/reopen operation. */
VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_domain_init_v1(
    VxRuntimeLifecycleDomainV1* domain);

VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_domain_validate_v1(
    const VxRuntimeLifecycleDomainV1* domain);

/* Only Runtime is a root.  CompiledModel and ExecutionContext can be created
 * only through the parent-pinning child initializer below.  The adapter must
 * serialize creation in domain -> parent -> child order; successful child
 * creation publishes its identity, parent count, and pin atomically. */
VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_state_init_root_v1(
    VxRuntimeLifecycleDomainV1* domain,
    VxRuntimeLifecycleStateV1* state);

/* Atomically publishes a child pin before asynchronous child construction.
 * The only accepted ownership edges are Runtime -> CompiledModel and
 * CompiledModel -> ExecutionContext. */
VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_state_init_child_v1(
    VxRuntimeLifecycleDomainV1* domain,
    VxRuntimeLifecycleStateV1* parent,
    VxRuntimeLifecycleStateV1* child,
    VxRuntimeLifecycleOwnerKindV1 child_kind);

VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_state_validate_v1(
    const VxRuntimeLifecycleStateV1* state);

VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_ticket_validate_v1(
    const VxRuntimeLifecycleTicketV1* ticket);

/* Admission owns one operation count immediately.  For a FIFO owner, begin
 * returns WAIT until this exact ticket is serving.  Parallel owners may have
 * several active tickets.  Every successful begin must be completed even when
 * the host operation fails; failure is an operation outcome, not queue poison. */
VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_operation_admit_v1(
    VxRuntimeLifecycleStateV1* state,
    VxRuntimeLifecycleOperationV1 operation,
    VxRuntimeLifecycleTicketV1* ticket);

VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_operation_begin_v1(
    VxRuntimeLifecycleStateV1* state,
    VxRuntimeLifecycleTicketV1* ticket);

VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_operation_complete_v1(
    VxRuntimeLifecycleStateV1* state,
    VxRuntimeLifecycleTicketV1* ticket);

/* close_request is the logical close boundary and is idempotent.  It rejects
 * all later operation/child admission.  Repeated requests return a projection
 * of the one canonical close ticket. */
VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_close_request_v1(
    VxRuntimeLifecycleStateV1* state,
    VxRuntimeLifecycleTicketV1* close_ticket);

VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_close_readiness_v1(
    const VxRuntimeLifecycleStateV1* state,
    VxRuntimeLifecycleCloseReadinessV1* readiness);

/* close_begin grants the physical host teardown boundary exactly once.
 * close_complete closes the logical owner even when host teardown failed; the
 * result bit lets an adapter preserve its language-specific close outcome. */
VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_close_begin_v1(
    VxRuntimeLifecycleStateV1* state,
    VxRuntimeLifecycleTicketV1* close_ticket);

VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_close_complete_v1(
    VxRuntimeLifecycleStateV1* state,
    VxRuntimeLifecycleTicketV1* close_ticket,
    VxRuntimeLifecycleCloseResultV1 result);

/* A child releases its parent pin only after the child reached CLOSED.  This
 * transition is idempotence-sensitive: a second release is DUPLICATE and can
 * never underflow the parent count. */
VX_RUNTIME_LIFECYCLE_CORE_API VxRuntimeLifecycleStatusV1
vx_runtime_lifecycle_parent_pin_release_v1(
    VxRuntimeLifecycleStateV1* parent,
    VxRuntimeLifecycleStateV1* child);

#ifdef __cplusplus
}
#endif

#endif
