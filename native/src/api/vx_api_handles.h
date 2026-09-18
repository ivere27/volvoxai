/* Opaque int64 handles for the protobuf API surface.
 *
 * proto/volvoxai.proto names every retained engine object with a plain int64.
 * Each module registry is the only place that maps such an id onto the engine's
 * opaque pointer. Ids are issued monotonically from one process-wide counter
 * and are never reused, so a stale id fails closed with HANDLE_DISPOSED
 * instead of aliasing a live object of the same kind.
 *
 * A successful insert() transfers one existing public reference to the
 * registry. acquire() takes an operation reference while holding the registry
 * mutex, so a concurrent remove() can retire the public id without destroying
 * an object that an accepted operation is still using. remove() drops the
 * registry's public reference after the id is no longer discoverable.
 */
#ifndef VOLVOXAI_API_HANDLES_H
#define VOLVOXAI_API_HANDLES_H

#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

typedef struct VxApiRegistry VxApiRegistry;
typedef struct VxRuntime VxRuntime;

/* A module owns one registry. Destroy only after its calls have drained. */
VxApiRegistry* vx_api_registry_create(void);
void vx_api_registry_destroy(VxApiRegistry* registry);
int vx_api_registry_track_runtime(VxApiRegistry* registry, VxRuntime* runtime);
void vx_api_registry_reap(VxApiRegistry* registry, uint32_t budget);

/* Private asynchronous call continuations. Only the module's serialized
 * entry thread changes the list; producers only set ready and post wakeups. */
typedef struct VxApiPending {
    struct VxApiPending* next;
    VxApiRegistry* registry;
    void (*poll)(struct VxApiPending* pending);
    atomic_int ready;
} VxApiPending;

void vx_api_registry_set_wakeup(VxApiRegistry* registry,
                                void (*wakeup)(void*), void* context);
void vx_api_pending_add(VxApiRegistry* registry, VxApiPending* pending);
void vx_api_pending_remove(VxApiPending* pending);
void vx_api_pending_notify(void* pending);
uint32_t vx_api_registry_poll(VxApiRegistry* registry, uint32_t budget);
int vx_api_registry_has_work(VxApiRegistry* registry);

typedef enum VxApiHandleKind {
    VX_API_HANDLE_RUNTIME = 0,
    VX_API_HANDLE_MODEL = 1,
    VX_API_HANDLE_COMPILED_MODEL = 2,
    VX_API_HANDLE_CONTEXT = 3,
    VX_API_HANDLE_RESULT = 4,
    VX_API_HANDLE_REQUEST = 5,
    VX_API_HANDLE_TRAINER = 6,
    VX_API_HANDLE_PTQ_PLAN = 7,
    VX_API_HANDLE_GRAPH_PLAN = 8,
    VX_API_HANDLE_TOKENIZER = 9,
    VX_API_HANDLE_BATCH_QUEUE = 10,
    VX_API_HANDLE_BUFFER = 11,
    VX_API_HANDLE_BUFFER_ACCESS = 12,
    VX_API_HANDLE_KIND_COUNT = 13
} VxApiHandleKind;

typedef void (*VxApiHandleRetainFn)(void* pointer);
typedef void (*VxApiHandleReleaseFn)(void* pointer);

/* Public protobuf handle ancestry. Core VxReport identities are deliberately
 * process-private and therefore cannot be exposed as the runtime/model/etc.
 * ids named by proto/volvoxai.proto. Every child copies this ancestry when it
 * enters the registry, so reports remain stable after a parent public handle
 * is released while the child still retains the engine object. */
typedef struct VxApiHandleLineage {
    uint64_t runtime_id;
    uint64_t model_id;
    uint64_t compiled_model_id;
    uint64_t context_id;
    uint64_t graph_plan_id;
} VxApiHandleLineage;

#define VX_API_HANDLE_LINEAGE_INIT { 0u, 0u, 0u, 0u, 0u }

typedef struct VxApiHandleLease {
    void* pointer;
    VxApiHandleReleaseFn release;
    VxApiHandleLineage lineage;
} VxApiHandleLease;

#define VX_API_HANDLE_LEASE_INIT { NULL, NULL, VX_API_HANDLE_LINEAGE_INIT }

/* Records pointer and its ownership callbacks under a fresh id. On success,
 * the registry owns the caller's existing reference. On failure, ownership
 * remains with the caller. A valid id is always positive. */
int64_t vx_api_handle_insert(VxApiRegistry* registry, VxApiHandleKind kind,
                             void* pointer,
                             VxApiHandleRetainFn retain,
                             VxApiHandleReleaseFn release);

/* As insert(), inheriting public handle ancestry from a parent. The newly
 * issued id is installed in the field corresponding to kind. */
int64_t vx_api_handle_insert_with_lineage(VxApiRegistry* registry,
    VxApiHandleKind kind,
    void* pointer,
    VxApiHandleRetainFn retain,
    VxApiHandleReleaseFn release,
    const VxApiHandleLineage* parent_lineage);

/* Acquires an operation lease. The retain callback runs before the registry
 * mutex is released, closing the lookup/remove race. Returns one on success
 * and zero for an unknown, wrong-kind, or retired id. */
int vx_api_handle_acquire(VxApiRegistry* registry, VxApiHandleKind kind,
                          int64_t id,
                          VxApiHandleLease* lease);

/* Drops an operation lease. Safe for an empty or already released lease. */
void vx_api_handle_lease_release(VxApiHandleLease* lease);

/* Atomically retires the public id and drops its registry-owned reference.
 * Returns one when a live id was retired and zero when it was already absent,
 * which makes protocol Release operations idempotent. */
int vx_api_handle_remove(VxApiRegistry* registry, VxApiHandleKind kind, int64_t id);

#endif /* VOLVOXAI_API_HANDLES_H */
