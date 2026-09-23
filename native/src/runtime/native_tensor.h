#ifndef VOLVOXAI_NATIVE_TENSOR_H
#define VOLVOXAI_NATIVE_TENSOR_H

/* Private backend-neutral storage. Public operations live in the proto and
 * generated dispatch; this is not another application ABI. */
#include "volvoxai_types.h"
#include "generated/backend_vocabulary.h"
#include <stdatomic.h>
#include "profiling.h"

typedef struct VxNativeStorage VxNativeStorage;
typedef struct VxNativePool VxNativePool;
typedef struct VxNativeTensorOps {
    VxNativeBufferKind kind;
    int (*device)(VxNativeBuffer* descriptor);
    int (*validate_external)(const VxNativeBuffer* source);
    int (*copy_input)(const void* host_identity, const VxNativeBuffer* source);
    int (*snapshot)(const void* host_identity, size_t bytes, VxNativeStorage* storage);
    int (*read)(const VxNativeStorage* storage, void* host, size_t bytes);
    /* Clear buffer.handle after releasing its API storage; a retained handle
     * reports failed retirement to optional memory accounting. */
    void (*destroy)(VxNativeStorage* storage);
    int (*batch_begin)(int outputs);
    int (*batch_end)(void);
    /* Private live graph view. Valid only inside a serialized context call
     * and across a rebind of its reserved physical domain. Never exported. */
    int (*view)(const void* host_identity, size_t bytes, VxNativeBuffer* view);
    /* A direct CPU mapping, without staging. Synchronization is separate. */
    int (*host_view)(const VxNativeStorage* storage, VxNativeBuffer* view);
    /* Pin backend state for imported storage without owning its allocation.
     * The producer's deleter owns the allocation itself. Both hooks are paired. */
    int (*retain_external)(const VxNativeBuffer* source);
    void (*release_external)(void);
    /* Wait for known producers, or conservatively finish untracked external
     * consumers. These are distinct from pool reuse dependency ordering. */
    int (*wait)(const VxNativeStorage* storage, int host);
    int (*finish_external)(void);
    /* Capture declared consumer streams into storage's completion dependency.
     * No host wait. Failure leaves the caller's access permission active. */
    VxStatus (*complete_external)(VxNativeStorage* storage, const uint64_t* streams, size_t count);
    /* Imported storage owns completion evidence but not its device allocation. */
    void (*release_completion)(VxNativeStorage* storage);
} VxNativeTensorOps;

struct VxNativeStorage {
    atomic_uint references;
    const VxNativeTensorOps* ops;
    VxNativeBuffer buffer;
    void* allocation;
    void* completion; /* Backend-owned producer dependency; never a public ID. */
    size_t capacity;
    VxNativePool* pool;
    VxNativeStorage* next;
    VxNativeStorage* pool_next;
};

const VxNativeTensorOps* vx_native_tensor_ops(VxBackendKind backend);
int vx_native_tensor_device(VxBackendKind backend, VxNativeBuffer* device);
int vx_native_buffer_validate(VxBackendKind backend, const VxNativeBuffer* buffer);
VxNativeStorage* vx_native_storage_acquire(const VxNativeBuffer* buffer);
VxNativePool* vx_native_pool_create(void);
void vx_native_pool_close(VxNativePool* pool);
void vx_native_pool_observe(VxNativePool* pool, const VxTraceScope* scope);
void vx_native_pool_memory(VxNativePool* pool, uint64_t* capacity, uint64_t* idle);
VxNativeStorage* vx_native_storage_snapshot(VxNativePool* pool, VxBackendKind backend, const void* host, size_t bytes);
void vx_native_storage_release(VxNativeStorage* storage);
int vx_native_storage_read(const VxNativeStorage* storage, void* host, size_t bytes);
int vx_native_tensor_copy_input(VxBackendKind backend, const void* host, const VxNativeBuffer* source, int internal_view);
int vx_native_tensor_batch_begin(VxBackendKind backend, int outputs);
int vx_native_tensor_batch_end(VxBackendKind backend);
int vx_native_tensor_view(VxBackendKind backend, const void* host, size_t bytes, VxNativeBuffer* view);

#endif
