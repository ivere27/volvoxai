#include "native_tensor.h"
#include <stdlib.h>
#include <string.h>
#include "vx_thread.h"

static pthread_mutex_t storage_mutex = PTHREAD_MUTEX_INITIALIZER;
static VxNativeStorage* storages;

/* Idle buffers belong to one execution context. Live results keep the pool
 * object alive after close, but never keep its idle allocations alive. */
#define VX_NATIVE_POOL_BYTES (64u * 1024u * 1024u)
#define VX_NATIVE_POOL_SLOTS 64u
struct VxNativePool {
    atomic_uint references;
    int closed;
    size_t bytes, count;
    VxNativeStorage* idle;
};

VxNativePool* vx_native_pool_create(void) {
    VxNativePool* pool = calloc(1, sizeof(*pool));
    if (pool) atomic_init(&pool->references, 1u);
    return pool;
}
static void pool_release(VxNativePool* pool) {
    if (pool && atomic_fetch_sub_explicit(&pool->references, 1u, memory_order_acq_rel) == 1u)
        free(pool);
}
static void storage_destroy(VxNativeStorage* storage) {
    VxNativePool* pool = storage->pool;
    pthread_mutex_lock(&storage_mutex);
    VxNativeStorage** link = &storages;
    while (*link && *link != storage) link = &(*link)->next;
    if (*link) *link = storage->next;
    pthread_mutex_unlock(&storage_mutex);
    storage->ops->destroy(storage);
    free(storage);
    pool_release(pool);
}
void vx_native_pool_close(VxNativePool* pool) {
    if (!pool) return;
    pthread_mutex_lock(&storage_mutex);
    pool->closed = 1;
    VxNativeStorage* idle = pool->idle;
    pool->idle = NULL;
    pool->bytes = pool->count = 0;
    pthread_mutex_unlock(&storage_mutex);
    while (idle) {
        VxNativeStorage* next = idle->pool_next;
        storage_destroy(idle);
        idle = next;
    }
    pool_release(pool);
}

static int host_device(VxNativeBuffer* view) {
    *view = (VxNativeBuffer){.kind = VX_NATIVE_BUFFER_HOST};
    return 1;
}
static int host_validate(const VxNativeBuffer* view) {
    return view->device_id == 0 && view->device_context == 0 &&
        view->handle && view->handle <= UINTPTR_MAX &&
        view->offset <= UINTPTR_MAX - view->handle &&
        view->length <= UINTPTR_MAX - view->handle - view->offset;
}
static int host_copy_input(const void* host, const VxNativeBuffer* view) {
    memcpy((void*)host, (const void*)(uintptr_t)(view->handle + view->offset), (size_t)view->length);
    return 1;
}
static int host_snapshot(const void* host, size_t bytes, VxNativeStorage* storage) {
    if (!storage->allocation) {
        storage->allocation = malloc(storage->capacity);
        storage->buffer.handle = (uint64_t)(uintptr_t)storage->allocation;
    }
    if (!storage->allocation) return 0;
    memcpy(storage->allocation, host, bytes);
    return 1;
}
static int host_read(const VxNativeStorage* storage, void* host, size_t bytes) {
    memcpy(host, storage->allocation, bytes);
    return 1;
}
static void host_destroy(VxNativeStorage* storage) { free(storage->allocation); }
static const VxNativeTensorOps host_ops = {
    VX_NATIVE_BUFFER_HOST, host_device, host_validate, host_copy_input,
    host_snapshot, host_read, host_destroy
};

#if VOLVOXAI_ENABLE_CUDA
extern const VxNativeTensorOps vx_cuda_tensor_ops;
#endif
#if VOLVOXAI_ENABLE_VULKAN
extern const VxNativeTensorOps vx_vulkan_tensor_ops;
#endif
#if VOLVOXAI_ENABLE_OPENGL
extern const VxNativeTensorOps vx_opengl_tensor_ops;
#endif
#if VOLVOXAI_ENABLE_METAL
extern const VxNativeTensorOps vx_metal_tensor_ops;
#endif

const VxNativeTensorOps* vx_native_tensor_ops(VxBackendKind backend) {
    switch (backend) {
        case VX_BACKEND_KIND_WASM:
        case VX_BACKEND_KIND_NATIVE_CPU: return &host_ops;
#if VOLVOXAI_ENABLE_CUDA
        case VX_BACKEND_KIND_CUDA: return &vx_cuda_tensor_ops;
#endif
#if VOLVOXAI_ENABLE_VULKAN
        case VX_BACKEND_KIND_VULKAN: return &vx_vulkan_tensor_ops;
#endif
#if VOLVOXAI_ENABLE_OPENGL
        case VX_BACKEND_KIND_OPENGL: return &vx_opengl_tensor_ops;
#endif
#if VOLVOXAI_ENABLE_METAL
        case VX_BACKEND_KIND_METAL: return &vx_metal_tensor_ops;
#endif
        default: return NULL;
    }
}
int vx_native_tensor_device(VxBackendKind backend, VxNativeBuffer* device) {
    const VxNativeTensorOps* ops = vx_native_tensor_ops(backend);
    return ops && device && ops->device(device);
}
int vx_native_tensor_view(VxBackendKind backend, const void* host, size_t bytes, VxNativeBuffer* view) {
    const VxNativeTensorOps* ops = vx_native_tensor_ops(backend);
    return ops && ops->view && ops->device(view) && ops->view(host, bytes, view);
}
static int contains(const VxNativeBuffer* owner, const VxNativeBuffer* view) {
    uint64_t offset = view->offset;
    if (owner->kind == VX_NATIVE_BUFFER_HOST || owner->kind == VX_NATIVE_BUFFER_CUDA) {
        if (view->handle < owner->handle || view->handle - owner->handle > UINT64_MAX - offset) return 0;
        offset += view->handle - owner->handle;
    } else if (owner->handle != view->handle) return 0;
    return owner->kind == view->kind && owner->device_id == view->device_id &&
        owner->device_context == view->device_context && offset >= owner->offset &&
        offset - owner->offset <= owner->length &&
        view->length <= owner->length - (offset - owner->offset);
}
VxNativeStorage* vx_native_storage_acquire(const VxNativeBuffer* view) {
    VxNativeStorage* found = NULL;
    pthread_mutex_lock(&storage_mutex);
    for (VxNativeStorage* item = storages; item; item = item->next) {
        if (atomic_load_explicit(&item->references, memory_order_relaxed) && contains(&item->buffer, view)) {
            atomic_fetch_add_explicit(&item->references, 1u, memory_order_relaxed);
            found = item;
            break;
        }
    }
    pthread_mutex_unlock(&storage_mutex);
    return found;
}
int vx_native_buffer_validate(VxBackendKind backend, const VxNativeBuffer* view) {
    const VxNativeTensorOps* ops = vx_native_tensor_ops(backend);
    VxNativeBuffer device = {0};
    if (!ops || !view || !view->handle || !view->length || view->length > SIZE_MAX ||
        view->offset > SIZE_MAX || view->offset > UINT64_MAX - view->length ||
        !ops->device(&device) || view->kind != device.kind ||
        view->device_id != device.device_id ||
        view->device_context != device.device_context) return 0;
    /* Buffer-to-buffer transfers require four-byte offsets on these APIs. */
    if ((view->kind == VX_NATIVE_BUFFER_VULKAN || view->kind == VX_NATIVE_BUFFER_METAL) &&
        (view->offset & 3u)) return 0;
    if (view->kind == VX_NATIVE_BUFFER_METAL && (view->length & 3u)) return 0;
    VxNativeStorage* owned = vx_native_storage_acquire(view);
    if (owned) { vx_native_storage_release(owned); return 1; }
    /* Rounded allocation capacity and idle pool memory are not public tensor
     * bytes. Never reinterpret an invalid owned view as an external pointer. */
    int known = 0;
    pthread_mutex_lock(&storage_mutex);
    for (VxNativeStorage* item = storages; item; item = item->next) {
        if (item->buffer.kind != view->kind) continue;
        known = item->buffer.handle == view->handle;
        if (!known && (view->kind == VX_NATIVE_BUFFER_HOST || view->kind == VX_NATIVE_BUFFER_CUDA))
            known = view->handle >= item->buffer.handle &&
                view->handle - item->buffer.handle < item->capacity;
        if (known) break;
    }
    pthread_mutex_unlock(&storage_mutex);
    if (known) return 0;
    return ops->validate_external && ops->validate_external(view);
}
VxNativeStorage* vx_native_storage_snapshot(VxNativePool* pool, VxBackendKind backend, const void* host, size_t bytes) {
    const VxNativeTensorOps* ops = vx_native_tensor_ops(backend);
    if (!ops || !host || !bytes) return NULL;
    VxNativeBuffer device = {0};
    if (!ops->device(&device)) return NULL;
    VxNativeStorage* storage = NULL;
    pthread_mutex_lock(&storage_mutex);
    if (pool && !pool->closed) {
        VxNativeStorage** best = NULL;
        for (VxNativeStorage** p = &pool->idle; *p; p = &(*p)->pool_next) {
            VxNativeStorage* item = *p;
            if (item->ops == ops && item->capacity >= bytes &&
                item->buffer.device_id == device.device_id &&
                item->buffer.device_context == device.device_context &&
                (!best || item->capacity < (*best)->capacity)) best = p;
        }
        if (best) {
            storage = *best;
            *best = storage->pool_next;
            pool->bytes -= storage->capacity;
            pool->count--;
        }
    }
    pthread_mutex_unlock(&storage_mutex);
    int fresh = !storage;
    if (fresh) {
        storage = calloc(1, sizeof(*storage));
        if (!storage) return NULL;
        storage->ops = ops;
        atomic_init(&storage->references, 0u);
        storage->buffer = device;
        storage->capacity = 256u;
        while (storage->capacity < bytes && storage->capacity <= SIZE_MAX / 2u)
            storage->capacity *= 2u;
        if (storage->capacity < bytes) storage->capacity = bytes;
        storage->pool = pool;
        if (pool) atomic_fetch_add_explicit(&pool->references, 1u, memory_order_relaxed);
    }
    storage->pool_next = NULL;
    if (!ops->snapshot(host, bytes, storage)) {
        /* Drain any queued copies before freeing their destinations. */
        if (ops->batch_end) (void)ops->batch_end();
        storage_destroy(storage);
        return NULL;
    }
    pthread_mutex_lock(&storage_mutex);
    storage->buffer.length = bytes;
    atomic_store_explicit(&storage->references, 1u, memory_order_release);
    if (fresh) {
        storage->next = storages;
        storages = storage;
    }
    pthread_mutex_unlock(&storage_mutex);
    return storage;
}
void vx_native_storage_release(VxNativeStorage* storage) {
    if (!storage) return;
    pthread_mutex_lock(&storage_mutex);
    int last = atomic_fetch_sub_explicit(&storage->references, 1u, memory_order_acq_rel) == 1u;
    if (last) {
        VxNativePool* pool = storage->pool;
        if (pool && !pool->closed && pool->count < VX_NATIVE_POOL_SLOTS &&
            storage->capacity <= VX_NATIVE_POOL_BYTES - pool->bytes) {
            storage->pool_next = pool->idle;
            pool->idle = storage;
            pool->bytes += storage->capacity;
            pool->count++;
            last = 0;
        }
    }
    pthread_mutex_unlock(&storage_mutex);
    if (last) storage_destroy(storage);
}
int vx_native_storage_read(const VxNativeStorage* storage, void* host, size_t bytes) {
    return storage && host && bytes <= storage->buffer.length && storage->ops->read(storage, host, bytes);
}
int vx_native_tensor_copy_input(VxBackendKind backend, const void* host, const VxNativeBuffer* source, int internal_view) {
    const VxNativeTensorOps* ops = vx_native_tensor_ops(backend);
    if (!ops || !host || !source || (!internal_view && !vx_native_buffer_validate(backend, source))) return 0;
    const VxNativeStorage* dependency = source->dependency;
    return (!dependency || !dependency->ops->wait || dependency->ops->wait(dependency, 0)) &&
        ops->copy_input(host, source);
}
int vx_native_tensor_batch_begin(VxBackendKind backend, int outputs) {
    const VxNativeTensorOps* ops = vx_native_tensor_ops(backend);
    return ops && (!ops->batch_begin || ops->batch_begin(outputs));
}
int vx_native_tensor_batch_end(VxBackendKind backend) {
    const VxNativeTensorOps* ops = vx_native_tensor_ops(backend);
    return ops && (!ops->batch_end || ops->batch_end());
}
