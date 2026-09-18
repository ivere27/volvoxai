/* Owner-scoped storage and explicit external access. No training dependency. */
#include "vx_api_buffer.h"
#include "vx_api.h"
#include "volvoxai_ffi.h"
#include "safetensors.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct VxApiBuffer {
    atomic_uint references;
    /* -1: exclusive external writer; >=0: readers including accepted calls. */
    atomic_int access;
    VxNativeBuffer memory;
    VxNativeStorage* storage;
    void* owner;
    void (*release_owner)(void*);
    int read_only;
};

typedef struct {
    atomic_uint references;
    atomic_int ended;
    VxApiBuffer* buffer;
    uint64_t offset, length;
    int write;
    int external;
} VxBufferAccessLease;

VxApiBuffer* vx_api_buffer_create(const VxNativeBuffer* memory,
    VxNativeStorage* storage, void* owner, void (*release)(void*), int read_only) {
    VxApiBuffer* buffer = calloc(1, sizeof(*buffer));
    if (!buffer) return NULL;
    atomic_init(&buffer->references, 1);
    atomic_init(&buffer->access, 0);
    buffer->memory = *memory;
    buffer->storage = storage;
    buffer->owner = owner;
    buffer->release_owner = release;
    buffer->read_only = read_only;
    return buffer;
}
void vx_api_buffer_retain(void* pointer) {
    VxApiBuffer* buffer = pointer;
    atomic_fetch_add_explicit(&buffer->references, 1, memory_order_relaxed);
}
void vx_api_buffer_release(void* pointer) {
    VxApiBuffer* buffer = pointer;
    if (buffer && atomic_fetch_sub_explicit(&buffer->references, 1, memory_order_acq_rel) == 1) {
        if (buffer->release_owner) buffer->release_owner(buffer->owner);
        free(buffer);
    }
}
int64_t vx_api_buffer_publish(VxApiRegistry* registry, VxApiBuffer* buffer) {
    return vx_api_handle_insert(registry, VX_API_HANDLE_BUFFER, buffer,
        vx_api_buffer_retain, vx_api_buffer_release);
}
static VxBackendKind vx_buffer_backend(VxNativeBufferKind kind) {
    switch (kind) {
        case VX_NATIVE_BUFFER_HOST: return VX_BACKEND_KIND_NATIVE_CPU;
        case VX_NATIVE_BUFFER_CUDA: return VX_BACKEND_KIND_CUDA;
        case VX_NATIVE_BUFFER_VULKAN: return VX_BACKEND_KIND_VULKAN;
        case VX_NATIVE_BUFFER_OPENGL: return VX_BACKEND_KIND_OPENGL;
        case VX_NATIVE_BUFFER_METAL: return VX_BACKEND_KIND_METAL;
        default: return VX_BACKEND_KIND_UNSPECIFIED;
    }
}
static int vx_buffer_wait(VxApiBuffer* buffer) {
    VxNativeStorage* storage = buffer->storage;
    return !storage || !storage->ops->wait || storage->ops->wait(storage, 1);
}
static void vx_buffer_access_retain(void* pointer) {
    VxBufferAccessLease* access = pointer;
    atomic_fetch_add_explicit(&access->references, 1, memory_order_relaxed);
}
static void vx_buffer_access_end_permission(VxBufferAccessLease* access) {
    if (atomic_exchange_explicit(&access->ended, 1, memory_order_acq_rel)) return;
    if (access->write) atomic_store_explicit(&access->buffer->access, 0, memory_order_release);
    else atomic_fetch_sub_explicit(&access->buffer->access, 1, memory_order_release);
}
static void vx_buffer_access_release(void* pointer) {
    VxBufferAccessLease* access = pointer;
    if (atomic_fetch_sub_explicit(&access->references, 1, memory_order_acq_rel) != 1) return;
    /* On device loss no further operation on that device is admissible. */
    VxNativeStorage* storage = access->buffer->storage;
    if (!atomic_load_explicit(&access->ended, memory_order_acquire) &&
        access->external && storage && storage->ops->finish_external)
        (void)storage->ops->finish_external();
    vx_buffer_access_end_permission(access);
    vx_api_buffer_release(access->buffer);
    free(access);
}
static VxStatus vx_buffer_access_take(VxApiBuffer* buffer, int write, VxBufferAccessLease** out) {
    if (write && buffer->read_only) return VX_STATUS_INVALID_ARGUMENT;
    VxBufferAccessLease* access = calloc(1, sizeof(*access));
    if (!access) return VX_STATUS_OUT_OF_MEMORY;
    int current = atomic_load_explicit(&buffer->access, memory_order_acquire);
    for (;;) {
        if (current < 0 || current == INT_MAX || (write && current)) { free(access); return VX_STATUS_BUSY; }
        if (atomic_compare_exchange_weak_explicit(&buffer->access, &current,
                write ? -1 : current + 1, memory_order_acq_rel, memory_order_acquire)) break;
    }
    atomic_init(&access->references, 1);
    atomic_init(&access->ended, 0);
    access->buffer = buffer;
    access->write = write;
    vx_api_buffer_retain(buffer);
    *out = access;
    return VX_STATUS_OK;
}
static VxStatus vx_buffer_view(VxApiRegistry* registry, const VolvoxaiV1BufferView* view,
    int write, VxBufferAccessLease** access, VxNativeBuffer* memory) {
    VxApiHandleLease handle = VX_API_HANDLE_LEASE_INIT;
    if (!view) return VX_STATUS_INVALID_ARGUMENT;
    if (!registry || !vx_api_handle_acquire(registry, VX_API_HANDLE_BUFFER, view->field_buffer_id, &handle))
        return VX_STATUS_HANDLE_DISPOSED;
    VxApiBuffer* buffer = handle.pointer;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    if (view->field_offset_bytes <= buffer->memory.length &&
        view->field_length_bytes <= buffer->memory.length - view->field_offset_bytes) {
        status = vx_buffer_access_take(buffer, write, access);
        if (status == VX_STATUS_OK) {
            (*access)->offset = view->field_offset_bytes;
            (*access)->length = view->field_length_bytes;
            *memory = buffer->memory;
            memory->offset += view->field_offset_bytes;
            memory->length = view->field_length_bytes;
        }
    }
    vx_api_handle_lease_release(&handle);
    return status;
}
/* Reuse a registered permission for scoped DLPack export. The capsule retains
 * allocation lifetime after EndBufferAccess has retired the permission. */
static VxStatus vx_buffer_access_join(VxApiRegistry* registry, const VolvoxaiV1BufferView* view,
    int64_t id, VxBufferAccessLease** out, VxNativeBuffer* memory) {
    VxApiHandleLease buffer_handle = VX_API_HANDLE_LEASE_INIT, access_handle = VX_API_HANDLE_LEASE_INIT;
    VxStatus status = VX_STATUS_HANDLE_DISPOSED;
    if (!view) return VX_STATUS_INVALID_ARGUMENT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_BUFFER, view->field_buffer_id, &buffer_handle) ||
        !vx_api_handle_acquire(registry, VX_API_HANDLE_BUFFER_ACCESS, id, &access_handle)) goto done;
    VxBufferAccessLease* access = access_handle.pointer;
    if (atomic_load_explicit(&access->ended, memory_order_acquire)) goto done;
    status = VX_STATUS_INVALID_ARGUMENT;
    if (!access->write || !access->external || access->buffer != buffer_handle.pointer ||
        view->field_offset_bytes < access->offset ||
        view->field_offset_bytes - access->offset > access->length ||
        view->field_length_bytes > access->length - (view->field_offset_bytes - access->offset)) goto done;
    *memory = access->buffer->memory;
    memory->offset += view->field_offset_bytes;
    memory->length = view->field_length_bytes;
    vx_buffer_access_retain(access);
    *out = access;
    status = VX_STATUS_OK;
done:
    vx_api_handle_lease_release(&access_handle);
    vx_api_handle_lease_release(&buffer_handle);
    return status;
}
VxStatus vx_api_buffer_resolve(VxApiScratch* scratch, const VolvoxaiV1BufferView* view, VxNativeBuffer* memory) {
    VxBufferAccessLease* access = NULL;
    VxStatus status = vx_buffer_view(scratch->registry, view, 0, &access, memory);
    if (status == VX_STATUS_OK) {
        memory->dependency = access->buffer->storage;
        /* Host bindings are consumed directly. Device dependencies must be
         * ordered later, on the execution context's actual submission queue. */
        if (memory->kind == VX_NATIVE_BUFFER_HOST && !vx_buffer_wait(access->buffer)) {
            vx_buffer_access_release(access);
            return VX_STATUS_EXECUTION_FAILED;
        }
    }
    if (status == VX_STATUS_OK && !vx_api_scratch_cleanup(scratch, vx_buffer_access_release, access)) {
        vx_buffer_access_release(access);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    return status;
}
VxStatus vx_api_borrowed_native(const VolvoxaiV1BorrowedBuffer* view, VxNativeBuffer* memory) {
#if defined(__wasm__)
    (void)view; (void)memory;
    return VX_STATUS_TRANSPORT_UNSUPPORTED;
#else
    if (!view || !view->field_resource) return VX_STATUS_INVALID_ARGUMENT;
    const VolvoxaiV1NativeResource* resource = view->field_resource;
    if (!resource->field_handle || resource->field_kind < VOLVOXAI_V1_NATIVE_RESOURCE_KIND_HOST ||
        resource->field_kind > VOLVOXAI_V1_NATIVE_RESOURCE_KIND_METAL ||
        resource->field_size_bytes > SIZE_MAX || view->field_offset_bytes > resource->field_size_bytes ||
        view->field_length_bytes > resource->field_size_bytes - view->field_offset_bytes)
        return VX_STATUS_INVALID_ARGUMENT;
    if (resource->field_kind == VOLVOXAI_V1_NATIVE_RESOURCE_KIND_HOST &&
        (resource->field_device_id || resource->field_device_context ||
         resource->field_handle > UINTPTR_MAX || resource->field_size_bytes > UINTPTR_MAX - resource->field_handle))
        return VX_STATUS_INVALID_ARGUMENT;
    *memory = (VxNativeBuffer){(VxNativeBufferKind)resource->field_kind,
        resource->field_handle, view->field_offset_bytes, view->field_length_bytes,
        resource->field_device_id, resource->field_device_context};
    return VX_STATUS_OK;
#endif
}
int vx_api_resource_descriptor(const SynurangLiteAllocator* allocator,
    VolvoxaiV1NativeResource** slot, const VxNativeBuffer* memory) {
    *slot = allocator->allocate(allocator->context, sizeof(**slot));
    if (!*slot) return 0;
    volvoxai_v1_native_resource_init_with_allocator(*slot, allocator);
    (*slot)->field_kind = (VolvoxaiV1NativeResourceKind)memory->kind;
    (*slot)->field_handle = memory->handle;
    (*slot)->field_size_bytes = memory->offset + memory->length;
    (*slot)->field_device_id = memory->device_id;
    (*slot)->field_device_context = memory->device_context;
    return 1;
}
int vx_api_borrowed_descriptor(const SynurangLiteAllocator* allocator,
    VolvoxaiV1BorrowedBuffer** slot, const VxNativeBuffer* memory) {
    *slot = allocator->allocate(allocator->context, sizeof(**slot));
    if (!*slot) return 0;
    volvoxai_v1_borrowed_buffer_init_with_allocator(*slot, allocator);
    (*slot)->field_offset_bytes = memory->offset;
    (*slot)->field_length_bytes = memory->length;
    return vx_api_resource_descriptor(allocator, &(*slot)->field_resource, memory);
}
int vx_api_buffer_tensor(const SynurangLiteAllocator* allocator, VolvoxaiV1Tensor* tensor,
    const VxTensorInfo* info, int64_t id) {
    tensor->field_dtype = (VolvoxaiV1DataType)info->dtype;
    if (info->name && synurang_lite_bytes_assign(allocator, &tensor->field_name,
        info->name, strlen(info->name)) != SYNURANG_LITE_OK) return 0;
    for (size_t i = 0; i < info->rank; i++) {
        int64_t* axis = volvoxai_v1_tensor_add_shape(tensor);
        if (!axis) return 0;
        *axis = info->shape[i];
    }
    if (id) {
        tensor->which_payload = 6;
        tensor->field_buffer = allocator->allocate(allocator->context, sizeof(*tensor->field_buffer));
        if (!tensor->field_buffer) return 0;
        volvoxai_v1_buffer_view_init_with_allocator(tensor->field_buffer, allocator);
        tensor->field_buffer->field_buffer_id = id;
        tensor->field_buffer->field_length_bytes = info->byte_size;
    }
    return 1;
}
VxStatus vx_api_tensor_bytes(const VolvoxaiV1Tensor* tensor, size_t* bytes) {
    if (!tensor || tensor->field_shape.len > VX_MAX_TENSOR_RANK) return VX_STATUS_INVALID_ARGUMENT;
    size_t bits = safetensors_dtype_bit_width((VxDataType)tensor->field_dtype), size = 1;
    if (!bits || bits % 8) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < tensor->field_shape.len; i++) {
        int64_t dim = tensor->field_shape.data[i];
        if (dim <= 0 || (uint64_t)dim > SIZE_MAX / size) return VX_STATUS_INVALID_ARGUMENT;
        size *= (size_t)dim;
    }
    if (size > SIZE_MAX / (bits / 8)) return VX_STATUS_INVALID_ARGUMENT;
    *bytes = size * (bits / 8);
    return VX_STATUS_OK;
}
static VxStatus vx_buffer_read_range(VxApiBuffer* buffer, const VxNativeBuffer* memory, void* host) {
    if (memory->kind == VX_NATIVE_BUFFER_HOST) {
        memmove(host, (const void*)(uintptr_t)(memory->handle + memory->offset), (size_t)memory->length);
        return VX_STATUS_OK;
    }
    if (!buffer->storage) return VX_STATUS_BACKEND_UNSUPPORTED;
    /* Existing backend readback adapters read whole logical allocations. A
     * sliced read is explicit staging, never an alleged zero-copy mapping. */
    size_t offset = (size_t)(memory->offset - buffer->memory.offset);
    if (!offset && memory->length == buffer->memory.length)
        return vx_native_storage_read(buffer->storage, host, (size_t)memory->length)
            ? VX_STATUS_OK : VX_STATUS_EXECUTION_FAILED;
    void* staging = malloc((size_t)buffer->memory.length);
    if (!staging) return VX_STATUS_OUT_OF_MEMORY;
    int ok = vx_native_storage_read(buffer->storage, staging, (size_t)buffer->memory.length);
    if (ok) memcpy(host, (char*)staging + offset, (size_t)memory->length);
    free(staging);
    return ok ? VX_STATUS_OK : VX_STATUS_EXECUTION_FAILED;
}
VxStatus vx_api_tensor_read(VxApiScratch* scratch, const VolvoxaiV1Tensor* tensor, void* host, size_t bytes) {
    size_t expected;
    VxStatus status = vx_api_tensor_bytes(tensor, &expected);
    if (status != VX_STATUS_OK || expected != bytes) return VX_STATUS_INVALID_ARGUMENT;
    if (tensor->which_payload == 5) {
        if (tensor->field_inline.len != bytes) return VX_STATUS_INVALID_ARGUMENT;
        memcpy(host, tensor->field_inline.data, bytes);
        return VX_STATUS_OK;
    }
    VxNativeBuffer memory;
    if (tensor->which_payload == 7) {
        status = vx_api_borrowed_native(tensor->field_borrowed, &memory);
        if (status != VX_STATUS_OK) return status;
        if (memory.length != bytes) return VX_STATUS_INVALID_ARGUMENT;
        if (memory.kind != VX_NATIVE_BUFFER_HOST) return VX_STATUS_BACKEND_UNSUPPORTED;
        memmove(host, (void*)(uintptr_t)(memory.handle + memory.offset), bytes);
        return VX_STATUS_OK;
    }
    if (tensor->which_payload != 6) return VX_STATUS_INVALID_ARGUMENT;
    VxBufferAccessLease* access = NULL;
    status = vx_buffer_view(scratch->registry, tensor->field_buffer, 0, &access, &memory);
    if (status != VX_STATUS_OK) return status;
    status = memory.length == bytes ? vx_buffer_read_range(access->buffer, &memory, host) : VX_STATUS_INVALID_ARGUMENT;
    vx_buffer_access_release(access);
    return status;
}

static int vx_buffer_report(const SynurangLiteAllocator* a, VolvoxaiV1OperationReport** slot, VxStatus status) {
    VxOperationCode code = status == VX_STATUS_OK ? VX_CODE_NONE : status == VX_STATUS_HANDLE_DISPOSED
        ? VX_CODE_HANDLE_DISPOSED : status == VX_STATUS_TRANSPORT_UNSUPPORTED ? VX_CODE_TRANSPORT_UNSUPPORTED
        : status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : status == VX_STATUS_BACKEND_UNSUPPORTED
        ? VX_CODE_BACKEND_UNSUPPORTED : status == VX_STATUS_BUSY ? VX_CODE_BUSY : VX_CODE_INVALID_ARGUMENT;
    return vx_api_report_fail(a, slot, status, VX_STAGE_READBACK, code,
        status == VX_STATUS_OK ? "buffer operation completed" : status == VX_STATUS_BUSY
        ? "buffer has conflicting external access" : "buffer operation could not be completed") ? 0 : -1;
}
static int vx_buffer_get_info(const VolvoxaiV1BufferHandle* request, VolvoxaiV1BufferInfo* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_BUFFER, request->field_buffer_id, &lease))
        return vx_buffer_report(response->_allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED);
    VxApiBuffer* buffer = lease.pointer;
    response->field_size_bytes = buffer->memory.length;
    response->field_kind = (VolvoxaiV1NativeResourceKind)buffer->memory.kind;
    response->field_device_id = buffer->memory.device_id;
    response->field_device_context = buffer->memory.device_context;
    response->field_cpu_accessible = buffer->memory.kind == VX_NATIVE_BUFFER_HOST;
    if (!response->field_cpu_accessible && buffer->storage && buffer->storage->ops->host_view) {
        VxNativeBuffer mapping;
        response->field_cpu_accessible = buffer->storage->ops->host_view(buffer->storage, &mapping);
    }
    response->field_read_only = buffer->read_only;
    vx_api_handle_lease_release(&lease);
    return vx_buffer_report(response->_allocator, &response->field_report, VX_STATUS_OK);
}
static VxApiBuffer* vx_buffer_host(void* data, size_t bytes) {
    VxNativeBuffer memory = {.kind = VX_NATIVE_BUFFER_HOST, .handle = (uint64_t)(uintptr_t)data, .length = bytes};
    return vx_api_buffer_create(&memory, NULL, data, free, 0);
}
static void vx_buffer_handles_unwind(VxApiRegistry* registry, VolvoxaiV1BufferHandles* response) {
    for (size_t i = 0; i < response->field_buffers.len; i++) {
        vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER, response->field_buffers.data[i].field_buffer_id);
        response->field_buffers.data[i].field_buffer_id = 0;
    }
    response->field_buffers.len = 0;
}
static int vx_buffer_allocate(const VolvoxaiV1AllocateBuffersRequest* request, VolvoxaiV1BufferHandles* response, void* registry) {
    VxStatus status = VX_STATUS_OK;
    for (size_t i = 0; i < request->field_sizes_bytes.len; i++) {
        uint64_t bytes = request->field_sizes_bytes.data[i];
        if (!bytes || bytes > SIZE_MAX) { status = VX_STATUS_INVALID_ARGUMENT; break; }
        void* data = calloc(1, (size_t)bytes);
        VxApiBuffer* buffer = data ? vx_buffer_host(data, (size_t)bytes) : NULL;
        if (!buffer) { free(data); status = VX_STATUS_OUT_OF_MEMORY; break; }
        VolvoxaiV1BufferHandle* handle = volvoxai_v1_buffer_handles_add_buffers(response);
        int64_t id = handle ? vx_api_buffer_publish(registry, buffer) : 0;
        if (!id) { vx_api_buffer_release(buffer); status = VX_STATUS_OUT_OF_MEMORY; break; }
        handle->field_buffer_id = id;
    }
    if (status != VX_STATUS_OK) vx_buffer_handles_unwind(registry, response);
    return vx_buffer_report(response->_allocator, &response->field_report, status);
}
static int vx_buffer_retain(const VolvoxaiV1BufferRefs* request, VolvoxaiV1BufferHandles* response, void* registry) {
    VxStatus status = VX_STATUS_OK;
    for (size_t i = 0; i < request->field_buffer_ids.len; i++) {
        VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
        if (!vx_api_handle_acquire(registry, VX_API_HANDLE_BUFFER, request->field_buffer_ids.data[i], &lease)) {
            status = VX_STATUS_HANDLE_DISPOSED; break;
        }
        VolvoxaiV1BufferHandle* handle = volvoxai_v1_buffer_handles_add_buffers(response);
        int64_t id = handle ? vx_api_buffer_publish(registry, lease.pointer) : 0;
        if (!id) { vx_api_handle_lease_release(&lease); status = VX_STATUS_OUT_OF_MEMORY; break; }
        /* Registry owns the acquired reference. */
        handle->field_buffer_id = id;
    }
    if (status != VX_STATUS_OK) vx_buffer_handles_unwind(registry, response);
    return vx_buffer_report(response->_allocator, &response->field_report, status);
}
static int vx_buffer_release(const VolvoxaiV1BufferRefs* request, VolvoxaiV1OperationReport* response, void* registry) {
    for (size_t i = 0; i < request->field_buffer_ids.len; i++)
        vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER, request->field_buffer_ids.data[i]);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}
static int vx_buffer_begin_access(const VolvoxaiV1BufferAccessRequest* request, VolvoxaiV1BufferAccess* response, void* registry) {
#if defined(__wasm__)
    (void)request; (void)registry;
    return vx_buffer_report(response->_allocator, &response->field_report, VX_STATUS_TRANSPORT_UNSUPPORTED);
#else
    VxBufferAccessLease* access = NULL;
    VxNativeBuffer memory;
    VxStatus status = request->field_mode == VOLVOXAI_V1_BUFFER_ACCESS_MODE_READ ||
        request->field_mode == VOLVOXAI_V1_BUFFER_ACCESS_MODE_WRITE ?
        vx_buffer_view(registry, request->field_view, request->field_mode == VOLVOXAI_V1_BUFFER_ACCESS_MODE_WRITE,
            &access, &memory) : VX_STATUS_INVALID_ARGUMENT;
    if (status != VX_STATUS_OK) goto done;
    if (request->field_host_mapping && memory.kind != VX_NATIVE_BUFFER_HOST) {
        VxNativeStorage* storage = access->buffer->storage;
        VxNativeBuffer mapping;
        if (!storage || !storage->ops->host_view || !storage->ops->host_view(storage, &mapping)) {
            status = VX_STATUS_BACKEND_UNSUPPORTED; goto done;
        }
        mapping.offset += memory.offset - access->buffer->memory.offset;
        mapping.length = memory.length;
        memory = mapping;
    }
    if (!vx_buffer_wait(access->buffer)) { status = VX_STATUS_EXECUTION_FAILED; goto done; }
    if (!vx_api_borrowed_descriptor(response->_allocator, &response->field_memory, &memory)) {
        status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    access->external = 1;
    response->field_access_id = vx_api_handle_insert(registry, VX_API_HANDLE_BUFFER_ACCESS,
        access, vx_buffer_access_retain, vx_buffer_access_release);
    if (!response->field_access_id) status = VX_STATUS_OUT_OF_MEMORY;
    else access = NULL;
done:
    if (access) vx_buffer_access_release(access);
    return vx_buffer_report(response->_allocator, &response->field_report, status);
#endif
}
static int vx_buffer_end_access(const VolvoxaiV1EndBufferAccessRequest* request, VolvoxaiV1OperationReport* response, void* registry) {
    VxStatus status = VX_STATUS_OK;
    VxApiHandleLease* leases = NULL;
    size_t count = request->field_access_ids.len;
    const VolvoxaiV1CudaStreamCompletion* cuda = request->field_cuda;
#if defined(__wasm__)
    if (cuda) { status = VX_STATUS_TRANSPORT_UNSUPPORTED; goto done; }
#endif
    if (cuda) {
        if (cuda->field_device_id < 0 || !cuda->field_streams.len || cuda->field_streams.len > 64) {
            status = VX_STATUS_INVALID_ARGUMENT; goto done;
        }
        for (size_t i = 0; i < cuda->field_streams.len; i++) {
            uint64_t stream = cuda->field_streams.data[i];
            if (!stream || stream == 2 || stream > UINTPTR_MAX) { status = VX_STATUS_INVALID_ARGUMENT; goto done; }
        }
    }
    if (count > SIZE_MAX / sizeof(*leases) || (count && !(leases = calloc(count, sizeof(*leases))))) {
        status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    /* Validate the complete batch before recording dependencies or retiring any
     * permission. Registry operation references also protect foreign deleters. */
    for (size_t i = 0; i < count; i++) {
        if (!vx_api_handle_acquire(registry, VX_API_HANDLE_BUFFER_ACCESS,
                request->field_access_ids.data[i], &leases[i])) continue;
        VxBufferAccessLease* access = leases[i].pointer;
        if (cuda && (access->buffer->memory.kind != VX_NATIVE_BUFFER_CUDA ||
                access->buffer->memory.device_id != cuda->field_device_id ||
                !access->buffer->storage || !access->buffer->storage->ops->complete_external)) {
            status = VX_STATUS_INVALID_ARGUMENT; goto done;
        }
        for (size_t j = 0; j < i; j++) {
            if (leases[j].pointer == access) { vx_api_handle_lease_release(&leases[i]); break; }
        }
    }
    for (size_t i = 0; i < count; i++) {
        VxBufferAccessLease* access = leases[i].pointer;
        if (!access || atomic_load_explicit(&access->ended, memory_order_acquire)) continue;
        VxNativeStorage* storage = access->buffer->storage;
        if (cuda) status = storage->ops->complete_external(storage, cuda->field_streams.data, cuda->field_streams.len);
        else if (access->external && storage && storage->ops->finish_external && !storage->ops->finish_external())
            status = VX_STATUS_EXECUTION_FAILED;
        if (status != VX_STATUS_OK) goto done;
    }
    for (size_t i = 0; i < count; i++) {
        if (leases[i].pointer) vx_buffer_access_end_permission(leases[i].pointer);
        vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER_ACCESS, request->field_access_ids.data[i]);
    }
done:
    if (leases) for (size_t i = 0; i < count; i++) vx_api_handle_lease_release(&leases[i]);
    free(leases);
    response->field_status = (VolvoxaiV1NativeStatus)status;
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}
/* The common array readback performs exactly one data copy. Multi-source
 * copies retain staging for simultaneous overlap semantics. */
static int vx_buffer_copy_into_one(const VolvoxaiV1CopyTensorsRequest* request,
    VolvoxaiV1TensorBatch* response, VxApiRegistry* registry) {
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(registry);
    const VolvoxaiV1Tensor* source = &request->field_sources.data[0];
    size_t bytes = 0, capacity = 0;
    void* destination = NULL;
    VxStatus status = vx_api_tensor_bytes(source, &bytes);
    if (status == VX_STATUS_OK) status = vx_api_borrowed_resolve(&request->field_into.data[0], &destination, &capacity);
    if (status == VX_STATUS_OK && capacity < bytes) status = VX_STATUS_BUFFER_TOO_SMALL;
    if (status == VX_STATUS_OK) {
        VolvoxaiV1Tensor* target = volvoxai_v1_tensor_batch_add_outputs(response);
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        info.name = vx_api_scratch_cstr(&scratch, &source->field_name);
        info.dtype = (VxDataType)source->field_dtype;
        info.rank = (uint32_t)source->field_shape.len;
        if (info.rank) memcpy(info.shape, source->field_shape.data, info.rank * sizeof(int64_t));
        info.byte_size = bytes;
        VxNativeBuffer memory;
        status = vx_api_borrowed_native(&request->field_into.data[0], &memory);
        memory.length = bytes;
        if (!target || !vx_api_buffer_tensor(response->_allocator, target, &info, 0)) status = VX_STATUS_OUT_OF_MEMORY;
        if (status == VX_STATUS_OK) {
            target->which_payload = 7;
            if (!vx_api_borrowed_descriptor(response->_allocator, &target->field_borrowed, &memory)) status = VX_STATUS_OUT_OF_MEMORY;
        }
        if (status == VX_STATUS_OK) status = vx_api_tensor_read(&scratch, source, destination, bytes);
    }
    vx_api_scratch_release(&scratch);
    return vx_buffer_report(response->_allocator, &response->field_report, status);
}

static int vx_buffer_copy(const VolvoxaiV1CopyTensorsRequest* request, VolvoxaiV1TensorBatch* response, void* registry) {
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(registry);
    VxStatus status = VX_STATUS_OK;
    size_t count = request->field_sources.len;
    if (count == 1 && request->field_into.len == 1 && !request->field_inline_result)
        return vx_buffer_copy_into_one(request, response, registry);
    if ((request->field_into.len && request->field_into.len != count) ||
        (request->field_into.len && request->field_inline_result)) status = VX_STATUS_INVALID_ARGUMENT;
    void** staging = count ? vx_api_scratch_alloc(&scratch, count * sizeof(*staging)) : NULL;
    size_t* sizes = count ? vx_api_scratch_alloc(&scratch, count * sizeof(*sizes)) : NULL;
    void** destinations = count ? vx_api_scratch_alloc(&scratch, count * sizeof(*destinations)) : NULL;
    if (count && (!staging || !sizes || !destinations)) status = VX_STATUS_OUT_OF_MEMORY;
    /* Read all sources before writes, so aliases and overlapping batches have
     * simultaneous-copy semantics. Allocation/publication also precedes writes. */
    for (size_t i = 0; i < count && status == VX_STATUS_OK; i++) {
        const VolvoxaiV1Tensor* source = &request->field_sources.data[i];
        status = vx_api_tensor_bytes(source, &sizes[i]);
        if (status != VX_STATUS_OK) break;
        staging[i] = vx_api_scratch_alloc(&scratch, sizes[i]);
        if (!staging[i]) { status = VX_STATUS_OUT_OF_MEMORY; break; }
        status = vx_api_tensor_read(&scratch, source, staging[i], sizes[i]);
        if (status != VX_STATUS_OK) break;
        if (request->field_into.len) {
            size_t capacity;
            status = vx_api_borrowed_resolve(&request->field_into.data[i], &destinations[i], &capacity);
            if (status == VX_STATUS_OK && capacity < sizes[i]) status = VX_STATUS_BUFFER_TOO_SMALL;
        }
    }
    for (size_t i = 0; i < count && status == VX_STATUS_OK; i++) {
        const VolvoxaiV1Tensor* source = &request->field_sources.data[i];
        VolvoxaiV1Tensor* target = volvoxai_v1_tensor_batch_add_outputs(response);
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        info.name = vx_api_scratch_cstr(&scratch, &source->field_name);
        info.dtype = (VxDataType)source->field_dtype;
        info.rank = (uint32_t)source->field_shape.len;
        if (info.rank) memcpy(info.shape, source->field_shape.data, info.rank * sizeof(int64_t));
        info.byte_size = sizes[i];
        if (!target || !vx_api_buffer_tensor(response->_allocator, target, &info, 0)) {
            status = VX_STATUS_OUT_OF_MEMORY; break;
        }
        if (request->field_inline_result) {
            target->which_payload = 5;
            if (synurang_lite_bytes_assign(response->_allocator, &target->field_inline, staging[i], sizes[i]) != SYNURANG_LITE_OK)
                status = VX_STATUS_OUT_OF_MEMORY;
        } else if (request->field_into.len) {
            VxNativeBuffer memory;
            status = vx_api_borrowed_native(&request->field_into.data[i], &memory);
            memory.length = sizes[i];
            target->which_payload = 7;
            if (status == VX_STATUS_OK && !vx_api_borrowed_descriptor(response->_allocator, &target->field_borrowed, &memory))
                status = VX_STATUS_OUT_OF_MEMORY;
        } else {
            void* data = malloc(sizes[i]);
            if (!data) { status = VX_STATUS_OUT_OF_MEMORY; break; }
            memcpy(data, staging[i], sizes[i]);
            VxApiBuffer* buffer = vx_buffer_host(data, sizes[i]);
            if (!buffer) { free(data); status = VX_STATUS_OUT_OF_MEMORY; break; }
            int64_t id = vx_api_buffer_publish(registry, buffer);
            if (!id) { vx_api_buffer_release(buffer); status = VX_STATUS_OUT_OF_MEMORY; break; }
            target->which_payload = 6;
            target->field_buffer = response->_allocator->allocate(response->_allocator->context, sizeof(*target->field_buffer));
            if (!target->field_buffer) {
                vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER, id); status = VX_STATUS_OUT_OF_MEMORY; break;
            }
            volvoxai_v1_buffer_view_init_with_allocator(target->field_buffer, response->_allocator);
            target->field_buffer->field_buffer_id = id;
            target->field_buffer->field_length_bytes = sizes[i];
        }
    }
    if (status == VX_STATUS_OK && request->field_into.len)
        for (size_t i = 0; i < count; i++) memcpy(destinations[i], staging[i], sizes[i]);
    if (status != VX_STATUS_OK) {
        for (size_t i = 0; i < response->field_outputs.len; i++) {
            VolvoxaiV1Tensor* tensor = &response->field_outputs.data[i];
            if (tensor->which_payload == 6 && tensor->field_buffer) {
                vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER, tensor->field_buffer->field_buffer_id);
                tensor->field_buffer->field_buffer_id = 0;
            }
        }
    }
    vx_api_scratch_release(&scratch);
    return vx_buffer_report(response->_allocator, &response->field_report, status);
}

#include "vx_api_dlpack.inc"

VX_API_UNARY(vx_buffer_get_info, VolvoxaiV1BufferHandle, VolvoxaiV1BufferInfo, volvoxai_v1_buffer_info, vx_buffer_get_buffer_info_respond)
VX_API_UNARY(vx_buffer_allocate, VolvoxaiV1AllocateBuffersRequest, VolvoxaiV1BufferHandles, volvoxai_v1_buffer_handles, vx_buffer_allocate_buffers_respond)
VX_API_UNARY(vx_buffer_retain, VolvoxaiV1BufferRefs, VolvoxaiV1BufferHandles, volvoxai_v1_buffer_handles, vx_buffer_retain_buffers_respond)
VX_API_UNARY(vx_buffer_release, VolvoxaiV1BufferRefs, VolvoxaiV1OperationReport, volvoxai_v1_operation_report, vx_buffer_release_buffers_respond)
VX_API_UNARY(vx_buffer_copy, VolvoxaiV1CopyTensorsRequest, VolvoxaiV1TensorBatch, volvoxai_v1_tensor_batch, vx_buffer_copy_tensors_respond)
VX_API_UNARY(vx_buffer_begin_access, VolvoxaiV1BufferAccessRequest, VolvoxaiV1BufferAccess, volvoxai_v1_buffer_access, vx_buffer_begin_buffer_access_respond)
VX_API_UNARY(vx_buffer_end_access, VolvoxaiV1EndBufferAccessRequest, VolvoxaiV1OperationReport, volvoxai_v1_operation_report, vx_buffer_end_buffer_access_respond)
/* Raw DLPack ownership must also unwind if response construction/write fails.
 * The local caller consumes its capsule only after receiving a success reply. */
static void vx_buffer_import_dlpack_call(SynurangStream* stream,
    const VolvoxaiV1ImportDLPackRequest* request, void* registry) {
    VolvoxaiV1TensorBatch response;
    volvoxai_v1_tensor_batch_init(&response);
    int result = vx_buffer_import_dlpack(request, &response, registry);
    SynurangStatus status = result ? SYNURANG_INTERNAL : vx_buffer_import_d_l_pack_respond(stream, &response);
#if !defined(__wasm__)
    if (status != SYNURANG_OK && response.field_outputs.len) {
        VolvoxaiV1Tensor* tensor = &response.field_outputs.data[0];
        if (tensor->field_buffer && tensor->field_buffer->field_buffer_id) {
            int64_t id = tensor->field_buffer->field_buffer_id;
            VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
            if (vx_api_handle_acquire(registry, VX_API_HANDLE_BUFFER, id, &lease)) {
                VxApiBuffer* buffer = lease.pointer;
                ((VxDLPackImport*)buffer->owner)->armed = 0;
                vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER, id);
                vx_api_handle_lease_release(&lease);
            }
        }
    }
#endif
    volvoxai_v1_tensor_batch_free(&response);
    if (status == SYNURANG_OK) (void)synurang_stream_finish(stream);
    else (void)synurang_stream_fail_error(stream, status, 13, "DLPack import response failed");
}
static void vx_buffer_export_dlpack_call(SynurangStream* stream,
    const VolvoxaiV1ExportDLPackRequest* request, void* registry) {
    VolvoxaiV1DLPackExport response;
    volvoxai_v1_dl_pack_export_init(&response);
    int result = vx_buffer_export_dlpack(request, &response, registry);
    SynurangStatus status = result ? SYNURANG_INTERNAL : vx_buffer_export_d_l_pack_respond(stream, &response);
#if !defined(__wasm__)
    if (status != SYNURANG_OK && response.field_managed_tensor) {
        if (response.field_versioned) {
            DLManagedTensorVersioned* tensor = (void*)(uintptr_t)response.field_managed_tensor;
            tensor->deleter(tensor);
        } else {
            DLManagedTensor* tensor = (void*)(uintptr_t)response.field_managed_tensor;
            tensor->deleter(tensor);
        }
    }
#endif
    volvoxai_v1_dl_pack_export_free(&response);
    if (status == SYNURANG_OK) (void)synurang_stream_finish(stream);
    else (void)synurang_stream_fail_error(stream, status, 13, "DLPack export response failed");
}
int vx_api_install_buffer_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxBufferServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.get_buffer_info.message = vx_buffer_get_info_call;
    handlers.allocate_buffers.message = vx_buffer_allocate_call;
    handlers.retain_buffers.message = vx_buffer_retain_call;
    handlers.release_buffers.message = vx_buffer_release_call;
    handlers.copy_tensors.message = vx_buffer_copy_call;
    handlers.begin_buffer_access.message = vx_buffer_begin_access_call;
    handlers.end_buffer_access.message = vx_buffer_end_access_call;
    handlers.import_d_l_pack.message = vx_buffer_import_dlpack_call;
    handlers.export_d_l_pack.message = vx_buffer_export_dlpack_call;
    return vx_buffer_register(instance, &handlers, registry);
}
