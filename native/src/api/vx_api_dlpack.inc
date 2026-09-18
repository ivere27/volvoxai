/* Standard C DLPack ownership. Python only adapts PyCapsule at its boundary. */
#include "../../third_party/dlpack/dlpack.h"

#if !defined(__wasm__)
typedef struct {
    void* managed;
    int versioned, armed, device_retained;
    VxNativeStorage storage;
} VxDLPackImport;
static void vx_dl_import_release(void* pointer) {
    VxDLPackImport* owner = pointer;
    /* A foreign deleter may recycle its allocation immediately. Finish only
     * this allocation's recorded work before giving it back to that producer. */
    if (owner->storage.ops && owner->storage.ops->release_completion)
        owner->storage.ops->release_completion(&owner->storage);
    if (owner->armed) {
        if (owner->versioned) {
            DLManagedTensorVersioned* tensor = owner->managed;
            if (tensor->deleter) tensor->deleter(tensor);
        } else {
            DLManagedTensor* tensor = owner->managed;
            if (tensor->deleter) tensor->deleter(tensor);
        }
    }
    if (owner->device_retained) owner->storage.ops->release_external();
    free(owner);
}
static VxDataType vx_dl_dtype(DLDataType dtype) {
    if (dtype.lanes != 1) return VX_DTYPE_UNSPECIFIED;
    if (dtype.code == kDLFloat) switch (dtype.bits) {
        case 16: return VX_DTYPE_F16; case 32: return VX_DTYPE_F32; case 64: return VX_DTYPE_F64;
    }
    if (dtype.code == kDLInt) switch (dtype.bits) {
        case 8: return VX_DTYPE_I8; case 16: return VX_DTYPE_I16; case 32: return VX_DTYPE_I32; case 64: return VX_DTYPE_I64;
    }
    if (dtype.code == kDLUInt) switch (dtype.bits) {
        case 8: return VX_DTYPE_U8; case 16: return VX_DTYPE_U16; case 32: return VX_DTYPE_U32; case 64: return VX_DTYPE_U64;
    }
    if (dtype.code == kDLBfloat && dtype.bits == 16) return VX_DTYPE_BF16;
    if (dtype.code == kDLBool && dtype.bits == 8) return VX_DTYPE_BOOL;
    return VX_DTYPE_UNSPECIFIED;
}
static DLDataType vx_dl_type(VxDataType dtype) {
    DLDataType result = {.lanes = 1, .bits = (uint8_t)safetensors_dtype_bit_width(dtype), .code = 255};
    switch (dtype) {
        case VX_DTYPE_F16: case VX_DTYPE_F32: case VX_DTYPE_F64: result.code = kDLFloat; break;
        case VX_DTYPE_I8: case VX_DTYPE_I16: case VX_DTYPE_I32: case VX_DTYPE_I64: result.code = kDLInt; break;
        case VX_DTYPE_U8: case VX_DTYPE_U16: case VX_DTYPE_U32: case VX_DTYPE_U64: result.code = kDLUInt; break;
        case VX_DTYPE_BF16: result.code = kDLBfloat; break;
        case VX_DTYPE_BOOL: result.code = kDLBool; break;
        default: break;
    }
    return result;
}
typedef struct {
    DLManagedTensor legacy;
    DLManagedTensorVersioned versioned;
    int64_t shape[VX_MAX_TENSOR_RANK];
    VxBufferAccessLease* access;
} VxDLPackExport;
static void vx_dl_export_delete(DLManagedTensor* tensor) {
    VxDLPackExport* owner = tensor->manager_ctx;
    vx_buffer_access_release(owner->access);
    free(owner);
}
static void vx_dl_export_delete_versioned(DLManagedTensorVersioned* tensor) {
    VxDLPackExport* owner = tensor->manager_ctx;
    vx_buffer_access_release(owner->access);
    free(owner);
}
#endif

static int vx_buffer_import_dlpack(const VolvoxaiV1ImportDLPackRequest* request,
    VolvoxaiV1TensorBatch* response, void* registry) {
#if defined(__wasm__)
    (void)request; (void)registry;
    return vx_buffer_report(response->_allocator, &response->field_report, VX_STATUS_TRANSPORT_UNSUPPORTED);
#else
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    VxDLPackImport* owner = NULL;
    VxApiBuffer* buffer = NULL;
    int64_t id = 0;
    if (!request->field_managed_tensor || request->field_managed_tensor > UINTPTR_MAX ||
        request->field_managed_tensor % _Alignof(DLManagedTensor)) goto done;
    void* managed = (void*)(uintptr_t)request->field_managed_tensor;
    DLTensor* tensor;
    int read_only = 0;
    if (request->field_versioned) {
        DLManagedTensorVersioned* value = managed;
        if (value->version.major != DLPACK_MAJOR_VERSION ||
            (value->flags & ~(DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED))) goto done;
        tensor = &value->dl_tensor;
        read_only = (value->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
    } else tensor = &((DLManagedTensor*)managed)->dl_tensor;
    if (!tensor->data || tensor->ndim < 0 || tensor->ndim > VX_MAX_TENSOR_RANK ||
        (tensor->ndim && !tensor->shape)) goto done;
    VxDataType dtype = vx_dl_dtype(tensor->dtype);
    if (dtype == VX_DTYPE_UNSPECIFIED) goto done;
    size_t elements = 1, itemsize = tensor->dtype.bits / 8;
    for (int i = tensor->ndim; i-- > 0;) {
        int64_t extent = tensor->shape[i];
        if (extent <= 0 || (uint64_t)extent > SIZE_MAX / elements ||
            (tensor->strides && extent > 1 && (uint64_t)tensor->strides[i] != elements)) goto done;
        elements *= (size_t)extent;
    }
    if (!itemsize || elements > SIZE_MAX / itemsize) goto done;
    size_t bytes = elements * itemsize;
    if (tensor->byte_offset > UINTPTR_MAX - (uintptr_t)tensor->data ||
        bytes > UINTPTR_MAX - (uintptr_t)tensor->data - tensor->byte_offset ||
        ((uintptr_t)tensor->data + tensor->byte_offset) % itemsize) goto done;
    VxNativeBuffer memory = {.handle = (uint64_t)(uintptr_t)tensor->data,
        .offset = tensor->byte_offset, .length = bytes, .device_id = tensor->device.device_id};
    if (tensor->device.device_type == kDLCPU && !tensor->device.device_id) memory.kind = VX_NATIVE_BUFFER_HOST;
    else if (tensor->device.device_type == kDLCUDA) memory.kind = VX_NATIVE_BUFFER_CUDA;
    else { status = VX_STATUS_BACKEND_UNSUPPORTED; goto done; }
    if (!vx_native_buffer_validate(vx_buffer_backend(memory.kind), &memory)) goto done;
    owner = calloc(1, sizeof(*owner));
    if (!owner) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    owner->managed = managed;
    owner->versioned = request->field_versioned;
    owner->storage.ops = vx_native_tensor_ops(vx_buffer_backend(memory.kind));
    owner->storage.buffer = memory;
    owner->storage.allocation = (void*)(uintptr_t)(memory.handle + memory.offset);
    if (owner->storage.ops->retain_external) {
        if (!owner->storage.ops->retain_external(&memory)) {
            free(owner); owner = NULL; status = VX_STATUS_BACKEND_UNSUPPORTED; goto done;
        }
        owner->device_retained = 1;
    }
    buffer = vx_api_buffer_create(&memory, &owner->storage, owner, vx_dl_import_release, read_only);
    if (!buffer) { vx_dl_import_release(owner); owner = NULL; status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    VolvoxaiV1Tensor* output = volvoxai_v1_tensor_batch_add_outputs(response);
    id = output ? vx_api_buffer_publish(registry, buffer) : 0;
    if (!id) { vx_api_buffer_release(buffer); owner = NULL; status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    info.dtype = dtype; info.byte_size = bytes; info.rank = (uint32_t)tensor->ndim;
    info.location = memory.kind == VX_NATIVE_BUFFER_HOST ? VX_MEMORY_HOST : VX_MEMORY_DEVICE;
    if (info.rank) memcpy(info.shape, tensor->shape, info.rank * sizeof(int64_t));
    if (!vx_api_buffer_tensor(response->_allocator, output, &info, id)) {
        vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER, id); owner = NULL;
        status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    /* Build the success report before consuming the foreign deleter. */
    if (vx_buffer_report(response->_allocator, &response->field_report, VX_STATUS_OK)) {
        vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER, id); return -1;
    }
    owner->armed = 1;
    return 0;
done:
    return vx_buffer_report(response->_allocator, &response->field_report, status);
#endif
}
static int vx_buffer_export_dlpack(const VolvoxaiV1ExportDLPackRequest* request,
    VolvoxaiV1DLPackExport* response, void* registry) {
#if defined(__wasm__)
    (void)request; (void)registry;
    return vx_buffer_report(response->_allocator, &response->field_report, VX_STATUS_TRANSPORT_UNSUPPORTED);
#else
    size_t bytes;
    VxStatus status = vx_api_tensor_bytes(request->field_tensor, &bytes);
    VxBufferAccessLease* access = NULL;
    VxNativeBuffer memory;
    if (status != VX_STATUS_OK || request->field_tensor->which_payload != 6) {
        status = VX_STATUS_INVALID_ARGUMENT; goto done;
    }
    status = request->field_access_id
        ? vx_buffer_access_join(registry, request->field_tensor->field_buffer, request->field_access_id, &access, &memory)
        : vx_buffer_view(registry, request->field_tensor->field_buffer, 1, &access, &memory);
    if (status != VX_STATUS_OK) goto done;
    if (memory.length != bytes) { status = VX_STATUS_INVALID_ARGUMENT; goto done; }
    if (memory.kind != VX_NATIVE_BUFFER_HOST && memory.kind != VX_NATIVE_BUFFER_CUDA) {
        status = VX_STATUS_BACKEND_UNSUPPORTED; goto done;
    }
    DLDataType dtype = vx_dl_type((VxDataType)request->field_tensor->field_dtype);
    if (dtype.code == 255 || (memory.handle + memory.offset) % (dtype.bits / 8)) {
        status = VX_STATUS_INVALID_ARGUMENT; goto done;
    }
    if (!vx_buffer_wait(access->buffer)) { status = VX_STATUS_EXECUTION_FAILED; goto done; }
    VxDLPackExport* owner = calloc(1, sizeof(*owner));
    if (!owner) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    access->external = 1;
    owner->access = access;
    DLTensor tensor = {.data = (void*)(uintptr_t)memory.handle,
        .device = {memory.kind == VX_NATIVE_BUFFER_HOST ? kDLCPU : kDLCUDA, memory.device_id},
        .dtype = dtype, .ndim = (int)request->field_tensor->field_shape.len,
        .shape = owner->shape, .byte_offset = memory.offset};
    if (tensor.ndim) memcpy(owner->shape, request->field_tensor->field_shape.data, tensor.ndim * sizeof(int64_t));
    owner->legacy = (DLManagedTensor){tensor, owner, vx_dl_export_delete};
    owner->versioned = (DLManagedTensorVersioned){.version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION},
        .manager_ctx = owner, .deleter = vx_dl_export_delete_versioned, .dl_tensor = tensor};
    if (vx_buffer_report(response->_allocator, &response->field_report, VX_STATUS_OK)) {
        free(owner); vx_buffer_access_release(access); return -1;
    }
    response->field_versioned = request->field_versioned;
    response->field_managed_tensor = (uint64_t)(uintptr_t)(request->field_versioned
        ? (void*)&owner->versioned : (void*)&owner->legacy);
    return 0;
done:
    if (access) vx_buffer_access_release(access);
    return vx_buffer_report(response->_allocator, &response->field_report, status);
#endif
}
