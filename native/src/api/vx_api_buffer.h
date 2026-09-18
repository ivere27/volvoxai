/* Common storage ownership, independent of execution and training. */
#ifndef VOLVOXAI_API_BUFFER_H
#define VOLVOXAI_API_BUFFER_H
#include "vx_api_convert.h"
#include "vx_api_handles.h"
#include "native_tensor.h"

typedef struct VxApiBuffer VxApiBuffer;
/* On success takes owner; on failure leaves it with the caller. */
VxApiBuffer* vx_api_buffer_create(const VxNativeBuffer* memory,
    VxNativeStorage* storage, void* owner, void (*release)(void*), int read_only);
void vx_api_buffer_retain(void* buffer);
void vx_api_buffer_release(void* buffer);
int64_t vx_api_buffer_publish(VxApiRegistry* registry, VxApiBuffer* buffer);
VxStatus vx_api_buffer_resolve(VxApiScratch* scratch, const VolvoxaiV1BufferView* view,
    VxNativeBuffer* memory);
VxStatus vx_api_borrowed_native(const VolvoxaiV1BorrowedBuffer* view, VxNativeBuffer* memory);
int vx_api_buffer_tensor(const SynurangLiteAllocator* allocator, VolvoxaiV1Tensor* tensor,
    const VxTensorInfo* info, int64_t id);
int vx_api_resource_descriptor(const SynurangLiteAllocator* allocator,
    VolvoxaiV1NativeResource** slot, const VxNativeBuffer* memory);
int vx_api_borrowed_descriptor(const SynurangLiteAllocator* allocator,
    VolvoxaiV1BorrowedBuffer** slot, const VxNativeBuffer* memory);
VxStatus vx_api_tensor_bytes(const VolvoxaiV1Tensor* tensor, size_t* bytes);
VxStatus vx_api_tensor_read(VxApiScratch* scratch, const VolvoxaiV1Tensor* tensor,
    void* host, size_t bytes);
#endif
