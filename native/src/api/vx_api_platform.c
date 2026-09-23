/* VxPlatformService — process-scoped, handle-free diagnostics.
 *
 * Every handler returns 0 and reports domain outcomes inside its response.
 * A non-zero return is reserved for failures that cannot produce a response
 * at all, which the generated dispatch turns into a transport error.
 */
#include "vx_api_convert.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"

#include <string.h>

#include "vx_api_contract.inc"

/* Backends compiled into this library. The engine has no runtime enumeration
 * entry point, so this mirrors the link-time toggles in native/CMakeLists.txt
 * rather than guessing. CPU is always present. */
static const char* const vx_api_compiled_backends[] = {
#if defined(__wasm__)
    "wasm",
#else
    "cpu",
#endif
#if defined(VOLVOXAI_ENABLE_VULKAN) && VOLVOXAI_ENABLE_VULKAN
    "vulkan",
#endif
#if defined(VOLVOXAI_ENABLE_OPENGL) && VOLVOXAI_ENABLE_OPENGL
    "opengl",
#endif
#if defined(VOLVOXAI_ENABLE_METAL) && VOLVOXAI_ENABLE_METAL
    "metal",
#endif
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    "cuda",
#endif
#if defined(VOLVOXAI_ENABLE_WEBGPU) && VOLVOXAI_ENABLE_WEBGPU
    "webgpu",
#endif
};

#define VX_API_COMPILED_BACKEND_COUNT \
    (sizeof(vx_api_compiled_backends) / sizeof(vx_api_compiled_backends[0]))

const char* const* vx_api_compiled_backend_names(size_t* count) {
    if (count) *count = VX_API_COMPILED_BACKEND_COUNT;
    return vx_api_compiled_backends;
}

static int vx_api_get_platform_info(const VolvoxaiV1Empty* request,
                                    VolvoxaiV1PlatformInfo* response,
                                    void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    SynurangLiteBytes* backends;
    size_t index;

    (void)request;
    response->field_api_version = VX_NATIVE_API_VERSION;
    response->field_max_tensor_rank = VX_MAX_TENSOR_RANK;
    /* An in-process library always shares the caller's address space, so a
     * BufferView payload is safe to map. A remote transport in front of this
     * engine must advertise REMOTE itself. */
    response->field_transport =
#if defined(__wasm__)
        VOLVOXAI_V1_TRANSPORT_PROFILE_REMOTE;
#else
        VOLVOXAI_V1_TRANSPORT_PROFILE_IN_PROCESS;
#endif
    /* VOLVOXAI_ENABLE_TRAINING is the profile switch every target already
     * sets, and it is what decides whether the Training and Quantization
     * handler tables get registered. Reporting anything else here would let a
     * caller ask a full library for PTQ and be told it is an inference
     * build. */
#if defined(VOLVOXAI_ENABLE_TRAINING) && VOLVOXAI_ENABLE_TRAINING
    response->field_profile = VOLVOXAI_V1_BUILD_PROFILE_FULL;
#else
    response->field_profile = VOLVOXAI_V1_BUILD_PROFILE_INFERENCE;
#endif
    if (synurang_lite_bytes_assign(allocator, &response->field_library_version,
                                   VOLVOXAI_VERSION,
                                   strlen(VOLVOXAI_VERSION)) !=
        SYNURANG_LITE_OK) {
        return -1;
    }

    backends = (SynurangLiteBytes*)allocator->allocate(
        allocator->context, sizeof(*backends) * VX_API_COMPILED_BACKEND_COUNT);
    if (!backends) return -1;
    memset(backends, 0, sizeof(*backends) * VX_API_COMPILED_BACKEND_COUNT);
    response->field_compiled_backends.data = backends;
    response->field_compiled_backends.len = VX_API_COMPILED_BACKEND_COUNT;
    response->field_compiled_backends.cap = VX_API_COMPILED_BACKEND_COUNT;
    for (index = 0; index < VX_API_COMPILED_BACKEND_COUNT; index++) {
        const char* name = vx_api_compiled_backends[index];
        if (synurang_lite_bytes_assign(allocator, &backends[index], name,
                                       strlen(name)) != SYNURANG_LITE_OK) {
            return -1;
        }
    }
    return 0;
}

static int vx_api_get_monotonic_time(const VolvoxaiV1Empty* request,
                                     VolvoxaiV1MonotonicTime* response,
                                     void* user_data) {
    (void)user_data;
    (void)request;
    response->field_nanoseconds = vx_runtime_monotonic_time_micros() * UINT64_C(1000);
    return 0;
}

static int vx_api_describe_status(const VolvoxaiV1DescribeStatusRequest* request,
                                  VolvoxaiV1StatusDescription* response,
                                  void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    const char* text = vx_status_string((VxStatus)request->field_status);

    response->field_status = request->field_status;
    if (!text) text = "UNKNOWN";
    if (synurang_lite_bytes_assign(allocator, &response->field_name, text,
                                   strlen(text)) != SYNURANG_LITE_OK) {
        return -1;
    }
    /* The engine exposes only the canonical token; a longer prose description
     * would have to be invented here, so it is deliberately left empty. */
    return 0;
}

VX_API_UNARY(vx_api_get_platform_info, VolvoxaiV1Empty, VolvoxaiV1PlatformInfo,
    volvoxai_v1_platform_info, vx_platform_get_platform_info_respond)
VX_API_UNARY(vx_api_describe_api, VolvoxaiV1DescribeApiRequest, VolvoxaiV1ApiDescription,
    volvoxai_v1_api_description, vx_platform_describe_api_respond)
VX_API_UNARY(vx_api_get_monotonic_time, VolvoxaiV1Empty, VolvoxaiV1MonotonicTime,
    volvoxai_v1_monotonic_time, vx_platform_get_monotonic_time_respond)
VX_API_UNARY(vx_api_describe_status, VolvoxaiV1DescribeStatusRequest, VolvoxaiV1StatusDescription,
    volvoxai_v1_status_description, vx_platform_describe_status_respond)

int vx_api_install_platform_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxPlatformServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.get_platform_info.message = vx_api_get_platform_info_call;
    handlers.describe_api.message = vx_api_describe_api_call;
    handlers.get_monotonic_time.message = vx_api_get_monotonic_time_call;
    handlers.describe_status.message = vx_api_describe_status_call;
    return vx_platform_register(instance, &handlers, registry);
}
