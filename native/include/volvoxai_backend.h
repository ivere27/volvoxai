#ifndef VOLVOXAI_BACKEND_H
#define VOLVOXAI_BACKEND_H

#include "volvoxai.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VX_BACKEND_ABI_VERSION UINT32_C(1)
#define VX_BACKEND_NAME_CAPACITY 64u

/* Provider-host service provider interface (SPI). These callbacks compose a
 * native runtime implementation; they are not application-facing Synurang FFI
 * operations. Shared statuses, stages, dtypes, and policies come from the
 * protobuf-derived declarations included through volvoxai.h. */

/* The sink copies each output before write() returns. A provider must write
 * every declared output exactly once with the model's exact name, execution
 * dtype, rank, dimensions, and byte size. Any mismatch rejects the complete
 * execution result. */
typedef struct VxBackendOutputSink {
    size_t struct_size;
    void* user_data;
    VxStatus (*write)(void* user_data,
                      const char* name,
                      VxDataType dtype,
                      const int64_t* shape,
                      uint32_t rank,
                      const void* data,
                      size_t byte_size);
} VxBackendOutputSink;

typedef struct VxBackendProvider {
    size_t struct_size;
    uint32_t abi_version;
    const char* name;
    void* user_data;
    uint32_t flags;

    VxStatus (*runtime_create)(void* user_data,
                               const VxRuntimeOptions* options,
                               void** out_runtime_instance,
                               VxReport* report);
    void (*runtime_destroy)(void* runtime_instance);

    /* A successful compile fills route_evidence and sets route_attested. When
     * operator_fallback is forbidden it must also leave
     * operator_fallback_used clear; the runtime rejects an unattested plan. */
    VxStatus (*compile)(void* runtime_instance,
                        const VxModelSource* source,
                        const VxBackendPolicy* policy,
                        void** out_compiled_instance,
                        VxReport* report);
    void (*compiled_destroy)(void* compiled_instance);

    VxStatus (*context_create)(void* compiled_instance,
                               const VxContextOptions* options,
                               void** out_context_instance,
                               VxReport* report);
    VxStatus (*context_set_input)(void* context_instance,
                                  const char* name,
                                  VxDataType dtype,
                                  const void* data,
                                  size_t byte_size,
                                  VxReport* report);
    VxStatus (*context_execute)(void* context_instance,
                                const VxBackendOutputSink* output_sink,
                                VxReport* report);
    /* Optional decode callbacks. A provider context created with decode
     * enabled returns BACKEND_UNSUPPORTED when the corresponding callback is
     * absent. Seed/step obey the same complete-output sink contract as
     * context_execute. */
    VxStatus (*context_decode_seed)(void* context_instance,
                                    const VxBackendOutputSink* output_sink,
                                    VxReport* report);
    VxStatus (*context_decode_step)(void* context_instance,
                                    int32_t position,
                                    const VxBackendOutputSink* output_sink,
                                    VxReport* report);
    VxStatus (*context_decode_reset)(void* context_instance,
                                     VxReport* report);
    /* Optional exact adapter rebind. package_path/version_name are borrowed
     * for the duration of the callback. A NULL package_path is a
     * provider-owned metadata route. */
    VxStatus (*context_select_adapter)(void* context_instance,
                                       uint64_t adapter_id,
                                       uint64_t adapter_revision,
                                       const char* package_path,
                                       const char* version_name,
                                       VxReport* report);
    /* Optional logical close. context_destroy remains required and runs once. */
    VxStatus (*context_close)(void* context_instance, VxReport* report);
    void (*context_destroy)(void* context_instance);
} VxBackendProvider;

/* Provider-host composition operation. The runtime copies the descriptor and
 * name, then creates exactly one provider runtime instance. Callback code and
 * user_data remain borrowed until the runtime and all retained descendants are
 * released. Names are canonical lower-case ASCII identifiers. The built-in
 * names cpu, vulkan, opengl, metal, nnapi, and cuda are reserved. Synurang
 * applications select already-composed names through protobuf BackendPolicy. */
VX_API VxStatus vx_runtime_register_provider(VxRuntime* runtime,
                                      const VxBackendProvider* provider,
                                      VxReport* report);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_BACKEND_H */
