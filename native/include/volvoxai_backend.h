#ifndef VOLVOXAI_BACKEND_H
#define VOLVOXAI_BACKEND_H

#include "volvoxai.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque discriminator for the current exact provider ABI. This value is not
 * a compatibility generation: providers must use this header's exact
 * descriptor layouts and callbacks, even if an older header used the same
 * numeric value. */
#define VX_BACKEND_ABI_VERSION UINT32_C(1)
#define VX_BACKEND_NAME_CAPACITY 64u
#define VX_BACKEND_SHAPE_PROOF_PROTOCOL \
    "canonical-symbolic-domain-proof/v1"
#define VX_BACKEND_RESOURCE_PROTOCOL "bounded-resource-maxima/v1"
#define VX_BACKEND_BATCH_PROTOCOL "provider-batch-contract/v1"
#define VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL \
    "typed-independent-batch-proof/v1"
/* Update this opaque marker for every provider-contract layout or semantic
 * change, including same-size changes. It is intentionally not a public
 * compatibility generation. */
#define VX_BACKEND_PROVIDER_EXACT_CONTRACT_MARKER \
    UINT64_C(0x565850524f564944)

/* Provider callback descriptors use exact struct sizes.
 * Provider-host service provider interface (SPI). These callbacks compose a
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

/* Frozen contract for one exact provider-compiled executable. Opaque identity
 * pointers remain owned by the provider and stable until compiled_destroy.
 * Reporting max_batch > 1 is an attestation that context_execute_batch enters
 * the backend once for the whole invocation; a host loop is not a batch. It
 * also attests that its dense executor preserves the Runtime's exact
 * graph-fingerprint-bound independent-request proof. The Runtime owns the
 * typed semantic proof and bounded axis validation; the provider must echo its
 * identity and may only narrow the proved batch range. */
typedef struct VxBackendBatchContract {
    size_t struct_size;
    const char* protocol;
    /* Exact core-authored semantic proof echoed by the compiled provider.
     * A provider may narrow the proved range, but it cannot promote a graph
     * for which the core supplied no proof identity. */
    const char* graph_fingerprint;
    const char* independent_batch_proof_identity;
    const void* resource_domain;
    const void* compatibility_token;
    uint32_t min_batch;
    uint32_t max_batch;
    uint32_t multiple_of;
    int32_t batch_axis;
    int32_t device_resident;
    uint64_t device_epoch;
} VxBackendBatchContract;

#define VX_BACKEND_BATCH_CONTRACT_INIT \
    { sizeof(VxBackendBatchContract), VX_BACKEND_BATCH_PROTOCOL, NULL, NULL, \
      NULL, NULL, 1u, 1u, 1u, 0, 0, 0 }

/* One invocation names exactly one mutable route context and one dense stacked
 * binding set. The provider may use request_ids for tracing and must publish
 * lane i through lane_output_sinks[i]. Each stacked input has the contract's
 * batch_axis dimension equal to batch_size. A callback failure rejects every
 * lane; the Runtime never publishes a partial batch. Providers therefore
 * validate the complete invocation before dispatch. The Runtime never
 * constructs B separate provider contexts or substitutes B calls to
 * context_execute. */
typedef struct VxBackendBatchInvocation {
    size_t struct_size;
    const VxTensorBinding* stacked_inputs;
    size_t stacked_input_count;
    uint32_t batch_size;
    const uint64_t* request_ids;
    const VxBackendOutputSink* lane_output_sinks;
    size_t lane_output_sink_count;
} VxBackendBatchInvocation;

#define VX_BACKEND_BATCH_INVOCATION_INIT \
    { sizeof(VxBackendBatchInvocation), NULL, 0, 0, NULL, NULL, 0 }

typedef enum VxBackendShapeDomainSupport {
    VX_BACKEND_SHAPE_DOMAIN_UNSUPPORTED = 0,
    VX_BACKEND_SHAPE_DOMAIN_FULL = 1
} VxBackendShapeDomainSupport;

/* Provider-wide declaration. FULL means compile either attests the complete
 * bounded domain supplied in VxBackendCompileInput or rejects it before
 * creating a compiled instance. */
typedef struct VxBackendShapeDomainCapability {
    size_t struct_size;
    const char* proof_protocol;
    const char* resource_protocol;
    VxBackendShapeDomainSupport support;
} VxBackendShapeDomainCapability;

#define VX_BACKEND_SHAPE_DOMAIN_CAPABILITY_INIT \
    { sizeof(VxBackendShapeDomainCapability), VX_BACKEND_SHAPE_PROOF_PROTOCOL, \
      VX_BACKEND_RESOURCE_PROTOCOL, \
      VX_BACKEND_SHAPE_DOMAIN_UNSUPPORTED }

/* Immutable callback-duration model view. graph_fingerprint and
 * shape_domain_proof_identity identify the exact graph and complete bounded
 * proof represented by the logical input/output specifications. */
typedef struct VxBackendCompileInput {
    size_t struct_size;
    const VxModelSource* source;
    const char* graph_fingerprint;
    const char* shape_domain_proof_identity;
    const VxTensorSpec* inputs;
    size_t input_count;
    const VxTensorSpec* outputs;
    size_t output_count;
    /* NULL means the exact graph failed the core's fail-closed typed
     * independent-request-axis proof and therefore remains scheduler B=1. */
    const char* independent_batch_proof_protocol;
    const char* independent_batch_proof_identity;
} VxBackendCompileInput;

#define VX_BACKEND_COMPILE_INPUT_INIT \
    { sizeof(VxBackendCompileInput), NULL, NULL, NULL, NULL, 0, NULL, 0, \
      NULL, NULL }

/* A successful provider compile must attest the exact input identities and
 * conservative resource maxima. resource_limit_bytes is meaningful only when
 * has_resource_limit is non-zero. Provider string pointers are borrowed only
 * until compile returns; the runtime validates them synchronously. */
typedef struct VxBackendShapeDomainAttestation {
    size_t struct_size;
    const char* graph_fingerprint;
    const char* shape_domain_proof_identity;
    uint64_t maximum_tensor_bytes;
    uint64_t maximum_resident_bytes;
    uint64_t resource_limit_bytes;
    int32_t has_resource_limit;
} VxBackendShapeDomainAttestation;

#define VX_BACKEND_SHAPE_DOMAIN_ATTESTATION_INIT \
    { sizeof(VxBackendShapeDomainAttestation), NULL, NULL, 0, 0, 0, 0 }

typedef struct VxBackendProvider {
    size_t struct_size;
    uint32_t abi_version;
    const char* name;
    void* user_data;
    uint32_t flags;
    VxBackendShapeDomainCapability shape_domain;

    VxStatus (*runtime_create)(void* user_data,
                               const VxRuntimeOptions* options,
                               void** out_runtime_instance,
                               VxReport* report);
    void (*runtime_destroy)(void* runtime_instance);

    /* A successful compile fills route_evidence and sets route_attested. When
     * operator_fallback is forbidden it must also leave
     * operator_fallback_used clear; the runtime rejects an unattested plan. */
    VxStatus (*compile)(void* runtime_instance,
                        const VxBackendCompileInput* input,
                        const VxBackendPolicy* policy,
                        void** out_compiled_instance,
                        VxBackendShapeDomainAttestation* attestation,
                        VxReport* report);
    void (*compiled_destroy)(void* compiled_instance);

    /* Optional as a pair. Absence is the explicit B=1 contract. The Runtime
     * currently invokes this batch executor only after it can construct one
     * legal public binding; it never substitutes repeated context_execute. */
    VxStatus (*compiled_batch_contract)(
        void* compiled_instance,
        VxBackendBatchContract* contract,
        VxReport* report);
    VxStatus (*context_execute_batch)(
        void* context_instance,
        const VxBackendBatchInvocation* invocation,
        VxReport* report);

    VxStatus (*context_create)(void* compiled_instance,
                               const VxContextOptions* options,
                               void** out_context_instance,
                               VxReport* report);
    /* Binding arrays and all referenced storage are borrowed only for the
     * synchronous callback. HOST is the sole accepted location.
     * Providers validate the complete batch/domain/resource transition before
     * mutating context state. A pre-commit failure preserves prior binding and
     * decode state; failures after dispatch are not required to roll back
     * arbitrary backend/device side effects. */
    VxStatus (*context_execute)(void* context_instance,
                                const VxTensorBinding* inputs,
                                size_t input_count,
                                const VxBackendOutputSink* output_sink,
                                VxReport* report);
    /* Optional decode callbacks. A provider context created with decode
     * enabled returns BACKEND_UNSUPPORTED when the corresponding callback is
     * absent. Seed/step obey the same complete-output sink contract as
     * context_execute. */
    VxStatus (*context_decode_seed)(void* context_instance,
                                    const VxTensorBinding* inputs,
                                    size_t input_count,
                                    const VxBackendOutputSink* output_sink,
                                    VxReport* report);
    VxStatus (*context_decode_step)(void* context_instance,
                                    int32_t position,
                                    const VxTensorBinding* inputs,
                                    size_t input_count,
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

    /* Required exact-layout discriminator. Both values are validated before
     * the Runtime reads or invokes any provider callback. This deliberately
     * rejects descriptors compiled from any other header even when their
     * opaque numeric ABI discriminator happens to match. */
    uint64_t exact_contract_marker;
    size_t exact_contract_extent;
} VxBackendProvider;

/* Provider-host composition operation. The runtime copies the descriptor and
 * name, then creates exactly one provider runtime instance. Callback code and
 * user_data remain borrowed until the runtime and all retained descendants are
 * released. Names are canonical lower-case ASCII identifiers. The built-in
 * names cpu, vulkan, opengl, metal, and cuda are reserved. Synurang
 * applications select already-composed names through protobuf BackendPolicy. */
VX_API VxStatus vx_runtime_register_provider(VxRuntime* runtime,
                                      const VxBackendProvider* provider,
                                      VxReport* report);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_BACKEND_H */
