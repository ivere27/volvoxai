/* Value types used only by the private C lifecycle implementation.
 *
 * Applications use proto/volvoxai.proto through generated clients. Backend
 * providers use native/include/volvoxai_backend.h. Keeping these bridge types
 * here prevents either internal lifecycle from becoming a second public API.
 */
#ifndef VOLVOXAI_LIFECYCLE_TYPES_H
#define VOLVOXAI_LIFECYCLE_TYPES_H

#include "volvoxai_types.h"

#define VX_NATIVE_API_VERSION UINT32_C(1)
#define VX_MAX_BACKEND_CANDIDATES 16u

/* Process-envelope sampling is an optional, read-only diagnostic surface. */
#define VX_PROCESS_MEMORY_SAMPLE_ABI_VERSION UINT32_C(1)
#define VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS UINT32_C(1)
#define VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS UINT32_C(2)
#define VX_PROCESS_MEMORY_AVAILABLE_MONOTONIC_TIME UINT32_C(4)

typedef struct VxProcessMemorySampleV1 {
    size_t struct_size;
    uint32_t abi_version;
    uint32_t available_mask;
    uint64_t rss_bytes;
    uint64_t peak_rss_bytes;
    uint64_t monotonic_nanoseconds;
} VxProcessMemorySampleV1;

#define VX_PROCESS_MEMORY_SAMPLE_V1_INIT { \
    sizeof(VxProcessMemorySampleV1), VX_PROCESS_MEMORY_SAMPLE_ABI_VERSION, \
    0, 0, 0, 0 \
}

typedef struct VxRevisionInfo {
    size_t struct_size;
    uint64_t graph_id;
    uint64_t graph_revision;
    uint64_t weight_id;
    uint64_t weight_revision;
    uint64_t adapter_id;
    uint64_t adapter_revision;
} VxRevisionInfo;

#define VX_REVISION_INFO_INIT \
    { sizeof(VxRevisionInfo), 0, 0, 0, 0, 0, 0 }

typedef struct VxAdapterSource {
    size_t struct_size;
    /* Stable logical name. Publishing the same name creates a new immutable
     * adapter revision. */
    const char* adapter_name;
    /* NULL publishes a metadata-only provider-owned route revision. */
    const char* package_path;
    const char* version_name;
} VxAdapterSource;

#define VX_ADAPTER_SOURCE_INIT \
    { sizeof(VxAdapterSource), NULL, NULL, NULL }

typedef struct VxAdapterRevision {
    size_t struct_size;
    uint64_t adapter_id;
    uint64_t adapter_revision;
} VxAdapterRevision;

#define VX_ADAPTER_REVISION_INIT \
    { sizeof(VxAdapterRevision), 0, 0 }

typedef struct VxTensorInfo {
    size_t struct_size;
    const char* name;
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[VX_MAX_TENSOR_RANK];
    size_t byte_size;
    VxMemoryLocation location;
} VxTensorInfo;

#define VX_TENSOR_INFO_INIT \
    { sizeof(VxTensorInfo), NULL, VX_DTYPE_F32, 0, {0}, 0, VX_MEMORY_HOST }

#define VX_RUNTIME_PRIORITY_MIN (-1000)
#define VX_RUNTIME_PRIORITY_MAX 1000

typedef struct VxRuntimeSubmitOptions {
    size_t struct_size;
    /* Higher values run first; age, deadline, then request id break ties. */
    int32_t priority;
    /* Absolute CLOCK_MONOTONIC microseconds; zero disables the deadline. */
    uint64_t deadline_monotonic_micros;
    VxRequestFreshness freshness;
    /* Required and nonzero for LATEST freshness. */
    uint64_t stream_key;
} VxRuntimeSubmitOptions;

#define VX_RUNTIME_SUBMIT_OPTIONS_INIT \
    { sizeof(VxRuntimeSubmitOptions), 0, 0, VX_REQUEST_FRESHNESS_ALL, 0 }

typedef struct VxRequestInfo {
    size_t struct_size;
    uint64_t request_id;
    VxRequestState state;
    /* BUSY while queued/running; otherwise the terminal execution status. */
    VxStatus status;
    size_t owned_input_bytes;
    int32_t deadline_missed;
} VxRequestInfo;

#define VX_REQUEST_INFO_INIT \
    { sizeof(VxRequestInfo), 0, VX_REQUEST_STATE_QUEUED, \
      VX_STATUS_BUSY, 0, 0 }

#define VX_REQUEST_WAIT_INFINITE UINT64_MAX

typedef struct VxAffineQuantization {
    size_t struct_size;
    int32_t defined;
    float scale;
    int32_t zero_point;
} VxAffineQuantization;

#define VX_AFFINE_QUANTIZATION_INIT \
    { sizeof(VxAffineQuantization), 0, 0.0f, 0 }

#endif /* VOLVOXAI_LIFECYCLE_TYPES_H */
