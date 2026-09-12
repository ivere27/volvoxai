#ifndef VOLVOXAI_RUNTIME_ACTIVATION_PLAN_H
#define VOLVOXAI_RUNTIME_ACTIVATION_PLAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_ACTIVATION_PLAN_API
#define VX_ACTIVATION_PLAN_API
#define VX_ACTIVATION_PLAN_API_DEFINED_HERE 1
#endif

/*
 * Backend-neutral temporal activation planner.
 *
 * The caller embeds one successful graph-plan response, followed by one
 * fixed record for every canonical graph-plan tensor. An optional successful
 * activation-plan response supplies the independently packed maximum-domain
 * layout used for concrete-size projection. All records are little-endian,
 * fixed-width, pointer-free, and owned by the caller.
 */
#define VOLVOXAI_ACTIVATION_PLAN_ABI_VERSION UINT32_C(1)
#define VOLVOXAI_ACTIVATION_PLAN_REQUEST_MAGIC UINT32_C(0x31415856)  /* "VXA1" */
#define VOLVOXAI_ACTIVATION_PLAN_RESPONSE_MAGIC UINT32_C(0x314c5856) /* "VXL1" */
#define VX_ACTIVATION_PLAN_INDEX_NONE UINT32_MAX

#if defined(__clang__) || defined(__GNUC__)
#define VX_ACTIVATION_PLAN_PACKED __attribute__((packed, aligned(4)))
#else
#error "The VolvoxAI activation-plan ABI requires packed-record support."
#endif

typedef int32_t VxActivationPlanStatusV1;
enum {
    VX_ACTIVATION_PLAN_STATUS_OK = 0,
    VX_ACTIVATION_PLAN_STATUS_INVALID_ARGUMENT = -1,
    VX_ACTIVATION_PLAN_STATUS_INVALID_WIRE = -2,
    VX_ACTIVATION_PLAN_STATUS_RESPONSE_TOO_SMALL = -3,
    VX_ACTIVATION_PLAN_STATUS_SCRATCH_TOO_SMALL = -4,
    VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN = -5,
    VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR = -6,
    VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM = -7,
    VX_ACTIVATION_PLAN_STATUS_INTERNAL = -8
};

typedef int32_t VxActivationPlanErrorCodeV1;
enum {
    VX_ACTIVATION_PLAN_ERROR_NONE = 0,
    VX_ACTIVATION_PLAN_ERROR_INVALID_HEADER = 1,
    VX_ACTIVATION_PLAN_ERROR_INVALID_LAYOUT = 2,
    VX_ACTIVATION_PLAN_ERROR_RESERVED_NONZERO = 3,
    VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN = 4,
    VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_TENSOR = 5,
    VX_ACTIVATION_PLAN_ERROR_INVALID_DTYPE = 6,
    VX_ACTIVATION_PLAN_ERROR_INVALID_SIZE = 7,
    VX_ACTIVATION_PLAN_ERROR_INVALID_ALIAS_ROOT = 8,
    VX_ACTIVATION_PLAN_ERROR_ALIAS_CYCLE = 9,
    VX_ACTIVATION_PLAN_ERROR_ALIAS_KIND = 10,
    VX_ACTIVATION_PLAN_ERROR_ALIAS_DTYPE = 11,
    VX_ACTIVATION_PLAN_ERROR_ALIAS_SIZE = 12,
    VX_ACTIVATION_PLAN_ERROR_ALIAS_LIFETIME = 13,
    VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW = 14,
    VX_ACTIVATION_PLAN_ERROR_INVALID_MAXIMUM = 15,
    VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH = 16,
    VX_ACTIVATION_PLAN_ERROR_PROJECTION_EXCEEDED = 17,
    VX_ACTIVATION_PLAN_ERROR_INVALID_BIRTH_WIDENING = 18
};

typedef uint32_t VxActivationPlanErrorSectionV1;
enum {
    VX_ACTIVATION_PLAN_ERROR_SECTION_NONE = 0,
    VX_ACTIVATION_PLAN_ERROR_SECTION_HEADER = 1,
    VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN = 2,
    VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR = 3,
    VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM = 4,
    VX_ACTIVATION_PLAN_ERROR_SECTION_PACKING = 5
};

enum {
    VX_ACTIVATION_PLAN_REQUEST_HAS_MAXIMUM = 1u,
    VX_ACTIVATION_PLAN_TENSOR_CROSS_CALL_LIVE = 1u,
    VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED = 2u,
    VX_ACTIVATION_PLAN_RESPONSE_PROJECTED = 1u,
    VX_ACTIVATION_PLAN_REGION_CROSS_CALL_LIVE = 1u
};

typedef struct VX_ACTIVATION_PLAN_PACKED VxActivationPlanRequestV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t total_bytes;
    uint32_t flags;
    uint32_t graph_plan_offset;
    uint32_t graph_plan_bytes;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t maximum_plan_offset;
    uint32_t maximum_plan_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
    uint32_t reserved4;
    uint32_t reserved5;
} VxActivationPlanRequestV1;

/* Tensor record index is the graph plan's canonical tensor index. */
typedef struct VX_ACTIVATION_PLAN_PACKED VxActivationPlanTensorV1 {
    int32_t dtype; /* Stable generated VxDataType numeric value. */
    uint32_t flags;
    uint64_t size_bytes;
    uint32_t alias_root_index;
    /*
     * Zero unless VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED is set. With that
     * flag this is UINT32_MAX for logical -1, or an earlier node index. The
     * core accepts widening only for VALUE tensors and never accepts a birth
     * equal to or later than the canonical graph-plan birth.
     */
    uint32_t widened_birth;
} VxActivationPlanTensorV1;

typedef struct VX_ACTIVATION_PLAN_PACKED VxActivationPlanResponseV1 {
    uint32_t magic;
    uint32_t abi_version;
    VxActivationPlanStatusV1 status;
    VxActivationPlanErrorCodeV1 error_code;
    VxActivationPlanErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
    uint32_t flags;
    uint32_t written_bytes;
    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    uint32_t tensor_count;
    uint32_t region_offset;
    uint32_t region_count;
    uint32_t arena_offset;
    uint32_t arena_count;
    uint64_t unique_bytes;
    uint64_t capacity_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
} VxActivationPlanResponseV1;

/* Regions are emitted in ascending owner_tensor_index order. */
typedef struct VX_ACTIVATION_PLAN_PACKED VxActivationPlanRegionV1 {
    uint32_t owner_tensor_index;
    int32_t dtype;
    uint32_t birth;
    uint32_t last_use;
    uint64_t offset_bytes;
    uint64_t size_bytes;
    uint32_t flags;
    uint32_t reserved;
} VxActivationPlanRegionV1;

/* Arenas use stable dtype-name order: F32, I32, I8, then U8. */
typedef struct VX_ACTIVATION_PLAN_PACKED VxActivationPlanArenaV1 {
    int32_t dtype;
    uint32_t reserved;
    uint64_t capacity_bytes;
} VxActivationPlanArenaV1;

#if defined(__cplusplus)
static_assert(sizeof(VxActivationPlanRequestV1) == 64,
              "activation-plan request ABI drift");
static_assert(sizeof(VxActivationPlanTensorV1) == 24,
              "activation-plan tensor ABI drift");
static_assert(sizeof(VxActivationPlanResponseV1) == 96,
              "activation-plan response ABI drift");
static_assert(sizeof(VxActivationPlanRegionV1) == 40,
              "activation-plan region ABI drift");
static_assert(sizeof(VxActivationPlanArenaV1) == 16,
              "activation-plan arena ABI drift");
#else
_Static_assert(sizeof(VxActivationPlanRequestV1) == 64,
               "activation-plan request ABI drift");
_Static_assert(sizeof(VxActivationPlanTensorV1) == 24,
               "activation-plan tensor ABI drift");
_Static_assert(sizeof(VxActivationPlanResponseV1) == 96,
               "activation-plan response ABI drift");
_Static_assert(sizeof(VxActivationPlanRegionV1) == 40,
               "activation-plan region ABI drift");
_Static_assert(sizeof(VxActivationPlanArenaV1) == 16,
               "activation-plan arena ABI drift");
#endif

VX_ACTIVATION_PLAN_API uint32_t vx_activation_plan_abi_version(void);

/*
 * Request/response are 4-byte aligned and scratch is 16-byte aligned. Buffers
 * are mutually disjoint. A null scratch pointer is accepted only with zero
 * bytes for a sizing query. No pointer or state is retained.
 */
VX_ACTIVATION_PLAN_API int32_t vx_activation_plan_compile_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    uint8_t* response,
    uint32_t response_bytes,
    uint8_t* scratch,
    uint32_t scratch_bytes);

#undef VX_ACTIVATION_PLAN_PACKED

#ifdef VX_ACTIVATION_PLAN_API_DEFINED_HERE
#undef VX_ACTIVATION_PLAN_API_DEFINED_HERE
#undef VX_ACTIVATION_PLAN_API
#endif

#ifdef __cplusplus
}
#endif

#endif
