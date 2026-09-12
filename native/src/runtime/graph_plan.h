#ifndef VOLVOXAI_RUNTIME_GRAPH_PLAN_H
#define VOLVOXAI_RUNTIME_GRAPH_PLAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_GRAPH_PLAN_API
#define VX_GRAPH_PLAN_API
#define VX_GRAPH_PLAN_API_DEFINED_HERE 1
#endif

/*
 * Backend-neutral immutable graph-topology compiler.
 *
 * The request, response, and scratch ranges are caller-owned. Wire records
 * are little-endian and contain only fixed-width scalar values and relative
 * offsets; they never contain C pointers, size_t values, compiler enums, or
 * target-dependent unions. The implementation is reentrant and does not
 * allocate, open files, acquire locks, or consult an engine singleton.
 *
 * A graph-plan response is immutable topology metadata. It deliberately does
 * not contain backend routes, executable handles, tensor payloads, concrete
 * shapes, or training state.
 */
#define VOLVOXAI_GRAPH_PLAN_ABI_VERSION UINT32_C(1)
#define VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC UINT32_C(0x31475856)  /* "VXG1" */
#define VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC UINT32_C(0x31505856) /* "VXP1" */
#define VX_GRAPH_PLAN_INDEX_NONE UINT32_MAX

#if defined(__clang__) || defined(__GNUC__)
#define VX_GRAPH_PLAN_PACKED __attribute__((packed, aligned(4)))
#else
#error "The VolvoxAI graph-plan ABI requires packed-record support."
#endif

typedef int32_t VxGraphPlanStatusV1;
enum {
    VX_GRAPH_PLAN_STATUS_OK = 0,
    VX_GRAPH_PLAN_STATUS_INVALID_ARGUMENT = -1,
    VX_GRAPH_PLAN_STATUS_INVALID_WIRE = -2,
    VX_GRAPH_PLAN_STATUS_RESPONSE_TOO_SMALL = -3,
    VX_GRAPH_PLAN_STATUS_SCRATCH_TOO_SMALL = -4,
    VX_GRAPH_PLAN_STATUS_INVALID_GRAPH = -5,
    VX_GRAPH_PLAN_STATUS_INTERNAL = -6
};

typedef int32_t VxGraphPlanErrorCodeV1;
enum {
    VX_GRAPH_PLAN_ERROR_NONE = 0,
    VX_GRAPH_PLAN_ERROR_INVALID_HEADER = 1,
    VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT = 2,
    VX_GRAPH_PLAN_ERROR_RESERVED_NONZERO = 3,
    VX_GRAPH_PLAN_ERROR_INVALID_UTF8 = 4,
    VX_GRAPH_PLAN_ERROR_EMPTY_NAME = 5,
    VX_GRAPH_PLAN_ERROR_NONCANONICAL_ORDER = 6,
    VX_GRAPH_PLAN_ERROR_INVALID_SOURCE_KIND = 7,
    VX_GRAPH_PLAN_ERROR_DUPLICATE_TENSOR = 8,
    VX_GRAPH_PLAN_ERROR_DUPLICATE_NODE = 9,
    VX_GRAPH_PLAN_ERROR_DUPLICATE_PORT = 10,
    VX_GRAPH_PLAN_ERROR_UNKNOWN_TENSOR = 11,
    VX_GRAPH_PLAN_ERROR_INVALID_OPERATOR = 12,
    VX_GRAPH_PLAN_ERROR_INVALID_NODE_RANGE = 13,
    VX_GRAPH_PLAN_ERROR_MISSING_NODE_OUTPUT = 14,
    VX_GRAPH_PLAN_ERROR_INVALID_PUBLIC_OUTPUT = 15,
    VX_GRAPH_PLAN_ERROR_DUPLICATE_PUBLIC_OUTPUT = 16,
    VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW = 17
};

typedef uint32_t VxGraphPlanErrorSectionV1;
enum {
    VX_GRAPH_PLAN_ERROR_SECTION_NONE = 0,
    VX_GRAPH_PLAN_ERROR_SECTION_HEADER = 1,
    VX_GRAPH_PLAN_ERROR_SECTION_SOURCE = 2,
    VX_GRAPH_PLAN_ERROR_SECTION_NODE = 3,
    VX_GRAPH_PLAN_ERROR_SECTION_EDGE = 4,
    VX_GRAPH_PLAN_ERROR_SECTION_PUBLIC_OUTPUT = 5
};

typedef uint32_t VxGraphPlanSourceKindV1;
enum {
    VX_GRAPH_PLAN_SOURCE_INPUT = 1,
    VX_GRAPH_PLAN_SOURCE_WEIGHT = 2
};

typedef uint32_t VxGraphPlanTensorKindV1;
enum {
    VX_GRAPH_PLAN_TENSOR_INPUT = 1,
    VX_GRAPH_PLAN_TENSOR_WEIGHT = 2,
    VX_GRAPH_PLAN_TENSOR_VALUE = 3
};

enum {
    VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT = 1u
};

/*
 * The five request sections must use the canonical packed layout immediately
 * following this header: sources, nodes, edges, public outputs, then the byte
 * string table. Name offsets in all request records are relative to the start
 * of that string table.
 */
typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanRequestV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t total_bytes;
    uint32_t flags;
    uint32_t source_offset;
    uint32_t source_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t edge_offset;
    uint32_t edge_count;
    uint32_t public_output_offset;
    uint32_t public_output_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
} VxGraphPlanRequestV1;

/* Inputs precede weights; names in each group use unsigned UTF-8 byte order. */
typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanSourceV1 {
    uint32_t name_offset;
    uint32_t name_length;
    VxGraphPlanSourceKindV1 kind;
    uint32_t reserved;
} VxGraphPlanSourceV1;

/*
 * Nodes remain in the package's required topological order. Every node owns
 * one consecutive input-edge range followed immediately by one consecutive,
 * non-empty output-edge range. Ranges across nodes partition the edge table.
 * Port names are strictly ordered by unsigned UTF-8 bytes independently in
 * each node's input and output range.
 */
typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanNodeV1 {
    uint32_t id_offset;
    uint32_t id_length;
    int32_t operator_kind; /* Stable generated VxOperatorKind numeric value. */
    uint32_t input_edge_first;
    uint32_t input_edge_count;
    uint32_t output_edge_first;
    uint32_t output_edge_count;
    uint32_t reserved;
} VxGraphPlanNodeV1;

/* Input edges reference an existing tensor; output edges declare one. */
typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanEdgeV1 {
    uint32_t port_offset;
    uint32_t port_length;
    uint32_t tensor_offset;
    uint32_t tensor_length;
    uint32_t reserved0;
    uint32_t reserved1;
} VxGraphPlanEdgeV1;

typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanPublicOutputV1 {
    uint32_t tensor_offset;
    uint32_t tensor_length;
} VxGraphPlanPublicOutputV1;

/*
 * A response header is written for every result except INVALID_ARGUMENT when
 * the response range cannot hold this header. required_* values are exact
 * after a structurally readable request header/table layout.
 */
typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanResponseV1 {
    uint32_t magic;
    uint32_t abi_version;
    VxGraphPlanStatusV1 status;
    VxGraphPlanErrorCodeV1 error_code;
    VxGraphPlanErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
    uint32_t written_bytes;
    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t edge_offset;
    uint32_t edge_count;
    uint32_t public_output_offset;
    uint32_t public_output_count;
    uint32_t reserved0;
    uint32_t reserved1;
} VxGraphPlanResponseV1;

/* Names remain in the immutable logical graph and are addressed by declaration. */
typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanTensorV1 {
    VxGraphPlanTensorKindV1 kind;
    /* Source-record index for INPUT/WEIGHT; request edge index for VALUE. */
    uint32_t declaration_index;
    uint32_t producer_node_index;
    uint32_t producer_edge_index;
    /*
     * UINT32_MAX encodes logical -1: INPUT birth begins before node zero,
     * VALUE birth is its producer node, and WEIGHT birth/last_use are both
     * UINT32_MAX. A public output lives through the caller boundary and has
     * last_use == node_count (the same convention used by a later cross-call
     * lifetime extension).
     */
    uint32_t birth;
    uint32_t last_use;
    uint32_t flags;
    /* Rank 0..tensor_count-1 by unsigned UTF-8 tensor-name bytes. */
    uint32_t canonical_name_rank;
} VxGraphPlanTensorV1;

typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanStepV1 {
    uint32_t request_node_index;
    int32_t operator_kind;
    uint32_t input_edge_first;
    uint32_t input_edge_count;
    uint32_t output_edge_first;
    uint32_t output_edge_count;
} VxGraphPlanStepV1;

typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanResolvedEdgeV1 {
    uint32_t request_edge_index;
    uint32_t tensor_index;
} VxGraphPlanResolvedEdgeV1;

typedef struct VX_GRAPH_PLAN_PACKED VxGraphPlanResolvedOutputV1 {
    uint32_t request_output_index;
    uint32_t tensor_index;
} VxGraphPlanResolvedOutputV1;

#if defined(__cplusplus)
static_assert(sizeof(VxGraphPlanRequestV1) == 64, "graph-plan request ABI drift");
static_assert(sizeof(VxGraphPlanSourceV1) == 16, "graph-plan source ABI drift");
static_assert(sizeof(VxGraphPlanNodeV1) == 32, "graph-plan node ABI drift");
static_assert(sizeof(VxGraphPlanEdgeV1) == 24, "graph-plan edge ABI drift");
static_assert(sizeof(VxGraphPlanPublicOutputV1) == 8, "graph-plan output ABI drift");
static_assert(sizeof(VxGraphPlanResponseV1) == 80, "graph-plan response ABI drift");
static_assert(sizeof(VxGraphPlanTensorV1) == 32, "graph-plan tensor ABI drift");
static_assert(sizeof(VxGraphPlanStepV1) == 24, "graph-plan step ABI drift");
static_assert(sizeof(VxGraphPlanResolvedEdgeV1) == 8, "graph-plan resolved-edge ABI drift");
static_assert(sizeof(VxGraphPlanResolvedOutputV1) == 8, "graph-plan resolved-output ABI drift");
#else
_Static_assert(sizeof(VxGraphPlanRequestV1) == 64, "graph-plan request ABI drift");
_Static_assert(sizeof(VxGraphPlanSourceV1) == 16, "graph-plan source ABI drift");
_Static_assert(sizeof(VxGraphPlanNodeV1) == 32, "graph-plan node ABI drift");
_Static_assert(sizeof(VxGraphPlanEdgeV1) == 24, "graph-plan edge ABI drift");
_Static_assert(sizeof(VxGraphPlanPublicOutputV1) == 8, "graph-plan output ABI drift");
_Static_assert(sizeof(VxGraphPlanResponseV1) == 80, "graph-plan response ABI drift");
_Static_assert(sizeof(VxGraphPlanTensorV1) == 32, "graph-plan tensor ABI drift");
_Static_assert(sizeof(VxGraphPlanStepV1) == 24, "graph-plan step ABI drift");
_Static_assert(sizeof(VxGraphPlanResolvedEdgeV1) == 8, "graph-plan resolved-edge ABI drift");
_Static_assert(sizeof(VxGraphPlanResolvedOutputV1) == 8, "graph-plan resolved-output ABI drift");
#endif

VX_GRAPH_PLAN_API uint32_t vx_graph_plan_abi_version(void);

/*
 * Buffers must be mutually disjoint. Request/response are 4-byte aligned and
 * scratch is 16-byte aligned. The portable core accepts a null scratch pointer
 * only when scratch_bytes is zero so callers can query exact requirements.
 * Scratch content is cleared before return. No pointer is retained.
 */
VX_GRAPH_PLAN_API int32_t vx_graph_plan_compile_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    uint8_t* response,
    uint32_t response_bytes,
    uint8_t* scratch,
    uint32_t scratch_bytes);

#undef VX_GRAPH_PLAN_PACKED

#ifdef VX_GRAPH_PLAN_API_DEFINED_HERE
#undef VX_GRAPH_PLAN_API_DEFINED_HERE
#undef VX_GRAPH_PLAN_API
#endif

#ifdef __cplusplus
}
#endif

#endif
