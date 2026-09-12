#ifndef VOLVOXAI_RUNTIME_GRAPH_BIND_H
#define VOLVOXAI_RUNTIME_GRAPH_BIND_H

#include "generated/operator_param_ids.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_GRAPH_BIND_API
#define VX_GRAPH_BIND_API
#define VX_GRAPH_BIND_API_DEFINED_HERE 1
#endif

/*
 * Backend-neutral concrete graph binding ABI.
 *
 * A definition is produced after package parsing and bounded-domain proof.
 * It contains the already-normalized logical tensor assertions, typed shape
 * parameters, and hydrated affine metadata, all indexed by one successful
 * graph_plan response. The original canonical graph_plan request accompanies
 * that response: this core recompiles it in caller scratch and accepts the
 * pair only when the produced response is byte-for-byte identical. Package
 * JSON parsing, parameter-schema normalization, and bounded-domain proof are
 * intentionally outside this ABI.
 *
 * The definition, graph-plan, binding, response, and scratch records are
 * little-endian, pointer-free, and caller-owned. No pointer, handle, model
 * state, allocator, FILE, lock, or thread primitive survives a call.
 */
#define VOLVOXAI_GRAPH_BIND_ABI_VERSION UINT32_C(1)
#define VOLVOXAI_GRAPH_BIND_DEFINITION_MAGIC UINT32_C(0x31445856) /* "VXD1" */
#define VOLVOXAI_GRAPH_BIND_REQUEST_MAGIC UINT32_C(0x31425856)    /* "VXB1" */
#define VOLVOXAI_GRAPH_BIND_RESPONSE_MAGIC UINT32_C(0x31595656)   /* "VXY1" */
#define VX_GRAPH_BIND_INDEX_NONE UINT32_MAX

#if defined(__clang__) || defined(__GNUC__)
#define VX_GRAPH_BIND_PACKED __attribute__((packed, aligned(4)))
#else
#error "The VolvoxAI graph-bind ABI requires packed-record support."
#endif

typedef int32_t VxGraphBindStatusV1;
enum {
    VX_GRAPH_BIND_STATUS_OK = 0,
    VX_GRAPH_BIND_STATUS_INVALID_ARGUMENT = -1,
    VX_GRAPH_BIND_STATUS_INVALID_WIRE = -2,
    VX_GRAPH_BIND_STATUS_RESPONSE_TOO_SMALL = -3,
    VX_GRAPH_BIND_STATUS_SCRATCH_TOO_SMALL = -4,
    VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN = -5,
    VX_GRAPH_BIND_STATUS_INVALID_DEFINITION = -6,
    VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED = -7,
    VX_GRAPH_BIND_STATUS_SHAPE_ERROR = -8,
    VX_GRAPH_BIND_STATUS_INTERNAL = -9
};

typedef int32_t VxGraphBindErrorCodeV1;
enum {
    VX_GRAPH_BIND_ERROR_NONE = 0,
    VX_GRAPH_BIND_ERROR_INVALID_HEADER = 1,
    VX_GRAPH_BIND_ERROR_INVALID_LAYOUT = 2,
    VX_GRAPH_BIND_ERROR_RESERVED_NONZERO = 3,
    VX_GRAPH_BIND_ERROR_INVALID_UTF8 = 4,
    VX_GRAPH_BIND_ERROR_NONCANONICAL_ORDER = 5,
    VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN = 6,
    VX_GRAPH_BIND_ERROR_COUNT_MISMATCH = 7,
    VX_GRAPH_BIND_ERROR_INVALID_DIMENSION = 8,
    VX_GRAPH_BIND_ERROR_INVALID_TENSOR = 9,
    VX_GRAPH_BIND_ERROR_INVALID_AXIS = 10,
    VX_GRAPH_BIND_ERROR_INVALID_NODE = 11,
    VX_GRAPH_BIND_ERROR_INVALID_EDGE = 12,
    VX_GRAPH_BIND_ERROR_INVALID_PARAM = 13,
    VX_GRAPH_BIND_ERROR_INVALID_QUANTIZATION = 14,
    VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET = 15,
    VX_GRAPH_BIND_ERROR_DTYPE_MISMATCH = 16,
    VX_GRAPH_BIND_ERROR_RANK_MISMATCH = 17,
    VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH = 18,
    VX_GRAPH_BIND_ERROR_SYMBOL_CONFLICT = 19,
    VX_GRAPH_BIND_ERROR_BOUND_VIOLATION = 20,
    VX_GRAPH_BIND_ERROR_MULTIPLE_OF_VIOLATION = 21,
    VX_GRAPH_BIND_ERROR_BYTE_LENGTH_MISMATCH = 22,
    VX_GRAPH_BIND_ERROR_UNBOUND_SYMBOL = 23,
    VX_GRAPH_BIND_ERROR_OPERATOR_SHAPE = 24,
    VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW = 25,
    VX_GRAPH_BIND_ERROR_OUTPUT_PORT_MISMATCH = 26,
    VX_GRAPH_BIND_ERROR_OUTPUT_DTYPE_MISMATCH = 27,
    VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH = 28
};

typedef uint32_t VxGraphBindErrorSectionV1;
enum {
    VX_GRAPH_BIND_ERROR_SECTION_NONE = 0,
    VX_GRAPH_BIND_ERROR_SECTION_DEFINITION = 1,
    VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN = 2,
    VX_GRAPH_BIND_ERROR_SECTION_BINDING = 3,
    VX_GRAPH_BIND_ERROR_SECTION_DIMENSION = 4,
    VX_GRAPH_BIND_ERROR_SECTION_TENSOR = 5,
    VX_GRAPH_BIND_ERROR_SECTION_NODE = 6,
    VX_GRAPH_BIND_ERROR_SECTION_EDGE = 7,
    VX_GRAPH_BIND_ERROR_SECTION_PARAM = 8,
    VX_GRAPH_BIND_ERROR_SECTION_RESPONSE = 9
};

typedef uint32_t VxGraphBindAxisKindV1;
enum {
    VX_GRAPH_BIND_AXIS_FIXED = 1,
    VX_GRAPH_BIND_AXIS_SYMBOL = 2
};


typedef int32_t VxGraphBindParamKindV1;
enum {
    VX_GRAPH_BIND_PARAM_NUMBER = 1,
    VX_GRAPH_BIND_PARAM_BOOLEAN = 2,
    VX_GRAPH_BIND_PARAM_STRING = 3,
    VX_GRAPH_BIND_PARAM_VALUE_ARRAY = 4
};

typedef uint32_t VxGraphBindParamValueKindV1;
enum {
    VX_GRAPH_BIND_PARAM_VALUE_NUMBER = 1,
    VX_GRAPH_BIND_PARAM_VALUE_DIMENSION = 2
};

/* Canonical packed layout: header, dimensions, tensors, axes, nodes, edges,
 * params, param values, affine scales, affine zero-points, UTF-8 strings. */
typedef struct VX_GRAPH_BIND_PACKED VxGraphBindDefinitionV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t total_bytes;
    uint32_t flags;
    uint32_t dimension_offset;
    uint32_t dimension_count;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t axis_offset;
    uint32_t axis_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t edge_offset;
    uint32_t edge_count;
    uint32_t param_offset;
    uint32_t param_count;
    uint32_t param_value_offset;
    uint32_t param_value_count;
    uint32_t quantization_scale_offset;
    uint32_t quantization_scale_count;
    uint32_t quantization_zero_point_offset;
    uint32_t quantization_zero_point_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
} VxGraphBindDefinitionV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindDimensionV1 {
    uint32_t name_offset; /* Relative to definition string table. */
    uint32_t name_length;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple_of;
    uint32_t reserved0;
    uint32_t reserved1;
} VxGraphBindDimensionV1;

/* Tensor record index is the graph-plan tensor index. */
typedef struct VX_GRAPH_BIND_PACKED VxGraphBindTensorV1 {
    int32_t dtype;
    uint32_t rank;
    uint32_t axis_first;
    VxTensorQuantizationKind quantization_scheme;
    uint32_t quantization_scale_bits;
    int32_t quantization_zero_point;
    uint32_t quantization_axis;
    uint32_t quantization_count;
    uint32_t quantization_value_first;
    /* Zero for an ordinary tensor; otherwise the original weight axis-0
     * bank slot count before a concrete resident subset is selected. */
    uint32_t bank_slot_count;
    uint32_t name_offset; /* Relative to definition string table. */
    uint32_t name_length;
} VxGraphBindTensorV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindAxisV1 {
    VxGraphBindAxisKindV1 kind;
    uint32_t dimension_index; /* SYMBOL only; FIXED requires UINT32_MAX. */
    uint64_t fixed_value;     /* FIXED only; SYMBOL requires zero. */
} VxGraphBindAxisV1;

/* Node record index is the graph-plan schedule index. */
typedef struct VX_GRAPH_BIND_PACKED VxGraphBindNodeV1 {
    uint32_t operator_offset; /* Relative to definition string table. */
    uint32_t operator_length;
    uint32_t id_offset; /* Relative to definition string table. */
    uint32_t id_length;
    uint32_t param_first;
    uint32_t param_count;
} VxGraphBindNodeV1;

/* Edge record index is the graph-plan resolved-edge response index. */
typedef struct VX_GRAPH_BIND_PACKED VxGraphBindEdgeV1 {
    uint32_t port_offset; /* Relative to definition string table. */
    uint32_t port_length;
    uint32_t tensor_index;
    uint32_t reserved;
} VxGraphBindEdgeV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindParamV1 {
    uint32_t name_offset; /* Relative to definition string table. */
    uint32_t name_length;
    VxGraphBindParamKindV1 kind;
    uint32_t reserved;
    /* STRING: string-table offset/byte length. VALUE_ARRAY: first/count. */
    uint32_t value_offset_or_first;
    uint32_t value_count;
    /* NUMBER: IEEE-754 binary64 bits. BOOLEAN: exactly zero or one. */
    uint64_t value_bits;
} VxGraphBindParamV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindParamValueV1 {
    VxGraphBindParamValueKindV1 kind;
    uint32_t reserved;
    /* NUMBER: binary64 bits. DIMENSION: uint32 index in low bits, high zero. */
    uint64_t value_bits;
} VxGraphBindParamValueV1;

/* Canonical packed layout: header, inputs, concrete axes, banks, bank slots. */
typedef struct VX_GRAPH_BIND_PACKED VxGraphBindRequestV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t total_bytes;
    uint32_t flags;
    uint32_t input_offset;
    uint32_t input_count;
    uint32_t axis_offset;
    uint32_t axis_count;
    uint32_t bank_offset;
    uint32_t bank_count;
    uint32_t bank_slot_offset;
    uint32_t bank_slot_count;
} VxGraphBindRequestV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindInputV1 {
    uint32_t tensor_index;
    int32_t dtype;
    uint32_t rank;
    uint32_t axis_first;
    uint64_t logical_size_bytes;
} VxGraphBindInputV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindBankV1 {
    uint32_t tensor_index;
    uint32_t slot_first;
    uint32_t slot_count;
    uint32_t reserved;
} VxGraphBindBankV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindResponseV1 {
    uint32_t magic;
    uint32_t abi_version;
    VxGraphBindStatusV1 status;
    VxGraphBindErrorCodeV1 error_code;
    VxGraphBindErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
    uint32_t written_bytes;
    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t axis_offset;
    uint32_t axis_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t symbol_offset;
    uint32_t symbol_count;
    uint32_t bank_offset;
    uint32_t bank_count;
    uint32_t bank_slot_offset;
    uint32_t bank_slot_count;
    uint64_t logical_activation_bytes;
    uint64_t weight_bytes;
    uint32_t error_path_offset;
    uint32_t error_path_length;
    uint32_t error_detail_offset;
    uint32_t error_detail_length;
    uint32_t reserved0;
    uint32_t reserved1;
} VxGraphBindResponseV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindResolvedTensorV1 {
    uint32_t kind; /* VxGraphPlanTensorKindV1 wire value. */
    int32_t dtype;
    uint32_t rank;
    uint32_t axis_first;
    uint64_t element_count;
    uint64_t size_bytes;
    VxTensorQuantizationKind quantization_scheme;
    uint32_t quantization_scale_bits;
    int32_t quantization_zero_point;
    uint32_t quantization_axis;
    uint32_t quantization_count;
    uint32_t quantization_scales_offset;
    uint32_t quantization_zero_points_offset;
    uint32_t flags;
} VxGraphBindResolvedTensorV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindResolvedNodeV1 {
    uint32_t shape_function_id_offset;
    uint32_t shape_function_id_length;
    uint32_t reserved0;
    uint32_t reserved1;
} VxGraphBindResolvedNodeV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindResolvedSymbolV1 {
    uint32_t dimension_index;
    uint32_t reserved;
    uint64_t value;
} VxGraphBindResolvedSymbolV1;

typedef struct VX_GRAPH_BIND_PACKED VxGraphBindResolvedBankV1 {
    uint32_t tensor_index;
    uint32_t slot_first;
    uint32_t slot_count;
    uint32_t reserved;
} VxGraphBindResolvedBankV1;

#if defined(__cplusplus)
static_assert(sizeof(VxGraphBindDefinitionV1) == 104, "graph-bind definition ABI drift");
static_assert(sizeof(VxGraphBindDimensionV1) == 40, "graph-bind dimension ABI drift");
static_assert(sizeof(VxGraphBindTensorV1) == 48, "graph-bind tensor ABI drift");
static_assert(sizeof(VxGraphBindAxisV1) == 16, "graph-bind axis ABI drift");
static_assert(sizeof(VxGraphBindNodeV1) == 24, "graph-bind node ABI drift");
static_assert(sizeof(VxGraphBindEdgeV1) == 16, "graph-bind edge ABI drift");
static_assert(sizeof(VxGraphBindParamV1) == 32, "graph-bind param ABI drift");
static_assert(sizeof(VxGraphBindParamValueV1) == 16, "graph-bind param value ABI drift");
static_assert(sizeof(VxGraphBindRequestV1) == 48, "graph-bind request ABI drift");
static_assert(sizeof(VxGraphBindInputV1) == 24, "graph-bind input ABI drift");
static_assert(sizeof(VxGraphBindBankV1) == 16, "graph-bind bank ABI drift");
static_assert(sizeof(VxGraphBindResponseV1) == 128, "graph-bind response ABI drift");
static_assert(sizeof(VxGraphBindResolvedTensorV1) == 64, "graph-bind tensor response ABI drift");
static_assert(sizeof(VxGraphBindResolvedNodeV1) == 16, "graph-bind node response ABI drift");
static_assert(sizeof(VxGraphBindResolvedSymbolV1) == 16, "graph-bind symbol response ABI drift");
static_assert(sizeof(VxGraphBindResolvedBankV1) == 16, "graph-bind bank response ABI drift");
#else
_Static_assert(sizeof(VxGraphBindDefinitionV1) == 104, "graph-bind definition ABI drift");
_Static_assert(sizeof(VxGraphBindDimensionV1) == 40, "graph-bind dimension ABI drift");
_Static_assert(sizeof(VxGraphBindTensorV1) == 48, "graph-bind tensor ABI drift");
_Static_assert(sizeof(VxGraphBindAxisV1) == 16, "graph-bind axis ABI drift");
_Static_assert(sizeof(VxGraphBindNodeV1) == 24, "graph-bind node ABI drift");
_Static_assert(sizeof(VxGraphBindEdgeV1) == 16, "graph-bind edge ABI drift");
_Static_assert(sizeof(VxGraphBindParamV1) == 32, "graph-bind param ABI drift");
_Static_assert(sizeof(VxGraphBindParamValueV1) == 16, "graph-bind param value ABI drift");
_Static_assert(sizeof(VxGraphBindRequestV1) == 48, "graph-bind request ABI drift");
_Static_assert(sizeof(VxGraphBindInputV1) == 24, "graph-bind input ABI drift");
_Static_assert(sizeof(VxGraphBindBankV1) == 16, "graph-bind bank ABI drift");
_Static_assert(sizeof(VxGraphBindResponseV1) == 128, "graph-bind response ABI drift");
_Static_assert(sizeof(VxGraphBindResolvedTensorV1) == 64, "graph-bind tensor response ABI drift");
_Static_assert(sizeof(VxGraphBindResolvedNodeV1) == 16, "graph-bind node response ABI drift");
_Static_assert(sizeof(VxGraphBindResolvedSymbolV1) == 16, "graph-bind symbol response ABI drift");
_Static_assert(sizeof(VxGraphBindResolvedBankV1) == 16, "graph-bind bank response ABI drift");
#endif

VX_GRAPH_BIND_API uint32_t vx_graph_bind_abi_version(void);

/* Definition, graph_plan request/response, and binding are read-only. The
 * graph_plan response must be the exact successful result of recompiling the
 * supplied request. Response is 8-byte aligned, scratch 16-byte aligned, and
 * every writable range must be disjoint from every other supplied range.
 * Scratch is cleared before return. No pointer is retained.
 *
 * A zero-byte scratch call starts sizing. SCRATCH_TOO_SMALL reports a strictly
 * increasing lower bound, not necessarily the final high-water mark: callers
 * must retry (and may grow geometrically) until another status is returned.
 * On success required_scratch_bytes is the exact high-water mark for that
 * request. RESPONSE_TOO_SMALL reports the exact required response size. */
VX_GRAPH_BIND_API int32_t vx_graph_bind_resolve_v1(
    const uint8_t* definition,
    uint32_t definition_bytes,
    const uint8_t* graph_plan_request,
    uint32_t graph_plan_request_bytes,
    const uint8_t* graph_plan_response,
    uint32_t graph_plan_response_bytes,
    const uint8_t* binding,
    uint32_t binding_bytes,
    uint8_t* response,
    uint32_t response_bytes,
    uint8_t* scratch,
    uint32_t scratch_bytes);

#undef VX_GRAPH_BIND_PACKED

#ifdef VX_GRAPH_BIND_API_DEFINED_HERE
#undef VX_GRAPH_BIND_API_DEFINED_HERE
#undef VX_GRAPH_BIND_API
#endif

#ifdef __cplusplus
}
#endif

#endif
