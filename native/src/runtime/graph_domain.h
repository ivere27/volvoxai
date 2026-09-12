#ifndef VOLVOXAI_RUNTIME_GRAPH_DOMAIN_H
#define VOLVOXAI_RUNTIME_GRAPH_DOMAIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_GRAPH_DOMAIN_API
#define VX_GRAPH_DOMAIN_API
#define VX_GRAPH_DOMAIN_API_DEFINED_HERE 1
#endif

/*
 * Backend-neutral bounded graph shape-domain proof.
 *
 * The core first runs the portable graph-bind resolver at the canonical
 * minimum binding to re-close definition/topology provenance and concrete
 * resolver parity. That run is never accepted as dynamic-domain evidence.
 * Complete bounded-domain acceptance comes only from allocator-free symbolic
 * fixed/symbol/affine contracts for every node. An operator outside the
 * explicitly implemented symbolic families returns UNSUPPORTED_DOMAIN; the
 * core never samples corners or extrapolates from a concrete result.
 *
 * The definition is the normalized VxGraphBindDefinitionV1 wire document.
 * The original graph-plan request and its successful response close topology
 * provenance: graph-bind recompiles the request and requires the response to
 * match byte for byte before this core can succeed.
 *
 * Response tensors and node edges retain the canonical logical descriptors
 * produced by the symbolic contracts. Relations preserve public-input self
 * witnesses, output constant/symbol witnesses, and exact affine witnesses.
 * Quantization arrays and the concrete parity result remain in the embedded
 * graph-bind response. Fact text is semantic but is not a byte-stable DTO.
 * The graph fingerprint is package identity rather than shape semantics; a
 * host may attach it only while retaining the exact immutable input documents.
 *
 * Every wire range is fixed-width, little-endian, pointer-free, caller-owned,
 * and discarded after the call. The implementation allocates no memory,
 * opens no files, acquires no locks, and consults no mutable engine state.
 */
#define VOLVOXAI_GRAPH_DOMAIN_ABI_VERSION UINT32_C(1)
#define VOLVOXAI_GRAPH_DOMAIN_RESPONSE_MAGIC UINT32_C(0x314F5856) /* "VXO1" */
#define VX_GRAPH_DOMAIN_INDEX_NONE UINT32_MAX

#if defined(__clang__) || defined(__GNUC__)
#define VX_GRAPH_DOMAIN_PACKED __attribute__((packed, aligned(4)))
#else
#error "The VolvoxAI graph-domain ABI requires packed-record support."
#endif

typedef int32_t VxGraphDomainStatusV1;
enum {
    VX_GRAPH_DOMAIN_STATUS_OK = 0,
    VX_GRAPH_DOMAIN_STATUS_INVALID_ARGUMENT = -1,
    VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE = -2,
    VX_GRAPH_DOMAIN_STATUS_RESPONSE_TOO_SMALL = -3,
    VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL = -4,
    VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN = -5,
    VX_GRAPH_DOMAIN_STATUS_GRAPH_BIND_FAILED = -6,
    VX_GRAPH_DOMAIN_STATUS_INTERNAL = -7
};

typedef int32_t VxGraphDomainErrorCodeV1;
enum {
    VX_GRAPH_DOMAIN_ERROR_NONE = 0,
    VX_GRAPH_DOMAIN_ERROR_INVALID_HEADER = 1,
    VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT = 2,
    VX_GRAPH_DOMAIN_ERROR_RESERVED_NONZERO = 3,
    VX_GRAPH_DOMAIN_ERROR_COUNT_MISMATCH = 4,
    VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW = 5,
    VX_GRAPH_DOMAIN_ERROR_PUBLIC_DOMAIN_NOT_SINGLETON = 6,
    VX_GRAPH_DOMAIN_ERROR_INVALID_DTYPE = 7,
    VX_GRAPH_DOMAIN_ERROR_UNSUPPORTED_OPERATOR_DOMAIN = 8,
    VX_GRAPH_DOMAIN_ERROR_OPERATOR_DOMAIN_REJECTED = 9,
    VX_GRAPH_DOMAIN_ERROR_OUTPUT_ASSERTION_MISMATCH = 10,
    VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT = 11,
    /* A nested graph-bind error is reported as this base plus its wire code. */
    VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE = 4096,
    VX_GRAPH_DOMAIN_ERROR_INTERNAL = 8191
};

typedef uint32_t VxGraphDomainErrorSectionV1;
enum {
    VX_GRAPH_DOMAIN_ERROR_SECTION_NONE = 0,
    VX_GRAPH_DOMAIN_ERROR_SECTION_HEADER = 1,
    VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION = 2,
    VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_PLAN = 3,
    VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN = 4,
    VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_BIND = 5,
    VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE = 6
};

typedef uint32_t VxGraphDomainProofModeV1;
enum {
    /* Complete exhaustive proof because the admitted public domain has one member. */
    VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE = 1,
    /* Algebraic proof over every legal public-input binding. */
    VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC = 2
};

typedef uint32_t VxGraphDomainAxisKindV1;
enum {
    VX_GRAPH_DOMAIN_AXIS_FIXED = 1,
    VX_GRAPH_DOMAIN_AXIS_SYMBOL = 2
};

typedef uint32_t VxGraphDomainRelationKindV1;
enum {
    VX_GRAPH_DOMAIN_RELATION_SELF = 1,
    VX_GRAPH_DOMAIN_RELATION_CONSTANT = 2,
    /* Exact target-to-source symbolic equality proved over the whole domain. */
    VX_GRAPH_DOMAIN_RELATION_SYMBOL = 3
};

typedef uint32_t VxGraphDomainFactCodeV1;
enum {
    VX_GRAPH_DOMAIN_FACT_SINGLETON_PUBLIC_DOMAIN = 1,
    VX_GRAPH_DOMAIN_FACT_PORTABLE_CONCRETE_NODE_ACCEPTED = 2,
    VX_GRAPH_DOMAIN_FACT_BOUNDED_SYMBOLIC_DOMAIN = 3,
    VX_GRAPH_DOMAIN_FACT_SYMBOLIC_NODE_ACCEPTED = 4,
    VX_GRAPH_DOMAIN_FACT_SYMBOLIC_DIRECT_PRESERVE = 5,
    VX_GRAPH_DOMAIN_FACT_SYMBOLIC_EXACT_BINARY = 6,
    VX_GRAPH_DOMAIN_FACT_SYMBOLIC_BROADCAST = 7,
    VX_GRAPH_DOMAIN_FACT_SYMBOLIC_STRUCTURAL = 8,
    VX_GRAPH_DOMAIN_FACT_SYMBOLIC_AFFINE = 9,
    VX_GRAPH_DOMAIN_FACT_SYMBOLIC_DIRECT_PROJECT = 10
};

/* Canonical packed layout after this header: dimensions, tensors, axes,
 * nodes, edges, relations, affine relations, public outputs, facts, strings,
 * then one embedded successful VxGraphBindResponseV1. */
typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainResponseV1 {
    uint32_t magic;
    uint32_t abi_version;
    VxGraphDomainStatusV1 status;
    VxGraphDomainErrorCodeV1 error_code;
    VxGraphDomainErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
    uint32_t written_bytes;
    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    VxGraphDomainProofModeV1 proof_mode;
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
    uint32_t relation_offset;
    uint32_t relation_count;
    uint32_t affine_relation_offset;
    uint32_t affine_relation_count;
    uint32_t public_output_offset;
    uint32_t public_output_count;
    uint32_t fact_offset;
    uint32_t fact_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t graph_bind_response_offset;
    uint32_t graph_bind_response_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} VxGraphDomainResponseV1;

/* Name offsets are relative to the graph-domain response string table. */
typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainDimensionV1 {
    uint32_t name_offset;
    uint32_t name_length;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple_of;
    uint32_t reserved0;
    uint32_t reserved1;
} VxGraphDomainDimensionV1;

/* Tensor record index is the graph-plan tensor index. Quantization arrays are
 * addressed by the same tensor record in the embedded graph-bind response. */
typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainTensorV1 {
    uint32_t name_offset;
    uint32_t name_length;
    uint32_t kind;
    int32_t dtype;
    uint32_t rank;
    uint32_t axis_first;
    int32_t quantization_scheme;
    uint32_t quantization_scale_bits;
    int32_t quantization_zero_point;
    uint32_t quantization_axis;
    uint32_t quantization_count;
    uint32_t graph_bind_tensor_index;
} VxGraphDomainTensorV1;

/* `value` is the fixed extent, or the sole value of a singleton-domain symbol.
 * It is zero for a genuinely dynamic symbol in BOUNDED_SYMBOLIC mode. */
typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainAxisV1 {
    VxGraphDomainAxisKindV1 kind;
    uint32_t dimension_index;
    uint64_t value;
} VxGraphDomainAxisV1;

typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainNodeV1 {
    uint32_t id_offset;
    uint32_t id_length;
    uint32_t operator_offset;
    uint32_t operator_length;
    uint32_t shape_function_id_offset;
    uint32_t shape_function_id_length;
    uint32_t input_edge_first;
    uint32_t input_edge_count;
    uint32_t output_edge_first;
    uint32_t output_edge_count;
    uint32_t fact_first;
    uint32_t fact_count;
    uint32_t request_node_index;
    uint32_t reserved;
} VxGraphDomainNodeV1;

typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainEdgeV1 {
    uint32_t port_offset;
    uint32_t port_length;
    uint32_t tensor_index;
    uint32_t reserved;
} VxGraphDomainEdgeV1;

typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainRelationV1 {
    uint32_t target_dimension_index;
    VxGraphDomainRelationKindV1 kind;
    uint32_t source_dimension_index;
    uint32_t reserved;
    uint64_t constant_value;
} VxGraphDomainRelationV1;

typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainAffineRelationV1 {
    uint32_t target_dimension_index;
    uint32_t source_dimension_index;
    int64_t offset;
} VxGraphDomainAffineRelationV1;

typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainPublicOutputV1 {
    uint32_t request_output_index;
    uint32_t tensor_index;
} VxGraphDomainPublicOutputV1;

typedef struct VX_GRAPH_DOMAIN_PACKED VxGraphDomainFactV1 {
    VxGraphDomainFactCodeV1 code;
    uint32_t text_offset;
    uint32_t text_length;
    uint32_t reserved;
} VxGraphDomainFactV1;

#if defined(__cplusplus)
static_assert(sizeof(VxGraphDomainResponseV1) == 144, "graph-domain response ABI drift");
static_assert(sizeof(VxGraphDomainDimensionV1) == 40, "graph-domain dimension ABI drift");
static_assert(sizeof(VxGraphDomainTensorV1) == 48, "graph-domain tensor ABI drift");
static_assert(sizeof(VxGraphDomainAxisV1) == 16, "graph-domain axis ABI drift");
static_assert(sizeof(VxGraphDomainNodeV1) == 56, "graph-domain node ABI drift");
static_assert(sizeof(VxGraphDomainEdgeV1) == 16, "graph-domain edge ABI drift");
static_assert(sizeof(VxGraphDomainRelationV1) == 24, "graph-domain relation ABI drift");
static_assert(sizeof(VxGraphDomainAffineRelationV1) == 16, "graph-domain affine ABI drift");
static_assert(sizeof(VxGraphDomainPublicOutputV1) == 8, "graph-domain output ABI drift");
static_assert(sizeof(VxGraphDomainFactV1) == 16, "graph-domain fact ABI drift");
#else
_Static_assert(sizeof(VxGraphDomainResponseV1) == 144, "graph-domain response ABI drift");
_Static_assert(sizeof(VxGraphDomainDimensionV1) == 40, "graph-domain dimension ABI drift");
_Static_assert(sizeof(VxGraphDomainTensorV1) == 48, "graph-domain tensor ABI drift");
_Static_assert(sizeof(VxGraphDomainAxisV1) == 16, "graph-domain axis ABI drift");
_Static_assert(sizeof(VxGraphDomainNodeV1) == 56, "graph-domain node ABI drift");
_Static_assert(sizeof(VxGraphDomainEdgeV1) == 16, "graph-domain edge ABI drift");
_Static_assert(sizeof(VxGraphDomainRelationV1) == 24, "graph-domain relation ABI drift");
_Static_assert(sizeof(VxGraphDomainAffineRelationV1) == 16, "graph-domain affine ABI drift");
_Static_assert(sizeof(VxGraphDomainPublicOutputV1) == 8, "graph-domain output ABI drift");
_Static_assert(sizeof(VxGraphDomainFactV1) == 16, "graph-domain fact ABI drift");
#endif

VX_GRAPH_DOMAIN_API uint32_t vx_graph_domain_abi_version(void);

/* All ranges are mutually disjoint. Definition and graph-plan ranges are
 * 4-byte aligned, response is 8-byte aligned, and scratch is 16-byte aligned.
 * Scratch may be null only when scratch_bytes is zero. A SCRATCH_TOO_SMALL
 * result supplies a strictly increasing retry lower bound; success reports
 * the exact scratch high-water used. Scratch is cleared before return and no
 * pointer or state is retained. */
VX_GRAPH_DOMAIN_API int32_t vx_graph_domain_prove_v1(
    const uint8_t* definition,
    uint32_t definition_bytes,
    const uint8_t* graph_plan_request,
    uint32_t graph_plan_request_bytes,
    const uint8_t* graph_plan_response,
    uint32_t graph_plan_response_bytes,
    uint8_t* response,
    uint32_t response_bytes,
    uint8_t* scratch,
    uint32_t scratch_bytes);

/* Allocation-free canonical decoder for a terminal graph-domain response.
 * It independently validates every success-body section/record/string and
 * the complete embedded graph-bind body against the immutable definition and
 * graph plan.  Semantic rejections are restricted to the error vocabulary and
 * index ranges represented by graph-domain v1.  The function does not replay
 * the proof and needs no workspace; it returns one exactly for a structurally
 * canonical terminal and zero otherwise. */
VX_GRAPH_DOMAIN_API int32_t vx_graph_domain_response_is_canonical_v1(
    const uint8_t* definition,
    uint32_t definition_bytes,
    const uint8_t* graph_plan_request,
    uint32_t graph_plan_request_bytes,
    const uint8_t* graph_plan_response,
    uint32_t graph_plan_response_bytes,
    const uint8_t* response,
    uint32_t response_bytes);

/* Strict immutable-cache boundary.  First run the independent canonical
 * decoder above, then re-run the portable proof over the same definition and
 * exact graph-plan pair and require a byte-identical terminal.  Structural
 * validation catches producer invariant drift; replay separately closes
 * provenance and mutation.  Graph-domain v1 does not retain a nested
 * graph-bind status/section (or nested shape-domain error code/index), so the
 * producer qualifies those fields before flattening and this validator admits
 * only the semantic vocabulary representable by the outer wire.
 *
 * `replay_response` and `replay_scratch` are caller-owned workspaces and must
 * obey the same alignment/disjointness rules as prove_v1.  The replay response
 * capacity must be at least `response_bytes`; the scratch capacity must be at
 * least the high-water used by the original proof.  The function returns one
 * only for a cacheable, byte-canonical terminal and zero otherwise.  Replay
 * scratch is cleared by prove_v1 and no pointer or state is retained. */
VX_GRAPH_DOMAIN_API int32_t vx_graph_domain_validate_response_v1(
    const uint8_t* definition,
    uint32_t definition_bytes,
    const uint8_t* graph_plan_request,
    uint32_t graph_plan_request_bytes,
    const uint8_t* graph_plan_response,
    uint32_t graph_plan_response_bytes,
    const uint8_t* response,
    uint32_t response_bytes,
    uint8_t* replay_response,
    uint32_t replay_response_bytes,
    uint8_t* replay_scratch,
    uint32_t replay_scratch_bytes);

#undef VX_GRAPH_DOMAIN_PACKED

#ifdef VX_GRAPH_DOMAIN_API_DEFINED_HERE
#undef VX_GRAPH_DOMAIN_API_DEFINED_HERE
#undef VX_GRAPH_DOMAIN_API
#endif

#ifdef __cplusplus
}
#endif

#endif
