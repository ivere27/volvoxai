#ifndef VOLVOXAI_RUNTIME_CALL_SEQUENCE_POLICY_H
#define VOLVOXAI_RUNTIME_CALL_SEQUENCE_POLICY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_CALL_SEQUENCE_POLICY_API
#define VX_CALL_SEQUENCE_POLICY_API
#define VX_CALL_SEQUENCE_POLICY_API_DEFINED_HERE 1
#endif

/*
 * Backend-neutral topology-only policy for one immutable call sequence.
 *
 * The request embeds the exact canonical graph-plan request and successful
 * response pair.  Before deriving any policy, the implementation recompiles
 * that request in caller-owned scratch and requires the complete response to
 * be byte-identical.  Host graph shadows, digests, and object identities are
 * not provenance.
 *
 * This ABI owns only node selection and cross-call liveness.  Dtype, logical
 * size, aliasing, fusion, lifetime birth widening, physical packing, backend
 * execution, and mutable call state are deliberately outside this contract.
 * All records are little-endian, fixed-width, pointer-free, and caller-owned.
 * The implementation allocates no memory, opens no files, acquires no locks,
 * and retains no pointer or state after a call.
 */
#define VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION UINT32_C(1)
#define VOLVOXAI_CALL_SEQUENCE_POLICY_PROTOCOL_VERSION UINT32_C(1)
#define VOLVOXAI_CALL_SEQUENCE_POLICY_REQUEST_MAGIC UINT32_C(0x31435856)  /* "VXC1" */
#define VOLVOXAI_CALL_SEQUENCE_POLICY_RESPONSE_MAGIC UINT32_C(0x314B5856) /* "VXK1" */
#define VX_CALL_SEQUENCE_POLICY_INDEX_NONE UINT32_MAX

#if defined(__clang__) || defined(__GNUC__)
#define VX_CALL_SEQUENCE_POLICY_PACKED __attribute__((packed, aligned(4)))
#else
#error "The VolvoxAI call-sequence-policy ABI requires packed-record support."
#endif

typedef int32_t VxCallSequencePolicyStatusV1;
enum {
    VX_CALL_SEQUENCE_POLICY_STATUS_OK = 0,
    VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_ARGUMENT = -1,
    VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE = -2,
    VX_CALL_SEQUENCE_POLICY_STATUS_RESPONSE_TOO_SMALL = -3,
    VX_CALL_SEQUENCE_POLICY_STATUS_SCRATCH_TOO_SMALL = -4,
    VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN = -5,
    VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_POLICY = -6,
    VX_CALL_SEQUENCE_POLICY_STATUS_INTERNAL = -7
};

typedef int32_t VxCallSequencePolicyErrorCodeV1;
enum {
    VX_CALL_SEQUENCE_POLICY_ERROR_NONE = 0,
    VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_HEADER = 1,
    VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_LAYOUT = 2,
    VX_CALL_SEQUENCE_POLICY_ERROR_RESERVED_NONZERO = 3,
    VX_CALL_SEQUENCE_POLICY_ERROR_ARITHMETIC_OVERFLOW = 4,
    VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN = 5,
    VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_POLICY_KIND = 6,
    VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_CHANGED_INPUT = 7,
    VX_CALL_SEQUENCE_POLICY_ERROR_NONCANONICAL_CHANGED_INPUT = 8,
    VX_CALL_SEQUENCE_POLICY_ERROR_INTERNAL = 8191
};

typedef uint32_t VxCallSequencePolicyErrorSectionV1;
enum {
    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_NONE = 0,
    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_HEADER = 1,
    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST = 2,
    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE = 3,
    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_POLICY = 4,
    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT = 5,
    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_RESPONSE = 6
};

typedef uint32_t VxCallSequencePolicyKindV1;
enum {
    VX_CALL_SEQUENCE_POLICY_FORWARD_ONLY = 1,
    VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY = 2,
    VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE = 3
};

typedef uint32_t VxCallSequenceSelectionKindV1;
enum {
    /* Every graph-plan schedule node is selected. */
    VX_CALL_SEQUENCE_SELECTION_ALL = 1,
    /* The selected-node flags encode one exact fixed dependency closure. */
    VX_CALL_SEQUENCE_SELECTION_EXPLICIT = 2
};

enum {
    VX_CALL_SEQUENCE_NODE_SELECTED = 1u,
    VX_CALL_SEQUENCE_TENSOR_CROSS_CALL_LIVE = 1u,
    VX_CALL_SEQUENCE_PROVENANCE_GRAPH_PLAN_REPLAY = 1u
};

/*
 * Canonical packed layout:
 *
 *   header, graph-plan request, zero alignment padding, graph-plan response,
 *   changed-input records
 *
 * The graph-plan response and changed-input section begin at the immediately
 * following four-byte-aligned offset.  Alignment padding must be zero.  When
 * changed_input_count is zero, changed_input_offset is also zero and the
 * request ends immediately after the graph-plan response.  Only the fixed
 * policy may carry changed-input records; its empty set remains legal.
 */
typedef struct VX_CALL_SEQUENCE_POLICY_PACKED VxCallSequencePolicyRequestV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t total_bytes;
    uint32_t flags;
    uint32_t graph_plan_request_offset;
    uint32_t graph_plan_request_bytes;
    uint32_t graph_plan_response_offset;
    uint32_t graph_plan_response_bytes;
    VxCallSequencePolicyKindV1 policy_kind;
    uint32_t changed_input_offset;
    uint32_t changed_input_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
    uint32_t reserved4;
} VxCallSequencePolicyRequestV1;

/* INPUT tensors occupy canonical source-name order in graph-plan tensor space;
 * records are strictly increasing by that canonical tensor index. */
typedef struct VX_CALL_SEQUENCE_POLICY_PACKED VxCallSequenceChangedInputV1 {
    uint32_t tensor_index;
    uint32_t reserved;
} VxCallSequenceChangedInputV1;

/*
 * Successful terminals always contain the complete node and tensor tables.
 * FORWARD_ONLY and ALL_CROSS_CALL_LIVE use selection_kind ALL and select the
 * full graph-plan schedule.  Only FIXED_INPUT_DEPENDENCY may return an
 * EXPLICIT (possibly empty) dependency closure.  No digest is a proof identity:
 * a cache binds this decoded evidence to the exact immutable retained request
 * bytes that passed validator B.
 */
typedef struct VX_CALL_SEQUENCE_POLICY_PACKED VxCallSequencePolicyResponseV1 {
    uint32_t magic;
    uint32_t abi_version;
    VxCallSequencePolicyStatusV1 status;
    VxCallSequencePolicyErrorCodeV1 error_code;
    VxCallSequencePolicyErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
    uint32_t written_bytes;
    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    uint32_t protocol_version;
    uint32_t graph_plan_abi_version;
    VxCallSequencePolicyKindV1 policy_kind;
    VxCallSequenceSelectionKindV1 selection_kind;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t selected_node_count;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t changed_input_count;
    uint32_t graph_plan_request_bytes;
    uint32_t graph_plan_response_bytes;
    uint32_t provenance_flags;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
    uint32_t reserved4;
} VxCallSequencePolicyResponseV1;

/* Record index and node_index both equal graph-plan schedule order. */
typedef struct VX_CALL_SEQUENCE_POLICY_PACKED VxCallSequenceNodeV1 {
    uint32_t node_index;
    uint32_t flags;
} VxCallSequenceNodeV1;

/* Record index and tensor_index both equal graph-plan tensor order. */
typedef struct VX_CALL_SEQUENCE_POLICY_PACKED VxCallSequenceTensorV1 {
    uint32_t tensor_index;
    uint32_t flags;
} VxCallSequenceTensorV1;

#if defined(__cplusplus)
static_assert(sizeof(VxCallSequencePolicyRequestV1) == 64,
              "call-sequence-policy request ABI drift");
static_assert(sizeof(VxCallSequenceChangedInputV1) == 8,
              "call-sequence-policy changed-input ABI drift");
static_assert(sizeof(VxCallSequencePolicyResponseV1) == 112,
              "call-sequence-policy response ABI drift");
static_assert(sizeof(VxCallSequenceNodeV1) == 8,
              "call-sequence-policy node ABI drift");
static_assert(sizeof(VxCallSequenceTensorV1) == 8,
              "call-sequence-policy tensor ABI drift");
#else
_Static_assert(sizeof(VxCallSequencePolicyRequestV1) == 64,
               "call-sequence-policy request ABI drift");
_Static_assert(sizeof(VxCallSequenceChangedInputV1) == 8,
               "call-sequence-policy changed-input ABI drift");
_Static_assert(sizeof(VxCallSequencePolicyResponseV1) == 112,
               "call-sequence-policy response ABI drift");
_Static_assert(sizeof(VxCallSequenceNodeV1) == 8,
               "call-sequence-policy node ABI drift");
_Static_assert(sizeof(VxCallSequenceTensorV1) == 8,
               "call-sequence-policy tensor ABI drift");
#endif

VX_CALL_SEQUENCE_POLICY_API uint32_t
vx_call_sequence_policy_abi_version(void);

/* Request and response are eight-byte aligned; scratch is sixteen-byte
 * aligned.  All ranges are mutually disjoint.  A null scratch pointer is
 * accepted only when scratch_bytes is zero for deterministic sizing.  The
 * complete provided scratch range is cleared before every materialized
 * return, and no pointer is retained. */
VX_CALL_SEQUENCE_POLICY_API int32_t vx_call_sequence_policy_compile_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    uint8_t* response,
    uint32_t response_bytes,
    uint8_t* scratch,
    uint32_t scratch_bytes);

/*
 * Structural validator A for an exact successful terminal.  It validates the
 * canonical request framing, graph-plan pair structure, response section
 * chain, every full node/tensor record, exact byte counts, reserved fields,
 * and the exact composed scratch formula.  FORWARD_ONLY and
 * ALL_CROSS_CALL_LIVE tables are exact.  For FIXED_INPUT_DEPENDENCY it checks
 * complete indices, flag vocabulary, selected-count consistency, and the
 * structural prohibition on weight CROSS flags, but does not recompute the
 * dependency closure or CROSS set.  It allocates no workspace.
 *
 * A does not establish graph-plan semantic provenance because it deliberately
 * does not invoke vx_graph_plan_compile_v1.  Sizing and error responses are
 * never cacheable, and A alone does not establish exact fixed semantics or form
 * an immutable-cache trust boundary.
 */
VX_CALL_SEQUENCE_POLICY_API int32_t
vx_call_sequence_policy_response_is_structurally_safe_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    const uint8_t* response,
    uint32_t response_bytes);

/*
 * Replay validator B, mandatory wherever a terminal is cached.  It first
 * applies A, then recompiles the same request with caller-owned replay ranges
 * and requires a byte-identical successful terminal, thereby establishing the
 * exact selected-node and CROSS sets as well as graph-plan provenance.
 * replay_response must
 * hold at least response_bytes; replay_scratch must hold at least the
 * terminal's exact required_scratch_bytes.  All four ranges are mutually
 * disjoint and obey the compile_v1 alignment rules.  Replay scratch is cleared
 * before return.
 */
VX_CALL_SEQUENCE_POLICY_API int32_t
vx_call_sequence_policy_validate_response_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    const uint8_t* response,
    uint32_t response_bytes,
    uint8_t* replay_response,
    uint32_t replay_response_bytes,
    uint8_t* replay_scratch,
    uint32_t replay_scratch_bytes);

#undef VX_CALL_SEQUENCE_POLICY_PACKED

#ifdef VX_CALL_SEQUENCE_POLICY_API_DEFINED_HERE
#undef VX_CALL_SEQUENCE_POLICY_API_DEFINED_HERE
#undef VX_CALL_SEQUENCE_POLICY_API
#endif

#ifdef __cplusplus
}
#endif

#endif
