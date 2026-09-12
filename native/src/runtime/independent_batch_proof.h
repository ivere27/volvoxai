#ifndef VOLVOXAI_RUNTIME_INDEPENDENT_BATCH_PROOF_H
#define VOLVOXAI_RUNTIME_INDEPENDENT_BATCH_PROOF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_INDEPENDENT_BATCH_PROOF_API
#define VX_INDEPENDENT_BATCH_PROOF_API
#define VX_INDEPENDENT_BATCH_PROOF_API_DEFINED_HERE 1
#endif

/*
 * Backend-neutral proof that one public symbolic dimension denotes mutually
 * independent scheduler lanes.
 *
 * The request embeds an immutable graph-bind definition and the exact
 * graph-plan request/response pair that produced it.  The core reruns the
 * complete graph-domain proof in caller scratch.  graph-domain in turn
 * recompiles and byte-compares the graph-plan response, so no hash, host
 * object identity, or caller assertion is accepted as provenance.
 *
 * All records are little-endian, fixed-width, pointer-free, and caller-owned.
 * The implementation allocates no memory, opens no files, acquires no locks,
 * and retains no pointer or state after the call.
 */
#define VOLVOXAI_INDEPENDENT_BATCH_PROOF_ABI_VERSION UINT32_C(1)
#define VOLVOXAI_INDEPENDENT_BATCH_PROOF_PROTOCOL_VERSION UINT32_C(1)
#define VOLVOXAI_INDEPENDENT_BATCH_REQUEST_MAGIC UINT32_C(0x31495856)  /* "VXI1" */
#define VOLVOXAI_INDEPENDENT_BATCH_RESPONSE_MAGIC UINT32_C(0x314A5856) /* "VXJ1" */
#define VX_INDEPENDENT_BATCH_INDEX_NONE UINT32_MAX

#if defined(__clang__) || defined(__GNUC__)
#define VX_INDEPENDENT_BATCH_PACKED __attribute__((packed, aligned(4)))
#else
#error "The VolvoxAI independent-batch ABI requires packed-record support."
#endif

typedef int32_t VxIndependentBatchStatusV1;
enum {
    VX_INDEPENDENT_BATCH_STATUS_OK = 0,
    VX_INDEPENDENT_BATCH_STATUS_INVALID_ARGUMENT = -1,
    VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE = -2,
    VX_INDEPENDENT_BATCH_STATUS_RESPONSE_TOO_SMALL = -3,
    VX_INDEPENDENT_BATCH_STATUS_SCRATCH_TOO_SMALL = -4,
    VX_INDEPENDENT_BATCH_STATUS_GRAPH_DOMAIN_FAILED = -5,
    VX_INDEPENDENT_BATCH_STATUS_INTERNAL = -6
};

typedef int32_t VxIndependentBatchErrorCodeV1;
enum {
    VX_INDEPENDENT_BATCH_ERROR_NONE = 0,
    VX_INDEPENDENT_BATCH_ERROR_INVALID_HEADER = 1,
    VX_INDEPENDENT_BATCH_ERROR_INVALID_LAYOUT = 2,
    VX_INDEPENDENT_BATCH_ERROR_RESERVED_NONZERO = 3,
    VX_INDEPENDENT_BATCH_ERROR_ARITHMETIC_OVERFLOW = 4,
    VX_INDEPENDENT_BATCH_ERROR_GRAPH_DOMAIN = 5,
    VX_INDEPENDENT_BATCH_ERROR_INTERNAL = 8191
};

typedef uint32_t VxIndependentBatchErrorSectionV1;
enum {
    VX_INDEPENDENT_BATCH_ERROR_SECTION_NONE = 0,
    VX_INDEPENDENT_BATCH_ERROR_SECTION_HEADER = 1,
    VX_INDEPENDENT_BATCH_ERROR_SECTION_DEFINITION = 2,
    VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_PLAN_REQUEST = 3,
    VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_PLAN_RESPONSE = 4,
    VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_DOMAIN = 5,
    VX_INDEPENDENT_BATCH_ERROR_SECTION_RESPONSE = 6
};

typedef uint32_t VxIndependentBatchReasonV1;
enum {
    VX_INDEPENDENT_BATCH_REASON_NONE = 0,
    VX_INDEPENDENT_BATCH_REASON_PUBLIC_AXIS = 1,
    VX_INDEPENDENT_BATCH_REASON_DIMENSION = 2,
    VX_INDEPENDENT_BATCH_REASON_WEIGHT_AXIS = 3,
    VX_INDEPENDENT_BATCH_REASON_REPEATED_AXIS = 4,
    VX_INDEPENDENT_BATCH_REASON_QUANTIZATION_AXIS = 5,
    VX_INDEPENDENT_BATCH_REASON_RESHAPE = 6,
    VX_INDEPENDENT_BATCH_REASON_TRANSPOSE = 7,
    VX_INDEPENDENT_BATCH_REASON_SQUEEZE = 8,
    VX_INDEPENDENT_BATCH_REASON_UNSQUEEZE = 9,
    VX_INDEPENDENT_BATCH_REASON_SLICE = 10,
    VX_INDEPENDENT_BATCH_REASON_CONCAT = 11,
    VX_INDEPENDENT_BATCH_REASON_EXPAND = 12,
    VX_INDEPENDENT_BATCH_REASON_GATHER = 13,
    VX_INDEPENDENT_BATCH_REASON_EMBEDDING = 14,
    VX_INDEPENDENT_BATCH_REASON_REDUCE_SUM = 15,
    VX_INDEPENDENT_BATCH_REASON_ARGMAX = 16,
    VX_INDEPENDENT_BATCH_REASON_SOFTMAX = 17,
    VX_INDEPENDENT_BATCH_REASON_BATCH_MATMUL = 18,
    VX_INDEPENDENT_BATCH_REASON_QBATCH_MATMUL = 19,
    VX_INDEPENDENT_BATCH_REASON_CONV2D = 20,
    VX_INDEPENDENT_BATCH_REASON_QCONV2D = 21,
    VX_INDEPENDENT_BATCH_REASON_GROUP_NORM = 22,
    VX_INDEPENDENT_BATCH_REASON_LAYER_NORM = 23,
    VX_INDEPENDENT_BATCH_REASON_LINEAR = 24,
    VX_INDEPENDENT_BATCH_REASON_QLINEAR = 25,
    VX_INDEPENDENT_BATCH_REASON_QGEMM = 26,
    VX_INDEPENDENT_BATCH_REASON_MATMUL = 27,
    VX_INDEPENDENT_BATCH_REASON_OPERATOR = 28,
    VX_INDEPENDENT_BATCH_REASON_NON_MAX_SUPPRESSION = 29
};

enum {
    VX_INDEPENDENT_BATCH_PROGRESS_INPUT = 1u,
    VX_INDEPENDENT_BATCH_PROGRESS_WEIGHT = 2u,
    VX_INDEPENDENT_BATCH_PROGRESS_VALUE = 4u,
    VX_INDEPENDENT_BATCH_PROGRESS_PUBLIC_OUTPUT = 8u
};

/* Canonical packed layout: header, definition, graph-plan request, graph-plan
 * response. Every section begins at the immediately following 4-byte aligned
 * offset; alignment padding must be zero, and trailing or reordered bytes are
 * rejected. */
typedef struct VX_INDEPENDENT_BATCH_PACKED VxIndependentBatchRequestV1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t total_bytes;
    uint32_t flags;
    uint32_t definition_offset;
    uint32_t definition_bytes;
    uint32_t graph_plan_request_offset;
    uint32_t graph_plan_request_bytes;
    uint32_t graph_plan_response_offset;
    uint32_t graph_plan_response_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
    uint32_t reserved4;
    uint32_t reserved5;
} VxIndependentBatchRequestV1;

/*
 * `supported == 0` with status OK is an authoritative semantic refusal.
 * String offsets are relative to the response string table.  No digest is a
 * proof identity: callers bind the decoded immutable evidence to the exact
 * retained request object/bytes that were fully revalidated by this call.
 */
typedef struct VX_INDEPENDENT_BATCH_PACKED VxIndependentBatchResponseV1 {
    uint32_t magic;
    uint32_t abi_version;
    VxIndependentBatchStatusV1 status;
    VxIndependentBatchErrorCodeV1 error_code;
    VxIndependentBatchErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
    uint32_t written_bytes;
    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    uint32_t supported;
    uint32_t protocol_version;
    uint32_t graph_plan_abi_version;
    uint32_t graph_bind_abi_version;
    uint32_t graph_domain_abi_version;
    uint32_t graph_domain_proof_mode;
    uint32_t batch_dimension_index;
    uint32_t batch_axis;
    uint64_t minimum_batch;
    uint64_t maximum_batch;
    uint64_t multiple_of;
    uint32_t covered_nodes;
    uint32_t node_count;
    uint32_t tensor_count;
    uint32_t progression_offset;
    uint32_t progression_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t batch_symbol_offset;
    uint32_t batch_symbol_length;
    VxIndependentBatchReasonV1 reason_code;
    uint32_t reason_offset;
    uint32_t reason_length;
    uint32_t failed_node_index;
    uint32_t failed_node_offset;
    uint32_t failed_node_length;
    uint32_t failed_tensor_index;
    uint32_t failed_tensor_offset;
    uint32_t failed_tensor_length;
    uint32_t definition_bytes;
    uint32_t graph_plan_request_bytes;
    uint32_t graph_plan_response_bytes;
    uint32_t graph_domain_response_bytes;
    uint32_t provenance_flags;
    uint32_t reserved0;
} VxIndependentBatchResponseV1;

typedef struct VX_INDEPENDENT_BATCH_PACKED VxIndependentBatchProgressV1 {
    uint32_t tensor_index;
    uint32_t producer_node_index;
    uint32_t batch_axis;
    uint32_t occurrences;
    uint32_t flags;
    uint32_t reserved;
} VxIndependentBatchProgressV1;

enum {
    /* Full graph-domain success, including byte-exact graph-plan recompile. */
    VX_INDEPENDENT_BATCH_PROVENANCE_GRAPH_DOMAIN = 1u
};

#if defined(__cplusplus)
static_assert(sizeof(VxIndependentBatchRequestV1) == 64,
              "independent-batch request ABI drift");
static_assert(sizeof(VxIndependentBatchResponseV1) == 192,
              "independent-batch response ABI drift");
static_assert(sizeof(VxIndependentBatchProgressV1) == 24,
              "independent-batch progression ABI drift");
#else
_Static_assert(sizeof(VxIndependentBatchRequestV1) == 64,
               "independent-batch request ABI drift");
_Static_assert(sizeof(VxIndependentBatchResponseV1) == 192,
               "independent-batch response ABI drift");
_Static_assert(sizeof(VxIndependentBatchProgressV1) == 24,
               "independent-batch progression ABI drift");
#endif

VX_INDEPENDENT_BATCH_PROOF_API uint32_t
vx_independent_batch_proof_abi_version(void);

/* Request and response are 8-byte aligned; scratch is 16-byte aligned. All
 * ranges are mutually disjoint. A zero-byte scratch call begins deterministic
 * sizing. SCRATCH_TOO_SMALL always reports a strictly increasing lower bound.
 * Scratch is cleared before return and no pointer is retained. */
VX_INDEPENDENT_BATCH_PROOF_API int32_t vx_independent_batch_prove_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    uint8_t* response,
    uint32_t response_bytes,
    uint8_t* scratch,
    uint32_t scratch_bytes);

/* Allocation-free structural decoder for the canonical successful-terminal
 * wire, including supported and semantic-refusal forms. It validates the
 * terminal section chain, progression/string records, reason/index/name
 * consistency, response reserved fields and version/provenance echoes against
 * a structurally framed nested request. It does not establish nested
 * definition/graph-plan semantic canonicality, exact scratch/domain high-water,
 * or provenance because it does not replay the proof. It needs no workspace;
 * sizing and error responses are never cacheable, and this decoder alone is
 * not a cache-trust boundary. */
VX_INDEPENDENT_BATCH_PROOF_API int32_t
vx_independent_batch_response_is_canonical_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    const uint8_t* response,
    uint32_t response_bytes);

/* Strict immutable-cache boundary, mandatory wherever a terminal is cached.
 * First apply the allocation-free decoder, then rerun the proof over the same
 * request and require a byte-identical successful terminal. Replay response
 * capacity must be at least response_bytes, and replay scratch capacity must
 * cover the terminal's required_scratch_bytes. All ranges obey prove_v1
 * alignment/disjointness;
 * replay scratch is cleared by prove_v1 and no state is retained. */
VX_INDEPENDENT_BATCH_PROOF_API int32_t
vx_independent_batch_validate_response_v1(
    const uint8_t* request,
    uint32_t request_bytes,
    const uint8_t* response,
    uint32_t response_bytes,
    uint8_t* replay_response,
    uint32_t replay_response_bytes,
    uint8_t* replay_scratch,
    uint32_t replay_scratch_bytes);

#undef VX_INDEPENDENT_BATCH_PACKED

#ifdef VX_INDEPENDENT_BATCH_PROOF_API_DEFINED_HERE
#undef VX_INDEPENDENT_BATCH_PROOF_API_DEFINED_HERE
#undef VX_INDEPENDENT_BATCH_PROOF_API
#endif

#ifdef __cplusplus
}
#endif

#endif
