#ifndef VOLVOXAI_RUNTIME_SHAPE_CONTRACT_H
#define VOLVOXAI_RUNTIME_SHAPE_CONTRACT_H

#include "volvoxai_enums.h"
#include "../generated/operator_vocabulary.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_SHAPE_CONTRACT_API
#define VX_SHAPE_CONTRACT_API
#endif

/* The graph v1 shape system is intentionally narrower than uint64_t. */
#define VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER UINT64_C(9007199254740991)

typedef enum VxShapeContractErrorCode {
    VX_SHAPE_CONTRACT_ERROR_NONE = 0,
    VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
    VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST,
    VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
    VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS,
    VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
    VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
    VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
    VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
    VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
    VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
    VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
    VX_SHAPE_CONTRACT_ERROR_INVALID_DOMAIN,
    VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY
} VxShapeContractErrorCode;

typedef struct VxShapeContractError {
    VxShapeContractErrorCode code;
    char path[192];
    char detail[320];
} VxShapeContractError;

typedef enum VxShapeQuantizationScheme {
    VX_SHAPE_QUANTIZATION_NONE = 0,
    VX_SHAPE_QUANTIZATION_PER_TENSOR = 1,
    VX_SHAPE_QUANTIZATION_PER_AXIS = 2
} VxShapeQuantizationScheme;

typedef struct VxShapeQuantization {
    VxShapeQuantizationScheme scheme;
    float scale;
    int32_t zero_point;
    size_t axis;
    size_t count;
    const float* scales;
    const int32_t* zero_points;
} VxShapeQuantization;

/* Request descriptors borrow all arrays for the duration of inference. */
typedef struct VxShapeTensorDescriptor {
    size_t rank;
    const uint64_t* shape;
    VxDataType dtype;
    VxShapeQuantization quantization;
} VxShapeTensorDescriptor;

typedef struct VxShapeNamedTensor {
    const char* name;
    VxShapeTensorDescriptor descriptor;
} VxShapeNamedTensor;

typedef enum VxShapeParamKind {
    VX_SHAPE_PARAM_NUMBER = 1,
    VX_SHAPE_PARAM_BOOLEAN = 2,
    VX_SHAPE_PARAM_STRING = 3,
    VX_SHAPE_PARAM_NUMBER_ARRAY = 4
} VxShapeParamKind;

typedef struct VxShapeNumberArray {
    size_t count;
    const double* values;
} VxShapeNumberArray;

typedef struct VxShapeParam {
    const char* name;
    VxShapeParamKind kind;
    union {
        double number;
        int boolean;
        const char* string;
        VxShapeNumberArray number_array;
    } value;
} VxShapeParam;

typedef struct VxConcreteShapeRequest {
    const VxShapeNamedTensor* inputs;
    size_t input_count;
    const VxShapeParam* params;
    size_t param_count;
    const VxShapeNamedTensor* declared_outputs;
    size_t declared_output_count;
} VxConcreteShapeRequest;

typedef enum VxShapeDeclaredOutputMode {
    VX_SHAPE_DECLARED_OUTPUT_NONE = 0,
    /* Concrete inference can run from concrete params alone. */
    VX_SHAPE_DECLARED_OUTPUT_OPTIONAL_CONCRETE = 1,
    /* Concrete inference consumes the graph-authored output descriptor. */
    VX_SHAPE_DECLARED_OUTPUT_REQUIRED_CONCRETE = 2
} VxShapeDeclaredOutputMode;

/* A successful result owns its descriptor arrays until clear is called. */
typedef struct VxConcreteShapeResult {
    const char* shape_function_id;
    VxShapeNamedTensor* outputs;
    size_t output_count;
    char** owned_names;
    uint64_t** owned_shapes;
    float** owned_scales;
    int32_t** owned_zero_points;
} VxConcreteShapeResult;

#define VX_CONCRETE_SHAPE_RESULT_INITIALIZER {0}

/* Non-owning projection returned only by the caller-owned scratch facade.
 * Its pointers become invalid as soon as the supplied scratch range is reused
 * or retired. It is discarded as plain metadata and is never passed to the
 * owning result_clear API. */
typedef struct VxBorrowedConcreteShapeResult {
    const char* shape_function_id;
    const VxShapeNamedTensor* outputs;
    size_t output_count;
} VxBorrowedConcreteShapeResult;

#define VX_BORROWED_CONCRETE_SHAPE_RESULT_INITIALIZER {0}

VX_SHAPE_CONTRACT_API
const char* vx_shape_contract_error_code_name(VxShapeContractErrorCode code);

/* Returns the generated stable ID for an implemented canonical operator. */
VX_SHAPE_CONTRACT_API
const char* vx_shape_contract_function_id(const char* operator_name);

/* One canonical policy used by graph-wide binding. OPTIONAL means concrete
 * params are authoritative and the inferred descriptor is checked/bound
 * afterward; it must not be synthesized from an unbound output symbol.
 * Concat is intentionally excluded because it consumes its graph-authored
 * descriptor only during logical bounded-domain proof. */
VX_SHAPE_CONTRACT_API
VxShapeDeclaredOutputMode vx_shape_contract_declared_output_mode(
    VxOperatorKind operator_kind);

/*
 * The result must be zero-initialized before its first use. Returns 0 on
 * success and -1 on failure. A failed inference leaves an existing result
 * unchanged; a success replaces and releases its old value.
 *
 * `declared_outputs` is required and must name every output port with its
 * rank. One evaluator now serves both the concrete and the symbolic domain and
 * it sizes its output slots from that declaration. `declared_output_mode`
 * still decides whether the declared extents are semantic input or only an
 * assertion the inferred result has to satisfy; a caller that cannot resolve an
 * extent yet passes a placeholder for a mode that infers.
 */
VX_SHAPE_CONTRACT_API
int vx_shape_contract_infer(const char* operator_name,
                            const VxConcreteShapeRequest* request,
                            VxConcreteShapeResult* result,
                            VxShapeContractError* error);

/*
 * Allocator-free inference facade for graph-wide portable orchestration.
 * Every successful result allocation and temporary comes from the supplied
 * 16-byte-aligned caller-owned scratch range. The callee retains no pointer or
 * state after return. The caller discards the borrowed metadata before the
 * range is reused or retired; result data is valid only while it remains live.
 *
 * On an allocation failure required_scratch_bytes is a monotonic retry lower
 * bound. It is exact on success. Result pointers borrow scratch and require no
 * clear operation; a failure resets the borrowed result to empty metadata.
 */
VX_SHAPE_CONTRACT_API
int vx_shape_contract_infer_with_scratch(
    const char* operator_name,
    const VxConcreteShapeRequest* request,
    VxBorrowedConcreteShapeResult* result,
    VxShapeContractError* error,
    void* scratch,
    size_t scratch_bytes,
    size_t* required_scratch_bytes);

VX_SHAPE_CONTRACT_API
void vx_shape_contract_result_clear(VxConcreteShapeResult* result);

#ifdef __cplusplus
}
#endif

#endif
