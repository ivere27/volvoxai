#ifndef VOLVOXAI_RUNTIME_SHAPE_CONTRACT_H
#define VOLVOXAI_RUNTIME_SHAPE_CONTRACT_H

#include "volvoxai_enums.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
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

const char* vx_shape_contract_error_code_name(VxShapeContractErrorCode code);

/* Returns the generated stable ID for an implemented canonical operator. */
const char* vx_shape_contract_function_id(const char* operator_name);

/*
 * The result must be zero-initialized before its first use. Returns 0 on
 * success and -1 on failure. A failed inference leaves an existing result
 * unchanged; a success replaces and releases its old value.
 */
int vx_shape_contract_infer(const char* operator_name,
                            const VxConcreteShapeRequest* request,
                            VxConcreteShapeResult* result,
                            VxShapeContractError* error);

void vx_shape_contract_result_clear(VxConcreteShapeResult* result);

#ifdef __cplusplus
}
#endif

#endif
