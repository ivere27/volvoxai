#ifndef VOLVOXAI_RUNTIME_SHAPE_DOMAIN_CONTRACT_H
#define VOLVOXAI_RUNTIME_SHAPE_DOMAIN_CONTRACT_H

#include "volvoxai_enums.h"
#include "../generated/operator_vocabulary.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VX_SHAPE_DOMAIN_CONTRACT_API
#define VX_SHAPE_DOMAIN_CONTRACT_API
#define VX_SHAPE_DOMAIN_CONTRACT_API_HEADER_LOCAL_EMPTY 1
#endif

/*
 * Allocator-free logical shape-domain contracts.
 *
 * Requests borrow normalized package metadata for one call. Results borrow
 * the caller's 16-byte-aligned scratch range and are discarded before that
 * range is reused. No pointer is retained, no global state is mutated, and no
 * concrete binding or sampled corner is accepted as bounded-domain evidence.
 */

#define VX_SHAPE_DOMAIN_INDEX_NONE UINT32_MAX
#define VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER UINT64_C(9007199254740991)

typedef int32_t VxShapeDomainStatus;
enum {
    VX_SHAPE_DOMAIN_STATUS_OK = 0,
    VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED = 1,
    VX_SHAPE_DOMAIN_STATUS_INVALID = -1,
    VX_SHAPE_DOMAIN_STATUS_SCRATCH_TOO_SMALL = -2
};

typedef int32_t VxShapeDomainErrorCode;
enum {
    VX_SHAPE_DOMAIN_ERROR_NONE = 0,
    VX_SHAPE_DOMAIN_ERROR_UNSUPPORTED_OPERATOR = 1,
    VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS = 2,
    VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE = 3,
    VX_SHAPE_DOMAIN_ERROR_INVALID_RANK = 4,
    VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS = 5,
    VX_SHAPE_DOMAIN_ERROR_INVALID_DIMENSION = 6,
    VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH = 7,
    VX_SHAPE_DOMAIN_ERROR_UNPROVABLE = 8,
    VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION = 9,
    VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW = 10,
    VX_SHAPE_DOMAIN_ERROR_INTERNAL = 11
};

typedef uint32_t VxShapeDomainDimensionKind;
enum {
    VX_SHAPE_DOMAIN_DIMENSION_FIXED = 1,
    VX_SHAPE_DOMAIN_DIMENSION_SYMBOL = 2,
    VX_SHAPE_DOMAIN_DIMENSION_AFFINE = 3
};

/* FIXED uses value as a positive extent and symbol_index == NONE. SYMBOL uses
 * symbol_index and value == 0. AFFINE means symbol_index + signed value. */
typedef struct VxShapeDomainDimension {
    VxShapeDomainDimensionKind kind;
    uint32_t symbol_index;
    int64_t value;
} VxShapeDomainDimension;

typedef struct VxShapeDomainConstraint {
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple_of;
} VxShapeDomainConstraint;

typedef struct VxShapeDomainProgression {
    uint64_t first;
    uint64_t last;
    uint64_t step;
    uint64_t count;
} VxShapeDomainProgression;

typedef struct VxShapeDomainString {
    const uint8_t* bytes;
    uint32_t length;
} VxShapeDomainString;

typedef struct VxShapeDomainQuantization {
    int32_t scheme;
    uint32_t scale_bits;
    int32_t zero_point;
    uint32_t axis;
    uint32_t count;
    const uint32_t* scales;
    const int32_t* zero_points;
} VxShapeDomainQuantization;

typedef struct VxShapeDomainTensorDescriptor {
    int32_t dtype;
    uint32_t rank;
    const VxShapeDomainDimension* dimensions;
    VxShapeDomainQuantization quantization;
} VxShapeDomainTensorDescriptor;

typedef struct VxShapeDomainNamedTensor {
    VxShapeDomainString name;
    uint32_t tensor_index;
    VxShapeDomainTensorDescriptor descriptor;
} VxShapeDomainNamedTensor;

typedef uint32_t VxShapeDomainParamKind;
enum {
    VX_SHAPE_DOMAIN_PARAM_NUMBER = 1,
    VX_SHAPE_DOMAIN_PARAM_BOOLEAN = 2,
    VX_SHAPE_DOMAIN_PARAM_STRING = 3,
    VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY = 4
};

typedef uint32_t VxShapeDomainParamValueKind;
enum {
    VX_SHAPE_DOMAIN_PARAM_VALUE_NUMBER = 1,
    VX_SHAPE_DOMAIN_PARAM_VALUE_DIMENSION = 2
};

typedef struct VxShapeDomainParamValue {
    VxShapeDomainParamValueKind kind;
    uint32_t dimension_index;
    double number;
} VxShapeDomainParamValue;

typedef struct VxShapeDomainParam {
    VxShapeDomainString name;
    VxShapeDomainParamKind kind;
    uint32_t reserved;
    double number;
    int32_t boolean;
    VxShapeDomainString string;
    const VxShapeDomainParamValue* values;
    uint32_t value_count;
} VxShapeDomainParam;

typedef struct VxShapeDomainAffineWitness {
    uint32_t target_dimension_index;
    uint32_t source_dimension_index;
    int64_t offset;
} VxShapeDomainAffineWitness;

typedef uint32_t VxShapeDomainFactKind;
enum {
    VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE = 1,
    VX_INTERNAL_SHAPE_DOMAIN_FACT_EXACT_BINARY = 2,
    VX_INTERNAL_SHAPE_DOMAIN_FACT_BROADCAST = 3,
    VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL = 4,
    VX_INTERNAL_SHAPE_DOMAIN_FACT_AFFINE = 5,
    VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT = 6
};

typedef struct VxShapeDomainProofRequest {
    VxOperatorKind operator_kind;
    VxShapeDomainString operator_name;
    const VxShapeDomainConstraint* environment;
    uint32_t environment_count;
    const VxShapeDomainNamedTensor* inputs;
    uint32_t input_count;
    const VxShapeDomainNamedTensor* declared_outputs;
    uint32_t declared_output_count;
    int use_declared_outputs;
    const VxShapeDomainParam* params;
    uint32_t param_count;
} VxShapeDomainProofRequest;

typedef struct VxShapeDomainProofResult {
    const char* shape_function_id;
    VxShapeDomainNamedTensor* outputs;
    uint32_t output_count;
    VxShapeDomainAffineWitness* affine_witnesses;
    uint32_t affine_witness_count;
    VxShapeDomainFactKind fact_kind;
} VxShapeDomainProofResult;

typedef struct VxShapeDomainError {
    VxShapeDomainErrorCode code;
    uint32_t index;
    uint32_t subindex;
} VxShapeDomainError;

VX_SHAPE_DOMAIN_CONTRACT_API int vx_shape_domain_legal_progression(
    const VxShapeDomainConstraint* environment,
    uint32_t environment_count,
    uint32_t symbol_index,
    VxShapeDomainProgression* progression);

VX_SHAPE_DOMAIN_CONTRACT_API int vx_shape_domain_fixed_value(
    const VxShapeDomainDimension* dimension,
    const VxShapeDomainConstraint* environment,
    uint32_t environment_count,
    uint64_t* value);

VX_SHAPE_DOMAIN_CONTRACT_API int vx_shape_domain_dimensions_equal(
    const VxShapeDomainDimension* left,
    const VxShapeDomainDimension* right,
    const VxShapeDomainConstraint* environment,
    uint32_t environment_count);

VX_SHAPE_DOMAIN_CONTRACT_API int32_t vx_shape_domain_contract_prove(
    const VxShapeDomainProofRequest* request,
    VxShapeDomainProofResult* result,
    VxShapeDomainError* error,
    void* scratch,
    uint32_t scratch_bytes,
    uint32_t* required_scratch_bytes);

#ifdef __cplusplus
}
#endif

#ifdef VX_SHAPE_DOMAIN_CONTRACT_API_HEADER_LOCAL_EMPTY
#undef VX_SHAPE_DOMAIN_CONTRACT_API_HEADER_LOCAL_EMPTY
#undef VX_SHAPE_DOMAIN_CONTRACT_API
#endif

#endif
