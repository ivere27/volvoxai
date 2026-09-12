#include "runtime/shape_contract.h"
#include "runtime/shape_domain_contract.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>


static VxShapeTensorDescriptor f32_tensor(
    const uint64_t* shape,
    size_t rank) {
    VxShapeTensorDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.rank = rank;
    descriptor.shape = shape;
    descriptor.dtype = VX_DTYPE_F32;
    return descriptor;
}


static VxShapeDomainString domain_string(const char* value) {
    VxShapeDomainString result;
    result.bytes = (const uint8_t*)value;
    result.length = (uint32_t)strlen(value);
    return result;
}


static VxShapeDomainTensorDescriptor domain_f32_tensor(
    const VxShapeDomainDimension* dimensions,
    uint32_t rank) {
    VxShapeDomainTensorDescriptor result;
    memset(&result, 0, sizeof(result));
    result.dtype = VX_DTYPE_F32;
    result.rank = rank;
    result.dimensions = dimensions;
    return result;
}


static int test_symbolic_domain_reuse(void) {
    static const VxShapeDomainConstraint environment[] = {
        {.minimum = 1, .maximum = 4, .multiple_of = 1},
    };
    static const VxShapeDomainDimension left_dimensions[] = {
        {VX_SHAPE_DOMAIN_DIMENSION_SYMBOL, 0, 0},
        {VX_SHAPE_DOMAIN_DIMENSION_FIXED, VX_SHAPE_DOMAIN_INDEX_NONE, 4},
    };
    static const VxShapeDomainDimension mismatch_dimensions[] = {
        {VX_SHAPE_DOMAIN_DIMENSION_SYMBOL, 0, 0},
        {VX_SHAPE_DOMAIN_DIMENSION_FIXED, VX_SHAPE_DOMAIN_INDEX_NONE, 3},
    };
    static const VxShapeDomainDimension valid_dimensions[] = {
        {VX_SHAPE_DOMAIN_DIMENSION_SYMBOL, 0, 0},
        {VX_SHAPE_DOMAIN_DIMENSION_FIXED, VX_SHAPE_DOMAIN_INDEX_NONE, 4},
    };
    VxShapeDomainNamedTensor inputs[2];
    VxShapeDomainNamedTensor outputs[1];
    VxShapeDomainProofRequest request;
    VxShapeDomainProofResult result;
    VxShapeDomainError error;
    uint32_t required_scratch_bytes = 0;
    _Alignas(16) unsigned char scratch[4096];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));
    memset(&request, 0, sizeof(request));
    memset(&result, 0, sizeof(result));
    memset(&error, 0, sizeof(error));
    inputs[0].name = domain_string("a");
    inputs[0].tensor_index = 0;
    inputs[0].descriptor = domain_f32_tensor(left_dimensions, 2);
    inputs[1].name = domain_string("b");
    inputs[1].tensor_index = 1;
    inputs[1].descriptor = domain_f32_tensor(mismatch_dimensions, 2);
    outputs[0].name = domain_string("out");
    outputs[0].tensor_index = 2;
    outputs[0].descriptor = domain_f32_tensor(valid_dimensions, 2);
    request.operator_kind = VX_OP_ADD;
    request.operator_name = domain_string("Add");
    request.environment = environment;
    request.environment_count = 1;
    request.inputs = inputs;
    request.input_count = 2;
    request.declared_outputs = outputs;
    request.declared_output_count = 1;
    request.use_declared_outputs = 1;

    if (
        vx_shape_domain_contract_prove(
            &request, &result, &error, scratch, sizeof(scratch),
            &required_scratch_bytes
        ) != VX_SHAPE_DOMAIN_STATUS_INVALID
        || error.code != VX_SHAPE_DOMAIN_ERROR_UNPROVABLE
        || error.index != 0
        || error.subindex != 1
        || result.shape_function_id != NULL
        || result.outputs != NULL
        || result.output_count != 0
        || required_scratch_bytes == 0
        || required_scratch_bytes > sizeof(scratch)
    ) {
        fprintf(stderr, "native semantics fixture: symbolic mismatch was not rejected transactionally\n");
        return 0;
    }

    /* Reuse the same request/result/error/scratch objects immediately. */
    inputs[1].descriptor = domain_f32_tensor(valid_dimensions, 2);
    if (
        vx_shape_domain_contract_prove(
            &request, &result, &error, scratch, sizeof(scratch),
            &required_scratch_bytes
        ) != VX_SHAPE_DOMAIN_STATUS_OK
        || error.code != VX_SHAPE_DOMAIN_ERROR_NONE
        || !result.shape_function_id
        || strcmp(result.shape_function_id, "volvox.shape.broadcast-arithmetic.v1") != 0
        || result.fact_kind != VX_INTERNAL_SHAPE_DOMAIN_FACT_EXACT_BINARY
        || result.output_count != 1
        || result.outputs[0].descriptor.rank != 2
        || result.outputs[0].descriptor.dimensions[0].kind
            != VX_SHAPE_DOMAIN_DIMENSION_SYMBOL
        || result.outputs[0].descriptor.dimensions[0].symbol_index != 0
        || result.outputs[0].descriptor.dimensions[1].kind
            != VX_SHAPE_DOMAIN_DIMENSION_FIXED
        || result.outputs[0].descriptor.dimensions[1].value != 4
    ) {
        fprintf(stderr, "native semantics fixture: valid symbolic request failed after rejection\n");
        return 0;
    }
    return 1;
}


int main(void) {
    static const uint64_t left_shape[] = {2, 4};
    /* Add supports broadcasting; [2, 1] is valid, while [2, 3] is not. */
    static const uint64_t mismatch_shape[] = {2, 3};
    static const uint64_t valid_shape[] = {2, 4};
    VxShapeNamedTensor inputs[] = {
        {.name = "a", .descriptor = {0}},
        {.name = "b", .descriptor = {0}},
    };
    /* The shape contract sizes its output slots from the declaration. */
    VxShapeNamedTensor declared_outputs[] = {
        {.name = "out", .descriptor = {0}},
    };
    VxConcreteShapeRequest request;
    VxConcreteShapeResult result = VX_CONCRETE_SHAPE_RESULT_INITIALIZER;
    VxShapeContractError error;

    inputs[0].descriptor = f32_tensor(left_shape, 2);
    inputs[1].descriptor = f32_tensor(mismatch_shape, 2);
    declared_outputs[0].descriptor = f32_tensor(valid_shape, 2);
    memset(&request, 0, sizeof(request));
    request.inputs = inputs;
    request.input_count = 2;
    request.declared_outputs = declared_outputs;
    request.declared_output_count = 1;
    memset(&error, 0, sizeof(error));

    if (
        vx_shape_contract_infer("Add", &request, &result, &error) != -1
        || error.code != VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH
        || result.shape_function_id != NULL
        || result.outputs != NULL
        || result.output_count != 0
    ) {
        fprintf(
            stderr,
            "native semantics fixture: incompatible shape was not rejected "
            "transactionally (status=%s, path=%s)\n",
            vx_shape_contract_error_code_name(error.code),
            error.path
        );
        vx_shape_contract_result_clear(&result);
        return 1;
    }

    /* Reuse the same request/result/error objects immediately after failure. */
    inputs[1].descriptor = f32_tensor(valid_shape, 2);
    if (
        vx_shape_contract_infer("Add", &request, &result, &error) != 0
        || error.code != VX_SHAPE_CONTRACT_ERROR_NONE
        || !result.shape_function_id
        || strcmp(result.shape_function_id, "volvox.shape.broadcast-arithmetic.v1") != 0
        || result.output_count != 1
        || strcmp(result.outputs[0].name, "out") != 0
        || result.outputs[0].descriptor.dtype != VX_DTYPE_F32
        || result.outputs[0].descriptor.rank != 2
        || result.outputs[0].descriptor.shape[0] != 2
        || result.outputs[0].descriptor.shape[1] != 4
    ) {
        fprintf(
            stderr,
            "native semantics fixture: valid request failed after rejection "
            "(status=%s, path=%s)\n",
            vx_shape_contract_error_code_name(error.code),
            error.path
        );
        vx_shape_contract_result_clear(&result);
        return 2;
    }

    vx_shape_contract_result_clear(&result);
    return test_symbolic_domain_reuse() ? 0 : 3;
}
