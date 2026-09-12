#ifndef VX_SHAPE_DOMAIN_CONTRACT_API
#define VX_SHAPE_DOMAIN_CONTRACT_API
#define VX_SHAPE_DOMAIN_CONTRACT_API_SOURCE_LOCAL_EMPTY 1
#endif

#include "shape_domain_contract.h"

#include "shape_contract.h"

#include <math.h>
#include <stdint.h>

typedef struct VxShapeDomainArena {
    uint8_t* bytes;
    uint32_t capacity;
    uint64_t cursor;
    uint64_t required;
} VxShapeDomainArena;

typedef struct VxShapeDomainBuild {
    const VxShapeDomainProofRequest* request;
    VxShapeDomainProofResult* result;
    VxShapeDomainError* error;
    VxShapeDomainArena* arena;
    uint32_t affine_capacity;
} VxShapeDomainBuild;

static void vx_sd_clear(void* destination, uint32_t bytes) {
    uint8_t* output = (uint8_t*)destination;
    for (uint32_t index = 0u; index < bytes; ++index) output[index] = 0u;
}

static void vx_sd_copy(void* destination, const void* source, uint32_t bytes) {
    uint8_t* output = (uint8_t*)destination;
    const uint8_t* input = (const uint8_t*)source;
    for (uint32_t index = 0u; index < bytes; ++index) output[index] = input[index];
}

/* Keep fixed-record copies scalar so a freestanding wasm32 -Oz build cannot
 * synthesize a libc memcpy import from an otherwise allocator-free core. */
static void vx_sd_descriptor_copy(
        VxShapeDomainTensorDescriptor* destination,
        const VxShapeDomainTensorDescriptor* source) {
    destination->dtype = source->dtype;
    destination->rank = source->rank;
    destination->dimensions = source->dimensions;
    destination->quantization.scheme = source->quantization.scheme;
    destination->quantization.scale_bits = source->quantization.scale_bits;
    destination->quantization.zero_point = source->quantization.zero_point;
    destination->quantization.axis = source->quantization.axis;
    destination->quantization.count = source->quantization.count;
    destination->quantization.scales = source->quantization.scales;
    destination->quantization.zero_points = source->quantization.zero_points;
}

static int vx_sd_align(uint64_t* cursor, uint32_t alignment) {
    uint64_t remainder;
    uint64_t add;
    if (!alignment) return -1;
    remainder = *cursor % alignment;
    add = remainder ? alignment - remainder : 0u;
    if (*cursor > UINT32_MAX - add) return -1;
    *cursor += add;
    return 0;
}

static void* vx_sd_allocate(VxShapeDomainArena* arena,
                            uint32_t count,
                            uint32_t item_bytes,
                            uint32_t alignment) {
    uint64_t cursor = arena->cursor;
    uint64_t bytes = (uint64_t)count * item_bytes;
    uint64_t end;
    if (vx_sd_align(&cursor, alignment) || bytes > UINT32_MAX - cursor) {
        arena->required = UINT32_MAX;
        return NULL;
    }
    end = cursor + bytes;
    if (end > arena->required) arena->required = end;
    arena->cursor = end;
    if (end > arena->capacity || (!arena->bytes && bytes)) return NULL;
    if (bytes) vx_sd_clear(arena->bytes + (uint32_t)cursor, (uint32_t)bytes);
    return bytes ? arena->bytes + (uint32_t)cursor : NULL;
}

static int32_t vx_sd_fail(VxShapeDomainBuild* build,
                          VxShapeDomainErrorCode code,
                          uint32_t index,
                          uint32_t subindex) {
    if (build && build->error) {
        build->error->code = code;
        build->error->index = index;
        build->error->subindex = subindex;
    }
    return VX_SHAPE_DOMAIN_STATUS_INVALID;
}

static int32_t vx_sd_unsupported(VxShapeDomainBuild* build) {
    if (build && build->error) {
        build->error->code = VX_SHAPE_DOMAIN_ERROR_UNSUPPORTED_OPERATOR;
        build->error->index = (uint32_t)build->request->operator_kind;
        build->error->subindex = VX_SHAPE_DOMAIN_INDEX_NONE;
    }
    return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
}

static int vx_sd_string_equal(const VxShapeDomainString* value,
                              const char* expected) {
    uint32_t length = 0u;
    if (!value || !expected) return 0;
    while (expected[length]) {
        if (length == UINT32_MAX) return 0;
        length++;
    }
    if (value->length != length || (length && !value->bytes)) return 0;
    for (uint32_t index = 0u; index < length; ++index)
        if (value->bytes[index] != (uint8_t)expected[index]) return 0;
    return 1;
}

static int vx_sd_dimension_valid(
        const VxShapeDomainDimension* dimension,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count) {
    if (!dimension) return 0;
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED)
        return dimension->symbol_index == VX_SHAPE_DOMAIN_INDEX_NONE &&
            dimension->value > 0 &&
            (uint64_t)dimension->value <= VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER;
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL)
        return dimension->symbol_index < environment_count &&
            dimension->value == 0 && environment;
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE)
        return dimension->symbol_index < environment_count && environment;
    return 0;
}

VX_SHAPE_DOMAIN_CONTRACT_API int vx_shape_domain_legal_progression(
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        uint32_t symbol_index,
        VxShapeDomainProgression* progression) {
    const VxShapeDomainConstraint* constraint;
    uint64_t remainder;
    uint64_t adjustment;
    uint64_t span;
    if (!environment || !progression || symbol_index >= environment_count)
        return -1;
    constraint = &environment[symbol_index];
    if (!constraint->minimum ||
        constraint->minimum > constraint->maximum ||
        !constraint->multiple_of ||
        constraint->maximum > VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER)
        return -1;
    remainder = constraint->minimum % constraint->multiple_of;
    adjustment = remainder ? constraint->multiple_of - remainder : 0u;
    if (adjustment > constraint->maximum - constraint->minimum) return -1;
    progression->first = constraint->minimum + adjustment;
    progression->last = constraint->maximum -
        constraint->maximum % constraint->multiple_of;
    progression->step = constraint->multiple_of;
    if (progression->first > progression->last) return -1;
    span = progression->last - progression->first;
    progression->count = span / progression->step + 1u;
    return 0;
}

static int vx_sd_add_signed(uint64_t source,
                            int64_t offset,
                            uint64_t* output) {
    if (offset >= 0) {
        uint64_t add = (uint64_t)offset;
        if (source > VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER - add) return -1;
        *output = source + add;
    } else {
        uint64_t subtract = (uint64_t)(-(offset + 1)) + 1u;
        if (source <= subtract) return -1;
        *output = source - subtract;
    }
    return *output && *output <= VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER ? 0 : -1;
}

VX_SHAPE_DOMAIN_CONTRACT_API int vx_shape_domain_fixed_value(
        const VxShapeDomainDimension* dimension,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        uint64_t* value) {
    const VxShapeDomainConstraint* constraint;
    if (!value || !vx_sd_dimension_valid(
            dimension, environment, environment_count)) return 0;
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
        *value = (uint64_t)dimension->value;
        return 1;
    }
    if (!environment || dimension->symbol_index >= environment_count) return 0;
    constraint = &environment[dimension->symbol_index];
    /* This intentionally mirrors the existing TS fixedDimensionValue: a
     * singleton legal progression whose declared min/max differ remains a
     * symbol in the observable logical proof. */
    if (constraint->minimum != constraint->maximum) return 0;
    if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL) {
        *value = constraint->minimum;
        return 1;
    }
    return vx_sd_add_signed(constraint->minimum, dimension->value, value) == 0;
}

VX_SHAPE_DOMAIN_CONTRACT_API int vx_shape_domain_dimensions_equal(
        const VxShapeDomainDimension* left,
        const VxShapeDomainDimension* right,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count) {
    uint64_t left_fixed;
    uint64_t right_fixed;
    if (!vx_sd_dimension_valid(left, environment, environment_count) ||
        !vx_sd_dimension_valid(right, environment, environment_count))
        return 0;
    if (left->kind == right->kind &&
        left->symbol_index == right->symbol_index &&
        left->value == right->value) return 1;
    if (left->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
        right->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE &&
        left->symbol_index == right->symbol_index && !right->value) return 1;
    if (right->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
        left->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE &&
        right->symbol_index == left->symbol_index && !left->value) return 1;
    return vx_shape_domain_fixed_value(
               left, environment, environment_count, &left_fixed) &&
        vx_shape_domain_fixed_value(
               right, environment, environment_count, &right_fixed) &&
        left_fixed == right_fixed;
}

static VxShapeDomainDimension vx_sd_fixed(uint64_t value) {
    VxShapeDomainDimension result;
    result.kind = VX_SHAPE_DOMAIN_DIMENSION_FIXED;
    result.symbol_index = VX_SHAPE_DOMAIN_INDEX_NONE;
    result.value = (int64_t)value;
    return result;
}

static VxShapeDomainDimension vx_sd_symbol(uint32_t symbol) {
    VxShapeDomainDimension result;
    result.kind = VX_SHAPE_DOMAIN_DIMENSION_SYMBOL;
    result.symbol_index = symbol;
    result.value = 0;
    return result;
}

static int vx_sd_shapes_equal(
        const VxShapeDomainTensorDescriptor* left,
        const VxShapeDomainTensorDescriptor* right,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count) {
    if (!left || !right || left->rank != right->rank) return 0;
    for (uint32_t axis = 0u; axis < left->rank; ++axis)
        if (!vx_shape_domain_dimensions_equal(&left->dimensions[axis],
                &right->dimensions[axis], environment, environment_count))
            return 0;
    return 1;
}

static int vx_sd_quantization_equal(
        const VxShapeDomainQuantization* left,
        const VxShapeDomainQuantization* right) {
    if (left->scheme != right->scheme || left->scale_bits != right->scale_bits ||
        left->zero_point != right->zero_point || left->axis != right->axis ||
        left->count != right->count) return 0;
    if (left->scheme != 2) return 1;
    if ((left->count && (!left->scales || !left->zero_points ||
                        !right->scales || !right->zero_points))) return 0;
    for (uint32_t index = 0u; index < left->count; ++index)
        if (left->scales[index] != right->scales[index] ||
            left->zero_points[index] != right->zero_points[index]) return 0;
    return 1;
}

static int vx_sd_runtime_dtype_valid(int32_t dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ||
        dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

static int vx_sd_constraint_valid(
        const VxShapeDomainConstraint* constraint) {
    VxShapeDomainProgression progression;
    if (!constraint) return 0;
    return vx_shape_domain_legal_progression(
        constraint, 1u, 0u, &progression) == 0;
}

static int vx_sd_quantization_valid(
        const VxShapeDomainTensorDescriptor* descriptor) {
    const VxShapeDomainQuantization* quantization;
    int32_t minimum;
    int32_t maximum;
    float scale;
    uint64_t extent;
    if (!descriptor) return 0;
    quantization = &descriptor->quantization;
    if (quantization->scheme == 0) return 1;
    if ((descriptor->dtype != VX_DTYPE_I8 &&
         descriptor->dtype != VX_DTYPE_U8) ||
        (quantization->scheme != 1 && quantization->scheme != 2)) return 0;
    minimum = descriptor->dtype == VX_DTYPE_I8 ? -128 : 0;
    maximum = descriptor->dtype == VX_DTYPE_I8 ? 127 : 255;
    if (quantization->scheme == 1) {
        vx_sd_copy(&scale, &quantization->scale_bits,
                   (uint32_t)sizeof(scale));
        return isfinite(scale) && scale > 0.0f &&
            quantization->zero_point >= minimum &&
            quantization->zero_point <= maximum;
    }
    if (quantization->axis >= descriptor->rank || !quantization->count ||
        !quantization->scales || !quantization->zero_points) return 0;
    if (!vx_shape_domain_fixed_value(
            &descriptor->dimensions[quantization->axis], NULL, 0u, &extent)) {
        /* Symbolic singleton axes need the environment and are checked by the
         * descriptor validator below. */
        extent = 0u;
    }
    for (uint32_t index = 0u; index < quantization->count; ++index) {
        vx_sd_copy(&scale, &quantization->scales[index],
                   (uint32_t)sizeof(scale));
        if (!isfinite(scale) || scale <= 0.0f ||
            quantization->zero_points[index] < minimum ||
            quantization->zero_points[index] > maximum) return 0;
    }
    return extent == 0u || extent == quantization->count;
}

static int vx_sd_descriptor_valid(
        const VxShapeDomainTensorDescriptor* descriptor,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count) {
    uint64_t maximum_product = 1u;
    uint64_t quantized_extent = 0u;
    if (!descriptor || !vx_sd_runtime_dtype_valid(descriptor->dtype) ||
        (descriptor->rank && !descriptor->dimensions) ||
        !vx_sd_quantization_valid(descriptor)) return 0;
    for (uint32_t axis = 0u; axis < descriptor->rank; ++axis) {
        const VxShapeDomainDimension* dimension = &descriptor->dimensions[axis];
        uint64_t minimum;
        uint64_t maximum;
        if (!vx_sd_dimension_valid(dimension, environment, environment_count))
            return 0;
        if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
            minimum = (uint64_t)dimension->value;
            maximum = (uint64_t)dimension->value;
        } else {
            minimum = environment[dimension->symbol_index].minimum;
            maximum = environment[dimension->symbol_index].maximum;
            if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE &&
                (vx_sd_add_signed(minimum, dimension->value, &minimum) ||
                 vx_sd_add_signed(maximum, dimension->value, &maximum)))
                return 0;
        }
        if (!minimum || maximum_product >
                VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER / maximum) return 0;
        maximum_product *= maximum;
    }
    if (descriptor->quantization.scheme == 2) {
        if (!vx_shape_domain_fixed_value(
                &descriptor->dimensions[descriptor->quantization.axis],
                environment, environment_count, &quantized_extent) ||
            quantized_extent != descriptor->quantization.count) return 0;
    }
    return 1;
}

static int vx_sd_string_valid(const VxShapeDomainString* value,
                              int allow_empty) {
    return value && (allow_empty || value->length) &&
        (!value->length || value->bytes);
}

static int vx_sd_request_metadata_valid(
        const VxShapeDomainProofRequest* request) {
    for (uint32_t index = 0u; index < request->environment_count; ++index)
        if (!vx_sd_constraint_valid(&request->environment[index])) return 0;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const VxShapeDomainNamedTensor* tensors = side
            ? request->declared_outputs : request->inputs;
        uint32_t count = side
            ? request->declared_output_count : request->input_count;
        for (uint32_t index = 0u; index < count; ++index) {
            if (!vx_sd_string_valid(&tensors[index].name, 0) ||
                !vx_sd_descriptor_valid(&tensors[index].descriptor,
                    request->environment, request->environment_count))
                return 0;
            for (uint32_t previous = 0u; previous < index; ++previous) {
                int equal = tensors[index].name.length ==
                    tensors[previous].name.length;
                for (uint32_t byte = 0u;
                     equal && byte < tensors[index].name.length; ++byte)
                    if (tensors[index].name.bytes[byte] !=
                        tensors[previous].name.bytes[byte]) equal = 0;
                if (equal) return 0;
            }
        }
    }
    for (uint32_t index = 0u; index < request->param_count; ++index) {
        const VxShapeDomainParam* param = &request->params[index];
        if (!vx_sd_string_valid(&param->name, 0) ||
            param->kind < VX_SHAPE_DOMAIN_PARAM_NUMBER ||
            param->kind > VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
            (param->kind == VX_SHAPE_DOMAIN_PARAM_STRING &&
             !vx_sd_string_valid(&param->string, 1)) ||
            (param->kind == VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY &&
             param->value_count && !param->values)) return 0;
        for (uint32_t previous = 0u; previous < index; ++previous) {
            int equal = param->name.length ==
                request->params[previous].name.length;
            for (uint32_t byte = 0u; equal && byte < param->name.length; ++byte)
                if (param->name.bytes[byte] !=
                    request->params[previous].name.bytes[byte]) equal = 0;
            if (equal) return 0;
        }
    }
    return 1;
}

static int vx_sd_unquantized(const VxShapeDomainTensorDescriptor* descriptor) {
    return descriptor && descriptor->quantization.scheme == 0;
}

static int32_t vx_sd_assert_float(VxShapeDomainBuild* build,
                                  const VxShapeDomainTensorDescriptor* value,
                                  uint32_t input_index) {
    if (!value || value->dtype != VX_DTYPE_F32)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!vx_sd_unquantized(value))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION,
                          input_index, VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static const VxShapeDomainNamedTensor* vx_sd_named(
        const VxShapeDomainNamedTensor* values,
        uint32_t count,
        const char* name) {
    for (uint32_t index = 0u; index < count; ++index)
        if (vx_sd_string_equal(&values[index].name, name)) return &values[index];
    return NULL;
}

static const VxShapeDomainNamedTensor* vx_sd_input(
        const VxShapeDomainProofRequest* request,
        const char* name) {
    return vx_sd_named(request->inputs, request->input_count, name);
}

static const VxShapeDomainParam* vx_sd_param(
        const VxShapeDomainProofRequest* request,
        const char* name) {
    for (uint32_t index = 0u; index < request->param_count; ++index)
        if (vx_sd_string_equal(&request->params[index].name, name))
            return &request->params[index];
    return NULL;
}

static int vx_sd_c_name_in(const VxShapeDomainString* name,
                           const char* const* allowed,
                           uint32_t allowed_count) {
    for (uint32_t index = 0u; index < allowed_count; ++index)
        if (vx_sd_string_equal(name, allowed[index])) return 1;
    return 0;
}

static int32_t vx_sd_allowed_params(VxShapeDomainBuild* build,
                                    const char* const* allowed,
                                    uint32_t allowed_count) {
    for (uint32_t index = 0u; index < build->request->param_count; ++index)
        if (!vx_sd_c_name_in(&build->request->params[index].name,
                             allowed, allowed_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                              index, VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_exact_ports(VxShapeDomainBuild* build,
                                 const char* const* inputs,
                                 uint32_t input_count,
                                 const char* const* outputs,
                                 uint32_t output_count) {
    if (build->request->input_count != input_count ||
        build->request->declared_output_count != output_count)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          build->request->input_count,
                          build->request->declared_output_count);
    for (uint32_t index = 0u; index < input_count; ++index)
        if (!vx_sd_input(build->request, inputs[index]))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                              index, 0u);
    for (uint32_t index = 0u; index < output_count; ++index)
        if (!vx_sd_named(build->request->declared_outputs,
                         output_count, outputs[index]))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                              index, 1u);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_optional_ports(VxShapeDomainBuild* build,
                                    const char* const* required_inputs,
                                    uint32_t required_input_count,
                                    const char* const* optional_inputs,
                                    uint32_t optional_input_count,
                                    const char* const* outputs,
                                    uint32_t output_count) {
    if (optional_input_count > UINT32_MAX - required_input_count ||
        build->request->input_count < required_input_count ||
        build->request->input_count >
            required_input_count + optional_input_count ||
        build->request->declared_output_count != output_count)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          build->request->input_count,
                          build->request->declared_output_count);
    for (uint32_t index = 0u; index < required_input_count; ++index)
        if (!vx_sd_input(build->request, required_inputs[index]))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                              index, 0u);
    for (uint32_t index = 0u; index < build->request->input_count; ++index) {
        const VxShapeDomainString* name = &build->request->inputs[index].name;
        if (!vx_sd_c_name_in(name, required_inputs, required_input_count) &&
            !vx_sd_c_name_in(name, optional_inputs, optional_input_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                              index, 0u);
    }
    for (uint32_t index = 0u; index < output_count; ++index)
        if (!vx_sd_named(build->request->declared_outputs,
                         output_count, outputs[index]))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                              index, 1u);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_number_integer(VxShapeDomainBuild* build,
                                    const VxShapeDomainParam* param,
                                    int64_t default_value,
                                    int64_t* output) {
    double number;
    int64_t integer;
    if (!output) return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
                                  VX_SHAPE_DOMAIN_INDEX_NONE,
                                  VX_SHAPE_DOMAIN_INDEX_NONE);
    if (!param) {
        *output = default_value;
        return VX_SHAPE_DOMAIN_STATUS_OK;
    }
    if (param->kind != VX_SHAPE_DOMAIN_PARAM_NUMBER ||
        !isfinite(param->number) ||
        param->number < -(double)VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER ||
        param->number > (double)VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    number = param->number;
    integer = (int64_t)number;
    if ((double)integer != number)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    *output = integer;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_axis(VxShapeDomainBuild* build,
                          const char* name,
                          uint32_t rank,
                          int64_t default_axis,
                          uint32_t* output) {
    int64_t raw = default_axis;
    if (!rank || vx_sd_number_integer(
            build, vx_sd_param(build->request, name), default_axis, &raw))
        return vx_sd_fail(build, rank ? VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS
                                     : VX_SHAPE_DOMAIN_ERROR_INVALID_RANK,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    if (raw < 0) raw += rank;
    if (raw < 0 || (uint64_t)raw >= rank)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    *output = (uint32_t)raw;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int vx_sd_param_array_integer(const VxShapeDomainParam* param,
                                     uint32_t index,
                                     int64_t* output) {
    double number;
    int64_t integer;
    if (!param || param->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY ||
        index >= param->value_count || !param->values || !output ||
        param->values[index].kind != VX_SHAPE_DOMAIN_PARAM_VALUE_NUMBER)
        return -1;
    number = param->values[index].number;
    if (!isfinite(number) ||
        number < -(double)VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER ||
        number > (double)VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER) return -1;
    integer = (int64_t)number;
    if ((double)integer != number) return -1;
    *output = integer;
    return 0;
}

static int vx_sd_target_dimension(const VxShapeDomainParamValue* value,
                                  VxShapeDomainDimension* output) {
    int64_t integer;
    if (!value || !output) return -1;
    if (value->kind == VX_SHAPE_DOMAIN_PARAM_VALUE_DIMENSION) {
        *output = vx_sd_symbol(value->dimension_index);
        return 0;
    }
    if (value->kind != VX_SHAPE_DOMAIN_PARAM_VALUE_NUMBER ||
        !isfinite(value->number) || value->number <= 0.0 ||
        value->number > (double)VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER)
        return -1;
    integer = (int64_t)value->number;
    if ((double)integer != value->number) return -1;
    *output = vx_sd_fixed((uint64_t)integer);
    return 0;
}

static int vx_sd_result_output_index(const VxShapeDomainBuild* build,
                                     const char* name,
                                     uint32_t* output_index) {
    for (uint32_t index = 0u; index < build->result->output_count; ++index) {
        if (vx_sd_string_equal(&build->result->outputs[index].name, name)) {
            *output_index = index;
            return 0;
        }
    }
    return -1;
}

static int32_t vx_sd_set_output(VxShapeDomainBuild* build,
                                const char* name,
                                int32_t dtype,
                                const VxShapeDomainDimension* dimensions,
                                uint32_t rank,
                                const VxShapeDomainQuantization* quantization) {
    uint32_t index;
    VxShapeDomainNamedTensor* output;
    if (vx_sd_result_output_index(build, name, &index))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS,
                          VX_SHAPE_DOMAIN_INDEX_NONE,
                          VX_SHAPE_DOMAIN_INDEX_NONE);
    output = &build->result->outputs[index];
    if (rank != output->descriptor.rank || (rank && !dimensions))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH,
                          index, rank);
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        if (!vx_sd_dimension_valid(&dimensions[axis],
                build->request->environment,
                build->request->environment_count))
            return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DIMENSION,
                              index, axis);
        ((VxShapeDomainDimension*)output->descriptor.dimensions)[axis] =
            dimensions[axis];
    }
    output->descriptor.dtype = dtype;
    if (quantization) output->descriptor.quantization = *quantization;
    else vx_sd_clear(&output->descriptor.quantization,
                     (uint32_t)sizeof(output->descriptor.quantization));
    /* A rule composes an output from validated inputs, but the composition
     * itself can leave the representable domain: a dtype the runtime does not
     * carry, an element product past the safe-integer bound, or a per-axis
     * quantization whose extent no longer matches. Prove the produced
     * descriptor here so every consumer of a proof gets that guarantee once,
     * instead of each one re-deriving it. */
    if (!vx_sd_descriptor_valid(&output->descriptor,
                                build->request->environment,
                                build->request->environment_count))
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INVALID_DIMENSION,
                          index, VX_SHAPE_DOMAIN_INDEX_NONE);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int32_t vx_sd_add_affine(VxShapeDomainBuild* build,
                                uint32_t target,
                                uint32_t source,
                                int64_t offset) {
    VxShapeDomainAffineWitness* witness;
    if (target >= build->request->environment_count ||
        source >= build->request->environment_count ||
        build->result->affine_witness_count >= build->affine_capacity)
        return vx_sd_fail(build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
                          target, source);
    witness = &build->result->affine_witnesses[
        build->result->affine_witness_count++];
    witness->target_dimension_index = target;
    witness->source_dimension_index = source;
    witness->offset = offset;
    return VX_SHAPE_DOMAIN_STATUS_OK;
}

static int vx_sd_port_suffix(const VxShapeDomainString* name,
                             const char* prefix,
                             uint32_t* suffix) {
    uint32_t prefix_length = 0u;
    uint64_t value = 0u;
    uint32_t cursor;
    while (prefix[prefix_length]) prefix_length++;
    if (!name || name->length <= prefix_length || !name->bytes) return -1;
    for (uint32_t index = 0u; index < prefix_length; ++index)
        if (name->bytes[index] != (uint8_t)prefix[index]) return -1;
    cursor = prefix_length;
    if (name->bytes[cursor] == (uint8_t)'0' &&
        cursor + 1u != name->length) return -1;
    for (; cursor < name->length; ++cursor) {
        uint8_t byte = name->bytes[cursor];
        if (byte < (uint8_t)'0' || byte > (uint8_t)'9') return -1;
        value = value * 10u + (byte - (uint8_t)'0');
        if (value > UINT32_MAX) return -1;
    }
    *suffix = (uint32_t)value;
    return 0;
}

static const VxShapeDomainNamedTensor* vx_sd_variadic_input(
        const VxShapeDomainProofRequest* request,
        const char* prefix,
        uint32_t suffix) {
    for (uint32_t index = 0u; index < request->input_count; ++index) {
        uint32_t parsed;
        if (!vx_sd_port_suffix(&request->inputs[index].name, prefix, &parsed) &&
            parsed == suffix) return &request->inputs[index];
    }
    return NULL;
}

static int vx_sd_variadic_inputs_valid(
        const VxShapeDomainProofRequest* request,
        const char* prefix,
        uint32_t minimum) {
    if (request->input_count < minimum) return 0;
    for (uint32_t index = 0u; index < request->input_count; ++index)
        if (!vx_sd_variadic_input(request, prefix, index)) return 0;
    return 1;
}

static int vx_sd_variadic_outputs_valid(
        const VxShapeDomainProofRequest* request,
        const char* prefix,
        uint32_t minimum) {
    if (request->declared_output_count < minimum) return 0;
    for (uint32_t expected = 0u;
         expected < request->declared_output_count; ++expected) {
        int found = 0;
        for (uint32_t index = 0u;
             index < request->declared_output_count; ++index) {
            uint32_t parsed;
            if (!vx_sd_port_suffix(
                    &request->declared_outputs[index].name,
                    prefix, &parsed) && parsed == expected) {
                found = 1;
                break;
            }
        }
        if (!found) return 0;
    }
    return 1;
}

static int vx_sd_checked_add(uint64_t left,
                             uint64_t right,
                             uint64_t* output) {
    if (!output || left > VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER - right) return -1;
    *output = left + right;
    return *output <= VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER ? 0 : -1;
}

static int vx_sd_checked_multiply(uint64_t left,
                                  uint64_t right,
                                  uint64_t* output) {
    if (!output || (right && left > VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER / right))
        return -1;
    *output = left * right;
    return *output <= VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER ? 0 : -1;
}

static int vx_sd_product_signature(
        const VxShapeDomainDimension* dimensions,
        uint32_t count,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        uint64_t* constant,
        uint32_t* dynamic_count,
        uint32_t* single_symbol,
        uint32_t* single_exponent) {
    uint64_t product = 1u;
    uint32_t symbol = VX_SHAPE_DOMAIN_INDEX_NONE;
    uint32_t exponent = 0u;
    uint32_t dynamics = 0u;
    for (uint32_t index = 0u; index < count; ++index) {
        uint64_t fixed;
        if (vx_shape_domain_fixed_value(
                &dimensions[index], environment, environment_count, &fixed)) {
            if (vx_sd_checked_multiply(product, fixed, &product)) return -1;
            continue;
        }
        if (dimensions[index].kind != VX_SHAPE_DOMAIN_DIMENSION_SYMBOL)
            return -1;
        dynamics++;
        if (symbol == VX_SHAPE_DOMAIN_INDEX_NONE ||
            symbol == dimensions[index].symbol_index) {
            symbol = dimensions[index].symbol_index;
            exponent++;
        } else {
            symbol = VX_SHAPE_DOMAIN_INDEX_NONE - 1u;
        }
    }
    *constant = product;
    *dynamic_count = dynamics;
    *single_symbol = symbol;
    *single_exponent = exponent;
    return 0;
}

static int vx_sd_products_equal(
        const VxShapeDomainDimension* left,
        uint32_t left_count,
        const VxShapeDomainDimension* right,
        uint32_t right_count,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count) {
    uint64_t left_constant = 1u;
    uint64_t right_constant = 1u;
    uint32_t counts[2];
    for (uint32_t side = 0u; side < 2u; ++side) {
        const VxShapeDomainDimension* values = side ? right : left;
        uint32_t value_count = side ? right_count : left_count;
        uint64_t* constant = side ? &right_constant : &left_constant;
        counts[side] = 0u;
        for (uint32_t index = 0u; index < value_count; ++index) {
            uint64_t fixed;
            if (vx_shape_domain_fixed_value(
                    &values[index], environment, environment_count, &fixed)) {
                if (vx_sd_checked_multiply(*constant, fixed, constant)) return 0;
            } else {
                if (values[index].kind != VX_SHAPE_DOMAIN_DIMENSION_SYMBOL)
                    return 0;
                counts[side]++;
            }
        }
    }
    if (left_constant != right_constant || counts[0] != counts[1]) return 0;
    for (uint32_t index = 0u; index < left_count; ++index) {
        uint32_t left_occurrences = 0u;
        uint32_t right_occurrences = 0u;
        if (vx_shape_domain_fixed_value(&left[index], environment,
                environment_count, &left_constant)) continue;
        for (uint32_t scan = 0u; scan < left_count; ++scan)
            if (left[scan].kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
                left[scan].symbol_index == left[index].symbol_index)
                left_occurrences++;
        for (uint32_t scan = 0u; scan < right_count; ++scan)
            if (right[scan].kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
                right[scan].symbol_index == left[index].symbol_index)
                right_occurrences++;
        if (left_occurrences != right_occurrences) return 0;
    }
    for (uint32_t index = 0u; index < right_count; ++index) {
        uint32_t left_occurrences = 0u;
        uint32_t right_occurrences = 0u;
        if (vx_shape_domain_fixed_value(&right[index], environment,
                environment_count, &right_constant)) continue;
        for (uint32_t scan = 0u; scan < left_count; ++scan)
            if (left[scan].kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
                left[scan].symbol_index == right[index].symbol_index)
                left_occurrences++;
        for (uint32_t scan = 0u; scan < right_count; ++scan)
            if (right[scan].kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
                right[scan].symbol_index == right[index].symbol_index)
                right_occurrences++;
        if (left_occurrences != right_occurrences) return 0;
    }
    return 1;
}

static int vx_sd_collapse_dimensions(
        const VxShapeDomainDimension* dimensions,
        uint32_t count,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        VxShapeDomainDimension* output) {
    uint64_t constant;
    uint32_t dynamic_count;
    uint32_t symbol;
    uint32_t exponent;
    if (vx_sd_product_signature(dimensions, count, environment,
            environment_count, &constant, &dynamic_count, &symbol, &exponent))
        return -1;
    if (!dynamic_count) {
        *output = vx_sd_fixed(constant);
        return 0;
    }
    if (dynamic_count == 1u && exponent == 1u && constant == 1u &&
        symbol < environment_count) {
        *output = vx_sd_symbol(symbol);
        return 0;
    }
    return -1;
}

static int vx_sd_broadcast_dimension(
        const VxShapeDomainDimension* left,
        const VxShapeDomainDimension* right,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        VxShapeDomainDimension* output) {
    uint64_t left_fixed;
    uint64_t right_fixed;
    if (vx_shape_domain_dimensions_equal(
            left, right, environment, environment_count)) {
        *output = *left;
        return 0;
    }
    if (vx_shape_domain_fixed_value(
            left, environment, environment_count, &left_fixed) &&
        left_fixed == 1u) {
        *output = *right;
        return 0;
    }
    if (vx_shape_domain_fixed_value(
            right, environment, environment_count, &right_fixed) &&
        right_fixed == 1u) {
        *output = *left;
        return 0;
    }
    if (vx_shape_domain_fixed_value(
            left, environment, environment_count, &left_fixed) &&
        vx_shape_domain_fixed_value(
            right, environment, environment_count, &right_fixed) &&
        left_fixed == right_fixed) {
        *output = vx_sd_fixed(left_fixed);
        return 0;
    }
    return -1;
}

static int vx_sd_broadcast_shape(
        const VxShapeDomainTensorDescriptor* left,
        const VxShapeDomainTensorDescriptor* right,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count,
        VxShapeDomainDimension* output,
        uint32_t* output_rank) {
    uint32_t rank = left->rank > right->rank ? left->rank : right->rank;
    if (rank > 8u) return -1;
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        int64_t left_axis = (int64_t)axis - (int64_t)(rank - left->rank);
        int64_t right_axis = (int64_t)axis - (int64_t)(rank - right->rank);
        VxShapeDomainDimension left_dimension = left_axis < 0
            ? vx_sd_fixed(1u) : left->dimensions[(uint32_t)left_axis];
        VxShapeDomainDimension right_dimension = right_axis < 0
            ? vx_sd_fixed(1u) : right->dimensions[(uint32_t)right_axis];
        if (vx_sd_broadcast_dimension(&left_dimension, &right_dimension,
                environment, environment_count, &output[axis])) return -1;
    }
    *output_rank = rank;
    return 0;
}

#include "shape_domain_contract_direct.inc"
#include "shape_domain_contract_structural.inc"
#include "shape_domain_contract_spatial.inc"
#include "shape_domain_contract_attention.inc"
#include "shape_domain_contract_quantized.inc"
#include "shape_domain_contract_extended_model.inc"
#include "shape_domain_contract_extended_sequence.inc"

VX_SHAPE_DOMAIN_CONTRACT_API int32_t vx_shape_domain_contract_prove(
        const VxShapeDomainProofRequest* request,
        VxShapeDomainProofResult* result,
        VxShapeDomainError* error,
        void* scratch,
        uint32_t scratch_bytes,
        uint32_t* required_scratch_bytes) {
    VxShapeDomainArena arena;
    VxShapeDomainBuild build;
    VxShapeDomainNamedTensor* outputs;
    VxShapeDomainDimension* dimensions;
    VxShapeDomainAffineWitness* affines;
    char* operator_name;
    uint64_t axis_total = 0u;
    uint32_t axis_position = 0u;
    int32_t status;
    if (result) vx_sd_clear(result, (uint32_t)sizeof(*result));
    if (error) {
        vx_sd_clear(error, (uint32_t)sizeof(*error));
        error->index = VX_SHAPE_DOMAIN_INDEX_NONE;
        error->subindex = VX_SHAPE_DOMAIN_INDEX_NONE;
    }
    if (required_scratch_bytes) *required_scratch_bytes = 0u;
    if (!request || !result || !error || !required_scratch_bytes ||
        (scratch_bytes && (!scratch || (uintptr_t)scratch % 16u)) ||
        (!scratch_bytes && scratch) ||
        (request->environment_count && !request->environment) ||
        (request->input_count && !request->inputs) ||
        (request->declared_output_count && !request->declared_outputs) ||
        (request->param_count && !request->params) ||
        !request->operator_name.length || !request->operator_name.bytes ||
        request->operator_name.length == UINT32_MAX ||
        !vx_sd_request_metadata_valid(request))
        return VX_SHAPE_DOMAIN_STATUS_INVALID;
    for (uint32_t index = 0u; index < request->declared_output_count; ++index) {
        if (axis_total + request->declared_outputs[index].descriptor.rank > UINT32_MAX)
            return VX_SHAPE_DOMAIN_STATUS_INVALID;
        axis_total += request->declared_outputs[index].descriptor.rank;
    }
    vx_sd_clear(&arena, (uint32_t)sizeof(arena));
    arena.bytes = (uint8_t*)scratch;
    arena.capacity = scratch_bytes;
    operator_name = (char*)vx_sd_allocate(
        &arena, request->operator_name.length + 1u, 1u, 1u);
    outputs = (VxShapeDomainNamedTensor*)vx_sd_allocate(
        &arena, request->declared_output_count,
        (uint32_t)sizeof(*outputs), 8u);
    dimensions = (VxShapeDomainDimension*)vx_sd_allocate(
        &arena, (uint32_t)axis_total,
        (uint32_t)sizeof(*dimensions), 8u);
    affines = (VxShapeDomainAffineWitness*)vx_sd_allocate(
        &arena, (uint32_t)axis_total,
        (uint32_t)sizeof(*affines), 8u);
    *required_scratch_bytes = arena.required > UINT32_MAX
        ? UINT32_MAX : (uint32_t)arena.required;
    if (arena.required == UINT32_MAX || !operator_name ||
        (request->declared_output_count && !outputs) ||
        (axis_total && (!dimensions || !affines)) ||
        arena.cursor > arena.capacity)
        return VX_SHAPE_DOMAIN_STATUS_SCRATCH_TOO_SMALL;
    vx_sd_copy(operator_name, request->operator_name.bytes,
               request->operator_name.length);
    operator_name[request->operator_name.length] = '\0';
    result->shape_function_id = vx_shape_contract_function_id(operator_name);
    result->outputs = outputs;
    result->output_count = request->declared_output_count;
    result->affine_witnesses = affines;
    for (uint32_t index = 0u; index < request->declared_output_count; ++index) {
        outputs[index].name = request->declared_outputs[index].name;
        outputs[index].tensor_index = request->declared_outputs[index].tensor_index;
        outputs[index].descriptor.rank =
            request->declared_outputs[index].descriptor.rank;
        outputs[index].descriptor.dimensions = axis_total
            ? dimensions + axis_position : NULL;
        axis_position += outputs[index].descriptor.rank;
    }
    build.request = request;
    build.result = result;
    build.error = error;
    build.arena = &arena;
    build.affine_capacity = (uint32_t)axis_total;
    if (!result->shape_function_id) {
        status = vx_sd_unsupported(&build);
        goto failure;
    }
    status = vx_sd_prove_direct(&build);
    if (status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED)
        status = vx_sd_prove_structural(&build);
    if (status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED)
        status = vx_sd_prove_spatial(&build);
    if (status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED)
        status = vx_sd_prove_attention(&build);
    if (status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED)
        status = vx_sd_prove_quantized(&build);
    if (status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED)
        status = vx_sd_prove_extended_model(&build);
    if (status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED)
        status = vx_sd_prove_extended_sequence(&build);
    *required_scratch_bytes = arena.required > UINT32_MAX
        ? UINT32_MAX : (uint32_t)arena.required;
    if (status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED) {
        status = vx_sd_unsupported(&build);
        goto failure;
    }
    if (status != VX_SHAPE_DOMAIN_STATUS_OK) goto failure;
    if (result->output_count != request->declared_output_count)
    {
        status = vx_sd_fail(&build, VX_SHAPE_DOMAIN_ERROR_INTERNAL,
                            result->output_count,
                            request->declared_output_count);
        goto failure;
    }
    return VX_SHAPE_DOMAIN_STATUS_OK;

failure:
    vx_sd_clear(result, (uint32_t)sizeof(*result));
    return status;
}

#ifdef VX_SHAPE_DOMAIN_CONTRACT_API_SOURCE_LOCAL_EMPTY
#undef VX_SHAPE_DOMAIN_CONTRACT_API_SOURCE_LOCAL_EMPTY
#undef VX_SHAPE_DOMAIN_CONTRACT_API
#endif
