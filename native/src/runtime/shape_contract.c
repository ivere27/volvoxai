#include "shape_contract.h"

#include "generated/kernel_registry.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct VxShapePortSpec {
    const char* const* required;
    size_t required_count;
    const char* const* optional;
    size_t optional_count;
} VxShapePortSpec;

static const char* const VX_PORT_INPUT[] = {"input"};
static const char* const VX_PORT_INPUT_SLOPE[] = {"input", "slope"};
static const char* const VX_PORT_INPUT_SCALE[] = {"input", "scale"};
static const char* const VX_PORT_ZERO_POINT[] = {"zero_point"};
static const char* const VX_PORT_INPUT_WEIGHT[] = {"input", "weight"};
static const char* const VX_PORT_BIAS[] = {"bias"};
static const char* const VX_PORT_INPUT_WEIGHT_BIAS[] = {"input", "weight", "bias"};
static const char* const VX_PORT_BINARY[] = {"a", "b"};
static const char* const VX_PORT_WHERE[] = {"condition", "a", "b"};
static const char* const VX_PORT_INPUT_INDICES[] = {"input", "indices"};
static const char* const VX_PORT_QKV[] = {"qkv"};
static const char* const VX_PORT_Q_K_V[] = {"q", "k", "v"};
static const char* const VX_PORT_MASK[] = {"mask"};
static const char* const VX_PORT_INPUT_MASK[] = {"input", "mask"};
static const char* const VX_PORT_POSITION_IDS[] = {"position_ids"};
static const char* const VX_PORT_MOE_ROUTER[] = {"input", "weight"};
static const char* const VX_PORT_MOE_LINEAR[] = {
    "input", "expert_weight", "route_indices", "route_weights"
};
static const char* const VX_PORT_EXPERT_BIAS[] = {"expert_bias"};
static const char* const VX_PORT_CROSS_ATTENTION[] = {"q", "kv", "weight"};
static const char* const VX_PORT_SCALE_BIAS[] = {"scale", "bias"};
static const char* const VX_PORT_BATCH_NORM_2D[] = {
    "input", "weight", "bias", "running_mean", "running_var"
};
static const char* const VX_PORT_MASK_EXACT[] = {"mask", "a", "b"};
static const char* const VX_PORT_SCAN[] = {"input", "delta", "A", "B", "C"};
static const char* const VX_PORT_SCAN_OPTIONAL[] = {"D", "z", "initial_state"};

static int vx_shape_fail(VxShapeContractError* error,
                         VxShapeContractErrorCode code,
                         const char* path,
                         const char* format,
                         ...) {
    va_list arguments;
    if (error) {
        error->code = code;
        snprintf(error->path, sizeof(error->path), "%s", path ? path : "");
        va_start(arguments, format);
        vsnprintf(error->detail, sizeof(error->detail), format, arguments);
        va_end(arguments);
    }
    return -1;
}

static void vx_shape_error_reset(VxShapeContractError* error) {
    if (!error) return;
    memset(error, 0, sizeof(*error));
}

const char* vx_shape_contract_error_code_name(VxShapeContractErrorCode code) {
    switch (code) {
        case VX_SHAPE_CONTRACT_ERROR_NONE: return "NONE";
        case VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR: return "UNKNOWN_OPERATOR";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST: return "INVALID_REQUEST";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS: return "INVALID_INPUT_PORTS";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS: return "INVALID_OUTPUT_PORTS";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR: return "INVALID_DESCRIPTOR";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE: return "INVALID_DTYPE";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_RANK: return "INVALID_RANK";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS: return "INVALID_PARAMS";
        case VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH: return "SHAPE_MISMATCH";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION: return "INVALID_QUANTIZATION";
        case VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM:
            return "UNSAFE_QUANTIZATION_TRANSFORM";
        case VX_SHAPE_CONTRACT_ERROR_INVALID_DOMAIN: return "INVALID_DOMAIN";
        case VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        default: return "UNKNOWN_ERROR";
    }
}

static int vx_shape_is_implemented_kind(VxOperatorKind kind) {
    switch (kind) {
        case VX_OP_IDENTITY:
        case VX_OP_RELU:
        case VX_OP_LEAKY_RELU:
        case VX_OP_PRELU:
        case VX_OP_GELU:
        case VX_OP_SILU:
        case VX_OP_SIGMOID:
        case VX_OP_HARD_SWISH:
        case VX_OP_HARD_SIGMOID:
        case VX_OP_TANH:
        case VX_OP_SIN:
        case VX_OP_COS:
        case VX_OP_CLIP:
        case VX_OP_SOFTMAX:
        case VX_OP_LOG_SOFTMAX:
        case VX_OP_CAST:
        case VX_OP_QUANTIZE_LINEAR:
        case VX_OP_DEQUANTIZE_LINEAR:
        case VX_OP_LAYER_NORM:
        case VX_OP_RMS_NORM:
        case VX_OP_GROUP_NORM:
        case VX_OP_LINEAR:
        case VX_OP_GEMM:
        case VX_OP_MATMUL:
        case VX_OP_EMBEDDING:
        case VX_OP_ADD:
        case VX_OP_MUL:
        case VX_OP_BATCH_MATMUL:
        case VX_OP_CONV_1D:
        case VX_OP_CONV_2D:
        case VX_OP_CONV_TRANSPOSE_2D:
        case VX_OP_MAX_POOL_2D:
        case VX_OP_AVERAGE_POOL_2D:
        case VX_OP_GLOBAL_AVERAGE_POOL:
        case VX_OP_RESIZE:
        case VX_OP_RESIZE_NEAREST_2D:
        case VX_OP_UPSAMPLE_NEAREST_2D:
        case VX_OP_SUB:
        case VX_OP_DIV:
        case VX_OP_EQUAL:
        case VX_OP_GREATER_OR_EQUAL:
        case VX_OP_WHERE:
        case VX_OP_REDUCE_SUM:
        case VX_OP_REDUCE_MEAN:
        case VX_OP_ARG_MAX:
        case VX_OP_TRANSPOSE:
        case VX_OP_FLATTEN:
        case VX_OP_SQUEEZE:
        case VX_OP_UNSQUEEZE:
        case VX_OP_RESHAPE:
        case VX_OP_EXPAND:
        case VX_OP_CONCAT:
        case VX_OP_SPLIT:
        case VX_OP_SLICE:
        case VX_OP_PAD:
        case VX_OP_GATHER:
        case VX_OP_GATHER_ELEMENTS:
        case VX_OP_SDPA:
        case VX_OP_CROSS_SDPA:
        case VX_OP_ROPE:
        case VX_OP_Q_LINEAR:
        case VX_OP_Q_MATMUL:
        case VX_OP_Q_GEMM:
        case VX_OP_Q_BATCH_MATMUL:
        case VX_OP_Q_CONV_2D:
        case VX_OP_Q_ADD:
        case VX_OP_Q_EMBEDDING:
        case VX_OP_Q_GELU:
        case VX_OP_Q_SILU:
        case VX_OP_Q_LAYER_NORM:
        case VX_OP_Q_GROUP_NORM:
        case VX_OP_Q_MASKED_MEAN:
        case VX_OP_Q_SDPA:
        case VX_OP_Q_ARG_MAX:
        case VX_OP_MOE_ROUTER:
        case VX_OP_MOE_LINEAR:
        case VX_OP_CROSS_ATTENTION:
        case VX_OP_BATCH_NORM_2D:
        case VX_OP_INTERPOLATE_1D:
        case VX_OP_NOT:
        case VX_OP_MASK:
        case VX_OP_BROADCAST:
        case VX_OP_CONCAT2:
        case VX_OP_REQUANTIZE_LINEAR:
        case VX_OP_SSM_SCAN:
        case VX_OP_SELECTIVE_SCAN:
        case VX_OP_SPATIAL_SOFTARGMAX_Y:
        case VX_OP_MEAN_HEIGHT:
        case VX_OP_PROFILE_X:
        case VX_OP_PROFILE_Y:
        case VX_OP_DROPOUT:
            return 1;
        default:
            return 0;
    }
}

static const VxGeneratedShapeContractRoute* vx_shape_implemented_route(
        const char* operator_name) {
    const VxGeneratedShapeContractRoute* route =
        vx_kernel_shape_contract_find(operator_name);
    if (!route ||
        route->classification != VX_SHAPE_CONTRACT_CLASSIFICATION_CANONICAL ||
        !vx_shape_is_implemented_kind(route->operator_kind)) {
        return NULL;
    }
    return route;
}

const char* vx_shape_contract_function_id(const char* operator_name) {
    const VxGeneratedShapeContractRoute* route =
        vx_shape_implemented_route(operator_name);
    return route ? route->shape_function_id : NULL;
}

static int vx_shape_runtime_dtype(VxDataType dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ||
           dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

static const char* vx_shape_dtype_name(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_F32: return "float32";
        case VX_DTYPE_I32: return "int32";
        case VX_DTYPE_I8: return "int8";
        case VX_DTYPE_U8: return "uint8";
        default: return "unsupported";
    }
}

static void vx_shape_field_path(char* destination,
                                size_t capacity,
                                const char* base,
                                const char* field) {
    snprintf(destination, capacity, "%s.%s", base, field);
}

static int vx_shape_checked_elements(const uint64_t* shape,
                                     size_t rank,
                                     const char* path,
                                     VxShapeContractError* error) {
    uint64_t product = 1;
    size_t axis;
    if (rank && !shape) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an array.");
    }
    for (axis = 0; axis < rank; ++axis) {
        uint64_t dimension = shape[axis];
        if (!dimension || dimension > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER ||
            product > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER / dimension) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                                 path,
                                 "must contain positive safe integers with a safe product.");
        }
        product *= dimension;
    }
    return 0;
}

static int vx_shape_validate_quantization(const VxShapeTensorDescriptor* descriptor,
                                          const char* path,
                                          VxShapeContractError* error) {
    const VxShapeQuantization* quantization = &descriptor->quantization;
    char quant_path[224];
    char field_path[256];
    int32_t minimum;
    int32_t maximum;
    size_t index;
    if (quantization->scheme == VX_SHAPE_QUANTIZATION_NONE) return 0;
    vx_shape_field_path(quant_path, sizeof(quant_path), path, "quantization");
    if (descriptor->dtype != VX_DTYPE_I8 && descriptor->dtype != VX_DTYPE_U8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             quant_path, "is not valid for dtype '%s'.",
                             vx_shape_dtype_name(descriptor->dtype));
    }
    minimum = descriptor->dtype == VX_DTYPE_I8 ? -128 : 0;
    maximum = descriptor->dtype == VX_DTYPE_I8 ? 127 : 255;
    if (quantization->scheme == VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        if (!isfinite(quantization->scale) || quantization->scale <= 0.0f) {
            vx_shape_field_path(field_path, sizeof(field_path), quant_path, "scale");
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 field_path,
                                 "must be positive and representable as float32.");
        }
        if (quantization->zero_point < minimum ||
            quantization->zero_point > maximum) {
            vx_shape_field_path(field_path, sizeof(field_path), quant_path,
                                "zero_point");
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 field_path, "is outside the storage dtype range.");
        }
        return 0;
    }
    if (quantization->scheme != VX_SHAPE_QUANTIZATION_PER_AXIS) {
        vx_shape_field_path(field_path, sizeof(field_path), quant_path, "scheme");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path, "must be per_tensor or per_axis.");
    }
    if (quantization->axis >= descriptor->rank) {
        vx_shape_field_path(field_path, sizeof(field_path), quant_path, "axis");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path, "must address an existing axis.");
    }
    if (!quantization->count || !quantization->scales) {
        vx_shape_field_path(field_path, sizeof(field_path), quant_path, "scales");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path, "must be a non-empty array.");
    }
    if (!quantization->zero_points) {
        vx_shape_field_path(field_path, sizeof(field_path), quant_path,
                            "zero_points");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path,
                             "must match scales and fit the storage dtype.");
    }
    if (quantization->count != descriptor->shape[quantization->axis]) {
        vx_shape_field_path(field_path, sizeof(field_path), quant_path, "scales");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path, "must match the quantized axis extent.");
    }
    for (index = 0; index < quantization->count; ++index) {
        if (!isfinite(quantization->scales[index]) ||
            quantization->scales[index] <= 0.0f) {
            snprintf(field_path, sizeof(field_path), "%s.scales[%zu]",
                     quant_path, index);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 field_path,
                                 "must be positive and representable as float32.");
        }
    }
    for (index = 0; index < quantization->count; ++index) {
        if (quantization->zero_points[index] < minimum ||
            quantization->zero_points[index] > maximum) {
            vx_shape_field_path(field_path, sizeof(field_path), quant_path,
                                "zero_points");
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 field_path,
                                 "must match scales and fit the storage dtype.");
        }
    }
    return 0;
}

static int vx_shape_validate_descriptor(const VxShapeTensorDescriptor* descriptor,
                                        const char* path,
                                        VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (!vx_shape_runtime_dtype(descriptor->dtype)) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "dtype");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             field_path, "has an unsupported runtime dtype.");
    }
    vx_shape_field_path(field_path, sizeof(field_path), path, "shape");
    if (vx_shape_checked_elements(descriptor->shape, descriptor->rank,
                                  field_path, error)) {
        return -1;
    }
    return vx_shape_validate_quantization(descriptor, path, error);
}

static int vx_shape_name_in(const char* name,
                            const char* const* names,
                            size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) {
        if (!strcmp(name, names[index])) return 1;
    }
    return 0;
}

static const VxShapeNamedTensor* vx_shape_find_input(
        const VxConcreteShapeRequest* request,
        const char* name) {
    size_t index;
    for (index = 0; index < request->input_count; ++index) {
        if (!strcmp(request->inputs[index].name, name)) return &request->inputs[index];
    }
    return NULL;
}

static VxShapePortSpec vx_shape_ports(VxOperatorKind kind) {
    VxShapePortSpec spec = {0};
    switch (kind) {
        case VX_OP_PRELU:
            spec.required = VX_PORT_INPUT_SLOPE;
            spec.required_count = 2;
            break;
        case VX_OP_QUANTIZE_LINEAR:
        case VX_OP_DEQUANTIZE_LINEAR:
            spec.required = VX_PORT_INPUT_SCALE;
            spec.required_count = 2;
            spec.optional = VX_PORT_ZERO_POINT;
            spec.optional_count = 1;
            break;
        case VX_OP_MOE_ROUTER:
            spec.required = VX_PORT_MOE_ROUTER;
            spec.required_count = 2;
            spec.optional = VX_PORT_BIAS;
            spec.optional_count = 1;
            break;
        case VX_OP_MOE_LINEAR:
            spec.required = VX_PORT_MOE_LINEAR;
            spec.required_count = 4;
            spec.optional = VX_PORT_EXPERT_BIAS;
            spec.optional_count = 1;
            break;
        case VX_OP_CROSS_ATTENTION:
            spec.required = VX_PORT_CROSS_ATTENTION;
            spec.required_count = 3;
            spec.optional = VX_PORT_SCALE_BIAS;
            spec.optional_count = 2;
            break;
        case VX_OP_BATCH_NORM_2D:
            spec.required = VX_PORT_BATCH_NORM_2D;
            spec.required_count = 5;
            break;
        case VX_OP_MASK:
            spec.required = VX_PORT_MASK_EXACT;
            spec.required_count = 3;
            break;
        case VX_OP_CONCAT2:
            spec.required = VX_PORT_BINARY;
            spec.required_count = 2;
            break;
        case VX_OP_SSM_SCAN:
        case VX_OP_SELECTIVE_SCAN:
            spec.required = VX_PORT_SCAN;
            spec.required_count = 5;
            spec.optional = VX_PORT_SCAN_OPTIONAL;
            spec.optional_count = 3;
            break;
        case VX_OP_LAYER_NORM:
        case VX_OP_LINEAR:
        case VX_OP_GEMM:
        case VX_OP_MATMUL:
        case VX_OP_CONV_1D:
        case VX_OP_CONV_2D:
        case VX_OP_CONV_TRANSPOSE_2D:
            spec.required = VX_PORT_INPUT_WEIGHT;
            spec.required_count = 2;
            spec.optional = VX_PORT_BIAS;
            spec.optional_count = 1;
            break;
        case VX_OP_Q_LINEAR:
        case VX_OP_Q_MATMUL:
        case VX_OP_Q_GEMM:
        case VX_OP_Q_LAYER_NORM:
        case VX_OP_Q_GROUP_NORM:
            spec.required = VX_PORT_INPUT_WEIGHT_BIAS;
            spec.required_count = 3;
            break;
        case VX_OP_Q_CONV_2D:
            spec.required = VX_PORT_INPUT_WEIGHT;
            spec.required_count = 2;
            spec.optional = VX_PORT_BIAS;
            spec.optional_count = 1;
            break;
        case VX_OP_RMS_NORM:
        case VX_OP_EMBEDDING:
        case VX_OP_Q_EMBEDDING:
            spec.required = VX_PORT_INPUT_WEIGHT;
            spec.required_count = 2;
            break;
        case VX_OP_GROUP_NORM:
            spec.required = VX_PORT_INPUT_WEIGHT_BIAS;
            spec.required_count = 3;
            break;
        case VX_OP_ADD:
        case VX_OP_MUL:
        case VX_OP_BATCH_MATMUL:
        case VX_OP_SUB:
        case VX_OP_DIV:
        case VX_OP_EQUAL:
        case VX_OP_GREATER_OR_EQUAL:
        case VX_OP_Q_ADD:
        case VX_OP_Q_BATCH_MATMUL:
            spec.required = VX_PORT_BINARY;
            spec.required_count = 2;
            break;
        case VX_OP_WHERE:
            spec.required = VX_PORT_WHERE;
            spec.required_count = 3;
            break;
        case VX_OP_GATHER:
        case VX_OP_GATHER_ELEMENTS:
            spec.required = VX_PORT_INPUT_INDICES;
            spec.required_count = 2;
            break;
        case VX_OP_SDPA:
            spec.required = VX_PORT_QKV;
            spec.required_count = 1;
            spec.optional = VX_PORT_MASK;
            spec.optional_count = 1;
            break;
        case VX_OP_CROSS_SDPA:
        case VX_OP_Q_SDPA:
            spec.required = VX_PORT_Q_K_V;
            spec.required_count = 3;
            spec.optional = VX_PORT_MASK;
            spec.optional_count = 1;
            break;
        case VX_OP_Q_MASKED_MEAN:
            spec.required = VX_PORT_INPUT_MASK;
            spec.required_count = 2;
            break;
        case VX_OP_ROPE:
            spec.required = VX_PORT_INPUT;
            spec.required_count = 1;
            spec.optional = VX_PORT_POSITION_IDS;
            spec.optional_count = 1;
            break;
        case VX_OP_CONCAT:
            /* Variadic input0..inputN ports are validated separately. */
            break;
        default:
            spec.required = VX_PORT_INPUT;
            spec.required_count = 1;
            break;
    }
    return spec;
}

static int vx_shape_validate_inputs(const VxConcreteShapeRequest* request,
                                    VxShapePortSpec spec,
                                    VxShapeContractError* error) {
    size_t index;
    size_t other;
    size_t validated = 0;
    const char* previous = NULL;
    if (request->input_count && !request->inputs) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                             "operator inputs", "must be an object.");
    }
    for (index = 0; index < request->input_count; ++index) {
        const char* name = request->inputs[index].name;
        if (!name || !name[0]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                                 "operator inputs", "contains an empty port name.");
        }
        for (other = index + 1; other < request->input_count; ++other) {
            if (request->inputs[other].name &&
                !strcmp(name, request->inputs[other].name)) {
                return vx_shape_fail(error,
                                     VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                                     "operator inputs", "contains a duplicate port.");
            }
        }
        if (!vx_shape_name_in(name, spec.required, spec.required_count) &&
            !vx_shape_name_in(name, spec.optional, spec.optional_count)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                                 "operator inputs", "contains an unexpected port.");
        }
    }
    for (index = 0; index < spec.required_count; ++index) {
        if (!vx_shape_find_input(request, spec.required[index])) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                                 "operator inputs", "is missing a required port.");
        }
    }

    /* TypeScript freezes descriptors in unsigned-name order. Match that error order. */
    while (validated < request->input_count) {
        const VxShapeNamedTensor* next = NULL;
        char path[224];
        for (index = 0; index < request->input_count; ++index) {
            const char* name = request->inputs[index].name;
            if (previous && strcmp(name, previous) <= 0) continue;
            if (!next || strcmp(name, next->name) < 0) next = &request->inputs[index];
        }
        if (!next) break;
        snprintf(path, sizeof(path), "operator input '%s'", next->name);
        if (vx_shape_validate_descriptor(&next->descriptor, path, error)) return -1;
        previous = next->name;
        validated++;
    }
    return 0;
}

static int vx_shape_concat_port_index(const char* name, size_t* output) {
    const char* cursor;
    size_t value = 0;
    if (!name || strncmp(name, "input", 5)) return -1;
    cursor = name + 5;
    if (!cursor[0] || (cursor[0] == '0' && cursor[1])) return -1;
    while (*cursor) {
        unsigned digit;
        if (*cursor < '0' || *cursor > '9') return -1;
        digit = (unsigned)(*cursor - '0');
        if (value > (SIZE_MAX - digit) / 10) return -1;
        value = value * 10 + digit;
        cursor++;
    }
    *output = value;
    return 0;
}

static int vx_shape_validate_concat_inputs(
        const VxConcreteShapeRequest* request,
        VxShapeContractError* error) {
    size_t index;
    size_t other;
    if (!request->inputs || request->input_count < 2) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
            "operator inputs",
            "must contain consecutive 'input0'..'inputN' ports with at least 2 inputs.");
    }
    for (index = 0; index < request->input_count; ++index) {
        size_t port_index;
        char expected[64];
        char path[224];
        const char* name = request->inputs[index].name;
        if (vx_shape_concat_port_index(name, &port_index)) {
            return vx_shape_fail(
                error, VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                "operator inputs",
                "must contain consecutive 'input0'..'inputN' ports with at least 2 inputs.");
        }
        for (other = index + 1; other < request->input_count; ++other) {
            if (request->inputs[other].name &&
                !strcmp(name, request->inputs[other].name)) {
                return vx_shape_fail(error,
                                     VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                                     "operator inputs", "contains a duplicate port.");
            }
        }
        if (port_index >= request->input_count) {
            return vx_shape_fail(error,
                                 VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                                 "operator inputs",
                                 "must not contain gaps in 'inputN' ports.");
        }
        snprintf(expected, sizeof(expected), "input%zu", port_index);
        if (strcmp(name, expected) ||
            !vx_shape_find_input(request, expected)) {
            return vx_shape_fail(error,
                                 VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                                 "operator inputs",
                                 "must not contain gaps in 'inputN' ports.");
        }
        snprintf(path, sizeof(path), "operator input '%s'", name);
        if (vx_shape_validate_descriptor(&request->inputs[index].descriptor,
                                         path, error)) return -1;
    }
    for (index = 0; index < request->input_count; ++index) {
        char expected[64];
        snprintf(expected, sizeof(expected), "input%zu", index);
        if (!vx_shape_find_input(request, expected)) {
            return vx_shape_fail(error,
                                 VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                                 "operator inputs",
                                 "must not contain gaps in 'inputN' ports.");
        }
    }
    return 0;
}

static const VxShapeParam* vx_shape_find_param(const VxConcreteShapeRequest* request,
                                               const char* name) {
    size_t index;
    for (index = 0; index < request->param_count; ++index) {
        if (!strcmp(request->params[index].name, name)) return &request->params[index];
    }
    return NULL;
}

static int vx_shape_validate_params(const VxConcreteShapeRequest* request,
                                    const char* const* allowed,
                                    size_t allowed_count,
                                    VxShapeContractError* error) {
    size_t index;
    size_t other;
    const char* unexpected = NULL;
    if (request->param_count && !request->params) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params", "must be an object.");
    }
    for (index = 0; index < request->param_count; ++index) {
        const char* name = request->params[index].name;
        if (!name || !name[0]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params", "contains an empty field name.");
        }
        for (other = index + 1; other < request->param_count; ++other) {
            if (request->params[other].name &&
                !strcmp(name, request->params[other].name)) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     "operator params", "contains a duplicate field.");
            }
        }
        if (!vx_shape_name_in(name, allowed, allowed_count) &&
            (!unexpected || strcmp(name, unexpected) < 0)) {
            unexpected = name;
        }
    }
    if (unexpected) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params", "has unsupported field '%s'.",
                             unexpected);
    }
    return 0;
}

static int vx_shape_number(const VxShapeParam* param,
                           const char* path,
                           double* value,
                           VxShapeContractError* error) {
    if (!param || param->kind != VX_SHAPE_PARAM_NUMBER) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be a number.");
    }
    *value = param->value.number;
    return 0;
}

static int vx_shape_is_safe_integer(double value) {
    return isfinite(value) && trunc(value) == value &&
           fabs(value) <= (double)VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER;
}

static int vx_shape_positive_integer_param(
        const VxConcreteShapeRequest* request,
        const char* name,
        uint64_t default_value,
        uint64_t* output,
        VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    double value = (double)default_value;
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (param && vx_shape_number(param, path, &value, error)) return -1;
    if (!vx_shape_is_safe_integer(value) || value <= 0.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be a positive safe integer.");
    }
    *output = (uint64_t)value;
    return 0;
}

static int vx_shape_boolean_param(const VxConcreteShapeRequest* request,
                                  const char* name,
                                  int default_value,
                                  int* output,
                                  VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (!param) {
        *output = default_value;
        return 0;
    }
    if (param->kind != VX_SHAPE_PARAM_BOOLEAN) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be boolean.");
    }
    *output = !!param->value.boolean;
    return 0;
}

static int vx_shape_rank_range(const VxShapeTensorDescriptor* descriptor,
                               size_t minimum,
                               size_t maximum,
                               const char* path,
                               VxShapeContractError* error) {
    if (descriptor->rank < minimum || descriptor->rank > maximum) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             path, "has an unsupported rank.");
    }
    return 0;
}

static int vx_shape_same(const VxShapeTensorDescriptor* left,
                         const VxShapeTensorDescriptor* right) {
    size_t axis;
    if (left->rank != right->rank) return 0;
    for (axis = 0; axis < left->rank; ++axis) {
        if (left->shape[axis] != right->shape[axis]) return 0;
    }
    return 1;
}

static int vx_shape_rank(const VxShapeTensorDescriptor* descriptor,
                         size_t minimum,
                         size_t exact,
                         int has_exact,
                         const char* path,
                         VxShapeContractError* error) {
    if ((has_exact && descriptor->rank != exact) ||
        (!has_exact && descriptor->rank < minimum)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             path, "has an unsupported rank.");
    }
    return 0;
}

static int vx_shape_float_tensor(const VxShapeTensorDescriptor* descriptor,
                                 const char* path,
                                 VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (descriptor->dtype != VX_DTYPE_F32) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "dtype");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             field_path, "must be 'float32'.");
    }
    if (descriptor->quantization.scheme != VX_SHAPE_QUANTIZATION_NONE) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "quantization");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path,
                             "is not valid for a float operator input.");
    }
    return 0;
}

static int vx_shape_unquantized(const VxShapeTensorDescriptor* descriptor,
                                const char* path,
                                VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (descriptor->quantization.scheme == VX_SHAPE_QUANTIZATION_NONE) return 0;
    vx_shape_field_path(field_path, sizeof(field_path), path, "quantization");
    return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                         field_path, "must be absent.");
}

static int vx_shape_vector(const VxShapeTensorDescriptor* descriptor,
                           uint64_t extent,
                           const char* path,
                           VxShapeContractError* error) {
    if (descriptor->rank != 1 || descriptor->shape[0] != extent) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path, "must match the required vector extent.");
    }
    return 0;
}

void vx_shape_contract_result_clear(VxConcreteShapeResult* result) {
    size_t index;
    if (!result) return;
    for (index = 0; index < result->output_count; ++index) {
        free(result->owned_zero_points ? result->owned_zero_points[index] : NULL);
        free(result->owned_scales ? result->owned_scales[index] : NULL);
        free(result->owned_shapes ? result->owned_shapes[index] : NULL);
        free(result->owned_names ? result->owned_names[index] : NULL);
    }
    free(result->owned_zero_points);
    free(result->owned_scales);
    free(result->owned_shapes);
    free(result->owned_names);
    free(result->outputs);
    memset(result, 0, sizeof(*result));
}

static int vx_shape_result_allocate(VxConcreteShapeResult* candidate,
                                    size_t count,
                                    VxShapeContractError* error) {
    memset(candidate, 0, sizeof(*candidate));
    if (!count || count > SIZE_MAX / sizeof(*candidate->outputs) ||
        count > SIZE_MAX / sizeof(*candidate->owned_names) ||
        count > SIZE_MAX / sizeof(*candidate->owned_shapes) ||
        count > SIZE_MAX / sizeof(*candidate->owned_scales) ||
        count > SIZE_MAX / sizeof(*candidate->owned_zero_points)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator outputs", "output count is too large.");
    }
    candidate->outputs = (VxShapeNamedTensor*)calloc(count,
                                                      sizeof(*candidate->outputs));
    candidate->owned_names = (char**)calloc(count,
                                             sizeof(*candidate->owned_names));
    candidate->owned_shapes = (uint64_t**)calloc(count,
                                                  sizeof(*candidate->owned_shapes));
    candidate->owned_scales = (float**)calloc(count,
                                               sizeof(*candidate->owned_scales));
    candidate->owned_zero_points = (int32_t**)calloc(
        count, sizeof(*candidate->owned_zero_points));
    if (!candidate->outputs || !candidate->owned_names ||
        !candidate->owned_shapes || !candidate->owned_scales ||
        !candidate->owned_zero_points) {
        candidate->output_count = count;
        vx_shape_contract_result_clear(candidate);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator outputs", "allocation failed.");
    }
    candidate->output_count = count;
    return 0;
}

static int vx_shape_clone_output_at(VxConcreteShapeResult* candidate,
                                    size_t output_index,
                                    const char* name,
                                    const uint64_t* shape,
                                    size_t rank,
                                    VxDataType dtype,
                                    const VxShapeQuantization* quantization,
                                    VxShapeContractError* error) {
    VxShapeTensorDescriptor descriptor;
    char path[224];
    size_t name_length;
    if (!candidate || output_index >= candidate->output_count || !name) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST,
                             "operator outputs", "builder state is invalid.");
    }
    snprintf(path, sizeof(path), "operator output '%s'", name);
    name_length = strlen(name);
    if (name_length == SIZE_MAX) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             path, "name is too large.");
    }
    candidate->owned_names[output_index] = (char*)malloc(name_length + 1);
    if (!candidate->owned_names[output_index]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             path, "allocation failed.");
    }
    memcpy(candidate->owned_names[output_index], name, name_length + 1);
    if (rank > SIZE_MAX / sizeof(*candidate->owned_shapes[output_index])) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             path, "rank is too large.");
    }
    if (rank) {
        candidate->owned_shapes[output_index] =
            (uint64_t*)malloc(rank * sizeof(*candidate->owned_shapes[output_index]));
        if (!candidate->owned_shapes[output_index]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 path, "allocation failed.");
        }
        memcpy(candidate->owned_shapes[output_index], shape,
               rank * sizeof(*candidate->owned_shapes[output_index]));
    }
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.rank = rank;
    descriptor.shape = candidate->owned_shapes[output_index];
    descriptor.dtype = dtype;
    if (quantization) {
        descriptor.quantization.scheme = quantization->scheme;
        if (quantization->scheme == VX_SHAPE_QUANTIZATION_PER_TENSOR) {
            descriptor.quantization.scale = quantization->scale;
            descriptor.quantization.zero_point = quantization->zero_point;
        } else if (quantization->scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
            descriptor.quantization.axis = quantization->axis;
            descriptor.quantization.count = quantization->count;
            descriptor.quantization.scales = quantization->scales;
            descriptor.quantization.zero_points = quantization->zero_points;
        }
    }
    if (quantization && quantization->scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        size_t bytes;
        if (quantization->count >
                SIZE_MAX / sizeof(*candidate->owned_scales[output_index]) ||
            quantization->count >
                SIZE_MAX / sizeof(*candidate->owned_zero_points[output_index])) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 path,
                                 "metadata is too large.");
        }
        bytes = quantization->count *
            sizeof(*candidate->owned_scales[output_index]);
        candidate->owned_scales[output_index] = (float*)malloc(bytes);
        candidate->owned_zero_points[output_index] = (int32_t*)malloc(
            quantization->count *
            sizeof(*candidate->owned_zero_points[output_index]));
        if (!candidate->owned_scales[output_index] ||
            !candidate->owned_zero_points[output_index]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 path,
                                 "allocation failed.");
        }
        memcpy(candidate->owned_scales[output_index], quantization->scales,
               bytes);
        memcpy(candidate->owned_zero_points[output_index],
               quantization->zero_points,
               quantization->count *
               sizeof(*candidate->owned_zero_points[output_index]));
        descriptor.quantization.scales = candidate->owned_scales[output_index];
        descriptor.quantization.zero_points =
            candidate->owned_zero_points[output_index];
    }
    if (vx_shape_validate_descriptor(&descriptor, path, error)) return -1;
    candidate->outputs[output_index].name = candidate->owned_names[output_index];
    candidate->outputs[output_index].descriptor = descriptor;
    return 0;
}

static int vx_shape_clone_output(VxConcreteShapeResult* candidate,
                                 const uint64_t* shape,
                                 size_t rank,
                                 VxDataType dtype,
                                 const VxShapeQuantization* quantization,
                                 VxShapeContractError* error) {
    if (vx_shape_result_allocate(candidate, 1, error) ||
        vx_shape_clone_output_at(candidate, 0, "out", shape, rank, dtype,
                                 quantization, error)) {
        vx_shape_contract_result_clear(candidate);
        return -1;
    }
    return 0;
}

static int vx_shape_activation_kind(VxOperatorKind kind) {
    switch (kind) {
        case VX_OP_RELU:
        case VX_OP_LEAKY_RELU:
        case VX_OP_GELU:
        case VX_OP_SILU:
        case VX_OP_SIGMOID:
        case VX_OP_HARD_SWISH:
        case VX_OP_HARD_SIGMOID:
        case VX_OP_TANH:
        case VX_OP_SIN:
        case VX_OP_COS:
        case VX_OP_CLIP:
        case VX_OP_SOFTMAX:
        case VX_OP_LOG_SOFTMAX:
            return 1;
        default:
            return 0;
    }
}

static int vx_shape_activation_params(VxOperatorKind kind,
                                      const VxConcreteShapeRequest* request,
                                      size_t rank,
                                      VxDataType dtype,
                                      VxShapeContractError* error) {
    static const char* const ALPHA[] = {"alpha"};
    static const char* const APPROXIMATE[] = {"approximate"};
    static const char* const CLIP_BOUNDS[] = {"min", "max"};
    static const char* const AXIS[] = {"axis"};
    const VxShapeParam* param;
    double value = 0.0;
    if (kind == VX_OP_LEAKY_RELU) {
        if (vx_shape_validate_params(request, ALPHA, 1, error)) return -1;
        param = vx_shape_find_param(request, "alpha");
        if (param) {
            if (vx_shape_number(param, "operator params.alpha", &value, error)) return -1;
            if (!isfinite(value)) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     "operator params.alpha", "must be finite.");
            }
        }
        return 0;
    }
    if (kind == VX_OP_GELU) {
        if (vx_shape_validate_params(request, APPROXIMATE, 1, error)) return -1;
        param = vx_shape_find_param(request, "approximate");
        if (param && (param->kind != VX_SHAPE_PARAM_STRING ||
                      !param->value.string ||
                      (strcmp(param->value.string, "none") &&
                       strcmp(param->value.string, "tanh")))) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.approximate",
                                 "must be 'none' or 'tanh'.");
        }
        return 0;
    }
    if (kind == VX_OP_CLIP) {
        double minimum = dtype == VX_DTYPE_I32 ? -2147483648.0 : -INFINITY;
        double maximum = dtype == VX_DTYPE_I32 ? 2147483647.0 : INFINITY;
        if (vx_shape_validate_params(request, CLIP_BOUNDS, 2, error)) return -1;
        param = vx_shape_find_param(request, "min");
        if (param) {
            if (param->kind != VX_SHAPE_PARAM_NUMBER) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     "operator params",
                                     "Clip min and max must be ordered numbers.");
            }
            minimum = param->value.number;
        }
        param = vx_shape_find_param(request, "max");
        if (param) {
            if (param->kind != VX_SHAPE_PARAM_NUMBER) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     "operator params",
                                     "Clip min and max must be ordered numbers.");
            }
            maximum = param->value.number;
        }
        if (isnan(minimum) || isnan(maximum) || minimum > maximum ||
            (dtype == VX_DTYPE_I32 &&
             (!vx_shape_is_safe_integer(minimum) ||
              !vx_shape_is_safe_integer(maximum) ||
              minimum < -2147483648.0 || maximum > 2147483647.0))) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params",
                                 "Clip min and max must be ordered numbers.");
        }
        return 0;
    }
    if (kind == VX_OP_SOFTMAX || kind == VX_OP_LOG_SOFTMAX) {
        double axis;
        if (vx_shape_validate_params(request, AXIS, 1, error)) return -1;
        if (!rank) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                                 "operator input 'input'.shape",
                                 "must have rank at least 1.");
        }
        param = vx_shape_find_param(request, "axis");
        axis = param ? 0.0 : -1.0;
        if (param && vx_shape_number(param, "operator params.axis", &axis, error)) return -1;
        if (!isfinite(axis) || trunc(axis) != axis) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.axis", "must be an integer.");
        }
        if (axis < 0.0) axis += (double)rank;
        if (axis != (double)(rank - 1)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.axis",
                                 "must resolve to the last axis.");
        }
        return 0;
    }
    return vx_shape_validate_params(request, NULL, 0, error);
}

static int vx_shape_quantization_parameter_inputs(
        const VxShapeTensorDescriptor* input,
        const VxShapeTensorDescriptor* scale,
        const VxShapeTensorDescriptor* zero_point,
        VxDataType dtype,
        const VxShapeQuantization* quantization,
        VxShapeContractError* error) {
    uint64_t extent;
    if (vx_shape_float_tensor(scale, "operator input 'scale'", error)) return -1;
    if (zero_point) {
        if (zero_point->dtype != dtype) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "operator input 'zero_point'.dtype",
                                 "must match the quantized storage dtype.");
        }
        if (vx_shape_unquantized(zero_point, "operator input 'zero_point'", error)) return -1;
    }
    if (quantization->scheme == VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        if (vx_shape_vector(scale, 1, "operator input 'scale'.shape", error)) return -1;
        if (zero_point &&
            vx_shape_vector(zero_point, 1,
                            "operator input 'zero_point'.shape", error)) return -1;
        return 0;
    }
    extent = input->shape[quantization->axis];
    if (vx_shape_vector(scale, extent, "operator input 'scale'.shape", error)) return -1;
    if (zero_point &&
        vx_shape_vector(zero_point, extent,
                        "operator input 'zero_point'.shape", error)) return -1;
    return 0;
}

static int vx_shape_declared_output(const VxConcreteShapeRequest* request,
                                    const VxShapeTensorDescriptor** output,
                                    VxShapeContractError* error) {
    if (request->declared_output_count != 1 || !request->declared_outputs ||
        !request->declared_outputs[0].name ||
        strcmp(request->declared_outputs[0].name, "out")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS,
                             "declared outputs",
                             "must contain exactly the 'out' descriptor.");
    }
    if (vx_shape_validate_descriptor(&request->declared_outputs[0].descriptor,
                                     "declared output 'out'", error)) return -1;
    *output = &request->declared_outputs[0].descriptor;
    return 0;
}

static int vx_shape_normalization_params(const VxConcreteShapeRequest* request,
                                         uint64_t feature,
                                         VxShapeContractError* error) {
    static const char* const FIELDS[] = {"eps", "d_model"};
    const VxShapeParam* param;
    double value = 0.0;
    if (vx_shape_validate_params(request, FIELDS, 2, error)) return -1;
    param = vx_shape_find_param(request, "eps");
    if (param) {
        if (vx_shape_number(param, "operator params.eps", &value, error)) return -1;
        if (!isfinite(value) || value <= 0.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.eps",
                                 "must be finite and positive.");
        }
    }
    param = vx_shape_find_param(request, "d_model");
    if (param) {
        if (vx_shape_number(param, "operator params.d_model", &value, error)) return -1;
        if (!vx_shape_is_safe_integer(value) || value <= 0.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.d_model",
                                 "must be a positive safe integer.");
        }
        if ((uint64_t)value != feature) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator params.d_model",
                                 "must equal the feature extent.");
        }
    }
    return 0;
}

typedef enum VxDenseLayout {
    VX_DENSE_LAYOUT_DIN_DOUT = 0,
    VX_DENSE_LAYOUT_DOUT_DIN = 1
} VxDenseLayout;

static int vx_shape_dense_layout(VxOperatorKind kind,
                                 const VxConcreteShapeRequest* request,
                                 VxDenseLayout* layout,
                                 VxShapeContractError* error) {
    static const char* const FIELDS[] = {"weight_layout", "transB"};
    const VxShapeParam* named;
    const VxShapeParam* transposed;
    if (vx_shape_validate_params(request, FIELDS, 2, error)) return -1;
    named = vx_shape_find_param(request, "weight_layout");
    transposed = vx_shape_find_param(request, "transB");
    if (named && transposed) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params",
                             "must not specify both weight_layout and transB.");
    }
    if (named) {
        if (named->kind != VX_SHAPE_PARAM_STRING || !named->value.string) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.weight_layout",
                                 "must be 'din_dout' or 'dout_din'.");
        }
        if (!strcmp(named->value.string, "din_dout")) {
            *layout = VX_DENSE_LAYOUT_DIN_DOUT;
            return 0;
        }
        if (!strcmp(named->value.string, "dout_din")) {
            *layout = VX_DENSE_LAYOUT_DOUT_DIN;
            return 0;
        }
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout",
                             "must be 'din_dout' or 'dout_din'.");
    }
    if (transposed) {
        if (transposed->kind != VX_SHAPE_PARAM_BOOLEAN) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.transB", "must be boolean.");
        }
        *layout = transposed->value.boolean
            ? VX_DENSE_LAYOUT_DOUT_DIN : VX_DENSE_LAYOUT_DIN_DOUT;
        return 0;
    }
    *layout = kind == VX_OP_LINEAR
        ? VX_DENSE_LAYOUT_DOUT_DIN : VX_DENSE_LAYOUT_DIN_DOUT;
    return 0;
}

static int vx_shape_spatial_kind(VxOperatorKind kind) {
    switch (kind) {
        case VX_OP_BATCH_MATMUL:
        case VX_OP_CONV_1D:
        case VX_OP_CONV_2D:
        case VX_OP_CONV_TRANSPOSE_2D:
        case VX_OP_MAX_POOL_2D:
        case VX_OP_AVERAGE_POOL_2D:
        case VX_OP_GLOBAL_AVERAGE_POOL:
        case VX_OP_RESIZE:
        case VX_OP_RESIZE_NEAREST_2D:
        case VX_OP_UPSAMPLE_NEAREST_2D:
            return 1;
        default:
            return 0;
    }
}

static int vx_shape_param_integer_value(double value,
                                        uint64_t minimum,
                                        const char* path,
                                        uint64_t* output,
                                        VxShapeContractError* error) {
    if (!vx_shape_is_safe_integer(value) || value < (double)minimum) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, minimum ? "must be a positive safe integer."
                                           : "must be a non-negative safe integer.");
    }
    *output = (uint64_t)value;
    return 0;
}

static int vx_shape_param_scalar(const VxConcreteShapeRequest* request,
                                 const char* name,
                                 uint64_t default_value,
                                 uint64_t minimum,
                                 int required,
                                 uint64_t* output,
                                 VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (!param) {
        if (required) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 path, "is required.");
        }
        *output = default_value;
        return 0;
    }
    if (param->kind == VX_SHAPE_PARAM_NUMBER) {
        return vx_shape_param_integer_value(param->value.number, minimum,
                                            path, output, error);
    }
    if (param->kind == VX_SHAPE_PARAM_NUMBER_ARRAY &&
        param->value.number_array.count == 1 &&
        param->value.number_array.values) {
        return vx_shape_param_integer_value(
            param->value.number_array.values[0], minimum, path, output, error);
    }
    return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                         path, "must be a scalar or one-element array.");
}

static int vx_shape_param_pair(const VxConcreteShapeRequest* request,
                               const char* name,
                               uint64_t default_value,
                               uint64_t minimum,
                               int required,
                               uint64_t output[2],
                               VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    const double* values = NULL;
    size_t count = 0;
    double scalar = 0.0;
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (!param) {
        if (required) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 path, "is required.");
        }
        output[0] = default_value;
        output[1] = default_value;
        return 0;
    }
    if (param->kind == VX_SHAPE_PARAM_NUMBER) {
        scalar = param->value.number;
        values = &scalar;
        count = 1;
    } else if (param->kind == VX_SHAPE_PARAM_NUMBER_ARRAY) {
        values = param->value.number_array.values;
        count = param->value.number_array.count;
    }
    if (!values || count < 1 || count > 2) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path,
                             "must be a scalar or a one/two-element array.");
    }
    if (vx_shape_param_integer_value(values[0], minimum, path,
                                     &output[0], error) ||
        vx_shape_param_integer_value(values[count == 1 ? 0 : 1], minimum,
                                     path, &output[1], error)) return -1;
    return 0;
}

static int vx_shape_param_string_default(
        const VxConcreteShapeRequest* request,
        const char* name,
        const char* default_value,
        const char** output,
        VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (!param) {
        *output = default_value;
        return 0;
    }
    if (param->kind != VX_SHAPE_PARAM_STRING || !param->value.string) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be a string.");
    }
    *output = param->value.string;
    return 0;
}

static int vx_shape_param_false_or_absent(
        const VxConcreteShapeRequest* request,
        const char* name,
        VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    if (!param) return 0;
    snprintf(path, sizeof(path), "operator params.%s", name);
    if ((param->kind == VX_SHAPE_PARAM_BOOLEAN && !param->value.boolean) ||
        (param->kind == VX_SHAPE_PARAM_NUMBER && param->value.number == 0.0)) {
        return 0;
    }
    return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                         path, "must be false, 0, or absent.");
}

static int vx_shape_param_activation(const VxConcreteShapeRequest* request,
                                     uint64_t* activation,
                                     VxShapeContractError* error) {
    if (vx_shape_param_scalar(request, "relu", 0, 0, 0,
                              activation, error)) return -1;
    if (*activation > 2) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.relu", "must be 0, 1, or 2.");
    }
    return 0;
}

static int vx_shape_checked_add_u64(uint64_t left,
                                    uint64_t right,
                                    const char* path,
                                    uint64_t* output,
                                    VxShapeContractError* error) {
    if (left > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER - right) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DOMAIN,
                             path, "exceeds the safe integer range.");
    }
    *output = left + right;
    return 0;
}

static int vx_shape_checked_multiply_u64(uint64_t left,
                                         uint64_t right,
                                         const char* path,
                                         uint64_t* output,
                                         VxShapeContractError* error) {
    if (left && right > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER / left) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DOMAIN,
                             path, "exceeds the safe integer range.");
    }
    *output = left * right;
    return 0;
}

static int vx_shape_window_output(uint64_t input,
                                  uint64_t kernel,
                                  uint64_t stride,
                                  uint64_t pad_before,
                                  uint64_t pad_after,
                                  uint64_t dilation,
                                  const char* path,
                                  uint64_t* output,
                                  VxShapeContractError* error) {
    uint64_t effective_minus_one = 0;
    uint64_t effective = 0;
    uint64_t padded = 0;
    uint64_t quotient;
    if (vx_shape_checked_multiply_u64(dilation, kernel - 1, path,
                                      &effective_minus_one, error) ||
        vx_shape_checked_add_u64(effective_minus_one, 1, path,
                                 &effective, error) ||
        vx_shape_checked_add_u64(input, pad_before, path, &padded, error) ||
        vx_shape_checked_add_u64(padded, pad_after, path, &padded, error)) {
        return -1;
    }
    if (padded < effective) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path, "produces a non-positive output extent.");
    }
    quotient = (padded - effective) / stride;
    if (vx_shape_checked_add_u64(quotient, 1, path, output, error)) return -1;
    if (!*output) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path, "produces a non-positive output extent.");
    }
    return 0;
}

static int vx_shape_transpose_output(uint64_t input,
                                     uint64_t kernel,
                                     uint64_t stride,
                                     uint64_t padding,
                                     const char* path,
                                     uint64_t* output,
                                     VxShapeContractError* error) {
    uint64_t base = 0;
    uint64_t doubled_padding = 0;
    if (vx_shape_checked_multiply_u64(input - 1, stride, path, &base, error) ||
        vx_shape_checked_add_u64(base, kernel, path, &base, error) ||
        vx_shape_checked_multiply_u64(2, padding, path,
                                      &doubled_padding, error)) return -1;
    if (base <= doubled_padding) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path, "produces a non-positive output extent.");
    }
    *output = base - doubled_padding;
    return 0;
}

static int vx_shape_quantization_equal(const VxShapeQuantization* left,
                                       const VxShapeQuantization* right) {
    size_t index;
    if (left->scheme != right->scheme) return 0;
    if (left->scheme == VX_SHAPE_QUANTIZATION_NONE) return 1;
    if (left->scheme == VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        return left->scale == right->scale &&
               left->zero_point == right->zero_point;
    }
    if (left->axis != right->axis || left->count != right->count) return 0;
    for (index = 0; index < left->count; ++index) {
        if (left->scales[index] != right->scales[index] ||
            left->zero_points[index] != right->zero_points[index]) return 0;
    }
    return 1;
}

static int vx_shape_spatial_pads(const VxConcreteShapeRequest* request,
                                 int symmetric_only,
                                 uint64_t output[4],
                                 VxShapeContractError* error) {
    const VxShapeParam* pads = vx_shape_find_param(request, "pads");
    const VxShapeParam* padding_param = vx_shape_find_param(request, "padding");
    uint64_t padding[2];
    size_t index;
    if (vx_shape_param_pair(request, "padding", 0, 0, 0,
                            padding, error)) return -1;
    if (!pads) {
        output[0] = padding[0];
        output[1] = padding[1];
        output[2] = padding[0];
        output[3] = padding[1];
        return 0;
    }
    if (pads->kind != VX_SHAPE_PARAM_NUMBER_ARRAY ||
        pads->value.number_array.count != 4 ||
        !pads->value.number_array.values) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.pads",
                             "must contain four non-negative safe integers.");
    }
    for (index = 0; index < 4; ++index) {
        if (vx_shape_param_integer_value(pads->value.number_array.values[index],
                                         0, "operator params.pads",
                                         &output[index], error)) return -1;
    }
    if (padding_param &&
        (output[0] != padding[0] || output[1] != padding[1] ||
         output[2] != padding[0] || output[3] != padding[1])) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params",
                             "padding and pads must describe the same symmetric padding.");
    }
    if (symmetric_only &&
        (output[0] != output[2] || output[1] != output[3])) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.pads",
                             "must be symmetric for this operator.");
    }
    return 0;
}

static const VxShapeTensorDescriptor* vx_shape_input_descriptor(
        const VxConcreteShapeRequest* request,
        const char* name) {
    const VxShapeNamedTensor* input = vx_shape_find_input(request, name);
    return input ? &input->descriptor : NULL;
}

static int vx_shape_infer_batch_matmul(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* left = vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right = vx_shape_input_descriptor(request, "b");
    uint64_t output_shape[8];
    size_t left_batch;
    size_t right_batch;
    size_t batch_rank;
    size_t axis;
    if (vx_shape_float_tensor(left, "operator input 'a'", error) ||
        vx_shape_float_tensor(right, "operator input 'b'", error) ||
        vx_shape_rank(left, 2, 0, 0, "operator input 'a'.shape", error) ||
        vx_shape_rank(right, 2, 0, 0, "operator input 'b'.shape", error)) {
        return -1;
    }
    if (left->rank > 8 || right->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "BatchMatMul operand ranks must not exceed 8.");
    }
    if (vx_shape_validate_params(request, NULL, 0, error)) return -1;
    if (left->shape[left->rank - 1] != right->shape[right->rank - 2]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'b'.shape",
                             "contracted dimensions must be equal.");
    }
    left_batch = left->rank - 2;
    right_batch = right->rank - 2;
    batch_rank = left_batch > right_batch ? left_batch : right_batch;
    for (axis = 0; axis < batch_rank; ++axis) {
        uint64_t left_extent = axis < batch_rank - left_batch
            ? 1 : left->shape[axis - (batch_rank - left_batch)];
        uint64_t right_extent = axis < batch_rank - right_batch
            ? 1 : right->shape[axis - (batch_rank - right_batch)];
        if (left_extent != right_extent && left_extent != 1 && right_extent != 1) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator inputs",
                                 "batch dimensions are not broadcast-compatible.");
        }
        output_shape[axis] = left_extent > right_extent
            ? left_extent : right_extent;
    }
    output_shape[batch_rank] = left->shape[left->rank - 2];
    output_shape[batch_rank + 1] = right->shape[right->rank - 1];
    return vx_shape_clone_output(candidate, output_shape, batch_rank + 2,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_conv1d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "stride", "padding", "groups", "relu", "data_layout", "weight_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight = vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias = vx_shape_input_descriptor(request, "bias");
    const char* data_layout;
    const char* weight_layout;
    uint64_t stride;
    uint64_t padding;
    uint64_t groups;
    uint64_t activation;
    uint64_t output_length;
    uint64_t output_shape[3];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank(input, 0, 3, 1, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 3, 1, "operator input 'weight'.shape", error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NLC",
                                      &data_layout, error) ||
        vx_shape_param_string_default(request, "weight_layout", "WIO",
                                      &weight_layout, error) ||
        vx_shape_param_scalar(request, "stride", 1, 1, 0,
                              &stride, error) ||
        vx_shape_param_scalar(request, "padding", 0, 0, 0,
                              &padding, error) ||
        vx_shape_param_scalar(request, "groups", 1, 1, 0,
                              &groups, error) ||
        vx_shape_param_activation(request, &activation, error)) return -1;
    (void)activation;
    if (strcmp(data_layout, "NLC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NLC'.");
    }
    if (strcmp(weight_layout, "WIO")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout", "must be 'WIO'.");
    }
    if (input->shape[2] % groups || weight->shape[2] % groups ||
        weight->shape[1] != input->shape[2] / groups) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape[1]",
                             "is incompatible with input channels and groups.");
    }
    if (bias &&
        (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, weight->shape[2],
                         "operator input 'bias'.shape", error))) return -1;
    if (vx_shape_window_output(input->shape[1], weight->shape[0], stride,
                               padding, padding, 1,
                               "operator input 'input'.shape[1]",
                               &output_length, error)) return -1;
    output_shape[0] = input->shape[0];
    output_shape[1] = output_length;
    output_shape[2] = weight->shape[2];
    return vx_shape_clone_output(candidate, output_shape, 3,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_conv2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "stride", "padding", "pads", "dilation", "groups", "relu",
        "data_layout", "weight_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight = vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias = vx_shape_input_descriptor(request, "bias");
    const char* data_layout;
    const char* weight_layout;
    uint64_t stride[2];
    uint64_t dilation[2];
    uint64_t pads[4];
    uint64_t groups;
    uint64_t activation;
    uint64_t output_channels = 0;
    uint64_t output_shape[4];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 4, 1, "operator input 'weight'.shape", error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error) ||
        vx_shape_param_string_default(request, "weight_layout", "HWIO",
                                      &weight_layout, error) ||
        vx_shape_param_pair(request, "stride", 1, 1, 0, stride, error) ||
        vx_shape_spatial_pads(request, 0, pads, error) ||
        vx_shape_param_pair(request, "dilation", 1, 1, 0, dilation, error) ||
        vx_shape_param_scalar(request, "groups", 1, 1, 0, &groups, error) ||
        vx_shape_param_activation(request, &activation, error)) return -1;
    (void)activation;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    if (!strcmp(weight_layout, "HWCM")) {
        if (groups != input->shape[3] || weight->shape[2] != input->shape[3]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'weight'.shape",
                                 "depthwise Conv2D requires HWCM geometry.");
        }
        if (vx_shape_checked_multiply_u64(input->shape[3], weight->shape[3],
                                          "Conv2D output channels",
                                          &output_channels, error)) return -1;
    } else if (!strcmp(weight_layout, "HWIO")) {
        if (input->shape[3] % groups || weight->shape[3] % groups ||
            weight->shape[2] != input->shape[3] / groups) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'weight'.shape",
                                 "grouped Conv2D requires compatible HWIO geometry.");
        }
        output_channels = weight->shape[3];
    } else {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout",
                             "must be 'HWIO' or 'HWCM'.");
    }
    if (bias &&
        (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, output_channels,
                         "operator input 'bias'.shape", error))) return -1;
    output_shape[0] = input->shape[0];
    if (vx_shape_window_output(input->shape[1], weight->shape[0], stride[0],
                               pads[0], pads[2], dilation[0],
                               "operator input 'input'.shape[1]",
                               &output_shape[1], error) ||
        vx_shape_window_output(input->shape[2], weight->shape[1], stride[1],
                               pads[1], pads[3], dilation[1],
                               "operator input 'input'.shape[2]",
                               &output_shape[2], error)) return -1;
    output_shape[3] = output_channels;
    return vx_shape_clone_output(candidate, output_shape, 4,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_conv_transpose2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "kernel", "stride", "padding", "data_layout", "weight_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight = vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias = vx_shape_input_descriptor(request, "bias");
    const char* data_layout;
    const char* weight_layout;
    uint64_t kernel[2];
    uint64_t stride[2];
    uint64_t padding[2];
    uint64_t output_shape[4];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 4, 1, "operator input 'weight'.shape", error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error) ||
        vx_shape_param_string_default(request, "weight_layout", "HWIO",
                                      &weight_layout, error) ||
        vx_shape_param_pair(request, "kernel", 1, 1, 1, kernel, error) ||
        vx_shape_param_pair(request, "stride", 1, 1, 0, stride, error) ||
        vx_shape_param_pair(request, "padding", 0, 0, 0,
                            padding, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    if (strcmp(weight_layout, "HWIO")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout", "must be 'HWIO'.");
    }
    if (kernel[0] != weight->shape[0] || kernel[1] != weight->shape[1]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.kernel",
                             "must equal the HWIO weight kernel extents.");
    }
    if (weight->shape[2] != input->shape[3]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape[2]",
                             "must equal input channels.");
    }
    if (bias &&
        (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, weight->shape[3],
                         "operator input 'bias'.shape", error))) return -1;
    output_shape[0] = input->shape[0];
    if (vx_shape_transpose_output(input->shape[1], kernel[0], stride[0],
                                  padding[0],
                                  "operator input 'input'.shape[1]",
                                  &output_shape[1], error) ||
        vx_shape_transpose_output(input->shape[2], kernel[1], stride[1],
                                  padding[1],
                                  "operator input 'input'.shape[2]",
                                  &output_shape[2], error)) return -1;
    output_shape[3] = weight->shape[3];
    return vx_shape_clone_output(candidate, output_shape, 4,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_pool2d(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const MAX_FIELDS[] = {
        "kernel", "stride", "padding", "pads", "dilation", "ceil_mode",
        "data_layout"
    };
    static const char* const AVERAGE_FIELDS[] = {
        "kernel", "stride", "padding", "pads", "dilation", "ceil_mode",
        "count_include_pad", "auto_pad", "data_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const char* data_layout;
    const char* auto_pad;
    uint64_t kernel[2];
    uint64_t stride[2];
    uint64_t dilation[2];
    uint64_t pads[4];
    uint64_t output_shape[4];
    int average = kind == VX_OP_AVERAGE_POOL_2D;
    if (vx_shape_rank(input, 0, 4, 1,
                      "operator input 'input'.shape", error) ||
        vx_shape_validate_params(
            request, average ? AVERAGE_FIELDS : MAX_FIELDS,
            average ? sizeof(AVERAGE_FIELDS) / sizeof(AVERAGE_FIELDS[0])
                    : sizeof(MAX_FIELDS) / sizeof(MAX_FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error) ||
        vx_shape_param_false_or_absent(request, "ceil_mode", error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    if (average) {
        if (vx_shape_param_false_or_absent(request, "count_include_pad", error) ||
            vx_shape_param_string_default(request, "auto_pad", "NOTSET",
                                          &auto_pad, error)) return -1;
        if (strcmp(auto_pad, "NOTSET") && strcmp(auto_pad, "")) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.auto_pad",
                                 "must be 'NOTSET', empty, or absent.");
        }
    }
    if (vx_shape_param_pair(request, "dilation", 1, 1, 0,
                            dilation, error)) return -1;
    if (dilation[0] != 1 || dilation[1] != 1) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.dilation",
                             "dilated pooling is not supported.");
    }
    if (vx_shape_param_pair(request, "kernel", 1, 1, 1, kernel, error) ||
        vx_shape_param_pair(request, "stride", 1, 1, 0, stride, error) ||
        vx_shape_spatial_pads(request, average, pads, error)) return -1;

    if (input->dtype == VX_DTYPE_F32) {
        if (vx_shape_float_tensor(input, "operator input 'input'", error)) return -1;
    } else if (!average &&
               (input->dtype == VX_DTYPE_I8 || input->dtype == VX_DTYPE_U8)) {
        if (input->quantization.scheme != VX_SHAPE_QUANTIZATION_PER_TENSOR) {
            return vx_shape_fail(error,
                                 VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 "operator input 'input'.quantization",
                                 "raw byte MaxPool2D requires per-tensor quantization.");
        }
    } else {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'input'.dtype",
                             average ? "AveragePool2D requires float32."
                                     : "MaxPool2D requires float32 or I8/U8.");
    }
    output_shape[0] = input->shape[0];
    if (vx_shape_window_output(input->shape[1], kernel[0], stride[0],
                               pads[0], pads[2], 1,
                               "operator input 'input'.shape[1]",
                               &output_shape[1], error) ||
        vx_shape_window_output(input->shape[2], kernel[1], stride[1],
                               pads[1], pads[3], 1,
                               "operator input 'input'.shape[2]",
                               &output_shape[2], error)) return -1;
    output_shape[3] = input->shape[3];
    return vx_shape_clone_output(candidate, output_shape, 4, input->dtype,
                                 &input->quantization, error);
}

static int vx_shape_infer_global_average_pool(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"data_layout"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const char* data_layout;
    uint64_t output_shape[4];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    output_shape[0] = input->shape[0];
    output_shape[1] = 1;
    output_shape[2] = 1;
    output_shape[3] = input->shape[3];
    return vx_shape_clone_output(candidate, output_shape, 4,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_resize_dtype(
        const VxShapeTensorDescriptor* input,
        const VxShapeTensorDescriptor* output,
        VxShapeContractError* error) {
    if (input->dtype == VX_DTYPE_F32) {
        if (vx_shape_float_tensor(input, "operator input 'input'", error)) return -1;
        if (output->dtype != VX_DTYPE_F32) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "declared output 'out'.dtype",
                                 "must be 'float32'.");
        }
        return vx_shape_unquantized(output, "declared output 'out'", error);
    }
    if ((input->dtype != VX_DTYPE_I8 && input->dtype != VX_DTYPE_U8) ||
        input->quantization.scheme != VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             "operator input 'input'.quantization",
                             "raw byte resize requires per-tensor I8/U8 quantization.");
    }
    if (output->dtype != input->dtype ||
        !vx_shape_quantization_equal(&input->quantization,
                                     &output->quantization)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             "declared output 'out'.quantization",
                             "must preserve the exact input byte domain.");
    }
    return 0;
}

static int vx_shape_infer_resize(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "mode", "coordinate_transformation_mode", "coordinate_transform_mode",
        "nearest_mode", "align_corners", "antialias", "data_layout"
    };
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* declared = NULL;
    const VxShapeParam* legacy_transform;
    const char* data_layout;
    const char* mode;
    const char* transform;
    const char* nearest_mode;
    int nearest;
    if (vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_declared_output(request, &declared, error) ||
        vx_shape_rank(declared, 0, 4, 1,
                      "declared output 'out'.shape", error) ||
        vx_shape_resize_dtype(input, declared, error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    legacy_transform = vx_shape_find_param(request, "coordinate_transform_mode");
    if (legacy_transform) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.coordinate_transform_mode",
                             "is not defined; use coordinate_transformation_mode.");
    }
    if (vx_shape_param_string_default(
            request, "mode",
            kind == VX_OP_RESIZE_NEAREST_2D ? "nearest" : "linear",
            &mode, error)) return -1;
    if (kind == VX_OP_RESIZE_NEAREST_2D && strcmp(mode, "nearest")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.mode", "must be 'nearest'.");
    }
    if (kind == VX_OP_RESIZE && strcmp(mode, "nearest") &&
        strcmp(mode, "linear")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.mode",
                             "must be 'nearest' or 'linear'.");
    }
    nearest = kind == VX_OP_RESIZE_NEAREST_2D || !strcmp(mode, "nearest");
    if (input->dtype != VX_DTYPE_F32 && !nearest) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.mode",
                             "raw I8/U8 resize requires explicit nearest mode.");
    }
    if (vx_shape_param_string_default(
            request, "coordinate_transformation_mode",
            nearest ? "asymmetric" : "half_pixel", &transform, error)) return -1;
    if ((nearest && strcmp(transform, "asymmetric")) ||
        (!nearest && strcmp(transform, "half_pixel"))) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.coordinate_transformation_mode",
                             nearest ? "must be 'asymmetric' for nearest resize."
                                     : "must be 'half_pixel' for linear resize.");
    }
    if (nearest) {
        if (vx_shape_param_string_default(request, "nearest_mode", "floor",
                                          &nearest_mode, error)) return -1;
        if (strcmp(nearest_mode, "floor")) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.nearest_mode",
                                 "must be 'floor'.");
        }
    } else if (vx_shape_find_param(request, "nearest_mode")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.nearest_mode",
                             "is valid only for nearest resize.");
    }
    if (vx_shape_param_false_or_absent(request, "align_corners", error) ||
        vx_shape_param_false_or_absent(request, "antialias", error)) return -1;
    if (declared->shape[0] != input->shape[0] ||
        declared->shape[3] != input->shape[3]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "declared output 'out'.shape",
                             "must preserve NHWC batch and channel extents.");
    }
    return vx_shape_clone_output(candidate, declared->shape, declared->rank,
                                 declared->dtype, &declared->quantization, error);
}

static int vx_shape_infer_upsample_nearest2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"data_layout"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const char* data_layout;
    uint64_t output_shape[4];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    output_shape[0] = input->shape[0];
    if (vx_shape_checked_multiply_u64(input->shape[1], 2,
                                      "operator input 'input'.shape[1]",
                                      &output_shape[1], error) ||
        vx_shape_checked_multiply_u64(input->shape[2], 2,
                                      "operator input 'input'.shape[2]",
                                      &output_shape[2], error)) return -1;
    output_shape[3] = input->shape[3];
    return vx_shape_clone_output(candidate, output_shape, 4,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_spatial_kind(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    switch (kind) {
        case VX_OP_BATCH_MATMUL:
            return vx_shape_infer_batch_matmul(request, candidate, error);
        case VX_OP_CONV_1D:
            return vx_shape_infer_conv1d(request, candidate, error);
        case VX_OP_CONV_2D:
            return vx_shape_infer_conv2d(request, candidate, error);
        case VX_OP_CONV_TRANSPOSE_2D:
            return vx_shape_infer_conv_transpose2d(request, candidate, error);
        case VX_OP_MAX_POOL_2D:
        case VX_OP_AVERAGE_POOL_2D:
            return vx_shape_infer_pool2d(kind, request, candidate, error);
        case VX_OP_GLOBAL_AVERAGE_POOL:
            return vx_shape_infer_global_average_pool(request, candidate, error);
        case VX_OP_RESIZE:
        case VX_OP_RESIZE_NEAREST_2D:
            return vx_shape_infer_resize(kind, request, candidate, error);
        case VX_OP_UPSAMPLE_NEAREST_2D:
            return vx_shape_infer_upsample_nearest2d(request, candidate, error);
        default:
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                                 "operator",
                                 "has no implemented spatial contract.");
    }
}

static int vx_shape_non_spatial_wave_b_kind(VxOperatorKind kind) {
    switch (kind) {
        case VX_OP_SUB:
        case VX_OP_DIV:
        case VX_OP_EQUAL:
        case VX_OP_GREATER_OR_EQUAL:
        case VX_OP_WHERE:
        case VX_OP_REDUCE_SUM:
        case VX_OP_REDUCE_MEAN:
        case VX_OP_ARG_MAX:
        case VX_OP_TRANSPOSE:
        case VX_OP_FLATTEN:
        case VX_OP_SQUEEZE:
        case VX_OP_UNSQUEEZE:
        case VX_OP_RESHAPE:
        case VX_OP_EXPAND:
        case VX_OP_CONCAT:
        case VX_OP_SPLIT:
        case VX_OP_SLICE:
        case VX_OP_PAD:
        case VX_OP_GATHER:
        case VX_OP_GATHER_ELEMENTS:
            return 1;
        default:
            return 0;
    }
}

static int vx_shape_param_signed_integer_value(double value,
                                               const char* path,
                                               int64_t* output,
                                               VxShapeContractError* error) {
    if (!vx_shape_is_safe_integer(value)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be an integer.");
    }
    *output = (int64_t)value;
    return 0;
}

static int vx_shape_normalize_axis(const VxConcreteShapeRequest* request,
                                   const char* name,
                                   size_t rank,
                                   int64_t default_axis,
                                   int allow_boundary,
                                   size_t* output,
                                   VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    int64_t axis = default_axis;
    int64_t upper = (int64_t)rank + (allow_boundary ? 1 : 0);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if ((!allow_boundary && rank == 0) || rank > (size_t)INT64_MAX) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             path, "requires an input rank of at least 1.");
    }
    if (param) {
        if (param->kind != VX_SHAPE_PARAM_NUMBER ||
            vx_shape_param_signed_integer_value(param->value.number, path,
                                                &axis, error)) return -1;
    }
    if (axis < 0) axis += (int64_t)rank;
    if (axis < 0 || axis >= upper) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "resolves outside the supported rank.");
    }
    *output = (size_t)axis;
    return 0;
}

static int vx_shape_boolean_default(const VxConcreteShapeRequest* request,
                                    const char* name,
                                    int default_value,
                                    int* output,
                                    VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (!param) {
        *output = default_value;
        return 0;
    }
    if (param->kind != VX_SHAPE_PARAM_BOOLEAN) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be boolean.");
    }
    *output = !!param->value.boolean;
    return 0;
}

static int vx_shape_param_number_array(
        const VxConcreteShapeRequest* request,
        const char* name,
        int required,
        const double** values,
        size_t* count,
        VxShapeContractError* error) {
    const VxShapeParam* param = vx_shape_find_param(request, name);
    char path[224];
    snprintf(path, sizeof(path), "operator params.%s", name);
    if (!param) {
        if (required) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 path, "is required.");
        }
        *values = NULL;
        *count = 0;
        return 0;
    }
    if (param->kind != VX_SHAPE_PARAM_NUMBER_ARRAY ||
        (param->value.number_array.count &&
         !param->value.number_array.values)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be an array.");
    }
    *values = param->value.number_array.values;
    *count = param->value.number_array.count;
    return 0;
}

static int vx_shape_product_range(const uint64_t* shape,
                                  size_t begin,
                                  size_t end,
                                  const char* path,
                                  uint64_t* output,
                                  VxShapeContractError* error) {
    uint64_t product = 1;
    size_t axis;
    for (axis = begin; axis < end; ++axis) {
        if (vx_shape_checked_multiply_u64(product, shape[axis], path,
                                          &product, error)) return -1;
    }
    *output = product;
    return 0;
}

static int vx_shape_broadcast(const VxShapeTensorDescriptor* left,
                              const VxShapeTensorDescriptor* right,
                              uint64_t output[8],
                              size_t* output_rank,
                              VxShapeContractError* error) {
    size_t rank = left->rank > right->rank ? left->rank : right->rank;
    size_t axis;
    if (rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "broadcast rank must not exceed 8.");
    }
    for (axis = 0; axis < rank; ++axis) {
        size_t left_offset = rank - left->rank;
        size_t right_offset = rank - right->rank;
        uint64_t a = axis < left_offset ? 1 : left->shape[axis - left_offset];
        uint64_t b = axis < right_offset ? 1 : right->shape[axis - right_offset];
        if (a != b && a != 1 && b != 1) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator inputs",
                                 "input shapes are not broadcast-compatible.");
        }
        output[axis] = a > b ? a : b;
    }
    *output_rank = rank;
    return 0;
}

static int vx_shape_optional_declared_output(
        const VxConcreteShapeRequest* request,
        const VxShapeTensorDescriptor** output,
        VxShapeContractError* error) {
    if (!request->declared_output_count) {
        *output = NULL;
        return 0;
    }
    return vx_shape_declared_output(request, output, error);
}

static int vx_shape_reshape_quantization(
        const VxShapeTensorDescriptor* input,
        const uint64_t* output_shape,
        size_t output_rank,
        VxShapeQuantization* output,
        VxShapeContractError* error) {
    const VxShapeQuantization* quantization = &input->quantization;
    size_t candidate = 0;
    size_t candidate_count = 0;
    size_t axis;
    if (quantization->scheme != VX_SHAPE_QUANTIZATION_PER_AXIS) {
        *output = *quantization;
        return 0;
    }
    for (axis = 0; axis < output_rank; ++axis) {
        uint64_t input_prefix;
        uint64_t output_prefix;
        uint64_t input_suffix;
        uint64_t output_suffix;
        if (input->shape[quantization->axis] != output_shape[axis]) continue;
        if (vx_shape_product_range(input->shape, 0, quantization->axis,
                                   "operator input 'input'.quantization input prefix",
                                   &input_prefix, error) ||
            vx_shape_product_range(output_shape, 0, axis,
                                   "operator input 'input'.quantization output prefix",
                                   &output_prefix, error) ||
            vx_shape_product_range(input->shape, quantization->axis + 1,
                                   input->rank,
                                   "operator input 'input'.quantization input suffix",
                                   &input_suffix, error) ||
            vx_shape_product_range(output_shape, axis + 1, output_rank,
                                   "operator input 'input'.quantization output suffix",
                                   &output_suffix, error)) return -1;
        if (input_prefix == output_prefix && input_suffix == output_suffix) {
            candidate = axis;
            candidate_count++;
        }
    }
    if (candidate_count != 1) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
            "operator input 'input'.quantization",
            "cannot map the per-axis coordinate uniquely through this reshape.");
    }
    *output = *quantization;
    output->axis = candidate;
    return 0;
}

static int vx_shape_infer_broadcast_arithmetic(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* left = vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right = vx_shape_input_descriptor(request, "b");
    uint64_t output_shape[8];
    size_t output_rank;
    if (vx_shape_float_tensor(left, "operator input 'a'", error) ||
        vx_shape_float_tensor(right, "operator input 'b'", error) ||
        vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_broadcast(left, right, output_shape, &output_rank, error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_broadcast_comparison(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* left = vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right = vx_shape_input_descriptor(request, "b");
    uint64_t output_shape[8];
    size_t output_rank;
    if (left->dtype != VX_DTYPE_I32 || right->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator inputs",
                             "comparison inputs must both be int32.");
    }
    if (vx_shape_unquantized(left, "operator input 'a'", error) ||
        vx_shape_unquantized(right, "operator input 'b'", error) ||
        vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_broadcast(left, right, output_shape, &output_rank, error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 VX_DTYPE_I32, NULL, error);
}

static int vx_shape_infer_where(const VxConcreteShapeRequest* request,
                                VxConcreteShapeResult* candidate,
                                VxShapeContractError* error) {
    const VxShapeTensorDescriptor* condition =
        vx_shape_input_descriptor(request, "condition");
    const VxShapeTensorDescriptor* left = vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right = vx_shape_input_descriptor(request, "b");
    VxShapeTensorDescriptor data_shape;
    uint64_t data_dimensions[8];
    uint64_t output_shape[8];
    size_t data_rank;
    size_t output_rank;
    if (condition->dtype != VX_DTYPE_F32 && condition->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'condition'.dtype",
                             "must be float32 or int32.");
    }
    if (vx_shape_unquantized(condition, "operator input 'condition'", error)) return -1;
    if (left->dtype != right->dtype) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator inputs 'a' and 'b'",
                             "must have the same dtype.");
    }
    if (left->dtype != VX_DTYPE_F32 && left->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator data inputs",
                             "must be float32 or int32.");
    }
    if (vx_shape_unquantized(left, "operator input 'a'", error) ||
        vx_shape_unquantized(right, "operator input 'b'", error) ||
        vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_broadcast(left, right, data_dimensions, &data_rank, error)) {
        return -1;
    }
    memset(&data_shape, 0, sizeof(data_shape));
    data_shape.rank = data_rank;
    data_shape.shape = data_dimensions;
    data_shape.dtype = left->dtype;
    if (vx_shape_broadcast(condition, &data_shape, output_shape,
                           &output_rank, error)) return -1;
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 left->dtype, NULL, error);
}

static int vx_shape_infer_reduction(VxOperatorKind kind,
                                    const VxConcreteShapeRequest* request,
                                    VxConcreteShapeResult* candidate,
                                    VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis", "keepdims"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    uint64_t output_shape[8];
    size_t axis;
    size_t output_rank;
    int keepdims;
    (void)kind;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 2, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, -1, 0,
                                &axis, error) ||
        vx_shape_boolean_default(request, "keepdims", 1, &keepdims, error)) {
        return -1;
    }
    if (axis != input->rank - 1) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.axis",
                             "must resolve to the last axis.");
    }
    output_rank = keepdims ? input->rank : input->rank - 1;
    if (output_rank) memcpy(output_shape, input->shape,
                            output_rank * sizeof(*output_shape));
    if (keepdims) output_shape[output_rank - 1] = 1;
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_arg_max(const VxConcreteShapeRequest* request,
                                  VxConcreteShapeResult* candidate,
                                  VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis", "keepdims", "select_last_index"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeParam* select_last;
    uint64_t output_shape[8];
    size_t axis;
    size_t input_axis;
    size_t output_rank = 0;
    int keepdims;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 3, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 0, 0,
                                &axis, error) ||
        vx_shape_boolean_default(request, "keepdims", 1, &keepdims, error)) {
        return -1;
    }
    select_last = vx_shape_find_param(request, "select_last_index");
    if (select_last &&
        (select_last->kind != VX_SHAPE_PARAM_NUMBER ||
         select_last->value.number != 0.0)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.select_last_index",
                             "must be 0 (first-index ties).");
    }
    for (input_axis = 0; input_axis < input->rank; ++input_axis) {
        if (input_axis == axis) {
            if (keepdims) output_shape[output_rank++] = 1;
        } else {
            output_shape[output_rank++] = input->shape[input_axis];
        }
    }
    return vx_shape_clone_output(candidate, output_shape, output_rank,
                                 VX_DTYPE_I32, NULL, error);
}

static int vx_shape_infer_transpose(const VxConcreteShapeRequest* request,
                                    VxConcreteShapeResult* candidate,
                                    VxShapeContractError* error) {
    static const char* const FIELDS[] = {"perm"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const double* raw = NULL;
    size_t count = 0;
    uint64_t output_shape[8];
    size_t perm[8];
    size_t axis;
    VxShapeQuantization quantization = input->quantization;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        input->rank > 8 ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_number_array(request, "perm", 0, &raw, &count, error)) {
        if (input->rank > 8 && error && error->code == VX_SHAPE_CONTRACT_ERROR_NONE) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                                 "operator input 'input'.shape",
                                 "must have rank at most 8.");
        }
        return -1;
    }
    if (!raw) count = input->rank;
    if (count != input->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.perm", "must be a permutation.");
    }
    for (axis = 0; axis < input->rank; ++axis) {
        size_t other;
        int64_t value = raw ? (int64_t)raw[axis]
                            : (int64_t)(input->rank - axis - 1);
        if (raw && (!vx_shape_is_safe_integer(raw[axis]) || value < 0 ||
                    (size_t)value >= input->rank)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.perm", "must be a permutation.");
        }
        perm[axis] = (size_t)value;
        for (other = 0; other < axis; ++other) {
            if (perm[other] == perm[axis]) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     "operator params.perm", "must be a permutation.");
            }
        }
        output_shape[axis] = input->shape[perm[axis]];
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        for (axis = 0; axis < input->rank; ++axis) {
            if (perm[axis] == quantization.axis) {
                quantization.axis = axis;
                break;
            }
        }
    }
    return vx_shape_clone_output(candidate, output_shape, input->rank,
                                 input->dtype, &quantization, error);
}

static int vx_shape_infer_flatten(const VxConcreteShapeRequest* request,
                                  VxConcreteShapeResult* candidate,
                                  VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    uint64_t output_shape[2];
    size_t axis;
    VxShapeQuantization quantization;
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 1, 1,
                                &axis, error) ||
        vx_shape_product_range(input->shape, 0, axis,
                               "Flatten prefix product", &output_shape[0], error) ||
        vx_shape_product_range(input->shape, axis, input->rank,
                               "Flatten suffix product", &output_shape[1], error) ||
        vx_shape_reshape_quantization(input, output_shape, 2,
                                      &quantization, error)) return -1;
    return vx_shape_clone_output(candidate, output_shape, 2, input->dtype,
                                 &quantization, error);
}

static int vx_shape_normalize_axes(const double* raw,
                                   size_t count,
                                   size_t normalization_rank,
                                   const char* path,
                                   size_t* axes,
                                   VxShapeContractError* error) {
    size_t index;
    if (!raw || !count) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             path, "must be a non-empty array.");
    }
    for (index = 0; index < count; ++index) {
        int64_t value = 0;
        size_t other;
        char item_path[224];
        snprintf(item_path, sizeof(item_path), "%s[%zu]", path, index);
        if (vx_shape_param_signed_integer_value(raw[index], item_path,
                                                &value, error)) return -1;
        if (value < 0) value += (int64_t)normalization_rank;
        if (value < 0 || (size_t)value >= normalization_rank) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 item_path, "resolves outside the supported rank.");
        }
        axes[index] = (size_t)value;
        for (other = 0; other < index; ++other) {
            if (axes[other] == axes[index]) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     path, "must contain unique axes.");
            }
        }
    }
    return 0;
}

static int vx_shape_infer_squeeze(const VxConcreteShapeRequest* request,
                                  VxConcreteShapeResult* candidate,
                                  VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axes"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const double* raw;
    size_t count;
    size_t* axes = NULL;
    uint64_t* output_shape = NULL;
    size_t output_rank;
    size_t input_axis;
    size_t output_axis = 0;
    VxShapeQuantization quantization;
    int status = -1;
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_number_array(request, "axes", 1, &raw, &count, error)) {
        return -1;
    }
    if (count > input->rank || count > SIZE_MAX / sizeof(*axes)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.axes", "contains too many axes.");
    }
    axes = (size_t*)malloc(count * sizeof(*axes));
    if (!axes) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                    "operator params.axes", "allocation failed.");
    if (vx_shape_normalize_axes(raw, count, input->rank,
                                "operator params.axes", axes, error)) goto cleanup;
    output_rank = input->rank - count;
    if (output_rank) {
        output_shape = (uint64_t*)malloc(output_rank * sizeof(*output_shape));
        if (!output_shape) {
            vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                          "operator output 'out'.shape", "allocation failed.");
            goto cleanup;
        }
    }
    for (input_axis = 0; input_axis < input->rank; ++input_axis) {
        size_t index;
        int selected = 0;
        for (index = 0; index < count; ++index) selected |= axes[index] == input_axis;
        if (selected) {
            if (input->shape[input_axis] != 1) {
                char path[224];
                snprintf(path, sizeof(path), "operator input 'input'.shape[%zu]",
                         input_axis);
                vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                              path, "must be 1 to squeeze it.");
                goto cleanup;
            }
        } else {
            output_shape[output_axis++] = input->shape[input_axis];
        }
    }
    if (vx_shape_reshape_quantization(input, output_shape, output_rank,
                                      &quantization, error) ||
        vx_shape_clone_output(candidate, output_shape, output_rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    free(axes);
    return status;
}

static int vx_shape_infer_unsqueeze(const VxConcreteShapeRequest* request,
                                    VxConcreteShapeResult* candidate,
                                    VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axes"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const double* raw;
    size_t count;
    size_t output_rank;
    size_t* axes = NULL;
    uint64_t* output_shape = NULL;
    size_t output_axis;
    size_t input_axis = 0;
    VxShapeQuantization quantization;
    int status = -1;
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_number_array(request, "axes", 1, &raw, &count, error) ||
        input->rank > SIZE_MAX - count) return -1;
    output_rank = input->rank + count;
    if (count > SIZE_MAX / sizeof(*axes) ||
        output_rank > SIZE_MAX / sizeof(*output_shape)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator params.axes", "rank is too large.");
    }
    axes = (size_t*)malloc(count * sizeof(*axes));
    output_shape = (uint64_t*)malloc(output_rank * sizeof(*output_shape));
    if (!axes || !output_shape) {
        vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                      "operator output 'out'.shape", "allocation failed.");
        goto cleanup;
    }
    if (vx_shape_normalize_axes(raw, count, output_rank,
                                "operator params.axes", axes, error)) goto cleanup;
    for (output_axis = 0; output_axis < output_rank; ++output_axis) {
        size_t index;
        int inserted = 0;
        for (index = 0; index < count; ++index) inserted |= axes[index] == output_axis;
        output_shape[output_axis] = inserted ? 1 : input->shape[input_axis++];
    }
    if (vx_shape_reshape_quantization(input, output_shape, output_rank,
                                      &quantization, error) ||
        vx_shape_clone_output(candidate, output_shape, output_rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    free(axes);
    return status;
}

static int vx_shape_target(const VxConcreteShapeRequest* request,
                           const VxShapeTensorDescriptor* input,
                           uint64_t** output_shape,
                           size_t* output_rank,
                           VxShapeContractError* error) {
    static const char* const FIELDS[] = {"shape"};
    const double* raw;
    size_t count;
    size_t axis;
    uint64_t* shape = NULL;
    const VxShapeTensorDescriptor* declared = NULL;
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_number_array(request, "shape", 1, &raw, &count, error) ||
        vx_shape_optional_declared_output(request, &declared, error)) return -1;
    if (count > SIZE_MAX / sizeof(*shape)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator params.shape", "rank is too large.");
    }
    if (declared && declared->dtype != input->dtype) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "declared output 'out'.dtype",
                             "must equal the input dtype.");
    }
    if (declared && declared->rank != count) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "declared output 'out'.shape",
                             "has a different rank from params.shape.");
    }
    if (count) {
        shape = (uint64_t*)malloc(count * sizeof(*shape));
        if (!shape) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                         "operator params.shape", "allocation failed.");
    }
    for (axis = 0; axis < count; ++axis) {
        char path[224];
        snprintf(path, sizeof(path), "operator params.shape[%zu]", axis);
        if (vx_shape_param_integer_value(raw[axis], 1, path,
                                         &shape[axis], error)) {
            free(shape);
            return -1;
        }
        if (declared && declared->shape[axis] != shape[axis]) {
            snprintf(path, sizeof(path), "declared output 'out'.shape[%zu]", axis);
            free(shape);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 path, "must equal params.shape.");
        }
    }
    *output_shape = shape;
    *output_rank = count;
    return 0;
}

static int vx_shape_infer_reshape(const VxConcreteShapeRequest* request,
                                  VxConcreteShapeResult* candidate,
                                  VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    uint64_t* output_shape = NULL;
    size_t output_rank = 0;
    uint64_t input_elements;
    uint64_t output_elements;
    VxShapeQuantization quantization;
    int status = -1;
    if (vx_shape_target(request, input, &output_shape, &output_rank, error) ||
        vx_shape_product_range(input->shape, 0, input->rank, "Reshape input",
                               &input_elements, error) ||
        vx_shape_product_range(output_shape, 0, output_rank, "Reshape output",
                               &output_elements, error)) goto cleanup;
    if (input_elements != output_elements) {
        vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                      "operator params.shape",
                      "must preserve the exact element count.");
        goto cleanup;
    }
    if (vx_shape_reshape_quantization(input, output_shape, output_rank,
                                      &quantization, error) ||
        vx_shape_clone_output(candidate, output_shape, output_rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    return status;
}

static int vx_shape_infer_expand(const VxConcreteShapeRequest* request,
                                 VxConcreteShapeResult* candidate,
                                 VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    uint64_t* output_shape = NULL;
    size_t output_rank = 0;
    size_t offset;
    size_t axis;
    VxShapeQuantization quantization = input->quantization;
    int status = -1;
    if (vx_shape_target(request, input, &output_shape, &output_rank, error)) return -1;
    if (output_rank < input->rank || output_rank > 8) {
        vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                      "operator params.shape",
                      "must have rank between input rank and 8.");
        goto cleanup;
    }
    offset = output_rank - input->rank;
    for (axis = 0; axis < output_rank; ++axis) {
        uint64_t input_dimension = axis < offset ? 1 : input->shape[axis - offset];
        if (input_dimension != 1 && input_dimension != output_shape[axis]) {
            char path[224];
            snprintf(path, sizeof(path), "operator params.shape[%zu]", axis);
            vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                          path, "is not a valid broadcast target.");
            goto cleanup;
        }
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        size_t output_axis = quantization.axis + offset;
        if (input->shape[quantization.axis] != output_shape[output_axis]) {
            char path[224];
            snprintf(path, sizeof(path), "operator params.shape[%zu]", output_axis);
            vx_shape_fail(error,
                          VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
                          path, "must not expand the per-axis quantization extent.");
            goto cleanup;
        }
        quantization.axis = output_axis;
    }
    if (vx_shape_clone_output(candidate, output_shape, output_rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    return status;
}

static int vx_shape_infer_concat(const VxConcreteShapeRequest* request,
                                 VxConcreteShapeResult* candidate,
                                 VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* first = vx_shape_input_descriptor(request, "input0");
    uint64_t output_shape[8];
    size_t axis;
    size_t input_index;
    VxShapeQuantization quantization = first->quantization;
    float* scales = NULL;
    int32_t* zero_points = NULL;
    int status = -1;
    if (first->rank < 1 || first->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "must all have one equal rank in [1, 8].");
    }
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_normalize_axis(request, "axis", first->rank, 0, 0,
                                &axis, error)) return -1;
    memcpy(output_shape, first->shape, first->rank * sizeof(*output_shape));
    output_shape[axis] = 0;
    for (input_index = 0; input_index < request->input_count; ++input_index) {
        char name[64];
        const VxShapeTensorDescriptor* input;
        size_t dimension;
        snprintf(name, sizeof(name), "input%zu", input_index);
        input = vx_shape_input_descriptor(request, name);
        if (input->rank != first->rank) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                                 "operator inputs",
                                 "must all have one equal rank in [1, 8].");
        }
        if (input->dtype != first->dtype) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "operator inputs", "must all have the same dtype.");
        }
        for (dimension = 0; dimension < first->rank; ++dimension) {
            if (dimension != axis && input->shape[dimension] != first->shape[dimension]) {
                char path[224];
                snprintf(path, sizeof(path),
                         "operator input 'input%zu'.shape[%zu]",
                         input_index, dimension);
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                     path, "must match input0.");
            }
        }
        if (vx_shape_checked_add_u64(output_shape[axis], input->shape[axis],
                                     "Concat axis sum", &output_shape[axis],
                                     error)) return -1;
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_NONE) {
        for (input_index = 1; input_index < request->input_count; ++input_index) {
            char name[64];
            snprintf(name, sizeof(name), "input%zu", input_index);
            if (vx_shape_input_descriptor(request, name)->quantization.scheme !=
                VX_SHAPE_QUANTIZATION_NONE) {
                return vx_shape_fail(error,
                                     VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                     "operator inputs",
                                     "Concat inputs must have compatible affine metadata.");
            }
        }
    } else if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_TENSOR ||
               quantization.axis != axis) {
        for (input_index = 1; input_index < request->input_count; ++input_index) {
            char name[64];
            snprintf(name, sizeof(name), "input%zu", input_index);
            if (!vx_shape_quantization_equal(
                    &quantization,
                    &vx_shape_input_descriptor(request, name)->quantization)) {
                return vx_shape_fail(error,
                                     VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                     "operator inputs",
                                     "Concat inputs must have identical affine metadata unless concatenating their common per-axis dimension.");
            }
        }
    } else {
        size_t count = (size_t)output_shape[axis];
        size_t offset = 0;
        if ((uint64_t)count != output_shape[axis] ||
            count > SIZE_MAX / sizeof(*scales) ||
            count > SIZE_MAX / sizeof(*zero_points)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 "operator inputs", "affine metadata is too large.");
        }
        scales = (float*)malloc(count * sizeof(*scales));
        zero_points = (int32_t*)malloc(count * sizeof(*zero_points));
        if (!scales || !zero_points) {
            vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                          "operator inputs", "allocation failed.");
            goto cleanup;
        }
        for (input_index = 0; input_index < request->input_count; ++input_index) {
            char name[64];
            const VxShapeQuantization* current;
            snprintf(name, sizeof(name), "input%zu", input_index);
            current = &vx_shape_input_descriptor(request, name)->quantization;
            if (current->scheme != VX_SHAPE_QUANTIZATION_PER_AXIS ||
                current->axis != axis) {
                vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                              "operator inputs", "have incompatible per-axis metadata.");
                goto cleanup;
            }
            memcpy(scales + offset, current->scales,
                   current->count * sizeof(*scales));
            memcpy(zero_points + offset, current->zero_points,
                   current->count * sizeof(*zero_points));
            offset += current->count;
        }
        quantization.count = count;
        quantization.scales = scales;
        quantization.zero_points = zero_points;
    }
    if (vx_shape_clone_output(candidate, output_shape, first->rank,
                              first->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(zero_points);
    free(scales);
    return status;
}

static int vx_shape_infer_split(const VxConcreteShapeRequest* request,
                                VxConcreteShapeResult* candidate,
                                VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis", "split", "num_outputs"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeParam* split = vx_shape_find_param(request, "split");
    const VxShapeParam* count_param = vx_shape_find_param(request, "num_outputs");
    const double* raw = NULL;
    size_t count = 0;
    size_t axis;
    uint64_t* sizes = NULL;
    uint64_t sum = 0;
    size_t index;
    size_t offset = 0;
    int status = -1;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 3, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 0, 0,
                                &axis, error)) return -1;
    if (split && count_param) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params",
                             "must specify split or num_outputs, not both.");
    }
    if (split) {
        if (vx_shape_param_number_array(request, "split", 1,
                                        &raw, &count, error)) return -1;
        if (!count) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.split",
                                 "must be a non-empty array.");
        }
    } else {
        double raw_count = 0.0;
        if (!count_param ||
            vx_shape_number(count_param, "operator params.num_outputs",
                            &raw_count, error) ||
            !vx_shape_is_safe_integer(raw_count) || raw_count <= 0.0 ||
            raw_count > (double)SIZE_MAX) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.num_outputs",
                                 "must be a positive safe integer.");
        }
        count = (size_t)raw_count;
        if (input->shape[axis] % count) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator params.num_outputs",
                                 "must divide the axis extent.");
        }
    }
    if (count > SIZE_MAX / sizeof(*sizes)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator params.split", "output count is too large.");
    }
    sizes = (uint64_t*)malloc(count * sizeof(*sizes));
    if (!sizes) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                     "operator params.split", "allocation failed.");
    for (index = 0; index < count; ++index) {
        char path[224];
        if (raw) {
            snprintf(path, sizeof(path), "operator params.split[%zu]", index);
            if (vx_shape_param_integer_value(raw[index], 1, path,
                                             &sizes[index], error)) goto cleanup;
        } else {
            sizes[index] = input->shape[axis] / count;
        }
        if (vx_shape_checked_add_u64(sum, sizes[index], "Split size sum",
                                     &sum, error)) goto cleanup;
    }
    if (sum != input->shape[axis]) {
        vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                      "operator params.split",
                      "must sum to the input axis extent.");
        goto cleanup;
    }
    if (vx_shape_result_allocate(candidate, count, error)) goto cleanup;
    for (index = 0; index < count; ++index) {
        uint64_t output_shape[8];
        VxShapeQuantization quantization = input->quantization;
        char name[64];
        memcpy(output_shape, input->shape, input->rank * sizeof(*output_shape));
        output_shape[axis] = sizes[index];
        if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS &&
            quantization.axis == axis) {
            quantization.count = (size_t)sizes[index];
            quantization.scales = input->quantization.scales + offset;
            quantization.zero_points = input->quantization.zero_points + offset;
        }
        snprintf(name, sizeof(name), "out%zu", index);
        if (vx_shape_clone_output_at(candidate, index, name, output_shape,
                                     input->rank, input->dtype,
                                     &quantization, error)) {
            vx_shape_contract_result_clear(candidate);
            goto cleanup;
        }
        offset += (size_t)sizes[index];
    }
    status = 0;
cleanup:
    free(sizes);
    return status;
}

static int vx_shape_infer_slice(const VxConcreteShapeRequest* request,
                                VxConcreteShapeResult* candidate,
                                VxShapeContractError* error) {
    static const char* const FIELDS[] = {"starts", "ends", "axes", "steps"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const double* starts_raw;
    const double* ends_raw;
    const double* axes_raw = NULL;
    const double* steps_raw = NULL;
    size_t starts_count;
    size_t ends_count;
    size_t axes_count;
    size_t steps_count;
    size_t* axes = NULL;
    uint64_t output_shape[8];
    size_t starts[8] = {0};
    size_t ends[8] = {0};
    size_t steps[8] = {0};
    size_t index;
    VxShapeQuantization quantization = input->quantization;
    if (input->rank < 1 || input->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator input 'input'.shape",
                             "must have rank in [1, 8].");
    }
    if (vx_shape_validate_params(request, FIELDS, 4, error) ||
        vx_shape_param_number_array(request, "starts", 1, &starts_raw,
                                    &starts_count, error) ||
        vx_shape_param_number_array(request, "ends", 1, &ends_raw,
                                    &ends_count, error) ||
        vx_shape_param_number_array(request, "axes", 0, &axes_raw,
                                    &axes_count, error) ||
        vx_shape_param_number_array(request, "steps", 0, &steps_raw,
                                    &steps_count, error)) return -1;
    if (!starts_count || starts_count != ends_count ||
        (axes_raw && axes_count != starts_count) ||
        (steps_raw && steps_count != starts_count)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params",
                             "starts, ends, axes, and steps must have equal nonzero lengths.");
    }
    axes = (size_t*)malloc(starts_count * sizeof(*axes));
    if (!axes) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                    "operator params.axes", "allocation failed.");
    if (axes_raw) {
        if (vx_shape_normalize_axes(axes_raw, axes_count, input->rank,
                                    "operator params.axes", axes, error)) {
            free(axes);
            return -1;
        }
    } else {
        if (starts_count > input->rank) {
            free(axes);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.axes",
                                 "resolves outside the supported rank.");
        }
        for (index = 0; index < starts_count; ++index) axes[index] = index;
    }
    memcpy(output_shape, input->shape, input->rank * sizeof(*output_shape));
    for (index = 0; index < input->rank; ++index) {
        starts[index] = 0;
        ends[index] = (size_t)input->shape[index];
        steps[index] = 1;
    }
    for (index = 0; index < starts_count; ++index) {
        int64_t raw_start = 0;
        int64_t raw_end = 0;
        uint64_t step = 1;
        uint64_t extent = input->shape[axes[index]];
        int64_t start;
        int64_t end;
        char path[224];
        snprintf(path, sizeof(path), "operator params.starts[%zu]", index);
        if (vx_shape_param_signed_integer_value(starts_raw[index], path,
                                                &raw_start, error)) goto slice_fail;
        snprintf(path, sizeof(path), "operator params.ends[%zu]", index);
        if (vx_shape_param_signed_integer_value(ends_raw[index], path,
                                                &raw_end, error)) goto slice_fail;
        if (steps_raw) {
            snprintf(path, sizeof(path), "operator params.steps[%zu]", index);
            if (vx_shape_param_integer_value(steps_raw[index], 1, path,
                                             &step, error)) goto slice_fail;
        }
        start = raw_start < 0 ? raw_start + (int64_t)extent : raw_start;
        end = raw_end < 0 ? raw_end + (int64_t)extent : raw_end;
        if (start < 0) start = 0;
        if (end < 0) end = 0;
        if ((uint64_t)start > extent) start = (int64_t)extent;
        if ((uint64_t)end > extent) end = (int64_t)extent;
        if (end <= start) {
            snprintf(path, sizeof(path), "operator params.ends[%zu]", index);
            vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                          path, "selects an empty axis, which v1 forbids.");
            goto slice_fail;
        }
        starts[axes[index]] = (size_t)start;
        ends[axes[index]] = (size_t)end;
        steps[axes[index]] = (size_t)step;
        output_shape[axes[index]] =
            ((uint64_t)(end - start) - 1) / step + 1;
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        size_t qaxis = quantization.axis;
        size_t count = 0;
        size_t position;
        for (position = starts[qaxis]; position < ends[qaxis];
             position += steps[qaxis]) count++;
        quantization.count = count;
        if (steps[qaxis] == 1) {
            quantization.scales += starts[qaxis];
            quantization.zero_points += starts[qaxis];
        } else {
            float* gathered_scales = (float*)malloc(count * sizeof(float));
            int32_t* gathered_zeros = (int32_t*)malloc(count * sizeof(int32_t));
            size_t out = 0;
            int clone_status;
            if (!gathered_scales || !gathered_zeros) {
                free(gathered_scales);
                free(gathered_zeros);
                vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                              "operator input 'input'.quantization",
                              "allocation failed.");
                goto slice_fail;
            }
            for (position = starts[qaxis]; position < ends[qaxis];
                 position += steps[qaxis]) {
                gathered_scales[out] = input->quantization.scales[position];
                gathered_zeros[out++] = input->quantization.zero_points[position];
            }
            quantization.scales = gathered_scales;
            quantization.zero_points = gathered_zeros;
            clone_status = vx_shape_clone_output(candidate, output_shape, input->rank,
                                                 input->dtype, &quantization, error);
            free(gathered_scales);
            free(gathered_zeros);
            free(axes);
            return clone_status;
        }
    }
    free(axes);
    return vx_shape_clone_output(candidate, output_shape, input->rank,
                                 input->dtype, &quantization, error);
slice_fail:
    free(axes);
    return -1;
}

static int vx_shape_infer_pad(const VxConcreteShapeRequest* request,
                              VxConcreteShapeResult* candidate,
                              VxShapeContractError* error) {
    static const char* const FIELDS[] = {"pads", "value"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeParam* value_param;
    const double* raw;
    size_t count;
    uint64_t* output_shape = NULL;
    size_t axis;
    VxShapeQuantization quantization = input->quantization;
    int status = -1;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 2, error) ||
        vx_shape_param_number_array(request, "pads", 1, &raw, &count, error)) {
        return -1;
    }
    if (count != input->rank * 2) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.pads",
                             "must have exactly twice the input rank entries.");
    }
    value_param = vx_shape_find_param(request, "value");
    if (value_param &&
        (value_param->kind != VX_SHAPE_PARAM_NUMBER ||
         !isfinite(value_param->value.number))) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.value", "must be finite.");
    }
    output_shape = (uint64_t*)malloc(input->rank * sizeof(*output_shape));
    if (!output_shape) return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                            "operator output 'out'.shape",
                                            "allocation failed.");
    for (axis = 0; axis < input->rank; ++axis) {
        uint64_t before = 0;
        uint64_t after = 0;
        char path[224];
        snprintf(path, sizeof(path), "operator params.pads[%zu]", axis);
        if (vx_shape_param_integer_value(raw[axis], 0, path, &before, error)) goto cleanup;
        snprintf(path, sizeof(path), "operator params.pads[%zu]", input->rank + axis);
        if (vx_shape_param_integer_value(raw[input->rank + axis], 0, path,
                                         &after, error) ||
            vx_shape_checked_add_u64(input->shape[axis], before,
                                     "Pad axis", &output_shape[axis], error) ||
            vx_shape_checked_add_u64(output_shape[axis], after,
                                     "Pad axis", &output_shape[axis], error)) goto cleanup;
        if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS &&
            quantization.axis == axis && (before || after)) {
            snprintf(path, sizeof(path), "operator params.pads[%zu]", axis);
            vx_shape_fail(error,
                          VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
                          path, "must not extend a per-axis quantization dimension.");
            goto cleanup;
        }
    }
    if (vx_shape_clone_output(candidate, output_shape, input->rank,
                              input->dtype, &quantization, error)) goto cleanup;
    status = 0;
cleanup:
    free(output_shape);
    return status;
}

static int vx_shape_indices(const VxShapeTensorDescriptor* indices,
                            VxShapeContractError* error) {
    if (indices->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'indices'.dtype", "must be int32.");
    }
    return vx_shape_unquantized(indices, "operator input 'indices'", error);
}

static int vx_shape_infer_gather(const VxConcreteShapeRequest* request,
                                 VxConcreteShapeResult* candidate,
                                 VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* indices = vx_shape_input_descriptor(request, "indices");
    uint64_t* output_shape;
    size_t output_rank;
    size_t axis;
    size_t output_axis = 0;
    size_t input_axis;
    VxShapeQuantization quantization = input->quantization;
    int status;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_indices(indices, error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 0, 0,
                                &axis, error)) return -1;
    if (input->rank - 1 > SIZE_MAX - indices->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "rank is too large.");
    }
    output_rank = input->rank - 1 + indices->rank;
    output_shape = output_rank
        ? (uint64_t*)malloc(output_rank * sizeof(*output_shape)) : NULL;
    if (output_rank && !output_shape) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "allocation failed.");
    }
    for (input_axis = 0; input_axis < axis; ++input_axis)
        output_shape[output_axis++] = input->shape[input_axis];
    for (input_axis = 0; input_axis < indices->rank; ++input_axis)
        output_shape[output_axis++] = indices->shape[input_axis];
    for (input_axis = axis + 1; input_axis < input->rank; ++input_axis)
        output_shape[output_axis++] = input->shape[input_axis];
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        if (quantization.axis == axis) {
            free(output_shape);
            return vx_shape_fail(
                error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
                "operator input 'input'.quantization.axis",
                "Gather indices would reorder the per-axis affine metadata.");
        }
        if (quantization.axis > axis) {
            if (indices->rank == 0) quantization.axis--;
            else quantization.axis += indices->rank - 1;
        }
    }
    status = vx_shape_clone_output(candidate, output_shape, output_rank,
                                   input->dtype, &quantization, error);
    free(output_shape);
    return status;
}

static int vx_shape_infer_gather_elements(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* input = vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* indices = vx_shape_input_descriptor(request, "indices");
    size_t axis;
    size_t dimension;
    VxShapeQuantization quantization = input->quantization;
    if (vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
        vx_shape_indices(indices, error)) return -1;
    if (indices->rank != input->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator input 'indices'.shape",
                             "must have the same rank as input.");
    }
    if (vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_normalize_axis(request, "axis", input->rank, 0, 0,
                                &axis, error)) return -1;
    for (dimension = 0; dimension < input->rank; ++dimension) {
        if (dimension != axis && indices->shape[dimension] > input->shape[dimension]) {
            char path[224];
            snprintf(path, sizeof(path), "operator input 'indices'.shape[%zu]",
                     dimension);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 path,
                                 "must not exceed the corresponding input dimension.");
        }
    }
    if (quantization.scheme == VX_SHAPE_QUANTIZATION_PER_AXIS) {
        if (quantization.axis == axis) {
            return vx_shape_fail(
                error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
                "operator input 'input'.quantization.axis",
                "GatherElements indices would reorder the per-axis affine metadata.");
        }
        if (indices->shape[quantization.axis] > quantization.count) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input indices",
                                 "exceeds the per-axis input extent.");
        }
        quantization.count = (size_t)indices->shape[quantization.axis];
    }
    return vx_shape_clone_output(candidate, indices->shape, indices->rank,
                                 input->dtype, &quantization, error);
}

static int vx_shape_infer_non_spatial_wave_b_kind(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    switch (kind) {
        case VX_OP_SUB:
        case VX_OP_DIV:
            return vx_shape_infer_broadcast_arithmetic(request, candidate, error);
        case VX_OP_EQUAL:
        case VX_OP_GREATER_OR_EQUAL:
            return vx_shape_infer_broadcast_comparison(request, candidate, error);
        case VX_OP_WHERE:
            return vx_shape_infer_where(request, candidate, error);
        case VX_OP_REDUCE_SUM:
        case VX_OP_REDUCE_MEAN:
            return vx_shape_infer_reduction(kind, request, candidate, error);
        case VX_OP_ARG_MAX:
            return vx_shape_infer_arg_max(request, candidate, error);
        case VX_OP_TRANSPOSE:
            return vx_shape_infer_transpose(request, candidate, error);
        case VX_OP_FLATTEN:
            return vx_shape_infer_flatten(request, candidate, error);
        case VX_OP_SQUEEZE:
            return vx_shape_infer_squeeze(request, candidate, error);
        case VX_OP_UNSQUEEZE:
            return vx_shape_infer_unsqueeze(request, candidate, error);
        case VX_OP_RESHAPE:
            return vx_shape_infer_reshape(request, candidate, error);
        case VX_OP_EXPAND:
            return vx_shape_infer_expand(request, candidate, error);
        case VX_OP_CONCAT:
            return vx_shape_infer_concat(request, candidate, error);
        case VX_OP_SPLIT:
            return vx_shape_infer_split(request, candidate, error);
        case VX_OP_SLICE:
            return vx_shape_infer_slice(request, candidate, error);
        case VX_OP_PAD:
            return vx_shape_infer_pad(request, candidate, error);
        case VX_OP_GATHER:
            return vx_shape_infer_gather(request, candidate, error);
        case VX_OP_GATHER_ELEMENTS:
            return vx_shape_infer_gather_elements(request, candidate, error);
        default:
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                                 "operator",
                                 "has no implemented non-spatial Wave-B contract.");
    }
}

static int vx_shape_attention_parameters(
        const VxConcreteShapeRequest* request,
        uint64_t* heads,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"heads", "causal", "scale"};
    const VxShapeParam* param;
    double value;
    float canonical;
    if (vx_shape_validate_params(request, FIELDS, 3, error)) return -1;
    param = vx_shape_find_param(request, "heads");
    if (!param || param->kind != VX_SHAPE_PARAM_NUMBER ||
        !vx_shape_is_safe_integer(param->value.number) ||
        param->value.number <= 0.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.heads",
                             "must be a positive safe integer.");
    }
    *heads = (uint64_t)param->value.number;
    param = vx_shape_find_param(request, "causal");
    if (!param || param->kind != VX_SHAPE_PARAM_BOOLEAN) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.causal", "must be boolean.");
    }
    param = vx_shape_find_param(request, "scale");
    if (!param) return 0;
    if (param->kind != VX_SHAPE_PARAM_NUMBER) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.scale",
                             "must be positive and representable as float32.");
    }
    value = param->value.number;
    canonical = (float)value;
    if (!isfinite(value) || value <= 0.0 ||
        !isfinite(canonical) || canonical <= 0.0f) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.scale",
                             "must be positive and representable as float32.");
    }
    return 0;
}

static int vx_shape_attention_rank(
        const VxShapeTensorDescriptor* descriptor,
        const char* path,
        VxShapeContractError* error) {
    if (descriptor->rank != 2 && descriptor->rank != 3) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             path, "must have rank 2 or 3.");
    }
    return 0;
}

static int vx_shape_attention_mask(
        const VxShapeTensorDescriptor* mask,
        uint64_t batch,
        uint64_t queries,
        uint64_t keys,
        VxShapeContractError* error) {
    int valid;
    if (!mask) return 0;
    if (mask->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'mask'.dtype",
                             "must be 'int32'.");
    }
    if (vx_shape_unquantized(mask, "operator input 'mask'", error)) return -1;
    valid = (mask->rank == 1 && mask->shape[0] == keys) ||
            (mask->rank == 2 && mask->shape[1] == keys &&
             (mask->shape[0] == batch || mask->shape[0] == queries)) ||
            (mask->rank == 3 && mask->shape[0] == batch &&
             mask->shape[1] == queries && mask->shape[2] == keys);
    if (!valid) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'mask'.shape",
                             "must have shape [K], [B,K], [Q,K], or [B,Q,K].");
    }
    return 0;
}

static int vx_shape_infer_sdpa(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* qkv =
        vx_shape_input_descriptor(request, "qkv");
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    uint64_t output_shape[3];
    uint64_t packed_feature;
    uint64_t feature;
    uint64_t sequence;
    uint64_t batch;
    uint64_t heads;
    size_t axis;
    if (vx_shape_float_tensor(qkv, "operator input 'qkv'", error) ||
        vx_shape_attention_rank(qkv, "operator input 'qkv'.shape", error)) {
        return -1;
    }
    packed_feature = qkv->shape[qkv->rank - 1];
    if (packed_feature % 3) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'qkv'.shape[%zu]",
                 qkv->rank - 1);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path,
                             "must be exactly 3 times the output feature extent.");
    }
    feature = packed_feature / 3;
    if (vx_shape_attention_parameters(request, &heads, error)) return -1;
    if (feature % heads) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.heads",
                             "must divide the output feature extent.");
    }
    sequence = qkv->shape[qkv->rank - 2];
    batch = qkv->rank == 2 ? 1 : qkv->shape[0];
    if (vx_shape_attention_mask(mask, batch, sequence, sequence, error)) return -1;
    for (axis = 0; axis < qkv->rank; ++axis) output_shape[axis] = qkv->shape[axis];
    output_shape[qkv->rank - 1] = feature;
    return vx_shape_clone_output(candidate, output_shape, qkv->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_cross_sdpa(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* query =
        vx_shape_input_descriptor(request, "q");
    const VxShapeTensorDescriptor* key =
        vx_shape_input_descriptor(request, "k");
    const VxShapeTensorDescriptor* value =
        vx_shape_input_descriptor(request, "v");
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    size_t sequence_axis;
    size_t feature_axis;
    uint64_t batch;
    uint64_t heads;
    uint64_t feature;
    if (vx_shape_float_tensor(query, "operator input 'q'", error) ||
        vx_shape_float_tensor(key, "operator input 'k'", error) ||
        vx_shape_float_tensor(value, "operator input 'v'", error) ||
        vx_shape_attention_rank(query, "operator input 'q'.shape", error) ||
        vx_shape_attention_rank(key, "operator input 'k'.shape", error) ||
        vx_shape_attention_rank(value, "operator input 'v'.shape", error)) {
        return -1;
    }
    if (key->rank != query->rank || value->rank != query->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "q, k, and v must have the same rank.");
    }
    sequence_axis = query->rank - 2;
    feature_axis = query->rank - 1;
    if (query->rank == 3 &&
        (key->shape[0] != query->shape[0] ||
         value->shape[0] != query->shape[0])) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "q, k, and v batch extents must match.");
    }
    if (value->shape[sequence_axis] != key->shape[sequence_axis]) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'v'.shape[%zu]",
                 sequence_axis);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path, "must equal the key sequence extent.");
    }
    if (key->shape[feature_axis] != query->shape[feature_axis] ||
        value->shape[feature_axis] != query->shape[feature_axis]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "q, k, and v feature extents must match.");
    }
    feature = query->shape[feature_axis];
    if (vx_shape_attention_parameters(request, &heads, error)) return -1;
    if (feature % heads) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.heads",
                             "must divide the feature extent.");
    }
    batch = query->rank == 2 ? 1 : query->shape[0];
    if (vx_shape_attention_mask(mask, batch, query->shape[sequence_axis],
                                key->shape[sequence_axis], error)) return -1;
    return vx_shape_clone_output(candidate, query->shape, query->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_rope_parameters(
        const VxConcreteShapeRequest* request,
        uint64_t width,
        uint64_t sequence,
        uint64_t* rotary_dimension,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "rotary_dim", "theta", "position_offset", "interleaved"
    };
    const VxShapeParam* param;
    uint64_t position_offset = 0;
    double value;
    float canonical;
    if (vx_shape_validate_params(request, FIELDS, 4, error)) return -1;
    *rotary_dimension = width;
    param = vx_shape_find_param(request, "rotary_dim");
    if (param) {
        if (param->kind != VX_SHAPE_PARAM_NUMBER ||
            !vx_shape_is_safe_integer(param->value.number) ||
            param->value.number <= 0.0 ||
            ((uint64_t)param->value.number % 2)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.rotary_dim",
                                 "must be a positive even safe integer.");
        }
        *rotary_dimension = (uint64_t)param->value.number;
    }
    if (*rotary_dimension % 2) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.rotary_dim",
                             "resolved rotary width must be even.");
    }
    if (*rotary_dimension > width) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.rotary_dim",
                             "must not exceed the feature extent.");
    }
    param = vx_shape_find_param(request, "theta");
    if (param && param->kind != VX_SHAPE_PARAM_NUMBER) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.theta",
                             "must be positive and representable as float32.");
    }
    value = param ? param->value.number : 10000.0;
    if (!isfinite(value) || value <= 0.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.theta",
                             "must be positive and representable as float32.");
    }
    canonical = (float)value;
    if (!isfinite(canonical) || canonical <= 0.0f) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.theta",
                             "must be positive and representable as float32.");
    }
    param = vx_shape_find_param(request, "position_offset");
    if (param) {
        if (param->kind != VX_SHAPE_PARAM_NUMBER ||
            !vx_shape_is_safe_integer(param->value.number) ||
            param->value.number < 0.0 ||
            param->value.number > (double)INT32_MAX) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.position_offset",
                                 "must be a non-negative int32 integer.");
        }
        position_offset = (uint64_t)param->value.number;
    }
    if (sequence - 1 > (uint64_t)INT32_MAX - position_offset) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.position_offset",
                             "plus the maximum sequence index must fit int32.");
    }
    param = vx_shape_find_param(request, "interleaved");
    if (param && param->kind != VX_SHAPE_PARAM_BOOLEAN) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.interleaved",
                             "must be boolean.");
    }
    return 0;
}

static int vx_shape_position_ids(
        const VxShapeTensorDescriptor* positions,
        size_t rank,
        uint64_t batch,
        uint64_t sequence,
        VxShapeContractError* error) {
    int valid;
    if (!positions) return 0;
    if (positions->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'position_ids'.dtype",
                             "must be 'int32'.");
    }
    if (vx_shape_unquantized(positions, "operator input 'position_ids'", error)) {
        return -1;
    }
    valid = (positions->rank == 1 && positions->shape[0] == sequence) ||
            (rank == 3 && positions->rank == 2 &&
             positions->shape[0] == batch && positions->shape[1] == sequence);
    if (!valid) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'position_ids'.shape",
                             "must have shape [S], or [B,S] for a rank-3 input.");
    }
    return 0;
}

static int vx_shape_infer_rope(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* positions =
        vx_shape_input_descriptor(request, "position_ids");
    uint64_t sequence;
    uint64_t batch;
    uint64_t rotary_dimension;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_attention_rank(input, "operator input 'input'.shape", error)) {
        return -1;
    }
    sequence = input->shape[input->rank - 2];
    batch = input->rank == 2 ? 1 : input->shape[0];
    if (vx_shape_rope_parameters(request, input->shape[input->rank - 1],
                                 sequence, &rotary_dimension, error) ||
        vx_shape_position_ids(positions, input->rank, batch, sequence, error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_attention_kind(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    switch (kind) {
        case VX_OP_SDPA:
            return vx_shape_infer_sdpa(request, candidate, error);
        case VX_OP_CROSS_SDPA:
            return vx_shape_infer_cross_sdpa(request, candidate, error);
        case VX_OP_ROPE:
            return vx_shape_infer_rope(request, candidate, error);
        default:
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                                 "operator",
                                 "has no implemented attention-family contract.");
    }
}

static int vx_shape_quantized_kind(VxOperatorKind kind) {
    switch (kind) {
        case VX_OP_Q_LINEAR:
        case VX_OP_Q_MATMUL:
        case VX_OP_Q_GEMM:
        case VX_OP_Q_BATCH_MATMUL:
        case VX_OP_Q_CONV_2D:
        case VX_OP_Q_ADD:
        case VX_OP_Q_EMBEDDING:
        case VX_OP_Q_GELU:
        case VX_OP_Q_SILU:
        case VX_OP_Q_LAYER_NORM:
        case VX_OP_Q_GROUP_NORM:
        case VX_OP_Q_MASKED_MEAN:
        case VX_OP_Q_SDPA:
        case VX_OP_Q_ARG_MAX:
            return 1;
        default:
            return 0;
    }
}

static int vx_shape_per_tensor_byte(
        const VxShapeTensorDescriptor* descriptor,
        const char* path,
        const VxShapeQuantization** quantization,
        VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (descriptor->dtype != VX_DTYPE_I8 && descriptor->dtype != VX_DTYPE_U8) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "dtype");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             field_path, "must be 'int8' or 'uint8'.");
    }
    if (descriptor->quantization.scheme != VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "quantization");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path, "must be per_tensor.");
    }
    if (quantization) *quantization = &descriptor->quantization;
    return 0;
}

static int vx_shape_axis_zero_byte_weight(
        const VxShapeTensorDescriptor* descriptor,
        const char* path,
        const VxShapeQuantization** quantization,
        VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (descriptor->dtype != VX_DTYPE_I8 && descriptor->dtype != VX_DTYPE_U8) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "dtype");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             field_path, "must be 'int8' or 'uint8'.");
    }
    if (descriptor->quantization.scheme != VX_SHAPE_QUANTIZATION_PER_AXIS ||
        descriptor->quantization.axis != 0) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "quantization");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                             field_path,
                             "must be per_axis along output/row axis 0.");
    }
    if (quantization) *quantization = &descriptor->quantization;
    return 0;
}

static int vx_shape_i32_tensor(const VxShapeTensorDescriptor* descriptor,
                               const char* path,
                               VxShapeContractError* error) {
    char field_path[224];
    if (!descriptor) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                             path, "must be an object.");
    }
    if (descriptor->dtype != VX_DTYPE_I32) {
        vx_shape_field_path(field_path, sizeof(field_path), path, "dtype");
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             field_path, "must be 'int32'.");
    }
    return vx_shape_unquantized(descriptor, path, error);
}

static int vx_shape_quantized_declared_output(
        const VxConcreteShapeRequest* request,
        const uint64_t* shape,
        size_t rank,
        const VxShapeTensorDescriptor** output,
        VxShapeContractError* error) {
    VxShapeTensorDescriptor expected;
    if (vx_shape_declared_output(request, output, error) ||
        vx_shape_per_tensor_byte(*output, "declared output 'out'", NULL,
                                 error)) {
        return -1;
    }
    memset(&expected, 0, sizeof(expected));
    expected.rank = rank;
    expected.shape = shape;
    if (!vx_shape_same(&expected, *output)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "declared output 'out'.shape",
                             "must equal the inferred output shape.");
    }
    return 0;
}

static uint64_t vx_shape_centered_magnitude(VxDataType dtype,
                                            int32_t zero_point) {
    int32_t minimum = dtype == VX_DTYPE_I8 ? -128 : 0;
    int32_t maximum = dtype == VX_DTYPE_I8 ? 127 : 255;
    uint64_t below = (uint64_t)((int64_t)zero_point - minimum);
    uint64_t above = (uint64_t)((int64_t)maximum - zero_point);
    return below > above ? below : above;
}

static uint64_t vx_shape_maximum_weight_magnitude(
        VxDataType dtype,
        const VxShapeQuantization* quantization) {
    uint64_t maximum = 0;
    size_t index;
    for (index = 0; index < quantization->count; ++index) {
        uint64_t magnitude = vx_shape_centered_magnitude(
            dtype, quantization->zero_points[index]);
        if (magnitude > maximum) maximum = magnitude;
    }
    return maximum;
}

static int vx_shape_i32_accumulator_bound(uint64_t terms,
                                          uint64_t left_magnitude,
                                          uint64_t right_magnitude,
                                          const char* path,
                                          VxShapeContractError* error) {
    uint64_t maximum = 0;
    if (vx_shape_checked_multiply_u64(terms, left_magnitude, path,
                                      &maximum, error) ||
        vx_shape_checked_multiply_u64(maximum, right_magnitude, path,
                                      &maximum, error)) {
        return -1;
    }
    if (maximum > (uint64_t)INT32_MAX) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DOMAIN,
                             path,
                             "may require an I32 accumulator magnitude of %llu.",
                             (unsigned long long)maximum);
    }
    return 0;
}

static int vx_shape_i32_sum_bound(uint64_t terms,
                                  uint64_t magnitude,
                                  const char* path,
                                  VxShapeContractError* error) {
    uint64_t maximum = 0;
    if (vx_shape_checked_multiply_u64(terms, magnitude, path,
                                      &maximum, error)) return -1;
    if (maximum > (uint64_t)INT32_MAX) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DOMAIN,
                             path,
                             "may require an I32 accumulator magnitude of %llu.",
                             (unsigned long long)maximum);
    }
    return 0;
}

static int vx_shape_positive_f32_ratio(float left,
                                       float right,
                                       float divisor,
                                       const char* path,
                                       VxShapeContractError* error) {
    float product = left * right;
    float multiplier = product / divisor;
    if (!isfinite(multiplier) || multiplier <= 0.0f) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
            path,
            "requantization multiplier must be positive and representable as float32.");
    }
    return 0;
}

static int vx_shape_validate_q_dense_multipliers(
        const VxShapeQuantization* input,
        const VxShapeQuantization* weight,
        const VxShapeQuantization* output,
        VxShapeContractError* error) {
    size_t index;
    char path[256];
    for (index = 0; index < weight->count; ++index) {
        snprintf(path, sizeof(path),
                 "operator input 'weight'.quantization.scales[%zu]", index);
        if (vx_shape_positive_f32_ratio(input->scale, weight->scales[index],
                                        output->scale, path, error)) return -1;
    }
    return 0;
}

static int vx_shape_infer_q_dense(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias =
        vx_shape_input_descriptor(request, "bias");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* input_quantization;
    const VxShapeQuantization* weight_quantization;
    uint64_t* output_shape;
    uint64_t output_feature;
    uint64_t contracted;
    size_t rank;
    int status;
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_rank(input, 1, 0, 0,
                      "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 2, 1,
                      "operator input 'weight'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'",
                                 &input_quantization, error) ||
        vx_shape_axis_zero_byte_weight(weight, "operator input 'weight'",
                                       &weight_quantization, error) ||
        vx_shape_i32_tensor(bias, "operator input 'bias'", error)) {
        return -1;
    }
    output_feature = weight->shape[0];
    contracted = weight->shape[1];
    if (input->shape[input->rank - 1] != contracted) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'input'.shape[%zu]",
                 input->rank - 1);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path,
                             "must equal the fixed contracted weight extent.");
    }
    if (vx_shape_vector(bias, output_feature,
                        "operator input 'bias'.shape", error)) return -1;
    rank = input->rank;
    if (rank > SIZE_MAX / sizeof(*output_shape)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "rank is too large.");
    }
    output_shape = (uint64_t*)malloc(rank * sizeof(*output_shape));
    if (!output_shape) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "allocation failed.");
    }
    memcpy(output_shape, input->shape, rank * sizeof(*output_shape));
    output_shape[rank - 1] = output_feature;
    status = vx_shape_quantized_declared_output(request, output_shape, rank,
                                                 &output, error);
    if (!status) {
        status = vx_shape_validate_q_dense_multipliers(
            input_quantization, weight_quantization, &output->quantization,
            error);
    }
    if (!status) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'input'.shape[%zu]",
                 input->rank - 1);
        status = vx_shape_i32_accumulator_bound(
            contracted,
            vx_shape_centered_magnitude(input->dtype,
                                        input_quantization->zero_point),
            vx_shape_maximum_weight_magnitude(weight->dtype,
                                               weight_quantization),
            path, error);
    }
    if (!status) {
        status = vx_shape_clone_output(candidate, output_shape, rank,
                                       output->dtype, &output->quantization,
                                       error);
    }
    free(output_shape);
    return status;
}

static int vx_shape_infer_q_batch_matmul(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* left =
        vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right =
        vx_shape_input_descriptor(request, "b");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* left_quantization;
    const VxShapeQuantization* right_quantization;
    uint64_t output_shape[8];
    size_t left_batch;
    size_t right_batch;
    size_t batch_rank;
    size_t axis;
    uint64_t contracted;
    char accumulator_path[224];
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_rank(left, 2, 0, 0, "operator input 'a'.shape", error) ||
        vx_shape_rank(right, 2, 0, 0, "operator input 'b'.shape", error) ||
        vx_shape_per_tensor_byte(left, "operator input 'a'",
                                 &left_quantization, error) ||
        vx_shape_per_tensor_byte(right, "operator input 'b'",
                                 &right_quantization, error)) return -1;
    if (left->rank > 8 || right->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "QBatchMatMul operand ranks must not exceed 8.");
    }
    contracted = left->shape[left->rank - 1];
    snprintf(accumulator_path, sizeof(accumulator_path),
             "operator input 'a'.shape[%zu]", left->rank - 1);
    if (contracted != right->shape[right->rank - 2]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'b'.shape",
                             "contracted matrix dimensions must match.");
    }
    left_batch = left->rank - 2;
    right_batch = right->rank - 2;
    batch_rank = left_batch > right_batch ? left_batch : right_batch;
    for (axis = 0; axis < batch_rank; ++axis) {
        uint64_t left_extent = axis < batch_rank - left_batch
            ? 1 : left->shape[axis - (batch_rank - left_batch)];
        uint64_t right_extent = axis < batch_rank - right_batch
            ? 1 : right->shape[axis - (batch_rank - right_batch)];
        if (left_extent != right_extent && left_extent != 1 && right_extent != 1) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator inputs",
                                 "batch dimensions are not broadcast-compatible.");
        }
        output_shape[axis] = left_extent > right_extent
            ? left_extent : right_extent;
    }
    output_shape[batch_rank] = left->shape[left->rank - 2];
    output_shape[batch_rank + 1] = right->shape[right->rank - 1];
    if (vx_shape_quantized_declared_output(request, output_shape,
                                           batch_rank + 2, &output, error) ||
        vx_shape_positive_f32_ratio(
            left_quantization->scale, right_quantization->scale,
            output->quantization.scale,
            "declared output 'out'.quantization.scale", error) ||
        vx_shape_i32_accumulator_bound(
            contracted,
            vx_shape_centered_magnitude(left->dtype,
                                        left_quantization->zero_point),
            vx_shape_centered_magnitude(right->dtype,
                                        right_quantization->zero_point),
            accumulator_path, error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, output_shape, batch_rank + 2,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_conv2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "stride", "padding", "pads", "dilation", "groups", "relu",
        "data_layout", "weight_layout"
    };
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias =
        vx_shape_input_descriptor(request, "bias");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* input_quantization;
    const VxShapeQuantization* weight_quantization;
    const char* data_layout;
    const char* weight_layout;
    uint64_t stride[2];
    uint64_t dilation[2];
    uint64_t pads[4];
    uint64_t groups;
    uint64_t activation;
    uint64_t output_shape[4];
    uint64_t terms = 0;
    if (vx_shape_rank(input, 0, 4, 1,
                      "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 4, 1,
                      "operator input 'weight'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'",
                                 &input_quantization, error) ||
        vx_shape_axis_zero_byte_weight(weight, "operator input 'weight'",
                                       &weight_quantization, error) ||
        vx_shape_validate_params(request, FIELDS,
                                 sizeof(FIELDS) / sizeof(FIELDS[0]), error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error) ||
        vx_shape_param_string_default(request, "weight_layout", "OHWI",
                                      &weight_layout, error) ||
        vx_shape_param_pair(request, "stride", 1, 1, 0, stride, error) ||
        vx_shape_spatial_pads(request, 0, pads, error) ||
        vx_shape_param_pair(request, "dilation", 1, 1, 0, dilation, error) ||
        vx_shape_param_scalar(request, "groups", 1, 1, 0, &groups, error) ||
        vx_shape_param_activation(request, &activation, error)) return -1;
    (void)activation;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    if (strcmp(weight_layout, "OHWI")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.weight_layout", "must be 'OHWI'.");
    }
    if (input->shape[3] != weight->shape[3] * groups ||
        weight->shape[0] % groups) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape",
                             "OHWI channels must match input channels and groups.");
    }
    if (bias &&
        (vx_shape_i32_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, weight->shape[0],
                         "operator input 'bias'.shape", error))) return -1;
    output_shape[0] = input->shape[0];
    if (vx_shape_window_output(input->shape[1], weight->shape[1], stride[0],
                               pads[0], pads[2], dilation[0],
                               "operator input 'input'.shape[1]",
                               &output_shape[1], error) ||
        vx_shape_window_output(input->shape[2], weight->shape[2], stride[1],
                               pads[1], pads[3], dilation[1],
                               "operator input 'input'.shape[2]",
                               &output_shape[2], error)) return -1;
    output_shape[3] = weight->shape[0];
    if (vx_shape_quantized_declared_output(request, output_shape, 4,
                                           &output, error) ||
        vx_shape_validate_q_dense_multipliers(
            input_quantization, weight_quantization, &output->quantization,
            error) ||
        vx_shape_checked_multiply_u64(weight->shape[1], weight->shape[2],
                                      "QConv2D accumulator terms", &terms,
                                      error) ||
        vx_shape_checked_multiply_u64(terms, weight->shape[3],
                                      "QConv2D accumulator terms", &terms,
                                      error) ||
        vx_shape_i32_accumulator_bound(
            terms,
            vx_shape_centered_magnitude(input->dtype,
                                        input_quantization->zero_point),
            vx_shape_maximum_weight_magnitude(weight->dtype,
                                               weight_quantization),
            "operator input 'weight'.shape", error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, output_shape, 4, output->dtype,
                                 &output->quantization, error);
}

static int vx_shape_infer_q_add(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"relu"};
    const VxShapeTensorDescriptor* left =
        vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right =
        vx_shape_input_descriptor(request, "b");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* left_quantization;
    const VxShapeQuantization* right_quantization;
    uint64_t activation;
    if (vx_shape_per_tensor_byte(left, "operator input 'a'",
                                 &left_quantization, error) ||
        vx_shape_per_tensor_byte(right, "operator input 'b'",
                                 &right_quantization, error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_param_activation(request, &activation, error)) return -1;
    (void)activation;
    if (!vx_shape_same(left, right)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "QAdd requires exactly equal shapes; Expand is explicit.");
    }
    if (vx_shape_quantized_declared_output(request, left->shape, left->rank,
                                           &output, error) ||
        vx_shape_positive_f32_ratio(
            left_quantization->scale, 1.0f, output->quantization.scale,
            "operator input 'a'.quantization.scale", error) ||
        vx_shape_positive_f32_ratio(
            right_quantization->scale, 1.0f, output->quantization.scale,
            "operator input 'b'.quantization.scale", error)) return -1;
    return vx_shape_clone_output(candidate, left->shape, left->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_embedding(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* weight_quantization;
    uint64_t* output_shape;
    size_t output_rank;
    size_t index;
    int status;
    char path[256];
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_i32_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 1, 0, 0,
                      "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 2, 1,
                      "operator input 'weight'.shape", error) ||
        vx_shape_axis_zero_byte_weight(weight, "operator input 'weight'",
                                       &weight_quantization, error)) return -1;
    if (input->rank == SIZE_MAX ||
        input->rank + 1 > SIZE_MAX / sizeof(*output_shape)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "rank is too large.");
    }
    output_rank = input->rank + 1;
    output_shape = (uint64_t*)malloc(output_rank * sizeof(*output_shape));
    if (!output_shape) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator output 'out'.shape", "allocation failed.");
    }
    memcpy(output_shape, input->shape, input->rank * sizeof(*output_shape));
    output_shape[input->rank] = weight->shape[1];
    status = vx_shape_quantized_declared_output(request, output_shape,
                                                 output_rank, &output, error);
    for (index = 0; !status && index < weight_quantization->count; ++index) {
        snprintf(path, sizeof(path),
                 "operator input 'weight'.quantization.scales[%zu]", index);
        status = vx_shape_positive_f32_ratio(
            weight_quantization->scales[index], 1.0f,
            output->quantization.scale, path, error);
    }
    if (!status) {
        status = vx_shape_clone_output(candidate, output_shape, output_rank,
                                       output->dtype, &output->quantization,
                                       error);
    }
    free(output_shape);
    return status;
}

static int vx_shape_infer_q_activation(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const APPROXIMATE[] = {"approximate"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* output;
    const VxShapeParam* param;
    if (vx_shape_per_tensor_byte(input, "operator input 'input'", NULL,
                                 error)) return -1;
    if (kind == VX_OP_Q_GELU) {
        if (vx_shape_validate_params(request, APPROXIMATE, 1, error)) return -1;
        param = vx_shape_find_param(request, "approximate");
        if (param && (param->kind != VX_SHAPE_PARAM_STRING ||
                      !param->value.string ||
                      strcmp(param->value.string, "none"))) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.approximate",
                                 "must be 'none'.");
        }
    } else if (vx_shape_validate_params(request, NULL, 0, error)) {
        return -1;
    }
    if (vx_shape_quantized_declared_output(request, input->shape, input->rank,
                                           &output, error)) return -1;
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_layer_norm(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias =
        vx_shape_input_descriptor(request, "bias");
    const VxShapeTensorDescriptor* output;
    const VxShapeParam* eps;
    uint64_t feature;
    if (vx_shape_rank(input, 1, 0, 0,
                      "operator input 'input'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'", NULL,
                                 error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_float_tensor(bias, "operator input 'bias'", error)) return -1;
    feature = input->shape[input->rank - 1];
    eps = vx_shape_find_param(request, "eps");
    if (eps) {
        double value;
        float canonical;
        if (vx_shape_number(eps, "operator params.eps", &value, error)) return -1;
        canonical = (float)value;
        if (!isfinite(value) || value <= 0.0 ||
            !isfinite(canonical) || canonical <= 0.0f) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.eps",
                                 "must be positive and representable as float32.");
        }
    }
    if (vx_shape_vector(weight, feature,
                        "operator input 'weight'.shape", error) ||
        vx_shape_vector(bias, feature,
                        "operator input 'bias'.shape", error) ||
        vx_shape_normalization_params(request, feature, error) ||
        vx_shape_quantized_declared_output(request, input->shape, input->rank,
                                           &output, error)) return -1;
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_group_norm(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"num_groups", "eps", "data_layout"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeTensorDescriptor* bias =
        vx_shape_input_descriptor(request, "bias");
    const VxShapeTensorDescriptor* output;
    const VxShapeParam* param;
    const char* data_layout;
    uint64_t channels;
    double value;
    if (vx_shape_rank(input, 0, 4, 1,
                      "operator input 'input'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'", NULL,
                                 error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_float_tensor(bias, "operator input 'bias'", error)) return -1;
    channels = input->shape[3];
    if (vx_shape_vector(weight, channels,
                        "operator input 'weight'.shape", error) ||
        vx_shape_vector(bias, channels,
                        "operator input 'bias'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 3, error) ||
        vx_shape_param_string_default(request, "data_layout", "NHWC",
                                      &data_layout, error)) return -1;
    if (strcmp(data_layout, "NHWC")) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.data_layout", "must be 'NHWC'.");
    }
    param = vx_shape_find_param(request, "eps");
    if (param) {
        float canonical;
        if (vx_shape_number(param, "operator params.eps", &value, error)) return -1;
        canonical = (float)value;
        if (!isfinite(value) || value <= 0.0 ||
            !isfinite(canonical) || canonical <= 0.0f) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.eps",
                                 "must be positive and representable as float32.");
        }
    }
    param = vx_shape_find_param(request, "num_groups");
    if (!param || param->kind != VX_SHAPE_PARAM_NUMBER ||
        !vx_shape_is_safe_integer(param->value.number) ||
        param->value.number <= 0.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.num_groups",
                             "must be a positive safe integer.");
    }
    if (channels % (uint64_t)param->value.number) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.num_groups",
                             "must divide the channel extent.");
    }
    if (vx_shape_quantized_declared_output(request, input->shape, input->rank,
                                           &output, error)) return -1;
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_masked_mean(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* input_quantization;
    uint64_t output_shape[2];
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_rank(input, 0, 3, 1,
                      "operator input 'input'.shape", error) ||
        vx_shape_rank(mask, 0, 2, 1,
                      "operator input 'mask'.shape", error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'",
                                 &input_quantization, error) ||
        vx_shape_i32_tensor(mask, "operator input 'mask'", error)) return -1;
    if (mask->shape[0] != input->shape[0] ||
        mask->shape[1] != input->shape[1]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'mask'.shape",
                             "must be [B,S] for input [B,S,D].");
    }
    output_shape[0] = input->shape[0];
    output_shape[1] = input->shape[2];
    if (vx_shape_quantized_declared_output(request, output_shape, 2,
                                           &output, error) ||
        vx_shape_positive_f32_ratio(
            input_quantization->scale, 1.0f, output->quantization.scale,
            "declared output 'out'.quantization.scale", error) ||
        vx_shape_i32_sum_bound(
            input->shape[1],
            vx_shape_centered_magnitude(input->dtype,
                                        input_quantization->zero_point),
            "operator input 'input'.shape[1]", error)) return -1;
    return vx_shape_clone_output(candidate, output_shape, 2, output->dtype,
                                 &output->quantization, error);
}

static int vx_shape_q_attention_parameters(
        const VxConcreteShapeRequest* request,
        uint64_t* heads,
        float* scale,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"heads", "causal", "scale"};
    const VxShapeParam* param;
    double value;
    if (vx_shape_validate_params(request, FIELDS, 3, error)) return -1;
    param = vx_shape_find_param(request, "heads");
    if (!param || param->kind != VX_SHAPE_PARAM_NUMBER ||
        !vx_shape_is_safe_integer(param->value.number) ||
        param->value.number <= 0.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.heads",
                             "must be a positive safe integer.");
    }
    *heads = (uint64_t)param->value.number;
    param = vx_shape_find_param(request, "causal");
    if (!param || param->kind != VX_SHAPE_PARAM_BOOLEAN) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.causal", "must be boolean.");
    }
    param = vx_shape_find_param(request, "scale");
    if (!param || param->kind != VX_SHAPE_PARAM_NUMBER) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.scale",
                             "must be positive and representable as float32.");
    }
    value = param->value.number;
    *scale = (float)value;
    if (!isfinite(value) || value <= 0.0 ||
        !isfinite(*scale) || *scale <= 0.0f) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.scale",
                             "must be positive and representable as float32.");
    }
    return 0;
}

static int vx_shape_infer_q_sdpa(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* query =
        vx_shape_input_descriptor(request, "q");
    const VxShapeTensorDescriptor* key =
        vx_shape_input_descriptor(request, "k");
    const VxShapeTensorDescriptor* value =
        vx_shape_input_descriptor(request, "v");
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* query_quantization;
    const VxShapeQuantization* key_quantization;
    const VxShapeQuantization* value_quantization;
    uint64_t heads = 0;
    uint64_t feature;
    uint64_t head_dimension;
    uint64_t batch;
    uint64_t queries;
    uint64_t keys;
    uint64_t maximum_dot = 0;
    uint64_t left_magnitude;
    uint64_t right_magnitude;
    float scale = 0.0f;
    float score_multiplier;
    float maximum_score;
    size_t sequence_axis;
    size_t feature_axis;
    if (vx_shape_q_attention_parameters(request, &heads, &scale, error) ||
        vx_shape_per_tensor_byte(query, "operator input 'q'",
                                 &query_quantization, error) ||
        vx_shape_per_tensor_byte(key, "operator input 'k'",
                                 &key_quantization, error) ||
        vx_shape_per_tensor_byte(value, "operator input 'v'",
                                 &value_quantization, error) ||
        vx_shape_attention_rank(query, "operator input 'q'.shape", error)) {
        return -1;
    }
    if (key->rank != query->rank || value->rank != query->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator inputs",
                             "q, k, and v must have the same rank.");
    }
    sequence_axis = query->rank - 2;
    feature_axis = query->rank - 1;
    batch = query->rank == 2 ? 1 : query->shape[0];
    queries = query->shape[sequence_axis];
    keys = key->shape[sequence_axis];
    feature = query->shape[feature_axis];
    if ((query->rank == 3 &&
         (key->shape[0] != batch || value->shape[0] != batch)) ||
        key->shape[feature_axis] != feature ||
        value->shape[feature_axis] != feature ||
        value->shape[sequence_axis] != keys) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
            "operator inputs",
            "QSDPA q/k/v batch, feature, and K/V sequence geometry must match.");
    }
    if (feature % heads || feature % 4) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'q'.shape[%zu]",
                 feature_axis);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             path, "D must be divisible by heads and by 4.");
    }
    head_dimension = feature / heads;
    if (head_dimension % 4 || head_dimension > 64) {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'q'.shape[%zu]",
                 feature_axis);
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH, path,
            "head_dim must be divisible by 4 and no greater than 64.");
    }
    if (vx_shape_attention_mask(mask, batch, queries, keys, error) ||
        vx_shape_quantized_declared_output(request, query->shape, query->rank,
                                           &output, error)) return -1;
    left_magnitude = vx_shape_centered_magnitude(
        query->dtype, query_quantization->zero_point);
    right_magnitude = vx_shape_centered_magnitude(
        key->dtype, key_quantization->zero_point);
    {
        char path[224];
        snprintf(path, sizeof(path), "operator input 'q'.shape[%zu]",
                 feature_axis);
        if (vx_shape_i32_accumulator_bound(head_dimension, left_magnitude,
                                           right_magnitude, path, error)) {
            return -1;
        }
    }
    if (vx_shape_checked_multiply_u64(head_dimension, left_magnitude,
                                      "operator params.scale", &maximum_dot,
                                      error) ||
        vx_shape_checked_multiply_u64(maximum_dot, right_magnitude,
                                      "operator params.scale", &maximum_dot,
                                      error)) return -1;
    score_multiplier = (query_quantization->scale * key_quantization->scale) *
        scale;
    maximum_score = (float)maximum_dot * score_multiplier;
    if (!isfinite(score_multiplier) || score_multiplier <= 0.0f ||
        !isfinite(maximum_score)) {
        return vx_shape_fail(
            error, VX_SHAPE_CONTRACT_ERROR_UNSAFE_QUANTIZATION_TRANSFORM,
            "operator params.scale",
            "quantized score scale and maximum score must be finite positive float32 values.");
    }
    if (vx_shape_positive_f32_ratio(
            value_quantization->scale, 1.0f, output->quantization.scale,
            "declared output 'out'.quantization.scale", error)) return -1;
    return vx_shape_clone_output(candidate, query->shape, query->rank,
                                 output->dtype, &output->quantization, error);
}

static int vx_shape_infer_q_arg_max(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeParam* axis;
    if (vx_shape_per_tensor_byte(input, "operator input 'input'", NULL,
                                 error)) return -1;
    if (input->rank < 2 || input->rank > 8) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator input 'input'.shape",
                             "must have rank 2 through 8.");
    }
    if (vx_shape_validate_params(request, FIELDS, 1, error)) return -1;
    axis = vx_shape_find_param(request, "axis");
    if (request->param_count != 1 || !axis ||
        axis->kind != VX_SHAPE_PARAM_NUMBER || axis->value.number != -1.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.axis",
                             "must be exactly -1 for canonical QArgMax.");
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank - 1,
                                 VX_DTYPE_I32, NULL, error);
}

static int vx_shape_infer_quantized_kind(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    switch (kind) {
        case VX_OP_Q_LINEAR:
        case VX_OP_Q_MATMUL:
        case VX_OP_Q_GEMM:
            return vx_shape_infer_q_dense(request, candidate, error);
        case VX_OP_Q_BATCH_MATMUL:
            return vx_shape_infer_q_batch_matmul(request, candidate, error);
        case VX_OP_Q_CONV_2D:
            return vx_shape_infer_q_conv2d(request, candidate, error);
        case VX_OP_Q_ADD:
            return vx_shape_infer_q_add(request, candidate, error);
        case VX_OP_Q_EMBEDDING:
            return vx_shape_infer_q_embedding(request, candidate, error);
        case VX_OP_Q_GELU:
        case VX_OP_Q_SILU:
            return vx_shape_infer_q_activation(kind, request, candidate, error);
        case VX_OP_Q_LAYER_NORM:
            return vx_shape_infer_q_layer_norm(request, candidate, error);
        case VX_OP_Q_GROUP_NORM:
            return vx_shape_infer_q_group_norm(request, candidate, error);
        case VX_OP_Q_MASKED_MEAN:
            return vx_shape_infer_q_masked_mean(request, candidate, error);
        case VX_OP_Q_SDPA:
            return vx_shape_infer_q_sdpa(request, candidate, error);
        case VX_OP_Q_ARG_MAX:
            return vx_shape_infer_q_arg_max(request, candidate, error);
        default:
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                                 "operator",
                                 "has no implemented quantized contract.");
    }
}

static int vx_shape_infer_final_not(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_i32_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank_range(input, 0, 8, "operator input 'input'.shape", error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 VX_DTYPE_I32, NULL, error);
}

static int vx_shape_infer_final_mask(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* mask =
        vx_shape_input_descriptor(request, "mask");
    const VxShapeTensorDescriptor* left =
        vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right =
        vx_shape_input_descriptor(request, "b");
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_unquantized(mask, "operator input 'mask'", error) ||
        vx_shape_unquantized(left, "operator input 'a'", error) ||
        vx_shape_unquantized(right, "operator input 'b'", error) ||
        vx_shape_rank_range(left, 0, 8, "operator data inputs", error)) {
        return -1;
    }
    if (mask->dtype != VX_DTYPE_F32 && mask->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'mask'.dtype",
                             "must be float32 or int32.");
    }
    if (left->dtype != right->dtype) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator data inputs", "must have the same dtype.");
    }
    if (left->dtype != VX_DTYPE_F32 && left->dtype != VX_DTYPE_I32) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                             "operator input 'a'.dtype",
                             "must be float32 or int32.");
    }
    if (!vx_shape_same(mask, left) || !vx_shape_same(left, right)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator inputs",
                             "Mask requires three exactly equal shapes; it never broadcasts.");
    }
    return vx_shape_clone_output(candidate, left->shape, left->rank,
                                 left->dtype, NULL, error);
}

static int vx_shape_infer_final_batch_norm2d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"eps"};
    static const char* const PARAMETER_NAMES[] = {
        "weight", "bias", "running_mean", "running_var"
    };
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeParam* eps;
    double value;
    size_t index;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error)) return -1;
    for (index = 0; index < 4; ++index) {
        char path[224];
        const VxShapeTensorDescriptor* parameter =
            vx_shape_input_descriptor(request, PARAMETER_NAMES[index]);
        snprintf(path, sizeof(path), "operator input '%s'", PARAMETER_NAMES[index]);
        if (vx_shape_float_tensor(parameter, path, error)) return -1;
        snprintf(path, sizeof(path), "operator input '%s'.shape", PARAMETER_NAMES[index]);
        if (vx_shape_vector(parameter, input->shape[3], path, error)) return -1;
    }
    eps = vx_shape_find_param(request, "eps");
    if (eps) {
        if (vx_shape_number(eps, "operator params.eps", &value, error)) return -1;
        if (!isfinite(value) || value <= 0.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.eps",
                                 "must be finite and positive.");
        }
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_final_interpolate1d(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"size"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    uint64_t size;
    uint64_t output_shape[3];
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 3, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_positive_integer_param(request, "size", 0, &size, error)) {
        return -1;
    }
    output_shape[0] = input->shape[0];
    output_shape[1] = input->shape[1];
    output_shape[2] = size;
    return vx_shape_clone_output(candidate, output_shape, 3,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_final_vision_profile(
        VxOperatorKind kind,
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    uint64_t output_shape[3];
    uint64_t doubled = 0;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank(input, 0, 4, 1, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, NULL, 0, error)) return -1;
    output_shape[0] = input->shape[0];
    if (kind == VX_OP_PROFILE_X || kind == VX_OP_PROFILE_Y) {
        if (vx_shape_checked_multiply_u64(input->shape[3], 2,
                                          "profile channel extent",
                                          &doubled, error)) return -1;
        output_shape[1] = doubled;
        output_shape[2] = kind == VX_OP_PROFILE_X
            ? input->shape[2] : input->shape[1];
    } else {
        output_shape[1] = input->shape[3];
        output_shape[2] = input->shape[2];
    }
    return vx_shape_clone_output(candidate, output_shape, 3,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_final_dropout(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"ratio", "p", "probability", "seed"};
    static const char* const PROBABILITY_NAMES[] = {"ratio", "p", "probability"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeParam* selected = NULL;
    const VxShapeParam* seed;
    size_t index;
    double value = 0.5;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_rank_range(input, 0, 8, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 4, error)) return -1;
    for (index = 0; index < 3; ++index) {
        const VxShapeParam* current =
            vx_shape_find_param(request, PROBABILITY_NAMES[index]);
        if (current && selected) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params",
                                 "must specify at most one Dropout probability field.");
        }
        if (current) selected = current;
    }
    if (selected && vx_shape_number(selected, "operator params.ratio", &value,
                                    error)) return -1;
    if (!isfinite(value) || value < 0.0 || value >= 1.0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.ratio",
                             "must be finite and in [0, 1).");
    }
    seed = vx_shape_find_param(request, "seed");
    if (seed) {
        if (vx_shape_number(seed, "operator params.seed", &value, error)) return -1;
        if (!vx_shape_is_safe_integer(value) || value < 0.0 ||
            value > 4294967295.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.seed",
                                 "must be an unsigned 32-bit integer.");
        }
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_final_requantize(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* output;
    const VxShapeQuantization* input_quantization;
    const VxShapeQuantization* output_quantization;
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_per_tensor_byte(input, "operator input 'input'",
                                 &input_quantization, error) ||
        vx_shape_quantized_declared_output(request, input->shape, input->rank,
                                           &output, error) ||
        vx_shape_per_tensor_byte(output, "declared output 'out'",
                                 &output_quantization, error) ||
        vx_shape_positive_f32_ratio(input_quantization->scale, 1.0f,
                                    output_quantization->scale,
                                    "declared output 'out'.quantization.scale",
                                    error)) {
        return -1;
    }
    return vx_shape_clone_output(candidate, input->shape, input->rank,
                                 output->dtype, output_quantization, error);
}

static int vx_shape_infer_final_concat2(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"axis", "sigmoid"};
    VxShapeNamedTensor inputs[2];
    VxShapeParam params[1];
    VxConcreteShapeRequest adapted;
    const VxShapeParam* axis = vx_shape_find_param(request, "axis");
    const VxShapeTensorDescriptor* left =
        vx_shape_input_descriptor(request, "a");
    const VxShapeTensorDescriptor* right =
        vx_shape_input_descriptor(request, "b");
    int sigmoid;
    if (vx_shape_validate_params(request, FIELDS, 2, error) ||
        vx_shape_boolean_param(request, "sigmoid", 0, &sigmoid, error)) {
        return -1;
    }
    if (sigmoid &&
        (vx_shape_float_tensor(left, "operator input 'a'", error) ||
         vx_shape_float_tensor(right, "operator input 'b'", error))) return -1;
    memset(&adapted, 0, sizeof(adapted));
    inputs[0].name = "input0";
    inputs[0].descriptor = *left;
    inputs[1].name = "input1";
    inputs[1].descriptor = *right;
    adapted.inputs = inputs;
    adapted.input_count = 2;
    if (axis) {
        params[0] = *axis;
        adapted.params = params;
        adapted.param_count = 1;
    }
    return vx_shape_infer_concat(&adapted, candidate, error);
}

static int vx_shape_infer_final_moe_router(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {
        "num_experts", "top_k", "temperature", "normalize"
    };
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeNamedTensor* named_bias = vx_shape_find_input(request, "bias");
    const VxShapeTensorDescriptor* bias =
        named_bias ? &named_bias->descriptor : NULL;
    const VxShapeParam* temperature;
    uint64_t experts;
    uint64_t num_experts;
    uint64_t top_k;
    uint64_t route_shape[8];
    double value;
    int normalize;
    if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank_range(input, 1, 8, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 2, 1, "operator input 'weight'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 4, error)) return -1;
    if (weight->shape[0] != input->shape[input->rank - 1]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape[0]",
                             "must equal the input feature extent.");
    }
    experts = weight->shape[1];
    if (vx_shape_positive_integer_param(request, "num_experts", experts,
                                        &num_experts, error) ||
        vx_shape_positive_integer_param(request, "top_k", 2, &top_k, error) ||
        vx_shape_boolean_param(request, "normalize", 1, &normalize, error)) {
        return -1;
    }
    (void)normalize;
    if (num_experts != experts) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator params.num_experts",
                             "must equal the router weight extent.");
    }
    if (top_k > experts) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.top_k",
                             "must not exceed the expert count.");
    }
    temperature = vx_shape_find_param(request, "temperature");
    if (temperature) {
        if (vx_shape_number(temperature, "operator params.temperature", &value,
                            error)) return -1;
        if (!isfinite(value) || value <= 0.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.temperature",
                                 "must be finite and positive.");
        }
    }
    if (bias &&
        (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
         vx_shape_vector(bias, experts, "operator input 'bias'.shape", error))) {
        return -1;
    }
    if (input->rank > 1) {
        memcpy(route_shape, input->shape,
               (input->rank - 1) * sizeof(*route_shape));
    }
    route_shape[input->rank - 1] = top_k;
    if (vx_shape_result_allocate(candidate, 2, error) ||
        vx_shape_clone_output_at(candidate, 0, "indices", route_shape,
                                 input->rank, VX_DTYPE_F32, NULL, error) ||
        vx_shape_clone_output_at(candidate, 1, "weights", route_shape,
                                 input->rank, VX_DTYPE_F32, NULL, error)) {
        vx_shape_contract_result_clear(candidate);
        return -1;
    }
    return 0;
}

static int vx_shape_infer_final_moe_linear(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "expert_weight");
    const VxShapeTensorDescriptor* indices =
        vx_shape_input_descriptor(request, "route_indices");
    const VxShapeTensorDescriptor* routes =
        vx_shape_input_descriptor(request, "route_weights");
    const VxShapeNamedTensor* named_bias =
        vx_shape_find_input(request, "expert_bias");
    const VxShapeTensorDescriptor* bias =
        named_bias ? &named_bias->descriptor : NULL;
    uint64_t output_shape[8];
    uint64_t experts;
    uint64_t output_feature;
    size_t axis;
    if (vx_shape_validate_params(request, NULL, 0, error) ||
        vx_shape_float_tensor(input, "operator input 'input'", error) ||
        vx_shape_float_tensor(weight, "operator input 'expert_weight'", error) ||
        vx_shape_float_tensor(indices, "operator input 'route_indices'", error) ||
        vx_shape_float_tensor(routes, "operator input 'route_weights'", error) ||
        vx_shape_rank_range(input, 1, 8, "operator input 'input'.shape", error) ||
        vx_shape_rank(weight, 0, 3, 1,
                      "operator input 'expert_weight'.shape", error)) return -1;
    experts = weight->shape[0];
    output_feature = weight->shape[2];
    if (weight->shape[1] != input->shape[input->rank - 1]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'expert_weight'.shape[1]",
                             "must equal the input feature extent.");
    }
    if (!vx_shape_same(indices, routes) || indices->rank != input->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator route inputs",
                             "must share the input token prefix and one common top-k axis.");
    }
    for (axis = 0; axis + 1 < input->rank; ++axis) {
        if (indices->shape[axis] != input->shape[axis]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator route inputs",
                                 "must share the input token prefix and one common top-k axis.");
        }
    }
    if (indices->shape[indices->rank - 1] > experts) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'route_indices'.shape",
                             "top-k extent must not exceed the expert count.");
    }
    if (bias) {
        if (vx_shape_float_tensor(bias, "operator input 'expert_bias'", error)) {
            return -1;
        }
        if (bias->rank != 2 || bias->shape[0] != experts ||
            bias->shape[1] != output_feature) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'expert_bias'.shape",
                                 "must equal [experts, output_feature].");
        }
    }
    if (input->rank > 1) {
        memcpy(output_shape, input->shape,
               (input->rank - 1) * sizeof(*output_shape));
    }
    output_shape[input->rank - 1] = output_feature;
    return vx_shape_clone_output(candidate, output_shape, input->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_infer_final_cross_attention(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"heads"};
    const VxShapeTensorDescriptor* query =
        vx_shape_input_descriptor(request, "q");
    const VxShapeTensorDescriptor* kv =
        vx_shape_input_descriptor(request, "kv");
    const VxShapeTensorDescriptor* weight =
        vx_shape_input_descriptor(request, "weight");
    const VxShapeNamedTensor* named_scale = vx_shape_find_input(request, "scale");
    const VxShapeNamedTensor* named_bias = vx_shape_find_input(request, "bias");
    const VxShapeTensorDescriptor* scale =
        named_scale ? &named_scale->descriptor : NULL;
    const VxShapeTensorDescriptor* bias =
        named_bias ? &named_bias->descriptor : NULL;
    uint64_t feature;
    uint64_t projection = 0;
    uint64_t heads;
    size_t feature_axis;
    if (vx_shape_float_tensor(query, "operator input 'q'", error) ||
        vx_shape_float_tensor(kv, "operator input 'kv'", error) ||
        vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
        vx_shape_rank_range(query, 2, 3, "operator input 'q'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error)) return -1;
    if (kv->rank != query->rank) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                             "operator input 'kv'.shape",
                             "must have the same rank as q.");
    }
    feature_axis = query->rank - 1;
    feature = query->shape[feature_axis];
    if (kv->shape[feature_axis] != feature ||
        (query->rank == 3 && kv->shape[0] != query->shape[0])) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'kv'.shape",
                             "must share q batch and feature dimensions.");
    }
    if (vx_shape_checked_multiply_u64(feature, 3,
                                      "CrossAttention projection extent",
                                      &projection, error)) return -1;
    if (weight->rank != 2 || weight->shape[0] != projection ||
        weight->shape[1] != feature) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'weight'.shape",
                             "must equal [3*feature, feature].");
    }
    if ((scale &&
         (vx_shape_float_tensor(scale, "operator input 'scale'", error) ||
          vx_shape_vector(scale, projection, "operator input 'scale'.shape", error))) ||
        (bias &&
         (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
          vx_shape_vector(bias, projection, "operator input 'bias'.shape", error)))) {
        return -1;
    }
    if (vx_shape_positive_integer_param(request, "heads", 8, &heads, error)) {
        return -1;
    }
    if (feature % heads) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params.heads",
                             "must divide the feature extent.");
    }
    return vx_shape_clone_output(candidate, query->shape, query->rank,
                                 VX_DTYPE_F32, NULL, error);
}

static int vx_shape_scan_bc_mode(const VxShapeTensorDescriptor* descriptor,
                                 const VxShapeTensorDescriptor* input,
                                 uint64_t state_width) {
    size_t rank = input->rank;
    uint64_t batch = rank == 3 ? input->shape[0] : 1;
    uint64_t sequence = input->shape[rank - 2];
    if (descriptor->rank == 1 && descriptor->shape[0] == state_width) return 0;
    if (descriptor->rank == 2 && descriptor->shape[0] == sequence &&
        descriptor->shape[1] == state_width) return 1;
    if (rank == 3 && descriptor->rank == 3 &&
        descriptor->shape[0] == batch && descriptor->shape[1] == sequence &&
        descriptor->shape[2] == state_width) return 2;
    return -1;
}

static int vx_shape_infer_final_scan(
        const VxConcreteShapeRequest* request,
        VxConcreteShapeResult* candidate,
        VxShapeContractError* error) {
    static const char* const FIELDS[] = {"delta_softplus"};
    static const char* const REQUIRED[] = {"input", "delta", "A", "B", "C"};
    static const char* const OPTIONAL[] = {"D", "z", "initial_state"};
    const VxShapeTensorDescriptor* input =
        vx_shape_input_descriptor(request, "input");
    const VxShapeTensorDescriptor* delta =
        vx_shape_input_descriptor(request, "delta");
    const VxShapeTensorDescriptor* a = vx_shape_input_descriptor(request, "A");
    const VxShapeTensorDescriptor* b = vx_shape_input_descriptor(request, "B");
    const VxShapeTensorDescriptor* c = vx_shape_input_descriptor(request, "C");
    uint64_t batch;
    uint64_t channels;
    uint64_t state_width;
    uint64_t state_shape[3];
    size_t index;
    int delta_softplus;
    for (index = 0; index < 5; ++index) {
        char path[224];
        const VxShapeTensorDescriptor* descriptor =
            vx_shape_input_descriptor(request, REQUIRED[index]);
        snprintf(path, sizeof(path), "operator input '%s'", REQUIRED[index]);
        if (vx_shape_float_tensor(descriptor, path, error)) return -1;
    }
    for (index = 0; index < 3; ++index) {
        const VxShapeNamedTensor* named =
            vx_shape_find_input(request, OPTIONAL[index]);
        if (named) {
            char path[224];
            snprintf(path, sizeof(path), "operator input '%s'", OPTIONAL[index]);
            if (vx_shape_float_tensor(&named->descriptor, path, error)) return -1;
        }
    }
    if (vx_shape_rank_range(input, 2, 3, "operator input 'input'.shape", error) ||
        vx_shape_validate_params(request, FIELDS, 1, error) ||
        vx_shape_boolean_param(request, "delta_softplus", 1,
                               &delta_softplus, error)) return -1;
    (void)delta_softplus;
    if (!vx_shape_same(input, delta)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'delta'.shape",
                             "must equal the input shape.");
    }
    batch = input->rank == 3 ? input->shape[0] : 1;
    channels = input->shape[input->rank - 1];
    if (a->rank != 2 || a->shape[0] != channels) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator input 'A'.shape",
                             "must be [channels, state_width].");
    }
    state_width = a->shape[1];
    if (vx_shape_scan_bc_mode(b, input, state_width) < 0 ||
        vx_shape_scan_bc_mode(c, input, state_width) < 0) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                             "operator B/C inputs",
                             "must use [N], [S,N], or [B,S,N] selective-scan layout.");
    }
    state_shape[0] = batch;
    state_shape[1] = channels;
    state_shape[2] = state_width;
    {
        const VxShapeNamedTensor* named_d = vx_shape_find_input(request, "D");
        const VxShapeNamedTensor* named_z = vx_shape_find_input(request, "z");
        const VxShapeNamedTensor* named_initial =
            vx_shape_find_input(request, "initial_state");
        VxShapeTensorDescriptor expected;
        if (named_d && vx_shape_vector(&named_d->descriptor, channels,
                                       "operator input 'D'.shape", error)) {
            return -1;
        }
        if (named_z && !vx_shape_same(&named_z->descriptor, input)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'z'.shape",
                                 "must equal the input shape.");
        }
        memset(&expected, 0, sizeof(expected));
        expected.rank = 3;
        expected.shape = state_shape;
        if (named_initial && !vx_shape_same(&named_initial->descriptor, &expected)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'initial_state'.shape",
                                 "must equal [batch, channels, state_width].");
        }
    }
    if (vx_shape_result_allocate(candidate, 2, error) ||
        vx_shape_clone_output_at(candidate, 0, "out", input->shape, input->rank,
                                 VX_DTYPE_F32, NULL, error) ||
        vx_shape_clone_output_at(candidate, 1, "state", state_shape, 3,
                                 VX_DTYPE_F32, NULL, error)) {
        vx_shape_contract_result_clear(candidate);
        return -1;
    }
    return 0;
}

static int vx_shape_infer_kind(VxOperatorKind kind,
                               const VxConcreteShapeRequest* request,
                               VxConcreteShapeResult* candidate,
                               VxShapeContractError* error) {
    const VxShapeTensorDescriptor* input;
    const VxShapeTensorDescriptor* weight;
    const VxShapeTensorDescriptor* bias;
    const VxShapeTensorDescriptor* other;
    const VxShapeTensorDescriptor* declared = NULL;
    static const char* const TO[] = {"to"};
    static const char* const GROUP_FIELDS[] = {"num_groups", "eps"};
    static const char* const RELU[] = {"relu"};
    uint64_t* output_shape;
    size_t output_rank;
    uint64_t feature;
    const VxShapeParam* param;
    double value = 0.0;
    VxDataType dtype;

    if (vx_shape_quantized_kind(kind)) {
        return vx_shape_infer_quantized_kind(kind, request, candidate, error);
    }
    if (vx_shape_spatial_kind(kind)) {
        return vx_shape_infer_spatial_kind(kind, request, candidate, error);
    }
    if (vx_shape_non_spatial_wave_b_kind(kind)) {
        return vx_shape_infer_non_spatial_wave_b_kind(
            kind, request, candidate, error);
    }
    if (kind == VX_OP_SDPA || kind == VX_OP_CROSS_SDPA || kind == VX_OP_ROPE) {
        return vx_shape_infer_attention_kind(kind, request, candidate, error);
    }
    switch (kind) {
        case VX_OP_MOE_ROUTER:
            return vx_shape_infer_final_moe_router(request, candidate, error);
        case VX_OP_MOE_LINEAR:
            return vx_shape_infer_final_moe_linear(request, candidate, error);
        case VX_OP_CROSS_ATTENTION:
            return vx_shape_infer_final_cross_attention(request, candidate, error);
        case VX_OP_BATCH_NORM_2D:
            return vx_shape_infer_final_batch_norm2d(request, candidate, error);
        case VX_OP_INTERPOLATE_1D:
            return vx_shape_infer_final_interpolate1d(request, candidate, error);
        case VX_OP_NOT:
            return vx_shape_infer_final_not(request, candidate, error);
        case VX_OP_MASK:
            return vx_shape_infer_final_mask(request, candidate, error);
        case VX_OP_BROADCAST:
            return vx_shape_infer_expand(request, candidate, error);
        case VX_OP_CONCAT2:
            return vx_shape_infer_final_concat2(request, candidate, error);
        case VX_OP_REQUANTIZE_LINEAR:
            return vx_shape_infer_final_requantize(request, candidate, error);
        case VX_OP_SSM_SCAN:
        case VX_OP_SELECTIVE_SCAN:
            return vx_shape_infer_final_scan(request, candidate, error);
        case VX_OP_SPATIAL_SOFTARGMAX_Y:
        case VX_OP_MEAN_HEIGHT:
        case VX_OP_PROFILE_X:
        case VX_OP_PROFILE_Y:
            return vx_shape_infer_final_vision_profile(
                kind, request, candidate, error);
        case VX_OP_DROPOUT:
            return vx_shape_infer_final_dropout(request, candidate, error);
        default:
            break;
    }

    {
        const VxShapeNamedTensor* named_input =
            vx_shape_find_input(request, "input");
        input = named_input ? &named_input->descriptor : NULL;
    }
    if (kind != VX_OP_ADD && kind != VX_OP_MUL && !input) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS,
                             "operator inputs", "is missing the input port.");
    }
    if (kind == VX_OP_IDENTITY) {
        if (vx_shape_validate_params(request, NULL, 0, error)) return -1;
        return vx_shape_clone_output(candidate, input->shape, input->rank,
                                     input->dtype, &input->quantization, error);
    }
    if (vx_shape_activation_kind(kind)) {
        if (kind == VX_OP_CLIP && input->dtype == VX_DTYPE_I32) {
            if (vx_shape_unquantized(input, "operator input 'input'", error)) return -1;
        } else if (vx_shape_float_tensor(input, "operator input 'input'", error)) {
            return -1;
        }
        if (vx_shape_activation_params(kind, request, input->rank,
                                       input->dtype, error)) return -1;
        return vx_shape_clone_output(candidate, input->shape, input->rank,
                                     input->dtype, NULL, error);
    }
    if (kind == VX_OP_PRELU) {
        other = &vx_shape_find_input(request, "slope")->descriptor;
        if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
            vx_shape_float_tensor(other, "operator input 'slope'", error) ||
            vx_shape_rank(input, 1, 0, 0, "operator input 'input'.shape", error) ||
            vx_shape_rank(other, 0, 1, 1, "operator input 'slope'.shape", error) ||
            vx_shape_validate_params(request, NULL, 0, error)) return -1;
        if (other->shape[0] != 1 &&
            other->shape[0] != input->shape[input->rank - 1]) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator input 'slope'.shape",
                                 "must be scalar or match the feature extent.");
        }
        return vx_shape_clone_output(candidate, input->shape, input->rank,
                                     VX_DTYPE_F32, NULL, error);
    }
    if (kind == VX_OP_CAST) {
        if (vx_shape_unquantized(input, "operator input 'input'", error) ||
            vx_shape_validate_params(request, TO, 1, error)) return -1;
        param = vx_shape_find_param(request, "to");
        if (!param || param->kind != VX_SHAPE_PARAM_STRING ||
            !param->value.string) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "operator params.to",
                                 "has an unsupported runtime dtype.");
        }
        if (!strcmp(param->value.string, "float32")) dtype = VX_DTYPE_F32;
        else if (!strcmp(param->value.string, "int32")) dtype = VX_DTYPE_I32;
        else if (!strcmp(param->value.string, "int8")) dtype = VX_DTYPE_I8;
        else if (!strcmp(param->value.string, "uint8")) dtype = VX_DTYPE_U8;
        else {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "operator params.to",
                                 "has an unsupported runtime dtype.");
        }
        return vx_shape_clone_output(candidate, input->shape, input->rank,
                                     dtype, NULL, error);
    }
    if (kind == VX_OP_QUANTIZE_LINEAR) {
        const VxShapeTensorDescriptor* scale =
            &vx_shape_find_input(request, "scale")->descriptor;
        const VxShapeNamedTensor* zero_named =
            vx_shape_find_input(request, "zero_point");
        if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
            vx_shape_validate_params(request, NULL, 0, error) ||
            vx_shape_declared_output(request, &declared, error)) return -1;
        if (!declared) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS,
                                 "declared outputs",
                                 "must contain exactly the 'out' descriptor.");
        }
        if (declared->dtype != VX_DTYPE_I8 && declared->dtype != VX_DTYPE_U8) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "declared output 'out'.dtype",
                                 "must be 'int8' or 'uint8'.");
        }
        if (declared->quantization.scheme == VX_SHAPE_QUANTIZATION_NONE) {
            return vx_shape_fail(error,
                                 VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 "declared output 'out'.quantization",
                                 "is required.");
        }
        if (!vx_shape_same(input, declared)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "declared output 'out'.shape",
                                 "must exactly equal the input shape.");
        }
        if (vx_shape_quantization_parameter_inputs(
                input, scale,
                zero_named ? &zero_named->descriptor : NULL,
                declared->dtype, &declared->quantization, error)) return -1;
        return vx_shape_clone_output(candidate, input->shape, input->rank,
                                     declared->dtype,
                                     &declared->quantization, error);
    }
    if (kind == VX_OP_DEQUANTIZE_LINEAR) {
        const VxShapeTensorDescriptor* scale =
            &vx_shape_find_input(request, "scale")->descriptor;
        const VxShapeNamedTensor* zero_named =
            vx_shape_find_input(request, "zero_point");
        if (input->dtype != VX_DTYPE_I8 && input->dtype != VX_DTYPE_U8) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "operator input 'input'.dtype",
                                 "must be 'int8' or 'uint8'.");
        }
        if (input->quantization.scheme == VX_SHAPE_QUANTIZATION_NONE) {
            return vx_shape_fail(error,
                                 VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 "operator input 'input'.quantization",
                                 "is required.");
        }
        if (vx_shape_validate_params(request, NULL, 0, error) ||
            vx_shape_quantization_parameter_inputs(
                input, scale,
                zero_named ? &zero_named->descriptor : NULL,
                input->dtype, &input->quantization, error)) return -1;
        return vx_shape_clone_output(candidate, input->shape, input->rank,
                                     VX_DTYPE_F32, NULL, error);
    }
    if (kind == VX_OP_LAYER_NORM || kind == VX_OP_RMS_NORM) {
        weight = &vx_shape_find_input(request, "weight")->descriptor;
        if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
            vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
            vx_shape_rank(input, 1, 0, 0,
                          "operator input 'input'.shape", error)) return -1;
        feature = input->shape[input->rank - 1];
        if (vx_shape_vector(weight, feature,
                            "operator input 'weight'.shape", error)) return -1;
        if (kind == VX_OP_LAYER_NORM &&
            (bias = (vx_shape_find_input(request, "bias")
                ? &vx_shape_find_input(request, "bias")->descriptor : NULL))) {
            if (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
                vx_shape_vector(bias, feature,
                                "operator input 'bias'.shape", error)) return -1;
        }
        if (vx_shape_normalization_params(request, feature, error)) return -1;
        return vx_shape_clone_output(candidate, input->shape, input->rank,
                                     VX_DTYPE_F32, NULL, error);
    }
    if (kind == VX_OP_GROUP_NORM) {
        weight = &vx_shape_find_input(request, "weight")->descriptor;
        bias = &vx_shape_find_input(request, "bias")->descriptor;
        if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
            vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
            vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
            vx_shape_rank(input, 0, 4, 1,
                          "operator input 'input'.shape", error)) return -1;
        feature = input->shape[3];
        if (vx_shape_vector(weight, feature,
                            "operator input 'weight'.shape", error) ||
            vx_shape_vector(bias, feature,
                            "operator input 'bias'.shape", error) ||
            vx_shape_validate_params(request, GROUP_FIELDS, 2, error)) return -1;
        param = vx_shape_find_param(request, "eps");
        if (param) {
            if (vx_shape_number(param, "operator params.eps", &value, error)) return -1;
            if (!isfinite(value) || value <= 0.0) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     "operator params.eps",
                                     "must be finite and positive.");
            }
        }
        param = vx_shape_find_param(request, "num_groups");
        if (!param) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.num_groups",
                                 "must be a positive safe integer.");
        }
        if (vx_shape_number(param, "operator params.num_groups",
                            &value, error)) return -1;
        if (!vx_shape_is_safe_integer(value) || value <= 0.0) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                 "operator params.num_groups",
                                 "must be a positive safe integer.");
        }
        if (feature % (uint64_t)value) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator params.num_groups",
                                 "must divide the channel extent.");
        }
        return vx_shape_clone_output(candidate, input->shape, input->rank,
                                     VX_DTYPE_F32, NULL, error);
    }
    if (kind == VX_OP_LINEAR || kind == VX_OP_GEMM || kind == VX_OP_MATMUL) {
        VxDenseLayout layout = VX_DENSE_LAYOUT_DIN_DOUT;
        uint64_t contracted;
        uint64_t output_feature;
        weight = &vx_shape_find_input(request, "weight")->descriptor;
        bias = vx_shape_find_input(request, "bias")
            ? &vx_shape_find_input(request, "bias")->descriptor : NULL;
        if (vx_shape_float_tensor(input, "operator input 'input'", error) ||
            vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
            vx_shape_rank(input, 1, 0, 0,
                          "operator input 'input'.shape", error) ||
            vx_shape_rank(weight, 0, 2, 1,
                          "operator input 'weight'.shape", error) ||
            vx_shape_dense_layout(kind, request, &layout, error)) return -1;
        contracted = layout == VX_DENSE_LAYOUT_DIN_DOUT
            ? weight->shape[0] : weight->shape[1];
        output_feature = layout == VX_DENSE_LAYOUT_DIN_DOUT
            ? weight->shape[1] : weight->shape[0];
        if (input->shape[input->rank - 1] != contracted) {
            char path[224];
            snprintf(path, sizeof(path), "operator input 'input'.shape[%zu]",
                     input->rank - 1);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 path, "must equal the contracted weight extent.");
        }
        if (bias &&
            (vx_shape_float_tensor(bias, "operator input 'bias'", error) ||
             vx_shape_vector(bias, output_feature,
                             "operator input 'bias'.shape", error))) return -1;
        output_rank = input->rank;
        if (!output_rank) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_RANK,
                                 "operator input 'input'.shape",
                                 "must have rank at least 1.");
        }
        if (output_rank > SIZE_MAX / sizeof(*output_shape)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 "operator output 'out'.shape", "rank is too large.");
        }
        output_shape = (uint64_t*)malloc(output_rank * sizeof(*output_shape));
        if (!output_shape) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 "operator output 'out'.shape", "allocation failed.");
        }
        memcpy(output_shape, input->shape, output_rank * sizeof(*output_shape));
        output_shape[output_rank - 1] = output_feature;
        if (vx_shape_clone_output(candidate, output_shape, output_rank,
                                  VX_DTYPE_F32, NULL, error)) {
            free(output_shape);
            return -1;
        }
        free(output_shape);
        return 0;
    }
    if (kind == VX_OP_EMBEDDING) {
        weight = &vx_shape_find_input(request, "weight")->descriptor;
        if (input->dtype != VX_DTYPE_I32) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE,
                                 "operator input 'input'.dtype",
                                 "must be 'int32'.");
        }
        if (vx_shape_unquantized(input, "operator input 'input'", error) ||
            vx_shape_float_tensor(weight, "operator input 'weight'", error) ||
            vx_shape_rank(input, 1, 0, 0,
                          "operator input 'input'.shape", error) ||
            vx_shape_rank(weight, 0, 2, 1,
                          "operator input 'weight'.shape", error) ||
            vx_shape_validate_params(request, NULL, 0, error)) return -1;
        if (input->rank == SIZE_MAX ||
            input->rank + 1 > SIZE_MAX / sizeof(*output_shape)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 "operator output 'out'.shape", "rank is too large.");
        }
        output_rank = input->rank + 1;
        output_shape = (uint64_t*)malloc(output_rank * sizeof(*output_shape));
        if (!output_shape) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 "operator output 'out'.shape", "allocation failed.");
        }
        memcpy(output_shape, input->shape, input->rank * sizeof(*output_shape));
        output_shape[input->rank] = weight->shape[1];
        if (vx_shape_clone_output(candidate, output_shape, output_rank,
                                  VX_DTYPE_F32, NULL, error)) {
            free(output_shape);
            return -1;
        }
        free(output_shape);
        return 0;
    }
    if (kind == VX_OP_ADD || kind == VX_OP_MUL) {
        const VxShapeTensorDescriptor* left =
            &vx_shape_find_input(request, "a")->descriptor;
        const VxShapeTensorDescriptor* right =
            &vx_shape_find_input(request, "b")->descriptor;
        if (vx_shape_float_tensor(left, "operator input 'a'", error) ||
            vx_shape_float_tensor(right, "operator input 'b'", error)) return -1;
        if (kind == VX_OP_ADD) {
            if (vx_shape_validate_params(request, RELU, 1, error)) return -1;
            param = vx_shape_find_param(request, "relu");
            value = 0.0;
            if (param && vx_shape_number(param, "operator params.relu", &value, error)) return -1;
            if (!vx_shape_is_safe_integer(value) || value < 0.0 || value > 2.0) {
                return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                                     "operator params.relu", "must be 0, 1, or 2.");
            }
        } else if (vx_shape_validate_params(request, NULL, 0, error)) {
            return -1;
        }
        if (!vx_shape_same(left, right)) {
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH,
                                 "operator inputs",
                                 "a and b must have exactly equal shapes.");
        }
        return vx_shape_clone_output(candidate, left->shape, left->rank,
                                     VX_DTYPE_F32, NULL, error);
    }
    return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                         "operator", "has no implemented canonical contract.");
}

int vx_shape_contract_infer(const char* operator_name,
                            const VxConcreteShapeRequest* request,
                            VxConcreteShapeResult* result,
                            VxShapeContractError* error) {
    const VxGeneratedShapeContractRoute* route;
    VxShapePortSpec ports;
    VxConcreteShapeResult candidate;
    vx_shape_error_reset(error);
    memset(&candidate, 0, sizeof(candidate));
    if (!request || !result) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST,
                             "shape inference request", "must be an object.");
    }
    if (!operator_name || !operator_name[0]) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                             "operator",
                             "must be a non-empty registered operator name.");
    }
    route = vx_shape_implemented_route(operator_name);
    if (!route) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR,
                             "operator",
                             "has no implemented canonical shape contract.");
    }
    ports = vx_shape_ports(route->operator_kind);
    if ((route->operator_kind == VX_OP_CONCAT
             ? vx_shape_validate_concat_inputs(request, error)
             : vx_shape_validate_inputs(request, ports, error)) ||
        vx_shape_infer_kind(route->operator_kind, request, &candidate, error)) {
        vx_shape_contract_result_clear(&candidate);
        return -1;
    }
    candidate.shape_function_id = route->shape_function_id;
    vx_shape_contract_result_clear(result);
    *result = candidate;
    return 0;
}
