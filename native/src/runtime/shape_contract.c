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

#include "shape_contract_spatial.inc"
#include "shape_contract_structural.inc"
#include "shape_contract_attention.inc"
#include "shape_contract_quantized.inc"
#include "shape_contract_extended.inc"

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
    if (vx_shape_structural_kind(kind)) {
        return vx_shape_infer_structural_kind(
            kind, request, candidate, error);
    }
    if (kind == VX_OP_SDPA || kind == VX_OP_CROSS_SDPA || kind == VX_OP_ROPE) {
        return vx_shape_infer_attention_kind(kind, request, candidate, error);
    }
    switch (kind) {
        case VX_OP_MOE_ROUTER:
            return vx_shape_infer_extended_moe_router(request, candidate, error);
        case VX_OP_MOE_LINEAR:
            return vx_shape_infer_extended_moe_linear(request, candidate, error);
        case VX_OP_CROSS_ATTENTION:
            return vx_shape_infer_extended_cross_attention(request, candidate, error);
        case VX_OP_BATCH_NORM_2D:
            return vx_shape_infer_extended_batch_norm2d(request, candidate, error);
        case VX_OP_INTERPOLATE_1D:
            return vx_shape_infer_extended_interpolate1d(request, candidate, error);
        case VX_OP_NOT:
            return vx_shape_infer_extended_not(request, candidate, error);
        case VX_OP_MASK:
            return vx_shape_infer_extended_mask(request, candidate, error);
        case VX_OP_BROADCAST:
            return vx_shape_infer_expand(request, candidate, error);
        case VX_OP_CONCAT2:
            return vx_shape_infer_extended_concat2(request, candidate, error);
        case VX_OP_REQUANTIZE_LINEAR:
            return vx_shape_infer_extended_requantize(request, candidate, error);
        case VX_OP_SSM_SCAN:
        case VX_OP_SELECTIVE_SCAN:
            return vx_shape_infer_extended_scan(request, candidate, error);
        case VX_OP_SPATIAL_SOFTARGMAX_Y:
        case VX_OP_MEAN_HEIGHT:
        case VX_OP_PROFILE_X:
        case VX_OP_PROFILE_Y:
            return vx_shape_infer_extended_vision_profile(
                kind, request, candidate, error);
        case VX_OP_DROPOUT:
            return vx_shape_infer_extended_dropout(request, candidate, error);
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
