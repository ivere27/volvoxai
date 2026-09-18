#include "shape_contract.h"

#include "shape_domain_contract.h"

#include "generated/kernel_registry.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VX_SHAPE_RESULT_STORAGE_HEAP = 0u,
    VX_SHAPE_RESULT_STORAGE_SCRATCH = 1u
};

typedef struct VxShapeScratchArena {
    unsigned char* base;
    size_t cursor;
    size_t capacity;
    size_t required;
    int failed;
} VxShapeScratchArena;

/* Every internal candidate is embedded first so existing operator helpers can
 * keep accepting VxConcreteShapeResult*. Allocator state stays outside the
 * owning public result layout and therefore does not change its private C ABI. */
typedef struct VxShapeResultBuilder {
    VxConcreteShapeResult result;
    uint32_t storage_kind;
    VxShapeScratchArena* scratch_arena;
} VxShapeResultBuilder;

static VxShapeResultBuilder* vx_shape_result_builder(
        VxConcreteShapeResult* result) {
    return (VxShapeResultBuilder*)result;
}

static void* vx_shape_result_allocate_bytes(VxConcreteShapeResult* result,
                                            size_t count,
                                            size_t item_bytes,
                                            int clear) {
    size_t bytes;
    VxShapeResultBuilder* builder;
    if (!result || (count && item_bytes > SIZE_MAX / count)) return NULL;
    builder = vx_shape_result_builder(result);
    bytes = count * item_bytes;
    if (builder->storage_kind == VX_SHAPE_RESULT_STORAGE_SCRATCH) {
        VxShapeScratchArena* arena = builder->scratch_arena;
        size_t aligned;
        size_t end;
        if (!arena) return NULL;
        if (!bytes) bytes = 1u;
        if (arena->cursor > SIZE_MAX - 15u) {
            arena->failed = 1;
            arena->required = SIZE_MAX;
            return NULL;
        }
        aligned = (arena->cursor + 15u) & ~(size_t)15u;
        if (bytes > SIZE_MAX - aligned) {
            arena->failed = 1;
            arena->required = SIZE_MAX;
            return NULL;
        }
        end = aligned + bytes;
        if (end > arena->required) arena->required = end;
        if (end > arena->capacity || !arena->base) {
            arena->failed = 1;
            return NULL;
        }
        arena->cursor = end;
        if (clear) memset(arena->base + aligned, 0, bytes);
        return arena->base + aligned;
    }
    return clear ? calloc(count, item_bytes) : malloc(bytes);
}

static void vx_shape_result_release_bytes(VxConcreteShapeResult* result,
                                          void* pointer) {
    if (!result || vx_shape_result_builder(result)->storage_kind ==
                       VX_SHAPE_RESULT_STORAGE_SCRATCH) {
        return;
    }
    free(pointer);
}


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

VX_SHAPE_CONTRACT_API
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

static const VxGeneratedShapeContractRoute* vx_shape_implemented_route(
        const char* operator_name) {
    const VxGeneratedShapeContractRoute* route =
        vx_kernel_shape_contract_find(vx_operator_kind_from_name(operator_name));
    /* The generated registry owns the implemented set: a canonical
     * classification is exactly the set of operators with a shape rule. A
     * hand-written second list would silently reject a newly generated
     * operator. */
    if (!route ||
        route->classification != VX_SHAPE_CONTRACT_CLASSIFICATION_CANONICAL) {
        return NULL;
    }
    return route;
}

VX_SHAPE_CONTRACT_API
const char* vx_shape_contract_function_id(const char* operator_name) {
    const VxGeneratedShapeContractRoute* route =
        vx_shape_implemented_route(operator_name);
    return route ? route->shape_function_id : NULL;
}

VX_SHAPE_CONTRACT_API
VxShapeDeclaredOutputMode vx_shape_contract_declared_output_mode(
        VxOperatorKind operator_kind) {
    switch (operator_kind) {
        case VX_OP_RESHAPE:
        case VX_OP_EXPAND:
        case VX_OP_BROADCAST:
        case VX_OP_QUANTIZE_LINEAR:
            return VX_SHAPE_DECLARED_OUTPUT_OPTIONAL_CONCRETE;
        case VX_OP_RESIZE:
        case VX_OP_RESIZE_NEAREST_2D:
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
        case VX_OP_REQUANTIZE_LINEAR:
            return VX_SHAPE_DECLARED_OUTPUT_REQUIRED_CONCRETE;
        default:
            return VX_SHAPE_DECLARED_OUTPUT_NONE;
    }
}

static void vx_shape_candidate_clear(VxConcreteShapeResult* result) {
    size_t index;
    VxShapeResultBuilder* builder;
    if (!result) return;
    builder = vx_shape_result_builder(result);
    if (builder->storage_kind == VX_SHAPE_RESULT_STORAGE_SCRATCH) {
        memset(result, 0, sizeof(*result));
        return;
    }
    for (index = 0; index < result->output_count; ++index) {
        vx_shape_result_release_bytes(
            result,
            result->owned_zero_points ? result->owned_zero_points[index] : NULL);
        vx_shape_result_release_bytes(
            result, result->owned_scales ? result->owned_scales[index] : NULL);
        vx_shape_result_release_bytes(
            result, result->owned_shapes ? result->owned_shapes[index] : NULL);
        vx_shape_result_release_bytes(
            result, result->owned_names ? result->owned_names[index] : NULL);
    }
    vx_shape_result_release_bytes(result, result->owned_zero_points);
    vx_shape_result_release_bytes(result, result->owned_scales);
    vx_shape_result_release_bytes(result, result->owned_shapes);
    vx_shape_result_release_bytes(result, result->owned_names);
    vx_shape_result_release_bytes(result, result->outputs);
    memset(result, 0, sizeof(*result));
}

VX_SHAPE_CONTRACT_API
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
    if (!candidate) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST,
                             "operator outputs", "builder state is invalid.");
    }
    memset(candidate, 0, sizeof(*candidate));
    if (!count || count > SIZE_MAX / sizeof(*candidate->outputs) ||
        count > SIZE_MAX / sizeof(*candidate->owned_names) ||
        count > SIZE_MAX / sizeof(*candidate->owned_shapes) ||
        count > SIZE_MAX / sizeof(*candidate->owned_scales) ||
        count > SIZE_MAX / sizeof(*candidate->owned_zero_points)) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator outputs", "output count is too large.");
    }
    candidate->outputs = (VxShapeNamedTensor*)vx_shape_result_allocate_bytes(
        candidate, count, sizeof(*candidate->outputs), 1);
    candidate->owned_names = (char**)vx_shape_result_allocate_bytes(
        candidate, count, sizeof(*candidate->owned_names), 1);
    candidate->owned_shapes = (uint64_t**)vx_shape_result_allocate_bytes(
        candidate, count, sizeof(*candidate->owned_shapes), 1);
    candidate->owned_scales = (float**)vx_shape_result_allocate_bytes(
        candidate, count, sizeof(*candidate->owned_scales), 1);
    candidate->owned_zero_points = (int32_t**)vx_shape_result_allocate_bytes(
        candidate, count, sizeof(*candidate->owned_zero_points), 1);
    if (!candidate->outputs || !candidate->owned_names ||
        !candidate->owned_shapes || !candidate->owned_scales ||
        !candidate->owned_zero_points) {
        candidate->output_count = count;
        vx_shape_candidate_clear(candidate);
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
    candidate->owned_names[output_index] =
        (char*)vx_shape_result_allocate_bytes(
            candidate, name_length + 1u, sizeof(char), 0);
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
            (uint64_t*)vx_shape_result_allocate_bytes(
                candidate, rank,
                sizeof(*candidate->owned_shapes[output_index]), 0);
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
        candidate->owned_scales[output_index] =
            (float*)vx_shape_result_allocate_bytes(
                candidate, quantization->count,
                sizeof(*candidate->owned_scales[output_index]), 0);
        candidate->owned_zero_points[output_index] =
            (int32_t*)vx_shape_result_allocate_bytes(
                candidate, quantization->count,
                sizeof(*candidate->owned_zero_points[output_index]), 0);
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
    /* vx_sd_set_output proves this descriptor before the proof is returned. */
    candidate->outputs[output_index].name = candidate->owned_names[output_index];
    candidate->outputs[output_index].descriptor = descriptor;
    return 0;
}

/* ---- Unified shape evaluator projection ---------------------------------
 *
 * shape_domain_contract.c owns exactly one rule body per operator. A concrete
 * dimension is the symbolic dimension FIXED(n) over an empty symbol
 * environment, so concrete inference is that same evaluator run on a fully
 * fixed request. This projects the concrete request into the domain evaluator
 * and the returned proof back into the owning concrete result.
 *
 * The domain evaluator sizes its output slots from the declared outputs, so a
 * caller must always supply output names and ranks. `use_declared_outputs`
 * still distinguishes a declaration that is semantic input (Reshape, Expand,
 * quantized outputs) from one that is only an assertion to check. */

/* The domain evaluator reports a port failure two ways. A per-port failure
 * carries the port index with subindex 0 for an input and 1 for a declared
 * output. An arity failure instead carries the request's own two counts, which
 * the caller can recognise because it supplied them. A rule always declares at
 * least one output, so an arity failure with no declared output is an output
 * port failure. */
static VxShapeContractErrorCode vx_shape_domain_error_code(
        const VxShapeDomainError* error,
        const VxConcreteShapeRequest* request) {
    VxShapeDomainErrorCode code = error->code;
    uint32_t subindex = error->subindex;
    /* The evaluator validates the whole request before it reaches a rule and
     * refuses a malformed one without naming a field. This projection always
     * supplies the pointers and counts that branch also guards, so the only
     * reachable cause is a descriptor the request could not carry. */
    if (code == VX_SHAPE_DOMAIN_ERROR_NONE)
        return VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR;
    if (code == VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS &&
        error->index == (uint32_t)request->input_count &&
        subindex == (uint32_t)request->declared_output_count)
        return request->declared_output_count
            ? VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS
            : VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS;
    switch (code) {
        case VX_SHAPE_DOMAIN_ERROR_UNSUPPORTED_OPERATOR:
            return VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR;
        case VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS:
            return subindex == 1u ? VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS
                                  : VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS;
        case VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE:
            return VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE;
        case VX_SHAPE_DOMAIN_ERROR_INVALID_RANK:
            return VX_SHAPE_CONTRACT_ERROR_INVALID_RANK;
        case VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS:
            return VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS;
        case VX_SHAPE_DOMAIN_ERROR_INVALID_DIMENSION:
            return VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR;
        case VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH:
            return VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH;
        /* Every dimension of a concrete request is FIXED, so a relation the
         * evaluator cannot prove is a relation that does not hold. Reserve
         * INVALID_DOMAIN for requests that carry an unresolved symbol. */
        case VX_SHAPE_DOMAIN_ERROR_UNPROVABLE:
            return VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH;
        case VX_SHAPE_DOMAIN_ERROR_INVALID_QUANTIZATION:
            return VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION;
        case VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW:
            return VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR;
        default:
            return VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST;
    }
}

static VxShapeDomainString vx_shape_domain_name(const char* name) {
    VxShapeDomainString value;
    value.bytes = (const uint8_t*)(name ? name : "");
    value.length = (uint32_t)(name ? strlen(name) : 0u);
    return value;
}

static uint32_t vx_shape_float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float vx_shape_bits_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

typedef struct VxShapeDomainProjection {
    VxShapeDomainNamedTensor* inputs;
    VxShapeDomainNamedTensor* outputs;
    VxShapeDomainDimension* dimensions;
    uint32_t* scale_bits;
    VxShapeDomainParam* params;
    VxShapeDomainParamValue* values;
    uint64_t* shape;
    unsigned char* scratch;
    size_t scratch_bytes;
    size_t dimension_cursor;
    size_t scale_cursor;
    size_t value_cursor;
} VxShapeDomainProjection;

/* Every projection buffer comes from the builder's own storage so heap and
 * caller-scratch modes behave identically. */
static void vx_shape_projection_release(VxConcreteShapeResult* owner,
                                        VxShapeDomainProjection* projection) {
    vx_shape_result_release_bytes(owner, projection->inputs);
    vx_shape_result_release_bytes(owner, projection->outputs);
    vx_shape_result_release_bytes(owner, projection->dimensions);
    vx_shape_result_release_bytes(owner, projection->scale_bits);
    vx_shape_result_release_bytes(owner, projection->params);
    vx_shape_result_release_bytes(owner, projection->values);
    vx_shape_result_release_bytes(owner, projection->shape);
    vx_shape_result_release_bytes(owner, projection->scratch);
    memset(projection, 0, sizeof(*projection));
}

static int vx_shape_project_tensor(VxShapeDomainProjection* projection,
                                   VxShapeDomainNamedTensor* target,
                                   const VxShapeNamedTensor* source,
                                   uint32_t tensor_index) {
    const VxShapeTensorDescriptor* descriptor = &source->descriptor;
    VxShapeDomainDimension* dimensions =
        projection->dimensions + projection->dimension_cursor;
    uint32_t* scales = projection->scale_bits + projection->scale_cursor;
    size_t axis;
    size_t index;
    memset(target, 0, sizeof(*target));
    target->name = vx_shape_domain_name(source->name);
    target->tensor_index = tensor_index;
    target->descriptor.dtype = (int32_t)descriptor->dtype;
    target->descriptor.rank = (uint32_t)descriptor->rank;
    for (axis = 0; axis < descriptor->rank; axis++) {
        dimensions[axis].kind = VX_SHAPE_DOMAIN_DIMENSION_FIXED;
        dimensions[axis].symbol_index = VX_SHAPE_DOMAIN_INDEX_NONE;
        dimensions[axis].value = (int64_t)descriptor->shape[axis];
    }
    target->descriptor.dimensions = dimensions;
    projection->dimension_cursor += descriptor->rank;
    target->descriptor.quantization.scheme = (int32_t)descriptor->quantization.scheme;
    target->descriptor.quantization.scale_bits =
        vx_shape_float_bits(descriptor->quantization.scale);
    target->descriptor.quantization.zero_point = descriptor->quantization.zero_point;
    target->descriptor.quantization.axis = (uint32_t)descriptor->quantization.axis;
    target->descriptor.quantization.count = (uint32_t)descriptor->quantization.count;
    if (descriptor->quantization.count) {
        if (!descriptor->quantization.scales ||
            !descriptor->quantization.zero_points) return -1;
        for (index = 0; index < descriptor->quantization.count; index++)
            scales[index] = vx_shape_float_bits(descriptor->quantization.scales[index]);
        target->descriptor.quantization.scales = scales;
        target->descriptor.quantization.zero_points =
            descriptor->quantization.zero_points;
        projection->scale_cursor += descriptor->quantization.count;
    }
    return 0;
}

static int vx_shape_project_params(VxShapeDomainProjection* projection,
                                   const VxConcreteShapeRequest* request) {
    size_t index;
    for (index = 0; index < request->param_count; index++) {
        const VxShapeParam* source = &request->params[index];
        VxShapeDomainParam* target = &projection->params[index];
        memset(target, 0, sizeof(*target));
        target->name = vx_shape_domain_name(source->name);
        switch (source->kind) {
            case VX_SHAPE_PARAM_NUMBER:
                target->kind = VX_SHAPE_DOMAIN_PARAM_NUMBER;
                target->number = source->value.number;
                break;
            case VX_SHAPE_PARAM_BOOLEAN:
                target->kind = VX_SHAPE_DOMAIN_PARAM_BOOLEAN;
                target->boolean = source->value.boolean;
                break;
            case VX_SHAPE_PARAM_STRING:
                target->kind = VX_SHAPE_DOMAIN_PARAM_STRING;
                target->string = vx_shape_domain_name(source->value.string);
                break;
            case VX_SHAPE_PARAM_NUMBER_ARRAY: {
                VxShapeDomainParamValue* values =
                    projection->values + projection->value_cursor;
                size_t item;
                if (source->value.number_array.count &&
                    !source->value.number_array.values) return -1;
                for (item = 0; item < source->value.number_array.count; item++) {
                    values[item].kind = VX_SHAPE_DOMAIN_PARAM_VALUE_NUMBER;
                    values[item].dimension_index = VX_SHAPE_DOMAIN_INDEX_NONE;
                    values[item].number = source->value.number_array.values[item];
                }
                target->kind = VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY;
                target->values = values;
                target->value_count = (uint32_t)source->value.number_array.count;
                projection->value_cursor += source->value.number_array.count;
                break;
            }
            default:
                return -1;
        }
    }
    return 0;
}

/* The domain evaluator seeds its output slots from the declared outputs this
 * projection built, so every returned name points back at a caller-owned
 * NUL-terminated string. Resolve it by matching rather than by position so a
 * future reordering inside the evaluator cannot silently mislabel an output. */
static const char* vx_shape_projected_name(
        const VxConcreteShapeRequest* request,
        const VxShapeDomainNamedTensor* output) {
    size_t index;
    for (index = 0; index < request->declared_output_count; index++) {
        const char* name = request->declared_outputs[index].name;
        /* memcmp is outside the freestanding wasm32 libc this file also
         * compiles against; strncmp with an explicit length check is not. */
        if (name && strlen(name) == output->name.length &&
            !strncmp(name, (const char*)output->name.bytes,
                     output->name.length))
            return name;
    }
    return NULL;
}

/* Most requests need far less than this; the corpus peak is 360 bytes. A short
 * first attempt keeps the common path to a single allocation and the evaluator
 * reports its exact requirement when the hint is not enough. */
#define VX_SHAPE_DOMAIN_SCRATCH_HINT 1024u

static int vx_shape_infer_via_domain(VxOperatorKind operator_kind,
                                     const char* operator_name,
                                     const VxConcreteShapeRequest* request,
                                     VxConcreteShapeResult* candidate,
                                     VxShapeContractError* error) {
    VxShapeDomainProjection projection;
    VxShapeDomainProofRequest domain_request;
    VxShapeDomainProofResult domain_result;
    VxShapeDomainError domain_error;
    size_t axis_total = 0;
    size_t scale_total = 0;
    size_t value_total = 0;
    size_t max_rank = 0;
    size_t index;
    uint32_t required = 0;
    int32_t status;

    memset(&projection, 0, sizeof(projection));
    for (index = 0; index < request->input_count; index++) {
        const VxShapeTensorDescriptor* descriptor =
            &request->inputs[index].descriptor;
        axis_total += descriptor->rank;
        scale_total += descriptor->quantization.count;
        if (descriptor->rank > max_rank) max_rank = descriptor->rank;
    }
    for (index = 0; index < request->declared_output_count; index++) {
        const VxShapeTensorDescriptor* descriptor =
            &request->declared_outputs[index].descriptor;
        axis_total += descriptor->rank;
        scale_total += descriptor->quantization.count;
        if (descriptor->rank > max_rank) max_rank = descriptor->rank;
    }
    for (index = 0; index < request->param_count; index++)
        if (request->params[index].kind == VX_SHAPE_PARAM_NUMBER_ARRAY)
            value_total += request->params[index].value.number_array.count;

    projection.inputs = (VxShapeDomainNamedTensor*)vx_shape_result_allocate_bytes(
        candidate, request->input_count + 1u, sizeof(*projection.inputs), 1);
    projection.outputs = (VxShapeDomainNamedTensor*)vx_shape_result_allocate_bytes(
        candidate, request->declared_output_count + 1u,
        sizeof(*projection.outputs), 1);
    projection.dimensions = (VxShapeDomainDimension*)vx_shape_result_allocate_bytes(
        candidate, axis_total + 1u, sizeof(*projection.dimensions), 1);
    projection.scale_bits = (uint32_t*)vx_shape_result_allocate_bytes(
        candidate, scale_total + 1u, sizeof(*projection.scale_bits), 1);
    projection.params = (VxShapeDomainParam*)vx_shape_result_allocate_bytes(
        candidate, request->param_count + 1u, sizeof(*projection.params), 1);
    projection.values = (VxShapeDomainParamValue*)vx_shape_result_allocate_bytes(
        candidate, value_total + 1u, sizeof(*projection.values), 1);
    projection.shape = (uint64_t*)vx_shape_result_allocate_bytes(
        candidate, max_rank + 1u, sizeof(*projection.shape), 1);
    projection.scratch_bytes = VX_SHAPE_DOMAIN_SCRATCH_HINT;
    projection.scratch = (unsigned char*)vx_shape_result_allocate_bytes(
        candidate, projection.scratch_bytes, 1u, 1);
    if (!projection.inputs || !projection.outputs || !projection.dimensions ||
        !projection.scale_bits || !projection.params || !projection.values ||
        !projection.shape || !projection.scratch) {
        vx_shape_projection_release(candidate, &projection);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                             "operator shape projection", "allocation failed.");
    }

    for (index = 0; index < request->input_count; index++)
        if (vx_shape_project_tensor(&projection, &projection.inputs[index],
                                    &request->inputs[index], (uint32_t)index)) {
            vx_shape_projection_release(candidate, &projection);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 "operator inputs",
                                 "quantization arrays are incomplete.");
        }
    for (index = 0; index < request->declared_output_count; index++)
        if (vx_shape_project_tensor(&projection, &projection.outputs[index],
                                    &request->declared_outputs[index],
                                    (uint32_t)index)) {
            vx_shape_projection_release(candidate, &projection);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION,
                                 "operator outputs",
                                 "quantization arrays are incomplete.");
        }
    if (vx_shape_project_params(&projection, request)) {
        vx_shape_projection_release(candidate, &projection);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS,
                             "operator params", "has an unsupported value kind.");
    }

    memset(&domain_request, 0, sizeof(domain_request));
    domain_request.operator_kind = operator_kind;
    domain_request.operator_name = vx_shape_domain_name(operator_name);
    domain_request.inputs = projection.inputs;
    domain_request.input_count = (uint32_t)request->input_count;
    domain_request.declared_outputs = projection.outputs;
    domain_request.declared_output_count = (uint32_t)request->declared_output_count;
    domain_request.use_declared_outputs =
        vx_shape_contract_declared_output_mode(operator_kind) !=
        VX_SHAPE_DECLARED_OUTPUT_NONE;
    domain_request.params = request->param_count ? projection.params : NULL;
    domain_request.param_count = (uint32_t)request->param_count;

    memset(&domain_result, 0, sizeof(domain_result));
    memset(&domain_error, 0, sizeof(domain_error));
    status = vx_shape_domain_contract_prove(
        &domain_request, &domain_result, &domain_error, projection.scratch,
        (uint32_t)projection.scratch_bytes, &required);
    if (status == VX_SHAPE_DOMAIN_STATUS_SCRATCH_TOO_SMALL &&
        required > projection.scratch_bytes) {
        vx_shape_result_release_bytes(candidate, projection.scratch);
        projection.scratch_bytes = required;
        projection.scratch = (unsigned char*)vx_shape_result_allocate_bytes(
            candidate, projection.scratch_bytes, 1u, 1);
        if (!projection.scratch) {
            vx_shape_projection_release(candidate, &projection);
            return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                 "operator shape projection",
                                 "scratch allocation failed.");
        }
        memset(&domain_result, 0, sizeof(domain_result));
        memset(&domain_error, 0, sizeof(domain_error));
        status = vx_shape_domain_contract_prove(
            &domain_request, &domain_result, &domain_error, projection.scratch,
            (uint32_t)projection.scratch_bytes, &required);
    }
    if (status != VX_SHAPE_DOMAIN_STATUS_OK) {
        VxShapeContractErrorCode code =
            status == VX_SHAPE_DOMAIN_STATUS_SCRATCH_TOO_SMALL
                ? VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY
                : vx_shape_domain_error_code(&domain_error, request);
        vx_shape_projection_release(candidate, &projection);
        return vx_shape_fail(error, code, "operator",
                             "does not satisfy its shape contract.");
    }

    if (vx_shape_result_allocate(candidate, domain_result.output_count, error)) {
        status = -1;
        goto done;
    }
    for (index = 0; index < domain_result.output_count; index++) {
        const VxShapeDomainNamedTensor* output = &domain_result.outputs[index];
        const VxShapeDomainQuantization* quantization =
            &output->descriptor.quantization;
        VxShapeQuantization concrete_quantization;
        const char* name = vx_shape_projected_name(request, output);
        size_t axis;
        if (!name) {
            status = vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS,
                                 "operator outputs",
                                 "the proof named an undeclared output port.");
            goto done;
        }
        for (axis = 0; axis < output->descriptor.rank; axis++) {
            if (output->descriptor.dimensions[axis].kind !=
                    VX_SHAPE_DOMAIN_DIMENSION_FIXED ||
                output->descriptor.dimensions[axis].value < 0) {
                status = vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR,
                                     "operator outputs",
                                     "a concrete request produced a non-fixed extent.");
                goto done;
            }
            projection.shape[axis] =
                (uint64_t)output->descriptor.dimensions[axis].value;
        }
        memset(&concrete_quantization, 0, sizeof(concrete_quantization));
        concrete_quantization.scheme =
            (VxShapeQuantizationScheme)quantization->scheme;
        concrete_quantization.scale = vx_shape_bits_float(quantization->scale_bits);
        concrete_quantization.zero_point = quantization->zero_point;
        concrete_quantization.axis = quantization->axis;
        concrete_quantization.count = quantization->count;
        concrete_quantization.zero_points = quantization->zero_points;
        if (quantization->count) {
            float* scales = (float*)vx_shape_result_allocate_bytes(
                candidate, quantization->count, sizeof(*scales), 0);
            size_t item;
            if (!scales) {
                status = vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY,
                                     "operator outputs", "allocation failed.");
                goto done;
            }
            for (item = 0; item < quantization->count; item++)
                scales[item] = vx_shape_bits_float(quantization->scales[item]);
            concrete_quantization.scales = scales;
            if (vx_shape_clone_output_at(
                    candidate, index, name,
                    projection.shape, output->descriptor.rank,
                    (VxDataType)output->descriptor.dtype,
                    &concrete_quantization, error)) {
                vx_shape_result_release_bytes(candidate, scales);
                status = -1;
                goto done;
            }
            vx_shape_result_release_bytes(candidate, scales);
            continue;
        }
        if (vx_shape_clone_output_at(
                candidate, index, name,
                projection.shape, output->descriptor.rank,
                (VxDataType)output->descriptor.dtype,
                quantization->scheme ? &concrete_quantization : NULL, error)) {
            status = -1;
            goto done;
        }
    }
    status = 0;
done:
    vx_shape_projection_release(candidate, &projection);
    return status;
}

static int vx_shape_contract_build_candidate(
        const char* operator_name,
        const VxConcreteShapeRequest* request,
        VxShapeResultBuilder* builder,
        VxShapeContractError* error,
        VxShapeScratchArena* scratch_arena) {
    const VxGeneratedShapeContractRoute* route;
    VxConcreteShapeResult* candidate;
    vx_shape_error_reset(error);
    if (!builder) {
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST,
                             "shape inference result", "must be an object.");
    }
    memset(builder, 0, sizeof(*builder));
    builder->storage_kind = scratch_arena
        ? VX_SHAPE_RESULT_STORAGE_SCRATCH : VX_SHAPE_RESULT_STORAGE_HEAP;
    builder->scratch_arena = scratch_arena;
    candidate = &builder->result;
    if (!request) {
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
    if (vx_shape_infer_via_domain(route->operator_kind, operator_name,
                                  request, candidate, error)) {
        vx_shape_candidate_clear(candidate);
        return -1;
    }
    candidate->shape_function_id = route->shape_function_id;
    return 0;
}

VX_SHAPE_CONTRACT_API
int vx_shape_contract_infer(const char* operator_name,
                            const VxConcreteShapeRequest* request,
                            VxConcreteShapeResult* result,
                            VxShapeContractError* error) {
    VxShapeResultBuilder builder;
    int status;
    if (!result) {
        vx_shape_error_reset(error);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST,
                             "shape inference result", "must be an object.");
    }
    status = vx_shape_contract_build_candidate(
        operator_name, request, &builder, error, NULL);
    if (status) return status;
    vx_shape_contract_result_clear(result);
    *result = builder.result;
    return 0;
}

VX_SHAPE_CONTRACT_API
int vx_shape_contract_infer_with_scratch(
        const char* operator_name,
        const VxConcreteShapeRequest* request,
        VxBorrowedConcreteShapeResult* result,
        VxShapeContractError* error,
        void* scratch,
        size_t scratch_bytes,
        size_t* required_scratch_bytes) {
    VxShapeScratchArena arena;
    VxShapeResultBuilder builder;
    int status;
    if (required_scratch_bytes) *required_scratch_bytes = 0u;
    if (result) memset(result, 0, sizeof(*result));
    if ((!scratch_bytes && scratch) ||
        (scratch_bytes &&
         (!scratch || (uintptr_t)scratch % (uintptr_t)16u != 0u ||
          scratch_bytes > UINTPTR_MAX - (uintptr_t)scratch))) {
        vx_shape_error_reset(error);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST,
                             "shape inference scratch",
                             "must be a valid 16-byte-aligned caller-owned range.");
    }
    memset(&arena, 0, sizeof(arena));
    arena.base = (unsigned char*)scratch;
    arena.capacity = scratch_bytes;
    if (!result) {
        vx_shape_error_reset(error);
        return vx_shape_fail(error, VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST,
                             "shape inference result", "must be an object.");
    }
    status = vx_shape_contract_build_candidate(
        operator_name, request, &builder, error, &arena);
    if (required_scratch_bytes) *required_scratch_bytes = arena.required;
    if (!status) {
        result->shape_function_id = builder.result.shape_function_id;
        result->outputs = builder.result.outputs;
        result->output_count = builder.result.output_count;
    }
    return status;
}
