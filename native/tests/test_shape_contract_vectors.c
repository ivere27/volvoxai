#include "shape_contract.h"

#include "cJSON.h"
#include "json_validation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct OwnedTensor {
    VxShapeNamedTensor named;
    uint64_t* shape;
    float* scales;
    int32_t* zero_points;
} OwnedTensor;

typedef struct OwnedRequest {
    VxConcreteShapeRequest request;
    OwnedTensor* inputs;
    VxShapeParam* params;
    double** param_number_arrays;
    OwnedTensor* declared_outputs;
} OwnedRequest;

typedef struct OperatorCoverage {
    const char* name;
    int success;
    int failure;
} OperatorCoverage;

#define VX_ARRAY_COUNT(values) (sizeof(values) / sizeof((values)[0]))

static int fail_at(const char* file, int line, const char* message) {
    fprintf(stderr, "FAIL %s:%d: %s\n", file, line, message);
    return -1;
}

#define REQUIRE(condition, message) \
    do { if (!(condition)) return fail_at(__FILE__, __LINE__, (message)); } while (0)

static size_t object_size(const cJSON* object) {
    size_t count = 0;
    const cJSON* item;
    cJSON_ArrayForEach(item, object) count++;
    return count;
}

static int json_safe_integer(const cJSON* value, double minimum, double maximum) {
    double number;
    if (!cJSON_IsNumber(value)) return 0;
    number = value->valuedouble;
    return isfinite(number) && trunc(number) == number &&
           number >= minimum && number <= maximum;
}

static VxDataType parse_dtype(const cJSON* value) {
    if (!cJSON_IsString(value) || !value->valuestring) return VX_DTYPE_UNSPECIFIED;
    if (!strcmp(value->valuestring, "float32")) return VX_DTYPE_F32;
    if (!strcmp(value->valuestring, "int32")) return VX_DTYPE_I32;
    if (!strcmp(value->valuestring, "int8")) return VX_DTYPE_I8;
    if (!strcmp(value->valuestring, "uint8")) return VX_DTYPE_U8;
    return VX_DTYPE_UNSPECIFIED;
}

static int name_in_fields(const char* name,
                          const char* const* fields,
                          size_t field_count) {
    size_t index;
    if (!name) return 0;
    for (index = 0; index < field_count; ++index) {
        if (!strcmp(name, fields[index])) return 1;
    }
    return 0;
}

static int object_has_only_fields(const cJSON* object,
                                  const char* const* fields,
                                  size_t field_count) {
    const cJSON* item;
    if (!cJSON_IsObject(object)) return 0;
    cJSON_ArrayForEach(item, object) {
        if (!name_in_fields(item->string, fields, field_count)) return 0;
    }
    return 1;
}

static int object_has_exact_fields(const cJSON* object,
                                   const char* const* fields,
                                   size_t field_count) {
    size_t index;
    if (!object_has_only_fields(object, fields, field_count) ||
        object_size(object) != field_count) return 0;
    for (index = 0; index < field_count; ++index) {
        if (!cJSON_GetObjectItemCaseSensitive(object, fields[index])) return 0;
    }
    return 1;
}

static int has_forbidden_value_field(const cJSON* value) {
    static const char* const FORBIDDEN[] = {
        "buffer", "data", "value", "values"
    };
    const cJSON* child;
    if (!cJSON_IsObject(value) && !cJSON_IsArray(value)) return 0;
    cJSON_ArrayForEach(child, value) {
        if (cJSON_IsObject(value) &&
            name_in_fields(child->string, FORBIDDEN, VX_ARRAY_COUNT(FORBIDDEN))) {
            return 1;
        }
        if (has_forbidden_value_field(child)) return 1;
    }
    return 0;
}

static int validate_string_array(const cJSON* array) {
    const cJSON* item;
    const cJSON* other;
    if (!cJSON_IsArray(array) || cJSON_GetArraySize(array) <= 0) return -1;
    cJSON_ArrayForEach(item, array) {
        if (!cJSON_IsString(item) || !item->valuestring || !item->valuestring[0]) {
            return -1;
        }
        for (other = item->next; other; other = other->next) {
            if (cJSON_IsString(other) && other->valuestring &&
                !strcmp(item->valuestring, other->valuestring)) return -1;
        }
    }
    return 0;
}

static int string_array_contains(const cJSON* array, const char* value) {
    const cJSON* item;
    if (!cJSON_IsArray(array) || !value) return 0;
    cJSON_ArrayForEach(item, array) {
        if (cJSON_IsString(item) && item->valuestring &&
            !strcmp(item->valuestring, value)) return 1;
    }
    return 0;
}

static int validate_descriptor_schema(const cJSON* descriptor) {
    static const char* const DESCRIPTOR_FIELDS[] = {
        "shape", "dtype", "quantization"
    };
    static const char* const PER_TENSOR_FIELDS[] = {
        "scheme", "scale", "zero_point"
    };
    static const char* const PER_AXIS_FIELDS[] = {
        "scheme", "axis", "scales", "zero_points"
    };
    const cJSON* shape;
    const cJSON* dtype;
    const cJSON* quantization;
    const cJSON* scheme;
    const cJSON* item;
    if (!object_has_only_fields(descriptor, DESCRIPTOR_FIELDS,
                                VX_ARRAY_COUNT(DESCRIPTOR_FIELDS))) return -1;
    shape = cJSON_GetObjectItemCaseSensitive(descriptor, "shape");
    dtype = cJSON_GetObjectItemCaseSensitive(descriptor, "dtype");
    if (!cJSON_IsArray(shape) || parse_dtype(dtype) == VX_DTYPE_UNSPECIFIED) {
        return -1;
    }
    cJSON_ArrayForEach(item, shape) {
        if (!json_safe_integer(item, 1.0,
                               (double)VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER)) {
            return -1;
        }
    }
    quantization = cJSON_GetObjectItemCaseSensitive(descriptor, "quantization");
    if (!quantization) return 0;
    if (!cJSON_IsObject(quantization)) return -1;
    scheme = cJSON_GetObjectItemCaseSensitive(quantization, "scheme");
    if (!cJSON_IsString(scheme) || !scheme->valuestring) return -1;
    if (!strcmp(scheme->valuestring, "per_tensor")) {
        return object_has_exact_fields(quantization, PER_TENSOR_FIELDS,
                                       VX_ARRAY_COUNT(PER_TENSOR_FIELDS)) ? 0 : -1;
    }
    if (strcmp(scheme->valuestring, "per_axis") ||
        !object_has_exact_fields(quantization, PER_AXIS_FIELDS,
                                 VX_ARRAY_COUNT(PER_AXIS_FIELDS))) return -1;
    if (!cJSON_IsArray(cJSON_GetObjectItemCaseSensitive(quantization, "scales")) ||
        !cJSON_IsArray(cJSON_GetObjectItemCaseSensitive(quantization,
                                                        "zero_points"))) return -1;
    return 0;
}

static int validate_request_schema(const cJSON* request) {
    static const char* const REQUEST_FIELDS[] = {
        "inputs", "params", "declaredOutputs"
    };
    static const char* const OUT_FIELD[] = {"out"};
    const cJSON* inputs;
    const cJSON* params;
    const cJSON* declared;
    const cJSON* item;
    if (!object_has_only_fields(request, REQUEST_FIELDS,
                                VX_ARRAY_COUNT(REQUEST_FIELDS))) return -1;
    inputs = cJSON_GetObjectItemCaseSensitive(request, "inputs");
    if (!cJSON_IsObject(inputs) || object_size(inputs) == 0) return -1;
    cJSON_ArrayForEach(item, inputs) {
        if (!item->string || !item->string[0] || validate_descriptor_schema(item)) {
            return -1;
        }
    }
    params = cJSON_GetObjectItemCaseSensitive(request, "params");
    if (params && !cJSON_IsObject(params)) return -1;
    declared = cJSON_GetObjectItemCaseSensitive(request, "declaredOutputs");
    if (declared &&
        (!object_has_exact_fields(declared, OUT_FIELD,
                                  VX_ARRAY_COUNT(OUT_FIELD)) ||
         validate_descriptor_schema(
             cJSON_GetObjectItemCaseSensitive(declared, "out")))) return -1;
    return has_forbidden_value_field(request) ? -1 : 0;
}

static const cJSON* find_family(const cJSON* families, const char* id) {
    const cJSON* family;
    if (!cJSON_IsArray(families) || !id) return NULL;
    cJSON_ArrayForEach(family, families) {
        const cJSON* family_id = cJSON_GetObjectItemCaseSensitive(family, "id");
        if (cJSON_IsString(family_id) && family_id->valuestring &&
            !strcmp(family_id->valuestring, id)) return family;
    }
    return NULL;
}

static size_t case_id_occurrences(const cJSON* success_cases,
                                  const cJSON* failure_cases,
                                  const char* id) {
    const cJSON* groups[] = {success_cases, failure_cases};
    size_t group;
    size_t count = 0;
    for (group = 0; group < VX_ARRAY_COUNT(groups); ++group) {
        const cJSON* vector;
        cJSON_ArrayForEach(vector, groups[group]) {
            const cJSON* candidate =
                cJSON_GetObjectItemCaseSensitive(vector, "id");
            if (cJSON_IsString(candidate) && candidate->valuestring &&
                !strcmp(candidate->valuestring, id)) count++;
        }
    }
    return count;
}

static size_t family_case_count(const cJSON* cases, const char* family_id) {
    const cJSON* vector;
    size_t count = 0;
    cJSON_ArrayForEach(vector, cases) {
        const cJSON* family = cJSON_GetObjectItemCaseSensitive(vector, "family");
        if (cJSON_IsString(family) && family->valuestring &&
            !strcmp(family->valuestring, family_id)) count++;
    }
    return count;
}

static int validate_vector_cases(const cJSON* cases,
                                 int success,
                                 const cJSON* families,
                                 const cJSON* success_cases,
                                 const cJSON* failure_cases) {
    static const char* const SUCCESS_FIELDS[] = {
        "id", "family", "operators", "request",
        "expected_shape_function_id", "expected"
    };
    static const char* const FAILURE_FIELDS[] = {
        "id", "family", "operators", "request",
        "expected_shape_function_id", "expected_error"
    };
    static const char* const ERROR_FIELDS[] = {"code", "path"};
    const char* const* fields = success ? SUCCESS_FIELDS : FAILURE_FIELDS;
    size_t field_count = success ? VX_ARRAY_COUNT(SUCCESS_FIELDS)
                                 : VX_ARRAY_COUNT(FAILURE_FIELDS);
    const cJSON* vector;
    if (!cJSON_IsArray(cases)) return -1;
    cJSON_ArrayForEach(vector, cases) {
        const cJSON* id;
        const cJSON* family_id;
        const cJSON* family;
        const cJSON* family_operators;
        const cJSON* operators;
        const cJSON* operator_name;
        const cJSON* shape_id;
        const cJSON* expected;
        if (!object_has_exact_fields(vector, fields, field_count)) return -1;
        id = cJSON_GetObjectItemCaseSensitive(vector, "id");
        family_id = cJSON_GetObjectItemCaseSensitive(vector, "family");
        operators = cJSON_GetObjectItemCaseSensitive(vector, "operators");
        shape_id = cJSON_GetObjectItemCaseSensitive(
            vector, "expected_shape_function_id");
        if (!cJSON_IsString(id) || !id->valuestring || !id->valuestring[0] ||
            case_id_occurrences(success_cases, failure_cases,
                                id->valuestring) != 1 ||
            !cJSON_IsString(family_id) || !family_id->valuestring ||
            !family_id->valuestring[0] || validate_string_array(operators) ||
            !cJSON_IsString(shape_id) || !shape_id->valuestring ||
            !shape_id->valuestring[0] ||
            validate_request_schema(
                cJSON_GetObjectItemCaseSensitive(vector, "request"))) return -1;
        family = find_family(families, family_id->valuestring);
        if (!family) return -1;
        family_operators = cJSON_GetObjectItemCaseSensitive(family, "operators");
        cJSON_ArrayForEach(operator_name, operators) {
            if (!string_array_contains(family_operators,
                                       operator_name->valuestring)) return -1;
        }
        expected = cJSON_GetObjectItemCaseSensitive(
            vector, success ? "expected" : "expected_error");
        if (success) {
            const cJSON* output;
            if (!cJSON_IsObject(expected) || object_size(expected) == 0) return -1;
            cJSON_ArrayForEach(output, expected) {
                if (!output->string || !output->string[0] ||
                    validate_descriptor_schema(output)) return -1;
            }
        } else if (!object_has_exact_fields(expected, ERROR_FIELDS,
                                            VX_ARRAY_COUNT(ERROR_FIELDS)) ||
                   !cJSON_IsString(cJSON_GetObjectItemCaseSensitive(
                       expected, "code")) ||
                   !cJSON_IsString(cJSON_GetObjectItemCaseSensitive(
                       expected, "path"))) {
            return -1;
        }
    }
    return 0;
}

static int validate_corpus_schema(const cJSON* root) {
    static const char* const ROOT_FIELDS[] = {
        "format", "families", "success_cases", "failure_cases"
    };
    static const char* const FAMILY_FIELDS[] = {"id", "operators"};
    const cJSON* format;
    const cJSON* families;
    const cJSON* success_cases;
    const cJSON* failure_cases;
    const cJSON* family;
    const cJSON* other_family;
    size_t operator_count = 0;
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !object_has_exact_fields(root, ROOT_FIELDS,
                                 VX_ARRAY_COUNT(ROOT_FIELDS))) return -1;
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    families = cJSON_GetObjectItemCaseSensitive(root, "families");
    success_cases = cJSON_GetObjectItemCaseSensitive(root, "success_cases");
    failure_cases = cJSON_GetObjectItemCaseSensitive(root, "failure_cases");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvox-operator-shape-vectors/v1") ||
        !cJSON_IsArray(families) || cJSON_GetArraySize(families) <= 0) return -1;
    cJSON_ArrayForEach(family, families) {
        const cJSON* id;
        const cJSON* operators;
        const cJSON* operator_name;
        if (!object_has_exact_fields(family, FAMILY_FIELDS,
                                     VX_ARRAY_COUNT(FAMILY_FIELDS))) return -1;
        id = cJSON_GetObjectItemCaseSensitive(family, "id");
        operators = cJSON_GetObjectItemCaseSensitive(family, "operators");
        if (!cJSON_IsString(id) || !id->valuestring || !id->valuestring[0] ||
            validate_string_array(operators)) return -1;
        for (other_family = family->next; other_family;
             other_family = other_family->next) {
            const cJSON* other_id =
                cJSON_GetObjectItemCaseSensitive(other_family, "id");
            if (cJSON_IsString(other_id) && other_id->valuestring &&
                !strcmp(id->valuestring, other_id->valuestring)) return -1;
        }
        cJSON_ArrayForEach(operator_name, operators) {
            const cJSON* candidate_family;
            size_t occurrences = 0;
            cJSON_ArrayForEach(candidate_family, families) {
                if (string_array_contains(cJSON_GetObjectItemCaseSensitive(
                        candidate_family, "operators"),
                        operator_name->valuestring)) occurrences++;
            }
            if (occurrences != 1 ||
                !vx_shape_contract_function_id(operator_name->valuestring)) return -1;
            operator_count++;
        }
        if (family_case_count(success_cases, id->valuestring) < 2 ||
            family_case_count(failure_cases, id->valuestring) < 1) return -1;
    }
    if (!operator_count ||
        validate_vector_cases(success_cases, 1, families,
                              success_cases, failure_cases) ||
        validate_vector_cases(failure_cases, 0, families,
                              success_cases, failure_cases)) return -1;
    return 0;
}

static int check_corpus_schema_rejections(const cJSON* root) {
    cJSON* copy;
    cJSON* success_cases;
    cJSON* failure_cases;
    cJSON* vector;
    cJSON* request;
    cJSON* params;
    cJSON* families;
    cJSON* first_family;
    cJSON* second_family;
    const cJSON* source_id;

    copy = cJSON_Duplicate(root, 1);
    REQUIRE(copy && cJSON_AddStringToObject(copy, "extra", "rejected"),
            "strict-root mutation allocation failed");
    REQUIRE(validate_corpus_schema(copy) != 0,
            "strict corpus schema accepted an extra root field");
    cJSON_Delete(copy);

    copy = cJSON_Duplicate(root, 1);
    REQUIRE(copy != NULL, "no-value mutation allocation failed");
    success_cases = cJSON_GetObjectItemCaseSensitive(copy, "success_cases");
    vector = cJSON_GetArrayItem(success_cases, 0);
    request = cJSON_GetObjectItemCaseSensitive(vector, "request");
    params = cJSON_AddObjectToObject(request, "params");
    REQUIRE(params && cJSON_AddNumberToObject(params, "data", 1.0),
            "no-value mutation construction failed");
    REQUIRE(validate_corpus_schema(copy) != 0,
            "strict corpus schema accepted tensor value data");
    cJSON_Delete(copy);

    copy = cJSON_Duplicate(root, 1);
    REQUIRE(copy != NULL, "duplicate-case mutation allocation failed");
    success_cases = cJSON_GetObjectItemCaseSensitive(copy, "success_cases");
    failure_cases = cJSON_GetObjectItemCaseSensitive(copy, "failure_cases");
    source_id = cJSON_GetObjectItemCaseSensitive(
        cJSON_GetArrayItem(success_cases, 0), "id");
    vector = cJSON_GetArrayItem(failure_cases, 0);
    REQUIRE(cJSON_IsString(source_id) && source_id->valuestring &&
            cJSON_ReplaceItemInObjectCaseSensitive(
                vector, "id", cJSON_CreateString(source_id->valuestring)),
            "duplicate-case mutation construction failed");
    REQUIRE(validate_corpus_schema(copy) != 0,
            "strict corpus schema accepted a duplicate case ID");
    cJSON_Delete(copy);

    copy = cJSON_Duplicate(root, 1);
    REQUIRE(copy != NULL, "family-route mutation allocation failed");
    families = cJSON_GetObjectItemCaseSensitive(copy, "families");
    success_cases = cJSON_GetObjectItemCaseSensitive(copy, "success_cases");
    vector = cJSON_GetArrayItem(success_cases, 0);
    source_id = cJSON_GetObjectItemCaseSensitive(vector, "family");
    second_family = NULL;
    cJSON_ArrayForEach(first_family, families) {
        const cJSON* candidate_id =
            cJSON_GetObjectItemCaseSensitive(first_family, "id");
        if (cJSON_IsString(candidate_id) && candidate_id->valuestring &&
            cJSON_IsString(source_id) && source_id->valuestring &&
            strcmp(candidate_id->valuestring, source_id->valuestring)) {
            second_family = first_family;
            break;
        }
    }
    source_id = cJSON_GetObjectItemCaseSensitive(second_family, "id");
    REQUIRE(cJSON_IsString(source_id) && source_id->valuestring &&
            cJSON_ReplaceItemInObjectCaseSensitive(
                vector, "family", cJSON_CreateString(source_id->valuestring)),
            "family-route mutation construction failed");
    REQUIRE(validate_corpus_schema(copy) != 0,
            "strict corpus schema accepted an operator in the wrong family");
    cJSON_Delete(copy);

    copy = cJSON_Duplicate(root, 1);
    REQUIRE(copy != NULL, "duplicate-family mutation allocation failed");
    families = cJSON_GetObjectItemCaseSensitive(copy, "families");
    first_family = cJSON_GetArrayItem(families, 0);
    second_family = cJSON_GetArrayItem(families, 1);
    source_id = cJSON_GetObjectItemCaseSensitive(first_family, "id");
    REQUIRE(cJSON_IsString(source_id) && source_id->valuestring &&
            cJSON_ReplaceItemInObjectCaseSensitive(
                second_family, "id", cJSON_CreateString(source_id->valuestring)),
            "duplicate-family mutation construction failed");
    REQUIRE(validate_corpus_schema(copy) != 0,
            "strict corpus schema accepted a duplicate family ID");
    cJSON_Delete(copy);
    return 0;
}

static void owned_tensor_clear(OwnedTensor* tensor) {
    if (!tensor) return;
    free(tensor->zero_points);
    free(tensor->scales);
    free(tensor->shape);
    memset(tensor, 0, sizeof(*tensor));
}

static int parse_descriptor(const cJSON* source,
                            const char* name,
                            OwnedTensor* output) {
    const cJSON* shape;
    const cJSON* dtype;
    const cJSON* quantization;
    const cJSON* item;
    size_t index = 0;
    memset(output, 0, sizeof(*output));
    REQUIRE(cJSON_IsObject(source), "descriptor must be an object");
    output->named.name = name;
    shape = cJSON_GetObjectItemCaseSensitive(source, "shape");
    dtype = cJSON_GetObjectItemCaseSensitive(source, "dtype");
    REQUIRE(cJSON_IsArray(shape), "descriptor shape must be an array");
    output->named.descriptor.rank = (size_t)cJSON_GetArraySize(shape);
    if (output->named.descriptor.rank) {
        REQUIRE(output->named.descriptor.rank <= SIZE_MAX / sizeof(*output->shape),
                "descriptor rank overflows size_t");
        output->shape = (uint64_t*)calloc(output->named.descriptor.rank,
                                         sizeof(*output->shape));
        REQUIRE(output->shape != NULL, "descriptor shape allocation failed");
    }
    cJSON_ArrayForEach(item, shape) {
        REQUIRE(json_safe_integer(item, 1.0,
                                  (double)VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER),
                "descriptor extent is not a positive safe integer");
        output->shape[index++] = (uint64_t)item->valuedouble;
    }
    output->named.descriptor.shape = output->shape;
    output->named.descriptor.dtype = parse_dtype(dtype);
    REQUIRE(output->named.descriptor.dtype != VX_DTYPE_UNSPECIFIED,
            "descriptor dtype is unsupported by the vector schema");

    quantization = cJSON_GetObjectItemCaseSensitive(source, "quantization");
    if (!quantization || cJSON_IsNull(quantization)) return 0;
    REQUIRE(cJSON_IsObject(quantization), "quantization must be an object");
    item = cJSON_GetObjectItemCaseSensitive(quantization, "scheme");
    REQUIRE(cJSON_IsString(item) && item->valuestring,
            "quantization scheme must be a string");
    if (!strcmp(item->valuestring, "per_tensor")) {
        const cJSON* scale = cJSON_GetObjectItemCaseSensitive(quantization, "scale");
        const cJSON* zero_point =
            cJSON_GetObjectItemCaseSensitive(quantization, "zero_point");
        REQUIRE(cJSON_IsNumber(scale), "per-tensor scale must be a number");
        REQUIRE(json_safe_integer(zero_point, -128.0, 255.0),
                "per-tensor zero point must be an integer");
        output->named.descriptor.quantization.scheme =
            VX_SHAPE_QUANTIZATION_PER_TENSOR;
        output->named.descriptor.quantization.scale = (float)scale->valuedouble;
        output->named.descriptor.quantization.zero_point =
            (int32_t)zero_point->valuedouble;
        return 0;
    }
    REQUIRE(!strcmp(item->valuestring, "per_axis"),
            "unsupported quantization scheme");
    {
        const cJSON* axis = cJSON_GetObjectItemCaseSensitive(quantization, "axis");
        const cJSON* scales = cJSON_GetObjectItemCaseSensitive(quantization, "scales");
        const cJSON* zero_points =
            cJSON_GetObjectItemCaseSensitive(quantization, "zero_points");
        size_t count;
        REQUIRE(json_safe_integer(axis, 0.0,
                                  (double)VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER),
                "per-axis axis must be a non-negative integer");
        REQUIRE(cJSON_IsArray(scales) && cJSON_IsArray(zero_points),
                "per-axis metadata must use arrays");
        count = (size_t)cJSON_GetArraySize(scales);
        REQUIRE(count > 0 && count == (size_t)cJSON_GetArraySize(zero_points),
                "per-axis metadata lengths differ");
        REQUIRE(count <= SIZE_MAX / sizeof(*output->scales) &&
                count <= SIZE_MAX / sizeof(*output->zero_points),
                "per-axis metadata size overflows");
        output->scales = (float*)calloc(count, sizeof(*output->scales));
        output->zero_points = (int32_t*)calloc(count, sizeof(*output->zero_points));
        REQUIRE(output->scales && output->zero_points,
                "per-axis metadata allocation failed");
        index = 0;
        cJSON_ArrayForEach(item, scales) {
            REQUIRE(cJSON_IsNumber(item), "per-axis scale must be a number");
            output->scales[index++] = (float)item->valuedouble;
        }
        index = 0;
        cJSON_ArrayForEach(item, zero_points) {
            REQUIRE(json_safe_integer(item, -128.0, 255.0),
                    "per-axis zero point must be an integer");
            output->zero_points[index++] = (int32_t)item->valuedouble;
        }
        output->named.descriptor.quantization.scheme =
            VX_SHAPE_QUANTIZATION_PER_AXIS;
        output->named.descriptor.quantization.axis = (size_t)axis->valuedouble;
        output->named.descriptor.quantization.count = count;
        output->named.descriptor.quantization.scales = output->scales;
        output->named.descriptor.quantization.zero_points = output->zero_points;
    }
    return 0;
}

static void owned_request_clear(OwnedRequest* owned) {
    size_t index;
    if (!owned) return;
    for (index = 0; index < owned->request.input_count; ++index)
        owned_tensor_clear(&owned->inputs[index]);
    for (index = 0; index < owned->request.declared_output_count; ++index)
        owned_tensor_clear(&owned->declared_outputs[index]);
    for (index = 0; index < owned->request.param_count; ++index)
        free(owned->param_number_arrays ? owned->param_number_arrays[index] : NULL);
    free(owned->declared_outputs);
    free(owned->param_number_arrays);
    free(owned->params);
    free(owned->inputs);
    memset(owned, 0, sizeof(*owned));
}

static int parse_named_tensor_object(const cJSON* object,
                                     OwnedTensor** owned_tensors,
                                     size_t* count) {
    const cJSON* item;
    size_t index = 0;
    *count = object_size(object);
    if (!*count) {
        *owned_tensors = NULL;
        return 0;
    }
    REQUIRE(*count <= SIZE_MAX / sizeof(**owned_tensors),
            "tensor object is too large");
    *owned_tensors = (OwnedTensor*)calloc(*count, sizeof(**owned_tensors));
    REQUIRE(*owned_tensors != NULL, "tensor object allocation failed");
    cJSON_ArrayForEach(item, object) {
        if (parse_descriptor(item, item->string, &(*owned_tensors)[index])) return -1;
        index++;
    }
    return 0;
}

/* OwnedTensor embeds VxShapeNamedTensor first, but its stride differs. Build a flat view. */
static VxShapeNamedTensor* flatten_named_tensors(const OwnedTensor* source,
                                                size_t count) {
    VxShapeNamedTensor* flat;
    size_t index;
    if (!count) return NULL;
    if (count > SIZE_MAX / sizeof(*flat)) return NULL;
    flat = (VxShapeNamedTensor*)calloc(count, sizeof(*flat));
    if (!flat) return NULL;
    for (index = 0; index < count; ++index) flat[index] = source[index].named;
    return flat;
}

static int parse_request(const cJSON* source, OwnedRequest* owned) {
    const cJSON* inputs;
    const cJSON* params;
    const cJSON* declared;
    const cJSON* item;
    VxShapeNamedTensor* flat_inputs;
    VxShapeNamedTensor* flat_declared;
    size_t index = 0;
    memset(owned, 0, sizeof(*owned));
    REQUIRE(cJSON_IsObject(source), "shape request must be an object");
    inputs = cJSON_GetObjectItemCaseSensitive(source, "inputs");
    REQUIRE(cJSON_IsObject(inputs) && object_size(inputs) > 0,
            "shape request inputs must be a non-empty object");
    if (parse_named_tensor_object(inputs, &owned->inputs,
                                  &owned->request.input_count)) return -1;
    flat_inputs = flatten_named_tensors(owned->inputs, owned->request.input_count);
    REQUIRE(flat_inputs != NULL, "flat input allocation failed");
    owned->request.inputs = flat_inputs;

    params = cJSON_GetObjectItemCaseSensitive(source, "params");
    if (params) {
        REQUIRE(cJSON_IsObject(params), "operator params must be an object");
        owned->request.param_count = object_size(params);
        if (owned->request.param_count) {
            REQUIRE(owned->request.param_count <= SIZE_MAX / sizeof(*owned->params),
                    "operator params are too large");
            owned->params = (VxShapeParam*)calloc(owned->request.param_count,
                                                  sizeof(*owned->params));
            owned->param_number_arrays = (double**)calloc(
                owned->request.param_count, sizeof(*owned->param_number_arrays));
            REQUIRE(owned->params != NULL && owned->param_number_arrays != NULL,
                    "operator params allocation failed");
        }
        cJSON_ArrayForEach(item, params) {
            VxShapeParam* param = &owned->params[index++];
            param->name = item->string;
            if (cJSON_IsBool(item)) {
                param->kind = VX_SHAPE_PARAM_BOOLEAN;
                param->value.boolean = cJSON_IsTrue(item);
            } else if (cJSON_IsNumber(item)) {
                param->kind = VX_SHAPE_PARAM_NUMBER;
                param->value.number = item->valuedouble;
            } else if (cJSON_IsString(item)) {
                param->kind = VX_SHAPE_PARAM_STRING;
                param->value.string = item->valuestring;
            } else if (cJSON_IsArray(item)) {
                const cJSON* element;
                size_t element_index = 0;
                size_t count = (size_t)cJSON_GetArraySize(item);
                double* values = NULL;
                if (count) {
                    REQUIRE(count <= SIZE_MAX / sizeof(*values),
                            "operator parameter array is too large");
                    values = (double*)calloc(count, sizeof(*values));
                    REQUIRE(values != NULL,
                            "operator parameter array allocation failed");
                }
                cJSON_ArrayForEach(element, item) {
                    REQUIRE(cJSON_IsNumber(element),
                            "operator parameter arrays must contain numbers");
                    values[element_index++] = element->valuedouble;
                }
                owned->param_number_arrays[index - 1] = values;
                param->kind = VX_SHAPE_PARAM_NUMBER_ARRAY;
                param->value.number_array.count = count;
                param->value.number_array.values = values;
            } else {
                return fail_at(__FILE__, __LINE__,
                               "unsupported operator parameter value in corpus");
            }
        }
        owned->request.params = owned->params;
    }

    declared = cJSON_GetObjectItemCaseSensitive(source, "declaredOutputs");
    if (declared) {
        REQUIRE(cJSON_IsObject(declared), "declared outputs must be an object");
        if (parse_named_tensor_object(declared, &owned->declared_outputs,
                                      &owned->request.declared_output_count)) return -1;
        flat_declared = flatten_named_tensors(owned->declared_outputs,
                                              owned->request.declared_output_count);
        REQUIRE(!owned->request.declared_output_count || flat_declared,
                "flat declared-output allocation failed");
        owned->request.declared_outputs = flat_declared;
    }
    return 0;
}

static void parsed_request_clear(OwnedRequest* owned) {
    free((void*)owned->request.declared_outputs);
    free((void*)owned->request.inputs);
    owned->request.declared_outputs = NULL;
    owned->request.inputs = NULL;
    owned_request_clear(owned);
}

static int quantization_equal(const VxShapeQuantization* actual,
                              const VxShapeQuantization* expected) {
    size_t index;
    if (actual->scheme != expected->scheme) return 0;
    if (actual->scheme == VX_SHAPE_QUANTIZATION_NONE) return 1;
    if (actual->scheme == VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        return actual->scale == expected->scale &&
               actual->zero_point == expected->zero_point;
    }
    if (actual->axis != expected->axis || actual->count != expected->count)
        return 0;
    for (index = 0; index < actual->count; ++index) {
        if (actual->scales[index] != expected->scales[index] ||
            actual->zero_points[index] != expected->zero_points[index]) return 0;
    }
    return 1;
}

static int descriptor_equal(const VxShapeTensorDescriptor* actual,
                            const VxShapeTensorDescriptor* expected) {
    size_t axis;
    if (actual->rank != expected->rank || actual->dtype != expected->dtype)
        return 0;
    for (axis = 0; axis < actual->rank; ++axis) {
        if (actual->shape[axis] != expected->shape[axis]) return 0;
    }
    return quantization_equal(&actual->quantization, &expected->quantization);
}

static const VxShapeTensorDescriptor* find_owned_descriptor(
        const OwnedTensor* tensors,
        size_t count,
        const char* name) {
    size_t index;
    for (index = 0; index < count; ++index) {
        if (tensors[index].named.name && name &&
            !strcmp(tensors[index].named.name, name)) {
            return &tensors[index].named.descriptor;
        }
    }
    return NULL;
}

static void owned_tensor_array_clear(OwnedTensor* tensors, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) owned_tensor_clear(&tensors[index]);
    free(tensors);
}

static char* read_text_file(const char* path) {
    FILE* stream;
    long length;
    char* source;
    size_t read_count;
    stream = fopen(path, "rb");
    if (!stream) return NULL;
    if (fseek(stream, 0, SEEK_END) || (length = ftell(stream)) < 0 ||
        fseek(stream, 0, SEEK_SET)) {
        fclose(stream);
        return NULL;
    }
    source = (char*)malloc((size_t)length + 1);
    if (!source) {
        fclose(stream);
        return NULL;
    }
    read_count = fread(source, 1, (size_t)length, stream);
    fclose(stream);
    if (read_count != (size_t)length) {
        free(source);
        return NULL;
    }
    source[length] = '\0';
    return source;
}

static OperatorCoverage* find_coverage(OperatorCoverage* coverage,
                                       size_t count,
                                       const char* name) {
    size_t index;
    for (index = 0; index < count; ++index) {
        if (!strcmp(coverage[index].name, name)) return &coverage[index];
    }
    return NULL;
}

static int build_manifest(const cJSON* families,
                          OperatorCoverage** coverage_out,
                          size_t* count_out) {
    const cJSON* family;
    const cJSON* operator_name;
    OperatorCoverage* coverage = NULL;
    size_t count = 0;
    REQUIRE(cJSON_IsArray(families), "families must be an array");
    cJSON_ArrayForEach(family, families) {
        const cJSON* operators =
            cJSON_GetObjectItemCaseSensitive(family, "operators");
        REQUIRE(cJSON_IsArray(operators), "family operators must be an array");
        cJSON_ArrayForEach(operator_name, operators) {
            OperatorCoverage* expanded;
            REQUIRE(cJSON_IsString(operator_name) && operator_name->valuestring,
                    "operator name must be a string");
            REQUIRE(find_coverage(coverage, count, operator_name->valuestring) == NULL,
                    "operator belongs to multiple families");
            expanded = (OperatorCoverage*)realloc(
                coverage, (count + 1) * sizeof(*coverage));
            REQUIRE(expanded != NULL, "operator manifest allocation failed");
            coverage = expanded;
            memset(&coverage[count], 0, sizeof(coverage[count]));
            coverage[count++].name = operator_name->valuestring;
        }
    }
    REQUIRE(count > 0, "shared manifest must contain at least one operator");
    {
        size_t index;
        for (index = 0; index < count; ++index) {
            REQUIRE(vx_shape_contract_function_id(coverage[index].name) != NULL,
                    "manifest operator has no native canonical contract");
        }
    }
    *coverage_out = coverage;
    *count_out = count;
    return 0;
}

static int run_success_cases(const cJSON* cases,
                             OperatorCoverage* coverage,
                             size_t coverage_count,
                             VxConcreteShapeResult* result) {
    const cJSON* vector;
    REQUIRE(cJSON_IsArray(cases), "success_cases must be an array");
    cJSON_ArrayForEach(vector, cases) {
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(vector, "id");
        const cJSON* operators =
            cJSON_GetObjectItemCaseSensitive(vector, "operators");
        const cJSON* request_json =
            cJSON_GetObjectItemCaseSensitive(vector, "request");
        const cJSON* shape_id =
            cJSON_GetObjectItemCaseSensitive(vector, "expected_shape_function_id");
        const cJSON* expected =
            cJSON_GetObjectItemCaseSensitive(vector, "expected");
        const cJSON* operator_name;
        OwnedRequest request;
        OwnedTensor* expected_tensors = NULL;
        size_t expected_count = 0;
        REQUIRE(cJSON_IsString(id) && id->valuestring,
                "success vector id must be a string");
        REQUIRE(cJSON_IsArray(operators), "success operators must be an array");
        REQUIRE(cJSON_IsString(shape_id) && shape_id->valuestring,
                "success shape ID must be a string");
        REQUIRE(cJSON_IsObject(expected), "success expected must be an object");
        if (parse_request(request_json, &request)) return -1;
        if (parse_named_tensor_object(expected, &expected_tensors,
                                      &expected_count)) {
            parsed_request_clear(&request);
            return -1;
        }
        cJSON_ArrayForEach(operator_name, operators) {
            const char* generated_id;
            OperatorCoverage* covered;
            VxShapeContractError error;
            REQUIRE(cJSON_IsString(operator_name) && operator_name->valuestring,
                    "success operator must be a string");
            generated_id = vx_shape_contract_function_id(operator_name->valuestring);
            REQUIRE(generated_id && !strcmp(generated_id, shape_id->valuestring),
                    "native generated shape-function ID differs from corpus");
            if (vx_shape_contract_infer(operator_name->valuestring,
                                        &request.request, result, &error)) {
                fprintf(stderr, "FAIL success %s:%s: %s at %s (%s)\n",
                        id->valuestring, operator_name->valuestring,
                        vx_shape_contract_error_code_name(error.code),
                        error.path, error.detail);
                owned_tensor_array_clear(expected_tensors, expected_count);
                parsed_request_clear(&request);
                return -1;
            }
            REQUIRE(result->shape_function_id &&
                    !strcmp(result->shape_function_id, shape_id->valuestring),
                    "inference result shape-function ID differs from corpus");
            if (result->output_count != expected_count) {
                fprintf(stderr, "FAIL success %s:%s: output-count mismatch\n",
                        id->valuestring, operator_name->valuestring);
                owned_tensor_array_clear(expected_tensors, expected_count);
                parsed_request_clear(&request);
                return -1;
            }
            {
                size_t output_index;
                for (output_index = 0; output_index < result->output_count;
                     ++output_index) {
                    const VxShapeTensorDescriptor* expected_descriptor =
                        find_owned_descriptor(expected_tensors, expected_count,
                                              result->outputs[output_index].name);
                    if (!expected_descriptor ||
                        !descriptor_equal(
                            &result->outputs[output_index].descriptor,
                            expected_descriptor)) {
                        fprintf(stderr,
                                "FAIL success %s:%s: output %s mismatch\n",
                                id->valuestring, operator_name->valuestring,
                                result->outputs[output_index].name);
                        owned_tensor_array_clear(expected_tensors, expected_count);
                        parsed_request_clear(&request);
                        return -1;
                    }
                }
            }
            covered = find_coverage(coverage, coverage_count,
                                    operator_name->valuestring);
            REQUIRE(covered != NULL, "success operator is absent from manifest");
            covered->success = 1;
        }
        owned_tensor_array_clear(expected_tensors, expected_count);
        parsed_request_clear(&request);
    }
    return 0;
}

static int run_failure_cases(const cJSON* cases,
                             OperatorCoverage* coverage,
                             size_t coverage_count,
                             VxConcreteShapeResult* result) {
    const cJSON* vector;
    REQUIRE(cJSON_IsArray(cases), "failure_cases must be an array");
    cJSON_ArrayForEach(vector, cases) {
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(vector, "id");
        const cJSON* operators =
            cJSON_GetObjectItemCaseSensitive(vector, "operators");
        const cJSON* request_json =
            cJSON_GetObjectItemCaseSensitive(vector, "request");
        const cJSON* shape_id =
            cJSON_GetObjectItemCaseSensitive(vector, "expected_shape_function_id");
        const cJSON* expected_error =
            cJSON_GetObjectItemCaseSensitive(vector, "expected_error");
        const cJSON* expected_code =
            cJSON_GetObjectItemCaseSensitive(expected_error, "code");
        const cJSON* expected_path =
            cJSON_GetObjectItemCaseSensitive(expected_error, "path");
        const cJSON* operator_name;
        OwnedRequest request;
        REQUIRE(cJSON_IsString(id) && id->valuestring,
                "failure vector id must be a string");
        REQUIRE(cJSON_IsArray(operators), "failure operators must be an array");
        REQUIRE(cJSON_IsString(shape_id) && shape_id->valuestring,
                "failure shape ID must be a string");
        REQUIRE(cJSON_IsString(expected_code) && expected_code->valuestring &&
                cJSON_IsString(expected_path) && expected_path->valuestring,
                "expected error must contain code and path strings");
        if (parse_request(request_json, &request)) return -1;
        cJSON_ArrayForEach(operator_name, operators) {
            const char* generated_id;
            const char* previous_id = result->shape_function_id;
            VxShapeNamedTensor* previous_outputs = result->outputs;
            size_t previous_output_count = result->output_count;
            OperatorCoverage* covered;
            VxShapeContractError error;
            REQUIRE(cJSON_IsString(operator_name) && operator_name->valuestring,
                    "failure operator must be a string");
            generated_id = vx_shape_contract_function_id(operator_name->valuestring);
            REQUIRE(generated_id && !strcmp(generated_id, shape_id->valuestring),
                    "failure vector generated ID differs from corpus");
            if (!vx_shape_contract_infer(operator_name->valuestring,
                                         &request.request, result, &error)) {
                fprintf(stderr, "FAIL failure %s:%s unexpectedly succeeded\n",
                        id->valuestring, operator_name->valuestring);
                parsed_request_clear(&request);
                return -1;
            }
            if (strcmp(vx_shape_contract_error_code_name(error.code),
                       expected_code->valuestring) ||
                strcmp(error.path, expected_path->valuestring)) {
                fprintf(stderr,
                        "FAIL failure %s:%s: got %s at %s, expected %s at %s\n",
                        id->valuestring, operator_name->valuestring,
                        vx_shape_contract_error_code_name(error.code), error.path,
                        expected_code->valuestring, expected_path->valuestring);
                parsed_request_clear(&request);
                return -1;
            }
            REQUIRE(result->shape_function_id == previous_id &&
                    result->outputs == previous_outputs &&
                    result->output_count == previous_output_count,
                    "failed inference changed the previous successful result");
            covered = find_coverage(coverage, coverage_count,
                                    operator_name->valuestring);
            REQUIRE(covered != NULL, "failure operator is absent from manifest");
            covered->failure = 1;
        }
        parsed_request_clear(&request);
    }
    return 0;
}

static const char* const VX_SPATIAL_OPERATORS[] = {
    "BatchMatMul",
    "Conv1D",
    "Conv2D",
    "ConvTranspose2D",
    "MaxPool2D",
    "AveragePool2D",
    "GlobalAveragePool",
    "Resize",
    "ResizeNearest2D",
    "UpsampleNearest2D"
};

static size_t spatial_case_id_occurrences(const cJSON* success_cases,
                                          const cJSON* failure_cases,
                                          const char* id) {
    const cJSON* groups[] = {success_cases, failure_cases};
    size_t group;
    size_t count = 0;
    for (group = 0; group < VX_ARRAY_COUNT(groups); ++group) {
        const cJSON* vector;
        cJSON_ArrayForEach(vector, groups[group]) {
            const cJSON* candidate =
                cJSON_GetObjectItemCaseSensitive(vector, "id");
            if (cJSON_IsString(candidate) && candidate->valuestring &&
                !strcmp(candidate->valuestring, id)) count++;
        }
    }
    return count;
}

static int validate_spatial_cases(const cJSON* cases,
                                  int success,
                                  const cJSON* success_cases,
                                  const cJSON* failure_cases,
                                  size_t* counts) {
    static const char* const SUCCESS_FIELDS[] = {
        "id", "operator", "request", "expected"
    };
    static const char* const FAILURE_FIELDS[] = {
        "id", "operator", "request", "expected_error"
    };
    static const char* const OUT_FIELD[] = {"out"};
    static const char* const ERROR_FIELDS[] = {"code", "path"};
    const char* const* fields = success ? SUCCESS_FIELDS : FAILURE_FIELDS;
    size_t field_count = success ? VX_ARRAY_COUNT(SUCCESS_FIELDS)
                                 : VX_ARRAY_COUNT(FAILURE_FIELDS);
    const cJSON* vector;
    if (!cJSON_IsArray(cases)) return -1;
    cJSON_ArrayForEach(vector, cases) {
        const cJSON* id;
        const cJSON* operator_name;
        const cJSON* expected;
        size_t operator_index;
        if (!object_has_exact_fields(vector, fields, field_count)) return -1;
        id = cJSON_GetObjectItemCaseSensitive(vector, "id");
        operator_name = cJSON_GetObjectItemCaseSensitive(vector, "operator");
        if (!cJSON_IsString(id) || !id->valuestring || !id->valuestring[0] ||
            spatial_case_id_occurrences(success_cases, failure_cases,
                                        id->valuestring) != 1 ||
            !cJSON_IsString(operator_name) || !operator_name->valuestring ||
            validate_request_schema(
                cJSON_GetObjectItemCaseSensitive(vector, "request"))) return -1;
        for (operator_index = 0;
             operator_index < VX_ARRAY_COUNT(VX_SPATIAL_OPERATORS);
             ++operator_index) {
            if (!strcmp(operator_name->valuestring,
                        VX_SPATIAL_OPERATORS[operator_index])) break;
        }
        if (operator_index == VX_ARRAY_COUNT(VX_SPATIAL_OPERATORS) ||
            !vx_shape_contract_function_id(operator_name->valuestring)) return -1;
        counts[operator_index]++;
        expected = cJSON_GetObjectItemCaseSensitive(
            vector, success ? "expected" : "expected_error");
        if (success) {
            if (!object_has_exact_fields(expected, OUT_FIELD,
                                         VX_ARRAY_COUNT(OUT_FIELD)) ||
                validate_descriptor_schema(
                    cJSON_GetObjectItemCaseSensitive(expected, "out"))) return -1;
        } else if (!object_has_exact_fields(expected, ERROR_FIELDS,
                                            VX_ARRAY_COUNT(ERROR_FIELDS)) ||
                   !cJSON_IsString(cJSON_GetObjectItemCaseSensitive(
                       expected, "code")) ||
                   !cJSON_IsString(cJSON_GetObjectItemCaseSensitive(
                       expected, "path"))) return -1;
    }
    return 0;
}

static int validate_spatial_corpus_schema(const cJSON* root) {
    static const char* const ROOT_FIELDS[] = {
        "format", "success_cases", "failure_cases"
    };
    const cJSON* format;
    const cJSON* success_cases;
    const cJSON* failure_cases;
    size_t success_counts[VX_ARRAY_COUNT(VX_SPATIAL_OPERATORS)] = {0};
    size_t failure_counts[VX_ARRAY_COUNT(VX_SPATIAL_OPERATORS)] = {0};
    size_t index;
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !object_has_exact_fields(root, ROOT_FIELDS,
                                 VX_ARRAY_COUNT(ROOT_FIELDS))) return -1;
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    success_cases = cJSON_GetObjectItemCaseSensitive(root, "success_cases");
    failure_cases = cJSON_GetObjectItemCaseSensitive(root, "failure_cases");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring,
               "volvox-operator-shape-spatial-vectors/v1") ||
        validate_spatial_cases(success_cases, 1, success_cases, failure_cases,
                               success_counts) ||
        validate_spatial_cases(failure_cases, 0, success_cases, failure_cases,
                               failure_counts)) return -1;
    for (index = 0; index < VX_ARRAY_COUNT(VX_SPATIAL_OPERATORS); ++index) {
        if (success_counts[index] < 2 || failure_counts[index] < 1) return -1;
    }
    return 0;
}

static int build_spatial_manifest(OperatorCoverage** coverage_out,
                                  size_t* count_out) {
    OperatorCoverage* coverage = (OperatorCoverage*)calloc(
        VX_ARRAY_COUNT(VX_SPATIAL_OPERATORS), sizeof(*coverage));
    size_t index;
    REQUIRE(coverage != NULL, "spatial operator manifest allocation failed");
    for (index = 0; index < VX_ARRAY_COUNT(VX_SPATIAL_OPERATORS); ++index) {
        coverage[index].name = VX_SPATIAL_OPERATORS[index];
        REQUIRE(vx_shape_contract_function_id(coverage[index].name) != NULL,
                "spatial operator has no native canonical contract");
    }
    *coverage_out = coverage;
    *count_out = VX_ARRAY_COUNT(VX_SPATIAL_OPERATORS);
    return 0;
}

static int run_spatial_success_cases(const cJSON* cases,
                                     OperatorCoverage* coverage,
                                     size_t coverage_count,
                                     VxConcreteShapeResult* result) {
    const cJSON* vector;
    cJSON_ArrayForEach(vector, cases) {
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(vector, "id");
        const cJSON* operator_name =
            cJSON_GetObjectItemCaseSensitive(vector, "operator");
        const cJSON* expected =
            cJSON_GetObjectItemCaseSensitive(vector, "expected");
        OwnedRequest request;
        OwnedTensor expected_tensor;
        VxShapeContractError error;
        OperatorCoverage* covered;
        if (parse_request(cJSON_GetObjectItemCaseSensitive(vector, "request"),
                          &request)) return -1;
        if (parse_descriptor(cJSON_GetObjectItemCaseSensitive(expected, "out"),
                             "out", &expected_tensor)) {
            parsed_request_clear(&request);
            return -1;
        }
        if (vx_shape_contract_infer(operator_name->valuestring,
                                    &request.request, result, &error)) {
            fprintf(stderr, "FAIL success %s:%s: %s at %s (%s)\n",
                    id->valuestring, operator_name->valuestring,
                    vx_shape_contract_error_code_name(error.code),
                    error.path, error.detail);
            owned_tensor_clear(&expected_tensor);
            parsed_request_clear(&request);
            return -1;
        }
        if (!result->shape_function_id ||
            strcmp(result->shape_function_id,
                   vx_shape_contract_function_id(operator_name->valuestring)) ||
            result->output_count != 1 ||
            strcmp(result->outputs[0].name, "out") ||
            !descriptor_equal(&result->outputs[0].descriptor,
                              &expected_tensor.named.descriptor)) {
            fprintf(stderr, "FAIL success %s:%s: descriptor/route mismatch\n",
                    id->valuestring, operator_name->valuestring);
            owned_tensor_clear(&expected_tensor);
            parsed_request_clear(&request);
            return -1;
        }
        covered = find_coverage(coverage, coverage_count,
                                operator_name->valuestring);
        REQUIRE(covered != NULL, "spatial success operator absent from manifest");
        covered->success = 1;
        owned_tensor_clear(&expected_tensor);
        parsed_request_clear(&request);
    }
    return 0;
}

static int run_spatial_failure_cases(const cJSON* cases,
                                     OperatorCoverage* coverage,
                                     size_t coverage_count,
                                     VxConcreteShapeResult* result) {
    const cJSON* vector;
    cJSON_ArrayForEach(vector, cases) {
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(vector, "id");
        const cJSON* operator_name =
            cJSON_GetObjectItemCaseSensitive(vector, "operator");
        const cJSON* expected_error =
            cJSON_GetObjectItemCaseSensitive(vector, "expected_error");
        const cJSON* expected_code =
            cJSON_GetObjectItemCaseSensitive(expected_error, "code");
        const cJSON* expected_path =
            cJSON_GetObjectItemCaseSensitive(expected_error, "path");
        OwnedRequest request;
        VxShapeContractError error;
        OperatorCoverage* covered;
        const char* previous_id = result->shape_function_id;
        VxShapeNamedTensor* previous_outputs = result->outputs;
        size_t previous_output_count = result->output_count;
        if (parse_request(cJSON_GetObjectItemCaseSensitive(vector, "request"),
                          &request)) return -1;
        if (!vx_shape_contract_infer(operator_name->valuestring,
                                     &request.request, result, &error)) {
            fprintf(stderr, "FAIL failure %s:%s unexpectedly succeeded\n",
                    id->valuestring, operator_name->valuestring);
            parsed_request_clear(&request);
            return -1;
        }
        if (strcmp(vx_shape_contract_error_code_name(error.code),
                   expected_code->valuestring) ||
            strcmp(error.path, expected_path->valuestring)) {
            fprintf(stderr,
                    "FAIL failure %s:%s: got %s at %s, expected %s at %s\n",
                    id->valuestring, operator_name->valuestring,
                    vx_shape_contract_error_code_name(error.code), error.path,
                    expected_code->valuestring, expected_path->valuestring);
            parsed_request_clear(&request);
            return -1;
        }
        REQUIRE(result->shape_function_id == previous_id &&
                result->outputs == previous_outputs &&
                result->output_count == previous_output_count,
                "failed spatial inference changed the previous result");
        covered = find_coverage(coverage, coverage_count,
                                operator_name->valuestring);
        REQUIRE(covered != NULL, "spatial failure operator absent from manifest");
        covered->failure = 1;
        parsed_request_clear(&request);
    }
    return 0;
}

static int check_direct_guards(void) {
    uint64_t ordinary_shape[] = {2, 3};
    VxShapeNamedTensor invalid_ports[] = {
        {.name = "input",
         .descriptor = {.rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32}},
        {.name = "extra",
         .descriptor = {.rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32}}
    };
    VxConcreteShapeRequest request = {
        .inputs = invalid_ports,
        .input_count = 2
    };
    VxConcreteShapeResult result = VX_CONCRETE_SHAPE_RESULT_INITIALIZER;
    VxShapeContractError error;
    REQUIRE(vx_shape_contract_infer("Identity", &request, &result, &error) != 0,
            "unexpected input port must fail");
    REQUIRE(error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_INPUT_PORTS &&
            !strcmp(error.path, "operator inputs"),
            "input-port error code/path mismatch");
    REQUIRE(vx_shape_contract_function_id("identity") == NULL,
            "operator lookup must remain case-sensitive");
    REQUIRE(vx_shape_contract_infer("NoSuchOperator", NULL,
                                    &result, &error) != 0 &&
            error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_REQUEST &&
            !strcmp(error.path, "shape inference request"),
            "invalid request must precede unknown-operator resolution");
    REQUIRE(vx_shape_contract_infer("NoSuchOperator", &request, &result, &error) != 0 &&
            error.code == VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR &&
            !strcmp(error.path, "operator"),
            "unknown operator error code/path mismatch");

    {
        VxShapeNamedTensor scalar_input = {
            .name = "input",
            .descriptor = {.rank = 0, .shape = NULL, .dtype = VX_DTYPE_F32}
        };
        request.inputs = &scalar_input;
        request.input_count = 1;
        REQUIRE(vx_shape_contract_infer("Softmax", &request, &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_RANK &&
                !strcmp(error.path, "operator input 'input'.shape"),
                "rank error code/path mismatch");
    }

    {
        VxShapeNamedTensor input = {
            .name = "input",
            .descriptor = {
                .rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32
            }
        };
        VxShapeParam invalid_approximate = {
            .name = "approximate",
            .kind = VX_SHAPE_PARAM_STRING,
            .value.string = "erf"
        };
        memset(&request, 0, sizeof(request));
        request.inputs = &input;
        request.input_count = 1;
        request.params = &invalid_approximate;
        request.param_count = 1;
        REQUIRE(vx_shape_contract_infer("GELU", &request, &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS &&
                !strcmp(error.path, "operator params.approximate"),
                "GELU approximate diagnostic mismatch");
    }

    {
        VxShapeNamedTensor input = {
            .name = "input",
            .descriptor = {
                .rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32
            }
        };
        VxShapeParam alpha = {
            .name = "alpha",
            .kind = VX_SHAPE_PARAM_NUMBER,
            .value.number = NAN
        };
        memset(&request, 0, sizeof(request));
        request.inputs = &input;
        request.input_count = 1;
        request.params = &alpha;
        request.param_count = 1;
        REQUIRE(vx_shape_contract_infer("LeakyReLU", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS &&
                !strcmp(error.path, "operator params.alpha"),
                "LeakyReLU alpha diagnostic mismatch");
    }

    {
        VxShapeNamedTensor input = {
            .name = "input",
            .descriptor = {
                .rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32
            }
        };
        VxShapeParam to = {
            .name = "to",
            .kind = VX_SHAPE_PARAM_STRING,
            .value.string = "float16"
        };
        memset(&request, 0, sizeof(request));
        request.inputs = &input;
        request.input_count = 1;
        request.params = &to;
        request.param_count = 1;
        REQUIRE(vx_shape_contract_infer("Cast", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE &&
                !strcmp(error.path, "operator params.to"),
                "Cast target-dtype diagnostic mismatch");
    }

    {
        VxShapeNamedTensor input = {
            .name = "input",
            .descriptor = {
                .rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32
            }
        };
        VxShapeParam bounds[] = {
            {.name = "min", .kind = VX_SHAPE_PARAM_NUMBER,
             .value.number = 2.0},
            {.name = "max", .kind = VX_SHAPE_PARAM_NUMBER,
             .value.number = 1.0}
        };
        memset(&request, 0, sizeof(request));
        request.inputs = &input;
        request.input_count = 1;
        request.params = bounds;
        request.param_count = VX_ARRAY_COUNT(bounds);
        REQUIRE(vx_shape_contract_infer("Clip", &request, &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS &&
                !strcmp(error.path, "operator params"),
                "Clip bound-order diagnostic mismatch");
    }

    {
        VxShapeNamedTensor input = {
            .name = "input",
            .descriptor = {
                .rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32
            }
        };
        VxShapeParam axis = {
            .name = "axis",
            .kind = VX_SHAPE_PARAM_NUMBER,
            .value.number = 0.0
        };
        memset(&request, 0, sizeof(request));
        request.inputs = &input;
        request.input_count = 1;
        request.params = &axis;
        request.param_count = 1;
        REQUIRE(vx_shape_contract_infer("Softmax", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS &&
                !strcmp(error.path, "operator params.axis"),
                "Softmax non-last-axis diagnostic mismatch");
    }

    {
        uint64_t input_shape[] = {2, 3};
        uint64_t weight_shape[] = {4, 3};
        VxShapeNamedTensor dense_inputs[] = {
            {.name = "input",
             .descriptor = {
                 .rank = 2, .shape = input_shape, .dtype = VX_DTYPE_F32
             }},
            {.name = "weight",
             .descriptor = {
                 .rank = 2, .shape = weight_shape, .dtype = VX_DTYPE_F32
             }}
        };
        VxShapeParam conflicting_layout[] = {
            {.name = "weight_layout", .kind = VX_SHAPE_PARAM_STRING,
             .value.string = "dout_din"},
            {.name = "transB", .kind = VX_SHAPE_PARAM_BOOLEAN,
             .value.boolean = 1}
        };
        memset(&request, 0, sizeof(request));
        request.inputs = dense_inputs;
        request.input_count = VX_ARRAY_COUNT(dense_inputs);
        request.params = conflicting_layout;
        request.param_count = VX_ARRAY_COUNT(conflicting_layout);
        REQUIRE(vx_shape_contract_infer("Linear", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS &&
                !strcmp(error.path, "operator params"),
                "dense layout-conflict diagnostic mismatch");
    }

    {
        uint64_t input_shape[] = {2, 3};
        uint64_t weight_shape[] = {3};
        VxShapeNamedTensor norm_inputs[] = {
            {.name = "input",
             .descriptor = {
                 .rank = 2, .shape = input_shape, .dtype = VX_DTYPE_F32
             }},
            {.name = "weight",
             .descriptor = {
                 .rank = 1, .shape = weight_shape, .dtype = VX_DTYPE_F32
             }}
        };
        VxShapeParam d_model = {
            .name = "d_model",
            .kind = VX_SHAPE_PARAM_NUMBER,
            .value.number = 4.0
        };
        memset(&request, 0, sizeof(request));
        request.inputs = norm_inputs;
        request.input_count = VX_ARRAY_COUNT(norm_inputs);
        request.params = &d_model;
        request.param_count = 1;
        REQUIRE(vx_shape_contract_infer("RMSNorm", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_SHAPE_MISMATCH &&
                !strcmp(error.path, "operator params.d_model"),
                "normalization d_model diagnostic mismatch");
    }

    {
        uint64_t input_shape[] = {1, 2, 2, 4};
        uint64_t affine_shape[] = {4};
        VxShapeNamedTensor group_inputs[] = {
            {.name = "input",
             .descriptor = {
                 .rank = 4, .shape = input_shape, .dtype = VX_DTYPE_F32
             }},
            {.name = "weight",
             .descriptor = {
                 .rank = 1, .shape = affine_shape, .dtype = VX_DTYPE_F32
             }},
            {.name = "bias",
             .descriptor = {
                 .rank = 1, .shape = affine_shape, .dtype = VX_DTYPE_F32
             }}
        };
        memset(&request, 0, sizeof(request));
        request.inputs = group_inputs;
        request.input_count = VX_ARRAY_COUNT(group_inputs);
        REQUIRE(vx_shape_contract_infer("GroupNorm", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS &&
                !strcmp(error.path, "operator params.num_groups"),
                "GroupNorm missing-group diagnostic mismatch");
    }

    {
        VxShapeNamedTensor binary_inputs[] = {
            {.name = "a",
             .descriptor = {
                 .rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32
             }},
            {.name = "b",
             .descriptor = {
                 .rank = 2, .shape = ordinary_shape, .dtype = VX_DTYPE_F32
             }}
        };
        VxShapeParam relu = {
            .name = "relu",
            .kind = VX_SHAPE_PARAM_NUMBER,
            .value.number = 3.0
        };
        memset(&request, 0, sizeof(request));
        request.inputs = binary_inputs;
        request.input_count = VX_ARRAY_COUNT(binary_inputs);
        request.params = &relu;
        request.param_count = 1;
        REQUIRE(vx_shape_contract_infer("Add", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_PARAMS &&
                !strcmp(error.path, "operator params.relu"),
                "Add relu diagnostic mismatch");
    }

    {
        uint64_t input_shape[] = {2};
        uint64_t scale_shape[] = {1};
        VxShapeNamedTensor quant_inputs[] = {
            {.name = "input",
             .descriptor = {
                 .rank = 1, .shape = input_shape, .dtype = VX_DTYPE_F32
             }},
            {.name = "scale",
             .descriptor = {
                 .rank = 1, .shape = scale_shape, .dtype = VX_DTYPE_F32
             }},
            {.name = "zero_point",
             .descriptor = {
                 .rank = 1, .shape = scale_shape, .dtype = VX_DTYPE_U8
             }}
        };
        VxShapeNamedTensor declared_output = {
            .name = "out",
            .descriptor = {
                .rank = 1,
                .shape = input_shape,
                .dtype = VX_DTYPE_I8,
                .quantization = {
                    .scheme = VX_SHAPE_QUANTIZATION_PER_TENSOR,
                    .scale = 0.25f,
                    .zero_point = 0
                }
            }
        };
        memset(&request, 0, sizeof(request));
        request.inputs = quant_inputs;
        request.input_count = VX_ARRAY_COUNT(quant_inputs);
        request.declared_outputs = &declared_output;
        request.declared_output_count = 1;
        REQUIRE(vx_shape_contract_infer("QuantizeLinear", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_DTYPE &&
                !strcmp(error.path,
                        "operator input 'zero_point'.dtype"),
                "quantization zero-point dtype diagnostic mismatch");
    }

    {
        uint64_t input_shape[] = {2, 3};
        uint64_t scale_shape[] = {3};
        float scales[] = {0.25f, 0.5f};
        int32_t zero_points[] = {0, 0};
        VxShapeNamedTensor quant_inputs[] = {
            {
                .name = "input",
                .descriptor = {
                    .rank = 2,
                    .shape = input_shape,
                    .dtype = VX_DTYPE_I8,
                    .quantization = {
                        .scheme = VX_SHAPE_QUANTIZATION_PER_AXIS,
                        .axis = 1,
                        .count = 2,
                        .scales = scales,
                        .zero_points = zero_points
                    }
                }
            },
            {
                .name = "scale",
                .descriptor = {
                    .rank = 1,
                    .shape = scale_shape,
                    .dtype = VX_DTYPE_F32
                }
            }
        };
        memset(&request, 0, sizeof(request));
        request.inputs = quant_inputs;
        request.input_count = 2;
        REQUIRE(vx_shape_contract_infer("DequantizeLinear", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION &&
                !strcmp(error.path,
                        "operator input 'input'.quantization.scales"),
                "per-axis extent error code/path mismatch");
    }

    {
        uint64_t input_shape[] = {2, 3};
        float one_scale = 0.25f;
        int32_t one_zero_point = 0;
        VxShapeNamedTensor oversized_metadata_input = {
            .name = "input",
            .descriptor = {
                .rank = 2,
                .shape = input_shape,
                .dtype = VX_DTYPE_I8,
                .quantization = {
                    .scheme = VX_SHAPE_QUANTIZATION_PER_AXIS,
                    .axis = 1,
                    .count = SIZE_MAX,
                    .scales = &one_scale,
                    .zero_points = &one_zero_point
                }
            }
        };
        memset(&request, 0, sizeof(request));
        request.inputs = &oversized_metadata_input;
        request.input_count = 1;
        REQUIRE(vx_shape_contract_infer("Identity", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION &&
                !strcmp(error.path,
                        "operator input 'input'.quantization.scales"),
                "per-axis count must be rejected before metadata dereference");
    }

    {
        uint64_t input_shape[] = {2};
        float scales[] = {1.0f, -1.0f};
        int32_t zero_points[] = {999, 0};
        VxShapeNamedTensor invalid_scale_input = {
            .name = "input",
            .descriptor = {
                .rank = 1,
                .shape = input_shape,
                .dtype = VX_DTYPE_I8,
                .quantization = {
                    .scheme = VX_SHAPE_QUANTIZATION_PER_AXIS,
                    .axis = 0,
                    .count = 2,
                    .scales = scales,
                    .zero_points = zero_points
                }
            }
        };
        memset(&request, 0, sizeof(request));
        request.inputs = &invalid_scale_input;
        request.input_count = 1;
        REQUIRE(vx_shape_contract_infer("Identity", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION &&
                !strcmp(error.path,
                        "operator input 'input'.quantization.scales[1]"),
                "all scales must precede zero-point validation");

        scales[1] = 1.0f;
        invalid_scale_input.descriptor.quantization.zero_points = NULL;
        REQUIRE(vx_shape_contract_infer("Identity", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_QUANTIZATION &&
                !strcmp(error.path,
                        "operator input 'input'.quantization.zero_points"),
                "missing zero-points must use the canonical diagnostic path");
    }

    {
        uint64_t index_shape[] = {VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER};
        uint64_t weight_shape[] = {1, 2};
        VxShapeNamedTensor embedding_inputs[] = {
            {
                .name = "input",
                .descriptor = {
                    .rank = 1,
                    .shape = index_shape,
                    .dtype = VX_DTYPE_I32
                }
            },
            {
                .name = "weight",
                .descriptor = {
                    .rank = 2,
                    .shape = weight_shape,
                    .dtype = VX_DTYPE_F32
                }
            }
        };
        memset(&request, 0, sizeof(request));
        request.inputs = embedding_inputs;
        request.input_count = 2;
        REQUIRE(vx_shape_contract_infer("Embedding", &request,
                                        &result, &error) != 0 &&
                error.code == VX_SHAPE_CONTRACT_ERROR_INVALID_DESCRIPTOR &&
                !strcmp(error.path, "operator output 'out'.shape"),
                "derived-output overflow error code/path mismatch");
    }
    vx_shape_contract_result_clear(&result);
    return 0;
}

int main(int argc, char** argv) {
    char* source;
    cJSON* root;
    const cJSON* format;
    OperatorCoverage* coverage = NULL;
    size_t coverage_count = 0;
    size_t index;
    VxConcreteShapeResult result = VX_CONCRETE_SHAPE_RESULT_INITIALIZER;
    int status = 1;
    if (argc != 2) {
        fprintf(stderr, "usage: %s tests/operator_shape_contract_vectors.json\n",
                argv[0]);
        return 2;
    }
    source = read_text_file(argv[1]);
    if (!source) {
        fprintf(stderr, "failed to read %s\n", argv[1]);
        return 2;
    }
    root = cJSON_Parse(source);
    free(source);
    if (!root || !vx_json_object_keys_unique_recursive(root)) {
        fprintf(stderr, "shared operator-shape corpus is invalid or ambiguous JSON\n");
        cJSON_Delete(root);
        return 1;
    }
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    if (cJSON_IsString(format) && format->valuestring &&
        !strcmp(format->valuestring,
                "volvox-operator-shape-spatial-vectors/v1")) {
        if (validate_spatial_corpus_schema(root) ||
            build_spatial_manifest(&coverage, &coverage_count) ||
            run_spatial_success_cases(
                cJSON_GetObjectItemCaseSensitive(root, "success_cases"),
                coverage, coverage_count, &result) ||
            run_spatial_failure_cases(
                cJSON_GetObjectItemCaseSensitive(root, "failure_cases"),
                coverage, coverage_count, &result)) {
            goto cleanup;
        }
        for (index = 0; index < coverage_count; ++index) {
            if (!coverage[index].success || !coverage[index].failure) {
                fprintf(stderr,
                        "spatial operator %s lacks shared legal/illegal coverage\n",
                        coverage[index].name);
                goto cleanup;
            }
        }
        printf("native spatial shape contracts passed: %zu operators, %d legal vectors, %d illegal vectors\n",
               coverage_count,
               cJSON_GetArraySize(cJSON_GetObjectItemCaseSensitive(
                   root, "success_cases")),
               cJSON_GetArraySize(cJSON_GetObjectItemCaseSensitive(
                   root, "failure_cases")));
        status = 0;
        goto cleanup;
    }
    if (validate_corpus_schema(root) || check_corpus_schema_rejections(root)) {
        fprintf(stderr, "shared operator-shape corpus violates the strict schema\n");
        goto cleanup;
    }
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    if (!cJSON_IsString(format) ||
        strcmp(format->valuestring, "volvox-operator-shape-vectors/v1")) {
        fprintf(stderr, "shared operator-shape corpus format mismatch\n");
        goto cleanup;
    }
    if (build_manifest(cJSON_GetObjectItemCaseSensitive(root, "families"),
                       &coverage, &coverage_count) ||
        run_success_cases(cJSON_GetObjectItemCaseSensitive(root, "success_cases"),
                          coverage, coverage_count, &result) ||
        run_failure_cases(cJSON_GetObjectItemCaseSensitive(root, "failure_cases"),
                          coverage, coverage_count, &result) ||
        check_direct_guards()) {
        goto cleanup;
    }
    for (index = 0; index < coverage_count; ++index) {
        if (!coverage[index].success || !coverage[index].failure) {
            fprintf(stderr, "operator %s lacks shared legal/illegal coverage\n",
                    coverage[index].name);
            goto cleanup;
        }
    }
    printf("native shared shape contracts passed: %zu operators, %d legal vectors, %d illegal vectors\n",
           coverage_count,
           cJSON_GetArraySize(cJSON_GetObjectItemCaseSensitive(root, "success_cases")),
           cJSON_GetArraySize(cJSON_GetObjectItemCaseSensitive(root, "failure_cases")));
    status = 0;

cleanup:
    vx_shape_contract_result_clear(&result);
    free(coverage);
    cJSON_Delete(root);
    return status;
}
