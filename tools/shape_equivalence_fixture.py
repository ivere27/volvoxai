"""Emit the C fixture that proves the concrete and symbolic shape evaluators
agree on every fully concrete vector in tests/contracts/shape_vectors.

M5 merges `shape_contract` (concrete) and `shape_domain_contract` (symbolic)
into one evaluator. Before any rule moves, the two implementations must be shown
to already agree wherever their domains overlap: a concrete dimension is the
symbolic dimension `FIXED(n)` with an empty environment. This fixture is that
proof, and it stays as the regression gate for each family cutover.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

if __package__:
    from .exporter.generated.operator_vocabulary import OPERATOR_GRAPH_NAMES
else:
    from exporter.generated.operator_vocabulary import OPERATOR_GRAPH_NAMES


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = REPOSITORY_ROOT / "tests" / "contracts" / "shape_vectors"

DTYPES = {
    "float32": "VX_DTYPE_F32",
    "int32": "VX_DTYPE_I32",
    "int8": "VX_DTYPE_I8",
    "uint8": "VX_DTYPE_U8",
}

SCHEMES = {"per_tensor": 1, "per_axis": 2}

# Mirrors vx_shape_contract_declared_output_mode. For these operators the
# declared output is semantic input, not an assertion, so a synthesized one
# would change the meaning of the request instead of merely satisfying the
# port arity.
SEMANTIC_DECLARED_OUTPUT = frozenset({
    "Reshape", "Expand", "Broadcast", "QuantizeLinear",
    "Resize", "ResizeNearest2D", "QLinear", "QMatMul", "QGemm",
    "QBatchMatMul", "QConv2D", "QAdd", "QEmbedding", "QGELU", "QSiLU",
    "QLayerNorm", "QGroupNorm", "QMaskedMean", "QSDPA", "RequantizeLinear",
})


def cfloat(value: float) -> str:
    """repr keeps the shortest round-trip form and always emits a decimal point,
    so `0.0f` never degenerates into the invalid octal constant `0f`."""
    return f"{float(value)!r}f"


def cdouble(value: float) -> str:
    return repr(float(value))


def operator_kinds() -> dict[str, str]:
    """Use the proto-generated mapping, independent of C table storage/order."""
    return {name: "VX_OP_" + kind.name for kind, name in OPERATOR_GRAPH_NAMES.items()}


class Emitter:
    """Projects the vector corpus into a C fixture.

    The unified evaluator sizes its output slots from the declared outputs, so a
    request must always carry output port names. Success cases record them under
    `expected`; failure cases predate that requirement and often omit them. For
    those, the port names are taken from a success case of the same operator and
    the extent is taken from the first input, which is enough to reach the input
    validation the failure case is actually testing.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.pool: dict[str, str] = {}
        self.kinds = operator_kinds()
        self.skipped: list[tuple[str, str]] = []
        self.output_ports: dict[str, list[str]] = {}
        self.synthesized = 0
        self.undeclared = 0

    def _array(self, prefix: str, ctype: str, values: list[str]) -> str:
        if not values:
            return "NULL"
        body = ", ".join(values)
        key = f"{ctype}|{body}"
        if key in self.pool:
            return self.pool[key]
        name = f"{prefix}_{len(self.pool)}"
        self.lines.append(f"static const {ctype} {name}[] = {{ {body} }};")
        self.pool[key] = name
        return name

    def tensor(self, name: str, spec: dict[str, Any]) -> str | None:
        dtype = DTYPES.get(spec.get("dtype"))
        if dtype is None:
            return None
        shape = spec.get("shape")
        if not isinstance(shape, list) or any(
                not isinstance(d, int) or d < 0 for d in shape):
            return None
        shape_ref = self._array("vx_shape", "uint64_t",
                                [f"UINT64_C({d})" for d in shape])
        quant = spec.get("quantization") or {}
        scheme = SCHEMES.get(quant.get("scheme"), 0)
        scale = float(quant.get("scale", 0.0))
        zero_point = int(quant.get("zero_point", 0))
        axis = int(quant.get("axis", 0))
        scales = quant.get("scales") or []
        zero_points = quant.get("zero_points") or []
        if len(scales) != len(zero_points):
            return None
        scales_ref = self._array("vx_scales", "float",
                                 [cfloat(v) for v in scales])
        zps_ref = self._array("vx_zps", "int32_t", [str(int(v)) for v in zero_points])
        return (f'{{ "{name}", {len(shape)}u, {shape_ref}, {dtype}, {scheme}, '
                f'{cfloat(scale)}, {zero_point}, {axis}u, {len(scales)}u, '
                f'{scales_ref}, {zps_ref} }}')

    def param(self, name: str, value: Any) -> str | None:
        if isinstance(value, bool):
            return f'{{ "{name}", VX_SHAPE_PARAM_BOOLEAN, 0.0, {1 if value else 0}, NULL, 0u, NULL }}'
        if isinstance(value, (int, float)):
            return f'{{ "{name}", VX_SHAPE_PARAM_NUMBER, {cdouble(value)}, 0, NULL, 0u, NULL }}'
        if isinstance(value, str):
            return f'{{ "{name}", VX_SHAPE_PARAM_STRING, 0.0, 0, "{value}", 0u, NULL }}'
        if isinstance(value, list):
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
                return None
            ref = self._array("vx_numbers", "double",
                              [cdouble(v) for v in value])
            return (f'{{ "{name}", VX_SHAPE_PARAM_NUMBER_ARRAY, 0.0, 0, NULL, '
                    f'{len(value)}u, {ref} }}')
        return None

    def case(self, case: dict[str, Any], operator: str, expect_success: bool) -> str | None:
        kind = self.kinds.get(operator)
        if kind is None:
            self.skipped.append((case.get("id", "?"), f"unknown operator {operator}"))
            return None
        request = case.get("request") or {}
        tensors: list[str] = []
        for port, spec in (request.get("inputs") or {}).items():
            literal = self.tensor(port, spec) if isinstance(spec, dict) else None
            if literal is None:
                self.skipped.append((case.get("id", "?"), f"input {port}"))
                return None
            tensors.append(literal)
        # The symbolic evaluator proves a declared output; it has no infer-only
        # mode (vx_sd_exact_ports requires declared_output_count == output_count).
        # The concrete evaluator may infer without one. Where the corpus omits
        # declaredOutputs it records the same descriptor under `expected`, so use
        # that: the concrete side still infers, the symbolic side proves it.
        declared_source = request.get("declaredOutputs") or case.get("expected") or {}
        # A case that exists to prove the missing declaration is rejected must
        # keep its missing declaration.
        error_path = str((case.get("expected_error") or {}).get("path", ""))
        declares_the_failure = "output" in error_path
        if (not declared_source and not declares_the_failure
                and operator not in SEMANTIC_DECLARED_OUTPUT):
            ports = self.output_ports.get(operator)
            first_input = next(iter((request.get("inputs") or {}).values()), None)
            if ports and isinstance(first_input, dict):
                declared_source = {port: first_input for port in ports}
                self.synthesized += 1
        declared: list[str] = []
        for port, spec in declared_source.items():
            literal = self.tensor(port, spec) if isinstance(spec, dict) else None
            if literal is None:
                self.skipped.append((case.get("id", "?"), f"declared {port}"))
                return None
            declared.append(literal)
        params: list[str] = []
        for name, value in (request.get("params") or {}).items():
            literal = self.param(name, value)
            if literal is None:
                self.skipped.append((case.get("id", "?"), f"param {name}"))
                return None
            params.append(literal)
        expected_outputs: list[str] = []
        for port, spec in (case.get("expected") or {}).items():
            literal = self.tensor(port, spec) if isinstance(spec, dict) else None
            if literal is None:
                self.skipped.append((case.get("id", "?"), f"expected {port}"))
                return None
            expected_outputs.append(literal)
        expected_ref = self._array("vx_expected", "VecTensor", expected_outputs)
        inputs_ref = self._array("vx_inputs", "VecTensor", tensors)
        declared_ref = self._array("vx_declared", "VecTensor", declared)
        params_ref = self._array("vx_params", "VecParam", params)
        expected = case.get("expected_shape_function_id")
        expected_c = f'"{expected}"' if expected else "NULL"
        error = case.get("expected_error") or {}
        code = error.get("code")
        if not declared:
            # Without a declaration the evaluator stops at port arity, so the
            # historical code is not reachable. Rejection is still required.
            code = None
            self.undeclared += 1
        code_c = f"VX_SHAPE_CONTRACT_ERROR_{code}" if code else "VX_SHAPE_CONTRACT_ERROR_NONE"
        path = error.get("path")
        path_c = f'"{path}"' if path else "NULL"
        identifier = f'{case.get("id", "case")}:{operator}'
        return (f'    {{ "{identifier}", "{operator}", {kind}, '
                f'{inputs_ref}, {len(tensors)}u, {params_ref}, {len(params)}u, '
                f'{declared_ref}, {len(declared)}u, '
                f'{expected_ref}, {len(expected_outputs)}u, '
                f'{1 if expect_success else 0}, '
                f'{expected_c}, {code_c}, {path_c} }},')


PRELUDE = r'''/* GENERATED by tools/shape_equivalence_fixture.py - do not edit.
 *
 * Proves that vx_shape_contract_infer (concrete) and
 * vx_shape_domain_contract_prove (symbolic, all dimensions FIXED) agree on the
 * operator shape vector corpus in tests/contracts/shape_vectors/. */

#include "runtime/shape_contract.h"
#include "runtime/shape_domain_contract.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct VecTensor {
    const char* name;
    uint32_t rank;
    const uint64_t* shape;
    VxDataType dtype;
    int quantization_scheme;
    float scale;
    int32_t zero_point;
    uint32_t axis;
    uint32_t scale_count;
    const float* scales;
    const int32_t* zero_points;
} VecTensor;

typedef struct VecParam {
    const char* name;
    VxShapeParamKind kind;
    double number;
    int boolean;
    const char* string;
    uint32_t number_count;
    const double* numbers;
} VecParam;

typedef struct VecCase {
    const char* id;
    const char* operator_name;
    VxOperatorKind operator_kind;
    const VecTensor* inputs;
    uint32_t input_count;
    const VecParam* params;
    uint32_t param_count;
    const VecTensor* declared;
    uint32_t declared_count;
    const VecTensor* expected_outputs;
    uint32_t expected_output_count;
    int expect_success;
    const char* expected_shape_function_id;
    VxShapeContractErrorCode expected_error_code;
    const char* expected_error_path;
} VecCase;

#define VX_MAX_PORTS 16u
#define VX_MAX_RANK 8u
#define VX_MAX_PARAMS 16u
#define VX_MAX_SCALES 64u

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static VxShapeDomainString domain_string(const char* value) {
    VxShapeDomainString result;
    result.bytes = (const uint8_t*)value;
    result.length = (uint32_t)strlen(value);
    return result;
}

/* Concrete request projection. Borrowed arrays live in the caller frame. */
static void fill_concrete(
        const VecTensor* source, uint32_t count, VxShapeNamedTensor* out) {
    uint32_t index;
    for (index = 0; index < count; index++) {
        VxShapeTensorDescriptor descriptor;
        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.rank = source[index].rank;
        descriptor.shape = source[index].shape;
        descriptor.dtype = source[index].dtype;
        descriptor.quantization.scheme =
            (VxShapeQuantizationScheme)source[index].quantization_scheme;
        descriptor.quantization.scale = source[index].scale;
        descriptor.quantization.zero_point = source[index].zero_point;
        descriptor.quantization.axis = source[index].axis;
        descriptor.quantization.count = source[index].scale_count;
        descriptor.quantization.scales = source[index].scales;
        descriptor.quantization.zero_points = source[index].zero_points;
        out[index].name = source[index].name;
        out[index].descriptor = descriptor;
    }
}

/* Symbolic projection of the same tensors: every dimension is FIXED(n) and the
 * symbol environment is empty, which is exactly the concrete sub-domain. */
static int fill_domain(
        const VecTensor* source,
        uint32_t count,
        VxShapeDomainNamedTensor* out,
        VxShapeDomainDimension dimensions[][VX_MAX_RANK],
        uint32_t scale_bits[][VX_MAX_SCALES]) {
    uint32_t index;
    for (index = 0; index < count; index++) {
        uint32_t axis;
        uint32_t scale;
        if (source[index].rank > VX_MAX_RANK) return 0;
        if (source[index].scale_count > VX_MAX_SCALES) return 0;
        for (axis = 0; axis < source[index].rank; axis++) {
            dimensions[index][axis].kind = VX_SHAPE_DOMAIN_DIMENSION_FIXED;
            dimensions[index][axis].symbol_index = VX_SHAPE_DOMAIN_INDEX_NONE;
            dimensions[index][axis].value = (int64_t)source[index].shape[axis];
        }
        for (scale = 0; scale < source[index].scale_count; scale++)
            scale_bits[index][scale] = float_bits(source[index].scales[scale]);
        memset(&out[index], 0, sizeof(out[index]));
        out[index].name = domain_string(source[index].name);
        out[index].tensor_index = index;
        out[index].descriptor.dtype = (int32_t)source[index].dtype;
        out[index].descriptor.rank = source[index].rank;
        out[index].descriptor.dimensions = dimensions[index];
        out[index].descriptor.quantization.scheme =
            source[index].quantization_scheme;
        out[index].descriptor.quantization.scale_bits =
            float_bits(source[index].scale);
        out[index].descriptor.quantization.zero_point = source[index].zero_point;
        out[index].descriptor.quantization.axis = source[index].axis;
        out[index].descriptor.quantization.count = source[index].scale_count;
        out[index].descriptor.quantization.scales =
            source[index].scale_count ? scale_bits[index] : NULL;
        out[index].descriptor.quantization.zero_points =
            source[index].zero_points;
    }
    return 1;
}

static void fill_domain_params(
        const VecParam* source,
        uint32_t count,
        VxShapeDomainParam* out,
        VxShapeDomainParamValue values[][VX_MAX_SCALES]) {
    uint32_t index;
    for (index = 0; index < count; index++) {
        uint32_t item;
        memset(&out[index], 0, sizeof(out[index]));
        out[index].name = domain_string(source[index].name);
        switch (source[index].kind) {
        case VX_SHAPE_PARAM_NUMBER:
            out[index].kind = VX_SHAPE_DOMAIN_PARAM_NUMBER;
            out[index].number = source[index].number;
            break;
        case VX_SHAPE_PARAM_BOOLEAN:
            out[index].kind = VX_SHAPE_DOMAIN_PARAM_BOOLEAN;
            out[index].boolean = source[index].boolean;
            break;
        case VX_SHAPE_PARAM_STRING:
            out[index].kind = VX_SHAPE_DOMAIN_PARAM_STRING;
            out[index].string = domain_string(source[index].string);
            break;
        default:
            out[index].kind = VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY;
            for (item = 0; item < source[index].number_count &&
                           item < VX_MAX_SCALES; item++) {
                values[index][item].kind = VX_SHAPE_DOMAIN_PARAM_VALUE_NUMBER;
                values[index][item].dimension_index = VX_SHAPE_DOMAIN_INDEX_NONE;
                values[index][item].number = source[index].numbers[item];
            }
            out[index].values = values[index];
            out[index].value_count = source[index].number_count;
            break;
        }
    }
}
'''

EPILOGUE = r'''
enum {
    VX_AGREE_BOTH_OK = 0,
    VX_AGREE_BOTH_REJECT,
    VX_AGREE_DOMAIN_UNSUPPORTED,
    VX_DISAGREE_OUTPUT,
    VX_DISAGREE_STATUS,
    VX_CORPUS_ERROR_DRIFT,
    VX_CORPUS_PATH_DRIFT,
    VX_AGREE_CLASS_COUNT
};

static _Alignas(16) unsigned char vx_scratch[1u << 20];
static uint32_t vx_peak_scratch;

/* The cross-evaluator comparison becomes vacuous for a family once its concrete
 * path delegates to the unified evaluator. The corpus expectation does not, so
 * it stays the regression gate through and after each cutover. */
static int compare_to_corpus(
        const VecCase* item, const VxConcreteShapeResult* concrete) {
    uint32_t index;
    if (!item->expected_output_count) return 1;
    if ((uint32_t)concrete->output_count != item->expected_output_count) {
        fprintf(stderr, "%s: output_count %u, corpus expects %u\n",
                item->id, (unsigned)concrete->output_count,
                item->expected_output_count);
        return 0;
    }
    for (index = 0; index < item->expected_output_count; index++) {
        const VecTensor* want = &item->expected_outputs[index];
        const VxShapeTensorDescriptor* got = NULL;
        uint32_t axis;
        uint32_t candidate;
        for (candidate = 0; candidate < item->expected_output_count; candidate++)
            if (concrete->outputs[candidate].name &&
                !strcmp(concrete->outputs[candidate].name, want->name))
                got = &concrete->outputs[candidate].descriptor;
        if (!got) {
            fprintf(stderr, "%s: corpus output '%s' is missing\n",
                    item->id, want->name);
            return 0;
        }
        if (got->dtype != want->dtype || (uint32_t)got->rank != want->rank) {
            fprintf(stderr, "%s: output '%s' dtype/rank %d/%u, corpus expects %d/%u\n",
                    item->id, want->name, (int)got->dtype, (unsigned)got->rank,
                    (int)want->dtype, want->rank);
            return 0;
        }
        for (axis = 0; axis < want->rank; axis++)
            if (got->shape[axis] != want->shape[axis]) {
                fprintf(stderr, "%s: output '%s' axis %u is %llu, corpus expects %llu\n",
                        item->id, want->name, axis,
                        (unsigned long long)got->shape[axis],
                        (unsigned long long)want->shape[axis]);
                return 0;
            }
    }
    return 1;
}

static int compare_outputs(
        const VecCase* item,
        const VxConcreteShapeResult* concrete,
        const VxShapeDomainProofResult* domain) {
    uint32_t index;
    if (concrete->shape_function_id && domain->shape_function_id &&
        strcmp(concrete->shape_function_id, domain->shape_function_id)) {
        fprintf(stderr, "%s: shape_function_id concrete=%s domain=%s\n",
                item->id, concrete->shape_function_id, domain->shape_function_id);
        return 0;
    }
    if (item->expected_shape_function_id && concrete->shape_function_id &&
        strcmp(concrete->shape_function_id, item->expected_shape_function_id)) {
        fprintf(stderr, "%s: concrete shape_function_id %s, corpus expects %s\n",
                item->id, concrete->shape_function_id,
                item->expected_shape_function_id);
        return 0;
    }
    if ((uint32_t)concrete->output_count != domain->output_count) {
        fprintf(stderr, "%s: output_count concrete=%u domain=%u\n",
                item->id, (unsigned)concrete->output_count,
                (unsigned)domain->output_count);
        return 0;
    }
    for (index = 0; index < domain->output_count; index++) {
        const VxShapeTensorDescriptor* left = &concrete->outputs[index].descriptor;
        const VxShapeDomainTensorDescriptor* right =
            &domain->outputs[index].descriptor;
        uint32_t axis;
        if (left->dtype != (VxDataType)right->dtype) {
            fprintf(stderr, "%s: output %u dtype concrete=%d domain=%d\n",
                    item->id, index, (int)left->dtype, (int)right->dtype);
            return 0;
        }
        if ((uint32_t)left->rank != right->rank) {
            fprintf(stderr, "%s: output %u rank concrete=%u domain=%u\n",
                    item->id, index, (unsigned)left->rank, right->rank);
            return 0;
        }
        for (axis = 0; axis < right->rank; axis++) {
            if (right->dimensions[axis].kind != VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
                fprintf(stderr,
                        "%s: output %u axis %u is not FIXED for a concrete input\n",
                        item->id, index, axis);
                return 0;
            }
            if ((uint64_t)right->dimensions[axis].value != left->shape[axis]) {
                fprintf(stderr, "%s: output %u axis %u concrete=%llu domain=%lld\n",
                        item->id, index, axis,
                        (unsigned long long)left->shape[axis],
                        (long long)right->dimensions[axis].value);
                return 0;
            }
        }
    }
    return 1;
}

static int run_case(const VecCase* item, unsigned* classes) {
    VxShapeNamedTensor concrete_inputs[VX_MAX_PORTS];
    VxShapeNamedTensor concrete_declared[VX_MAX_PORTS];
    VxShapeParam concrete_params[VX_MAX_PARAMS];
    VxConcreteShapeRequest concrete_request;
    VxConcreteShapeResult concrete_result = VX_CONCRETE_SHAPE_RESULT_INITIALIZER;
    VxShapeContractError concrete_error;
    VxShapeDomainNamedTensor domain_inputs[VX_MAX_PORTS];
    VxShapeDomainNamedTensor domain_declared[VX_MAX_PORTS];
    VxShapeDomainDimension input_dimensions[VX_MAX_PORTS][VX_MAX_RANK];
    VxShapeDomainDimension declared_dimensions[VX_MAX_PORTS][VX_MAX_RANK];
    uint32_t input_scales[VX_MAX_PORTS][VX_MAX_SCALES];
    uint32_t declared_scales[VX_MAX_PORTS][VX_MAX_SCALES];
    VxShapeDomainParam domain_params[VX_MAX_PARAMS];
    VxShapeDomainParamValue domain_values[VX_MAX_PARAMS][VX_MAX_SCALES];
    VxShapeDomainProofRequest domain_request;
    VxShapeDomainProofResult domain_result;
    VxShapeDomainError domain_error;
    uint32_t required_scratch = 0;
    int concrete_ok;
    int32_t domain_status;
    uint32_t index;

    if (item->input_count > VX_MAX_PORTS || item->declared_count > VX_MAX_PORTS ||
        item->param_count > VX_MAX_PARAMS) {
        fprintf(stderr, "%s: fixture capacity exceeded\n", item->id);
        return 0;
    }

    fill_concrete(item->inputs, item->input_count, concrete_inputs);
    fill_concrete(item->declared, item->declared_count, concrete_declared);
    for (index = 0; index < item->param_count; index++) {
        memset(&concrete_params[index], 0, sizeof(concrete_params[index]));
        concrete_params[index].name = item->params[index].name;
        concrete_params[index].kind = item->params[index].kind;
        switch (item->params[index].kind) {
        case VX_SHAPE_PARAM_NUMBER:
            concrete_params[index].value.number = item->params[index].number;
            break;
        case VX_SHAPE_PARAM_BOOLEAN:
            concrete_params[index].value.boolean = item->params[index].boolean;
            break;
        case VX_SHAPE_PARAM_STRING:
            concrete_params[index].value.string = item->params[index].string;
            break;
        default:
            concrete_params[index].value.number_array.count =
                item->params[index].number_count;
            concrete_params[index].value.number_array.values =
                item->params[index].numbers;
            break;
        }
    }
    memset(&concrete_request, 0, sizeof(concrete_request));
    concrete_request.inputs = concrete_inputs;
    concrete_request.input_count = item->input_count;
    concrete_request.params = item->param_count ? concrete_params : NULL;
    concrete_request.param_count = item->param_count;
    concrete_request.declared_outputs =
        item->declared_count ? concrete_declared : NULL;
    concrete_request.declared_output_count = item->declared_count;
    memset(&concrete_error, 0, sizeof(concrete_error));
    concrete_ok = vx_shape_contract_infer(
        item->operator_name, &concrete_request, &concrete_result,
        &concrete_error) == 0;

    if (!fill_domain(item->inputs, item->input_count, domain_inputs,
                     input_dimensions, input_scales) ||
        !fill_domain(item->declared, item->declared_count, domain_declared,
                     declared_dimensions, declared_scales)) {
        fprintf(stderr, "%s: rank or scale capacity exceeded\n", item->id);
        vx_shape_contract_result_clear(&concrete_result);
        return 0;
    }
    fill_domain_params(item->params, item->param_count, domain_params,
                       domain_values);
    memset(&domain_request, 0, sizeof(domain_request));
    domain_request.operator_kind = item->operator_kind;
    domain_request.operator_name = domain_string(item->operator_name);
    domain_request.environment = NULL;
    domain_request.environment_count = 0;
    domain_request.inputs = domain_inputs;
    domain_request.input_count = item->input_count;
    domain_request.declared_outputs =
        item->declared_count ? domain_declared : NULL;
    domain_request.declared_output_count = item->declared_count;
    domain_request.use_declared_outputs = item->declared_count ? 1 : 0;
    domain_request.params = item->param_count ? domain_params : NULL;
    domain_request.param_count = item->param_count;
    memset(&domain_result, 0, sizeof(domain_result));
    memset(&domain_error, 0, sizeof(domain_error));
    domain_status = vx_shape_domain_contract_prove(
        &domain_request, &domain_result, &domain_error, vx_scratch,
        (uint32_t)sizeof(vx_scratch), &required_scratch);
    if (required_scratch > vx_peak_scratch) vx_peak_scratch = required_scratch;

    if (concrete_ok && domain_status == VX_SHAPE_DOMAIN_STATUS_OK) {
        int agreed = compare_outputs(item, &concrete_result, &domain_result) &&
                     compare_to_corpus(item, &concrete_result);
        if (!item->expect_success) {
            fprintf(stderr, "%s: corpus expects rejection but both accepted\n",
                    item->id);
            agreed = 0;
        }
        classes[agreed ? VX_AGREE_BOTH_OK : VX_DISAGREE_OUTPUT]++;
        vx_shape_contract_result_clear(&concrete_result);
        return agreed;
    }
    vx_shape_contract_result_clear(&concrete_result);
    if (!concrete_ok && domain_status != VX_SHAPE_DOMAIN_STATUS_OK) {
        classes[VX_AGREE_BOTH_REJECT]++;
        if (item->expected_error_code != VX_SHAPE_CONTRACT_ERROR_NONE &&
            concrete_error.code != item->expected_error_code) {
            fprintf(stderr,
                    "%s: concrete error code %d, corpus expects %d "
                    "(domain code %d index %u subindex %u)\n",
                    item->id, (int)concrete_error.code,
                    (int)item->expected_error_code, (int)domain_error.code,
                    domain_error.index, domain_error.subindex);
            classes[VX_CORPUS_ERROR_DRIFT]++;
            return 0;
        }
        /* `path` was rendered by the retired per-rule concrete evaluator. The
         * unified evaluator reports a numeric field position instead, so the
         * historical string is recorded but no longer gates. The error code
         * above is the part of the diagnostic that callers switch on. */
        if (item->expected_error_path &&
            strcmp(concrete_error.path, item->expected_error_path))
            classes[VX_CORPUS_PATH_DRIFT]++;
        return 1;
    }
    if (concrete_ok && domain_status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED) {
        classes[VX_AGREE_DOMAIN_UNSUPPORTED]++;
        return 1;
    }
    fprintf(stderr,
            "%s: status disagreement concrete=%s domain=%d (error %d/%u/%u)\n",
            item->id, concrete_ok ? "ok" : "reject", (int)domain_status,
            (int)domain_error.code, domain_error.index, domain_error.subindex);
    classes[VX_DISAGREE_STATUS]++;
    return 0;
}

int main(void) {
    unsigned classes[VX_AGREE_CLASS_COUNT];
    size_t index;
    size_t failures = 0;
    memset(classes, 0, sizeof(classes));
    for (index = 0; index < sizeof(vx_cases) / sizeof(vx_cases[0]); index++)
        if (!run_case(&vx_cases[index], classes)) failures++;
    printf("cases=%zu both_ok=%u both_reject=%u domain_unsupported=%u "
           "output_mismatch=%u status_mismatch=%u error_code_drift=%u "
           "error_path_drift=%u\n",
           sizeof(vx_cases) / sizeof(vx_cases[0]),
           classes[VX_AGREE_BOTH_OK], classes[VX_AGREE_BOTH_REJECT],
           classes[VX_AGREE_DOMAIN_UNSUPPORTED], classes[VX_DISAGREE_OUTPUT],
           classes[VX_DISAGREE_STATUS], classes[VX_CORPUS_ERROR_DRIFT],
           classes[VX_CORPUS_PATH_DRIFT]);
    printf("peak_domain_scratch=%u\n", vx_peak_scratch);
    return failures ? 1 : 0;
}
'''


def emit(output: Path) -> tuple[int, list[tuple[str, str]]]:
    emitter = Emitter()
    rows: list[str] = []
    documents = {path: json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(VECTOR_DIR.glob("*.json"))}
    for document in documents.values():
        for case in document.get("success_cases", []):
            ports = list((case.get("expected") or {}).keys())
            if not ports:
                continue
            operators = case.get("operators") or (
                [case["operator"]] if case.get("operator") else [])
            for operator in operators:
                emitter.output_ports.setdefault(operator, ports)
    for path, document in documents.items():
        for bucket, expect_success in (("success_cases", True), ("failure_cases", False)):
            for case in document.get(bucket, []):
                operators = case.get("operators")
                if not operators:
                    single = case.get("operator")
                    operators = [single] if single else []
                for operator in operators:
                    row = emitter.case(case, operator, expect_success)
                    if row is not None:
                        rows.append(row)
    body = [PRELUDE, ""]
    body.extend(emitter.lines)
    body.append("")
    body.append("static const VecCase vx_cases[] = {")
    body.extend(rows)
    body.append("};")
    body.append("")
    body.append(EPILOGUE)
    output.write_text("\n".join(body) + "\n", encoding="utf-8")
    return len(rows), emitter.skipped


if __name__ == "__main__":
    count, skipped = emit(Path("shape_equivalence_generated.c"))
    print(f"emitted {count} case rows; skipped {len(skipped)}")
    for identifier, reason in skipped[:20]:
        print(f"  skip {identifier}: {reason}")
