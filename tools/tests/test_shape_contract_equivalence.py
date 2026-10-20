"""Concrete and symbolic shape evaluators must agree on the concrete domain.

`shape_contract.c` evaluates concrete shapes and `shape_domain_contract.c`
proves symbolic domains. This gate compiles both against the operator shape
vector corpus and requires that, for every fully concrete vector, the
symbolic evaluator with all dimensions `FIXED(n)` produces the same status,
shape function id, dtypes and extents as the concrete evaluator.

A `domain_unsupported` result is recorded but not failed: the symbolic side may
explicitly refuse an input the concrete side accepts. A contradiction - one side
accepting what the other rejects, or different outputs - fails.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.exporter.operator_shape_contracts import prove_operator_shape_domain
from tools.exporter.shape_system import ShapeEnvironment
from tools.shape_equivalence_fixture import emit

NATIVE_INCLUDE = REPOSITORY_ROOT / "native" / "include"
NATIVE_SOURCE = REPOSITORY_ROOT / "native" / "src"
SHAPE_CONTRACT = NATIVE_SOURCE / "runtime" / "shape_contract.c"
SHAPE_DOMAIN_CONTRACT = NATIVE_SOURCE / "runtime" / "shape_domain_contract.c"
KERNEL_REGISTRY = NATIVE_SOURCE / "generated" / "kernel_registry.c"


def _compiler_command() -> list[str] | None:
    configured = os.environ.get("CC")
    command = shlex.split(configured) if configured else ["cc"]
    if not command or shutil.which(command[0]) is None:
        return None
    return command


HEAD_FIXTURE = r'''
#include <assert.h>
#include <stdio.h>
#include "runtime/shape_domain_contract.h"
#define TEXT(value) {(const uint8_t *)(value), sizeof(value) - 1u}

static VxShapeDomainDimension fixed(int value) {
    return (VxShapeDomainDimension){VX_SHAPE_DOMAIN_DIMENSION_FIXED,
        VX_SHAPE_DOMAIN_INDEX_NONE, value};
}
static VxShapeDomainDimension symbol(unsigned index) {
    return (VxShapeDomainDimension){VX_SHAPE_DOMAIN_DIMENSION_SYMBOL, index, 0};
}
static void prove(VxShapeDomainProofRequest *request) {
    _Alignas(16) unsigned char scratch[16384];
    VxShapeDomainProofResult result = {0};
    VxShapeDomainError error = {0};
    uint32_t required = 0;
    int status = vx_shape_domain_contract_prove(request, &result, &error,
        scratch, sizeof(scratch), &required);
    fprintf(stderr, "status=%d code=%d index=%u subindex=%u\n",
        status, error.code, error.index, error.subindex);
    assert(status != VX_SHAPE_DOMAIN_STATUS_SCRATCH_TOO_SMALL);
    if (status == VX_SHAPE_DOMAIN_STATUS_OK) {
        assert(result.output_count == 1);
        assert(result.outputs[0].descriptor.dtype == VX_DTYPE_F32);
        assert(result.outputs[0].descriptor.rank == 3);
        for (unsigned axis = 0; axis < 3; axis++)
            assert(vx_shape_domain_dimensions_equal(
                &result.outputs[0].descriptor.dimensions[axis],
                &request->inputs[0].descriptor.dimensions[axis],
                request->environment, request->environment_count));
    }
    printf("%d\n", status == VX_SHAPE_DOMAIN_STATUS_OK);
}
int main(void) {
    VxShapeDomainConstraint environment[] = {{1,5,1}, {16,32,8}};
    VxShapeDomainDimension shape[] = {fixed(3), symbol(0), symbol(1)};
    VxShapeDomainNamedTensor input = {.name=TEXT("input"), .tensor_index=0,
        .descriptor={.dtype=VX_DTYPE_F32,.rank=3,.dimensions=shape}};
    VxShapeDomainNamedTensor output = {.name=TEXT("out"), .tensor_index=1,
        .descriptor={.dtype=VX_DTYPE_F32,.rank=3,.dimensions=shape}};
    VxShapeDomainParam params[] = {
        {.name=TEXT("heads"),.kind=VX_SHAPE_DOMAIN_PARAM_NUMBER,.number=4},
        {.name=TEXT("rotary_dim"),.kind=VX_SHAPE_DOMAIN_PARAM_NUMBER,.number=4}
    };
    VxShapeDomainProofRequest request = {.operator_kind=VX_OP_ROPE,
        .operator_name=TEXT("RoPE"),.environment=environment,.environment_count=2,
        .inputs=&input,.input_count=1,.params=params,.param_count=1,
        .declared_outputs=&output,.declared_output_count=1,.use_declared_outputs=1};
    prove(&request);
    environment[1].multiple_of=4; prove(&request);
    request.param_count=2; prove(&request);
    params[1].number=6; prove(&request);
    params[1].number=4; environment[1].minimum=20;
    environment[1].maximum=28; prove(&request);

    environment[1]=(VxShapeDomainConstraint){1,9,1};
    VxShapeDomainDimension q[] = {fixed(3),symbol(0),fixed(512)};
    VxShapeDomainDimension kv[] = {fixed(3),symbol(1),fixed(256)};
    VxShapeDomainNamedTensor inputs[] = {
        {.name=TEXT("q"),.tensor_index=0,
         .descriptor={.dtype=VX_DTYPE_F32,.rank=3,.dimensions=q}},
        {.name=TEXT("k"),.tensor_index=1,
         .descriptor={.dtype=VX_DTYPE_F32,.rank=3,.dimensions=kv}},
        {.name=TEXT("v"),.tensor_index=2,
         .descriptor={.dtype=VX_DTYPE_F32,.rank=3,.dimensions=kv}}
    };
    VxShapeDomainParam cross[] = {
        {.name=TEXT("heads"),.kind=VX_SHAPE_DOMAIN_PARAM_NUMBER,.number=4},
        {.name=TEXT("kv_heads"),.kind=VX_SHAPE_DOMAIN_PARAM_NUMBER,.number=2},
        {.name=TEXT("causal"),.kind=VX_SHAPE_DOMAIN_PARAM_BOOLEAN,.boolean=1}
    };
    request.operator_kind=VX_OP_CROSS_SDPA;
    request.operator_name=(VxShapeDomainString)TEXT("CrossSDPA");
    request.inputs=inputs; request.input_count=3;
    output.descriptor=inputs[0].descriptor;
    request.params=cross; request.param_count=3;
    prove(&request);
    cross[1].number=1; kv[2]=fixed(128); prove(&request);
    cross[1].number=3; kv[2]=fixed(384); prove(&request);
    cross[1].number=2; kv[2]=fixed(260); prove(&request);
    return 0;
}
'''


LIFETIME_FIXTURE = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static size_t live, allocations, fail_at;
static void *tracked_malloc(size_t bytes) {
    if (++allocations == fail_at) return NULL;
    void *result = malloc(bytes ? bytes : 1);
    if (result) live++;
    return result;
}
static void *tracked_calloc(size_t count, size_t bytes) {
    void *result = tracked_malloc(count * bytes);
    if (result) memset(result, 0, count * bytes);
    return result;
}
static void tracked_free(void *pointer) {
    if (pointer) { assert(live); live--; free(pointer); }
}
#define malloc tracked_malloc
#define calloc tracked_calloc
#define free tracked_free
#include "runtime/shape_contract.c"
#undef malloc
#undef calloc
#undef free

static void exercise(const char *op, VxConcreteShapeRequest *request,
                     const uint64_t expected[2], int quantized) {
    VxConcreteShapeResult result = {0};
    VxShapeContractError error = {0};
    allocations = fail_at = 0;
    assert(vx_shape_contract_infer(op, request, &result, &error) == 0);
    assert(result.output_count == 1);
    const VxShapeTensorDescriptor *out = &result.outputs[0].descriptor;
    assert(out->rank == 2 && !memcmp(out->shape, expected, 2 * sizeof(uint64_t)));
    if (quantized) {
        assert(out->quantization.count == 3);
        assert(out->quantization.scales[2] == .5f);
    }
    size_t total = allocations;
    vx_shape_contract_result_clear(&result);
    assert(live == 0);
    for (size_t failure = 1; failure <= total; failure++) {
        allocations = 0; fail_at = failure;
        assert(vx_shape_contract_infer(op, request, &result, &error) != 0);
        assert(error.code == VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY);
        vx_shape_contract_result_clear(&result);
        assert(live == 0);
    }
    fail_at = 0;
    for (int i = 0; i < 100; i++) {
        assert(vx_shape_contract_infer(op, request, &result, &error) == 0);
        vx_shape_contract_result_clear(&result);
        assert(live == 0);
    }
    _Alignas(16) unsigned char scratch[16384];
    VxBorrowedConcreteShapeResult borrowed = {0};
    size_t required = 0;
    allocations = 0; fail_at = 1;
    assert(vx_shape_contract_infer_with_scratch(op, request, &borrowed,
        &error, scratch, sizeof(scratch), &required) == 0);
    assert(allocations == 0 && live == 0 && required <= sizeof(scratch));
    assert(borrowed.output_count == 1);
    assert(!memcmp(borrowed.outputs[0].descriptor.shape, expected, 2 * sizeof(uint64_t)));
    fail_at = 0;
}
int main(void) {
    uint64_t input_shape[] = {1, 2}, weight_shape[] = {3, 2}, output_shape[] = {1, 3};
    VxShapeNamedTensor inputs[] = {
        {.name="input", .descriptor={.rank=2, .shape=input_shape, .dtype=VX_DTYPE_F32}},
        {.name="weight", .descriptor={.rank=2, .shape=weight_shape, .dtype=VX_DTYPE_F32}}
    };
    VxShapeNamedTensor output = {.name="out", .descriptor={.rank=2, .shape=output_shape, .dtype=VX_DTYPE_F32}};
    VxShapeParam layout = {.name="weight_layout", .kind=VX_SHAPE_PARAM_STRING, .value.string="dout_din"};
    VxConcreteShapeRequest request = {.inputs=inputs, .input_count=2, .params=&layout,
        .param_count=1, .declared_outputs=&output, .declared_output_count=1};
    exercise("Linear", &request, output_shape, 0);
    float scales[] = {.125f, .25f, .5f}; int32_t zeros[] = {0, 0, 0};
    inputs[0].descriptor = (VxShapeTensorDescriptor){.rank=2, .shape=output_shape,
        .dtype=VX_DTYPE_I8, .quantization={.scheme=VX_SHAPE_QUANTIZATION_PER_AXIS,
            .axis=1, .count=3, .scales=scales, .zero_points=zeros}};
    output.descriptor = inputs[0].descriptor;
    request.input_count=1; request.params=NULL; request.param_count=0;
    exercise("Identity", &request, output_shape, 1);
    return 0;
}
'''


class ShapeContractEquivalenceTests(unittest.TestCase):
    def test_symbolic_evaluator_matches_concrete_on_fixed_dimensions(self) -> None:
        compiler = _compiler_command()
        if compiler is None:
            self.fail("a C compiler is required for the shape equivalence gate")

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "shape_equivalence_generated.c"
            executable = Path(directory) / "shape_equivalence"
            case_count, skipped = emit(fixture)
            self.assertGreater(case_count, 0, "the vector corpus produced no cases")
            self.assertEqual(
                skipped, [], f"the emitter could not project {len(skipped)} vectors")

            subprocess.run(
                [
                    *compiler,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(NATIVE_INCLUDE),
                    "-I",
                    str(NATIVE_SOURCE),
                    str(fixture),
                    str(SHAPE_CONTRACT),
                    str(SHAPE_DOMAIN_CONTRACT),
                    str(KERNEL_REGISTRY),
                    str(NATIVE_SOURCE / "generated" / "operator_vocabulary.c"),
                    "-lm",
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [str(executable)],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

    def test_c_and_python_agree_on_unsampled_bounded_domains(self):
        compiler = _compiler_command()
        self.assertIsNotNone(compiler, "a C compiler is required")
        expected = []
        for minimum, maximum, multiple, rotary in (
            (16, 32, 8, None), (16, 32, 4, None),
            (16, 32, 4, 4), (16, 32, 4, 6), (20, 28, 4, 4),
        ):
            environment = ShapeEnvironment([
                {"name": "S", "min": 1, "max": 5},
                {"name": "W", "min": minimum, "max": maximum, "multiple_of": multiple},
            ])
            params = {"heads": 4}
            if rotary is not None:
                params["rotary_dim"] = rotary
            expected.append(int(prove_operator_shape_domain("RoPE", {
                "environment": environment, "params": params,
                "inputs": {"input": {"shape": [3, "S", "W"], "dtype": "float32"}},
            }).supported))
        environment = ShapeEnvironment([
            {"name": "Q", "min": 1, "max": 5},
            {"name": "K", "min": 1, "max": 9},
        ])
        for kv_heads, width in ((2, 256), (1, 128), (3, 384), (2, 260)):
            expected.append(int(prove_operator_shape_domain("CrossSDPA", {
                "environment": environment,
                "params": {"heads": 4, "kv_heads": kv_heads, "causal": True},
                "inputs": {
                    "q": {"shape": [3, "Q", 512], "dtype": "float32"},
                    "k": {"shape": [3, "K", width], "dtype": "float32"},
                    "v": {"shape": [3, "K", width], "dtype": "float32"},
                },
            }).supported))
        self.assertEqual(expected, [1, 0, 1, 0, 1, 1, 1, 0, 0])
        source = REPOSITORY_ROOT / "native/src"
        with tempfile.TemporaryDirectory() as directory:
            fixture, executable = Path(directory) / "heads.c", Path(directory) / "heads"
            fixture.write_text(HEAD_FIXTURE)
            compiled = subprocess.run([
                *compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(source), "-I", str(REPOSITORY_ROOT / "native/include"),
                str(fixture), str(source / "runtime/shape_contract.c"),
                str(source / "runtime/shape_domain_contract.c"),
                str(source / "generated/kernel_registry.c"),
                str(source / "generated/operator_vocabulary.c"), "-lm", "-o", str(executable),
            ], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            actual = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(actual.returncode, 0, actual.stdout + actual.stderr)
            self.assertEqual([int(line) for line in actual.stdout.splitlines()], expected,
                             actual.stderr)

    def test_projection_and_output_lifetimes(self):
        compiler = _compiler_command()
        self.assertIsNotNone(compiler, "a C compiler is required")
        source = REPOSITORY_ROOT / "native/src"
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "lifetime.c"
            executable = Path(directory) / "lifetime"
            fixture.write_text(LIFETIME_FIXTURE)
            command = [*compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-I", str(source), "-I", str(REPOSITORY_ROOT / "native/include"),
                       str(fixture), str(source / "runtime/shape_domain_contract.c"),
                       str(source / "generated/kernel_registry.c"),
                       str(source / "generated/operator_vocabulary.c"), "-lm", "-o", str(executable)]
            for invocation in (command, [str(executable)]):
                result = subprocess.run(invocation, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
