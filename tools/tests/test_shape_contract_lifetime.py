"""Shape inference releases heap temporaries on success and allocation failure."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.tests.test_shape_contract_equivalence import _compiler_command, REPOSITORY_ROOT


FIXTURE = r'''
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


class ShapeContractLifetimeTests(unittest.TestCase):
    def test_projection_and_output_lifetimes(self):
        compiler = _compiler_command()
        self.assertIsNotNone(compiler, "a C compiler is required")
        source = REPOSITORY_ROOT / "native/src"
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "lifetime.c"
            executable = Path(directory) / "lifetime"
            fixture.write_text(FIXTURE)
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
