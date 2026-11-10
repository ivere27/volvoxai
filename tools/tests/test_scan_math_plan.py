"""Checked scan geometry shared by the four WGSL device backends."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


class ScanMathPlanTests(unittest.TestCase):
    def test_webgpu_unary_dispatch_carries_the_shader_operation_word(self):
        root = Path(__file__).resolve().parents[2]
        table = (root / "native/src/backends/webgpu_dispatch.inc").read_text()
        for operation, operator in enumerate(("EXP", "SIN", "COS", "SQRT", "LOG", "ERF", "FLOOR", "CEIL", "ROUND", "ATAN", "SOFTPLUS", "NEG")):
            with self.subTest(operator=operator):
                entry = re.search(
                    rf"\{{ VX_OP_{operator}, VX_SHADER_INFERENCE_UNARY_MATH,\s*"
                    r"\w+, \d+,\s*(\w+), 4,", table)
                self.assertIsNotNone(entry)
                parameters = re.search(
                    rf"{entry.group(1)}\[\] = \{{(.*?)\n\}};", table, re.S)
                self.assertIsNotNone(parameters)
                words = re.findall(r"\{ ([^}]+) \}", parameters.group(1))
                selector = [field.strip() for field in words[1].split(",")]
                self.assertEqual(selector[0], "VX_WEBGPU_VALUE_CONST")
                # vx_webgpu_value reads CONST from word (the fourth field),
                # while a (the third field) indexes axes/parameter IDs.
                self.assertEqual(int(selector[3]), operation)

    def test_axes_bounds_aliases_and_launch_geometry(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        source = r'''
#include <assert.h>
#include "native/src/backends/scan_math_plan.h"
int main(void) {
    float input[30] = {0}, output[30] = {0};
    int shape[3] = {2, 5, 3};
    const unsigned scans[3] = {15, 6, 10}, inner[3] = {15, 3, 1};
    VxScanMathPlan plan;
    for (int axis = -3; axis < 3; ++axis)
        for (int exclusive = 0; exclusive < 2; ++exclusive)
            for (int reverse = 0; reverse < 2; ++reverse) {
                int normalized = axis < 0 ? axis + 3 : axis;
                assert(vx_cumsum_plan(input, output, 30, shape, 3, axis,
                    exclusive, reverse, VX_DTYPE_F32, &plan));
                assert(plan.params[0] == scans[normalized]);
                assert(plan.params[1] == (unsigned)shape[normalized]);
                assert(plan.params[2] == inner[normalized]);
                assert(plan.params[3] == (unsigned)exclusive);
                assert(plan.params[4] == (unsigned)reverse);
                assert(plan.params[5] == 0 && plan.bytes == sizeof(input));
                assert(vx_cumsum_plan(input, output, 30, shape, 3, axis,
                    exclusive, reverse, VX_DTYPE_I32, &plan));
                assert(plan.params[5] == 1);
            }
    assert(!vx_cumsum_plan(input, input, 30, shape, 3, 1, 0, 0, VX_DTYPE_F32, &plan));
    assert(!vx_cumsum_plan(input, input + 1, 30, shape, 3, 1, 0, 0, VX_DTYPE_F32, &plan));
    assert(!vx_cumsum_plan(input, output, 29, shape, 3, 1, 0, 0, VX_DTYPE_F32, &plan));
    assert(!vx_cumsum_plan(input, output, 30, shape, 3, 3, 0, 0, VX_DTYPE_F32, &plan));
    assert(!vx_cumsum_plan(input, output, 30, shape, 3, -4, 0, 0, VX_DTYPE_F32, &plan));
    assert(!vx_cumsum_plan(input, output, 30, shape, 3, 1, 2, 0, VX_DTYPE_F32, &plan));
    assert(!vx_cumsum_plan(input, output, 30, shape, 3, 1, 0, -1, VX_DTYPE_F32, &plan));
    assert(!vx_cumsum_plan(input, output, 30, shape, 3, 1, 0, 0, VX_DTYPE_F16, &plan));
    int invalid[3] = {2, 0, 3};
    assert(!vx_cumsum_plan(input, output, 30, invalid, 3, 1, 0, 0, VX_DTYPE_F32, &plan));
    int huge[3] = {INT_MAX, INT_MAX, INT_MAX};
    assert(!vx_cumsum_plan(input, output, 30, huge, 3, 1, 0, 0, VX_DTYPE_F32, &plan));
    for (int op = 0; op < 12; ++op) {
        assert(vx_unary_math_plan(input, output, 30, op, &plan));
        assert(plan.params[0] == 30 && plan.params[1] == (unsigned)op);
    }
    assert(!vx_unary_math_plan(input, output, 30, 12, &plan));
    assert(!vx_unary_math_plan(input, output, 0, 0, &plan));
    uint32_t grid[3];
    vx_scan_math_grid(65535u * 64u + 1u, grid);
    assert(grid[0] == 65535u && grid[1] == 2u && grid[2] == 1u);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            test = path / "scan_math.c"
            test.write_text(source)
            binary = path / "scan_math"
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(root), str(test), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
