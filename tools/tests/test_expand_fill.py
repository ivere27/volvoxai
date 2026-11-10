"""Production Expand must preserve bits across trailing broadcast runs."""
from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class ExpandFillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-expand-fill-")
        cls.addClassCleanup(cls.directory.cleanup)
        runtime = (ROOT / "native/src/runtime/engine_runtime_f32_cpu.inc").read_text()
        kernel = (ROOT / "native/src/kernels/tensor_basic_ops.inc").read_text()
        source = r'''
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include "engine_internal.h"
static T tensors[2];
static T* node_input_default(Node* node) {(void)node;return &tensors[0];}
T* t_find(const char* name) {(void)name;return &tensors[1];}
'''
        source += function_source(kernel, "fill_u32")
        for name in ("tensor_contiguous_strides", "tensor_broadcast_strides",
                     "tensor_plain_32", "expand_plain_32_forward"):
            source += function_source(runtime, name)
        source += r'''
int expand(const uint32_t* input, uint32_t* output, const int* in_shape,
           const int* out_shape, int in_rank, int out_rank, int integer) {
    memset(tensors,0,sizeof(tensors));
    tensors[0].data=(float*)input;tensors[1].data=(float*)output;
    for(int side=0;side<2;++side) {
        T* tensor=&tensors[side];
        tensor->ndim=side?out_rank:in_rank;
        tensor->dtype=integer?T_I32:T_F32;tensor->elem_size=4;tensor->numel=1;
        for(int axis=0;axis<tensor->ndim;++axis) {
            tensor->shape[axis]=(side?out_shape:in_shape)[axis];
            tensor->numel*=tensor->shape[axis];
        }
    }
    Node node={0};return expand_plain_32_forward(&node);
}
'''
        unit = Path(cls.directory.name) / "expand.c"
        library = unit.with_suffix(".so")
        unit.write_text(source)
        result = subprocess.run([compiler, "-std=c11", "-O3", "-shared", "-fPIC",
            "-Wall", "-Wextra", "-Werror", "-DVOLVOXAI_ENABLE_TRAINING=0",
            *["-I" + str(ROOT / path) for path in ("native/include", "native/third_party",
                "native/src", "native/src/runtime", "native/src/kernels", "native/src/backends")],
            str(unit), "-o", str(library)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.expand.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
        cls.lib.expand.restype = ctypes.c_int

    def run_expand(self, source, shape, integer):
        source = np.asarray(source)
        in_shape = np.asarray(source.shape, dtype=np.int32)
        out_shape = np.asarray(shape, dtype=np.int32)
        # Guard both edges, including non-SIMD tail lengths.
        guarded = np.full(int(np.prod(shape)) + 2, 0xDEADBEEF, np.uint32)
        result = self.lib.expand(source.ctypes.data, guarded[1:].ctypes.data,
            in_shape.ctypes.data, out_shape.ctypes.data, source.ndim, len(shape), integer)
        self.assertEqual(result, 0)
        np.testing.assert_array_equal(guarded[[0, -1]], [0xDEADBEEF, 0xDEADBEEF])
        np.testing.assert_array_equal(guarded[1:-1].reshape(shape), np.broadcast_to(source, shape))

    def test_bitwise_trailing_runs_and_mixed_axes(self):
        patterns = np.asarray([0, 0x80000000, 0x7FC01234, 0x7F800000,
                               0xFF800000, 0xFFFFFFFF, 1, 0x00800000], np.uint32)
        shapes = [((), (3, 5)), ((8, 1), (8, 7921)), ((1, 8, 1, 1), (3, 8, 5, 7)),
                  ((2, 1, 3, 1), (2, 4, 3, 17)), ((1, 3), (5, 3)),
                  ((2, 1, 3), (2, 5, 3)), ((1, 1, 1), (3, 2, 9)),
                  ((2, 3), (2, 3)), ((1, 8, 1, 1), (2, 8, 13, 1))]
        for input_shape, output_shape in shapes:
            for integer in (0, 1):
                with self.subTest(input_shape=input_shape, output_shape=output_shape, integer=integer):
                    count = int(np.prod(input_shape))
                    source = np.resize(patterns, count).reshape(input_shape)
                    self.run_expand(source, output_shape, integer)
                    # Same shape with changed live inputs catches value caching.
                    self.run_expand(np.bitwise_xor(source, np.uint32(0x12345678)), output_shape, integer)

    def test_random_broadcast_ranks(self):
        rng = np.random.default_rng(42)
        for rank in range(1, 9):
            for _ in range(12):
                shape = tuple(int(x) for x in rng.integers(1, 4, size=rank))
                input_rank = int(rng.integers(0, rank + 1))
                source_shape = tuple(x if rng.integers(2) else 1 for x in shape[rank-input_rank:])
                source = rng.integers(0, 2**32, size=source_shape, dtype=np.uint32)
                self.run_expand(source, shape, 1)


if __name__ == "__main__":
    unittest.main()
