"""Run the authored CUDA Expand reduction body against a broadcast oracle."""
import ctypes
import itertools
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class CudaTrainingExpand(unittest.TestCase):
    def test_kernel_reduction_rank_promotion_axes_tails_and_accumulation(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        source_text = '''#include <stdint.h>
using vx_u32 = uint32_t;
static vx_u32 global_index;
static vx_u32 vx_training_global_x() { return global_index; }
#define VX_CUDA_GLOBAL
'''
        source_text += (ROOT / 'native/src/backends/cuda/kernels/cuda_training_shape_kernels.inc').read_text()
        source_text += '''
extern "C" void expand(const float* dy, float* dx, const vx_u32* p) {
    for (global_index = 0; global_index < ((p[2] + 63u) / 64u) * 64u; ++global_index)
        vx_cuda_training_expand_backward_f32(dy, dx, p);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            source, library = Path(temporary) / 'expand.cpp', Path(temporary) / 'expand.so'
            source.write_text(source_text)
            subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC',
                            '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                            str(source), '-o', str(library)], check=True)
            expand = ctypes.CDLL(str(library)).expand
            expand.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_uint32)]
            expand.restype = None
            outputs = [(73,), (2, 3), (2, 3, 4), (2, 3, 2, 3),
                       (2, 1, 2, 1, 2, 1, 2, 3)]
            for output_shape in outputs:
                for rank in range(1, len(output_shape) + 1):
                    for mask in itertools.product((False, True), repeat=rank):
                        shape = tuple(d if keep else 1 for d, keep in zip(output_shape[-rank:], mask))
                        with self.subTest(input=shape, output=output_shape):
                            elements, count = int(np.prod(shape)), int(np.prod(output_shape))
                            mapping = np.broadcast_to(np.arange(elements).reshape(shape), output_shape).ravel()
                            dy = ((np.arange(count) % 17 - 8) / 16).astype(np.float32)
                            initial = (np.arange(elements) % 7 / 8).astype(np.float32)
                            expected = initial.copy()
                            dx = initial.copy()
                            p = np.array([rank, len(output_shape), elements, count]
                                         + list(shape) + [0] * (8 - rank)
                                         + list(output_shape) + [0] * (8 - len(output_shape)), np.uint32)
                            for _ in range(2):
                                np.add.at(expected, mapping, dy)
                                expand(dy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       dx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))
                                np.testing.assert_array_equal(dx, expected)


if __name__ == '__main__':
    unittest.main()
