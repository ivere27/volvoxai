"""Portable unary arithmetic, including TTS rounding and stable negative tails."""
import ctypes
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.exporter.operator_shape_contracts import prove_operator_shape_domain
from tools.exporter.shape_system import ShapeEnvironment

ROOT = Path(__file__).resolve().parents[2]
OPERATORS = ('Exp', 'Sin', 'Cos', 'Sqrt', 'Log', 'Erf', 'Floor', 'Ceil', 'Round', 'Atan', 'Softplus', 'Neg')


class UnaryMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('cc')
        if not compiler:
            raise unittest.SkipTest('C compiler unavailable')
        cls.directory = tempfile.TemporaryDirectory(prefix='volvox-unary-')
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(cls.directory.name) / 'unary.c'
        source.write_text('#include <math.h>\n#define VX_FASTMATH_X86_AVX2 0\n#define accurate_expf expf\n' +
                          (ROOT / 'native/src/kernels/activations_hard.inc').read_text())
        library = source.with_suffix('.so')
        subprocess.run([compiler, '-std=c11', '-O2', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                        '-shared', '-fPIC', str(source), '-lm', '-o', str(library)], check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.unary_math_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    def test_all_unary_bounded_contracts(self):
        environment = ShapeEnvironment([{'name': 'T', 'min': 1, 'max': 33}])
        for operator in OPERATORS:
            with self.subTest(operator=operator):
                proof = prove_operator_shape_domain(operator, {'environment': environment,
                    'inputs': {'input': {'shape': [2, 'T', 4], 'dtype': 'float32'}}, 'params': {}})
                self.assertTrue(proof.supported, proof)
                self.assertEqual(proof.outputs['out'].shape, (2, 'T', 4))

    def test_independent_double_precision_oracles(self):
        rng = np.random.default_rng(4015)
        values = np.concatenate([np.linspace(-110, 110, 22001),
            rng.integers(0, 2**32, 5000, dtype=np.uint32).view(np.float32),
            [-np.inf, np.inf, np.nan, -0., 0., -2.5, -1.5, -.5, .5, 1.5, 2.5, 8388607.5]]).astype(np.float32)
        references = (np.exp, np.sin, np.cos, np.sqrt, np.log,
            np.vectorize(math.erf), np.floor, np.ceil, np.rint, np.arctan,
            lambda x: np.logaddexp(0, x), np.negative)
        for operation, reference in enumerate(references):
            with self.subTest(operator=OPERATORS[operation]), np.errstate(all='ignore'):
                expected = reference(values.astype(np.float64)).astype(np.float32)
                actual = np.empty_like(values)
                self.lib.unary_math_f32(values.ctypes.data, actual.ctypes.data, values.size, operation)
                np.testing.assert_allclose(actual, expected, rtol=4e-7, atol=0 if operation != 9 else 2e-7)
                if operation in (3, 5, 6, 7, 8, 9):
                    self.assertEqual(bool(np.signbit(actual[-10])), bool(np.signbit(expected[-10])))
        # Tiny finite Softplus outputs must survive log(1+exp(x)) cancellation.
        tails = np.array([-100, -40, -20], dtype=np.float32)
        actual = np.empty_like(tails)
        self.lib.unary_math_f32(tails.ctypes.data, actual.ctypes.data, 3, 10)
        self.assertTrue(np.all(actual > 0))


if __name__ == '__main__':
    unittest.main()
