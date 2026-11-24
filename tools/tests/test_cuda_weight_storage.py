"""Numerically execute authored compact-weight CUDA bodies without a GPU."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class CudaWeightStorageTests(unittest.TestCase):
    def test_authored_storage_kernels(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='volvox-cuda-weight-storage-') as directory:
            binary = Path(directory) / 'test'
            built = subprocess.run([
                compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-UNDEBUG', '-ffp-contract=off',
                str(root / 'native/tests/test_cuda_weight_storage.cpp'), '-o', str(binary),
            ], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            if checked.returncode == 77:
                self.skipTest('Cooperative CUDA emulator requires Unix ucontext')
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn('decode, dense, lookup and MoE numerical PASS', checked.stdout)


if __name__ == '__main__':
    unittest.main()
