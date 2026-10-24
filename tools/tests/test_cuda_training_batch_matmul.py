"""Production CUDA BMM host admission and batch-index arithmetic, without a GPU."""
from pathlib import Path
import ctypes
import itertools
import math
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CudaTrainingBatchMatMul(unittest.TestCase):
    def test_host_contract_before_device_access(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'host'
            subprocess.run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=undefined', str(ROOT / 'native/tests/test_cuda_training_batch_matmul_host.c'),
                '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=30)

    def test_production_broadcast_index_order(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        production = (ROOT / 'native/src/backends/cuda/kernels/cuda_training_batch_matmul_kernels.inc').read_text()
        production = production[:production.index('extern "C" VX_CUDA_GLOBAL')]
        wrapper = '''\nextern "C" void mapped(const vx_u32* p, vx_u32 source, vx_u32 reduced,
            vx_u32 axis, vx_u32* out) {
            VxCudaTrainingBatchMatMulParams params;
            for(int i=0;i<32;i++) params.values[i]=p[i];
            vx_cuda_training_bmm_batch(params,source,reduced,axis,out[0],out[1],out[2]);
        }\n'''
        with tempfile.TemporaryDirectory() as tmp:
            source, library = Path(tmp) / 'mapping.cpp', Path(tmp) / 'mapping.so'
            source.write_text('#include <stdint.h>\nusing vx_u32=uint32_t;\n#define VX_CUDA_DEVICE\n' + production + wrapper)
            subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC','-Wall','-Wextra',
                            '-Werror',str(source),'-o',str(library)],check=True)
            function = ctypes.CDLL(str(library)).mapped
            u32 = ctypes.c_uint32
            function.argtypes = [ctypes.POINTER(u32),u32,u32,u32,ctypes.POINTER(u32)]
            # Exhaust every broadcast layout through rank six, then asymmetric rank eight.
            shapes = [dims for rank in range(5) for dims in itertools.product((1,2,3),repeat=rank)]
            shapes += [(2,1,3,1,2,2)]
            for dims in shapes:
                for assignment in itertools.product((1,2,3),repeat=len(dims)):
                    a_dims = [d if choice & 1 else 1 for d,choice in zip(dims,assignment)]
                    b_dims = [d if choice & 2 else 1 for d,choice in zip(dims,assignment)]
                    p = [0]*32; p[0]=len(dims); p[1:8]=[2,3,4,math.prod(a_dims)*6,math.prod(b_dims)*12,math.prod(dims)*8,math.prod(dims)]
                    ast,bst,ost=6,12,1
                    for axis in reversed(range(len(dims))):
                        p[8+axis]=dims[axis]; p[14+axis]=ast if a_dims[axis]>1 else 0
                        p[20+axis]=bst if b_dims[axis]>1 else 0; p[26+axis]=ost
                        ast*=a_dims[axis]; bst*=b_dims[axis]; ost*=dims[axis]
                    expected = [{},{}]
                    for batch,coord in enumerate(itertools.product(*(range(d) for d in dims))):
                        ai=bi=0
                        for axis,c in enumerate(coord):
                            ai=ai*a_dims[axis]+(c if a_dims[axis]>1 else 0)
                            bi=bi*b_dims[axis]+(c if b_dims[axis]>1 else 0)
                        entry=(batch,ai*6,bi*12)
                        expected[0].setdefault(ai,[]).append(entry)
                        expected[1].setdefault(bi,[]).append(entry)
                    for side in range(2):
                        for source_batch,entries in expected[side].items():
                            for reduced,entry in enumerate(entries):
                                actual=(u32*3)()
                                function((u32*32)(*p),source_batch,reduced,14+side*6,actual)
                                self.assertEqual(tuple(actual),entry)


if __name__ == '__main__':
    unittest.main()
