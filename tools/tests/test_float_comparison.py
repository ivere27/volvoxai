"""F32 comparisons retain IEEE ordering and produce exact I32 booleans."""
from __future__ import annotations
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from tools.tests import function_source
from tools.exporter.operator_shape_contracts import infer_concrete_operator_shapes, prove_operator_shape_domain
from tools.exporter.operator_shape_contracts_common import OperatorShapeContractError
from tools.exporter.shape_system import ShapeEnvironment

ROOT = Path(__file__).resolve().parents[2]


class FloatComparisonTests(unittest.TestCase):
    def test_native_gpu_scalar_extent_proof_and_invalid_domains(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = '''
#include <assert.h>
#include <stdint.h>
#define VX_MAX_TENSOR_RANK 8
typedef struct { uint32_t rank; int64_t maximums[8]; } VxDeclaredTensor;
''' + function_source((ROOT / "native/src/runtime/public_api_native_gpu_domain.inc").read_text(),
                     "vx_native_gpu_tensor_maximum_elements") + '''
int main(void) {
    VxDeclaredTensor scalar={0}, tensor={2,{3,7}}, invalid={1,{0}};
    uint64_t elements=0;
    assert(vx_native_gpu_tensor_maximum_elements(&scalar,&elements) && elements==1);
    assert(vx_native_gpu_tensor_maximum_elements(&tensor,&elements) && elements==21);
    assert(!vx_native_gpu_tensor_maximum_elements(&invalid,&elements));
    invalid.rank=9;
    assert(!vx_native_gpu_tensor_maximum_elements(&invalid,&elements));
    invalid.rank=2; invalid.maximums[0]=INT64_MAX; invalid.maximums[1]=3;
    assert(!vx_native_gpu_tensor_maximum_elements(&invalid,&elements));
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-scalar-domain-") as directory:
            root = Path(directory)
            (root / "test.c").write_text(source)
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                            str(root / "test.c"), "-o", str(root / "test")], check=True)
            subprocess.run([str(root / "test")], check=True)

    def test_same_dtype_broadcast_concrete_and_bounded_contracts(self):
        for op in ("Equal", "GreaterOrEqual"):
            for dtype in ("int32", "float32"):
                values = {"a": {"dtype": dtype, "shape": [2, 1]}, "b": {"dtype": dtype, "shape": [1, 5]}}
                inferred = infer_concrete_operator_shapes(op, {"inputs": values})
                self.assertEqual((inferred["out"].shape, inferred["out"].dtype), ((2, 5), "int32"))
                values["a"]["shape"] = ["T", 1]
                proof = prove_operator_shape_domain(op, {"inputs": values,
                    "environment": ShapeEnvironment([{"name": "T", "min": 1, "max": 13}])})
                self.assertTrue(proof.supported, proof)
                self.assertEqual(proof.outputs["out"].shape, ("T", 5))
            with self.assertRaises(OperatorShapeContractError):
                infer_concrete_operator_shapes(op, {"inputs": {
                    "a": {"dtype": "float32", "shape": [2]}, "b": {"dtype": "int32", "shape": [2]}}})

    def test_cuda_production_comparison_nan_zero_infinity_and_integer_abi(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        text = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_typed_graph_kernels.inc").read_text()
        kernel = text[text.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_compare_i32('):text.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_not_i32(')]
        source = r'''
#include <cstdint>
#define VX_CUDA_GLOBAL
using vx_u32=uint32_t; using vx_i32=int32_t;
struct VxCudaBroadcastParams { vx_u32 rank; vx_u32 output_strides[8], a_strides[8], b_strides[8]; };
static unsigned index;
static unsigned vx_global_x() { return index; }
'''+kernel+r'''
extern "C" void compare(const void* a, const void* b, int32_t* out, unsigned rows, unsigned columns, int op) {
    VxCudaBroadcastParams p={}; p.rank=2;
    p.output_strides[0]=columns; p.output_strides[1]=1;
    p.a_strides[0]=1; p.b_strides[1]=1;
    for(index=0; index<rows*columns+3; ++index)
        vx_cuda_compare_i32((const int32_t*)a,(const int32_t*)b,out,rows*columns,p,op);
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-compare-") as directory:
            root = Path(directory)
            cpp = root / "compare.cpp"
            cpp.write_text(source)
            built = subprocess.run([compiler, "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                str(cpp), "-o", str(root / "compare.so")], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            library = ctypes.CDLL(str(root / "compare.so"))
            library.compare.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint] * 2 + [ctypes.c_int]
            for dtype in (np.float32, np.int32):
                a = np.array([-np.inf, -3.5, -0., 0., 2.25, np.inf, np.nan] if dtype == np.float32 else [-2**31, -7, 0, 4, 2**31-1], dtype=dtype)[:, None]
                b = a[:, 0].copy()[None, :]
                for operation, oracle in ((0, np.equal), (1, np.greater_equal)):
                    expected = oracle(a, b).astype(np.int32)
                    actual = np.empty_like(expected)
                    library.compare(a.ctypes.data, b.ctypes.data, actual.ctypes.data, len(a), b.size,
                                    operation + (16 if dtype == np.float32 else 0))
                    np.testing.assert_array_equal(actual, expected)

    def test_native_typed_control_metadata_dtype_scalar_and_invalid_operation(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = r'''
#include <assert.h>
#include "native/src/backends/typed_control_plan.h"
int main(void) {
    float a[2]={0}, b[3]={0}; int32_t output[6];
    uint32_t out_stride[8]={3,1}, a_stride[8]={1,0}, b_stride[8]={0,1};
    VxTypedControlMetadata metadata; size_t ab,bb,ob;
    assert(vx_typed_control_compare_plan((int32_t*)a,2,(int32_t*)b,3,output,6,out_stride,a_stride,b_stride,2,17,&metadata,&ab,&bb,&ob));
    assert(metadata.values[2]==1 && metadata.values[3]==VX_DTYPE_F32 && metadata.values[4]==VX_DTYPE_I32);
    assert(ab==8 && bb==12 && ob==24);
    assert(!vx_typed_control_compare_plan((int32_t*)a,2,(int32_t*)b,3,output,6,out_stride,a_stride,b_stride,2,18,&metadata,&ab,&bb,&ob));
    assert(vx_typed_control_compare_plan((int32_t*)a,1,(int32_t*)b,1,output,1,out_stride,a_stride,b_stride,0,16,&metadata,&ab,&bb,&ob));
    assert(metadata.values[1]==0 && metadata.values[2]==0 && metadata.values[3]==VX_DTYPE_F32);
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-compare-plan-") as directory:
            root = Path(directory)
            (root / "test.c").write_text(source)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(root / "test.c"), "-o", str(root / "test")], check=True)
            subprocess.run([str(root / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
