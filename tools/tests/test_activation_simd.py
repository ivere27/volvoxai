"""Production activation dispatch: exceptional values, tails and SIMD accuracy."""
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from tools.tests.kernel_pool import bind_kernel_pool

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#include "fast_math.inc"
#include "elementwise_f32.inc"
#include "activations_hard.inc"
#include "activations.inc"
void activation_reference(const float* input, float* output, int n, int op) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        if (op == 0) output[i] = tanhf(x);
        else if (op == 1) output[i] = 1.0f / (1.0f + accurate_expf(-x));
        else if (op == 2) output[i] = x * (1.0f / (1.0f + accurate_expf(-x)));
        else output[i] = 0.5f * x * (1.0f + gelu_erf_f32(x * 0.7071067811865475f));
    }
}
'''


class ActivationSimdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-activation-simd-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        source = cls.root / "activations.c"
        source.write_text(SOURCE)
        library = cls.root / "activations.so"
        subprocess.run([compiler, "-std=c11", "-O3", "-ffp-contract=off",
                        "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-shared", "-fPIC", "-pthread", "-I" + str(ROOT / "native/src/kernels"),
                        str(source), str(ROOT / "native/src/kernels/cpu_features.c"),
                        str(ROOT / "native/src/kernels/thread_pool.c"), "-lm", "-o", str(library)], check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.vx_set_num_threads.argtypes = [ctypes.c_int]
        cls.addClassCleanup(bind_kernel_pool(cls.lib))
        for name in ("tanh", "sigmoid", "silu", "gelu"):
            getattr(cls.lib, name + "_f32").argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        cls.lib.clip_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_float, ctypes.c_float]
        cls.lib.activation_reference.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_int, ctypes.c_int]

    def run_activation(self, name, values, inplace=False):
        values = np.ascontiguousarray(values, dtype=np.float32)
        actual = values.copy() if inplace else np.empty_like(values)
        source = actual if inplace else values
        getattr(self.lib, name + "_f32")(source.ctypes.data, actual.ctypes.data, values.size)
        return actual

    def test_tanh_full_range_and_small_values(self):
        rng = np.random.default_rng(16180)
        small = np.concatenate([np.geomspace(1e-45, .25, 8192),
                                -np.geomspace(1e-45, .25, 8192)]).astype(np.float32)
        edges = np.array([-2, 2, -2**-12, 2**-12, -10, 10, -0., 0., -np.inf, np.inf, np.nan], dtype=np.float32)
        with np.errstate(invalid="ignore"):
            edges = np.concatenate([edges, np.nextafter(edges, np.float32(np.inf)),
                                     np.nextafter(edges, np.float32(-np.inf))])
        corpus = (small, np.linspace(-20, 20, 100003, dtype=np.float32),
                  rng.integers(0, 2**32, 100003, dtype=np.uint32).view(np.float32),
                  np.tile(edges, 17))
        for threads in (1, 4):
            self.lib.vx_set_num_threads(threads)
            for values in corpus:
                for inplace in (False, True):
                    with self.subTest(threads=threads, size=values.size, inplace=inplace):
                        actual = self.run_activation("tanh", values, inplace)
                        with np.errstate(invalid="ignore"):
                            expected = np.tanh(values.astype(np.float64)).astype(np.float32)
                        np.testing.assert_allclose(actual, expected, rtol=8e-7, atol=0)
                        zero = values == 0
                        np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(values[zero]))
                        self.assertTrue(np.all(np.abs(actual[~np.isnan(values)]) <= 1))

    def test_exceptional_values_in_every_lane_and_scalar_tail(self):
        self.lib.vx_set_num_threads(1)
        for length in (7, 8, 9, 15, 16, 17, 31, 32, 33, 2047, 2048, 2049,
                       16383, 16384, 16385):
            values = np.resize(np.array([np.nan, -np.inf, np.inf, -0., 0.,
                                         1e-8, -1e-8, 1e-40, -1e-40], dtype=np.float32), length)
            for op, name in enumerate(("tanh", "sigmoid", "silu", "gelu")):
                expected = np.empty_like(values)
                self.lib.activation_reference(values.ctypes.data, expected.ctypes.data, length, op)
                for inplace in (False, True):
                    with self.subTest(name=name, length=length, inplace=inplace):
                        actual = self.run_activation(name, values, inplace)
                        np.testing.assert_allclose(actual, expected, rtol=8e-7, atol=0)
                        zeros = expected == 0
                        np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))

    def test_clip_matches_scalar_conditionals_bitwise(self):
        rng = np.random.default_rng(2718)
        values = rng.integers(0, 2**32, 4099, dtype=np.uint32).view(np.float32)
        values[:10] = [np.nan, -np.inf, np.inf, -0., 0., -1., 1., 1e-40, -1e-40, np.nan]
        for minimum, maximum in ((-1., 1.), (0., 0.), (-0., -0.),
                                 (-np.inf, np.inf), (np.nan, 1.), (-1., np.nan)):
            with np.errstate(invalid="ignore"):
                expected = np.where(values < minimum, np.float32(minimum), values)
                expected = np.where(expected > maximum, np.float32(maximum), expected)
            for inplace in (False, True):
                actual = values.copy() if inplace else np.empty_like(values)
                source = actual if inplace else values
                self.lib.clip_f32(source.ctypes.data, actual.ctypes.data, values.size, minimum, maximum)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_shared_exp_consumers_keep_their_existing_finite_contract(self):
        values = np.linspace(-80, 80, 100003, dtype=np.float32)
        for op, name in ((1, "sigmoid"), (2, "silu"), (3, "gelu")):
            expected = np.empty_like(values)
            self.lib.activation_reference(values.ctypes.data, expected.ctypes.data, values.size, op)
            actual = self.run_activation(name, values)
            # GELU retains its documented erf approximation; the sigmoid/SiLU
            # kernels retain the scalar bounded-exp arithmetic.
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-7 if op == 3 else 0)

    def test_wasm_simd_uses_production_tanh_and_scalar_math(self):
        compiler = shutil.which("clang-17") or shutil.which("clang")
        node = shutil.which("node")
        if not compiler or not node:
            self.skipTest("WASM compiler and Node required")
        source = self.root / "wasm.c"
        source.write_text("#include <math.h>\n" + SOURCE + r'''
int vx_kernels_thread_count(void) { return 1; }
void vx_kernels_parallel_for(int n, int grain, VxKernelParallelFn fn, void* ctx) {
    (void)grain; fn(ctx, 0, n);
}
static float input[8193], output[8193];
float* inputs(void) { return input; }
float* outputs(void) { return output; }
void run(int n) { tanh_f32(input, output, n); }
void run_silu(int n) { silu_f32(input, output, n); }
void run_gelu(int n) { gelu_f32(input, output, n); }
void run_sigmoid(int n) { sigmoid_f32(input, output, n); }
void run_helper(int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4)
        wasm_v128_store(output + i, vx_tanh_wasm_simd(wasm_v128_load(input + i)));
    for (; i < n; i++) output[i] = tanhf(input[i]);
}
''')
        module = self.root / "tanh.wasm"
        subprocess.run([compiler, "--target=wasm32", "-std=c11", "-O3", "-ffp-contract=off",
                        "-ffreestanding", "-msimd128", "-nostdlib",
                        "-I" + str(ROOT / "native/src/kernels"),
                        "-I" + str(ROOT / "native/src/runtime/wasm_freestanding/include"),
                        str(source), str(ROOT / "native/src/runtime/wasm_math.c"),
                        "-Wl,--no-entry", "-Wl,--export=inputs", "-Wl,--export=outputs",
                        "-Wl,--export=run", "-Wl,--export=run_helper",
                        "-Wl,--export=run_silu", "-Wl,--export=run_gelu",
                        "-Wl,--export=run_sigmoid", "-o", str(module)], check=True)
        script = r'''
const fs=require('fs');
WebAssembly.instantiate(fs.readFileSync(process.argv[1]),{}).then(({instance:{exports:e}})=>{
  const input=new Float32Array(e.memory.buffer,e.inputs(),8193);
  const output=new Float32Array(e.memory.buffer,e.outputs(),8193);
  let maxAbsoluteError=0;
  for(const run of [e.run,e.run_helper]) {
  for(const length of [7,8,9,15,16,17,31,32,33,8193]) {
    for(const mode of [0,1,2]) {
      for(let i=0;i<length;i++) input[i]=mode===0?(i%2?-1:1)*2**(-(i%149)):
        mode===1?(i/length*40-20):[NaN,-Infinity,Infinity,-0,0,1e-8,-1e-8][i%7];
      run(length);
      for(let i=0;i<length;i++) {
        const expected=Math.fround(Math.tanh(input[i])), actual=output[i];
        if(Number.isNaN(expected)) {if(!Number.isNaN(actual)) throw Error('NaN lost'); continue;}
        if(input[i]===0 && !Object.is(actual,input[i])) throw Error('signed zero lost');
        const delta=Math.abs(actual-expected);
        maxAbsoluteError=Math.max(maxAbsoluteError,delta);
        if(delta>8e-7*Math.abs(expected)) throw Error(JSON.stringify({i,length,input:input[i],actual,expected}));
      }
    }
  }
  }
  // Identical values retain their bits in a full vector, a one-element tail,
  // and a reversed array containing exceptional neighbors.
  const bits=new Uint32Array(e.memory.buffer,e.outputs(),8193);
  for(const run of [e.run,e.run_silu,e.run_gelu,e.run_sigmoid]) {
    const values=Float32Array.from({length:8193},(_,i)=>i%31<7 ?
      [NaN,Infinity,-Infinity,-0,0,1e-40,-1e-40][i%31] : (i/8193*16-8));
    const expected=new Uint32Array(values.length);
    const expectedValues=new Float32Array(expected.buffer);
    // WASM permits arithmetic NaN payload/sign canonicalization; finite bits,
    // infinities and both zero signs remain exact.
    const same=(i,j)=>bits[i]===expected[j] ||
      (Number.isNaN(output[i]) && Number.isNaN(expectedValues[j]));
    for(let i=0;i<values.length;i++) {input[0]=values[i];run(1);expected[i]=bits[0];}
    for(const length of [1,3,4,7,8,9,15,16,17,31,32,33,8193]) {
      input.set(values.subarray(0,length));run(length);
      for(let i=0;i<length;i++) if(!same(i,i))
        throw Error(JSON.stringify({kind:'partition',fn:run.name,length,i,input:values[i],actual:bits[i],expected:expected[i]}));
      for(let i=0;i<length;i++) input[i]=values[length-1-i];
      run(length);
      for(let i=0;i<length;i++) if(!same(i,length-1-i))
        throw Error(JSON.stringify({kind:'permutation',fn:run.name,length,i,input:values[length-1-i],actual:bits[i],expected:expected[length-1-i]}));
    }
  }
  console.log(JSON.stringify({maxAbsoluteError}));
});
'''
        result = subprocess.run([node, "-e", script, str(module)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertLessEqual(json.loads(result.stdout)["maxAbsoluteError"], 8e-7)


if __name__ == "__main__":
    unittest.main()
