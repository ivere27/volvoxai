"""Independent ONNX checks for time-major CPU and cooperative CUDA LSTM."""
from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


from tools.lstm_fixtures import lstm_model, lstm_tail_fixture


class LstmNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        c_compiler = shutil.which("cc")
        if not compiler or not c_compiler:
            raise unittest.SkipTest("C and C++ compilers required")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-lstm-")
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        scalar = function_source((ROOT / "native/src/kernels/broadcast_ops.inc").read_text(), "matmul_f32_out_in_scalar")
        scalar = scalar.replace("matmul_f32_out_in_scalar", "matmul_f32_out_in")
        cpu = (ROOT / "native/src/kernels/lstm.inc").read_text()
        cuda = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_lstm_kernels.inc").read_text()
        source = r'''
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstdlib>
#include <pthread.h>
#include <thread>
#include <vector>
#include "lstm_workspace.h"
extern "C" size_t workspace_bytes(uint64_t time, uint64_t batch, uint64_t hidden) {
    return vx_lstm_workspace_bytes(time, batch, hidden);
}
#define VX_CUDA_GLOBAL
#define VX_CUDA_SHARED static
using vx_u32 = uint32_t;
struct Dim { unsigned x, y, z; };
static thread_local Dim threadIdx, blockIdx, blockDim;
static vx_u32 vx_global_x() { return blockIdx.x*blockDim.x+threadIdx.x; }
static pthread_barrier_t barrier;
static void vx_sync_threads() { pthread_barrier_wait(&barrier); }
static float vx_add_rn(float a, float b) { return a+b; }
static float vx_mul_rn(float a, float b) { return a*b; }
// Host libm substitutes only the CUDA hardware exp/tanh intrinsics. All gate
// reductions, state transitions, synchronization and indexing are production.
static float vx_exp(float a) { return std::exp(a); }
static float vx_tanh(float a) { return std::tanh(a); }
'''+cuda+r'''
extern "C" int cuda_lstm(const float* x, const float* w, const float* r,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, unsigned time, unsigned batch, unsigned width,
        unsigned hidden, unsigned direction) {
    unsigned flags = (bias?1:0) | (initial_h?2:0) | (initial_c?4:0);
    pthread_barrier_init(&barrier, nullptr, 256);
    std::vector<std::thread> lanes;
    for (unsigned lane=0; lane<256; ++lane) lanes.emplace_back([=] {
        threadIdx.x=lane;
        for (blockIdx.x=0; blockIdx.x<batch*(direction==2?2:1); ++blockIdx.x) {
            vx_cuda_lstm_f32(x,w,r,bias,initial_h,initial_c,output,time,batch,width,hidden,direction,flags);
            vx_sync_threads();
        }
    });
    for (auto& lane:lanes) lane.join();
    pthread_barrier_destroy(&barrier);
    return 1;
}
extern "C" int cuda_lstm_prepared(const float* x, const float* w, const float* r,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, unsigned time, unsigned batch, unsigned width,
        unsigned hidden, unsigned direction) {
    unsigned directions=direction==2?2:1, gates_width=4*hidden;
    unsigned flags=(bias?1:0)|(initial_h?2:0)|(initial_c?4:0);
    std::vector<float> wt(directions*gates_width*width),rt(directions*gates_width*hidden);
    // Independent arithmetic-free transpose here; the physical CUDA corpus
    // separately exercises the authored shared-memory transpose and its tails.
    for(unsigned d=0;d<directions;d++)for(unsigned row=0;row<gates_width;row++) {
        for(unsigned k=0;k<width;k++)wt[(d*width+k)*gates_width+row]=w[(d*gates_width+row)*width+k];
        for(unsigned k=0;k<hidden;k++)rt[(d*hidden+k)*gates_width+row]=r[(d*gates_width+row)*hidden+k];
    }
    unsigned count=time*directions*batch*gates_width;
    std::vector<float> projection(2*count),gates(directions*batch*gates_width),cells(directions*batch*hidden);
    blockDim.x=128;
    for(unsigned i=0;i<count+3;i++) {
        blockIdx.x=i/128;threadIdx.x=i%128;
        vx_cuda_lstm_project_f32(x,wt.data(),projection.data(),time,batch,width,hidden,directions);
    }
    for(unsigned step=0;step<time;step++) {
        for(unsigned i=0;i<directions*batch*gates_width+3;i++) {
            blockIdx.x=i/128;threadIdx.x=i%128;
            vx_cuda_lstm_recurrent_f32(projection.data(),rt.data(),bias,initial_h,output,gates.data(),
                time,batch,hidden,direction,flags,step);
        }
        for(unsigned i=0;i<directions*batch*hidden+3;i++) {
            blockIdx.x=i/128;threadIdx.x=i%128;
            vx_cuda_lstm_update_f32(gates.data(),initial_c,cells.data(),output,
                time,batch,hidden,direction,flags,step);
        }
    }
    return 1;
}
'''
        # Compile the production CPU kernel as C, with its actual SIMD math
        # dependencies. Track its allocator boundary to verify scratch ownership.
        cpu_source = r'''
#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include "fast_math.inc"
typedef union { size_t bytes; max_align_t alignment; } Allocation;
static _Atomic size_t live_bytes, last_bytes;
static _Atomic int fail_allocation;
static void* tracked_malloc(size_t bytes) {
    atomic_store(&last_bytes, bytes);
    if (atomic_load(&fail_allocation)) return NULL;
    Allocation* allocation = malloc(sizeof(*allocation) + bytes);
    if (!allocation) return NULL;
    allocation->bytes = bytes;
    atomic_fetch_add(&live_bytes, bytes);
    return allocation + 1;
}
static void tracked_free(void* pointer) {
    if (!pointer) return;
    Allocation* allocation = (Allocation*)pointer - 1;
    atomic_fetch_sub(&live_bytes, allocation->bytes);
    free(allocation);
}
static void* tracked_realloc(void* pointer, size_t bytes) {
    if (!pointer) return tracked_malloc(bytes);
    Allocation* old = (Allocation*)pointer - 1;
    size_t old_bytes = old->bytes;
    Allocation* allocation = realloc(old, sizeof(*allocation) + bytes);
    if (!allocation) return NULL;
    allocation->bytes = bytes;
    atomic_fetch_sub(&live_bytes, old_bytes);
    atomic_fetch_add(&live_bytes, bytes);
    atomic_store(&last_bytes, bytes);
    return allocation + 1;
}
size_t workspace_live_bytes(void) { return atomic_load(&live_bytes); }
size_t workspace_last_allocation(void) { return atomic_load(&last_bytes); }
void workspace_fail_allocation(int fail) { atomic_store(&fail_allocation, fail); }
#define malloc tracked_malloc
#define realloc tracked_realloc
#define free tracked_free
''' + scalar + '\n' + cpu
        cpu_path = root / "lstm_cpu.c"
        cpu_path.write_text(cpu_source)
        objects = []
        for source_path in (cpu_path, ROOT / "native/src/kernels/cpu_features.c"):
            obj = root / (source_path.stem + ".o")
            result = subprocess.run([c_compiler, "-std=c11", "-O2", "-ffp-contract=off",
                                     "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                                     "-fPIC", "-I", str(ROOT / "native/src/kernels"),
                                     "-c", str(source_path), "-o", str(obj)], capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            objects.append(str(obj))
        cpp = root / "lstm.cpp"
        cpp.write_text(source)
        result = subprocess.run([compiler, "-std=c++14", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                                 "-pthread", "-shared", "-fPIC", "-I", str(ROOT / "native/src/kernels"),
                                 str(cpp), *objects, "-o", str(root / "lstm.so")], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.library = ctypes.CDLL(str(root / "lstm.so"))
        cls.library.workspace_live_bytes.restype = ctypes.c_size_t
        cls.library.workspace_last_allocation.restype = ctypes.c_size_t
        cls.library.workspace_fail_allocation.argtypes = [ctypes.c_int]
        cls.library.workspace_bytes.argtypes = [ctypes.c_uint64] * 3
        cls.library.workspace_bytes.restype = ctypes.c_size_t
        for name in ("lstm_f32", "cuda_lstm", "cuda_lstm_prepared"):
            getattr(cls.library, name).argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_uint] * 5
            getattr(cls.library, name).restype = ctypes.c_int

    def test_workspace_covers_live_arrays_and_rejects_overflow(self):
        self.assertEqual(self.library.workspace_bytes(11, 2, 256), (11 * 2 * 4 * 256 + 2 * 6 * 256) * 4)
        for shape in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (2**31, 1, 1),
                      (2**30, 2, 1), (1, 1, 2**30), (2**64-1, 2**64-1, 2**64-1)):
            self.assertEqual(self.library.workspace_bytes(*shape), 0)

    def test_workspace_is_released_after_each_call_and_thread_exit(self):
        x = np.zeros((1, 1, 1), dtype=np.float32)
        w = np.zeros((1, 32, 1), dtype=np.float32)
        r = np.zeros((1, 32, 8), dtype=np.float32)

        def execute():
            output = np.empty((1, 1, 1, 8), dtype=np.float32)
            self.assertEqual(self.library.lstm_f32(x.ctypes.data, w.ctypes.data,
                r.ctypes.data, None, None, None, output.ctypes.data, 1, 1, 1, 8, 0), 1)
            np.testing.assert_array_equal(output, 0)
            self.assertEqual(self.library.workspace_live_bytes(), 0)
            self.assertEqual(self.library.workspace_last_allocation(),
                             self.library.workspace_bytes(1, 1, 8))

        execute()
        execute()
        for _ in range(3):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(execute).result()
            self.assertEqual(self.library.workspace_live_bytes(), 0)

    def test_workspace_allocation_failure_leaves_output_unchanged(self):
        x = np.zeros(1, dtype=np.float32)
        w = np.zeros(32, dtype=np.float32)
        r = np.zeros(256, dtype=np.float32)
        output = np.full(8, 123, dtype=np.float32)
        self.library.workspace_fail_allocation(1)
        try:
            self.assertEqual(self.library.lstm_f32(x.ctypes.data, w.ctypes.data,
                r.ctypes.data, None, None, None, output.ctypes.data, 1, 1, 1, 8, 0), 0)
        finally:
            self.library.workspace_fail_allocation(0)
        np.testing.assert_array_equal(output, 123)
        self.assertEqual(self.library.workspace_live_bytes(), 0)

    def test_cpu_gate_helpers_preserve_small_values_and_nan(self):
        # Widths cross the AVX2/NEON vector boundary and include scalar tails.
        # Only candidate gates are nonzero: h = .5*tanh(.5*tanh(candidate)).
        for hidden in (7, 8, 9, 16, 17):
            x = np.ones((1, 1, 1), dtype=np.float32)
            w = np.zeros((1, 4 * hidden, 1), dtype=np.float32)
            r = np.zeros((1, 4 * hidden, hidden), dtype=np.float32)
            candidates = np.resize(np.array([1e-8, -1e-8, 1e-20, -1e-20,
                                              .25, -.25, 1., -1.], dtype=np.float32), hidden)
            for exceptional in (False, True):
                current = candidates.copy()
                if exceptional:
                    current[0] = np.nan
                    current[1] = np.inf
                    current[2] = -np.inf
                w[0, 3 * hidden:, 0] = current
                output = np.empty((1, 1, 1, hidden), dtype=np.float32)
                self.assertEqual(self.library.lstm_f32(x.ctypes.data, w.ctypes.data,
                    r.ctypes.data, None, None, None, output.ctypes.data, 1, 1, 1, hidden, 0), 1)
                expected = (.5 * np.tanh(.5 * np.tanh(current.astype(np.float64)))).astype(np.float32)
                np.testing.assert_allclose(output.reshape(-1), expected, rtol=8e-7, atol=0)

    def compare(self, x, w, r, bias, h, c, direction, expected=None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        model = lstm_model(x, w, r, bias, h, c, direction)
        reference = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        if expected is None:
            expected = reference
        else:
            np.testing.assert_allclose(reference, expected, rtol=1e-6, atol=1e-7)
        legacy = None
        for name in ("lstm_f32", "cuda_lstm", "cuda_lstm_prepared"):
            with self.subTest(kernel=name):
                actual = np.empty_like(expected)
                pointers = [value.ctypes.data if value is not None else None for value in (x, w, r, bias, h, c, actual)]
                self.assertEqual(getattr(self.library, name)(*pointers, *x.shape, r.shape[-1], direction), 1)
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                if name == "cuda_lstm": legacy = actual.copy()
                if name == "cuda_lstm_prepared":
                    np.testing.assert_array_equal(actual.view(np.uint32), legacy.view(np.uint32))

    def test_cuda_projection_boundary_correction_and_repeated_inputs(self):
        # 1 + half an ULP leaves a nonzero Kahan correction. The first
        # recurrent product supplies the second half; resetting correction at
        # that boundary changes a gate and the final waveform's bits.
        time, batch, width, hidden = 3, 2, 33, 65
        x = np.zeros((time, batch, width), np.float32)
        x[..., 0] = 1
        x[..., 1] = 2**-24
        w = np.zeros((2, 4*hidden, width), np.float32)
        w[:, 3*hidden:, :2] = 1
        r = np.zeros((2, 4*hidden, hidden), np.float32)
        r[:, 3*hidden:, 0] = 2**-24
        h = np.ones((2, batch, hidden), np.float32)
        c = np.zeros_like(h)
        self.compare(x, w, r, None, h, c, 2)
        # Same backing arrays, changed live values and learned projections.
        x[..., 0] = -.25
        w[:, 3*hidden:, 1] = -2
        r[:, 3*hidden:, 0] = .125
        c.fill(.5)
        self.compare(x, w, r, None, h, c, 2)

    def test_large_finite_cell_and_gate_tails(self):
        fixture = lstm_tail_fixture()
        x, _, _, bias, _, initial_c, _ = fixture
        expected = np.empty((len(x), 2, 1, 4), dtype=np.float32)
        # The three sigmoid gates have zero preactivation, hence exactly 1/2.
        # Independent scalar double libm avoids a tanh implementation using
        # exp(+2*x), which overflows for these otherwise finite states.
        for direction in range(2):
            cell = [float(value) for value in initial_c[direction, 0]]
            for step in range(len(x)):
                time = step if direction == 0 else len(x)-1-step
                for channel in range(4):
                    cell[channel] = .5 * cell[channel] + .5 * math.tanh(float(bias[direction, 12+channel]))
                    expected[time, direction, 0, channel] = .5 * math.tanh(cell[channel])
        self.compare(*fixture, expected=expected)

    def test_directions_variable_time_batch_and_optional_states(self):
        rng = np.random.default_rng(718)
        for direction in (0, 1, 2):
            for time, batch, width, hidden, optional in ((1, 1, 3, 1, False), (7, 2, 5, 9, True), (3, 1, 7, 257, True)):
                directions = 2 if direction == 2 else 1
                def random(shape): return rng.normal(0, .15, shape).astype(np.float32)
                x = random((time, batch, width))
                w, r = random((directions, 4 * hidden, width)), random((directions, 4 * hidden, hidden))
                bias = random((directions, 8 * hidden)) if optional else None
                h, c = (random((directions, batch, hidden)) for _ in range(2)) if optional else (None, None)
                with self.subTest(direction=direction, time=time, batch=batch, hidden=hidden, optional=optional):
                    self.compare(x, w, r, bias, h, c, direction)

    def test_real_kokoro_first_lstm_weights(self):
        source = ROOT / "build/tts-examples/assets/kokoro/onnx/model.onnx"
        if not source.exists():
            self.skipTest("Prepare Kokoro to qualify its learned LSTM weights")
        model = onnx.load(source)
        node = next(node for node in model.graph.node if node.op_type == "LSTM")
        names = set(node.input[1:4])
        weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer if tensor.name in names}
        w, r, bias = (weights[name] for name in node.input[1:4])
        del model
        rng = np.random.default_rng(812)
        x = rng.normal(0, .5, (11, 1, w.shape[-1])).astype(np.float32)
        self.compare(x, w, r, bias, None, None, 2)


if __name__ == "__main__":
    unittest.main()
