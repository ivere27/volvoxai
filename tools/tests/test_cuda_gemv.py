"""Production GEMV arithmetic/launch checks; physical qualification is separate."""
from pathlib import Path
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CUDA_INCLUDE = ROOT / "native/src/backends/cuda"


from tools.tests import function_source as function


class CudaGemvTests(unittest.TestCase):
    def test_gemm_probe_multirow_geometry_reference_and_legacy_tile(self):
        helper = ROOT / "tools/llm_baseline/probe_cuda_gemv.py"
        sys.path.insert(0, str(helper.parent))
        try:
            spec = importlib.util.spec_from_file_location("cuda_gemv_probe_test", helper)
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
        finally:
            sys.path.pop(0)
        cases = probe.geometries("gemm")
        self.assertEqual(len(cases), 92)
        self.assertEqual({g["rows"] for g in cases}, {1, 15, 16, 17, 31, 32, 33, 39, 63, 65, 128, 256})
        geometry = probe.tiled_geometry((ROOT /
            "native/src/backends/cuda/kernels/cuda_forward_linear_moe_kernels.inc").read_text())
        symbols = set()
        for g in cases:
            compact, original = probe.launch(g, "tiled", 8), probe.launch(g, "tiled", 16)
            self.assertEqual(compact["grid"], [(g["n"] + 15) // 16, (g["rows"] + 15) // 16, 1])
            self.assertEqual(compact["block"], [8, 16, 1])
            self.assertEqual(original["block"], [16, 16, 1])
            self.assertEqual(compact["static_shared_bytes"], 2176)
            self.assertEqual(original["grid"], compact["grid"])
            selected = probe.launch(g, "tiled", **geometry)
            wide = g["rows"] > 32 and g["n"] >= 1024
            tile_rows, tile_columns = (32, 64) if wide else (16, 16)
            self.assertEqual(selected["grid"], [(g["n"] + tile_columns - 1) // tile_columns,
                                               (g["rows"] + tile_rows - 1) // tile_rows, 1])
            self.assertEqual(selected["block"], [8, 16, 1])
            self.assertEqual(selected["static_shared_bytes"], 6336 if wide else 8704)
            symbol = probe.kernel(g, "tiled", geometry["split_entries"])
            symbols.add(symbol)
            self.assertEqual("_register" in symbol, wide)
        self.assertEqual(symbols, {"vx_cuda_matmul_f32", "vx_cuda_matmul_out_in_f32",
                                  "vx_cuda_matmul_register_f32", "vx_cuda_matmul_register_out_in_f32"})
        self.assertTrue(any(g["rows"] == 31 and g["n"] == 1537 for g in cases))
        self.assertTrue(any(g["rows"] >= 128 and g["n"] == 64 for g in cases))
        with self.assertRaises(ValueError):
            probe.tiled_geometry("void vx_cuda_matmul_register_f32( input_tile[32][17] weight_tile[16][65] input_tile[4][16][17]")
        old_compact = probe.tiled_geometry("input_tile[16][17] weight_tile[16][17]")
        self.assertEqual(old_compact["tiled_block_x"], 16)
        old_adaptive = probe.tiled_geometry(
            "input_tile[32][17] weight_tile[16][65] vx_cuda_matmul_compact_body")
        for g in cases:
            self.assertEqual(probe.launch(g, "tiled", **old_adaptive)["static_shared_bytes"], 6336)
            self.assertNotIn("_register", probe.kernel(g, "tiled"))
        for layout, weight in (("IN_OUT", [1., 2., 3., 4.]),
                               ("OUT_IN", [1., 3., 2., 4.])):
            g = dict(rows=2, k=2, n=2, layout=layout, bias=True)
            self.assertEqual(list(probe.reference(g, [2., 3., 5., 7.], weight, [.5, -.5])),
                             [11.5, 15.5, 26.5, 37.5])

    def test_probe_declares_and_binds_optional_out_in_gemv_arguments(self):
        import ctypes as C
        import hashlib
        from types import SimpleNamespace
        from unittest.mock import patch
        helper = ROOT / "tools/llm_baseline/probe_cuda_gemv.py"
        sys.path.insert(0, str(helper.parent))
        try:
            spec = importlib.util.spec_from_file_location("cuda_gemv_probe_abi_test", helper)
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
        finally:
            sys.path.pop(0)

        class BindingComplete(Exception):
            pass

        class PreparationDriver:
            def __init__(self):
                self.calls, self.images = [], []

            def call(self, name, *args):
                self.calls.append(name)
                if name == "DeviceGetName":
                    args[0].value = b"CPU argument preparation only"
                elif name in ("DeviceGet", "DriverGetVersion"):
                    C.cast(args[0], C.POINTER(C.c_int))[0] = 1
                elif name in ("CtxCreate", "StreamCreate", "ModuleLoadData", "ModuleGetFunction", "EventCreate"):
                    C.cast(args[0], C.POINTER(C.c_void_p))[0] = len(self.calls)
                elif name == "MemAlloc":
                    C.cast(args[0], C.POINTER(C.c_uint64))[0] = 0x10000 + len(self.calls) * 0x1000
                elif name == "MemcpyHtoD":
                    self.images.append(C.string_at(args[1], args[2]))

        with tempfile.TemporaryDirectory(prefix="volvox-gemv-probe-abi-") as temp:
            ptx, library = Path(temp) / "fixture.ptx", Path(temp) / "fixture.so"
            ptx.write_bytes(b"CPU preparation fixture only\n")
            library.write_bytes(b"CPU preparation fixture only\n")
            for mode in ("gemv", "tiled"):
                for layout in ("IN_OUT", "OUT_IN"):
                    for stored in (False, True):
                        g = dict(label="ABI-binding", rows=1, k=3, n=2,
                                 layout=layout, bias=False, seed=98743)
                        optional = mode == "gemv" and layout == "OUT_IN"
                        abi = ["ptr"] * 4 + ["u32"] * 3
                        if optional:
                            abi += ["ptr", "ptr", "u32"]
                        self.assertEqual(probe.kernel_argument_abi(g, mode), abi + ["argument_storage_ptr"])
                        driver = PreparationDriver()
                        real_prepare = probe.prepare_arguments

                        def capture(driver_, scalars, allocations, *, stored):
                            self.assertEqual(len(scalars), len(abi))
                            self.assertEqual([C.sizeof(v) for v in scalars],
                                             [8 if kind == "ptr" else 4 for kind in abi])
                            self.assertEqual([v.value for v in scalars[4:7]], [1, 3, 2])
                            self.assertEqual(scalars[2].value, 0)
                            if optional:
                                self.assertEqual([v.value for v in scalars[7:]], [0, 0, 0])
                            values, params = real_prepare(driver_, scalars, allocations, stored=stored)
                            self.assertEqual(len(params), len(abi) + 1)
                            if stored:
                                image = driver.images[-1]
                                self.assertEqual(len(image), 8 * len(abi))
                                if optional:
                                    self.assertEqual(image[56:], bytes(24))
                                self.assertTrue(all(v.value == 0 for v in values[:-1]))
                            else:
                                self.assertEqual(values[-1].value, 0)
                            raise BindingComplete()

                        args = SimpleNamespace(source_root=ROOT, ptx=ptx, library=library,
                            mode=mode, argument_storage=stored, warmups=1, samples=3)
                        with self.subTest(mode=mode, layout=layout, stored=stored), \
                                patch.object(probe, "Driver", return_value=driver), \
                                patch.object(probe, "prepare_arguments", side_effect=capture):
                            with self.assertRaises(BindingComplete):
                                probe.run(args, {"configs": [{"geometry": g,
                                    "input_sha256": {name: hashlib.sha256(values.tobytes()).hexdigest()
                                        for name, values in zip(("input", "weight", "bias"), probe.fixture(g))},
                                    "reference_f64_sha256": hashlib.sha256(
                                        probe.reference(g, *probe.fixture(g)).tobytes()).hexdigest(),
                                    "kernel": probe.kernel(g, mode)}]})
                        self.assertNotIn("LaunchKernel", driver.calls)
                        self.assertEqual(driver.calls[-1], "CtxDestroy")

    def test_compact_gemm_compile_admission_proves_static_shared_and_geometry(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = (ROOT / "native/src/runtime/public_api_native_gpu_domain.inc").read_text()
        start = source.index("typedef struct VxNativeGpuDomainLimits {")
        end = source.index("} VxNativeGpuDomainLimits;", start) + len("} VxNativeGpuDomainLimits;")
        fixture = '''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
typedef int VxBackendKind;
enum {VX_BACKEND_KIND_CUDA=0,VX_BACKEND_KIND_VULKAN=1};
'''+source[start:end]+'''
static int vx_native_gpu_one_d_launch_proven(uint64_t elements,VxBackendKind backend,
        const VxNativeGpuDomainLimits* limits) { return elements && backend==0 && limits; }
'''+function(source, "vx_native_gpu_dense_launch_proven")+'''
int main(void) {
    VxNativeGpuDomainLimits l={0};
    l.maximum_block[0]=8;l.maximum_block[1]=16;l.maximum_block[2]=1;
    l.maximum_threads_per_block=128;l.maximum_shared_memory_bytes=8704;
    l.maximum_grid[0]=2147483647u;l.maximum_grid[1]=65535;l.maximum_grid[2]=1;
    assert(vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    assert(vx_native_gpu_dense_launch_proven(128,1024,3072,0,0,&l));
    assert(vx_native_gpu_dense_launch_proven(65535u*16u,1,1,0,0,&l));
    assert(!vx_native_gpu_dense_launch_proven(65535u*16u+1u,1,1,0,0,&l));
    assert(vx_native_gpu_dense_launch_proven(65535u*32u,1,1024,0,0,&l));
    assert(!vx_native_gpu_dense_launch_proven(65535u*32u+1u,1,1024,0,0,&l));
    l.maximum_shared_memory_bytes=8703;
    assert(!vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    assert(!vx_native_gpu_dense_launch_proven(128,1024,3072,0,0,&l));
    l.maximum_shared_memory_bytes=8704;
    assert(vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    assert(vx_native_gpu_dense_launch_proven(256,576,64,0,0,&l));
    l.maximum_shared_memory_bytes=8703;
    assert(!vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    assert(vx_native_gpu_dense_launch_proven(31,576,1536,1,0,&l));
    l.maximum_shared_memory_bytes=8704;l.maximum_threads_per_block=127;
    assert(!vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    l.maximum_threads_per_block=128;l.maximum_block[0]=7;
    assert(!vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    l.maximum_block[0]=8;l.maximum_block[1]=15;
    assert(!vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    l.maximum_block[1]=16;l.maximum_grid[0]=95;
    assert(!vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    l.maximum_grid[0]=96;l.maximum_grid[1]=1;
    assert(!vx_native_gpu_dense_launch_proven(33,576,1536,0,0,&l));
    l.maximum_grid[1]=2;
    assert(vx_native_gpu_dense_launch_proven(31,576,1536,0,0,&l));
    assert(!vx_native_gpu_dense_launch_proven(UINT32_MAX,1,2,0,0,&l));
    assert(!vx_native_gpu_dense_launch_proven(0,1,1,0,0,&l));
    assert(!vx_native_gpu_dense_launch_proven(1,0,1,0,0,&l));
    assert(!vx_native_gpu_dense_launch_proven(1,1,1,0,0,NULL));
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-compact-gemm-domain-") as tmp:
            path, binary = Path(tmp) / "proof.c", Path(tmp) / "proof"
            path.write_text(fixture)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                str(path), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout+checked.stderr)

    def test_f64_layouts_bias_tails_guards_determinism_and_gemm(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        production = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_linear_moe_kernels.inc").read_text()
        production = production[:production.index("/* LoRA inference")]
        production += (ROOT / "native/src/backends/cuda/kernels/cuda_forward_gemv_pair_kernels.inc").read_text()
        production, count = re.subn(
            r'(static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE float vx_cuda_gemv_shuffle_down\(.*?\{).*?\n\}',
            r'\1\n    return host_shuffle(value, delta);\n}', production, flags=re.S)
        self.assertEqual(count, 1)
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-gemv-") as tmp:
            source, binary = Path(tmp) / "production.inc", Path(tmp) / "probe"
            source.write_text((ROOT / "native/src/backends/cuda/kernels/cuda_forward_argument_storage.inc").read_text() + production)
            build = subprocess.run([compiler, "-std=c++17", "-O3", "-ffp-contract=off",
                "-Wall", "-Wextra", "-Werror", f'-DTEST_CUDA_GEMV_SOURCE="{source}"',
                str(ROOT / "native/tests/test_cuda_gemv.cpp"), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(checked.stdout.count("PASS"), 354, checked.stdout)
            self.assertEqual(checked.stdout.count("alignment-rebind"), 4, checked.stdout)
            self.assertEqual(checked.stdout.count("dual OUT_IN"), 32, checked.stdout)
            self.assertEqual(checked.stdout.count("paired OUT_IN"), 32, checked.stdout)
            self.assertIn("compact OUT_IN rows=31 k=257 n=129", checked.stdout)
            self.assertIn("register OUT_IN rows=39 k=5 n=1027", checked.stdout)
            self.assertIn("register IN_OUT rows=65 k=17 n=1057", checked.stdout)
            for route in ("compact", "register"):
                for layout in ("IN_OUT", "OUT_IN"):
                    self.assertIn(route + " " + layout, checked.stdout)

    def test_actual_host_route_preserves_fallback_coherence_and_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        dense = (ROOT / "native/src/backends/cuda/host/cuda_graph_dense_moe_host.inc").read_text()
        extracted = function(memory, "size_product_u32") + function(memory, "ranges_overlap")
        extracted += dense[:dense.index("static int cuda_lora_workspace_reserve")].replace(
            function(dense, "cuda_graph_gemv_pair_out_in_f32", "int "), "")
        fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
typedef uintptr_t CUdeviceptr;
typedef unsigned CUfunction;
typedef struct { CUdeviceptr device; int host_dirty,device_dirty; const void* host; int is_weight; } CudaTensorSlot;
typedef struct { CudaTensorSlot* slot; CUdeviceptr device; } CudaTensorView;
enum { k_matmul=1,k_matmul_out_in,k_matmul_register,k_matmul_register_out_in,k_gemv,k_gemv_out_in,
    k_matmul_split,k_matmul_split_out_in,k_matmul_split_reduce };
typedef struct { CUdeviceptr matmul_workspace; struct { CUdeviceptr output; } matmul_tuning; } CudaContextState;
static CudaContextState capsule;
static CudaContextState* cuda_context_state_get(void) { return &capsule; }
static uint32_t mock_split_k_slices=4u;
static int fail_choice;
static unsigned choices;
static int cuda_matmul_choose_slices(CudaContextState* state,
    CUdeviceptr input,CUdeviceptr weight,CUdeviceptr bias,
    uint32_t rows,uint32_t input_width,uint32_t output_width,int out_in,uint32_t* selected) {
    (void)state;(void)input;(void)weight;(void)bias;(void)rows;
    (void)input_width;(void)output_width;(void)out_in;++choices;
    *selected=mock_split_k_slices;return !fail_choice;
}
#define CUDA_FORWARD_FUNCTION(k) k
static CudaTensorSlot slots[4];
#define graph_slots slots
static int cuda_domain_enforced,cuda_forward_scope_active,cuda_forward_context_ready,cuda_forward_context_failed;
static int graph_find_containing_slot(const void* p,size_t n) { (void)p;(void)n;return -1; }
static int graph_slot_is_resident_weight(const CudaTensorSlot* slot) { return slot&&slot->is_weight; }
static unsigned cuda_max_shared_memory_per_block=65536,grid_limit=1000000,block_limit=1024;
static unsigned launches,last_kernel,last_grid,last_bx,last_by,marks,syncs,views,refreshes;
static int fail_views,fail_launch,fail_sync;
static unsigned fail_kernel,grid_z_limit=65535,split_partials,split_reductions;
static CUdeviceptr partial_owner,output_device_override;
static unsigned next_constant=1;
static int graph_get_view(const void* p,size_t n,int invariant,CudaTensorView* view) {
    (void)p;(void)n;(void)invariant; ++refreshes;view->slot=&slots[0];return !fail_views;
}
static int graph_ensure_input_view(const void* p,size_t n,int invariant,CudaTensorView* view) {
    (void)n;(void)invariant; ++views;view->device=(uintptr_t)p;view->slot=&slots[0];return !fail_views;
}
static CudaTensorSlot* graph_ensure_input(const void* p,size_t n,int invariant) {
    (void)n;(void)invariant;unsigned i=next_constant;next_constant=3-next_constant;
    slots[i].device=(uintptr_t)p;return fail_views?NULL:&slots[i];
}
static int graph_output_view(void* p,size_t n,CudaTensorView* view) {
    (void)n;view->device=output_device_override?output_device_override:(uintptr_t)p;view->slot=&slots[3];return !fail_views;
}
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    return x&&y&&z&&x<=grid_limit&&z<=grid_z_limit&&bx<=block_limit&&bx*by*bz<=1024&&shared<=cuda_max_shared_memory_per_block;
}
static int cuda_launch(unsigned kernel,unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(y&&z&&bz&&!shared&&args);++launches;
    if(kernel==k_matmul_split||kernel==k_matmul_split_out_in) {
        assert(z==mock_split_k_slices&&bx==8&&by==16&&*(uint32_t*)args[3]>=17);
        assert(*(uint32_t*)args[6]==mock_split_k_slices);
        assert(*(CUdeviceptr*)args[2]==capsule.matmul_workspace);
        assert(*(CUdeviceptr*)args[0]==0x10000000&&*(CUdeviceptr*)args[1]==0x20000000);
        partial_owner=*(CUdeviceptr*)args[2];++split_partials;
    } else if(kernel==k_matmul_split_reduce) {
        assert(z==1&&bx==128&&by==1&&*(uint32_t*)args[5]>=17);
        assert(*(CUdeviceptr*)args[3]==partial_owner&&*(uint32_t*)args[8]<=1);
        assert(*(uint32_t*)args[9]==mock_split_k_slices);
        ++split_reductions;
    } else assert(*(uint32_t*)args[4]);
    last_kernel=kernel;last_grid=x;last_bx=bx;last_by=by;
    return !fail_launch&&kernel!=fail_kernel&&cuda_launch_dimensions_valid(x,y,z,bx,by,bz,shared);
}
static void graph_mark_device(CudaTensorSlot* slot) { assert(slot==&slots[3]);++marks; }
static int cuda_graph_sync_host(void* output,size_t n,int constant) { assert(output&&n&&!constant);++syncs;return !fail_sync; }
'''
        cases = r'''
int main(void) {
    static float x[64]={0},w[64*64]={0},b[64]={0},y[64*64]={0};
    static float tx[32*129],tw[129*65],tb[65],ty[32*65];
    assert(cuda_graph_matmul_layout_f32(x,w,b,y,1,33,19,0));
    assert(last_kernel==k_gemv&&last_grid==3&&last_bx==128&&last_by==1&&marks==1&&!syncs&&!refreshes);
    assert(cuda_graph_matmul_layout_f32(x,w,NULL,y,1,33,19,1));
    assert(last_kernel==k_gemv_out_in&&last_grid==5&&marks==2);
    assert(cuda_graph_matmul_f32(x,w,b,y,17,3,19));
    assert(last_kernel==k_matmul&&last_grid==2&&last_bx==8&&last_by==16);
    assert(cuda_graph_matmul_layout_f32(x,w,b,y,2,3,19,1)&&last_kernel==k_matmul_out_in);
    assert(cuda_graph_matmul_f32(x,w,b,y,31,3,1025)&&last_kernel==k_matmul);
    assert(cuda_graph_matmul_layout_f32(x,w,b,y,31,3,1025,1)&&last_kernel==k_matmul_out_in);
    assert(cuda_graph_matmul_f32(x,w,b,y,33,3,1025)&&last_kernel==k_matmul_register&&last_grid==17);
    assert(cuda_graph_matmul_layout_f32(x,w,b,y,33,3,1025,1)&&last_kernel==k_matmul_register_out_in);
    assert(cuda_graph_matmul_f32(x,w,b,y,256,3,65)&&last_kernel==k_matmul&&last_grid==5);
    assert(cuda_graph_matmul_layout_f32(tx,tw,tb,ty,31,129,65,1));
    assert(last_kernel==k_matmul_out_in&&last_grid==5&&last_bx==8&&last_by==16);
    cuda_max_shared_memory_per_block=8703;
    assert(!cuda_graph_matmul_layout_f32(tx,tw,tb,ty,31,129,65,1));
    cuda_max_shared_memory_per_block=8704;
    assert(cuda_graph_matmul_layout_f32(tx,tw,tb,ty,31,129,65,1)&&last_kernel==k_matmul_out_in);
    assert(cuda_graph_matmul_layout_f32(tx,tw,tb,tx,31,129,65,1)&&last_kernel==k_matmul_out_in);
    grid_limit=2;
    assert(!cuda_graph_matmul_layout_f32(tx,tw,tb,ty,31,129,65,1));
    grid_limit=1000000;
    block_limit=1024;cuda_max_shared_memory_per_block=65536;
    unsigned old_launches=launches,old_route_marks=marks,old_route_syncs=syncs;
    assert(cuda_graph_matmul_layout_f32(tx,tw,NULL,ty,31,129,65,1));
    assert(last_kernel==k_matmul_out_in&&launches==old_launches+1&&marks==old_route_marks+1&&syncs==old_route_syncs);
    assert(cuda_graph_matmul_layout_f32(tx,tw,tb,ty,31,129,65,1)&&last_kernel==k_matmul_out_in);
    assert(cuda_graph_matmul_layout_f32(tx,tw,NULL,ty,31,129,65,0)&&last_kernel==k_matmul);
    assert(cuda_graph_matmul_layout_f32(tx,tw,NULL,ty,1,129,65,1)&&last_kernel==k_gemv_out_in);
    assert(cuda_graph_matmul_layout_f32(tx,tw,NULL,tx,31,129,65,1)&&last_kernel==k_matmul_out_in);
    old_launches=launches;old_route_marks=marks;fail_launch=1;
    assert(!cuda_graph_matmul_layout_f32(tx,tw,NULL,ty,31,129,65,1));
    assert(launches==old_launches+1&&marks==old_route_marks&&syncs==old_route_syncs);
    fail_launch=0;
    assert(cuda_graph_matmul_layout_f32(tx,tw,NULL,ty,31,129,65,1)&&last_kernel==k_matmul_out_in);
    assert(marks==old_route_marks+1&&syncs==old_route_syncs);
    cuda_max_shared_memory_per_block=8704;
    assert(cuda_graph_matmul_f32(x,w,b,y,31,3,1025)&&last_kernel==k_matmul);
    assert(cuda_graph_matmul_f32(x,w,b,y,33,3,1025));
    cuda_max_shared_memory_per_block=8703;
    assert(!cuda_graph_matmul_f32(x,w,b,y,31,3,1025));
    assert(cuda_graph_matmul_f32(x,w,b,y,33,3,1025)&&last_kernel==k_matmul_register);
    cuda_max_shared_memory_per_block=6336;
    assert(cuda_graph_matmul_f32(x,w,b,y,33,3,1025)&&last_kernel==k_matmul_register);
    assert(!cuda_graph_matmul_f32(x,w,b,y,31,3,1025));
    cuda_max_shared_memory_per_block=6335;
    assert(!cuda_graph_matmul_f32(x,w,b,y,33,3,1025));
    cuda_max_shared_memory_per_block=65536;
    assert(cuda_graph_matmul_f32(x,w,b,x,1,33,19)&&last_kernel==k_matmul);
    assert(cuda_graph_matmul_f32(x,w,b,w+1,1,33,19)&&last_kernel==k_matmul);
    assert(cuda_graph_matmul_f32(x,w,b,b+1,1,33,19)&&last_kernel==k_matmul);
    cuda_max_shared_memory_per_block=575;
    assert(!cuda_graph_matmul_f32(x,w,b,y,1,33,19));
    assert(cuda_graph_matmul_layout_f32(x,w,b,y,1,33,19,1)&&last_kernel==k_gemv_out_in);
    cuda_max_shared_memory_per_block=65536;grid_limit=2;
    assert(cuda_graph_matmul_f32(x,w,b,y,1,33,19)&&last_kernel==k_matmul);
    grid_limit=1000000;block_limit=64;
    assert(cuda_graph_matmul_f32(x,w,b,y,1,33,19)&&last_kernel==k_matmul);
    block_limit=1024;
    assert(cuda_matmul(x,w,b,y,1,33,19)&&last_kernel==k_gemv&&syncs==1&&refreshes==1);
    assert(slots[0].host_dirty==1&&!slots[0].device_dirty);
    unsigned before=launches,old_views=views;
    assert(!cuda_graph_matmul_f32(NULL,w,b,y,1,33,19));
    assert(!cuda_graph_matmul_f32(x,w,b,y,0,33,19));
    assert(!cuda_graph_matmul_f32(x,w,b,y,1,-1,19));
    assert(!cuda_graph_matmul_layout_f32(x,w,b,y,1,33,19,2));
    assert(!cuda_graph_matmul_f32(x,w,b,y,1,2147483647,2147483647));
    assert(launches==before&&views==old_views);
    fail_views=1;assert(!cuda_graph_matmul_f32(x,w,b,y,1,33,19)&&launches==before);fail_views=0;
    unsigned old_marks=marks;fail_launch=1;
    assert(!cuda_graph_matmul_f32(x,w,b,y,1,33,19)&&marks==old_marks);fail_launch=0;
    fail_sync=1;assert(!cuda_matmul(x,w,b,y,1,33,19)&&marks==old_marks+1);
    fail_sync=0;
    const float* sx=(const float*)(uintptr_t)0x10000000;
    const float* sw=(const float*)(uintptr_t)0x20000000;
    const float* sb=(const float*)(uintptr_t)0x30000000;
    float* sy=(float*)(uintptr_t)0x40000000;
    capsule.matmul_workspace=0x50000000;
    for(mock_split_k_slices=2;mock_split_k_slices<=16;mock_split_k_slices*=2)
      for(int layout=0;layout<2;layout++) for(int biased=0;biased<2;biased++) {
        unsigned count=launches,marked=marks,part=split_partials,red=split_reductions;
        assert(cuda_graph_matmul_layout_f32(sx,sw,biased?sb:NULL,sy,17,1025,1025,layout)==1);
        assert(launches==count+2&&marks==marked+1&&split_partials==part+1&&split_reductions==red+1);
        assert(last_kernel==k_matmul_split_reduce);
    }
    mock_split_k_slices=4;
    unsigned count=launches,marked=marks;
    fail_choice=1;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,1024,2048,0)==0);
    assert(launches==count&&marks==marked);
    fail_choice=0;
    fail_kernel=k_matmul_split;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,1024,2048,0)==0);
    assert(launches==count+1&&marks==marked&&last_kernel==k_matmul_split);
    count=launches;fail_kernel=k_matmul_split_reduce;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,1024,2048,1)==0);
    assert(launches==count+2&&marks==marked&&last_kernel==k_matmul_split_reduce);
    fail_kernel=0;
    unsigned part=split_partials,red=split_reductions;
    capsule.matmul_workspace=0;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,1024,2048,1)&&last_kernel==k_matmul_out_in);
    capsule.matmul_workspace=0x50000000;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,16,1024,2048,1)&&last_kernel==k_matmul_out_in);
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,33,1024,2048,1)&&last_kernel==k_matmul_register_out_in);
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,24,1023,2048,1)&&last_kernel==k_matmul_out_in);
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,24,1025,1023,1)&&last_kernel==k_matmul_out_in);
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,32,1024,4097,1)&&last_kernel==k_matmul_out_in);
    cuda_max_shared_memory_per_block=17151;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,32,1024,4096,1)&&last_kernel==k_matmul_out_in);
    cuda_max_shared_memory_per_block=17152;grid_z_limit=3;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,32,1024,4096,1)&&last_kernel==k_matmul_out_in);
    grid_z_limit=4;grid_limit=512;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,32,1024,4096,1)&&last_kernel==k_matmul_out_in);
    grid_limit=1000000;block_limit=64;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,32,1024,4096,1)&&last_kernel==k_matmul_out_in);
    block_limit=1024;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,(float*)sx,31,1024,2048,1)&&last_kernel==k_matmul_out_in);
    output_device_override=(CUdeviceptr)sw+4;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,1024,2048,1)&&last_kernel==k_matmul_out_in);
    output_device_override=0;
    capsule.matmul_workspace=UINTPTR_MAX-8;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,1024,2048,1)&&last_kernel==k_matmul_out_in);
    capsule.matmul_workspace=(CUdeviceptr)sx;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,1024,2048,1)&&last_kernel==k_matmul_out_in);
    assert(part==split_partials&&red==split_reductions);
    capsule.matmul_workspace=0x50000000;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,32,1024,4096,1)&&last_kernel==k_matmul_split_reduce);
    assert(vx_cuda_matmul_split_workspace_bytes(32,1024,4096,4)==VX_CUDA_MATMUL_WORKSPACE_BYTES);
    assert(vx_cuda_matmul_split_workspace_bytes(17,1024,7710,4)==2097120u);
    assert(!vx_cuda_matmul_split_workspace_bytes(17,1024,7711,4));
    assert(!vx_cuda_matmul_split_workspace_bytes(UINT32_MAX,UINT32_MAX,UINT32_MAX,4));
    assert(!vx_cuda_matmul_split_workspace_bytes(32,UINT32_MAX,UINT32_MAX,4));
    assert(!vx_cuda_matmul_split_workspace_bytes(0,1024,1024,4));
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,24,3072,1024,1)&&last_kernel==k_matmul_split_reduce);
    mock_split_k_slices=16;grid_z_limit=65535;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,32,3072,1024,1)&&last_kernel==k_matmul_split_reduce);
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,32,3072,1025,1)&&last_kernel==k_matmul_out_in);
    mock_split_k_slices=0;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,3072,1024,1)&&last_kernel==k_matmul_out_in);
    mock_split_k_slices=4;capsule.matmul_tuning.output=(CUdeviceptr)sw;
    unsigned choices_before=choices;
    assert(cuda_graph_matmul_layout_f32(sx,sw,NULL,sy,31,3072,1024,1)&&last_kernel==k_matmul_out_in);
    assert(choices==choices_before);
    capsule.matmul_tuning.output=0;
    const CUdeviceptr views_to_overlap[4] = {(CUdeviceptr)sx,(CUdeviceptr)sw,(CUdeviceptr)sb,(CUdeviceptr)sy};
    for(unsigned i=0;i<4;i++) {
        /* Overlap only the reserved tail, beyond this shape's logical output. */
        capsule.matmul_tuning.output=views_to_overlap[i]-700000u;
        choices_before=choices;
        assert(cuda_graph_matmul_layout_f32(sx,sw,sb,sy,31,3072,1024,1)&&last_kernel==k_matmul_out_in);
        assert(choices==choices_before);
        capsule.matmul_tuning.output=0;
        capsule.matmul_workspace=views_to_overlap[i]-1500000u;
        assert(cuda_graph_matmul_layout_f32(sx,sw,sb,sy,31,3072,1024,1)&&last_kernel==k_matmul_out_in);
        assert(choices==choices_before);
        capsule.matmul_workspace=0x50000000;
    }

    puts("PASS production GEMV/split host dispatch: boundaries, both launches, terminal failures and T fallbacks");return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-gemv-route-") as tmp:
            source, binary = Path(tmp) / "route.c", Path(tmp) / "route"
            plan = (ROOT / "native/src/backends/cuda/dense_launch_plan.h").read_text()
            source.write_text(plan + fixture + extracted + cases)
            build = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(CUDA_INCLUDE), str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_row_tactics_bind_current_owners_before_partial_writes(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        dense = (ROOT / "native/src/backends/cuda/host/cuda_graph_dense_moe_host.inc").read_text()
        body = function(memory, "size_product_u32") + function(memory, "ranges_overlap")
        body += dense[:dense.index("static int cuda_lora_workspace_reserve")]
        fixture = r"""
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
typedef uint64_t CUdeviceptr;
typedef struct { const void* host; size_t bytes; CUdeviceptr device; int is_weight,host_dirty,device_dirty; } CudaTensorSlot;
typedef struct { CudaTensorSlot* slot; CUdeviceptr device; } CudaTensorView;
static CudaTensorSlot graph_slots[5];
static int cuda_domain_enforced=1,cuda_forward_scope_active=1,cuda_forward_context_ready=1,cuda_forward_context_failed;
static int available=1,ensures,launches,marks,fail_ensure,fail_launch,grid_ok=1,paired=1,residual_first;
static unsigned int cuda_max_shared_memory_per_block=65536;
typedef unsigned CUfunction;
enum { k_matmul=1,k_matmul_out_in,k_matmul_register,k_matmul_register_out_in,k_gemv,k_gemv_out_in,
    k_matmul_split,k_matmul_split_out_in,k_matmul_split_reduce,k_gemv_pair_out_in };
#define CUDA_FORWARD_FUNCTION(k) (available ? k : 0)
typedef struct { CUdeviceptr matmul_workspace; struct { CUdeviceptr output; } matmul_tuning; } CudaContextState;
static CudaContextState* cuda_context_state_get(void) { assert(0);return NULL; }
static int cuda_matmul_choose_slices(CudaContextState* state,CUdeviceptr input,CUdeviceptr weight,CUdeviceptr bias,
    uint32_t rows,uint32_t input_width,uint32_t output_width,int out_in,uint32_t* selected) {
    (void)state;(void)input;(void)weight;(void)bias;(void)rows;(void)input_width;(void)output_width;(void)out_in;(void)selected;
    assert(0);return 0;
}
static int graph_get_view(const void* p,size_t n,int constant,CudaTensorView* view) {
    (void)p;(void)n;(void)constant;(void)view;assert(0);return 0;
}
static CudaTensorSlot* graph_ensure_input(const void* p,size_t n,int constant) {
    (void)p;(void)n;(void)constant;assert(0);return NULL;
}
static int graph_output_view(void* p,size_t n,CudaTensorView* view) {
    (void)p;(void)n;(void)view;assert(0);return 0;
}
static int cuda_graph_sync_host(void* p,size_t n,int constant) {
    (void)p;(void)n;(void)constant;assert(0);return 0;
}
static int graph_find_containing_slot(const void* p,size_t n) {
    for(int i=0;i<5;++i) {
        uintptr_t start=(uintptr_t)graph_slots[i].host,value=(uintptr_t)p;
        if(value>=start&&value-start<=graph_slots[i].bytes&&n<=graph_slots[i].bytes-(value-start)) return i;
    }
    return -1;
}
static int graph_slot_is_resident_weight(const CudaTensorSlot* slot) { return slot->is_weight; }
static int graph_ensure_input_view(const void* p,size_t n,int weight,CudaTensorView* view) {
    ++ensures;int index=graph_find_containing_slot(p,n);
    assert(index==ensures-1&&weight==graph_slots[index].is_weight);
    assert(p==graph_slots[index].host&&n==graph_slots[index].bytes);
    view->slot=&graph_slots[index];view->device=graph_slots[index].device+(uintptr_t)p-(uintptr_t)graph_slots[index].host;
    return ensures!=fail_ensure;
}
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    assert(x&&y==1&&z==1&&bx==128&&by==1&&bz==1&&!shared);return grid_ok;
}
static int cuda_launch(unsigned fn,unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(fn&&x==5&&y==1&&z==1&&bx==128&&by==1&&bz==1&&!shared);
    assert(ensures==5);
    assert(*(uint64_t*)args[0]==graph_slots[0].device+33*4);
    assert(*(uint64_t*)args[1]==graph_slots[1].device);
    if (paired) {
        assert(fn==k_gemv_pair_out_in);
        assert(*(uint64_t*)args[2]==graph_slots[2].device);
        assert(*(uint64_t*)args[3]==graph_slots[3].device+17*4&&*(uint64_t*)args[4]==graph_slots[4].device+17*4);
        assert(*(uint32_t*)args[5]==1&&*(uint32_t*)args[6]==33&&*(uint32_t*)args[7]==17);
    } else {
        assert(fn==k_gemv_out_in&&!*(uint64_t*)args[2]);
        assert(*(uint64_t*)args[3]==graph_slots[3].device+17*4);
        assert(*(uint32_t*)args[4]==1&&*(uint32_t*)args[5]==33&&*(uint32_t*)args[6]==17);
        assert(*(uint64_t*)args[7]==graph_slots[2].device+17*4&&*(uint64_t*)args[8]==graph_slots[4].device+17*4);
        assert(*(uint32_t*)args[9]==(uint32_t)residual_first);
    }
    ++launches;return !fail_launch;
}
static void graph_mark_device(CudaTensorSlot* slot) { assert(slot==&graph_slots[3]||slot==&graph_slots[4]);++marks; }
static float x[99],w0[561],w1[561],y0[51],y1[51];
static void reset(void) {
    graph_slots[0]=(CudaTensorSlot){x,sizeof(x),0x10000,0,0,0};
    graph_slots[1]=(CudaTensorSlot){w0,sizeof(w0),0x20000,1,0,0};
    graph_slots[2]=(CudaTensorSlot){w1,paired?sizeof(w1):sizeof(y0),0x30000,paired,0,0};
    graph_slots[3]=(CudaTensorSlot){y0,sizeof(y0),0x40000,0,0,0};
    graph_slots[4]=(CudaTensorSlot){y1,sizeof(y1),0x50000,0,0,0};
    ensures=launches=marks=fail_ensure=fail_launch=cuda_forward_context_failed=0;
    available=grid_ok=cuda_domain_enforced=cuda_forward_scope_active=cuda_forward_context_ready=1;
}
"""
        cases = r"""
static int run(const float* input,const float* weight0,const float* second,float* output0,float* output1,
    int sequence,int row,int d_in,int d_out) {
    return paired ? cuda_graph_gemv_pair_out_in_f32(input,weight0,second,output0,output1,sequence,row,d_in,d_out)
                  : cuda_graph_gemv_add_dual_f32(input,weight0,second,output0,output1,sequence,row,d_in,d_out,residual_first);
}
int main(void) {
  for(paired=0;paired<2;paired++) for(residual_first=0;residual_first<2;residual_first++) {
    reset();assert(run(x,w0,w1,y0,y1,3,1,33,17)==1&&ensures==5&&launches==1&&marks==2);
    for(int index=0;index<5;++index) {
        reset();graph_slots[index].bytes--;assert(!run(x,w0,w1,y0,y1,3,1,33,17)&&!ensures&&!launches&&!marks);
        reset();graph_slots[index].device=UINT64_MAX-4;
        assert(!run(x,w0,w1,y0,y1,3,1,33,17)&&!ensures&&!launches&&!marks);
        reset();graph_slots[index].device=0;assert(!run(x,w0,w1,y0,y1,3,1,33,17)&&!ensures);
        reset();graph_slots[index].is_weight=!graph_slots[index].is_weight;assert(!run(x,w0,w1,y0,y1,3,1,33,17)&&!ensures);
        reset();fail_ensure=index+1;assert(run(x,w0,w1,y0,y1,3,1,33,17)==-1&&ensures==index+1&&!launches&&!marks);
    }
    reset();assert(!run(x,w0,w1,x+1,y1,3,1,33,17)&&!ensures);
    reset();graph_slots[4].device=graph_slots[3].device+4;assert(!run(x,w0,w1,y0,y1,3,1,33,17)&&!ensures);
    if(paired) { reset();available=0;assert(!run(x,w0,w1,y0,y1,3,1,33,17)&&!ensures); }
    reset();grid_ok=0;assert(!run(x,w0,w1,y0,y1,3,1,33,17)&&!ensures);
    reset();cuda_forward_context_failed=1;assert(!run(x,w0,w1,y0,y1,3,1,33,17)&&!ensures);
    reset();assert(!run(x,w0,w1,y0,y1,3,3,33,17)&&!ensures);
    reset();assert(!run(x,w0,w1,y0,y1,3,1,INT32_MAX,INT32_MAX)&&!ensures);
    reset();fail_launch=1;assert(run(x,w0,w1,y0,y1,3,1,33,17)==-1&&ensures==5&&launches==1&&!marks);
  }
    puts("PASS both row tactics: full owner reads, aliases, overflow and terminal failures");return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="volvox-projection-pair-host-") as tmp:
            source, binary = Path(tmp) / "proof.c", Path(tmp) / "proof"
            source.write_text((ROOT / "native/src/backends/cuda/dense_launch_plan.h").read_text() + fixture + body + cases)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(CUDA_INCLUDE), str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout+checked.stderr)

    def test_split_capacity_and_default_wave_policy(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        cases = r"""
#include <assert.h>
int main(void) {
    const uint32_t slices[4] = {2, 4, 8, 16};
    const uint32_t max_n17[4] = {15420, 7710, 3855, 1927};
    const uint32_t max_n32[4] = {8192, 4096, 2048, 1024};
    for (unsigned i = 0; i < 4; ++i) {
        uint32_t p = slices[i];
        assert(vx_cuda_matmul_split_slices_supported(p));
        assert(vx_cuda_matmul_split_workspace_bytes(17, 4097, max_n17[i], p));
        assert(!vx_cuda_matmul_split_workspace_bytes(17, 4097, max_n17[i]+1, p));
        assert(vx_cuda_matmul_split_workspace_bytes(32, 3072, max_n32[i], p) == VX_CUDA_MATMUL_WORKSPACE_BYTES);
        assert(!vx_cuda_matmul_split_workspace_bytes(32, 3072, max_n32[i]+1, p));
        assert(!vx_cuda_matmul_split_workspace_bytes(16, 1024, 1024, p));
        assert(!vx_cuda_matmul_split_workspace_bytes(33, 1024, 1024, p));
        assert(!vx_cuda_matmul_split_workspace_bytes(17, 1023, 1024, p));
        assert(!vx_cuda_matmul_split_workspace_bytes(17, 1024, 1023, p));
        assert(!vx_cuda_matmul_split_workspace_bytes(UINT32_MAX, UINT32_MAX, UINT32_MAX, p));
        assert(!vx_cuda_matmul_split_workspace_bytes(32, UINT32_MAX, UINT32_MAX, p));
        assert(vx_cuda_matmul_split_workspace_bytes(17, UINT32_MAX, 1024, p));
    }
    const uint32_t invalid[] = {0, 1, 3, 5, 32, UINT32_MAX};
    for (unsigned i = 0; i < sizeof(invalid)/sizeof(invalid[0]); ++i) {
        assert(!vx_cuda_matmul_split_slices_supported(invalid[i]));
        assert(!vx_cuda_matmul_split_workspace_bytes(17, 1024, 1024, invalid[i]));
    }
    assert(vx_cuda_matmul_split_default_slices(31, 1024, 1024, 82, 1) == 8);
    assert(vx_cuda_matmul_split_default_slices(31, 1024, 1024, 82, 2) == 8);
    assert(vx_cuda_matmul_split_default_slices(31, 2047, 1024, 82, 2) == 8);
    assert(vx_cuda_matmul_split_default_slices(31, 2048, 1024, 82, 2) == 16);
    assert(vx_cuda_matmul_split_default_slices(31, 3072, 1024, 82, 2) == 16);
    assert(vx_cuda_matmul_split_default_slices(32, 3072, 2048, 82, 4) == 8);
    assert(vx_cuda_matmul_split_default_slices(31, 1024, 1024, 1, 1) == 2);
    assert(vx_cuda_matmul_split_default_slices(31, 3072, 1024, UINT32_MAX, UINT32_MAX) == 16);
    assert(!vx_cuda_matmul_split_default_slices(31, 3072, 1024, 0, 2));
    assert(!vx_cuda_matmul_split_default_slices(31, 3072, 1024, 82, 0));
    assert(!vx_cuda_matmul_split_default_slices(0, 3072, 1024, 82, 2));
    assert(!vx_cuda_matmul_split_default_slices(17, 1023, 1024, 82, 2));
    assert(!vx_cuda_matmul_split_default_slices(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX));
    assert(!vx_cuda_packed_linear_row_cache_bytes(0));
    assert(vx_cuda_packed_linear_row_cache_bytes(1) == 128);
    assert(vx_cuda_packed_linear_row_cache_bytes(128) == 128);
    assert(vx_cuda_packed_linear_row_cache_bytes(129) == 256);
    assert(vx_cuda_packed_linear_row_cache_bytes(16384) == 16384);
    assert(!vx_cuda_packed_linear_row_cache_bytes(16385));
    assert(!vx_cuda_packed_linear_row_cache_bytes(UINT32_MAX));
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="volvox-split-plan-") as tmp:
            source, binary = Path(tmp) / "plan.c", Path(tmp) / "plan"
            source.write_text((ROOT / "native/src/backends/cuda/dense_launch_plan.h").read_text() + cases)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(CUDA_INCLUDE), str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout+checked.stderr)

    def test_packed_adaptive_tile_grid_and_device_coverage(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        cases = r"""
#include "dense_launch_plan.h"
#include <assert.h>
int main(void) {
    /* Direct launches without device metadata preserve the existing widths. */
    assert(vx_cuda_packed_linear_tile_columns(16, 4032, 8, 0) == 32);
    assert(vx_cuda_packed_linear_tile_columns(16, 4033, 8, 0) == 64);
    assert(vx_cuda_packed_linear_tile_columns(16, 16256, 8, 0) == 64);
    assert(vx_cuda_packed_linear_tile_columns(16, 16257, 8, 0) == 128);
    assert(vx_cuda_packed_linear_tile_columns(16, UINT32_MAX, 16, 0) == 32);
    assert(vx_cuda_packed_linear_tile_columns(17, 7808, 16, 0) == 32);
    assert(vx_cuda_packed_linear_tile_columns(17, 7809, 16, 0) == 64);
    assert(vx_cuda_packed_linear_tile_columns(31, 2048, 8, 0) == 64);
    /* Narrow the tile before abandoning matrix arithmetic when the wider
     * route would leave SMs idle. Device metadata remains fixed at replay. */
    assert(vx_cuda_packed_linear_tile_columns(31, 2048, 8, 82) == 32);
    assert(vx_cuda_packed_linear_matrix_tiles(31, 2048, 8, 82, 75) == 128);
    assert(vx_cuda_packed_linear_tile_columns(16, 5184, 8, 82) == 32);
    assert(vx_cuda_packed_linear_tile_columns(16, 5185, 8, 82) == 64);
    assert(vx_cuda_packed_linear_tile_columns(512, 512, 8, 128) == 128);
    assert(vx_cuda_packed_linear_tile_columns(512, 512, 8, 129) == 64);
    assert(vx_cuda_packed_linear_tile_columns(512, 512, 8, 256) == 64);
    assert(vx_cuda_packed_linear_tile_columns(512, 512, 8, 257) == 32);
    assert(vx_cuda_packed_linear_tile_columns(32, 7809, 16, 246) == 64);
    assert(vx_cuda_packed_linear_tile_columns(32, 7809, 16, 247) == 32);
    assert(!vx_cuda_packed_linear_matrix_tiles(15, 4096, 8, 82, 75));
    assert(!vx_cuda_packed_linear_matrix_tiles(16, 512, 8, 82, 75));
    assert(!vx_cuda_packed_linear_matrix_tiles(16, 2592, 8, 82, 75));
    assert(vx_cuda_packed_linear_matrix_tiles(16, 2624, 8, 82, 75) == 82);
    assert(!vx_cuda_packed_linear_matrix_tiles(32, 320, 16, 82, 75));
    assert(vx_cuda_packed_linear_matrix_tiles(32, 1312, 16, 82, 75) == 82);
    assert(vx_cuda_packed_linear_matrix_tiles(512, 512, 8, 82, 75) == 128);
    assert(vx_cuda_packed_linear_matrix_tiles(1536, 256, 8, 82, 75) == 192);
    assert(vx_cuda_packed_linear_matrix_tiles(512, 512, 16, 82, 75) == 256);
    assert(vx_cuda_packed_linear_matrix_tiles(1536, 254, 16, 82, 75) == 384);
    assert(!vx_cuda_packed_linear_matrix_tiles(32, 1, 16, 82, 75));
    assert(!vx_cuda_packed_linear_matrix_tiles(32, 512, 8, 0, 75));
    assert(!vx_cuda_packed_linear_matrix_tiles(32, 512, 8, 82, 74));
    assert(!vx_cuda_packed_linear_matrix_tiles(32, 512, 4, 82, 75));
    assert(!vx_cuda_packed_linear_tile_columns(0, 512, 8, 82));
    assert(!vx_cuda_packed_linear_tile_columns(32, 0, 8, 82));
    assert(!vx_cuda_packed_linear_tile_columns(32, 512, 0, 82));
    /* Maximal dimensions cannot wrap the selector into a smaller tile, nor
     * can a multiplication overflow produce an apparently legal launch. */
    assert(vx_cuda_packed_linear_tile_columns(UINT64_MAX, UINT32_MAX, 8, UINT32_MAX) == 128);
    assert(vx_cuda_packed_linear_tile_columns(UINT64_MAX, UINT32_MAX, 16, UINT32_MAX) == 64);
    assert(!vx_cuda_packed_linear_matrix_tiles(UINT64_MAX, UINT32_MAX, 8, 82, 75));
    assert(!vx_cuda_packed_linear_matrix_tiles(UINT64_MAX, UINT32_MAX, 16, 82, 75));
    assert(vx_cuda_packed_linear_matrix_tiles(UINT64_MAX, 1, 8, 82, 75) ==
        UINT64_MAX / 16u + 1u);
    const uint64_t rows[] = {16, 17, 31, 32, 33, 511, 512, 1536};
    const uint32_t widths[] = {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 254, 512, 2048, 32000};
    const uint32_t capacities[] = {1, 64, 82, 128, 129, 246, 247, 512};
    for (unsigned r = 0; r < sizeof(rows)/sizeof(rows[0]); ++r)
    for (unsigned n = 0; n < sizeof(widths)/sizeof(widths[0]); ++n)
    for (unsigned c = 0; c < sizeof(capacities)/sizeof(capacities[0]); ++c)
    for (unsigned bits = 8; bits <= 16; bits += 8) {
        uint64_t row_tiles = (rows[r] + 15u)/16u;
        uint32_t tile = vx_cuda_packed_linear_tile_columns(rows[r], widths[n], bits, capacities[c]);
        uint64_t columns = ((uint64_t)widths[n] + tile - 1u)/tile;
        uint64_t logical_tiles = row_tiles * columns;
        uint64_t launched = vx_cuda_packed_linear_matrix_tiles(rows[r], widths[n], bits, capacities[c], 75);
        assert(tile == 32 || tile == 64 || (bits == 8 && tile == 128));
        assert(launched == (logical_tiles >= capacities[c] ? logical_tiles : 0));
        /* Every submitted CTA reaches a live row and column. No leftover
         * ceil(N/32) grid can silently launch redundant early-return blocks. */
        if (launched) {
            assert((launched - 1u)/columns*16u < rows[r]);
            assert((launched - 1u)%columns*tile < widths[n]);
            assert(launched/columns*16u >= rows[r]);
        }
        if (!launched) assert(row_tiles*((widths[n]+31u)/32u) < capacities[c]);
    }
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="volvox-packed-plan-") as tmp:
            source, binary = Path(tmp) / "plan.c", Path(tmp) / "plan"
            source.write_text(cases)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(CUDA_INCLUDE), str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_split_f64_bias_tails_specials_and_stored_records(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        kernels = ROOT / "native/src/backends/cuda/kernels"
        compact = (kernels / "cuda_forward_linear_moe_kernels.inc").read_text()
        compact = compact[:compact.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_f32')]
        production = (kernels / "cuda_forward_argument_storage.inc").read_text() + compact
        production += (kernels / "cuda_forward_gemm_splitk_kernels.inc").read_text()
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-split-") as tmp:
            source, binary = Path(tmp) / "production.inc", Path(tmp) / "probe"
            source.write_text(production)
            built = subprocess.run([compiler, "-std=c++17", "-O2", "-ffp-contract=off",
                f'-DTEST_CUDA_SPLIT_SOURCE="{source}"',
                str(ROOT / "native/tests/test_cuda_matmul_split.cpp"), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            if checked.returncode == 77:
                self.skipTest("cooperative CUDA host emulator requires GCC x86/SSE on Unix")
            self.assertEqual(checked.returncode, 0, checked.stdout+checked.stderr)
            self.assertIn("SUMMARY cases=168 pass=true payload_differences=0", checked.stdout)

    def test_optional_split_workspace_lifetime_and_failure_paths(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        path = ROOT / "native/src/backends/cuda/host/cuda_backend_lifecycle_host.inc"
        lifecycle = path.read_text()
        body = function(lifecycle, "cuda_matmul_workspace_prepare")
        body += function(lifecycle, "cuda_matmul_workspace_release")
        init = function(lifecycle, "cuda_init", "int ")
        self.assertEqual(init.count("cuda_matmul_workspace_prepare(state)"), 1)
        self.assertLess(init.index("if (cuda_ready) return 0;"), init.index("cuda_matmul_workspace_prepare(state)"))
        cleanup = function(lifecycle, "cuda_cleanup", "void ")
        self.assertLess(cleanup.index("cuda_matmul_workspace_release(state, replay_handles_cleared)"),
                        cleanup.index("cuda_device_state.reference_count--"))
        prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
typedef uintptr_t CUdeviceptr;
typedef int CUresult;
typedef struct { CUdeviceptr matmul_workspace; } CudaContextState;
typedef struct { int active; } CudaContextGuard;
enum { CUDA_SUCCESS=0,CUDA_ERROR_OUT_OF_MEMORY=2 };
static unsigned allocations,frees,waits,enters,leaves,errors;
static int alloc_status,free_status,wait_status,enter_ok=1,leave_ok=1,limits_ok=1;
static int cuda_stream=7,cuda_graph_allocation_failed=13,cuda_forward_context_failed=17;
static CUdeviceptr next_device=0x10000000;
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,
        unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    assert(x==1&&y==1&&bz==1);
    assert((z==4&&bx==8&&by==16&&shared==17152)||(z==1&&bx==128&&by==1&&!shared));
    return limits_ok;
}
static CUresult cuda_memory_allocate(CUdeviceptr* p,size_t n) {
    assert(n==VX_CUDA_CONTEXT_WORKSPACE_BYTES);++allocations;
    if(!alloc_status) { *p=next_device;next_device+=n; } return alloc_status;
}
static CUresult cuda_memory_free(CUdeviceptr p) { assert(p);++frees;return free_status; }
static int cuda_ok(CUresult s,const char* op) { assert(op);if(s) ++errors;return !s; }
static int cuda_context_enter(CudaContextGuard* guard) { ++enters;guard->active=enter_ok;return enter_ok; }
static int cuda_context_leave(CudaContextGuard* guard) { assert(guard->active);++leaves;return leave_ok; }
static CUresult cuda_trace_wait_stream(int stream) { assert(stream==7);++waits;return wait_status; }
'''
        cases = r'''
int main(void) {
    CudaContextState a={0},b={0};
    limits_ok=0;assert(cuda_matmul_workspace_prepare(&a)&&!a.matmul_workspace&&!allocations);
    limits_ok=1;alloc_status=CUDA_ERROR_OUT_OF_MEMORY;
    assert(cuda_matmul_workspace_prepare(&a)&&!a.matmul_workspace&&allocations==1&&!errors);
    assert(cuda_graph_allocation_failed==13&&cuda_forward_context_failed==17);
    alloc_status=700;assert(!cuda_matmul_workspace_prepare(&a)&&!a.matmul_workspace&&errors==1);
    alloc_status=0;assert(cuda_matmul_workspace_prepare(&a)&&a.matmul_workspace);
    unsigned n=allocations;CUdeviceptr saved=a.matmul_workspace;
    assert(cuda_matmul_workspace_prepare(&a)&&allocations==n&&a.matmul_workspace==saved);
    assert(cuda_matmul_workspace_prepare(&b)&&b.matmul_workspace!=saved);
    assert(!cuda_matmul_workspace_release(&a,0)&&!enters&&!waits&&!frees&&a.matmul_workspace==saved);
    enter_ok=0;assert(!cuda_matmul_workspace_release(&a,1)&&!waits&&!frees&&a.matmul_workspace==saved);
    enter_ok=1;wait_status=700;
    assert(!cuda_matmul_workspace_release(&a,1)&&waits==1&&!frees&&a.matmul_workspace==saved);
    wait_status=0;free_status=700;
    assert(!cuda_matmul_workspace_release(&a,1)&&frees==1&&a.matmul_workspace==saved);
    free_status=0;assert(cuda_matmul_workspace_release(&a,1)&&!a.matmul_workspace);
    n=enters;assert(cuda_matmul_workspace_release(&a,1)&&enters==n);
    leave_ok=0;assert(!cuda_matmul_workspace_release(&b,1)&&!b.matmul_workspace);
    assert(cuda_graph_allocation_failed==13&&cuda_forward_context_failed==17);
    puts("PASS optional split workspace: distinct owners, OOM fallback, fatal errors and release failure ownership");
}
'''
        plan = (ROOT / "native/src/backends/cuda/dense_launch_plan.h").read_text()
        with tempfile.TemporaryDirectory(prefix="volvox-split-lifetime-") as tmp:
            source, binary = Path(tmp) / "proof.c", Path(tmp) / "proof"
            source.write_text(plan + prefix + body + cases)
            for training in (0, 1):
                built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    f"-DVOLVOXAI_ENABLE_TRAINING={training}", "-I", str(CUDA_INCLUDE), str(source), "-o", str(binary)],
                    capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
                checked = subprocess.run([str(binary)], capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, checked.stdout+checked.stderr)


if __name__ == "__main__":
    unittest.main()
