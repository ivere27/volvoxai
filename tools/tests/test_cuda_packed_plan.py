"""Packed variant identity, admission, geometry and workspace stay one contract."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class CudaPackedPlanTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_CUDA_PTX"),
                         "set VOLVOXAI_TEST_CUDA_PTX for selected-route CUDA numerics")
    def test_tiny_row_selection_preserves_cuda_arithmetic_and_row_isolation(self):
        import ctypes as ct
        import numpy as np
        from tools.tests.test_packed_linear import PackedLinearCudaTests, reference

        # Exercise the real selector, then the selected device tactic. The
        # independent oracle and device guard regions check more than the
        # variant identity asserted by the host-only boundary tests below.
        source = r'''
#include "native/src/backends/cuda/packed_launch_plan.h"
int select_route(unsigned rows,unsigned k,unsigned n,unsigned bits,unsigned weights,unsigned* output) {
    VxCudaPackedLaunchPlan plan;
    if(!vx_cuda_packed_select(rows,(uint64_t)rows*n,k,n,128,bits,weights,86,0,1,1,82,49152,&plan))return 0;
    output[0]=plan.execution.route;
    output[1]=vx_cuda_packed_linear_prepares_input(bits,n,plan.execution.route);
    return 1;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-packed-selection-") as tmp:
            path, library = Path(tmp)/"selection.c", Path(tmp)/"selection.so"
            path.write_text(source)
            subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-I", str(ROOT),
                            str(path), "-o", str(library)], check=True)
            select = ct.CDLL(str(library)).select_route
            select.argtypes = [ct.c_uint]*5 + [ct.POINTER(ct.c_uint)]
            select.restype = ct.c_int
            PackedLinearCudaTests.setUpClass()
            try:
                cuda = PackedLinearCudaTests("test_shared_tiles_tails_nonfinite_bias_and_physical_guards")
                rng = np.random.default_rng(20261027)
                for rows,n,k,bits,weights in ((2,4096,2048,32,4), (4,4096,2048,8,4),
                        (4,2624,2049,16,4), (8,2624,2049,16,4),
                        (2,8192,128,8,8), (2,4096,127,16,4)):
                    with self.subTest(rows=rows,n=n,k=k,bits=bits,weights=weights):
                        x = rng.normal(0,.2,(rows,k)).astype(np.float32)
                        x[0] = 0
                        if rows>2:
                            x[-1,0] = np.nan
                        if rows>4:
                            x[1] *= np.float32(2**-120)
                        offset = 1 << (weights-1)
                        q = rng.integers(-offset,offset,(n,k),dtype=np.int16).astype(np.int8)
                        scales = rng.uniform(.001,.03,(n,(k+127)//128)).astype(np.float32)
                        bias = rng.normal(0,.02,n).astype(np.float32)
                        plan = (ct.c_uint*2)()
                        self.assertTrue(select(rows,k,n,bits,weights,plan))
                        actual = cuda.evaluate(x,q,scales,bias,bits,weights,plan[0],
                                               prepared=bool(plan[1]),sm_count=82)
                        expected = reference(x,q,scales,bias,bits)
                        if bits==8:
                            np.testing.assert_array_equal(actual,expected)
                        else:
                            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5,equal_nan=True)
            finally:
                PackedLinearCudaTests.tearDownClass()

    def test_registered_variants_requested_support_and_exact_launch_geometry(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = ROOT / "native/src/backends/cuda/host"
        source = r'''
#include <assert.h>
#include <string.h>
#include "native/src/backends/cuda/packed_launch_plan.h"
typedef unsigned CUfunction;
enum {
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol, storage) handle,
#include "cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
};
#define CUDA_FORWARD_FUNCTION(handle) handle
'''
        source += function_source((host / "cuda_graph_llm_host.inc").read_text(), "cuda_packed_function")
        # Reproduce a fork adding only a new row threshold, without registering
        # its function. The real selection body must refuse that incomplete edit.
        selection = function_source((host.parent / "packed_launch_plan.h").read_text(),
                                    "vx_cuda_packed_prepared_variant")
        selection = selection.replace("vx_cuda_packed_prepared_variant(", "unregistered_m32(")
        selection = selection.replace(
            "unsigned bm=vx_cuda_packed_linear_prepared_tile_rows(rows,k,n,bits,weights,arch);",
            "unsigned bm=32u;")
        source += selection
        source += r'''
int main(void) {
    VxCudaPackedLaunchPlan p;
    assert(unregistered_m32(1536,2048,2048,8,4,86)==VX_CUDA_PACKED_VARIANT_COUNT);
    assert(unregistered_m32(1536,2048,2048,16,4,86)==VX_CUDA_PACKED_VARIANT_COUNT);
    for(unsigned i=0;i<VX_CUDA_PACKED_VARIANT_COUNT;i++) {
        const VxCudaPackedVariant* v=&vx_cuda_packed_variants[i];
        unsigned bits=(v->arithmetic_mask&1)?8:(v->arithmetic_mask&2)?16:32;
        assert(vx_cuda_packed_plan(i,65,65*257,257,257,128,bits,4,86,1,1,65536,&p));
        for(unsigned lowbits=2;lowbits<=3;lowbits++) {
            VxCudaPackedLaunchPlan denied;
            assert(!vx_cuda_packed_plan(i,65,65*257,257,257,128,bits,lowbits,86,1,1,65536,&denied));
        }
        assert(p.variant==i&&p.threads==v->threads&&p.tile_rows==v->tile_rows&&p.tile_k==v->tile_k);
        assert(p.execution.tile_columns==v->execution.tile_columns);
        assert(p.total_shared_bytes==p.shared_bytes+v->static_shared_bytes);
        assert(p.grid==(v->execution.route==VX_CUDA_PACKED_PREPARE?65:
            !v->tile_rows?(65*257+1)/2:
            ((65+v->tile_rows-1)/v->tile_rows)*((257+v->execution.tile_columns-1)/v->execution.tile_columns)));
        assert(cuda_packed_function(i));
        for(unsigned j=0;j<i;j++)assert(strcmp(v->id,vx_cuda_packed_variants[j].id));
        if(v->group_size)assert(!vx_cuda_packed_plan(i,65,65*257,257,257,32,bits,4,86,1,1,65536,&p));
        if(v->prepared_only)assert(!vx_cuda_packed_plan(i,65,65*257,257,257,128,bits,4,86,1,0,65536,&p));
        if(v->minimum_arch)assert(!vx_cuda_packed_plan(i,65,65*257,257,257,128,bits,4,v->minimum_arch-1,1,1,65536,&p));
        if(v->fast_only)assert(!vx_cuda_packed_plan(i,65,65*257,257,257,128,bits,4,86,0,1,65536,&p));
    }
    assert(vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_A16_CACHE,1,65,257,65,128,16,4,86,0,1,49152,&p));
    assert(p.shared_bytes==384*2&&p.total_shared_bytes==384*2+192);
    assert(vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_F32_CACHE,1,65,257,65,128,32,4,86,0,1,49152,&p));
    assert(p.shared_bytes==384*4&&p.total_shared_bytes==384*4+192);
    assert(cuda_packed_function(VX_CUDA_PACKED_VARIANT_A8_PREPARED_M16)==k_packed_a8_m16);
    assert(cuda_packed_function(VX_CUDA_PACKED_VARIANT_A8_PREPARED_M64)==k_packed_a8_m64);
    assert(!cuda_packed_function(VX_CUDA_PACKED_VARIANT_COUNT));
    for(unsigned g=32;g<=256;g*=2)for(unsigned bits=8;bits<=32;bits*=2) {
        assert(vx_cuda_packed_select(31,31*65,257,65,g,bits,8,86,1,1,1,82,49152,&p));
        if(g!=128)assert(p.variant==VX_CUDA_PACKED_VARIANT_GENERIC&&p.tile_k==32&&p.grid==(31*65+1)/2);
        assert(vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_GENERIC,31,31*65,257,65,g,bits,4,86,0,0,49152,&p));
    }
    /* G128 and BK16 retain distinct semantics; exactly this registered kernel
     * owns BM16/BN32. A nonexistent M32 configuration cannot acquire its ID. */
    assert(vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_A8_M16_N32,33,33*65,257,65,128,8,4,86,0,0,49152,&p));
    assert(p.tile_rows==16&&p.execution.tile_columns==32&&p.tile_k==16&&p.grid==9);
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_COUNT,33,33*65,257,65,128,8,4,86,0,0,49152,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_GENERIC,33,33*65,257,65,0,8,4,86,0,0,49152,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_GENERIC,33,33*65,257,65,96,8,4,86,0,0,49152,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_GENERIC,33,33*65,257,65,128,8,3,86,0,0,49152,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_GENERIC,UINT64_MAX,UINT64_MAX,257,65,128,8,4,86,0,0,49152,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_A8_M16_N32,33,33*65-1,257,65,128,8,4,86,0,0,49152,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_A8_M16_N32,33,33*65,257,65,128,16,4,86,0,0,49152,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_A8_M16_N32,33,33*65,257,65,128,8,4,86,0,0,2000,&p));
    assert(vx_cuda_packed_prepared_variant(1536,2048,2048,8,4,86)==VX_CUDA_PACKED_VARIANT_A8_PREPARED_M64);
    assert(vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_A8_PREPARED_M64,1536,1536*2048,2048,2048,128,8,4,86,0,1,49152,&p));
    assert(vx_cuda_packed_workspace(&p,1536,2048,2048,8,4,86,1));
    assert(p.workspace_stride==2048&&p.workspace_limit==32*1024*1024&&p.chunk_capacity>=1536);
    /* The shared entry contains A8 staging160 plus float validity32 bytes.
     * Test exact total boundaries, not only ordinary device capacities. */
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_GENERIC,1,65,128,65,32,8,4,86,0,0,191,&p));
    assert(vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_GENERIC,1,65,128,65,32,8,4,86,0,0,192,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_A8_CACHE,1,65,128,65,128,8,4,86,0,0,319,&p));
    assert(vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_A8_CACHE,1,65,128,65,128,8,4,86,0,0,320,&p));
    assert(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_PREPARE,1,65,128,65,128,8,4,86,0,1,223,&p));
    assert(vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_PREPARE,1,65,128,65,128,8,4,86,0,1,224,&p));
    p.variant=VX_CUDA_PACKED_VARIANT_COUNT;
    assert(!vx_cuda_packed_workspace(&p,1536,2048,2048,8,4,86,1));
    /* Padding small batches to M16 needs enough active rows as well as SM
     * coverage. Check both sides of each arithmetic-specific row boundary,
     * including long reductions and both supported CUDA weight widths. */
    for(unsigned weights=4;weights<=8;weights*=2)
    for(unsigned rows=2;rows<=16;rows++)
    for(unsigned k=2048;k<=8192;k*=4) {
        assert(vx_cuda_packed_select(rows,(uint64_t)rows*4096u,k,4096,128,8,weights,86,0,1,1,82,49152,&p));
        assert(p.variant==(rows>=4u?VX_CUDA_PACKED_VARIANT_A8_M16_N32:VX_CUDA_PACKED_VARIANT_A8_CACHE));
        assert(vx_cuda_packed_select(rows,(uint64_t)rows*4096u,k,4096,128,16,weights,86,0,1,1,82,49152,&p));
        assert(p.variant==(rows>=4u?VX_CUDA_PACKED_VARIANT_A16_M16_N32:VX_CUDA_PACKED_VARIANT_A16_CACHE));
        for(unsigned fast=0;fast<=1;fast++) {
            assert(vx_cuda_packed_select(rows,(uint64_t)rows*4096u,k,4096,128,32,weights,86,fast,1,1,82,49152,&p));
            assert(p.variant==(rows>=8u?VX_CUDA_PACKED_VARIANT_F32_M16_N32:VX_CUDA_PACKED_VARIANT_F32_CACHE));
        }
        assert(vx_cuda_packed_select(rows,(uint64_t)rows*256u,2048,256,128,8,4,86,0,1,1,82,49152,&p));
        assert(p.variant==VX_CUDA_PACKED_VARIANT_A8_CACHE);
    }
    /* Single-group reductions and wide A8 remain qualified below the active
     * row floor. The first K beyond each exception returns to row-cache. */
    for(unsigned bits=8;bits<=32;bits*=2) {
        assert(vx_cuda_packed_select(2,8192,128,4096,128,bits,4,86,0,1,1,82,49152,&p));
        assert(p.tile_rows==16u && p.grid==128u);
        assert(vx_cuda_packed_select(2,8192,129,4096,128,bits,4,86,0,1,1,82,49152,&p));
        assert(p.execution.route==VX_CUDA_PACKED_ROW_CACHE);
    }
    assert(vx_cuda_packed_select(2,16384,2048,8192,128,8,4,86,0,1,1,82,49152,&p));
    assert(p.variant==VX_CUDA_PACKED_VARIANT_A8_M16_N64);
    assert(vx_cuda_packed_select(2,16384,2049,8192,128,8,4,86,0,1,1,82,49152,&p));
    assert(p.variant==VX_CUDA_PACKED_VARIANT_A8_CACHE);
    assert(vx_cuda_packed_select(2,16384,2048,8192,128,8,4,86,0,1,1,132,49152,&p));
    assert(p.variant==VX_CUDA_PACKED_VARIANT_A8_CACHE);
    /* F16 tails and narrow long reductions do not amortize M16 at four rows.
     * Aligned wider reductions still qualify, and eight rows recover reuse. */
    for(unsigned weights=4;weights<=8;weights*=2) {
        assert(vx_cuda_packed_select(4,16384,2049,4096,128,16,weights,86,0,1,1,82,49152,&p));
        assert(p.variant==VX_CUDA_PACKED_VARIANT_A16_CACHE);
        assert(vx_cuda_packed_select(4,10496,8192,2624,128,16,weights,86,0,1,1,82,49152,&p));
        assert(p.variant==VX_CUDA_PACKED_VARIANT_A16_CACHE);
        assert(vx_cuda_packed_select(4,10496,2048,2624,128,16,weights,86,0,1,1,82,49152,&p));
        assert(p.variant==VX_CUDA_PACKED_VARIANT_A16_M16_N32);
        assert(vx_cuda_packed_select(8,20992,2049,2624,128,16,weights,86,0,1,1,82,49152,&p));
        assert(p.variant==VX_CUDA_PACKED_VARIANT_A16_M16_N32);
    }
    assert(vx_cuda_packed_select(2,5248,128,2624,128,8,4,86,0,1,1,82,49152,&p));
    assert(p.variant==VX_CUDA_PACKED_VARIANT_A8_CACHE);
    assert(vx_cuda_packed_select(2,16384,128,8192,32,8,4,86,0,1,1,82,49152,&p));
    assert(p.variant==VX_CUDA_PACKED_VARIANT_GENERIC);
    assert(vx_cuda_packed_select(2,8192,128,4096,128,8,4,86,0,0,1,82,49152,&p));
    assert(p.variant==VX_CUDA_PACKED_VARIANT_A8_CACHE);
    assert(vx_cuda_packed_select(1,4096,2048,4096,128,8,4,86,0,1,1,82,49152,&p));
    assert(p.variant==VX_CUDA_PACKED_VARIANT_A8_CACHE);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-packed-plan-") as tmp:
            path, binary = Path(tmp) / "plan.c", Path(tmp) / "plan"
            path.write_text(source)
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                "-I", str(ROOT), "-I", str(host), str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
