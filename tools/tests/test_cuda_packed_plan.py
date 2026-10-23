"""Packed variant identity, admission, geometry and workspace stay one contract."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class CudaPackedPlanTests(unittest.TestCase):
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
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-packed-plan-") as tmp:
            path, binary = Path(tmp) / "plan.c", Path(tmp) / "plan"
            path.write_text(source)
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                "-I", str(ROOT), "-I", str(host), str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
