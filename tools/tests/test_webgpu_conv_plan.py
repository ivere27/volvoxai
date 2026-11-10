"""Exercise concrete vector-route admission, grouped layouts and shape changes."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class WebGpuConvPlanTests(unittest.TestCase):
    def test_layout_guards_and_concrete_shape_reselection(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = (ROOT / "native/src/backends/webgpu_backend.c").read_text()
        start = source.index("static int vx_webgpu_plan_conv2d(")
        end = source.index("/* Defined with the strided routes", start)
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
enum { VX_PORT_WEIGHT, VX_PORT_BIAS, VX_CONV_WEIGHT_LAYOUT_HWIO,
       VX_CONV_WEIGHT_LAYOUT_HWCM, VX_SHADER_INFERENCE_CONV2_D,
       VX_SHADER_INFERENCE_CONV2_D_REGULAR_OUT16 };
#define VX_WEBGPU_WORKGROUP_LIMIT 65535u
typedef struct { int ndim, shape[4]; void* data; } T;
typedef struct { int weight_layout,groups,relu,stride[2],dilation[2],pads[2]; } Params;
typedef struct { T* ports[2]; Params parsed_params; int fuse_relu6; } Node;
typedef struct { unsigned shader_id,workgroup[3]; } Variant;
typedef struct { unsigned variant_count; Variant variants[1]; } VxWebGpuNodePlan;
static T* vx_webgpu_port(const Node* n,int p) { return n->ports[p]; }
static int vx_webgpu_f32_extent(const T* t,uint32_t* count,size_t* bytes) {
    if(!t) return 0;
    *count=1; for(int i=0;i<t->ndim;i++) *count *= t->shape[i];
    *bytes=(size_t)*count*4; return 1;
}
static void* vx_webgpu_arena(size_t bytes) { static float zero[64]; assert(bytes<=sizeof(zero)); return zero; }
static int vx_webgpu_bind(VxWebGpuNodePlan* p,const void* data,size_t bytes) { (void)p;(void)data;return bytes>0; }
static int overwrite_count;
static int vx_webgpu_bind_overwrite(VxWebGpuNodePlan* p,const void* data,size_t bytes) {
    overwrite_count++; return vx_webgpu_bind(p,data,bytes);
}
static int vx_webgpu_params(VxWebGpuNodePlan* p,const uint32_t* v,unsigned n) { (void)p;(void)v;return n==17; }
'''
        main = r'''
int main(void) {
    float data[1];
    T x={4,{2,1,19,8},data},w={4,{1,3,8,32},data},b={1,{32},data},y={4,{2,1,19,32},data};
    Node node={.ports={&w,&b},.parsed_params={.weight_layout=VX_CONV_WEIGHT_LAYOUT_HWIO,
        .groups=1,.stride={1,1},.dilation={1,1},.pads={0,1}}};
    VxWebGpuNodePlan plan={0};
    assert(vx_webgpu_plan_conv2d(&node,&x,&y,&plan));
    assert(overwrite_count==1);
    assert(plan.variants[0].shader_id==VX_SHADER_INFERENCE_CONV2_D_REGULAR_OUT16);
    assert(plan.variants[0].workgroup[0]==3 && plan.variants[0].workgroup[1]==1 && plan.variants[0].workgroup[2]==4);
    /* The same node changes shape/layout: never reuse the prior route. */
    y.shape[3]=w.shape[3]=b.shape[0]=17;
    assert(vx_webgpu_plan_conv2d(&node,&x,&y,&plan));
    assert(plan.variants[0].shader_id==VX_SHADER_INFERENCE_CONV2_D && plan.variants[0].workgroup[2]==34);
    y.shape[3]=w.shape[3]=b.shape[0]=32;
    node.parsed_params.groups=2;w.shape[2]=4;
    assert(vx_webgpu_plan_conv2d(&node,&x,&y,&plan));
    assert(plan.variants[0].shader_id==VX_SHADER_INFERENCE_CONV2_D);
    node.parsed_params.groups=8;node.parsed_params.weight_layout=VX_CONV_WEIGHT_LAYOUT_HWCM;
    w.shape[2]=8;w.shape[3]=4;
    assert(vx_webgpu_plan_conv2d(&node,&x,&y,&plan));
    assert(plan.variants[0].shader_id==VX_SHADER_INFERENCE_CONV2_D);
    node.parsed_params.groups=1;x.shape[3]=w.shape[2]=1;w.shape[3]=32;
    assert(vx_webgpu_plan_conv2d(&node,&x,&y,&plan));
    assert(plan.variants[0].shader_id==VX_SHADER_INFERENCE_CONV2_D);
    node.parsed_params.weight_layout=VX_CONV_WEIGHT_LAYOUT_HWIO;x.shape[3]=w.shape[2]=8;
    node.ports[VX_PORT_BIAS]=NULL;
    assert(vx_webgpu_plan_conv2d(&node,&x,&y,&plan));
    assert(plan.variants[0].shader_id==VX_SHADER_INFERENCE_CONV2_D_REGULAR_OUT16);
    y.shape[0]=x.shape[0]=65536;
    assert(!vx_webgpu_plan_conv2d(&node,&x,&y,&plan));
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-webgpu-conv-plan-") as directory:
            root = Path(directory)
            (root / "test.c").write_text(harness + source[start:end] + main)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                str(root / "test.c"), "-o", str(root / "test")], check=True)
            subprocess.run([str(root / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
