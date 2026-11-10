"""LSTM optional inputs must not resize an existing WebGPU input span."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class WebGpuLstmPlanTests(unittest.TestCase):
    def test_optional_bindings_keep_distinct_span_extents(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
enum { T_F32=1, VX_PORT_WEIGHT, VX_PORT_RECURRENT_WEIGHT, VX_PORT_BIAS,
       VX_PORT_INITIAL_H, VX_PORT_INITIAL_C, VX_NODE_PARAM_DIRECTION,
       VX_SHADER_INFERENCE_LSTM };
typedef struct { int ndim,shape[8],dtype; float* data; } T;
typedef struct { T* ports[8]; int direction; } Node;
typedef struct { const void* data; size_t bytes; } Binding;
typedef struct { unsigned shader_id,workgroup[3]; } Variant;
typedef struct { Binding bindings[8]; unsigned count,variant_count; Variant variants[1]; uint32_t params[8]; } VxWebGpuNodePlan;
static T* vx_webgpu_port(const Node* node,int port) { return node->ports[port]; }
static int vx_node_param_i32(const Node* node,int key,int fallback) { (void)key; (void)fallback; return node->direction; }
static int vx_webgpu_bind(VxWebGpuNodePlan* plan,const void* data,size_t bytes) {
    /* The bridge owns one buffer extent per host address. A second extent at
       that address would replace the first buffer and invalidate its binding. */
    for(unsigned i=0;i<plan->count;i++)
        if(plan->bindings[i].data==data && plan->bindings[i].bytes!=bytes)return 0;
    assert(plan->count<8); plan->bindings[plan->count++]=(Binding){data,bytes};return 1;
}
int vx_webgpu_bind_scratch(VxWebGpuNodePlan* plan,const void* data,size_t words) {
    static float scratch[8]; assert(words==1 && plan->count<8);
    memcpy(scratch+plan->count,data,4); return vx_webgpu_bind(plan,scratch+plan->count,4);
}
static int vx_webgpu_params(VxWebGpuNodePlan* plan,const uint32_t* words,size_t count) {
    assert(count==8); memcpy(plan->params,words,32); return 1;
}
#include "webgpu_lstm_plan.inc"
int main(void) {
    float x[30],w[140],r[196],bias[56],h[14],c[14],y[42];
    T input={3,{3,2,5},T_F32,x},weight={3,{1,28,5},T_F32,w},recurrent={3,{1,28,7},T_F32,r};
    T b={2,{1,56},T_F32,bias},initial_h={3,{1,2,7},T_F32,h},initial_c={3,{1,2,7},T_F32,c};
    T output={4,{3,1,2,7},T_F32,y}; Node node={0};
    node.ports[VX_PORT_WEIGHT]=&weight; node.ports[VX_PORT_RECURRENT_WEIGHT]=&recurrent;
    for(unsigned optional=0;optional<3;optional++) {
        VxWebGpuNodePlan plan={0};
        node.ports[VX_PORT_BIAS]=optional?&b:NULL;
        node.ports[VX_PORT_INITIAL_H]=optional==2?&initial_h:NULL;
        node.ports[VX_PORT_INITIAL_C]=optional==2?&initial_c:NULL;
        assert(vx_webgpu_plan_lstm(&node,&input,&output,&plan));
        assert(plan.count==7 && plan.bindings[0].data==x && plan.bindings[0].bytes==sizeof(x));
        assert(plan.params[5]==(optional==0?0u:optional==1?1u:7u));
        assert(plan.variants[0].workgroup[0]==2 && plan.variant_count==1);
        for(unsigned i=3;i<6;i++)assert(plan.bindings[i].data!=x);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-webgpu-lstm-plan-") as directory:
            root = Path(directory)
            (root / "test.c").write_text(source)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "native/src/backends"), str(root / "test.c"), "-o", str(root / "test")], check=True)
            subprocess.run([str(root / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
