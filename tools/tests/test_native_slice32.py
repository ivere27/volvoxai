"""Execute translated Slice bits with the actual native3 typed route guards."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tools.tests.test_webgpu_llm_workspace import function_source


def slice_branch(source, start):
    first = source.index("if (op == VX_OP_SLICE)", start)
    opening = source.index("{", first)
    depth, closing = 1, opening + 1
    while depth:
        depth += (source[closing] == "{") - (source[closing] == "}")
        closing += 1
    return source[first:closing], closing


PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
typedef uint32_t uint;
typedef struct {uint x,y,z;} uvec3;
static uvec3 gl_GlobalInvocationID;
static const uint* _group_0_binding_0_cs;
static uint* _group_0_binding_1_cs;
enum {VX_OP_SLICE=1,VX_PORT_DATA,T_F32=2,T_I32=3,T_I8=4};
typedef struct {float* data;int dtype,ndim,shape[8];size_t elem_size;} T;
typedef struct {struct {int has_slice,slice_rank,starts[8],steps[8],axes[8];} parsed_params;} Node;
static T* fallback;
static int dispatches;
static T* nin(Node* node,int port) {(void)node;(void)port;return fallback;}
static void materialize_tensor_f32(T* tensor) {(void)tensor;}
'''

WRAPPER = r'''
static int run_shader(const float* input,float* output,const int* shape,int rank,
        const int* result_shape,int result_rank,const int* starts,const int* steps) {
    assert(rank==result_rank && rank>0 && rank<=4);
    uint in4[4]={1,1,1,1},out4[4]={1,1,1,1};
    for(int i=0;i<rank;i++) {in4[4-rank+i]=shape[i];out4[4-rank+i]=result_shape[i];}
    uint total=out4[0]*out4[1]*out4[2]*out4[3];
    uint words[16]={out4[0],out4[1],out4[2],out4[3],in4[1],in4[2],in4[3],
        starts[0],starts[1],starts[2],starts[3],steps[0],steps[1],steps[2],steps[3],total};
    memcpy(&_group_0_binding_2_cs,words,sizeof(words));
    _group_0_binding_0_cs=(const uint*)input;_group_0_binding_1_cs=(uint*)output;
    /* Actual translated shader body runs every invocation, including padded
     * dispatch tail invocations guarded by total. No F32 conversion occurs. */
    for(uint i=0;i<(total+63u)/64u*64u;i++) {gl_GlobalInvocationID=(uvec3){i,0,0};translated_main();}
    dispatches++;return 1;
}
#define vk_graph_slice4d_f32 run_shader
#define opengl_graph_slice4d_f32 run_shader
#define metal_graph_slice4d_f32 run_shader
'''

FIXTURE = r'''
int main(void) {
    uint patterns[8]={1u,2u,0xffffffffu,0x7fffffffu,0x80000000u,0x01000001u,0x7f800001u,0xffc01234u};
    uint input[24],output[32];for(unsigned plane=0;plane<3;plane++)for(unsigned i=0;i<8;i++)input[plane*8+i]=patterns[(i+plane)%8];
    T source={(float*)input,T_I32,2,{3,8},4},target={(float*)output,T_I32,2,{1,8},4};
    Node node={0};node.parsed_params.has_slice=1;node.parsed_params.slice_rank=1;
    node.parsed_params.axes[0]=0;node.parsed_params.steps[0]=1;const char* backend=NULL;
    /* The three actual MRoPE coordinate planes retain tiny positive IDs,
     * negative/large signed IDs and every 32-bit pattern exactly. */
    for(int dtype=T_F32;dtype<=T_I32;dtype++) {
        source.dtype=target.dtype=dtype;
        for(int plane=0;plane<3;plane++) {
            memset(output,0xa5,sizeof(output));node.parsed_params.starts[0]=plane;
            assert(run_slice(&node,&source,&target,&backend));assert(backend && !strcmp(backend,EXPECTED_BACKEND));
            assert(!memcmp(output,input+plane*8,8*sizeof(uint)));
            for(unsigned i=8;i<32;i++)assert(output[i]==0xa5a5a5a5u);
        }
    }
    /* Independent strided oracle slices axes0 and1 rather than the special
     * whole-plane path, including F32 NaN/subnormal/negative-zero bit payloads. */
    target.shape[0]=2;target.shape[1]=3;
    node.parsed_params.slice_rank=2;node.parsed_params.axes[0]=0;node.parsed_params.axes[1]=1;
    node.parsed_params.starts[0]=1;node.parsed_params.starts[1]=1;
    node.parsed_params.steps[0]=1;node.parsed_params.steps[1]=2;
    for(int dtype=T_F32;dtype<=T_I32;dtype++) {
        source.dtype=target.dtype=dtype;memset(output,0xa5,sizeof(output));
        assert(run_slice(&node,&source,&target,&backend));
        for(unsigned p=0;p<2;p++)for(unsigned j=0;j<3;j++)assert(output[p*3+j]==input[(p+1)*8+1+j*2]);
        for(unsigned i=6;i<32;i++)assert(output[i]==0xa5a5a5a5u);
    }
    int before=dispatches;target.dtype=T_F32;source.dtype=T_I32;
    assert(!run_slice(&node,&source,&target,&backend) && dispatches==before);
    target.dtype=source.dtype=T_I8;target.elem_size=source.elem_size=1;
    assert(!run_slice(&node,&source,&target,&backend) && dispatches==before);
    target.dtype=source.dtype=T_I32;target.elem_size=source.elem_size=4;
    node.parsed_params.steps[1]=0;assert(!run_slice(&node,&source,&target,&backend) && dispatches==before);
    node.parsed_params.steps[1]=4;assert(!run_slice(&node,&source,&target,&backend) && dispatches==before);
    node.parsed_params.steps[1]=1;node.parsed_params.axes[1]=0;
    assert(!run_slice(&node,&source,&target,&backend) && dispatches==before);
    puts("PASS native F32/I32 Slice guards and translated raw-bit numerical kernel");
}
'''


class NativeSlice32Tests(unittest.TestCase):
    def test_actual_translated_slice_and_native_typed_guards(self):
        compiler = shutil.which("cc")
        root = Path(__file__).resolve().parents[2]
        shader_compiler = root / "tools/metal_shader_compiler/target/release/volvoxai-native-shader-compiler"
        if not compiler or not shader_compiler.is_file():
            self.skipTest("C/Naga shader compiler unavailable")
        runtime = (root / "native/src/runtime/engine_runtime_f32_gpu.inc").read_text()
        cpu = (root / "native/src/runtime/engine_runtime_f32_cpu.inc").read_text()
        with tempfile.TemporaryDirectory(prefix="volvoxai-native-slice32-") as temporary:
            directory = Path(temporary)
            core, es, metal = directory / "core", directory / "es", directory / "metal"
            subprocess.run([str(shader_compiler), "glsl", str(root / "shaders/inference"), str(core), str(es), "slice"], check=True)
            subprocess.run([str(shader_compiler), "msl", str(root / "shaders/inference"), str(metal), "slice"], check=True)
            glsl = (core / "slice.comp").read_text()
            self.assertRegex(glsl, r"buffer[^\n]*\{ uint _group_0_binding_0_cs\[\]")
            self.assertRegex((es / "slice.comp").read_text(), r"buffer[^\n]*\{ uint _group_0_binding_1_cs\[\]")
            self.assertIn("typedef uint type_1[1];", (metal / "slice.metal").read_text())
            params = re.search(r"struct Params \{.*?\};", glsl, re.S).group()
            shader = function_source(glsl, "main").replace("void main()", "static void translated_main(void)")
            cursor = 0
            for backend in ("vulkan", "opengl", "metal"):
                with self.subTest(backend=backend):
                    branch, cursor = slice_branch(runtime, cursor)
                    entry = function_source(runtime, "try_" + backend + "_graph_node")
                    gate = re.search(r"if \(o->dtype != T_F32.*?return 0;", entry, re.S).group()
                    source = PREAMBLE + params + "\nstatic struct Params _group_0_binding_2_cs;\n" + shader
                    source += WRAPPER + function_source(cpu, "slice_canonical_start")
                    source += "static int run_slice(Node* n,T* in,T* o,const char** node_backend) {int op=VX_OP_SLICE;\n" + gate + "\n" + branch + "\nreturn 0;}\n"
                    source += '#define EXPECTED_BACKEND "' + backend + '-graph"\n' + FIXTURE
                    unit, binary = directory / (backend + ".c"), directory / (backend + "-test")
                    unit.write_text(source)
                    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(unit), "-o", str(binary)], check=True)
                    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
                    self.assertIn("PASS native F32/I32 Slice", result.stdout)


if __name__ == "__main__":
    unittest.main()
