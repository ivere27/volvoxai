"""Exercise the production WebGPU allocator and combined-slot proof on CPU."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests import function_source




class WebGpuWorkspaceTests(unittest.TestCase):
    def test_actual_combined_allocator_rounding_and_owner_restore(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        backend = (root / "native/src/backends/webgpu_backend.c").read_text()
        plans = (root / "native/src/backends/webgpu_llm_plans.inc").read_text()
        preamble = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef struct VxWebGpuArenaBlock {
    struct VxWebGpuArenaBlock* next;
    size_t capacity;
    _Alignas(16) unsigned char data[];
} VxWebGpuArenaBlock;
typedef struct {
    uint64_t maxBufferSize, maxStorageBufferBindingSize;
} Limits;
typedef struct {
    Limits limits;
    VxWebGpuArenaBlock* arena;
    size_t arena_used;
    void* spans;
    void* llm_workspace[2];
    size_t llm_workspace_bytes[2];
    void* llm_zero;
} VxWebGpuState;
typedef struct {VxWebGpuState* webgpu_state;} VxEngineState;
typedef struct {VxEngineState* prior;} VxEngineStateScope;
static VxEngineState* current;
static int validating, mark_calls;
#define g_webgpu (*current->webgpu_state)
#define g_webgpu_arena_used (g_webgpu.arena_used)
static VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* state) {
    VxEngineStateScope saved={current}; current=state; return saved;
}
static void vx_engine_state_scope_leave(VxEngineStateScope scope) {current=scope.prior;}
static void vx_webgpu_mark_host(void* user, void* host, size_t bytes, int weight) {
    (void)user;(void)host;(void)bytes;(void)weight;mark_calls++;
}
'''
        translation = preamble
        translation += function_source(backend, "vx_webgpu_arena")
        translation += function_source(plans, "vx_webgpu_llm_reserve")
        translation += function_source(backend, "vx_webgpu_llm_proof_capacity")
        translation += function_source(backend, "vx_webgpu_prove_workspace_domain")
        translation += r'''
static uint64_t expected(const uint64_t slots[2]) {
    /* Both power-of-two-sized workspaces fill one block each. The nine
     * optional bindings require a third minimum-size arena block. */
    return slots[0]+slots[1]+3u*sizeof(VxWebGpuArenaBlock)+
        slots[0]+slots[1]+65536u+36u;
}
int main(void) {
    VxWebGpuState live={.limits={1ull<<32,1ull<<32}};
    int marker;
    live.llm_workspace[0]=&marker;live.llm_workspace_bytes[0]=7;
    VxEngineState state={&live}, prior={&live};current=&prior;
    uint64_t slots[2]={524288u,1048576u}, capacity=0;
    assert(vx_webgpu_prove_workspace_domain(&state,slots,&capacity));
    assert(capacity==expected(slots));
    assert(current==&prior && state.webgpu_state==&live && validating==0 && mark_calls==0);
    assert(!live.arena && live.llm_workspace[0]==&marker && live.llm_workspace_bytes[0]==7);
    /* Separately proving each node is smaller than the combined persistent
     * maxima, even though the graph executes the nodes serially. */
    uint64_t one[2]={slots[0],512u}, first, second;
    assert(vx_webgpu_prove_workspace_domain(&state,one,&first));
    one[0]=16;one[1]=slots[1];assert(vx_webgpu_prove_workspace_domain(&state,one,&second));
    assert(capacity>first && capacity>second);
    live.limits.maxStorageBufferBindingSize=slots[1]-1;
    assert(!vx_webgpu_prove_workspace_domain(&state,slots,&capacity) && capacity==0);
    assert(state.webgpu_state==&live && current==&prior && !live.arena && mark_calls==0);
    live.limits.maxStorageBufferBindingSize=1ull<<32;
    slots[1]=(uint64_t)UINT32_MAX;
    assert(!vx_webgpu_prove_workspace_domain(&state,slots,&capacity));
    slots[0]=slots[1]=0;assert(vx_webgpu_prove_workspace_domain(&state,slots,&capacity));
    assert(capacity==65536u+sizeof(VxWebGpuArenaBlock)+36u);
    validating=1;slots[0]=16;assert(vx_webgpu_prove_workspace_domain(&state,slots,&capacity));
    assert(validating==1 && current==&prior && mark_calls==0);
    puts("PASS WebGPU combined workspace rounding, limits and owner restoration");
}
'''
        with tempfile.TemporaryDirectory(prefix="volvoxai-webgpu-workspace-") as temporary:
            source = Path(temporary) / "test.c"
            binary = Path(temporary) / "test"
            source.write_text(translation)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("PASS WebGPU combined workspace", result.stdout)


if __name__ == "__main__":
    unittest.main()
