"""An empty captured adapter run skips the scheduler scan without changing routes."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "native/src/runtime"

FIXTURE = r'''
#include <assert.h>
#include <stdio.h>
#include "engine_internal.h"
#include "runtime_state.c"
#include "adapter_runtime.c"
#include "lora_linear.inc"

static unsigned target_queries, checks;
static int observed_target(const char* name) {
    target_queries++;
    return vx_adapter_run_has_target(name);
}
#define vx_adapter_run_has_target observed_target
SCHEDULER
REFERENCE
#undef vx_adapter_run_has_target

static void check(int versions, int target, unsigned queries) {
    assert(vx_adapter_run_has_versions()==versions);
    target_queries=0;
    assert(run_has_adapter_targets()==target && target_queries==queries);
    target_queries=0;
    assert(run_has_adapter_targets_reference()==target);
    if(!versions) assert(target_queries==5);
    checks++;
}
static void stage(const char* version, const char* weight, float a, float b) {
    VxAdapterTargetSpec target={.weight_name=weight,.kind=VX_ADAPTER_LORA,
        .d_in=1,.d_out=1,.rank=1,.alpha=1,.scale=1,
        .a={.data=&a,.nbytes=sizeof(a),.dtype=VX_DTYPE_F32,.rows=1,.cols=1},
        .b={.data=&b,.nbytes=sizeof(b),.dtype=VX_DTYPE_F32,.rows=1,.cols=1}};
    VxAdapterVersionSpec spec={.version_id=version,.targets=&target,.target_count=1};
    assert(!vx_adapter_stage(&spec));
}
int main(void) {
    VxEngineState state={0}, sibling={0};
    Node nodes[3]={0}; Ref inputs[5]={0};
    const char* names[]={"input","untouched","other","weightA","weightB"};
    for(int i=0;i<5;i++) strcpy(inputs[i].name,names[i]);
    nodes[0].ins=inputs;nodes[0].nin=2;
    nodes[1].ins=inputs+2;nodes[1].nin=2;
    nodes[2].ins=inputs+4;nodes[2].nin=1;
    state.nodes=nodes;state.node_count=3;
    g_current_engine_state=&state;
    check(0,0,0);
    assert(!vx_adapter_run_begin());check(0,0,0);
    stage("A","weightA",2,3);stage("B","weightB",3,4);
    stage("unrelated","not-in-graph",1,1);
    assert(!vx_adapter_activate("A"));
    /* A captured empty run stays empty when the default changes. */
    check(0,0,0);assert(!vx_adapter_run_begin());check(0,0,0);
    vx_adapter_run_end();check(0,0,0);vx_adapter_run_end();check(0,0,0);

    assert(!vx_adapter_run_begin());check(1,1,4);
    assert(!vx_adapter_activate("B"));assert(!vx_adapter_remove("A"));
    assert(vx_adapter_run_has_target("weightA") && !vx_adapter_run_has_target("weightB"));
    assert(!vx_adapter_run_begin());check(1,1,4);
    assert(vx_adapter_request_begin("B")==-1);
    vx_adapter_run_end();check(1,1,4);vx_adapter_run_end();check(0,0,0);
    assert(!vx_adapter_run_begin());check(1,1,5);vx_adapter_run_end();

    /* Explicit empty requests override a nonempty global default. */
    assert(!vx_adapter_request_begin(""));assert(!vx_adapter_run_begin());check(0,0,0);
    vx_adapter_request_end();check(0,0,0); /* held run protects the request */
    assert(!vx_adapter_run_begin());check(0,0,0);vx_adapter_run_end();vx_adapter_run_end();vx_adapter_request_end();
    const char* empty[]={NULL,"",NULL};
    assert(!vx_adapter_request_begin_many(empty,NULL,3));assert(!vx_adapter_run_begin());
    check(0,0,0);vx_adapter_run_end();vx_adapter_request_end();
    assert(!vx_adapter_request_begin_many(NULL,NULL,0));assert(!vx_adapter_run_begin());
    check(0,0,0);vx_adapter_run_end();vx_adapter_request_end();

    stage("A2","weightA",2,3);
    /* NULL captures B now; activating A2 before run_begin cannot replace it. */
    assert(!vx_adapter_request_begin(NULL));assert(!vx_adapter_activate("A2"));
    assert(!vx_adapter_run_begin());check(1,1,5);
    assert(vx_adapter_run_has_target("weightB") && !vx_adapter_run_has_target("weightA"));
    vx_adapter_run_end();vx_adapter_request_end();
    assert(!vx_adapter_request_begin("B"));assert(!vx_adapter_run_begin());check(1,1,5);
    assert(!vx_adapter_activate(NULL));assert(!vx_adapter_remove("B"));check(1,1,5);
    vx_adapter_run_end();vx_adapter_request_end();

    /* A nonempty unrelated adapter must keep the ordinary complete scan. */
    assert(!vx_adapter_activate("unrelated"));assert(!vx_adapter_run_begin());check(1,0,5);
    vx_adapter_run_end();assert(!vx_adapter_activate(NULL));
    assert(vx_adapter_request_begin("missing")==-1);
    assert(!vx_adapter_run_begin());check(0,0,0);vx_adapter_run_end();

    stage("B2","weightB",3,4);
    const char* mixed[]={NULL,"A2","B2",NULL};const float scales[]={1,.5f,2,1};
    assert(!vx_adapter_request_begin_many(mixed,scales,4));assert(!vx_adapter_run_begin());check(1,1,4);
    assert(!vx_adapter_remove("A2") && !vx_adapter_remove("B2"));
    assert(vx_adapter_run_has_target("weightA") && vx_adapter_run_has_target("weightB"));
    VxAdapterLinearSegment segment[4];
    assert(vx_adapter_run_linear_segments("weightA",4,0,4,4,1,1,segment,4)==1);
    assert(segment[0].row_start==1 && segment[0].row_count==1 && segment[0].scale==.5f);
    assert(vx_adapter_run_linear_segments("weightB",4,0,4,4,1,1,segment,4)==1);
    assert(segment[0].row_start==2 && segment[0].row_count==1 && segment[0].scale==2);
    float x[]={2,3,4,5},ya[]={10,10,10,10},yb[]={10,10,10,10};
    assert(vx_adapter_apply_linear("weightA",x,NULL,VX_DTYPE_F32,NULL,ya,4,1,1,0,4,0,4)==1);
    assert(vx_adapter_apply_linear("weightB",x,NULL,VX_DTYPE_F32,NULL,yb,4,1,1,0,4,0,4)==1);
    assert(ya[0]==10 && ya[1]==19 && ya[2]==10 && ya[3]==10);
    assert(yb[0]==10 && yb[1]==10 && yb[2]==106 && yb[3]==10);
    /* Context-local TLS ownership must not leak a captured route. */
    sibling.nodes=nodes;sibling.node_count=3;g_current_engine_state=&sibling;
    assert(!vx_adapter_run_begin());check(0,0,0);vx_adapter_run_end();
    g_current_engine_state=&state;check(1,1,4);
    assert(!vx_adapter_run_begin());check(1,1,4);vx_adapter_run_end();
    vx_adapter_run_end();vx_adapter_request_end();check(0,0,0);
    vx_adapter_reset();state.adapter_registry_state_destroy(state.adapter_registry_state);
    g_current_engine_state=&sibling;
    vx_adapter_reset();sibling.adapter_registry_state_destroy(sibling.adapter_registry_state);
    g_current_engine_state=NULL;
    printf("PASS actual adapter TLS: %u scheduler/reference checks; mixed-route numerical outputs exact\n",checks);
}
'''


class AdapterRunScanTest(unittest.TestCase):
    def test_actual_captured_adapter_routes_and_scheduler_scan(self):
        compiler = shutil.which(os.environ.get("CC", "cc"))
        if not compiler:
            self.skipTest("C compiler unavailable")
        scheduler = function_source((RUNTIME / "incremental_runtime.c").read_text(), "run_has_adapter_targets")
        guard = "    if (!vx_adapter_run_has_versions()) return 0;\n"
        self.assertEqual(scheduler.count(guard), 1)
        reference = scheduler.replace(guard, "").replace("run_has_adapter_targets(", "run_has_adapter_targets_reference(")
        unit_text = FIXTURE.replace("SCHEDULER", scheduler).replace("REFERENCE", reference)
        includes = ["native/include", "native/src", "native/src/runtime", "native/src/kernels",
                    "native/src/backends", "native/third_party", "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-adapter-run-scan-") as temporary:
            unit = Path(temporary) / "test.c"
            unit.write_text(unit_text)
            for training in (0, 1):
                binary = Path(temporary) / f"test-{training}"
                command = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                           "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections", "-pthread",
                           f"-DVOLVOXAI_ENABLE_TRAINING={training}"]
                command += [f"-DVOLVOXAI_ENABLE_{backend}=0" for backend in ("CUDA", "VULKAN", "OPENGL", "METAL", "WEBGPU")]
                command += [f"-I{ROOT / path}" for path in includes]
                command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
                command += [str(unit), str(ROOT / "native/third_party/cJSON.c"), "-lm", "-o", str(binary)]
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([str(binary)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS actual adapter TLS:", result.stdout)
                print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
