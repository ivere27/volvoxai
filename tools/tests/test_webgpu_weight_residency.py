"""Immutable model classification excludes caller inputs and training storage."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class WebGPUWeightResidencyTests(unittest.TestCase):
    def test_exact_compiled_weight_storage_and_training_refusal(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler required")
        source = r'''
#include <assert.h>
#include <string.h>
#include "engine_internal.h"
static VxEngineState state;
VxEngineState* vx_engine_state_current(void) { return &state; }
int volvoxai_engine_tensor_is_model_weight_locked(const char* name) {
    return !strcmp(name,"stored") || !strcmp(name,"converted");
}
'''
        source += function_source((ROOT / "native/src/backends/webgpu_backend.c").read_text(),
                                  "vx_webgpu_immutable_weight_span")
        source += r'''
int main(void) {
    float values[32]={0};T tensors[3]={0};state.tensors=tensors;state.tensor_count=3;
    for(int i=0;i<3;++i) {
        tensors[i].data=values+i*8;tensors[i].numel=8;tensors[i].elem_size=4;
    }
    strcpy(tensors[0].name,"stored");strcpy(tensors[1].name,"converted");
    strcpy(tensors[2].name,"activation");
    assert(!vx_webgpu_immutable_weight_span(values,32));
    state.weight_files_borrowed=1;
    assert(vx_webgpu_immutable_weight_span(values,32));
    assert(vx_webgpu_immutable_weight_span(values,4));
    assert(vx_webgpu_immutable_weight_span(values+8,32));
    assert(!vx_webgpu_immutable_weight_span(values,36));
    assert(!vx_webgpu_immutable_weight_span(values+1,4));
    assert(!vx_webgpu_immutable_weight_span(values+16,32));
    tensors[0].is_graph_input=1;assert(!vx_webgpu_immutable_weight_span(values,32));
    tensors[0].is_graph_input=0;
    state.native_training_mode=1;assert(!vx_webgpu_immutable_weight_span(values,32));
    state.native_training_mode=0;
    tensors[0].data=values+24;
    assert(!vx_webgpu_immutable_weight_span(values,32));
    assert(vx_webgpu_immutable_weight_span(values+24,32));
    assert(!vx_webgpu_immutable_weight_span(NULL,32));
    assert(!vx_webgpu_immutable_weight_span(values,0));
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-weight-residency-") as directory:
            path = Path(directory) / "test.c";path.write_text(source)
            binary = path.with_suffix("")
            built = subprocess.run([compiler,"-std=c11","-O2","-Wall","-Wextra","-Werror",
                "-DVOLVOXAI_ENABLE_TRAINING=1",
                *["-I"+str(ROOT/p) for p in ("native/include","native/third_party","native/src",
                    "native/src/runtime","native/src/kernels","native/src/backends")],
                str(path),"-o",str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            run = subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)


if __name__ == "__main__": unittest.main()
