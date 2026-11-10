"""Production convolution dispatch requires every ISA used by its selected TU."""
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "native/src/kernels"


class ConvIsaDispatchTests(unittest.TestCase):
    def test_avx2_without_fma_retains_baseline_for_every_entrypoint(self):
        compiler = shutil.which("cc")
        if not compiler or platform.machine().lower() not in ("x86_64", "amd64", "i386", "i686"):
            self.skipTest("x86 C compiler required")
        entries = (KERNELS / "conv_f32_isa_entrypoints.h").read_text()
        declarations = re.findall(
            r"VX_CONV_F32_ENTRY(?:_VOID)?\(([^,]+),\s*(\w+),\s*\((.*?)\),\s*\((.*?)\)\)",
            entries, re.DOTALL)
        self.assertGreater(len(declarations), 1)
        calls = []
        for _, name, _, arguments in declarations:
            zeros = ",".join("0" for _ in arguments.split(",")) if arguments.strip() else ""
            calls.append(f"selected=-1; calls=0; (void){name}({zeros}); "
                         "assert(selected==(avx2&&fma)); assert(calls==1);")
        source = r'''
#include <assert.h>
#include <string.h>
/* Mock only hardware admission and terminal kernel bodies. The entire
 * production dispatcher and entrypoint inventory compile unchanged. */
#define VOLVOXAI_KERNEL_PLATFORM_H
typedef struct { int has_avx2; } VxKernelPlatform;
static VxKernelPlatform mock_platform;
static int fma, selected, calls;
static const VxKernelPlatform* vx_kernel_platform(void) { return &mock_platform; }
static int mock_cpu_supports(const char* feature) {
    assert(!strcmp(feature,"fma")); return fma;
}
#define __builtin_cpu_supports(feature) mock_cpu_supports(feature)
#include "conv_f32_dispatch.c"
#define VX_CONV_F32_ENTRY(ret,name,params,args) \
    ret name##_baseline params { selected=0; calls++; return (ret)0; } \
    ret name##_avx2 params { selected=1; calls++; return (ret)0; }
#define VX_CONV_F32_ENTRY_VOID(ret,name,params,args) \
    ret name##_baseline params { selected=0; calls++; } \
    ret name##_avx2 params { selected=1; calls++; }
#include "conv_f32_isa_entrypoints.h"
int main(void) {
    for(int avx2=0;avx2<=1;avx2++) for(fma=0;fma<=1;fma++) {
        mock_platform.has_avx2=avx2;
''' + "\n".join(calls) + r'''
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-conv-isa-dispatch-") as temporary:
            root = Path(temporary)
            path, binary = root / "dispatch.c", root / "dispatch"
            path.write_text(source)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-Wno-unused-parameter", "-I" + str(KERNELS), str(path), "-o", str(binary)],
                           check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
