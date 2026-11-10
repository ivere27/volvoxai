"""Check packed Conv affine rounding with contraction enabled in production code."""
from pathlib import Path
import json
import platform
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "native/src/kernels"


class PackedConvAffineTests(unittest.TestCase):
    def test_separate_f32_rounding_at_halfway(self):
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        flags = []
        if platform.machine().lower() in ("x86_64", "amd64"):
            cpuinfo = Path("/proc/cpuinfo")
            if not cpuinfo.exists() or " fma " not in cpuinfo.read_text():
                self.skipTest("FMA unavailable")
            flags = ["-mfma"]
        elif platform.machine().lower() not in ("aarch64", "arm64"):
            self.skipTest("FMA test requires x86 FMA or AArch64")
        with tempfile.TemporaryDirectory(prefix="volvox-qconv-affine-") as directory:
            source = Path(directory) / "halfway.c"
            source.write_text(
                '#define main all_qconv_tests\n'
                f'#include "{ROOT / "native/tests/test_qconv4.c"}"\n'
                '#undef main\n#undef CHECK\n'
                '#define main all_packed_tests\n'
                '#define test_affine_halfway test_packed_affine_halfway\n'
                f'#include "{ROOT / "native/tests/test_packed_quant_gemm.c"}"\n'
                '#undef main\n#undef test_affine_halfway\n'
                'int main(void) { test_affine_halfway(); return !test_packed_affine_halfway(); }\n')
            binary = Path(directory) / "halfway"
            subprocess.run([compiler, "-O3", "-ffp-contract=fast", *flags,
                            "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                            "-I" + str(KERNELS), "-I" + str(ROOT / "native/include"),
                            "-pthread", str(source),
                            str(KERNELS / "kernels.c"), str(KERNELS / "cpu_features.c"),
                            str(KERNELS / "quant_cpu_isa.c"), "-lm", "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_release_wasm_halfway(self):
        compiler = shutil.which("clang-17") or shutil.which("clang")
        node = shutil.which("node")
        if not compiler or not node:
            self.skipTest("WASM compiler and Node required")
        recipe = json.loads(subprocess.check_output([node, "--input-type=module", "-e",
            "import {wasmProfile} from './tools/release_profiles.mjs';"
            "console.log(JSON.stringify(wasmProfile('inference',process.cwd()).recipe.parent.objects.find(o=>o.id==='kernels')));"],
            cwd=ROOT, text=True))
        with tempfile.TemporaryDirectory(prefix="volvox-qconv-wasm-") as directory:
            module = Path(directory) / "halfway.wasm"
            subprocess.run([compiler, *recipe["flags"], recipe["optimization"], recipe["source"],
                            "-Wl,--no-entry", "-Wl,--export=qconv2d_packed_a8",
                            "-Wl,--export=qconv2d_packed_a8_workspace_bytes",
                            "-Wl,--export=alloc_bytes", "-Wl,--export=reset_heap",
                            "-o", str(module)], cwd=ROOT, check=True)
            script = r'''
const fs=require('fs');
WebAssembly.instantiate(fs.readFileSync(process.argv[1]),{}).then(({instance:{exports:e}})=>{
  for(const channels of [18,4097]) for(const bits of [2,3,4]) {
    e.reset_heap();const capacity=e.qconv2d_packed_a8_workspace_bytes();
    const xp=e.alloc_bytes(channels),wp=e.alloc_bytes(Math.ceil(channels*bits/8));
    const sp=e.alloc_bytes(4),yp=e.alloc_bytes(4),scratch=e.alloc_bytes(capacity);
    const x=new Uint8Array(e.memory.buffer,xp,channels);x.fill(0);x.fill(255,0,17);x[17]=40;
    const w=new Uint8Array(e.memory.buffer,wp,Math.ceil(channels*bits/8));w.fill(bits===3?0:bits===2?0xaa:0x88);
    const code=(1<<(bits-1))+1;
    for(let k=0;k<channels;k++)for(let bit=0;bit<bits;bit++) {
      const at=k*bits+bit;w[at>>3]=(w[at>>3]&~(1<<(at%8)))|(((code>>bit)&1)<<(at%8));
    }
    new Float32Array(e.memory.buffer,sp,1)[0]=.008;
    const ok=e.qconv2d_packed_a8(bits,xp,wp,0,sp,yp,1,1,1,channels,1,1,1,1,1,channels,
      1,1,1,1,0,0,0,0,1,0,.015,0,.03,-17,5,6,scratch,capacity);
    if(ok!==1||new Int8Array(e.memory.buffer,yp,1)[0]!==0)throw Error(`halfway K=${channels} bits=${bits}`);
  }
  console.log('six packed Conv affine ties passed');
});
'''
            result = subprocess.run([node, "-e", script, str(module)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
