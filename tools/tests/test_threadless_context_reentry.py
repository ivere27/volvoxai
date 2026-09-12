from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPOSITORY_ROOT
    / "tests"
    / "contracts"
    / "threadless_context_reentry_wasm.c"
)


def _command_from_environment(
    variable: str, candidates: list[str]
) -> list[str] | None:
    configured = os.environ.get(variable)
    if configured:
        command = shlex.split(configured)
        return command if command and shutil.which(command[0]) else None
    for candidate in candidates:
        if shutil.which(candidate):
            return [candidate]
    return None


class ThreadlessGuardTests(unittest.TestCase):
    def test_context_reentry_and_request_wait_guards(self) -> None:
        compiler = _command_from_environment("WASM_CC", ["clang-17", "clang"])
        node = _command_from_environment("NODE", ["node"])
        if compiler is None:
            self.fail("clang with a wasm32 target is required")
        if node is None:
            self.fail("Node.js is required to execute the threadless contract")

        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory) / "threadless_context_reentry.wasm"
            subprocess.run(
                [
                    *compiler,
                    "--target=wasm32",
                    "-std=c11",
                    "-O2",
                    "-msimd128",
                    "-nostdlib",
                    "-ffreestanding",
                    "-ffunction-sections",
                    "-fdata-sections",
                    "-DVOLVOXAI_ENABLE_TRAINING=0",
                    "-DVOLVOXAI_ENABLE_VULKAN=0",
                    "-DVOLVOXAI_ENABLE_OPENGL=0",
                    "-DVOLVOXAI_ENABLE_CUDA=0",
                    "-DVOLVOXAI_ENABLE_METAL=0",
                    "-I",
                    str(
                        REPOSITORY_ROOT
                        / "native"
                        / "src"
                        / "runtime"
                        / "wasm_freestanding"
                        / "include"
                    ),
                    "-I",
                    str(REPOSITORY_ROOT / "native" / "include"),
                    "-I",
                    str(REPOSITORY_ROOT / "native" / "src"),
                    "-I",
                    str(REPOSITORY_ROOT / "native" / "src" / "runtime"),
                    "-I",
                    str(REPOSITORY_ROOT / "native" / "src" / "kernels"),
                    "-I",
                    str(REPOSITORY_ROOT / "native" / "src" / "backends"),
                    "-I",
                    str(REPOSITORY_ROOT / "native" / "third_party"),
                    "-I",
                    str(
                        REPOSITORY_ROOT
                        / "native"
                        / "third_party"
                        / "synurang"
                        / "include"
                    ),
                    "-I",
                    str(REPOSITORY_ROOT / "runtime" / "generated" / "c" / "inference"),
                    str(FIXTURE),
                    "-Wl,--no-entry",
                    "-Wl,--allow-undefined",
                    "-Wl,--gc-sections",
                    "-Wl,--export=volvoxai_test_threadless_context_reentry",
                    "-Wl,--export=volvoxai_test_threadless_request_wait",
                    "-o",
                    str(module),
                ],
                check=True,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            script = """
const fs = require('node:fs');
const bytes = fs.readFileSync(process.argv[1]);
const compiled = new WebAssembly.Module(bytes);
const imports = {};
for (const entry of WebAssembly.Module.imports(compiled)) {
  if (entry.kind !== 'function') {
    throw new Error(`unexpected ${entry.kind} import ${entry.module}.${entry.name}`);
  }
  (imports[entry.module] ??= {})[entry.name] = () => 0;
}
const instance = new WebAssembly.Instance(compiled, imports);
const status = instance.exports.volvoxai_test_threadless_context_reentry();
if (status !== 0) throw new Error(`threadless context contract failed: ${status}`);
const waitStatus = instance.exports.volvoxai_test_threadless_request_wait();
if (waitStatus !== 0) throw new Error(`threadless request-wait contract failed: ${waitStatus}`);
"""
            result = subprocess.run(
                [*node, "-e", script, str(module)],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
