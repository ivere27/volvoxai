from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKER = REPOSITORY_ROOT / "tools" / "pack_native_shaders.py"
FIXTURE = REPOSITORY_ROOT / "tests" / "contracts" / "shader_override_contract.c"
NATIVE_SOURCE = REPOSITORY_ROOT / "native" / "src"
XZ_SOURCE = REPOSITORY_ROOT / "native" / "third_party" / "xz-embedded"
LOG_PREFIX = "[VolvoxAI] Using shader override: VOLVOXAI_SHADER_DIR="


def _compiler_command() -> list[str] | None:
    configured = os.environ.get("CC")
    command = shlex.split(configured) if configured else ["cc"]
    if not command or shutil.which(command[0]) is None:
        return None
    return command


class ShaderOverrideContractTests(unittest.TestCase):
    def test_external_override_is_used_and_logged_once(self) -> None:
        compiler = _compiler_command()
        if compiler is None:
            self.fail("a C compiler is required for the native shader-store contract")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            compiled_root = root / "compiled"
            missing_root = root / "missing"
            override_root = root / "override"
            (compiled_root / "spv").mkdir(parents=True)
            (override_root / "spv").mkdir(parents=True)
            source_root.mkdir()
            missing_root.mkdir()

            (source_root / "test.wgsl").write_text(
                "@compute @workgroup_size(1) fn main() {}\n",
                encoding="utf-8",
            )
            (compiled_root / "spv" / "test.spv").write_bytes(
                b"embedded shader fixture"
            )
            (override_root / "spv" / "test.spv").write_bytes(
                b"external shader fixture"
            )

            generated_c = root / "embedded_shaders.c"
            generated_h = root / "embedded_shaders.h"
            subprocess.run(
                [
                    sys.executable,
                    str(PACKER),
                    "--compiled-dir",
                    str(compiled_root),
                    "--shader-source-dir",
                    str(source_root),
                    "--profile",
                    "inference",
                    "--output-c",
                    str(generated_c),
                    "--output-h",
                    str(generated_h),
                    "--backend",
                    "spv",
                    "--source-scope",
                    "inference",
                ],
                check=True,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )

            executable = root / "shader_override_contract"
            subprocess.run(
                [
                    *compiler,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pthread",
                    "-I",
                    str(root),
                    "-I",
                    str(NATIVE_SOURCE),
                    "-I",
                    str(XZ_SOURCE),
                    str(FIXTURE),
                    str(NATIVE_SOURCE / "shader_store.c"),
                    str(generated_c),
                    str(XZ_SOURCE / "xz_crc32.c"),
                    str(XZ_SOURCE / "xz_dec_lzma2.c"),
                    str(XZ_SOURCE / "xz_dec_stream.c"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [str(executable), str(missing_root), str(override_root)],
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
            self.assertEqual(result.stderr.count(LOG_PREFIX), 1)
            self.assertIn(LOG_PREFIX + str(override_root), result.stderr)
            self.assertNotIn(str(missing_root), result.stderr)


if __name__ == "__main__":
    unittest.main()
