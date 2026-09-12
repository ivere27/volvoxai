from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class NativeCudaBuildContractTests(unittest.TestCase):
    def _tool(self, name: str) -> str:
        executable = shutil.which(name)
        if executable is None:
            self.fail(f"{name} is required for the CUDA build contract")
        return executable

    def _write_fake_nvcc(self, path: Path, log: Path) -> None:
        path.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json
                import sys
                from pathlib import Path

                arguments = sys.argv[1:]
                with Path({str(log)!r}).open("a", encoding="utf-8") as output_log:
                    output_log.write(json.dumps(arguments) + "\\n")
                output = Path(arguments[arguments.index("-o") + 1])
                source = next(
                    argument for argument in arguments if argument.endswith(".cu")
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    ".version 7.5\\n"
                    + ".visible .entry "
                    + Path(source).stem
                    + "_contract_probe() {{ ret; }}\\n",
                    encoding="utf-8",
                )
                """
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)

    def _configure(self, build: Path, fake_nvcc: Path, fast: bool) -> str:
        command = [
            self._tool("cmake"),
            "-S",
            str(REPOSITORY_ROOT),
            "-B",
            str(build),
            f"-DCMAKE_C_COMPILER={self._tool('cc')}",
            "-DVOLVOXAI_ENABLE_VULKAN=OFF",
            "-DVOLVOXAI_ENABLE_OPENGL=OFF",
            "-DVOLVOXAI_ENABLE_METAL=OFF",
            "-DVOLVOXAI_ENABLE_CUDA=ON",
            f"-DVOLVOXAI_NVCC_EXECUTABLE={fake_nvcc}",
        ]
        if fast:
            command.append("-DVOLVOXAI_CUDA_FAST_FP32=ON")
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def _build(self, build: Path, target: str) -> None:
        result = subprocess.run(
            [self._tool("cmake"), "--build", str(build), "--target", target],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"target {target}\n{result.stdout}{result.stderr}",
        )

    def _assert_embedded(
        self, ptx: Path, source: Path, header: Path, symbol: str
    ) -> None:
        payload = ptx.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        source_bytes = source.read_bytes()
        header_bytes = header.read_bytes()
        for generated in (source_bytes, header_bytes):
            self.assertIn(b"DO NOT EDIT", generated)
            self.assertIn(digest.encode("ascii"), generated)
        source_text = source_bytes.decode("utf-8")
        encoded = bytes(
            int(value, 16)
            for value in re.findall(r"0x([0-9a-f]{2})", source_text)
        )
        self.assertEqual(encoded, payload + b"\x00")
        self.assertIn(f"const unsigned char {symbol}[]", source_text)
        self.assertIn(f"extern const unsigned char {symbol}[]", header_bytes.decode())

    def test_cmake_enforces_fp_mode_and_profile_specific_ptx_embedding(self) -> None:
        for fast in (False, True):
            with self.subTest(fast=fast), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                build = root / "build"
                log = root / "nvcc.jsonl"
                fake_nvcc = root / "nvcc"
                self._write_fake_nvcc(fake_nvcc, log)

                configure_output = self._configure(build, fake_nvcc, fast)
                contract = "fast-fma" if fast else "strict-no-fma"
                expected_flag = "--fmad=true" if fast else "--fmad=false"
                self.assertIn(f"CUDA FP32 contract: {contract}", configure_output)

                generated = build / "gen/cuda" / contract
                self._build(build, "volvoxai_cuda_codegen")
                forward_ptx = generated / "cuda_kernels.ptx"
                forward_c = generated / "embedded_cuda_ptx.c"
                forward_h = generated / "embedded_cuda_ptx.h"
                self.assertTrue(forward_ptx.is_file())
                self.assertFalse((generated / "cuda_training_kernels.ptx").exists())
                self._assert_embedded(
                    forward_ptx,
                    forward_c,
                    forward_h,
                    "volvoxai_cuda_ptx",
                )

                self._build(build, "volvoxai_cuda_training_codegen")
                training_ptx = generated / "cuda_training_kernels.ptx"
                training_c = generated / "embedded_cuda_training_ptx.c"
                training_h = generated / "embedded_cuda_training_ptx.h"
                self._assert_embedded(
                    training_ptx,
                    training_c,
                    training_h,
                    "volvoxai_cuda_training_ptx",
                )
                self.assertNotIn(
                    "volvoxai_cuda_training_ptx", forward_c.read_text(encoding="utf-8")
                )

                invocations = [
                    json.loads(line)
                    for line in log.read_text(encoding="utf-8").splitlines()
                ]
                compile_invocations = [
                    arguments for arguments in invocations if "-o" in arguments
                ]
                self.assertEqual(len(compile_invocations), 2)
                for arguments in compile_invocations:
                    self.assertIn(expected_flag, arguments)
                    opposite = "--fmad=false" if fast else "--fmad=true"
                    self.assertNotIn(opposite, arguments)


if __name__ == "__main__":
    unittest.main()
