#!/usr/bin/env python3
"""Verify the CUDA PTX build's strict and fast FP32 contracts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake") or "cmake"
C_COMPILER = shutil.which("cc") or "cc"


class CudaFp32BuildContractTests(unittest.TestCase):
    def _write_fake_cuda_compiler(self, path: Path, log_path: Path) -> None:
        path.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json
                import sys
                from pathlib import Path

                args = sys.argv[1:]
                if "--print-targets" in args:
                    print("nvptx64 - NVIDIA PTX 64-bit")
                    raise SystemExit(0)
                Path({str(log_path)!r}).write_text(json.dumps(args), encoding="utf-8")
                output = Path(args[args.index("-o") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    ".version 7.5\\n.visible .entry contract_probe() {{ ret; }}\\n",
                    encoding="utf-8",
                )
                """
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)

    def _configure(
        self,
        build_dir: Path,
        compiler_kind: str,
        fake_compiler: Path,
        fast_fp32: bool,
    ) -> str:
        command = [
            CMAKE,
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            f"-DCMAKE_C_COMPILER={C_COMPILER}",
            "-DVOLVOXAI_ENABLE_VULKAN=OFF",
            "-DVOLVOXAI_ENABLE_OPENGL=OFF",
            "-DVOLVOXAI_ENABLE_CUDA=ON",
            "-DVOLVOXAI_ENABLE_METAL=OFF",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ]
        if fast_fp32:
            command.append("-DVOLVOXAI_CUDA_FAST_FP32=ON")
        if compiler_kind == "nvcc":
            command.append(f"-DVOLVOXAI_NVCC_EXECUTABLE={fake_compiler}")
        else:
            command.extend(
                (
                    "-DVOLVOXAI_NVCC_EXECUTABLE=OFF",
                    f"-DVOLVOXAI_CUDA_CLANGXX_EXECUTABLE={fake_compiler}",
                )
            )
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def _assert_training_embedding_is_full_profile_only(
        self, build_dir: Path
    ) -> None:
        entries = json.loads(
            (build_dir / "compile_commands.json").read_text(encoding="utf-8")
        )

        def commands_for(filename: str) -> list[str]:
            commands = []
            for entry in entries:
                if Path(entry["file"]).name != filename:
                    continue
                command = entry.get("command")
                if command is None:
                    command = " ".join(entry.get("arguments", []))
                commands.append(command)
            return commands

        forward_commands = commands_for("embedded_cuda_ptx.c")
        training_commands = commands_for("embedded_cuda_training_ptx.c")
        self.assertTrue(
            any("CMakeFiles/volvoxai.dir" in command for command in forward_commands),
            forward_commands,
        )
        self.assertTrue(
            any(
                "CMakeFiles/volvoxai-full.dir" in command
                for command in forward_commands
            ),
            forward_commands,
        )
        self.assertTrue(training_commands)
        self.assertTrue(
            any(
                "CMakeFiles/volvoxai-full.dir" in command
                for command in training_commands
            ),
            training_commands,
        )
        self.assertTrue(
            all("VOLVOXAI_ENABLE_TRAINING=1" in command for command in training_commands),
            training_commands,
        )
        self.assertFalse(
            any("CMakeFiles/volvoxai.dir" in command for command in training_commands),
            training_commands,
        )

    def _build_codegen(
        self, build_dir: Path, target: str = "volvoxai_cuda_codegen"
    ) -> str:
        result = subprocess.run(
            [CMAKE, "--build", str(build_dir), "--target", target],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def _assert_compiler_contract(
        self, compiler_kind: str, strict_flag: str, fast_flag: str
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=f"volvoxai-cuda-{compiler_kind}-contract-"
        ) as temporary:
            root = Path(temporary)
            build_dir = root / "build"
            compiler_log = root / "compiler-args.json"
            fake_compiler = root / compiler_kind
            self._write_fake_cuda_compiler(fake_compiler, compiler_log)

            configure_output = self._configure(
                build_dir, compiler_kind, fake_compiler, fast_fp32=False
            )
            self.assertIn("CUDA FP32 contract: strict-no-fma", configure_output)
            self._assert_training_embedding_is_full_profile_only(build_dir)
            self._build_codegen(build_dir)
            strict_args = json.loads(compiler_log.read_text(encoding="utf-8"))
            self.assertIn(strict_flag, strict_args)
            self.assertNotIn(fast_flag, strict_args)
            strict_output = Path(strict_args[strict_args.index("-o") + 1])
            self.assertIn("strict-no-fma", strict_output.parts)
            self.assertTrue(strict_output.is_file())
            self.assertFalse(
                strict_output.with_name("cuda_training_kernels.ptx").exists()
            )
            self.assertFalse(
                strict_output.with_name("embedded_cuda_training_ptx.c").exists()
            )

            self._build_codegen(build_dir, "volvoxai_cuda_training_codegen")
            strict_training_args = json.loads(
                compiler_log.read_text(encoding="utf-8")
            )
            self.assertIn(strict_flag, strict_training_args)
            self.assertNotIn(fast_flag, strict_training_args)
            self.assertIn(
                "native/src/backends/cuda_training_kernels.cu",
                strict_training_args,
            )
            strict_training_output = Path(
                strict_training_args[strict_training_args.index("-o") + 1]
            )
            self.assertIn("strict-no-fma", strict_training_output.parts)
            self.assertTrue(strict_training_output.is_file())
            self.assertNotEqual(strict_output, strict_training_output)

            configure_output = self._configure(
                build_dir, compiler_kind, fake_compiler, fast_fp32=True
            )
            self.assertIn("CUDA FP32 contract: fast-fma", configure_output)
            self._build_codegen(build_dir)
            fast_args = json.loads(compiler_log.read_text(encoding="utf-8"))
            self.assertIn(fast_flag, fast_args)
            self.assertNotIn(strict_flag, fast_args)
            fast_output = Path(fast_args[fast_args.index("-o") + 1])
            self.assertIn("fast-fma", fast_output.parts)
            self.assertTrue(fast_output.is_file())

            self._build_codegen(build_dir, "volvoxai_cuda_training_codegen")
            fast_training_args = json.loads(
                compiler_log.read_text(encoding="utf-8")
            )
            self.assertIn(fast_flag, fast_training_args)
            self.assertNotIn(strict_flag, fast_training_args)
            self.assertIn(
                "native/src/backends/cuda_training_kernels.cu",
                fast_training_args,
            )
            fast_training_output = Path(
                fast_training_args[fast_training_args.index("-o") + 1]
            )
            self.assertIn("fast-fma", fast_training_output.parts)
            self.assertTrue(fast_training_output.is_file())
            self.assertNotEqual(fast_output, fast_training_output)

            self.assertNotEqual(strict_output, fast_output)
            self.assertTrue(strict_output.is_file())
            self.assertNotEqual(strict_training_output, fast_training_output)
            self.assertTrue(strict_training_output.is_file())

    def test_nvcc_contract_defaults_to_strict_and_fast_is_explicit(self) -> None:
        self._assert_compiler_contract(
            "nvcc", strict_flag="--fmad=false", fast_flag="--fmad=true"
        )

    def test_clang_contract_defaults_to_strict_and_fast_is_explicit(self) -> None:
        self._assert_compiler_contract(
            "clang++",
            strict_flag="-ffp-contract=off",
            fast_flag="-ffp-contract=fast",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cmake")
    parser.add_argument("--c-compiler")
    options, unittest_args = parser.parse_known_args()
    if options.cmake:
        CMAKE = options.cmake
    if options.c_compiler:
        C_COMPILER = options.c_compiler
    unittest.main(argv=[sys.argv[0], *unittest_args])
