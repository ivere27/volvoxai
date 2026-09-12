from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPOSITORY_ROOT / "tests" / "contracts" / "native_semantics_contract.c"
NATIVE_INCLUDE = REPOSITORY_ROOT / "native" / "include"
NATIVE_SOURCE = REPOSITORY_ROOT / "native" / "src"
SHAPE_CONTRACT = NATIVE_SOURCE / "runtime" / "shape_contract.c"
SHAPE_DOMAIN_CONTRACT = NATIVE_SOURCE / "runtime" / "shape_domain_contract.c"
KERNEL_REGISTRY = NATIVE_SOURCE / "generated" / "kernel_registry.c"


def _compiler_command() -> list[str] | None:
    configured = os.environ.get("CC")
    command = shlex.split(configured) if configured else ["cc"]
    if not command or shutil.which(command[0]) is None:
        return None
    return command


class NativeSemanticsContractTests(unittest.TestCase):
    def test_shape_rejection_does_not_poison_next_request(self) -> None:
        compiler = _compiler_command()
        if compiler is None:
            self.fail("a C compiler is required for the native semantics contract")

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "native_semantics_contract"
            subprocess.run(
                [
                    *compiler,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(NATIVE_INCLUDE),
                    "-I",
                    str(NATIVE_SOURCE),
                    str(FIXTURE),
                    str(SHAPE_CONTRACT),
                    str(SHAPE_DOMAIN_CONTRACT),
                    str(KERNEL_REGISTRY),
                    str(NATIVE_SOURCE / "generated" / "operator_vocabulary.c"),
                    "-lm",
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [str(executable)],
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
