"""Concrete and symbolic shape evaluators must agree on the concrete domain.

`shape_contract.c` evaluates concrete shapes and `shape_domain_contract.c`
proves symbolic domains. This gate compiles both against the operator shape
vector corpus and requires that, for every fully concrete vector, the
symbolic evaluator with all dimensions `FIXED(n)` produces the same status,
shape function id, dtypes and extents as the concrete evaluator.

A `domain_unsupported` result is recorded but not failed: the symbolic side may
explicitly refuse an input the concrete side accepts. A contradiction - one side
accepting what the other rejects, or different outputs - fails.
"""

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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.shape_equivalence_fixture import emit

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


class ShapeContractEquivalenceTests(unittest.TestCase):
    def test_symbolic_evaluator_matches_concrete_on_fixed_dimensions(self) -> None:
        compiler = _compiler_command()
        if compiler is None:
            self.fail("a C compiler is required for the shape equivalence gate")

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "shape_equivalence_generated.c"
            executable = Path(directory) / "shape_equivalence"
            case_count, skipped = emit(fixture)
            self.assertGreater(case_count, 0, "the vector corpus produced no cases")
            self.assertEqual(
                skipped, [], f"the emitter could not project {len(skipped)} vectors")

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
                    str(fixture),
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


if __name__ == "__main__":
    unittest.main()
