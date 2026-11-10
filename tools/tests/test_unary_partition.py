"""Elementwise results must be independent of chunks, adjacent lanes and tails."""
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "native/src/kernels"


class UnaryPartitionTests(unittest.TestCase):
    def test_production_dispatch_partition_and_oracle(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        configurations = [("baseline", ["-ffp-contract=off"])]
        if platform.machine().lower() in ("x86_64", "amd64"):
            # This is the same compiler feature check used by production trig.
            with tempfile.TemporaryDirectory() as directory:
                probe = Path(directory) / "features.c"
                probe.write_text('int main(void) {return !(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"));}')
                binary = Path(directory) / "features"
                subprocess.run([compiler, str(probe), "-o", str(binary)], check=True)
                if subprocess.run([str(binary)]).returncode == 0:
                    configurations.append(("fma-contract", ["-mavx2", "-mfma", "-ffp-contract=fast"]))
        for name, flags in configurations:
            with self.subTest(configuration=name), tempfile.TemporaryDirectory() as directory:
                binary = Path(directory) / "partition"
                subprocess.run([compiler, "-std=c11", "-O3", *flags, "-pthread",
                                "-I" + str(KERNELS),
                                str(Path(__file__).with_name("unary_partition_probe.c")),
                                str(KERNELS / "sequence_ops.c"), str(KERNELS / "cpu_features.c"),
                                str(KERNELS / "thread_pool.c"), "-lm", "-o", str(binary)], check=True)
                result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("unary partition/oracle passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
