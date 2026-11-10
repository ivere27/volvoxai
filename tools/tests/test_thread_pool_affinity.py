"""Affinity must count one allowed representative of each SMT sibling list."""
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest


class AffinityTopologyTests(unittest.TestCase):
    def test_partial_sibling_ranges(self):
        compiler = shutil.which("cc")
        if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64") or not compiler:
            self.skipTest("Linux x86 topology policy and C compiler required")
        source = r'''
#include "thread_pool.c"
int main(void) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed); CPU_SET(1, &allowed); CPU_SET(2, &allowed);
    if (vx_first_allowed_sibling("0-3\n", &allowed) != 1) return 1;
    CPU_ZERO(&allowed); CPU_SET(5, &allowed); CPU_SET(6, &allowed); CPU_SET(9, &allowed);
    if (vx_first_allowed_sibling("0,4-7,9\n", &allowed) != 5) return 2;
    if (vx_first_allowed_sibling("0,9\n", &allowed) != 9) return 3;
    if (vx_first_allowed_sibling("0-3\n", &allowed) != -1) return 4;
    CPU_ZERO(&allowed); CPU_SET(0, &allowed);
    if (vx_first_allowed_sibling("0\n", &allowed) != 0) return 5;
    if (vx_first_allowed_sibling("3-1\n", &allowed) != -1) return 6;
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "affinity.c"
            path.write_text(source)
            binary = path.with_suffix("")
            kernels = Path(__file__).resolve().parents[2] / "native/src/kernels"
            subprocess.run([compiler, "-std=c11", "-O2", "-pthread", "-I" + str(kernels),
                            str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=30)
