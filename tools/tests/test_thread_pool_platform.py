"""Non-Linux POSIX preprocessing must not require Linux affinity APIs.

This deliberately uses host pthreads and headers with Linux-only names poisoned.
It checks the fallback compile/ownership contract, not macOS SDK execution.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#undef __linux__
#undef CPU_ZERO
#undef CPU_ISSET
#undef CPU_SETSIZE
#pragma GCC poison cpu_set_t CPU_ZERO CPU_ISSET CPU_SETSIZE sched_getaffinity
#include "thread_pool.c"
static int visits[17];
static void work(void* context, int begin, int end) {
    (void)context;
    for (int i = begin; i < end; i++) visits[i]++;
}
int main(void) {
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(0);
    if (!pool) return 1;
    VxKernelThreadPoolScope scope = vx_kernel_thread_pool_scope_enter(pool);
    int count = vx_kernels_thread_count();
#if defined(__ARM_NEON)
    if (count < 1 || count > 2) return 2;
#else
    if (count < 1 || count > 4) return 2;
#endif
    vx_kernels_parallel_for(17, 3, work, NULL);
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    for (int i = 0; i < 17; i++) if (visits[i] != 1) return 3;
    return vx_kernels_thread_count() != 1;
}
'''


class ThreadPoolPlatformTests(unittest.TestCase):
    def test_non_linux_posix_auto_fallback(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        for arm in (False, True):
            with self.subTest(arm_branch=arm), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "nonlinux.c"
                source.write_text(SOURCE)
                binary = Path(directory) / "nonlinux"
                subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                "-pthread", "-I" + str(ROOT / "native/src/kernels"),
                                *( ["-D__ARM_NEON=1"] if arm else []),
                                str(source), "-o", str(binary)], check=True)
                result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15,
                                        env={"VOLVOXAI_THREADS": "0"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
