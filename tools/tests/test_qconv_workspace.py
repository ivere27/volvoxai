"""Private QConv memory limits include live immutable packs and bounded patches."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class QConvWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-qconv-workspace-")
        cls.addClassCleanup(cls.directory.cleanup)
        folder = Path(cls.directory.name)
        proof = (ROOT / "native/src/runtime/public_api_bounded_domain_proof.inc").read_text()
        first = proof.index("static int vx_execution_qconv_packed_weight_bound(")
        last = proof.index("\n/* These portable kernels", first)
        source = r"""
#include <stdint.h>
#include <stddef.h>
#include "qconv_workspace.h"
typedef struct { void* packed_weight; uint32_t packed_weight_bytes; } QConv2DMetadata;
typedef struct { int node_count; QConv2DMetadata* qconv_metadata; } VxEngineState;
""" + proof[first:last] + r"""
uint64_t resident(const uint32_t* sizes, const uint8_t* live, int count) {
    QConv2DMetadata metadata[64] = {0};
    if (count < 0 || count > 64) return UINT64_MAX;
    for (int i=0; i<count; i++) {
        metadata[i].packed_weight = live[i] ? (void*)&metadata[i] : NULL;
        metadata[i].packed_weight_bytes = sizes[i];
    }
    VxEngineState state = {count, metadata};
    uint64_t bytes;
    return vx_execution_qconv_packed_weight_bound(&state, &bytes) ? bytes : UINT64_MAX;
}
size_t patches(uint64_t rows, uint64_t columns) {
    return vx_qconv_im2col_allocation_bytes(rows, columns);
}
"""
        (folder / "workspace.c").write_text(source)
        library = folder / "workspace.so"
        subprocess.run([compiler, "-std=c11", "-O2", "-shared", "-fPIC",
                        "-Wall", "-Wextra", "-Werror",
                        "-I" + str(ROOT / "native/src/kernels"),
                        str(folder / "workspace.c"), "-o", str(library)], check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.resident.argtypes = [ctypes.POINTER(ctypes.c_uint32),
                                     ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        cls.lib.resident.restype = ctypes.c_uint64
        cls.lib.patches.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        cls.lib.patches.restype = ctypes.c_size_t

    def resident(self, sizes, live):
        return self.lib.resident((ctypes.c_uint32 * len(sizes))(*sizes),
                                 (ctypes.c_uint8 * len(live))(*live), len(sizes))

    def test_simultaneous_packs_use_a_sum_and_64_bit_accounting(self):
        self.assertEqual(self.resident([361008, 394288, 74544], [1, 1, 1]), 829840)
        self.assertEqual(self.resident([0xFFFFFFFF] * 4, [1] * 4), 17179869180)
        self.assertEqual(self.resident([], []), 0)

    def test_release_and_failed_allocation_do_not_charge_absent_storage(self):
        self.assertEqual(self.resident([361008, 394288], [1, 0]), 361008)
        self.assertEqual(self.resident([0, 0], [0, 0]), 0)
        self.assertEqual(self.resident([0], [1]), 0xFFFFFFFFFFFFFFFF)

    def test_patch_cap_extent_and_overflow_boundaries(self):
        self.assertEqual(self.lib.patches(7921, 11 * 128), 11152768)
        self.assertEqual(self.lib.patches(1024, 131072), 128 * 1024 * 1024)
        for rows, columns in [(1025, 131072), (0, 1), (1, 0),
                              (2**64 - 1, 2**64 - 1), (1, 2**32), (2**32, 1)]:
            self.assertEqual(self.lib.patches(rows, columns), 0)


if __name__ == "__main__":
    unittest.main()
