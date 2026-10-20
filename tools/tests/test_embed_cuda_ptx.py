"""The PTX packer must remain compatible with the production bounded decoder."""
import ctypes
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tools.embed_cuda_ptx import write_embedded_ptx
from tools.tests import function_source


ROOT = Path(__file__).resolve().parents[2]


class EmbeddedCudaPtxTests(unittest.TestCase):
    def test_production_decoder_preserves_module_and_rejects_invalid_streams(self):
        compiler = shutil.which("cc")
        self.assertIsNotNone(compiler, "a C compiler is required for decoder validation")
        decoder = function_source((ROOT / "native/src/shader_store.c").read_text(),
                                  "volvoxai_shader_decode_xz")
        preamble = r'''
#include <pthread.h>
#include "native/src/shader_store.h"
#include "native/third_party/xz-embedded/xz.h"
#define VOLVOXAI_SHADER_STORE_MAX_BLOCK_SIZE ((size_t)64 * 1024 * 1024)
static pthread_once_t g_crc32_once = PTHREAD_ONCE_INIT;
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "decoder.c"
            source.write_text(preamble + decoder)
            library = root / "decoder.so"
            xz = ROOT / "native/third_party/xz-embedded"
            subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-std=c11",
                            "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(ROOT),
                            str(source), *(str(xz / name) for name in
                                ("xz_crc32.c", "xz_dec_lzma2.c", "xz_dec_stream.c")),
                            "-o", str(library)], check=True, capture_output=True)
            loaded = ctypes.CDLL(str(library))
            decode = loaded.volvoxai_shader_decode_xz
            decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_void_p, ctypes.c_size_t]
            decode.restype = ctypes.c_int
            entry = b".version 7.0\n.target sm_75\n.address_size 64\n.visible .entry probe() { ret; }\n"
            comments = b"".join(f"// symbol_{i:08x}: %{i % 4096}, {i * 2654435761 % 2**32}\n".encode()
                                for i in range(32768))
            for payload in (entry, entry + comments + comments[:65536]):
                with self.subTest(bytes=len(payload)):
                    ptx = root / "module.ptx"
                    ptx.write_bytes(payload)
                    c, h = root / "embedded.c", root / "embedded.h"
                    write_embedded_ptx(ptx, c, h)
                    body = c.read_text().split("volvoxai_cuda_ptx_xz[] = {", 1)[1].split("};", 1)[0]
                    encoded = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-f]{2})", body))
                    expected = payload + b"\0"
                    output = ctypes.create_string_buffer(b"\xA5" * (len(expected) + 32), len(expected) + 32)
                    target = ctypes.byref(output, 16)
                    original = ctypes.create_string_buffer(encoded, len(encoded))
                    self.assertEqual(decode(original, len(encoded), target, len(expected)), 0)
                    self.assertEqual(output.raw[16:-16], expected)
                    self.assertEqual(output.raw[:16] + output.raw[-16:], b"\xA5" * 32)
                    self.assertEqual(original.raw, encoded)
                    damaged = bytearray(encoded)
                    damaged[len(damaged) // 2] ^= 1
                    for invalid in (encoded[:-1], encoded + b"\0", bytes(damaged)):
                        memory = ctypes.create_string_buffer(invalid, len(invalid))
                        self.assertNotEqual(decode(memory, len(invalid), target, len(expected)), 0)
                    self.assertNotEqual(decode(original, len(encoded), target, len(expected) - 1), 0)


if __name__ == "__main__":
    unittest.main()
