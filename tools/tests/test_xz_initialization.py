"""Metadata and shader decoding must share one process-wide CRC initializer."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.tests import function_source


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform.startswith("linux"),
                     "CRC call counting uses the ELF linker --wrap option")
class XzInitializationTests(unittest.TestCase):
    def test_metadata_and_shader_decoding_share_crc_initialization(self):
        compiler = shutil.which("cc")
        self.assertIsNotNone(compiler, "a C compiler is required for XZ validation")
        native = ROOT / "native"
        xz = native / "third_party/xz-embedded"
        decoder = function_source((native / "src/shader_store.c").read_text(),
                                  "volvoxai_shader_decode_xz")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shader_decoder = root / "shader_decoder.c"
            shader_decoder.write_text(
                '#include "shader_store.h"\n'
                '#include "vx_xz.h"\n'
                '#include "xz.h"\n'
                '#define VOLVOXAI_SHADER_STORE_MAX_BLOCK_SIZE ((size_t)64 * 1024 * 1024)\n'
                + decoder)
            # Cover both contract payloads and the threadless primitive used by
            # WASM. The concurrent native case calls both real decoding paths.
            for full, threadless in ((False, False), (True, False), (False, True)):
                with self.subTest(full=full, threadless=threadless):
                    generated = ROOT / "runtime/generated/c"
                    if not full:
                        generated /= "inference"
                    executable = root / f"decode-{int(full)}-{int(threadless)}"
                    includes = (native / "include", native / "src",
                                native / "src/runtime", native / "src/api",
                                native / "third_party/synurang/include", xz, generated)
                    command = [
                        compiler, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-D_POSIX_C_SOURCE=200809L",
                        f"-DVOLVOXAI_ENABLE_TRAINING={int(full)}",
                        *(["-DVOLVOXAI_NO_THREADS=1"] if threadless else ["-pthread"]),
                        "-ffunction-sections", "-fdata-sections",
                        *(flag for path in includes for flag in ("-I", str(path))),
                        str(ROOT / "tests/contracts/xz_initialization_contract.c"),
                        str(shader_decoder), str(native / "src/runtime/vx_xz.c"),
                        str(native / "src/api/vx_api_convert.c"),
                        str(generated / "volvoxai_lite.c"),
                        *(str(xz / name) for name in
                          ("xz_crc32.c", "xz_dec_lzma2.c", "xz_dec_stream.c")),
                        "-Wl,--gc-sections", "-Wl,--wrap=xz_crc32_init", "-lm",
                        "-o", str(executable),
                    ]
                    compiled = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    result = subprocess.run([str(executable)], capture_output=True,
                                            text=True, timeout=30)
                    self.assertEqual(result.returncode, 0,
                                     f"{result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
