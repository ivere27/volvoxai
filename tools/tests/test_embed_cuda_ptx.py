import re
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import embed_cuda_ptx as embedder  # noqa: E402


class CudaPtxEmbeddingTests(unittest.TestCase):
    def test_generated_files_are_deterministic_and_nul_terminated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ptx_path = root / "kernel.ptx"
            output_c = root / "generated" / "embedded_cuda_ptx.c"
            output_h = root / "generated" / "embedded_cuda_ptx.h"
            ptx = b".version 7.5\n.visible .entry add() { ret; }\n"
            ptx_path.write_bytes(ptx)

            embedder.write_embedded_ptx(ptx_path, output_c, output_h)
            first_c = output_c.read_bytes()
            first_h = output_h.read_bytes()
            embedder.write_embedded_ptx(ptx_path, output_c, output_h)

            self.assertEqual(first_c, output_c.read_bytes())
            self.assertEqual(first_h, output_h.read_bytes())
            digest = embedder.input_hash(ptx)
            for generated in (first_c, first_h):
                self.assertIn(b"DO NOT EDIT", generated)
                self.assertIn(digest.encode("ascii"), generated)

            source = first_c.decode("utf-8")
            encoded = bytes(
                int(value, 16)
                for value in re.findall(r"0x([0-9a-f]{2})", source)
            )
            self.assertEqual(encoded, ptx + b"\x00")
            self.assertIn(
                "const size_t volvoxai_cuda_ptx_size = "
                "sizeof(volvoxai_cuda_ptx) - 1u;",
                source,
            )

            header = first_h.decode("utf-8")
            self.assertIn("#include <stddef.h>", header)
            self.assertIn(
                "extern const unsigned char volvoxai_cuda_ptx[];", header
            )
            self.assertIn(
                "extern const size_t volvoxai_cuda_ptx_size;", header
            )

    def test_changed_input_changes_hash_and_generated_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ptx_path = root / "kernel.ptx"
            output_c = root / "embedded_cuda_ptx.c"
            output_h = root / "embedded_cuda_ptx.h"
            ptx_path.write_bytes(b"first\n")
            embedder.write_embedded_ptx(ptx_path, output_c, output_h)
            first = output_c.read_bytes()

            ptx_path.write_bytes(b"second\n")
            embedder.write_embedded_ptx(ptx_path, output_c, output_h)
            self.assertNotEqual(first, output_c.read_bytes())

    def test_custom_symbol_keeps_modules_linkable_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ptx_path = root / "training.ptx"
            output_c = root / "embedded_cuda_training_ptx.c"
            output_h = root / "embedded_cuda_training_ptx.h"
            ptx_path.write_bytes(
                b".version 7.5\n.visible .entry training_probe() { ret; }\n"
            )

            embedder.write_embedded_ptx(
                ptx_path,
                output_c,
                output_h,
                symbol="volvoxai_cuda_training_ptx",
            )

            source = output_c.read_text(encoding="utf-8")
            header = output_h.read_text(encoding="utf-8")
            self.assertIn(
                "const unsigned char volvoxai_cuda_training_ptx[]", source
            )
            self.assertIn(
                "const size_t volvoxai_cuda_training_ptx_size = "
                "sizeof(volvoxai_cuda_training_ptx) - 1u;",
                source,
            )
            self.assertNotIn("volvoxai_cuda_ptx[]", source)
            self.assertIn(
                "extern const unsigned char volvoxai_cuda_training_ptx[];",
                header,
            )

    def test_invalid_custom_symbol_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            embedder.render_header("0" * 64, "not-a-c-symbol")

    def test_empty_or_preterminated_ptx_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ptx_path = root / "kernel.ptx"
            output_c = root / "embedded_cuda_ptx.c"
            output_h = root / "embedded_cuda_ptx.h"
            for invalid in (b"", b"ptx\x00trailing"):
                with self.subTest(ptx=invalid):
                    ptx_path.write_bytes(invalid)
                    with self.assertRaises(ValueError):
                        embedder.write_embedded_ptx(ptx_path, output_c, output_h)


if __name__ == "__main__":
    unittest.main()
