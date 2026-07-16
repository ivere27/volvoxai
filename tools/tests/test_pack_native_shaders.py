import lzma
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import pack_native_shaders as packer  # noqa: E402


class NativeShaderPackTests(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[Path, Path]:
        sources = root / "shaders"
        compiled = root / "compiled"
        (sources / "inference").mkdir(parents=True)
        (sources / "training").mkdir()
        (sources / "inference" / "add.wgsl").write_text(
            "@compute @workgroup_size(1) fn main() {}\n", encoding="utf-8"
        )
        (sources / "training" / "addBackward.wgsl").write_text(
            "@compute @workgroup_size(1) fn input_main() {}\n", encoding="utf-8"
        )

        for backend in packer.BACKENDS:
            (compiled / backend.name).mkdir(parents=True)
        (compiled / "spv" / "add.spv").write_bytes(b"\x03\x02\x23\x07forward")
        (compiled / "spv" / "addBackward.spv").write_bytes(b"backward-spv")
        (compiled / "glsl" / "add.comp").write_bytes(b"abcde")
        (compiled / "glsl" / "addBackward_input_main.comp").write_bytes(
            b"backward-glsl"
        )
        (compiled / "gles" / "add.comp").write_bytes(b"gles-forward")
        (compiled / "metal" / "add.metal").write_bytes(b"metal-forward")
        return compiled, sources

    def test_blocks_are_aligned_crc32_xz_and_scope_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, sources = self.make_inputs(Path(temporary))
            pack = packer.build_pack(compiled, sources, "full")

            self.assertEqual(len(pack.blocks), 8)
            self.assertEqual(
                [(block.backend.name, block.scope) for block in pack.blocks],
                [
                    ("spv", "inference"),
                    ("spv", "training"),
                    ("glsl", "inference"),
                    ("glsl", "training"),
                    ("gles", "inference"),
                    ("gles", "training"),
                    ("metal", "inference"),
                    ("metal", "training"),
                ],
            )
            record_by_path = {record.path: record for record in pack.records}
            self.assertEqual(
                sorted(record_by_path),
                [
                    "gles/add.comp",
                    "glsl/add.comp",
                    "glsl/addBackward_input_main.comp",
                    "metal/add.metal",
                    "spv/add.spv",
                    "spv/addBackward.spv",
                ],
            )
            for block_index, block in enumerate(pack.blocks):
                self.assertEqual(block.compressed_data[:6], b"\xfd7zXZ\x00")
                self.assertEqual(block.compressed_data[7], lzma.CHECK_CRC32)
                # This also guards against accidentally restoring preset 9's
                # default 64 MiB dictionary in an embedded-runtime artifact.
                payload = lzma.decompress(
                    block.compressed_data,
                    memlimit=4 * packer.XZ_DICTIONARY_SIZE,
                )
                self.assertEqual(len(payload), block.uncompressed_size)
                self.assertEqual(len(payload) % packer.ALIGNMENT, 0)
                for record in pack.records:
                    if record.block != block_index:
                        continue
                    self.assertEqual(record.offset % packer.ALIGNMENT, 0)
                    source_path = compiled / record.path
                    self.assertEqual(
                        payload[record.offset : record.offset + record.size],
                        source_path.read_bytes(),
                    )

    def test_inference_pack_excludes_and_does_not_hash_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, sources = self.make_inputs(Path(temporary))
            first = packer.build_pack(compiled, sources, "inference")
            (sources / "training" / "addBackward.wgsl").write_text(
                "changed training source\n", encoding="utf-8"
            )
            (compiled / "spv" / "addBackward.spv").write_bytes(b"changed training binary")
            second = packer.build_pack(compiled, sources, "inference")

            self.assertEqual(len(first.blocks), 4)
            self.assertEqual(first, second)
            self.assertNotIn("addBackward", "\n".join(record.path for record in first.records))

    def test_browser_only_source_marker_is_excluded_from_native_pack_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, sources = self.make_inputs(Path(temporary))
            browser_shader = sources / "inference" / "packedDot.wgsl"
            browser_shader.write_bytes(
                packer.BROWSER_ONLY_MARKER
                + b"\nrequires packed_4x8_integer_dot_product;\n"
            )

            first = packer.build_pack(compiled, sources, "inference")
            browser_shader.write_bytes(
                packer.BROWSER_ONLY_MARKER
                + b"\nrequires packed_4x8_integer_dot_product;\n// changed browser source\n"
            )
            second = packer.build_pack(compiled, sources, "inference")

            self.assertEqual(first, second)
            discovered = packer.discover_source_shaders(sources)
            self.assertNotIn("packedDot", {source.stem for source in discovered})

    def test_native_spv_only_source_is_packed_only_for_vulkan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, sources = self.make_inputs(Path(temporary))
            source = sources / "inference" / "packedDot.wgsl"
            source.write_bytes(
                packer.NATIVE_SPV_ONLY_MARKER
                + b"\nrequires packed_4x8_integer_dot_product;\n"
            )
            (compiled / "spv" / "packedDot.spv").write_bytes(b"packed-dot-spv")

            native = packer.build_pack(
                compiled, sources, "inference",
                backend_names=("spv", "glsl", "gles", "metal"),
            )
            non_vulkan = packer.build_pack(
                compiled, sources, "inference",
                backend_names=("glsl", "gles", "metal"),
            )

            self.assertIn("spv/packedDot.spv", [record.path for record in native.records])
            self.assertNotIn(
                "packedDot", "\n".join(record.path for record in non_vulkan.records)
            )
            self.assertNotEqual(native.input_hash, non_vulkan.input_hash)

    def test_backend_filter_and_empty_pack_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiled, sources = self.make_inputs(root)
            linux = packer.build_pack(
                compiled, sources, "inference", backend_names=("spv", "glsl", "gles")
            )
            empty = packer.build_pack(
                compiled, sources, "inference", backend_names=()
            )

            self.assertEqual(
                [block.backend.name for block in linux.blocks],
                ["spv", "glsl", "gles"],
            )
            self.assertNotIn("metal/", "\n".join(record.path for record in linux.records))
            self.assertEqual(empty.blocks, ())
            self.assertEqual(empty.records, ())
            self.assertNotEqual(linux.input_hash, empty.input_hash)
            empty_c = packer.render_c(empty, "embedded_shaders.h")
            self.assertIn("volvoxai_embedded_shader_block_count = 0", empty_c)
            self.assertIn("volvoxai_embedded_shader_record_count = 0", empty_c)

    def test_flat_sources_require_explicit_training_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "shaders"
            compiled = root / "compiled"
            sources.mkdir()
            (sources / "forward.wgsl").write_bytes(b"forward source")
            (sources / "backward.wgsl").write_bytes(b"backward source")
            for backend in packer.BACKENDS:
                (compiled / backend.name).mkdir(parents=True)
            (compiled / "spv" / "forward.spv").write_bytes(b"forward")
            (compiled / "spv" / "backward_input_main.spv").write_bytes(b"backward")

            with self.assertRaisesRegex(packer.PackError, "--training-list"):
                packer.build_pack(compiled, sources, "inference")

            training_list = root / "training-shaders.txt"
            training_list.write_text("# Training overlay\nbackward.wgsl\n", encoding="utf-8")
            inference = packer.build_pack(
                compiled, sources, "inference", training_list=training_list
            )
            full = packer.build_pack(compiled, sources, "full", training_list=training_list)
            self.assertEqual(
                [record.path for record in inference.records], ["spv/forward.spv"]
            )
            self.assertEqual(
                [record.path for record in full.records],
                ["spv/backward_input_main.spv", "spv/forward.spv"],
            )

    def test_generated_files_are_deterministic_and_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiled, sources = self.make_inputs(root)
            pack = packer.build_pack(compiled, sources, "full")
            output_c = root / "generated" / "embedded_shaders.c"
            output_h = root / "generated" / "embedded_shaders.h"

            packer.write_pack(pack, output_c, output_h)
            first_c = output_c.read_bytes()
            first_h = output_h.read_bytes()
            packer.write_pack(pack, output_c, output_h)

            self.assertEqual(first_c, output_c.read_bytes())
            self.assertEqual(first_h, output_h.read_bytes())
            c_text = first_c.decode("utf-8")
            h_text = first_h.decode("utf-8")
            for text in (c_text, h_text):
                self.assertIn("DO NOT EDIT", text)
                self.assertIn(pack.input_hash, text)
            self.assertIn("volvoxai_embedded_shader_blocks", h_text)
            self.assertIn("volvoxai_embedded_shader_records", h_text)
            self.assertIn('"glsl/addBackward_input_main.comp"', c_text)


if __name__ == "__main__":
    unittest.main()
