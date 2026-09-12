from __future__ import annotations

import lzma
import re
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import pack_native_shaders as packer  # noqa: E402


class NativeGpuShaderAssetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        source_root = root / "shaders"
        compiled_root = root / "compiled"
        (source_root / "inference").mkdir(parents=True)
        (source_root / "training").mkdir()
        (source_root / "inference" / "forward.wgsl").write_text(
            "@compute @workgroup_size(1) fn main() {}\n", encoding="utf-8"
        )
        (source_root / "training" / "backward.wgsl").write_text(
            "@compute @workgroup_size(1) fn input_main() {}\n",
            encoding="utf-8",
        )
        (source_root / "inference" / "browserOnly.wgsl").write_bytes(
            packer.BROWSER_ONLY_MARKER
            + b"\nrequires packed_4x8_integer_dot_product;\n"
        )
        (source_root / "inference" / "nativeDot.wgsl").write_bytes(
            packer.NATIVE_SPV_ONLY_MARKER
            + b"\nrequires packed_4x8_integer_dot_product;\n"
        )

        payloads = {
            "spv": (".spv", b"spv"),
            "glsl": (".comp", b"glsl"),
            "gles": (".comp", b"gles"),
            "metal": (".metal", b"metal"),
        }
        for backend, (suffix, payload) in payloads.items():
            directory = compiled_root / backend
            directory.mkdir(parents=True)
            (directory / f"forward{suffix}").write_bytes(payload + b"-forward")
            (directory / f"backward_input_main{suffix}").write_bytes(
                payload + b"-backward"
            )
        (compiled_root / "spv" / "nativeDot.spv").write_bytes(
            b"spv-native-only"
        )
        return compiled_root, source_root

    def test_profiles_backend_filter_and_source_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, sources = self._fixture(Path(temporary))
            inference = packer.build_pack(compiled, sources, "inference")
            full = packer.build_pack(compiled, sources, "full")
            filtered = packer.build_pack(
                compiled,
                sources,
                "inference",
                backend_names=("glsl", "metal"),
            )

            inference_paths = {record.path for record in inference.records}
            full_paths = {record.path for record in full.records}
            filtered_paths = {record.path for record in filtered.records}
            self.assertFalse(any("backward" in path for path in inference_paths))
            self.assertTrue(any("backward" in path for path in full_paths))
            self.assertIn("spv/nativeDot.spv", inference_paths)
            self.assertFalse(any("nativeDot" in path for path in filtered_paths))
            self.assertFalse(any("browserOnly" in path for path in full_paths))
            self.assertEqual(
                {block.backend.name for block in filtered.blocks},
                {"glsl", "metal"},
            )

            # Browser-only inputs are outside every native pack identity.
            before = inference
            (sources / "inference" / "browserOnly.wgsl").write_bytes(
                packer.BROWSER_ONLY_MARKER + b"\n// changed browser source\n"
            )
            self.assertEqual(
                before, packer.build_pack(compiled, sources, "inference")
            )

            # SPIR-V-only inputs affect a Vulkan-capable pack, but are outside
            # a non-Vulkan backend filter and its hash.
            (sources / "inference" / "nativeDot.wgsl").write_bytes(
                packer.NATIVE_SPV_ONLY_MARKER + b"\n// changed SPIR-V source\n"
            )
            self.assertNotEqual(
                before, packer.build_pack(compiled, sources, "inference")
            )
            self.assertEqual(
                filtered,
                packer.build_pack(
                    compiled,
                    sources,
                    "inference",
                    backend_names=("glsl", "metal"),
                ),
            )

            # Training inputs are outside the inference pack identity.
            before = packer.build_pack(compiled, sources, "inference")
            (sources / "training" / "backward.wgsl").write_text(
                "changed training source\n", encoding="utf-8"
            )
            (compiled / "spv" / "backward_input_main.spv").write_bytes(
                b"changed training artifact"
            )
            self.assertEqual(
                before, packer.build_pack(compiled, sources, "inference")
            )
            self.assertNotEqual(
                full, packer.build_pack(compiled, sources, "full")
            )

    def test_pack_is_deterministic_aligned_bounded_xz_and_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiled, sources = self._fixture(root)
            first = packer.build_pack(compiled, sources, "full")
            second = packer.build_pack(compiled, sources, "full")
            self.assertEqual(first, second)
            self.assertRegex(first.input_hash, r"^[0-9a-f]{64}$")
            self.assertEqual(
                [record.path for record in first.records],
                sorted(record.path for record in first.records),
            )

            for block_index, block in enumerate(first.blocks):
                self.assertTrue(block.compressed_data.startswith(b"\xfd7zXZ\x00"))
                self.assertEqual(block.compressed_data[7], lzma.CHECK_CRC32)
                payload = lzma.decompress(
                    block.compressed_data,
                    memlimit=4 * packer.XZ_DICTIONARY_SIZE,
                )
                self.assertEqual(len(payload), block.uncompressed_size)
                self.assertEqual(len(payload) % packer.ALIGNMENT, 0)
                for record in first.records:
                    if record.block != block_index:
                        continue
                    self.assertEqual(record.offset % packer.ALIGNMENT, 0)
                    self.assertEqual(
                        payload[record.offset : record.offset + record.size],
                        (compiled / record.path).read_bytes(),
                    )

            output_c = root / "generated" / "embedded_shaders.c"
            output_h = root / "generated" / "embedded_shaders.h"
            packer.write_pack(first, output_c, output_h)
            c_before = output_c.read_bytes()
            h_before = output_h.read_bytes()
            packer.write_pack(first, output_c, output_h)
            self.assertEqual(c_before, output_c.read_bytes())
            self.assertEqual(h_before, output_h.read_bytes())
            for generated in (c_before, h_before):
                self.assertIn(b"DO NOT EDIT", generated)
                self.assertIn(first.input_hash.encode("ascii"), generated)
            self.assertIn(
                f"UINT32_C({packer.XZ_DICTIONARY_SIZE})".encode("ascii"),
                h_before,
            )

    def test_cpu_only_pack_is_explicit_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, sources = self._fixture(Path(temporary))
            first = packer.build_pack(
                compiled, sources, "inference", backend_names=()
            )
            second = packer.build_pack(
                compiled, sources, "inference", backend_names=()
            )
            self.assertEqual(first, second)
            self.assertEqual(first.blocks, ())
            self.assertEqual(first.records, ())
            generated = packer.render_c(first, "embedded_shaders.h")
            self.assertIn("volvoxai_embedded_shader_block_count = 0", generated)
            self.assertIn("volvoxai_embedded_shader_record_count = 0", generated)

    def test_cmake_keeps_inference_and_full_shader_roots_disjoint(self) -> None:
        composition = (REPOSITORY_ROOT / "native/CMakeLists.txt").read_text(
            encoding="utf-8"
        )

        inference_start = composition.index("set(INFERENCE_COMPILED_SHADER_DIR ")
        inference_end = composition.index(
            "add_custom_target(volvoxai_compile_inference_shaders",
            inference_start,
        )
        inference_rule = composition[inference_start:inference_end]
        self.assertIn("VOLVOXAI_SHADER_PROFILE=inference", inference_rule)
        self.assertIn("DEPENDS ${WGSL_INFERENCE_SOURCES}", inference_rule)
        self.assertNotIn("WGSL_TRAINING_SOURCES", inference_rule)
        self.assertNotIn("${REPO}/native/shaders", inference_rule)

        pack_start = composition.index("function(volvox_shader_pack ")
        pack_end = composition.index("endfunction()", pack_start)
        pack_rule = composition[pack_start:pack_end]
        inference_branch = pack_rule[
            pack_rule.index('if(pack_profile STREQUAL "inference")') :
            pack_rule.index('elseif(pack_profile STREQUAL "full")')
        ]
        full_branch = pack_rule[
            pack_rule.index('elseif(pack_profile STREQUAL "full")') :
            pack_rule.index("else()")
        ]
        self.assertIn("${REPO}/shaders/inference", inference_branch)
        self.assertIn("--source-scope inference", inference_branch)
        self.assertNotIn("shaders/training", inference_branch)
        self.assertIn("${REPO}/native/shaders", full_branch)
        self.assertRegex(
            composition,
            re.compile(
                r"set\(INFERENCE_SHADER_SRCS\s+\$\{SHADER_STORE_SRCS\} "
                r"\$\{EMB_INFERENCE\}\)"
            ),
        )
        self.assertRegex(
            composition,
            re.compile(
                r"set\(FULL_SHADER_SRCS\s+\$\{SHADER_STORE_SRCS\}\s+"
                r"\$\{EMB_FULL\}\)"
            ),
        )
        self.assertIn("add_dependencies(${target} ${pack_target})", composition)
        self.assertRegex(
            composition,
            re.compile(
                r"volvox_add_release_object_shards\(\s*"
                r"volvoxai_inference_release_shader inference\s+"
                r"INFERENCE_SHADER_SRCS INFERENCE_RELEASE_HOT_SRCS "
                r"INFERENCE_RELEASE_COLD_SRCS\s+"
                r"volvoxai_pack_inference_shaders "
                r"INFERENCE_RELEASE_SHADER_SHARDS\)",
            ),
        )
        self.assertRegex(
            composition,
            re.compile(
                r"volvox_add_release_object_shards\(\s*"
                r"volvoxai_full_release_shader full\s+"
                r"FULL_SHADER_SRCS FULL_RELEASE_HOT_SRCS "
                r"FULL_RELEASE_COLD_SRCS\s+"
                r"volvoxai_pack_full_shaders FULL_RELEASE_SHADER_SHARDS\)",
            ),
        )

        filter_start = composition.index('set(PACK_BACKENDS "")')
        filter_end = composition.index("file(GLOB WGSL_INFERENCE_SOURCES", filter_start)
        backend_filter = composition[filter_start:filter_end]
        self.assertIn("VOLVOXAI_ENABLE_VULKAN", backend_filter)
        self.assertIn("--backend spv", backend_filter)
        self.assertIn("VOLVOXAI_ENABLE_OPENGL", backend_filter)
        self.assertIn("--backend glsl --backend gles", backend_filter)
        self.assertIn("VOLVOXAI_ENABLE_METAL", backend_filter)
        self.assertIn("--backend metal", backend_filter)
        self.assertIn("--backend none", backend_filter)
        self.assertNotIn("VOLVOXAI_ENABLE_CUDA", backend_filter)


if __name__ == "__main__":
    unittest.main()
