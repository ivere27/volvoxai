from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from exporter.generated import kernel_registry  # noqa: E402


CUDA_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+\.inc)"', re.MULTILINE)
CUDA_DEFINITION = re.compile(
    r'extern\s+"C"\s+VX_CUDA_GLOBAL\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
)
CUDA_HOST_FUNCTION = re.compile(
    r"VOLVOXAI_CUDA_FORWARD_FUNCTION\(\s*"
    r"(?P<requirement>REQUIRED|DEFERRED)\s*,\s*"
    r"[A-Za-z_][A-Za-z0-9_]*\s*,\s*"
    r'"(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)"\s*\)',
    re.MULTILINE,
)


def _cuda_composition(root: Path) -> tuple[Path, ...]:
    visited: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        try:
            resolved.relative_to(REPOSITORY_ROOT / "native/src/backends")
        except ValueError as error:
            raise AssertionError(f"CUDA include escapes backend root: {resolved}") from error
        if not resolved.is_file():
            raise AssertionError(f"missing CUDA composition input: {resolved}")
        visited.add(resolved)
        ordered.append(resolved)
        source = resolved.read_text(encoding="utf-8")
        for include in CUDA_INCLUDE.finditer(source):
            visit(resolved.parent / include.group(1))

    visit(root)
    return tuple(ordered)


class NativeGpuRegistryAssetTests(unittest.TestCase):
    def test_native_shader_variants_have_source_and_engine_descriptors(self) -> None:
        engine_sources = {
            "vulkan": (
                REPOSITORY_ROOT / "native/src/backends/vulkan_engine.c",
                "spv",
                "spv",
            ),
            "opengl": (
                REPOSITORY_ROOT / "native/src/backends/opengl_engine.c",
                "OGL_SHADER_DIR",
                "comp",
            ),
            "metal": (
                REPOSITORY_ROOT / "native/src/backends/metal_engine.m",
                "METAL_SHADER_DIR",
                "metal",
            ),
        }
        variants_by_backend = {
            backend: [
                variant
                for variant in kernel_registry.KERNEL_VARIANTS
                if variant["backend"] == backend
            ]
            for backend in engine_sources
        }

        for backend, variants in variants_by_backend.items():
            self.assertTrue(variants, f"registry has no {backend} physical variants")
            engine_path, path_prefix, suffix = engine_sources[backend]
            engine = engine_path.read_text(encoding="utf-8")
            for variant in variants:
                entrypoint = variant["entrypoint_id"]
                with self.subTest(backend=backend, variant=variant["id"]):
                    source = (
                        REPOSITORY_ROOT
                        / "shaders/inference"
                        / f"{entrypoint}.wgsl"
                    )
                    self.assertTrue(
                        source.is_file(),
                        f"{variant['id']} has no authoritative WGSL source",
                    )
                    if backend == "vulkan":
                        descriptor = re.compile(
                            r'\{\s*"'
                            + re.escape(entrypoint)
                            + r'"\s*,\s*"'
                            + re.escape(path_prefix)
                            + r"/"
                            + re.escape(entrypoint)
                            + r"\."
                            + suffix
                            + r'"'
                        )
                    else:
                        descriptor = re.compile(
                            r'\{\s*"'
                            + re.escape(entrypoint)
                            + r'"\s*,\s*'
                            + re.escape(path_prefix)
                            + r'\s*"/'
                            + re.escape(entrypoint)
                            + r"\."
                            + suffix
                            + r'"'
                        )
                    self.assertRegex(
                        engine,
                        descriptor,
                        f"{variant['id']} is not connected to a production descriptor",
                    )
                    for operator in variant["operators"]:
                        self.assertIn(
                            operator,
                            kernel_registry.RUNTIME_OPERATORS_BY_BACKEND[backend],
                        )

    def test_cuda_variants_are_defined_and_required_by_the_host_loader(self) -> None:
        cuda_root = REPOSITORY_ROOT / "native/src/backends/cuda_kernels.cu"
        composition = _cuda_composition(cuda_root)
        device_source = "\n".join(
            path.read_text(encoding="utf-8") for path in composition
        )
        definitions = set(CUDA_DEFINITION.findall(device_source))

        host_registry = (
            REPOSITORY_ROOT
            / "native/src/backends/cuda/host/cuda_forward_function_registry_host.inc"
        ).read_text(encoding="utf-8")
        host_functions = {
            match.group("symbol"): match.group("requirement")
            for match in CUDA_HOST_FUNCTION.finditer(host_registry)
        }
        variants = [
            variant
            for variant in kernel_registry.KERNEL_VARIANTS
            if variant["backend"] == "cuda"
        ]
        self.assertTrue(variants, "registry has no CUDA physical variants")
        for variant in variants:
            entrypoint = variant["entrypoint_id"]
            with self.subTest(variant=variant["id"]):
                self.assertIn(entrypoint, definitions)
                self.assertEqual(host_functions.get(entrypoint), "REQUIRED")
                for operator in variant["operators"]:
                    self.assertIn(
                        operator,
                        kernel_registry.RUNTIME_OPERATORS_BY_BACKEND["cuda"],
                    )


if __name__ == "__main__":
    unittest.main()
