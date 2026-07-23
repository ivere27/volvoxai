from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.generate_kernel_registry import (
    ROOT,
    Variant,
    load_registry,
    render_c,
    render_c_full,
    render_python,
    render_python_full,
    render_ts,
    render_ts_full,
)


REGISTRY_PROTO = ROOT / "proto/kernel_registry.proto"
PUBLIC_PROTO = ROOT / "proto/volvoxai.proto"


class KernelRegistryGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(REGISTRY_PROTO, PUBLIC_PROTO)

    def test_canonical_inventory_separates_runtime_and_exporter_support(self):
        by_id = {backend.runtime_id: backend for backend in self.registry.backends}
        self.assertEqual(len(by_id["cpu-js"].runtime_operators), 92)
        self.assertEqual(len(by_id["wasm"].runtime_operators), 92)
        self.assertEqual(len(by_id["webgpu"].runtime_operators), 87)
        self.assertEqual(len(by_id["webnn"].runtime_operators), 16)
        self.assertEqual(len(by_id["native-cpu"].runtime_operators), 77)
        self.assertEqual(len(by_id["nnapi"].runtime_operators), 3)
        self.assertEqual(len(by_id["cpu-js"].qualified_operators), 64)
        self.assertEqual(len(by_id["native-cpu"].qualified_operators), 62)
        self.assertEqual(len(by_id["webnn"].qualified_operators), 8)
        self.assertEqual(by_id["native-cpu"].support_mode, "RUNTIME_SUPPORT_MODE_DIRECT")
        unsupported_native = {
            "OPERATOR_KIND_CROSS_ATTENTION",
            "OPERATOR_KIND_CONV_1D",
            "OPERATOR_KIND_CONV_TRANSPOSE_2D",
            "OPERATOR_KIND_AVERAGE_POOL_2D",
            "OPERATOR_KIND_INTERPOLATE_1D",
            "OPERATOR_KIND_MASK",
            "OPERATOR_KIND_PAD",
            "OPERATOR_KIND_GATHER_ELEMENTS",
            "OPERATOR_KIND_BROADCAST",
            "OPERATOR_KIND_CONCAT2",
            "OPERATOR_KIND_NON_MAX_SUPPRESSION",
            "OPERATOR_KIND_SPATIAL_SOFTARGMAX_Y",
            "OPERATOR_KIND_MEAN_HEIGHT",
            "OPERATOR_KIND_PROFILE_X",
            "OPERATOR_KIND_PROFILE_Y",
        }
        self.assertEqual(
            set(by_id["cpu-js"].runtime_operators) - set(by_id["native-cpu"].runtime_operators),
            unsupported_native,
        )
        for backend_id in ("vulkan", "opengl", "metal", "cuda"):
            self.assertEqual(by_id[backend_id].qualified_operators, ())
            self.assertEqual(by_id[backend_id].support_mode, "RUNTIME_SUPPORT_MODE_DYNAMIC")

    def test_wasm_has_an_explicit_route_for_every_runtime_operator(self):
        wasm = next(backend for backend in self.registry.backends if backend.runtime_id == "wasm")
        self.assertFalse(wasm.default_route)
        self.assertEqual(
            {route.operator for route in wasm.routes},
            set(wasm.runtime_operators),
        )
        routes = {
            self.registry.graph_names[route.operator]: route.route
            for route in wasm.routes
        }
        self.assertEqual(routes["QLinear"], "qlinear")
        self.assertEqual(routes["LogSoftmax"], "softmax")

    def test_full_variants_do_not_leak_into_inference_outputs(self):
        full = Variant(
            "native-cpu.test.backward",
            "BACKEND_KIND_NATIVE_CPU",
            ("OPERATOR_KIND_LINEAR",),
            "KERNEL_PROFILE_FULL",
            "KERNEL_PHASE_BACKWARD",
            "vx_test_training_backward",
            "test.training-predicate",
            (),
            1,
        )
        registry = dataclasses.replace(
            self.registry, variants=self.registry.variants + (full,)
        )
        inference = render_python(registry) + render_ts(registry) + render_c(registry)
        full_outputs = (
            render_python_full(registry) + render_ts_full(registry) + render_c_full(registry)
        )
        self.assertNotIn(b"vx_test_training_backward", inference)
        self.assertIn(b"vx_test_training_backward", full_outputs)

    def test_rejects_duplicate_operator_inside_a_named_set(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        needle = (
            'name: "runtime-linear"\n'
            "    operator: OPERATOR_KIND_MATMUL\n"
        )
        mutated = source.replace(
            needle,
            needle + "    operator: OPERATOR_KIND_MATMUL\n",
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate operator"):
            self._load_mutated(mutated)

    def test_rejects_unknown_operator_runtime_registration(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated = source.replace(
            'name: "runtime-linear"\n    operator: OPERATOR_KIND_MATMUL',
            'name: "runtime-linear"\n    operator: OPERATOR_KIND_NOT_DECLARED',
            1,
        )
        with self.assertRaisesRegex(ValueError, "unknown operator"):
            self._load_mutated(mutated)

    def test_rejects_inference_training_phase(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated = source.replace(
            "phase: KERNEL_PHASE_FORWARD",
            "phase: KERNEL_PHASE_BACKWARD",
            1,
        )
        with self.assertRaisesRegex(ValueError, "non-forward phase"):
            self._load_mutated(mutated)

    def test_generated_files_are_current_and_protoc_accepts_schema(self):
        command = [sys.executable, str(ROOT / "tools/generate_kernel_registry.py"), "--check"]
        if shutil.which("protoc"):
            command.append("--protoc-check")
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def _load_mutated(self, source: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel_registry.proto"
            path.write_text(source, encoding="utf-8")
            return load_registry(path, PUBLIC_PROTO)


if __name__ == "__main__":
    unittest.main()
