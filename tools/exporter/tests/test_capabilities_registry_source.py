from __future__ import annotations

import unittest

from tools.exporter import capabilities
from tools.exporter.generated import kernel_registry


def identity_graph() -> dict[str, object]:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 4], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {"out": {
                "tensor": "y", "shape": [1, 4], "dtype": "float32",
            }},
            "params": {},
        }],
    }


class GeneratedCapabilitySourceTests(unittest.TestCase):
    def test_capability_tables_are_the_generated_registry_projections(self):
        self.assertIs(capabilities.ATOMIC_TARGETS, kernel_registry.ATOMIC_TARGETS)
        self.assertIs(capabilities.TARGETS, kernel_registry.TARGETS)
        self.assertIs(capabilities._PROFILE_MEMBERS, kernel_registry.PROFILE_MEMBERS)
        self.assertIs(capabilities._OPS_BY_TARGET, kernel_registry.OPS_BY_TARGET)

    def test_profiles_expand_from_generated_members_in_registry_order(self):
        for profile, members in kernel_registry.PROFILE_MEMBERS.items():
            with self.subTest(profile=profile):
                self.assertEqual(capabilities.expand_targets([profile]), members)

        self.assertEqual(
            capabilities.expand_targets(["browser", "native-cpu", "browser"]),
            (*kernel_registry.PROFILE_MEMBERS["browser"], "native-cpu"),
        )

    def test_validation_uses_generated_strict_target_qualification(self):
        graph = identity_graph()
        for target, admitted in kernel_registry.OPS_BY_TARGET.items():
            with self.subTest(target=target):
                result = capabilities.validate_graph(graph, [target])
                capability_codes = {
                    diagnostic.code
                    for diagnostic in result.diagnostics
                    if diagnostic.stage == "capability"
                }
                if "Identity" in admitted:
                    self.assertNotIn("VXCAP001", capability_codes)
                else:
                    self.assertIn("VXCAP001", capability_codes)

        native_gpu_targets = (
            "backend:vulkan",
            "backend:opengl",
            "backend:metal",
            "backend:cuda",
        )
        expected = kernel_registry.OPS_BY_TARGET["backend:cuda"]
        self.assertTrue(expected)
        self.assertNotIn("Identity", expected)
        for target in native_gpu_targets:
            with self.subTest(native_gpu_target=target):
                self.assertEqual(kernel_registry.OPS_BY_TARGET[target], expected)
                result = capabilities.validate_graph(graph, [target])
                capability_codes = {
                    diagnostic.code
                    for diagnostic in result.diagnostics
                    if diagnostic.stage == "capability"
                }
                self.assertIn("VXCAP001", capability_codes)


if __name__ == "__main__":
    unittest.main()
