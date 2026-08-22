#!/usr/bin/env python3
"""Reject process-global Metal graph/request/training state."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METAL = ROOT / "native/src/backends/metal_engine.m"
RUNTIME_HEADER = ROOT / "native/src/runtime/runtime_state.h"
RUNTIME_IMPL = ROOT / "native/src/runtime/runtime_state.c"
METAL_TEST = ROOT / "native/tests/test_metal_training.m"


def typedef_body(source: str, name: str) -> str:
    end = source.index(f"}} {name};")
    start = source.rfind("typedef struct {", 0, end)
    if start < 0:
        raise AssertionError(f"missing typedef struct for {name}")
    return source[start + len("typedef struct {") : end]


class MetalStateOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metal = METAL.read_text(encoding="utf-8")
        cls.runtime_header = RUNTIME_HEADER.read_text(encoding="utf-8")
        cls.runtime_impl = RUNTIME_IMPL.read_text(encoding="utf-8")
        cls.metal_test = METAL_TEST.read_text(encoding="utf-8")

    def test_metal_has_one_explicit_mutable_device_global(self) -> None:
        self.assertRegex(
            self.metal,
            r"static\s+MetalDeviceState\s+g_metal_device_state\s*=",
        )
        for name in (
            "device",
            "commandQueue",
            "graph_slots",
            "graph_slot_count",
            "graph_command",
            "graph_retained_bindings",
            "graph_forward_active",
            "graph_forward_error",
            "qgroupnorm_stats_buffer",
            "qlayernorm_stats_buffer",
            "qconv_zero_bias_backings",
            "training_active",
            "training_kernels",
            "training_slots",
            "training_command",
            "training_encoder",
            "training_transients",
        ):
            with self.subTest(name=name):
                self.assertNotRegex(
                    self.metal,
                    rf"(?m)^\s*static\s+(?!const\b)[^();\n]*\b{re.escape(name)}\b",
                )

    def test_metal_device_state_is_narrow_synchronized_and_refcounted(self) -> None:
        body = typedef_body(self.metal, "MetalDeviceState")
        for expected in (
            "pthread_mutex_t mutex",
            "reference_count",
            "id<MTLDevice> device",
            "id<MTLCommandQueue> command_queue",
            "MetalPipelineCacheEntry pipelines",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, body)
        for forbidden in (
            "MetalTensorSlot",
            "MetalTrainingTensorSlot",
            "graph_command",
            "scratch_buffer",
            "zero_bias",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)
        self.assertIn("pthread_mutex_lock(&g_metal_device_state.mutex)", self.metal)
        self.assertIn("pthread_mutex_unlock(&g_metal_device_state.mutex)", self.metal)

    def test_metal_context_owns_graph_command_scratch_and_training_state(self) -> None:
        body = typedef_body(self.metal, "MetalContextState")
        for expected in (
            "MetalTensorSlot graph_slot_storage",
            "graph_command_buffer",
            "graph_retained_binding_storage",
            "qgroupnorm_scratch_buffer",
            "qlayernorm_scratch_buffer",
            "qconv_zero_bias_storage",
            "shape_signature",
            "shape_generation",
            "capacity_generation",
            "MetalTrainingKernel training_kernel_storage",
            "MetalTrainingTensorSlot training_slot_storage",
            "training_command_buffer",
            "training_transient_storage",
            "device_acquired",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, body)
        self.assertIn("void* metal_context_state;", self.runtime_header)
        self.assertIn("metal_context_state_destroy", self.runtime_header)
        self.assertIn("state->metal_context_state_destroy", self.runtime_impl)

    def test_metal_lifecycle_test_covers_threads_isolation_and_survivor(self) -> None:
        for evidence in (
            "test_engine_state_isolation",
            "pthread_create",
            "shared_output",
            "survivor_output",
            "metal_cleanup();",
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, self.metal_test)

    def test_metal_dynamic_shape_rebind_is_exact_and_transactional(self) -> None:
        for evidence in (
            "metal_graph_bind_shape",
            "!strcmp(state->shape_signature, signature)",
            "candidate = (char*)malloc",
            "graph_flush_commands()",
            "graph_slot_ensure_capacity",
            "candidate = create_buffer",
            "test_dynamic_shape_capacity_lifecycle",
            "pooled_capacity_bytes",
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, self.metal + self.metal_test)

    def test_shader_override_path_remains_store_owned(self) -> None:
        self.assertIn("metal_set_shader_root", self.metal)
        self.assertIn("volvoxai_shader_store_set_override_root", self.metal)
        self.assertIn("volvoxai_shader_store_get", self.metal)


if __name__ == "__main__":
    unittest.main()
