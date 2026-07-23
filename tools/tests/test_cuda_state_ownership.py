#!/usr/bin/env python3
"""Reject process-global CUDA graph, request, profiling, or training state."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CUDA_ROOT = ROOT / "native/src/backends/cuda_engine.c"
CUDA_FRAGMENTS = ROOT / "native/src/backends/cuda/host"
CUDA_QUANTIZE_W8 = CUDA_FRAGMENTS / "cuda_quantize_w8_host.inc"
RUNTIME_STATE = ROOT / "native/src/runtime/runtime_state.h"
RUNTIME_STATE_IMPL = ROOT / "native/src/runtime/runtime_state.c"

FORBIDDEN_STATIC_NAMES = (
    "graph_slots",
    "graph_slot_count",
    "graph_slot_hash",
    "graph_slot_hash_valid",
    "graph_slot_epoch",
    "cuda_replay",
    "qact_lut_cache",
    "cuda_profile",
    "cuda_profile_path_history",
    "cuda_profile_scope_serial",
    "cuda_forward_context_guard",
    "cuda_forward_scope_active",
    "cuda_forward_context_ready",
    "cuda_forward_context_failed",
    "cuda_training_context_guard",
    "cuda_training_active",
    "cuda_training_context_ready",
    "cuda_training_failed",
    "cuda_training_optimizer_mirrors",
    "cuda_training_attention_workspace",
    "cuda_training_basic_workspace",
    "cuda_lora_workspace",
    "cuda_launch_count",
    "cuda_graph_capture_count",
    "cuda_graph_replay_count",
)


class CudaStateOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sources = [CUDA_ROOT, *sorted(CUDA_FRAGMENTS.glob("*.inc"))]
        cls.source = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        cls.root_source = CUDA_ROOT.read_text(encoding="utf-8")
        cls.runtime_header = RUNTIME_STATE.read_text(encoding="utf-8")
        cls.runtime_impl = RUNTIME_STATE_IMPL.read_text(encoding="utf-8")
        cls.quantize_w8 = CUDA_QUANTIZE_W8.read_text(encoding="utf-8")

    def test_only_explicit_device_state_is_file_static(self) -> None:
        self.assertRegex(
            self.root_source,
            r"static\s+CudaDeviceState\s+cuda_device_state\s*=",
        )
        for name in FORBIDDEN_STATIC_NAMES:
            with self.subTest(name=name):
                self.assertNotRegex(
                    self.source,
                    rf"(?m)^\s*static\s+[^();\n]*\b{re.escape(name)}\b",
                )

    def test_device_state_is_synchronized_and_narrow(self) -> None:
        device_match = re.search(
            r"typedef\s+struct\s*\{(?P<body>.*?)\}\s+CudaDeviceState\s*;",
            self.root_source,
            re.DOTALL,
        )
        self.assertIsNotNone(device_match)
        body = device_match.group("body")
        self.assertIn("pthread_mutex_t mutex", body)
        self.assertIn("CUcontext context", body)
        self.assertIn("CUmodule module", body)
        self.assertIn("CUstream stream", body)
        for forbidden in (
            "CudaTensorSlot",
            "CudaReplayState",
            "CudaQactLutCache",
            "workspace",
            "optimizer",
            "profile",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)

    def test_engine_capsule_owns_mutable_cuda_state(self) -> None:
        capsule_match = re.search(
            r"typedef\s+struct\s*\{(?P<body>.*?)\}\s+CudaContextState\s*;",
            self.root_source,
            re.DOTALL,
        )
        self.assertIsNotNone(capsule_match)
        body = capsule_match.group("body")
        for field in (
            "graph_slots",
            "graph_slot_hash",
            "graph_slot_epoch",
            "CudaReplayState replay",
            "CudaQactLutCache qact_lut_cache",
            "lora_workspace",
            "training_optimizer_mirrors",
            "CudaProfileState profile",
        ):
            with self.subTest(field=field):
                self.assertIn(field, body)
        self.assertIn("void* cuda_context_state;", self.runtime_header)
        self.assertIn("cuda_context_state_destroy", self.runtime_header)
        self.assertIn("state->cuda_context_state_destroy", self.runtime_impl)

    def test_shared_stream_operations_take_device_lock(self) -> None:
        self.assertIn(
            "pthread_mutex_lock(&cuda_device_state.mutex)",
            self.source,
        )
        self.assertIn("owns_device_lock", self.root_source)
        self.assertIn("reference_count", self.root_source)

    def test_each_profile_capsule_maps_cached_module_functions(self) -> None:
        self.assertIn("cuda_profile_register_loaded_forward_functions();",
                      self.source)
        self.assertIn("cuda_profile_register_loaded_training_functions();",
                      self.source)

    def test_w8_authoring_does_not_recursively_take_device_lock(self) -> None:
        self.assertIn("cuda_context_enter_owned(&guard)", self.quantize_w8)
        self.assertIn("cuda_launch_in_owned_context(", self.quantize_w8)
        self.assertNotRegex(self.quantize_w8, r"\bcuda_launch\s*\(")


if __name__ == "__main__":
    unittest.main()
