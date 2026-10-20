"""Serial comparison retains independent bounded outputs and exact constants."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from tools import llm_cached_compare
from tools.exporter.hf_decoder import DecoderConfig, build_decoder_graph


class SerialCaptureOwnershipTests(unittest.TestCase):
    def test_trimmed_arrays_survive_producer_reuse_without_padded_storage(self):
        original = {"logits": np.arange(1 * 128 * 32, dtype=np.float32).reshape(1, 128, 32),
                    "kv": np.ones((1, 128, 8), dtype=np.float32)}
        retained = llm_cached_compare._retain_fresh_arrays(original, 3, {"logits": 32, "kv": 8})
        expected = original["logits"][:, :3].copy()
        self.assertEqual(sum(a.nbytes for a in retained.values()), 3 * (32 + 8) * 4)
        for name in original:
            self.assertFalse(np.shares_memory(retained[name], original[name]))
            self.assertIsNone(retained[name].base)
            original[name].fill(-1)
        np.testing.assert_array_equal(retained["logits"], expected)
        np.testing.assert_array_equal(retained["kv"], np.ones((1, 3, 8), dtype=np.float32))

    def test_capture_contract_refuses_missing_names_wrong_shapes_and_dtypes(self):
        for arrays, length in (({}, 2), ({"x": np.ones((1, 2, 3), dtype=np.float64)}, 2),
                               ({"x": np.ones((2, 2, 3), dtype=np.float32)}, 2),
                               ({"x": np.ones((1, 2, 4), dtype=np.float32)}, 2),
                               ({"x": np.ones((1, 2, 3), dtype=np.float32)}, 3),
                               ({"x": np.ones((1, 2, 3), dtype=np.float32)}, 0)):
            with self.subTest(length=length), self.assertRaises(ValueError):
                llm_cached_compare._retain_fresh_arrays(arrays, length, {"x": 3})

    def test_native_snapshot_inventory_detects_owner_file_retirement(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"TMPDIR": temporary}):
            path = Path(temporary) / "volvoxai-snapshot-fixture"
            self.assertEqual(llm_cached_compare._native_snapshot_inventory(), [])
            path.write_bytes(b"model")
            inventory = llm_cached_compare._native_snapshot_inventory()
            self.assertEqual([(row["name"], row["bytes"]) for row in inventory], [(path.name, 5)])
            path.unlink()
            self.assertEqual(llm_cached_compare._native_snapshot_inventory(), [])


class SharedScaledInitializerTests(unittest.TestCase):
    def fixture(self):
        config = {"model_type": "llama", "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5, "hidden_act": "silu", "tie_word_embeddings": False,
            "rope_theta": 500000.0, "rope_scaling": {"rope_type": "llama3", "factor": 32.0,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}}
        decoder = DecoderConfig.from_hf(config, max_context=16)
        generator = np.random.default_rng(1701)
        base = {name: generator.standard_normal(shape).astype(np.float32)
                for name, shape in decoder.weight_shapes().items()}
        _, weights = build_decoder_graph(decoder, base, compact_kv=True, capture_kv=True)
        policy = {"format": "volvox-hf-baseline/v1", "models": {"fixture": {"config": config}},
                  "reference": {"qualification": {"max_context": 16}}}
        return policy, base, weights

    def test_scaled_constants_are_checked_exactly_without_weight_copy(self):
        policy, _, weights = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);policy_path = root / "policy.json";source = root / "weights.safetensors"
            policy_path.write_text(json.dumps(policy));save_file(weights, str(source));before = source.read_bytes()
            with patch("builtins.print"):
                llm_cached_compare.prepare("fixture", root / "packages", policy_path=policy_path, weights_path=source)
            self.assertEqual(source.read_bytes(), before)
            for name in ("execution", "capture"):
                self.assertEqual((root / "packages" / name / "model.safetensors").resolve(), source)

    def test_missing_or_changed_scaled_constants_refused_before_graph_publication(self):
        policy, base, complete = self.fixture()
        changed = dict(complete);changed["hf.llama3.cos"] = complete["hf.llama3.cos"].copy()
        changed["hf.llama3.cos"][0, 0] += np.float32(.01)
        wrong_dtype = dict(complete);wrong_dtype["hf.llama3.cos"] = complete["hf.llama3.cos"].astype(np.float16)
        for weights in (base, changed, wrong_dtype):
            with self.subTest(keys=len(weights)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary);policy_path = root / "policy.json";source = root / "weights.safetensors"
                policy_path.write_text(json.dumps(policy));save_file(weights, str(source))
                with self.assertRaisesRegex(ValueError, "derived graph initializer"):
                    llm_cached_compare.prepare("fixture", root / "packages", policy_path=policy_path, weights_path=source)
                self.assertFalse((root / "packages" / "execution" / "graph.json").exists())
                self.assertFalse((root / "packages" / "preparation.json").exists())


class CachedNativeRouteTests(unittest.TestCase):
    def evidence(self, backend):
        return SimpleNamespace(status=1, backend=backend, device="NVIDIA GeForce RTX 3090",
            fallback=SimpleNamespace(operator_fallback_used=False),
            route=SimpleNamespace(provider=backend, attested=True, builtin=True,
                                  active_nodes=17, selected_nodes=17,
                                  fallback_nodes=0, missing_nodes=0, digest=42))

    def test_all_device_routes_require_complete_builtin_attestation(self):
        for backend in ("cuda", "vulkan", "opengl"):
            evidence = self.evidence(backend)
            record = llm_cached_compare._checked_native_gpu_route(evidence, backend, 1, "execute")
            self.assertEqual(record["backend"], backend)
            self.assertEqual(record["active_nodes"], record["selected_nodes"])
            for owner, field, bad in (("report", "status", 2), ("report", "backend", "cpu"),
                    ("route", "provider", "cpu"), ("route", "attested", False),
                    ("route", "builtin", False), ("route", "active_nodes", 0),
                    ("route", "selected_nodes", 16), ("route", "fallback_nodes", 1),
                    ("route", "missing_nodes", 1), ("fallback", "operator_fallback_used", True)):
                value = copy.deepcopy(evidence)
                setattr(value if owner == "report" else getattr(value, owner), field, bad)
                with self.subTest(backend=backend, fault=field), self.assertRaises(ValueError):
                    llm_cached_compare._checked_native_gpu_route(value, backend, 1, "execute")

    def test_portable_provider_is_exact_and_cuda_cpu_contracts_are_preserved(self):
        for backend in ("vulkan", "opengl"):
            value = self.evidence(backend); value.route.provider = "external." + backend
            with self.assertRaises(ValueError):
                llm_cached_compare._checked_native_gpu_route(value, backend, 1, "compile")
        cuda = self.evidence("cuda"); cuda.route.provider = "builtin.cuda"
        self.assertEqual(llm_cached_compare._checked_native_gpu_route(cuda, "cuda", 1, "compile")["provider"], "builtin.cuda")
        self.assertIsNone(llm_cached_compare._checked_native_gpu_route(None, "cpu", 1, "cpu"))


if __name__ == "__main__":
    unittest.main()
