"""Offline checks for additional pinned checkpoints and policy provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from tools import llm_cached_compare, llm_reference
from tools.exporter.hf_decoder import DecoderConfig
from tools.llm_baseline import measure_native_baseline as measurement


class AlternatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.alias = "tinyllama-1.1b-chat"
        self.policy = llm_reference.read_policy(model=self.alias)
        self.policy["models"] = {self.alias: self.policy["models"][self.alias]}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.policy_path = Path(temporary.name) / "selected-policy.json"
        self.policy_path.write_text(json.dumps(self.policy))

    def test_model_reference_overrides_keep_scaled_rope_source_and_do_not_leak(self):
        scaled = llm_reference.read_policy(model="llama3.2-1b-gsm8k")
        original = llm_reference.read_policy(model="smollm2-135m")
        rope_source = "transformers/modeling_rope_utils.py"
        self.assertIn(rope_source, scaled["reference"]["transformers_source_sha256"])
        self.assertNotIn(rope_source, original["reference"]["transformers_source_sha256"])
        for field in ("logits", "layer_kv", "greedy_ids", "independent_operator_fixture"):
            self.assertEqual(scaled["reference"]["engine_tolerance"][field],
                             original["reference"]["engine_tolerance"][field])
        self.assertIn("F16", scaled["reference"]["engine_tolerance"]["rationale"])
        self.assertEqual(scaled["models"]["llama3.2-1b-gsm8k"]["config"]["rope_scaling"]["rope_type"], "llama3")

    def test_unknown_model_reference_override_is_refused(self):
        self.policy["models"][self.alias]["reference_overrides"] = {"unknown_budget": 1}
        self.policy_path.write_text(json.dumps(self.policy))
        with self.assertRaisesRegex(ValueError, "reference override"):
            llm_reference.read_policy(self.policy_path, model=self.alias)

    def test_measurement_workloads_keep_distinct_graphs_selection_and_profiles(self):
        cached = measurement.read_policy()
        generated = measurement.read_policy(workload="full_prefix_generate")
        self.assertEqual(cached["cached_decode_steps"], 4)
        self.assertEqual(cached["prefix_lengths"], [1, 16, 31])
        self.assertIn({"profile": "inference", "backend": "cpu"}, cached["cells"])
        self.assertNotIn({"profile": "inference", "backend": "cpu"}, generated["cells"])
        self.assertEqual(generated["generation_output_byte_budget"], 16)
        self.assertEqual([v["name"] for v in generated["selection_variants"]],
                         ["greedy", "random-default", "random-filtered"])
        for model in cached["models"]:
            self.assertEqual(cached["models"][model]["weights_sha256"], generated["models"][model]["weights_sha256"])
            for field in ("execution_graph_sha256", "reference_manifest_sha256"):
                self.assertNotEqual(cached["models"][model][field], generated["models"][model][field])
        with self.assertRaisesRegex(RuntimeError, "unplanned workload"):
            measurement.read_policy(workload="unknown")

    def test_measurement_worker_refuses_pins_for_another_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private"
            private.mkdir()
            pins = root / "pins.json"
            pins.write_text(json.dumps({"format": "volvox-native-llm-baseline-pins/v1",
                "model": "smollm2-135m", "profile": "full", "backend": "cpu",
                "workload": "cached_execute"}))
            with patch.dict("os.environ", {}), patch("sys.argv", ["measure_native_baseline.py",
                    "--model", "smollm2-135m", "--profile", "full", "--backend", "cpu",
                    "--workload", "full_prefix_generate", "--pins", str(pins),
                    "--tmpdir", str(private), "--out", str(root / "out.json")]):
                with self.assertRaisesRegex(RuntimeError, "workload differs"):
                    measurement.main()

    def reference_manifest(self):
        pinned = self.policy["models"][self.alias]
        return {"status": "complete", "model": self.alias,
            "repo_id": pinned["repo_id"], "revision": pinned["revision"],
            "config": pinned["config"], "reference_policy": self.policy["reference"],
            "execution_dtype": "float32", "provenance": {
                "policy_sha256": llm_reference.sha256_file(self.policy_path)}}

    def test_additional_checkpoint_keeps_original_numerical_policy(self):
        baseline = llm_reference.read_policy()
        self.assertEqual(self.policy["reference"], baseline["reference"])
        self.assertEqual(set(baseline["models"]), {"smollm2-135m", "qwen3-0.6b",
            "tinyllama-1.1b-chat", "llama3.2-1b-gsm8k"})
        config = DecoderConfig.from_hf(self.policy["models"][self.alias]["config"],
            max_context=self.policy["reference"]["qualification"]["max_context"])
        self.assertFalse(config.tied_embeddings)
        self.assertIn("lm_head.weight", config.weight_shapes())
        self.assertEqual((config.layers, config.query_heads, config.kv_heads), (22, 32, 4))

    def test_reference_rejects_different_policy_bytes_and_model_identity(self):
        manifest = self.reference_manifest()
        policy, pinned = llm_reference.validate_reference_policy(manifest, self.policy_path)
        self.assertEqual(policy, self.policy)
        self.assertEqual(pinned["repo_id"], "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        with tempfile.TemporaryDirectory() as temporary:
            equivalent = Path(temporary) / "same-values.json"
            equivalent.write_text(json.dumps(self.policy, separators=(",", ":")))
            with self.assertRaisesRegex(ValueError, "fixed model/numerical policy"):
                llm_reference.validate_reference_policy(manifest, equivalent)
        for field in ("repo_id", "revision", "model", "execution_dtype"):
            changed = copy.deepcopy(manifest)
            changed[field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                llm_reference.validate_reference_policy(changed, self.policy_path)

    def test_reference_cli_selects_alias_from_requested_policy(self):
        with patch("sys.argv", ["llm_reference.py", "metadata", "--model", self.alias,
                "--policy", str(self.policy_path), "--snapshot", "/tmp/no-download-test"]), \
                patch.object(llm_reference, "download_snapshot") as download:
            llm_reference.main()
        download.assert_called_once_with(Path("/tmp/no-download-test"),
            self.policy["models"][self.alias], metadata_only=True)
        with patch("sys.argv", ["llm_reference.py", "metadata", "--model", "unknown-model",
                "--snapshot", "/tmp/no-download-test"]), \
                patch.object(llm_reference, "download_snapshot") as download, \
                patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                llm_reference.main()
        download.assert_not_called()

    def test_capture_cli_passes_policy_file_to_provenance(self):
        with patch("sys.argv", ["llm_reference.py", "capture", "--model", self.alias,
                "--policy", str(self.policy_path), "--snapshot", "/tmp/snapshot-test",
                "--output", "/tmp/capture-test"]), \
                patch.object(llm_reference, "capture_snapshot", return_value={"cases": []}) as capture, \
                patch("builtins.print"):
            llm_reference.main()
        capture.assert_called_once_with(Path("/tmp/snapshot-test"), Path("/tmp/capture-test"),
            self.alias, self.policy, policy_path=self.policy_path)

    def test_provenance_rejects_policy_that_differs_from_selected_file(self):
        changed = copy.deepcopy(self.policy)
        changed["reference"]["engine_tolerance"]["logits"]["absolute"] *= 2
        with self.assertRaisesRegex(ValueError, "provenance file"):
            llm_reference._provenance(changed, self.policy_path)

    def test_provenance_hashes_selected_policy_file(self):
        policy = copy.deepcopy(self.policy)
        policy["reference"]["transformers_version"] = "test-version"
        policy["reference"]["transformers_source_sha256"] = {}
        fake_transformers = SimpleNamespace(__version__="test-version", __file__=__file__)
        fake_torch = SimpleNamespace(version=SimpleNamespace(git_version="test-commit"))
        with tempfile.TemporaryDirectory() as temporary:
            policy_path = Path(temporary) / "chosen-policy.json"
            policy_path.write_text(json.dumps(policy))
            with patch.dict("sys.modules", {"transformers": fake_transformers, "torch": fake_torch}), \
                    patch.object(llm_reference.importlib, "import_module",
                        return_value=SimpleNamespace(__version__="test-version")):
                provenance = llm_reference._provenance(policy, policy_path)
            self.assertEqual(provenance["policy_sha256"], llm_reference.sha256_file(policy_path))
            self.assertNotEqual(provenance["policy_sha256"],
                llm_reference.sha256_file(llm_reference.POLICY_PATH))

    def test_cached_prepare_shares_selected_f32_weights(self):
        config = {"model_type": "llama", "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 16,
            "rms_norm_eps": 1e-5, "hidden_act": "silu", "tie_word_embeddings": False}
        decoder = DecoderConfig.from_hf(config)
        generator = np.random.default_rng(1701)
        weights = {name: generator.standard_normal(shape).astype(np.float32)
            for name, shape in decoder.weight_shapes().items()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_path, weight_path = root / "policy.json", root / "f32.safetensors"
            policy = {"format": "volvox-hf-baseline/v1", "models": {"tiny-fixture": {
                "config": config}}, "reference": {"qualification": {"max_context": 16}}}
            policy_path.write_text(json.dumps(policy))
            save_file(weights, str(weight_path))
            destination = root / "cached"
            with patch("builtins.print"):
                llm_cached_compare.prepare("tiny-fixture", destination,
                    policy_path=policy_path, weights_path=weight_path)
            for package in ("execution", "capture"):
                shared = destination / package / "model.safetensors"
                self.assertTrue(shared.is_symlink())
                self.assertEqual(shared.resolve(), weight_path)
            receipt = json.loads((destination / "preparation.json").read_text())
            self.assertEqual(receipt["policy"]["sha256"], llm_reference.sha256_file(policy_path))
            self.assertEqual(receipt["source_weights"]["sha256"], llm_reference.sha256_file(weight_path))


if __name__ == "__main__":
    unittest.main()
