"""Independent metric examples and public-runtime teacher-forcing alignment."""
import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from tools.evaluate_quantization_quality import (
    aggregate_language_metrics, evaluate_model_on_cases, score_multiple_choice,
    score_rows, softmax_log_prob, validate_cases, main, vx,
)
from tools.exporter.hf_decoder import DecoderConfig, build_decoder_graph
from tools.exporter.packed_linear import pack_linear_weights
from tools.llm_baseline.generate_modern_incremental_stress import tiny_dense_config


class QualityInputTests(unittest.TestCase):
    def assert_cli_error(self, arguments, message):
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["quality", *arguments]), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_policy_models_and_output_must_be_explicit(self):
        options = {"--policy": "policy.json", "--model": "f32=model", "--out": "report.json"}
        for missing in options:
            arguments = [part for option, argument in options.items() if option != missing
                         for part in (option, argument)]
            with self.subTest(missing=missing):
                self.assert_cli_error(arguments, f"required: {missing}")

    def test_invalid_or_missing_requested_models_fail_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            package = path / "model"
            package.mkdir()
            (package / "graph.json").write_text("{}")
            (package / "model.safetensors").write_bytes(b"weights")
            output = path / "report.json"
            common = ["--policy", str(path / "policy.json"), "--out", str(output)]
            valid = f"f32={package}"
            invalid = [(["f32"], "--model requires NAME="),
                       ([f"={package}"], "--model requires NAME="),
                       ([f" ={package}"], "--model requires NAME="),
                       (["f32="], "--model requires NAME="),
                       ([valid, valid], "Duplicate model name f32"),
                       ([f"w4={package}"], "Required reference model f32 was not provided"),
                       ([valid, f"w4={path / 'missing'}"], "Model w4 is missing required file")]
            for models, message in invalid:
                arguments = common + [value for model in models for value in ("--model", model)]
                with self.subTest(models=models):
                    self.assert_cli_error(arguments, message)
                    self.assertFalse(output.exists())
            for filename in ("graph.json", "model.safetensors"):
                artifact = package / filename
                contents = artifact.read_bytes()
                artifact.unlink()
                with self.subTest(missing_file=filename):
                    self.assert_cli_error(common + ["--model", valid], f"Model f32 is missing required file: {artifact}")
                    self.assertFalse(output.exists())
                artifact.write_bytes(contents)


class QualityMetricTests(unittest.TestCase):
    def test_likelihood_and_kl_against_hand_calculated_probabilities(self):
        reference = np.log([[.75, .25], [.2, .8]])
        candidate = np.log([[.5, .5], [.4, .6]])
        score = score_rows(candidate, [0, 1], reference)
        self.assertAlmostEqual(score["mean_nll"], (-np.log(.5) - np.log(.6)) / 2)
        expected = (.75 * np.log(.75 / .5) + .25 * np.log(.25 / .5)
                    + .2 * np.log(.2 / .4) + .8 * np.log(.8 / .6)) / 2
        self.assertAlmostEqual(score["mean_kl_nats"], expected)
        self.assertEqual(score["target_top1_correct"], 2)
        self.assertEqual(score["top1_agreement_pct"], 100)
        np.testing.assert_allclose(softmax_log_prob(candidate + 12345)[0], candidate, atol=1e-11)

    def test_language_aggregate_weights_tokens_and_excludes_answer_candidates(self):
        scores = {"one": score_rows(np.log([[.5, .5]]), [0]),
                  "three": score_rows(np.log([[.25, .75]] * 3), [0] * 3),
                  "answer": score_rows(np.log([[.01, .99]]), [0])}
        aggregate = aggregate_language_metrics(scores, {"one": {}, "three": {}, "answer": {"task": {"id": "x"}}})
        self.assertEqual(aggregate["token_count"], 4)
        self.assertAlmostEqual(aggregate["mean_nll"], (-np.log(.5) - 3 * np.log(.25)) / 4)
        self.assertAlmostEqual(aggregate["perplexity"], np.exp(aggregate["mean_nll"]))

    def test_choice_scores_use_total_likelihood_and_declared_character_normalization(self):
        cases = {f"choice{i}": {"task": {"id": "q", "choice_index": i, "choice_count": 2,
                                        "answer_index": 1, "normalization_chars": 1 if i == 0 else 4}}
                 for i in range(2)}
        result = score_multiple_choice({"choice0": {"nll_sum": 2}, "choice1": {"nll_sum": 3}}, cases)
        self.assertEqual(result["accuracy_pct"], 0)
        self.assertEqual(result["character_normalized_accuracy_pct"], 100)
        with self.assertRaises(ValueError):
            score_multiple_choice({"choice0": {"nll_sum": 2}}, {"choice0": cases["choice0"]})

    def test_invalid_tokens_shapes_capacities_and_nonfinite_logits_refused(self):
        case = {"initial_ids": [1, 2], "teacher_ids": [3, 4], "capacity": 3}
        validate_cases({"ok": case}, 5, 3)
        for edit in ({"capacity": 2}, {"teacher_ids": [5]}, {"initial_ids": []}):
            with self.assertRaises(ValueError):
                validate_cases({"bad": dict(case, **edit)}, 5, 3)
        with self.assertRaises(ValueError):
            score_rows([[1, np.nan]], [0])
        with self.assertRaises(ValueError):
            score_rows([[1, 2]], [2])

    def test_public_runtime_prefill_rows_match_incremental_teacher_forcing(self):
        raw = tiny_dense_config(hidden_size=32, intermediate_size=64, head_dim=8,
                                num_attention_heads=4, num_key_value_heads=2,
                                num_hidden_layers=1, vocab_size=23, max_position_embeddings=256)
        config = DecoderConfig.from_hf(raw, max_context=256)
        rng = np.random.default_rng(907)
        weights = {name: rng.normal(0, .05, shape).astype(np.float32)
                   for name, shape in config.weight_shapes().items()}
        for name in weights:
            if "norm" in name:
                weights[name].fill(1)
        graph, weights = build_decoder_graph(config, weights, compact_kv=True, cached=True)
        # A short request still needs output admission for the graph maximum.
        # 256 declared rows versus 4–16 actual rows reproduces conservative
        # result-budget admission used by the 2048-row checkpoint smoke.
        base = {"initial_ids": [1, 7, 12], "teacher_ids": [2, 15, 6, 9], "capacity": 8}
        cases = {"prefill": dict(base, execution="prefill"), "incremental": dict(base, execution="incremental"),
                 "shorter": {"initial_ids": [9, 2], "teacher_ids": [3, 8], "capacity": 4, "execution": "prefill"},
                 "larger": {"initial_ids": [4, 7, 9, 3], "teacher_ids": [1, 2, 8], "capacity": 16, "execution": "prefill"},
                 "repeated": dict(base, execution="prefill")}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "graph.json").write_text(json.dumps(graph))
            save_file(weights, path / "model.safetensors")
            original_create = vx.VxInferenceServiceClient.create_execution_context
            original_release = vx.VxInferenceServiceClient.release_execution_context
            ownership = {"live": 0, "peak": 0}
            def create(client, request):
                handle = original_create(client, request)
                ownership["live"] += 1
                ownership["peak"] = max(ownership["peak"], ownership["live"])
                return handle
            def release(client, request):
                result = original_release(client, request)
                ownership["live"] -= 1
                return result
            with patch.object(vx.VxInferenceServiceClient, "create_execution_context", create), \
                    patch.object(vx.VxInferenceServiceClient, "release_execution_context", release):
                results = evaluate_model_on_cases(path / "graph.json", path / "model.safetensors", cases, cpu_threads=1)
            self.assertEqual(ownership, {"live": 0, "peak": 1})
            fresh = evaluate_model_on_cases(path / "graph.json", path / "model.safetensors", cases,
                                            cpu_threads=1, reuse_prefill_context=False)
            for name in cases:
                np.testing.assert_allclose(results[name], fresh[name], atol=2e-6, rtol=2e-5)
            # The full CLI matrix retains metrics, removes temporary full-vocab
            # reference rows, and exposes every requested arithmetic separately.
            source = path / "source"
            packed = path / "packed"
            source.mkdir(); packed.mkdir()
            (source / "graph.json").write_text(json.dumps(graph))
            save_file(weights, source / "model.safetensors")
            packed_graph, packed_weights = pack_linear_weights(graph, weights, weight_bits=4)
            (packed / "graph.json").write_text(json.dumps(packed_graph))
            save_file(packed_weights, packed / "model.safetensors")
            (path / "policy.json").write_text(json.dumps({"cases": {"sample": dict(base, execution="prefill")}}))
            arguments = ["quality", "--policy", str(path / "policy.json"), "--model", f"f32={source}",
                         "--model", f"w4={packed}", "--packed-arithmetic", "f32", "f16", "dynamic-i8", "f32-fast",
                         "--out", str(path / "report.json")]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((path / "report.json").read_text())
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(len(report["variants"]), 5)
            self.assertTrue(report["teacher_forcing_smoke"]["passed"])
            self.assertEqual(len(report["variant_reuse_smokes"]), 4)
            self.assertTrue(all(value["status"] == "PASSED" for value in report["variant_reuse_smokes"].values()))
            self.assertEqual(list(path.glob("*.reference.*")), [])
        np.testing.assert_allclose(results["prefill"], results["incremental"], atol=2e-6, rtol=2e-5)
        self.assertEqual(results["prefill"].shape, (4, 23))


if __name__ == "__main__":
    unittest.main()
