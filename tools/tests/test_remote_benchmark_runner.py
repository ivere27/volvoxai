"""Measurement contracts: requested work never substitutes for actual work."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools import remote_benchmark_runner as runner


class BenchmarkManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        (self.base / "graph.json").write_text("{}")
        (self.base / "weights.bin").write_bytes(b"weights")
        self.config = {"models": [{"name": "w4", "graph_path": "graph.json",
            "weight_paths": ["weights.bin"], "packed_arithmetic": "f32-fast"}],
            "cases": {"short": {"initial_ids": [1, 2], "budget": 4, "capacity": 8,
                                  "decode_chunk_size": 1}}}

    def test_relative_paths_and_case_chunking(self):
        models, protocol = runner.validate_manifest(self.config, self.base)
        self.assertEqual(models[0]["graph_path"], str(self.base / "graph.json"))
        self.assertEqual(models[0]["packed_arithmetic"], "f32-fast")
        self.assertEqual(protocol["measured_cycles"], 5)
        self.assertEqual(protocol["decode_chunk_size"], 0)

    def test_requested_missing_model_and_file_fail(self):
        with self.assertRaisesRegex(ValueError, "unknown requested"):
            runner.validate_manifest(self.config, self.base, {"typo"})
        (self.base / "weights.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            runner.validate_manifest(self.config, self.base)

    def test_invalid_budget_capacity_and_protocol_fail(self):
        for update in ({"budget": 0}, {"capacity": 3}, {"decode_chunk_size": -1}, {"initial_ids": [True]}):
            config = {**self.config, "cases": {"bad": {**self.config["cases"]["short"], **update}}}
            with self.subTest(update=update), self.assertRaises(ValueError):
                runner.validate_manifest(config, self.base)
        self.config["protocol"] = {"measured_cycles": 0}
        with self.assertRaises(ValueError):
            runner.validate_manifest(self.config, self.base)


class TimingSummaryTests(unittest.TestCase):
    @staticmethod
    def cycle(ttft, decode, tokens=3, phase="measured", split=False):
        return {"phase": phase, "context_create_ms": 1, "ttft_ms": ttft,
                "decode_ms": decode, "total_ms": ttft + decode, "generated_tokens": tokens + 1,
                "decode_tokens": tokens, "ids": [1] * (tokens + 1),
                "decode_calls": ([{"tokens": 1, "call_ms": decode / tokens, "delivery_interval_ms": decode / tokens}] * tokens if split
                                 else [{"tokens": tokens, "call_ms": decode, "delivery_interval_ms": decode}]) if tokens else []}

    def test_percentiles_exclude_cold_and_grouped_is_not_token_latency(self):
        cycles = [self.cycle(1000, 1000, phase="cold"), self.cycle(10, 6), self.cycle(20, 12)]
        summary = runner.summarize_cycles(cycles, 100)
        self.assertEqual(summary["ttft_ms"]["p50"], 15)
        self.assertEqual(summary["ttft_ms"]["p95"], 19.5)
        self.assertEqual(summary["decode_tokens_per_sec"]["p50"], 375)
        self.assertEqual(summary["prefill_tokens_per_sec"]["p50"], 7500)
        self.assertIsNone(summary["inter_token_ms"])
        self.assertEqual(summary["amortized_decode_ms_per_token"]["p50"], 3)

    def test_single_token_decode_and_prefill_only(self):
        summary = runner.summarize_cycles([self.cycle(10, 6, split=True)], 100)
        self.assertEqual(summary["inter_token_ms"]["count"], 3)
        self.assertEqual(summary["inter_token_ms"]["p50"], 2)
        prefill = runner.summarize_cycles([self.cycle(10, 0, tokens=0)], 100)
        self.assertIsNone(prefill["decode_tokens_per_sec"])
        self.assertIsNone(prefill["inter_token_ms"])


class GenerationContractTests(unittest.TestCase):
    def setUp(self):
        self.pb = NS(DataType=NS(DATA_TYPE_I32=3), NativeStatus=NS(NATIVE_STATUS_OK=0))
        self.report = NS(status=0, backend="cuda", device="test GPU", fallback=None,
            route=NS(provider="cuda", builtin=True, attested=True, active_nodes=3,
                     selected_nodes=3, fallback_nodes=0, missing_nodes=0, digest=1))
        self.patch = patch.object(runner, "pb", self.pb)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_actual_committed_tokens_must_match_output(self):
        response = NS(outputs=[NS(dtype=3, inline=np.array([1, 2], np.int32).tobytes())],
                      progress=NS(steps=2))
        self.assertEqual(runner.response_ids(response), [1, 2])
        response.progress.steps = 3
        with self.assertRaisesRegex(RuntimeError, "committed"):
            runner.response_ids(response)

    def test_wall_latency_includes_between_call_work_and_context_is_released(self):
        for name in ("CreateExecutionContextRequest", "AutoregressiveExecution", "GenerateRequest",
                     "GenerationInputs", "GenerationBudget", "OutputOptions", "ExecutionContextRef"):
            setattr(self.pb, name, lambda **kwargs: NS(**kwargs))
        self.pb.SamplingMode = NS(SAMPLING_MODE_GREEDY=1)
        self.pb.OutputStorage = NS(OUTPUT_STORAGE_INLINE=1)
        client = Mock()
        client.create_execution_context.return_value = NS(context_id=7)
        client.generate.side_effect = [NS(report=self.report,
            outputs=[NS(dtype=3, inline=np.array([token], np.int32).tobytes())],
            progress=NS(steps=1, can_continue=token < 3, stop_reason=0)) for token in (1, 2, 3)]
        with patch.object(runner.time, "perf_counter_ns", side_effect=range(1000000, 9000000, 1000000)):
            cycle = runner.run_cycle(client, 3, {"budget": 3}, [], 1, "measured")
        self.assertEqual(cycle["ttft_ms"], 1)
        self.assertEqual(cycle["decode_ms"], 4)
        self.assertEqual(cycle["total_ms"], 5)
        self.assertEqual(cycle["generate_call_ms"], 3)
        self.assertEqual([call["delivery_interval_ms"] for call in cycle["decode_calls"]], [2, 2])
        self.assertEqual(cycle["ids"], [1, 2, 3])
        client.release_execution_context.assert_called_once()

    def test_cuda_route_must_be_complete(self):
        self.assertEqual(runner.route_evidence(self.report)["active_nodes"], 3)
        for field, value in (("selected_nodes", 2), ("fallback_nodes", 1), ("attested", False)):
            previous = getattr(self.report.route, field)
            setattr(self.report.route, field, value)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.route_evidence(self.report)
            setattr(self.report.route, field, previous)


if __name__ == "__main__":
    unittest.main()
