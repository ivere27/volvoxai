"""Tiny public generation policies on native CPU or required CUDA.

Set VOLVOXAI_LIBRARY to either release profile. Opt in to physical CUDA with
VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda; missing CUDA then fails admission.
The zero-Q/K causal fixture has analytical logits and needs no downloaded
model, disk weights, timing gate or Python numerical generation fallback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx

PB = vx.pb


def fixture():
    inputs = {"tokens": {"dtype": "int32", "shape": [1, 6]},
        "keep": {"dtype": "int32", "shape": [1, 6]},
        "lengths": {"dtype": "int32", "shape": [1]},
        "table": {"dtype": "float32", "shape": [2, 2]},
        "query": {"dtype": "float32", "shape": [1, 6, 2]},
        "key": {"dtype": "float32", "shape": [1, 6, 2]}}
    def node(name, op, ports, shape, params=None, dtype="float32"):
        return {"id": name, "opType": op, "inputs": ports, "params": params or {},
            "outputs": {"out": {"tensor": name, "dtype": dtype, "shape": shape}}}
    outputs = ["ids", "logits"]
    return {"format": "volvox-graph/v1", "dimensions": {}, "inputs": inputs,
        "nodes": [node("value", "Embedding", {"input": "tokens", "weight": "table"}, [1, 6, 2]),
            node("logits", "CrossSDPA", {"q": "query", "k": "key", "v": "value", "mask": "keep"},
                [1, 6, 2], {"heads": 1, "causal": True}),
            node("ids", "ArgMax", {"input": "logits"}, [1, 6], {"axis": 2, "keepdims": False}, "int32")],
        "outputs": outputs, "execution": {"invariant_inputs": ["lengths", "table", "query", "key"],
            "incremental": {"active_lengths_input": "lengths", "sequence_axis": 1, "update_inputs": ["tokens", "keep"]},
            "entrypoints": [{"name": name, "effect": effect, "allowed_ranges": ranges,
                "inputs": ports, "outputs": outputs} for name, effect, ranges, ports in
                (("evaluate", 1, [1], list(inputs)), ("initialize", 2, [1], list(inputs)),
                 ("transition", 3, [3], ["tokens", "keep"]))],
            "generation": {"initialize_entrypoint": "initialize", "transition_entrypoint": "transition",
                "token_input": "tokens", "keep_input": "keep", "token_output": "ids", "logits_output": "logits",
                "logits_axis": 2, "id_domain_size": 2, "output_name": "generated", "sequence_axis": 1}}}


def initial(length=2):
    values = {"tokens": np.array([[0, 1, 0, 0, 0, 0]], dtype=np.int32),
        "keep": np.array([[1, 1, 0, 0, 0, 0]], dtype=np.int32),
        "lengths": np.array([length], dtype=np.int32),
        "table": np.array([[0, 3], [2, 0]], dtype=np.float32),
        "query": np.zeros((1, 6, 2), dtype=np.float32), "key": np.zeros((1, 6, 2), dtype=np.float32)}
    return PB.GenerationInputs(values=[PB.Tensor(name=name, shape=list(value.shape), inline=value.tobytes(),
        dtype=PB.DataType.DATA_TYPE_I32 if value.dtype == np.int32 else PB.DataType.DATA_TYPE_F32)
        for name, value in values.items()])


class GenerationPolicyTest(unittest.TestCase):
    def setUp(self):
        self.backend = os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND", "cpu")
        if self.backend not in ("cpu", "cuda"):
            self.skipTest("this policy fixture admits CPU and opt-in CUDA")
        self.host = vx.open_library()
        self.addCleanup(self.host.close)
        self.inference = vx.VxInferenceServiceClient(self.host)
        runtime = self.inference.create_runtime(PB.CreateRuntimeRequest(cpu_threads=1))
        self.addCleanup(self.inference.release_runtime, PB.RuntimeRef(runtime_id=runtime.runtime_id))
        model = self.inference.load_model(PB.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=PB.ModelPackage(graph_document=json.dumps(fixture()).encode())))
        self.addCleanup(self.inference.release_model, PB.ModelRef(model_id=model.model_id))
        self.compiled = self.inference.compile_model(PB.CompileModelRequest(model_id=model.model_id,
            policy=PB.BackendPolicy(mode=PB.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=[self.backend], operator_fallback=PB.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        self.addCleanup(self.inference.release_compiled_model,
            PB.CompiledModelRef(compiled_model_id=self.compiled.compiled_model_id))
        self.route(self.compiled.report)

    def route(self, report):
        self.assertEqual(report.backend, self.backend)
        self.assertFalse(report.fallback and report.fallback.operator_fallback_used)
        if self.backend == "cuda":
            self.assertIsNotNone(report.route)
            self.assertTrue(report.route.attested and report.route.builtin)
            self.assertEqual(report.route.selected_nodes, report.route.active_nodes)
            self.assertGreater(report.route.active_nodes, 0)
            self.assertEqual((report.route.fallback_nodes, report.route.missing_nodes), (0, 0))

    def context(self, greedy=False, **options):
        context = self.inference.create_execution_context(PB.CreateExecutionContextRequest(
            compiled_model_id=self.compiled.compiled_model_id,
            autoregressive=PB.AutoregressiveExecution(max_steps=5,
                sampling=PB.SamplingMode.SAMPLING_MODE_GREEDY if greedy else PB.SamplingMode.SAMPLING_MODE_RANDOM, **options)))
        self.addCleanup(self.inference.release_execution_context, PB.ExecutionContextRef(context_id=context.context_id))
        return context.context_id

    def step(self, context, count, first=False):
        packet = self.inference.generate(PB.GenerateRequest(context_id=context, initial_inputs=initial() if first else None,
            budget=PB.GenerationBudget(max_steps=count)))
        self.route(packet.report)
        output = packet.outputs[0]
        self.assertEqual((output.name, output.dtype), ("generated", PB.DataType.DATA_TYPE_I32))
        self.assertFalse(output.HasField("buffer"))
        ids = np.frombuffer(output.inline, dtype=np.int32).tolist()
        self.assertEqual(output.shape, [1, len(ids)])
        self.assertEqual(packet.progress.steps, len(ids))
        return ids, packet

    def run_policy(self, chunks, greedy=False, **options):
        context = self.context(greedy, **options)
        result = []
        for count in chunks:
            ids, packet = self.step(context, count, not result)
            result.extend(ids)
            self.assertEqual(packet.progress.total_steps, len(result))
            if not packet.progress.can_continue:
                break
        return result, packet, context

    def test_fixed_seeds_and_chunk_invariance(self):
        # Independent SplitMix64 draws + analytical uniform-attention logits
        # [2*count(1)/n, 3*count(0)/n], never engine-produced reference IDs.
        cases = [({}, [0, 1, 1, 1, 0]), ({"seed": 7}, [1, 0, 0, 1, 1]),
            ({"seed": 2**64-1}, [0, 0, 1, 1, 0])]
        for options, expected in cases:
            for chunks in ((5,), (1, 1, 1, 1, 1)):
                with self.subTest(options=options, chunks=chunks):
                    ids, packet, _ = self.run_policy(chunks, **options)
                    self.assertEqual(ids, expected)
                    self.assertEqual(packet.progress.stop_reason, PB.ExecutionStopReason.EXECUTION_STOP_REASON_STEP_LIMIT)

    def test_temperature_and_top_filters(self):
        ids, _, _ = self.run_policy((1, 2, 2), seed=7, temperature=1000)
        self.assertEqual(ids, [1, 0, 0, 0, 1])
        expected, _, _ = self.run_policy((5,), greedy=True)
        for options in ({"seed": 7, "top_k": 1}, {"seed": 7, "top_p": .4}):
            ids, _, _ = self.run_policy((1, 2, 2), **options)
            self.assertEqual(ids, expected)

    def test_eos_id_stop_and_cross_call_sequence_matching(self):
        for options, reason, expected in (({"stop_ids": [1]}, PB.ExecutionStopReason.EXECUTION_STOP_REASON_ID_STOP, [1]),
                ({"stop_ids": [1], "eos_ids": [1]}, PB.ExecutionStopReason.EXECUTION_STOP_REASON_EOS, [1]),
                ({"stop_sequences": [PB.IdStopSequence(ids=[0, 1])]}, PB.ExecutionStopReason.EXECUTION_STOP_REASON_ID_STOP, [1, 0, 1])):
            for chunks in ((5,), (1, 1, 1, 1, 1)):
                with self.subTest(options=options, chunks=chunks):
                    ids, packet, context = self.run_policy(chunks, greedy=True, **options)
                    self.assertEqual(ids, expected)  # Prompt [0,1] never matches.
                    self.assertEqual(packet.progress.stop_reason, reason)
                    self.assertFalse(packet.progress.can_continue)
                    with self.assertRaises(vx.VolvoxAIError):
                        self.step(context, 1)

    def test_context_rng_isolation_and_atomic_invalid_initial_inputs(self):
        left, right = self.context(seed=7), self.context(seed=2**64-1)
        with self.assertRaises(vx.VolvoxAIError) as failure:
            self.inference.generate(PB.GenerateRequest(context_id=left, initial_inputs=initial(0),
                budget=PB.GenerationBudget(max_steps=5)))
        self.assertEqual(failure.exception.status, PB.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
        info = self.inference.get_execution_context_info(PB.ExecutionContextRef(context_id=left))
        self.assertFalse(info.initialized)
        self.assertEqual(info.total_steps, 0)
        a, b = [], []
        for index in range(5):
            a.extend(self.step(left, 1, index == 0)[0])
            b.extend(self.step(right, 1, index == 0)[0])
        self.assertEqual(a, [1, 0, 0, 1, 1])
        self.assertEqual(b, [0, 0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
