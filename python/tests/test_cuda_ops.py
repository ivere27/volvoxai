"""Opt-in CUDA composition oracles; requested CUDA never skips or falls back.

Where→NGram source proofs, Gather axis/index tests, and ReduceMean domain checks.
Set VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda to run the physical device tests.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import volvoxai as vx
from execution_fixture import stateless_graph
from test_modern_moe import save_weights


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND") == "cuda",
                     "explicit CUDA qualification was not requested")
class CudaWhereIndexSourceTest(unittest.TestCase):
    @contextmanager
    def session(self, *, public_right=False, immutable_eos=23):
        shape = ["B", "S"]
        tensor = lambda name, dims: {"tensor": name, "dtype": "int32", "shape": dims}
        inputs = {name: {"dtype": "int32", "shape": shape} for name in ("condition", "left")}
        weights = {"multipliers": np.array([[7, 0], [13, 0], [17, 0]], np.int32),
                   "vocab_sizes": np.array([17, 19], np.int32),
                   "offsets": np.array([0, 17], np.int32)}
        nodes = []
        if public_right:
            inputs["right"] = {"dtype": "int32", "shape": shape}
        else:
            weights["eos"] = np.array([immutable_eos], np.int32)
            nodes.append({"id": "eos-expand", "opType": "Expand", "inputs": {"input": "eos"},
                          "params": {"shape": shape}, "outputs": {"out": tensor("right", shape)}})
        nodes.extend([
            {"id": "select", "opType": "Where",
             "inputs": {"condition": "condition", "a": "left", "b": "right"},
             "params": {},
             "outputs": {"out": tensor("selected", shape)}},
            {"id": "ngram", "opType": "NGramIds",
             "inputs": {"token_ids": "selected", "multipliers": "multipliers",
                        "vocab_sizes": "vocab_sizes", "offsets": "offsets"},
             "params": {"ngram_size": 3, "heads_per_ngram": 1,
                        "eos_token_id": 23, "vocab_size": 24},
             "outputs": {"out": tensor("ids", ["B", "S", 2])}},
        ])
        graph = stateless_graph({"format": "volvox-graph/v1",
            "dimensions": {"B": {"min": 1, "max": 2}, "S": {"min": 1, "max": 6}},
            "inputs": inputs, "nodes": nodes, "outputs": ["ids"]})
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-where-") as temporary:
            directory = Path(temporary)
            (directory / "graph.json").write_text(json.dumps(graph))
            save_weights(directory / "model.safetensors", weights)
            with vx.InferenceSession(directory, backend="cuda") as session:
                self.assertEqual(session.report.compilation.policy_mode,
                                 vx.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE)
                self.assertEqual(session.report.compilation.operator_fallback,
                                 vx.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)
                execute = session._engine.execute

                def observed(request, **kwargs):
                    response = execute(request, **kwargs)
                    self.assertEqual(response.report.backend, "cuda")
                    self.assertFalse(response.report.fallback.operator_fallback_used)
                    route = response.report.route
                    self.assertTrue(route.builtin and route.attested)
                    self.assertGreater(route.active_nodes, 0)
                    self.assertEqual(route.selected_nodes, route.active_nodes)
                    self.assertEqual((route.fallback_nodes, route.missing_nodes), (0, 0))
                    return response

                session._engine.execute = observed
                yield session

    @staticmethod
    def oracle(values):
        tokens = np.where(values["condition"], values["left"], values.get("right", 23))
        expected = np.empty((*tokens.shape, 2), np.int32)
        for batch, row in enumerate(tokens):
            for position, current in enumerate(row):
                for head, (size, offset) in enumerate(((17, 0), (19, 17))):
                    mixed = int(current) * 7
                    stopped = False
                    for shift in range(1, head + 2):
                        prior = 23
                        if position >= shift and not stopped:
                            prior = int(row[position - shift])
                            stopped = prior == 23
                        mixed ^= prior * (13 if shift == 1 else 17)
                    expected[batch, position, head] = mixed % size + offset
        return expected

    def assert_ready(self, session, bindings):
        info = session._engine.get_execution_context_info(vx.pb.ExecutionContextRef(
            context_id=session._context.context_id))
        self.assertEqual(info.state, vx.pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_READY)
        self.assertEqual(info.bindings_available, bindings)

    def test_expanded_immutable_eos_and_mixed_mask_match_ngram_oracle(self):
        with self.session() as session:
            for batch, sequence in ((1, 3), (2, 6), (1, 4)):
                values = {"left": (np.arange(batch * sequence, dtype=np.int32).reshape(batch, sequence) + 3) % 24,
                          "condition": (np.arange(batch * sequence, dtype=np.int32).reshape(batch, sequence) % 3 != 1).astype(np.int32)}
                np.testing.assert_array_equal(session.run(values)["ids"], self.oracle(values))

    def test_each_unselected_public_branch_refuses_before_commit(self):
        for bad_name, select_left in (("left", 0), ("right", 1)):
            with self.subTest(branch=bad_name), self.session(public_right=True) as session:
                good = {"left": np.array([[3, 23, 7]], np.int32),
                        "right": np.array([[4, 6, 23]], np.int32),
                        "condition": np.array([[1, 0, 1]], np.int32)}
                bad = {"left": np.array([[8, 9]], np.int32),
                       "right": np.array([[11, 12]], np.int32),
                       "condition": np.full((1, 2), select_left, np.int32)}
                bad[bad_name][0, 1] = 24
                with self.assertRaises(vx.VolvoxAIError) as failed:
                    session.run(bad)
                self.assertEqual(failed.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
                self.assert_ready(session, False)
                expected = self.oracle(good)
                with session.run_tensors(good) as retained:
                    np.testing.assert_array_equal(retained["ids"].numpy(), expected)
                    with self.assertRaises(vx.VolvoxAIError) as failed:
                        session.run({}, reuse_inputs=list(good))
                    self.assertEqual(failed.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
                    self.assert_ready(session, True)
                    np.testing.assert_array_equal(retained["ids"].numpy(), expected)
                    for invalid in (-1, 24):
                        bad[bad_name][0, 1] = invalid
                        with self.assertRaises(vx.VolvoxAIError) as failed:
                            session.run(bad)
                        self.assertEqual(failed.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
                        self.assert_ready(session, True)
                        np.testing.assert_array_equal(retained["ids"].numpy(), expected)
                        np.testing.assert_array_equal(session.run(
                            {name: good[name] for name in ("left", "right")},
                            reuse_inputs=["condition"])["ids"], expected)

    def test_invalid_immutable_branch_is_refused_at_compile(self):
        for invalid in (-1, 24):
            with self.subTest(eos=invalid), self.assertRaises(vx.VolvoxAIError) as failed:
                with self.session(immutable_eos=invalid):
                    self.fail("out-of-vocabulary immutable Where branch was admitted")
            self.assertEqual(failed.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND") == "cuda",
                     "explicit CUDA qualification was not requested")
class CudaGatherAxisTest(unittest.TestCase):
    @contextmanager
    def session(self, axis, *, fixed_indices=None):
        normalized = axis % 3
        index_shape = list(fixed_indices.shape) if fixed_indices is not None else [2]
        source_shape = ["B", "S", 5]
        output_shape = source_shape[:normalized] + index_shape + source_shape[normalized + 1:]
        inputs = {"x": {"dtype": "float32", "shape": source_shape}}
        if fixed_indices is None:
            inputs["indices"] = {"dtype": "int32", "shape": index_shape}
        graph = stateless_graph({"format": "volvox-graph/v1",
            "dimensions": {"B": {"min": 1, "max": 2}, "S": {"min": 2, "max": 4}},
            "inputs": inputs, "nodes": [{"id": "gather", "opType": "Gather",
                "inputs": {"input": "x", "indices": "indices"}, "params": {"axis": axis},
                "outputs": {"out": {"tensor": "y", "dtype": "float32", "shape": output_shape}}}],
            "outputs": ["y"]})
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-gather-") as temporary:
            directory = Path(temporary)
            (directory / "graph.json").write_text(json.dumps(graph))
            save_weights(directory / "model.safetensors", {} if fixed_indices is None else {"indices": fixed_indices})
            with vx.InferenceSession(directory, backend="cuda") as session:
                self.assertEqual(session.backend, "cuda")
                self.assertEqual(session.report.compilation.policy_mode,
                                 vx.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE)
                self.assertEqual(session.report.compilation.operator_fallback,
                                 vx.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)
                execute = session._engine.execute

                def observed(request, **kwargs):
                    response = execute(request, **kwargs)
                    self.assertEqual(response.report.backend, "cuda")
                    self.assertFalse(response.report.fallback.operator_fallback_used)
                    route = response.report.route
                    self.assertTrue(route.builtin and route.attested)
                    self.assertGreater(route.active_nodes, 0)
                    self.assertEqual(route.selected_nodes, route.active_nodes)
                    self.assertEqual((route.fallback_nodes, route.missing_nodes), (0, 0))
                    return response

                session._engine.execute = observed
                yield session

    @staticmethod
    def values(batch, sequence):
        return (np.arange(batch * sequence * 5, dtype=np.float32).reshape(batch, sequence, 5) - 7) / 4

    def test_axis_two_and_negative_axis_with_immutable_rank_two_indices(self):
        indices = np.array([[4, 0], [-5, -1]], np.int32)
        for axis in (2, -1):
            with self.subTest(axis=axis), self.session(axis, fixed_indices=indices) as session:
                for batch, sequence in ((1, 2), (2, 4), (2, 3)):
                    values = self.values(batch, sequence)
                    np.testing.assert_array_equal(session.run({"x": values})["y"],
                                                  np.take(values, indices, axis=axis))

    def test_middle_axis_public_bounds_refuse_before_input_commit(self):
        with self.session(1) as session:
            engine, pb = session._engine, vx.pb
            reference = pb.ExecutionContextRef(context_id=session._context.context_id)
            first = self.values(2, 2)
            with self.assertRaises(vx.VolvoxAIError) as failed:
                session.run({"x": first, "indices": np.array([-3, 0], np.int32)})
            self.assertEqual(failed.exception.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
            info = engine.get_execution_context_info(reference)
            self.assertEqual(info.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_READY)
            self.assertFalse(info.bindings_available)
            indices = np.array([-1, 0], np.int32)
            expected = np.take(first, indices, axis=1)
            np.testing.assert_array_equal(session.run({"x": first, "indices": indices})["y"], expected)
            for invalid in (4, -5):
                with self.subTest(index=invalid), self.assertRaises(vx.VolvoxAIError) as failed:
                    session.run({"x": self.values(1, 4) + 100,
                                 "indices": np.array([invalid, 0], np.int32)})
                self.assertEqual(failed.exception.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
                info = engine.get_execution_context_info(reference)
                self.assertEqual(info.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_READY)
                self.assertTrue(info.bindings_available)
                np.testing.assert_array_equal(session.run({"indices": indices}, reuse_inputs=["x"])["y"], expected)
            for batch, sequence in ((1, 4), (2, 3)):
                values = self.values(batch, sequence)
                np.testing.assert_array_equal(session.run({"x": values, "indices": indices})["y"],
                                              np.take(values, indices, axis=1))

    def test_immutable_out_of_axis_range_is_refused_at_compile(self):
        for invalid in (5, -6):
            with self.subTest(index=invalid), self.assertRaises(vx.VolvoxAIError) as failed:
                with self.session(2, fixed_indices=np.array([invalid], np.int32)):
                    self.fail("invalid immutable Gather index was admitted")
            self.assertEqual(failed.exception.status, vx.pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED)


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND") == "cuda",
                     "explicit CUDA qualification was not requested")
class CudaReduceMeanDomainTest(unittest.TestCase):
    def test_last_axis_mean_with_both_keepdims_and_dynamic_rows(self):
        pb = vx.pb
        for keepdims in (False, True):
            output_shape = ["B", "S", 5] + ([1] if keepdims else [])
            graph = stateless_graph({"format": "volvox-graph/v1",
                "dimensions": {"B": {"min": 1, "max": 2}, "S": {"min": 1, "max": 4}},
                "inputs": {"x": {"dtype": "float32", "shape": ["B", "S", 5, 3]}},
                "nodes": [{"id": "mean", "opType": "ReduceMean", "inputs": {"input": "x"},
                    "params": {"axis": -1, "keepdims": keepdims},
                    "outputs": {"out": {"tensor": "y", "dtype": "float32", "shape": output_shape}}}],
                "outputs": ["y"]})
            with self.subTest(keepdims=keepdims), tempfile.TemporaryDirectory(prefix="volvox-cuda-mean-") as temporary:
                directory = Path(temporary)
                (directory / "graph.json").write_text(json.dumps(graph))
                save_weights(directory / "model.safetensors", {})
                with vx.InferenceSession(directory, backend="cuda") as session:
                    self.assertEqual(session.report.compilation.policy_mode,
                                     pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE)
                    self.assertEqual(session.report.compilation.operator_fallback,
                                     pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)
                    for batch, sequence in ((1, 1), (2, 4), (2, 2)):
                        values = (np.arange(batch * sequence * 15, dtype=np.float32).reshape(batch, sequence, 5, 3) - 19) / 4
                        packet = session._engine.execute(pb.ExecuteRequest(context_id=session._context.context_id,
                            scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty()),
                            inputs=pb.InputBindings(values=[pb.Tensor(name="x", dtype=pb.DataType.DATA_TYPE_F32,
                                shape=list(values.shape), inline=values.tobytes())])))
                        self.assertEqual(packet.report.backend, "cuda")
                        self.assertFalse(packet.report.fallback.operator_fallback_used)
                        route = packet.report.route
                        self.assertTrue(route.builtin and route.attested)
                        self.assertGreater(route.active_nodes, 0)
                        self.assertEqual(route.selected_nodes, route.active_nodes)
                        self.assertEqual((route.fallback_nodes, route.missing_nodes), (0, 0))
                        expected = np.mean(values, axis=-1, keepdims=keepdims)
                        output = packet.outputs[0]
                        self.assertEqual(output.dtype, pb.DataType.DATA_TYPE_F32)
                        self.assertEqual(list(output.shape), list(expected.shape))
                        np.testing.assert_allclose(np.frombuffer(output.inline, np.float32).reshape(output.shape),
                                                   expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
