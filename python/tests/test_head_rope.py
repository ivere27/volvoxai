"""Head-local normalization/RoPE through the public CPU graph API."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx

from execution_fixture import stateless_graph


def head_graph(heads=4, width=128):
    """Keep S symbolic while separating heads before normalization/rotation."""
    nodes = []

    def node(name, operator, inputs, shape, params=None):
        nodes.append({"id": name, "opType": operator, "inputs": inputs,
                      "outputs": {"out": {"tensor": name, "shape": shape,
                                          "dtype": "float32"}},
                      "params": params or {}})

    node("split", "Reshape", {"input": "x"}, ["S", heads, width],
         {"shape": ["S", heads, width]})
    node("head_major", "Transpose", {"input": "split"}, [heads, "S", width],
         {"perm": [1, 0, 2]})
    node("normalized", "RMSNorm", {"input": "head_major", "weight": "norm"},
         [heads, "S", width], {"eps": 1e-6})
    node("rotated", "RoPE", {"input": "normalized", "position_ids": "positions"},
         [heads, "S", width],
         {"rotary_dim": width, "theta": 1000000.0, "interleaved": False})
    node("row_major", "Transpose", {"input": "rotated"}, ["S", heads, width],
         {"perm": [1, 0, 2]})
    node("y", "Reshape", {"input": "row_major"}, [1, "S", heads * width],
         {"shape": [1, "S", heads * width]})
    return stateless_graph({
        "format": "volvox-graph/v1", "dimensions": {"S": {"min": 1, "max": 5}},
        "inputs": {"x": {"shape": [1, "S", heads * width], "dtype": "float32"},
                   "positions": {"shape": ["S"], "dtype": "int32"},
                   "norm": {"shape": [width], "dtype": "float32"}},
        "nodes": nodes, "outputs": ["y"],
    })


def inputs(sequence, heads=4, width=128):
    values = np.arange(sequence * heads * width, dtype=np.float64)
    values = (np.sin(values * .17) + .25 * np.cos(values * .31)).astype(np.float32)
    return {"x": values.reshape(1, sequence, heads * width),
            "positions": (7 + np.arange(sequence, dtype=np.int32) * 3),
            "norm": np.linspace(.7, 1.3, width, dtype=np.float32)}


def reference(values, heads=4, width=128):
    # Independent NumPy formula: each head has its own RMS reduction and pairs.
    x = values["x"].astype(np.float64).reshape(-1, heads, width)
    normalized = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    normalized *= values["norm"].astype(np.float64)
    half = width // 2
    inverse_frequency = 1000000.0 ** (-np.arange(half, dtype=np.float64) * 2 / width)
    angles = values["positions"].astype(np.float64)[:, None, None] * inverse_frequency
    cosine, sine = np.cos(angles), np.sin(angles)
    left, right = normalized[..., :half], normalized[..., half:]
    rotated = np.concatenate((left * cosine - right * sine,
                              right * cosine + left * sine), axis=-1)
    return rotated.astype(np.float32).reshape(values["x"].shape)


class HeadRoPETest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "graph.json").write_text(json.dumps(head_graph()))

    def test_dynamic_head_norm_and_half_split_rope_match_independent_reference(self):
        with vx.InferenceSession(self.directory, backend="cpu", cpu_threads=1) as session:
            self.assertEqual(session.backend, "cpu")
            self.assertFalse(session.report.compilation.tier_fallback_used)
            for sequence in (1, 3, 5, 1):
                with self.subTest(sequence=sequence):
                    values = inputs(sequence)
                    actual = session.run(values)["y"]
                    np.testing.assert_allclose(actual, reference(values), rtol=2e-5, atol=2e-5)

    def test_invalid_positions_fail_context_and_fresh_context_recovers(self):
        with vx.InferenceSession(self.directory, backend="cpu", cpu_threads=1) as session:
            values = inputs(3)
            expected = reference(values)
            with session.run_tensors(values) as retained:
                np.testing.assert_allclose(retained["y"].numpy(), expected,
                                           rtol=2e-5, atol=2e-5)
                invalid = {**values, "positions": np.array([7, -1, 13], dtype=np.int32)}
                with self.assertRaises(vx.VolvoxAIError) as failed:
                    session.run(invalid)
                self.assertEqual(failed.exception.status,
                                 vx.pb.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED)
                self.assertEqual(failed.exception.stage,
                                 vx.pb.OperationStage.OPERATION_STAGE_EXECUTE)
                info = session._engine.get_execution_context_info(vx.pb.ExecutionContextRef(
                    context_id=session._context.context_id))
                self.assertEqual(info.state, vx.pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_FAILED)
                session.new_context()
                np.testing.assert_allclose(session.run(values)["y"], expected,
                                           rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(retained["y"].numpy(), expected,
                                           rtol=2e-5, atol=2e-5)

    def test_d128_attention_matches_gqa_with_explicit_kv_head_expansion(self):
        heads, kv_heads, width = 4, 2, 128
        qshape, kvshape = [1, "S", heads * width], [1, "S", kv_heads * width]
        expanded = []
        for name in ("k", "v"):
            expanded.append({"id": name + "_expanded", "opType": "Gather",
                "inputs": {"input": name, "indices": "repeat_indices"},
                "outputs": {"out": {"tensor": name + "_expanded", "shape": qshape,
                                    "dtype": "float32"}}, "params": {"axis": 2}})
        graph = stateless_graph({"format": "volvox-graph/v1",
            "dimensions": {"S": {"min": 1, "max": 5}},
            "inputs": {"q": {"shape": qshape, "dtype": "float32"},
                       "k": {"shape": kvshape, "dtype": "float32"},
                       "v": {"shape": kvshape, "dtype": "float32"},
                       "repeat_indices": {"shape": [heads * width], "dtype": "int32"},
                       "keep": {"shape": ["S"], "dtype": "int32"}},
            "nodes": [*expanded, {"id": "attention", "opType": "CrossSDPA",
                "inputs": {"q": "q", "k": "k_expanded", "v": "v_expanded", "mask": "keep"},
                "outputs": {"out": {"tensor": "y", "shape": qshape, "dtype": "float32"}},
                "params": {"heads": heads, "causal": True}}], "outputs": ["y"]})
        (self.directory / "graph.json").write_text(json.dumps(graph))
        repeats = np.array([(head // (heads // kv_heads)) * width + channel
                            for head in range(heads) for channel in range(width)], dtype=np.int32)
        with vx.InferenceSession(self.directory, backend="cpu", cpu_threads=1) as session:
            for sequence in (1, 3, 5):
                with self.subTest(sequence=sequence):
                    q = inputs(sequence)["x"]
                    axis = np.arange(sequence * kv_heads * width, dtype=np.float64)
                    k = (.3 * np.cos(axis * .11)).astype(np.float32).reshape(1, sequence, -1)
                    v = (.4 * np.sin(axis * .23)).astype(np.float32).reshape(1, sequence, -1)
                    keep = np.ones(sequence, dtype=np.int32)
                    if sequence > 1:
                        keep[-1] = 0
                    actual = session.run({"q": q, "k": k, "v": v,
                                          "repeat_indices": repeats, "keep": keep})["y"]
                    expected = np.zeros((sequence, heads, width), dtype=np.float64)
                    qr = q.reshape(sequence, heads, width).astype(np.float64)
                    kr = k.reshape(sequence, kv_heads, width).astype(np.float64)
                    vr = v.reshape(sequence, kv_heads, width).astype(np.float64)
                    for position in range(sequence):
                        visible = np.flatnonzero(keep[:position + 1])
                        for head in range(heads):
                            kv = head // (heads // kv_heads)
                            scores = kr[visible, kv] @ qr[position, head] / np.sqrt(width)
                            probabilities = np.exp(scores - scores.max())
                            probabilities /= probabilities.sum()
                            expected[position, head] = probabilities @ vr[visible, kv]
                    np.testing.assert_allclose(actual, expected.reshape(q.shape),
                                               rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
