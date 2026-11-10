"""Tests for PackedLSTM operator: packing helpers, shape/capability contracts, and native runtime execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from tools.exporter.capabilities import validate_graph, validate_runtime_descriptors
from tools.exporter.packed_linear import pack_lstm_weight, pack_lstm_weights
import volvoxai as vx


def make_lstm_fixture(seq_len=3, batch=2, width=16, hidden=8, direction=2):
    rng = np.random.default_rng(42)
    directions = 2 if direction == 2 else 1
    x = rng.normal(0.0, 1.0, size=(seq_len, batch, width)).astype(np.float32)
    w = rng.normal(0.0, 0.5, size=(directions, 4 * hidden, width)).astype(np.float32)
    r = rng.normal(0.0, 0.5, size=(directions, 4 * hidden, hidden)).astype(np.float32)
    b = rng.normal(0.0, 0.1, size=(directions, 8 * hidden)).astype(np.float32)

    graph = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [seq_len, batch, width], "dtype": "float32"}},
        "nodes": [
            {
                "id": "lstm_node",
                "opType": "LSTM",
                "inputs": {
                    "input": "x",
                    "weight": "w_tensor",
                    "recurrent_weight": "r_tensor",
                    "bias": "b_tensor",
                },
                "outputs": {
                    "out": {
                        "tensor": "y",
                        "shape": [seq_len, directions, batch, hidden],
                        "dtype": "float32",
                    }
                },
                "params": {"direction": direction},
            }
        ],
        "outputs": ["y"],
        "execution": {
            "stateless": {},
            "entrypoints": [
                {
                    "name": "evaluate",
                    "effect": 1,
                    "allowed_ranges": [1],
                    "inputs": ["x"],
                    "outputs": ["y"],
                }
            ],
        },
    }
    weights = {"w_tensor": w, "r_tensor": r, "b_tensor": b}
    return graph, weights, x


class TestPackedLSTM(unittest.TestCase):
    def test_pack_lstm_weight_shapes_and_tail_bits(self):
        rng = np.random.default_rng(123)
        for directions in (1, 2):
            for hidden in (4, 8):
                for width in (15, 33):  # non-multiples of 32 to test tail bits
                    w = rng.normal(0.0, 1.0, size=(directions, 4 * hidden, width)).astype(np.float32)
                    for bits in (2, 3, 4, 8):
                        for group in (32, 64):
                            words, scales = pack_lstm_weight(w, weight_bits=bits, group_size=group)
                            exp_words = (width * bits + 31) // 32
                            exp_groups = (width + group - 1) // group
                            self.assertEqual(words.shape, (directions, 4 * hidden, exp_words))
                            self.assertEqual(scales.shape, (directions, 4 * hidden, exp_groups))
                            self.assertEqual(words.dtype, np.int32)
                            self.assertEqual(scales.dtype, np.float32)
                            self.assertTrue(np.all(scales > 0))

                            shift = (width * bits) % 32
                            if shift != 0:
                                offset = 1 << (bits - 1)
                                per_word = 32 // bits if bits != 3 else 0
                                pad_word = 0 if bits == 3 else sum(offset << (i * bits) for i in range(per_word))
                                tail = (words[:, :, -1].view(np.uint32) >> shift)
                                expected_tail = (pad_word >> shift)
                                np.testing.assert_array_equal(tail, expected_tail)

    def test_pack_lstm_weights_graph_transformation(self):
        graph, weights, _ = make_lstm_fixture()
        packed_graph, packed_weights = pack_lstm_weights(graph, weights, weight_bits=4, group_size=32)

        node = packed_graph["nodes"][0]
        self.assertEqual(node["opType"], "PackedLSTM")
        self.assertEqual(node["params"]["weight_bits"], 4)
        self.assertEqual(node["params"]["group_size"], 32)
        self.assertEqual(node["params"]["direction"], 2)

        self.assertIn("scale", node["inputs"])
        self.assertIn("recurrent_scale", node["inputs"])
        self.assertIn(node["inputs"]["weight"], packed_weights)
        self.assertIn(node["inputs"]["scale"], packed_weights)
        self.assertIn(node["inputs"]["recurrent_weight"], packed_weights)
        self.assertIn(node["inputs"]["recurrent_scale"], packed_weights)
        self.assertIn("b_tensor", packed_weights)

        # Original unquantized float weights should be purged
        self.assertNotIn("w_tensor", packed_weights)
        self.assertNotIn("r_tensor", packed_weights)

    def test_packed_lstm_capabilities_and_descriptors(self):
        graph, weights, _ = make_lstm_fixture(width=15)
        packed_graph, packed_weights = pack_lstm_weights(graph, weights, weight_bits=4, group_size=32)

        desc_res = validate_runtime_descriptors(packed_graph, weights=packed_weights)
        self.assertTrue(desc_res.supported, [d.message for d in desc_res.diagnostics])

        graph_res = validate_graph(packed_graph, ("native-cpu", "wasm"), weights=packed_weights)
        self.assertTrue(graph_res.supported, [d.message for d in graph_res.diagnostics])

        # Test rejection of corrupted scales
        bad_weights = dict(packed_weights)
        scale_name = packed_graph["nodes"][0]["inputs"]["scale"]
        bad_scale = np.array(bad_weights[scale_name])
        bad_scale[0, 0, 0] = -0.5
        bad_weights[scale_name] = bad_scale
        bad_res = validate_runtime_descriptors(packed_graph, weights=bad_weights)
        self.assertFalse(bad_res.supported)
        self.assertIn("VXDESC_PACKED_SCALE", [d.code for d in bad_res.diagnostics])

        # Test rejection of corrupted tail bits
        bad_weights2 = dict(packed_weights)
        w_name = packed_graph["nodes"][0]["inputs"]["weight"]
        bad_w = np.array(bad_weights2[w_name])
        bad_w[0, 0, -1] ^= 0x7F000000
        bad_weights2[w_name] = bad_w
        bad_tail_res = validate_runtime_descriptors(packed_graph, weights=bad_weights2)
        self.assertFalse(bad_tail_res.supported)
        self.assertIn("VXDESC_PACKED_TAIL", [d.code for d in bad_tail_res.diagnostics])

        # Test rejection of 2-bit on webgpu backend
        packed_g2, packed_w2 = pack_lstm_weights(graph, weights, weight_bits=2, group_size=32)
        gpu_res = validate_graph(packed_g2, ("webgpu",), weights=packed_w2)
        self.assertFalse(gpu_res.supported)
        self.assertIn("VXCAP_PACKED_LOWBIT_BACKEND", [d.code for d in gpu_res.diagnostics])

    def test_packed_lstm_native_execution(self):
        # Run end-to-end forward execution in C runtime across W8 and W4
        for bits in (8, 4):
            with self.subTest(bits=bits):
                graph, weights, x = make_lstm_fixture(seq_len=4, batch=2, width=16, hidden=8, direction=2)
                packed_graph, packed_weights = pack_lstm_weights(graph, weights, weight_bits=bits, group_size=32)

                # Dequantize packed weights to run on an identical float LSTM reference
                node = packed_graph["nodes"][0]
                w_words = packed_weights[node["inputs"]["weight"]]
                w_scales = packed_weights[node["inputs"]["scale"]]
                r_words = packed_weights[node["inputs"]["recurrent_weight"]]
                r_scales = packed_weights[node["inputs"]["recurrent_scale"]]

                def unpack(words, scales, width, group_size):
                    d, gates_hidden, _ = words.shape
                    per_word = 32 // bits
                    offset = 1 << (bits - 1)
                    unpacked = np.zeros((d, gates_hidden, width), dtype=np.float32)
                    for di in range(d):
                        for gi in range(gates_hidden):
                            for col in range(width):
                                word_idx = (col * bits) // 32
                                shift = (col * bits) % 32
                                code = (int(words[di, gi, word_idx].view(np.uint32)) >> shift) & ((1 << bits) - 1)
                                q = code - offset
                                scale = scales[di, gi, col // group_size]
                                unpacked[di, gi, col] = float(q) * float(scale)
                    return unpacked

                w_dequant = unpack(w_words, w_scales, width=16, group_size=32)
                r_dequant = unpack(r_words, r_scales, width=8, group_size=32)

                ref_graph = copy.deepcopy(graph)
                ref_weights = {
                    "w_tensor": w_dequant,
                    "r_tensor": r_dequant,
                    "b_tensor": weights["b_tensor"],
                }

                with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
                    p1 = Path(tmp1)
                    (p1 / "graph.json").write_text(json.dumps(packed_graph))
                    (p1 / "model.safetensors").write_bytes(save(packed_weights))

                    p2 = Path(tmp2)
                    (p2 / "graph.json").write_text(json.dumps(ref_graph))
                    (p2 / "model.safetensors").write_bytes(save(ref_weights))

                    with vx.InferenceSession(p1, cpu_threads=1) as s1:
                        actual = s1.run({"x": x})["y"]

                    with vx.InferenceSession(p2, cpu_threads=1) as s2:
                        expected = s2.run({"x": x})["y"]

                    # PackedLSTM and reference dequantized LSTM must produce matching output
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
