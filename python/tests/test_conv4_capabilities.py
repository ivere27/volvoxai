"""Packed QConv4 descriptor and backend proof must inspect logical nibble codes."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.exporter.capabilities import classify_package, validate_graph, validate_runtime_descriptors


def fixture():
    graph = {'format': 'volvox-graph/v1', 'dimensions': {},
        'inputs': {'x': {'shape': [1, 1, 3, 1], 'dtype': 'int8'}},
        'nodes': [{'id': 'conv4', 'opType': 'QConv2D', 'inputs': {'input': 'x', 'weight': 'w'},
            'outputs': {'out': {'tensor': 'y', 'shape': [1, 1, 1, 1], 'dtype': 'int8'}},
            'params': {'weight_bits': 4, 'kernel_size': [1, 3], 'data_layout': 'NHWC', 'weight_layout': 'OHWI'}}],
        'outputs': ['y'], 'quantization': {'format': 'volvox-affine-safetensors/v1', 'tensors': {
            'x': {'scheme': 'per_tensor', 'scale_tensor': 'xs', 'zero_point_tensor': 'xz'},
            'w': {'scheme': 'per_axis', 'axis': 0, 'scale_tensor': 'ws', 'zero_point_tensor': 'wz'},
            'y': {'scheme': 'per_tensor', 'scale_tensor': 'ys', 'zero_point_tensor': 'yz'}}},
        'execution': {'stateless': {}, 'entrypoints': [{'name': 'evaluate', 'effect': 1,
            'allowed_ranges': [1], 'inputs': ['x'], 'outputs': ['y']}]}}
    weights = {'w': np.array([[0xf0, 0x88]], np.uint8),
        'xs': np.array([1], np.float32), 'xz': np.array([0], np.int8),
        'ws': np.array([.05], np.float32), 'wz': np.array([8], np.uint8),
        'ys': np.array([1], np.float32), 'yz': np.array([0], np.int8)}
    return graph, weights


class Conv4CapabilitiesTest(unittest.TestCase):
    def test_cpu_wasm_and_honest_package_classification(self):
        graph, weights = fixture()
        self.assertEqual(classify_package(graph, weights), 'hybrid')
        result = validate_runtime_descriptors(graph, weights=weights)
        self.assertTrue(result.supported, result.diagnostics)
        result = validate_graph(graph, ('native-cpu', 'wasm'), weights=weights)
        self.assertTrue(result.supported, result.diagnostics)
        result = validate_graph(graph, ('webgpu',), weights=weights)
        self.assertIn('VXCAP_CONV4_BACKEND', [item.code for item in result.diagnostics])

    def test_tail_zero_and_geometry_refusals(self):
        for kind in ('tail', 'zero', 'shape', 'kernel', 'dtype'):
            with self.subTest(kind=kind):
                graph, weights = fixture()
                if kind == 'tail':
                    weights['w'][0, 1] = 0x78
                elif kind == 'zero':
                    weights['wz'][0] = 7
                elif kind == 'shape':
                    weights['w'] = np.array([[0xf0, 0x88, 0x88]], np.uint8)
                elif kind == 'kernel':
                    del graph['nodes'][0]['params']['kernel_size']
                else:
                    weights['w'] = weights['w'].astype(np.int8)
                result = validate_runtime_descriptors(graph, weights=weights)
                self.assertFalse(result.supported)

    def test_bounds_use_i4_domain_instead_of_physical_bytes(self):
        graph, weights = fixture()
        graph['nodes'][0]['inputs']['bias'] = 'bias'
        # Admission reserves the full I4 domain (3*8*128) plus bias.
        # Physical bytes240,136 would yield an incorrect larger bound.
        weights['bias'] = np.array([2**31 - 1 - 3*8*128], np.int32)
        self.assertTrue(validate_runtime_descriptors(graph, weights=weights).supported)
        weights['bias'] += 1
        result = validate_runtime_descriptors(graph, weights=weights)
        self.assertIn('VXDESC_QCONV_BOUND', [item.code for item in result.diagnostics])

    def test_weight_only_packed_conv4_and_subchannel_group_scales(self):
        import json
        import tempfile
        from safetensors.numpy import save
        sys.path.insert(0, str(ROOT / "python"))
        import volvoxai as vx

        rng = np.random.default_rng(42)
        out_c, kh, kw, in_c = 16, 1, 3, 24  # terms = 72 -> 3 groups of 32
        terms = kh * kw * in_c
        row_bytes = (terms + 1) // 2
        codes = rng.integers(-8, 8, size=(out_c, terms), dtype=np.int8)
        packed = np.full((out_c, row_bytes), 0x88, dtype=np.uint8)
        u_codes = (codes.astype(np.int32) + 8).astype(np.uint8)
        packed[:, :] = u_codes[:, 0::2] | (packed & 0xF0)
        packed[:, :terms // 2] = (u_codes[:, 1::2] << 4) | (packed[:, :terms // 2] & 0x0F)
        bias = rng.normal(0.0, 0.1, size=(out_c,)).astype(np.float32)
        x = rng.normal(0.0, 0.5, size=(1, 1, 8, in_c)).astype(np.float32)

        for group_size, scale_dtype in ((0, np.float32), (32, np.float32), (32, np.float16)):
            with self.subTest(group_size=group_size, scale_dtype=str(scale_dtype)):
                if group_size == 0:
                    scales = rng.uniform(0.01, 0.08, size=(out_c,)).astype(scale_dtype)
                    w_f32 = (codes.astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(out_c, kh, kw, in_c)
                else:
                    num_groups = (terms + group_size - 1) // group_size
                    scales = rng.uniform(0.01, 0.08, size=(out_c, num_groups)).astype(scale_dtype)
                    expanded_scales = np.repeat(scales.astype(np.float32), group_size, axis=1)[:, :terms]
                    w_f32 = (codes.astype(np.float32) * expanded_scales).reshape(out_c, kh, kw, in_c)

                expected = np.empty((1, 1, 6, out_c), dtype=np.float32)
                for ox in range(6):
                    patch = x[0, 0, ox:ox + kw, :]  # [kw, in_c]
                    expected[0, 0, ox, :] = np.einsum("oki,ki->o", w_f32[:, 0, :, :], patch) + bias

                graph = {
                    "format": "volvox-graph/v1",
                    "dimensions": {},
                    "inputs": {"x": {"shape": [1, 1, 8, in_c], "dtype": "float32"}},
                    "nodes": [{
                        "id": "qconv_wo",
                        "opType": "QConv2D",
                        "inputs": {
                            "input": "x",
                            "weight": "w",
                            "weight_scale": "ws",
                            "bias": "b",
                        },
                        "outputs": {"out": {"tensor": "y", "shape": [1, 1, 6, out_c], "dtype": "float32"}},
                        "params": {
                            "weight_only": True,
                            "weight_bits": 4,
                            "kernel_size": [kh, kw],
                            "data_layout": "NHWC",
                            "weight_layout": "OHWI",
                            "stride": [1, 1],
                            "dilation": [1, 1],
                            "pads": [0, 0, 0, 0],
                            "groups": 1,
                        },
                    }],
                    "outputs": ["y"],
                    "execution": {
                        "stateless": {},
                        "entrypoints": [{
                            "name": "evaluate",
                            "effect": 1,
                            "allowed_ranges": [1],
                            "inputs": ["x"],
                            "outputs": ["y"],
                        }],
                    },
                }
                weights = {"w": packed, "ws": scales, "b": bias}
                res = validate_runtime_descriptors(graph, weights=weights)
                self.assertTrue(res.supported, res.diagnostics)
                with tempfile.TemporaryDirectory() as tmp:
                    d = Path(tmp)
                    (d / "graph.json").write_text(json.dumps(graph))
                    (d / "model.safetensors").write_bytes(save(weights))
                    with vx.InferenceSession(d, cpu_threads=1) as sess:
                        actual = sess.run({"x": x})["y"]
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
