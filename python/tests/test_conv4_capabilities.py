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


if __name__ == '__main__':
    unittest.main()
