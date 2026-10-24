"""Low-bit physical storage, arithmetic references and GPU admission contracts."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_conv4_capabilities import fixture
from tools.exporter.capabilities import classify_package, validate_graph, validate_runtime_descriptors
from tools.exporter.errors import ExporterError
from tools.exporter.operator_shape_contracts import infer_concrete_operator_shapes, OperatorShapeContractError
from tools.exporter.packed_linear import pack_weight, packed_linear_reference


def pack_codes(codes, bits, width):
    # Set each bit separately, independently of the production packer.
    result = np.zeros((len(codes), (codes.shape[1] * bits + width - 1) // width),
                      np.uint32 if width == 32 else np.uint8)
    zero = 1 << (bits - 1)
    count = codes.shape[1] if bits == 3 else result.shape[1] * width // bits
    for row in range(len(codes)):
        for k in range(count):
            value = int(codes[row, k]) + zero if k < codes.shape[1] else zero
            for b in range(bits):
                if value & (1 << b):
                    bit = k * bits + b
                    result[row, bit // width] |= 1 << (bit % width)
    return result


class LowbitCapabilitiesTest(unittest.TestCase):
    def test_shape_vector_corpus_matches_python(self):
        vectors = json.loads((ROOT / 'tests/contracts/shape_vectors/lowbit_packed_vectors.json').read_text())
        for case in vectors['success_cases']:
            with self.subTest(case=case['id']):
                actual = infer_concrete_operator_shapes(case['operators'][0], case['request'])
                for port, expected in case['expected'].items():
                    self.assertEqual(list(actual[port].shape), expected['shape'])
                    self.assertEqual(actual[port].dtype, expected['dtype'])
        for case in vectors['failure_cases']:
            with self.subTest(case=case['id']):
                with self.assertRaises(OperatorShapeContractError) as failed:
                    infer_concrete_operator_shapes(case['operators'][0], case['request'])
                self.assertEqual(failed.exception.code, case['expected_error']['code'])

    def test_conv_storage_tail_magnitude_and_backend_proof(self):
        for bits in (2, 3):
            graph, weights = fixture()
            graph['nodes'][0]['params']['weight_bits'] = bits
            weights['w'] = pack_codes(np.array([[-2, 1, 0]], np.int8), bits, 8)
            weights['wz'][0] = 1 << (bits - 1)
            with self.subTest(bits=bits):
                self.assertEqual(classify_package(graph, weights), 'hybrid')
                result = validate_graph(graph, ('native-cpu', 'wasm'), weights=weights)
                self.assertTrue(result.supported, result.diagnostics)
                for target in ('webgpu', 'backend:cuda', 'backend:metal', 'backend:vulkan'):
                    result = validate_graph(graph, (target,), weights=weights)
                    self.assertIn('VXCAP_CONV4_BACKEND', [item.code for item in result.diagnostics])
                graph['nodes'][0]['inputs']['bias'] = 'bias'
                bound = 3 * (1 << (bits - 1)) * 128
                weights['bias'] = np.array([2**31 - 1 - bound], np.int32)
                self.assertTrue(validate_runtime_descriptors(graph, weights=weights).supported)
                weights['bias'] += 1
                self.assertFalse(validate_runtime_descriptors(graph, weights=weights).supported)
                del graph['nodes'][0]['inputs']['bias']
                weights['w'][0, -1] ^= 1 << (3 * bits % 8)
                self.assertFalse(validate_runtime_descriptors(graph, weights=weights).supported)

    def test_linear_gpu_refusal_preserves_cpu_storage_support(self):
        for bits in (2,3):
            graph = {'format':'volvox-graph/v1','dimensions':{},
                'inputs':{'x':{'shape':[2,33],'dtype':'float32'}},
                'nodes':[{'id':'packed','opType':'PackedLinear',
                    'inputs':{'input':'x','weight':'w','scale':'s'},
                    'outputs':{'out':{'tensor':'y','shape':[2,3],'dtype':'float32'}},
                    'params':{'weight_bits':bits,'group_size':32}}],
                'outputs':['y'],'execution':{'stateless':{},'entrypoints':[
                    {'name':'evaluate','effect':1,'allowed_ranges':[1],'inputs':['x'],'outputs':['y']}]}}
            words,scales = pack_weight(np.ones((3,33),np.float32),weight_bits=bits,group_size=32)
            weights={'w':words,'s':scales}
            result=validate_graph(graph,('native-cpu','wasm'),weights=weights)
            self.assertTrue(result.supported,result.diagnostics)
            for target in ('webgpu','backend:cuda','backend:metal','backend:vulkan'):
                result=validate_graph(graph,(target,),weights=weights)
                self.assertIn('VXCAP_PACKED_LOWBIT_BACKEND',[item.code for item in result.diagnostics])

    def test_linear_lowbit_packing_and_reference(self):
        for bits in (2, 3):
            for k in (1, 10, 11, 21, 22, 31, 32, 33, 65):
                for group in (32, 64, 128, 256):
                    high = (1 << (bits - 1)) - 1
                    # Exact codes with maximum=high keep RTN scale exactly one.
                    codes = ((np.arange(3*k).reshape(3,k) % (2*high+1)) - high).astype(np.int8)
                    codes[:, ::group] = high
                    source = codes.astype(np.float32)
                    words, scales = pack_weight(source, weight_bits=bits, group_size=group)
                    expected = pack_codes(codes, bits, 32)
                    np.testing.assert_array_equal(words.view(np.uint32), expected)
                    np.testing.assert_array_equal(scales, np.ones_like(scales))
                    x = np.ones((2,k), np.float32)
                    for arithmetic in (8,16,32):
                        actual = packed_linear_reference(x, words, scales, weight_bits=bits,
                                                        group_size=group, activation_bits=arithmetic)
                        np.testing.assert_array_equal(actual, np.tile(codes.astype(np.float32).sum(1),(2,1)))
                    if k*bits%32:
                        words[0,-1] = np.array(int(words[0,-1]) ^ (1 << (k*bits%32))).astype(np.int32)
                        with self.assertRaises(ExporterError):
                            packed_linear_reference(x, words, scales, weight_bits=bits, group_size=group)


if __name__ == '__main__':
    unittest.main()
