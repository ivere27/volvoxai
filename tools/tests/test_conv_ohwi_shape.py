"""OHWI geometry matches HWIO after permuting the physical weight axes."""
import copy
import json
from pathlib import Path
import unittest

from tools.exporter.operator_shape_contracts import (
    OperatorShapeContractError, infer_concrete_operator_shapes, prove_operator_shape_domain,
)
from tools.exporter.shape_system import ShapeEnvironment


class ConvOHWIShape(unittest.TestCase):
    def test_geometry_and_refusal_vectors(self):
        path = Path(__file__).resolve().parents[2] / 'tests/contracts/shape_vectors/conv_ohwi_vectors.json'
        vectors = json.loads(path.read_text())
        for case in vectors['success_cases']:
            with self.subTest(case=case['id']):
                actual = infer_concrete_operator_shapes('Conv2D', case['request'])
                self.assertEqual(list(actual['out'].shape), case['expected']['out']['shape'])
                hwio = copy.deepcopy(case['request'])
                o, h, w, i = hwio['inputs']['weight']['shape']
                hwio['inputs']['weight']['shape'] = [h, w, i, o]
                hwio['params']['weight_layout'] = 'HWIO'
                self.assertEqual(actual, infer_concrete_operator_shapes('Conv2D', hwio))
        for case in vectors['failure_cases']:
            with self.subTest(case=case['id']):
                with self.assertRaises(OperatorShapeContractError) as error:
                    infer_concrete_operator_shapes('Conv2D', case['request'])
                self.assertEqual(error.exception.code, case['expected_error']['code'])

    def test_bounded_batch_spatial_domain_uses_kernel_axes(self):
        environment = ShapeEnvironment([
            {'name': 'B', 'min': 1, 'max': 3},
            {'name': 'H', 'min': 8, 'max': 12},
            {'name': 'W', 'min': 9, 'max': 15},
        ])
        request = {'environment': environment,
            'inputs': {'input': {'shape': ['B', 'H', 'W', 4], 'dtype': 'float32'},
                       'weight': {'shape': [6, 3, 2, 4], 'dtype': 'float32'}},
            'params': {'weight_layout': 'OHWI', 'pads': [2, 0, 2, 1],
                       'stride': [1, 1], 'dilation': [2, 1]}}
        self.assertTrue(prove_operator_shape_domain('Conv2D', request).supported)
        for b in (1, 3):
            for h in (8, 9, 12):
                for w in (9, 10, 15):
                    concrete = copy.deepcopy({key: value for key, value in request.items()
                                              if key != 'environment'})
                    concrete['inputs']['input']['shape'] = [b, h, w, 4]
                    result = infer_concrete_operator_shapes('Conv2D', concrete)
                    self.assertEqual(result['out'].shape, (b, h, w, 6))
        # The v1 symbolic contract still refuses nonconstant floor-window
        # expressions; admitting OHWI must not weaken that proof boundary.
        request['params']['stride'] = [2, 3]
        self.assertFalse(prove_operator_shape_domain('Conv2D', request).supported)


if __name__ == '__main__':
    unittest.main()
