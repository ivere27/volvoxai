"""BatchMatMul VJPs against an independent float64 finite-difference loss."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph
import volvoxai as vx


def cross_entropy(logits, targets):
    rows = logits.reshape(-1, logits.shape[-1])
    shifted = rows - rows.max(axis=1, keepdims=True)
    return np.mean(np.log(np.exp(shifted).sum(axis=1)) -
                   shifted[np.arange(len(rows)), targets.reshape(-1)])


class BatchMatMulBackward(unittest.TestCase):
    backend = 'cpu'

    def check_case(self, a_shape, b_shape, transpose=False, same_operand=False,
                   constant_b=False, fanout=False):
        rng = np.random.default_rng(113)
        a = rng.normal(0, .3, a_shape).astype(np.float32)
        b = a if same_operand else rng.normal(0, .3, b_shape).astype(np.float32)
        x = rng.uniform(.5, 1.5, a_shape).astype(np.float32)
        a64, b64 = a.astype(np.float64), b.astype(np.float64)

        def predict(aa, bb):
            left = aa * x
            right = left if same_operand else np.swapaxes(bb, -2, -1) if transpose else bb
            result = left @ right
            return result + result if fanout else result

        expected = predict(a64, b64)
        targets = (np.arange(expected.size // expected.shape[-1]) % expected.shape[-1]).reshape(expected.shape[:-1]).astype(np.int32)
        nodes = [{'id': 'input_weight', 'opType': 'Mul', 'params': {}, 'inputs': {'a': 'x', 'b': 'a'},
            'outputs': {'out': {'tensor': 'left', 'dtype': 'float32', 'shape': list(a_shape)}}}]
        right = 'left' if same_operand else 'b'
        if transpose:
            perm = list(range(b.ndim)); perm[-2], perm[-1] = perm[-1], perm[-2]
            nodes.append({'id': 'transpose', 'opType': 'Transpose', 'inputs': {'input': 'b'},
                'params': {'perm': perm}, 'outputs': {'out': {'tensor': 'right', 'dtype': 'float32',
                    'shape': list(np.swapaxes(b, -2, -1).shape)}}})
            right = 'right'
        nodes.append({'id': 'batched', 'opType': 'BatchMatMul', 'params': {},
            'inputs': {'a': 'left', 'b': right}, 'outputs': {'out': {'tensor': 'product' if fanout else 'logits',
                'dtype': 'float32', 'shape': list(expected.shape)}}})
        if fanout:
            # The shared gradient receives two Add contributions before BMM runs.
            nodes.append({'id': 'sum', 'opType': 'Add', 'params': {},
                'inputs': {'a': 'product', 'b': 'product'}, 'outputs': {'out': {
                    'tensor': 'logits', 'dtype': 'float32', 'shape': list(expected.shape)}}})
        graph = stateless_graph({'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'x': {'dtype': 'float32', 'shape': list(a_shape)}},
            'nodes': nodes, 'outputs': ['logits']})
        weights = {'a': a} if same_operand else {'a': a, 'b': b}
        trainable = ['a'] if constant_b else list(weights)
        gradients = {}
        for name, value in weights.items():
            if name not in trainable:
                continue
            numerical = np.empty(value.shape, np.float64)
            variable = a64 if name == 'a' else b64
            for index in np.ndindex(value.shape):
                saved = variable[index]
                variable[index] = saved + 1e-4
                upper = cross_entropy(predict(a64, b64), targets)
                variable[index] = saved - 1e-4
                lower = cross_entropy(predict(a64, b64), targets)
                variable[index] = saved
                numerical[index] = (upper - lower) / 2e-4
            gradients[name] = numerical
        with tempfile.TemporaryDirectory(prefix='volvoxai-bmm-backward-') as directory:
            model = Path(directory)
            (model / 'graph.json').write_text(json.dumps(graph))
            save_file(weights, model / 'model.safetensors')
            with vx.TrainingSession(model, loss=vx.CrossEntropyLoss('logits'),
                                   optimizer=vx.SGD(lr=.1), cpu_threads=1,
                                   backend=self.backend, trainable_names=trainable) as trainer:
                with trainer.step(x, targets, output_names=['logits']) as result:
                    self.assertEqual(result.backend, self.backend)
                    self.assertFalse(result.report.fallback.operator_fallback_used)
                    np.testing.assert_allclose(result['logits'].numpy(), expected, rtol=2e-6, atol=2e-7)
                    self.assertAlmostEqual(result.loss, cross_entropy(expected, targets), places=6)
                with trainer.parameters(list(weights)) as actual:
                    for name, original in weights.items():
                        if name not in trainable:
                            np.testing.assert_array_equal(actual[name].numpy(), original)
                            continue
                        gradient = (original.astype(np.float64) - actual[name].numpy()) / .1
                        np.testing.assert_allclose(gradient, gradients[name], rtol=3e-4, atol=7e-7,
                                                   err_msg=name)

    def test_rank_two(self):
        self.check_case((2, 3), (3, 4))

    def test_multi_batch_and_singleton_reduction(self):
        self.check_case((2, 2, 3), (1, 3, 4))

    def test_each_operand_broadcasts_a_different_axis(self):
        self.check_case((2, 1, 2, 3), (1, 3, 3, 4))

    def test_transpose_and_rank_broadcast(self):
        self.check_case((2, 3), (2, 4, 3), transpose=True)

    def test_shared_operand_sums_both_contributions(self):
        self.check_case((2, 2, 2), (2, 2, 2), same_operand=True)

    def test_rank_eight(self):
        self.check_case((2, 1, 1, 1, 1, 1, 2, 3), (1, 1, 1, 1, 1, 2, 3, 2))

    def test_constant_right_operand(self):
        self.check_case((2, 2, 3), (3, 4), constant_b=True)

    def test_shared_output_gradient_accumulation(self):
        self.check_case((2, 1, 2, 3), (1, 3, 3, 4), fanout=True)

    def test_partial_tiles(self):
        self.check_case((17, 19), (19, 21))


@unittest.skipUnless(os.environ.get('VOLVOXAI_TEST_NATIVE_GPU_BACKEND') == 'cuda',
                     'set VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda on physical hardware')
class CudaBatchMatMulBackward(BatchMatMulBackward):
    backend = 'cuda'


if __name__ == '__main__':
    unittest.main()
