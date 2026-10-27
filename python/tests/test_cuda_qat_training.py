"""Explicit physical CUDA qualification; opt in with VOLVOXAI_TEST_CUDA_QAT=1.

The backend request is exact: unavailable CUDA fails, never becomes a skip or
CPU fallback. CPU reference sessions use the same immutable original package.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file, load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import volvoxai as vx
from execution_fixture import stateless_graph
from test_qat_training import fit_rows, qat


@unittest.skipUnless(os.environ.get('VOLVOXAI_TEST_CUDA_QAT') == '1',
                     'Set VOLVOXAI_TEST_CUDA_QAT=1 for required CUDA execution')
class CudaQatTraining(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='volvoxai-cuda-qat-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.rng = np.random.default_rng(813)

    def write(self, nodes, shape, weights):
        graph = stateless_graph({'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'x': {'dtype': 'float32', 'shape': shape}},
            'nodes': nodes, 'outputs': ['logits']})
        (self.model / 'graph.json').write_text(json.dumps(graph))
        save_file(weights, self.model / 'model.safetensors')

    def trainer(self, backend, policy=None, **kwargs):
        loss = kwargs.pop('loss', vx.CrossEntropyLoss('logits'))
        return vx.TrainingSession(self.model, backend=backend, cpu_threads=1,
            loss=loss, qat=policy, **kwargs)

    @staticmethod
    def values(trainer, names):
        with trainer.parameters(names) as values:
            return {name: values[name].numpy().copy() for name in names}

    @staticmethod
    def state(trainer):
        return trainer._engine.get_trainer_state(vx.pb.TrainerRef(trainer_id=trainer._trainer.trainer_id))

    def linear(self, group, layout):
        k, n = group + 1, 3
        canonical = self.rng.normal(0, .2, (n, k)).astype(np.float32)
        canonical[:, 0] = [3, -3, 2]
        first = self.rng.normal(0, .1, (2, k)).astype(np.float32)
        raw = canonical.T.copy() if layout == 'din_dout' else canonical.copy()
        self.write([
            {'id': 'first', 'opType': 'Linear', 'inputs': {'input': 'x', 'weight': 'first'},
             'outputs': {'out': {'tensor': 'hidden', 'dtype': 'float32', 'shape': [2, k]}},
             'params': {'weight_layout': 'din_dout'}},
            {'id': 'projection', 'opType': 'Linear', 'inputs': {'input': 'hidden', 'weight': 'weight'},
             'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': [2, n]}},
             'params': {'weight_layout': layout}}], [2, 2], {'weight': raw, 'first': first})
        return canonical, first, raw

    def test_linear_bits_groups_layouts_fake_forward_and_master_ste(self):
        x = np.array([[.4, -.2], [.1, .7]], np.float32)
        labels = [0, 2]
        for bits in (2, 3, 4):
            for group in (32, 64, 128, 256):
                for layout in ('din_dout', 'dout_din'):
                    for algorithm in ('rtn', 'mse'):
                        with self.subTest(bits=bits, group=group, layout=layout, algorithm=algorithm):
                            canonical, first, raw = self.linear(group, layout)
                            fake, mask = fit_rows(canonical, bits, group, mse=algorithm == 'mse')
                            hidden = x @ first
                            expected_logits = hidden @ fake.T
                            p = np.exp(expected_logits.astype(np.float64) - expected_logits.max(axis=1, keepdims=True))
                            p /= p.sum(axis=1, keepdims=True)
                            p[np.arange(2), labels] -= 1
                            p /= 2
                            delta = (p.T @ hidden) * mask
                            expected_weight = raw - .05 * (delta.T if layout == 'din_dout' else delta)
                            expected_first = first - .05 * (x.T @ (p @ fake))
                            with self.trainer('cuda', qat(bits, group, algorithm), optimizer=vx.SGD(lr=.05)) as trainer:
                                with trainer.step(x, labels, output_names=['logits']) as result:
                                    self.assertEqual(result.backend, 'cuda')
                                    np.testing.assert_allclose(result['logits'].numpy(), expected_logits, rtol=4e-5, atol=3e-6)
                                values = self.values(trainer, ['weight', 'first'])
                                np.testing.assert_allclose(values['weight'], expected_weight, rtol=4e-5, atol=4e-6)
                                np.testing.assert_allclose(values['first'], expected_first, rtol=4e-5, atol=4e-6)
                                np.testing.assert_array_equal(values['weight'][~(mask.T if layout == 'din_dout' else mask)],
                                                              raw[~(mask.T if layout == 'din_dout' else mask)])
                                self.assertGreater(self.state(trainer).qat_workspace_high_water_bytes, 0)

    def test_conv_bits_layouts_and_mixed_linear_cpu_parity(self):
        for bits in (2, 3, 4):
            for layout in ('HWIO', 'OHWI'):
                for algorithm in ('rtn', 'mse'):
                    with self.subTest(bits=bits, layout=layout, algorithm=algorithm):
                        canonical = self.rng.normal(0, .2, (4, 3)).astype(np.float32)
                        raw = canonical.T.reshape(1, 1, 3, 4).copy() if layout == 'HWIO' else canonical.reshape(4, 1, 1, 3).copy()
                        head = self.rng.normal(0, .1, (4, 3)).astype(np.float32)
                        self.write([
                            {'id': 'conv', 'opType': 'Conv2D', 'inputs': {'input': 'x', 'weight': 'conv'},
                             'outputs': {'out': {'tensor': 'features', 'dtype': 'float32', 'shape': [2, 1, 1, 4]}},
                             'params': {'data_layout': 'NHWC', 'weight_layout': layout}},
                            {'id': 'flat', 'opType': 'Reshape', 'inputs': {'input': 'features'},
                             'outputs': {'out': {'tensor': 'flat', 'dtype': 'float32', 'shape': [2, 4]}}, 'params': {'shape': [2, 4]}},
                            {'id': 'projection', 'opType': 'Linear', 'inputs': {'input': 'flat', 'weight': 'head'},
                             'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': [2, 3]}},
                             'params': {'weight_layout': 'din_dout'}}], [2, 1, 1, 3], {'conv': raw, 'head': head})
                        policy = qat(bits, 0, algorithm, 'conv')
                        policy.nodes.extend(qat(3 if bits != 3 else 2, 32, algorithm).nodes)
                        x = self.rng.normal(0, .4, (2, 1, 1, 3)).astype(np.float32)
                        values, outputs = {}, {}
                        for backend in ('cpu', 'cuda'):
                            with self.trainer(backend, policy, optimizer=vx.SGD(lr=.05)) as trainer:
                                with trainer.step(x, [0, 2], output_names=['logits']) as result:
                                    self.assertEqual(result.backend, backend)
                                    outputs[backend] = result['logits'].numpy().copy()
                                values[backend] = self.values(trainer, ['conv', 'head'])
                        np.testing.assert_allclose(outputs['cuda'], outputs['cpu'], rtol=4e-5, atol=3e-6)
                        for name in values['cpu']:
                            np.testing.assert_allclose(values['cuda'][name], values['cpu'][name], rtol=4e-5, atol=4e-6)

    def test_expand_parameter_gradients_axes_tails_and_shared_accumulation(self):
        for source_shape, output_shape in (
                ((3,), (2, 73, 3)), ((73, 1, 3), (2, 73, 5, 3)), ((1, 3), (2, 3)),
                ((2, 1, 3), (2, 5, 3)), ((1, 3, 1), (2, 3, 3)),
                ((1, 1, 1, 1, 1, 1, 3), (2, 1, 2, 1, 2, 1, 2, 3)),
                ((2, 3), (2, 3))):
            with self.subTest(source=source_shape, expanded=output_shape):
                shift = self.rng.normal(0, .1, source_shape).astype(np.float32)
                weight = self.rng.normal(0, .2, (3, 3)).astype(np.float32)
                nodes = [
                    {'id': f'expand-{i}', 'opType': 'Expand', 'inputs': {'input': 'shift'},
                     'outputs': {'out': {'tensor': f'expanded-{i}', 'dtype': 'float32', 'shape': list(output_shape)}},
                     'params': {'shape': list(output_shape)}} for i in range(2)]
                nodes += [
                    {'id': 'sum', 'opType': 'Add', 'inputs': {'a': 'expanded-0', 'b': 'expanded-1'},
                     'outputs': {'out': {'tensor': 'expanded', 'dtype': 'float32', 'shape': list(output_shape)}}, 'params': {}},
                    {'id': 'add', 'opType': 'Add', 'inputs': {'a': 'x', 'b': 'expanded'},
                     'outputs': {'out': {'tensor': 'features', 'dtype': 'float32', 'shape': list(output_shape)}}, 'params': {}},
                    {'id': 'projection', 'opType': 'Linear', 'inputs': {'input': 'features', 'weight': 'weight'},
                     'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': list(output_shape)}},
                     'params': {'weight_layout': 'dout_din'}}]
                self.write(nodes, list(output_shape), {'shift': shift, 'weight': weight})
                x = self.rng.normal(0, .3, output_shape).astype(np.float32)
                labels = np.arange(int(np.prod(output_shape[:-1])), dtype=np.int32) % 3
                weights, logits = {}, {}
                for backend in ('cpu', 'cuda'):
                    with self.trainer(backend, qat(3, 32, 'mse'), optimizer=vx.SGD(lr=.05)) as trainer:
                        for _ in range(2):
                            with trainer.step(x, labels, output_names=['logits']) as result:
                                self.assertEqual(result.backend, backend)
                                logits[backend] = result['logits'].numpy().copy()
                        weights[backend] = self.values(trainer, ['shift', 'weight'])
                np.testing.assert_allclose(logits['cuda'], logits['cpu'], rtol=4e-5, atol=3e-6)
                for name in weights['cpu']:
                    np.testing.assert_allclose(weights['cuda'][name], weights['cpu'][name], rtol=4e-5, atol=4e-6)
                self.assertFalse(np.array_equal(weights['cuda']['shift'], shift))

    def test_checkpoint_resume_rollback_accumulation_and_failure(self):
        self.linear(32, 'din_dout')
        source = (self.model / 'model.safetensors').read_bytes()
        x = np.array([[.4, -.2], [.1, .7]], np.float32)
        checkpoint = self.root / 'qat.pb'
        policy = qat(2, 32, 'mse')
        # Two microbatches each contain two valid labels. The public API
        # requires their full-window normalizer explicitly for accumulation.
        loss = vx.CrossEntropyLoss('logits', normalizer=4)
        with self.trainer('cuda', policy, optimizer=vx.AdamW(lr=.003), loss=loss) as trainer:
            trainer.step(x, [0, 2], accumulation_steps=2)
            with self.assertRaises(vx.VolvoxAIError): trainer.save_checkpoint(self.root / 'incomplete.pb')
            trainer.step(x, [0, 2], accumulation_steps=2)
            trainer.save_checkpoint(checkpoint)
            saved = self.values(trainer, ['weight', 'first'])
            trainer.step(x, [0, 2])
            continued = self.values(trainer, ['weight', 'first'])
            trainer.save_checkpoint(self.root / 'continued.pb')
            trainer.rollback()
            initial = load_file(self.model / 'model.safetensors')
            for name, value in self.values(trainer, ['weight', 'first']).items():
                np.testing.assert_array_equal(value, initial[name])
        with self.trainer('cuda', checkpoint=checkpoint, loss=loss) as trainer:
            trainer.step(x, [0, 2])
            resumed = self.values(trainer, ['weight', 'first'])
            trainer.save_checkpoint(self.root / 'resumed.pb')
            for name in continued: np.testing.assert_array_equal(resumed[name], continued[name])
            a = vx.pb.TrainerCheckpoint.FromString((self.root / 'continued.pb').read_bytes())
            b = vx.pb.TrainerCheckpoint.FromString((self.root / 'resumed.pb').read_bytes())
            self.assertEqual(a.optimizer, b.optimizer)
            self.assertEqual(a.qat, b.qat)
            trainer.rollback()
            for name, value in self.values(trainer, ['weight', 'first']).items():
                np.testing.assert_array_equal(value, saved[name])
            bad = x.copy(); bad[0, 0] = np.nan
            with self.assertRaises(vx.VolvoxAIError): trainer.step(bad, [0, 2])
            trainer.rollback()
            for name, value in self.values(trainer, ['weight', 'first']).items():
                np.testing.assert_array_equal(value, saved[name])
        self.assertEqual((self.model / 'model.safetensors').read_bytes(), source)


if __name__ == '__main__':
    unittest.main()
