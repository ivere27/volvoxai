"""Weight-only QAT: independent fake-forward/STE and master lifecycle checks."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph
import volvoxai as vx

pb = vx.pb


def qat(bits=2, group=32, algorithm='rtn', node='projection'):
    return pb.QatOptions(nodes=[pb.QatNode(node_id=node,
        weights=pb.WeightQuantization(format=pb.WeightFormat(bits=bits, group_size=group),
            algorithm=pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN if algorithm == 'rtn'
            else pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP))])


def fit_rows(original, bits, group=32, mse=False, conv=False):
    """Scalar scale search, independent of C fitting and packed storage."""
    original = np.ascontiguousarray(original, np.float32)
    fake = np.empty_like(original)
    mask = np.empty(original.shape, bool)
    high, low = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    width = original.shape[1] if conv else group
    for row in range(len(original)):
        for start in range(0, original.shape[1], width):
            values = original[row, start:start + width]
            maximum = max(abs(float(v)) for v in values)
            base = np.float32(maximum / high if maximum else 1)
            best, chosen = float('inf'), base
            steps = range((99 if bits == 2 else 50 if bits == 3 or conv else 25) + 1) if mse else [0]
            for step in steps:
                fraction = np.float32((100 - step) / 100) if bits <= 3 or conv else np.float32(1) - np.float32(step) * np.float32(.01)
                scale = np.float32(base * fraction)
                codes = [max(low, min(high, round(float(np.float32(v / scale))))) for v in values]
                restored = [np.float32(q * scale) for q in codes]
                error = sum((float(v) - float(r)) ** 2 for v, r in zip(values, restored))
                if error < best:
                    best, chosen = error, scale
            for k, value in enumerate(values, start):
                ratio = np.float32(value / chosen)
                fake[row, k] = np.float32(max(low, min(high, round(float(ratio)))) * chosen)
                mask[row, k] = low <= ratio <= high
    return fake, mask


class QatTraining(unittest.TestCase):
    def setUp(self):
        try:
            vx.find_library()
        except vx.VolvoxAIError as error:
            self.skipTest(str(error))
        temp = tempfile.TemporaryDirectory(prefix='volvoxai-qat-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / 'source'
        self.model.mkdir()

    def linear(self, weight, first=None, node='projection'):
        k, n = weight.shape
        nodes = []
        tensors = {'weight': weight}
        input_width = k if first is None else first.shape[0]
        port = 'x'
        if first is not None:
            tensors['first'] = first
            nodes.append({'id': 'first', 'opType': 'Linear', 'inputs': {'input': 'x', 'weight': 'first'},
                'outputs': {'out': {'tensor': 'hidden', 'dtype': 'float32', 'shape': [1, k]}},
                'params': {'weight_layout': 'din_dout'}})
            port = 'hidden'
        nodes.append({'id': node, 'opType': 'Linear', 'inputs': {'input': port, 'weight': 'weight'},
            'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': [1, n]}},
            'params': {'weight_layout': 'din_dout'}})
        self.write({'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'x': {'dtype': 'float32', 'shape': [1, input_width]}},
            'nodes': nodes, 'outputs': ['logits']}, tensors)

    def write(self, graph, tensors):
        (self.model / 'graph.json').write_text(json.dumps(stateless_graph(graph)))
        save_file(tensors, self.model / 'model.safetensors')

    def trainer(self, **options):
        return vx.TrainingSession(self.model, loss=vx.CrossEntropyLoss('logits'), cpu_threads=1, **options)

    @staticmethod
    def state(trainer):
        return trainer._engine.get_trainer_state(pb.TrainerRef(trainer_id=trainer._trainer.trainer_id))

    @staticmethod
    def values(trainer, names=('weight',)):
        with trainer.parameters(names) as tensors:
            return {name: tensors[name].numpy().copy() for name in names}

    def test_independent_forward_weight_ste_and_fake_input_gradient(self):
        rng = np.random.default_rng(19)
        weight = rng.normal(0, .2, (33, 2)).astype(np.float32)
        first = rng.normal(0, .1, (2, 33)).astype(np.float32)
        x = np.array([[.6, -.3]], np.float32)
        for bits in (2, 3, 4):
            with self.subTest(bits=bits):
                self.linear(weight, first)
                fake, mask = fit_rows(weight.T, bits)
                hidden = x @ first
                logits = hidden @ fake.T
                probabilities = np.exp(logits.astype(np.float64) - logits.max())
                probabilities /= probabilities.sum()
                dy = probabilities.copy(); dy[0, 1] -= 1
                expected_w = weight - .125 * (hidden.T @ dy) * mask.T
                expected_first = first - .125 * (x.T @ (dy @ fake))
                with self.trainer(qat=qat(bits), optimizer=vx.SGD(lr=.125)) as trainer:
                    with trainer.step(x, [1], output_names=['logits']) as result:
                        np.testing.assert_allclose(result['logits'].numpy(), logits, rtol=2e-6, atol=2e-7)
                    actual = self.values(trainer, ('weight', 'first'))
                    np.testing.assert_allclose(actual['weight'], expected_w, rtol=2e-6, atol=2e-7)
                    np.testing.assert_allclose(actual['first'], expected_first, rtol=2e-6, atol=2e-7)
                    self.assertGreater(self.state(trainer).qat_workspace_high_water_bytes, 0)

    def test_mse_clipping_ste_preserves_outlier_masters(self):
        weight = np.ones((32, 2), np.float32)
        weight[:, 1] *= -.8
        weight[0] = [4, -4]
        self.linear(weight)
        x = np.full((1, 32), .03, np.float32)
        fake, mask = fit_rows(weight.T, 2, mse=True)
        self.assertTrue(np.any(~mask))
        logits = x @ fake.T
        p = np.exp(logits.astype(np.float64) - logits.max()); p /= p.sum(); p[0, 1] -= 1
        expected = weight - .2 * (x.T @ p) * mask.T
        with self.trainer(qat=qat(algorithm='mse'), optimizer=vx.SGD(lr=.2)) as trainer:
            with trainer.step(x, [1], output_names=['logits']) as result:
                np.testing.assert_allclose(result['logits'].numpy(), logits, atol=2e-6)
            actual = self.values(trainer)['weight']
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
            np.testing.assert_array_equal(actual[~mask.T], weight[~mask.T])

    def test_conv_output_channel_fake_forward_and_ste(self):
        canonical = np.array([[.9, -.4, .2], [-.7, .3, -.1]], np.float32)
        raw = canonical.T.reshape(1, 1, 3, 2).copy()
        graph = {'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'x': {'dtype': 'float32', 'shape': [1, 1, 1, 3]}},
            'nodes': [{'id': 'conv', 'opType': 'Conv2D', 'inputs': {'input': 'x', 'weight': 'weight'},
                'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': [1, 1, 1, 2]}},
                'params': {'data_layout': 'NHWC', 'weight_layout': 'HWIO'}}], 'outputs': ['logits']}
        x = np.array([[[[.5, -.2, .7]]]], np.float32)
        for bits in (2, 3, 4):
            with self.subTest(bits=bits):
                self.write(graph, {'weight': raw})
                fake, mask = fit_rows(canonical, bits, conv=True)
                logits = x.reshape(1, 3) @ fake.T
                p = np.exp(logits.astype(np.float64) - logits.max()); p /= p.sum(); p[0, 0] -= 1
                expected = raw - (.1 * (x.reshape(1, 3).T @ p) * mask.T).reshape(raw.shape)
                with self.trainer(qat=qat(bits, 0, node='conv'), optimizer=vx.SGD(lr=.1)) as trainer:
                    with trainer.step(x, [0], output_names=['logits']) as result:
                        np.testing.assert_allclose(result['logits'].numpy().reshape(1, 2), logits, atol=2e-6)
                    np.testing.assert_allclose(self.values(trainer)['weight'], expected, atol=2e-7)

    def test_loss_recovery_checkpoint_resume_and_rollback(self):
        weight = np.full((32, 2), .1, np.float32); weight[0] = [.3, -.2]
        self.linear(weight)
        original = (self.model / 'model.safetensors').read_bytes()
        x = np.zeros((1, 32), np.float32); x[0, 0] = 1
        checkpoint = self.root / 'qat.pb'
        with self.trainer(qat=qat(), optimizer=vx.AdamW(lr=.05, weight_decay=0)) as trainer:
            losses = [trainer.step(x, [1]).loss for _ in range(30)]
            self.assertLess(losses[-1], losses[0] * .5)
            trainer.save_checkpoint(checkpoint)
            saved = self.values(trainer)['weight']
            trainer.step(x, [1]); continued = self.values(trainer)['weight']
            trainer.rollback()
            np.testing.assert_array_equal(self.values(trainer)['weight'], weight)
        payload = pb.TrainerCheckpoint.FromString(checkpoint.read_bytes())
        self.assertEqual(payload.qat, qat())
        with self.trainer(checkpoint=checkpoint) as resumed:
            self.assertEqual(self.state(resumed).qat, qat())
            resumed.step(x, [1])
            np.testing.assert_array_equal(self.values(resumed)['weight'], continued)
            resumed.rollback()
            np.testing.assert_array_equal(self.values(resumed)['weight'], saved)
        with self.assertRaises(vx.VolvoxAIError):
            self.trainer(checkpoint=checkpoint, qat=qat(4))
        self.assertEqual((self.model / 'model.safetensors').read_bytes(), original)

    def test_master_export_ptq_preserves_same_fake_weights(self):
        rng = np.random.default_rng(5)
        weight = rng.normal(0, .2, (33, 2)).astype(np.float32)
        self.linear(weight)
        for bits in (2, 3, 4):
            with self.subTest(bits=bits), self.trainer(qat=qat(bits), optimizer=vx.SGD(lr=0)) as trainer:
                trainer.step(np.ones((1, 33), np.float32), [0])
                masters = trainer.save(self.root / f'masters{bits}')
                np.testing.assert_array_equal(load_file(masters.weight_paths[0])['weight'], weight)
                vx.quantize(masters.path, output=self.root / f'packed{bits}',
                    weight_bits=bits, group_size=32, algorithm='rtn', cpu_threads=1)
                graph = json.loads((self.root / f'packed{bits}/graph.json').read_text())
                tensors = load_file(self.root / f'packed{bits}/model.safetensors')
                node = graph['nodes'][0]
                packed, scales = tensors[node['inputs']['weight']], tensors[node['inputs']['scale']]
                actual = np.empty((2, 33), np.float32)
                for row in range(2):
                    stream = int.from_bytes(packed[row].tobytes(), 'little')
                    for k in range(33):
                        q = ((stream >> (bits * k)) & ((1 << bits) - 1)) - (1 << (bits - 1))
                        actual[row, k] = np.float32(q * scales[row, k // 32])
                np.testing.assert_array_equal(actual, fit_rows(weight.T, bits)[0])

    def test_busy_shared_views_accumulation_and_failed_step_restore_masters(self):
        weight = np.full((32, 2), .1, np.float32); weight[0] = [.3, -.2]
        self.linear(weight)
        x = np.ones((1, 32), np.float32) * .1
        with vx.TrainingSession(self.model, loss=vx.CrossEntropyLoss('logits', normalizer=2),
                qat=qat(), optimizer=vx.SGD(lr=.1), cpu_threads=1) as trainer:
            with trainer.parameters(['weight'], mode='shared_read') as views:
                with self.assertRaises(vx.VolvoxAIError): trainer.step(x, [0])
                np.testing.assert_array_equal(views['weight'].numpy(), weight)
            trainer.step(x, [0], accumulation_steps=2)
            np.testing.assert_array_equal(self.values(trainer)['weight'], weight)
            with self.assertRaises(vx.VolvoxAIError): trainer.save_checkpoint(self.root / 'unfinished.pb')
            trainer.rollback()
            bad = x.copy(); bad[0, 0] = np.nan
            with self.assertRaises(vx.VolvoxAIError): trainer.step(bad, [0])
            np.testing.assert_array_equal(self.values(trainer)['weight'], weight)
            trainer.step(x, [0])

    def test_policy_admission_and_unquantized_default(self):
        weight = np.full((32, 2), .1, np.float32)
        long_id = 'projection-' + 'x' * 300
        self.linear(weight, node=long_id)
        with self.trainer(qat=qat(node=long_id), optimizer=vx.SGD(lr=0)) as trainer:
            trainer.step(np.ones((1, 32), np.float32), [0])
        self.linear(weight)
        for policy in (pb.QatOptions(), qat(8), qat(node='missing'), qat(group=0)):
            with self.subTest(policy=str(policy)), self.assertRaises(vx.VolvoxAIError):
                self.trainer(qat=policy)
        bad = qat(); bad.nodes[0].weights.algorithm = pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE
        with self.assertRaises(vx.VolvoxAIError): self.trainer(qat=bad)
        with self.assertRaises(vx.VolvoxAIError): self.trainer(qat=qat(), backend='vulkan')
        with self.trainer(optimizer=vx.SGD(lr=0)) as trainer:
            trainer.step(np.ones((1, 32), np.float32), [0])
            self.assertFalse(self.state(trainer).HasField('qat'))
            self.assertEqual(self.state(trainer).qat_workspace_high_water_bytes, 0)

    def test_nonfinite_gradient_reports_master_norm_and_restores_weights(self):
        # MSE chooses exactly scale1/code1; the outlier master remains4. Exact
        # cancellation keeps logits finite while a small loss denominator
        # makes the backward overflow. STE must not hide that failure.
        weight = np.full((32, 2), np.float32(28 / 31), np.float32); weight[0] = 4
        self.linear(weight)
        fake, _ = fit_rows(weight.T, 2, mse=True)
        np.testing.assert_array_equal(fake, np.ones_like(fake))
        x = np.zeros((1, 32), np.float32); x[0, 1:3] = [3e38, -3e38]
        with vx.TrainingSession(self.model, loss=vx.CrossEntropyLoss('logits', normalizer=.1),
                qat=qat(algorithm='mse'), optimizer=vx.SGD(lr=.1), cpu_threads=1) as trainer:
            with self.assertRaises(vx.VolvoxAIError) as caught:
                trainer.step(x, [0], statistics=True)
            self.assertEqual(caught.exception.code, pb.OperationCode.OPERATION_CODE_TRAINING_GRADIENT_NONFINITE)
            stats = caught.exception.response.gradients.parameters[0]
            self.assertGreater(stats.nonfinite_count, 0)
            self.assertAlmostEqual(stats.parameter_norm, np.linalg.norm(weight.astype(np.float64)), places=6)
            np.testing.assert_array_equal(self.values(trainer)['weight'], weight)

    def test_shared_weights_require_identical_selected_weight_only_consumers(self):
        weight = np.full((32, 2), .1, np.float32)
        self.linear(weight)
        graph = json.loads((self.model / 'graph.json').read_text())
        first = graph['nodes'][0]
        first['outputs']['out']['tensor'] = 'a'
        second = json.loads(json.dumps(first)); second['id'] = 'other'
        second['outputs']['out']['tensor'] = 'b'
        graph['nodes'] += [second, {'id': 'sum', 'opType': 'Add', 'inputs': {'a': 'a', 'b': 'b'},
            'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': [1, 2]}}, 'params': {}}]
        self.write(graph, {'weight': weight})
        with self.assertRaises(vx.VolvoxAIError): self.trainer(qat=qat())
        shared = qat(); shared.nodes.extend(qat(4, node='other').nodes)
        with self.assertRaises(vx.VolvoxAIError): self.trainer(qat=shared)
        shared.nodes[1].weights.format.bits = 2
        with self.trainer(qat=shared, optimizer=vx.SGD(lr=.1)) as trainer:
            trainer.step(np.ones((1, 32), np.float32) * .1, [0])
        # A selected tensor used as both input and weight is not a weight-only
        # fake-quant contract, even when both roles belong to the same node.
        square = np.eye(32, dtype=np.float32)
        self.linear(square)
        graph = json.loads((self.model / 'graph.json').read_text())
        graph['nodes'][0]['inputs']['input'] = 'weight'
        graph['nodes'][0]['outputs']['out']['shape'] = [32, 32]
        self.write(graph, {'weight': square})
        with vx.InferenceSession(self.model, cpu_threads=1):
            pass
        with self.assertRaises(vx.VolvoxAIError): self.trainer(qat=qat())


if __name__ == '__main__':
    unittest.main()
