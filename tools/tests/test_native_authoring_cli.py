"""Native CLI orchestration against real full binaries; no mocked service calls.

VOLVOXAI_NATIVE_CLI=/path/to/volvoxai python -m unittest discover \
    -s tools/tests -p test_native_authoring_cli.py -v
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
import volvoxai as vx


class NativeAuthoringCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = Path(os.environ.get('VOLVOXAI_NATIVE_CLI', ROOT / 'native/volvoxai')).resolve()
        if not cls.cli.is_file():
            raise unittest.SkipTest('Build the full native CLI first')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='volvox-cli-authoring-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'model'
        self.source.mkdir()
        self.rng = np.random.default_rng(453)
        self.weights = {'weight': self.rng.normal(0, .2, (33, 3)).astype(np.float32)}
        self.write_graph([{'id': 'projection', 'opType': 'Linear',
            'inputs': {'input': 'x', 'weight': 'weight'},
            'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': ['B', 3]}},
            'params': {'weight_layout': 'din_dout'}}], ["B", 33])
        self.x = self.rng.normal(0, .5, (2, 33)).astype(np.float32)
        self.x.tofile(self.root / 'a.f32')
        (self.x[:1] * -.7).astype(np.float32).tofile(self.root / 'b.f32')
        np.array([0, 2], np.int32).tofile(self.root / 'a.i32')
        np.array([1], np.int32).tofile(self.root / 'b.i32')
        self.batches = self.json('batches.json', {'batches': [
            {'inputs': ['x[2,33]=a.f32'], 'targets': 'a.i32'},
            {'inputs': ['x[1,33]=b.f32'], 'targets': 'b.i32'}]})
        self.calibration = self.json('calibration.json', {'batches': [
            {'inputs': ['x[2,33]=a.f32'], 'sampleName': 'two', 'sampleCount': 2, 'profileName': 'wide'},
            {'inputs': ['x[1,33]=b.f32'], 'sample_name': 'one', 'sample_count': 1, 'profile_name': 'short'}]})

    def json(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return path

    def write_graph(self, nodes, shape):
        graph = {'format': 'volvox-graph/v1', 'dimensions': {'B': {'min': 1, 'max': 2}},
            'inputs': {'x': {'dtype': 'float32', 'shape': shape}}, 'outputs': ['logits'],
            'execution': {'stateless': {}, 'entrypoints': [{'name': 'evaluate', 'effect': 1,
                'allowed_ranges': [1], 'inputs': ['x'], 'outputs': ['logits']}]}, 'nodes': nodes}
        (self.source / 'graph.json').write_text(json.dumps(graph))
        save_file(self.weights, self.source / 'model.safetensors')

    def call(self, *args, ok=True):
        result = subprocess.run([str(self.cli), *map(str, args)], cwd=self.root,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    @staticmethod
    def weight_policy(bits, group=32, algorithm='WEIGHT_ALGORITHM_MSE_CLIP'):
        return {'format': {'bits': bits, 'groupSize': group}, 'algorithm': algorithm}

    def train_args(self, output, **extra):
        args = ['train', self.source, '--logits', 'logits', '--trainable', 'weight',
                '--threads', '1', '--output-weights', self.root / output]
        for k, value in extra.items():
            args += ['--' + k.replace('_', '-'), value]
        return args

    def test_train_more_than_32_weights_matches_sdk(self):
        nodes = [{'id': 'projection', 'opType': 'Linear',
            'inputs': {'input': 'x', 'weight': 'weight'},
            'outputs': {'out': {'tensor': 'hidden-0', 'dtype': 'float32', 'shape': ['B', 3]}},
            'params': {'weight_layout': 'din_dout'}}]
        for i in range(50):
            name = f'bias-{i}'
            self.weights[name] = self.rng.normal(0, .01, (3,)).astype(np.float32)
            nodes.append({'id': f'add-{i}', 'opType': 'Add',
                'inputs': {'a': f'hidden-{i}', 'b': name},
                'outputs': {'out': {'tensor': f'hidden-{i+1}' if i != 49 else 'logits',
                                    'dtype': 'float32', 'shape': ['B', 3]}}, 'params': {}})
        self.write_graph(nodes, ['B', 33])
        output = self.root / 'all-51.safetensors'
        args = ['train', self.source, '--batches', self.batches, '--logits', 'logits',
                '--output-weights', output, '--cpu', '--threads', '1', '--learning-rate', '.01',
                '--max-activation-capacity-bytes', str(4 * 1024**3)]
        for name in self.weights:
            args += ['--trainable', name]
        self.call(*args)
        actual = load_file(output)
        with vx.TrainingSession(self.source, loss=vx.CrossEntropyLoss('logits'),
                optimizer=vx.SGD(lr=.01), cpu_threads=1, trainable_names=list(self.weights),
                shape_options=vx.pb.TrainerShapeOptions(max_activation_capacity_bytes=4 * 1024**3)) as trainer:
            trainer.step(self.x, [0, 2])
            trainer.step((self.x[:1] * -.7).astype(np.float32), [1])
            with trainer.parameters(list(self.weights)) as expected:
                self.assertEqual(len(actual), 51)
                for name in self.weights:
                    np.testing.assert_allclose(actual[name], expected[name].numpy(), rtol=2e-6, atol=1e-7)
                    self.assertFalse(np.array_equal(actual[name], self.weights[name]), name)

    def test_activation_capacity_admission_and_invalid_values(self):
        for index, value in enumerate(('0', '-1', '+1', '1.5', '1KiB', '18446744073709551616')):
            with self.subTest(value=value):
                self.call(*self.train_args(f'invalid-capacity-{index}.safetensors', batches=self.batches,
                    max_activation_capacity_bytes=value), ok=False)
        failed = self.root / 'small-capacity.json'
        self.call(*self.train_args('small-capacity.safetensors', batches=self.batches,
            max_activation_capacity_bytes='1', report_json=failed), ok=False)
        self.assertFalse((self.root / 'small-capacity.safetensors').exists())
        self.call(*self.train_args('capacity-duplicate.safetensors', batches=self.batches),
            '--max-activation-capacity-bytes', '4096', '--max-activation-capacity-bytes', '8192', ok=False)

    def test_ptq_cuda_choice_without_observers_creates_no_executor(self):
        output = self.root / 'unused-cuda'
        report = self.root / 'unused-cuda.json'
        self.call('ptq', self.source, '--weight-bits', 3, '--group-size', 32,
                  '--algorithm', 'rtn', '--cuda', '--out', output,
                  '--report-json', report)
        plan = json.loads(report.read_text())['plan']
        self.assertFalse(plan['requiresCalibration'])
        self.assertEqual(plan['calibrationBackend'], '')
        self.call('ptq', self.source, '--cuda', '--cpu', '--dry-run', ok=False)

    def test_ptq_real_packed_bits_and_independent_inference(self):
        for bits in (2, 3, 4, 8):
            with self.subTest(bits=bits):
                out = self.root / f'packed{bits}'
                report = self.root / f'ptq{bits}.json'
                self.call('ptq', self.source, '--weight-bits', bits, '--group-size', 32,
                          '--algorithm', 'mse-clip', '--out', out, '--report-json', report)
                g = json.loads((out / 'graph.json').read_text())
                node = next(n for n in g['nodes'] if n['opType'] == 'PackedLinear')
                self.assertEqual(node['params']['weight_bits'], bits)
                tensors = load_file(out / 'model.safetensors')
                codes = tensors[node['inputs']['weight']].view(np.uint32)
                scales = tensors[node['inputs']['scale']]
                self.assertEqual(codes.nbytes, 3 * ((33 * bits + 31) // 32) * 4)
                # Independent bit-at-a-time signed-code reconstruction, including I3 word crossing.
                restored = np.empty((3, 33), np.float32)
                for o in range(3):
                    for k in range(33):
                        code = sum(((int(codes[o, (k * bits + j) // 32]) >> ((k * bits + j) % 32)) & 1) << j for j in range(bits))
                        restored[o, k] = (code - (1 << (bits - 1))) * scales[o, k // 32]
                output = self.root / f'inference{bits}.f32'
                self.call('run', out, '--cpu', '--threads', 1, '--input', f'x[2,33]={self.root / "a.f32"}',
                          '--output', f'logits={output}')
                np.testing.assert_allclose(np.fromfile(output, np.float32).reshape(2, 3),
                                           self.x @ restored.T, atol=2e-6, rtol=2e-6)
                self.assertTrue(json.loads(report.read_text())['passed'])
                integers = np.empty((3, 33), np.int32)
                for o in range(3):
                    for k in range(33):
                        integers[o, k] = sum(((int(codes[o, (k * bits + j) // 32]) >> ((k * bits + j) % 32)) & 1) << j for j in range(bits)) - (1 << (bits - 1))
                expected = np.zeros((2, 3), np.float32)
                for row in range(2):
                    activation_scale = np.float32(np.max(np.abs(self.x[row])) / np.float32(127))
                    quantized = np.rint(self.x[row] / activation_scale).clip(-127, 127).astype(np.int32)
                    for o in range(3):
                        for start in range(0, 33, 32):
                            dot = np.dot(quantized[start:start+32], integers[o, start:start+32])
                            expected[row, o] += np.float32(dot) * np.float32(activation_scale * scales[o, start//32])
                a8_output, a8_report = self.root / f'a8-{bits}.f32', self.root / f'a8-{bits}.json'
                self.call('run', out, '--cpu', '--threads', 1, '--packed-arithmetic', 'dynamic-i8',
                          '--input', f'x[2,33]={self.root / "a.f32"}', '--output', f'logits={a8_output}', '--report-json', a8_report)
                np.testing.assert_allclose(np.fromfile(a8_output, np.float32).reshape(2, 3), expected, atol=1e-7, rtol=2e-6)
                arithmetic = json.loads(a8_report.read_text())
                self.assertEqual(arithmetic['request']['packedArithmetic'], 'PACKED_ARITHMETIC_DYNAMIC_I8')
                self.assertEqual(arithmetic['resolvedPackedArithmetic'], 'PACKED_ARITHMETIC_DYNAMIC_I8')
                failed = self.root / f'a4-{bits}.json'
                self.call('run', out, '--cpu', '--packed-arithmetic', 'dynamic-i4',
                          '--input', f'x[2,33]={self.root / "a.f32"}', '--report-json', failed, ok=False)
                self.assertEqual(json.loads(failed.read_text())['operation'], 'CompileModel')


    def test_rtn_zero_enum_and_unknown_algorithms(self):
        out = self.root / 'rtn-flag'
        report = self.root / 'rtn-flag.json'
        self.call('ptq', self.source, '--algorithm', 'rtn', '--weight-bits', 2,
                  '--out', out, '--report-json', report)
        self.assertEqual(json.loads(report.read_text())['plan']['resolvedNodes'][0]
                         ['weights']['algorithm'], 'WEIGHT_ALGORITHM_RTN')
        for index, algorithm in enumerate(('WEIGHT_ALGORITHM_RTN', 0, None)):
            with self.subTest(algorithm=algorithm):
                weights = self.weight_policy(2, algorithm=algorithm)
                if algorithm is None:
                    del weights['algorithm']  # Canonical protobuf JSON omits the RTN zero value.
                rule = self.json(f'rtn-rule-{index}.json', [
                    {'nodeId': 'projection', 'weights': weights}])
                exported = self.root / f'rtn-rule-{index}'
                self.call('ptq', self.source, '--rules', rule, '--out', exported)
                self.assertEqual((out / 'model.safetensors').read_bytes(),
                                 (exported / 'model.safetensors').read_bytes())
                qat = self.json(f'rtn-qat-{index}.json', {'nodes': [
                    {'nodeId': 'projection', 'weights': weights}]})
                training_report = self.root / f'rtn-train-{index}.json'
                self.call(*self.train_args(f'rtn-train-{index}.safetensors',
                          batches=self.batches, qat=qat, report_json=training_report))
                self.assertEqual(json.loads(training_report.read_text())['finalState']
                                 ['qat']['nodes'][0]['weights']['algorithm'], 'WEIGHT_ALGORITHM_RTN')
        self.call('ptq', self.source, '--algorithm', 'unknown', '--dry-run', ok=False)
        for index, algorithm in enumerate(('WEIGHT_ALGORITHM_UNKNOWN', 'rtn', 3, -1)):
            with self.subTest(invalid=algorithm):
                policy = {'nodeId': 'projection', 'weights': self.weight_policy(2, algorithm=algorithm)}
                rule = self.json(f'bad-algorithm-rule-{index}.json', [policy])
                failed_out = self.root / f'bad-algorithm-rule-{index}'
                self.call('ptq', self.source, '--rules', rule, '--out', failed_out, ok=False)
                self.assertFalse((failed_out / 'model.safetensors').exists())
                qat = self.json(f'bad-algorithm-qat-{index}.json', {'nodes': [policy]})
                output = f'bad-algorithm-train-{index}.safetensors'
                self.call(*self.train_args(output, batches=self.batches, qat=qat), ok=False)
                self.assertFalse((self.root / output).exists())

    def test_importance_named_profiles_and_canonical_rules(self):
        rules = self.json('rules.json', [{'nodeGlob': 'proj*', 'weights': self.weight_policy(3, algorithm='WEIGHT_ALGORITHM_IMPORTANCE')}])
        out, report = self.root / 'importance', self.root / 'importance.json'
        self.call('ptq', self.source, '--weight-bits', 2, '--algorithm', 'importance',
                  '--rules', rules, '--calibration', self.calibration, '--threads', 1,
                  '--out', out, '--report-json', report)
        r = json.loads(report.read_text())
        self.assertEqual(r['plan']['calibration']['batches'], 2)
        self.assertEqual(r['plan']['calibration']['samples'], 3)
        self.assertEqual(r['plan']['calibration']['importanceTensors'], 1)
        self.assertEqual(r['plan']['resolvedNodes'][0]['weights']['format']['bits'], 3)
        self.assertTrue(r['plan']['calibration']['coverageComplete'])
        dry = self.root / 'dry.json'
        self.call('ptq', self.source, '--algorithm', 'importance', '--dry-run', '--report-json', dry)
        self.assertFalse(json.loads(dry.read_text())['plan']['ready'])

    def test_keep_float_unknown_duplicate_alias_and_literal_backslash(self):
        self.write_graph([
            {'id': 'activate', 'opType': 'SiLU', 'inputs': {'input': 'x'},
             'outputs': {'out': {'tensor': 'hidden', 'dtype': 'float32', 'shape': ['B', 33]}}, 'params': {}},
            {'id': 'projection', 'opType': 'Linear', 'inputs': {'input': 'hidden', 'weight': 'weight'},
             'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': ['B', 3]}},
             'params': {'weight_layout': 'din_dout'}}], ['B', 33])
        keep = self.json('keep.json', [{'operator_name': 'SiLU', 'keepFloat': {}}])
        report = self.root / 'keep-report.json'
        self.call('ptq', self.source, '--rules', keep, '--dry-run', '--report-json', report)
        self.assertTrue(json.loads(report.read_text())['passed'])
        self.assertEqual(json.loads(report.read_text())['plan']['resolvedNodes'][0]['nodeId'], 'projection')
        invalid = [
            '[{"nodeId":"projection","node_id":"projection","keepFloat":{}}]',
            '[{"nodeId":"projection","keepFloat":{},"keep_float":{}}]',
            '[{"nodeId":"projection","keepFloat":{},"typo":0}]',
            '[{"nodeId":"projection","nodeId":"projection","keepFloat":{}}]',
            '[{"nodeId":"projection","keepFloat":true}]',
            '[{"nodeId":"missing","keepFloat":{}}]',
        ]
        for i, text in enumerate(invalid):
            path = self.root / f'invalid{i}.json'; path.write_text(text)
            self.call('ptq', self.source, '--rules', path, '--dry-run', ok=False)
        # A literal backslash-u0000 is a legal selector string, not an embedded NUL.
        path = self.json('literal.json', [{'nodeId': r'\u0000', 'keepFloat': {}}])
        result = self.call('ptq', self.source, '--rules', path, '--dry-run', ok=False)
        self.assertIn('CreateQuantizationPlan', result.stderr)
        self.assertNotIn('Invalid JSON document', result.stderr)
        # Large valid manifests/policies are not subject to an unrelated fixed byte cap.
        path = self.root / 'large.json'; path.write_text(' ' * (8 * 1024 * 1024 + 1) + keep.read_text())
        self.call('ptq', self.source, '--rules', path, '--dry-run')

    def test_static_conv_i2_i3_and_explicit_int8_rtn(self):
        self.weights = {'conv': self.rng.normal(0, .2, (1, 3, 1, 4)).astype(np.float32)}
        self.write_graph([{'id': 'conv', 'opType': 'Conv2D',
            'inputs': {'input': 'x', 'weight': 'conv'},
            'outputs': {'out': {'tensor': 'logits', 'dtype': 'float32', 'shape': [1, 3, 4, 4]}},
            'params': {'stride': [1, 2], 'dilation': [1, 1],
                       'groups': 2, 'pads': [0, 1, 0, 1], 'data_layout': 'NHWC', 'weight_layout': 'HWIO'}}], [1, 3, 7, 2])
        self.rng.normal(0, .5, (1, 3, 7, 2)).astype(np.float32).tofile(self.root / 'conv.f32')
        calibration = self.json('conv-cal.json', {'batches': [{'inputs': ['x=conv.f32']}]})
        for bits in (2, 3, 8):
            out = self.root / f'conv{bits}'
            self.call('ptq', self.source, '--weight-bits', bits, '--group-size', 0,
                      '--algorithm', 'rtn' if bits == 8 else 'importance', '--static-activations', 'i8',
                      '--calibration', calibration, '--out', out, '--threads', 1)
            node = next(n for n in json.loads((out / 'graph.json').read_text())['nodes'] if n['opType'] == 'QConv2D')
            self.assertEqual(node['params'].get('weight_bits', 8), bits)
            weight = load_file(out / 'model.safetensors')[node['inputs']['weight']]
            self.assertEqual(weight.dtype, np.int8 if bits == 8 else np.uint8)
            self.assertEqual(weight.nbytes, 4 * ((3 * bits + 7) // 8))

    def test_qat_adamw_varying_batches_epochs_exact_checkpoint_resume(self):
        qat = self.json('qat.json', {'nodes': [{'nodeId': 'projection', 'weights': self.weight_policy(3)}]})
        before = hashlib.sha256((self.source / 'model.safetensors').read_bytes()).hexdigest()
        continuous, first, resumed = self.root / 'continuous.pb', self.root / 'first.pb', self.root / 'resumed.pb'
        optimizer = dict(optimizer='adamw', learning_rate='.005', beta1='.8', beta2='.95', epsilon='0.00001', weight_decay='.01')
        self.call(*self.train_args('continuous.safetensors', qat=qat, batches=self.batches,
                  epochs=2, checkpoint_out=continuous, **optimizer))
        self.call(*self.train_args('first.safetensors', qat=qat, batches=self.batches,
                  checkpoint_out=first, **optimizer))
        report = self.root / 'resume.json'
        self.call(*self.train_args('resumed.safetensors', batches=self.batches, checkpoint_in=first,
                  checkpoint_out=resumed, report_json=report))
        self.assertEqual((self.root / 'continuous.safetensors').read_bytes(), (self.root / 'resumed.safetensors').read_bytes())
        a = vx.pb.TrainerCheckpoint.FromString(continuous.read_bytes())
        b = vx.pb.TrainerCheckpoint.FromString(resumed.read_bytes())
        self.assertEqual(a.optimizer_step, 4)
        self.assertEqual(a.optimizer, b.optimizer)
        self.assertEqual(a.qat, b.qat)
        self.assertNotEqual((self.root / 'resumed.safetensors').read_bytes(), (self.source / 'model.safetensors').read_bytes())
        self.assertEqual(hashlib.sha256((self.source / 'model.safetensors').read_bytes()).hexdigest(), before)
        r = json.loads(report.read_text())
        self.assertEqual(r['initialState']['optimizerStep'], 2)
        self.assertEqual(r['finalState']['optimizerConfig']['kind'], 'TRAINING_OPTIMIZER_KIND_ADAMW')
        self.assertAlmostEqual(r['finalState']['optimizerConfig']['learningRate'], .005)
        wrong = self.json('wrong-qat.json', {'nodes': [{'node_id': 'projection', 'weights': self.weight_policy(2)}]})
        self.call(*self.train_args('conflict.safetensors', batches=self.batches, qat=wrong, checkpoint_in=first), ok=False)
        self.assertFalse((self.root / 'conflict.safetensors').exists())

    def test_mixed_qat_2_3_4_nodes(self):
        self.weights = {'weight': self.rng.normal(0, .1, (33, 32)).astype(np.float32),
                        'middle': self.rng.normal(0, .1, (32, 32)).astype(np.float32),
                        'head': self.rng.normal(0, .1, (32, 3)).astype(np.float32)}
        nodes = []
        for name, source, output, width in [('first', 'x', 'h1', 32), ('second', 'h1', 'h2', 32), ('last', 'h2', 'logits', 3)]:
            weight = {'first': 'weight', 'second': 'middle', 'last': 'head'}[name]
            nodes.append({'id': name, 'opType': 'Linear', 'inputs': {'input': source, 'weight': weight},
                          'outputs': {'out': {'tensor': output, 'dtype': 'float32', 'shape': ['B', width]}},
                          'params': {'weight_layout': 'din_dout'}})
        self.write_graph(nodes, ['B', 33])
        qat = self.json('mixed-qat.json', {'nodes': [{'nodeId': name, 'weights': self.weight_policy(bits)}
                                                    for name, bits in [('first', 2), ('second', 3), ('last', 4)]]})
        report = self.root / 'mixed-report.json'
        self.call(*self.train_args('mixed.safetensors', batches=self.batches, qat=qat, report_json=report),
                  '--trainable', 'middle', '--trainable', 'head')
        nodes = json.loads(report.read_text())['finalState']['qat']['nodes']
        self.assertEqual([n['weights']['format']['bits'] for n in nodes], [2, 3, 4])

    def test_final_short_accumulation_window_and_ignored_targets(self):
        np.array([0, -100], np.int32).tofile(self.root / 'ignored.i32')
        common = dict(targets=self.root / 'ignored.i32', input=f'x[2,33]={self.root / "a.f32"}', learning_rate='.01')
        self.call(*self.train_args('two.safetensors', microbatches=2, **common))
        self.call(*self.train_args('three.safetensors', microbatches=3, accumulation_steps=2, **common))
        np.testing.assert_allclose(load_file(self.root / 'two.safetensors')['weight'],
                                   load_file(self.root / 'three.safetensors')['weight'], rtol=0, atol=2e-7)
        result = self.call(*self.train_args('unsupported.safetensors', batches=self.batches, accumulation_steps=2), ok=False)
        self.assertIn('requires --accumulation-steps 1', result.stderr)
        self.assertFalse((self.root / 'unsupported.safetensors').exists())

    def test_fresh_default_sgd_and_all_ignored_noop(self):
        common = dict(targets=self.root / 'a.i32', input=f'x[2,33]={self.root / "a.f32"}', learning_rate='.01')
        self.call(*self.train_args('default.safetensors', **common))
        self.call(*self.train_args('sgd.safetensors', optimizer='sgd', **common))
        self.assertEqual((self.root / 'default.safetensors').read_bytes(), (self.root / 'sgd.safetensors').read_bytes())
        logits = self.x.astype(np.float64) @ self.weights['weight'].astype(np.float64)
        probs = np.exp(logits-logits.max(axis=1, keepdims=True)); probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(2), [0, 2]] -= 1
        expected = self.weights['weight'] - .01 * (self.x.T @ probs) / 2
        np.testing.assert_allclose(load_file(self.root / 'default.safetensors')['weight'], expected, rtol=0, atol=5e-8)
        np.array([-100, -100], np.int32).tofile(self.root / 'empty.i32')
        common['targets'] = self.root / 'empty.i32'
        self.call(*self.train_args('ignored.safetensors', microbatches=3, accumulation_steps=2, **common))
        np.testing.assert_array_equal(load_file(self.root / 'ignored.safetensors')['weight'], self.weights['weight'])
        self.call(*self.train_args('bad-arithmetic.safetensors', batches=self.batches), '--packed-arithmetic', 'dynamic-i8', ok=False)

    def test_existing_outputs_and_dangling_symlink_refused(self):
        original = (self.source / 'model.safetensors').read_bytes()
        self.call(*self.train_args('model/model.safetensors', batches=self.batches), ok=False)
        self.assertEqual((self.source / 'model.safetensors').read_bytes(), original)
        path = self.root / 'report-link.json'; path.symlink_to(self.root / 'absent')
        self.call('ptq', self.source, '--dry-run', '--report-json', path, ok=False)
        self.assertTrue(path.is_symlink()); self.assertFalse((self.root / 'absent').exists())


if __name__ == '__main__':
    unittest.main()
