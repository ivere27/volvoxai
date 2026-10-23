"""Actual I2/I3 PTQ storage and diagonal input-energy fitting via public RPCs.

The references enumerate weight candidates and im2col patches independently.
They do not assert full reconstruction loss or end-task quality improvement.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load, save

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import volvoxai as vx
from test_ptq_conv4_end_to_end import source as conv_source

pb = vx.pb
RTN = pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN
MSE = pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP
IMPORTANCE = pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE


def weight_policy(bits, group, algorithm):
    return pb.WeightQuantization(format=pb.WeightFormat(bits=bits, group_size=group),
                                 algorithm=algorithm)


def fit(rows, bits, algorithm, importance, grouped):
    """One independent candidate enumeration per group/output channel."""
    rows = rows.reshape(len(rows), -1)
    high, low = 2**(bits - 1) - 1, -2**(bits - 1)
    group = 32 if grouped else rows.shape[1]
    scales = np.empty((len(rows), (rows.shape[1] + group - 1) // group), np.float32)
    codes = np.empty(rows.shape, np.int8)
    for r in range(len(rows)):
        for g, begin in enumerate(range(0, rows.shape[1], group)):
            values = rows[r, begin:begin + group]
            base = np.float32(np.max(np.abs(values)) / np.float32(high))
            if base == 0:
                base = np.float32(1)
            last = 100 if algorithm == RTN else 1 if bits == 2 else 50 if bits == 3 or not grouped else 75
            best_loss, best_scale = float('inf'), base
            for percent in range(100, last - 1, -1):
                factor = np.float32(percent / 100)
                if grouped and bits >= 4:
                    factor = np.float32(1) - np.float32(100 - percent) * np.float32(.01)
                scale = np.float32(base * factor)
                q = np.clip(np.rint(values / scale), low, high)
                delta = values.astype(np.float64) - (q * scale).astype(np.float64)
                energy = importance[r, begin:begin + len(values)] if algorithm == IMPORTANCE else 1
                loss = float(np.sum(delta * delta * energy))
                if loss < best_loss:
                    best_loss, best_scale = loss, scale
            scales[r, g] = best_scale
            codes[r, begin:begin + len(values)] = np.clip(np.rint(values / best_scale), low, high)
    return codes, scales


def pack(codes, bits, word_bits):
    """Set every bit independently, including canonical physical tail bits."""
    count = (codes.shape[1] * bits + word_bits - 1) // word_bits
    words = np.zeros((len(codes), count), np.uint32 if word_bits == 32 else np.uint8)
    logical = count * word_bits // bits if bits != 3 else codes.shape[1]
    zero = 2**(bits - 1)
    for row in range(len(codes)):
        for column in range(logical):
            code = int(codes[row, column]) + zero if column < codes.shape[1] else zero
            for bit in range(bits):
                position = column * bits + bit
                if code & (1 << bit):
                    words[row, position // word_bits] |= np.array(1 << (position % word_bits), dtype=words.dtype)
    return words.view(np.int32) if word_bits == 32 else words


def infer(package, values):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / 'graph.json').write_bytes(package.graph)
        (directory / 'model.safetensors').write_bytes(package.weight_shards[0])
        with vx.InferenceSession(directory, cpu_threads=1, packed_arithmetic='dynamic-i8') as session:
            return session.run(values)


def calibrate(client, identifier, name, values):
    return client.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
        quantization_plan_id=identifier, sample_name=name, sample_count=1,
        inputs=[pb.Tensor(name=key, dtype=pb.DataType.DATA_TYPE_F32,
                          shape=list(value.shape), inline=value.tobytes())
                for key, value in values.items()]))


def export(client, identifier):
    return client.export_quantized_model(pb.ExportQuantizedModelRequest(
        quantization_plan_id=identifier)).bytes


class LowbitImportancePTQTest(unittest.TestCase):
    def test_grouped_mixed_rules_snapshot_atomicity_and_dynamic_i8_oracle(self):
        rng = np.random.default_rng(412)
        weights = {'w': rng.normal(0, .3, (3, 35)).astype(np.float32),
                   'head': rng.normal(0, .4, (2, 35)).astype(np.float32),
                   'bias': np.array([.125, -.05, .01], np.float32)}
        graph = {'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'x': {'shape': [3, 35], 'dtype': 'float32'}}, 'nodes': [
                {'id': 'body', 'opType': 'Linear', 'inputs': {'input': 'x', 'weight': 'w', 'bias': 'bias'},
                 'outputs': {'out': {'tensor': 'body-out', 'shape': [3, 3], 'dtype': 'float32'}},
                 'params': {'weight_layout': 'dout_din'}},
                {'id': 'head', 'opType': 'Linear', 'inputs': {'input': 'x', 'weight': 'head'},
                 'outputs': {'out': {'tensor': 'head-out', 'shape': [3, 2], 'dtype': 'float32'}},
                 'params': {'weight_layout': 'dout_din'}}],
            'outputs': ['body-out', 'head-out'], 'execution': {'stateless': {}, 'entrypoints': [
                {'name': 'evaluate', 'effect': 1, 'allowed_ranges': [1],
                 'inputs': ['x'], 'outputs': ['body-out', 'head-out']}]}}
        batches = [rng.normal(0, 1, (3, 35)).astype(np.float32) for _ in range(2)]
        batches[0][:, ::3] *= 50
        batches[1][:, 1::3] *= 90
        energy = sum(np.sum(batch.astype(np.float64)**2, axis=0) for batch in batches)
        for bits in (2, 3):
            for algorithm in (RTN, MSE, IMPORTANCE):
                with self.subTest(bits=bits, algorithm=algorithm), vx.open_library() as host:
                    client = vx.VxQuantizationServiceClient(host)
                    config = pb.QuantizationConfig(weights=weight_policy(bits, 32, algorithm), rules=[
                        pb.QuantizationRule(node_id='head', weights=weight_policy(4, 32, MSE))])
                    original = bytearray(save(weights))
                    plan = client.create_quantization_plan(pb.CreateQuantizationPlanRequest(
                        bytes=pb.ModelBytes(graph=json.dumps(graph).encode(), weight_shards=[bytes(original)]),
                        config=config, calibration=pb.CalibrationOptions(cpu_threads=1)))
                    ref = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
                    try:
                        original[:] = bytes(len(original))
                        self.assertEqual(plan.info.requires_calibration, algorithm == IMPORTANCE)
                        self.assertEqual({node.node_id: node.weights.format.bits for node in plan.info.resolved_nodes if node.HasField('weights')}, {'body': bits, 'head': 4})
                        if algorithm == IMPORTANCE:
                            with self.assertRaises(vx.VolvoxAIError):
                                export(client, plan.quantization_plan_id)
                            first_export = None
                            for index, batch in enumerate(batches):
                                calibrate(client, plan.quantization_plan_id, f'batch-{index}', {'x': batch})
                                if index == 0:
                                    first_export = export(client, plan.quantization_plan_id)
                                    first_bytes = first_export.SerializeToString()
                        package = export(client, plan.quantization_plan_id)
                        if algorithm == IMPORTANCE:
                            self.assertEqual(first_export.SerializeToString(), first_bytes)
                            # Different feature energies must affect the fit;
                            # later export cannot reuse a stale ready snapshot.
                            first_arrays = load(first_export.weight_shards[0])
                            later_arrays = load(package.weight_shards[0])
                            first_doc = json.loads(first_export.graph)
                            first_body = next(n for n in first_doc['nodes'] if n['id'] == 'body')
                            scale_name = first_body['inputs']['scale']
                            self.assertFalse(np.array_equal(first_arrays[scale_name], later_arrays[scale_name]))
                            invalid = batches[0].copy(); invalid[0, 0] = np.nan
                            with self.assertRaises(vx.VolvoxAIError):
                                calibrate(client, plan.quantization_plan_id, 'failed', {'x': invalid})
                            self.assertEqual(export(client, plan.quantization_plan_id), package)
                            self.assertEqual(client.inspect_quantization_plan(ref).info.calibration.samples, 2)
                        self.assertEqual(export(client, plan.quantization_plan_id), package)
                    finally:
                        client.release_quantization_plan(ref)
                document, arrays = json.loads(package.graph), load(package.weight_shards[0])
                value = rng.normal(0, .7, (3, 35)).astype(np.float32); value[2] = 0
                actual = infer(package, {'x': value})
                for node in document['nodes']:
                    local_bits = node['params']['weight_bits']
                    local_algorithm = algorithm if node['id'] == 'body' else MSE
                    source = weights['w' if node['id'] == 'body' else 'head']
                    q, scales = fit(source, local_bits, local_algorithm,
                                    np.broadcast_to(energy, source.shape), True)
                    np.testing.assert_array_equal(arrays[node['inputs']['weight']], pack(q, local_bits, 32))
                    np.testing.assert_array_equal(arrays[node['inputs']['scale']], scales)
                    reference = np.zeros((len(value), len(q)), np.float32)
                    for r, row in enumerate(value):
                        sx = np.float32(np.max(np.abs(row)) / np.float32(127)) if np.any(row) else np.float32(1)
                        iq = np.rint(row / sx).astype(np.int32)
                        for o in range(len(q)):
                            total = np.float32(0)
                            for g, begin in enumerate(range(0, len(row), 32)):
                                dot = int(np.dot(iq[begin:begin + 32].astype(np.int64), q[o, begin:begin + 32].astype(np.int64)))
                                total = np.float32(total + np.float32(np.float32(dot) * np.float32(sx * scales[o, g])))
                            if 'bias' in node['inputs']:
                                total = np.float32(total + weights['bias'][o])
                            reference[r, o] = total
                    np.testing.assert_array_equal(actual[node['outputs']['out']['tensor']], reference)

    def test_conv_importance_im2col_groups_padding_dilation_and_real_packing(self):
        for bits, odd, algorithm in ((2, False, IMPORTANCE), (3, False, IMPORTANCE),
                                      (4, False, IMPORTANCE), (2, True, MSE),
                                      (3, True, RTN)):
            with self.subTest(bits=bits, odd=odd, algorithm=algorithm):
                graph, weight_bytes, canonical, bias, image = conv_source(odd, with_bias=not odd)
                node = graph['nodes'][1]
                if not odd:
                    node['params']['dilation'] = [2, 1]
                    node['outputs']['out']['shape'][1] = 4
                images = [image, np.ascontiguousarray(image[:, ::-1] * np.float32(1.7))]
                params = node['params']; shape = node['outputs']['out']['shape']
                kh, kw, ic = canonical.shape[1:]; groups = params['groups']
                energy = np.zeros((groups, kh, kw, ic), np.float64)
                # Explicit padded patch construction is independent of the
                # engine's source-index accumulation and handles zero padding.
                for sample in images:
                    padded = np.pad(sample.astype(np.float64), ((0, 0), (params['pads'][0], params['pads'][2]),
                                    (params['pads'][1], params['pads'][3]), (0, 0)))
                    for oy in range(shape[1]):
                        for ox in range(shape[2]):
                            patch = padded[:, oy * params['stride'][0]:oy * params['stride'][0] + kh * params['dilation'][0]:params['dilation'][0],
                                           ox * params['stride'][1]:ox * params['stride'][1] + kw * params['dilation'][1]:params['dilation'][1]]
                            for group in range(groups):
                                energy[group] += np.sum(patch[..., group * ic:(group + 1) * ic]**2, axis=0)
                energy_rows = np.repeat(energy.reshape(groups, -1), len(canonical) // groups, axis=0)
                config = pb.QuantizationConfig(weights=weight_policy(bits, 0, algorithm),
                    static_activations=pb.StaticActivations(dtype=pb.DataType.DATA_TYPE_I8,
                        scheme=pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC))
                with vx.open_library() as host:
                    client = vx.VxQuantizationServiceClient(host)
                    plan = client.create_quantization_plan(pb.CreateQuantizationPlanRequest(
                        bytes=pb.ModelBytes(graph=json.dumps(graph).encode(), weight_shards=[weight_bytes]),
                        config=config, calibration=pb.CalibrationOptions(cpu_threads=1)))
                    ref = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
                    try:
                        self.assertTrue(plan.info.requires_calibration)
                        for index, sample in enumerate(images):
                            calibrate(client, plan.quantization_plan_id, str(index), {'image': sample})
                        package = export(client, plan.quantization_plan_id)
                        invalid = image.copy(); invalid.reshape(-1)[-1] = np.nan
                        with self.assertRaises(vx.VolvoxAIError):
                            calibrate(client, plan.quantization_plan_id, 'bad', {'image': invalid})
                        self.assertEqual(export(client, plan.quantization_plan_id), package)
                    finally:
                        client.release_quantization_plan(ref)
                document, arrays = json.loads(package.graph), load(package.weight_shards[0])
                conv = next(n for n in document['nodes'] if n['opType'] == 'QConv2D')
                self.assertEqual(conv['params']['weight_bits'], bits)
                q, scales = fit(canonical, bits, algorithm, energy_rows, False)
                np.testing.assert_array_equal(arrays[conv['inputs']['weight']], pack(q, bits, 8))
                affine = document['quantization']['tensors']
                wa = affine[conv['inputs']['weight']]
                np.testing.assert_array_equal(arrays[wa['scale_tensor']], scales[:, 0])
                np.testing.assert_array_equal(arrays[wa['zero_point_tensor']], 2**(bits - 1))
                ia = affine[conv['inputs']['input']]
                xs = arrays[ia['scale_tensor']][0]
                np.testing.assert_array_equal(arrays[conv['inputs']['bias']], np.rint(bias / (xs * scales[:, 0])).astype(np.int32))
                actual = infer(package, {'image': image})['features']
                # Same exact codes/affines in the established I8 Conv kernel
                # isolate new bitstream storage from the numerical fit.
                expanded = copy.deepcopy(document)
                target = next(n for n in expanded['nodes'] if n['opType'] == 'QConv2D')
                del target['params']['weight_bits'], target['params']['kernel_size']
                arrays[target['inputs']['weight']] = q.reshape(canonical.shape)
                arrays[wa['zero_point_tensor']] = np.zeros(len(q), np.int8)
                control = pb.ModelBytes(graph=json.dumps(expanded).encode(), weight_shards=[save(arrays)])
                np.testing.assert_array_equal(actual, infer(control, {'image': image})['features'])

    def test_static_mixed_rules_keep_the_requested_weight_bits(self):
        for bits, higher in ((2, 4), (2, 3), (3, 4)):
            with self.subTest(bits=bits, higher=higher):
                graph, weight_bytes, _, _, image = conv_source(True)
                other = copy.deepcopy(graph['nodes'][1])
                other['id'] = 'conv-high'
                other['outputs']['out']['tensor'] = 'higher'
                graph['nodes'].append(other)
                graph['outputs'].append('higher')
                graph['execution']['entrypoints'][0]['outputs'].append('higher')
                config = pb.QuantizationConfig(weights=weight_policy(bits, 0, IMPORTANCE),
                    static_activations=pb.StaticActivations(dtype=pb.DataType.DATA_TYPE_I8,
                        scheme=pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC),
                    rules=[pb.QuantizationRule(node_id='conv-high', weights=weight_policy(higher, 0, MSE))])
                with vx.open_library() as host:
                    client = vx.VxQuantizationServiceClient(host)
                    plan = client.create_quantization_plan(pb.CreateQuantizationPlanRequest(
                        bytes=pb.ModelBytes(graph=json.dumps(graph).encode(), weight_shards=[weight_bytes]),
                        config=config, calibration=pb.CalibrationOptions(cpu_threads=1)))
                    ref = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
                    try:
                        self.assertEqual({n.node_id: n.weights.format.bits for n in plan.info.resolved_nodes if n.HasField('weights')},
                                         {'conv': bits, 'conv-high': higher})
                        calibrate(client, plan.quantization_plan_id, 'mixed', {'image': image})
                        package = export(client, plan.quantization_plan_id)
                    finally:
                        client.release_quantization_plan(ref)
                document = json.loads(package.graph)
                self.assertEqual({n['id']: n['params']['weight_bits'] for n in document['nodes']
                                  if n['opType'] == 'QConv2D'}, {'conv': bits, 'conv-high': higher})
                self.assertTrue(all(np.isfinite(value).all() for value in infer(package, {'image': image}).values()))


if __name__ == '__main__':
    unittest.main()
