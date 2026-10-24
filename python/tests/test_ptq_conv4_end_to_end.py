"""Real Conv4 PTQ: per-channel fit, nibble storage, bias and integer oracle.

The exported package is executed through the public native inference API.
The reference loops implement asymmetric activation arithmetic independently.
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
import volvoxai as vx

pb = vx.pb


def policy(algorithm=pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP, dtype=pb.DataType.DATA_TYPE_I8):
    return pb.QuantizationConfig(weights=pb.WeightQuantization(
        format=pb.WeightFormat(bits=4, group_size=0), algorithm=algorithm),
        static_activations=pb.StaticActivations(dtype=dtype,
            scheme=pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC))


def source(odd, with_bias=True):
    shape = (4, 1, 3, 1) if odd else (6, 2, 3, 2)
    image_shape = (1, 3, 7, 2) if odd else (1, 5, 6, 4)
    output_shape = (1, 3, 4, 4) if odd else (1, 5, 3, 6)
    pads = [0, 1, 0, 1] if odd else [1, 0, 0, 2]
    rng = np.random.default_rng(77)
    canonical = rng.normal(0, .3, shape).astype(np.float32)
    canonical[-1] = 0
    bias = np.linspace(-.2, .3, shape[0], dtype=np.float32)
    graph = {'format': 'volvox-graph/v1', 'dimensions': {},
        'inputs': {'image': {'shape': list(image_shape), 'dtype': 'float32'}},
        'nodes': [{'id': 'conv', 'opType': 'Conv2D',
            'inputs': {'input': 'image', 'weight': 'weight', 'bias': 'bias'},
            'outputs': {'out': {'tensor': 'features', 'shape': list(output_shape), 'dtype': 'float32'}},
            'params': {'data_layout': 'NHWC', 'weight_layout': 'HWIO', 'groups': 2,
                'stride': [1, 2], 'dilation': [1, 1], 'pads': pads}}],
        'outputs': ['features'], 'execution': {'stateless': {}, 'entrypoints': [
            {'name': 'evaluate', 'effect': 1, 'allowed_ranges': [1],
             'inputs': ['image'], 'outputs': ['features']}]}}
    # A retained reshape precedes Conv: a global Conv4 policy must not reject
    # unrelated operators merely because they carry no I4-compatible weight.
    graph['nodes'][0]['inputs']['input'] = 'viewed'
    graph['nodes'].insert(0, {'id': 'view', 'opType': 'Reshape',
        'inputs': {'input': 'image'}, 'outputs': {'out': {'tensor': 'viewed',
            'shape': list(image_shape), 'dtype': 'float32'}},
        'params': {'shape': list(image_shape)}})
    tensors = {'weight': np.ascontiguousarray(canonical.transpose(1, 2, 3, 0))}
    if with_bias:
        tensors['bias'] = bias
    else:
        del graph['nodes'][1]['inputs']['bias']
        bias = np.zeros_like(bias)
    weights = save(tensors)
    image = rng.uniform(-.8, 1.3, image_shape).astype(np.float32)
    return graph, weights, canonical, bias, image


def reference_fit(weights, mse):
    rows = weights.reshape(weights.shape[0], -1)
    base = np.abs(rows).max(axis=1) / np.float32(7)
    base[base == 0] = 1
    best = np.full(len(rows), np.inf)
    scales = base.copy()
    for percent in range(100, 49 if mse else 99, -1):
        candidate = base * np.float32(percent / 100)
        codes = np.clip(np.rint(rows / candidate[:, None]), -8, 7)
        delta = rows.astype(np.float64) - (codes * candidate[:, None]).astype(np.float64)
        loss = np.sum(delta * delta, axis=1)
        changed = loss < best
        best[changed] = loss[changed]
        scales[changed] = candidate[changed]
    codes = np.clip(np.rint(rows / scales[:, None]), -8, 7).astype(np.int8)
    return codes.reshape(weights.shape), scales


class Conv4PTQTest(unittest.TestCase):
    def test_exported_storage_and_integer_arithmetic(self):
        for odd, mse, unsigned, with_bias in ((False, True, False, True), (True, True, True, True),
                                                (True, False, False, True), (False, True, False, False)):
            with self.subTest(odd=odd, mse=mse, unsigned=unsigned, bias=with_bias):
                graph, weight_bytes, canonical, bias, image = source(odd, with_bias)
                algorithm = pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP if mse else pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN
                dtype = pb.DataType.DATA_TYPE_U8 if unsigned else pb.DataType.DATA_TYPE_I8
                with vx.open_library() as host:
                    client = vx.VxQuantizationServiceClient(host)
                    plan = client.create_quantization_plan(pb.CreateQuantizationPlanRequest(
                        bytes=pb.ModelBytes(graph=json.dumps(graph).encode(), weight_shards=[weight_bytes]),
                        config=policy(algorithm, dtype), calibration=pb.CalibrationOptions(cpu_threads=1)))
                    ref = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
                    try:
                        self.assertTrue(plan.info.requires_calibration)
                        self.assertFalse(plan.info.ready)
                        self.assertEqual(len(plan.info.quantized), 1)
                        self.assertEqual(plan.info.quantized[0].format.bits, 4)
                        self.assertEqual(plan.info.quantized[0].format.group_size, 0)
                        client.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                            quantization_plan_id=plan.quantization_plan_id, sample_name='oracle', sample_count=1,
                            inputs=[pb.Tensor(name='image', dtype=pb.DataType.DATA_TYPE_F32,
                                shape=list(image.shape), inline=image.tobytes())]))
                        info = client.inspect_quantization_plan(ref).info
                        self.assertTrue(info.ready)
                        self.assertEqual(info.calibration.samples, 1)
                        exported = client.export_quantized_model(pb.ExportQuantizedModelRequest(
                            quantization_plan_id=plan.quantization_plan_id)).bytes
                        again = client.export_quantized_model(pb.ExportQuantizedModelRequest(
                            quantization_plan_id=plan.quantization_plan_id)).bytes
                        self.assertEqual(exported, again)
                    finally:
                        client.release_quantization_plan(ref)
                converted = json.loads(exported.graph)
                arrays = load(exported.weight_shards[0])
                conv = next(node for node in converted['nodes'] if node['opType'] == 'QConv2D')
                self.assertEqual(conv['params']['weight_bits'], 4)
                self.assertEqual(conv['params']['kernel_size'], list(canonical.shape[1:3]))
                codes, scales = reference_fit(canonical, mse)
                row = codes.reshape(len(codes), -1)
                packed = arrays[conv['inputs']['weight']]
                self.assertEqual(packed.dtype, np.uint8)
                self.assertEqual(packed.shape, (len(codes), (row.shape[1] + 1) // 2))
                expected = np.full(packed.shape, 0x88, np.uint8)
                expected[:, :] = (row[:, ::2] + 8).astype(np.uint8) | (expected & 0xf0)
                expected[:, :row.shape[1] // 2] = ((row[:, 1::2] + 8).astype(np.uint8) << 4) | (expected[:, :row.shape[1] // 2] & 15)
                np.testing.assert_array_equal(packed, expected)
                if odd:
                    np.testing.assert_array_equal(packed[:, -1] >> 4, 8)
                affine = converted['quantization']['tensors']
                wa = affine[conv['inputs']['weight']]
                np.testing.assert_array_equal(arrays[wa['scale_tensor']], scales)
                np.testing.assert_array_equal(arrays[wa['zero_point_tensor']], np.full(len(codes), 8, np.uint8))
                ia, oa = (affine[name] for name in (conv['inputs']['input'], conv['outputs']['out']['tensor']))
                xs, ys = (arrays[a['scale_tensor']][0] for a in (ia, oa))
                xz, yz = (int(arrays[a['zero_point_tensor']][0]) for a in (ia, oa))
                expected_bias = np.rint(bias / (xs * scales)).astype(np.int32)
                np.testing.assert_array_equal(arrays[conv['inputs']['bias']], expected_bias)
                low, high = (0, 255) if unsigned else (-128, 127)
                # QuantizeLinear evaluates F32 input/scale, then integer ties-even.
                quantized = np.clip(np.rint(image / xs + np.float32(xz)), low, high).astype(np.int32)
                shape = conv['outputs']['out']['shape']
                expected_output = np.empty(shape, np.float32)
                kh, kw, ic = canonical.shape[1:]
                pads = conv['params']['pads']
                for y in range(shape[1]):
                    for x in range(shape[2]):
                        for o in range(shape[3]):
                            total = int(expected_bias[o])
                            group = o // (shape[3] // 2)
                            for ky in range(kh):
                                for kx in range(kw):
                                    iy, ix = y + ky - pads[0], x * 2 + kx - pads[1]
                                    if 0 <= iy < image.shape[1] and 0 <= ix < image.shape[2]:
                                        for i in range(ic):
                                            total += (int(quantized[0, iy, ix, group * ic + i]) - xz) * int(codes[o, ky, kx, i])
                            multiplier = np.float32(np.float32(xs * scales[o]) / ys)
                            value = np.clip(np.rint(np.float32(total) * multiplier + np.float32(yz)), low, high)
                            expected_output[0, y, x, o] = np.float32(value - yz) * ys
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    (directory / 'graph.json').write_bytes(exported.graph)
                    (directory / 'model.safetensors').write_bytes(exported.weight_shards[0])
                    with vx.InferenceSession(directory, cpu_threads=1) as session:
                        actual = session.run({'image': image})['features']
                    np.testing.assert_array_equal(actual, expected_output)
                    # An unpacked I8 oracle also proves packed bytes address logical OHWI.
                    expanded = copy.deepcopy(converted)
                    target = next(node for node in expanded['nodes'] if node['opType'] == 'QConv2D')
                    del target['params']['weight_bits'], target['params']['kernel_size']
                    arrays[target['inputs']['weight']] = codes
                    arrays[wa['zero_point_tensor']] = np.zeros(len(codes), np.int8)
                    (directory / 'graph.json').write_text(json.dumps(expanded))
                    (directory / 'model.safetensors').write_bytes(save(arrays))
                    with vx.InferenceSession(directory, cpu_threads=1) as session:
                        np.testing.assert_array_equal(session.run({'image': image})['features'], actual)

    def test_python_convenience_static_conv4(self):
        graph, weights, _, _, image = source(True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / 'source'
            original.mkdir()
            (original / 'graph.json').write_text(json.dumps(graph))
            (original / 'model.safetensors').write_bytes(weights)
            package = vx.quantize(original, output=directory / 'result', weight_bits=4,
                group_size=0, algorithm='mse-clip', static_activations=policy().static_activations,
                calibration_data=[{'image': image}], cpu_threads=1)
            self.assertTrue(package.requires_calibration)
            self.assertEqual(package.calibration_samples, 1)
            document = json.loads(package.graph_path.read_text())
            self.assertEqual(next(node for node in document['nodes'] if node['opType'] == 'QConv2D')['params']['weight_bits'], 4)
            with vx.InferenceSession(package, cpu_threads=1) as session:
                self.assertTrue(np.isfinite(session.run({'image': image})['features']).all())

    def test_conv4_accumulator_bound_is_checked_before_execution(self):
        from test_conv4_capabilities import fixture

        graph, weights = fixture()
        graph['nodes'][0]['inputs']['bias'] = 'bias'
        maximum_bias = 2**31 - 1 - 3 * 8 * 128
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'graph.json').write_text(json.dumps(graph))
            for sign in (1, -1):
                weights['bias'] = np.array([sign * maximum_bias], np.int32)
                (directory / 'model.safetensors').write_bytes(save(weights))
                with vx.InferenceSession(directory, cpu_threads=1) as session:
                    output = session.run({'x': np.zeros((1, 1, 3, 1), np.int8)})['y']
                    np.testing.assert_array_equal(output, 127 if sign > 0 else -128)
                weights['bias'] = np.array([sign * (maximum_bias + 1)], np.int32)
                (directory / 'model.safetensors').write_bytes(save(weights))
                # The immutable descriptor is rejected while loading/compiling;
                # no Execute call may be needed to discover an invalid bound.
                with self.assertRaises(vx.VolvoxAIError):
                    with vx.InferenceSession(directory, cpu_threads=1):
                        pass

    def test_invalid_static_conv4_policies(self):
        graph, weights, _, _, _ = source(True)
        for algorithm, reduce_range in ((pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE, True),
                                         (pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP, True)):
            config = policy(algorithm)
            config.static_activations.reduce_weight_range = reduce_range
            with vx.open_library() as host:
                client = vx.VxQuantizationServiceClient(host)
                with self.assertRaises(vx.VolvoxAIError):
                    client.create_quantization_plan(pb.CreateQuantizationPlanRequest(
                        bytes=pb.ModelBytes(graph=json.dumps(graph).encode(), weight_shards=[weights]), config=config))

    def test_static_conv4_does_not_silently_pack_linear(self):
        graph = {'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'x': {'shape': [1, 3], 'dtype': 'float32'}},
            'nodes': [{'id': 'dense', 'opType': 'Linear', 'inputs': {'input': 'x', 'weight': 'w'},
                'outputs': {'out': {'tensor': 'y', 'shape': [1, 2], 'dtype': 'float32'}},
                'params': {'weight_layout': 'dout_din'}}], 'outputs': ['y'],
            'execution': {'stateless': {}, 'entrypoints': [{'name': 'evaluate', 'effect': 1,
                'allowed_ranges': [1], 'inputs': ['x'], 'outputs': ['y']}]}}
        with vx.open_library() as host:
            with self.assertRaisesRegex(vx.VolvoxAIError, 'unsupported weight/activation combination'):
                vx.VxQuantizationServiceClient(host).create_quantization_plan(pb.CreateQuantizationPlanRequest(
                    bytes=pb.ModelBytes(graph=json.dumps(graph).encode(),
                        weight_shards=[save({'w': np.ones((2, 3), np.float32)})]), config=policy()))


if __name__ == '__main__':
    unittest.main()
