"""CUDA PTQ device observations through the public calibration lifecycle.

Run on physical CUDA with VOLVOXAI_TEST_CUDA=1. The exported low-bit model is
executed on CPU: CUDA authoring does not imply CUDA low-bit inference support.
"""
from __future__ import annotations

import copy
import json
import os
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
from test_ptq_lowbit_importance import calibrate, export, fit, infer, pack, weight_policy

pb = vx.pb
RTN = pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN
MSE = pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP
IMPORTANCE = pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE


def static_policy(bits, algorithm):
    return pb.QuantizationConfig(weights=weight_policy(bits, 0, algorithm),
        static_activations=pb.StaticActivations(dtype=pb.DataType.DATA_TYPE_I8,
            scheme=pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC))


@unittest.skipUnless(os.environ.get('VOLVOXAI_TEST_CUDA') == '1', 'physical CUDA test is opt-in')
class CudaPTQTest(unittest.TestCase):
    def calibrate_package(self, graph, weight_bytes, config, batches, backend, files=False):
        with tempfile.TemporaryDirectory() as directory, vx.open_library() as host:
            client = vx.VxQuantizationServiceClient(host)
            source = pb.ModelBytes(graph=json.dumps(graph).encode(), weight_shards=[weight_bytes])
            request = pb.CreateQuantizationPlanRequest(bytes=source, config=config,
                calibration=pb.CalibrationOptions(backend=backend, cpu_threads=1))
            if files:
                base = Path(directory)
                (base / 'graph.json').write_bytes(source.graph)
                (base / 'model.safetensors').write_bytes(source.weight_shards[0])
                request.ClearField('bytes')
                request.files.CopyFrom(pb.ModelFiles(graph_path=str(base / 'graph.json'),
                    weight_paths=[str(base / 'model.safetensors')]))
            plan = client.create_quantization_plan(request)
            ref = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
            self.assertEqual(plan.info.calibration_backend, backend)
            try:
                # Accepted snapshots must survive all caller/source mutation.
                request.bytes.graph = b'invalid after acceptance'
                if files:
                    (base / 'graph.json').write_bytes(b'invalid')
                    (base / 'model.safetensors').write_bytes(b'invalid')
                with self.assertRaises(vx.VolvoxAIError):
                    export(client, plan.quantization_plan_id)
                for index, values in enumerate(batches):
                    calibrate(client, plan.quantization_plan_id, str(index), values)
                package = export(client, plan.quantization_plan_id)
                inspected = client.inspect_quantization_plan(ref).info
                self.assertEqual(inspected.calibration_backend, backend)
                self.assertEqual(inspected.calibration.samples, len(batches))
                for special in (np.nan, np.inf, -np.inf):
                    invalid = {name: value.copy() for name, value in batches[0].items()}
                    next(iter(invalid.values())).reshape(-1)[-1] = special
                    with self.assertRaises(vx.VolvoxAIError):
                        calibrate(client, plan.quantization_plan_id, 'reusable-failed-name', invalid)
                    self.assertEqual(export(client, plan.quantization_plan_id), package)
                    self.assertEqual(client.inspect_quantization_plan(ref).info.calibration.samples, len(batches))
                # Failure must not poison scheduling flags, sample name, or device slots.
                calibrate(client, plan.quantization_plan_id, 'reusable-failed-name', batches[0])
                self.assertEqual(client.inspect_quantization_plan(ref).info.calibration.samples, len(batches) + 1)
                self.assertEqual(export(client, plan.quantization_plan_id),
                                 export(client, plan.quantization_plan_id))
                return package
            finally:
                client.release_quantization_plan(ref)

    def test_static_conv_ranges_importance_grouped_geometry_and_snapshot(self):
        for bits, with_bias in ((2, True), (3, True), (4, True),
                                (8, True), (8, False)):
            with self.subTest(bits=bits, with_bias=with_bias):
                graph, weights, _, _, image = conv_source(False, with_bias=with_bias)
                graph['nodes'][1]['params']['dilation'] = [2, 1]
                graph['nodes'][1]['outputs']['out']['shape'][1] = 4
                batches = [{'image': image}, {'image': np.ascontiguousarray(image[:, ::-1] * np.float32(1.7))}]
                config = static_policy(bits, RTN if bits == 8 else IMPORTANCE)
                cpu = self.calibrate_package(graph, weights, config, batches, 'cpu')
                gpu = self.calibrate_package(graph, weights, config, batches, 'cuda', files=True)
                cpu_graph, gpu_graph = json.loads(cpu.graph), json.loads(gpu.graph)
                cpu_arrays, gpu_arrays = load(cpu.weight_shards[0]), load(gpu.weight_shards[0])
                cpu_conv = next(n for n in cpu_graph['nodes'] if n['opType'] == 'QConv2D')
                gpu_conv = next(n for n in gpu_graph['nodes'] if n['opType'] == 'QConv2D')
                np.testing.assert_array_equal(cpu_arrays[cpu_conv['inputs']['weight']], gpu_arrays[gpu_conv['inputs']['weight']])
                np.testing.assert_array_equal(cpu_arrays[cpu_conv['inputs']['bias']], gpu_arrays[gpu_conv['inputs']['bias']])
                if not with_bias:
                    # Static authoring retains a synthetic zero I32 bias;
                    # CUDA calibration must preserve that host export policy.
                    np.testing.assert_array_equal(gpu_arrays[gpu_conv['inputs']['bias']],
                                                  np.zeros(6, dtype=np.int32))
                for port in ('input', 'weight', 'out'):
                    name = gpu_conv['inputs'][port] if port != 'out' else gpu_conv['outputs']['out']['tensor']
                    affine = gpu_graph['quantization']['tensors'][name]
                    np.testing.assert_allclose(gpu_arrays[affine['scale_tensor']], cpu_arrays[affine['scale_tensor']], rtol=2e-5, atol=1e-7)
                    np.testing.assert_array_equal(gpu_arrays[affine['zero_point_tensor']], cpu_arrays[affine['zero_point_tensor']])
                np.testing.assert_allclose(infer(gpu, batches[0])['features'], infer(cpu, batches[0])['features'], rtol=3e-5, atol=2e-6)

    def test_linear_importance_exact_moments_and_real_lowbit_storage(self):
        rng = np.random.default_rng(912)
        # More than 256*256 values exercises the bounded range-reduction
        # grid-stride path, not just a single CUDA block.
        rows, channels, outputs = 1025, 67, 5
        weights = rng.normal(0, .4, (outputs, channels)).astype(np.float32)
        graph = {'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'x': {'shape': [rows, channels], 'dtype': 'float32'}},
            'nodes': [{'id': 'linear', 'opType': 'Linear',
                'inputs': {'input': 'x', 'weight': 'w'},
                'outputs': {'out': {'tensor': 'y', 'shape': [rows, outputs], 'dtype': 'float32'}},
                'params': {'weight_layout': 'dout_din'}}], 'outputs': ['y'],
            'execution': {'stateless': {}, 'entrypoints': [{'name': 'evaluate', 'effect': 1,
                'allowed_ranges': [1], 'inputs': ['x'], 'outputs': ['y']}]}}
        samples = [rng.normal(0, 1, (rows, channels)).astype(np.float32) for _ in range(2)]
        samples[0][:, ::3] *= 90
        samples[1][:, 1::3] *= 50
        energy = sum(np.sum(x.astype(np.float64)**2, axis=0) for x in samples)
        for bits in (2, 3, 4):
            with self.subTest(bits=bits):
                config = pb.QuantizationConfig(weights=weight_policy(bits, 32, IMPORTANCE))
                batches = [{'x': x} for x in samples]
                cpu = self.calibrate_package(graph, save({'w': weights}), config, batches, 'cpu')
                gpu = self.calibrate_package(graph, save({'w': weights}), config, batches, 'cuda')
                self.assertEqual(gpu.graph, cpu.graph)
                self.assertEqual(gpu.weight_shards, cpu.weight_shards)
                node = json.loads(gpu.graph)['nodes'][0]
                arrays = load(gpu.weight_shards[0])
                q, scales = fit(weights, bits, IMPORTANCE, np.broadcast_to(energy, weights.shape), True)
                np.testing.assert_array_equal(arrays[node['inputs']['weight']], pack(q, bits, 32))
                np.testing.assert_array_equal(arrays[node['inputs']['scale']], scales)
                self.assertTrue(np.isfinite(infer(gpu, batches[0])['y']).all())

    def test_packed_suffix_can_be_omitted_but_required_prefix_cannot_fallback(self):
        for bits in (2, 3):
            with self.subTest(bits=bits):
                graph, weights, _, _, image = conv_source(True)
                arrays = load(weights)
                arrays['head'] = np.linspace(-.4, .5, 12, dtype=np.float32).reshape(3, 4)
                graph['nodes'].append({'id': 'head', 'opType': 'Linear',
                    'inputs': {'input': 'features', 'weight': 'head'},
                    'outputs': {'out': {'tensor': 'logits', 'shape': [1, 3, 4, 3], 'dtype': 'float32'}},
                    'params': {'weight_layout': 'dout_din'}})
                graph['outputs'] = ['logits']
                graph['execution']['entrypoints'][0]['outputs'] = ['logits']
                with vx.open_library() as host:
                    client = vx.VxQuantizationServiceClient(host)
                    config = pb.QuantizationConfig(weights=weight_policy(bits, 32, MSE), rules=[
                        pb.QuantizationRule(node_id='conv', keep_float=pb.Empty())])
                    plan = client.create_quantization_plan(pb.CreateQuantizationPlanRequest(
                        bytes=pb.ModelBytes(graph=json.dumps(graph).encode(), weight_shards=[save(arrays)]), config=config))
                    packed = export(client, plan.quantization_plan_id)
                    client.release_quantization_plan(pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id))
                packed_graph = json.loads(packed.graph)
                config = static_policy(4, MSE)
                config.rules.append(pb.QuantizationRule(node_id='head', keep_float=pb.Empty()))
                gpu = self.calibrate_package(packed_graph, packed.weight_shards[0], config, [{'image': image}], 'cuda')
                self.assertTrue(np.isfinite(infer(gpu, {'image': image})['logits']).all())
                final_head = next(n for n in json.loads(gpu.graph)['nodes'] if n['id'] == 'head')
                self.assertEqual(final_head['params']['weight_bits'], bits)

                # Move an unsupported PackedLinear into the required prefix.
                # Same input/output width keeps the graph's shape contract valid.
                blocked = copy.deepcopy(packed_graph)
                head = next(n for n in blocked['nodes'] if n['id'] == 'head')
                conv = next(n for n in blocked['nodes'] if n['id'] == 'conv')
                head['inputs']['input'] = 'viewed'
                head['outputs']['out']['tensor'] = 'blocked'
                head['outputs']['out']['shape'] = [1, 3, 7, 3]
                # A separate independent unsupported branch before Conv is
                # still in the conservative prefix and must fail closed.
                # Match its K=4 with a declared input rather than Conv's C=2.
                blocked['inputs']['extra'] = {'shape': [1, 3, 7, 4], 'dtype': 'float32'}
                blocked['execution']['entrypoints'][0]['inputs'].append('extra')
                head['inputs']['input'] = 'extra'
                blocked['nodes'].remove(head)
                blocked['nodes'].insert(1, head)
                blocked['outputs'] = ['features', 'blocked']
                blocked['execution']['entrypoints'][0]['outputs'] = ['features', 'blocked']
                with vx.open_library() as host:
                    client = vx.VxQuantizationServiceClient(host)
                    plan = client.create_quantization_plan(pb.CreateQuantizationPlanRequest(
                        bytes=pb.ModelBytes(graph=json.dumps(blocked).encode(), weight_shards=packed.weight_shards),
                        config=config, calibration=pb.CalibrationOptions(backend='cuda')))
                    ref = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
                    try:
                        for attempt in range(2):
                            with self.assertRaises(vx.VolvoxAIError):
                                calibrate(client, plan.quantization_plan_id, str(attempt),
                                    {'image': image, 'extra': np.zeros((1, 3, 7, 4), np.float32)})
                            self.assertEqual(client.inspect_quantization_plan(ref).info.calibration.samples, 0)
                    finally:
                        client.release_quantization_plan(ref)


if __name__ == '__main__':
    unittest.main()
