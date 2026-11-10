"""CPU backward/QAT thread parity against independent parameter-delta oracles.

Run with VOLVOXAI_LIBRARY pointing to the full-profile library under test.
The test never installs a runtime or changes the source model files.
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import volvoxai as vx
from execution_fixture import stateless_graph
from test_qat_training import fit_rows, qat


def node(name, op, inputs, shape, params=None):
    return {'id': name, 'opType': op, 'inputs': inputs,
            'outputs': {'out': {'tensor': name, 'dtype': 'float32', 'shape': list(shape)}},
            'params': params or {}}


def derivative(logits, targets):
    values = logits.astype(np.float64).reshape(len(targets), -1)
    p = np.exp(values - values.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    p[np.arange(len(targets)), targets] -= 1
    return (p / len(targets)).astype(np.float32).reshape(logits.shape)


def unbroadcast(value, shape):
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(axis=axis, keepdims=True)
    return value.reshape(shape)


class CpuTrainingThreads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cpu-training-threads-')
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name)
        self.rng = np.random.default_rng(1029)

    def random(self, shape, scale=.2):
        return self.rng.normal(0, scale, shape).astype(np.float32)

    def compare(self, x, weights, nodes, logits, grads, *, targets=None, policy=None, microbatches=1):
        targets = np.asarray([int(np.argmax(logits))] if targets is None else targets, np.int32)
        if targets.size == 1 and logits.shape != (1, logits.size):
            nodes += [node('logits', 'Reshape', {'input': nodes[-1]['id']}, (1, logits.size),
                           {'shape': [1, logits.size]})]
        elif nodes[-1]['id'] != 'logits':
            nodes += [node('logits', 'Reshape', {'input': nodes[-1]['id']}, logits.shape,
                           {'shape': list(logits.shape)})]
        graph = stateless_graph({'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'x': {'dtype': 'float32', 'shape': list(x.shape)}},
            'nodes': nodes, 'outputs': ['logits']})
        (self.model / 'graph.json').write_text(json.dumps(graph))
        save_file({key: np.ascontiguousarray(value) for key, value in weights.items()},
                  self.model / 'model.safetensors')
        originals = {file.name: file.read_bytes() for file in self.model.iterdir()}
        results = {}
        for threads in (1, 2, 4):
            with vx.TrainingSession(self.model, cpu_threads=threads,
                    loss=vx.CrossEntropyLoss('logits', normalizer=targets.size * microbatches),
                    optimizer=vx.SGD(lr=.25), qat=policy) as trainer:
                for step in range(microbatches):
                    with trainer.step(x, targets, accumulation_steps=microbatches,
                                      output_names=['logits']) as output:
                        np.testing.assert_allclose(output['logits'].numpy().reshape(logits.shape), logits,
                                                   rtol=5e-5, atol=3e-6)
                    if step + 1 < microbatches:
                        with trainer.parameters(list(grads)) as tensors:
                            for name in grads:
                                np.testing.assert_array_equal(tensors[name].numpy(), weights[name])
                with trainer.parameters(list(grads)) as tensors:
                    results[threads] = {name: tensors[name].numpy().copy() for name in grads}
            for name, gradient in grads.items():
                actual_gradient = (weights[name] - results[threads][name]) / np.float32(.25)
                # Compare the recovered gradient, not only the loss or a tiny weight update.
                np.testing.assert_allclose(actual_gradient, gradient, rtol=3e-4, atol=2e-6,
                                           err_msg=f'{name}, cpu_threads={threads}')
        for threads in (2, 4):
            for name in grads:
                np.testing.assert_array_equal(results[threads][name], results[1][name],
                                              err_msg=f'{name}: thread-count drift')
        self.assertEqual(originals, {file.name: file.read_bytes() for file in self.model.iterdir()})

    def test_linear_layouts_shared_gradients_and_microbatch_accumulation(self):
        for layout in ('din_dout', 'dout_din'):
            for shared in (False, True):
                with self.subTest(layout=layout, shared=shared):
                    x = self.random((17, 65)); scale = self.random(x.shape, .5)
                    w = self.random((65, 33)); b = self.random((33,))
                    value = x * scale
                    nodes = [node('scaled', 'Mul', {'a': 'x', 'b': 'scale'}, x.shape)]
                    args = {'input': 'scaled', 'weight': 'weight', 'bias': 'bias'}
                    nodes.append(node('projection', 'Linear', args, (17, 33), {'weight_layout': layout}))
                    factor = 2 if shared else 1
                    if shared:
                        nodes += [node('projection2', 'Linear', args, (17, 33), {'weight_layout': layout}),
                                  node('sum', 'Add', {'a': 'projection', 'b': 'projection2'}, (17, 33))]
                    logits = factor * (value @ w + b)
                    targets = np.arange(17, dtype=np.int32) % 33
                    dy = factor * derivative(logits, targets)
                    dw = value.T @ dy
                    self.compare(x, {'scale': scale, 'weight': w if layout == 'din_dout' else w.T,
                                     'bias': b}, nodes, logits,
                        {'scale': (dy @ w.T) * x, 'weight': dw if layout == 'din_dout' else dw.T,
                         'bias': dy.sum(axis=0)}, targets=targets, microbatches=2)

    def test_batch_matmul_broadcast_and_alias_accumulate_both_operands(self):
        shapes = [((2, 3, 13, 17), (2, 3, 17, 19)),
                  ((2, 1, 13, 17), (1, 3, 17, 19)),
                  ((13, 17), (2, 3, 17, 19)), ((2, 17, 17), None)]
        for a_shape, b_shape in shapes:
            with self.subTest(a=a_shape, b=b_shape):
                a = self.random(a_shape); b = a if b_shape is None else self.random(b_shape)
                x = np.ones_like(a)
                logits = np.matmul(a, b)
                target = [int(np.argmax(logits))]
                dy = derivative(logits, target)
                da = unbroadcast(np.matmul(dy, np.swapaxes(b, -1, -2)), a.shape)
                db = unbroadcast(np.matmul(np.swapaxes(a, -1, -2), dy), b.shape)
                weights = {'a': a}; grads = {'a': da + db if b_shape is None else da}
                if b_shape is not None:
                    weights['b'] = b; grads['b'] = db
                nodes = [node('av', 'Mul', {'a': 'x', 'b': 'a'}, a.shape),
                         node('product', 'BatchMatMul', {'a': 'av', 'b': 'av' if b_shape is None else 'b'}, logits.shape)]
                self.compare(x, weights, nodes, logits, grads, microbatches=2)

    def conv_case(self, layout, groups, *, relu=0, f16=False, policy=None):
        n, h, width, channels, out_channels, kh, kw = 2, 12, 13, 4, 8, 3, 2
        stride, dilation, pads = (2, 1), (2, 1), (1, 2, 2, 0)
        oh = (h + pads[0] + pads[2] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (width + pads[1] + pads[3] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        x = self.random((n, h, width, channels)); scale = self.random(x.shape, .7)
        value = x * scale
        shape = ((kh, kw, channels // groups, out_channels) if layout == 'HWIO' else
                 (out_channels, kh, kw, channels // groups) if layout == 'OHWI' else
                 (kh, kw, channels, out_channels // channels) if layout == 'HWCM' else
                 (1, kh, kw, out_channels))
        w = self.random(shape).astype(np.float16 if f16 else np.float32)
        b = np.linspace(-1, 7, out_channels, dtype=np.float32) if relu else self.random((out_channels,))
        fake, mask = w.astype(np.float32), np.ones(w.shape, bool)
        if policy is not None:
            canonical = w.transpose(3, 0, 1, 2).reshape(out_channels, -1) if layout == 'HWIO' else w.reshape(out_channels, -1)
            fake_rows, mask_rows = fit_rows(canonical, 3, group=0, mse=True, conv=True)
            fake = fake_rows.reshape(out_channels, kh, kw, channels // groups)
            mask = mask_rows.reshape(fake.shape)
            if layout == 'HWIO':
                fake, mask = fake.transpose(1, 2, 3, 0), mask.transpose(1, 2, 3, 0)
        output = np.zeros((n, oh, ow, out_channels), np.float32)
        visits = []
        for bn in range(n):
            for oy in range(oh):
                for ox in range(ow):
                    for oc in range(out_channels):
                        oi = (bn, oy, ox, oc); output[oi] = b[oc]
                        for icl in range(channels // groups):
                            ic = oc // (out_channels // groups) * (channels // groups) + icl
                            for ky in range(kh):
                                iy = oy * stride[0] - pads[0] + ky * dilation[0]
                                if not 0 <= iy < h: continue
                                for kx in range(kw):
                                    ix = ox * stride[1] - pads[1] + kx * dilation[1]
                                    if not 0 <= ix < width: continue
                                    ii = (bn, iy, ix, ic)
                                    wi = ((ky, kx, icl, oc) if layout == 'HWIO' else
                                          (oc, ky, kx, icl) if layout == 'OHWI' else
                                          (ky, kx, ic, oc % (out_channels // channels)) if layout == 'HWCM' else
                                          (0, ky, kx, oc))
                                    output[oi] += value[ii] * fake[wi]
                                    visits.append((oi, ii, wi))
        logits = np.maximum(output, 0) if relu else output.copy()
        if relu == 2: logits = np.minimum(logits, 6)
        target = [int(np.argmax(logits))]
        dy = derivative(logits, target)
        if relu: dy *= output > 0
        if relu == 2: dy *= output < 6
        dx = np.zeros_like(value); dw = np.zeros(w.shape, np.float32)
        for oi, ii, wi in visits:
            dx[ii] += dy[oi] * fake[wi]; dw[wi] += dy[oi] * value[ii]
        nodes = [node('scaled', 'Mul', {'a': 'x', 'b': 'scale'}, x.shape),
                 node('conv', 'Conv2D', {'input': 'scaled', 'weight': 'weight', 'bias': 'bias'}, logits.shape,
                      {'data_layout': 'NHWC', 'weight_layout': layout, 'groups': groups,
                       'stride': list(stride), 'dilation': list(dilation), 'pads': list(pads), 'relu': relu})]
        grads = {'scale': dx * x, 'bias': dy.sum(axis=(0, 1, 2))}
        if not f16: grads['weight'] = dw * mask
        self.compare(x, {'scale': scale, 'weight': w, 'bias': b}, nodes, logits, grads,
                     policy=policy, microbatches=2)

    def test_conv_layouts_groups_depthwise_asymmetric_pads_stride_dilation(self):
        for layout, groups in [('HWIO', 1), ('HWIO', 2), ('OHWI', 1), ('HWCM', 4)]:
            with self.subTest(layout=layout, groups=groups): self.conv_case(layout, groups)

    def test_conv_relu_relu6_and_frozen_f16_weight_input_gradient(self):
        for relu in (1, 2):
            with self.subTest(relu=relu): self.conv_case('HWIO', 1, relu=relu)
        self.conv_case('HWIO', 1, f16=True)

    def test_qat_parallel_scale_fitting_and_ste(self):
        for layout in ('HWIO', 'OHWI'):
            with self.subTest(layout=layout): self.conv_case(layout, 1, policy=qat(3, 0, 'mse', 'conv'))
        for bits in (2, 3, 4):
            with self.subTest(bits=bits):
                x = self.random((17, 65)); w = self.random((65, 33)); scale = self.random(x.shape, .7)
                fake, mask = fit_rows(w.T, bits, group=32, mse=True)
                value = x * scale; logits = value @ fake.T
                targets = np.arange(17, dtype=np.int32) % 33
                dy = derivative(logits, targets)
                nodes = [node('scaled', 'Mul', {'a': 'x', 'b': 'scale'}, x.shape),
                         node('projection', 'Linear', {'input': 'scaled', 'weight': 'weight'}, logits.shape,
                              {'weight_layout': 'din_dout'})]
                self.compare(x, {'scale': scale, 'weight': w}, nodes, logits,
                    {'scale': (dy @ fake) * x, 'weight': (value.T @ dy) * mask.T},
                    targets=targets, policy=qat(bits, 32, 'mse'), microbatches=2)


if __name__ == '__main__': unittest.main(verbosity=2)
