"""End-to-end ONNX export and native execution for non-autoregressive TTS + vocoder primitives."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.export_safetensors import export_onnx_model
from tools.tts_vocoder_fixtures import build_kokoro_fixture, build_tts_vocoder_fixture


class NonAutoregressiveTtsVocoderExportTests(unittest.TestCase):
    def _export_and_compare(self, nodes, inputs, initializers, expected, **export_options):
        def value_info(name, value):
            dtype = TensorProto.INT32 if value.dtype == np.int32 else TensorProto.FLOAT
            return helper.make_tensor_value_info(name, dtype, list(value.shape))

        model = helper.make_model(helper.make_graph(
            nodes, "tts_primitives",
            [value_info(name, value) for name, value in inputs.items()],
            [value_info(name, value) for name, value in expected.items()],
            [numpy_helper.from_array(value, name) for name, value in initializers.items()],
        ), opset_imports=[helper.make_opsetid("", 17)])
        return self._compare_fixture(model, inputs, expected, **export_options)

    def _compare_fixture(self, model, inputs, expected, **export_options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.onnx"
            onnx.save(model, source)
            graph = export_onnx_model(str(source), str(root / "pkg/model.safetensors"), **export_options)
            native_bin = Path(__file__).resolve().parents[2] / "native/volvoxai-lite"
            if not native_bin.exists():
                self.skipTest("native inference binary required for numerical parity")
            command = [str(native_bin), "run", str(root / "pkg")]
            for index, value in enumerate(inputs.values()):
                suffix = "i32" if value.dtype == np.int32 else "f32"
                path = root / f"input{index}.{suffix}"
                value.tofile(path)
                shape = ",".join(str(dimension) for dimension in value.shape)
                command += ["--input", f"input{index}[{shape}]={path}"]
            for start in range(0, len(expected), 16):
                outputs = []
                for index in range(start, min(start + 16, len(expected))):
                    suffix = "i32" if list(expected.values())[index].dtype == np.int32 else "f32"
                    outputs += ["--output", f"output{index}={root}/output{index}.{suffix}"]
                result = subprocess.run([*command, *outputs], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for index, value in enumerate(expected.values()):
                suffix = "i32" if value.dtype == np.int32 else "f32"
                actual = np.fromfile(root / f"output{index}.{suffix}", dtype=value.dtype).reshape(value.shape)
                if value.dtype == np.int32:
                    np.testing.assert_array_equal(actual, value)
                else:
                    np.testing.assert_allclose(actual, value, rtol=1e-5, atol=1e-5)
            return graph

    def test_constant_unary_operands_are_materialized(self):
        constant = np.asarray([[-1.25, 0.0, 0.75]], dtype=np.float32)
        x = np.asarray([[2.0, 3.0, 4.0]], dtype=np.float32)
        operations = {
            "Exp": np.exp(constant), "Sin": np.sin(constant), "Cos": np.cos(constant),
            "LeakyRelu": np.where(constant >= 0, constant, constant * np.float32(0.2)),
            "CumSum": np.cumsum(constant, axis=1),
        }
        nodes, expected = [], {}
        for op, value in operations.items():
            attrs = {"alpha": 0.2} if op == "LeakyRelu" else {}
            nodes.extend([
                helper.make_node(op, ["constant", "axis"] if op == "CumSum" else ["constant"],
                                 [f"{op}_constant"], **attrs),
                helper.make_node("Add", ["x", f"{op}_constant"], [op]),
            ])
            expected[op] = x + value
        for weight_dtype in ("float32", "float16"):
            with self.subTest(weight_dtype=weight_dtype):
                self._export_and_compare(nodes, {"x": x},
                                         {"constant": constant, "axis": np.asarray(1, dtype=np.int64)}, expected,
                                         weight_dtype=weight_dtype)

    def test_cumsum_modes_dtypes_and_axes(self):
        inputs = {
            "f32": (np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 12) * 0.125,
            "i32": np.arange(24, dtype=np.int32).reshape(2, 3, 4) - 12,
        }
        inputs["i32"][0, 0, :2] = [2**31 - 1, 1]
        nodes, constants, expected = [], {}, {}
        for name, value in inputs.items():
            for axis in range(-3, 3):
                axis_name = f"axis_{axis}"
                constants[axis_name] = np.asarray(axis, dtype=np.int64)
                for exclusive in (0, 1):
                    for reverse in (0, 1):
                        output = f"{name}_{axis}_{exclusive}_{reverse}"
                        nodes.append(helper.make_node("CumSum", [name, axis_name], [output],
                                                      exclusive=exclusive, reverse=reverse))
                        sequence = np.flip(value, axis) if reverse else value
                        result = np.cumsum(sequence, axis=axis, dtype=value.dtype)
                        if exclusive:
                            result = np.roll(result, 1, axis=axis)
                            first = [slice(None)] * value.ndim
                            first[axis] = 0
                            result[tuple(first)] = 0
                        expected[output] = np.flip(result, axis) if reverse else result
        self._export_and_compare(nodes, inputs, constants, expected)

    def test_exp_full_range_and_cumsum_f32_rounding(self):
        x = np.asarray([-90, 88.5, 89, np.inf, -np.inf, np.nan], dtype=np.float32)
        sequence = np.asarray([2**24, 1, -(2**24), 1], dtype=np.float32)
        with np.errstate(over="ignore"):
            exponential = np.exp(x)
        self._export_and_compare([
            helper.make_node("Exp", ["x"], ["exp"]),
            helper.make_node("CumSum", ["sequence", "axis"], ["scan"]),
        ], {"x": x, "sequence": sequence}, {"axis": np.asarray(0, dtype=np.int64)}, {
            "exp": exponential,
            "scan": np.cumsum(sequence, dtype=np.float32),
        })

    def test_conv_transpose_output_shape_overrides_padding(self):
        x = np.asarray([[[1, 2, 4]]], dtype=np.float32)
        w = np.asarray([[[1, 2, 3]]], dtype=np.float32)
        full = np.asarray([[[1, 2, 5, 4, 10, 8, 12]]], dtype=np.float32)
        # Even and odd inferred padding; supplied pads must be ignored.
        for length, start in ((5, 1), (4, 2)):
            with self.subTest(length=length):
                self._export_and_compare([
                    helper.make_node("ConvTranspose", ["x", "w"], ["y"],
                                     strides=[2], pads=[0, 0], output_shape=[length]),
                ], {"x": x}, {"w": w}, {"y": full[:, :, start:start + length]})

    def test_conv_transpose_shared_weights_with_different_strides(self):
        x = np.asarray([[[1, 2, 4]]], dtype=np.float32)
        w = np.asarray([[[1, 2, 3]]], dtype=np.float32)
        b = np.asarray([0.5], dtype=np.float32)
        self._export_and_compare([
            helper.make_node("ConvTranspose", ["x", "w", "b"], ["s1"], strides=[1]),
            helper.make_node("ConvTranspose", ["x", "w", "b"], ["s2"], strides=[2]),
        ], {"x": x}, {"w": w, "b": b}, {
            "s1": np.asarray([[[1, 4, 11, 14, 12]]], dtype=np.float32) + b[None, :, None],
            "s2": np.asarray([[[1, 2, 5, 4, 10, 8, 12]]], dtype=np.float32) + b[None, :, None],
        })

    def test_conv_transpose_multichannel_asymmetric_and_sparse_phases(self):
        rng = np.random.default_rng(93)
        x = rng.standard_normal((2, 3, 4), dtype=np.float32)
        for stride, kernel, pads in ((2, 5, [1, 2]), (4, 2, [0, 1])):
            with self.subTest(stride=stride, kernel=kernel):
                w = rng.standard_normal((3, 2, kernel), dtype=np.float32) * 0.1
                b = rng.standard_normal((2,), dtype=np.float32) * 0.1
                full_length = (x.shape[2] - 1) * stride + kernel
                full = np.zeros((2, 2, full_length), dtype=np.float32)
                for t in range(x.shape[2]):
                    for k in range(kernel):
                        full[:, :, t * stride + k] += x[:, :, t] @ w[:, :, k]
                expected = full[:, :, pads[0]:full_length - pads[1]] + b[None, :, None]
                self._export_and_compare([
                    helper.make_node("ConvTranspose", ["x", "w", "b"], ["y"], strides=[stride], pads=pads),
                ], {"x": x}, {"w": w, "b": b}, {"y": expected})

    def test_conv1d_grouped_depthwise_stride_and_dilation(self):
        rng = np.random.default_rng(51)
        x = rng.standard_normal((2, 4, 9), dtype=np.float32)
        for groups, out_channels, stride, dilation, pad in ((1, 6, 2, 1, 1), (2, 6, 1, 2, 2), (4, 8, 2, 2, 2)):
            with self.subTest(groups=groups):
                w = rng.standard_normal((out_channels, 4 // groups, 3), dtype=np.float32) * 0.1
                b = rng.standard_normal((out_channels,), dtype=np.float32) * 0.1
                length = (x.shape[2] + 2 * pad - dilation * 2 - 1) // stride + 1
                padded = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
                expected = np.zeros((2, out_channels, length), dtype=np.float32)
                for c in range(out_channels):
                    begin = (c // (out_channels // groups)) * (4 // groups)
                    for t in range(length):
                        for k in range(3):
                            expected[:, c, t] += padded[:, begin:begin + 4 // groups, t * stride + k * dilation] @ w[c, :, k]
                expected += b[None, :, None]
                self._export_and_compare([
                    helper.make_node("Conv", ["x", "w", "b"], ["y"], group=groups,
                                     strides=[stride], dilations=[dilation], pads=[pad, pad]),
                ], {"x": x}, {"w": w, "b": b}, {"y": expected})

    def test_conv1d_shared_depthwise_weights_with_different_groups(self):
        rng = np.random.default_rng(52)
        w = rng.standard_normal((8, 1, 3), dtype=np.float32) * 0.1
        inputs, expected, nodes = {}, {}, []
        for groups in (2, 4):
            x = rng.standard_normal((2, groups, 6), dtype=np.float32)
            name, output = f"x{groups}", f"y{groups}"
            inputs[name] = x
            padded = np.pad(x, ((0, 0), (0, 0), (1, 1)))
            value = np.zeros((2, 8, 6), dtype=np.float32)
            for channel in range(8):
                source_channel = channel // (8 // groups)
                for kernel_index in range(3):
                    value[:, channel, :] += padded[:, source_channel, kernel_index:kernel_index + 6] * w[channel, 0, kernel_index]
            expected[output] = value
            nodes.append(helper.make_node("Conv", [name, "w"], [output], group=groups, pads=[1, 1]))
        self._export_and_compare(nodes, inputs, {"w": w}, expected)

    def test_instance_norm_rank4_multiple_batches(self):
        rng = np.random.default_rng(71)
        x = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)
        scale = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
        bias = np.asarray([-0.1, 0.0, 0.1], dtype=np.float32)
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        variance = np.var(x, axis=(2, 3), keepdims=True)
        expected = (x - mean) / np.sqrt(variance + 1e-4) * scale[None, :, None, None] + bias[None, :, None, None]
        self._export_and_compare([
            helper.make_node("InstanceNormalization", ["x", "scale", "bias"], ["y"], epsilon=1e-4),
        ], {"x": x}, {"scale": scale, "bias": bias}, {"y": expected})

    def test_non_autoregressive_tts_vocoder_single_execute_parity(self):
        self._compare_fixture(*build_tts_vocoder_fixture())

    def test_kokoro_snake_and_instancenorm_parity(self):
        self._compare_fixture(*build_kokoro_fixture())


if __name__ == "__main__":
    unittest.main()
