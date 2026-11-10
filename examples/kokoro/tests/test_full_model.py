import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper

from examples.kokoro.full_model import lower_geometry
from examples.kokoro.prepare import session


class GeometryLoweringTests(unittest.TestCase):
    def compare(self, node, values, input_shape, output_shape, input_values, *, atol=1e-6, rtol=1e-5):
        graph = helper.make_graph([node], "kokoro-geometry",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, output_shape)],
            [numpy_helper.from_array(value, name) for name, value in values.items()])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
        lowered, evidence = lower_geometry(model)
        expected = session(model.SerializeToString(), 1).run(None, {"x": input_values})[0]
        actual = session(lowered.SerializeToString(), 1).run(None, {"x": input_values})[0]
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        self.assertEqual(len(evidence), 1)

    def test_windowed_real_stft(self):
        values = {"hop": np.array(5, dtype=np.int64), "length": np.array(20, dtype=np.int64),
            "window": (.5 - .5 * np.cos(2 * np.pi * np.arange(20) / 20)).astype(np.float32)}
        for length, rank in ((40, 3), (95, 2)):
            shape = [1, length, 1] if rank == 3 else [1, length]
            signal = np.random.default_rng(length).normal(0, .2, shape).astype(np.float32)
            self.compare(helper.make_node("STFT", ["x", "hop", "window", "length"], ["y"], onesided=1),
                values, shape, [1, (length - 20) // 5 + 1, 11, 2], signal, atol=2e-7)

    def test_actual_resize_coordinate_modes(self):
        for size, scale, mode, coordinates in [
            (7, 2., "nearest", "asymmetric"), (3, 300., "nearest", "asymmetric"),
            (900, 1 / 300., "linear", "half_pixel"), (7, 300., "linear", "half_pixel"),
        ]:
            with self.subTest(size=size, scale=scale, mode=mode):
                scales = np.array([1, 1, scale], dtype=np.float32)
                output_size = int(np.floor(size * float(scales[2])))
                signal = np.random.default_rng(size).normal(0, .2, (1, 2, size)).astype(np.float32)
                self.compare(helper.make_node("Resize", ["x", "", "scales"], ["y"], mode=mode,
                    coordinate_transformation_mode=coordinates, nearest_mode="floor"),
                    {"scales": scales}, [1, 2, size], [1, 2, output_size], signal)

    def test_phase_resize_preserves_trilinear_accumulation(self):
        # Kokoro interpolates an accumulated oscillator phase. One ULP here
        # changes subsequent Sin/STFT phase features, so preserve all eight
        # source trilinear terms even along the unchanged singleton axis.
        size, channels = 132, 9
        scales = np.array([1, 1, 300], dtype=np.float32)
        signal = np.random.default_rng(47).uniform(0, 80000, (1, channels, size)).astype(np.float32)
        self.compare(helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="linear",
            coordinate_transformation_mode="half_pixel"), {"scales": scales},
            [1, channels, size], [1, channels, size * 300], signal, atol=0, rtol=0)

    def test_depthwise_transposed_convolution(self):
        rng = np.random.default_rng(311)
        for channels, length in ((3, 7), (7, 11)):
            signal = rng.normal(0, .2, (1, channels, length)).astype(np.float32)
            weights = rng.normal(0, .2, (channels, 1, 3)).astype(np.float32)
            bias = rng.normal(0, .2, channels).astype(np.float32)
            self.compare(helper.make_node("ConvTranspose", ["x", "w", "b"], ["y"],
                group=channels, strides=[2], pads=[1, 1], output_padding=[1]),
                {"w": weights, "b": bias}, [1, channels, length], [1, channels, length * 2], signal)

    def test_reflect_padding_is_live_gather(self):
        signal = np.arange(26, dtype=np.float32).reshape(1, 2, 13)
        for before, after in [(10, 10), (1, 0)]:
            self.compare(helper.make_node("Pad", ["x", "pads"], ["y"], mode="reflect"),
                {"pads": np.array([0, 0, before, 0, 0, after], dtype=np.int64)},
                [1, 2, 13], [1, 2, 13 + before + after], signal, atol=0)


if __name__ == "__main__":
    unittest.main()
