"""Public weight storage values round-trip partial words and groups."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx

pb = vx.pb


def tensor(name, values, dtype=pb.DataType.DATA_TYPE_F32):
    values = np.ascontiguousarray(values)
    return pb.Tensor(name=name, shape=values.shape, dtype=dtype, inline=values.tobytes())


class QuantizedWeightTest(unittest.TestCase):
    def setUp(self):
        host = vx.open_library()
        self.addCleanup(host.close)
        self.client = vx.VxQuantizationServiceClient(host)

    def quantize(self, values, bits=4, algorithm=pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN,
                 group_size=128, **options):
        return self.client.quantize_weight(pb.QuantizeWeightRequest(
            weight=tensor("weight", values),
            quantization=pb.WeightQuantization(format=pb.WeightFormat(bits=bits, group_size=group_size),
                                               algorithm=algorithm), **options))

    def test_all_group_boundaries_preserved_scales_and_transpose(self):
        for group_size in (32, 64, 128, 256):
            for bits in (2, 3, 4, 8):
                for width in (1, 3, 7, group_size - 1, group_size, group_size + 1,
                              group_size * 2 + 3):
                    groups = (width + group_size - 1) // group_size
                    scales = np.arange(1, 2 * groups + 1, dtype=np.float32).reshape(2, groups) / 8
                    original = np.random.default_rng(width + bits).normal(0, 1, (2, width)).astype(np.float32)
                    offset = 1 << (bits - 1)
                    per_element = scales[:, np.arange(width) // group_size]
                    expected = np.clip(np.rint(original / per_element), -offset, offset - 1) * per_element
                    for transpose in (False, True):
                        with self.subTest(group_size=group_size, bits=bits, width=width, transpose=transpose):
                            packed = self.quantize(original.T if transpose else original, bits,
                                group_size=group_size, transpose_source=transpose,
                                preserve_scales=scales.flatten()).weight
                            self.assertEqual(packed.format.group_size, group_size)
                            self.assertEqual(list(packed.scales.shape), [2, groups])
                            output = self.client.dequantize_weight(pb.DequantizeWeightRequest(weight=packed)).tensor
                            np.testing.assert_array_equal(np.frombuffer(output.inline, np.float32).reshape(output.shape), expected)

    def test_group_metadata_is_checked_and_per_channel_remains_separate(self):
        for group_size in (1, 16, 48, 512):
            with self.subTest(group_size=group_size), self.assertRaises(vx.VolvoxAIError):
                self.quantize(np.ones((2, 129), np.float32), group_size=group_size)
        with self.assertRaises(vx.VolvoxAIError):
            self.quantize(np.ones((2, 129), np.float32), bits=4, group_size=0)
        packed = self.quantize(np.ones((2, 129), np.float32), bits=8, group_size=0).weight
        self.assertEqual(packed.codes.dtype, pb.DataType.DATA_TYPE_I8)
        self.assertEqual(list(packed.scales.shape), [2])
        packed = self.quantize(np.ones((2, 129), np.float32)).weight
        packed.format.group_size = 32
        with self.assertRaises(vx.VolvoxAIError):
            self.client.dequantize_weight(pb.DequantizeWeightRequest(weight=packed))

    def test_partial_word_and_group_roundtrip_preserves_logical_shape(self):
        for bits in (2, 3, 4, 8):
            for width in (129, 130):
                original = np.random.default_rng(width + bits).normal(0, .3, (3, width)).astype(np.float32)
                original[1] = 0
                for transpose in (False, True):
                    with self.subTest(bits=bits, width=width, transpose=transpose):
                        packed = self.quantize(original.T if transpose else original, bits,
                                               transpose_source=transpose).weight
                        self.assertEqual(list(packed.logical_shape), [3, width])
                        self.assertEqual(list(packed.codes.shape), [3, (width * bits + 31) // 32])
                        scales = np.frombuffer(packed.scales.inline, np.float32).reshape(3, 2)
                        output = self.client.dequantize_weight(pb.DequantizeWeightRequest(weight=packed)).tensor
                        decoded = np.frombuffer(output.inline, np.float32).reshape(output.shape)
                        self.assertEqual(decoded.shape, original.shape)
                        bound = scales[:, np.arange(width) // 128] * .501 + 1e-7
                        self.assertTrue(np.all(np.abs(decoded - original) <= bound))
                        restored = self.client.dequantize_weight(pb.DequantizeWeightRequest(
                            weight=packed, transpose_destination=True)).tensor
                        np.testing.assert_array_equal(np.frombuffer(restored.inline, np.float32).reshape(restored.shape), decoded.T)

    def test_mse_clip_does_not_increase_weight_error(self):
        original = np.random.default_rng(42).normal(0, .2, (2, 259)).astype(np.float32)
        original[0, 5] = 2
        for group_size in (32, 64, 128, 256):
            with self.subTest(group_size=group_size):
                errors = []
                for algorithm in (pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN, pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP):
                    packed = self.quantize(original, algorithm=algorithm, group_size=group_size).weight
                    restored = self.client.dequantize_weight(pb.DequantizeWeightRequest(weight=packed)).tensor
                    values = np.frombuffer(restored.inline, np.float32).reshape(original.shape)
                    errors.append(np.square(original.astype(np.float64) - values).sum())
                self.assertLessEqual(errors[1], errors[0] + 1e-10)

    def test_w3_dense_bitstream_matches_independent_bytes_and_refuses_dirty_tail(self):
        codes = np.tile(np.arange(-4, 4, dtype=np.int32), 5)[:35]
        values = codes.astype(np.float32).reshape(1, -1)
        packed = self.quantize(values, bits=3, group_size=32, preserve_scales=[1., 1.]).weight
        stream = sum(int(value + 4) << (3 * index) for index, value in enumerate(codes))
        expected = stream.to_bytes(((codes.size * 3 + 31) // 32) * 4, 'little')
        self.assertEqual(packed.codes.inline, expected)
        self.assertEqual(list(packed.codes.shape), [1, 4])
        output = self.client.dequantize_weight(pb.DequantizeWeightRequest(weight=packed)).tensor
        np.testing.assert_array_equal(np.frombuffer(output.inline, np.float32), values[0])
        packed.codes.inline = (stream | (1 << (codes.size * 3))).to_bytes(len(expected), 'little')
        with self.assertRaises(vx.VolvoxAIError):
            self.client.dequantize_weight(pb.DequantizeWeightRequest(weight=packed))

    def test_mse_clip_preserves_finite_f32_reconstruction_at_extreme_magnitudes(self):
        words = np.full((1, 128), 0x7f028f5b, dtype=np.uint32)
        words[0, 0] = 0xff7fffff
        original = words.view(np.float32)
        for bits in (4, 8):
            with self.subTest(bits=bits):
                errors = []
                for algorithm in (pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN, pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP):
                    packed = self.quantize(original, bits=bits, algorithm=algorithm).weight
                    restored = self.client.dequantize_weight(pb.DequantizeWeightRequest(weight=packed)).tensor
                    values = np.frombuffer(restored.inline, np.float32).reshape(original.shape)
                    self.assertTrue(np.isfinite(values).all(), pb.WeightAlgorithm.Name(algorithm))
                    errors.append(np.square(original.astype(np.float64) - values.astype(np.float64)).sum())
                self.assertLessEqual(errors[1], errors[0])

    def test_invalid_storage_and_fitting_are_refused(self):
        packed = self.quantize(np.ones((2, 129), np.float32)).weight
        malformed = []
        missing = pb.QuantizedWeight(); missing.CopyFrom(packed); del missing.logical_shape[:]
        malformed.append(missing)
        shape = pb.QuantizedWeight(); shape.CopyFrom(packed); shape.logical_shape[1] += 8
        malformed.append(shape)
        scale = pb.QuantizedWeight(); scale.CopyFrom(packed)
        scale.scales.inline = np.asarray([0, 1, 1, 1], np.float32).tobytes(); malformed.append(scale)
        tail = pb.QuantizedWeight(); tail.CopyFrom(packed)
        words = np.frombuffer(tail.codes.inline, np.uint32).copy().reshape(tail.codes.shape)
        words[0, -1] ^= np.uint32(1 << 4)
        tail.codes.inline = words.tobytes(); malformed.append(tail)
        for value in malformed:
            with self.subTest(value=value.logical_shape), self.assertRaises(vx.VolvoxAIError):
                self.client.dequantize_weight(pb.DequantizeWeightRequest(weight=value))
        for options in ({"algorithm": pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE},
                        {"preserve_scales": [1]},
                        {"algorithm": pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP, "preserve_scales": [1, 1, 1, 1]}):
            with self.subTest(options=options), self.assertRaises(vx.VolvoxAIError):
                self.quantize(np.ones((2, 129), np.float32), **options)


if __name__ == "__main__":
    unittest.main()
