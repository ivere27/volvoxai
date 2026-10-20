"""Importer refusals and native numerical comparisons to independent HF models."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import hashlib
import subprocess
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from tools.exporter.errors import ExporterError
from tools.llm_baseline.generate_modern_incremental_stress import tiny_dense_config as tiny_config
from tools.exporter.hf_decoder import DecoderConfig, build_decoder_graph, read_hf_checkpoint
from tools.exporter.runtime_ir import import_runtime_package
from tools.exporter.capabilities import validate_runtime_descriptors
from tools.exporter.native_execution import NativeGraphRunner

try:
    import torch
    from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config,
                              Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM)
    from safetensors.torch import save_file as save_torch_file
    HAVE_HF = True
except ImportError:
    HAVE_HF = False


def random_weights(config):
    generator = np.random.default_rng(1107)
    return {name: (generator.standard_normal(shape) * 0.05).astype(np.float32)
            for name, shape in config.weight_shapes().items()}


def scaled_config(**changes):
    result = tiny_config(hidden_size=64, intermediate_size=128, head_dim=16,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=131072,
        rope_theta=500000.0, rope_scaling={"rope_type": "llama3", "factor": 32.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192})
    result.update(changes)
    return result


class CheckpointStorageTests(unittest.TestCase):
    """Raw SafeTensors fixtures avoid torch's own decoding as the only oracle."""

    def test_metadata_inventory_preserves_original_storage_without_payload_reads(self):
        from safetensors.numpy import safe_open
        from tools.exporter.hf_graph import read_checkpoint_metadata

        class MetadataOnlyHandle:
            def __init__(self, *args, **kwargs):
                self.reader = safe_open(*args, **kwargs)
            def __enter__(self):
                self.reader.__enter__()
                return self
            def __exit__(self, *args):
                return self.reader.__exit__(*args)
            def keys(self):
                return self.reader.keys()
            def get_slice(self, name):
                return self.reader.get_slice(name)
            def get_tensor(self, name):
                raise AssertionError("metadata inventory must not read a tensor payload")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.safetensors"
            self.write_shard(path, {"weight": ("BF16", np.array([[0x3f80, 0xc000]], np.uint16))})
            with patch("tools.exporter.hf_graph.safe_open", MetadataOnlyHandle), \
                    patch("tools.exporter.hf_graph._mapped_tensor") as mapped:
                config, tensors = read_checkpoint_metadata(root, source_config={})
            mapped.assert_not_called()
            self.assertEqual(config, {})
            self.assertEqual(tensors["weight"][:3], (path, "BF16", (1, 2)))
            self.assertEqual(path.read_bytes()[tensors["weight"][3]:], bytes.fromhex("803f00c0"))

    @staticmethod
    def write_shard(path, tensors):
        header, payload = {}, bytearray()
        for name, (dtype, array) in tensors.items():
            start = len(payload)
            payload.extend(array.tobytes())
            header[name] = {"dtype": dtype, "shape": list(array.shape),
                            "data_offsets": [start, len(payload)]}
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)

    @staticmethod
    def config():
        return {"quantization_config": {"quant_method": "fp8", "scale_fmt": "ue8m0",
            "weight_block_size": [32, 32], "expert_dtype": "fp4"}}

    def read(self, root, tensors, config=None):
        from tools.exporter.hf_graph import read_checkpoint_storage
        self.write_shard(root / "model.safetensors", tensors)
        return read_checkpoint_storage(root, source_config=self.config() if config is None else config)

    def test_fp8_all_finite_codes_and_cross_shard_block_scales(self):
        from tools.exporter.hf_graph import read_checkpoint_storage
        # Every finite E4M3FN code, non-square dimensions and both block tails.
        codes = np.array([i for i in range(256) if i not in (127, 255)], np.uint8)
        raw = np.resize(codes, (35, 67))
        scales = np.array([[125, 127, 130], [128, 126, 129]], np.uint8)
        weight, scale = "model.layers.0.self_attn.wq_a.weight", "model.layers.0.self_attn.wq_a.weight_scale_inv"
        def scalar(code):
            sign = -1. if code & 128 else 1.
            exponent, fraction = (int(code) >> 3) & 15, int(code) & 7
            return sign * (math.ldexp(fraction, -9) if exponent == 0 else
                           math.ldexp(8 + fraction, exponent - 10))
        expected = np.array([[math.ldexp(scalar(raw[i, j]), int(scales[i // 32, j // 32]) - 127)
                              for j in range(67)] for i in range(35)], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_shard(root / "a.safetensors", {weight: ("F8_E4M3", raw)})
            self.write_shard(root / "z.safetensors", {scale: ("F8_E8M0", scales)})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map":
                {weight: "a.safetensors", scale: "z.safetensors"}}))
            _, weights, dtypes = read_checkpoint_storage(root, source_config=self.config())
            self.assertEqual(set(weights), {weight})
            self.assertEqual(dtypes, {weight: "F8_E4M3", scale: "F8_E8M0"})
            np.testing.assert_array_equal(weights[weight].view(np.uint32), expected.view(np.uint32))

    def test_fp4_low_nibble_first_and_engram_per_row_scales(self):
        # Each sign/exponent/mantissa code occurs in both nibble positions.
        codes = np.arange(16, dtype=np.uint8)
        packed = np.tile(codes | ((15 - codes) << 4), (2, 2))
        scales = np.array([[126, 129], [128, 125]], np.uint8)
        weight, scale = "layers.0.ffn.experts.3.w2.weight", "layers.0.ffn.experts.3.w2.scale"
        def fp4(code):
            exponent, mantissa = (code >> 1) & 3, code & 1
            magnitude = .5 * mantissa if exponent == 0 else math.ldexp(1. + .5 * mantissa, exponent - 1)
            return -magnitude if code & 8 else magnitude
        expected = np.array([[math.ldexp(fp4(int(packed[i, j // 2]) >> (4 * (j % 2)) & 15),
                                         int(scales[i, j // 32]) - 127)
                              for j in range(64)] for i in range(2)], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dtype in ("I8", "U8"):
                _, arrays, _ = self.read(root, {weight: (dtype, packed), scale: ("F8_E8M0", scales)})
                np.testing.assert_array_equal(arrays[weight].view(np.uint32), expected.view(np.uint32))
            name = "layers.1.engram.embed.weight"
            _, arrays, _ = self.read(root, {name: ("F8_E4M3", np.full((3, 64), 56, np.uint8)),
                "layers.1.engram.embed.scale": ("F8_E8M0", np.array([[127, 128], [126, 129], [125, 130]], np.uint8))})
            np.testing.assert_array_equal(arrays[name], np.repeat(np.array([[1, 2], [.5, 4], [.25, 8]], np.float32), 32, axis=1))

    def test_declared_128_block_geometry_and_fp4_incomplete_block_refusal(self):
        w, s = "layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale"
        with tempfile.TemporaryDirectory() as directory:
            config = self.config()
            config["quantization_config"]["weight_block_size"] = [128, 128]
            _, arrays, _ = self.read(Path(directory), {
                w: ("F8_E4M3", np.full((129, 129), 56, np.uint8)),
                s: ("F8_E8M0", np.array([[126, 127], [128, 129]], np.uint8))}, config)
            expected = np.repeat(np.repeat(np.array([[.5, 1], [2, 4]], np.float32), 128, 0), 128, 1)[:129, :129]
            np.testing.assert_array_equal(arrays[w], expected)
            with self.assertRaisesRegex(ExporterError, "complete 32-value blocks"):
                self.read(Path(directory), {
                    "layers.0.ffn.experts.0.w1.weight": ("I8", np.ones((2, 15), np.uint8)),
                    "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", np.full((2, 1), 127, np.uint8))})

    def test_quantized_storage_refuses_ambiguous_unsupported_and_nonfinite_inputs(self):
        w, s = "layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale"
        valid = {w: ("F8_E4M3", np.full((32, 32), 56, np.uint8)),
                 s: ("F8_E8M0", np.full((1, 1), 127, np.uint8))}
        cases = [({w: valid[w]}, None),
            ({**valid, w.replace("weight", "weight_scale_inv"): valid[s]}, None),
            ({**valid, "unused.scale": valid[s]}, None),
            ({**valid, s: ("F8_E8M0", np.ones((32, 1), np.uint8))}, None),
            ({**valid, s: ("U8", np.full((1, 1), 127, np.uint8))}, None),
            ({**valid, w: ("I8", valid[w][1])}, None),
            ({**valid, s: ("F8_E8M0", np.full((1, 1), 255, np.uint8))}, None),
            ({**valid, w: ("F8_E4M3", np.full((32, 32), 127, np.uint8))}, None),
            ({**valid, w: ("F8_E4M3", np.full((32, 32), 126, np.uint8)),
              s: ("F8_E8M0", np.full((1, 1), 254, np.uint8))}, None),
            (valid, {}),
            (valid, {"quantization_config": {**self.config()["quantization_config"], "scale_fmt": "e4m3"}})]
        with tempfile.TemporaryDirectory() as directory:
            for tensors, config in cases:
                with self.subTest(names=list(tensors), config=config), self.assertRaises(ExporterError):
                    self.read(Path(directory), tensors, config)
            # UE8M0 byte zero is 2^-127, not zero and not the BF16 bias.
            _, arrays, _ = self.read(Path(directory), {**valid,
                s: ("F8_E8M0", np.zeros((1, 1), np.uint8))})
            self.assertTrue(np.all(arrays[w] == np.float32(math.ldexp(1., -127))))

    def test_quantized_storage_preserves_ordinary_tensors_and_early_validation(self):
        from tools.exporter.hf_graph import read_checkpoint_storage
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = {"dense.weight": ("F32", np.array([[1., -2.]], np.float32)),
                "half.weight": ("F16", np.array([.5, -4.], np.float16)),
                "norm.weight": ("BF16", np.array([0x3f80, 0xc000], np.uint16)),
                "layers.0.attn.wq_a.weight": ("F8_E4M3", np.full((32, 32), 56, np.uint8)),
                "layers.0.attn.wq_a.scale": ("F8_E8M0", np.full((1, 1), 127, np.uint8))}
            _, arrays, _ = self.read(root, tensors)
            np.testing.assert_array_equal(arrays["dense.weight"], [[1., -2.]])
            np.testing.assert_array_equal(arrays["half.weight"], [.5, -4.])
            np.testing.assert_array_equal(arrays["norm.weight"], [1., -2.])
            with patch("tools.exporter.hf_graph.safe_open") as opened:
                def refuse(config):
                    raise ValueError("architecture refused before payload IO")
                with self.assertRaisesRegex(ValueError, "architecture refused"):
                    read_checkpoint_storage(root, source_config=self.config(), config_validator=refuse)
                opened.assert_not_called()


class DecoderImporterTests(unittest.TestCase):
    def test_structure_builder_reuses_original_weights_without_reading_payloads(self):
        from tools.exporter.hf_decoder import build_decoder_structure
        for raw in (tiny_config(), scaled_config()):
            config = DecoderConfig.from_hf(raw, max_context=16)
            with patch("tools.exporter.hf_decoder.read_hf_checkpoint") as reader, \
                    patch("tools.exporter.hf_decoder._checked_weights") as checked:
                graph, derived = build_decoder_structure(config, compact_kv=True, cached=True)
                reader.assert_not_called()
                checked.assert_not_called()
            self.assertEqual(set(derived), {"hf.llama3.cos", "hf.llama3.sin"} if config.table_rope else set())
            source = random_weights(config)
            exported, weights = build_decoder_graph(config, source, compact_kv=True, cached=True)
            self.assertEqual(graph, exported)
            self.assertEqual(set(weights), set(source) | set(derived))
            for name, value in derived.items():
                np.testing.assert_array_equal(value, weights[name])

    def test_immutable_integer_metadata_preserves_bits_and_rejects_other_weights(self):
        from tools.exporter.hf_graph import read_checkpoint_storage
        names = ["model.layers.0.mlp.gate.tid2eid", "layers.0.ffn.gate.tid2eid",
                 "mtp.1.ffn.gate.tid2eid"] + [
            prefix + suffix for prefix in ("model.layers.2.ple.ple_embedding.",
                "model.language_model.layers.2.ple.ple_embedding.")
            for suffix in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")]
        values = np.array([-(1 << 62), (1 << 62) + 31], np.int64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(tiny_config()))
            save_file({name: values for name in names}, root / "model.safetensors")
            _, arrays, dtypes = read_checkpoint_storage(root)
            for name in names:
                self.assertEqual(dtypes[name], "I64")
                self.assertEqual(arrays[name].dtype, np.int64)
                np.testing.assert_array_equal(arrays[name], values)
            for name in ("model.layers.x.mlp.gate.tid2eid",
                         "model.layers.0.mlp.gate.tid2eid.extra",
                         "model.layers.0.self_attn.q_proj.weight",
                         "layers.x.ffn.gate.tid2eid", "mtp.1.ffn.gate.tid2eid.extra",
                         "layers.0.ffn.gate.weight",
                         "model.layers.0.ple.ple_embedding.other"):
                save_file({name: values}, root / "model.safetensors")
                with self.subTest(name=name), self.assertRaises(ExporterError):
                    read_checkpoint_storage(root)

    def test_llama3_configuration_and_early_export_refusals(self):
        from tools.exporter.hf_decoder import export_hf_decoder
        scaling = scaled_config()["rope_scaling"]
        broken = [{"factor": False}, {"factor": 0.5}, {"factor": float("nan")},
            {"low_freq_factor": 0}, {"high_freq_factor": 1},
            {"high_freq_factor": 1.00000001}, {"original_max_position_embeddings": True},
            {"original_max_position_embeddings": 131073}, {"attention_factor": 2},
            {"type": "default"}, {"rope_type": "yarn"}]
        for changes in broken:
            with self.subTest(changes=changes), self.assertRaises(ExporterError):
                DecoderConfig.from_hf(scaled_config(rope_scaling={**scaling, **changes}), max_context=128)
        with self.assertRaises(ExporterError):
            DecoderConfig.from_hf(scaled_config(model_type="qwen3"), max_context=128)
        with self.assertRaises(ExporterError):
            DecoderConfig.from_hf(scaled_config(rope_parameters=scaling), max_context=128)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(scaled_config()))
            for options in ({"compact_kv": True}, {"max_context": 128}):
                with patch("tools.exporter.hf_decoder.read_hf_checkpoint") as read, \
                        self.assertRaises(ExporterError):
                    export_hf_decoder(root, root / "out.safetensors", **options)
                read.assert_not_called()

    @unittest.skipUnless(HAVE_HF, "independent HF oracle requires torch and transformers")
    def test_llama3_native_logits_and_kv_at_absolute_positions(self):
        torch.manual_seed(1717)
        torch.set_num_threads(1)
        raw = scaled_config()
        hf = LlamaConfig(**raw)
        hf._attn_implementation = "eager"
        model = LlamaForCausalLM(hf).float().eval()
        config = DecoderConfig.from_hf(raw, max_context=128)
        graph, weights = build_decoder_graph(config,
            {name: value.detach().numpy() for name, value in model.state_dict().items()},
            compact_kv=True, capture_kv=True)
        self.assertFalse(any(node["opType"] in {"Gather", "Transpose", "RoPE"} for node in graph["nodes"]))
        self.assertEqual(sum(node["opType"] == "RotaryEmbedding" for node in graph["nodes"]), 2 * config.layers)
        self.assertEqual(sum(node["opType"] == "Linear" for node in graph["nodes"]), 7 * config.layers + 1)
        self.assertTrue(validate_runtime_descriptors(graph, weights=weights).supported)
        from transformers.modeling_rope_utils import _compute_llama3_parameters
        inverse, _ = _compute_llama3_parameters(hf)
        wavelength = 2 * torch.pi / (1 / (500000 ** (torch.arange(0, 16, 2).float() / 16)))
        self.assertTrue(torch.any(wavelength < 2048))
        self.assertTrue(torch.any((wavelength >= 2048) & (wavelength <= 8192)))
        self.assertTrue(torch.any(wavelength > 8192))
        phase = torch.arange(128).float()[:, None] * inverse[None]
        phase = torch.cat((phase, phase), -1)
        np.testing.assert_allclose(weights["hf.llama3.cos"], phase.cos().numpy(), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(weights["hf.llama3.sin"], phase.sin().numpy(), atol=1e-7, rtol=1e-7)
        ir = import_runtime_package(graph, weights)
        root = Path(__file__).resolve().parents[2]
        for positions in ([127], [0, 31, 127], list(range(5, 12))):
            length = len(positions)
            inputs = {"token_ids": ((np.arange(length)[None] + 2) % 32).astype(np.int32),
                      "position_ids": np.array(positions, np.int32),
                      "attention_mask": np.ones((1, length), np.int32)}
            with torch.inference_mode():
                expected = model(input_ids=torch.tensor(inputs["token_ids"]).long(),
                    position_ids=torch.tensor(inputs["position_ids"][None]).long(),
                    attention_mask=torch.tensor(inputs["attention_mask"]).long(), use_cache=True)
            for profile in ("volvoxai", "volvoxai-lite"):
                with self.subTest(positions=positions, profile=profile), patch.dict(os.environ,
                        {"VOLVOXAI_NATIVE_RUNNER": str(root / "native" / profile)}):
                    runner = NativeGraphRunner(ir.bind_shape_profile({"S": length}), weights)
                    actual = runner.run(runner.bind(inputs)).outputs
                np.testing.assert_allclose(actual["logits"], expected.logits.numpy(), atol=1e-5, rtol=1e-5)
                np.testing.assert_array_equal(actual["logits"].argmax(-1), expected.logits.numpy().argmax(-1))
                for layer, cached in enumerate(expected.past_key_values.layers):
                    for suffix, tensor in (("k_rope", cached.keys), ("v", cached.values)):
                        value = actual[f"layer.{layer}.{suffix}"].reshape(1, length, 2, 16).transpose(0, 2, 1, 3)
                        np.testing.assert_allclose(value, tensor.numpy(), atol=1e-5, rtol=1e-5)
        for profile in ("volvoxai", "volvoxai-lite"):
            with patch.dict(os.environ, {"VOLVOXAI_NATIVE_RUNNER": str(root / "native" / profile)}):
                runner = NativeGraphRunner(ir.bind_shape_profile({"S": 1}), weights)
                for position in (-1, 128):
                    invalid = {"token_ids": np.array([[2]], np.int32),
                        "position_ids": np.array([position], np.int32), "attention_mask": np.ones((1, 1), np.int32)}
                    with self.subTest(profile=profile, invalid_position=position), self.assertRaises(ExporterError):
                        runner.run(runner.bind(invalid))

    @unittest.skipUnless(HAVE_HF, "independent HF oracle requires torch and transformers")
    def test_llama3_frequency_endpoints_match_hf_f32_tables(self):
        from tools.exporter.hf_decoder import _llama3_initializers
        from transformers.modeling_rope_utils import _compute_llama3_parameters
        raw = scaled_config()
        unscaled = 1 / (500000 ** (torch.arange(0, 16, 2).float() / 16))
        wavelength = 2 * torch.pi / unscaled
        # Put two frequencies exactly on the F32 high/low boundaries so the
        # independent oracle exercises both inclusive interpolation endpoints.
        low = 8192 / float(wavelength[4])
        high = 8192 / float(wavelength[3])
        self.assertEqual(np.float32(8192 / low), np.float32(wavelength[4]))
        self.assertEqual(np.float32(8192 / high), np.float32(wavelength[3]))
        for factor in (1.0, 3.75, 32.0):
            with self.subTest(factor=factor):
                raw["rope_scaling"] = {**raw["rope_scaling"], "factor": factor,
                    "low_freq_factor": low, "high_freq_factor": high}
                config = DecoderConfig.from_hf(raw, max_context=128)
                weights = _llama3_initializers(config)
                inverse, attention_factor = _compute_llama3_parameters(LlamaConfig(**raw))
                self.assertEqual(attention_factor, 1.0)
                phase = torch.arange(128).float()[:, None] * inverse[None]
                phase = torch.cat((phase, phase), -1)
                for function in ("cos", "sin"):
                    actual = weights[f"hf.llama3.{function}"]
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_allclose(actual, getattr(phase, function)().numpy(), atol=1e-7, rtol=1e-7)

    def test_scaled_tables_storage_is_independent_of_heads_and_layers(self):
        from tools.exporter.hf_decoder import _llama3_initializers, _qwen2_initializers
        raw = scaled_config(hidden_size=4096, head_dim=128, num_attention_heads=32,
                            num_key_value_heads=8, num_hidden_layers=32)
        config = DecoderConfig.from_hf(raw, max_context=8192)
        tables = _llama3_initializers(config)
        self.assertEqual(set(tables), {"hf.llama3.cos", "hf.llama3.sin"})
        self.assertEqual(sum(value.nbytes for value in tables.values()), 8 * 1024 * 1024)
        for value in tables.values():
            self.assertEqual(value.shape, (8192, 128))
        alternate = DecoderConfig.from_hf({**raw, "num_attention_heads": 16,
            "num_key_value_heads": 16, "num_hidden_layers": 2}, max_context=8192)
        for name, value in _llama3_initializers(alternate).items():
            np.testing.assert_array_equal(value, tables[name])
        for family, scaling, initializer in (("llama", raw["rope_scaling"], _llama3_initializers),
                ("qwen2", {"rope_type": "linear", "factor": 2.0}, _qwen2_initializers)):
            excessive = DecoderConfig.from_hf({**raw, "model_type": family,
                "max_position_embeddings": 1 << 21, "rope_scaling": scaling})
            with self.subTest(family=family), self.assertRaisesRegex(ExporterError, "1 GiB"):
                initializer(excessive)

    @unittest.skipUnless(HAVE_HF, "independent HF oracle requires torch and transformers")
    def test_llama3_cached_transitions_and_generate_match_hf_on_both_profiles(self):
        self._assert_cached_hf(scaled_config(), LlamaConfig, LlamaForCausalLM, maximum=128, prefix=124)

    @unittest.skipUnless(HAVE_HF, "independent HF oracle requires torch and transformers")
    def test_qwen2_yarn_cached_transitions_and_generate_match_hf_on_both_profiles(self):
        raw = tiny_config(model_type="qwen2", hidden_size=64, intermediate_size=128,
            head_dim=16, rope_scaling={"rope_type": "yarn", "factor": 2.0,
                "original_max_position_embeddings": 8})
        self._assert_cached_hf(raw, Qwen2Config, Qwen2ForCausalLM, maximum=16, prefix=12)

    def _assert_cached_hf(self, raw, config_class, model_class, *, maximum, prefix):
        from tools.llm_reference import capture_sequence, sha256_file
        torch.manual_seed(1717)
        torch.set_num_threads(1)
        hf = config_class(**raw)
        hf._attn_implementation = "eager"
        model = model_class(hf).float().eval()
        config = DecoderConfig.from_hf(raw, max_context=maximum)
        source = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        from tools.llm_reference import read_policy
        policy = read_policy(model="llama3.2-1b-gsm8k")
        policy["models"] = {"scaled-fixture": {"repo_id": "synthetic-independent-HF-fixture", "revision": "seed1717",
            "config": raw, "head_dim": config.head_dim}}
        policy["reference"]["qualification"]["prompts"] = ["Synthetic token-ID fixture; no text tokenizer is used."]
        policy["reference"]["qualification"]["prefix_lengths"] = [prefix]
        policy["reference"]["qualification"]["max_context"] = maximum
        with tempfile.TemporaryDirectory(prefix="volvox-scaled-cached-") as directory:
            root = Path(directory)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy))
            for package_name, options in (("execution", {"cached": True}), ("capture", {"capture_kv": True})):
                package = root / package_name
                package.mkdir()
                graph, weights = build_decoder_graph(config, source, compact_kv=True, **options)
                (package / "graph.json").write_text(json.dumps(graph))
                if package_name == "execution": save_file(weights, root / "model.safetensors")
                (package / "model.safetensors").symlink_to(root / "model.safetensors")
            reference = root / "reference"
            reference.mkdir()
            tensors, observation = capture_sequence(model, (torch.arange(prefix)[None] % 30) + 1,
                decode_steps=4, tolerance=policy["reference"]["cached_full_parity"])
            tensor_path = reference / "fixture.safetensors"
            save_torch_file(tensors, tensor_path)
            case = {"id": f"prompt0-prefix{prefix:03d}", "prompt_index": 0, "prefix_length": prefix,
                "prompt": policy["reference"]["qualification"]["prompts"][0],
                "reference_path": tensor_path.name, "reference_bytes": tensor_path.stat().st_size,
                "reference_sha256": sha256_file(tensor_path), "engine_input_ids": {
                    "sha256": hashlib.sha256(tensors["prefill.input_ids"].numpy().astype(np.int32).tobytes()).hexdigest()},
                **observation}
            manifest = {"status": "complete", "model": "scaled-fixture", "repo_id": "synthetic-independent-HF-fixture",
                "revision": "seed1717", "config": raw, "execution_dtype": "float32", "reference_policy": policy["reference"],
                "provenance": {"policy_sha256": sha256_file(policy_path)}, "cases": [case]}
            (reference / "manifest.json").write_text(json.dumps(manifest))
            for profile in ("full", "inference"):
                output = root / (profile + ".json")
                command = [sys.executable, str(Path(__file__).resolve().parents[2] / "tools/llm_cached_compare.py"),
                    "--policy", str(policy_path), "--package", str(root / "execution"), "--capture-package", str(root / "capture"),
                    "--reference", str(reference), "--profile", profile, "--threads", "1", "--out", str(output)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], "PASS")
                self.assertTrue(report["generation_qualified"])
                self.assertEqual(len(report["cases"][0]["steps"]), 5)

    def test_family_dispatch_and_shared_reader_validate_before_storage(self):
        from tools.exporter.hf_decoder import export_hf_decoder
        for family, class_name, function_name in (("qwen3_moe", "ExtendedDecoderConfig", "export_hf_extended"),
                ("deepseek_v3", "DeepSeekDecoderConfig", "export_hf_deepseek")):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                raw = {"model_type": family}
                (path / "config.json").write_text(json.dumps(raw))
                with patch("tools.exporter.hf_decoder.importlib.import_module") as load, \
                        patch("tools.exporter.hf_decoder.read_checkpoint_storage") as storage:
                    parser = unittest.mock.Mock()
                    exporter = unittest.mock.Mock(return_value={"outputs": ["chosen"]})
                    load.return_value = SimpleNamespace(**{class_name: SimpleNamespace(from_hf=parser),
                        function_name: exporter})
                    read_hf_checkpoint(path)
                    parser.assert_called_once_with(raw)
                    storage.assert_called_once_with(path, source_config=raw)
                    result = export_hf_decoder(path, path / "out", max_context=8,
                        output_names=["chosen"], compact_kv=True, cached=True)
                    self.assertEqual(result["outputs"], ["chosen"])
                    exporter.assert_called_once_with(path, path / "out", max_context=8,
                        output_names=["chosen"], capture_kv=False, compact_kv=True,
                        cached=True, report_callback=None)
                with patch("tools.exporter.hf_decoder.read_checkpoint_storage") as storage:
                    def refuse(_):
                        raise ExporterError("synthetic configuration refusal")
                    with self.assertRaises(ExporterError):
                        read_hf_checkpoint(path, config_validator=refuse)
                    storage.assert_not_called()

    def test_compact_decoder_preserves_head_axis_and_declares_typed_state(self):
        config = DecoderConfig.from_hf(tiny_config(model_type="qwen3", head_dim=8))
        graph, weights = build_decoder_graph(config, random_weights(config),
                                              compact_kv=True, cached=True)
        self.assertNotIn("hf.repeat_kv.indices", weights)
        self.assertFalse(any(node["opType"] in {"Gather", "Transpose"} for node in graph["nodes"]))
        self.assertEqual(graph["outputs"], ["logits", "generated"])
        self.assertEqual(graph["execution"]["incremental"]["position_input"], "position_ids")
        self.assertEqual(graph["execution"]["generation"]["id_domain_size"], config.vocab_size)
        self.assertEqual(graph["execution"]["generation"]["token_output"], "generated")
        ir = import_runtime_package(graph, weights)
        for length in (1, 3, 16):
            bound = ir.bind_shape_profile({"S": length})
            self.assertEqual(bound.tensors["layer.0.k_rope.norm"].shape, (1, length, 2, 8))
            self.assertEqual(bound.tensors["layer.0.k_rope"].shape, (1, length, 16))
            self.assertEqual(bound.tensors["generated"].shape, (1, length))
        self.assertTrue(validate_runtime_descriptors(graph, weights=weights).supported)
        for options in ({"cached": True}, {"cached": True, "compact_kv": True, "capture_kv": True}):
            with self.assertRaisesRegex(ExporterError, "private cache tensors"):
                build_decoder_graph(config, weights, **options)

    def test_dynamic_shape_proof_head_width_and_tied_storage(self):
        config = DecoderConfig.from_hf(tiny_config(model_type="qwen3", head_dim=8))
        graph, weights = build_decoder_graph(config, random_weights(config), capture_kv=True)
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(config.query_heads * config.head_dim, 32)
        self.assertNotIn("lm_head.weight", weights)
        self.assertEqual(weights["hf.repeat_kv.indices"].tolist(),
                         list(range(8)) * 2 + list(range(8, 16)) * 2)
        self.assertEqual(graph["outputs"], ["logits", "layer.0.k_rope", "layer.0.v", "layer.1.k_rope", "layer.1.v"])
        self.assertEqual(graph["execution"]["entrypoints"][0]["inputs"], list(graph["inputs"]))
        ir = import_runtime_package(graph, weights)
        for length in (1, 3, 16):
            bound = ir.bind_shape_profile({"S": length})
            self.assertEqual(bound.tensors["logits"].shape, (1, length, 32))
        self.assertTrue(validate_runtime_descriptors(graph, weights=weights).supported)

    def test_config_refusals(self):
        changes = [{"model_type": "not_a_decoder"}, {"model_type": []}, {"hidden_act": "gelu"},
                   {"num_attention_heads": 3}, {"head_dim": 3},
                   {"rope_scaling": {"type": "linear", "factor": 2}},
                   {"rope_parameters": {"rope_type": "yarn", "rope_theta": 10000}},
                   {"rope_parameters": []}, {"rope_scaling": False},
                   {"rope_scaling": {"rope_type": "default", "type": "linear"}},
                   {"sliding_window": 8}, {"use_sliding_window": True},
                   {"layer_types": ["full_attention", "sliding_attention"]},
                   {"quantization_config": {"bits": 4}}, {"rms_norm_eps": float("nan")},
                   {"rope_theta": float("inf")}, {"rope_theta": 1e-99},
                   {"num_hidden_layers": True}, {"partial_rotary_factor": 0.5}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ExporterError):
                DecoderConfig.from_hf(tiny_config(**change))
        with self.assertRaises(ExporterError):
            DecoderConfig.from_hf(tiny_config(), max_context=17)

    def test_weight_and_tied_head_refusals(self):
        config = DecoderConfig.from_hf(tiny_config())
        for error in ("missing", "shape", "nonfinite", "integer", "extra", "tied"):
            weights = random_weights(config)
            if error == "missing":
                del weights["model.norm.weight"]
            elif error == "shape":
                weights["model.norm.weight"] = np.ones(15, np.float32)
            elif error == "nonfinite":
                weights["model.norm.weight"][0] = np.nan
            elif error == "integer":
                weights["model.norm.weight"] = np.ones(16, np.int32)
            elif error == "extra":
                weights["unknown.adapter.weight"] = np.ones(2, np.float32)
            else:
                weights["lm_head.weight"] = weights["model.embed_tokens.weight"] + 1
            with self.subTest(error=error), self.assertRaises(ExporterError):
                build_decoder_graph(config, weights)

    def test_qwen_mlp_bias_true_is_refused(self):
        with self.assertRaisesRegex(ExporterError, "qwen3 MLP projections are bias-free"):
            DecoderConfig.from_hf(tiny_config(model_type="qwen3", head_dim=8, mlp_bias=True))

    def test_qwen2_biases_defaults_and_disabled_sliding_metadata(self):
        raw = tiny_config(model_type="qwen2", architectures=["Qwen2ForCausalLM"],
            use_sliding_window=False, sliding_window=32768, max_window_layers=28)
        config = DecoderConfig.from_hf(raw)
        self.assertEqual(config.head_dim, 4)
        self.assertTrue(config.attention_bias)
        self.assertFalse(config.output_attention_bias)
        shapes = config.weight_shapes()
        for layer in range(config.layers):
            prefix = f"model.layers.{layer}.self_attn."
            for name in ("q_proj", "k_proj", "v_proj"):
                self.assertIn(prefix + name + ".bias", shapes)
            for name in ("o_proj.bias", "q_norm.weight", "k_norm.weight"):
                self.assertNotIn(prefix + name, shapes)
        graph, weights = build_decoder_graph(config, random_weights(config), compact_kv=True)
        self.assertTrue(validate_runtime_descriptors(graph, weights=weights).supported)
        self.assertFalse(any(".q_rope.norm" in n["id"] for n in graph["nodes"]))
        for change in ({"use_sliding_window": True}, {"head_dim": None}, {"mlp_bias": True},
                       {"architectures": ["Qwen2ForSequenceClassification"]},
                       {"layer_types": ["full_attention", "sliding_attention"]}):
            with self.subTest(change=change), self.assertRaises(ExporterError):
                DecoderConfig.from_hf({**raw, **change})
        for omitted in (True, False):
            defaults = {**raw, "hidden_size": 256, "num_attention_heads": 64}
            defaults.pop("num_key_value_heads")
            if not omitted:
                defaults["num_key_value_heads"] = None
            parsed = DecoderConfig.from_hf(defaults)
            self.assertEqual(parsed.kv_heads, 32 if omitted else 64)
            if HAVE_HF:
                hf = Qwen2Config(**defaults)
                self.assertEqual(parsed.kv_heads, hf.num_key_value_heads)
                self.assertEqual(hf.sliding_window, None)

    def test_qwen2_scaled_rope_admission_and_weight_refusals(self):
        from tools.exporter.hf_decoder import export_hf_decoder
        raw = tiny_config(model_type="qwen2", rope_scaling={"type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 8})
        config = DecoderConfig.from_hf(raw, max_context=16)
        for change in ({"factor": 0.5}, {"factor": True}, {"beta_fast": 0},
                       {"beta_fast": 1, "beta_slow": 2}, {"truncate": "yes"},
                       {"attention_factor": float("nan")}, {"original_max_position_embeddings": 17},
                       {"mscale": 2}, {"rope_type": "dynamic"}, {"rope_type": []}):
            with self.subTest(change=change), self.assertRaises(ExporterError):
                DecoderConfig.from_hf({**raw, "rope_scaling": {**raw["rope_scaling"], **change}})
        for broken in ("missing_q_bias", "extra_o_bias", "extra_q_norm"):
            weights = random_weights(config)
            prefix = "model.layers.0.self_attn."
            if broken == "missing_q_bias":
                del weights[prefix + "q_proj.bias"]
            elif broken == "extra_o_bias":
                weights[prefix + "o_proj.bias"] = np.zeros(16, np.float32)
            else:
                weights[prefix + "q_norm.weight"] = np.ones(4, np.float32)
            with self.subTest(broken=broken), self.assertRaises(ExporterError):
                build_decoder_graph(config, weights, compact_kv=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(raw))
            for options in ({"compact_kv": True}, {"max_context": 16}):
                with patch("tools.exporter.hf_decoder.read_hf_checkpoint") as read, self.assertRaises(ExporterError):
                    export_hf_decoder(path, path / "out.safetensors", **options)
                read.assert_not_called()

    def test_qwen_omitted_head_dim_uses_family_default_and_null_is_refused(self):
        config = DecoderConfig.from_hf(tiny_config(model_type="qwen3"))
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.weight_shapes()["model.layers.0.self_attn.q_proj.weight"], (512, 16))
        graph, weights = build_decoder_graph(config, random_weights(config))
        self.assertTrue(validate_runtime_descriptors(graph, weights=weights).supported)
        with self.assertRaisesRegex(ExporterError, "qwen3 head_dim must not be null"):
            DecoderConfig.from_hf(tiny_config(model_type="qwen3", head_dim=None))
        if HAVE_HF:
            self.assertEqual(config.head_dim, Qwen3Config(hidden_size=16, num_attention_heads=4).head_dim)

    def test_family_defaults_match_hf_config(self):
        for family in ("llama", "qwen3"):
            for kv in ("omitted", None):
                raw = tiny_config(model_type=family, num_attention_heads=64)
                del raw["num_key_value_heads"]
                del raw["rms_norm_eps"]
                del raw["tie_word_embeddings"]
                if family == "llama":
                    # The adapter refuses omitted non-divisible dimensions;
                    # this fixture covers its accepted inferred Llama domain.
                    raw["hidden_size"] = 256
                if kv is None:
                    raw["num_key_value_heads"] = None
                config = DecoderConfig.from_hf(raw)
                self.assertEqual(config.kv_heads, 32 if family == "qwen3" and kv == "omitted" else 64)
                if HAVE_HF:
                    hf = (LlamaConfig if family == "llama" else Qwen3Config)(**raw)
                    self.assertEqual(config.head_dim, hf.head_dim)
                    self.assertEqual(config.kv_heads, hf.num_key_value_heads)
                    self.assertEqual(config.epsilon, hf.rms_norm_eps)
                    self.assertEqual(config.theta, hf.rope_parameters["rope_theta"])
                    self.assertEqual(config.attention_bias, hf.attention_bias)
                    self.assertFalse(config.mlp_bias)
                    self.assertEqual(config.tied_embeddings, hf.tie_word_embeddings)

    def test_rope_descriptor_acceptance_and_refusals(self):
        config = DecoderConfig.from_hf(tiny_config())
        graph, weights = build_decoder_graph(config, random_weights(config))
        for error in ("position_shape", "position_dtype", "odd_dim", "theta", "output_shape"):
            broken = copy.deepcopy(graph)
            rope = next(n for n in broken["nodes"] if n["opType"] == "RoPE")
            if error == "position_shape":
                broken["inputs"]["position_ids"]["shape"] = [1, "S"]
            elif error == "position_dtype":
                broken["inputs"]["position_ids"]["dtype"] = "float32"
            elif error == "odd_dim":
                rope["params"]["rotary_dim"] = 3
            elif error == "theta":
                rope["params"]["theta"] = 0
            else:
                rope["outputs"]["out"]["shape"][-1] = 6
            with self.subTest(error=error):
                self.assertFalse(validate_runtime_descriptors(broken, weights=weights).supported)

    def test_indexed_source_refusals(self):
        config = DecoderConfig.from_hf(tiny_config())
        weights = random_weights(config)
        with tempfile.TemporaryDirectory(prefix="volvox-hf-import-") as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(tiny_config()))
            save_file(weights, root / "one.safetensors")
            mapping = {name: "one.safetensors" for name in weights}
            index = root / "model.safetensors.index.json"
            for error in ("traversal", "missing", "wrong_shard", "duplicate_json"):
                broken = dict(mapping)
                if error == "traversal":
                    broken[next(iter(broken))] = "../one.safetensors"
                elif error == "missing":
                    broken["nonexistent.weight"] = "one.safetensors"
                elif error == "wrong_shard":
                    del broken[next(iter(broken))]
                index.write_text(json.dumps({"weight_map": broken}))
                if error == "duplicate_json":
                    index.write_text('{"weight_map": {}, "weight_map": {}}')
                with self.subTest(error=error), self.assertRaises(ExporterError):
                    read_hf_checkpoint(root)

    def test_single_f16_source_and_unsupported_export_options(self):
        from tools.export_safetensors import export_model
        config = DecoderConfig.from_hf(tiny_config())
        expected = {name: value.astype(np.float16) for name, value in random_weights(config).items()}
        with tempfile.TemporaryDirectory(prefix="volvox-hf-import-") as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(tiny_config()))
            save_file(expected, root / "model.safetensors")
            _, widened, dtypes = read_hf_checkpoint(root)
            self.assertEqual(set(dtypes.values()), {"F16"})
            for name in expected:
                np.testing.assert_array_equal(widened[name], expected[name].astype(np.float32))
            for options in ({"quant_mode": "require-w8a8"}, {"weight_dtype": "float16"},
                            {"specialize_inputs": {"token_ids": 1}}, {"dimension_bounds": {"S": (1, 8)}},
                            {"allow_silu_numerical_migration": True}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    export_model(str(root), str(root / "out.safetensors"), **options)

    @unittest.skipUnless(HAVE_HF, "independent HF oracle requires torch and transformers")
    def test_bf16_shards_widen_exactly_and_cli_publishes(self):
        from tools.export_safetensors import export_model
        from tools.exporter.cli import main
        config = DecoderConfig.from_hf(tiny_config())
        source = {name: torch.tensor(value).to(torch.bfloat16) for name, value in random_weights(config).items()}
        with tempfile.TemporaryDirectory(prefix="volvox-hf-import-") as directory:
            root, package = Path(directory) / "source", Path(directory) / "package"
            root.mkdir()
            (root / "config.json").write_text(json.dumps(tiny_config()))
            names, mapping = sorted(source), {}
            for i, shard_names in enumerate((names[::2], names[1::2])):
                filename = f"model-{i}.safetensors"
                save_torch_file({name: source[name] for name in shard_names}, root / filename)
                mapping.update({name: filename for name in shard_names})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
            _, widened, dtypes = read_hf_checkpoint(root)
            self.assertEqual(set(dtypes.values()), {"BF16"})
            for name in source:
                np.testing.assert_array_equal(widened[name], source[name].float().numpy())
            # Exact descriptor publication is exercised separately from target
            # registry qualification; native-cpu registration is required.
            arguments = ["--model", str(root), "--out", str(package / "model.safetensors"),
                         "--target", "native-cpu", "--max-context", "8", "--capture-kv"]
            self.assertEqual(main(export_model, arguments), 0)
            self.assertTrue((package / "graph.json").is_file())
            self.assertNotEqual(main(export_model, arguments), 0)

    @unittest.skipUnless(HAVE_HF, "independent HF oracle requires torch and transformers")
    def test_qwen2_native_logits_and_kv_match_hf_default_linear_and_yarn(self):
        from tools.exporter.hf_decoder import _qwen2_initializers
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        root = Path(__file__).resolve().parents[2]
        runners = [root / "native" / name for name in ("volvoxai", "volvoxai-lite")]
        self.assertTrue(all(path.is_file() for path in runners), "both native profiles are required")
        torch.set_num_threads(1)
        variants = (None, {"rope_type": "linear", "factor": 4.0},
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8},
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8,
             "attention_factor": 1.2, "beta_fast": 8.0, "beta_slow": 2.0, "truncate": False})
        for scaling in variants:
            torch.manual_seed(4321)
            raw = tiny_config(model_type="qwen2", hidden_size=64, intermediate_size=128,
                head_dim=16, max_position_embeddings=32, rope_theta=1000000.0,
                use_sliding_window=False, sliding_window=32768, rope_scaling=scaling)
            hf = Qwen2Config(**raw)
            hf._attn_implementation = "eager"
            model = Qwen2ForCausalLM(hf).float().eval()
            # Nonzero, distinct biases make silently dropping Q/K/V bias visible.
            with torch.no_grad():
                for layer in model.model.layers:
                    for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                        projection.bias.uniform_(-0.05, 0.05)
            config = DecoderConfig.from_hf(hf.to_dict(), max_context=32)
            source_weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
            if scaling:
                inverse, attention = ROPE_INIT_FUNCTIONS[scaling["rope_type"]](hf)
                phase = torch.arange(32).float()[:, None] * inverse[None]
                phase = torch.cat((phase, phase), -1)
                tables = _qwen2_initializers(config)
                np.testing.assert_allclose(tables["hf.qwen2.cos"], (phase.cos() * attention).numpy(), atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(tables["hf.qwen2.sin"], (phase.sin() * attention).numpy(), atol=2e-6, rtol=2e-6)
            for compact in ((False, True) if scaling is None else (True,)):
                graph, weights = build_decoder_graph(config, source_weights, capture_kv=True, compact_kv=compact)
                self.assertTrue(validate_runtime_descriptors(graph, weights=weights).supported)
                ir = import_runtime_package(graph, weights)
                for positions in ([31], [2, 17, 31], list(range(3, 10))):
                    length = len(positions)
                    inputs = {"token_ids": (np.arange(length)[None] + 2).astype(np.int32),
                        "position_ids": np.array(positions, np.int32),
                        "attention_mask": np.ones((1, length), np.int32)}
                    if length > 1:
                        inputs["attention_mask"][0, 1] = 0
                    with torch.inference_mode():
                        expected = model(input_ids=torch.tensor(inputs["token_ids"]).long(),
                            position_ids=torch.tensor(inputs["position_ids"][None]).long(),
                            attention_mask=torch.tensor(inputs["attention_mask"]).long(), use_cache=True)
                    for executable in runners:
                        with self.subTest(scaling=scaling, compact=compact, positions=positions, profile=executable.name), \
                                patch.dict(os.environ, {"VOLVOXAI_NATIVE_RUNNER": str(executable)}):
                            runner = NativeGraphRunner(ir.bind_shape_profile({"S": length}), weights)
                            actual = runner.run(runner.bind(inputs)).outputs
                            np.testing.assert_allclose(actual["logits"], expected.logits.numpy(), atol=1e-5, rtol=1e-5)
                            np.testing.assert_array_equal(actual["logits"].argmax(-1), expected.logits.numpy().argmax(-1))
                            for i in range(config.layers):
                                layer = expected.past_key_values.layers[i]
                                for name, tensor in ((f"layer.{i}.k_rope", layer.keys), (f"layer.{i}.v", layer.values)):
                                    compact_tensor = actual[name].reshape(1, length, config.kv_heads, config.head_dim).transpose(0, 2, 1, 3)
                                    np.testing.assert_allclose(compact_tensor, tensor.numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(HAVE_HF, "independent HF oracle requires torch and transformers")
    def test_native_f32_matches_hf_logits_and_compact_kv(self):
        runners = [Path(os.environ["VOLVOXAI_NATIVE_RUNNER"])] if os.environ.get("VOLVOXAI_NATIVE_RUNNER") else [
            Path(__file__).resolve().parents[2] / "native" / "volvoxai",
            Path(__file__).resolve().parents[2] / "native" / "volvoxai-lite"]
        self.assertTrue(all(path.is_file() for path in runners), "build both native profiles before numerical qualification")
        torch.set_num_threads(1)
        cases = [(LlamaConfig, LlamaForCausalLM, 16, 4, 2, 4, True),
                 (Qwen3Config, Qwen3ForCausalLM, 16, 4, 2, 8, True),
                 (LlamaConfig, LlamaForCausalLM, 256, 2, 2, 128, False)]
        for config_class, model_class, hidden, heads, kv_heads, dim, tied in cases:
            torch.manual_seed(1717)
            hf = config_class(hidden_size=hidden, intermediate_size=hidden * 2,
                              num_hidden_layers=2, num_attention_heads=heads,
                              num_key_value_heads=kv_heads, head_dim=dim, vocab_size=32,
                              max_position_embeddings=16, tie_word_embeddings=tied)
            hf._attn_implementation = "eager"
            model = model_class(hf).eval()
            config = DecoderConfig.from_hf(hf.to_dict())
            graph, weights = build_decoder_graph(config, {k: v.detach().numpy() for k, v in model.state_dict().items()}, capture_kv=True)
            ir = import_runtime_package(graph, weights)
            for length in (1, 3, 7):
                inputs = {"token_ids": (np.arange(length)[None] + 2).astype(np.int32),
                          "position_ids": (np.arange(length) + 5).astype(np.int32),
                          "attention_mask": np.ones((1, length), np.int32)}
                if length > 1:
                    inputs["attention_mask"][0, 1] = 0
                with torch.no_grad():
                    reference = model(input_ids=torch.tensor(inputs["token_ids"]).long(),
                                      position_ids=torch.tensor(inputs["position_ids"][None]).long(),
                                      attention_mask=torch.tensor(inputs["attention_mask"]).long(), use_cache=True)
                for executable in runners:
                    with self.subTest(family=config.family, dim=dim, length=length, profile=executable.name), patch.dict(
                            os.environ, {"VOLVOXAI_NATIVE_RUNNER": str(executable)}):
                        runner = NativeGraphRunner(ir.bind_shape_profile({"S": length}), weights)
                        actual = runner.run(runner.bind(inputs)).outputs
                        expected = reference.logits.numpy()
                        np.testing.assert_allclose(actual["logits"], expected, atol=1e-5, rtol=1e-5)
                        np.testing.assert_array_equal(actual["logits"].argmax(-1), expected.argmax(-1))
                        for i in range(config.layers):
                            cache = reference.past_key_values
                            if hasattr(cache, "layers"):
                                k, v = cache.layers[i].keys, cache.layers[i].values
                            else:
                                k, v = cache[i]
                            for name, target in ((f"layer.{i}.k_rope", k), (f"layer.{i}.v", v)):
                                compact = actual[name].reshape(1, length, kv_heads, dim).transpose(0, 2, 1, 3)
                                np.testing.assert_allclose(compact, target.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
