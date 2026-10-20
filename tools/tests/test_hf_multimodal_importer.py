"""Complete two-graph image workflows against official multimodal HF logits."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from tools.whisper_audio import prepare_whisper_features
from tools.llm_baseline.generate_modern_incremental_stress import tiny_config, tiny_vision_config, execute_native as execute

from tools.export_safetensors import export_model
from tools.exporter.cli import main as exporter_main
from tools.exporter.errors import ExporterError
from tools.exporter.hf_extended import ExtendedDecoderConfig,build_extended_decoder_graph
from tools.exporter.hf_vision import (QwenVisionConfig,build_qwen_vision_graph,_fixed_grid,
    Llama4VisionConfig as NativeLlama4VisionConfig,build_llama4_vision_graph)
from tools.exporter.hf_whisper import (WhisperConfig, build_whisper_encoder_graph,
    build_whisper_decoder_graph)
from tools.exporter.runtime_ir import import_runtime_package
from tools.exporter.native_execution import NativeGraphRunner


try:
    import torch
    from transformers import (Qwen3_5Config,Qwen3_5ForConditionalGeneration,
        Llama4Config,Llama4ForConditionalGeneration,
        WhisperConfig as HFWhisperConfig, WhisperForConditionalGeneration,
        Qwen3_5VisionConfig, Llama4VisionConfig)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from transformers.models.llama4.modeling_llama4 import Llama4VisionModel
    from transformers.vision_utils import get_vision_position_ids, get_vision_bilinear_indices_and_weights
    HAVE_HF=True
except ImportError:HAVE_HF=False


@unittest.skipUnless(HAVE_HF,"independent oracle needs torch and transformers")
class MultimodalNativeReferenceTests(unittest.TestCase):
    def test_qwen3_5_image_features_and_three_axis_rotary_logits(self):
        torch.manual_seed(5315);torch.set_num_threads(1)
        raw=tiny_config("qwen3_5_text")
        raw["head_dim"]=16
        raw["rope_parameters"]["mrope_section"]=[1,1,2]
        hf=Qwen3_5Config(text_config=raw,vision_config=tiny_vision_config(),image_token_id=23,
            video_token_id=22,vision_start_token_id=21,vision_end_token_id=20)
        hf._attn_implementation="eager"
        model=Qwen3_5ForConditionalGeneration(hf).float().eval()
        source={name:value.detach().numpy() for name,value in model.state_dict().items()}
        grid=[[1,2,4]]
        vg,vw=build_qwen_vision_graph(QwenVisionConfig.from_hf(hf.vision_config.to_dict()),source,grid_thw=grid)
        config=ExtendedDecoderConfig.from_hf(hf.to_dict(),max_context=16)
        graph,weights=build_extended_decoder_graph(config,source,image_features_count=2)
        pixels=np.random.default_rng(8153).standard_normal((8,24)).astype(np.float32)
        axes=np.array([[0,1,1,3],[0,1,1,3],[0,1,2,3]],np.int32)
        inputs={"token_ids":np.array([[1,23,23,2]],np.int32),"position_ids":np.arange(4,dtype=np.int32),
            "attention_mask":np.ones((1,4),np.int32),"mrope_position_ids":axes,
            "image_feature_ids":np.array([0,0,1,0],np.int32),"image_token_mask":np.array([[0,1,1,0]],np.int32)}
        with torch.inference_mode():
            expected=model(input_ids=torch.from_numpy(inputs["token_ids"]).long(),
                attention_mask=torch.from_numpy(inputs["attention_mask"]).long(),position_ids=torch.from_numpy(axes[:,None]).long(),
                pixel_values=torch.from_numpy(pixels),image_grid_thw=torch.tensor(grid),use_cache=False).logits.numpy()
        for profile in ("volvoxai","volvoxai-lite"):
            with self.subTest(profile=profile):
                features=execute(profile,vg,vw,{"pixel_patches":pixels})["image_features"]
                actual=execute(profile,graph,weights,{**inputs,"image_features":features})["logits"]
                np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
                np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))

    def test_llama4_tiles_projector_and_image_conditioned_logits(self):
        torch.manual_seed(4511);torch.set_num_threads(1)
        vision=dict(hidden_size=16,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,
            num_channels=3,image_size=8,patch_size=2,projector_input_dim=16,projector_output_dim=16,
            vision_output_dim=16,pixel_shuffle_ratio=.5)
        hf=Llama4Config(text_config=tiny_config("llama4_text"),vision_config=vision,image_token_index=23)
        hf._attn_implementation="eager"
        model=Llama4ForConditionalGeneration(hf).float().eval()
        source={name:value.detach().numpy() for name,value in model.state_dict().items()}
        vg,vw=build_llama4_vision_graph(NativeLlama4VisionConfig.from_hf(hf.vision_config.to_dict()),source,tiles=1,text_hidden_size=16)
        config=ExtendedDecoderConfig.from_hf(hf.to_dict(),max_context=16)
        graph,weights=build_extended_decoder_graph(config,source,image_features_count=4)
        pixels=np.random.default_rng(1154).standard_normal((1,3,8,8)).astype(np.float32)
        positions=np.arange(6,dtype=np.int32)
        inputs={"token_ids":np.array([[1,23,23,23,23,2]],np.int32),"position_ids":positions,
            "sequence_ids":positions,"attention_mask":np.ones((1,6),np.int32),
            "chunk_attention_mask":((positions[:,None]//2)==(positions[None,:]//2))[None].astype(np.int32),
            "image_feature_ids":np.array([0,0,1,2,3,0],np.int32),"image_token_mask":np.array([[0,1,1,1,1,0]],np.int32)}
        with torch.inference_mode():
            expected=model(input_ids=torch.from_numpy(inputs["token_ids"]).long(),
                attention_mask=torch.from_numpy(inputs["attention_mask"]).long(),position_ids=torch.from_numpy(positions[None]).long(),
                pixel_values=torch.from_numpy(pixels),use_cache=False).logits.numpy()
        for profile in ("volvoxai","volvoxai-lite"):
            with self.subTest(profile=profile):
                features=execute(profile,vg,vw,{"pixel_values":pixels})["image_features"]
                actual=execute(profile,graph,weights,{**inputs,"image_features":features})["logits"]
                np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
                np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))


def tiny_whisper():
    return dict(model_type="whisper", d_model=16, vocab_size=32, num_mel_bins=4,
        max_source_positions=8, max_target_positions=16, encoder_layers=2,
        decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=24, decoder_ffn_dim=24, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, decoder_start_token_id=1, dropout=0., attention_dropout=0.)


class WhisperAdmissionTests(unittest.TestCase):
    def test_unsupported_structures_fail_before_weight_io(self):
        for changes in ({"activation_function": "relu"}, {"scale_embedding": True},
                {"encoder_attention_heads": 3}, {"model_type": "llama"},
                {"is_encoder_decoder": False}, {"quantization_config": {"quant_method": "fp8"}}):
            with self.subTest(changes=changes), self.assertRaises(ExporterError):
                WhisperConfig.from_hf({**tiny_whisper(), **changes})
        with self.assertRaises(ExporterError): WhisperConfig.from_hf(tiny_whisper(), max_context=17)


@unittest.skipUnless(HAVE_HF, "independent HF oracle needs torch and transformers")
class WhisperNativeReferenceTests(unittest.TestCase):
    def test_explicit_equal_kv_heads_preserves_mha_variants(self):
        torch.manual_seed(7113)
        torch.set_num_threads(1)
        hf = HFWhisperConfig(**{**tiny_whisper(), "encoder_attention_heads": 1,
            "decoder_attention_heads": 4})
        hf._attn_implementation = "eager"
        model = WhisperForConditionalGeneration(hf).float().eval()
        source = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        config = WhisperConfig.from_hf(hf.to_dict(), max_context=16)
        encoder, ew = build_whisper_encoder_graph(config, source)
        decoder, dw = build_whisper_decoder_graph(config, source)
        features = np.random.default_rng(5103).standard_normal((1, 4, 16)).astype(np.float32)
        tokens = np.array([[1, 8, 3]], np.int32)
        keep = np.array([[1, 0, 1]], np.int32)
        with torch.inference_mode():
            expected_memory = model.model.encoder(torch.from_numpy(features)).last_hidden_state
            expected_logits = model(encoder_outputs=(expected_memory,),
                decoder_input_ids=torch.from_numpy(tokens).long(),
                decoder_attention_mask=torch.from_numpy(keep).long(), use_cache=False).logits.numpy()
        root = Path(__file__).resolve().parents[2]
        for role, graph, weights, heads in (("encoder", encoder, ew, 1), ("decoder", decoder, dw, 4)):
            attention = [node for node in graph["nodes"] if node["opType"] == "CrossSDPA"]
            self.assertTrue(attention)
            for node in attention:
                self.assertEqual(node["params"]["heads"], heads)
                self.assertEqual(node["params"]["kv_heads"], heads)
                self.assertEqual(config.hidden % heads, 0)
            legacy = json.loads(json.dumps(graph))
            for node in legacy["nodes"]:
                if node["opType"] == "CrossSDPA":
                    del node["params"]["kv_heads"]
            for profile in ("volvoxai", "volvoxai-lite"):
                with self.subTest(role=role, profile=profile), patch.dict(os.environ,
                        {"VOLVOXAI_NATIVE_RUNNER": str(root / "native" / profile)}):
                    inputs = {"input_features": features} if role == "encoder" else {
                        "token_ids": tokens, "position_ids": np.arange(3, dtype=np.int32),
                        "attention_mask": keep, "encoder_memory": expected_memory.numpy()}
                    output = "encoder_memory" if role == "encoder" else "logits"
                    results = []
                    for candidate in (graph, legacy):
                        ir = import_runtime_package(candidate, weights)
                        if role == "decoder": ir = ir.bind_shape_profile({"S": 3})
                        runner = NativeGraphRunner(ir, weights)
                        results.append(runner.run(runner.bind(inputs)).outputs[output])
                    np.testing.assert_array_equal(results[0], results[1])
                    np.testing.assert_allclose(results[0], expected_memory.numpy()
                        if role == "encoder" else expected_logits, atol=2e-5, rtol=2e-5)

    def test_encoder_decoder_and_padded_prefix(self):
        torch.manual_seed(7109)
        torch.set_num_threads(1)
        hf = HFWhisperConfig(**tiny_whisper())
        hf._attn_implementation = "eager"
        model = WhisperForConditionalGeneration(hf).float().eval()
        source = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        config = WhisperConfig.from_hf(hf.to_dict(), max_context=16)
        encoder, ew = build_whisper_encoder_graph(config, source)
        decoder, dw = build_whisper_decoder_graph(config, source, generate=True)
        features = np.random.default_rng(5101).standard_normal((1, 4, 16)).astype(np.float32)
        tokens = np.array([[1, 7, 9, 4]], np.int32)
        keep = np.array([[1, 1, 0, 1]], np.int32)
        with torch.inference_mode():
            expected_memory = model.model.encoder(torch.from_numpy(features)).last_hidden_state
            expected_logits = model(encoder_outputs=(expected_memory,), decoder_input_ids=torch.from_numpy(tokens).long(),
                decoder_attention_mask=torch.from_numpy(keep).long(), use_cache=False).logits.numpy()
        suppressed = int(expected_logits[0, -1].argmax())
        masked, mw = build_whisper_decoder_graph(config, source, generate=True, suppressed_ids=[suppressed])
        self.assertIn(masked["execution"]["generation"]["logits_output"], masked["outputs"])
        expected_masked = expected_logits.copy(); expected_masked[..., suppressed] = -np.finfo(np.float32).max
        for invalid in ((True,), (-1,), (32,), (2, 2)):
            with self.assertRaises(ExporterError):
                build_whisper_decoder_graph(config, source, generate=True, suppressed_ids=invalid)
        root = Path(__file__).resolve().parents[2]
        for profile in ("volvoxai", "volvoxai-lite"):
            with self.subTest(profile=profile), patch.dict(os.environ, {"VOLVOXAI_NATIVE_RUNNER": str(root / "native" / profile)}):
                runner = NativeGraphRunner(import_runtime_package(encoder, ew), ew)
                memory = runner.run(runner.bind({"input_features": features})).outputs["encoder_memory"]
                np.testing.assert_allclose(memory, expected_memory.numpy(), atol=2e-5, rtol=2e-5)
                ir = import_runtime_package(decoder, dw).bind_shape_profile({"S": 4})
                runner = NativeGraphRunner(ir, dw)
                actual = runner.run(runner.bind({"token_ids": tokens, "position_ids": np.arange(4, dtype=np.int32),
                    "attention_mask": keep, "encoder_memory": memory})).outputs
                np.testing.assert_allclose(actual["logits"], expected_logits, atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(actual["generated"], expected_logits.argmax(-1))
                ir = import_runtime_package(masked, mw).bind_shape_profile({"S": 4})
                runner = NativeGraphRunner(ir, mw)
                actual = runner.run(runner.bind({"token_ids": tokens, "position_ids": np.arange(4, dtype=np.int32),
                    "attention_mask": keep, "encoder_memory": memory})).outputs
                np.testing.assert_allclose(actual["logits"], expected_logits, atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(actual["generated"], expected_masked.argmax(-1))
        self.assertEqual(decoder["execution"]["invariant_inputs"], ["encoder_memory"])
        self.assertNotIn("input_features", decoder["inputs"])
        self.assertTrue(all("audio.encoder" not in node["id"] for node in decoder["nodes"]))

    def test_cli_split_packages_and_incompatible_options(self):
        torch.manual_seed(7111)
        config = HFWhisperConfig(**tiny_whisper())
        model = WhisperForConditionalGeneration(config).float().eval()
        with tempfile.TemporaryDirectory(prefix="volvox-whisper-workflow-") as directory:
            root = Path(directory); source = root / "source"; source.mkdir()
            (source / "config.json").write_text(json.dumps(config.to_dict()))
            save_file({name: value.detach().numpy() for name, value in model.state_dict().items()}, str(source / "model.safetensors"))
            for role, options in (("encoder", ["--hf-whisper-encoder"]), ("decoder", ["--max-context", "16", "--generate"])):
                out = root / role / "model.safetensors"
                output, error = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                    status = exporter_main(export_model, ["--model", str(source), "--out", str(out), "--target", "native-cpu", *options])
                self.assertEqual(status, 0, error.getvalue())
                graph = json.loads((out.parent / "graph.json").read_text())
                self.assertEqual(graph["outputs"][0], "encoder_memory" if role == "encoder" else "logits")
                self.assertIn("input_features" if role == "encoder" else "encoder_memory", graph["inputs"])
            for options in ({"hf_whisper_encoder": True, "generate_decoder": True},
                    {"hf_whisper_encoder": True, "max_context": 16}, {"cached_decoder": True},
                    {"capture_kv": True}, {"hf_image_features": 1}):
                with self.subTest(options=options), self.assertRaises((ExporterError, ValueError)):
                    export_model(str(source), str(root / "bad" / "model.safetensors"), **options)


class BrowserMediaProcessorsTests(unittest.TestCase):
    def test_whisper_numpy_reference_across_energy_duration_and_ownership(self):
        from transformers import WhisperFeatureExtractor
        root = Path(__file__).resolve().parents[2]
        processor = json.loads((root / "tools/llm_baseline/multimodal_sensor.json").read_text())["audio"]["processor_config"]
        hf = WhisperFeatureExtractor(**{k: v for k, v in processor.items() if k != "mel_filters"})
        clock = np.arange(32000, dtype=np.float32) / np.float32(16000)
        chirp = np.sin(2 * np.pi * (110 * clock + 300 * clock * clock)).astype(np.float32)
        impulse = np.zeros(6400, np.float32); impulse[[0, 160, 399, 400, 6399]] = [1, -.5, .25, -1, .75]
        cases = {"silence": np.zeros(160, np.float32), "impulse": impulse, "chirp": chirp,
            "low_gain": chirp * np.float32(1e-4), "high_gain": chirp * np.float32(8),
            "noise": np.random.default_rng(8241).normal(0, .03, 17777).astype(np.float32),
            "truncated": np.resize(chirp, 480123).astype(np.float32)}
        with tempfile.TemporaryDirectory(prefix="volvox-browser-media-unit-") as directory:
            work = Path(directory); (work / "processor.json").write_text(json.dumps(processor))
            for name, pcm in cases.items(): (work / (name + ".f32")).write_bytes(pcm.tobytes())
            code = """
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {whisperFeatures} from './tools/llm_baseline/browser_media_processors.mjs';
const directory=process.argv[1],processor=JSON.parse(fs.readFileSync(directory+'/processor.json'));
for(const name of JSON.parse(process.argv[2])){
  const bytes=fs.readFileSync(directory+'/'+name+'.f32'),pcm=new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  const result=whisperFeatures(pcm,processor);assert.deepEqual(result.shape,[1,80,3000]);
  const held=result.values.slice();pcm.fill(0);assert.deepEqual(result.values,held);
  fs.writeFileSync(directory+'/'+name+'.out.f32',new Uint8Array(result.values.buffer));
}
for(const pcm of [new Float64Array(3),new Float32Array(),Float32Array.of(NaN)])assert.throws(()=>whisperFeatures(pcm,processor));
assert.throws(()=>whisperFeatures(new Float32Array(3),{...processor,sampling_rate:8000}));
"""
            subprocess.run(["node", "--input-type=module", "-e", code, str(work), json.dumps(list(cases))], cwd=root, check=True)
            for name, pcm in cases.items():
                with self.subTest(signal=name):
                    padded = np.zeros((1, 480000), np.float32); length = min(len(pcm), 480000); padded[0, :length] = pcm[:length]
                    expected = hf._np_extract_fbank_features(padded, "cpu")
                    observed = np.fromfile(work / (name + ".out.f32"), np.float32).reshape(1, 80, 3000)
                    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-5)


class WhisperAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = json.loads((Path(__file__).resolve().parents[2] /
            "tools/llm_baseline/multimodal_sensor.json").read_text())["audio"]["processor_config"]

    def test_waveforms_match_independent_hf_and_own_their_snapshot(self):
        from transformers import WhisperFeatureExtractor
        import torch
        torch.set_num_threads(1)
        hf = WhisperFeatureExtractor(**{k: v for k, v in self.processor.items() if k != "mel_filters"})
        clock = np.arange(24000, dtype=np.float32) / np.float32(16000)
        wave = (.2 * np.sin(2 * np.pi * 220 * clock) + .05 * np.sin(2 * np.pi * 730 * clock)).astype(np.float32)
        for pcm in (np.zeros(8000, np.float32), wave):
            with self.subTest(samples=len(pcm)):
                padded = np.zeros((1, 480000), np.float32)
                padded[0, :len(pcm)] = pcm
                expected = hf._np_extract_fbank_features(padded, "cpu")
                actual = prepare_whisper_features(pcm, self.processor)
                np.testing.assert_array_equal(actual, expected)
                torch_features = hf(pcm.copy(), sampling_rate=16000, return_tensors="np").input_features
                self.assertLessEqual(float(np.abs(actual - torch_features).max()), 5e-5)
                snapshot = actual.copy()
                pcm[:] = 0
                np.testing.assert_array_equal(actual, snapshot)
                self.assertEqual(actual.shape, (1, 80, 3000))
                self.assertEqual(actual.dtype, np.float32)

    def test_numpy_branch_is_exact_across_signal_energy_and_duration(self):
        from transformers import WhisperFeatureExtractor
        hf = WhisperFeatureExtractor(**{k: v for k, v in self.processor.items() if k != "mel_filters"})
        clock = np.arange(32000, dtype=np.float32) / np.float32(16000)
        chirp = np.sin(2 * np.pi * (110 * clock + 300 * clock * clock)).astype(np.float32)
        noise = np.random.default_rng(8241).normal(0, .03, 17777).astype(np.float32)
        impulse = np.zeros(6400, np.float32)
        impulse[[0, 160, 399, 400, 6399]] = [1, -.5, .25, -1, .75]
        long_wave = np.resize(chirp, 480123).astype(np.float32)
        for name, pcm in (("silence", np.zeros(160, np.float32)), ("impulse", impulse),
                ("chirp", chirp), ("low_gain", chirp * np.float32(1e-4)),
                ("high_gain", chirp * np.float32(8)), ("noise", noise), ("truncated", long_wave)):
            with self.subTest(signal=name):
                padded = np.zeros((1, 480000), np.float32)
                length = min(len(pcm), padded.shape[1])
                padded[0, :length] = pcm[:length]
                expected = hf._np_extract_fbank_features(padded, "cpu")
                actual = prepare_whisper_features(pcm, self.processor)
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertFalse(np.shares_memory(actual, pcm))

    def test_bad_input_and_implicit_resampling_refused(self):
        for pcm in (np.zeros((1, 3), np.float32), np.zeros(3, np.float64), np.array([np.nan], np.float32), np.zeros(0, np.float32)):
            with self.assertRaises(ValueError): prepare_whisper_features(pcm, self.processor)
        with self.assertRaises(ValueError): prepare_whisper_features(np.zeros(3, np.float32), self.processor, sampling_rate=8000)


class VisionAdmissionTests(unittest.TestCase):
    def test_published_qwen_nested_model_types(self):
        baseline=QwenVisionConfig.from_hf(tiny_vision_config())
        for family in ("qwen3_5","qwen3_5_moe","qwen4_exp"):
            with self.subTest(family=family):
                self.assertEqual(QwenVisionConfig.from_hf({**tiny_vision_config(),"model_type":family}),baseline)

    def test_grid_and_structural_flag_refusals(self):
        config = QwenVisionConfig.from_hf(tiny_vision_config())
        for grid in ([[1, 3, 2]], [[0, 2, 2]], [[1., 2., 2.]], [[1, 2]], [[1, 10000, 10000]]):
            with self.subTest(grid=grid), self.assertRaises(ExporterError):
                _fixed_grid(config, grid)
        for flags in ({"deepstack_visual_indexes": [0]}, {"hidden_act": "relu"},
                {"num_position_embeddings": 15}, {"rope_parameters": {"rope_type": "dynamic"}}):
            with self.subTest(flags=flags), self.assertRaises(ExporterError):
                QwenVisionConfig.from_hf({**tiny_vision_config(), **flags})


@unittest.skipUnless(HAVE_HF, "independent oracle needs torch and transformers")
class VisionReferenceTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(3925)
        torch.set_num_threads(1)
        hf = Qwen3_5VisionConfig(**tiny_vision_config())
        hf._attn_implementation = "eager"
        return hf, Qwen3_5VisionModel(hf).float().eval()

    def test_fixed_position_initializers_against_hf(self):
        hf, model = self._model()
        config = QwenVisionConfig.from_hf(hf.to_dict())
        source = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        for grid in ([[1, 2, 2]], [[1, 2, 4], [2, 2, 2]]):
            with self.subTest(grid=grid):
                graph, weights = build_qwen_vision_graph(config, source, grid_thw=grid)
                _, coords, _ = _fixed_grid(config, grid)
                expected_coords = get_vision_position_ids(torch.tensor(grid), config.merge)
                np.testing.assert_array_equal(coords, expected_coords.numpy())
                indices, factors = get_vision_bilinear_indices_and_weights(torch.tensor(grid),
                    num_grid_per_side=4, spatial_merge_size=2)
                with torch.inference_mode():
                    expected = (model.pos_embed(indices) * factors[:, :, None]).sum(0).numpy()
                    angles = model.rotary_pos_emb(expected_coords).flatten(1)
                    phase = torch.cat((angles, angles), dim=-1)
                np.testing.assert_allclose(weights["hf.vision.position"], expected, atol=1e-7, rtol=1e-7)
                np.testing.assert_allclose(weights["hf.vision.cos"][0], phase.cos().numpy(), atol=1e-7, rtol=1e-7)
                np.testing.assert_allclose(weights["hf.vision.sin"][0], phase.sin().numpy(), atol=1e-7, rtol=1e-7)
                self.assertEqual(graph["inputs"]["pixel_patches"]["shape"], [len(coords), 24])

    def test_native_patch_attention_and_spatial_merger(self):
        hf, model = self._model()
        config = QwenVisionConfig.from_hf(hf.to_dict())
        source = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        rng = np.random.default_rng(9131)
        root = Path(__file__).resolve().parents[2]
        for grid in ([[1, 2, 2]], [[1, 2, 4], [2, 2, 2]]):
            graph, weights = build_qwen_vision_graph(config, source, grid_thw=grid)
            patches = rng.standard_normal(graph["inputs"]["pixel_patches"]["shape"]).astype(np.float32)
            with torch.inference_mode():
                expected = model(torch.from_numpy(patches), grid_thw=torch.tensor(grid)).pooler_output.numpy()
            ir = import_runtime_package(graph, weights)
            for profile in ("volvoxai", "volvoxai-lite"):
                with self.subTest(grid=grid, profile=profile), patch.dict(os.environ,
                        {"VOLVOXAI_NATIVE_RUNNER": str(root / "native" / profile)}):
                    runner = NativeGraphRunner(ir, weights)
                    actual = runner.run(runner.bind({"pixel_patches": patches})).outputs["image_features"]
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_llama4_tiles_rotary_shuffle_adapter_and_projector(self):
        torch.manual_seed(1470)
        torch.set_num_threads(1)
        hf=Llama4VisionConfig(hidden_size=16,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=2,num_channels=3,image_size=8,patch_size=2,
            projector_input_dim=16,projector_output_dim=16,vision_output_dim=16,pixel_shuffle_ratio=.5)
        hf._attn_implementation="eager"
        model=Llama4VisionModel(hf).float().eval()
        config=NativeLlama4VisionConfig.from_hf(hf.to_dict())
        source={name:value.detach().numpy() for name,value in model.state_dict().items()}
        rng=np.random.default_rng(7104)
        projector=(rng.standard_normal((20,16))*.05).astype(np.float32)
        source["multi_modal_projector.linear_1.weight"]=projector
        root=Path(__file__).resolve().parents[2]
        for tiles in (1,2):
            graph,weights=build_llama4_vision_graph(config,source,tiles=tiles,text_hidden_size=20)
            pixels=rng.standard_normal((tiles,3,8,8)).astype(np.float32)
            with torch.inference_mode():
                expected=model(torch.from_numpy(pixels)).last_hidden_state.reshape(-1,16) @ torch.from_numpy(projector.T)
            ir=import_runtime_package(graph,weights)
            for profile in ("volvoxai","volvoxai-lite"):
                with self.subTest(tiles=tiles,profile=profile),patch.dict(os.environ,
                        {"VOLVOXAI_NATIVE_RUNNER":str(root/"native"/profile)}):
                    runner=NativeGraphRunner(ir,weights)
                    actual=runner.run(runner.bind({"pixel_values":pixels})).outputs["image_features"]
                    np.testing.assert_allclose(actual,expected.numpy(),atol=2e-5,rtol=2e-5)


if __name__=="__main__":unittest.main()

