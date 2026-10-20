"""Extended decoder graphs compared with independently constructed HF models."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib

import numpy as np
from safetensors.numpy import save_file

from tools.export_safetensors import export_model
from tools.exporter.cli import main as exporter_main
from tools.exporter.errors import ExporterError
from tools.exporter.generated.volvox_enums import ExecutionEffect, ExecutionRangeKind
from tools.exporter.hf_decoder import read_hf_checkpoint
from tools.exporter.hf_deepseek import (
    DeepSeekDecoderConfig, _checked_weights, _rotary_tables,
    build_deepseek_decoder_graph, export_hf_deepseek,
)
from tools.exporter.hf_extended import (
    ExtendedDecoderConfig, build_extended_decoder_graph, export_hf_extended)
from tools.exporter.hf_graph import GraphBuilder
from tools.exporter.runtime_ir import import_runtime_package
from tools.exporter.native_execution import NativeGraphRunner
from tools.exporter.hf_vision import QwenVisionConfig, build_qwen_vision_graph
from tools.llm_baseline.generate_modern_incremental_stress import (tiny_config, tiny_deepseek_config,
    tiny_qwen4, tiny_vision_config, execute_native as execute, qwen4_reference)

try:
    import torch
    from transformers import (Qwen3MoeConfig, Qwen3MoeForCausalLM,
        Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM,
        Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM, Llama4TextConfig, Llama4ForCausalLM,
        DeepseekV2Config, DeepseekV2ForCausalLM,
        DeepseekV3Config, DeepseekV3ForCausalLM,
        DeepseekV4Config, DeepseekV4ForCausalLM)
    HAVE_HF = True
except ImportError:
    HAVE_HF = False


def random_deepseek_weights(config, layout="packed"):
    rng = np.random.default_rng(912)
    result = {name: (rng.normal(size=shape) * 0.1).astype(np.float32)
              for name, shape in config.weight_shapes(expert_layout=layout).items()}
    for name, value in result.items():
        if name.endswith(".gate.tid2eid"):
            result[name] = np.tile(np.arange(config.top_k, dtype=np.int64), (config.vocab_size, 1))
        elif name.endswith("norm.weight"):
            value[:] = 1 + rng.normal(size=value.shape) * 0.03
    return result


class ExtendedAdmissionTests(unittest.TestCase):
    def test_real_family_geometry_and_alias_refusal(self):
        for family in ("qwen3_moe", "qwen3_next", "qwen3_5_text", "qwen3_5_moe_text", "llama4_text"):
            with self.subTest(family=family):
                config = ExtendedDecoderConfig.from_hf(tiny_config(family), max_context=16)
                self.assertEqual(config.max_context, 16)
                self.assertEqual(config.rotary_dim, 8 if family in ("llama4_text", "qwen3_moe") else 4)
        for model_type in ("qwen3_8", "qwen3.8"):
            with self.subTest(model_type=model_type), self.assertRaises(ExporterError):
                ExtendedDecoderConfig.from_hf(tiny_config(model_type))

    def test_structural_flags_and_quantized_storage_refused(self):
        for changes in ({"quantization_config": {"quant_method": "fp8"}},
                {"use_sliding_window": True}, {"mlp_bias": True},
                {"num_experts_per_tok": 4}, {"output_gate_type": "relu"},
                {"linear_num_value_heads": 3, "linear_num_key_heads": 2}):
            with self.subTest(changes=changes), self.assertRaises(ExporterError):
                ExtendedDecoderConfig.from_hf({**tiny_config("qwen3_next"), **changes})
        bad = tiny_config("qwen3_5_text")
        bad["rope_parameters"]["rope_type"] = "dynamic"
        with self.assertRaises(ExporterError):
            ExtendedDecoderConfig.from_hf(bad)

    def test_llama_dense_and_routed_widths_are_distinct(self):
        config = ExtendedDecoderConfig.from_hf(tiny_config("llama4_text"))
        shapes = config.weight_shapes()
        self.assertEqual(shapes["model.layers.0.feed_forward.gate_proj.weight"], (32, 16))
        self.assertEqual(shapes["model.layers.1.feed_forward.shared_expert.gate_proj.weight"], (24, 16))


@unittest.skipUnless(HAVE_HF, "independent oracle needs torch and transformers")
class ExtendedNativeReferenceTests(unittest.TestCase):
    def _compare(self, family, config_cls, model_cls, *, individual_experts=False):
        torch.manual_seed(2603)
        torch.set_num_threads(1)
        raw = tiny_config(family)
        hf = config_cls(**raw)
        hf._attn_implementation = "eager"
        model = model_cls(hf).float().eval()
        config = ExtendedDecoderConfig.from_hf(hf.to_dict(), max_context=16)
        source = {name: tensor.detach().numpy() for name, tensor in model.state_dict().items()}
        if individual_experts:
            for layer in config.moe_layers:
                prefix = f"model.layers.{layer}.mlp.experts."
                gate_up = source.pop(prefix + "gate_up_proj")
                down = source.pop(prefix + "down_proj")
                gate, up = np.split(gate_up, 2, axis=1)
                for expert in range(config.experts):
                    for label, tensor in (("gate_proj", gate[expert]), ("up_proj", up[expert]), ("down_proj", down[expert])):
                        source[prefix + str(expert) + "." + label + ".weight"] = tensor
        graph, weights = build_extended_decoder_graph(config, source, capture_kv=True)
        ir = import_runtime_package(graph, weights)
        root = Path(__file__).resolve().parents[2]
        for length, masked, positions in ((1, False, [15]), (3, False, [0, 3, 15]), (5, True, [0, 1, 2, 3, 4])):
            inputs = {"token_ids": ((np.arange(length)[None] + 2) % 24).astype(np.int32),
                "position_ids": np.asarray(positions, np.int32),
                "attention_mask": np.ones((1, length), np.int32)}
            if masked:
                inputs["attention_mask"][0, 0] = 0
            if "sequence_ids" in graph["inputs"]:
                inputs["sequence_ids"] = np.arange(length, dtype=np.int32)
            if "chunk_attention_mask" in graph["inputs"]:
                # HF chunk boundaries start after left padding, independently
                # of the caller's absolute rotary position IDs.
                sequence = np.arange(length) - int(np.argmax(inputs["attention_mask"][0] != 0))
                inputs["chunk_attention_mask"] = ((sequence[:, None] // config.chunk_size == sequence[None, :] // config.chunk_size)
                    & (inputs["attention_mask"] != 0))[None].astype(np.int32)
            # Transformers 5.13's padding helper erroneously requires B>1.
            # Current official main removes that guard. Limit the oracle fix to
            # the upstream helper; all projections, recurrence, attention and
            # routed experts remain the independent HF implementation.
            module = importlib.import_module(model.__class__.__module__)
            def mask_padding(hidden, mask):
                return hidden if mask is None else hidden * mask[:, :, None].to(hidden.dtype)
            mask_patch = (patch.object(module, "apply_mask_to_padding_states", mask_padding)
                if family not in ("llama4_text", "qwen3_moe") else patch.dict({}, {}))
            with torch.inference_mode(), mask_patch:
                expected = model(input_ids=torch.from_numpy(inputs["token_ids"]).long(),
                    position_ids=torch.from_numpy(inputs["position_ids"][None]).long(),
                    attention_mask=torch.from_numpy(inputs["attention_mask"]).long(), use_cache=False).logits.numpy()
            for profile in ("volvoxai", "volvoxai-lite"):
                with self.subTest(family=family, length=length, masked=masked, profile=profile), patch.dict(os.environ,
                        {"VOLVOXAI_NATIVE_RUNNER": str(root / "native" / profile)}):
                    runner = NativeGraphRunner(ir.bind_shape_profile({"S": length}), weights)
                    actual = runner.run(runner.bind(inputs)).outputs["logits"]
                    # Logits on padding queries have no model meaning. HF eager
                    # finite-min masking and the engine's zero all-masked row
                    # intentionally differ there; keep every valid query.
                    selected = inputs["attention_mask"].astype(bool)
                    np.testing.assert_allclose(actual[selected], expected[selected], atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(actual[selected].argmax(-1), expected[selected].argmax(-1))

    def test_qwen3_moe_packed(self):
        self._compare("qwen3_moe", Qwen3MoeConfig, Qwen3MoeForCausalLM)

    def test_qwen3_moe_individual(self):
        self._compare("qwen3_moe", Qwen3MoeConfig, Qwen3MoeForCausalLM, individual_experts=True)

    def test_qwen3_next(self):
        self._compare("qwen3_next", Qwen3NextConfig, Qwen3NextForCausalLM)

    def test_qwen3_5_dense(self):
        self._compare("qwen3_5_text", Qwen3_5TextConfig, Qwen3_5ForCausalLM)

    def test_qwen3_5_moe(self):
        self._compare("qwen3_5_moe_text", Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM)

    def test_llama4_dense_routed_chunked_no_rope(self):
        self._compare("llama4_text", Llama4TextConfig, Llama4ForCausalLM)


FAMILIES = ("qwen3_moe", "qwen3_next", "qwen3_5_text", "qwen3_5_moe_text", "llama4_text")


def fixture(family):
    config = ExtendedDecoderConfig.from_hf(tiny_config(family), max_context=16)
    random = np.random.default_rng(1709)
    weights = {name: random.uniform(-.02, .02, shape).astype(np.float32)
               for name, shape in config.weight_shapes().items()}
    return config, weights


class ExtendedCachedImporterTests(unittest.TestCase):
    def test_cache_contract_keeps_prefix_math_and_private_state(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                config, weights = fixture(family)
                reference, reference_weights = build_extended_decoder_graph(config, weights, generate=True)
                cached, cached_weights = build_extended_decoder_graph(config, weights, cached=True)
                prefix_nodes = copy.deepcopy(cached["nodes"])
                for node in prefix_nodes:
                    if node["opType"] == "GatedDeltaNet":
                        node["outputs"].pop("state")
                self.assertEqual(prefix_nodes, reference["nodes"])
                self.assertEqual(cached["outputs"], reference["outputs"])
                self.assertEqual(set(cached_weights), set(reference_weights))
                for name in cached_weights:
                    np.testing.assert_array_equal(cached_weights[name], reference_weights[name])
                execution = cached["execution"]
                entries = {entry["name"]: entry for entry in execution["entrypoints"]}
                self.assertEqual(entries["initialize"]["effect"], int(ExecutionEffect.INITIALIZE))
                self.assertEqual(entries["transition"]["effect"], int(ExecutionEffect.TRANSITION))
                self.assertEqual(entries["transition"]["allowed_ranges"],
                    [int(ExecutionRangeKind.SLOT_ROWS), int(ExecutionRangeKind.DEPENDENCIES)])
                self.assertNotIn("full_prefix", execution["generation"])
                self.assertEqual(execution["generation"]["incremental_controls"]["position_input"], "position_ids")
                self.assertEqual(execution["invariant_inputs"], ["active_lengths"])
                self.assertNotIn("active_lengths", execution["incremental"]["update_inputs"])
                self.assertEqual(cached["inputs"]["active_lengths"], {"shape": [1], "dtype": "int32"})
                recurrent = [node for node in cached["nodes"] if node["opType"] == "GatedDeltaNet"]
                self.assertEqual(len(recurrent), config.layer_types.count("linear_attention"))
                for node in recurrent:
                    state = node["outputs"]["state"]
                    self.assertEqual(state["shape"], [1, config.linear_value_heads,
                        config.linear_key_dim, config.linear_value_dim])
                    self.assertEqual(state["dtype"], "float32")
                    self.assertNotIn("initial_state", node["inputs"])
                    self.assertNotIn(state["tensor"], cached["inputs"])
                    self.assertNotIn(state["tensor"], cached["outputs"])
                ir = import_runtime_package(cached, cached_weights)
                for length in (1, 5, 16):
                    bound = ir.bind_shape_profile({"S": length})
                    self.assertEqual(bound.tensors["generated"].shape, (1, length))
                    for node in recurrent:
                        self.assertEqual(bound.tensors[node["outputs"]["state"]["tensor"]].shape,
                            (1, config.linear_value_heads, config.linear_key_dim, config.linear_value_dim))

    def test_multimodal_chunk_controls_keep_conditioning_immutable(self):
        for family in ("llama4_text", "qwen3_5_text", "qwen3_5_moe_text"):
            with self.subTest(family=family):
                config, weights = fixture(family)
                graph, _ = build_extended_decoder_graph(config, weights, cached=True, image_features_count=2)
                execution = graph["execution"]
                controls = execution["generation"]["incremental_controls"]
                self.assertEqual(controls["zero_append_inputs"], ["image_feature_ids", "image_token_mask"])
                self.assertEqual(execution["invariant_inputs"], ["active_lengths", "image_features"])
                self.assertNotIn("image_features", execution["incremental"]["update_inputs"])
                if family == "llama4_text":
                    self.assertEqual((controls["sequence_input"], controls["chunk_mask_input"], controls["chunk_size"]),
                        ("sequence_ids", "chunk_attention_mask", 2))
                else:
                    self.assertEqual(controls["multimodal_position_input"], "mrope_position_ids")
                    self.assertEqual(graph["inputs"]["mrope_position_ids"]["shape"], [3, "S"])

    def test_invalid_cache_options_reject_before_checkpoint_storage(self):
        config, weights = fixture("qwen3_next")
        for options in ({"cached": 1}, {"cached": "true"}, {"cached": True, "capture_kv": True}):
            with self.subTest(options=options), patch("tools.exporter.hf_extended._text_weights") as load:
                with self.assertRaises(ExporterError):
                    build_extended_decoder_graph(config, weights, **options)
                load.assert_not_called()
            with self.subTest(export_options=options), patch("tools.exporter.hf_extended.read_checkpoint_storage") as load:
                with self.assertRaises(ExporterError):
                    export_hf_extended("unopened-source", "unused-output", **options)
                load.assert_not_called()

    def test_cached_export_report_names_retained_recurrence(self):
        config, weights = fixture("qwen3_next")
        with tempfile.TemporaryDirectory(prefix="volvox-extended-cache-export-") as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(tiny_config("qwen3_next")))
            save_file(weights, str(source / "model.safetensors"))
            report = []
            out = Path(temporary) / "output.safetensors"
            graph = export_hf_extended(source, out, max_context=16, cached=True, report_callback=report.append)
            self.assertTrue(out.exists())
            self.assertEqual(json.loads((out.parent / "graph.json").read_text()), graph)
            features = report[0]["features"]["hf_decoder"]
            self.assertIn("state:incremental-kv", features)
            self.assertIn("state:gated-delta-and-causal-convolution-history", features)
            self.assertIn("generation:incremental-rows", features)
            self.assertNotIn("generation:full-prefix-recomputation", features)


@unittest.skipUnless(HAVE_HF, "fixture needs torch and transformers")
class ExtendedExportWorkflowTests(unittest.TestCase):
    def export(self, arguments):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = exporter_main(export_model, arguments)
        self.assertEqual(status, 0, error.getvalue())

    def test_qwen_separate_image_decoder_and_generate_contract(self):
        import torch
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        torch.manual_seed(5139)
        config = Qwen3_5Config(text_config=tiny_config("qwen3_5_text"),
            vision_config=tiny_vision_config(), image_token_id=23)
        config._attn_implementation = "eager"
        model = Qwen3_5ForConditionalGeneration(config).float().eval()
        source = {name:value.detach().numpy() for name,value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix="volvox-hf-workflow-") as temporary:
            directory = Path(temporary) / "source"; directory.mkdir()
            (directory / "config.json").write_text(json.dumps(config.to_dict()))
            save_file(source, str(directory / "model.safetensors"))
            encoder = Path(temporary) / "encoder" / "model.safetensors"
            decoder = Path(temporary) / "decoder" / "model.safetensors"
            self.export(["--model", str(directory), "--out", str(encoder),
                "--target", "native-cpu", "--hf-vision-grid", "[[1,2,2]]"])
            self.export(["--model", str(directory), "--out", str(decoder),
                "--target", "native-cpu", "--max-context", "16", "--hf-image-features", "1", "--generate"])
            image = json.loads((encoder.parent / "graph.json").read_text())
            text = json.loads((decoder.parent / "graph.json").read_text())
            self.assertEqual(image["outputs"], ["image_features"])
            self.assertEqual(image["inputs"]["pixel_patches"]["shape"], [4,24])
            self.assertEqual(text["inputs"]["image_features"]["shape"], [1,16])
            generation = text["execution"]["generation"]
            self.assertEqual(generation["initialize_entrypoint"], "evaluate")
            self.assertEqual(generation["transition_entrypoint"], "evaluate")
            self.assertEqual(generation["full_prefix"]["multimodal_position_input"], "mrope_position_ids")
            self.assertEqual(generation["full_prefix"]["zero_append_inputs"], ["image_feature_ids","image_token_mask"])
            self.assertEqual(text["execution"]["invariant_inputs"], ["image_features"])
            self.assertIn("generated", text["outputs"])
            cached = export_model(str(directory), str(Path(temporary)/"cache.safetensors"),
                cached_decoder=True, hf_image_features=1)
            generation = cached["execution"]["generation"]
            self.assertEqual(generation["initialize_entrypoint"], "initialize")
            self.assertEqual(generation["transition_entrypoint"], "transition")
            self.assertNotIn("full_prefix", generation)
            self.assertEqual(generation["incremental_controls"]["multimodal_position_input"], "mrope_position_ids")
            self.assertEqual(generation["incremental_controls"]["zero_append_inputs"],
                ["image_feature_ids", "image_token_mask"])
            self.assertEqual(cached["execution"]["invariant_inputs"], ["active_lengths", "image_features"])
            with self.assertRaises(ValueError):
                export_model(str(directory), str(Path(temporary)/"bad.safetensors"),
                    hf_vision_grid=[[1,2,2]], generate_decoder=True)

    def test_llama4_vision_tiles_and_text_position_controls(self):
        import torch
        from transformers import Llama4Config, Llama4ForConditionalGeneration
        vision=dict(hidden_size=16,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,
            num_channels=3,image_size=8,patch_size=2,projector_input_dim=16,projector_output_dim=16,
            vision_output_dim=16,pixel_shuffle_ratio=.5)
        config=Llama4Config(text_config=tiny_config("llama4_text"),vision_config=vision,image_token_index=23)
        config._attn_implementation="eager"
        model=Llama4ForConditionalGeneration(config).float().eval()
        source={name:value.detach().numpy() for name,value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix="volvox-hf-workflow-") as temporary:
            directory=Path(temporary)/"source";directory.mkdir()
            (directory/"config.json").write_text(json.dumps(config.to_dict()))
            save_file(source,str(directory/"model.safetensors"))
            encoder=Path(temporary)/"encoder"/"model.safetensors"
            decoder=Path(temporary)/"decoder"/"model.safetensors"
            self.export(["--model",str(directory),"--out",str(encoder),
                "--target","native-cpu","--hf-vision-tiles","2"])
            self.export(["--model",str(directory),"--out",str(decoder),
                "--target","native-cpu","--max-context","16","--hf-image-features","8","--generate"])
            image=json.loads((encoder.parent/"graph.json").read_text())
            text=json.loads((decoder.parent/"graph.json").read_text())
            self.assertEqual(image["inputs"]["pixel_values"]["shape"],[2,3,8,8])
            self.assertEqual(text["inputs"]["image_features"]["shape"],[8,16])
            controls=text["execution"]["generation"]["full_prefix"]
            self.assertEqual(controls["sequence_input"],"sequence_ids")
            self.assertEqual((controls["chunk_mask_input"],controls["chunk_size"]),("chunk_attention_mask",2))
            self.assertNotIn("multimodal_position_input",controls)


class DeepSeekImporterTests(unittest.TestCase):
    def test_official_v4_namespace_is_only_a_checkpoint_name_adapter(self):
        from tools.exporter.hf_deepseek import _v4_source_name
        config = DeepSeekDecoderConfig.from_hf(tiny_deepseek_config("deepseek_v4"))
        canonical = random_deepseek_weights(config)
        # Expand the compact fixture's packed expert banks to the checkpoint's
        # individual layout before exercising the published inference names.
        if any(name.endswith("experts.gate_up_proj") for name in canonical):
            for layer in range(config.layers):
                prefix = f"model.layers.{layer}.mlp.experts."
                bank = canonical.pop(prefix + "gate_up_proj")
                down = canonical.pop(prefix + "down_proj")
                for expert in range(config.experts):
                    canonical[prefix + f"{expert}.gate_proj.weight"] = bank[expert, :config.moe_intermediate_size]
                    canonical[prefix + f"{expert}.up_proj.weight"] = bank[expert, config.moe_intermediate_size:]
                    canonical[prefix + f"{expert}.down_proj.weight"] = down[expert]
        official = {_v4_source_name(name): value for name, value in canonical.items()}
        graph, weights = build_deepseek_decoder_graph(config, canonical)
        renamed_graph, renamed_weights = build_deepseek_decoder_graph(config, official)
        self.assertEqual(graph, renamed_graph)
        self.assertEqual(set(weights), set(renamed_weights))
        for name in weights:
            np.testing.assert_array_equal(weights[name], renamed_weights[name])
        with self.assertRaises(ExporterError):
            _checked_weights(config, {**official, "model.embed_tokens.weight": canonical["model.embed_tokens.weight"]})
        # The published DSpark config declares its three stages through the
        # target/rate arrays despite retaining num_nextn_predict_layers=1.
        raw = tiny_deepseek_config("deepseek_v4", num_nextn_predict_layers=1,
            dspark_target_layer_ids=[0, 1, 2], compress_ratios=[0, 4, 128, 0, 0, 0])
        raw.pop("layer_types", None)
        dspark = DeepSeekDecoderConfig.from_hf(raw)
        self.assertEqual(dspark.auxiliary_layers, 3)
        source = random_deepseek_weights(dspark)
        _checked_weights(dspark, {**source, "mtp.2.aux.weight": np.ones(2, np.float32)})
        for name in ("mtp.3.aux.weight", "mtp.unknown.weight"):
            with self.subTest(name=name), self.assertRaises(ExporterError):
                _checked_weights(dspark, {**source, name: np.ones(2, np.float32)})

    def test_checked_geometry_namespace_and_storage_refusals(self):
        for family in ("deepseek_v2", "deepseek_v3", "deepseek_v4"):
            raw = tiny_deepseek_config(family)
            config = DeepSeekDecoderConfig.from_hf(raw)
            self.assertEqual(config.family, family)
            weights = random_deepseek_weights(config)
            _checked_weights(config, weights)
            for changes in ({"num_experts_per_tok": 5}, {"qk_rope_head_dim": 3},
                            {"max_position_embeddings": True}, {"hidden_act": "gelu"},
                            {"quantization_config": "fp8"}, {"moe_layer_freq": 2}):
                with self.subTest(family=family, changes=changes), self.assertRaises(ExporterError):
                    DeepSeekDecoderConfig.from_hf({**raw, **changes})
            for changed in ("missing", "extra", "nan", "shape"):
                bad = dict(weights)
                key = "model.embed_tokens.weight"
                if changed == "missing": del bad[key]
                elif changed == "extra": bad["wrong.expert.weight"] = np.ones(1, np.float32)
                elif changed == "nan": bad[key] = np.full(bad[key].shape, np.nan, np.float32)
                else: bad[key] = bad[key][:-1]
                with self.subTest(family=family, changed=changed), self.assertRaises(ExporterError):
                    _checked_weights(config, bad)
            with self.assertRaises(ExporterError):
                build_deepseek_decoder_graph(config, weights, cached=True, capture_kv=True)
        # Source quantization describes architecture, not accepted storage.
        raw = tiny_deepseek_config("deepseek_v4", quantization_config={"quant_method": "fp8"})
        config = DeepSeekDecoderConfig.from_hf(raw)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(raw))
            weights = random_deepseek_weights(config)
            save_file(weights, root / "model.safetensors")
            _, loaded, dtypes = read_hf_checkpoint(root)
            table = "model.layers.0.mlp.gate.tid2eid"
            self.assertEqual(dtypes[table], "I64")
            np.testing.assert_array_equal(loaded[table], weights[table])
            self.assertEqual(loaded[table].dtype, np.int64)
            # Unrecognized integer tensors never get admitted as F32 weights.
            save_file({"not_a_hash_buffer": np.ones(2, np.int64)}, root / "model.safetensors")
            with self.assertRaises(ExporterError):
                read_hf_checkpoint(root)

    def test_hash_bounds_and_declared_auxiliary_omission(self):
        config = DeepSeekDecoderConfig.from_hf(tiny_deepseek_config("deepseek_v4", num_nextn_predict_layers=1))
        weights = random_deepseek_weights(config)
        table = "model.layers.0.mlp.gate.tid2eid"
        for value in (-1, config.experts, 1 << 32):
            bad = {**weights, table: np.full(weights[table].shape, value, np.int64)}
            with self.subTest(value=value), self.assertRaises(ExporterError):
                _checked_weights(config, bad)
        # Duplicate IDs are meaningful repeated weighted routes in upstream.
        _checked_weights(config, {**weights, table: np.zeros(weights[table].shape, np.int64)})
        _checked_weights(config, {**weights, "model.layers.3.aux.weight": np.ones(2, np.float32)})
        with self.assertRaises(ExporterError):
            _checked_weights(config, {**weights, "model.layers.4.aux.weight": np.ones(2, np.float32)})

    def test_bounded_runtime_ir_and_compressed_bank_tables(self):
        extreme = DeepSeekDecoderConfig.from_hf(tiny_deepseek_config(rope_theta=1e-45, qk_rope_head_dim=64))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"), self.assertRaises(ExporterError):
            _rotary_tables(extreme, extreme.rope_main, interleaved=False)
        for family in ("deepseek_v2", "deepseek_v3", "deepseek_v4"):
            config = DeepSeekDecoderConfig.from_hf(tiny_deepseek_config(family))
            graph, weights = build_deepseek_decoder_graph(config, random_deepseek_weights(config), capture_kv=True)
            ir = import_runtime_package(graph, weights)
            for length in (1, 5, 16):
                ir.bind_shape_profile({"S": length}).verify_concrete_bound()
            with self.assertRaises(ExporterError):
                ir.bind_shape_profile({"S": 17})
            if family == "deepseek_v4":
                compressors = [node for node in graph["nodes"] if node["opType"] == "KVCompress"]
                self.assertEqual(len(compressors), 3)
                for node in compressors:
                    table = weights[node["inputs"]["cos"]]
                    rate = node["params"]["compress_rate"]
                    expected, _ = _rotary_tables(config, config.rope_compress, interleaved=True, rate=rate)
                    np.testing.assert_array_equal(table, expected)
                    self.assertEqual(table.shape, (16, 2))
                    self.assertNotEqual(node["inputs"]["cos"], "position_ids")
        with tempfile.TemporaryDirectory() as directory, patch("tools.exporter.hf_deepseek.read_checkpoint_storage") as reader:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(tiny_deepseek_config()))
            for options in ({}, {"max_context": 8, "cached": True, "capture_kv": True}):
                with self.assertRaises(ExporterError):
                    export_hf_deepseek(root, root / "out.safetensors", **options)
            reader.assert_not_called()

    def test_incremental_contract_retains_mla_and_completed_compressor_rows(self):
        for family in ("deepseek_v2", "deepseek_v3", "deepseek_v4"):
            with self.subTest(family=family):
                config = DeepSeekDecoderConfig.from_hf(tiny_deepseek_config(family))
                source = random_deepseek_weights(config)
                reference, reference_weights = build_deepseek_decoder_graph(config, source, generate=True)
                graph, weights = build_deepseek_decoder_graph(config, source, cached=True)
                # Caching changes the execution contract, not the causal math.
                self.assertEqual(graph["nodes"], reference["nodes"])
                self.assertEqual(set(weights), set(reference_weights))
                for name in weights:
                    np.testing.assert_array_equal(weights[name], reference_weights[name])
                self.assertEqual(graph["outputs"], ["logits", "generated"])
                self.assertEqual(graph["inputs"]["active_lengths"], {"shape": [1], "dtype": "int32"})
                execution = graph["execution"]
                self.assertNotIn("full_prefix", execution["generation"])
                self.assertEqual(execution["generation"]["incremental_controls"], {"position_input": "position_ids"})
                self.assertEqual(execution["incremental"]["active_lengths_input"], "active_lengths")
                self.assertEqual(execution["incremental"]["update_inputs"], ["token_ids", "position_ids", "attention_mask"])
                self.assertEqual(execution["invariant_inputs"], ["active_lengths"])
                ir = import_runtime_package(graph, weights)
                for capacity in (1, 5, config.max_context):
                    ir.bind_shape_profile({"S": capacity}).verify_concrete_bound()
                attention = [node for node in graph["nodes"] if node["opType"] == "SparseAttention"]
                self.assertEqual(len(attention), config.layers)
                if family == "deepseek_v4":
                    self.assertEqual(len([node for node in graph["nodes"] if node["opType"] == "KVCompress"]), 3)
                for options in ({"cached": 1}, {"cached": True, "generate": 1}):
                    with self.assertRaises(ExporterError):
                        build_deepseek_decoder_graph(config, source, **options)

    def test_incremental_checkpoint_export_declares_private_cache(self):
        raw = tiny_deepseek_config("deepseek_v4")
        config = DeepSeekDecoderConfig.from_hf(raw, max_context=8)
        source = random_deepseek_weights(config)
        reports = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(raw))
            with patch("tools.exporter.hf_deepseek.read_checkpoint_storage",
                    return_value=(raw, source, {name: "F32" for name in source})) as reader:
                graph = export_hf_deepseek(root, root / "out.safetensors", max_context=8,
                    cached=True, report_callback=reports.append)
            reader.assert_called_once()
            self.assertEqual(json.loads((root / "graph.json").read_text()), graph)
            self.assertTrue((root / "out.safetensors").is_file())
            self.assertIn("incremental", graph["execution"])
        self.assertIn("state:retained-incremental-kv-compressor-rows", reports[0]["features"]["hf_decoder"])
        self.assertIn("generation:incremental", reports[0]["features"]["hf_decoder"])

    @unittest.skipUnless(HAVE_HF, "independent reference needs Transformers and torch")
    def test_v4_completed_banks_preserve_history_only_with_timeline_positions(self):
        torch.manual_seed(746)
        torch.set_num_threads(1)
        hf = DeepseekV4Config(**tiny_deepseek_config("deepseek_v4"))
        hf._attn_implementation = "eager"
        model = DeepseekV4ForCausalLM(hf).float().eval()
        tokens = torch.arange(3, 11)[None]
        with torch.inference_mode():
            for offset in (0, 4):
                positions = torch.arange(8)[None] + offset
                outputs = [model(input_ids=tokens[:, :length], position_ids=positions[:, :length],
                    attention_mask=torch.ones(1, length, dtype=torch.long), use_cache=False).logits.numpy()
                    for length in (3, 4, 5, 8)]
                errors = [np.max(np.abs(value[:, :3] - outputs[-1][:, :3])) for value in outputs[:-1]]
                if offset == 0:
                    for value in outputs[:-1]:
                        np.testing.assert_allclose(value[:, :3], outputs[-1][:, :3], atol=2e-7, rtol=2e-6)
                else:
                    # The official block bias uses absolute position IDs.
                    # Such positions expose later banks to earlier queries,
                    # so cached admission must require position_ids[i] == i.
                    self.assertTrue(all(error > 0.01 for error in errors))

    @unittest.skipUnless(HAVE_HF, "independent reference needs Transformers and torch")
    def test_rotary_tables_match_official_reference(self):
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
        for family, C, R in (("deepseek_v3", DeepseekV3Config, DeepseekV3RotaryEmbedding),
                             ("deepseek_v4", DeepseekV4Config, DeepseekV4RotaryEmbedding)):
            for kind in ("default", "linear", "yarn"):
                raw = tiny_deepseek_config(family)
                params = {"rope_type": kind}
                if kind != "default": params["factor"] = 4.0
                if kind == "yarn":
                    params.update(original_max_position_embeddings=8, beta_fast=8, beta_slow=1, mscale=0.7, mscale_all_dim=0.2)
                raw["rope_parameters"] = params
                hf = C(**raw)
                config = DeepSeekDecoderConfig.from_hf(hf.to_dict())
                rope = R(hf)
                for rate in (1, 2):
                    positions = torch.arange(16)[None] * rate
                    with torch.no_grad():
                        kwargs = {"layer_type": "compress"} if family == "deepseek_v4" else {}
                        cos, sin = rope(torch.zeros(1, 16, 16), positions, **kwargs)
                    tables = _rotary_tables(config, config.rope_compress if family == "deepseek_v4" else config.rope_main,
                        interleaved=family == "deepseek_v4", rate=rate)
                    for actual, expected in zip(tables, (cos.numpy()[0], sin.numpy()[0])):
                        np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-7)

    @unittest.skipUnless(HAVE_HF, "independent reference needs Transformers and torch")
    def test_native_mla_dense_routing_bias_and_rotary_layout(self):
        variants = [
            (DeepseekV2Config, DeepseekV2ForCausalLM, tiny_deepseek_config("deepseek_v2")),
            (DeepseekV2Config, DeepseekV2ForCausalLM, tiny_deepseek_config("deepseek_v2", q_lora_rank=None, attention_bias=True,
                topk_method="group_limited_greedy")),
            (DeepseekV3Config, DeepseekV3ForCausalLM, tiny_deepseek_config("deepseek_v3")),
            (DeepseekV3Config, DeepseekV3ForCausalLM, tiny_deepseek_config("deepseek_v3", rope_interleave=False,
                rms_norm_eps=0.001,
                rope_parameters={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8,
                    "beta_fast": 8, "beta_slow": 1, "mscale": 0.7, "mscale_all_dim": 0.2})),
        ]
        for C, M, raw in variants:
            self._compare_model(C, M, raw, sequences=([0], [0, 1, 2, 3, 4], [2, 4, 7, 11]))

    @unittest.skipUnless(HAVE_HF, "independent reference needs Transformers and torch")
    def test_native_v4_mhc_hca_csa_hash_and_grouped_inverse_rope(self):
        for kind in ("default", "yarn"):
            params = {"rope_type": kind}
            if kind == "yarn":
                params.update(factor=4.0, original_max_position_embeddings=8, beta_fast=8, beta_slow=1)
            raw = tiny_deepseek_config("deepseek_v4", rope_parameters=params)
            # Seed745 is preserved in all-v3.log: exact zero index scores made
            # unspecified torch.topk choose1 while the portable policy chose0.
            # Seed746 supplies non-tied top scores (minimum gap4.038e-5).
            self._compare_model(DeepseekV4Config, DeepseekV4ForCausalLM, raw,
                                sequences=([0], list(range(5)), list(range(9)), list(range(2, 11))), seed=746)

    def test_native_compressed_index_ties_choose_lowest_bank_index(self):
        b = GraphBuilder(2, ["out"], inputs={
            "position_ids": {"shape": [2], "dtype": "int32"},
            "attention_mask": {"shape": [1, 2], "dtype": "int32"}})
        for name, value in {
            "q": np.zeros((1, 2, 1, 2), np.float32),
            "k": np.zeros((1, 2, 1, 2), np.float32),
            "v": np.zeros((1, 2, 1, 2), np.float32),
            "bank": np.array([[[2, 5], [7, 11]]], np.float32),
            "iq": np.zeros((1, 2, 1, 2), np.float32),
            "ik": np.zeros((1, 2, 2), np.float32),
            "iw": np.ones((1, 2, 1), np.float32),
        }.items(): b.add(name, value)
        b.tensor("out", "SparseAttention", {"q": "q", "k": "k", "v": "v",
            "compressed_kv": "bank", "mask": "attention_mask", "position_ids": "position_ids",
            "index_q": "iq", "index_k": "ik", "index_weights": "iw"}, [1, 2, 1, 2],
            {"causal": True, "scale": 1.0, "compress_rate": 1, "top_k": 1})
        graph, weights = b.finish()
        ir = import_runtime_package(graph, weights).bind_shape_profile({"S": 2})
        inputs = {"position_ids": np.array([0, 1], np.int32), "attention_mask": np.zeros((1, 2), np.int32)}
        root = Path(__file__).resolve().parents[2]
        for profile in ("volvoxai", "volvoxai-lite"):
            with self.subTest(profile=profile), patch.dict(os.environ,
                    {"VOLVOXAI_NATIVE_RUNNER": str(root / "native" / profile)}):
                runner = NativeGraphRunner(ir, weights)
                actual = runner.run(runner.bind(inputs)).outputs["out"]
                np.testing.assert_array_equal(actual, np.array([[[[2, 5]], [[2, 5]]]], np.float32))

    def _compare_model(self, C, M, raw, sequences, seed=745):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        hf = C(**raw)
        hf._attn_implementation = "eager"
        model = M(hf).float().eval()
        # Nonzero biases and hash IDs make those contracts observable.
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name.endswith(".bias"):
                    value.copy_(torch.linspace(-0.02, 0.03, value.numel()).reshape(value.shape))
            for name, value in model.named_buffers():
                if name.endswith(".gate.tid2eid"):
                    value.copy_(torch.arange(value.numel()).reshape(value.shape) % hf.n_routed_experts)
        config = DeepSeekDecoderConfig.from_hf(hf.to_dict())
        graph, weights = build_deepseek_decoder_graph(config,
            {name: tensor.detach().numpy() for name, tensor in model.state_dict().items()}, capture_kv=True)
        ir = import_runtime_package(graph, weights)
        normalized_inputs = {}
        hooks = []
        for layer, block in enumerate(model.model.layers):
            def capture(module, args, output, layer=layer):
                normalized_inputs[layer] = output
            hooks.append(block.input_layernorm.register_forward_hook(capture))
        root = Path(__file__).resolve().parents[2]
        for positions in sequences:
            length = len(positions)
            inputs = {"token_ids": ((np.arange(length)[None] + 3) % 32).astype(np.int32),
                "position_ids": np.array(positions, np.int32),
                "attention_mask": np.ones((1, length), np.int32)}
            if length > 1:
                inputs["attention_mask"][0, 1] = 0
            with torch.inference_mode():
                expected = model(input_ids=torch.tensor(inputs["token_ids"]).long(),
                    position_ids=torch.tensor(inputs["position_ids"][None]).long(),
                    attention_mask=torch.tensor(inputs["attention_mask"]).long(), use_cache=False).logits.numpy()
                expected_kv = {}
                for layer, block in enumerate(model.model.layers):
                    attention = block.self_attn
                    x = normalized_inputs[layer]
                    ids = torch.tensor(inputs["position_ids"][None]).long()
                    if config.family == "deepseek_v4":
                        from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb
                        which = "main" if config.rates[layer] == 0 else "compress"
                        cos, sin = model.model.rotary_emb(x, ids, layer_type=which)
                        kv = attention.kv_norm(attention.kv_proj(x)).reshape(1, length, 1, config.head_dim).transpose(1, 2)
                        kv = apply_rotary_pos_emb(kv, cos, sin).transpose(1, 2)
                        expected_kv[layer] = (kv.numpy(), kv.numpy())
                    else:
                        latent = attention.kv_a_proj_with_mqa(x)
                        latent, kr = torch.split(latent, [config.kv_lora_rank, config.rotary_dim], dim=-1)
                        expanded = attention.kv_b_proj(attention.kv_a_layernorm(latent)).reshape(
                            1, length, config.heads, config.nope_dim + config.value_dim)
                        kn, value = torch.split(expanded, [config.nope_dim, config.value_dim], dim=-1)
                        kr = kr.reshape(1, 1, length, config.rotary_dim)
                        embeddings = model.model.rotary_emb(x, ids)
                        if config.family == "deepseek_v2":
                            from transformers.models.deepseek_v2.modeling_deepseek_v2 import apply_rotary_emb
                            _, kr = apply_rotary_emb(kr, kr, embeddings)
                        else:
                            from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
                                apply_rotary_pos_emb, apply_rotary_pos_emb_interleave)
                            apply = apply_rotary_pos_emb_interleave if config.interleaved else apply_rotary_pos_emb
                            _, kr = apply(kr, kr, *embeddings)
                        key = torch.cat((kn, kr.transpose(1, 2).expand(1, length, config.heads, config.rotary_dim)), -1)
                        expected_kv[layer] = (key.numpy(), value.numpy())
            for profile in ("volvoxai", "volvoxai-lite"):
                with self.subTest(family=config.family, variant=raw, positions=positions, profile=profile), \
                        patch.dict(os.environ, {"VOLVOXAI_NATIVE_RUNNER": str(root / "native" / profile)}):
                    runner = NativeGraphRunner(ir.bind_shape_profile({"S": length}), weights)
                    actual = runner.run(runner.bind(inputs)).outputs
                    np.testing.assert_allclose(actual["logits"], expected, atol=1e-5, rtol=1e-5)
                    np.testing.assert_array_equal(actual["logits"].argmax(-1), expected.argmax(-1))
                    for layer, pair in expected_kv.items():
                        for port, expected_tensor in zip(("k_rope", "v"), pair):
                            np.testing.assert_allclose(actual[f"layer.{layer}.{port}"], expected_tensor, atol=1e-5, rtol=1e-5)
        for hook in hooks:
            hook.remove()


class Qwen4AdmissionTests(unittest.TestCase):
    def test_real_topology_and_hashed_metadata(self):
        config=ExtendedDecoderConfig.from_hf(tiny_qwen4(),max_context=16)
        self.assertEqual(config.family,"qwen4_exp")
        self.assertEqual(config.ple_layers,(0,))
        multipliers,sizes,offsets,rows=config.ngram_tables(0)
        self.assertTrue(np.all(multipliers>0));self.assertTrue(np.all(multipliers%2==1))
        self.assertEqual(sizes.tolist(),[5,7,11,13]);self.assertEqual(offsets.tolist(),[0,5,12,23])
        self.assertEqual(rows,36)
        for changes in ({"indexer_kv_heads":2},{"indexer_budget":3},{"hc_count":1},
                {"ple_layer_ids":[2]},{"eos_token_id":24},{"ple_embed_dim":9}):
            with self.subTest(changes=changes),self.assertRaises(ExporterError):
                ExtendedDecoderConfig.from_hf({**tiny_qwen4(),**changes})

    def test_incremental_contract_keeps_private_recurrent_and_lexical_history(self):
        config=ExtendedDecoderConfig.from_hf(tiny_qwen4(),max_context=16)
        rng=np.random.default_rng(711)
        source={name:(rng.standard_normal(shape)*.1).astype(np.float32)
            for name,shape in config.weight_shapes().items()}
        reference,reference_weights=build_extended_decoder_graph(config,source,generate=True)
        graph,weights=build_extended_decoder_graph(config,source,cached=True)
        self.assertEqual(graph["outputs"],["logits","generated"])
        self.assertEqual([node["id"] for node in graph["nodes"]].count("generated"),1)
        self.assertEqual(graph["inputs"]["active_lengths"],{"shape":[1],"dtype":"int32"})
        self.assertEqual(set(graph["inputs"]),{"token_ids","position_ids","attention_mask","active_lengths"})
        execution=graph["execution"]
        self.assertNotIn("full_prefix",execution["generation"])
        self.assertEqual(execution["generation"]["incremental_controls"],{"position_input":"position_ids"})
        self.assertEqual(execution["incremental"]["update_inputs"],["token_ids","position_ids","attention_mask"])
        self.assertEqual(execution["invariant_inputs"],["active_lengths"])
        recurrent=[node for node in graph["nodes"] if node["opType"]=="GatedDeltaNet"]
        self.assertEqual(len(recurrent),1)
        self.assertNotIn("initial_state",recurrent[0]["inputs"])
        self.assertEqual(recurrent[0]["outputs"]["state"],
            {"tensor":"layer.0.delta_state","shape":[1,2,4,4],"dtype":"float32"})
        self.assertNotIn("layer.0.delta_state",graph["outputs"])
        # QSA keeps raw index-key history, and PLE keeps token/ngram and
        # dilated causal convolution history in the same retained timeline.
        indexer=next(node for node in graph["nodes"] if node["opType"]=="QSAIndexer")
        self.assertEqual(indexer["outputs"]["out"]["shape"],[1,"S","S"])
        self.assertTrue(indexer["params"]["causal"])
        self.assertEqual(len([node for node in graph["nodes"] if node["opType"]=="NGramIds"]),1)
        ple_conv=next(node for node in graph["nodes"] if node["id"]=="layer.0.ple.conv")
        self.assertEqual(ple_conv["params"]["dilation"],[1,3])
        self.assertEqual(ple_conv["params"]["pads"],[0,6,0,0])
        normalized=[]
        for node in graph["nodes"]:
            if node["opType"]=="GatedDeltaNet":
                node={**node,"outputs":{"out":node["outputs"]["out"]}}
            normalized.append(node)
        self.assertEqual(normalized,reference["nodes"])
        self.assertEqual(set(weights),set(reference_weights))
        for name in weights:
            np.testing.assert_array_equal(weights[name],reference_weights[name])
        ir=import_runtime_package(graph,weights)
        for capacity in (1,3,16):
            ir.bind_shape_profile({"S":capacity}).verify_concrete_bound()
        for options in ({"cached":True,"capture_kv":True},{"cached":1}):
            with self.assertRaises(ExporterError):
                build_extended_decoder_graph(config,source,**options)


class Qwen4NativeReferenceTests(unittest.TestCase):
    def test_reference_compatibility_restores_bindings_after_failure(self):
        from transformers import AutoModel
        import transformers.configuration_utils as configuration
        import transformers.integrations as integrations
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.generic as generic
        import transformers.vision_utils as vision

        absent = object()
        attributes = [(configuration, "ALLOWED_LAYER_TYPES"),
            (integrations, "use_kernel_func_from_hub_with_fallback"),
            (generic, "get_max_seqlen"), (vision, "get_vision_attention_seqlens"),
            (vision, "get_vision_position_ids"),
            (vision, "get_vision_interpolation_indices_and_weights")]
        before = [getattr(module, name, absent) for module, name in attributes]
        prefix = "transformers.models.qwen4_exp"
        modules = {name: value for name, value in sys.modules.items() if name.startswith(prefix)}
        model_mapping = dict(AutoModel._model_mapping._extra_content)
        rope_mapping = dict(modeling_utils.ROPE_INIT_FUNCTIONS)
        with self.assertRaisesRegex(RuntimeError, "fixture owner failed"):
            with qwen4_reference() as (config_cls, _):
                self.assertEqual(config_cls(**tiny_qwen4()).model_type, "qwen4_exp_text")
                raise RuntimeError("fixture owner failed")
        for (module, name), value in zip(attributes, before):
            self.assertIs(getattr(module, name, absent), value)
        self.assertEqual({name: value for name, value in sys.modules.items() if name.startswith(prefix)}, modules)
        self.assertEqual(AutoModel._model_mapping._extra_content, model_mapping)
        self.assertEqual(modeling_utils.ROPE_INIT_FUNCTIONS, rope_mapping)

    def test_official_qsa_and_ple_past_rows_are_prefix_stable(self):
        import torch
        reference_scope = contextlib.ExitStack()
        self.addCleanup(reference_scope.close)
        config_cls,model_cls=reference_scope.enter_context(qwen4_reference())
        hf=config_cls(**tiny_qwen4())
        hf._attn_implementation="eager"
        torch.manual_seed(4819);torch.set_num_threads(1)
        model=model_cls(hf).float().eval()
        tokens=torch.tensor([[2,3,23,4,5,6,7,8]])
        # Position IDs only rotate already-causal QSA keys. Unlike V4 bank
        # visibility, this selector remains causal with irregular positions.
        positions=torch.tensor([[0,3,15,7,4,9,12,13]])
        mask=torch.tensor([[0,1,1,1,1,1,1,1]])
        with torch.inference_mode():
            outputs=[model(input_ids=tokens[:,:length],position_ids=positions[:,:length],
                attention_mask=mask[:,:length],use_cache=False).logits.numpy()
                for length in (3,4,5,8)]
        for output in outputs[:-1]:
            np.testing.assert_allclose(output[:,1:3],outputs[-1][:,1:3],atol=2e-6,rtol=2e-5)

    def _compare(self,sharded):
        import torch
        reference_scope = contextlib.ExitStack()
        self.addCleanup(reference_scope.close)
        config_cls,model_cls=reference_scope.enter_context(qwen4_reference())
        torch.manual_seed(4819);torch.set_num_threads(1)
        hf=config_cls(**tiny_qwen4())
        hf._attn_implementation="eager"
        model=model_cls(hf).float().eval()
        # Exercise learned (rather than zero-initialized) grouped normalization.
        with torch.no_grad():
            for name,value in model.named_parameters():
                if name.endswith(("hc_norm.weight","norm_key.weight","norm_query.weight","norm_conv.weight")):
                    value.copy_(torch.linspace(-.05,.07,value.numel()))
                elif name.endswith(".ple.conv1d.weight"):
                    value.copy_(torch.linspace(-.04,.06,value.numel()).reshape(value.shape))
        source={name:value.detach().numpy() for name,value in model.state_dict().items()}
        if sharded:
            name="model.layers.0.ple.ple_embedding.ngram_embedding.weight"
            bank=source.pop(name)
            for shard,value in enumerate(np.array_split(bank,4,axis=0)):
                source[name[:-len("weight")]+f"shard_{shard}.weight"]=value
        config=ExtendedDecoderConfig.from_hf(hf.to_dict(),max_context=16)
        graph,weights=build_extended_decoder_graph(config,source)
        ir=import_runtime_package(graph,weights)
        root=Path(__file__).resolve().parents[2]
        for tokens,positions,mask in (([2],[15],[1]),([2,3,4],[0,3,15],[1,1,1]),
                ([2,3,23,4,5,6,7],[0,1,2,3,4,5,6],[1,1,1,1,1,1,1]),
                ([2,3,4,5,6],[0,1,2,3,4],[0,1,1,1,1])):
            length=len(tokens)
            inputs={"token_ids":np.asarray([tokens],np.int32),"position_ids":np.asarray(positions,np.int32),"attention_mask":np.asarray([mask],np.int32)}
            with torch.inference_mode():
                expected=model(input_ids=torch.from_numpy(inputs["token_ids"]).long(),
                    position_ids=torch.from_numpy(inputs["position_ids"][None]).long(),
                    attention_mask=torch.from_numpy(inputs["attention_mask"]).long(),use_cache=False).logits.numpy()
            for profile in ("volvoxai","volvoxai-lite"):
                with self.subTest(sharded=sharded,tokens=tokens,profile=profile),patch.dict(os.environ,
                        {"VOLVOXAI_NATIVE_RUNNER":str(root/"native"/profile)}):
                    runner=NativeGraphRunner(ir.bind_shape_profile({"S":length}),weights)
                    actual=runner.run(runner.bind(inputs)).outputs["logits"]
                    selected=inputs["attention_mask"].astype(bool)
                    np.testing.assert_allclose(actual[selected],expected[selected],atol=2e-5,rtol=2e-5)
                    np.testing.assert_array_equal(actual[selected].argmax(-1),expected[selected].argmax(-1))

    def test_unsplit_hc_qsa_ple_and_eos(self):self._compare(False)
    def test_official_sharded_ngram_layout(self):self._compare(True)

    def test_real_qwen4_image_conditioned_hyperconnection_logits(self):
        import torch
        from tools.exporter.hf_vision import QwenVisionConfig,build_qwen_vision_graph
        reference_scope = contextlib.ExitStack()
        self.addCleanup(reference_scope.close)
        reference_scope.enter_context(qwen4_reference())
        module=sys.modules["transformers.models.qwen4_exp.modeling_qwen4_exp"]
        configs=sys.modules["transformers.models.qwen4_exp.configuration_qwen4_exp"]
        text=tiny_qwen4();text["head_dim"]=16
        text["rope_parameters"]["mrope_section"]=[1,1,2]
        vision={**tiny_vision_config(),"model_type":"qwen4_exp_vision",
            "rope_parameters":{"rope_type":"axial","rope_theta":10000.}}
        hf=configs.Qwen4ExpConfig(text_config=text,vision_config=vision,
            image_token_id=23,video_token_id=22,vision_start_token_id=21,vision_end_token_id=20)
        hf._attn_implementation="eager"
        torch.manual_seed(4815);torch.set_num_threads(1)
        model=module.Qwen4ExpForConditionalGeneration(hf).float().eval()
        source={name:value.detach().numpy() for name,value in model.state_dict().items()}
        config=ExtendedDecoderConfig.from_hf(hf.to_dict(),max_context=16)
        vg,vw=build_qwen_vision_graph(QwenVisionConfig.from_hf(hf.vision_config.to_dict()),source,grid_thw=[[1,2,4]])
        graph,weights=build_extended_decoder_graph(config,source,image_features_count=2)
        pixels=np.random.default_rng(4158).standard_normal((8,24)).astype(np.float32)
        axes=np.array([[0,1,1,3],[0,1,1,3],[0,1,2,3]],np.int32)
        inputs={"token_ids":np.array([[1,23,23,2]],np.int32),"position_ids":np.arange(4,dtype=np.int32),
            "attention_mask":np.ones((1,4),np.int32),"mrope_position_ids":axes,
            "image_feature_ids":np.array([0,0,1,0],np.int32),"image_token_mask":np.array([[0,1,1,0]],np.int32)}
        with torch.inference_mode():
            expected_features=model.model.visual(torch.from_numpy(pixels),grid_thw=torch.tensor([[1,2,4]])).pooler_output.numpy()
            expected=model(input_ids=torch.from_numpy(inputs["token_ids"]).long(),
                attention_mask=torch.from_numpy(inputs["attention_mask"]).long(),
                position_ids=torch.from_numpy(np.concatenate((inputs["position_ids"][None],axes),axis=0)[:,None]).long(),
                pixel_values=torch.from_numpy(pixels),image_grid_thw=torch.tensor([[1,2,4]]),use_cache=False).logits.numpy()
        for profile in ("volvoxai","volvoxai-lite"):
            with self.subTest(profile=profile):
                features=execute(profile,vg,vw,{"pixel_patches":pixels})["image_features"]
                np.testing.assert_allclose(features,expected_features,atol=2e-5,rtol=2e-5)
                actual=execute(profile,graph,weights,{**inputs,"image_features":features})["logits"]
                np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
                np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))


if __name__ == "__main__":
    unittest.main()
