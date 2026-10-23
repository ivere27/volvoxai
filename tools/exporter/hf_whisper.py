"""Whisper F32 encoder and whole-prefix decoder as ordinary tensor graphs.

Applications supply log-mel features and explicit decoder prompt IDs. Audio
decoding, language/task prompts, timestamp rules and logits processors remain
caller policy. The separate encoder output is immutable decoder conditioning;
Generate reevaluates the decoder prefix without running the audio encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .execution_contract import full_prefix_decoder_execution
from .hf_graph import (GraphBuilder, positive_integer as _positive_integer,
    read_checkpoint_storage, write_hf_package)


def _fail(message):
    raise ExporterError(Diagnostic("VXHF_WHISPER", message, "hf-whisper"))


@dataclass(frozen=True)
class WhisperConfig:
    hidden: int
    vocabulary: int
    mel_bins: int
    source_positions: int
    target_positions: int
    max_context: int
    encoder_layers: int
    decoder_layers: int
    encoder_heads: int
    decoder_heads: int
    encoder_intermediate: int
    decoder_intermediate: int
    tied: bool

    @classmethod
    def from_hf(cls, source: Mapping, *, max_context=None):
        if not isinstance(source, Mapping) or source.get("model_type") != "whisper":
            _fail("configuration must identify Whisper")
        if source.get("architectures") not in (None, ["WhisperForConditionalGeneration"]):
            _fail("architecture must identify WhisperForConditionalGeneration")
        if source.get("is_encoder_decoder", True) is not True:
            _fail("Whisper requires an encoder-decoder configuration")
        if source.get("quantization_config") not in (None, {}):
            _fail("quantized source storage is unsupported")
        if source.get("activation_function", "gelu") != "gelu":
            _fail("Whisper supports exact GELU only")
        if source.get("scale_embedding", False) is not False:
            _fail("scaled embeddings are outside this Whisper import contract")
        hidden = _positive_integer(source.get("d_model", 384), "d_model")
        heads = [_positive_integer(source.get(name, 6), name)
                 for name in ("encoder_attention_heads", "decoder_attention_heads")]
        if any(hidden % n for n in heads):
            _fail("attention heads must divide d_model")
        target = _positive_integer(source.get("max_target_positions", 448), "max_target_positions")
        context = target if max_context is None else _positive_integer(max_context, "max_context")
        if context > target:
            _fail("max_context exceeds the learned decoder positions")
        tied = source.get("tie_word_embeddings", True)
        if not isinstance(tied, bool):
            _fail("tie_word_embeddings must be boolean")
        return cls(hidden, _positive_integer(source.get("vocab_size", 51865), "vocab_size"),
            _positive_integer(source.get("num_mel_bins", 80), "num_mel_bins"),
            _positive_integer(source.get("max_source_positions", 1500), "max_source_positions"),
            target, context, _positive_integer(source.get("encoder_layers", 4), "encoder_layers"),
            _positive_integer(source.get("decoder_layers", 4), "decoder_layers"), *heads,
            _positive_integer(source.get("encoder_ffn_dim", 1536), "encoder_ffn_dim"),
            _positive_integer(source.get("decoder_ffn_dim", 1536), "decoder_ffn_dim"), tied)

    def weight_shapes(self):
        h = self.hidden
        result = {"model.encoder.conv1.weight": (h, self.mel_bins, 3),
            "model.encoder.conv1.bias": (h,), "model.encoder.conv2.weight": (h, h, 3),
            "model.encoder.conv2.bias": (h,),
            "model.encoder.embed_positions.weight": (self.source_positions, h),
            "model.decoder.embed_positions.weight": (self.target_positions, h),
            "model.decoder.embed_tokens.weight": (self.vocabulary, h)}
        for side, layers, intermediate in (("encoder", self.encoder_layers, self.encoder_intermediate),
                ("decoder", self.decoder_layers, self.decoder_intermediate)):
            for suffix in ("weight", "bias"):
                result[f"model.{side}.layer_norm.{suffix}"] = (h,)
            for layer in range(layers):
                prefix = f"model.{side}.layers.{layer}."
                for norm in ("self_attn_layer_norm", "final_layer_norm", *
                        (("encoder_attn_layer_norm",) if side == "decoder" else ())):
                    result[prefix + norm + ".weight"] = result[prefix + norm + ".bias"] = (h,)
                for attention in ("self_attn", *(("encoder_attn",) if side == "decoder" else ())):
                    for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                        p = prefix + attention + "." + projection
                        result[p + ".weight"] = (h, h)
                        if projection != "k_proj": result[p + ".bias"] = (h,)
                result[prefix + "fc1.weight"] = (intermediate, h)
                result[prefix + "fc1.bias"] = (intermediate,)
                result[prefix + "fc2.weight"] = (h, intermediate)
                result[prefix + "fc2.bias"] = (h,)
        if not self.tied: result["proj_out.weight"] = (self.vocabulary, h)
        return result


def _weights(config, source):
    expected = config.weight_shapes()
    names = set(source)
    if config.tied and "proj_out.weight" in names:
        if not np.array_equal(source["proj_out.weight"], source.get("model.decoder.embed_tokens.weight")):
            _fail("tied output projection differs from decoder embeddings")
        names.remove("proj_out.weight")
    if names != set(expected):
        _fail(f"weight names differ: missing={sorted(set(expected)-names)}, unexpected={sorted(names-set(expected))}")
    result = {}
    for name, shape in expected.items():
        value = np.asarray(source[name])
        if value.shape != shape or value.dtype not in (np.dtype("float32"), np.dtype("float16")) or not np.isfinite(value).all():
            _fail(f"weight {name} must contain finite floating shape {shape}")
        result[name] = np.ascontiguousarray(value, dtype=np.float32)
    return result


def _norm(b, name, value, prefix, shape):
    return b.tensor(name, "LayerNorm", {"input": value, "weight": prefix + ".weight",
        "bias": prefix + ".bias"}, shape, {"eps": 1e-5})


def _attention(b, name, value, memory, prefix, shape, keys, heads, *, causal=False, mask=None):
    h = shape[-1]
    q = b.linear(name + ".q", value, prefix + ".q_proj.weight", h,
        prefix + ".q_proj.bias", shape)
    # Whisper scales Q before the dot product. Keeping this operation separate
    # preserves HF's F32 ordering rather than scaling the completed scores.
    scale = name + ".scale"
    b.add(scale, np.array([1 / math.sqrt(h // heads)], np.float32))
    scale = b.tensor(name + ".scale_rows", "Expand", {"input": scale}, shape, {"shape": shape})
    q = b.tensor(name + ".scaled_q", "Mul", {"a": q, "b": scale}, shape)
    k = b.linear(name + ".k", memory, prefix + ".k_proj.weight", h, shape=[1, keys, h])
    v = b.linear(name + ".v", memory, prefix + ".v_proj.weight", h,
        prefix + ".v_proj.bias", [1, keys, h])
    inputs = {"q": q, "k": k, "v": v}
    if mask is not None: inputs["mask"] = mask
    attended = b.tensor(name + ".attention", "CrossSDPA", inputs, shape,
        {"heads": heads, "kv_heads": heads, "causal": causal, "scale": 1.0})
    return b.linear(name + ".output", attended, prefix + ".out_proj.weight", h,
        prefix + ".out_proj.bias", shape)


def _mlp(b, name, value, prefix, shape, intermediate):
    normalized = _norm(b, name + ".norm", value, prefix + "final_layer_norm", shape)
    up_shape = [*shape[:-1], intermediate]
    up = b.linear(name + ".up", normalized, prefix + "fc1.weight", intermediate,
        prefix + "fc1.bias", up_shape)
    up = b.tensor(name + ".gelu", "GELU", {"input": up}, up_shape, {"approximate": "none"})
    down = b.linear(name + ".down", up, prefix + "fc2.weight", shape[-1], prefix + "fc2.bias", shape)
    return b.tensor(name + ".residual", "Add", {"a": value, "b": down}, shape)


def build_whisper_encoder_graph(config, source_weights, *, output_name="encoder_memory"):
    weights = _weights(config, source_weights)
    h, n, bins = config.hidden, config.source_positions, config.mel_bins
    b = GraphBuilder(1, [output_name], {"input_features": {"shape": [1, bins, n * 2], "dtype": "float32"}})
    b.graph["dimensions"] = {}
    b.weights.update(weights)
    value = b.tensor("audio.features_rows", "Transpose", {"input": "input_features"},
        [1, n * 2, bins], {"perm": [0, 2, 1]})
    value = b.reshape("audio.features_nhwc", value, [1, 1, n * 2, bins])
    for index, width, stride, channels in ((1, n * 2, 1, bins), (2, n, 2, h)):
        prefix, name = f"model.encoder.conv{index}", f"audio.conv{index}"
        kernel = weights[prefix + ".weight"].transpose(2, 1, 0)[None]
        b.add(name + ".kernel", kernel)
        value = b.tensor(name, "Conv2D", {"input": value, "weight": name + ".kernel",
            "bias": prefix + ".bias"}, [1, 1, width, h],
            {"stride": [1, stride], "pads": [0, 1, 0, 1], "data_layout": "NHWC", "weight_layout": "HWIO"})
        value = b.tensor(name + ".gelu", "GELU", {"input": value}, [1, 1, width, h], {"approximate": "none"})
    shape = [1, n, h]
    value = b.reshape("audio.encoder_rows", value, shape)
    positions = b.reshape("audio.encoder_positions", "model.encoder.embed_positions.weight", shape)
    value = b.tensor("audio.positioned", "Add", {"a": value, "b": positions}, shape)
    for layer in range(config.encoder_layers):
        prefix, name = f"model.encoder.layers.{layer}.", f"audio.encoder.{layer}"
        normalized = _norm(b, name + ".norm", value, prefix + "self_attn_layer_norm", shape)
        attention = _attention(b, name, normalized, normalized, prefix + "self_attn", shape, n, config.encoder_heads)
        value = b.tensor(name + ".residual", "Add", {"a": value, "b": attention}, shape)
        value = _mlp(b, name + ".mlp", value, prefix, shape, config.encoder_intermediate)
    _norm(b, output_name, value, "model.encoder.layer_norm", shape)
    return b.finish()


def build_whisper_decoder_graph(config, source_weights, *, output_name="logits", generate=False,
        suppressed_ids=()):
    weights = _weights(config, source_weights)
    h, s, n = config.hidden, "S", config.source_positions
    b = GraphBuilder(config.max_context, [output_name])
    b.graph["inputs"]["encoder_memory"] = {"shape": [1, n, h], "dtype": "float32"}
    b.weights.update(weights)
    shape = [1, s, h]
    tokens = b.tensor("audio.tokens", "Embedding", {"input": "token_ids", "weight": "model.decoder.embed_tokens.weight"}, shape)
    positions = b.tensor("audio.positions", "Embedding", {"input": "position_ids", "weight": "model.decoder.embed_positions.weight"}, [s, h])
    positions = b.reshape("audio.position_rows", positions, shape)
    value = b.tensor("audio.decoder_positioned", "Add", {"a": tokens, "b": positions}, shape)
    for layer in range(config.decoder_layers):
        prefix, name = f"model.decoder.layers.{layer}.", f"audio.decoder.{layer}"
        normalized = _norm(b, name + ".norm", value, prefix + "self_attn_layer_norm", shape)
        attention = _attention(b, name + ".self", normalized, normalized, prefix + "self_attn", shape, s,
            config.decoder_heads, causal=True, mask="attention_mask")
        value = b.tensor(name + ".self_residual", "Add", {"a": value, "b": attention}, shape)
        normalized = _norm(b, name + ".cross_norm", value, prefix + "encoder_attn_layer_norm", shape)
        attention = _attention(b, name + ".cross", normalized, "encoder_memory", prefix + "encoder_attn", shape, n, config.decoder_heads)
        value = b.tensor(name + ".cross_residual", "Add", {"a": value, "b": attention}, shape)
        value = _mlp(b, name + ".mlp", value, prefix, shape, config.decoder_intermediate)
    value = _norm(b, "audio.decoder_norm", value, "model.decoder.layer_norm", shape)
    b.linear(output_name, value, "model.decoder.embed_tokens.weight" if config.tied else "proj_out.weight",
        config.vocabulary, shape=[1, s, config.vocabulary])
    suppressed_ids = tuple(suppressed_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= config.vocabulary
            for value in suppressed_ids) or len(set(suppressed_ids)) != len(suppressed_ids):
        _fail("caller suppression IDs must be unique integers in the declared vocabulary")
    selection = output_name
    if suppressed_ids:
        if not generate: _fail("caller suppression requires generation")
        keep = np.ones((1, 1, config.vocabulary), np.int32)
        keep[0, 0, list(suppressed_ids)] = 0
        b.add("audio.caller_selection_keep", keep)
        b.add("audio.caller_selection_floor", np.array([-np.finfo(np.float32).max], np.float32))
        mask_shape = [1, s, config.vocabulary]
        mask = b.tensor("audio.selection_keep", "Expand", {"input": "audio.caller_selection_keep"}, mask_shape,
            {"shape": mask_shape}, "int32")
        floor = b.tensor("audio.selection_floor", "Expand", {"input": "audio.caller_selection_floor"}, mask_shape,
            {"shape": mask_shape})
        selection = b.tensor("audio.selection_logits", "Where", {"condition": mask, "a": output_name, "b": floor}, mask_shape)
        b.graph["outputs"].append(selection)
    if not generate: return b.finish()
    if output_name == "generated": _fail("logits output conflicts with generated IDs")
    b.tensor("generated", "ArgMax", {"input": selection}, [1, s], {"axis": -1, "keepdims": False}, "int32")
    b.graph["outputs"].append("generated")
    return b.finish(full_prefix_decoder_execution(b.graph["inputs"], b.graph["outputs"],
        token_input="token_ids", token_output="generated", keep_input="attention_mask",
        position_input="position_ids", logits_output=selection, id_domain_size=config.vocabulary,
        invariant_inputs=("encoder_memory",)))


def export_hf_whisper(directory, out_path, *, encoder=False, max_context=None, output_names=None,
        generate=False, report_callback=None, weight_precision=None, group_size=128):
    if encoder and (max_context is not None or generate):
        _fail("encoder export excludes decoder context and generation options")
    source, source_weights, source_dtypes = read_checkpoint_storage(directory, config_validator=WhisperConfig.from_hf)
    config = WhisperConfig.from_hf(source, max_context=max_context)
    if output_names is not None and len(output_names) != 1:
        _fail("Whisper export declares one principal output")
    name = output_names[0] if output_names else "encoder_memory" if encoder else "logits"
    builder = build_whisper_encoder_graph if encoder else build_whisper_decoder_graph
    graph, weights = builder(config, source_weights, output_name=name, **({} if encoder else {"generate": generate}))
    graph = write_hf_package(out_path, graph, weights, weight_precision=weight_precision, group_size=group_size)
    if report_callback:
        report_callback({"features": {"hf_whisper": ["execution:F32", "batch:1",
            "scope:audio-encoder" if encoder else "scope:conditioned-decoder",
            "conditioning:encoder_memory", "policy:caller-features-prompt-and-logits-processing",
            "state:full-prefix-greedy" if generate else "state:stateless"],
            "source_storage": sorted(set(source_dtypes.values()))}})
    return graph
