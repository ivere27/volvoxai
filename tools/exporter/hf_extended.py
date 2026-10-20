"""Inference graph import for Qwen MoE/hybrid decoders and Llama 4.

The exporter lowers checkpoint layouts into the common graph vocabulary. It
does not carry a model executor, tokenizer, training dependency or a second
runtime contract. Whole-prefix evaluation starts from zero recurrent state.
Cached execution declares retained KV, convolution history and delta state;
the engine owns their initialization and single-row transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .execution_contract import full_prefix_decoder_execution, incremental_decoder_execution
from .hf_decoder import DecoderConfig
from .hf_graph import (GraphBuilder, positive_float as _positive_float, positive_integer as _positive_integer,
    read_config as _read_json, read_checkpoint_storage, write_hf_package)
from .quantization_storage import GRAPH_FORMAT
from .runtime_ir import import_runtime_package
from .runtime_names import is_runtime_graph_name


EXTENDED_MODEL_TYPES = frozenset({"qwen3_moe", "qwen3_next", "llama4", "llama4_text",
    "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text"})


def _fail(message: str, code: str = "VXHF_CONFIG") -> None:
    raise ExporterError(Diagnostic(code, message, "hf-extended"))


def _boolean(source: Mapping[str, Any], name: str, default: bool) -> bool:
    value = source.get(name, default)
    if not isinstance(value, bool):
        _fail(f"{name} must be boolean")
    return value


def _indices(source, name, layers, default):
    value = source.get(name)
    value = list(default) if value is None else value
    if not isinstance(value, (list, tuple)) or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < layers
            for index in value) or len(set(value)) != len(value):
        _fail(f"{name} must contain unique valid layer indices")
    return tuple(value)


@dataclass(frozen=True)
class ExtendedDecoderConfig:
    family: str
    decoder: DecoderConfig
    layer_types: tuple[str, ...]
    moe_layers: tuple[int, ...]
    experts: int = 0
    experts_per_token: int = 1
    expert_intermediate: int = 1
    shared_intermediate: int = 0
    normalize_routes: bool = True
    rotary_dim: int = 0
    norm_offset: bool = False
    attention_gate: str | None = None
    rope_layers: tuple[bool, ...] = ()
    qk_l2_norm: bool = False
    chunk_size: int | None = None
    temperature_tuning: bool = False
    temperature_floor: int = 8192
    temperature_scale: float = .1
    linear_key_heads: int = 1
    linear_value_heads: int = 1
    linear_key_dim: int = 1
    linear_value_dim: int = 1
    linear_kernel: int = 1
    vision_config: Mapping[str, Any] | None = None
    source_model_type: str = ""
    mrope_sections: tuple[int, int, int] = (0, 0, 0)

    def __getattr__(self, name):
        # Shared decoder geometry remains a single checked representation.
        return getattr(self.decoder, name)

    @classmethod
    def from_hf(cls, source: Mapping[str, Any], *, max_context: int | None = None):
        original = source.get("model_type")
        if original in {"qwen4_exp", "qwen4_exp_text"}:
            from .hf_qwen4 import Qwen4DecoderConfig
            return Qwen4DecoderConfig.from_hf(source,max_context=max_context)
        if original not in EXTENDED_MODEL_TYPES:
            _fail(f"unsupported extended decoder model_type {original!r}")
        outer = original in {"llama4", "qwen3_5", "qwen3_5_moe"} and "text_config" in source
        text = source.get("text_config") if outer else source
        if not isinstance(text, Mapping):
            _fail("text_config must contain the decoder configuration")
        family = {"llama4_text": "llama4", "qwen3_5_text": "qwen3_5",
                  "qwen3_5_moe_text": "qwen3_5_moe"}.get(text.get("model_type", original), original)
        if family not in {"llama4", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_moe"}:
            _fail("text_config model_type does not identify a supported decoder")
        expected_inner = {"llama4": "llama4_text", "qwen3_5": "qwen3_5_text",
                          "qwen3_5_moe": "qwen3_5_moe_text"}.get(original)
        if outer and text.get("model_type", expected_inner) != expected_inner:
            _fail("outer and text_config model_type disagree")
        architectures = {
            "qwen3_moe": {"Qwen3MoeForCausalLM"}, "qwen3_next": {"Qwen3NextForCausalLM"},
            "llama4": {"Llama4ForConditionalGeneration", "Llama4ForCausalLM"},
            "qwen3_5": {"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"},
            "qwen3_5_moe": {"Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"},
        }[family]
        if source.get("architectures") is not None and (
                not isinstance(source["architectures"], list) or len(source["architectures"]) != 1 or
                source["architectures"][0] not in architectures):
            _fail(f"architectures must identify one of {sorted(architectures)}")
        for scope in (source, text):
            if scope.get("quantization_config") not in (None, {}):
                _fail("FP8/FP4 and other quantized source storage are not implemented; use F32/F16/BF16")
            for name in ("is_encoder_decoder", "use_sliding_window", "attn_logit_softcapping", "final_logit_softcapping"):
                if scope.get(name) not in (None, False, 0):
                    _fail(f"{name} is unsupported")
            if scope.get("pretraining_tp", 1) != 1:
                _fail("pretraining_tp other than 1 is unsupported")
        if text.get("hidden_act", "silu") not in {"silu", "swish"}:
            _fail("the decoder requires a SiLU gated MLP and convolution")
        layers = _positive_integer(text.get("num_hidden_layers"), "num_hidden_layers")
        hidden = _positive_integer(text.get("hidden_size"), "hidden_size")
        hybrid = family in {"qwen3_next", "qwen3_5", "qwen3_5_moe"}
        interval = _positive_integer(text.get("full_attention_interval", 4), "full_attention_interval")
        defaults = [("full_attention" if (index + 1) % interval == 0 else "linear_attention")
                    if hybrid else "full_attention" for index in range(layers)]
        types = text.get("layer_types", defaults)
        if not isinstance(types, (list, tuple)) or len(types) != layers or any(
                kind not in ({"full_attention", "linear_attention"} if hybrid else
                             {"full_attention", "chunked_attention"} if family == "llama4" else {"full_attention"})
                for kind in types):
            _fail("layer_types do not match the supported decoder attention architecture")
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        if not isinstance(rope, Mapping):
            _fail("rope_parameters/rope_scaling must be an object or null")
        if text.get("rope_parameters") and text.get("rope_scaling"):
            _fail("conflicting RoPE descriptions")
        allowed_rope = {"rope_type", "type", "rope_theta", "partial_rotary_factor", "mrope_interleaved", "mrope_section"}
        if set(rope) - allowed_rope or rope.get("rope_type", rope.get("type", "default")) != "default":
            _fail("only default bounded RoPE is supported for this architecture")
        if "type" in rope and "rope_type" in rope and rope["type"] != rope["rope_type"]:
            _fail("conflicting RoPE type values")
        head_dim = _positive_integer(text.get("head_dim", 128 if family in {"llama4", "qwen3_moe"} else 256), "head_dim")
        fraction = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", .25 if hybrid else 1.0))
        fraction = _positive_float(fraction, "partial_rotary_factor")
        rotary = int(head_dim * fraction)
        if fraction > 1 or rotary <= 0 or rotary % 2 or rotary > head_dim:
            _fail("partial_rotary_factor must produce a positive even rotary dimension")
        # Modern HF serializers retain the legacy top-level default even when
        # rope_parameters overrides it. The RoPE object is authoritative.
        if family.startswith("qwen3_5"):
            sections = rope.get("mrope_section", [11, 11, 10])
            if not isinstance(sections, (list, tuple)) or len(sections) != 3 or any(
                    not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in sections) or sum(sections) != rotary // 2:
                _fail("mrope_section must partition the rotary frequency dimension")
            _boolean(rope, "mrope_interleaved", True)
        elif "mrope_section" in rope or "mrope_interleaved" in rope:
            _fail("multimodal RoPE flags are only supported for Qwen3.5-compatible decoders")
        intermediate = (text.get("intermediate_size_mlp", 16384) if family == "llama4" else
            text.get("intermediate_size", text.get("moe_intermediate_size", hidden * 2)))
        sanitized = {"model_type": "llama" if family == "llama4" else "qwen3",
            "vocab_size": text.get("vocab_size"), "hidden_size": hidden, "intermediate_size": intermediate,
            "num_hidden_layers": layers, "num_attention_heads": text.get("num_attention_heads"),
            "num_key_value_heads": text.get("num_key_value_heads", 8 if family == "llama4" else 4 if family == "qwen3_moe" else 2),
            "head_dim": head_dim, "max_position_embeddings": text.get("max_position_embeddings"),
            "rms_norm_eps": text.get("rms_norm_eps", 1e-5 if family == "llama4" else 1e-6),
            "rope_theta": rope.get("rope_theta", text.get("rope_theta", 500000 if family == "llama4" else 1000000 if family == "qwen3_moe" else 10000000)),
            "tie_word_embeddings": text.get("tie_word_embeddings", source.get("tie_word_embeddings", False)),
            "attention_bias": text.get("attention_bias", False), "mlp_bias": False}
        base = DecoderConfig.from_hf(sanitized, max_context=max_context)
        if text.get("mlp_bias", False) is not False:
            _fail("extended decoder MLP projections are bias-free")
        step = _positive_integer(text.get("interleave_moe_layer_step" if family == "llama4" else "decoder_sparse_step", 1), "MoE layer interval")
        expert_count = text.get("num_local_experts" if family == "llama4" else "num_experts",
            text.get("num_local_experts", 128) if family == "qwen3_moe" else
            16 if family == "llama4" else 0 if family == "qwen3_5" else 256 if family == "qwen3_5_moe" else 512)
        if not isinstance(expert_count, int) or isinstance(expert_count, bool) or not 0 <= expert_count <= 2**31 - 1:
            _fail("expert count must be a nonnegative int32")
        top = _positive_integer(text.get("num_experts_per_tok", 1 if family == "llama4" else 8 if family != "qwen3_next" else 10), "num_experts_per_tok")
        if expert_count and top > expert_count:
            _fail("num_experts_per_tok exceeds expert count")
        if family == "qwen3_5" and expert_count:
            _fail("dense Qwen3.5 text architecture cannot declare routed experts")
        if family == "llama4":
            moe = _indices(text, "moe_layers", layers, range(step - 1, layers, step)) if expert_count else ()
        else:
            dense_only = _indices(text, "mlp_only_layers", layers, ())
            moe = tuple(i for i in range(layers) if expert_count and (i + 1) % step == 0 and i not in dense_only)
        shared = 0 if family == "qwen3_moe" else _positive_integer(text.get("intermediate_size", 8192), "intermediate_size") if family == "llama4" else (
            _positive_integer(text.get("shared_expert_intermediate_size", 512), "shared_expert_intermediate_size") if expert_count else 0)
        expert_width = _positive_integer(text.get("intermediate_size", 8192) if family == "llama4" else text.get("moe_intermediate_size", 768 if family == "qwen3_moe" else 512), "moe_intermediate_size")
        normalize = _boolean(text, "norm_topk_prob", False if family == "qwen3_moe" else True)
        if family == "qwen3_5_moe" and not normalize:
            _fail("Qwen3.5 MoE always normalizes selected routing scores")
        if family == "llama4":
            flags = text.get("no_rope_layers")
            no_rope_interval = _positive_integer(text.get("no_rope_layer_interval", 4), "no_rope_layer_interval")
            flags = [int((i + 1) % no_rope_interval != 0) for i in range(layers)] if flags is None else flags
            if not isinstance(flags, (list, tuple)) or len(flags) != layers or any(x not in (0, 1) for x in flags):
                _fail("no_rope_layers must contain one 0/1 flag per layer")
            rope_layers = tuple(bool(x) for x in flags)
            types = text.get("layer_types", ["chunked_attention" if flag else "full_attention" for flag in flags])
            chunk = text.get("attention_chunk_size", 8192)
            chunk = None if chunk is None else _positive_integer(chunk, "attention_chunk_size")
            if any(kind == "chunked_attention" for kind in types) and chunk is None:
                _fail("chunked attention requires attention_chunk_size")
        else:
            rope_layers, chunk = tuple(True for _ in range(layers)), None
        linear = []
        for name, default in (("linear_num_key_heads", 16), ("linear_num_value_heads", 32),
                              ("linear_key_head_dim", 128), ("linear_value_head_dim", 128), ("linear_conv_kernel_dim", 4)):
            linear.append(_positive_integer(text.get(name, default), name) if hybrid else 1)
        if hybrid and linear[1] % linear[0]:
            _fail("linear value head count must be divisible by linear key head count")
        source_gate = text.get("output_gate_type", "silu") if hybrid else None
        if source_gate not in (None, "sigmoid", "swish", "silu"):
            _fail("output_gate_type must be sigmoid or swish")
        # The official Qwen3-Next/Qwen3.5-compatible classes fix the full
        # attention gate to sigmoid and the recurrent normalization gate to
        # SiLU. In particular, Qwen3.8's legacy 'swish' field does not change
        # full attention into a SiLU gate.
        gate = "sigmoid" if hybrid else None
        if hybrid and not _boolean(text, "attn_output_gate", True):
            _fail("this architecture requires the packed full-attention output gate")
        tuning = _boolean(text, "attn_temperature_tuning", True) if family == "llama4" else False
        scale = text.get("attn_scale", .1)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            _fail("attn_scale must be finite and nonnegative")
        vision = source.get("vision_config") if outer else None
        if vision is not None and not isinstance(vision, Mapping):
            _fail("vision_config must be an object")
        return cls(family, base, tuple(types), moe, expert_count, top, expert_width, shared,
            normalize, rotary, hybrid, gate, rope_layers,
            _boolean(text, "use_qk_norm", True) if family == "llama4" else False, chunk, tuning,
            _positive_integer(text.get("floor_scale", 8192), "floor_scale"), float(scale),
            *linear, dict(vision) if vision is not None else None, original,
            tuple(rope.get("mrope_section", [11, 11, 10])) if family.startswith("qwen3_5") else (0, 0, 0))

    def weight_shapes(self, *, expert_layout: str = "packed") -> dict[str, tuple[int, ...]]:
        h, q, kv, d = self.hidden_size, self.query_heads * self.head_dim, self.kv_heads * self.head_dim, self.head_dim
        result = {"model.embed_tokens.weight": (self.vocab_size, h), "model.norm.weight": (h,)}
        if not self.tied_embeddings:
            result["lm_head.weight"] = (self.vocab_size, h)
        for index, kind in enumerate(self.layer_types):
            p = f"model.layers.{index}."
            result[p + "input_layernorm.weight"] = result[p + "post_attention_layernorm.weight"] = (h,)
            if kind == "linear_attention":
                key, value = self.linear_key_heads * self.linear_key_dim, self.linear_value_heads * self.linear_value_dim
                prefix = p + "linear_attn."
                if self.family == "qwen3_next":
                    result[prefix + "in_proj_qkvz.weight"] = (key * 2 + value * 2, h)
                    result[prefix + "in_proj_ba.weight"] = (self.linear_value_heads * 2, h)
                else:
                    for name, width in (("qkv", key * 2 + value), ("z", value), ("a", self.linear_value_heads), ("b", self.linear_value_heads)):
                        result[prefix + "in_proj_" + name + ".weight"] = (width, h)
                result[prefix + "conv1d.weight"] = (key * 2 + value, 1, self.linear_kernel)
                result[prefix + "A_log"] = result[prefix + "dt_bias"] = (self.linear_value_heads,)
                result[prefix + "norm.weight"] = (self.linear_value_dim,)
                result[prefix + "out_proj.weight"] = (h, value)
            else:
                for name, shape in (("q_proj", (q * (2 if self.attention_gate else 1), h)), ("k_proj", (kv, h)), ("v_proj", (kv, h)), ("o_proj", (h, q))):
                    result[p + "self_attn." + name + ".weight"] = shape
                    if self.attention_bias:
                        result[p + "self_attn." + name + ".bias"] = (shape[0],)
                if self.family != "llama4":
                    result[p + "self_attn.q_norm.weight"] = result[p + "self_attn.k_norm.weight"] = (d,)
            prefix = p + ("feed_forward." if self.family == "llama4" else "mlp.")
            if index in self.moe_layers:
                result[prefix + ("router.weight" if self.family == "llama4" else "gate.weight")] = (self.experts, h)
                if expert_layout == "packed":
                    result[prefix + "experts.gate_up_proj"] = ((self.experts, h, self.expert_intermediate * 2) if self.family == "llama4" else (self.experts, self.expert_intermediate * 2, h))
                    result[prefix + "experts.down_proj"] = ((self.experts, self.expert_intermediate, h) if self.family == "llama4" else (self.experts, h, self.expert_intermediate))
                elif expert_layout == "individual" and self.family != "llama4":
                    for expert in range(self.experts):
                        result[prefix + f"experts.{expert}.gate_proj.weight"] = (self.expert_intermediate, h)
                        result[prefix + f"experts.{expert}.up_proj.weight"] = (self.expert_intermediate, h)
                        result[prefix + f"experts.{expert}.down_proj.weight"] = (h, self.expert_intermediate)
                else:
                    _fail("unsupported expert checkpoint layout", "VXHF_WEIGHTS")
                if self.shared_intermediate:
                    for name, shape in (("gate_proj", (self.shared_intermediate, h)), ("up_proj", (self.shared_intermediate, h)), ("down_proj", (h, self.shared_intermediate))):
                        result[prefix + "shared_expert." + name + ".weight"] = shape
                    if self.family != "llama4":
                        result[prefix + "shared_expert_gate.weight"] = (1, h)
            else:
                width = self.intermediate_size
                for name, shape in (("gate_proj", (width, h)), ("up_proj", (width, h)), ("down_proj", (h, width))):
                    result[prefix + name + ".weight"] = shape
        return result


def _text_weights(config: ExtendedDecoderConfig, source):
    result = {}
    for name, value in source.items():
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model."):]
        elif name.startswith("language_model.model."):
            name = "model." + name[len("language_model.model."):]
        elif name.startswith("language_model.lm_head."):
            name = "lm_head." + name[len("language_model.lm_head."):]
        # These are separate, named subgraphs. Text-only export reports that
        # the image encoder and optional speculative head were not exported.
        elif config.vision_config is not None and name.startswith(("model.visual.", "model.vision_model.", "model.multi_modal_projector.",
                "visual.", "vision_model.", "multi_modal_projector.")):
            continue
        elif name.startswith("mtp."):
            continue
        if name in result:
            _fail(f"duplicate normalized checkpoint tensor {name}", "VXHF_WEIGHTS")
        result[name] = value
    layout = "individual" if any(".experts.0." in name for name in result) else "packed"
    expected = config.weight_shapes(expert_layout=layout)
    optional_head = {"lm_head.weight"} if config.tied_embeddings else set()
    if set(result) - set(expected) - optional_head:
        _fail(f"unexpected checkpoint tensors: {sorted(set(result) - set(expected) - optional_head)}", "VXHF_WEIGHTS")
    if set(expected) - set(result):
        _fail(f"missing checkpoint tensors: {sorted(set(expected) - set(result))}", "VXHF_WEIGHTS")
    checked = {}
    for name, shape in expected.items():
        array = np.asarray(result[name])
        if array.dtype not in (np.dtype("float32"), np.dtype("float16")) or array.shape != shape or not np.isfinite(array).all():
            _fail(f"tensor {name} must contain finite floating shape {shape}; received {array.dtype} {array.shape}", "VXHF_WEIGHTS")
        checked[name] = np.ascontiguousarray(array, dtype=np.float32)
    if config.tied_embeddings and "lm_head.weight" in result:
        head = np.asarray(result["lm_head.weight"])
        if head.dtype not in (np.dtype("float32"), np.dtype("float16")) or not np.array_equal(head.astype(np.float32), checked["model.embed_tokens.weight"]):
            _fail("tied lm_head.weight must equal model.embed_tokens.weight", "VXHF_WEIGHTS")
    return checked, layout


def build_extended_decoder_graph(config: ExtendedDecoderConfig, source_weights: Mapping[str, Any], *,
        output_name="logits", capture_kv=False, compact_kv=True, cached=False, generate=False,
        image_features_count=None):
    """Lower a prefix graph with optional retained single-row execution.

    The graph uses compact KV heads regardless of the legacy dense export
    preference. Cached recurrent nodes expose their final state as a private
    intermediate; the common execution context carries it between row calls.
    """
    if not isinstance(cached, bool):
        _fail("cached must be a boolean")
    if cached and capture_kv:
        _fail("cached execution requires private cache tensors")
    if config.family=="qwen4_exp":
        from .hf_qwen4 import build_qwen4_decoder_graph
        return build_qwen4_decoder_graph(config,source_weights,output_name=output_name,
            capture_kv=capture_kv,compact_kv=compact_kv,cached=cached,generate=generate,
            image_features_count=image_features_count)
    weights, layout = _text_weights(config, source_weights)
    if not is_runtime_graph_name(output_name) or output_name in weights:
        _fail("logits output must be a valid tensor name distinct from weights")
    b = GraphBuilder(config.max_context, [output_name])
    b.weights.update(weights)
    s, h, d = "S", config.hidden_size, config.head_dim
    shape = [1, s, h]
    if config.norm_offset:
        for name in list(b.weights):
            if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")) or name == "model.norm.weight":
                b.weights[name] = b.weights[name] + np.float32(1)
    b.add("hf.zero", np.zeros((1,), np.float32))

    # Bounded lookup tables avoid a dense rotation matrix and preserve partial
    # or adjacent-pair rotation independently of the flattened head geometry.
    exponent = np.arange(0, config.rotary_dim, 2, dtype=np.float32) / np.float32(config.rotary_dim)
    inverse = np.float32(1) / np.power(np.float32(config.theta), exponent)
    phase = np.arange(config.max_context, dtype=np.float32)[:, None] * inverse[None]
    interleaved = config.family == "llama4"
    if not interleaved:
        phase = np.concatenate((phase, phase), axis=1)
    if not np.isfinite(phase).all():
        _fail("bounded rotary position tables must remain finite")
    if image_features_count is not None:
        image_features_count = _positive_integer(image_features_count, "image_features_count")
        if config.family not in {"qwen3_5", "qwen3_5_moe", "llama4"}:
            _fail("image feature injection is available for multimodal decoder architectures only")
        b.graph["inputs"].update({"image_features": {"shape": [image_features_count, h], "dtype": "float32"},
            "image_feature_ids": {"shape": [s], "dtype": "int32"}, "image_token_mask": {"shape": [1, s], "dtype": "int32"}})
        if config.family.startswith("qwen3_5"):
            b.graph["inputs"]["mrope_position_ids"] = {"shape": [3, s], "dtype": "int32"}
    rotary_tables = {}
    for function in ("cos", "sin"):
        weight = b.add("hf.rope." + function, getattr(np, function)(phase).astype(np.float32))
        if image_features_count is not None and config.family.startswith("qwen3_5"):
            lookups = []
            for axis in range(3):
                selected = b.tensor(weight + ".axis." + str(axis), "Slice", {"input": "mrope_position_ids"}, [1, s],
                    {"starts": [axis], "ends": [axis + 1], "axes": [0], "steps": [1]}, "int32")
                selected = b.reshape(weight + ".axis_ids." + str(axis), selected, [s], "int32")
                lookups.append(b.tensor(weight + ".lookup." + str(axis), "Embedding", {"input": selected, "weight": weight}, [s, phase.shape[1]]))
            lookup = lookups[0]
            for axis in (1, 2):
                indices = "hf.mrope.frequency_mask." + str(axis)
                if indices not in b.weights:
                    frequencies = np.arange(config.rotary_dim // 2)
                    selected = ((frequencies % 3 == axis) & (frequencies < config.mrope_sections[axis] * 3)).astype(np.int32)
                    b.add(indices, np.tile(selected, 2)[None])
                condition = b.tensor(weight + ".expand_axis_mask." + str(axis), "Expand", {"input": indices},
                    [s, phase.shape[1]], {"shape": [s, phase.shape[1]]}, "int32")
                lookup = b.tensor(weight + ".select_axis." + str(axis), "Where", {"condition": condition, "a": lookups[axis], "b": lookup}, [s, phase.shape[1]])
        else:
            lookup = b.tensor(weight + ".lookup", "Embedding", {"input": "position_ids", "weight": weight}, [s, phase.shape[1]])
        rotary_tables[function] = b.reshape(weight + ".rows", lookup, [1, s, phase.shape[1]])

    def norm(name, value, weight, outshape=shape):
        return b.norm(name, value, weight, outshape, config.epsilon)

    def linear(name, value, prefix, width):
        return b.linear(name, value, prefix + ".weight", width,
            prefix + ".bias" if prefix + ".bias" in b.weights else None)

    def rope(name, value):
        return b.tensor(name, "RotaryEmbedding", {"input": value, **rotary_tables},
            [1, s, config.query_heads if ".q_" in name else config.kv_heads, d],
            {"rotary_dim": config.rotary_dim, "interleaved": interleaved, "position_offset": 0})

    def dense(name, value, prefix, width):
        gate = linear(name + ".gate", value, prefix + "gate_proj", width)
        up = linear(name + ".up", value, prefix + "up_proj", width)
        gate = b.tensor(name + ".silu", "SiLU", {"input": gate}, [1, s, width])
        product = b.tensor(name + ".product", "Mul", {"a": gate, "b": up}, [1, s, width])
        return linear(name + ".down", product, prefix + "down_proj", h)

    temperature = None
    if config.temperature_tuning and not all(config.rope_layers):
        # Sequence offset is a distinct input because caller RoPE positions may
        # contain gaps. Full-prefix generation supplies arange(S), not those IDs.
        b.graph["inputs"]["sequence_ids"] = {"shape": [s], "dtype": "int32"}
        positions = np.arange(config.max_context, dtype=np.float32)
        table = (np.log1p(np.floor((positions + np.float32(1)) / np.float32(config.temperature_floor))) *
                 np.float32(config.temperature_scale) + np.float32(1))[:, None]
        b.add("hf.temperature", table)
        lookup = b.tensor("hf.temperature.lookup", "Embedding", {"input": "sequence_ids", "weight": "hf.temperature"}, [s, 1])
        temperature = b.reshape("hf.temperature.rows", lookup, [1, s, 1, 1])
    if "chunked_attention" in config.layer_types:
        # This mask includes padding and chunk boundaries. It is kept separate
        # from BK padding, which recurrent layers need before their projections.
        b.graph["inputs"]["chunk_attention_mask"] = {"shape": [1, s, s], "dtype": "int32"}
    if config.qk_l2_norm:
        b.add("hf.qk.unit", np.ones((d,), np.float32))

    value = b.tensor("embedding", "Embedding", {"input": "token_ids", "weight": "model.embed_tokens.weight"}, shape)
    if image_features_count is not None:
        image = b.tensor("image.lookup", "Embedding", {"input": "image_feature_ids", "weight": "image_features"}, [s, h])
        image = b.reshape("image.rows", image, shape)
        mask = b.reshape("image.mask", "image_token_mask", [1, s, 1], "int32")
        mask = b.tensor("image.mask_expand", "Expand", {"input": mask}, shape, {"shape": shape}, "int32")
        value = b.tensor("image.embedding", "Where", {"condition": mask, "a": image, "b": value}, shape)
    for index, kind in enumerate(config.layer_types):
        p, name = f"model.layers.{index}.", f"layer.{index}"
        normalized = norm(name + ".attention_norm", value, p + "input_layernorm.weight")
        if kind == "linear_attention":
            prefix = p + "linear_attn."
            hk, hv, dk, dv = config.linear_key_heads, config.linear_value_heads, config.linear_key_dim, config.linear_value_dim
            key, val = hk * dk, hv * dv
            if config.family == "qwen3_next":
                packed = b.weights[prefix + "in_proj_qkvz.weight"].reshape(hk, 2 * dk + 2 * (hv // hk) * dv, h)
                splits = np.split(packed, [dk, dk * 2, dk * 2 + (hv // hk) * dv], axis=1)
                qkv = np.concatenate([part.reshape(-1, h) for part in splits[:3]], axis=0)
                b.add(prefix + "hf_qkv.weight", qkv)
                b.add(prefix + "hf_z.weight", splits[3].reshape(val, h))
                ba = b.weights[prefix + "in_proj_ba.weight"].reshape(hk, 2 * hv // hk, h)
                part_b, part_a = np.split(ba, 2, axis=1)
                b.add(prefix + "hf_a.weight", part_a.reshape(hv, h))
                b.add(prefix + "hf_b.weight", part_b.reshape(hv, h))
                qkv_prefix, z_prefix, a_prefix, beta_prefix = [prefix + "hf_" + suffix for suffix in ("qkv", "z", "a", "b")]
            else:
                qkv_prefix, z_prefix, a_prefix, beta_prefix = [prefix + "in_proj_" + suffix for suffix in ("qkv", "z", "a", "b")]
            mask = b.reshape(name + ".padding.rows", "attention_mask", [1, s, 1], "int32")
            mask = b.tensor(name + ".padding.expand", "Expand", {"input": mask}, shape, {"shape": shape}, "int32")
            zero = b.tensor(name + ".zero.expand", "Expand", {"input": "hf.zero"}, shape, {"shape": shape})
            masked = b.tensor(name + ".masked_hidden", "Where", {"condition": mask, "a": normalized, "b": zero}, shape)
            projected = linear(name + ".qkv", masked, qkv_prefix, key * 2 + val)
            conv_input = b.reshape(name + ".conv_input", projected, [1, 1, s, key * 2 + val])
            conv_weight = b.add(prefix + "hf_conv.weight", b.weights[prefix + "conv1d.weight"][:, 0, :].T[None, :, :, None])
            conv = b.tensor(name + ".conv", "Conv2D", {"input": conv_input, "weight": conv_weight},
                [1, 1, s, key * 2 + val], {"groups": key * 2 + val, "pads": [0, config.linear_kernel - 1, 0, 0],
                    "data_layout": "NHWC", "weight_layout": "HWCM"})
            conv = b.reshape(name + ".conv_rows", conv, [1, s, key * 2 + val])
            conv = b.tensor(name + ".conv_silu", "SiLU", {"input": conv}, [1, s, key * 2 + val])
            qkv = []
            for label, start, width, heads, dim in (("q", 0, key, hk, dk), ("k", key, key, hk, dk), ("v", key * 2, val, hv, dv)):
                indices = b.add(prefix + "hf_" + label + ".indices", np.arange(start, start + width, dtype=np.int32))
                part = b.tensor(name + ".conv_" + label, "Gather", {"input": conv, "indices": indices}, [1, s, width], {"axis": 2})
                qkv.append(b.reshape(name + "." + label + "_heads", part, [1, s, heads, dim]))
            a = linear(name + ".a", masked, a_prefix, hv)
            beta = linear(name + ".b", masked, beta_prefix, hv)
            delta_inputs = {"q": qkv[0], "k": qkv[1], "v": qkv[2],
                "a": a, "b": beta, "a_log": prefix + "A_log", "dt_bias": prefix + "dt_bias"}
            delta_params = {"normalize": True, "eps": 1e-6, "scale": 1 / math.sqrt(dk)}
            if cached:
                delta = b.multi(name + ".delta", "GatedDeltaNet", delta_inputs,
                    {"out": (name + ".delta", [1, s, hv, dv], "float32"),
                     "state": (name + ".delta_state", [1, hv, dk, dv], "float32")}, delta_params)["out"]
            else:
                delta = b.tensor(name + ".delta", "GatedDeltaNet", delta_inputs,
                    [1, s, hv, dv], delta_params)
            delta = norm(name + ".delta_norm", delta, prefix + "norm.weight", [1, s, hv, dv])
            z = linear(name + ".z", masked, z_prefix, val)
            z = b.reshape(name + ".z_heads", z, [1, s, hv, dv])
            z = b.tensor(name + ".z_silu", "SiLU", {"input": z}, [1, s, hv, dv])
            delta = b.tensor(name + ".delta_gated", "Mul", {"a": delta, "b": z}, [1, s, hv, dv])
            delta = b.reshape(name + ".delta_flat", delta, [1, s, val])
            projected = linear(name + ".attention_output", delta, prefix + "out_proj", h)
        else:
            prefix = p + "self_attn."
            query_prefix = prefix + "q_proj"
            if config.attention_gate:
                packed = b.weights[query_prefix + ".weight"].reshape(config.query_heads, 2 * d, h)
                b.add(prefix + "hf_q.weight", packed[:, :d].reshape(config.query_heads * d, h))
                b.add(prefix + "hf_gate.weight", packed[:, d:].reshape(config.query_heads * d, h))
                if query_prefix + ".bias" in b.weights:
                    bias = b.weights[query_prefix + ".bias"].reshape(config.query_heads, 2 * d)
                    b.add(prefix + "hf_q.bias", bias[:, :d].reshape(-1))
                    b.add(prefix + "hf_gate.bias", bias[:, d:].reshape(-1))
                query_prefix = prefix + "hf_q"
            q = linear(name + ".q", normalized, query_prefix, config.query_heads * d)
            k = linear(name + ".k", normalized, prefix + "k_proj", config.kv_heads * d)
            v = linear(name + ".v", normalized, prefix + "v_proj", config.kv_heads * d)
            q = b.reshape(name + ".q_heads", q, [1, s, config.query_heads, d])
            k = b.reshape(name + ".k_heads", k, [1, s, config.kv_heads, d])
            v = b.reshape(name + ".v_heads", v, [1, s, config.kv_heads, d])
            if config.family != "llama4":
                q = norm(name + ".q_norm", q, prefix + "q_norm.weight", [1, s, config.query_heads, d])
                k = norm(name + ".k_norm", k, prefix + "k_norm.weight", [1, s, config.kv_heads, d])
            if config.rope_layers[index]:
                q, k = rope(name + ".q_rope", q), rope(name + ".k_rope", k)
                if config.qk_l2_norm:
                    q = norm(name + ".q_l2", q, "hf.qk.unit", [1, s, config.query_heads, d])
                    k = norm(name + ".k_l2", k, "hf.qk.unit", [1, s, config.kv_heads, d])
            elif temperature:
                expanded = b.tensor(name + ".temperature_expand", "Expand", {"input": temperature},
                    [1, s, config.query_heads, d], {"shape": [1, s, config.query_heads, d]})
                q = b.tensor(name + ".q_temperature", "Mul", {"a": q, "b": expanded}, [1, s, config.query_heads, d])
            if capture_kv:
                b.graph["outputs"].extend([k, v])
            attention = b.tensor(name + ".attention", "SparseAttention", {"q": q, "k": k, "v": v,
                "mask": "chunk_attention_mask" if kind == "chunked_attention" else "attention_mask"},
                [1, s, config.query_heads, d], {"causal": True, "scale": 1 / math.sqrt(d)})
            attention = b.reshape(name + ".attention_flat", attention, [1, s, config.query_heads * d])
            if config.attention_gate:
                gate = linear(name + ".attention_gate", normalized, prefix + "hf_gate", config.query_heads * d)
                gate = b.tensor(name + ".attention_gate_activation", "Sigmoid" if config.attention_gate == "sigmoid" else "SiLU", {"input": gate}, [1, s, config.query_heads * d])
                attention = b.tensor(name + ".attention_gated", "Mul", {"a": attention, "b": gate}, [1, s, config.query_heads * d])
            projected = linear(name + ".attention_output", attention, prefix + "o_proj", h)
        value = b.tensor(name + ".attention_residual", "Add", {"a": value, "b": projected}, shape)
        normalized = norm(name + ".mlp_norm", value, p + "post_attention_layernorm.weight")
        prefix = p + ("feed_forward." if config.family == "llama4" else "mlp.")
        if index in config.moe_layers:
            bank = prefix + "experts."
            if layout == "packed":
                packed = b.weights[bank + "gate_up_proj"]
                if config.family != "llama4":
                    packed = packed.transpose(0, 2, 1)
                gate_weight, up_weight = np.split(packed, 2, axis=2)
                down_weight = b.weights[bank + "down_proj"]
                if config.family != "llama4":
                    down_weight = down_weight.transpose(0, 2, 1)
            else:
                gate_weight, up_weight, down_weight = [np.stack([b.weights[bank + str(expert) + "." + label + ".weight"].T for expert in range(config.experts)]) for label in ("gate_proj", "up_proj", "down_proj")]
            for label, tensor in (("gate", gate_weight), ("up", up_weight), ("down", down_weight)):
                b.add(bank + "hf_" + label, tensor)
            router = prefix + ("router.weight" if config.family == "llama4" else "gate.weight")
            b.add(prefix + "hf_router", b.weights[router].T)
            route = b.multi(name + ".router", "MoERouter", {"input": normalized, "weight": prefix + "hf_router"},
                {"indices": (name + ".route_indices", [1, s, config.experts_per_token], "float32"),
                 "weights": (name + ".route_weights", [1, s, config.experts_per_token], "float32")},
                {"num_experts": config.experts, "top_k": config.experts_per_token,
                 "scoring_func": "sigmoid" if config.family == "llama4" else "softmax",
                 "normalize": False if config.family == "llama4" else config.normalize_routes})
            down = b.tensor(name + ".experts", "MoEGatedMLP", {"input": normalized,
                "gate_weight": bank + "hf_gate", "up_weight": bank + "hf_up", "down_weight": bank + "hf_down",
                "route_indices": route["indices"], "route_weights": route["weights"]}, shape,
                {"route_input_scale": config.family == "llama4"})
            if config.shared_intermediate:
                shared = dense(name + ".shared", normalized, prefix + "shared_expert.", config.shared_intermediate)
                if config.family != "llama4":
                    gate = linear(name + ".shared_gate", normalized, prefix + "shared_expert_gate", 1)
                    gate = b.tensor(name + ".shared_sigmoid", "Sigmoid", {"input": gate}, [1, s, 1])
                    gate = b.tensor(name + ".shared_expand", "Expand", {"input": gate}, shape, {"shape": shape})
                    shared = b.tensor(name + ".shared_gated", "Mul", {"a": shared, "b": gate}, shape)
                down = b.tensor(name + ".shared_sum", "Add", {"a": down, "b": shared}, shape)
        else:
            down = dense(name + ".mlp", normalized, prefix, config.intermediate_size)
        value = b.tensor(name + ".mlp_residual", "Add", {"a": value, "b": down}, shape)
    value = norm("final_norm", value, "model.norm.weight")
    b.linear(output_name, value, "model.embed_tokens.weight" if config.tied_embeddings else "lm_head.weight", config.vocab_size)
    return _finish_extended(b,config,output_name,generate,cached=cached)


def _finish_extended(builder,config,output_name,generate,cached=False):
    if not (generate or cached):return builder.finish()
    if output_name=="generated":
        _fail("generation logits output conflicts with generated ID output")
    builder.tensor("generated","ArgMax",{"input":output_name},[1,"S"],
        {"axis":-1,"keepdims":False},"int32")
    builder.graph["outputs"].append("generated")
    inputs=builder.graph["inputs"]
    zero_controls=[name for name in ("image_feature_ids","image_token_mask") if name in inputs]
    controlled={"token_ids","attention_mask","position_ids","sequence_ids","chunk_attention_mask","mrope_position_ids",*zero_controls}
    controls = dict(token_input="token_ids",token_output="generated",
        keep_input="attention_mask",position_input="position_ids",logits_output=output_name,id_domain_size=config.vocab_size,
        sequence_input="sequence_ids" if "sequence_ids" in inputs else None,
        chunk_mask_input="chunk_attention_mask" if "chunk_attention_mask" in inputs else None,
        chunk_size=config.chunk_size if "chunk_attention_mask" in inputs else None,zero_append_inputs=zero_controls,
        multimodal_position_input="mrope_position_ids" if "mrope_position_ids" in inputs else None,
        invariant_inputs=[name for name in inputs if name not in controlled])
    if cached:
        inputs["active_lengths"] = {"shape": [1], "dtype": "int32"}
        execution = incremental_decoder_execution(inputs, builder.graph["outputs"],
            incremental_controls=True, **controls)
    else:
        execution = full_prefix_decoder_execution(inputs, builder.graph["outputs"], **controls)
    return builder.finish(execution)


def export_hf_extended(directory, out_path, *, max_context=None, output_names=None,
        capture_kv=False, compact_kv=False, cached=False, generate=False,
        image_features_count=None, report_callback=None, weight_precision=None):
    if not isinstance(cached, bool):
        _fail("cached must be a boolean")
    if cached and capture_kv:
        _fail("cached execution requires private cache tensors")
    source = _read_json(Path(directory) / "config.json")
    config = ExtendedDecoderConfig.from_hf(source, max_context=max_context)
    if output_names is not None and len(output_names) != 1:
        _fail("HF decoder export has exactly one logits output")
    source, source_weights, source_dtypes = read_checkpoint_storage(directory, config_validator=ExtendedDecoderConfig.from_hf)
    graph, weights = build_extended_decoder_graph(config, source_weights,
        output_name=output_names[0] if output_names else "logits", capture_kv=capture_kv,
        compact_kv=compact_kv, cached=cached, generate=generate, image_features_count=image_features_count)
    graph = write_hf_package(out_path, graph, weights, weight_precision=weight_precision)
    if report_callback:
        features = [f"family:{config.family}", "execution:F32", "batch:1",
            "state:incremental-kv" if cached else "state:full-prefix-zero-recurrent-state",
            "attention:rank4-compact-kv", f"context-bound:{config.max_context}", "rope:bounded-position-tables"]
        if cached and "linear_attention" in config.layer_types:
            features.append("state:gated-delta-and-causal-convolution-history")
        if cached:
            features.append("generation:incremental-rows")
        elif generate:
            features.append("generation:full-prefix-recomputation")
        if config.vision_config is not None:
            features.append("scope:image-conditioned-text-decoder" if image_features_count is not None else "scope:text-decoder;image-encoder-separate-package")
        if source.get("text_config", source).get("mtp_num_hidden_layers", 0):
            features.append("scope:main-decoder;optional-speculative-MTP-head-not-included")
        report_callback({"features": {"hf_decoder": features, "source_storage": sorted(set(source_dtypes.values()))}})
    return graph
