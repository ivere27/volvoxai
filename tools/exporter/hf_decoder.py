"""Local Hugging Face decoder checkpoints to ordinary CPU/F32 graphs.

The shared builder handles batch-one F32 graphs and their typed execution
metadata. Full-prefix evaluation may materialize repeated KV heads with Gather;
the compact option addresses grouped heads directly and can declare incremental
state with greedy engine feedback. Tokenization and source BF16 execution are
outside this builder.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .hf_graph import (positive_float as _positive_float, positive_integer as _positive_integer,
    read_config as _read_json, read_checkpoint_storage, write_hf_package)
from .execution_contract import stateless_execution
from .quantization_storage import GRAPH_FORMAT
from .runtime_names import is_runtime_graph_name
from .runtime_ir import import_runtime_package


def _fail(message: str, code: str = "VXHF_CONFIG") -> None:
    raise ExporterError(Diagnostic(code, message, "hf-decoder"))


@dataclass(frozen=True)
class DecoderConfig:
    family: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    max_context: int
    epsilon: float
    theta: float
    tied_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    llama3_scaling: tuple[float, float, float, int] | None = None
    qwen2_scaling: tuple[str, float, float, float, float, int, bool] | None = None

    @property
    def output_attention_bias(self) -> bool:
        # Qwen2 always has biased Q/K/V and a bias-free output projection.
        return self.attention_bias and self.family != "qwen2"

    @property
    def table_rope(self) -> bool:
        return self.llama3_scaling is not None or self.qwen2_scaling is not None

    @property
    def rope_table_prefix(self) -> str:
        return "hf.llama3" if self.llama3_scaling else "hf.qwen2"

    @classmethod
    def from_hf(cls, source: Mapping[str, Any], *, max_context: int | None = None):
        family = source.get("model_type")
        if not isinstance(family, str) or family not in {"llama", "qwen2", "qwen3"}:
            _fail("dense decoder configurations require llama, qwen2 or qwen3")
        architecture = {"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM",
                        "qwen3": "Qwen3ForCausalLM"}[family]
        if source.get("architectures") not in (None, [architecture]):
            _fail(f"architectures must identify {architecture}")
        if source.get("hidden_act", "silu") != "silu":
            _fail("only the SiLU gated MLP is supported")
        for name in ("quantization_config", "attention_chunk_size",
                     "attn_logit_softcapping", "final_logit_softcapping"):
            if source.get(name) not in (None, False, 0):
                _fail(f"{name} is unsupported by full-prefix decoder import")
        sliding_enabled = source.get("use_sliding_window", False)
        if not isinstance(sliding_enabled, bool):
            _fail("use_sliding_window must be boolean")
        # Official Qwen2Config discards sliding_window when its flag is false;
        # many full-attention Qwen2.5 checkpoints retain that inactive metadata.
        if family != "qwen2" and source.get("sliding_window") not in (None, False, 0):
            _fail("sliding_window is unsupported by full-prefix decoder import")
        if sliding_enabled or source.get("is_encoder_decoder", False):
            _fail("sliding-window and encoder-decoder configurations are unsupported")
        for name in ("partial_rotary_factor", "rope_fraction"):
            if source.get(name, 1.0) != 1.0:
                _fail(f"{name} must be 1; partial RoPE is unsupported")
        hidden = _positive_integer(source.get("hidden_size"), "hidden_size")
        heads = _positive_integer(source.get("num_attention_heads"), "num_attention_heads")
        kv_heads = source.get("num_key_value_heads", 32 if family in {"qwen2", "qwen3"} else None)
        if kv_heads is None:
            kv_heads = heads
        kv_heads = _positive_integer(kv_heads, "num_key_value_heads")
        if heads % kv_heads:
            _fail("num_attention_heads must be divisible by num_key_value_heads")
        if family in {"qwen2", "qwen3"} and "head_dim" in source and source["head_dim"] is None:
            _fail(f"{family} head_dim must not be null")
        # Qwen3Config defaults to 128 even when hidden_size / heads differs.
        # LlamaConfig derives its omitted head dimension from hidden_size.
        head_dim = source.get("head_dim", 128 if family == "qwen3" else None)
        if head_dim is None:
            if hidden % heads:
                _fail("hidden_size must be divisible by num_attention_heads without explicit head_dim")
            head_dim = hidden // heads
        head_dim = _positive_integer(head_dim, "head_dim")
        if head_dim % 2:
            _fail("head_dim must be even for full half-split RoPE")
        layers = _positive_integer(source.get("num_hidden_layers"), "num_hidden_layers")
        if source.get("layer_types") not in (None, ["full_attention"] * layers):
            _fail("every layer must use full_attention")
        if source.get("pretraining_tp", 1) != 1:
            _fail("pretraining_tp other than 1 is unsupported")
        for name in ("rope_parameters", "rope_scaling"):
            if source.get(name) is not None and not isinstance(source[name], Mapping):
                _fail(f"{name} must be an object or null")
        rope = source.get("rope_parameters") or source.get("rope_scaling") or {}
        if "rope_type" in rope and "type" in rope and rope["rope_type"] != rope["type"]:
            _fail("conflicting RoPE type values")
        if source.get("rope_parameters") and source.get("rope_scaling"):
            _fail("conflicting rope_parameters and rope_scaling are unsupported")
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if not isinstance(rope_type, str):
            _fail("RoPE type must be a string")
        allowed = {"rope_type", "type", "rope_theta"}
        if rope_type == "llama3" and family == "llama":
            allowed |= {"factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}
        elif family == "qwen2" and rope_type in {"linear", "yarn"}:
            allowed |= {"factor"}
            if rope_type == "yarn":
                allowed |= {"attention_factor", "beta_fast", "beta_slow",
                            "original_max_position_embeddings", "truncate"}
        elif rope_type != "default":
            _fail("unsupported RoPE: dense import accepts default, Llama3, and Qwen2 linear/YaRN")
        if set(rope) - allowed:
            _fail("scaled or custom RoPE settings are unsupported")
        theta = rope.get("rope_theta", source.get("rope_theta", 10000.0))
        if "rope_theta" in rope and "rope_theta" in source and theta != source["rope_theta"]:
            _fail("conflicting RoPE theta values")
        theta = _positive_float(theta, "rope_theta")
        declared_max = _positive_integer(source.get("max_position_embeddings"), "max_position_embeddings")
        maximum = declared_max if max_context is None else _positive_integer(max_context, "max_context")
        if maximum > declared_max:
            _fail("max_context exceeds source max_position_embeddings")
        scaling = None
        qwen2_scaling = None
        if rope_type == "llama3":
            factor = _positive_float(rope.get("factor"), "llama3 factor")
            low = _positive_float(rope.get("low_freq_factor"), "llama3 low_freq_factor")
            high = _positive_float(rope.get("high_freq_factor"), "llama3 high_freq_factor")
            original = _positive_integer(rope.get("original_max_position_embeddings"),
                                         "llama3 original_max_position_embeddings")
            if factor < 1 or high <= low or np.float32(high) <= np.float32(low) or original > declared_max:
                _fail("llama3 requires factor >= 1, distinct F32 high > low, and original context <= source context")
            scaling = (factor, low, high, original)
        if family == "qwen2" and rope_type in {"linear", "yarn"}:
            factor = _positive_float(rope.get("factor"), f"{rope_type} factor")
            if factor < 1:
                _fail(f"{rope_type} factor must be >= 1")
            original = _positive_integer(source.get("original_max_position_embeddings",
                rope.get("original_max_position_embeddings", declared_max)),
                                         "original_max_position_embeddings")
            if original > declared_max:
                _fail("original context exceeds source max_position_embeddings")
            attention = _positive_float(rope.get("attention_factor") if rope.get("attention_factor") is not None
                else (1 + 0.1 * math.log(factor) if rope_type == "yarn" else 1), "attention_factor")
            fast = _positive_float(rope.get("beta_fast", 32), "beta_fast")
            slow = _positive_float(rope.get("beta_slow", 1), "beta_slow")
            truncate = rope.get("truncate", True)
            if not isinstance(truncate, bool) or fast < slow:
                _fail("YaRN requires boolean truncate and beta_fast >= beta_slow")
            if rope_type == "yarn" and float(theta) <= 1:
                _fail("YaRN rope_theta must exceed 1 for its correction range")
            qwen2_scaling = (rope_type, factor, attention, fast, slow, original, truncate)
        booleans = {}
        for name in ("tie_word_embeddings", "attention_bias", "mlp_bias"):
            value = source.get(name, False)
            if not isinstance(value, bool):
                _fail(f"{name} must be boolean")
            booleans[name] = value
        if family in {"qwen2", "qwen3"} and booleans["mlp_bias"]:
            _fail(f"{family} MLP projections are bias-free; mlp_bias=True is unsupported")
        if family == "qwen2":
            booleans["attention_bias"] = True
        return cls(family, _positive_integer(source.get("vocab_size"), "vocab_size"),
                   hidden, _positive_integer(source.get("intermediate_size"), "intermediate_size"),
                   layers, heads, kv_heads, head_dim, maximum,
                   _positive_float(source.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
                   theta, booleans["tie_word_embeddings"],
                   booleans["attention_bias"], booleans["mlp_bias"], scaling, qwen2_scaling)

    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        h, m, q, kv = self.hidden_size, self.intermediate_size, self.query_heads * self.head_dim, self.kv_heads * self.head_dim
        result = {"model.embed_tokens.weight": (self.vocab_size, h), "model.norm.weight": (h,)}
        if not self.tied_embeddings:
            result["lm_head.weight"] = (self.vocab_size, h)
        for i in range(self.layers):
            prefix = f"model.layers.{i}."
            result[prefix + "input_layernorm.weight"] = (h,)
            result[prefix + "post_attention_layernorm.weight"] = (h,)
            for name, shape in (("q_proj", (q, h)), ("k_proj", (kv, h)),
                                ("v_proj", (kv, h)), ("o_proj", (h, q))):
                result[prefix + "self_attn." + name + ".weight"] = shape
                has_bias = self.output_attention_bias if name == "o_proj" else self.attention_bias
                if has_bias:
                    result[prefix + "self_attn." + name + ".bias"] = (shape[0],)
            if self.family == "qwen3":
                for name in ("q_norm", "k_norm"):
                    result[prefix + "self_attn." + name + ".weight"] = (self.head_dim,)
            for name, shape in (("gate_proj", (m, h)), ("up_proj", (m, h)), ("down_proj", (h, m))):
                result[prefix + "mlp." + name + ".weight"] = shape
                if self.mlp_bias:
                    result[prefix + "mlp." + name + ".bias"] = (shape[0],)
        return result


def _checked_weights(config: DecoderConfig, source: Mapping[str, Any]) -> dict[str, np.ndarray]:
    shapes = config.weight_shapes()
    optional_head = {"lm_head.weight"} if config.tied_embeddings else set()
    if set(source) - set(shapes) - optional_head:
        _fail(f"unexpected checkpoint tensors: {sorted(set(source) - set(shapes) - optional_head)}", "VXHF_WEIGHTS")
    if set(shapes) - set(source):
        _fail(f"missing checkpoint tensors: {sorted(set(shapes) - set(source))}", "VXHF_WEIGHTS")
    result = {}
    for name, shape in shapes.items():
        array = np.asarray(source[name])
        if array.dtype not in (np.dtype("float32"), np.dtype("float16")) or array.shape != shape:
            _fail(f"tensor {name} must have floating shape {shape}; received {array.dtype} {array.shape}", "VXHF_WEIGHTS")
        array = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            _fail(f"tensor {name} contains non-finite values", "VXHF_WEIGHTS")
        result[name] = array
    if config.tied_embeddings and "lm_head.weight" in source:
        head = np.asarray(source["lm_head.weight"])
        if head.dtype not in (np.dtype("float32"), np.dtype("float16")) or head.shape != (config.vocab_size, config.hidden_size):
            _fail("tied lm_head.weight must have the same floating matrix shape as embeddings", "VXHF_WEIGHTS")
        head = head.astype(np.float32, copy=False)
        if not np.array_equal(head, result["model.embed_tokens.weight"]):
            _fail("tied lm_head.weight must equal model.embed_tokens.weight", "VXHF_WEIGHTS")
    return result


def check_tied_checkpoint_weights(inventory):
    """Compare duplicated tied weights in bounded chunks without F32 export."""
    from .hf_graph import _mapped_tensor
    dtypes = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}
    descriptors = [inventory[name] for name in ("model.embed_tokens.weight", "lm_head.weight")]
    if descriptors[0][2] != descriptors[1][2] or any(item[1] not in dtypes for item in descriptors):
        _fail("tied lm_head.weight must match embedding shape and floating storage", "VXHF_WEIGHTS")
    arrays = [_mapped_tensor(item, dtypes[item[1]]).reshape(-1) for item in descriptors]
    try:
        for start in range(0, arrays[0].size, 262144):
            values = []
            for array, descriptor in zip(arrays, descriptors):
                chunk = array[start:start + 262144]
                values.append((chunk.astype(np.uint32) << 16).view(np.float32)
                              if descriptor[1] == "BF16" else chunk.astype(np.float32, copy=False))
            if not all(np.isfinite(value).all() for value in values) or not np.array_equal(*values):
                _fail("tied lm_head.weight must equal model.embed_tokens.weight", "VXHF_WEIGHTS")
    finally:
        arrays.clear()


def _llama3_initializers(config: DecoderConfig) -> dict[str, np.ndarray]:
    """Fixed Llama3 frequencies, shared by every query/key head and layer."""
    factor, low, high, original = config.llama3_scaling
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exponent = np.arange(0, config.head_dim, 2, dtype=np.float32) / np.float32(config.head_dim)
        inverse = np.float32(1) / np.power(np.float32(config.theta), exponent)
        wavelength = np.float32(2 * math.pi) / inverse
        low_wavelength, high_wavelength = np.float32(original / low), np.float32(original / high)
        selected = np.where(wavelength > low_wavelength, inverse / np.float32(factor), inverse)
        smooth = (np.float32(original) / wavelength - np.float32(low)) / np.float32(high - low)
        interpolated = ((np.float32(1) - smooth) * selected / np.float32(factor)
                        + smooth * selected)
        medium = (wavelength >= high_wavelength) & (wavelength <= low_wavelength)
        inverse = np.where(medium, interpolated, selected).astype(np.float32)
    return _rotary_initializers(config, inverse)


def _qwen2_initializers(config: DecoderConfig) -> dict[str, np.ndarray]:
    """Fixed linear/YaRN frequencies from Transformers' RoPE definition.

    Like Llama3, this bounded lowering uses ordinary graph operators. It does
    not approximate a sequence-dependent NTK update or silently change theta.
    """
    kind, factor, attention, fast, slow, original, truncate = config.qwen2_scaling
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exponent = np.arange(0, config.head_dim, 2, dtype=np.float32) / np.float32(config.head_dim)
        powers = np.power(np.float32(config.theta), exponent)
        inverse = np.float32(1) / powers
        if kind == "linear":
            inverse /= np.float32(factor)
        else:
            def correction(rotations):
                return config.head_dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(config.theta))
            low, high = correction(fast), correction(slow)
            if truncate:
                low, high = math.floor(low), math.ceil(high)
            low, high = max(low, 0), min(high, config.head_dim - 1)
            if low == high:
                high += 0.001
            ramp = np.clip((np.arange(config.head_dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
            extrapolation = np.float32(1) - ramp
            interpolated = np.float32(1) / (np.float32(factor) * powers)
            inverse = interpolated * (np.float32(1) - extrapolation) + inverse * extrapolation
    return _rotary_initializers(config, inverse, attention)


def _rotary_initializers(config: DecoderConfig, inverse, attention=1.0) -> dict[str, np.ndarray]:
    """Bounded half-split tables without per-head copies or rotation matrices."""
    if 8 * config.max_context * config.head_dim > 1 << 30:
        _fail("scaled RoPE exceeds 1 GiB of derived initializers; reduce the context/head geometry")
    with np.errstate(over="ignore", invalid="ignore"):
        phase = np.arange(config.max_context, dtype=np.float32)[:, None] * inverse[None]
        phase = np.concatenate((phase, phase), axis=1)
        cosine = np.cos(phase) * np.float32(attention)
        sine = np.sin(phase) * np.float32(attention)
    if not all(np.isfinite(value).all() for value in (inverse, cosine, sine)):
        _fail("scaled frequencies and position tables must remain finite F32")
    return {config.rope_table_prefix + ".cos": cosine,
            config.rope_table_prefix + ".sin": sine}


def build_decoder_graph(config: DecoderConfig, source_weights: Mapping[str, Any], *, output_name: str = "logits", capture_kv: bool = False, compact_kv: bool = False, cached: bool = False):
    """Return a bounded dynamic-S full-forward graph and F32 initializer set."""
    weights = _checked_weights(config, source_weights)
    graph, derived = build_decoder_structure(config, output_name=output_name,
        capture_kv=capture_kv, compact_kv=compact_kv, cached=cached)
    weights.update(derived)
    # The exporter validates payloads and the complete typed RuntimeIR before
    # publication. Direct loading validates original shard metadata in C.
    import_runtime_package(graph, weights)
    return graph, weights


def build_decoder_structure(config: DecoderConfig, *, output_name: str = "logits", capture_kv: bool = False, compact_kv: bool = False, cached: bool = False):
    """Build graph topology and small derived constants without checkpoint IO.

    Original weight names/layouts come from config.weight_shapes(). A direct
    loader can bind its original shards, supplying only these derived constants
    as an additional shard. Loading/compilation must validate actual metadata.
    """
    if config.table_rope and not compact_kv:
        _fail("table-based scaled RoPE requires compact_kv; materialized-repeat lowering is unsupported")
    weights = {}
    if cached and (not compact_kv or capture_kv):
        _fail("cached execution requires compact KV and private cache tensors")
    if not is_runtime_graph_name(output_name) or output_name in config.weight_shapes():
        _fail("logits output must be a valid graph tensor name distinct from weights")
    s, h, d = "S", config.hidden_size, config.head_dim
    graph = {"format": GRAPH_FORMAT, "dimensions": {s: {"min": 1, "max": config.max_context}},
             "inputs": {"token_ids": {"shape": [1, s], "dtype": "int32"},
                        "position_ids": {"shape": [s], "dtype": "int32"},
                        "attention_mask": {"shape": [1, s], "dtype": "int32"}},
             "nodes": [], "outputs": [output_name]}

    def node(name, op, inputs, shape, params=None):
        graph["nodes"].append({"id": name, "opType": op, "inputs": inputs,
                               "outputs": {"out": {"tensor": name, "dtype": "float32", "shape": list(shape)}},
                               "params": params or {}})
        return name

    def norm(name, value, weight, shape):
        return node(name, "RMSNorm", {"input": value, "weight": weight}, shape, {"eps": config.epsilon})

    def linear(name, value, prefix, width, bias):
        inputs = {"input": value, "weight": prefix + ".weight"}
        if bias:
            inputs["bias"] = prefix + ".bias"
        return node(name, "Linear", inputs, [1, s, width], {"weight_layout": "dout_din"})

    scaled_tables = {}
    if config.table_rope:
        weights.update(_llama3_initializers(config) if config.llama3_scaling else _qwen2_initializers(config))
        for function in ("cos", "sin"):
            name = config.rope_table_prefix + "." + function
            embedded = node(name + ".lookup", "Embedding", {"input": "position_ids", "weight": name}, [s, d])
            scaled_tables[function] = node(name + ".rows", "Reshape", {"input": embedded},
                                            [1, s, d], {"shape": [1, s, d]})

    def head_rope(name, value, heads, norm_weight=None):
        if config.table_rope:
            value = node(name + ".split", "Reshape", {"input": value},
                         [1, s, heads, d], {"shape": [1, s, heads, d]})
            if norm_weight:
                value = norm(name + ".norm", value, norm_weight, [1, s, heads, d])
            value = node(name + ".rotate", "RotaryEmbedding", {"input": value, **scaled_tables},
                         [1, s, heads, d], {"rotary_dim": d, "interleaved": False})
            return node(name, "Reshape", {"input": value},
                        [1, s, heads * d], {"shape": [1, s, heads * d]})
        if compact_kv:
            if norm_weight:
                value = node(name + ".split", "Reshape", {"input": value},
                             [1, s, heads, d], {"shape": [1, s, heads, d]})
                value = norm(name + ".norm", value, norm_weight, [1, s, heads, d])
                value = node(name + ".flat", "Reshape", {"input": value},
                             [1, s, heads * d], {"shape": [1, s, heads * d]})
            return node(name, "RoPE", {"input": value, "position_ids": "position_ids"},
                        [1, s, heads * d], {"heads": heads, "rotary_dim": d,
                            "theta": config.theta, "interleaved": False})
        # [1,S,H*D] -> [H,S,D] puts each head in RoPE's batch axis.
        value = node(name + ".split", "Reshape", {"input": value}, [s, heads, d], {"shape": [s, heads, d]})
        value = node(name + ".heads", "Transpose", {"input": value}, [heads, s, d], {"perm": [1, 0, 2]})
        if norm_weight:
            value = norm(name + ".norm", value, norm_weight, [heads, s, d])
        value = node(name + ".rotate", "RoPE", {"input": value, "position_ids": "position_ids"},
                     [heads, s, d], {"rotary_dim": d, "theta": config.theta, "interleaved": False})
        value = node(name + ".sequence", "Transpose", {"input": value}, [s, heads, d], {"perm": [1, 0, 2]})
        return node(name, "Reshape", {"input": value}, [1, s, heads * d], {"shape": [1, s, heads * d]})

    value = node("embedding", "Embedding", {"input": "token_ids", "weight": "model.embed_tokens.weight"}, [1, s, h])
    repeat = config.query_heads // config.kv_heads
    if repeat != 1 and not compact_kv:
        weights["hf.repeat_kv.indices"] = np.array([
            (head // repeat) * d + feature for head in range(config.query_heads) for feature in range(d)], dtype=np.int32)
    for i in range(config.layers):
        p, name = f"model.layers.{i}.", f"layer.{i}"
        normalized = norm(name + ".attention_norm", value, p + "input_layernorm.weight", [1, s, h])
        q = linear(name + ".q", normalized, p + "self_attn.q_proj", config.query_heads * d, config.attention_bias)
        k = linear(name + ".k", normalized, p + "self_attn.k_proj", config.kv_heads * d, config.attention_bias)
        v = linear(name + ".v", normalized, p + "self_attn.v_proj", config.kv_heads * d, config.attention_bias)
        q = head_rope(name + ".q_rope", q, config.query_heads, p + "self_attn.q_norm.weight" if config.family == "qwen3" else None)
        k = head_rope(name + ".k_rope", k, config.kv_heads, p + "self_attn.k_norm.weight" if config.family == "qwen3" else None)
        if capture_kv:
            graph["outputs"].extend([k, v])
        if repeat != 1 and not compact_kv:
            k = node(name + ".repeat_k", "Gather", {"input": k, "indices": "hf.repeat_kv.indices"}, [1, s, config.query_heads * d], {"axis": 2})
            v = node(name + ".repeat_v", "Gather", {"input": v, "indices": "hf.repeat_kv.indices"}, [1, s, config.query_heads * d], {"axis": 2})
        attention = node(name + ".attention", "CrossSDPA", {"q": q, "k": k, "v": v, "mask": "attention_mask"},
                         [1, s, config.query_heads * d], {"heads": config.query_heads,
                         **({"kv_heads": config.kv_heads} if compact_kv else {}),
                         "causal": True, "scale": 1 / math.sqrt(d)})
        projected = linear(name + ".attention_output", attention, p + "self_attn.o_proj", h, config.output_attention_bias)
        value = node(name + ".attention_residual", "Add", {"a": value, "b": projected}, [1, s, h])
        normalized = norm(name + ".mlp_norm", value, p + "post_attention_layernorm.weight", [1, s, h])
        gate = linear(name + ".gate", normalized, p + "mlp.gate_proj", config.intermediate_size, config.mlp_bias)
        up = linear(name + ".up", normalized, p + "mlp.up_proj", config.intermediate_size, config.mlp_bias)
        gate = node(name + ".silu", "SiLU", {"input": gate}, [1, s, config.intermediate_size])
        product = node(name + ".gate_product", "Mul", {"a": gate, "b": up}, [1, s, config.intermediate_size])
        down = linear(name + ".down", product, p + "mlp.down_proj", h, config.mlp_bias)
        value = node(name + ".mlp_residual", "Add", {"a": value, "b": down}, [1, s, h])
    value = norm("final_norm", value, "model.norm.weight", [1, s, h])
    head = "model.embed_tokens.weight" if config.tied_embeddings else "lm_head.weight"
    node(output_name, "Linear", {"input": value, "weight": head}, [1, s, config.vocab_size], {"weight_layout": "dout_din"})
    if cached:
        if output_name == "generated":
            _fail("cached logits output conflicts with generated ID output")
        node("generated", "ArgMax", {"input": output_name}, [1, s],
             {"axis": -1, "keepdims": False})
        graph["nodes"][-1]["outputs"]["out"]["dtype"] = "int32"
        graph["outputs"].append("generated")
        graph["inputs"]["active_lengths"] = {"shape": [1], "dtype": "int32"}
        from .execution_contract import incremental_decoder_execution
        graph["execution"] = incremental_decoder_execution(
            graph["inputs"], graph["outputs"], token_input="token_ids",
            token_output="generated", keep_input="attention_mask",
            position_input="position_ids", logits_output=output_name,
            id_domain_size=config.vocab_size)
    else:
        graph["execution"] = stateless_execution(graph["inputs"], graph["outputs"])
    return graph, weights


_EXTENDED_FAMILIES = frozenset({"qwen3_moe", "llama4", "llama4_text", "qwen3_next",
    "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
    "qwen4_exp", "qwen4_exp_text"})
_DEEPSEEK_FAMILIES = frozenset({"deepseek_v2", "deepseek_v3", "deepseek_v4", "deepseek_v41", "deepseek_v41_text"})


def _family_module(family):
    if not isinstance(family, str):
        _fail("model_type must be a string")
    if family in _EXTENDED_FAMILIES:
        return importlib.import_module(".hf_extended", __package__)
    if family in ("deepseek_v41", "deepseek_v41_text"):
        return importlib.import_module(".hf_deepseek_v41", __package__)
    if family in _DEEPSEEK_FAMILIES:
        return importlib.import_module(".hf_deepseek", __package__)
    return None


def read_hf_checkpoint(directory: str | Path, *, config_validator=None):
    """Validate architecture before local weight IO; allow family-specific admission.

    Family modules can supply their own validator without duplicating the exact
    local-shard/storage checks or introducing a module-import cycle.
    """
    directory = Path(directory)
    source_config = _read_json(directory / "config.json")
    if config_validator is None:
        if source_config.get("model_type") == "whisper":
            from .hf_whisper import WhisperConfig
            config_validator = WhisperConfig.from_hf
    if config_validator is None:
        module = _family_module(source_config.get("model_type"))
        config_validator = (module.DeepSeekV41Config.from_hf if source_config.get("model_type") in ("deepseek_v41", "deepseek_v41_text") else module.DeepSeekDecoderConfig.from_hf
            if source_config.get("model_type") in _DEEPSEEK_FAMILIES else module.ExtendedDecoderConfig.from_hf
            ) if module else DecoderConfig.from_hf
    config_validator(source_config)
    return read_checkpoint_storage(directory, source_config=source_config)


def export_hf_decoder(directory, out_path, *, max_context=None, output_names=None, capture_kv=False,
                      compact_kv=False, cached=False, generate=False, image_features_count=None,
                      report_callback=None, weight_precision=None, group_size=128):
    source = _read_json(Path(directory) / "config.json")
    if source.get("model_type") == "whisper":
        if capture_kv or compact_kv or cached or image_features_count is not None:
            _fail("Whisper uses a separate encoder and whole-prefix decoder; KV/image options are unsupported")
        from .hf_whisper import export_hf_whisper
        return export_hf_whisper(directory, out_path, max_context=max_context,
            output_names=output_names, generate=generate, report_callback=report_callback,
            **({"weight_precision": weight_precision, "group_size": group_size} if weight_precision else {}))
    module = _family_module(source.get("model_type"))
    if module:
        if source.get("model_type") in _DEEPSEEK_FAMILIES and image_features_count is not None:
            _fail("DeepSeek text decoder does not accept image features")
        exporter = (module.export_hf_deepseek_v41 if source.get("model_type") in ("deepseek_v41", "deepseek_v41_text") else module.export_hf_deepseek if source.get("model_type") in _DEEPSEEK_FAMILIES
                    else module.export_hf_extended)
        return exporter(directory, out_path, max_context=max_context,
            output_names=output_names, capture_kv=capture_kv, compact_kv=compact_kv,
            cached=cached, report_callback=report_callback,
            **({"weight_precision": weight_precision, "group_size": group_size} if weight_precision else {}),
            **({"generate": True} if generate else {}),
            **({"image_features_count": image_features_count} if image_features_count is not None else {}))
    if image_features_count is not None:
        _fail("this dense decoder architecture does not accept image features")
    cached = cached or generate
    if generate:
        compact_kv = True
    preliminary = DecoderConfig.from_hf(source, max_context=max_context)
    if preliminary.table_rope:
        if max_context is None:
            _fail("scaled RoPE export requires an explicit max_context for its bounded position tables")
        if not compact_kv:
            _fail("table-based scaled RoPE requires compact_kv; materialized-repeat lowering is unsupported")
    source, source_weights, source_dtypes = read_hf_checkpoint(directory)
    config = DecoderConfig.from_hf(source, max_context=max_context)
    if output_names is not None and len(output_names) != 1:
        _fail("HF decoder export has exactly one logits output")
    graph, weights = build_decoder_graph(config, source_weights, output_name=output_names[0] if output_names else "logits",
        capture_kv=capture_kv, compact_kv=compact_kv, cached=cached)
    graph = write_hf_package(out_path, graph, weights, weight_precision=weight_precision, group_size=group_size)
    if report_callback:
        report_callback({"features": {"hf_decoder": [f"family:{config.family}", "execution:F32",
                            "batch:1", "attention:compact-kv" if compact_kv else "attention:materialized-repeat-kv",
                            "state:incremental-greedy" if cached else "state:stateless", f"context-bound:{config.max_context}"]
                           + (["rope:llama3-shared-tables", f"absolute-position-range:0..{config.max_context-1}"]
                              if config.llama3_scaling else [])
                           + ([f"rope:qwen2-{config.qwen2_scaling[0]}-shared-tables",
                               f"absolute-position-range:0..{config.max_context-1}"] if config.qwen2_scaling else []),
                         "source_storage": sorted(set(source_dtypes.values())),
                         "kv_capture": ([f"layer.{i}.k_rope:K:1,S,Hkv*D:after-QKnorm-RoPE;layer.{i}.v:V:1,S,Hkv*D:raw-projection"
                                          for i in range(config.layers)] if capture_kv else [])}})
    return graph


def import_hf_decoder(directory, *, max_context=None, compact_kv=False, cached=False):
    config, weights, _ = read_hf_checkpoint(directory)
    graph, initializers = build_decoder_graph(DecoderConfig.from_hf(config, max_context=max_context), weights,
                                              compact_kv=compact_kv, cached=cached)
    return import_runtime_package(graph, initializers), initializers

