"""DeepSeek V2/V3 MLA and V4 mHC/compressed-attention prefix graphs.

Checkpoint storage is handled by the shared local reader. The graph executes
F32 values through common operators. Checkpoint dequantization and native
low-precision execution have separate, explicitly reported qualification scopes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .hf_graph import (GraphBuilder, positive_float as _positive_float, positive_integer as _positive_integer,
    read_config as _read_json, read_checkpoint_storage, write_hf_package)
from .runtime_names import is_runtime_graph_name


def _fail(message, code="VXHF_CONFIG"):
    raise ExporterError(Diagnostic(code, message, "hf-deepseek"))


def _integer(value, name, *, zero=False):
    if zero and value == 0 and isinstance(value, int) and not isinstance(value, bool):
        return 0
    return _positive_integer(value, name)


def _boolean(source, name, default=False):
    value = source.get(name, default)
    if not isinstance(value, bool):
        _fail(f"{name} must be boolean")
    return value


def _rope_parameters(source, *, theta, rotary, kind=None):
    parameters = source.get("rope_parameters") or source.get("rope_scaling") or {}
    if not isinstance(parameters, Mapping):
        _fail("RoPE parameters must be an object")
    if source.get("rope_parameters") and source.get("rope_scaling"):
        _fail("conflicting rope_parameters and rope_scaling")
    if kind and isinstance(parameters.get(kind), Mapping):
        parameters = parameters[kind]
    result = dict(parameters)
    rope_type = result.get("rope_type", result.get("type", "default"))
    if rope_type not in ("default", "linear", "yarn"):
        _fail("DeepSeek prefix import supports fixed default, linear or YaRN RoPE")
    if "type" in result and "rope_type" in result and result["type"] != result["rope_type"]:
        _fail("conflicting RoPE types")
    allowed = {"rope_type", "type", "rope_theta", "partial_rotary_factor"}
    if rope_type != "default":
        allowed.add("factor")
    if rope_type == "yarn":
        allowed |= {"original_max_position_embeddings", "attention_factor", "beta_fast", "beta_slow",
                    "mscale", "mscale_all_dim", "truncate"}
    if set(result) - allowed:
        _fail(f"unsupported RoPE parameters: {sorted(set(result) - allowed)}")
    result["rope_type"] = rope_type
    result["rope_theta"] = _positive_float(result.get("rope_theta", theta), "rope_theta")
    if rope_type != "default":
        factor = _positive_float(result.get("factor"), "RoPE factor")
        if factor < 1:
            _fail("RoPE factor must be >= 1")
        result["factor"] = factor
    if rope_type == "yarn":
        if result["rope_theta"] <= 1:
            _fail("YaRN theta must exceed 1")
        result["original_max_position_embeddings"] = _positive_integer(
            source.get("original_max_position_embeddings", result.get("original_max_position_embeddings",
                source.get("max_position_embeddings"))), "original_max_position_embeddings")
        for name, default in (("beta_fast", 32), ("beta_slow", 1)):
            result[name] = _positive_float(default if result.get(name) is None else result[name], name)
        if result["beta_fast"] < result["beta_slow"]:
            _fail("YaRN beta_fast must be >= beta_slow")
        if not isinstance(result.get("truncate", True), bool):
            _fail("YaRN truncate must be boolean")
        for name in ("mscale", "mscale_all_dim", "attention_factor"):
            if result.get(name) is not None:
                value = result[name]
                if name != "attention_factor" and value == 0 and not isinstance(value, bool):
                    result[name] = 0.0
                else:
                    result[name] = _positive_float(value, name)
    result["rotary_dim"] = rotary
    return result


@dataclass(frozen=True)
class DeepSeekDecoderConfig:
    family: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    layers: int
    heads: int
    max_context: int
    epsilon: float
    tied_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    q_lora_rank: int | None
    kv_lora_rank: int
    nope_dim: int
    rotary_dim: int
    value_dim: int
    interleaved: bool
    experts: int
    top_k: int
    shared_experts: int
    groups: int
    topk_groups: int
    normalize_routes: bool
    routed_scale: float
    scoring: str
    first_dense: int
    layer_types: tuple[str, ...]
    mlp_types: tuple[str, ...]
    rates: tuple[int, ...]
    rope_main: Mapping[str, Any]
    rope_compress: Mapping[str, Any]
    head_dim: int = 1
    streams: int = 1
    sinkhorn_iterations: int = 1
    hc_epsilon: float = 1e-6
    window: int = 1
    output_groups: int = 1
    output_rank: int = 1
    index_heads: int = 1
    index_dim: int = 1
    index_top_k: int = 1
    swiglu_limit: float | None = None
    auxiliary_layers: int = 0
    source_quantization: Mapping[str, Any] | None = None

    @classmethod
    def from_hf(cls, source, *, max_context=None):
        family = source.get("model_type")
        if family not in ("deepseek_v2", "deepseek_v3", "deepseek_v4"):
            _fail("DeepSeek requires model_type deepseek_v2, deepseek_v3 or deepseek_v4")
        architecture = {"deepseek_v2": "DeepseekV2ForCausalLM", "deepseek_v3": "DeepseekV3ForCausalLM",
                        "deepseek_v4": "DeepseekV4ForCausalLM"}[family]
        if source.get("architectures") not in (None, [architecture]):
            _fail(f"architectures must identify {architecture}")
        if source.get("hidden_act", "silu") != "silu" or source.get("pretraining_tp", 1) != 1:
            _fail("DeepSeek import requires SiLU and pretraining_tp=1")
        for name in ("is_encoder_decoder", "router_jitter_noise", "attn_logit_softcapping", "final_logit_softcapping"):
            if source.get(name) not in (None, False, 0):
                _fail(f"{name} is unsupported")
        quantization = source.get("quantization_config")
        if quantization is not None and not isinstance(quantization, Mapping):
            _fail("quantization_config must be an object or null")
        integer = lambda name, default=None: _positive_integer(source.get(name, default), name)
        layers, hidden, heads = integer("num_hidden_layers"), integer("hidden_size"), integer("num_attention_heads")
        maximum = integer("max_position_embeddings")
        bound = maximum if max_context is None else _positive_integer(max_context, "max_context")
        if bound > maximum:
            _fail("max_context exceeds the source context")
        is_v4 = family == "deepseek_v4"
        q_lora = source.get("q_lora_rank", 1024 if is_v4 else 1536)
        if q_lora is not None:
            q_lora = _positive_integer(q_lora, "q_lora_rank")
        if is_v4 and q_lora is None:
            _fail("V4 requires q_lora_rank")
        head_dim = integer("head_dim", 512) if is_v4 else 1
        fraction = source.get("partial_rotary_factor")
        fraction = 0.125 if fraction is None else _positive_float(fraction, "partial_rotary_factor")
        rotary = source.get("qk_rope_head_dim", int(head_dim * _positive_float(fraction, "partial_rotary_factor"))) if is_v4 else source.get("qk_rope_head_dim", 64)
        rotary = _positive_integer(rotary, "qk_rope_head_dim")
        if rotary % 2 or (is_v4 and rotary > head_dim):
            _fail("the rotary width must be positive, even and fit its head")
        if is_v4 and source.get("partial_rotary_factor") is not None and int(head_dim * fraction) != rotary:
            _fail("partial_rotary_factor and qk_rope_head_dim disagree")
        kv_heads = source.get("num_key_value_heads", 1 if is_v4 else heads)
        kv_heads = heads if kv_heads is None and not is_v4 else kv_heads
        if kv_heads != (1 if is_v4 else heads):
            _fail("DeepSeek requires shared-KV MQA for V4 and expanded MLA heads for V2/V3")
        intermediate = integer("moe_intermediate_size", 2048 if is_v4 else 1407 if family == "deepseek_v2" else 2048)
        dense = intermediate if is_v4 else integer("intermediate_size", 11008 if family == "deepseek_v2" else 18432)
        experts = integer("n_routed_experts", 64 if family == "deepseek_v2" else 256)
        top = integer("num_experts_per_tok", 6 if is_v4 else 8 if family == "deepseek_v3" else 2)
        shared = integer("n_shared_experts", 2 if family == "deepseek_v2" else 1)
        if top > experts or (is_v4 and shared != 1):
            _fail("selected experts exceed the bank, or V4 shared_experts differs from its fixed shared MLP")
        first = _integer(source.get("first_k_dense_replace", 3 if family == "deepseek_v3" else 0), "first_k_dense_replace", zero=True)
        if source.get("moe_layer_freq", 1) != 1:
            _fail("only the declared all-MoE tail is supported")
        method = source.get("topk_method", "greedy" if family == "deepseek_v2" else "noaux_tc")
        if method not in (("greedy", "group_limited_greedy") if family == "deepseek_v2" else ("noaux_tc",)):
            _fail("unsupported DeepSeek expert routing method")
        groups = integer("n_group", 8) if family == "deepseek_v3" or method == "group_limited_greedy" else 1
        selected = integer("topk_group", 4) if groups != 1 else 1
        group_score = 2 if family == "deepseek_v3" else 1
        if experts % groups or selected > groups or experts // groups < group_score or top > selected * (experts // groups):
            _fail("expert groups cannot provide the selected top-k bank")
        scoring = source.get("scoring_func", "softmax" if family == "deepseek_v2" else "sqrtsoftplus" if is_v4 else "sigmoid")
        if scoring not in (("softmax",) if family == "deepseek_v2" else ("sigmoid",) if not is_v4 else ("sqrtsoftplus", "sigmoid", "softmax")):
            _fail("unsupported architecture-specific scoring function")
        normalize = False if family == "deepseek_v2" else _boolean(source, "norm_topk_prob", True)
        if is_v4:
            # Both upstream V4 routers normalize unconditionally; the inherited
            # metadata flag does not change their arithmetic.
            normalize = True
        attention_bias, mlp_bias = _boolean(source, "attention_bias"), _boolean(source, "mlp_bias")
        if (is_v4 and attention_bias) or (family == "deepseek_v3" and mlp_bias):
            _fail("this DeepSeek attention/MLP architecture is bias-free")
        auxiliary = _integer(source.get("num_nextn_predict_layers", source.get("num_mtp_layers", 0)), "num_nextn_predict_layers", zero=True)
        theta = _positive_float(source.get("rope_theta", 10000), "rope_theta")
        main = {} if is_v4 else _rope_parameters(source, theta=theta, rotary=rotary)
        types, mlp_types, rates = (), (), ()
        extra = {}
        if is_v4:
            supplied_rope = source.get("rope_parameters") or {}
            if isinstance(supplied_rope.get("main"), Mapping) and isinstance(supplied_rope.get("compress"), Mapping):
                main = _rope_parameters(source, theta=theta, rotary=rotary, kind="main")
                compress = _rope_parameters(source, theta=theta, rotary=rotary, kind="compress")
            else:
                compress = _rope_parameters(source, theta=_positive_float(source.get("compress_rope_theta", 160000), "compress_rope_theta"), rotary=rotary)
                main = {"rope_type": "default", "rope_theta": theta, "rotary_dim": rotary}
            if compress["rope_type"] == "yarn":
                compress.setdefault("attention_factor", 1.0)
            raw_types = source.get("layer_types")
            if raw_types is None and source.get("compress_ratios") is not None:
                ratios = source["compress_ratios"]
                if not isinstance(ratios, list) or any(r not in (0, 4, 128) for r in ratios):
                    _fail("compress_ratios must contain legacy 0/4/128 entries")
                # DSpark releases carry three explicit draft stages even when
                # the inherited num_nextn_predict_layers metadata remains one.
                if source.get("dspark_target_layer_ids") is not None:
                    targets = source["dspark_target_layer_ids"]
                    if (not isinstance(targets, list) or len(targets) != 3 or
                            any(type(i) is not int or not 0 <= i < layers for i in targets) or
                            len(ratios) != layers + 3 or any(ratios[layers:])):
                        _fail("DSpark requires three declared zero-compression draft stages")
                    auxiliary = 3
                raw_types = [{0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}[r] for r in ratios]
            if raw_types is None:
                raw_types = ["heavily_compressed_attention"] * min(layers, 2) + [
                    "heavily_compressed_attention" if i % 2 == 0 else "compressed_sparse_attention" for i in range(max(layers - 2, 0))]
            if not isinstance(raw_types, (list, tuple)) or len(raw_types) < layers:
                _fail("V4 requires an attention type for every layer")
            types = tuple(raw_types[:layers])
            rate_map = source.get("compress_rates") or {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
            if not isinstance(rate_map, Mapping) or set(rate_map) - {"compressed_sparse_attention", "heavily_compressed_attention"}:
                _fail("invalid V4 compression rate map")
            rate_map = {**rate_map}
            for kind, field, default in (("compressed_sparse_attention", "compress_rate_csa", 4), ("heavily_compressed_attention", "compress_rate_hca", 128)):
                rate_map[kind] = _positive_integer(source.get(field, rate_map.get(kind, default)), field)
            if any(kind not in (*rate_map, "sliding_attention") for kind in types):
                _fail("unsupported V4 attention layer type")
            rates = tuple(0 if kind == "sliding_attention" else rate_map[kind] for kind in types)
            raw_mlp = source.get("mlp_layer_types")
            hashes = _integer(source.get("num_hash_layers", 3), "num_hash_layers", zero=True)
            raw_mlp = ["hash_moe"] * min(layers, hashes) + ["moe"] * max(0, layers - hashes) if raw_mlp is None else raw_mlp
            if not isinstance(raw_mlp, (list, tuple)) or len(raw_mlp) < layers or any(x not in ("hash_moe", "moe") for x in raw_mlp[:layers]):
                _fail("V4 requires hash_moe or moe for every layer")
            mlp_types = tuple(raw_mlp[:layers])
            output_groups = integer("o_groups", 8)
            if heads % output_groups:
                _fail("o_groups must divide the head count")
            index_dim = integer("index_head_dim", 128)
            if rotary > index_dim and "compressed_sparse_attention" in types:
                _fail("rotary width exceeds the indexer head")
            extra = dict(head_dim=head_dim, streams=integer("hc_mult", 4),
                sinkhorn_iterations=integer("hc_sinkhorn_iters", 20), hc_epsilon=_positive_float(source.get("hc_eps", 1e-6), "hc_eps"),
                window=integer("sliding_window", 128), output_groups=output_groups,
                output_rank=integer("o_lora_rank", 1024), index_heads=integer("index_n_heads", 64),
                index_dim=index_dim, index_top_k=integer("index_topk", 512),
                swiglu_limit=_positive_float(source.get("swiglu_limit", 10), "swiglu_limit"))
        else:
            compress = main
        return cls(family, integer("vocab_size"), hidden, dense, intermediate, layers, heads, bound,
            _positive_float(source.get("rms_norm_eps", 1e-6), "rms_norm_eps"), _boolean(source, "tie_word_embeddings"),
            attention_bias, mlp_bias, q_lora, 1 if is_v4 else integer("kv_lora_rank", 512),
            1 if is_v4 else integer("qk_nope_head_dim", 128), rotary, head_dim if is_v4 else integer("v_head_dim", 128),
            True if is_v4 or family == "deepseek_v2" else _boolean(source, "rope_interleave", True),
            experts, top, shared, groups, selected, normalize, _positive_float(source.get("routed_scaling_factor", 1 if family == "deepseek_v2" else 1.5 if is_v4 else 2.5), "routed_scaling_factor"),
            scoring, first, types, mlp_types, rates, main, compress, auxiliary_layers=auxiliary,
            source_quantization=dict(quantization) if quantization else None, **extra)

    def weight_shapes(self, *, expert_layout="packed"):
        h, n, r = self.hidden_size, self.heads, self.rotary_dim
        result = {"model.embed_tokens.weight": (self.vocab_size, h), "model.norm.weight": (h,)}
        if not self.tied_embeddings:
            result["lm_head.weight"] = (self.vocab_size, h)
        if self.family == "deepseek_v4":
            m = self.streams
            result.update({"model.hc_head.hc_fn": (m, m * h), "model.hc_head.hc_base": (m,), "model.hc_head.hc_scale": (1,)})
        for layer in range(self.layers):
            p = f"model.layers.{layer}."
            result[p + "input_layernorm.weight"] = result[p + "post_attention_layernorm.weight"] = (h,)
            a = p + "self_attn."
            if self.family == "deepseek_v4":
                d, g, o = self.head_dim, self.output_groups, self.output_rank
                result.update({a + "q_a_proj.weight": (self.q_lora_rank, h), a + "q_a_norm.weight": (self.q_lora_rank,),
                    a + "q_b_proj.weight": (n * d, self.q_lora_rank), a + "kv_proj.weight": (d, h), a + "kv_norm.weight": (d,),
                    a + "o_a_proj.weight": (g * o, n * d // g), a + "o_b_proj.weight": (h, g * o), a + "sinks": (n,)})
                if self.rates[layer]:
                    groups = 2 if self.layer_types[layer] == "compressed_sparse_attention" else 1
                    c = a + "compressor."
                    result.update({c + "kv_proj.weight": (groups * d, h), c + "gate_proj.weight": (groups * d, h),
                        c + "position_bias": (self.rates[layer], groups * d), c + "kv_norm.weight": (d,)})
                    if groups == 2:
                        c += "indexer."
                        di = self.index_dim
                        result.update({c + "kv_proj.weight": (2 * di, h), c + "gate_proj.weight": (2 * di, h),
                            c + "position_bias": (self.rates[layer], 2 * di), c + "kv_norm.weight": (di,),
                            c + "q_b_proj.weight": (self.index_heads * di, self.q_lora_rank),
                            c + "scorer.weights_proj.weight": (self.index_heads, h)})
                m = self.streams
                for name in ("attn_hc.", "ffn_hc."):
                    result.update({p + name + "fn": (2 * m + m * m, m * h),
                        p + name + "base": (2 * m + m * m,), p + name + "scale": (3,)})
            else:
                qd, v = self.nope_dim + r, self.value_dim
                if self.q_lora_rank is None:
                    result[a + "q_proj.weight"] = (n * qd, h)
                else:
                    result.update({a + "q_a_proj.weight": (self.q_lora_rank, h), a + "q_a_layernorm.weight": (self.q_lora_rank,),
                        a + "q_b_proj.weight": (n * qd, self.q_lora_rank)})
                    if self.attention_bias:
                        result[a + "q_a_proj.bias"] = (self.q_lora_rank,)
                result.update({a + "kv_a_proj_with_mqa.weight": (self.kv_lora_rank + r, h),
                    a + "kv_a_layernorm.weight": (self.kv_lora_rank,), a + "kv_b_proj.weight": (n * (self.nope_dim + v), self.kv_lora_rank),
                    a + "o_proj.weight": (h, n * v)})
                if self.attention_bias:
                    result[a + "kv_a_proj_with_mqa.bias"] = (self.kv_lora_rank + r,)
                    result[a + "o_proj.bias"] = (h,)
            p += "mlp."
            if self.family == "deepseek_v4" or layer >= self.first_dense:
                result[p + "gate.weight"] = (self.experts, h)
                if self.family == "deepseek_v4" and self.mlp_types[layer] == "hash_moe":
                    result[p + "gate.tid2eid"] = (self.vocab_size, self.top_k)
                elif self.family != "deepseek_v2":
                    result[p + "gate.e_score_correction_bias"] = (self.experts,)
                if expert_layout == "packed":
                    result[p + "experts.gate_up_proj"] = (self.experts, 2 * self.moe_intermediate_size, h)
                    result[p + "experts.down_proj"] = (self.experts, h, self.moe_intermediate_size)
                elif expert_layout == "individual":
                    for expert in range(self.experts):
                        for name, shape in (("gate_proj", (self.moe_intermediate_size, h)), ("up_proj", (self.moe_intermediate_size, h)), ("down_proj", (h, self.moe_intermediate_size))):
                            result[p + f"experts.{expert}." + name + ".weight"] = shape
                else:
                    _fail("unknown expert checkpoint layout", "VXHF_WEIGHTS")
                self._mlp_shapes(result, p + "shared_experts.", self.moe_intermediate_size * self.shared_experts)
            else:
                self._mlp_shapes(result, p, self.intermediate_size)
        return result

    def _mlp_shapes(self, result, prefix, width):
        for name, shape in (("gate_proj", (width, self.hidden_size)), ("up_proj", (width, self.hidden_size)), ("down_proj", (self.hidden_size, width))):
            result[prefix + name + ".weight"] = shape
            if self.mlp_bias:
                result[prefix + name + ".bias"] = (shape[0],)


def _v4_source_name(name):
    """Translate one canonical V4 tensor to the official inference namespace."""
    roots = {"model.embed_tokens.weight": "embed.weight", "model.norm.weight": "norm.weight",
        "lm_head.weight": "head.weight", "model.hc_head.hc_fn": "hc_head_fn",
        "model.hc_head.hc_base": "hc_head_base", "model.hc_head.hc_scale": "hc_head_scale"}
    if name in roots:
        return roots[name]
    name = name.removeprefix("model.")
    for old, new in ((".input_layernorm.", ".attn_norm."), (".post_attention_layernorm.", ".ffn_norm."),
            (".attn_hc.fn", ".hc_attn_fn"), (".attn_hc.base", ".hc_attn_base"), (".attn_hc.scale", ".hc_attn_scale"),
            (".ffn_hc.fn", ".hc_ffn_fn"), (".ffn_hc.base", ".hc_ffn_base"), (".ffn_hc.scale", ".hc_ffn_scale"),
            (".self_attn.compressor.indexer.scorer.weights_proj.", ".attn.indexer.weights_proj."),
            (".self_attn.compressor.indexer.q_b_proj.", ".attn.indexer.wq_b."),
            (".self_attn.compressor.indexer.", ".attn.indexer.compressor."),
            (".self_attn.", ".attn."), (".mlp.", ".ffn."),
            (".q_a_proj.", ".wq_a."), (".q_b_proj.", ".wq_b."), (".q_a_norm.", ".q_norm."),
            (".kv_proj.", ".wkv."), (".gate_proj.", ".w1."), (".up_proj.", ".w3."), (".down_proj.", ".w2."),
            (".o_a_proj.", ".wo_a."), (".o_b_proj.", ".wo_b."),
            (".compressor.w1.", ".compressor.wgate."), (".compressor.kv_norm.", ".compressor.norm."),
            (".position_bias", ".ape"), (".sinks", ".attn_sink"), (".gate.e_score_correction_bias", ".gate.bias")):
        name = name.replace(old, new)
    return name


def _checked_weights(config, source):
    if config.family == "deepseek_v4" and "embed.weight" in source:
        if any(name.startswith("model.") or name == "lm_head.weight" for name in source):
            _fail("official and canonical V4 tensor namespaces cannot be mixed", "VXHF_WEIGHTS")
        expected = config.weight_shapes(expert_layout="individual")
        mapping = {_v4_source_name(name): name for name in expected}
        if config.tied_embeddings:
            mapping["head.weight"] = "lm_head.weight"
        translated = {}
        for name, value in source.items():
            target = mapping.get(name, name)
            if target in translated:
                _fail(f"ambiguous official/canonical tensor {target}", "VXHF_WEIGHTS")
            translated[target] = value
        source = translated
    packed = any(name.endswith(".experts.gate_up_proj") for name in source)
    expected = config.weight_shapes(expert_layout="packed" if packed else "individual")
    extras = set(source) - set(expected)
    if config.tied_embeddings:
        extras.discard("lm_head.weight")
    # Next-token prediction modules do not belong to the causal decoder stack.
    # Only the source-declared auxiliary layer interval is omitted.
    omitted = set()
    for name in extras:
        parts = name.split(".")
        if config.auxiliary_layers and (len(parts) > 2 and parts[0] == "mtp" and parts[1].isdigit()
                and int(parts[1]) < config.auxiliary_layers or
                len(parts) > 3 and parts[:2] == ["model", "layers"] and parts[2].isdigit()
                and config.layers <= int(parts[2]) < config.layers + config.auxiliary_layers):
            omitted.add(name)
    extras -= omitted
    missing = set(expected) - set(source)
    if missing or extras:
        _fail(f"checkpoint tensor namespace mismatch; missing={sorted(missing)}, extra={sorted(extras)}", "VXHF_WEIGHTS")
    result = {}
    for name, shape in expected.items():
        value = np.asarray(source[name])
        if name.endswith(".gate.tid2eid"):
            if value.dtype not in (np.dtype("int32"), np.dtype("int64")) or value.shape != shape or np.any(value < 0) or np.any(value >= config.experts):
                _fail(f"hash expert table {name} must have integer shape {shape} and IDs inside the expert bank", "VXHF_WEIGHTS")
            result[name] = np.ascontiguousarray(value, dtype=np.int32)
            continue
        if value.dtype not in (np.dtype("float32"), np.dtype("float16")) or value.shape != shape or not np.isfinite(value).all():
            _fail(f"tensor {name} must have finite floating shape {shape}; received {value.dtype} {value.shape}", "VXHF_WEIGHTS")
        result[name] = np.ascontiguousarray(value, dtype=np.float32)
    if config.tied_embeddings and "lm_head.weight" in source:
        value = np.asarray(source["lm_head.weight"])
        embedding = result["model.embed_tokens.weight"]
        if value.dtype not in (np.dtype("float32"), np.dtype("float16")) or value.shape != embedding.shape or not np.array_equal(value.astype(np.float32), embedding):
            _fail("tied lm_head must equal the embedding matrix", "VXHF_WEIGHTS")
    return result


def _rotary_tables(config, parameters, *, interleaved, rate=1):
    r = config.rotary_dim
    if config.max_context * r * 8 > 1 << 30:
        _fail("bounded rotary initializers exceed the 1GiB derived-table limit")
    theta = np.float32(parameters["rope_theta"])
    inverse = np.float32(1) / np.power(theta, np.arange(0, r, 2, dtype=np.float32) / np.float32(r))
    attention = np.float32(1)
    kind = parameters["rope_type"]
    if kind == "linear":
        inverse /= np.float32(parameters["factor"])
    elif kind == "yarn":
        factor = parameters["factor"]
        original = parameters["original_max_position_embeddings"]
        def correction(rotations):
            return r * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(float(theta)))
        low, high = correction(parameters["beta_fast"]), correction(parameters["beta_slow"])
        if parameters.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, r - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(r // 2, dtype=np.float32) - low) / (high - low), 0, 1)
        inverse = inverse * (1 - ramp) + inverse / np.float32(factor) * ramp
        def mscale(value):
            return 1 if factor <= 1 else 0.1 * value * math.log(factor) + 1
        attention = np.float32(parameters.get("attention_factor",
            mscale(parameters.get("mscale", 1.0)) / mscale(parameters.get("mscale_all_dim", 0.0))))
    if not np.isfinite(inverse).all() or not np.isfinite(attention):
        _fail("derived rotary frequencies and scaling must remain finite F32")
    phase = (np.arange(config.max_context, dtype=np.float32) * np.float32(rate))[:, None] * inverse[None]
    if not interleaved:
        phase = np.concatenate((phase, phase), axis=1)
    cosine, sine = np.cos(phase) * attention, np.sin(phase) * attention
    if not np.isfinite(cosine).all() or not np.isfinite(sine).all():
        _fail("derived rotary position tables must remain finite F32")
    return np.ascontiguousarray(cosine), np.ascontiguousarray(sine)


def add_deepseek_dense_mlp(b, config, weights, name, x, prefix, width):
    h, s = config.hidden_size, "S"
    def linear(name, x, prefix, width):
        return b.linear(name, x, prefix + ".weight", width,
            prefix + ".bias" if prefix + ".bias" in weights else None)
    gate = linear(name + ".gate", x, prefix + "gate_proj", width)
    up = linear(name + ".up", x, prefix + "up_proj", width)
    if config.swiglu_limit is not None:
        gate = b.tensor(name + ".gate.clip", "Clip", {"input": gate}, [1, s, width],
            {"max": config.swiglu_limit})
        up = b.tensor(name + ".up.clip", "Clip", {"input": up}, [1, s, width],
            {"min": -config.swiglu_limit, "max": config.swiglu_limit})
    activated = b.tensor(name + ".silu", "SiLU", {"input": gate}, [1, s, width])
    mixed = b.tensor(name + ".mul", "Mul", {"a": activated, "b": up}, [1, s, width])
    return linear(name + ".down", mixed, prefix + "down_proj", h)


def add_deepseek_moe(b, config, weights, name, x, prefix, layer):
    h, s = config.hidden_size, "S"
    hs = [1, s, h]
    m = config.moe_intermediate_size
    if prefix + "experts.gate_up_proj" in weights:
        banks = weights[prefix + "experts.gate_up_proj"]
        gate, up = banks[:, :m, :].transpose(0, 2, 1), banks[:, m:, :].transpose(0, 2, 1)
        down = weights[prefix + "experts.down_proj"].transpose(0, 2, 1)
    else:
        gate, up, down = [np.stack([weights[prefix + f"experts.{i}." + kind + ".weight"].T
            for i in range(config.experts)]) for kind in ("gate_proj", "up_proj", "down_proj")]
    router_inputs = {"input": x, "weight": b.add(name + ".router.weight", weights[prefix + "gate.weight"].T)}
    router_params = {"num_experts": config.experts, "top_k": config.top_k,
        "temperature": 1.0, "normalize": config.normalize_routes,
        "scoring_func": config.scoring, "num_groups": config.groups,
        "topk_groups": config.topk_groups, "group_score_top_k": 2 if config.family == "deepseek_v3" else 1,
        "routed_scaling_factor": config.routed_scale, "normalization_epsilon": 1e-20}
    if prefix + "gate.e_score_correction_bias" in weights:
        router_inputs["correction_bias"] = prefix + "gate.e_score_correction_bias"
    if prefix + "gate.tid2eid" in weights:
        router_inputs.update(token_ids="token_ids", hash_indices=prefix + "gate.tid2eid")
    routes = b.multi(name + ".routes", "MoERouter", router_inputs,
        {"indices": (name + ".indices", [1, s, config.top_k], "float32"),
         "weights": (name + ".weights", [1, s, config.top_k], "float32")}, router_params)
    mixed = b.tensor(name + ".experts", "MoEGatedMLP",
        {"input": x, "route_indices": routes["indices"], "route_weights": routes["weights"],
         "gate_weight": b.add(name + ".gate.bank", gate),
         "up_weight": b.add(name + ".up.bank", up),
         "down_weight": b.add(name + ".down.bank", down)}, hs,
        {"route_input_scale": False, **({"swiglu_limit": config.swiglu_limit} if config.swiglu_limit is not None else {})})
    shared = add_deepseek_dense_mlp(b, config, weights, name + ".shared", x, prefix + "shared_experts.", m * config.shared_experts)
    return b.tensor(name + ".sum", "Add", {"a": mixed, "b": shared}, hs)


def collapse_streams(b, name, hidden, pre, streams, width):
    """Apply mHC pre coefficients separately from producing the next mix."""
    shape = [1, "S", streams, width]
    expanded = b.reshape(name + ".pre.rows", pre, [1, "S", streams, 1])
    expanded = b.tensor(name + ".pre.expand", "Expand", {"input": expanded}, shape, {"shape": shape})
    product = b.tensor(name + ".product", "Mul", {"a": hidden, "b": expanded}, shape)
    # Reduction kernels operate on the last axis; transpose streams there.
    product = b.tensor(name + ".stream_axis", "Transpose", {"input": product},
        [1, "S", width, streams], {"perm": [0, 1, 3, 2]})
    return b.tensor(name, "ReduceSum", {"input": product}, [1, "S", width], {"axis": -1, "keepdims": False})


def build_deepseek_decoder_graph(config, source_weights, *, output_name="logits",
                                 capture_kv=False, compact_kv=True, cached=False, generate=False):
    """Lower a causal prefix or explicit retained-row incremental execution."""
    if not isinstance(cached, bool) or not isinstance(generate, bool):
        _fail("cached and generate must be boolean")
    if cached and capture_kv:
        _fail("cached execution requires private retained KV/compressor tensors")
    if not isinstance(output_name, str) or not is_runtime_graph_name(output_name):
        _fail("output_name must be a portable runtime tensor name")
    weights = _checked_weights(config, source_weights)
    outputs = [output_name]
    if capture_kv:
        outputs += [f"layer.{i}.{port}" for i in range(config.layers) for port in ("k_rope", "v")]
    b = GraphBuilder(config.max_context, outputs)
    for name, value in weights.items():
        b.add(name, value)
    h, n, r = config.hidden_size, config.heads, config.rotary_dim
    hs, s = [1, "S", h], "S"

    def linear(name, x, prefix, width, shape=None):
        return b.linear(name, x, prefix + ".weight", width,
            prefix + ".bias" if prefix + ".bias" in weights else None, shape)

    def gather(name, x, indices, axis, shape):
        index = b.add(name + ".indices", np.asarray(indices, dtype=np.int32))
        return b.tensor(name, "Gather", {"input": x, "indices": index}, shape, {"axis": axis})

    def concat(name, x, y, axis, shape):
        return b.tensor(name, "Concat2", {"a": x, "b": y}, shape, {"axis": axis})

    rotary_cache = {}
    def table_pair(key, parameters, interleaved, rate=1):
        cache_key = (key, interleaved, rate)
        if cache_key not in rotary_cache:
            cos, sin = _rotary_tables(config, parameters, interleaved=interleaved, rate=rate)
            rotary_cache[cache_key] = (b.add(key + f".{rate}.cos", cos), b.add(key + f".{rate}.sin", sin))
        return rotary_cache[cache_key]

    def query_pair(key, parameters, interleaved):
        cos, sin = table_pair(key, parameters, interleaved)
        width = r // 2 if interleaved else r
        return tuple(b.reshape(key + "." + suffix + ".batch",
            b.tensor(key + "." + suffix + ".lookup", "Embedding",
                {"input": "position_ids", "weight": table}, [s, width]), [1, s, width])
            for suffix, table in (("cos", cos), ("sin", sin)))

    # V3 interleaved checkpoints rotate even/odd source channels into a
    # half-packed result. Reorder that source slice, then use half-split math.
    def rotary(name, x, heads, width, pair, *, interleaved, offset, repack=False):
        if repack:
            order = list(range(offset)) + list(range(offset, offset + r, 2)) + list(range(offset + 1, offset + r, 2)) + list(range(offset + r, width))
            x = gather(name + ".packing", x, order, 3, [1, s, heads, width])
        return b.tensor(name, "RotaryEmbedding", {"input": x, "cos": pair[0], "sin": pair[1]},
            [1, s, heads, width], {"rotary_dim": r, "position_offset": offset, "interleaved": interleaved})

    def dense_mlp(name, x, prefix, width):
        return add_deepseek_dense_mlp(b, config, weights, name, x, prefix, width)

    def moe(name, x, prefix, layer):
        return add_deepseek_moe(b, config, weights, name, x, prefix, layer)

    pairs = {}
    def get_pair(which):
        if which not in pairs:
            params = config.rope_main if which == "main" else config.rope_compress
            pairs[which] = query_pair("hf.deepseek." + which, params, config.family in ("deepseek_v2", "deepseek_v4"))
        return pairs[which]

    def mla(name, x, prefix, layer):
        qd, vd = config.nope_dim + r, config.value_dim
        if config.q_lora_rank is None:
            q = linear(name + ".q", x, prefix + "q_proj", n * qd)
        else:
            q = linear(name + ".qa", x, prefix + "q_a_proj", config.q_lora_rank)
            q = b.norm(name + ".qa.norm", q, prefix + "q_a_layernorm.weight", [1, s, config.q_lora_rank], 1e-6)
            q = linear(name + ".qb", q, prefix + "q_b_proj", n * qd)
        q = b.reshape(name + ".q.heads", q, [1, s, n, qd])
        q = rotary(name + ".q.rope", q, n, qd, get_pair("main"), interleaved=config.family == "deepseek_v2",
            offset=config.nope_dim, repack=config.family == "deepseek_v3" and config.interleaved)
        latent = linear(name + ".kva", x, prefix + "kv_a_proj_with_mqa", config.kv_lora_rank + r)
        kbase = gather(name + ".kv.latent", latent, range(config.kv_lora_rank), 2, [1, s, config.kv_lora_rank])
        kbase = b.norm(name + ".kv.norm", kbase, prefix + "kv_a_layernorm.weight", [1, s, config.kv_lora_rank], 1e-6)
        expanded = linear(name + ".kvb", kbase, prefix + "kv_b_proj", n * (config.nope_dim + vd))
        expanded = b.reshape(name + ".kv.heads", expanded, [1, s, n, config.nope_dim + vd])
        kn = gather(name + ".k.nope", expanded, range(config.nope_dim), 3, [1, s, n, config.nope_dim])
        value = gather(f"layer.{layer}.v", expanded, range(config.nope_dim, config.nope_dim + vd), 3, [1, s, n, vd])
        kr = gather(name + ".k.rotary", latent, range(config.kv_lora_rank, config.kv_lora_rank + r), 2, [1, s, r])
        kr = b.reshape(name + ".k.rotary.heads", kr, [1, s, 1, r])
        kr = rotary(name + ".k.rope", kr, 1, r, get_pair("main"), interleaved=config.family == "deepseek_v2",
            offset=0, repack=config.family == "deepseek_v3" and config.interleaved)
        kr = gather(name + ".k.repeat", kr, [0] * n, 2, [1, s, n, r])
        key = concat(f"layer.{layer}.k_rope", kn, kr, 3, [1, s, n, qd])
        scale = qd ** -0.5
        p = config.rope_main
        if config.family == "deepseek_v3" and p["rope_type"] != "default" and p.get("mscale_all_dim"):
            factor = p["factor"]
            scale *= (1 if factor <= 1 else 0.1 * p["mscale_all_dim"] * math.log(factor) + 1) ** 2
        attention = b.tensor(name + ".attention", "SparseAttention", {"q": q, "k": key, "v": value, "mask": "attention_mask"},
            [1, s, n, vd], {"causal": True, "scale": scale})
        return linear(name + ".out", b.reshape(name + ".flat", attention, [1, s, n * vd]), prefix + "o_proj", h)

    def compress(name, x, prefix, width, rate, groups):
        kv = linear(name + ".kv", x, prefix + "kv_proj", width * groups)
        gate = linear(name + ".gate", x, prefix + "gate_proj", width * groups)
        cos, sin = table_pair("hf.deepseek.compress", config.rope_compress, True, rate)
        return b.tensor(name, "KVCompress", {"input": kv, "a": gate, "bias": prefix + "position_bias",
            "weight": prefix + "kv_norm.weight", "cos": cos, "sin": sin}, [1, s, width],
            {"num_groups": groups, "compress_rate": rate, "eps": config.epsilon,
             "rotary_dim": r, "position_offset": width - r, "interleaved": True})

    def v4_attention(name, x, prefix, layer):
        d, which = config.head_dim, "main" if config.rates[layer] == 0 else "compress"
        pair = get_pair(which)
        qa = linear(name + ".qa", x, prefix + "q_a_proj", config.q_lora_rank)
        qa = b.norm(name + ".qa.norm", qa, prefix + "q_a_norm.weight", [1, s, config.q_lora_rank], config.epsilon)
        q = linear(name + ".qb", qa, prefix + "q_b_proj", n * d)
        q = b.reshape(name + ".q.heads", q, [1, s, n, d])
        q = b.norm(name + ".q.norm", q, b.add(name + ".q.norm.weight", np.ones(d, np.float32)), [1, s, n, d], config.epsilon)
        q = rotary(name + ".q.rope", q, n, d, pair, interleaved=True, offset=d - r)
        kv = linear(name + ".kv", x, prefix + "kv_proj", d)
        kv = b.norm(name + ".kv.norm", kv, prefix + "kv_norm.weight", [1, s, d], config.epsilon)
        kv = rotary(f"layer.{layer}.k_rope", b.reshape(name + ".kv.heads", kv, [1, s, 1, d]),
            1, d, pair, interleaved=True, offset=d - r)
        b.reshape(f"layer.{layer}.v", kv, [1, s, 1, d])
        inputs = {"q": q, "k": kv, "v": kv, "mask": "attention_mask", "position_ids": "position_ids", "sinks": prefix + "sinks"}
        params = {"causal": True, "scale": d ** -0.5, "window_size": config.window}
        if config.rates[layer]:
            groups = 2 if config.layer_types[layer] == "compressed_sparse_attention" else 1
            rate, c = config.rates[layer], prefix + "compressor."
            inputs["compressed_kv"] = compress(name + ".compressed", x, c, d, rate, groups)
            params["compress_rate"] = rate
            if groups == 2:
                ci, di = c + "indexer.", config.index_dim
                inputs["index_k"] = compress(name + ".index.compressed", x, ci, di, rate, 2)
                iq = linear(name + ".index.q", qa, ci + "q_b_proj", config.index_heads * di)
                iq = b.reshape(name + ".index.q.heads", iq, [1, s, config.index_heads, di])
                inputs["index_q"] = rotary(name + ".index.q.rope", iq, config.index_heads, di,
                    get_pair("compress"), interleaved=True, offset=di - r)
                inputs["index_weights"] = linear(name + ".index.weights", x, ci + "scorer.weights_proj", config.index_heads)
                params["top_k"] = config.index_top_k
        out = b.tensor(name + ".attention", "SparseAttention", inputs, [1, s, n, d], params)
        minus = b.tensor(name + ".minus.expand", "Expand",
            {"input": b.add(name + ".minus.one", np.array([-1], np.float32))},
            [1, s, r // 2], {"shape": [1, s, r // 2]})
        negsin = b.tensor(name + ".inverse.sin", "Mul", {"a": pair[1], "b": minus}, [1, s, r // 2])
        out = rotary(name + ".inverse.rope", out, n, d, (pair[0], negsin), interleaved=True, offset=d - r)
        flat = b.reshape(name + ".flat", out, [1, s, n * d])
        width = n * d // config.output_groups
        groups = []
        oa = weights[prefix + "o_a_proj.weight"]
        for group in range(config.output_groups):
            part = gather(name + f".group.{group}", flat, range(group * width, (group + 1) * width), 2, [1, s, width])
            w = b.add(name + f".group.{group}.weight", oa[group * config.output_rank:(group + 1) * config.output_rank])
            groups.append(b.linear(name + f".group.{group}.out", part, w, config.output_rank))
        projected = groups[0]
        for group in range(1, len(groups)):
            projected = concat(name + f".group.concat.{group}", projected, groups[group], 2, [1, s, (group + 1) * config.output_rank])
        return linear(name + ".out", projected, prefix + "o_b_proj", h)

    hidden = b.tensor("embedding", "Embedding", {"input": "token_ids", "weight": "model.embed_tokens.weight"}, hs)
    if config.family == "deepseek_v4":
        hidden = gather("embedding.streams", b.reshape("embedding.one.stream", hidden, [1, s, 1, h]),
            [0] * config.streams, 2, [1, s, config.streams, h])
    for layer in range(config.layers):
        name, p = f"layer.{layer}", f"model.layers.{layer}."
        if config.family == "deepseek_v4":
            stream_shape = [1, s, config.streams, h]
            for role in ("attn", "ffn"):
                hp = p + role + "_hc."
                connection = b.multi(name + "." + role + ".hc", "HyperConnection",
                    {"input": hidden, "weight": hp + "fn", "bias": hp + "base", "scale": hp + "scale"},
                    {"pre": (name + "." + role + ".pre", [1, s, config.streams], "float32"),
                     "post": (name + "." + role + ".post", [1, s, config.streams], "float32"),
                     "comb": (name + "." + role + ".comb", [1, s, config.streams, config.streams], "float32")},
                    {"eps": config.epsilon, "normalization_epsilon": config.hc_epsilon, "iterations": config.sinkhorn_iterations})
                collapsed = collapse_streams(b, name + "." + role + ".input", hidden,
                    connection["pre"], config.streams, h)
                x = b.norm(name + "." + role + ".norm", collapsed,
                    p + ("input_layernorm.weight" if role == "attn" else "post_attention_layernorm.weight"), hs, config.epsilon)
                update = (v4_attention(name + ".attn", x, p + "self_attn.", layer) if role == "attn"
                    else moe(name + ".mlp", x, p + "mlp.", layer))
                hidden = b.tensor(name + "." + role + ".residual", "HyperResidual",
                    {"input": hidden, "update": update, "post": connection["post"], "comb": connection["comb"]}, stream_shape)
        else:
            x = b.norm(name + ".attn.norm", hidden, p + "input_layernorm.weight", hs, config.epsilon)
            update = mla(name + ".attn", x, p + "self_attn.", layer)
            hidden = b.tensor(name + ".attn.residual", "Add", {"a": hidden, "b": update}, hs)
            x = b.norm(name + ".mlp.norm", hidden, p + "post_attention_layernorm.weight", hs, config.epsilon)
            update = (dense_mlp(name + ".mlp", x, p + "mlp.", config.intermediate_size) if layer < config.first_dense
                else moe(name + ".mlp", x, p + "mlp.", layer))
            hidden = b.tensor(name + ".mlp.residual", "Add", {"a": hidden, "b": update}, hs)
    if config.family == "deepseek_v4":
        hidden = b.tensor("model.hyper.head", "HyperHead", {"input": hidden, "weight": "model.hc_head.hc_fn",
            "bias": "model.hc_head.hc_base", "scale": "model.hc_head.hc_scale"}, hs,
            {"eps": config.epsilon, "normalization_epsilon": config.hc_epsilon})
    hidden = b.norm("model.final.norm", hidden, "model.norm.weight", hs, config.epsilon)
    b.linear(output_name, hidden, "model.embed_tokens.weight" if config.tied_embeddings else "lm_head.weight", config.vocab_size)
    if generate or cached:
        from .execution_contract import full_prefix_decoder_execution, incremental_decoder_execution
        if output_name=="generated": _fail("logits output conflicts with the generated ID output")
        b.tensor("generated","ArgMax",{"input":output_name},[1,"S"],{"axis":2,"keepdims":False,"select_last_index":0},dtype="int32")
        b.graph["outputs"].append("generated")
        if cached:
            b.graph["inputs"]["active_lengths"] = {"shape": [1], "dtype": "int32"}
            return b.finish(incremental_decoder_execution(b.graph["inputs"], b.graph["outputs"],
                token_input="token_ids", token_output="generated", keep_input="attention_mask",
                position_input="position_ids", logits_output=output_name, id_domain_size=config.vocab_size,
                incremental_controls=True))
        return b.finish(full_prefix_decoder_execution(b.graph["inputs"],b.graph["outputs"],
            token_input="token_ids",token_output="generated",keep_input="attention_mask",
            position_input="position_ids",logits_output=output_name,id_domain_size=config.vocab_size))
    return b.finish()


def export_hf_deepseek(directory, out_path, *, max_context=None, output_names=None,
                       capture_kv=False, compact_kv=False, cached=False, generate=False, report_callback=None,
                       weight_precision=None, group_size=128):
    source = _read_json(Path(directory) / "config.json")
    if max_context is None:
        _fail("DeepSeek export requires explicit max_context for bounded rotary/compressor tables")
    if not isinstance(cached, bool) or not isinstance(generate, bool):
        _fail("cached and generate must be boolean")
    if cached and capture_kv:
        _fail("cached execution requires private retained KV/compressor tensors")
    if output_names is not None and len(output_names) != 1:
        _fail("DeepSeek export has exactly one logits output")
    config = DeepSeekDecoderConfig.from_hf(source, max_context=max_context)
    _, source_weights, storage = read_checkpoint_storage(directory, config_validator=DeepSeekDecoderConfig.from_hf)
    graph, weights = build_deepseek_decoder_graph(config, source_weights,
        output_name=output_names[0] if output_names else "logits", capture_kv=capture_kv,
        compact_kv=compact_kv, cached=cached, generate=generate)
    graph = write_hf_package(out_path, graph, weights, weight_precision=weight_precision, group_size=group_size,
        storage_conversion="F32/F16/BF16;FP8-E4M3FN/FP4-E2M1+UE8M0->F32;checked-I32-metadata")
    if report_callback:
        report_callback({"features": {"hf_decoder": [f"family:{config.family}", "execution:F32", "batch:1",
            "state:retained-incremental-kv-compressor-rows" if cached else "state:complete-stateless-prefix",
            f"context-bound:{config.max_context}", "performance:unqualified",
            "generation:incremental" if cached else "generation:full-prefix" if generate else "generation:undeclared"],
            "source_storage": sorted(set(storage.values())),
            "source_quantization_config": config.source_quantization,
            "raw_FP8_FP4_storage": "checked-dequantization-to-F32;native-low-precision-unqualified",
            "auxiliary_MTP_layers": "excluded-from-causal-prefix"}})
    return graph
