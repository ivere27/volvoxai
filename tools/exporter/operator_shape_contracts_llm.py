"""Concrete and bounded shape contracts for modern decoder primitives.

Sequence axes may vary within the authored domain. Head counts and channel
widths are fixed; kernels never infer a tensor extent from its values.
"""
from __future__ import annotations

import math
from .operator_shape_contracts_common import (
    _assert_allowed_fields, _assert_float_tensor, _assert_i32_tensor,
    _boolean_param, _dimensions_provably_equal, _fail,
    _finite_positive_param, _fixed_dimension_value, _logical_outputs,
    _logical_request_parts, _outputs, _positive_safe_integer_param, _request_parts,
)

PORTS = {
    "GatedDeltaNet": (("q", "k", "v", "a", "b", "a_log", "dt_bias"), ("initial_state",), ("out", "state")),
    "SparseAttention": (("q", "k", "v"), ("mask", "position_ids", "compressed_kv", "sinks", "index_q", "index_k", "index_weights", "compressed_mask"), ("out",)),
    "KVCompress": (("input", "a", "bias", "weight"), ("cos", "sin"), ("out",)),
    "HyperConnection": (("input", "weight", "bias", "scale"), (), ("pre", "post", "comb")),
    "HyperHead": (("input", "weight", "bias", "scale"), (), ("out",)),
    "HyperResidual": (("input", "update", "post", "comb"), (), ("out",)),
    "RotaryEmbedding": (("input", "cos", "sin"), (), ("out",)),
    "NGramIds": (("token_ids", "multipliers", "vocab_sizes", "offsets"), (), ("out",)),
    "SparseIndexer": (("q", "k", "index_weights"), ("candidates",), ("out", "candidates")),
    "QSAIndexer": (("q", "k", "weight", "cos", "sin"), ("mask",), ("out",)),
    "SignedSqrt": (("input",), (), ("out",)),
    "EmbeddingShard": (("input", "weight"), (), ("out",)),
    "PackedLinear": (("input", "weight", "scale"), ("bias",), ("out",)),
}

def _shapes(operator, request, logical):
    required, optional, output_names = PORTS[operator]
    if logical:
        env, inputs, params, declared = _logical_request_parts(request, required, optional)
    else:
        inputs, params, declared = _request_parts(request, required, optional)
        env = None

    def eq(a, b):
        return _dimensions_provably_equal(a, b, env) if logical else a == b

    def check(condition, path="operator inputs"):
        if not condition:
            _fail("SHAPE_MISMATCH", path, f"does not satisfy {operator}'s shape contract.")

    def shape(name, expected):
        value = inputs[name].shape
        check(len(value) == len(expected) and all(eq(a, b) for a, b in zip(value, expected)), name)

    def fixed(value):
        result = _fixed_dimension_value(value, env) if logical else value
        check(isinstance(result, int) and result > 0, "operator fixed dimensions")
        return result

    for name, tensor in inputs.items():
        (_assert_i32_tensor if operator == "NGramIds" or (operator == "EmbeddingShard" and name == "input") or (operator == "PackedLinear" and name == "weight") or name in ("mask", "position_ids", "candidates", "compressed_mask") else _assert_float_tensor)(tensor, name)
    result = {}
    if operator == "PackedLinear":
        _assert_allowed_fields(params, ("group_size", "activation_bits", "weight_bits"), "operator params")
        if _positive_safe_integer_param(params, "group_size") != 128:
            _fail("INVALID_PARAMS", "group_size", "PackedLinear requires group_size=128.")
        if _positive_safe_integer_param(params, "activation_bits") not in (8, 16):
            _fail("INVALID_PARAMS", "activation_bits", "PackedLinear requires activation_bits=8 or 16.")
        weight_bits = _positive_safe_integer_param(params, "weight_bits", 4)
        if weight_bits not in (4, 8):
            _fail("INVALID_PARAMS", "weight_bits", "PackedLinear requires weight_bits=4 or 8.")
        x, w = inputs["input"].shape, inputs["weight"].shape
        check(1 <= len(x) <= 8 and len(w) == 2)
        k, n = fixed(x[-1]), fixed(w[0])
        check(k <= 2147483647 and n <= 2147483647)
        per_word = 32 // weight_bits
        shape("weight", (n, (k + per_word - 1) // per_word))
        shape("scale", (n, (k + 127) // 128))
        if "bias" in inputs: shape("bias", (n,))
        result["out"] = (*x[:-1], n)
    elif operator == "EmbeddingShard":
        _assert_allowed_fields(params, ("vocab_size", "vocab_offset"), "operator params")
        vocab = _positive_safe_integer_param(params, "vocab_size")
        offset = params.get("vocab_offset")
        check(isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 and vocab <= 2147483647)
        weight = inputs["weight"].shape
        check(len(weight) == 2 and len(inputs["input"].shape) <= 7)
        rows, width = fixed(weight[0]), fixed(weight[1])
        check(offset + rows <= vocab)
        result["out"] = (*inputs["input"].shape, width)
    elif operator == "SignedSqrt":
        _assert_allowed_fields(params, ("eps",), "operator params")
        _finite_positive_param(params, "eps")
        result["out"] = inputs["input"].shape
    elif operator == "NGramIds":
        _assert_allowed_fields(params, ("ngram_size", "heads_per_ngram", "eos_token_id", "pad_token_id", "vocab_size"), "operator params")
        check({"ngram_size", "heads_per_ngram", "eos_token_id", "vocab_size"} <= set(params))
        n = _positive_safe_integer_param(params, "ngram_size")
        p = _positive_safe_integer_param(params, "heads_per_ngram")
        vocab = _positive_safe_integer_param(params, "vocab_size")
        eos = params["eos_token_id"]
        pad = params.get("pad_token_id", eos)
        check(n >= 2 and isinstance(eos, int) and not isinstance(eos, bool) and -1 <= eos < vocab <= 2147483647 and (n - 1) * p <= 2147483647)
        check(isinstance(pad, int) and not isinstance(pad, bool) and 0 <= pad < vocab)
        tokens = inputs["token_ids"].shape
        check(len(tokens) == 2)
        shape("multipliers", (n, 2)); shape("vocab_sizes", ((n - 1) * p,)); shape("offsets", ((n - 1) * p,))
        result["out"] = (*tokens, (n - 1) * p)
    elif operator == "SparseIndexer":
        _assert_allowed_fields(params, ("compress_rate", "top_k", "candidate_block_size", "candidate_top_k"), "operator params")
        _positive_safe_integer_param(params, "compress_rate"); _positive_safe_integer_param(params, "top_k")
        q = inputs["q"].shape
        check(len(q) == 4)
        b, sequence, heads, width = q; fixed(heads); fixed(width)
        shape("k", (b, sequence, width)); shape("index_weights", (b, sequence, heads))
        result["out"] = (b, sequence, sequence)
        if "candidates" in inputs: shape("candidates", result["out"])
        generate = (declared is not None and "candidates" in declared) or (declared is None and "candidate_top_k" in params)
        if generate:
            check("candidates" not in inputs)
            _positive_safe_integer_param(params, "candidate_block_size"); _positive_safe_integer_param(params, "candidate_top_k")
            result["candidates"] = result["out"]
        else: check("candidate_block_size" not in params and "candidate_top_k" not in params)
    elif operator == "QSAIndexer":
        _assert_allowed_fields(params, ("causal", "eps", "compress_rate", "top_k", "rotary_dim"), "operator params")
        _boolean_param(params, "causal", True); _finite_positive_param(params, "eps")
        _positive_safe_integer_param(params, "compress_rate"); _positive_safe_integer_param(params, "top_k")
        q, k = inputs["q"].shape, inputs["k"].shape
        check(len(q) == 4 and len(k) == 3)
        b, qs, hi, d = q; ks = k[1]
        fixed(hi); d = fixed(d); check(d <= 512)
        r = _positive_safe_integer_param(params, "rotary_dim", d)
        check(r % 2 == 0 and r <= d)
        shape("k", (b, ks, d)); shape("weight", (d,))
        shape("cos", (b, ks, r)); shape("sin", (b, ks, r))
        if "mask" in inputs:
            mask = inputs["mask"].shape
            check(any(len(mask) == len(target) and all(eq(x, y) for x, y in zip(mask, target)) for target in ((b, ks), (b, qs, ks))))
        result["out"] = (b, qs, ks)
    elif operator == "GatedDeltaNet":
        _assert_allowed_fields(params, ("normalize", "eps", "scale"), "operator params")
        _boolean_param(params, "normalize", True)
        _finite_positive_param(params, "eps")
        _finite_positive_param(params, "scale")
        q, v = inputs["q"].shape, inputs["v"].shape
        check(len(q) == len(v) == 4)
        b, s, hk, dk = q
        hv, dv = v[2:]
        check(fixed(hv) % fixed(hk) == 0)
        fixed(dk); fixed(dv)
        shape("k", q); shape("v", (b, s, hv, dv))
        for name in ("a", "b"): shape(name, (b, s, hv))
        for name in ("a_log", "dt_bias"): shape(name, (hv,))
        state = (b, hv, dk, dv)
        if "initial_state" in inputs: shape("initial_state", state)
        result["out"] = v
        if declared is None or "state" in declared: result["state"] = state
    elif operator == "SparseAttention":
        _assert_allowed_fields(params, ("causal", "scale", "window_size", "compress_rate", "top_k"), "operator params")
        _boolean_param(params, "causal", True)
        _finite_positive_param(params, "scale")
        for name in ("window_size", "top_k"):
            if name in params and params[name] != 0: _positive_safe_integer_param(params, name)
        if "compress_rate" in params: _positive_safe_integer_param(params, "compress_rate")
        q, k, v = (inputs[name].shape for name in ("q", "k", "v"))
        check(len(q) == len(k) == len(v) == 4)
        b, qs, hq, dk = q
        ks, hk, dv = k[1], k[2], v[3]
        check(fixed(hq) % fixed(hk) == 0)
        fixed(dk); fixed(dv)
        shape("k", (b, ks, hk, dk)); shape("v", (b, ks, hk, dv))
        if "mask" in inputs:
            mask = inputs["mask"].shape
            check(any(len(mask) == len(target) and all(eq(x, y) for x, y in zip(mask, target))
                      for target in ((ks,), (b, ks), (qs, ks), (b, qs, ks))), "mask")
        if "position_ids" in inputs: shape("position_ids", (qs,))
        if "sinks" in inputs: shape("sinks", (hq,))
        compressed = "compressed_kv" in inputs
        if compressed:
            check(eq(qs, ks) and fixed(hk) == 1 and eq(dk, dv))
            _positive_safe_integer_param(params, "compress_rate")
            shape("compressed_kv", (b, ks, dk))
        else: check("compress_rate" not in params and "top_k" not in params)
        index_names = ("index_q", "index_k", "index_weights")
        check(sum(name in inputs for name in index_names) in (0, 3))
        if "index_q" in inputs:
            check(compressed)
            iq = inputs["index_q"].shape
            check(len(iq) == 4)
            hi, di = iq[2:]; fixed(hi); fixed(di)
            shape("index_q", (b, qs, hi, di)); shape("index_k", (b, ks, di))
            shape("index_weights", (b, qs, hi))
            _positive_safe_integer_param(params, "top_k")
        if "compressed_mask" in inputs:
            check(compressed and not any(name in inputs for name in index_names))
            shape("compressed_mask", (b, qs, ks))
        result["out"] = (b, qs, hq, dv)
    elif operator == "KVCompress":
        _assert_allowed_fields(params, ("num_groups", "compress_rate", "eps", "rotary_dim", "position_offset", "interleaved"), "operator params")
        groups = _positive_safe_integer_param(params, "num_groups", 1)
        rate = _positive_safe_integer_param(params, "compress_rate")
        _finite_positive_param(params, "eps")
        x, w = inputs["input"].shape, inputs["weight"].shape
        check(len(x) == 3 and len(w) == 1 and groups in (1, 2))
        d = fixed(w[0]); check(fixed(x[2]) == groups * d)
        shape("a", x); shape("bias", (rate, groups * d))
        check(("cos" in inputs) == ("sin" in inputs))
        if "cos" in inputs:
            r = _positive_safe_integer_param(params, "rotary_dim", d)
            offset = params.get("position_offset", 0)
            check(isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 and r % 2 == 0 and offset + r <= d)
            interleaved = _boolean_param(params, "interleaved", False)
            tables = inputs["cos"].shape
            check(len(tables) == 2 and fixed(tables[1]) == (r // 2 if interleaved else r))
            capacity = fixed(tables[0])
            maximum_s = env.get(x[1]).max if logical and isinstance(x[1], str) else x[1]
            check(capacity >= maximum_s)
            shape("sin", tables)
        else:
            check(not any(name in params for name in ("rotary_dim", "position_offset", "interleaved")))
        result["out"] = (*x[:2], d)
    elif operator in ("HyperConnection", "HyperHead", "HyperResidual"):
        x = inputs["input"].shape
        check(len(x) == 4)
        b, s, m, d = x
        m, d = fixed(m), fixed(d)
        if operator == "HyperResidual":
            _assert_allowed_fields(params, (), "operator params")
            shape("update", (b, s, d)); shape("post", (b, s, m)); shape("comb", (b, s, m, m))
            result["out"] = x
        else:
            _assert_allowed_fields(params, ("eps", "normalization_epsilon") + (("iterations",) if operator == "HyperConnection" else ()), "operator params")
            _finite_positive_param(params, "eps")
            _finite_positive_param(params, "normalization_epsilon")
            if operator == "HyperConnection": _positive_safe_integer_param(params, "iterations", 20)
            count = m if operator == "HyperHead" else 2 * m + m * m
            shape("weight", (count, m * d)); shape("bias", (count,))
            shape("scale", (1 if operator == "HyperHead" else 3,))
            if operator == "HyperConnection":
                result["pre"] = (b, s, m)
                result["post"] = (b, s, m); result["comb"] = (b, s, m, m)
            else:
                result["out"] = (b, s, d)
    elif operator == "RotaryEmbedding":
        _assert_allowed_fields(params, ("rotary_dim", "position_offset", "interleaved"), "operator params")
        x = inputs["input"].shape
        check(len(x) == 4)
        d = fixed(x[3]); fixed(x[2])
        r = _positive_safe_integer_param(params, "rotary_dim", d)
        offset = params.get("position_offset", 0)
        check(isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0)
        check(r % 2 == 0 and offset + r <= d)
        interleaved = _boolean_param(params, "interleaved", False)
        shape("cos", (*x[:2], r // 2 if interleaved else r))
        shape("sin", inputs["cos"].shape)
        result["out"] = x
    if declared is not None:
        check(isinstance(declared, dict) or hasattr(declared, "keys"), "declared outputs")
        if set(declared) != set(result):
            _fail("INVALID_OUTPUT_PORTS", "declared outputs", "does not match the operator output ports.")
    descriptors = {name: (shape, "int32" if operator in ("NGramIds", "QSAIndexer", "SparseIndexer") else "float32", None) for name, shape in result.items()}
    return _logical_outputs(descriptors, env) if logical else _outputs(descriptors)

def infer_llm(operator, request):
    return _shapes(operator, request, False)

def prove_llm(operator, request):
    return _shapes(operator, request, True), ("fixed feature geometry and provably equal bounded sequence axes",)
