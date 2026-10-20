"""Group-128 INT4/INT8 packing shared by HF architectures and numerical checks.

PackedLinear keeps public activations F32, rounds them at the operator boundary
to FP16 or dynamic per-row INT8, and accumulates in F32. Offset-binary values
occupy I32 words, with the first K element in the least significant bits.
This is the engine's explicit format; importing AWQ/GPTQ/GGUF is separate.
"""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .operator_shape_contracts_common import _dense_layout


GROUP_SIZE = 128


def _fail(message):
    raise ExporterError(Diagnostic("VXPACKED001", message, "packed-linear"))


def _packing(weight_bits):
    if type(weight_bits) is not int or weight_bits not in (4, 8):
        _fail("weight_bits must be 4 or 8")
    return 32 // weight_bits, 1 << (weight_bits - 1)


def pack_weight(weight, *, weight_bits=4):
    """Return I32 packed words and F32 [N,ceil(K/128)] scales.

    Words hold eight 4-bit or four 8-bit values. Quantization uses nearest-even
    rounding and scale=max(abs(group))/(2**(weight_bits-1)-1). Zero groups use
    scale one. A nonzero subnormal scale is kept positive. Work is bounded to
    256 output rows and one group, without copying the full weight.
    """
    per_word, offset = _packing(weight_bits)
    weight = np.asarray(weight)
    if weight.ndim != 2 or min(weight.shape) <= 0 or weight.dtype != np.dtype("float32"):
        _fail("weights must be a nonempty F32 [N,K] matrix")
    outputs, width = weight.shape
    zero_word = sum(offset << (index * weight_bits) for index in range(per_word))
    words = np.full((outputs, (width + per_word - 1) // per_word), zero_word, np.uint32)
    scales = np.empty((outputs, (width + GROUP_SIZE - 1) // GROUP_SIZE), np.float32)
    shifts = np.arange(per_word, dtype=np.uint32) * np.uint32(weight_bits)
    smallest = np.nextafter(np.float32(0), np.float32(1))
    for row in range(0, outputs, 256):
        end = min(row + 256, outputs)
        for start in range(0, width, GROUP_SIZE):
            block = weight[row:end, start:start + GROUP_SIZE]
            if not np.isfinite(block).all():
                _fail("weights must contain only finite F32 values")
            maximum = np.abs(block).max(axis=1)
            scale = np.where(maximum == 0, np.float32(1),
                             np.maximum(maximum / np.float32(offset - 1), smallest))
            scales[row:end, start // GROUP_SIZE] = scale
            quantized = np.clip(np.rint(block / scale[:, None]), -offset, offset - 1).astype(np.int32)
            padded = np.full((end - row, ((block.shape[1] + per_word - 1) // per_word) * per_word), offset, np.uint32)
            padded[:, :block.shape[1]] = (quantized + offset).astype(np.uint32)
            packed = (padded.reshape(end - row, -1, per_word) << shifts).sum(axis=2, dtype=np.uint32)
            words[row:end, start // per_word:start // per_word + packed.shape[1]] = packed
    return words.view(np.int32), scales


def _checked_storage(words, scales, width, weight_bits):
    per_word, offset = _packing(weight_bits)
    words, scales = np.asarray(words), np.asarray(scales)
    if (not isinstance(width, int) or isinstance(width, bool) or width <= 0
            or words.dtype != np.dtype("int32") or words.ndim != 2 or words.shape[0] <= 0
            or words.shape[1] != (width + per_word - 1) // per_word or scales.dtype != np.dtype("float32")
            or scales.shape != (words.shape[0], (width + GROUP_SIZE - 1) // GROUP_SIZE)
            or not np.isfinite(scales).all() or not (scales > 0).all()):
        _fail("invalid packed words, group scales or logical K width")
    if width % per_word:
        shift = (width % per_word) * weight_bits
        zero_word = sum(offset << (index * weight_bits) for index in range(per_word))
        if np.any((words[:, -1].view(np.uint32) >> shift) != (zero_word >> shift)):
            _fail("unused packed elements must encode zero")
    indices = np.arange(width)
    quantized = ((words.view(np.uint32)[:, indices // per_word] >>
                  ((indices % per_word) * weight_bits).astype(np.uint32)) & np.uint32(2 * offset - 1)).astype(np.int32) - offset
    return quantized, scales


def packed_linear_reference(inputs, words, scales, *, weight_bits=4, activation_bits=16, bias=None):
    """Independent NumPy implementation of the explicit scalar arithmetic.

    Device reductions may use a different summation order. The reference uses
    ascending K for FP16 and ascending groups for INT8. Nonfinite source input
    rows produce NaN; FP16 conversion overflow of finite values remains Inf.
    """
    inputs = np.asarray(inputs)
    if inputs.dtype != np.dtype("float32") or inputs.ndim < 1 or min(inputs.shape) <= 0:
        _fail("activations must be nonempty F32 tensors")
    if activation_bits not in (8, 16) or isinstance(activation_bits, bool):
        _fail("activation_bits must be 8 or 16")
    width = inputs.shape[-1]
    quantized, scales = _checked_storage(words, scales, width, weight_bits)
    rows = inputs.reshape(-1, width)
    bad = ~np.isfinite(rows).all(axis=1)
    clean = np.where(bad[:, None], np.float32(0), rows)
    result = np.zeros((len(rows), len(quantized)), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        if activation_bits == 16:
            rounded = clean.astype(np.float16).astype(np.float32)
            weights = (quantized.astype(np.float32) * scales[:, np.arange(width) // GROUP_SIZE])
            weights = weights.astype(np.float16).astype(np.float32)
            for index in range(width):
                result += rounded[:, index, None] * weights[None, :, index]
        else:
            maximum = np.abs(clean).max(axis=1)
            activation_scale = np.where(maximum == 0, np.float32(1),
                np.maximum(maximum / np.float32(127), np.float32(2 ** -126)))
            rounded = np.clip(np.rint(clean / activation_scale[:, None]), -127, 127).astype(np.int32)
            for start in range(0, width, GROUP_SIZE):
                dot = rounded[:, start:start + GROUP_SIZE] @ quantized[:, start:start + GROUP_SIZE].T
                result += dot.astype(np.float32) * (activation_scale[:, None] * scales[None, :, start // GROUP_SIZE])
        if bias is not None:
            bias = np.asarray(bias)
            if bias.dtype != np.dtype("float32") or bias.shape != (len(quantized),):
                _fail("bias must be F32 [N]")
            result += bias
    result[bad] = np.float32(np.nan)
    return result.reshape(*inputs.shape[:-1], len(quantized))


def pack_linear_weights(graph: Mapping, weights: Mapping, *, weight_bits=4, activation_bits=16):
    """Clone a graph and pack each immutable Linear weight exactly once.

    Other operators retain their original weights. This includes an embedding
    tied to lm_head: its F32 matrix remains present alongside the packed head.
    Tensor buffers are shared until a selected weight is packed, and unused
    originals are omitted from the returned package.
    """
    _packing(weight_bits)
    if activation_bits not in (8, 16) or isinstance(activation_bits, bool):
        _fail("activation_bits must be 8 or 16")
    result = copy.deepcopy(graph)
    tensors = dict(weights)
    packed = {}
    converted = set()
    for node in result.get("nodes", ()):
        if node.get("opType") != "Linear":
            continue
        layout = _dense_layout("Linear", node.get("params", {}))
        inputs = node.get("inputs", {})
        name = inputs.get("weight")
        if name not in weights:
            _fail("PackedLinear requires an immutable weight tensor")
        key = name, layout
        if key not in packed:
            digest = hashlib.sha256((name + "\0" + layout).encode()).hexdigest()[:24]
            word_name, scale_name = f"hf.packed.{digest}.weight", f"hf.packed.{digest}.scale"
            if word_name in tensors or scale_name in tensors:
                _fail("packed weight names conflict with existing tensors")
            weight = np.asarray(weights[name])
            words, scales = pack_weight(weight if layout == "dout_din" else weight.T, weight_bits=weight_bits)
            tensors[word_name], tensors[scale_name] = words, scales
            packed[key] = word_name, scale_name
        word_name, scale_name = packed[key]
        node["opType"] = "PackedLinear"
        node["inputs"] = {**inputs, "weight": word_name, "scale": scale_name}
        node["params"] = {"group_size": GROUP_SIZE, "weight_bits": weight_bits, "activation_bits": activation_bits}
        converted.add(name)
    if not packed:
        _fail("weight precision requested but the graph has no immutable Linear weights")
    retained = {name for node in result["nodes"] for name in node.get("inputs", {}).values()}
    retained.update(result.get("outputs", ()))
    # Bank-resident tensors are independently addressable by the model API.
    retained.update(result.get("banks", {}))
    for name in converted - retained:
        tensors.pop(name, None)
    return result, tensors
