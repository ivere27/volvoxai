"""Grouped INT2/INT3/INT4/INT8 packing shared by HF architectures and numerical checks.

PackedLinear keeps public activations F32. Compilation independently selects
F32, F16 or dynamic per-row I8 arithmetic with F32 accumulation. Offset-binary values
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


SUPPORTED_GROUP_SIZES = (32, 64, 128, 256)


def _fail(message):
    raise ExporterError(Diagnostic("VXPACKED001", message, "packed-linear"))


def _packing(weight_bits):
    if type(weight_bits) is not int or weight_bits not in (2, 3, 4, 8):
        _fail("weight_bits must be 2, 3, 4 or 8")
    return 32 // weight_bits, 1 << (weight_bits - 1)


def _group_size(group_size):
    if type(group_size) is not int or group_size not in SUPPORTED_GROUP_SIZES:
        _fail("group_size must be 32, 64, 128 or 256")
    return group_size


def pack_weight(weight, *, weight_bits=4, group_size=128):
    """Return I32 packed words and F32 [N,ceil(K/group_size)] scales.

    I3 codes cross I32 word boundaries; other widths divide each word evenly. Quantization uses nearest-even
    rounding and scale=max(abs(group))/(2**(weight_bits-1)-1). Zero groups use
    scale one. A nonzero subnormal scale is kept positive; an overflowing F32
    reconstruction endpoint lowers the scale by one ULP. Work is bounded to
    256 output rows and one group, without copying the full weight.
    """
    per_word, offset = _packing(weight_bits)
    group_size = _group_size(group_size)
    weight = np.asarray(weight)
    if weight.ndim != 2 or min(weight.shape) <= 0 or weight.dtype != np.dtype("float32"):
        _fail("weights must be a nonempty F32 [N,K] matrix")
    outputs, width = weight.shape
    zero_word = 0 if weight_bits == 3 else sum(offset << (index * weight_bits) for index in range(per_word))
    words = np.full((outputs, (width * weight_bits + 31) // 32), zero_word, np.uint32)
    scales = np.empty((outputs, (width + group_size - 1) // group_size), np.float32)
    shifts = np.arange(per_word, dtype=np.uint32) * np.uint32(weight_bits)
    smallest = np.nextafter(np.float32(0), np.float32(1))
    for row in range(0, outputs, 256):
        end = min(row + 256, outputs)
        for start in range(0, width, group_size):
            block = weight[row:end, start:start + group_size]
            if not np.isfinite(block).all():
                _fail("weights must contain only finite F32 values")
            maximum = np.abs(block).max(axis=1)
            scale = np.where(maximum == 0, np.float32(1),
                             np.maximum(maximum / np.float32(offset - 1), smallest))
            # F32 division can round max/127 upward: keep the actual decoder's
            # endpoint finite even when a source value is FLT_MAX.
            with np.errstate(over="ignore"):
                overflowing = ~np.isfinite(np.float32(offset - 1) * scale)
            scale[overflowing] = np.nextafter(scale[overflowing], np.float32(0), dtype=np.float32)
            scales[row:end, start // group_size] = scale
            quantized = np.clip(np.rint(block / scale[:, None]), -offset, offset - 1).astype(np.int32)
            if weight_bits == 3:
                codes = (quantized + offset).astype(np.uint32)
                for j in range(block.shape[1]):
                    word, shift = divmod((start + j) * 3, 32)
                    words[row:end, word] |= codes[:, j] << np.uint32(shift)
                    if shift > 29:
                        words[row:end, word + 1] |= codes[:, j] >> np.uint32(32 - shift)
            else:
                padded = np.full((end - row, ((block.shape[1] + per_word - 1) // per_word) * per_word), offset, np.uint32)
                padded[:, :block.shape[1]] = (quantized + offset).astype(np.uint32)
                packed = (padded.reshape(end - row, -1, per_word) << shifts).sum(axis=2, dtype=np.uint32)
                words[row:end, start // per_word:start // per_word + packed.shape[1]] = packed
    return words.view(np.int32), scales


def _checked_storage(words, scales, width, weight_bits, group_size):
    group_size = _group_size(group_size)
    per_word, offset = _packing(weight_bits)
    words, scales = np.asarray(words), np.asarray(scales)
    if (not isinstance(width, int) or isinstance(width, bool) or width <= 0
            or words.dtype != np.dtype("int32") or words.ndim != 2 or words.shape[0] <= 0
            or words.shape[1] != (width * weight_bits + 31) // 32 or scales.dtype != np.dtype("float32")
            or scales.shape != (words.shape[0], (width + group_size - 1) // group_size)
            or not np.isfinite(scales).all() or not (scales > 0).all()):
        _fail("invalid packed words, group scales or logical K width")
    if width * weight_bits % 32:
        shift = width * weight_bits % 32
        zero_word = 0 if weight_bits == 3 else sum(offset << (index * weight_bits) for index in range(per_word))
        if np.any((words[:, -1].view(np.uint32) >> shift) != (zero_word >> shift)):
            _fail("unused packed elements/bits must use canonical zero padding")
    indices = np.arange(width)
    if weight_bits == 3:
        bit = indices * 3
        word, shift = bit // 32, (bit % 32).astype(np.uint32)
        codes = words.view(np.uint32)[:, word] >> shift
        crosses = shift > 29
        codes[:, crosses] |= words.view(np.uint32)[:, word[crosses] + 1] << (np.uint32(32) - shift[crosses])
        quantized = (codes & np.uint32(7)).astype(np.int32) - 4
    else:
        quantized = ((words.view(np.uint32)[:, indices // per_word] >>
                      ((indices % per_word) * weight_bits).astype(np.uint32)) & np.uint32(2 * offset - 1)).astype(np.int32) - offset
    return quantized, scales


def packed_linear_reference(inputs, words, scales, *, weight_bits=4, group_size=128, activation_bits=16, bias=None):
    """Independent NumPy implementation of the explicit scalar arithmetic.

    Device reductions may use a different summation order. The reference uses
    ascending K for FP16 and ascending groups for INT8. Nonfinite source input
    rows produce NaN; FP16 conversion overflow of finite values remains Inf.
    """
    inputs = np.asarray(inputs)
    if inputs.dtype != np.dtype("float32") or inputs.ndim < 1 or min(inputs.shape) <= 0:
        _fail("activations must be nonempty F32 tensors")
    if activation_bits not in (8, 16, 32) or isinstance(activation_bits, bool):
        _fail("activation_bits must be 8, 16 or 32")
    width = inputs.shape[-1]
    quantized, scales = _checked_storage(words, scales, width, weight_bits, group_size)
    rows = inputs.reshape(-1, width)
    bad = ~np.isfinite(rows).all(axis=1)
    clean = np.where(bad[:, None], np.float32(0), rows)
    result = np.zeros((len(rows), len(quantized)), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        if activation_bits in (16, 32):
            rounded = clean.astype(np.float16).astype(np.float32) if activation_bits == 16 else clean
            weights = (quantized.astype(np.float32) * scales[:, np.arange(width) // group_size])
            if activation_bits == 16:
                weights = weights.astype(np.float16).astype(np.float32)
            for index in range(width):
                result += rounded[:, index, None] * weights[None, :, index]
        else:
            maximum = np.abs(clean).max(axis=1)
            activation_scale = np.where(maximum == 0, np.float32(1),
                np.maximum(maximum / np.float32(127), np.float32(2 ** -126)))
            rounded = np.clip(np.rint(clean / activation_scale[:, None]), -127, 127).astype(np.int32)
            for start in range(0, width, group_size):
                dot = rounded[:, start:start + group_size] @ quantized[:, start:start + group_size].T
                result += dot.astype(np.float32) * (activation_scale[:, None] * scales[None, :, start // group_size])
        if bias is not None:
            bias = np.asarray(bias)
            if bias.dtype != np.dtype("float32") or bias.shape != (len(quantized),):
                _fail("bias must be F32 [N]")
            result += bias
    result[bad] = np.float32(np.nan)
    return result.reshape(*inputs.shape[:-1], len(quantized))


def pack_linear_weights(graph: Mapping, weights: Mapping, *, weight_bits=4, group_size=128):
    """Clone a graph and pack each immutable Linear weight exactly once.

    Other operators retain their original weights. This includes an embedding
    tied to lm_head: its F32 matrix remains present alongside the packed head.
    Tensor buffers are shared until a selected weight is packed, and unused
    originals are omitted from the returned package.
    """
    _packing(weight_bits)
    group_size = _group_size(group_size)
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
            words, scales = pack_weight(weight if layout == "dout_din" else weight.T, weight_bits=weight_bits, group_size=group_size)
            tensors[word_name], tensors[scale_name] = words, scales
            packed[key] = word_name, scale_name
        word_name, scale_name = packed[key]
        node["opType"] = "PackedLinear"
        node["inputs"] = {**inputs, "weight": word_name, "scale": scale_name}
        node["params"] = {"group_size": group_size, "weight_bits": weight_bits}
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
