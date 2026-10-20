"""Shared graph construction and local package IO for Hugging Face importers.

Architecture modules own configuration semantics and graph lowering. These
helpers validate checkpoint storage, collect immutable initializers and publish
packages; RuntimeIR remains the common shape authority. They execute no model.
Packed formats require their own explicit storage conversion contract.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from .errors import Diagnostic, ExporterError
from .execution_contract import stateless_execution
from .quantization_storage import GRAPH_FORMAT
from .runtime_ir import import_runtime_package


def _fail(message: str, code: str = "VXHF_CONFIG") -> None:
    raise ExporterError(Diagnostic(code, message, "hf-decoder"))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}", "VXHF_SOURCE")
        result[key] = value
    return result


def read_config(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"),
                            object_pairs_hook=_unique_object,
                            parse_constant=lambda value: _fail(
                                f"non-finite JSON value {value}", "VXHF_SOURCE"))
    except (OSError, UnicodeError, ValueError) as error:
        _fail(f"cannot read {path.name}: {error}", "VXHF_SOURCE")
    if not isinstance(result, dict):
        _fail(f"{path.name} must contain an object", "VXHF_SOURCE")
    return result


def positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= 2**31 - 1:
        _fail(f"{name} must be a positive int32 integer")
    return value


def positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a positive finite F32 value")
    if not math.isfinite(float(value)) or value <= 0 or value > np.finfo(np.float32).max:
        _fail(f"{name} must be a positive finite F32 value")
    if np.float32(value) == 0:
        _fail(f"{name} must remain positive as F32")
    return float(value)


def _quantized_weights(tensors, source):
    """Bind encoded weights to their explicit scale tensors before payload IO."""
    encoded = {name for name, (_, dtype, _, _) in tensors.items()
               if dtype in {"F8_E4M3", "I8", "U8"}}
    if not encoded:
        return {}
    config = source.get("quantization_config")
    if not isinstance(config, dict) or config.get("quant_method") != "fp8" or \
            config.get("scale_fmt") != "ue8m0" or \
            config.get("weight_block_size") not in ([32, 32], [128, 128]):
        _fail("encoded weights require explicit FP8 block geometry and UE8M0 scales", "VXHF_WEIGHTS")
    plans, used_scales = {}, set()
    for name in sorted(encoded):
        _, dtype, shape, _ = tensors[name]
        if not name.endswith(".weight") or len(shape) != 2 or min(shape) <= 0:
            _fail(f"encoded tensor {name} must be a nonempty matrix weight", "VXHF_WEIGHTS")
        prefix = name[:-len("weight")]
        scales = [key for key in (prefix + "scale", prefix + "weight_scale_inv") if key in tensors]
        if len(scales) != 1:
            _fail(f"encoded tensor {name} requires exactly one scale tensor", "VXHF_WEIGHTS")
        scale = scales[0]
        if tensors[scale][1] != "F8_E8M0":
            _fail(f"scale tensor {scale} must use UE8M0 storage", "VXHF_WEIGHTS")
        rows, columns = shape
        block_rows, block_columns = config["weight_block_size"]
        if dtype in {"I8", "U8"}:
            # Official DeepSeek HF exports pack two E2M1 values per byte,
            # low nibble first; this is not arbitrary int8 quantization.
            if config.get("expert_dtype") != "fp4" or not re.search(
                    r"(?:^|\.)experts\.\d+\.(?:w[123]|gate_proj|up_proj|down_proj)\.weight$", name):
                _fail(f"packed tensor {name} is not a declared FP4 expert", "VXHF_WEIGHTS")
            columns *= 2
            block_rows, block_columns = 1, 32
            if columns % block_columns:
                _fail(f"FP4 tensor {name} requires complete 32-value blocks", "VXHF_WEIGHTS")
        elif name.endswith(".engram.embed.weight"):
            # Engram lookup dequantizes individual rows, unlike linear GEMM.
            block_rows = 1
        expected = ((rows + block_rows - 1) // block_rows,
                    (columns + block_columns - 1) // block_columns)
        if tensors[scale][2] != expected or scale in used_scales:
            _fail(f"scale tensor {scale} has incorrect block geometry", "VXHF_WEIGHTS")
        used_scales.add(scale)
        plans[name] = (scale, (rows, columns), (block_rows, block_columns))
    for name, (_, dtype, _, _) in tensors.items():
        if (dtype == "F8_E8M0" or name.endswith((".weight_scale_inv", ".scale"))) and name not in used_scales:
            _fail(f"unpaired quantization scale {name}", "VXHF_WEIGHTS")
    return plans


def _mapped_tensor(tensor, dtype):
    path, _, shape, offset = tensor
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)


def _dequantize_weight(tensor, scale_tensor, shape, block):
    """Decode into the final F32 array with at most one encoded block temporary.

    FP8 E4M3FN and packed FP4 E2M1 use UE8M0 multiplication scales (not
    reciprocals, including the HF name `weight_scale_inv`). No torch or CUDA
    dependency, activation quantization, or expanded full-matrix scale buffer.
    """
    raw, scales = _mapped_tensor(tensor, "u1"), _mapped_tensor(scale_tensor, "u1")
    packed = tensor[1] in {"I8", "U8"}
    if packed:
        lookup = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           -0., -.5, -1., -1.5, -2., -3., -4., -6.], np.float32)
    else:
        bits = np.arange(256, dtype=np.uint8)
        exponent, mantissa = (bits >> 3) & 15, bits & 7
        lookup = np.ldexp(1 + mantissa.astype(np.float32) / 8, exponent.astype(np.int16) - 7)
        lookup[exponent == 0] = mantissa[exponent == 0].astype(np.float32) * np.float32(2**-9)
        lookup = np.copysign(lookup, np.where(bits & 128, -1., 1.)).astype(np.float32)
        lookup[(exponent == 15) & (mantissa == 7)] = np.nan
    result = np.empty(shape, dtype=np.float32)
    rows, columns = shape
    block_rows, block_columns = block
    for i in range(0, rows, block_rows):
        for j in range(0, columns, block_columns):
            exponent = int(scales[i // block_rows, j // block_columns])
            if exponent == 255:
                _fail("UE8M0 scale contains NaN", "VXHF_WEIGHTS")
            target = result[i:i + block_rows, j:j + block_columns]
            if packed:
                values = raw[i:i + block_rows, j // 2:(j + block_columns) // 2]
                target[:, 0::2] = lookup[values & 15]
                target[:, 1::2] = lookup[values >> 4]
            else:
                target[:] = lookup[raw[i:i + block_rows, j:j + block_columns]]
            with np.errstate(over="ignore", invalid="ignore"):
                target *= np.float32(math.ldexp(1., exponent - 127))
            if not np.isfinite(target).all():
                _fail("decoded weight contains non-finite F32 values", "VXHF_WEIGHTS")
    return result


def read_checkpoint_metadata(directory: str | Path, *, source_config=None, config_validator=None):
    """Validate local shards without reading or converting tensor payloads.

    Return config and name -> (path, storage dtype, shape, payload byte offset).
    Both direct loading and the converting exporter use this source inventory.
    """
    directory = Path(directory)
    source_config = read_config(directory / "config.json") if source_config is None else source_config
    if config_validator is not None:
        config_validator(source_config)
    index_path = directory / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        weight_map = read_config(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            _fail("shard index requires a nonempty weight_map", "VXHF_SOURCE")
        files = sorted(set(weight_map.values())) if all(isinstance(v, str) for v in weight_map.values()) else []
    else:
        files = ["model.safetensors"]
    if not files:
        _fail("invalid shard filename in weight_map", "VXHF_SOURCE")
    tensors = {}
    for filename in files:
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            _fail("shard paths must be plain local SafeTensors filenames", "VXHF_SOURCE")
        path = directory / filename
        try:
            # The official reader validates header geometry and payload spans.
            with safe_open(path, framework="numpy") as handle:
                with path.open("rb") as stream:
                    header_size = int.from_bytes(stream.read(8), "little")
                    header = json.loads(stream.read(header_size), object_pairs_hook=_unique_object)
                for name in handle.keys():
                    if name in tensors or (weight_map is not None and weight_map.get(name) != filename):
                        _fail(f"duplicate or incorrectly indexed tensor {name}", "VXHF_SOURCE")
                    descriptor = handle.get_slice(name)
                    tensors[name] = (path, descriptor.get_dtype(), tuple(descriptor.get_shape()),
                                     8 + header_size + header[name]["data_offsets"][0])
        except ExporterError:
            raise
        except Exception as error:
            _fail(f"cannot read checkpoint shard {filename}: {error}", "VXHF_SOURCE")
    if weight_map is not None and set(weight_map) != set(tensors):
        _fail("shard index contains missing tensor names", "VXHF_SOURCE")
    return source_config, tensors


def read_checkpoint_storage(directory: str | Path, *, source_config=None, config_validator=None):
    """Read local storage into F32, admitting architecture before weight IO.

    Encoded matrices are decoded block by block; the existing graph exporter
    still owns the complete decoded F32 checkpoint in memory.
    """
    source_config, tensors = read_checkpoint_metadata(directory,
        source_config=source_config, config_validator=config_validator)
    plans = _quantized_weights(tensors, source_config)
    scales = {plan[0] for plan in plans.values()}
    weights, source_dtypes = {}, {name: tensor[1] for name, tensor in tensors.items()}
    for path in sorted({tensor[0] for tensor in tensors.values()}):
        try:
            with safe_open(path, framework="numpy") as handle:
                for name in handle.keys():
                    if name in scales:
                        continue
                    dtype = tensors[name][1]
                    if name in plans:
                        scale, shape, block = plans[name]
                        array = _dequantize_weight(tensors[name], tensors[scale], shape, block)
                    elif dtype == "BF16":
                        # NumPy has no built-in BF16. Widen its exact bits,
                        # after validation, without adding a torch dependency.
                        bits = _mapped_tensor(tensors[name], "<u2")
                        array = (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)
                        del bits
                    elif dtype in {"F32", "F16"}:
                        array = handle.get_tensor(name).astype(np.float32, copy=False)
                    elif dtype in {"I32", "I64"} and _immutable_integer_buffer(name):
                        # These checkpoint buffers describe routing/hash geometry.
                        # Preserve exact integers; each family checks their shape
                        # and values before lowering to portable I32 initializers.
                        array = handle.get_tensor(name)
                    else:
                        _fail(f"source tensor {name} has unsupported storage {dtype}", "VXHF_WEIGHTS")
                    weights[name] = np.ascontiguousarray(array)
        except ExporterError:
            raise
        except Exception as error:
            _fail(f"cannot read checkpoint shard {path.name}: {error}", "VXHF_SOURCE")
    return source_config, weights, source_dtypes


def _immutable_integer_buffer(name):
    parts = name.split(".")
    # Released DeepSeek checkpoints use the native inference namespace.
    # Draft-stage tables remain integer metadata until their declared stage
    # is explicitly omitted by architecture admission.
    if (len(parts) == 5 and parts[0] in {"layers", "mtp"} and parts[1].isdigit()
            and parts[2:] == ["ffn", "gate", "tid2eid"]):
        return True
    if parts[:2] == ["model", "language_model"]:
        parts = ["model", *parts[2:]]
    if len(parts) < 5 or parts[:2] != ["model", "layers"] or not parts[2].isdigit():
        return False
    tail = ".".join(parts[3:])
    return tail == "mlp.gate.tid2eid" or tail in {
        "ple.ple_embedding.layer_multipliers",
        "ple.ple_embedding.ngram_heads_vocab_sizes",
        "ple.ple_embedding.ngram_heads_offsets",
    }

def write_hf_package(out_path, graph, weights, *, storage_conversion="F32/F16/BF16->F32", weight_precision=None):
    """Write an already validated graph and its converted initializers."""
    if weight_precision is not None:
        if weight_precision not in ("w4a16", "w4a8", "w8a16"):
            _fail("weight_precision must be w4a16, w4a8 or w8a16")
        from .packed_linear import pack_linear_weights
        from .runtime_ir import import_runtime_package
        graph, weights = pack_linear_weights(graph, weights,
            weight_bits=8 if weight_precision == "w8a16" else 4,
            activation_bits=8 if weight_precision == "w4a8" else 16)
        import_runtime_package(graph, weights)
    save_file(weights, out_path, metadata={"format": "pt",
        "volvoxai.hf.storage_conversion": storage_conversion,
        **({"volvoxai.hf.weight_precision": weight_precision} if weight_precision else {})})
    (Path(out_path).parent / "graph.json").write_text(
        json.dumps(graph, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return graph


class GraphBuilder:
    def __init__(self, max_context, outputs, inputs=None):
        self.graph = {"format": GRAPH_FORMAT,
            "dimensions": {"S": {"min": 1, "max": max_context}},
            "inputs": inputs or {
                "token_ids": {"shape": [1, "S"], "dtype": "int32"},
                "position_ids": {"shape": ["S"], "dtype": "int32"},
                "attention_mask": {"shape": [1, "S"], "dtype": "int32"}},
            "nodes": [], "outputs": list(outputs)}
        self.weights = {}

    def add(self, name, array):
        if name in self.weights:
            raise ValueError(f"initializer already defined: {name}")
        self.weights[name] = np.ascontiguousarray(array)
        return name

    def tensor(self, name, op, inputs, shape, params=None, dtype="float32", port="out"):
        self.graph["nodes"].append({"id": name, "opType": op, "inputs": inputs,
            "outputs": {port: {"tensor": name, "dtype": dtype, "shape": list(shape)}},
            "params": params or {}})
        return name

    def multi(self, name, op, inputs, outputs, params=None):
        self.graph["nodes"].append({"id": name, "opType": op, "inputs": inputs,
            "outputs": {port: {"tensor": tensor, "dtype": dtype, "shape": list(shape)}
                for port, (tensor, shape, dtype) in outputs.items()}, "params": params or {}})
        return {port: tensor for port, (tensor, _, _) in outputs.items()}

    def reshape(self, name, value, shape, dtype="float32"):
        return self.tensor(name, "Reshape", {"input": value}, shape, {"shape": list(shape)}, dtype)

    def linear(self, name, value, weight, width, bias=None, shape=None):
        inputs = {"input": value, "weight": weight}
        if bias is not None:
            inputs["bias"] = bias
        return self.tensor(name, "Linear", inputs, shape or [1, "S", width], {"weight_layout": "dout_din"})

    def norm(self, name, value, weight, shape, eps):
        return self.tensor(name, "RMSNorm", {"input": value, "weight": weight}, shape, {"eps": eps})

    def finish(self, execution=None):
        self.graph["execution"] = execution or stateless_execution(self.graph["inputs"], self.graph["outputs"])
        # Lowered bank layouts replace their source tensors. Publish only the
        # initializers the emitted graph reads, avoiding duplicate expert and
        # patch banks or synthetic topology placeholders in model packages.
        referenced = {value for node in self.graph["nodes"] for value in node["inputs"].values()}
        referenced.update(self.graph["outputs"])
        self.weights = {name:value for name,value in self.weights.items() if name in referenced}
        import_runtime_package(self.graph, self.weights)
        return self.graph, self.weights
