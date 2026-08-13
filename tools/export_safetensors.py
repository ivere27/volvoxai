#!/usr/bin/env python3
"""Export ONNX or TFLite graphs as model-agnostic VolvoxAI packages.

Model-family graph construction and vocabulary policy live with their examples.
"""
import importlib
import json
import re
import sys
from pathlib import Path
from safetensors.numpy import save_file


_REPOSITORY_ROOT = str(Path(__file__).resolve().parent.parent)
if __package__ is None and _REPOSITORY_ROOT not in sys.path:
    # Direct script execution otherwise exposes only tools/ as sys.path[0],
    # while exporter modules consistently use their repository package name.
    sys.path.insert(0, _REPOSITORY_ROOT)


def _exporter_module(name: str):
    """Resolve exporter modules for package, script, and repository imports."""

    roots = (f"{__package__}.exporter",) if __package__ else (
        "exporter",
        "tools.exporter",
    )
    missing: ModuleNotFoundError | None = None
    for root in roots:
        module_name = f"{root}.{name}"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            # Fall through only when this candidate package itself is absent;
            # a missing frontend dependency must retain its original error.
            if error.name is None or not (
                module_name == error.name or module_name.startswith(error.name + ".")
            ):
                raise
            missing = error
    assert missing is not None
    raise missing


classify_package = _exporter_module("capabilities").classify_package
GRAPH_FORMAT = _exporter_module("quantization_storage").GRAPH_FORMAT
concrete_broadcast_shape = _exporter_module(
    "typed_broadcast"
).concrete_broadcast_shape


def dumps_with_compact_lists(obj, indent=2):
    indent_unit = " " * indent

    def render(value, level=0):
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = list(value.items())
            lines = ["{"]
            for i, (k, v) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                lines.append(
                    f"{indent_unit * (level + 1)}{json.dumps(k)}: {render(v, level + 1)}{comma}"
                )
            lines.append(f"{indent_unit * level}}}")
            return "\n".join(lines)
        if isinstance(value, (list, tuple)):
            if not value:
                return "[]"
            if all(not isinstance(item, (dict, list, tuple)) for item in value):
                return "[" + ", ".join(json.dumps(item) for item in value) + "]"
            lines = ["["]
            for i, item in enumerate(value):
                comma = "," if i < len(value) - 1 else ""
                lines.append(f"{indent_unit * (level + 1)}{render(item, level + 1)}{comma}")
            lines.append(f"{indent_unit * level}]")
            return "\n".join(lines)
        return json.dumps(value)

    return render(obj) + "\n"


def _path_suggests_float16(path: str) -> bool:
    name = Path(path).name.lower()
    return any(token in name for token in ("float16", "fp16", "f16"))


def _resolve_output_names(output_count: int, requested=None):
    """Return model-agnostic canonical names in source-output order."""
    if requested is None:
        return [f"output{index}" for index in range(output_count)]
    names = list(requested)
    if len(names) != output_count:
        raise ValueError(
            f"Expected {output_count} --output-name value(s), received {len(names)}"
        )
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Output names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("Output names must be unique")
    return names


def _apply_image_normalizations(inputs, requested=None):
    """Reject application preprocessing at the closed graph boundary."""
    if not requested:
        return
    raise ValueError(
        "--image-normalization is application preprocessing and cannot be "
        "embedded in closed volvox-graph/v1 input descriptors; configure it "
        "in the consuming application"
    )


def _require_closed_dynamic_v1(graph):
    """Accept only the sole executable package schema from delegated frontends."""

    required_root = {
        "format", "dimensions", "inputs", "nodes", "outputs",
    }
    if not isinstance(graph, dict):
        raise RuntimeError("delegated exporter did not return a graph object")
    if graph.get("format") != GRAPH_FORMAT:
        raise RuntimeError("delegated exporter returned a non-current graph discriminator")
    if not required_root <= set(graph) or set(graph) - required_root - {"banks", "quantization"}:
        raise RuntimeError("delegated exporter returned a non-closed graph root")
    if not isinstance(graph.get("dimensions"), dict) or not isinstance(graph.get("inputs"), dict):
        raise RuntimeError("delegated exporter returned malformed shape declarations")
    for name, descriptor in graph["inputs"].items():
        if not isinstance(name, str) or not isinstance(descriptor, dict) or set(descriptor) != {"shape", "dtype"}:
            raise RuntimeError("delegated exporter returned a non-closed input descriptor")
    if not isinstance(graph.get("nodes"), list):
        raise RuntimeError("delegated exporter returned malformed nodes")
    for node in graph["nodes"]:
        if not isinstance(node, dict) or set(node) != {"id", "opType", "inputs", "outputs", "params"}:
            raise RuntimeError("delegated exporter returned a non-closed node descriptor")
        outputs = node.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise RuntimeError("delegated exporter returned malformed node outputs")
        if any(
            not isinstance(descriptor, dict)
            or set(descriptor) != {"tensor", "dtype", "shape"}
            for descriptor in outputs.values()
        ):
            raise RuntimeError("delegated exporter returned split or extended node outputs")
    if not isinstance(graph.get("outputs"), list):
        raise RuntimeError("delegated exporter returned malformed public outputs")


def export_onnx_model(
    model_path: str,
    out_path: str,
    weight_dtype: str = "auto",
    output_names=None,
    image_normalizations=None,
    input_shapes=None,
    dimension_bounds=None,
    anonymous_dimension_bounds=None,
    input_dtypes=None,
    output_dtypes=None,
    specialize_inputs=None,
    quant_mode: str = "preserve",
    allow_silu_numerical_migration: bool = False,
    allow_quantized_bias_folding_numerical_migration: bool = False,
    allow_static_qdq_qbatch_matmul_numerical_migration: bool = False,
    allow_static_qdq_groupnorm_silu_numerical_migration: bool = False,
    enable_static_qdq_layout_optimization: bool = True,
    enable_exact_common_subexpression_elimination: bool = False,
    report_callback=None,
):
    """Compile a static ONNX graph through the typed current frontend.

    The source-faithful import boundary preserves QDQ, integer ABIs, LoRA
    branches, and static selectors before verified RuntimeIR lowering.
    """
    if quant_mode not in ("preserve", "require-w8a8"):
        raise ValueError(f"Unsupported quant mode: {quant_mode}")
    compile_onnx_model = _exporter_module("frontend_onnx").compile_onnx_model

    graph = compile_onnx_model(
        model_path,
        out_path,
        weight_dtype=weight_dtype,
        output_names=output_names,
        image_normalizations=image_normalizations,
        input_shapes=input_shapes,
        dimension_bounds=dimension_bounds,
        anonymous_dimension_bounds=anonymous_dimension_bounds,
        input_dtypes=input_dtypes,
        output_dtypes=output_dtypes,
        specialize_inputs=specialize_inputs,
        allow_silu_numerical_migration=allow_silu_numerical_migration,
        allow_quantized_bias_folding_numerical_migration=(
            allow_quantized_bias_folding_numerical_migration
        ),
        allow_static_qdq_qbatch_matmul_numerical_migration=(
            allow_static_qdq_qbatch_matmul_numerical_migration
        ),
        allow_static_qdq_groupnorm_silu_numerical_migration=(
            allow_static_qdq_groupnorm_silu_numerical_migration
        ),
        enable_static_qdq_layout_optimization=(
            enable_static_qdq_layout_optimization
        ),
        enable_exact_common_subexpression_elimination=(
            enable_exact_common_subexpression_elimination
        ),
        report_callback=report_callback,
    )
    _require_closed_dynamic_v1(graph)
    package_class = classify_package(graph)
    if quant_mode == "require-w8a8" and package_class != "w8a8-v1":
        raise RuntimeError(
            "require-w8a8 requested, but the source graph does not describe a "
            "complete canonical W8A8 activation island"
        )
    return graph


def export_tflite_model(
    model_path: str,
    out_path: str,
    weight_dtype: str = "auto",
    output_names=None,
    image_normalizations=None,
    allow_silu_numerical_migration: bool = False,
    allow_quantized_bias_folding_numerical_migration: bool = False,
    allow_static_qdq_qbatch_matmul_numerical_migration: bool = False,
    allow_static_qdq_groupnorm_silu_numerical_migration: bool = False,
    enable_static_qdq_layout_optimization: bool = True,
    enable_exact_common_subexpression_elimination: bool = False,
):
    import math
    import struct
    from collections import Counter

    import numpy as np
    from flatbuffers import number_types as N
    from flatbuffers.table import Table

    import_tflite_source = _exporter_module("importers").import_tflite_source

    # Decode and verify the complete FlatBuffer through the official LiteRT
    # schema before any target lowering. The source IR is the lossless audit
    # boundary; recognizers below may only lower constructs they can prove.
    source_ir = import_tflite_source(model_path)

    print(f"[Export] Loading TFLite flatbuffer {model_path}...")
    raw = bytearray(Path(model_path).read_bytes())
    if len(raw) < 8 or raw[4:8] != b"TFL3":
        raise RuntimeError(f"{model_path} is not a TFLite flatbuffer with TFL3 identifier")

    def table_at(pos):
        return Table(raw, pos)

    def off(table, field):
        return table.Offset(4 + field * 2)

    def scalar(table, field, default=0, flags=N.Int32Flags):
        return table.GetSlot(4 + field * 2, default, flags)

    def vec_len(table, field):
        o = off(table, field)
        return table.VectorLen(o) if o else 0

    def vec_table(table, field, idx):
        o = off(table, field)
        if not o:
            raise IndexError(field)
        start = table.Vector(o)
        return table_at(table.Indirect(start + idx * 4))

    def vec_i32(table, field):
        o = off(table, field)
        if not o:
            return []
        start = table.Vector(o)
        return [table.Get(N.Int32Flags, start + i * 4) for i in range(table.VectorLen(o))]

    def vec_i64(table, field):
        o = off(table, field)
        if not o:
            return []
        start = table.Vector(o)
        return [table.Get(N.Int64Flags, start + i * 8) for i in range(table.VectorLen(o))]

    def vec_f32(table, field):
        o = off(table, field)
        if not o:
            return []
        start = table.Vector(o)
        return [float(table.Get(N.Float32Flags, start + i * 4)) for i in range(table.VectorLen(o))]

    def vec_bytes(table, field):
        o = off(table, field)
        if not o:
            return b""
        start = table.Vector(o)
        return bytes(raw[start:start + table.VectorLen(o)])

    def string_field(table, field):
        o = off(table, field)
        return table.String(o + table.Pos).decode("utf-8", "replace") if o else ""

    def table_field(table, field):
        o = off(table, field)
        return table_at(table.Indirect(o + table.Pos)) if o else None

    root = table_at(struct.unpack_from("<I", raw, 0)[0])
    if vec_len(root, 2) != 1:
        raise RuntimeError("Direct TFLite export currently supports one subgraph")
    subgraph = vec_table(root, 2, 0)
    opcodes = [vec_table(root, 1, i) for i in range(vec_len(root, 1))]
    buffers = [vec_table(root, 4, i) for i in range(vec_len(root, 4))]
    tensor_tables = [vec_table(subgraph, 0, i) for i in range(vec_len(subgraph, 0))]
    op_tables = [vec_table(subgraph, 3, i) for i in range(vec_len(subgraph, 3))]

    builtin_names = {
        0: "ADD",
        2: "CONCATENATION",
        3: "CONV_2D",
        4: "DEPTHWISE_CONV_2D",
        6: "DEQUANTIZE",
        14: "LOGISTIC",
        17: "MAX_POOL_2D",
        22: "RESHAPE",
        97: "RESIZE_NEAREST_NEIGHBOR",
        114: "QUANTIZE",
    }
    dtype_np = {
        0: np.float32,
        1: np.float16,
        2: np.int32,
        3: np.uint8,
        4: np.int64,
        9: np.int8,
    }
    dtype_names = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        9: "int8",
    }

    def product(shape):
        total = 1
        for d in shape:
            total *= int(d)
        return int(total)

    tensors_meta = []
    for idx, table in enumerate(tensor_tables):
        dtype = scalar(table, 1, 0, N.Uint8Flags)
        q = table_field(table, 4)
        tensors_meta.append({
            "shape": [int(v) for v in vec_i32(table, 0)],
            "dtype": int(dtype),
            "buffer": int(scalar(table, 2, 0, N.Uint32Flags)),
            "name": string_field(table, 3) or f"tensor_{idx}",
            "scale": vec_f32(q, 2) if q else [],
            "zero_point": [int(v) for v in vec_i64(q, 3)] if q else [],
            "quantized_dimension": int(scalar(q, 6, 0, N.Int32Flags)) if q else 0,
        })

    input_tids = vec_i32(subgraph, 1)
    output_tids = vec_i32(subgraph, 2)

    const_alias = {}

    def buffer_data(buffer_idx):
        if buffer_idx < 0 or buffer_idx >= len(buffers):
            return b""
        return vec_bytes(buffers[buffer_idx], 0)

    def resolve_const(tid):
        seen = set()
        while tid in const_alias and tid not in seen:
            seen.add(tid)
            tid = const_alias[tid]
        return tid

    def tensor_array(tid):
        source_tid = resolve_const(tid)
        meta = tensors_meta[source_tid]
        data = buffer_data(meta["buffer"])
        if not data:
            return None
        dtype = dtype_np.get(meta["dtype"])
        if dtype is None:
            raise RuntimeError(f"Unsupported TFLite tensor dtype {meta['dtype']} on tensor {source_tid}")
        arr = np.frombuffer(data, dtype=dtype)
        expected = product(meta["shape"]) if meta["shape"] else 1
        if expected and arr.size < expected:
            raise RuntimeError(f"TFLite tensor {source_tid} buffer is smaller than declared shape")
        if expected:
            arr = arr[:expected].reshape(meta["shape"])
        return arr

    requested = (weight_dtype or "auto").lower()
    if requested in ("float16", "fp16", "f16"):
        float_storage = "float16"
    elif requested in ("float32", "fp32", "f32"):
        float_storage = "float32"
    elif requested == "auto":
        has_f16_const = any(m["dtype"] == 1 and buffer_data(m["buffer"]) for m in tensors_meta)
        float_storage = "float16" if has_f16_const or _path_suggests_float16(model_path) else "float32"
    else:
        raise ValueError(f"Unsupported weight dtype: {requested}")
    store_float16 = float_storage == "float16"
    print(f"[Export] TFLite float tensor storage: {float_storage}")

    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    name_map = {}
    used = set()
    next_tmp = 0
    next_w = 0

    def short(original, preferred=None, weight=False):
        nonlocal next_tmp, next_w
        if original in name_map:
            return name_map[original]
        if preferred:
            base = preferred
        elif weight:
            base = f"w{next_w}"
            next_w += 1
        else:
            base = f"v{next_tmp}"
            next_tmp += 1
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:96].strip("._-") or "t"
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        name_map[original] = candidate
        return candidate

    def tkey(tid):
        return f"tflite_{tid}"

    inputs_def = {}
    affine_descriptors = {}
    value_name = {}
    value_shape = {}
    value_layout = {}

    # Canonical W8A8 graph metadata is carried by the typed tensor edge, not
    # by an adjacent Conv/Linear parameter list.  Keep the TFLite descriptor
    # extraction in one place so inputs, node outputs, and physical weights
    # agree exactly.
    def tensor_dtype(tid):
        return dtype_names.get(tensors_meta[resolve_const(tid)]["dtype"])

    def tensor_quantization(tid, *, axis_override=None, axis_length=None, force_per_axis=False):
        source_tid = resolve_const(tid)
        meta = tensors_meta[source_tid]
        dtype = dtype_names.get(meta["dtype"])
        if dtype not in ("int8", "uint8"):
            return None
        scales = [float(value) for value in meta.get("scale", [])]
        zero_points = [int(value) for value in meta.get("zero_point", [])] or [0] * len(scales)
        if not scales or len(zero_points) not in (1, len(scales)):
            return None
        if not all(math.isfinite(value) and value > 0.0 for value in scales):
            return None
        minimum, maximum = (-128, 127) if dtype == "int8" else (0, 255)
        if not all(isinstance(value, int) and minimum <= value <= maximum for value in zero_points):
            return None
        if len(zero_points) == 1 and len(scales) > 1:
            zero_points = zero_points * len(scales)
        if len(scales) == 1 and not force_per_axis:
            return {
                "scheme": "per_tensor",
                "scale": float(scales[0]),
                "zero_point": int(zero_points[0]),
            }
        axis = int(meta.get("quantized_dimension", 0)) if axis_override is None else axis_override
        if not isinstance(axis, int) or axis < 0 or axis >= len(meta["shape"]):
            return None
        channels = int(meta["shape"][axis]) if axis_length is None else int(axis_length)
        if channels <= 0:
            return None
        if len(scales) == 1:
            scales = scales * channels
            zero_points = zero_points * channels
        if len(scales) != channels:
            return None
        return {
            "scheme": "per_axis",
            "axis": int(axis),
            "scales": [float(value) for value in scales],
            "zero_points": [int(value) for value in zero_points],
        }

    for idx, tid in enumerate(input_tids):
        name = short(tkey(tid), f"input{idx}")
        shape = tensors_meta[tid]["shape"]
        dtype = dtype_names.get(tensors_meta[tid]["dtype"], "float32")
        inputs_def[name] = {"shape": shape, "dtype": dtype}
        quantization = tensor_quantization(tid)
        if quantization is not None:
            affine_descriptors[name] = quantization
        value_name[tid] = name
        value_shape[tid] = shape
        value_layout[tid] = "NHWC" if len(shape) == 4 else "other"
    _apply_image_normalizations(inputs_def, image_normalizations)

    canonical_outputs = _resolve_output_names(len(output_tids), output_names)
    for tid, preferred in zip(output_tids, canonical_outputs):
        short(tkey(tid), preferred)

    tensors = {}
    weight_names = {}
    nodes = []

    def save_array(name, arr):
        arr = np.asarray(arr)
        if arr.dtype.kind == "f":
            dtype = np.float16 if store_float16 else np.float32
            arr = np.asarray(arr, dtype=dtype)
        elif arr.dtype not in (np.int8, np.uint8, np.int32, np.int64):
            arr = np.asarray(arr, dtype=np.float32)
        tensors[name] = np.ascontiguousarray(arr)

    def qparams(tid):
        meta = tensors_meta[resolve_const(tid)]
        scale = [float(v) for v in meta.get("scale", [])]
        zp = [int(v) for v in meta.get("zero_point", [])]
        return {
            "scale": scale,
            "zero_point": zp,
            "axis": int(meta.get("quantized_dimension", 0)),
        }

    def q_scalar(q, key, default):
        values = q.get(key, [])
        return values[0] if values else default

    def ensure_weight(tid, transform="raw", channels=None):
        source_tid = resolve_const(tid)
        key = (source_tid, transform, channels)
        if key in weight_names:
            return weight_names[key]
        arr = tensor_array(tid)
        if arr is None:
            raise RuntimeError(f"TFLite tensor {tid} is not a constant weight")
        if transform == "conv":
            if arr.ndim != 4:
                raise RuntimeError(f"Conv weight tensor {tid} is rank {arr.ndim}, expected 4")
            # TFLite stores Conv2D filters as OHWI. Keep that artifact layout;
            # native prepares a compute cache for the image-layout microkernels.
        elif transform == "conv_hwio":
            if arr.ndim != 4:
                raise RuntimeError(f"Conv weight tensor {tid} is rank {arr.ndim}, expected 4")
            # OHWI [O,H,W,I] -> HWIO, the layout the ordinary Conv2D microkernels
            # index directly, so no runtime transpose or second copy is needed.
            arr = np.ascontiguousarray(np.transpose(arr, (1, 2, 3, 0)))
        elif transform == "depthwise":
            if arr.ndim != 4 or arr.shape[0] != 1:
                raise RuntimeError(f"Depthwise weight tensor {tid} has unsupported shape {list(arr.shape)}")
            # TFLite depthwise filters are 1HWO, where O = input_channels * multiplier.
            # Keep that artifact layout; native prepares HWCM for compute.
        elif transform == "depthwise_hwcm":
            if arr.ndim != 4 or arr.shape[0] != 1:
                raise RuntimeError(f"Depthwise weight tensor {tid} has unsupported shape {list(arr.shape)}")
            # 1HWO -> HWCM. O is already channel-major over the multiplier, so the
            # split is a reshape; HWCM is what selects the depthwise microkernels.
            if not channels or arr.shape[3] % channels:
                raise RuntimeError(
                    f"Depthwise weight tensor {tid} output channels {arr.shape[3]} "
                    f"are not a multiple of input channels {channels}"
                )
            arr = np.ascontiguousarray(
                arr.reshape(arr.shape[1], arr.shape[2], channels, arr.shape[3] // channels)
            )
        elif transform == "depthwise_q":
            if arr.ndim != 4 or arr.shape[0] != 1:
                raise RuntimeError(f"Quantized depthwise weight tensor {tid} has unsupported shape {list(arr.shape)}")
            # The canonical QConv2D contract is OHWI for both regular and
            # depthwise convolution. TFLite's 1HWO becomes [O,H,W,1].
            arr = np.transpose(arr, (3, 1, 2, 0)).copy()
        elif transform != "raw":
            raise RuntimeError(f"Unknown weight transform: {transform}")
        name = short(f"{tkey(source_tid)}_{transform}", weight=True)
        save_array(name, arr)
        weight_names[key] = name
        return name

    def ensure_quantized_conv_weight(tid, transform):
        name = ensure_weight(tid, transform)
        source_tid = resolve_const(tid)
        source_meta = tensors_meta[source_tid]
        if transform == "depthwise_q":
            # Per-output scale values are ordered by the original O axis 3;
            # after conversion that becomes canonical axis 0.
            descriptor = tensor_quantization(source_tid, axis_override=0,
                axis_length=int(tensors[name].shape[0]), force_per_axis=True)
            if descriptor is not None:
                descriptor["axis"] = 0
                channels = int(tensors[name].shape[0])
                if len(descriptor["scales"]) != channels:
                    descriptor = None
        else:
            descriptor = tensor_quantization(source_tid, force_per_axis=True)
            if descriptor is not None and descriptor.get("axis") != 0:
                descriptor = None
        if descriptor is None:
            raise RuntimeError(f"TFLite quantized convolution weight {tid} lacks canonical output-channel quantization metadata")
        affine_descriptors[name] = descriptor
        return name

    def ensure_qparam_weight(tid, kind):
        source_tid = resolve_const(tid)
        key = (source_tid, kind)
        if key in weight_names:
            return weight_names[key]
        q = qparams(source_tid)
        if kind == "scale":
            arr = np.asarray(q["scale"] or [1.0], dtype=np.float32)
        elif kind == "zero_point":
            arr = np.asarray(q["zero_point"] or [0], dtype=np.int32)
        else:
            raise RuntimeError(f"Unknown quant parameter kind: {kind}")
        name = short(f"{tkey(source_tid)}_{kind}", weight=True)
        save_array(name, arr)
        weight_names[key] = name
        return name

    def ensure_typed_zero_point(tid):
        source_tid = resolve_const(tid)
        key = (source_tid, "typed_zero_point")
        if key in weight_names:
            return weight_names[key]
        dtype = tensor_dtype(source_tid)
        q = qparams(source_tid)
        if dtype not in ("int8", "uint8") or len(q["zero_point"] or [0]) != 1:
            raise RuntimeError(f"TFLite tensor {tid} needs a scalar I8/U8 zero point")
        array_dtype = np.int8 if dtype == "int8" else np.uint8
        name = short(f"{tkey(source_tid)}_typed_zero_point", weight=True)
        save_array(name, np.asarray([q_scalar(q, "zero_point", 0)], dtype=array_dtype))
        weight_names[key] = name
        return name

    def add_node(
        op, inputs, out_name, out_shape, params=None, out_tid=None, *,
        out_dtype=None,
    ):
        shape = [int(v) for v in out_shape]
        dtype = tensor_dtype(out_tid) if out_tid is not None else out_dtype
        if dtype not in ("float32", "int32", "int8", "uint8"):
            source = f"TFLite tensor {out_tid}" if out_tid is not None else "synthetic output"
            raise RuntimeError(
                f"Unsupported exported tensor dtype {dtype!r} on {source}"
            )
        canonical_params = dict(params or {})
        if op in {"Reshape", "Expand"}:
            canonical_params.setdefault("shape", list(shape))
        node = {
            "id": f"node_{len(nodes)}",
            "opType": op,
            "inputs": inputs,
            "outputs": {
                "out": {"tensor": out_name, "dtype": dtype, "shape": shape},
            },
            "params": canonical_params,
        }
        if out_tid is not None:
            quantization = tensor_quantization(out_tid)
            if quantization is not None:
                affine_descriptors[out_name] = quantization
        nodes.append(node)

    def set_value(tid, name, shape, layout):
        value_name[tid] = name
        value_shape[tid] = [int(v) for v in shape]
        value_layout[tid] = layout

    def name_for_value(tid):
        if tid in value_name:
            return value_name[tid]
        if tensor_array(tid) is not None:
            return ensure_weight(tid, "raw")
        raise RuntimeError(f"TFLite tensor {tid} has no produced value")

    def same_pads(in_size, out_size, kernel, stride, dilation):
        effective = dilation * (kernel - 1) + 1
        total = max((out_size - 1) * stride + effective - in_size, 0)
        before = total // 2
        return before, total - before

    def conv_pads(padding, in_shape, out_shape, kernel, stride, dilation, layout="NHWC"):
        if padding != 0:
            return [0, 0, 0, 0]
        in_h, in_w = in_shape[1], in_shape[2]
        out_h, out_w = out_shape[1], out_shape[2]
        top, bottom = same_pads(in_h, out_h, kernel[0], stride[0], dilation[0])
        left, right = same_pads(in_w, out_w, kernel[1], stride[1], dilation[1])
        return [top, left, bottom, right]

    def activation_param(act):
        if act == 0:
            return 0
        if act == 1:
            return 1
        if act == 3:
            return 2
        raise RuntimeError(f"Unsupported fused TFLite activation {act}")

    op_counter = Counter()
    folded_dequantize = 0
    lowered_quantize = 0

    for op_idx, op_table in enumerate(op_tables):
        opcode_idx = scalar(op_table, 0, 0, N.Uint32Flags)
        builtin = scalar(opcodes[opcode_idx], 0, 0, N.Int8Flags)
        op_name = builtin_names.get(builtin, str(builtin))
        op_counter[op_name] += 1
        inputs = vec_i32(op_table, 1)
        outputs = vec_i32(op_table, 2)
        options = table_field(op_table, 4)

        if op_name == "DEQUANTIZE":
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError("Unsupported TFLite DEQUANTIZE arity")
            if tensor_array(inputs[0]) is not None:
                const_alias[outputs[0]] = resolve_const(inputs[0])
                folded_dequantize += 1
                continue
            in_name = name_for_value(inputs[0])
            out_shape = tensors_meta[outputs[0]]["shape"]
            out_name = short(tkey(outputs[0]))
            scale = ensure_qparam_weight(inputs[0], "scale")
            add_node("DequantizeLinear", {
                "input": in_name,
                "scale": scale,
                "zero_point": ensure_typed_zero_point(inputs[0]),
            }, out_name, out_shape, out_tid=outputs[0])
            set_value(outputs[0], out_name, out_shape, "NHWC" if len(out_shape) == 4 else "other")
            continue

        if not outputs:
            continue
        out_tid = outputs[0]
        out_tflite_shape = tensors_meta[out_tid]["shape"]
        out_name = short(tkey(out_tid))

        if op_name == "QUANTIZE":
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError("Unsupported TFLite QUANTIZE arity")
            in_name = name_for_value(inputs[0])
            output_quantization = tensor_quantization(out_tid)
            if output_quantization is None or output_quantization.get("scheme") != "per_tensor":
                raise RuntimeError(f"TFLite QUANTIZE output {out_tid} needs scalar I8/U8 quantization metadata")
            input_dtype = tensor_dtype(inputs[0])
            if input_dtype in ("int8", "uint8"):
                input_quantization = tensor_quantization(inputs[0])
                if input_quantization is None or input_quantization.get("scheme") != "per_tensor":
                    raise RuntimeError(f"TFLite QUANTIZE input {inputs[0]} needs scalar I8/U8 quantization metadata")
                add_node(
                    "RequantizeLinear", {"input": in_name}, out_name,
                    out_tflite_shape, out_tid=out_tid,
                )
            elif input_dtype == "float32":
                add_node("QuantizeLinear", {
                    "input": in_name,
                    "scale": ensure_qparam_weight(out_tid, "scale"),
                    "zero_point": ensure_typed_zero_point(out_tid),
                }, out_name, out_tflite_shape, out_tid=out_tid)
            else:
                raise RuntimeError(f"TFLite QUANTIZE input {inputs[0]} has unsupported dtype {input_dtype}")
            set_value(out_tid, out_name, out_tflite_shape, "NHWC" if len(out_tflite_shape) == 4 else "other")
            lowered_quantize += 1
            continue

        if op_name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            if len(inputs) < 2:
                raise RuntimeError(f"{op_name} at op {op_idx} has too few inputs")
            in_name = name_for_value(inputs[0])
            in_shape = value_shape.get(inputs[0], tensors_meta[inputs[0]]["shape"])
            if value_layout.get(inputs[0], "NHWC") != "NHWC" or len(in_shape) != 4 or len(out_tflite_shape) != 4:
                raise RuntimeError(f"{op_name} expects rank-4 tensors")
            out_shape = out_tflite_shape
            weight_meta = tensors_meta[resolve_const(inputs[1])]
            use_qconv = (
                weight_meta["dtype"] in (3, 9)
                and tensor_quantization(inputs[0]) is not None
                and tensor_quantization(out_tid) is not None
            )
            if op_name == "CONV_2D":
                if use_qconv:
                    weight = ensure_quantized_conv_weight(inputs[1], "conv")
                    weight_layout = "OHWI"
                else:
                    weight = ensure_weight(inputs[1], "conv_hwio")
                    weight_layout = "HWIO"
                weight_shape = list(tensors[weight].shape)
                kernel = weight_shape[1:3] if weight_layout == "OHWI" else weight_shape[0:2]
                groups = 1
                stride = [
                    scalar(options, 2, 1, N.Int32Flags) if options else 1,
                    scalar(options, 1, 1, N.Int32Flags) if options else 1,
                ]
                dilation = [
                    scalar(options, 5, 1, N.Int32Flags) if options else 1,
                    scalar(options, 4, 1, N.Int32Flags) if options else 1,
                ]
                act = scalar(options, 3, 0, N.Int8Flags) if options else 0
            else:
                groups = int(in_shape[3])
                if use_qconv:
                    weight = ensure_quantized_conv_weight(inputs[1], "depthwise_q")
                    weight_layout = "OHWI"
                else:
                    weight = ensure_weight(inputs[1], "depthwise_hwcm", channels=groups)
                    weight_layout = "HWCM"
                weight_shape = list(tensors[weight].shape)
                kernel = weight_shape[1:3] if weight_layout == "OHWI" else weight_shape[0:2]
                stride = [
                    scalar(options, 2, 1, N.Int32Flags) if options else 1,
                    scalar(options, 1, 1, N.Int32Flags) if options else 1,
                ]
                dilation = [
                    scalar(options, 6, 1, N.Int32Flags) if options else 1,
                    scalar(options, 5, 1, N.Int32Flags) if options else 1,
                ]
                act = scalar(options, 4, 0, N.Int8Flags) if options else 0
            padding = scalar(options, 0, 0, N.Int8Flags) if options else 0
            pads = conv_pads(padding, in_shape, out_shape, kernel, stride, dilation, layout="NHWC")
            node_inputs = {"input": in_name, "weight": weight}
            if len(inputs) > 2 and inputs[2] >= 0:
                if use_qconv and tensor_dtype(inputs[2]) != "int32":
                    raise RuntimeError(f"TFLite QConv2D bias {inputs[2]} must use I32 accumulator storage")
                node_inputs["bias"] = ensure_weight(inputs[2], "raw")
            params = {
                "stride": stride,
                "dilation": dilation,
                "groups": groups,
                "pads": pads,
                "data_layout": "NHWC",
                "weight_layout": weight_layout,
            }
            relu = activation_param(act)
            if relu:
                params["relu"] = relu
            add_node("QConv2D" if use_qconv else "Conv2D", node_inputs, out_name, out_shape, params, out_tid=out_tid)
            set_value(out_tid, out_name, out_shape, "NHWC")
            continue

        if op_name == "ADD":
            a_name = name_for_value(inputs[0])
            b_name = name_for_value(inputs[1])
            out_shape = out_tflite_shape
            act = scalar(options, 0, 0, N.Int8Flags) if options else 0
            params = {}
            relu = activation_param(act)
            if relu:
                params["relu"] = relu
            quantized_add = (tensor_quantization(inputs[0]) is not None and
                             tensor_quantization(inputs[1]) is not None and
                             tensor_quantization(out_tid) is not None)
            if quantized_add:
                input_shapes = [
                    value_shape.get(tid, tensors_meta[tid]["shape"])
                    for tid in inputs[:2]
                ]
                if concrete_broadcast_shape(
                    input_shapes[0], input_shapes[1],
                ) != tuple(out_shape):
                    raise RuntimeError(
                        "TFLite quantized ADD output is not the exact static broadcast shape"
                    )
                names = [a_name, b_name]
                for port, (tid, shape) in enumerate(
                    zip(inputs[:2], input_shapes)
                ):
                    if tuple(shape) == tuple(out_shape):
                        continue
                    descriptor = tensor_quantization(tid)
                    if descriptor is None or descriptor.get("scheme") != "per_tensor":
                        raise RuntimeError(
                            "TFLite quantized ADD broadcast requires per-tensor I8/U8 operands"
                        )
                    expanded = short(f"{tkey(out_tid)}_qadd_input{port}_expanded")
                    add_node(
                        "Expand", {"input": names[port]}, expanded, out_shape,
                        out_dtype=tensor_dtype(tid),
                    )
                    affine_descriptors[expanded] = descriptor
                    names[port] = expanded
                a_name, b_name = names
            add_node("QAdd" if quantized_add else "Add", {"a": a_name, "b": b_name},
                     out_name, out_shape, params, out_tid=out_tid)
            set_value(out_tid, out_name, out_shape, "NHWC" if len(out_tflite_shape) == 4 else "other")
            continue

        if op_name == "MAX_POOL_2D":
            in_name = name_for_value(inputs[0])
            in_shape = value_shape.get(inputs[0], tensors_meta[inputs[0]]["shape"])
            out_shape = out_tflite_shape
            stride = [
                scalar(options, 2, 1, N.Int32Flags) if options else 1,
                scalar(options, 1, 1, N.Int32Flags) if options else 1,
            ]
            kernel = [
                scalar(options, 4, 1, N.Int32Flags) if options else 1,
                scalar(options, 3, 1, N.Int32Flags) if options else 1,
            ]
            padding = scalar(options, 0, 0, N.Int8Flags) if options else 0
            pads = conv_pads(padding, in_shape, out_shape, kernel, stride, [1, 1], layout="NHWC")
            act = scalar(options, 5, 0, N.Int8Flags) if options else 0
            if act:
                raise RuntimeError("Direct TFLite export does not yet support fused activation on MAX_POOL_2D")
            add_node("MaxPool2D", {"input": in_name}, out_name, out_shape, {
                "kernel": kernel,
                "stride": stride,
                "pads": pads,
                "data_layout": "NHWC",
            }, out_tid=out_tid)
            set_value(out_tid, out_name, out_shape, "NHWC")
            continue

        if op_name == "RESIZE_NEAREST_NEIGHBOR":
            in_name = name_for_value(inputs[0])
            out_shape = out_tflite_shape
            add_node("ResizeNearest2D", {"input": in_name}, out_name, out_shape, {
                "mode": "nearest",
                "coordinate_transformation_mode": "asymmetric",
                "data_layout": "NHWC",
            }, out_tid=out_tid)
            set_value(out_tid, out_name, out_shape, "NHWC")
            continue

        if op_name == "RESHAPE":
            in_tid = inputs[0]
            in_name = name_for_value(in_tid)
            add_node("Reshape", {"input": in_name}, out_name, out_tflite_shape, out_tid=out_tid)
            set_value(out_tid, out_name, out_tflite_shape, "NHWC" if len(out_tflite_shape) == 4 else "other")
            continue

        if op_name == "CONCATENATION":
            rank = len(out_tflite_shape)
            axis = scalar(options, 0, 0, N.Int32Flags) if options else 0
            act = scalar(options, 1, 0, N.Int8Flags) if options else 0
            if act:
                raise RuntimeError("Direct TFLite export does not yet support fused activation on CONCATENATION")
            if rank == 4:
                out_shape = out_tflite_shape
                node_inputs = {f"input{i}": name_for_value(tid) for i, tid in enumerate(inputs)}
                layout = "NHWC"
            else:
                out_shape = out_tflite_shape
                node_inputs = {f"input{i}": name_for_value(tid) for i, tid in enumerate(inputs)}
                layout = "other"
            add_node("Concat", node_inputs, out_name, out_shape, {"axis": axis}, out_tid=out_tid)
            set_value(out_tid, out_name, out_shape, layout)
            continue

        if op_name == "LOGISTIC":
            in_name = name_for_value(inputs[0])
            in_shape = value_shape.get(inputs[0], tensors_meta[inputs[0]]["shape"])
            if tensor_quantization(inputs[0]) is not None and tensor_quantization(out_tid) is not None:
                f32_input = short(f"{tkey(out_tid)}_sigmoid_input")
                f32_output = short(f"{tkey(out_tid)}_sigmoid_f32")
                add_node("DequantizeLinear", {
                    "input": in_name,
                    "scale": ensure_qparam_weight(inputs[0], "scale"),
                    "zero_point": ensure_typed_zero_point(inputs[0]),
                }, f32_input, in_shape, out_dtype="float32")
                add_node(
                    "Sigmoid", {"input": f32_input}, f32_output, in_shape,
                    out_dtype="float32",
                )
                add_node("QuantizeLinear", {
                    "input": f32_output,
                    "scale": ensure_qparam_weight(out_tid, "scale"),
                    "zero_point": ensure_typed_zero_point(out_tid),
                }, out_name, in_shape, out_tid=out_tid)
            else:
                add_node("Sigmoid", {"input": in_name}, out_name, in_shape, out_tid=out_tid)
            set_value(out_tid, out_name, in_shape, value_layout.get(inputs[0], "other"))
            continue

        raise RuntimeError(f"Unsupported TFLite op for direct Volvox export: {op_name} ({builtin})")

    for tid in output_tids:
        wanted = short(tkey(tid))
        current = name_for_value(tid)
        if current != wanted:
            add_node("Identity", {"input": current}, wanted, value_shape.get(tid, tensors_meta[tid]["shape"]), out_tid=tid)
            set_value(tid, wanted, value_shape.get(tid, tensors_meta[tid]["shape"]), value_layout.get(tid, "other"))

    raw_node_count = len(nodes)
    declared_tensors = set(inputs_def) | set(tensors)
    for node in nodes:
        declared_tensors.update(
            descriptor.get("tensor")
            for descriptor in (node.get("outputs") or {}).values()
            if isinstance(descriptor, dict)
            and isinstance(descriptor.get("tensor"), str)
        )
    affine_descriptors = {
        name: descriptor for name, descriptor in affine_descriptors.items()
        if name in declared_tensors
    }
    package_class = classify_package(
        {"inputs": inputs_def, "nodes": nodes},
        tensors,
    )

    graph = {
        "format": GRAPH_FORMAT,
        "dimensions": {},
        "inputs": inputs_def,
        "outputs": [short(tkey(tid)) for tid in output_tids],
        "nodes": nodes,
    }

    externalize_quantization = _exporter_module(
        "quantization_storage"
    ).externalize_quantization
    quantization_report = externalize_quantization(
        graph, tensors, affine_descriptors
    )

    typed_pipeline = _exporter_module("optimizer.typed_pipeline")
    optimize_runtime_package = typed_pipeline.optimize_runtime_package
    graph, optimized_tensors, typed_report = optimize_runtime_package(
        graph,
        tensors,
        source_name=f"{Path(model_path).name}:lowered",
        allow_silu_numerical_migration=allow_silu_numerical_migration,
        allow_quantized_bias_folding_numerical_migration=(
            allow_quantized_bias_folding_numerical_migration
        ),
        allow_static_qdq_qbatch_matmul_numerical_migration=(
            allow_static_qdq_qbatch_matmul_numerical_migration
        ),
        allow_static_qdq_groupnorm_silu_numerical_migration=(
            allow_static_qdq_groupnorm_silu_numerical_migration
        ),
        enable_static_qdq_layout_optimization=(
            enable_static_qdq_layout_optimization
        ),
        enable_exact_common_subexpression_elimination=(
            enable_exact_common_subexpression_elimination
        ),
        shape_profile={},
    )
    tensors = {name: np.asarray(value) for name, value in optimized_tensors.items()}

    save_file(tensors, str(out_path))
    graph_path = out_dir / "graph.json"
    graph_path.write_text(dumps_with_compact_lists(graph, indent=2), encoding="utf-8")
    print(
        f"[Export] TFLite ops={len(op_tables)} "
        f"folded_dequantize={folded_dequantize} "
        f"lowered_nodes={raw_node_count} "
        f"optimized_nodes={len(graph['nodes'])} weights={len(tensors)} "
        f"package_class={package_class} "
        f"affine_parameters={quantization_report.scales_created + quantization_report.zero_points_created} "
        f"optimizer_runs={len(typed_report.runs)} "
        f"source_ir_nodes={len(source_ir.nodes)} "
        f"ops={','.join(f'{name}:{count}' for name, count in sorted(op_counter.items())) or 'none'}"
    )
    print(f"[Export] Wrote {out_path} and {graph_path}")


def export_model(
    model_path: str,
    out_path: str,
    weight_dtype: str = "auto",
    output_names=None,
    image_normalizations=None,
    input_shapes=None,
    dimension_bounds=None,
    anonymous_dimension_bounds=None,
    input_dtypes=None,
    output_dtypes=None,
    specialize_inputs=None,
    quant_mode: str = "preserve",
    allow_silu_numerical_migration: bool = False,
    allow_quantized_bias_folding_numerical_migration: bool = False,
    allow_static_qdq_qbatch_matmul_numerical_migration: bool = False,
    allow_static_qdq_groupnorm_silu_numerical_migration: bool = False,
    enable_static_qdq_layout_optimization: bool = True,
    enable_exact_common_subexpression_elimination: bool = False,
    report_callback=None,
):
    if quant_mode not in ("preserve", "require-w8a8"):
        raise ValueError(f"Unsupported quant mode: {quant_mode}")
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        export_onnx_model(
            model_path,
            out_path,
            weight_dtype=weight_dtype,
            output_names=output_names,
            image_normalizations=image_normalizations,
            input_shapes=input_shapes,
            dimension_bounds=dimension_bounds,
            anonymous_dimension_bounds=anonymous_dimension_bounds,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            specialize_inputs=specialize_inputs,
            quant_mode=quant_mode,
            allow_silu_numerical_migration=allow_silu_numerical_migration,
            allow_quantized_bias_folding_numerical_migration=(
                allow_quantized_bias_folding_numerical_migration
            ),
            allow_static_qdq_qbatch_matmul_numerical_migration=(
                allow_static_qdq_qbatch_matmul_numerical_migration
            ),
            allow_static_qdq_groupnorm_silu_numerical_migration=(
                allow_static_qdq_groupnorm_silu_numerical_migration
            ),
            enable_static_qdq_layout_optimization=(
                enable_static_qdq_layout_optimization
            ),
            enable_exact_common_subexpression_elimination=(
                enable_exact_common_subexpression_elimination
            ),
            report_callback=report_callback,
        )
        return
    if suffix == ".tflite":
        if (input_shapes or dimension_bounds or anonymous_dimension_bounds or
                input_dtypes or output_dtypes or specialize_inputs):
            raise ValueError(
                "TFLite dimension/input shape/dtype bindings and specialization are not "
                "enabled until the TFLite frontend uses the shared static IR"
            )
        export_tflite_model(
            model_path,
            out_path,
            weight_dtype=weight_dtype,
            output_names=output_names,
            image_normalizations=image_normalizations,
            allow_silu_numerical_migration=allow_silu_numerical_migration,
            allow_quantized_bias_folding_numerical_migration=(
                allow_quantized_bias_folding_numerical_migration
            ),
            allow_static_qdq_qbatch_matmul_numerical_migration=(
                allow_static_qdq_qbatch_matmul_numerical_migration
            ),
            allow_static_qdq_groupnorm_silu_numerical_migration=(
                allow_static_qdq_groupnorm_silu_numerical_migration
            ),
            enable_static_qdq_layout_optimization=(
                enable_static_qdq_layout_optimization
            ),
            enable_exact_common_subexpression_elimination=(
                enable_exact_common_subexpression_elimination
            ),
        )
        if quant_mode == "require-w8a8":
            graph = json.loads((Path(out_path).parent / "graph.json").read_text(encoding="utf-8"))
            if classify_package(graph) != "w8a8-v1":
                raise ValueError("require-w8a8 requested, but the TFLite source is not canonical W8A8")
        return
    raise ValueError(
        f"Unsupported source {model_path!r}; the generic exporter accepts only "
        ".onnx and .tflite files. Use an example-owned exporter for model-family "
        "checkpoints."
    )


if __name__ == "__main__":
    exporter_main = _exporter_module("cli").main
    raise SystemExit(exporter_main(export_model))
