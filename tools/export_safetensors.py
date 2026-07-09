#!/usr/bin/env python3
"""
VolvoxAI - Universal Safetensors Exporter
Automatically detects model architecture and exports PyTorch weights to Safetensors format,
embedding the VolvoxAI Graph topology into the config.json.
"""
import os
import argparse
import json
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import save_file

def quantize_to_int8(tensor):
    if not tensor.is_floating_point(): return tensor, None
    max_val = tensor.abs().max()
    scale = max_val / 127.0
    if scale == 0: return tensor.to(torch.int8), torch.tensor([1.0], dtype=torch.float32)
    return torch.round(tensor / scale).to(torch.int8), scale.view(1).float()


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


def _attr(node, name, default=None):
    import onnx
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def _shape_from_value_info(value_info):
    tt = value_info.type.tensor_type
    return [d.dim_value if d.dim_value else 0 for d in tt.shape.dim]


def _dequantize_array(x, scale, zero_point, axis=0):
    import numpy as np
    xf = x.astype(np.float32)
    sf = np.asarray(scale, dtype=np.float32)
    zp = np.asarray(zero_point, dtype=np.float32)
    if sf.ndim == 0 or sf.size == 1:
        return (xf - float(zp.reshape(-1)[0] if zp.size else 0.0)) * float(sf.reshape(-1)[0])
    shape = [1] * xf.ndim
    shape[axis] = sf.shape[0]
    return (xf - zp.reshape(shape)) * sf.reshape(shape)


def _path_suggests_float16(path: str) -> bool:
    name = Path(path).name.lower()
    return any(token in name for token in ("float16", "fp16", "f16"))


def _resolve_onnx_float_storage(model_path: str, model, requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested in ("float16", "fp16", "f16"):
        return "float16"
    if requested in ("float32", "fp32", "f32"):
        return "float32"
    if requested != "auto":
        raise ValueError(f"Unsupported weight dtype: {requested}")

    import onnx
    if any(init.data_type == onnx.TensorProto.FLOAT16 for init in model.graph.initializer):
        return "float16"
    if _path_suggests_float16(model_path):
        return "float16"
    return "float32"


def optimize_export_nodes(nodes):
    """Fold simple single-consumer activation nodes into their producer."""
    removed = {}

    def out_name(node):
        return (node.get("outputs") or {}).get("out")

    def out_shape(node):
        return (node.get("outputs_shape") or {}).get("out", [])

    def add_removed(op):
        removed[op] = removed.get(op, 0) + 1

    changed = True
    while changed:
        changed = False
        uses = {}
        producers = {}
        for idx, node in enumerate(nodes):
            out = out_name(node)
            if out:
                producers[out] = idx
            for inp in (node.get("inputs") or {}).values():
                uses[inp] = uses.get(inp, 0) + 1

        next_nodes = []
        for idx, node in enumerate(nodes):
            op = node.get("op", "")
            inputs = node.get("inputs") or {}
            params = node.get("params") or {}
            src = inputs.get("input")

            if op == "Clip" and src and uses.get(src, 0) == 1:
                if float(params.get("min", -1e30)) == 0.0 and float(params.get("max", 1e30)) == 6.0:
                    prod_idx = producers.get(src, -1)
                    if 0 <= prod_idx < len(nodes):
                        prod = nodes[prod_idx]
                        if prod.get("op") in ("Conv2D", "QConv2D"):
                            prod.setdefault("params", {})["relu"] = 2
                            prod.setdefault("outputs", {})["out"] = out_name(node)
                            prod.setdefault("outputs_shape", {})["out"] = out_shape(node)
                            add_removed(op)
                            changed = True
                            continue

            if op == "Sigmoid" and src and uses.get(src, 0) == 1:
                prod_idx = producers.get(src, -1)
                if 0 <= prod_idx < len(nodes):
                    prod = nodes[prod_idx]
                    if prod.get("op") == "Concat":
                        prod.setdefault("params", {})["sigmoid"] = 1
                        prod.setdefault("outputs", {})["out"] = out_name(node)
                        prod.setdefault("outputs_shape", {})["out"] = out_shape(node)
                        add_removed(op)
                        changed = True
                        continue

            next_nodes.append(node)
        nodes = next_nodes

    return nodes, removed


def export_onnx_model(model_path: str, out_path: str, weight_dtype: str = "auto"):
    import numpy as np
    import onnx
    from onnx import numpy_helper, shape_inference

    print(f"[Export] Loading ONNX graph {model_path}...")
    model = shape_inference.infer_shapes(onnx.load(model_path))
    float_storage = _resolve_onnx_float_storage(model_path, model, weight_dtype)
    store_float16 = float_storage == "float16"
    print(f"[Export] ONNX float tensor storage: {float_storage}")
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_map = {}
    elem_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.HasField("tensor_type"):
            shape_map[vi.name] = _shape_from_value_info(vi)
            elem_map[vi.name] = vi.type.tensor_type.elem_type

    initializer_names = {i.name for i in model.graph.initializer}
    arrays = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for name, arr in arrays.items():
        shape_map.setdefault(name, list(arr.shape))

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

    graph_inputs = [i for i in model.graph.input if i.name not in initializer_names]
    graph_input_names = {i.name for i in graph_inputs}
    inputs_def = {}
    for idx, inp in enumerate(graph_inputs):
        s = short(inp.name, f"input{idx}")
        inputs_def[s] = {"shape": shape_map.get(inp.name, []), "dtype": "float32", "source_name": inp.name}

    for out in model.graph.output:
        shape = shape_map.get(out.name, [])
        preferred = "boxes" if shape and shape[-1] == 4 else "scores"
        short(out.name, preferred)

    alias = {}
    dq_info = {}
    tensors = {}
    nodes = []
    skipped_qdq = 0

    def resolve(name):
        while name in alias:
            name = alias[name]
        return name

    def const_array(name):
        name = resolve(name)
        return arrays.get(name)

    def clean_shape(shape):
        return [int(d) if int(d) > 0 else 0 for d in shape] if shape else []

    def is_concrete(shape):
        return bool(shape) and all(int(d) > 0 for d in shape)

    def shape_of(name):
        rn = resolve(name)
        if rn in arrays:
            return list(arrays[rn].shape)
        return clean_shape(shape_map.get(rn) or shape_map.get(name) or [])

    def out_shape(name):
        return clean_shape(shape_map.get(name, []))

    def product(shape):
        v = 1
        for d in shape:
            if d <= 0:
                return 0
            v *= d
        return v

    def broadcast_shape(a, b):
        if not a:
            return b
        if not b:
            return a
        out = []
        for i in range(1, max(len(a), len(b)) + 1):
            da = a[-i] if i <= len(a) else 1
            db = b[-i] if i <= len(b) else 1
            if da == db:
                out.append(da)
            elif da == 1:
                out.append(db)
            elif db == 1:
                out.append(da)
            elif da == 0:
                out.append(db)
            elif db == 0:
                out.append(da)
            else:
                raise RuntimeError(f"Cannot broadcast ONNX shapes {a} and {b}")
        return list(reversed(out))

    def conv_shape(node):
        x = shape_of(node.input[0])
        w = shape_of(node.input[1])
        if len(x) != 4 or len(w) != 4:
            return []
        strides = list(_attr(node, "strides", [1, 1]))
        dilations = list(_attr(node, "dilations", [1, 1]))
        pads = list(_attr(node, "pads", [0, 0, 0, 0]))
        kh = dilations[0] * (w[2] - 1) + 1
        kw = dilations[1] * (w[3] - 1) + 1
        oh = (x[2] + pads[0] + pads[2] - kh) // strides[0] + 1 if x[2] and w[2] else 0
        ow = (x[3] + pads[1] + pads[3] - kw) // strides[1] + 1 if x[3] and w[3] else 0
        return [x[0], w[0], oh, ow]

    def pool_shape(node):
        x = shape_of(node.input[0])
        if len(x) != 4:
            return []
        kernel = list(_attr(node, "kernel_shape", [1, 1]))
        strides = list(_attr(node, "strides", kernel))
        pads = list(_attr(node, "pads", [0, 0, 0, 0]))
        dilations = list(_attr(node, "dilations", [1, 1]))
        kh = dilations[0] * (kernel[0] - 1) + 1
        kw = dilations[1] * (kernel[1] - 1) + 1
        oh = (x[2] + pads[0] + pads[2] - kh) // strides[0] + 1 if x[2] else 0
        ow = (x[3] + pads[1] + pads[3] - kw) // strides[1] + 1 if x[3] else 0
        return [x[0], x[1], oh, ow]

    def resize_shape(node):
        x = shape_of(node.input[0])
        if not x:
            return []
        sizes = const_array(node.input[3]) if len(node.input) > 3 and node.input[3] else None
        if sizes is not None:
            return [int(v) for v in np.asarray(sizes).reshape(-1)]
        scales = const_array(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        if scales is not None:
            sv = np.asarray(scales, dtype=np.float32).reshape(-1)
            if len(sv) == len(x):
                return [int(round(float(d) * float(s))) if d else 0 for d, s in zip(x, sv)]
        return []

    def reshape_shape(node):
        x = shape_of(node.input[0])
        target = const_array(node.input[1]) if len(node.input) > 1 else None
        if target is None:
            return []
        allowzero = int(_attr(node, "allowzero", 0))
        out = [int(v) for v in np.asarray(target).reshape(-1)]
        for i, d in enumerate(out):
            if d == 0 and not allowzero and i < len(x):
                out[i] = x[i]
        if -1 in out:
            infer_idx = out.index(-1)
            known = 1
            for d in out:
                if d != -1:
                    known *= d
            total = product(x)
            out[infer_idx] = total // known if total and known else 0
        return out

    def concat_shape(node):
        shapes = [shape_of(inp) for inp in node.input]
        shapes = [s for s in shapes if s]
        if not shapes:
            return []
        out = list(shapes[0])
        axis = int(_attr(node, "axis", 1))
        if axis < 0:
            axis += len(out)
        out[axis] = 0
        for s in shapes:
            if axis >= len(s):
                return []
            out[axis] += s[axis]
            for i, d in enumerate(s):
                if i != axis and out[i] == 0:
                    out[i] = d
        return out

    def transpose_shape(node):
        x = shape_of(node.input[0])
        if not x:
            return []
        perm = list(_attr(node, "perm", list(range(len(x)))))
        return [x[i] for i in perm]

    def ensure_weight(name):
        name = resolve(name)
        if name in tensors:
            return short(name, weight=True)
        arr = arrays.get(name)
        if arr is None:
            return short(name)
        dtype = np.float16 if store_float16 and np.asarray(arr).dtype.kind == "f" else np.float32
        tensors[short(name, weight=True)] = torch.from_numpy(np.asarray(arr, dtype=dtype).copy()).contiguous()
        return short(name)

    def ensure_weight_preserve(name):
        name = resolve(name)
        if name in tensors:
            return short(name, weight=True)
        arr = arrays.get(name)
        if arr is None:
            return short(name)
        arr = np.asarray(arr)
        if arr.dtype == np.int8:
            tensors[short(name, weight=True)] = torch.from_numpy(arr.copy()).contiguous()
        elif arr.dtype == np.uint8:
            tensors[short(name, weight=True)] = torch.from_numpy(arr.copy()).contiguous()
        elif arr.dtype == np.int32:
            tensors[short(name, weight=True)] = torch.from_numpy(arr.copy()).contiguous()
        else:
            dtype = np.float16 if store_float16 and arr.dtype.kind == "f" else np.float32
            tensors[short(name, weight=True)] = torch.from_numpy(np.asarray(arr, dtype=dtype).copy()).contiguous()
        return short(name)

    def q_scalar(arr, default=0):
        if arr is None:
            return default
        a = np.asarray(arr).reshape(-1)
        return int(a[0]) if a.size else default

    def add_node(op, inputs, output, params=None, shape=None, prefer_shape=False):
        declared = out_shape(output)
        final_shape = clean_shape(shape) if shape and (prefer_shape or not is_concrete(declared)) else declared
        if not final_shape and shape:
            final_shape = clean_shape(shape)
        if final_shape:
            shape_map[output] = final_shape
        nodes.append({
            "op": op,
            "inputs": inputs,
            "outputs": {"out": short(output)},
            "outputs_shape": {"out": final_shape},
            **({"params": params} if params else {}),
        })

    for node in model.graph.node:
        op = node.op_type
        output = node.output[0] if node.output else ""
        if op == "DequantizeLinear":
            x = resolve(node.input[0])
            scale = const_array(node.input[1]) if len(node.input) > 1 else None
            zp = const_array(node.input[2]) if len(node.input) > 2 else np.array(0, dtype=np.float32)
            dq_info[output] = {
                "input": x,
                "scale": resolve(node.input[1]) if len(node.input) > 1 else "",
                "zero_point": resolve(node.input[2]) if len(node.input) > 2 else "",
                "axis": int(_attr(node, "axis", 0)),
            }
            if x in arrays and scale is not None:
                arrays[output] = _dequantize_array(arrays[x], scale, zp, int(_attr(node, "axis", 0))).astype(np.float32)
                shape_map[output] = list(arrays[output].shape)
            elif x in graph_input_names and scale is not None:
                in_shape = shape_of(x)
                scale_v = float(np.asarray(scale).reshape(-1)[0])
                zp_v = float(np.asarray(zp).reshape(-1)[0]) if zp is not None and np.asarray(zp).size else 0.0
                neg_name = f"{output}__neg_zero_point"
                scale_name = f"{output}__scale"
                add_out = f"{output}__centered"
                arrays[neg_name] = np.asarray([-zp_v], dtype=np.float32)
                arrays[scale_name] = np.asarray([scale_v], dtype=np.float32)
                shape_map[neg_name] = [1]
                shape_map[scale_name] = [1]
                add_node("Add", {"a": short(x), "b": ensure_weight(neg_name)}, add_out, shape=in_shape, prefer_shape=True)
                add_node("Mul", {"a": short(add_out), "b": ensure_weight(scale_name)}, output, shape=in_shape, prefer_shape=True)
            else:
                alias[output] = x
                shape_map[output] = shape_of(x)
            skipped_qdq += 1
            continue
        if op == "QuantizeLinear":
            alias[output] = resolve(node.input[0])
            shape_map[output] = shape_of(node.input[0])
            skipped_qdq += 1
            continue

        if op == "Conv":
            inferred_shape = conv_shape(node)
            attrs = {
                "stride": list(_attr(node, "strides", [1, 1])),
                "dilation": list(_attr(node, "dilations", [1, 1])),
                "groups": int(_attr(node, "group", 1)),
                "data_layout": "NHWC",
                "weight_layout": "OIHW",
            }
            pads = list(_attr(node, "pads", [0, 0, 0, 0]))
            attrs["pads"] = pads
            attrs["padding"] = [pads[0], pads[1]]
            input_q = dq_info.get(node.input[0])
            weight_q = dq_info.get(node.input[1])
            weight_source = weight_q["input"] if weight_q else ""
            weight_arr = arrays.get(weight_source)
            input_scale = const_array(input_q["scale"]) if input_q and input_q.get("scale") else None
            input_zp = const_array(input_q["zero_point"]) if input_q and input_q.get("zero_point") else None
            use_qconv = (
                input_q is not None and weight_q is not None and weight_arr is not None
                and np.asarray(weight_arr).dtype in (np.int8, np.uint8)
                and input_scale is not None and np.asarray(input_scale).size == 1
            )
            if use_qconv:
                attrs["input_scale"] = float(np.asarray(input_scale).reshape(-1)[0])
                attrs["input_zero_point"] = q_scalar(input_zp, 0)
                attrs["weight_axis"] = int(weight_q.get("axis", 0))
                inputs = {
                    "input": short(resolve(node.input[0])),
                    "weight": ensure_weight_preserve(weight_source),
                    "weight_scale": ensure_weight_preserve(weight_q["scale"]),
                }
                if weight_q.get("zero_point"):
                    inputs["weight_zero_point"] = ensure_weight_preserve(weight_q["zero_point"])
                if len(node.input) > 2 and node.input[2]:
                    inputs["bias"] = ensure_weight(node.input[2])
                add_node("QConv2D", inputs, output, attrs, inferred_shape)
            else:
                inputs = {"input": short(resolve(node.input[0])), "weight": ensure_weight(node.input[1])}
                if len(node.input) > 2 and node.input[2]:
                    inputs["bias"] = ensure_weight(node.input[2])
                add_node("Conv2D", inputs, output, attrs, inferred_shape)
        elif op == "Add":
            inferred_shape = broadcast_shape(shape_of(node.input[0]), shape_of(node.input[1]))
            add_node("Add", {"a": short(resolve(node.input[0])), "b": short(resolve(node.input[1]))}, output, shape=inferred_shape)
        elif op == "Clip":
            mn = const_array(node.input[1]) if len(node.input) > 1 and node.input[1] else None
            mx = const_array(node.input[2]) if len(node.input) > 2 and node.input[2] else None
            params = {}
            if mn is not None: params["min"] = float(np.asarray(mn).reshape(-1)[0])
            if mx is not None: params["max"] = float(np.asarray(mx).reshape(-1)[0])
            add_node("Clip", {"input": short(resolve(node.input[0]))}, output, params, shape_of(node.input[0]))
        elif op == "MaxPool":
            pads = list(_attr(node, "pads", [0, 0, 0, 0]))
            add_node("MaxPool2D", {"input": short(resolve(node.input[0]))}, output, {
                "kernel": list(_attr(node, "kernel_shape", [1, 1])),
                "stride": list(_attr(node, "strides", [1, 1])),
                "pads": pads,
                "padding": [pads[0], pads[1]],
                "data_layout": "NHWC",
            }, pool_shape(node))
        elif op == "Resize":
            add_node("ResizeNearest2D", {"input": short(resolve(node.input[0]))}, output, {
                "mode": "nearest",
                "coordinate_transformation_mode": (_attr(node, "coordinate_transformation_mode", b"asymmetric") or b"").decode("utf-8", "ignore"),
                "data_layout": "NHWC",
            }, resize_shape(node), prefer_shape=True)
        elif op == "Concat":
            add_node("Concat", {f"input{i}": short(resolve(inp)) for i, inp in enumerate(node.input)}, output,
                     {"axis": int(_attr(node, "axis", 1)), "count": len(node.input)}, concat_shape(node))
        elif op == "Reshape":
            add_node("Reshape", {"input": short(resolve(node.input[0]))}, output, shape=reshape_shape(node), prefer_shape=True)
        elif op == "Sigmoid":
            add_node("Sigmoid", {"input": short(resolve(node.input[0]))}, output, shape=shape_of(node.input[0]))
        elif op == "Transpose":
            add_node("Transpose", {"input": short(resolve(node.input[0]))}, output,
                     {"perm": list(_attr(node, "perm", []))}, transpose_shape(node))
        else:
            raise RuntimeError(f"Unsupported ONNX op for Volvox export: {op}")

    for out in model.graph.output:
        resolved = resolve(out.name)
        if short(resolved) != short(out.name):
            add_node("Identity", {"input": short(resolved)}, out.name, shape=shape_of(resolved), prefer_shape=True)

    raw_node_count = len(nodes)
    nodes, optimized_removed = optimize_export_nodes(nodes)

    config = {
        "format": "volvoxai-onnx-v1",
        "source": {
            "onnx": Path(model_path).name,
            "skipped_qdq_nodes": skipped_qdq,
            "float_storage": float_storage,
            "raw_nodes": raw_node_count,
            "optimized_nodes_removed": optimized_removed,
        },
        "inputs": inputs_def,
        "outputs": {out.name: short(out.name) for out in model.graph.output},
        "nodes": nodes,
    }

    save_file(tensors, str(out_path))
    config_path = out_dir / "config.json"
    config_path.write_text(dumps_with_compact_lists(config, indent=2), encoding="utf-8")
    print(f"[Export] ONNX lowered nodes={raw_node_count} optimized_nodes={len(nodes)} weights={len(tensors)} skipped_qdq={skipped_qdq}")
    print(f"[Export] Wrote {out_path} and {config_path}")


def build_kie_graph(cfg):
    nodes = []
    d_model = cfg.get("d_model", 320)
    for i in range(4):
        in_name = "images" if i == 0 else f"stem_{i-1}_out"
        nodes.append({"op": "Conv2D", "inputs": {"input": in_name, "weight": f"stem.{i}.net.0.weight"}, "outputs": {"out": f"stem_{i}_out"}, "outputs_shape": {"out": [1, 32, 112, 112]}, "params": {"stride": 2 if i < 2 else 1, "padding": 1}})
        
    last_out = "stem_3_out"
    for i in range(cfg.get("enc_layers", 6)):
        prefix = f"encoder.layers.{i}"
        nodes.append({"op": "LayerNorm", "inputs": {"input": last_out, "weight": f"{prefix}.norm1.weight", "bias": f"{prefix}.norm1.bias"}, "outputs": {"out": f"enc_{i}_n1"}, "outputs_shape": {"out": [1, 196, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"enc_{i}_n1", "weight": f"{prefix}.self_attn.in_proj_weight", "scale": f"{prefix}.self_attn.in_proj_weight_scale", "bias": f"{prefix}.self_attn.in_proj_bias"}, "outputs": {"out": f"enc_{i}_qkv"}, "outputs_shape": {"out": [1, 196, d_model*3]}})
        nodes.append({"op": "SDPA", "inputs": {"qkv": f"enc_{i}_qkv"}, "outputs": {"out": f"enc_{i}_attn"}, "outputs_shape": {"out": [1, 196, d_model]}, "params": {"heads": cfg.get("heads", 8)}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"enc_{i}_attn", "weight": f"{prefix}.self_attn.out_proj.weight", "scale": f"{prefix}.self_attn.out_proj.weight_scale", "bias": f"{prefix}.self_attn.out_proj.bias"}, "outputs": {"out": f"enc_{i}_attn_proj"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": last_out, "b": f"enc_{i}_attn_proj"}, "outputs": {"out": f"enc_{i}_add1"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "LayerNorm", "inputs": {"input": f"enc_{i}_add1", "weight": f"{prefix}.norm2.weight", "bias": f"{prefix}.norm2.bias"}, "outputs": {"out": f"enc_{i}_n2"}, "outputs_shape": {"out": [1, 196, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"enc_{i}_n2", "weight": f"{prefix}.linear1.weight", "scale": f"{prefix}.linear1.weight_scale", "bias": f"{prefix}.linear1.bias"}, "outputs": {"out": f"enc_{i}_ff1"}, "outputs_shape": {"out": [1, 196, d_model*4]}})
        nodes.append({"op": "GELU", "inputs": {"input": f"enc_{i}_ff1"}, "outputs": {"out": f"enc_{i}_gelu"}, "outputs_shape": {"out": [1, 196, d_model*4]}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"enc_{i}_gelu", "weight": f"{prefix}.linear2.weight", "scale": f"{prefix}.linear2.weight_scale", "bias": f"{prefix}.linear2.bias"}, "outputs": {"out": f"enc_{i}_ff2"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": f"enc_{i}_add1", "b": f"enc_{i}_ff2"}, "outputs": {"out": f"enc_{i}_out"}, "outputs_shape": {"out": [1, 196, d_model]}})
        last_out = f"enc_{i}_out"
        
    nodes.append({"op": "Embedding", "inputs": {"input": "q_tokens", "weight": "q_pos"}, "outputs": {"out": "dec_in"}, "outputs_shape": {"out": [1, 192, d_model]}})
    dec_out = "dec_in"
    for i in range(cfg.get("dec_layers", 4)):
        prefix = f"decoder.layers.{i}"
        nodes.append({"op": "LayerNorm", "inputs": {"input": dec_out, "weight": f"{prefix}.norm1.weight", "bias": f"{prefix}.norm1.bias"}, "outputs": {"out": f"dec_{i}_n1"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{i}_n1", "weight": f"{prefix}.self_attn.in_proj_weight", "scale": f"{prefix}.self_attn.in_proj_weight_scale", "bias": f"{prefix}.self_attn.in_proj_bias"}, "outputs": {"out": f"dec_{i}_qkv"}, "outputs_shape": {"out": [1, 192, d_model*3]}})
        nodes.append({"op": "SDPA", "inputs": {"qkv": f"dec_{i}_qkv"}, "outputs": {"out": f"dec_{i}_attn"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"heads": cfg.get("heads", 8)}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{i}_attn", "weight": f"{prefix}.self_attn.out_proj.weight", "scale": f"{prefix}.self_attn.out_proj.weight_scale", "bias": f"{prefix}.self_attn.out_proj.bias"}, "outputs": {"out": f"dec_{i}_attn_proj"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": dec_out, "b": f"dec_{i}_attn_proj"}, "outputs": {"out": f"dec_{i}_add1"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "LayerNorm", "inputs": {"input": f"dec_{i}_add1", "weight": f"{prefix}.norm2.weight", "bias": f"{prefix}.norm2.bias"}, "outputs": {"out": f"dec_{i}_n2"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{i}_n2", "weight": f"{prefix}.multihead_attn.q_proj_weight", "scale": f"{prefix}.multihead_attn.q_proj_scale", "bias": f"{prefix}.multihead_attn.q_proj_bias"}, "outputs": {"out": f"dec_{i}_q"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "MatMul", "inputs": {"input": last_out, "weight": f"{prefix}.multihead_attn.k_proj_weight", "scale": f"{prefix}.multihead_attn.k_proj_scale", "bias": f"{prefix}.multihead_attn.k_proj_bias"}, "outputs": {"out": f"dec_{i}_k"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "MatMul", "inputs": {"input": last_out, "weight": f"{prefix}.multihead_attn.v_proj_weight", "scale": f"{prefix}.multihead_attn.v_proj_scale", "bias": f"{prefix}.multihead_attn.v_proj_bias"}, "outputs": {"out": f"dec_{i}_v"}, "outputs_shape": {"out": [1, 196, d_model]}})
        nodes.append({"op": "CrossSDPA", "inputs": {"q": f"dec_{i}_q", "k": f"dec_{i}_k", "v": f"dec_{i}_v"}, "outputs": {"out": f"dec_{i}_cross"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"heads": cfg.get("heads", 8)}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{i}_cross", "weight": f"{prefix}.multihead_attn.out_proj.weight", "scale": f"{prefix}.multihead_attn.out_proj.weight_scale", "bias": f"{prefix}.multihead_attn.out_proj.bias"}, "outputs": {"out": f"dec_{i}_cross_proj"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": f"dec_{i}_add1", "b": f"dec_{i}_cross_proj"}, "outputs": {"out": f"dec_{i}_add2"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "LayerNorm", "inputs": {"input": f"dec_{i}_add2", "weight": f"{prefix}.norm3.weight", "bias": f"{prefix}.norm3.bias"}, "outputs": {"out": f"dec_{i}_n3"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"d_model": d_model}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{i}_n3", "weight": f"{prefix}.linear1.weight", "scale": f"{prefix}.linear1.weight_scale", "bias": f"{prefix}.linear1.bias"}, "outputs": {"out": f"dec_{i}_ff1"}, "outputs_shape": {"out": [1, 192, d_model*4]}})
        nodes.append({"op": "GELU", "inputs": {"input": f"dec_{i}_ff1"}, "outputs": {"out": f"dec_{i}_gelu"}, "outputs_shape": {"out": [1, 192, d_model*4]}})
        nodes.append({"op": "MatMul", "inputs": {"input": f"dec_{i}_gelu", "weight": f"{prefix}.linear2.weight", "scale": f"{prefix}.linear2.weight_scale", "bias": f"{prefix}.linear2.bias"}, "outputs": {"out": f"dec_{i}_ff2"}, "outputs_shape": {"out": [1, 192, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": f"dec_{i}_add2", "b": f"dec_{i}_ff2"}, "outputs": {"out": f"dec_{i}_out"}, "outputs_shape": {"out": [1, 192, d_model]}})
        dec_out = f"dec_{i}_out"

    nodes.append({"op": "LayerNorm", "inputs": {"input": dec_out, "weight": "encoder.layers.0.norm1.weight", "bias": "encoder.layers.0.norm1.bias"}, "outputs": {"out": "final_norm"}, "outputs_shape": {"out": [1, 192, d_model]}, "params": {"d_model": d_model}})
    nodes.append({"op": "MatMul", "inputs": {"input": "final_norm", "weight": "encoder.layers.0.self_attn.out_proj.weight", "scale": "encoder.layers.0.self_attn.out_proj.weight_scale", "bias": "out_bias"}, "outputs": {"out": "logits"}, "outputs_shape": {"out": [1, 192, cfg.get("vocab_size", 530)]}})
    return nodes

def build_gptneo_graph(cfg, sd, out_tensors):
    nodes = []
    d_model = cfg.hidden_size
    n_layers = cfg.num_layers
    
    for k, v in sd.items():
        k_new = k.replace("transformer.", "")
        out_tensors[k_new] = v.float().contiguous()
        
    nodes.append({"op": "Embedding", "inputs": {"input": "tokens", "weight": "wte.weight"}, "outputs": {"out": "emb_tok"}, "outputs_shape": {"out": [1, 256, d_model]}})
    nodes.append({"op": "Embedding", "inputs": {"input": "positions", "weight": "wpe.weight"}, "outputs": {"out": "emb_pos"}, "outputs_shape": {"out": [1, 256, d_model]}})
    nodes.append({"op": "Add", "inputs": {"a": "emb_tok", "b": "emb_pos"}, "outputs": {"out": "hidden_0"}, "outputs_shape": {"out": [1, 256, d_model]}})
    
    last_hidden = "hidden_0"
    for i in range(n_layers):
        prefix = f"h.{i}"
        nodes.append({"op": "LayerNorm", "inputs": {"input": last_hidden, "weight": f"{prefix}.ln_1.weight", "bias": f"{prefix}.ln_1.bias"}, "outputs": {"out": f"ln1_{i}"}, "outputs_shape": {"out": [1, 256, d_model]}, "params": {"eps": cfg.layer_norm_epsilon, "d_model": d_model}})
        
        q_w = out_tensors.pop(f"{prefix}.attn.attention.q_proj.weight").t()
        k_w = out_tensors.pop(f"{prefix}.attn.attention.k_proj.weight").t()
        v_w = out_tensors.pop(f"{prefix}.attn.attention.v_proj.weight").t()
        out_tensors[f"{prefix}.attn.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=1).contiguous()
        
        nodes.append({"op": "MatMul", "inputs": {"input": f"ln1_{i}", "weight": f"{prefix}.attn.qkv_proj.weight"}, "outputs": {"out": f"qkv_{i}"}, "outputs_shape": {"out": [1, 256, d_model*3]}})
        nodes.append({"op": "SDPA", "inputs": {"qkv": f"qkv_{i}"}, "outputs": {"out": f"attn_{i}"}, "outputs_shape": {"out": [1, 256, d_model]}, "params": {"heads": cfg.num_heads, "scale": 1.0}})
        
        out_tensors[f"{prefix}.attn.out_proj.weight"] = out_tensors.pop(f"{prefix}.attn.attention.out_proj.weight").t().contiguous()
        out_tensors[f"{prefix}.attn.out_proj.bias"] = out_tensors.pop(f"{prefix}.attn.attention.out_proj.bias").contiguous()
        
        nodes.append({"op": "MatMul", "inputs": {"input": f"attn_{i}", "weight": f"{prefix}.attn.out_proj.weight", "bias": f"{prefix}.attn.out_proj.bias"}, "outputs": {"out": f"attn_proj_{i}"}, "outputs_shape": {"out": [1, 256, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": last_hidden, "b": f"attn_proj_{i}"}, "outputs": {"out": f"add1_{i}"}, "outputs_shape": {"out": [1, 256, d_model]}})
        nodes.append({"op": "LayerNorm", "inputs": {"input": f"add1_{i}", "weight": f"{prefix}.ln_2.weight", "bias": f"{prefix}.ln_2.bias"}, "outputs": {"out": f"ln2_{i}"}, "outputs_shape": {"out": [1, 256, d_model]}, "params": {"eps": cfg.layer_norm_epsilon, "d_model": d_model}})
        
        out_tensors[f"{prefix}.mlp.c_fc.weight"] = out_tensors.pop(f"{prefix}.mlp.c_fc.weight").t().contiguous()
        out_tensors[f"{prefix}.mlp.c_fc.bias"] = out_tensors.pop(f"{prefix}.mlp.c_fc.bias").contiguous()
        
        nodes.append({"op": "MatMul", "inputs": {"input": f"ln2_{i}", "weight": f"{prefix}.mlp.c_fc.weight", "bias": f"{prefix}.mlp.c_fc.bias"}, "outputs": {"out": f"mlp1_{i}"}, "outputs_shape": {"out": [1, 256, cfg.hidden_size * 4]}})
        nodes.append({"op": "GELU", "inputs": {"input": f"mlp1_{i}"}, "outputs": {"out": f"mlp_act_{i}"}, "outputs_shape": {"out": [1, 256, cfg.hidden_size * 4]}})
        
        out_tensors[f"{prefix}.mlp.c_proj.weight"] = out_tensors.pop(f"{prefix}.mlp.c_proj.weight").t().contiguous()
        out_tensors[f"{prefix}.mlp.c_proj.bias"] = out_tensors.pop(f"{prefix}.mlp.c_proj.bias").contiguous()
        
        nodes.append({"op": "MatMul", "inputs": {"input": f"mlp_act_{i}", "weight": f"{prefix}.mlp.c_proj.weight", "bias": f"{prefix}.mlp.c_proj.bias"}, "outputs": {"out": f"mlp2_{i}"}, "outputs_shape": {"out": [1, 256, d_model]}})
        nodes.append({"op": "Add", "inputs": {"a": f"add1_{i}", "b": f"mlp2_{i}"}, "outputs": {"out": f"hidden_{i+1}"}, "outputs_shape": {"out": [1, 256, d_model]}})
        last_hidden = f"hidden_{i+1}"

    nodes.append({"op": "LayerNorm", "inputs": {"input": last_hidden, "weight": "ln_f.weight", "bias": "ln_f.bias"}, "outputs": {"out": "final_norm"}, "outputs_shape": {"out": [1, 256, d_model]}, "params": {"eps": cfg.layer_norm_epsilon, "d_model": d_model}})
    if "lm_head.weight" in out_tensors:
        out_tensors["lm_head.weight"] = out_tensors.pop("lm_head.weight").t()
    else:
        out_tensors["lm_head.weight"] = out_tensors["wte.weight"].t()
        
    nodes.append({"op": "MatMul", "inputs": {"input": "final_norm", "weight": "lm_head.weight"}, "outputs": {"out": "logits"}, "outputs_shape": {"out": [1, 256, cfg.vocab_size]}})
    return nodes


def export_tflite_model(model_path: str, out_path: str, weight_dtype: str = "auto"):
    import math
    import struct
    from collections import Counter

    import numpy as np
    from flatbuffers import number_types as N
    from flatbuffers.table import Table

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
    value_name = {}
    value_shape = {}
    value_layout = {}
    for idx, tid in enumerate(input_tids):
        name = short(tkey(tid), f"input{idx}")
        shape = tensors_meta[tid]["shape"]
        dtype = dtype_names.get(tensors_meta[tid]["dtype"], "float32")
        inputs_def[name] = {"shape": shape, "dtype": dtype, "source_name": tensors_meta[tid]["name"]}
        value_name[tid] = name
        value_shape[tid] = shape
        value_layout[tid] = "NHWC" if len(shape) == 4 else "other"

    for idx, tid in enumerate(output_tids):
        shape = tensors_meta[tid]["shape"]
        preferred = "boxes" if shape and shape[-1] == 4 else ("scores" if shape and shape[-1] > 4 else f"output{idx}")
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
        tensors[name] = torch.from_numpy(arr.copy()).contiguous()

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

    def ensure_weight(tid, transform="raw"):
        source_tid = resolve_const(tid)
        key = (source_tid, transform)
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
        elif transform == "depthwise":
            if arr.ndim != 4 or arr.shape[0] != 1:
                raise RuntimeError(f"Depthwise weight tensor {tid} has unsupported shape {list(arr.shape)}")
            # TFLite depthwise filters are 1HWO, where O = input_channels * multiplier.
            # Keep that artifact layout; native prepares HWCM for compute.
        elif transform != "raw":
            raise RuntimeError(f"Unknown weight transform: {transform}")
        name = short(f"{tkey(source_tid)}_{transform}", weight=True)
        save_array(name, arr)
        weight_names[key] = name
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

    def ensure_dequant_bias(tid):
        source_tid = resolve_const(tid)
        key = (source_tid, "dequant_bias")
        if key in weight_names:
            return weight_names[key]
        arr = tensor_array(source_tid)
        if arr is None:
            raise RuntimeError(f"TFLite bias tensor {tid} is not constant")
        q = qparams(source_tid)
        scales = np.asarray(q["scale"] or [1.0], dtype=np.float32)
        zps = np.asarray(q["zero_point"] or [0], dtype=np.float32)
        if scales.size == 1:
            out = (arr.astype(np.float32) - float(zps.reshape(-1)[0])) * float(scales.reshape(-1)[0])
        else:
            out = (arr.astype(np.float32) - zps.reshape(arr.shape)) * scales.reshape(arr.shape)
        name = short(f"{tkey(source_tid)}_dequant_bias", weight=True)
        save_array(name, out.astype(np.float32))
        weight_names[key] = name
        return name

    def add_node(op, inputs, out_name, out_shape, params=None):
        node = {
            "op": op,
            "inputs": inputs,
            "outputs": {"out": out_name},
            "outputs_shape": {"out": [int(v) for v in out_shape]},
        }
        if params:
            node["params"] = params
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
            q = qparams(inputs[0])
            scale = ensure_qparam_weight(inputs[0], "scale")
            zero_point = ensure_qparam_weight(inputs[0], "zero_point")
            add_node("DequantizeLinear", {
                "input": in_name,
                "scale": scale,
                "zero_point": zero_point,
            }, out_name, out_shape, {
                "scale": float(q_scalar(q, "scale", 1.0)),
                "zero_point": int(q_scalar(q, "zero_point", 0)),
                "data_layout": "NHWC" if len(out_shape) == 4 else "other",
            })
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
            in_q = qparams(inputs[0])
            out_q = qparams(outputs[0])
            params = {
                "input_scale": float(q_scalar(in_q, "scale", 0.0)),
                "input_zero_point": int(q_scalar(in_q, "zero_point", 0)),
                "output_scale": float(q_scalar(out_q, "scale", 1.0)),
                "output_zero_point": int(q_scalar(out_q, "zero_point", 0)),
                "data_layout": "NHWC" if len(out_tflite_shape) == 4 else "other",
            }
            add_node("QuantizeLinear", {"input": in_name}, out_name, out_tflite_shape, params)
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
            in_q = qparams(inputs[0])
            out_q = qparams(out_tid)
            weight_q = qparams(inputs[1])
            weight_meta = tensors_meta[resolve_const(inputs[1])]
            use_qconv = (
                weight_meta["dtype"] in (3, 9)
                and bool(in_q["scale"])
                and bool(weight_q["scale"])
            )
            if op_name == "CONV_2D":
                weight = ensure_weight(inputs[1], "conv")
                weight_shape = list(tensors[weight].shape)
                kernel = weight_shape[1:3]
                groups = 1
                weight_layout = "OHWI"
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
                weight = ensure_weight(inputs[1], "depthwise")
                weight_shape = list(tensors[weight].shape)
                kernel = weight_shape[1:3]
                groups = int(in_shape[3])
                weight_layout = "1HWO"
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
            if use_qconv:
                node_inputs["weight_scale"] = ensure_qparam_weight(inputs[1], "scale")
                if any(v != 0 for v in weight_q["zero_point"]):
                    node_inputs["weight_zero_point"] = ensure_qparam_weight(inputs[1], "zero_point")
            if len(inputs) > 2 and inputs[2] >= 0:
                node_inputs["bias"] = ensure_dequant_bias(inputs[2]) if use_qconv else ensure_weight(inputs[2], "raw")
            params = {
                "stride": stride,
                "dilation": dilation,
                "groups": groups,
                "pads": pads,
                "padding": [pads[0], pads[1]],
                "data_layout": "NHWC",
                "weight_layout": weight_layout,
            }
            if use_qconv:
                params["input_scale"] = float(q_scalar(in_q, "scale", 1.0))
                params["input_zero_point"] = int(q_scalar(in_q, "zero_point", 0))
                params["output_scale"] = float(q_scalar(out_q, "scale", 0.0))
                params["output_zero_point"] = int(q_scalar(out_q, "zero_point", 0))
                params["weight_axis"] = int(weight_q.get("axis", 0))
            relu = activation_param(act)
            if relu:
                params["relu"] = relu
            add_node("QConv2D" if use_qconv else "Conv2D", node_inputs, out_name, out_shape, params)
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
            add_node("Add", {"a": a_name, "b": b_name}, out_name, out_shape, params)
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
                "padding": [pads[0], pads[1]],
                "data_layout": "NHWC",
            })
            set_value(out_tid, out_name, out_shape, "NHWC")
            continue

        if op_name == "RESIZE_NEAREST_NEIGHBOR":
            in_name = name_for_value(inputs[0])
            out_shape = out_tflite_shape
            add_node("ResizeNearest2D", {"input": in_name}, out_name, out_shape, {
                "mode": "nearest",
                "coordinate_transformation_mode": "asymmetric",
                "data_layout": "NHWC",
            })
            set_value(out_tid, out_name, out_shape, "NHWC")
            continue

        if op_name == "RESHAPE":
            in_tid = inputs[0]
            in_name = name_for_value(in_tid)
            add_node("Reshape", {"input": in_name}, out_name, out_tflite_shape)
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
            add_node("Concat", node_inputs, out_name, out_shape, {"axis": axis, "count": len(inputs)})
            set_value(out_tid, out_name, out_shape, layout)
            continue

        if op_name == "LOGISTIC":
            in_name = name_for_value(inputs[0])
            in_shape = value_shape.get(inputs[0], tensors_meta[inputs[0]]["shape"])
            add_node("Sigmoid", {"input": in_name}, out_name, in_shape)
            set_value(out_tid, out_name, in_shape, value_layout.get(inputs[0], "other"))
            continue

        raise RuntimeError(f"Unsupported TFLite op for direct Volvox export: {op_name} ({builtin})")

    for tid in output_tids:
        wanted = short(tkey(tid))
        current = name_for_value(tid)
        if current != wanted:
            add_node("Identity", {"input": current}, wanted, value_shape.get(tid, tensors_meta[tid]["shape"]))
            set_value(tid, wanted, value_shape.get(tid, tensors_meta[tid]["shape"]), value_layout.get(tid, "other"))

    raw_node_count = len(nodes)
    nodes, optimized_removed = optimize_export_nodes(nodes)

    config = {
        "format": "volvoxai-tflite-v1",
        "source": {
            "tflite": Path(model_path).name,
            "float_storage": float_storage,
            "raw_tflite_ops": len(op_tables),
            "folded_dequantize_nodes": folded_dequantize,
            "lowered_quantize_nodes": lowered_quantize,
            "raw_nodes": raw_node_count,
            "optimized_nodes_removed": optimized_removed,
            "op_histogram": dict(op_counter),
            "internal_layout": "NHWC",
            "conv_weight_layout": "OHWI",
            "depthwise_weight_layout": "1HWO",
        },
        "inputs": inputs_def,
        "outputs": {tensors_meta[tid]["name"]: short(tkey(tid)) for tid in output_tids},
        "nodes": nodes,
    }

    save_file(tensors, str(out_path))
    config_path = out_dir / "config.json"
    config_path.write_text(dumps_with_compact_lists(config, indent=2), encoding="utf-8")
    print(f"[Export] TFLite ops={len(op_tables)} folded_dequantize={folded_dequantize} lowered_nodes={raw_node_count} optimized_nodes={len(nodes)} weights={len(tensors)}")
    print(f"[Export] Wrote {out_path} and {config_path}")


def export_model(model_id_or_path: str, out_path: str, weight_dtype: str = "auto"):
    print(f"[Export] Detecting architecture for {model_id_or_path}...")
    if model_id_or_path.lower().endswith(".onnx"):
        export_onnx_model(model_id_or_path, out_path, weight_dtype=weight_dtype)
        return
    if model_id_or_path.lower().endswith(".tflite"):
        export_tflite_model(model_id_or_path, out_path, weight_dtype=weight_dtype)
        return
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tensors = {}
    
    # Check if it's a local KIE checkpoint or a HuggingFace CausalLM model
    if os.path.exists(model_id_or_path) and not os.path.isdir(model_id_or_path):
        # Local KIE Checkpoint
        print(f"[Export] Discovered local PyTorch Checkpoint. Using Vision-Encoder-Decoder Graph Builder.")
        ckpt = torch.load(model_id_or_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        cfg = ckpt.get("config", {})
        
        for key, tensor in state_dict.items():
            if "multihead_attn.in_proj_weight" in key:
                q_w, k_w, v_w = tensor.chunk(3, dim=0)
                prefix = key.replace(".in_proj_weight", "")
                for name, w in zip(["q_proj_weight", "k_proj_weight", "v_proj_weight"], [q_w, k_w, v_w]):
                    q_tensor, scale = quantize_to_int8(w)
                    out_tensors[f"{prefix}.{name}"] = q_tensor
                    out_tensors[f"{prefix}.{name.replace('weight', 'scale')}"] = scale.view(1)
            elif "multihead_attn.in_proj_bias" in key:
                q_b, k_b, v_b = tensor.chunk(3, dim=0)
                prefix = key.replace(".in_proj_bias", "")
                out_tensors[f"{prefix}.q_proj_bias"] = q_b.float()
                out_tensors[f"{prefix}.k_proj_bias"] = k_b.float()
                out_tensors[f"{prefix}.v_proj_bias"] = v_b.float()
            elif tensor.dim() == 2 and ("weight" in key and "norm" not in key):
                q_tensor, scale = quantize_to_int8(tensor)
                out_tensors[key] = q_tensor
                out_tensors[f"{key}_scale"] = scale.view(1)
            else:
                out_tensors[key] = tensor.float()
                
        inputs_def = {"images": {"shape": [1, 3, 224, 224], "dtype": "float32"}, "q_tokens": {"shape": [1, 192], "dtype": "int32"}}
        nodes = build_kie_graph(cfg)
        
    else:
        # HuggingFace Transformer Model
        print(f"[Export] Fetching HuggingFace Config. Using GPT-Neo Graph Builder.")
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
        cfg = model.config
        sd = model.state_dict()
        
        inputs_def = {"tokens": {"shape": [1, 256], "dtype": "int32"}, "positions": {"shape": [1, 256], "dtype": "int32"}}
        nodes = build_gptneo_graph(cfg, sd, out_tensors)
        
        import numpy as np
        tok_in = np.ones((1, 256), dtype=np.int32) * 50256
        tok_in[0, 0:5] = [123, 456, 789, 1011, 1213]
        pos_in = np.arange(256, dtype=np.int32).reshape(1, 256)
        with open(out_dir / "tokens.i32", "wb") as f: f.write(tok_in.tobytes())
        with open(out_dir / "positions.i32", "wb") as f: f.write(pos_in.tobytes())

    print(f"[Export] Saving standard Safetensors to {out_path}...")
    final_tensors = {k: v.clone().contiguous() for k, v in out_tensors.items()}
    save_file(final_tensors, out_path)
    
    config_path = out_dir / "config.json"
    print(f"[Export] Saving graph topology to {config_path}...")
    with open(config_path, "w") as f:
        f.write(dumps_with_compact_lists({"inputs": inputs_def, "nodes": nodes}, indent=2))
        
    print(f"[Export] Success! Safetensors size: {Path(out_path).stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--out", required=True, help="Output safetensors path (e.g. models/model.safetensors)")
    parser.add_argument(
        "--weight-dtype",
        choices=["auto", "float32", "float16"],
        default="auto",
        help="Storage dtype for floating ONNX tensors. auto preserves FLOAT16 initializers and treats fp16/float16 filenames as float16.",
    )
    args = parser.parse_args()
    export_model(args.model, args.out, weight_dtype=args.weight_dtype)
