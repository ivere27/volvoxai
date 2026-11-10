"""Complete Kokoro lowering with one host shape boundary after duration.

All learned and sample-valued operations stay in the emitted graphs. The
caller reads the final cumulative duration solely to choose the second graph's
extent. It never computes an alignment or supplies cached model activations.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper

from examples.kokoro.prepare import extract, specialized_head
from examples.kokoro.signal_geometry import bluestein_stft_nodes

TEXT = "/encoder/text_encoder/Transpose_2_output_0"
PREDICTOR = "/encoder/Transpose_output_0"
DURATIONS = "/encoder/CumSum_output_0"
FRAME_TOTAL = "/encoder/Cast_1_output_0"
DEFAULT_MAX_FRAMES = 2048


def info(name: str, shape, dtype=onnx.TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def boundary_types(tokens: int):
    if tokens < 3 or tokens > 512:
        raise ValueError("Kokoro expects 3..512 padded token IDs")
    return {TEXT: info(TEXT, [1, 512, tokens]),
        PREDICTOR: info(PREDICTOR, [1, 640, tokens]),
        DURATIONS: info(DURATIONS, [tokens], onnx.TensorProto.INT64)}


def encoder(model, tokens: int, max_frames: int = DEFAULT_MAX_FRAMES):
    if not isinstance(max_frames, int) or not tokens <= max_frames <= (2**31 - 20) // 600 or tokens * (max_frames + 1) > 2**31 - 1:
        raise ValueError("Duration bound must fit an exact positive I32 cumulative sum")
    result = extract(model, {
        "input_ids": info("input_ids", [1, tokens], onnx.TensorProto.INT64),
        "style": info("style", [1, 256]), "speed": info("speed", [1]),
    }, boundary_types(tokens))
    # The source emits I64 durations, while Volvox stores execution integers
    # as I32. Saturating each positive duration at max_frames+1 before the
    # cast cannot alter any admitted request; every saturated request still
    # exceeds the host's maximum total. The sum itself provably fits I32.
    nodes, found = [], False
    for node in result.graph.node:
        if node.name == "/encoder/Cast":
            target = {value.name: helper.get_attribute_value(value) for value in node.attribute}.get("to")
            if target != onnx.TensorProto.INT64:
                raise ValueError("Unexpected Kokoro duration cast")
            nodes.append(helper.make_node("Clip", [node.input[0], "kokoro_duration_min", "kokoro_duration_max"], ["kokoro_bounded_durations"]))
            node.input[0] = "kokoro_bounded_durations"
            for value in node.attribute:
                if value.name == "to":
                    value.i = onnx.TensorProto.INT32
            found = True
        elif node.name == "/encoder/Gather":
            # The duration tensor has exactly one batch. Selecting batch zero
            # is a shape-only view, including after the bounded I32 cast.
            node = helper.make_node("Reshape", [node.input[0], "kokoro_duration_shape"], list(node.output), name=node.name)
        nodes.append(node)
    if not found:
        raise ValueError("Missing source duration cast")
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    result.graph.initializer.extend([
        numpy_helper.from_array(np.array(0, dtype=np.float32), "kokoro_duration_min"),
        numpy_helper.from_array(np.array(max_frames + 1, dtype=np.float32), "kokoro_duration_max"),
        numpy_helper.from_array(np.array([tokens], dtype=np.int64), "kokoro_duration_shape"),
    ])
    result.graph.output[-1].type.tensor_type.elem_type = onnx.TensorProto.INT32
    return result


def decoder(model, tokens: int, frames: int):
    if not isinstance(frames, int) or frames < tokens or frames > (2**31 - 20) // 600:
        raise ValueError("Invalid predicted duration extent")
    inputs = {**boundary_types(tokens), "style": info("style", [1, 256])}
    source = extract(model, inputs, {"waveform": info("waveform", [1, frames * 600])})
    # Only Range's extent becomes a shape constant. Its comparisons continue
    # to consume the real cumulative durations bound as an execution input.
    for node in source.graph.node:
        if node.name == "/encoder/Range":
            if node.input[1] != FRAME_TOTAL:
                raise ValueError("Unexpected Kokoro duration graph")
            node.input[1] = "kokoro_duration_extent"
    source.graph.initializer.append(numpy_helper.from_array(np.array(frames, dtype=np.int64), "kokoro_duration_extent"))
    return extract(source, inputs, {"waveform": info("waveform", [1, frames * 600])})


def fold_static_programs(model):
    """Reuse the frontend's constant/shape proof; never evaluate live tensors."""
    from tools.exporter.frontend_onnx import OnnxCompiler
    with tempfile.TemporaryDirectory(prefix="kokoro-shape-") as temporary:
        path = Path(temporary) / "source.onnx"
        onnx.save(model, path)
        compiler = OnnxCompiler(str(path))
    values = {value.name: copy.deepcopy(value) for value in compiler.model.graph.initializer}
    nodes = []
    for index, node in enumerate(compiler.model.graph.node):
        if index in compiler.folded:
            for name in node.output:
                values[name] = numpy_helper.from_array(compiler.arrays[name], name)
        else:
            nodes.append(copy.deepcopy(node))
    result = copy.deepcopy(compiler.model)
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    del result.graph.initializer[:]
    result.graph.initializer.extend(values.values())
    # Old symbolic intermediate metadata must not override newly proved shapes.
    del result.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(result, strict_mode=True, data_prop=True)


def lower_geometry(model, *, operators=None):
    """Lower the pinned model's STFT, resize and reflection geometry.

    STFT is a windowed Bluestein FFT graph. Resize indices and weights
    depend only on tensor extents/scales; samples remain runtime Gather/Mul/Add.
    See ONNX STFT-17, Resize-19 and Pad-19 for source contracts.
    """
    model = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    shapes = {value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    rewritten, evidence = [], []
    def constant(name, value):
        array = np.asarray(value)
        model.graph.initializer.append(numpy_helper.from_array(array, name))
        constants[name] = array
        return name
    for index, node in enumerate(model.graph.node):
        if operators is not None and node.op_type not in operators:
            rewritten.append(copy.deepcopy(node))
            continue
        attrs = {value.name: helper.get_attribute_value(value) for value in node.attribute}
        stem = f"kokoro_geometry_{index}"
        shape = shapes.get(node.input[0], []) if node.input else []
        if node.op_type == "ConvTranspose" and attrs.get("group", 1) > 1:
            weights = constants[node.input[1]]
            stride = attrs.get("strides", [1])[0]
            dilation = attrs.get("dilations", [1])[0]
            padding = attrs.get("pads", [0, 0])
            extra = attrs.get("output_padding", [0])[0]
            groups = attrs["group"]
            if len(shape) != 3 or weights.ndim != 3 or weights.shape[:2] != (groups, 1) or shape[1] != groups or attrs.get("auto_pad", b"NOTSET") != b"NOTSET" or "output_shape" in attrs:
                raise ValueError(f"Unsupported grouped transposed convolution: {node.name}")
            extent = stride * (shape[2] - 1) + 1 + extra
            positions = np.arange(extent, dtype=np.int64)
            indices = np.minimum(positions // stride, shape[2] - 1)
            mask = ((positions % stride == 0) & (positions // stride < shape[2])).astype(np.float32).reshape(1, 1, extent)
            conv_pads = [dilation * (weights.shape[2] - 1) - value for value in padding]
            if min(conv_pads) < 0:
                raise ValueError("Grouped transposed convolution requires nonnegative transformed padding")
            gathered, inserted = stem + "_gathered", stem + "_inserted"
            weight_name = constant(stem + "_reversed_weight", weights[:, :, ::-1].copy())
            rewritten.extend([
                helper.make_node("Gather", [node.input[0], constant(stem + "_indices", indices)], [gathered], axis=2),
                helper.make_node("Mul", [gathered, constant(stem + "_mask", mask)], [inserted]),
                helper.make_node("Conv", [inserted, weight_name, *node.input[2:]], list(node.output), group=groups,
                    dilations=[dilation], strides=[1], pads=conv_pads, kernel_shape=[weights.shape[2]]),
            ])
            evidence.append({"source": node.name, "kind": "depthwise_transpose_zero_insert", "groups": groups, "stride": stride})
        elif node.op_type == "STFT":
            hop = int(constants[node.input[1]].item())
            window = constants[node.input[2]].reshape(-1).astype(np.float32)
            length = int(constants[node.input[3]].item())
            if len(shape) not in (2, 3) or shape[0] != 1 or (len(shape) == 3 and shape[2] != 1) or shape[1] < length or length != 20 or hop != 5 or window.size != length or attrs.get("onesided", 1) != 1:
                raise ValueError(f"Unsupported STFT geometry at {node.name}: {shape}")
            rewritten.extend(bluestein_stft_nodes(node.input[0], node.output[0], shape, window, stem, constant))
            evidence.append({"source": node.name, "kind": "f32_bluestein_graph", "length": length, "hop": hop, "fft_extent": 64})
        elif node.op_type == "Resize":
            scales = constants.get(node.input[2]) if len(node.input) > 2 else None
            mode, coordinate = attrs.get("mode", b"nearest"), attrs.get("coordinate_transformation_mode", b"half_pixel")
            if len(shape) != 3 or min(shape) <= 0 or scales is None or scales.shape != (3,) or not np.array_equal(scales[:2], [1, 1]) or attrs.get("antialias", 0) != 0:
                raise ValueError(f"Unsupported resize geometry at {node.name}: {shape}")
            size = int(np.floor(shape[2] * float(scales[2])))
            positions = np.arange(size, dtype=np.float32)
            if coordinate == b"asymmetric":
                positions = positions / np.float32(scales[2])
            elif coordinate == b"half_pixel":
                positions = (positions + np.float32(.5)) / np.float32(scales[2]) - np.float32(.5)
            else:
                raise ValueError(f"Unsupported resize coordinates: {coordinate}")
            if mode == b"nearest" and attrs.get("nearest_mode", b"round_prefer_floor") == b"floor":
                indices = np.clip(np.floor(positions), 0, shape[2] - 1).astype(np.int64)
                rewritten.append(helper.make_node("Gather", [node.input[0], constant(stem + "_indices", indices)], list(node.output), axis=2))
            elif mode == b"linear":
                if shape[0] != 1:
                    raise ValueError("Kokoro linear Resize expects singleton batch")
                positions = np.clip(positions, 0, shape[2] - 1)
                left = np.floor(positions).astype(np.int64)
                right = np.minimum(left + 1, shape[2] - 1)
                # A rank-three source Resize uses trilinear interpolation,
                # including eight sequential contributions when two axes
                # have unit scale. Keep that F32 order: reducing it to two
                # terms changes the following oscillator's accumulated phase.
                wx = [np.abs(positions - right.astype(np.float32)), np.abs(positions - left.astype(np.float32))]
                for weight in wx:
                    weight[left == right] = np.float32(.5)
                channels = np.arange(shape[1], dtype=np.int64)
                next_channels = np.minimum(channels + 1, shape[1] - 1)
                wy = [np.ones(shape[1], dtype=np.float32), np.zeros(shape[1], dtype=np.float32)]
                for weight in wy:
                    weight[-1] = np.float32(.5)
                terms = []
                for y, channel_indices in enumerate((channels, next_channels)):
                    row = stem + f"_row_{y}"
                    rewritten.append(helper.make_node("Gather", [node.input[0], constant(row + "_indices", channel_indices)], [row], axis=1))
                    for x, sample_indices in enumerate((left, right)):
                        term = stem + f"_term_{y}_{x}"
                        gathered = term + "_gathered"
                        weight = (wx[x][None, None, :] * wy[y][None, :, None]) * np.float32(.5)
                        rewritten.extend([
                            helper.make_node("Gather", [row, constant(term + "_indices", sample_indices)], [gathered], axis=2),
                            helper.make_node("Mul", [gathered, constant(term + "_weight", weight)], [term]),
                        ])
                        terms.append(term)
                previous = terms[0]
                for term_index, term in enumerate((terms * 2)[1:]):
                    target = node.output[0] if term_index == 6 else stem + f"_sum_{term_index}"
                    rewritten.append(helper.make_node("Add", [previous, term], [target]))
                    previous = target
            else:
                raise ValueError(f"Unsupported resize mode: {mode}")
            evidence.append({"source": node.name, "kind": "resize_gather", "mode": mode.decode(), "input_extent": shape[2], "output_extent": size})
        elif node.op_type == "Pad":
            pads = constants.get(node.input[1])
            if len(shape) != 3 or min(shape) <= 0 or pads is None or pads.shape != (6,) or np.any(pads[[0, 1, 3, 4]]) or attrs.get("mode") != b"reflect":
                raise ValueError(f"Unsupported pad geometry at {node.name}: {shape}")
            before, after = int(pads[2]), int(pads[5])
            if min(before, after) < 0 or max(before, after) >= shape[2]:
                raise ValueError("Reflection pads must be smaller than the source extent")
            indices = np.pad(np.arange(shape[2], dtype=np.int64), (before, after), mode="reflect")
            rewritten.append(helper.make_node("Gather", [node.input[0], constant(stem + "_indices", indices)], list(node.output), axis=2))
            evidence.append({"source": node.name, "kind": "reflection_gather", "before": before, "after": after})
        else:
            rewritten.append(copy.deepcopy(node))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    used = {name for node in model.graph.node for name in node.input}
    retained = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    onnx.checker.check_model(model)
    return model, evidence


def lower_decoder(model, tokens: int, frames: int):
    source = decoder(model, tokens, frames)
    # The source inverse STFT has a constant window-index program. Only that
    # geometry is specialized, using the duration-derived output extent.
    boundaries = {value.name: value for value in source.graph.input}
    source = specialized_head(source, None, np.empty((1, 1, frames * 600 + 20), dtype=np.float32),
        np.empty((1, frames * 600), dtype=np.float32), boundaries=boundaries)
    source = fold_static_programs(source)
    source, geometry = lower_geometry(source)
    source = extract(source, boundaries, {"waveform": info("waveform", [1, frames * 600])})
    return fold_static_programs(source), {"geometry": geometry, "normalization": "original centered ReduceMean/Sub/Pow/Sqrt/Div expressions preserved"}
