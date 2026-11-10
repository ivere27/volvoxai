"""Exact fixed-shape lowering of the pinned Piper VITS checkpoint.

Only shape/index programs and the sparse implementation of elementwise splines
are rewritten. Learned tensors remain graph inputs, activations, or the original
checkpoint initializers. Random draws become explicit request inputs.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DURATION_NOISE = "piper_duration_noise"
ACOUSTIC_NOISE = "piper_acoustic_noise"
DURATION_OUTPUTS = {
    "durations": "/Ceil_output_0",
    "frames": "/Cast_output_0",
    "mean": "/enc_p/Split_output_0",
    "log_scale": "/enc_p/Split_output_1",
    "mask": "/enc_p/Cast_1_output_0",
}


class Builder:
    def __init__(self, prefix):
        self.prefix, self.nodes, self.initializers = prefix, [], []
        self.serial = 0

    def constant(self, value, dtype=np.float32):
        name = f"{self.prefix}/c{self.serial}"
        self.serial += 1
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, kind, *inputs, output=None, **attributes):
        name = f"{self.prefix}/n{self.serial}"
        self.serial += 1
        output = output or name
        self.nodes.append(helper.make_node(kind, list(inputs), [output], name=name, **attributes))
        return output

    def reshape(self, value, shape):
        return self.op("Reshape", value, self.constant(shape, np.int64))

    def slice(self, value, start, end, axis=1):
        return self.op("Slice", value, self.constant([start], np.int64),
                       self.constant([end], np.int64), self.constant([axis], np.int64))


def dense_spline(prefix: str, tokens: int, boundary_derivative: np.float32):
    """Evaluate every independent spline lane, then select its identity tail.

    The source compacts lanes in [-5, 5] with NonZero/GatherND and scatters
    results back. Dense lanes are equivalent because all spline calculations
    are independent along this dimension. Out-of-domain lanes use a safe zero
    during calculation and retain their exact identity value in the final Where.
    """
    b = Builder(prefix + "/dense")
    y = b.reshape(prefix + "/Split_output_1", [tokens, 1])
    lo, hi, zero = b.constant(-5), b.constant(5), b.constant(0)
    inside = b.op("And", b.op("GreaterOrEqual", y, lo), b.op("LessOrEqual", y, hi))
    safe_y = b.op("Where", inside, y, zero)
    widths = b.reshape(prefix + "/Div_output_0", [tokens, 10])
    heights = b.reshape(prefix + "/Div_1_output_0", [tokens, 10])
    derivatives = b.reshape(prefix + "/Slice_2_output_0", [tokens, 9])
    edge = b.constant(np.full((tokens, 1), boundary_derivative, np.float32))
    derivatives = b.op("Concat", edge, derivatives, edge, axis=1)
    derivatives = b.op("Add", b.op("Softplus", derivatives), b.constant(.001))
    axis = b.constant(1, np.int64)
    reduce_axis = b.constant([1], np.int64)

    def bins(raw):
        unit = b.op("Add", b.op("Mul", b.op("Softmax", raw, axis=1),
                    b.constant(np.float32(1) - np.float32(10) * np.float32(.001))), b.constant(.001))
        cumulative = b.op("CumSum", unit, axis)
        cumulative = b.op("Concat", b.constant(np.zeros((tokens, 1), np.float32)), cumulative, axis=1)
        cumulative = b.op("Add", b.op("Mul", cumulative, b.constant(10)), lo)
        cumulative = b.op("Concat", b.constant(np.full((tokens, 1), -5, np.float32)),
                          b.slice(cumulative, 1, 10), b.constant(np.full((tokens, 1), 5, np.float32)), axis=1)
        size = b.op("Sub", b.slice(cumulative, 1, 11), b.slice(cumulative, 0, 10))
        return cumulative, size

    cumulative_widths, widths = bins(widths)
    cumulative_heights, heights = bins(heights)
    search_heights = b.op("Concat", b.slice(cumulative_heights, 0, 10),
                         b.constant(np.full((tokens, 1), np.float32(5) + np.float32(1e-6), np.float32)), axis=1)
    comparisons = b.op("Cast", b.op("GreaterOrEqual", safe_y, search_heights), to=TensorProto.FLOAT)
    index = b.op("Sub", b.op("ReduceSum", comparisons, reduce_axis, keepdims=1), b.constant(1))
    selection = b.op("Cast", b.op("Equal", index, b.constant(np.arange(10, dtype=np.float32)[None, :])), to=TensorProto.FLOAT)

    def select(value):
        return b.op("ReduceSum", b.op("Mul", value, selection), reduce_axis, keepdims=1)

    x0 = select(b.slice(cumulative_widths, 0, 10))
    width = select(widths)
    y0 = select(b.slice(search_heights, 0, 10))
    delta = select(b.op("Div", heights, widths))
    left = select(b.slice(derivatives, 0, 10))
    right = select(b.slice(derivatives, 1, 11))
    height = select(heights)
    difference = b.op("Sub", safe_y, y0)
    curvature = b.op("Sub", b.op("Add", left, right), b.op("Mul", delta, b.constant(2)))
    product = b.op("Mul", difference, curvature)
    quadratic = b.op("Add", product, b.op("Mul", height, b.op("Sub", delta, left)))
    linear = b.op("Sub", b.op("Mul", height, left), product)
    constant = b.op("Mul", b.op("Neg", delta), difference)
    discriminant = b.op("Sub", b.op("Mul", linear, linear),
                        b.op("Mul", b.op("Mul", quadratic, b.constant(4)), constant))
    root = b.op("Div", b.op("Mul", constant, b.constant(2)),
                b.op("Sub", b.op("Neg", linear), b.op("Sqrt", discriminant)))
    result = b.op("Add", b.op("Mul", root, width), x0)
    selected = b.op("Where", inside, result, y)
    b.op("Reshape", selected, b.constant([1, 1, tokens], np.int64), output=prefix + "/ScatterND_9_output_0")
    return b.nodes, b.initializers


def lower_splines(model: onnx.ModelProto, tokens: int) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    source = list(result.graph.node)
    for flow in (7, 5, 3):
        prefix = f"/dp/flows.{flow}"
        start = next(i for i, node in enumerate(source) if node.name == prefix + "/GreaterOrEqual")
        end = next(i for i, node in enumerate(source) if node.name == prefix + "/ScatterND_9")
        derivative_node = next(node for node in source if node.name == prefix + "/ConstantOfShape_1")
        derivative = numpy_helper.to_array(helper.get_attribute_value(derivative_node.attribute[0])).item()
        replacement, initializers = dense_spline(prefix, tokens, np.float32(derivative))
        source[start:end + 1] = replacement
        result.graph.initializer.extend(initializers)
    result.graph.ClearField("node")
    result.graph.node.extend(source)
    result.graph.ClearField("value_info")
    return result


def externalize_noise(model: onnx.ModelProto, tokens: int, frames: int | str = "frames") -> onnx.ModelProto:
    result = copy.deepcopy(model)
    for node in result.graph.node:
        if node.op_type != "RandomNormalLike":
            continue
        duration = node.name.startswith("/dp/")
        name = DURATION_NOISE if duration else ACOUSTIC_NOISE
        shape = [1, 2, tokens] if duration else [1, 192, frames]
        result.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        node.op_type = "Identity"
        node.ClearField("attribute")
        del node.input[:]
        node.input.append(name)
    return result


def prune(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove dead shape/index graphs after model-specific rewrites."""
    needed = {value.name for value in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    result = copy.deepcopy(model)
    result.graph.ClearField("node")
    result.graph.node.extend(reversed(kept))
    initializers = [value for value in result.graph.initializer if value.name in needed]
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(initializers)
    inputs = [value for value in result.graph.input if value.name in needed]
    result.graph.ClearField("input")
    result.graph.input.extend(inputs)
    return result


def extract(model, inputs, outputs):
    inferred = onnx.shape_inference.infer_shapes(model)
    result = onnx.utils.Extractor(inferred).extract_model(inputs, outputs)
    result.graph.ClearField("value_info")
    return prune(result)


def lower_static_layouts(model):
    """Lower padding, immutable splits and reversed axes to ordinary layouts."""
    shapes = {value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
              for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    arrays = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    shapes.update({name: list(value.shape) for name, value in arrays.items()})
    rewritten = []
    for node in model.graph.node:
        attributes = {value.name: helper.get_attribute_value(value) for value in node.attribute}
        if node.op_type == "ReduceSum" and len(node.input) > 1:
            axes = sorted(int(value) % len(shapes[node.input[0]]) for value in arrays[node.input[1]].reshape(-1))
            if len(axes) > 1:
                if axes != list(range(axes[0], axes[-1] + 1)):
                    raise ValueError(f"{node.name}: reduction axes must be contiguous")
                shape = shapes[node.input[0]]
                collapsed = shape[:axes[0]] + [int(np.prod([shape[axis] for axis in axes]))] + shape[axes[-1] + 1:]
                b = Builder(node.name + "/static_reduce")
                value = b.reshape(node.input[0], collapsed)
                value = b.op("ReduceSum", value, b.constant([axes[0]], np.int64), keepdims=0)
                if attributes.get("keepdims", 1):
                    final_shape = [1 if axis in axes else size for axis, size in enumerate(shape)]
                    value = b.reshape(value, final_shape)
                b.op("Identity", value, output=node.output[0])
                rewritten.extend(b.nodes)
                model.graph.initializer.extend(b.initializers)
                continue
        if node.op_type == "Split":
            shape = shapes[node.input[0]]
            axis = int(attributes.get("axis", 0)) % len(shape)
            sizes = (arrays[node.input[1]].reshape(-1).tolist() if len(node.input) > 1 else
                     attributes.get("split", [shape[axis] // len(node.output)] * len(node.output)))
            if sum(sizes) != shape[axis] or len(sizes) != len(node.output):
                raise ValueError(f"{node.name}: split extents do not cover input")
            b, offset = Builder(node.name + "/static_split"), 0
            for output, size in zip(node.output, sizes):
                sliced = b.slice(node.input[0], offset, offset + int(size), axis)
                b.op("Identity", sliced, output=output)
                offset += int(size)
            rewritten.extend(b.nodes)
            model.graph.initializer.extend(b.initializers)
            continue
        if node.op_type == "Slice" and len(node.input) > 4 and node.input[4] and np.any(arrays[node.input[4]] < 0):
            shape = shapes[node.input[0]]
            starts, ends = arrays[node.input[1]].reshape(-1), arrays[node.input[2]].reshape(-1)
            axes = arrays[node.input[3]].reshape(-1) if node.input[3] else range(len(starts))
            steps = arrays[node.input[4]].reshape(-1)
            b, current = Builder(node.name + "/static_reverse"), node.input[0]
            for start, end, axis, step in zip(starts, ends, axes, steps):
                axis = int(axis) % len(shape)
                indices = np.arange(shape[axis], dtype=np.int64)[slice(int(start), int(end), int(step))]
                current = b.op("Gather", current, b.constant(indices, np.int64), axis=axis)
            b.op("Identity", current, output=node.output[0])
            rewritten.extend(b.nodes)
            model.graph.initializer.extend(b.initializers)
            continue
        if node.op_type != "Pad":
            rewritten.append(node)
            continue
        if attributes.get("mode", b"constant") != b"constant":
            raise ValueError(f"{node.name}: only constant padding is admitted")
        shape = list(shapes[node.input[0]])
        if not shape or any(extent <= 0 for extent in shape):
            raise ValueError(f"{node.name}: padding requires a concrete positive shape")
        pads = arrays[node.input[1]].reshape(-1)
        if len(pads) != 2 * len(shape):
            raise ValueError(f"{node.name}: unexpected padding rank")
        fill = arrays[node.input[2]].item() if len(node.input) > 2 and node.input[2] else 0.0
        b = Builder(node.name + "/static_pad")
        current = node.input[0]
        for axis in range(len(shape)):
            before, after = int(pads[axis]), int(pads[axis + len(shape)])
            if before < 0 or after < 0:
                current = b.slice(current, max(-before, 0), shape[axis] + min(after, 0), axis)
                shape[axis] += min(before, 0) + min(after, 0)
            pieces = []
            if before > 0:
                padded = list(shape)
                padded[axis] = before
                pieces.append(b.constant(np.full(padded, fill, np.float32)))
            pieces.append(current)
            if after > 0:
                padded = list(shape)
                padded[axis] = after
                pieces.append(b.constant(np.full(padded, fill, np.float32)))
            if len(pieces) > 1:
                current = b.op("Concat", *pieces, axis=axis)
                shape[axis] += max(before, 0) + max(after, 0)
        b.op("Identity", current, output=node.output[0])
        rewritten.extend(b.nodes)
        model.graph.initializer.extend(b.initializers)
    model.graph.ClearField("node")
    model.graph.node.extend(rewritten)
    return model


def simplify_static(model, shapes):
    """Fold only graph constants and fixed shape metadata; never request values."""
    import onnxsim
    model.graph.ClearField("value_info")
    arrays = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Gather" and node.input[0] == "scales":
            index = arrays[node.input[1]]
            if index.ndim == 0:
                name = node.input[1] + "__scale_vector"
                model.graph.initializer.append(numpy_helper.from_array(index.reshape(1), name))
                node.input[1] = name
    # Producer symbolic metadata is not authoritative after fixing input shapes.
    for value in model.graph.output:
        for index, dimension in enumerate(value.type.tensor_type.shape.dim):
            dimension.ClearField("dim_value")
            dimension.dim_param = f"output_{value.name}_{index}"
    simplified, valid = onnxsim.simplify(model, overwrite_input_shapes=shapes, check_n=0)
    if not valid:
        raise ValueError("constant/shape simplification failed")
    simplified = lower_static_layouts(simplified)
    simplified, valid = onnxsim.simplify(simplified, check_n=0)
    if not valid:
        raise ValueError("static padding simplification failed")
    return simplified


def duration_model(source: Path, tokens: int, *, dense=True):
    model = externalize_noise(onnx.load(source), tokens)
    if dense:
        model = lower_splines(model, tokens)
    model = extract(model, ["input", "input_lengths", "scales", DURATION_NOISE], list(DURATION_OUTPUTS.values()))
    # Frame allocation is bounded by the adapter and represented by runtime I32.
    for node in model.graph.node:
        if node.name == "/Cast":
            for attribute in node.attribute:
                if attribute.name == "to":
                    attribute.i = TensorProto.INT32
    for value in model.graph.output:
        if value.name == DURATION_OUTPUTS["frames"]:
            value.type.tensor_type.elem_type = TensorProto.INT32
    return simplify_static(model, {"input": [1, tokens], "input_lengths": [1], "scales": [3], DURATION_NOISE: [1, 2, tokens]})


def acoustic_model(source: Path, tokens: int, frames: int):
    if frames <= 0:
        raise ValueError("frame count must be positive")
    model = externalize_noise(onnx.load(source), tokens, frames)
    inputs = [*DURATION_OUTPUTS.values(), "scales", ACOUSTIC_NOISE]
    model = extract(model, inputs, ["output"])
    # Range allocates the duration-dependent frame axis. The caller obtains
    # this extent from the Volvox duration stage and validates it before using
    # this specialization; the duration/mask tensors remain runtime inputs.
    nodes = [node for node in model.graph.node if node.name != "/Range"]
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.initializer.append(numpy_helper.from_array(np.arange(frames, dtype=np.int32), "/Range_output_0"))
    for value in model.graph.input:
        if value.name == DURATION_OUTPUTS["frames"]:
            value.type.tensor_type.elem_type = TensorProto.INT32
    shapes = {
        DURATION_OUTPUTS["durations"]: [1, 1, tokens], DURATION_OUTPUTS["frames"]: [1],
        DURATION_OUTPUTS["mean"]: [1, 192, tokens], DURATION_OUTPUTS["log_scale"]: [1, 192, tokens],
        DURATION_OUTPUTS["mask"]: [1, 1, tokens], "scales": [3], ACOUSTIC_NOISE: [1, 192, frames],
    }
    return simplify_static(prune(model), shapes)

