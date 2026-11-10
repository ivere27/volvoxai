"""Time-major ONNX LSTM contracts with explicit default activation semantics."""
from .operator_shape_contracts_common import (
    _assert_allowed_fields, _assert_float_tensor, _assert_i32_tensor, _assert_rank,
    _constant_logical_shape, _fail, _fixed_dimension_value, _logical_output,
    _logical_request_parts, _logical_shapes_provably_equal, _output,
    _positive_safe_integer_param, _request_parts,
)


REQUIRED = ("input", "weight", "recurrent_weight")
OPTIONAL = ("bias", "initial_h", "initial_c")

PACKED_REQUIRED = ("input", "weight", "scale", "recurrent_weight", "recurrent_scale")
PACKED_OPTIONAL = ("bias", "initial_h", "initial_c")


def _shape(inputs, params, environment=None):
    _assert_allowed_fields(params, ("direction",), "operator params")
    direction = params.get("direction", 0)
    if isinstance(direction, bool) or not isinstance(direction, int) or direction not in (0, 1, 2):
        _fail("INVALID_PARAMS", "operator params.direction", "must be 0 (forward), 1 (reverse), or 2 (bidirectional).")
    for name, value in inputs.items():
        _assert_float_tensor(value, f"operator input '{name}'")
    x, w, r = (inputs[name] for name in REQUIRED)
    for name, value in zip(REQUIRED, (x, w, r)):
        _assert_rank(value.shape, 0, 3, f"operator input '{name}'.shape")
    ws = _constant_logical_shape(w.shape, "operator input 'weight'.shape")
    rs = _constant_logical_shape(r.shape, "operator input 'recurrent_weight'.shape")
    directions, hidden = (2 if direction == 2 else 1), rs[2]
    width = x.shape[2] if environment is None else _fixed_dimension_value(x.shape[2], environment)
    if rs != (directions, 4 * hidden, hidden) or ws != (directions, 4 * hidden, width):
        _fail("SHAPE_MISMATCH", "operator inputs", "LSTM input projection and recurrence shapes do not match direction/hidden/input widths.")
    for name in OPTIONAL:
        if name not in inputs:
            continue
        expected = (directions, 8 * hidden) if name == "bias" else (directions, x.shape[1], hidden)
        equal = inputs[name].shape == expected if environment is None else _logical_shapes_provably_equal(inputs[name].shape, expected, environment)
        if not equal:
            _fail("SHAPE_MISMATCH", f"operator input '{name}'.shape", "does not match LSTM direction, batch and hidden extents.")
    return (x.shape[0], directions, x.shape[1], hidden)


def infer_lstm(_operator, request):
    inputs, params, _ = _request_parts(request, REQUIRED, OPTIONAL)
    return _output(_shape(inputs, params), "float32")


def prove_lstm(_operator, request):
    environment, inputs, params, _ = _logical_request_parts(request, REQUIRED, OPTIONAL)
    return (_logical_output(_shape(inputs, params, environment), "float32", environment),
            ("LSTM preserves the bounded time/batch domains and uses fixed projection/hidden extents",))


def _packed_shape(inputs, params, environment=None):
    _assert_allowed_fields(params, ("direction", "group_size", "weight_bits"), "operator params")
    direction = params.get("direction", 0)
    if isinstance(direction, bool) or not isinstance(direction, int) or direction not in (0, 1, 2):
        _fail("INVALID_PARAMS", "operator params.direction", "must be 0 (forward), 1 (reverse), or 2 (bidirectional).")
    group_size = _positive_safe_integer_param(params, "group_size", 32)
    if group_size not in (32, 64, 128, 256):
        _fail("INVALID_PARAMS", "operator params.group_size", "PackedLSTM requires group_size=32, 64, 128 or 256.")
    weight_bits = _positive_safe_integer_param(params, "weight_bits", 4)
    if weight_bits not in (2, 3, 4, 8):
        _fail("INVALID_PARAMS", "operator params.weight_bits", "PackedLSTM requires weight_bits=2, 3, 4 or 8.")

    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _assert_i32_tensor(inputs["weight"], "operator input 'weight'")
    _assert_float_tensor(inputs["scale"], "operator input 'scale'")
    _assert_i32_tensor(inputs["recurrent_weight"], "operator input 'recurrent_weight'")
    _assert_float_tensor(inputs["recurrent_scale"], "operator input 'recurrent_scale'")

    x = inputs["input"]
    w = inputs["weight"]
    ws = inputs["scale"]
    r = inputs["recurrent_weight"]
    rs = inputs["recurrent_scale"]

    for name, value in (("input", x), ("weight", w), ("scale", ws), ("recurrent_weight", r), ("recurrent_scale", rs)):
        _assert_rank(value.shape, 0, 3, f"operator input '{name}'.shape")

    w_shape = _constant_logical_shape(w.shape, "operator input 'weight'.shape")
    ws_shape = _constant_logical_shape(ws.shape, "operator input 'scale'.shape")
    r_shape = _constant_logical_shape(r.shape, "operator input 'recurrent_weight'.shape")
    rs_shape = _constant_logical_shape(rs.shape, "operator input 'recurrent_scale'.shape")

    directions = 2 if direction == 2 else 1
    if r_shape[0] != directions or r_shape[1] % 4 != 0:
        _fail("SHAPE_MISMATCH", "operator input 'recurrent_weight'.shape", "must have shape [directions, 4*hidden, recurrent_words].")
    hidden = r_shape[1] // 4
    if hidden <= 0:
        _fail("SHAPE_MISMATCH", "operator input 'recurrent_weight'.shape", "hidden size must be positive.")

    width = x.shape[2] if environment is None else _fixed_dimension_value(x.shape[2], environment)
    if not isinstance(width, int) or width <= 0:
        _fail("SHAPE_MISMATCH", "operator input 'input'.shape", "input width must be a positive integer.")

    w_words = (width * weight_bits + 31) // 32
    w_groups = (width + group_size - 1) // group_size
    r_words = (hidden * weight_bits + 31) // 32
    r_groups = (hidden + group_size - 1) // group_size

    if w_shape != (directions, 4 * hidden, w_words):
        _fail("SHAPE_MISMATCH", "operator input 'weight'.shape", f"expected {(directions, 4 * hidden, w_words)}, got {w_shape}.")
    if ws_shape != (directions, 4 * hidden, w_groups):
        _fail("SHAPE_MISMATCH", "operator input 'scale'.shape", f"expected {(directions, 4 * hidden, w_groups)}, got {ws_shape}.")
    if r_shape != (directions, 4 * hidden, r_words):
        _fail("SHAPE_MISMATCH", "operator input 'recurrent_weight'.shape", f"expected {(directions, 4 * hidden, r_words)}, got {r_shape}.")
    if rs_shape != (directions, 4 * hidden, r_groups):
        _fail("SHAPE_MISMATCH", "operator input 'recurrent_scale'.shape", f"expected {(directions, 4 * hidden, r_groups)}, got {rs_shape}.")

    for name in PACKED_OPTIONAL:
        if name not in inputs:
            continue
        _assert_float_tensor(inputs[name], f"operator input '{name}'")
        expected = (directions, 8 * hidden) if name == "bias" else (directions, x.shape[1], hidden)
        equal = inputs[name].shape == expected if environment is None else _logical_shapes_provably_equal(inputs[name].shape, expected, environment)
        if not equal:
            _fail("SHAPE_MISMATCH", f"operator input '{name}'.shape", "does not match LSTM direction, batch and hidden extents.")

    return (x.shape[0], directions, x.shape[1], hidden)


def infer_packed_lstm(_operator, request):
    inputs, params, _ = _request_parts(request, PACKED_REQUIRED, PACKED_OPTIONAL)
    return _output(_packed_shape(inputs, params), "float32")


def prove_packed_lstm(_operator, request):
    environment, inputs, params, _ = _logical_request_parts(request, PACKED_REQUIRED, PACKED_OPTIONAL)
    return (_logical_output(_packed_shape(inputs, params, environment), "float32", environment),
            ("PackedLSTM preserves the bounded time/batch domains and uses fixed projection/hidden extents",))
