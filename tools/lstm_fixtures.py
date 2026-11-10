"""Independent ONNX LSTM fixtures shared by numerical tests and device qualification."""
from onnx import TensorProto, helper, numpy_helper


def lstm_tail_fixture():
    """Finite cell/gate tails, including states beyond exp's F32 range."""
    import numpy as np
    x = np.zeros((4, 1, 1), dtype=np.float32)
    w = np.zeros((2, 16, 1), dtype=np.float32)
    r = np.zeros((2, 16, 4), dtype=np.float32)
    bias = np.zeros((2, 32), dtype=np.float32)
    bias[:, 12:16] = [100, -100, -100, 100]
    h = np.zeros((2, 1, 4), dtype=np.float32)
    c = np.array([[[100, -100, 1e20, -1e20]]] * 2, dtype=np.float32)
    return x, w, r, bias, h, c, 2


def lstm_model(x, w, r, bias=None, h=None, c=None, direction=0):
    values = {"x": x, "w": w, "r": r}
    values.update({name: value for name, value in (("b", bias), ("h", h), ("c", c)) if value is not None})
    names = ["x", "w", "r", "b" if bias is not None else "", "", "h" if h is not None else "", "c" if c is not None else ""]
    directions = 2 if direction == 2 else 1
    model = helper.make_model(helper.make_graph([
        helper.make_node("LSTM", names, ["y"], hidden_size=r.shape[-1],
                         direction=("forward", "reverse", "bidirectional")[direction]),
    ], "lstm", [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [x.shape[0], directions, x.shape[1], r.shape[-1]])],
        [numpy_helper.from_array(value, name) for name, value in values.items() if name != "x"]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    return model

