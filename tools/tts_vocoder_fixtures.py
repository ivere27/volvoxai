"""Deterministic synthetic TTS/vocoder fixtures shared by CPU and GPU validation.

These graphs cover operator compositions used by duration-regulated vocoders and
Kokoro-style normalization/periodic activation, not complete pretrained models.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

Fixture = tuple[onnx.ModelProto, dict[str, np.ndarray], dict[str, np.ndarray]]


def build_tts_vocoder_fixture() -> Fixture:
    rng = np.random.default_rng(42)
    s_len = 4
    channels = 4
    t_mel = 8
    t_audio = 16

    w_pre = rng.standard_normal((channels, channels, 3), dtype=np.float32) * 0.2
    b_pre = rng.standard_normal((channels,), dtype=np.float32) * 0.05
    w_up = rng.standard_normal((channels, channels, 4), dtype=np.float32) * 0.2
    b_up = rng.standard_normal((channels,), dtype=np.float32) * 0.05
    w_dil = rng.standard_normal((channels, channels, 3), dtype=np.float32) * 0.2
    b_dil = rng.standard_normal((channels,), dtype=np.float32) * 0.05
    w_post = rng.standard_normal((1, channels, 3), dtype=np.float32) * 0.2
    b_post = rng.standard_normal((1,), dtype=np.float32) * 0.05
    frame_indices = np.arange(t_mel, dtype=np.int32).reshape(1, t_mel, 1)

    nodes = [
        # Exp(log_durations) -> Cast(INT32) -> CumSum -> interval mask -> MatMul length regulator
        helper.make_node("Exp", ["log_durations"], ["exp_durations"], name="exp_dur"),
        helper.make_node("Cast", ["exp_durations"], ["durations"], name="cast_dur", to=TensorProto.INT32),
        helper.make_node("CumSum", ["durations", "axis_one"], ["cum_end"], name="cumsum_end"),
        helper.make_node("CumSum", ["durations", "axis_one"], ["cum_start"], name="cumsum_start", exclusive=1),
        helper.make_node("Unsqueeze", ["cum_end", "axes_1"], ["cum_end_3d"], name="unsq_end"),
        helper.make_node("Unsqueeze", ["cum_start", "axes_1"], ["cum_start_3d"], name="unsq_start"),
        helper.make_node("GreaterOrEqual", ["frame_grid", "cum_start_3d"], ["ge_start"], name="ge_start"),
        helper.make_node("GreaterOrEqual", ["frame_grid", "cum_end_3d"], ["ge_end"], name="ge_end"),
        helper.make_node("Not", ["ge_end"], ["lt_end"], name="lt_end"),
        helper.make_node("Where", ["ge_start", "lt_end", "bool_zero"], ["in_span"], name="in_span"),
        helper.make_node("Cast", ["in_span"], ["alignment"], name="cast_align", to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["alignment", "hidden"], ["mel_nlc"], name="regulate_matmul"),
        helper.make_node("Transpose", ["mel_nlc"], ["mel_nct"], name="mel_to_nct", perm=[0, 2, 1]),
        # Pre-Conv1D (dilation=1) + LeakyRelu
        helper.make_node(
            "Conv", ["mel_nct", "w_pre", "b_pre"], ["pre_conv"],
            name="pre_conv1d", kernel_shape=[3], strides=[1], pads=[1, 1], dilations=[1],
        ),
        helper.make_node("LeakyRelu", ["pre_conv"], ["pre_act"], name="pre_lrelu", alpha=0.1),
        # 1D ConvTranspose upsampler (stride=2, kernel=4, pads=[1, 1])
        helper.make_node(
            "ConvTranspose", ["pre_act", "w_up", "b_up"], ["up_conv"],
            name="upsample_conv_transpose1d", kernel_shape=[4], strides=[2], pads=[1, 1],
        ),
        # HiFi-GAN style dilated 1D Conv ResBlock (dilation=3) + LeakyRelu + residual Add
        helper.make_node(
            "Conv", ["up_conv", "w_dil", "b_dil"], ["dil_conv"],
            name="resblock_dilated_conv1d", kernel_shape=[3], strides=[1], pads=[3, 3], dilations=[3],
        ),
        helper.make_node("LeakyRelu", ["dil_conv"], ["dil_act"], name="resblock_lrelu", alpha=0.1),
        helper.make_node("Add", ["up_conv", "dil_act"], ["res_out"], name="res_add"),
        # Post-Conv1D + Tanh -> waveform
        helper.make_node(
            "Conv", ["res_out", "w_post", "b_post"], ["post_conv"],
            name="post_conv1d", kernel_shape=[3], strides=[1], pads=[1, 1], dilations=[1],
        ),
        helper.make_node("Tanh", ["post_conv"], ["waveform"], name="waveform_tanh"),
    ]

    initializers = [
        numpy_helper.from_array(np.asarray(1, dtype=np.int64), "axis_one"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axes_1"),
        numpy_helper.from_array(frame_indices, "frame_grid"),
        numpy_helper.from_array(np.zeros((1, t_mel, s_len), dtype=np.bool_), "bool_zero"),
        numpy_helper.from_array(w_pre, "w_pre"),
        numpy_helper.from_array(b_pre, "b_pre"),
        numpy_helper.from_array(w_up, "w_up"),
        numpy_helper.from_array(b_up, "b_up"),
        numpy_helper.from_array(w_dil, "w_dil"),
        numpy_helper.from_array(b_dil, "b_dil"),
        numpy_helper.from_array(w_post, "w_post"),
        numpy_helper.from_array(b_post, "b_post"),
    ]

    graph_proto = helper.make_graph(
        nodes,
        "non_ar_tts_vocoder",
        [
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, s_len, channels]),
            helper.make_tensor_value_info("log_durations", TensorProto.FLOAT, [1, s_len]),
        ],
        [
            helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, 1, t_audio]),
        ],
        initializers,
    )
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    model = onnx.shape_inference.infer_shapes(model)

    hidden = rng.standard_normal((1, s_len, channels), dtype=np.float32)
    # log([2.5, 1.5, 3.5, 2.5]) truncates to int32 [2, 1, 3, 2]
    log_durations = np.log(np.asarray([[2.5, 1.5, 3.5, 2.5]], dtype=np.float32))
    durations = np.exp(log_durations).astype(np.int32)

    # Reference calculation in NumPy matching exact ONNX semantics
    cum_end = np.cumsum(durations, axis=1)
    cum_start = cum_end - durations
    grid = frame_indices
    in_span = (grid >= cum_start[:, None, :]) & (grid < cum_end[:, None, :])
    mel_nlc = in_span.astype(np.float32) @ hidden
    mel_nct = np.transpose(mel_nlc, (0, 2, 1))

    def conv1d_ref(x, w, b, pad=0, dilation=1):
        n, c_in, length = x.shape
        c_out, _, k = w.shape
        padded = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
        out_len = length + 2 * pad - dilation * (k - 1)
        out = np.zeros((n, c_out, out_len), dtype=np.float32)
        for ki in range(k):
            out += np.einsum("ncl,oc->nol", padded[:, :, ki * dilation : ki * dilation + out_len], w[:, :, ki])
        return out + b[None, :, None]

    def conv_transpose1d_ref(x, w, b, stride=1, pad=0):
        n, c_in, length = x.shape
        _, c_out, k = w.shape
        full_len = (length - 1) * stride + k
        full = np.zeros((n, c_out, full_len), dtype=np.float32)
        for t in range(length):
            for ki in range(k):
                full[:, :, t * stride + ki] += x[:, :, t] @ w[:, :, ki]
        cropped = full[:, :, pad : full_len - pad]
        return cropped + b[None, :, None]

    pre = conv1d_ref(mel_nct, w_pre, b_pre, pad=1, dilation=1)
    pre = np.where(pre >= 0, pre, pre * np.float32(0.1))
    up = conv_transpose1d_ref(pre, w_up, b_up, stride=2, pad=1)
    dil = conv1d_ref(up, w_dil, b_dil, pad=3, dilation=3)
    dil = np.where(dil >= 0, dil, dil * np.float32(0.1))
    res = up + dil
    expected = np.tanh(conv1d_ref(res, w_post, b_post, pad=1, dilation=1))
    return model, {"hidden": hidden, "log_durations": log_durations}, {"waveform": expected}


def build_kokoro_fixture() -> Fixture:
    rng = np.random.default_rng(123)
    b, c, l = 1, 4, 8
    x = rng.standard_normal((b, c, l), dtype=np.float32)
    scale = rng.standard_normal((c,), dtype=np.float32) * 0.5 + 1.0
    bias = rng.standard_normal((c,), dtype=np.float32) * 0.1
    alpha_val = np.asarray([0.5], dtype=np.float32)

    nodes = [
        helper.make_node("InstanceNormalization", ["x", "scale", "bias"], ["norm_out"], name="adain_inorm", epsilon=1e-5),
        # Snake1D activation: y = norm_out + (1/alpha) * sin(alpha * norm_out)**2
        helper.make_node("Mul", ["norm_out", "alpha"], ["scaled_x"], name="snake_scale"),
        helper.make_node("Sin", ["scaled_x"], ["sin_x"], name="snake_sin"),
        helper.make_node("Mul", ["sin_x", "sin_x"], ["sin2_x"], name="snake_sin2"),
        helper.make_node("Div", ["sin2_x", "alpha"], ["snake_term"], name="snake_div"),
        helper.make_node("Add", ["norm_out", "snake_term"], ["snake_out"], name="snake_add"),
        # Cosine basis projection:
        helper.make_node("Cos", ["snake_out"], ["cos_out"], name="stft_cos"),
    ]

    initializers = [
        numpy_helper.from_array(scale, "scale"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(alpha_val, "alpha"),
    ]

    graph_proto = helper.make_graph(
        nodes,
        "kokoro_subgraph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [b, c, l])],
        [helper.make_tensor_value_info("cos_out", TensorProto.FLOAT, [b, c, l])],
        initializer=initializers,
    )
    model_proto = helper.make_model(graph_proto, producer_name="test_kokoro", opset_imports=[helper.make_opsetid("", 17)], ir_version=8)

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm_ref = (x - mean) / np.sqrt(var + 1e-5) * scale[None, :, None] + bias[None, :, None]
    scaled_ref = norm_ref * alpha_val
    snake_ref = norm_ref + (1.0 / alpha_val) * (np.sin(scaled_ref) ** 2)
    expected = np.cos(snake_ref)
    return model_proto, {"x": x}, {"cos_out": expected}
