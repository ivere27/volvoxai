#!/usr/bin/env python3
"""Generic PyTorch interpreter — the scalable oracle for reaching *all* operators.

Instead of one reference function per case, this walks any authored graph
(graph.json + inputs/ + weights/) through a torch op-map. Adding a new operator
to the parity suite is then: one case in cases.mjs + one entry in OP below. Used
for both the L1 single-node op cases and the L2 mixed graphs.

Usage:  python3 tests/parity/graphs/torch_interp.py <cases_dir> <sig_dir>
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PARITY, "external"))
from sigutil import signature, write_json_atomic  # noqa: E402


def _rows(x):
    return x.reshape(-1, x.shape[-1])


def _matmul(T, node):
    x = T[node["inputs"]["input"]]
    W = T[node["inputs"]["weight"]]
    k = x.shape[-1]
    xr = _rows(x)
    # [K,N] => din (x@W); [N,K] => dout (x@W^T). Square is disambiguated by trying din first.
    y = xr @ W if W.shape[0] == k else xr @ W.t()
    if "bias" in node["inputs"]:
        y = y + T[node["inputs"]["bias"]]
    return y


def _layernorm(T, node):
    x = _rows(T[node["inputs"]["input"]])
    p = node.get("params", {})
    w = T[node["inputs"]["weight"]]
    b = T[node["inputs"]["bias"]] if "bias" in node["inputs"] else None
    return F.layer_norm(x, [x.shape[-1]], w, b, eps=p.get("eps", 1e-6))


def _rmsnorm(T, node):
    x = _rows(T[node["inputs"].get("input", node["inputs"].get("x"))])
    w = T[node["inputs"]["weight"]]
    eps = node.get("params", {}).get("eps", 1e-6)
    return x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps) * w


def _inp(T, node):
    return T[node["inputs"].get("input", node["inputs"].get("x", node["inputs"].get("data")))]


def _prelu(T, node):
    w = T[node["inputs"].get("slope", node["inputs"].get("weight"))]
    return F.prelu(_inp(T, node), w)


def _slice(T, node):
    x = _inp(T, node)
    p = node.get("params", {})
    starts, ends = p["starts"], p["ends"]
    axes = p.get("axes", list(range(len(starts))))
    steps = p.get("steps", [1] * len(starts))
    out = x
    for s, e, ax, st in zip(starts, ends, axes, steps):
        idx = [slice(None)] * x.dim()
        idx[ax] = slice(s, e, st)
        out = out[tuple(idx)]
    return out.contiguous()


def _flatten(T, node):
    x = _inp(T, node)
    axis = node.get("params", {}).get("axis", 1)
    if axis < 0:
        axis += x.dim()
    return x.reshape(math.prod(x.shape[:axis]), math.prod(x.shape[axis:]))


def _squeeze(T, node):
    out = _inp(T, node)
    axes = node.get("params", {}).get("axes", [])
    if not axes:
        return out.squeeze()
    for axis in sorted((a if a >= 0 else a + out.dim() for a in axes), reverse=True):
        out = out.squeeze(axis)
    return out


def _unsqueeze(T, node):
    out = _inp(T, node)
    for axis in sorted(node.get("params", {}).get("axes", [])):
        out = out.unsqueeze(axis)
    return out


# --- vision ops: VolvoxAI is NHWC; torch wants NCHW, so transpose around it -----
def _globalavgpool(T, node):
    return _inp(T, node).mean(dim=(1, 2), keepdim=True)  # NHWC → mean over H,W


def _batchnorm(T, node):
    x, i, p = _inp(T, node), node["inputs"], node.get("params", {})
    scale = T[i.get("scale", i.get("weight"))]
    bias = T[i.get("bias", i.get("b"))]
    mean = T[i.get("mean", i.get("running_mean"))]
    var = T[i.get("var", i.get("running_var"))]
    return (x - mean) / torch.sqrt(var + p.get("eps", 1e-5)) * scale + bias  # channel = last dim


def _groupnorm(T, node):
    x, i, p = _inp(T, node), node["inputs"], node.get("params", {})
    xn = x.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
    b = T[i["bias"]] if "bias" in i else None
    y = F.group_norm(xn, p.get("num_groups", p.get("groups", 1)), T[i["weight"]], b, eps=p.get("eps", 1e-5))
    return y.permute(0, 2, 3, 1).contiguous()  # → NHWC


def _avgpool2d(T, node):
    p = node.get("params", {})
    k = tuple(p.get("kernel", [2, 2]))
    xn = _inp(T, node).permute(0, 3, 1, 2)  # NHWC → NCHW
    y = F.avg_pool2d(xn, kernel_size=k, stride=tuple(p.get("stride", k)))
    return y.permute(0, 2, 3, 1).contiguous()


def _upsample2x(T, node):
    xn = _inp(T, node).permute(0, 3, 1, 2)  # NHWC → NCHW
    y = F.interpolate(xn, scale_factor=2, mode="nearest")
    return y.permute(0, 2, 3, 1).contiguous()


def _conv2d(T, node):
    x, i, p = _inp(T, node), node["inputs"], node.get("params", {})
    xn = x.permute(0, 3, 1, 2)                      # NHWC → NCHW
    layout = p.get("weight_layout", "OHWI")
    w = T[i["weight"]]
    if layout == "HWIO":                             # [kh,kw,I/g,O] → OIHW
        wn = w.permute(3, 2, 0, 1)
    elif layout == "HWCM":                           # [kh,kw,C,M] → [C*M,1,kh,kw]
        kh, kw, channels, multiplier = w.shape
        wn = w.permute(2, 3, 0, 1).reshape(channels * multiplier, 1, kh, kw)
    else:                                            # OHWI → OIHW
        wn = w.permute(0, 3, 1, 2)
    b = T[i["bias"]] if "bias" in i else None
    pads = p.get("pads", [0, 0, 0, 0])               # [t,l,b,r]; test cases use symmetric
    y = F.conv2d(xn, wn, b, stride=tuple(p.get("stride", [1, 1])),
                 padding=(pads[0], pads[1]), dilation=tuple(p.get("dilation", [1, 1])),
                 groups=p.get("groups", 1))
    if p.get("relu"):
        y = torch.clamp(y, 0, 6)                      # relu != 0 → fused ReLU6
    return y.permute(0, 2, 3, 1).contiguous()         # → NHWC


def _conv1d(T, node):
    # NLC activations [N,L,C]; WIO weights [k, in_per_group, out_c].
    x, i, p = _inp(T, node), node["inputs"], node.get("params", {})
    xn = x.permute(0, 2, 1)                          # NLC → NCL
    wn = T[i["weight"]].permute(2, 1, 0)             # WIO → OIW
    b = T[i["bias"]] if "bias" in i else None
    y = F.conv1d(xn, wn, b, stride=p.get("stride", 1),
                 padding=p.get("padding", 0), groups=p.get("groups", 1))
    if p.get("relu"):
        y = torch.clamp(y, min=0)
    return y.permute(0, 2, 1).contiguous()           # NCL → NLC


def _conv_transpose2d(T, node):
    # NHWC activations; HWIO weights [kh, kw, in_c, out_c] → torch wants [I,O,kh,kw].
    x, i, p = _inp(T, node), node["inputs"], node.get("params", {})
    xn = x.permute(0, 3, 1, 2)                       # NHWC → NCHW
    wn = T[i["weight"]].permute(2, 3, 0, 1)          # HWIO → IOHW
    b = T[i["bias"]] if "bias" in i else None
    y = F.conv_transpose2d(xn, wn, b, stride=tuple(p.get("stride", [1, 1])),
                           padding=tuple(p.get("padding", [0, 0])))
    return y.permute(0, 2, 3, 1).contiguous()        # NCHW → NHWC


def _pad(T, node):
    x, p = _inp(T, node), node.get("params", {})
    pads, v = p.get("pads", []), p.get("value", 0.0)
    if len(pads) == 8:
        pt, pl, pb, pr = pads[1], pads[2], pads[5], pads[6]
    elif len(pads) == 4:
        pt, pl, pb, pr = pads
    else:
        pt = pl = pb = pr = 0
    return F.pad(x, (0, 0, pl, pr, pt, pb, 0, 0), value=v)  # NHWC spatial pad


# Per-case hydrated quant metadata plus the loaded current v1 graph.
_STATE = {"cfg": {}, "Q": {}}


def _output_tensor(node, port="out"):
    return node["outputs"][port]["tensor"]


def _affine(T, tensor_name):
    descriptor = _STATE["cfg"]["quantization"]["tensors"][tensor_name]
    scale = T[descriptor["scale_tensor"]]
    zero = T[descriptor["zero_point_tensor"]]
    if descriptor["scheme"] == "per_tensor":
        return {"scale": scale.reshape(-1)[0], "zp": zero.reshape(-1)[0]}
    return {"scale": scale.reshape(-1), "zp": zero.reshape(-1)}


def _quantize(T, node):
    i = node["inputs"]
    zp = T[i["zero_point"]] if "zero_point" in i else 0
    output_name = _output_tensor(node)
    oq = _affine(T, output_name)
    _STATE["Q"][output_name] = oq
    return torch.clamp(torch.round(T[i["input"]] / T[i["scale"]]) + zp, -128, 127)  # int8 grid


def _qconv2d(T, node):
    # Faithful W8A8: int8 conv, int32 bias in the accumulator domain, per-axis
    # weight scales, requantize to the output int8 grid. (gemmlowp/TFLite style.)
    i, p = node["inputs"], node.get("params", {})
    iq = _STATE["Q"][i["input"]]
    in_s, in_zp = iq["scale"], iq["zp"]
    wsc = _affine(T, i["weight"])["scale"].to(torch.float64)
    oq = _affine(T, _output_tensor(node))
    out_s, out_zp = oq["scale"], oq["zp"]
    xn = (T[i["input"]] - in_zp).permute(0, 3, 1, 2)   # NHWC → NCHW
    wn = T[i["weight"]].permute(0, 3, 1, 2)             # OHWI → OIHW
    pads = p.get("pads", [0, 0, 0, 0])
    acc = F.conv2d(xn, wn, None, stride=tuple(p.get("stride", [1, 1])), padding=(pads[0], pads[1]),
                   dilation=tuple(p.get("dilation", [1, 1])), groups=p.get("groups", 1))
    if "bias" in i:
        acc = acc + T[i["bias"]].reshape(1, -1, 1, 1)   # int32 bias per output channel
    real = acc * in_s * wsc.reshape(1, -1, 1, 1)
    if p.get("relu"):
        real = torch.clamp(real, 0, 6)
    out_q = torch.clamp(torch.round(real / out_s) + out_zp, -128, 127)
    _STATE["Q"][_output_tensor(node)] = {"scale": out_s, "zp": out_zp}
    return out_q.permute(0, 2, 3, 1).contiguous()      # NCHW → NHWC


def _dequantize(T, node):
    i = node["inputs"]
    zp = T[i["zero_point"]] if "zero_point" in i else 0
    return (T[i["input"]] - zp) * T[i["scale"]]


def _sdpa(T, node):
    # qkv [B,S,3d]: Q=[..0:d], K=[..d:2d], V=[..2d:3d]; heads contiguous within each.
    qkv, p = T[node["inputs"]["qkv"]], node.get("params", {})
    if qkv.dim() == 2:
        qkv = qkv.unsqueeze(0)
    B, S, d3 = qkv.shape
    d = d3 // 3
    H = p.get("heads", 8)
    hd = d // H
    scale = p.get("scale", 1.0 / math.sqrt(hd))
    causal = p.get("causal", True) is not False

    def head(t):
        return t.reshape(B, S, H, hd).permute(0, 2, 1, 3)
    q, k, v = head(qkv[..., :d]), head(qkv[..., d:2 * d]), head(qkv[..., 2 * d:])
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    return o.permute(0, 2, 1, 3).reshape(B, S, d)


OP = {
    "Add": lambda T, n: T[n["inputs"]["a"]] + T[n["inputs"]["b"]],
    "Sub": lambda T, n: T[n["inputs"]["a"]] - T[n["inputs"]["b"]],
    "Mul": lambda T, n: T[n["inputs"]["a"]] * T[n["inputs"]["b"]],
    "Div": lambda T, n: T[n["inputs"]["a"]] / T[n["inputs"]["b"]],
    "ReLU": lambda T, n: torch.relu(_inp(T, n)),
    "GELU": lambda T, n: F.gelu(_inp(T, n)),
    "SiLU": lambda T, n: F.silu(_inp(T, n)),
    "Sigmoid": lambda T, n: torch.sigmoid(_inp(T, n)),
    "Tanh": lambda T, n: torch.tanh(_inp(T, n)),
    "LeakyReLU": lambda T, n: F.leaky_relu(_inp(T, n), n.get("params", {}).get("alpha", 0.01)),
    "Clip": lambda T, n: torch.clamp(_inp(T, n), n.get("params", {}).get("min"), n.get("params", {}).get("max")),
    "HardSigmoid": lambda T, n: F.hardsigmoid(_inp(T, n)),
    "HardSwish": lambda T, n: F.hardswish(_inp(T, n)),
    "PReLU": _prelu,
    "Sin": lambda T, n: torch.sin(_inp(T, n)),
    "Cos": lambda T, n: torch.cos(_inp(T, n)),
    "MatMul": _matmul,
    "Linear": _matmul,
    "LayerNorm": _layernorm,
    "RMSNorm": _rmsnorm,
    "Softmax": lambda T, n: F.softmax(_rows(_inp(T, n)), dim=-1),
    "LogSoftmax": lambda T, n: F.log_softmax(_rows(_inp(T, n)), dim=-1),
    "ReduceMean": lambda T, n: _rows(_inp(T, n)).mean(dim=-1),
    "ReduceSum": lambda T, n: _rows(_inp(T, n)).sum(dim=-1),
    "Transpose": lambda T, n: _inp(T, n).permute(*n["params"]["perm"]).contiguous(),
    "Reshape": lambda T, n: _inp(T, n).reshape(*n["outputs"]["out"]["shape"]),
    "Flatten": _flatten,
    "Identity": lambda T, n: _inp(T, n),
    "Squeeze": _squeeze,
    "Unsqueeze": _unsqueeze,
    "Slice": _slice,
    "Expand": lambda T, n: _inp(T, n).expand(*n["outputs"]["out"]["shape"]).contiguous(),
    "Broadcast": lambda T, n: _inp(T, n).expand(*n["outputs"]["out"]["shape"]).contiguous(),
    "Concat": lambda T, n: torch.cat([T[v] for k, v in sorted(n["inputs"].items()) if k.startswith("input")], dim=n["params"]["axis"]),
    "GlobalAveragePool": _globalavgpool,
    "BatchNorm2D": _batchnorm,
    "GroupNorm": _groupnorm,
    "AveragePool2D": _avgpool2d,
    "UpsampleNearest2D": _upsample2x,
    "Pad": _pad,
    "MeanHeight": lambda T, n: _inp(T, n).mean(dim=1).permute(0, 2, 1).contiguous(),  # NHWC → [N,C,W]
    "ArgMax": lambda T, n: _inp(T, n).argmax(dim=n.get("params", {}).get("axis", -1)).to(torch.float64),
    "Conv2D": _conv2d,
    "Conv1D": _conv1d,
    "ConvTranspose2D": _conv_transpose2d,
    "QuantizeLinear": _quantize,
    "DequantizeLinear": _dequantize,
    "QConv2D": _qconv2d,
    "SDPA": _sdpa,
}


def run_case(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    cfg = json.load(open(os.path.join(d, "graph.json")))
    T = {}
    for name, spec in cfg["inputs"].items():
        arr = np.fromfile(os.path.join(d, "inputs", f"{name}.f32"), dtype=np.float32).astype(np.float64)
        T[name] = torch.from_numpy(arr).reshape(spec["shape"])
    DT_NP = {"f32": np.float32, "i8": np.int8, "i32": np.int32, "u8": np.uint8}
    wdt = meta.get("wDtypes", {})
    for name, shape in meta.get("wShapes", {}).items():
        dt = wdt.get(name, "f32")
        arr = np.fromfile(os.path.join(d, "weights", f"{name}.{dt}"), dtype=DT_NP[dt]).astype(np.float64)
        T[name] = torch.from_numpy(arr).reshape(shape)
    _STATE["cfg"] = cfg
    _STATE["Q"] = {}
    for node in cfg["nodes"]:
        if node["opType"] not in OP:
            return None  # op not yet mapped → skip (reported as uncovered)
        output = OP[node["opType"]](T, node)
        declared_shape = node.get("outputs", {}).get("out", {}).get("shape")
        if declared_shape is not None:
            expected = math.prod(declared_shape)
            if output.numel() != expected:
                raise ValueError(
                    f'{node["opType"]} produced {output.numel()} elements; '
                    f'declared output shape {declared_shape} requires {expected}'
                )
            # Several row-wise reference helpers intentionally flatten leading
            # dimensions while calculating. Restore the graph-declared logical
            # shape so the v2 structural signature checks the same tensor ABI as
            # VolvoxAI instead of silently accepting a flattened oracle output.
            output = output.reshape(tuple(declared_shape))
        T[_output_tensor(node)] = output
    final = cfg["outputs"][0]
    return T[final]


def main():
    cases_dir, sig_dir = sys.argv[1], sys.argv[2]
    os.makedirs(sig_dir, exist_ok=True)
    ran = 0
    for cid in sorted(os.listdir(cases_dir)):
        d = os.path.join(cases_dir, cid)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        try:
            out = run_case(d)
        except Exception as e:  # noqa: BLE001
            print(f"  torch-interp {cid}: error {e}", file=sys.stderr)
            out = None
        if out is None:
            continue
        output = out.detach().numpy()
        write_json_atomic(
            os.path.join(sig_dir, f"{cid}.torch.json"),
            {"id": cid, "tier": "torch", "sig": signature(output, topk=0, shape=list(output.shape))},
        )
        ran += 1
    print(f"torch-interp: wrote {ran} reference signatures from {cases_dir}")


if __name__ == "__main__":
    main()
