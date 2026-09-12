"""A small FP32 graph in PTQ-ready form, and a calibration profile for it.

The PTQ authoring stage has preconditions the general exporter pipeline
normally establishes: dense nodes carry explicit ``input``/``weight`` ports and
a declared ``weight_layout``, weights are immutable initializers, and
activations are dynamic F32. A fixture that already satisfies them lets a
parity test exercise authoring itself rather than the several passes that lead
up to it.

Everything here is derived from the tensor's own name, so the fixture is
identical on every machine and every run. That is what makes a byte-level
golden meaningful: a difference in the output is a difference in the
implementation, never in the input.

The shape is a transformer block reduced to its quantizable skeleton —
LayerNorm, three dense projections, GELU, and a residual Add — which covers
the dense, activation, and elementwise paths that ``typed_ptq`` treats
differently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import numpy as np

D_MODEL = 8
D_FF = 16
SEQUENCE = 4
FORMAT = "volvox-graph/v1"


def _values(name: str, count: int) -> np.ndarray:
    """Deterministic weights: the tensor's name is the whole seed."""

    digest = hashlib.sha256(name.encode("utf-8")).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return generator.uniform(-0.5, 0.5, size=count).astype(np.float32)


def tensors() -> dict[str, np.ndarray]:
    """The immutable payloads the graph refers to."""

    shapes = {
        "ln.weight": (D_MODEL,),
        "ln.bias": (D_MODEL,),
        # [dout, din]: the layout the packing path reads directly, so a
        # fixture in it exercises the pipeline rather than a transpose.
        "fc_in.weight": (D_FF, D_MODEL),
        "fc_in.bias": (D_FF,),
        "fc_out.weight": (D_MODEL, D_FF),
        "fc_out.bias": (D_MODEL,),
        "proj.weight": (D_MODEL, D_MODEL),
        "proj.bias": (D_MODEL,),
    }
    built: dict[str, np.ndarray] = {}
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        built[name] = _values(name, count).reshape(shape)
    # A LayerNorm scale near one keeps the activations in a range a per-tensor
    # affine can represent, which is the regime PTQ is for.
    built["ln.weight"] = (built["ln.weight"] + 1.0).astype(np.float32)
    return built


def document() -> dict[str, Any]:
    """The FP32 graph, already in the form dense PTQ requires."""

    def dense(node_id: str, prefix: str, source: str, out: str, width: int,
              bias: bool) -> dict[str, Any]:
        inputs = {"input": source, "weight": f"{prefix}.weight"}
        if bias:
            inputs["bias"] = f"{prefix}.bias"
        return {
            "id": node_id,
            "opType": "Linear",
            "inputs": inputs,
            "outputs": {
                "out": {
                    "tensor": out,
                    "shape": [1, SEQUENCE, width],
                    "dtype": "float32",
                }
            },
            # dout_din means [out, in]. PTQ reads the declaration rather than
            # inferring an orientation from the extents, which are square for
            # proj and would be ambiguous.
            "params": {"weight_layout": "dout_din"},
        }

    return {
        "format": FORMAT,
        "dimensions": {},
        "inputs": {
            "hidden": {"shape": [1, SEQUENCE, D_MODEL], "dtype": "float32"},
        },
        "nodes": [
            {
                "id": "node_0",
                "opType": "LayerNorm",
                "inputs": {
                    "input": "hidden",
                    "weight": "ln.weight",
                    "bias": "ln.bias",
                },
                "outputs": {
                    "out": {
                        "tensor": "normed",
                        "shape": [1, SEQUENCE, D_MODEL],
                        "dtype": "float32",
                    }
                },
                "params": {"eps": 1e-05, "d_model": D_MODEL},
            },
            dense("node_1", "fc_in", "normed", "expanded", D_FF, bias=True),
            {
                "id": "node_2",
                "opType": "GELU",
                "inputs": {"input": "expanded"},
                "outputs": {
                    "out": {
                        "tensor": "activated",
                        "shape": [1, SEQUENCE, D_FF],
                        "dtype": "float32",
                    }
                },
                "params": {},
            },
            dense("node_3", "fc_out", "activated", "contracted", D_MODEL,
                  bias=True),
            {
                "id": "node_4",
                "opType": "Add",
                "inputs": {"a": "hidden", "b": "contracted"},
                "outputs": {
                    "out": {
                        "tensor": "residual",
                        "shape": [1, SEQUENCE, D_MODEL],
                        "dtype": "float32",
                    }
                },
                "params": {},
            },
            # Every dense node carries a bias. A Linear without one is a real
            # case and both authoring implementations handle it — they emit
            # the packed int32 bias regardless, because it carries the
            # activation zero point's contribution — but the C package writer
            # requires a source bias to fold, so a bias-less fixture could not
            # be written end to end.
            dense("node_5", "proj", "residual", "output", D_MODEL, bias=True),
        ],
        "outputs": ["output"],
    }


def ranges(names: Iterable[str]) -> dict[str, dict[str, float]]:
    """Calibration ranges for exactly the tensors PTQ asks about.

    Derived from each name, so they are reproducible, and always straddling
    zero with an asymmetric span — which is the case that distinguishes a
    correct zero-point from one that happened to land on zero.
    """

    built: dict[str, dict[str, float]] = {}
    for name in names:
        digest = hashlib.sha256(f"range:{name}".encode("utf-8")).digest()
        low = -0.5 - (int.from_bytes(digest[0:4], "big") / 2**32) * 3.5
        high = 0.5 + (int.from_bytes(digest[4:8], "big") / 2**32) * 3.5
        built[name] = {"min": round(low, 6), "max": round(high, 6)}
    return built


SAMPLE_COUNT = 64
SAMPLE_DIGEST = "5" * 64


# --- a second graph, for operators the golden fixture does not reach --------
#
# The golden fixture is deliberately stable: changing it moves digests that
# exist to catch change. Operators added to authoring after it was frozen get
# their parity coverage here instead, where the shape is free to grow.

def batch_matmul_document() -> dict[str, Any]:
    """Two dynamic operands multiplied — no immutable weight anywhere.

    This is the case that distinguishes BatchMatMul from Linear: there is
    nothing to pack, so both operands are calibrated and the result is a
    byte-domain product.
    """

    return {
        "format": FORMAT,
        "dimensions": {},
        "inputs": {
            "left": {"shape": [1, SEQUENCE, D_MODEL], "dtype": "float32"},
            "right": {"shape": [1, D_MODEL, SEQUENCE], "dtype": "float32"},
        },
        "nodes": [
            {
                "id": "node_0",
                "opType": "BatchMatMul",
                "inputs": {"a": "left", "b": "right"},
                "outputs": {
                    "out": {
                        "tensor": "scores",
                        "shape": [1, SEQUENCE, SEQUENCE],
                        "dtype": "float32",
                    }
                },
                "params": {},
            },
            {
                "id": "node_1",
                "opType": "SiLU",
                "inputs": {"input": "scores"},
                "outputs": {
                    "out": {
                        "tensor": "activated_scores",
                        "shape": [1, SEQUENCE, SEQUENCE],
                        "dtype": "float32",
                    }
                },
                "params": {},
            },
        ],
        "outputs": ["activated_scores"],
    }


def batch_matmul_tensors() -> dict[str, np.ndarray]:
    """None: every value in that graph is dynamic."""

    return {}


CHANNELS = 4
HEIGHT = 6
WIDTH = 6


def conv_document() -> dict[str, Any]:
    """One NHWC convolution with stated geometry.

    Conv2D is the other operator that packs a weight, and its per-channel
    scales are indexed by the same axis a Linear's are — but only after the
    OHWI kernel is laid out that way. Stating the geometry rather than
    defaulting it is what makes the template describe the convolution the
    kernel will actually perform.
    """

    return {
        "format": FORMAT,
        "dimensions": {},
        "inputs": {
            "image": {
                "shape": [1, HEIGHT, WIDTH, CHANNELS],
                "dtype": "float32",
            },
        },
        "nodes": [
            {
                "id": "node_0",
                "opType": "Conv2D",
                "inputs": {
                    "input": "image",
                    "weight": "conv.weight",
                    "bias": "conv.bias",
                },
                "outputs": {
                    "out": {
                        "tensor": "features",
                        "shape": [1, HEIGHT, WIDTH, CHANNELS],
                        "dtype": "float32",
                    }
                },
                "params": {
                    "stride": [1, 1],
                    "dilation": [1, 1],
                    "padding": [1, 1],
                    "groups": 1,
                    "data_layout": "NHWC",
                    # A source Conv2D is HWIO; OHWI is the packed form the
                    # quantized kernel reads, and authoring is what moves
                    # between them.
                    "weight_layout": "HWIO",
                },
            },
            {
                "id": "node_1",
                "opType": "GELU",
                "inputs": {"input": "features"},
                "outputs": {
                    "out": {
                        "tensor": "activated_features",
                        "shape": [1, HEIGHT, WIDTH, CHANNELS],
                        "dtype": "float32",
                    }
                },
                "params": {},
            },
        ],
        "outputs": ["activated_features"],
    }


def conv_tensors() -> dict[str, np.ndarray]:
    """HWIO: [kh, kw, in_channels, out_channels]."""

    shapes = {"conv.weight": (3, 3, CHANNELS, CHANNELS), "conv.bias": (CHANNELS,)}
    return {
        name: _values(name, int(np.prod(shape))).reshape(shape)
        for name, shape in shapes.items()
    }


def cross_sdpa_document(*, mask: bool = False, scale: bool = False) -> dict[str, Any]:
    """Attention with q, k and v calibrated independently.

    Nothing is packed here — the three operands are all dynamic — so QSDPA is
    a byte node, and the optional keep mask is copied through untouched
    because it selects rather than scales.
    """

    heads = 2
    inputs = {"q": "query", "k": "key", "v": "value"}
    params: dict[str, Any] = {"heads": heads, "causal": False}
    if scale:
        params["scale"] = 0.125
    declared = {
        name: {"shape": [1, SEQUENCE, D_MODEL], "dtype": "float32"}
        for name in ("query", "key", "value")
    }
    if mask:
        # No mask_encoding: the operator registry allows only heads, causal
        # and scale, and an I32 mask is a keep mask by its dtype. PTQ reads
        # the field when a graph carries one, to refuse an additive mask that
        # slipped through.
        inputs["mask"] = "keep"
        declared["keep"] = {"shape": [SEQUENCE, SEQUENCE], "dtype": "int32"}

    return {
        "format": FORMAT,
        "dimensions": {},
        "inputs": declared,
        "nodes": [
            {
                "id": "node_0",
                "opType": "CrossSDPA",
                "inputs": inputs,
                "outputs": {
                    "out": {
                        "tensor": "attended",
                        "shape": [1, SEQUENCE, D_MODEL],
                        "dtype": "float32",
                    }
                },
                "params": params,
            },
        ],
        "outputs": ["attended"],
    }


def cross_sdpa_tensors() -> dict[str, np.ndarray]:
    return {}


def din_dout_document() -> dict[str, Any]:
    """A dense node whose weight is stored the other way round.

    Both layouts are legal in a source graph and the packed form is always
    [dout, din], so authoring has to read the declaration rather than guess
    from the extents. A fixture with unequal in and out widths is what makes
    guessing wrong observable — a square weight hides it.
    """

    return {
        "format": FORMAT,
        "dimensions": {},
        "inputs": {
            "hidden": {"shape": [1, SEQUENCE, D_MODEL], "dtype": "float32"},
        },
        "nodes": [
            {
                "id": "node_0",
                "opType": "Linear",
                "inputs": {"input": "hidden", "weight": "up.weight",
                           "bias": "up.bias"},
                "outputs": {
                    "out": {
                        "tensor": "wide",
                        "shape": [1, SEQUENCE, D_FF],
                        "dtype": "float32",
                    }
                },
                "params": {"weight_layout": "din_dout"},
            },
        ],
        "outputs": ["wide"],
    }


def din_dout_tensors() -> dict[str, np.ndarray]:
    """din_dout: [in, out], so the bias length is the second extent."""

    shapes = {"up.weight": (D_MODEL, D_FF), "up.bias": (D_FF,)}
    return {
        name: _values(name, int(np.prod(shape))).reshape(shape)
        for name, shape in shapes.items()
    }
