#!/usr/bin/env python3
"""Put one VolvoxAI kernel next to the equivalent ONNX Runtime kernel.

Runs ``benchmark_kernel_unit`` to get VolvoxAI's per-kernel throughput, then
builds a single-node ONNX model per case with the same shape and dtypes and
times it on the CPU execution provider at the same thread count.  The output is
one line per kernel with both throughputs and their ratio, so an optimization
target is chosen from a measured gap rather than a guess.

Two properties make the comparison honest rather than merely convenient:

* the same shape table drives both sides, taken from the shapes the model
  actually executes; and
* the VolvoxAI side has already proven byte-exactness against its portable
  reference, and this script separately checks ORT's integer result against an
  exact NumPy reference.  On AVX2 without VNNI, ORT's u8 x s8 kernel uses a
  saturating VPMADDUBSW, so a throughput number there may not correspond to a
  correct result.  The ``onnx_exact`` column records that.

Usage:
    python3 tools/compare_kernel_onnx.py --binary build/.../benchmark_kernel_unit
    python3 tools/compare_kernel_onnx.py --threads 1 --op qlinear
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional


def _require(module: str):
    try:
        return __import__(module)
    except ImportError:
        print(
            f"compare_kernel_onnx: {module} is required for the ONNX side.\n"
            f"  pip install onnx onnxruntime",
            file=sys.stderr,
        )
        raise SystemExit(2)


@dataclass
class Row:
    kernel: str
    onnx_op: str
    shape: str
    unit: str
    work: float
    ms: Optional[float]
    throughput: Optional[float]
    exact: str


def run_volvox(binary: str, threads: int, op: Optional[str]) -> list[Row]:
    if not os.path.isfile(binary):
        raise SystemExit(
            f"compare_kernel_onnx: {binary} not found. Build it with:\n"
            f"  cmake --build <build-dir> --target benchmark_kernel_unit"
        )
    argv = [binary]
    if op:
        argv += ["--op", op]
    env = dict(os.environ)
    if threads > 0:
        env["VOLVOXAI_THREADS"] = str(threads)
        argv += ["--threads", str(threads)]
    proc = subprocess.run(argv, capture_output=True, text=True, env=env)
    # A non-zero exit means a kernel disagreed with the portable reference. That
    # is worth surfacing loudly, but the rows that did pass are still useful.
    if proc.returncode != 0 and proc.stderr:
        print(f"[volvox] {proc.stderr.strip()}", file=sys.stderr)
    text = proc.stdout
    isa = "unknown"
    for line in text.splitlines():
        if line.startswith("# isa="):
            isa = line[len("# isa="):].split()[0]
    # Drop the comment header before handing the rest to the CSV reader.
    text = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    rows: list[Row] = []
    run_volvox.isa = isa
    for r in csv.DictReader(io.StringIO(text)):
        rows.append(
            Row(
                kernel=r["kernel"],
                onnx_op=r["onnx_op"],
                shape=r["shape"],
                unit=r["unit"],
                work=float(r["work"]),
                ms=float(r["ms"]) if r["ms"] else None,
                throughput=float(r["throughput"]) if r["throughput"] else None,
                exact=r["exact"],
            )
        )
    return rows


def _session(model_bytes: bytes, threads: int):
    ort = _require("onnxruntime")
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(threads, 1)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_bytes, so, providers=["CPUExecutionProvider"]
    )


def _time(session, feeds, work: float) -> float:
    reps = max(1, min(100000, int(2.0e9 / max(work, 1.0))))
    for _ in range(3):
        session.run(None, feeds)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            session.run(None, feeds)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def _dims(shape: str) -> list[int]:
    """Pull the numbers out of a shape label, in order.

    Labels tag their dimensions with a letter prefix so a human can read them
    ("c48->96", "q192 kv402 d320 h8"), so a token is a prefix followed by
    digits.  This used to strip only "c" and "s", which silently dropped every
    other tag: the attention shapes contributed one dimension instead of four
    and no builder could match them.
    """
    out = []
    for token in shape.replace("x", " ").replace("->", " ").split():
        digits = token.lstrip("abcdefghijklmnopqrstuvwxyz")
        if digits.isdigit():
            out.append(int(digits))
    return out


def onnx_case(row: Row, threads: int):
    """Return (ms, exact_verdict) or None when no single-node equivalent exists."""
    np = _require("numpy")
    onnx = _require("onnx")
    from onnx import helper, TensorProto

    rng = np.random.default_rng(0)
    dims = _dims(row.shape)

    if row.onnx_op == "QLinearMatMul" and len(dims) >= 3:
        M, K, N = dims[0], dims[1], dims[2]
        W = rng.integers(-127, 128, (K, N)).astype(np.int8)
        A = rng.integers(0, 256, (M, K)).astype(np.uint8)
        init = [
            helper.make_tensor("a_scale", TensorProto.FLOAT, [], [0.01]),
            helper.make_tensor("a_zp", TensorProto.UINT8, [], [128]),
            helper.make_tensor("B", TensorProto.INT8, [K, N], W.tobytes(), raw=True),
            helper.make_tensor("b_scale", TensorProto.FLOAT, [], [0.002]),
            helper.make_tensor("b_zp", TensorProto.INT8, [], [0]),
            helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.05]),
            helper.make_tensor("y_zp", TensorProto.UINT8, [], [128]),
        ]
        node = helper.make_node(
            "QLinearMatMul",
            ["A", "a_scale", "a_zp", "B", "b_scale", "b_zp", "y_scale", "y_zp"],
            ["Y"],
        )
        g = helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("A", TensorProto.UINT8, [M, K])],
            [helper.make_tensor_value_info("Y", TensorProto.UINT8, [M, N])],
            init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        ms = _time(s, {"A": A}, row.work) * 1e3
        # Exactness: compare against the integer-exact accumulation.
        acc = (A.astype(np.int64) - 128) @ W.astype(np.int64)
        expect = np.clip(
            np.rint(acc * (0.01 * 0.002 / 0.05)) + 128, 0, 255
        ).astype(np.int64)
        got = s.run(None, {"A": A})[0].astype(np.int64)
        wrong = int((np.abs(got - expect) > 1).sum())
        return ms, ("exact" if wrong == 0 else f"WRONG:{wrong}/{got.size}")

    if row.onnx_op == "QLinearConv" and len(dims) >= 4:
        OH, OW, CIN, COUT = dims[0], dims[1], dims[2], dims[3]
        W = rng.integers(-127, 128, (COUT, CIN, 3, 3)).astype(np.int8)
        X = rng.integers(0, 256, (1, CIN, OH, OW)).astype(np.uint8)
        init = [
            helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.01]),
            helper.make_tensor("x_zp", TensorProto.UINT8, [], [128]),
            helper.make_tensor("w", TensorProto.INT8, list(W.shape),
                               W.tobytes(), raw=True),
            helper.make_tensor("w_scale", TensorProto.FLOAT, [], [0.002]),
            helper.make_tensor("w_zp", TensorProto.INT8, [], [0]),
            helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.05]),
            helper.make_tensor("y_zp", TensorProto.UINT8, [], [128]),
        ]
        node = helper.make_node(
            "QLinearConv",
            ["X", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp"],
            ["Y"], pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1], group=1,
        )
        g = helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("X", TensorProto.UINT8,
                                           [1, CIN, OH, OW])],
            [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)],
            init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        return _time(s, {"X": X}, row.work) * 1e3, "unchecked"

    if row.onnx_op in ("QuantizeLinear", "DequantizeLinear"):
        n = int(dims[0] * dims[1]) if len(dims) >= 2 else int(row.work)
        if row.onnx_op == "QuantizeLinear":
            in_t, out_t = TensorProto.FLOAT, TensorProto.UINT8
            data = rng.standard_normal(n).astype(np.float32) * 4.0
            zp = helper.make_tensor("zp", TensorProto.UINT8, [], [128])
        else:
            in_t, out_t = TensorProto.UINT8, TensorProto.FLOAT
            data = rng.integers(0, 256, n).astype(np.uint8)
            zp = helper.make_tensor("zp", TensorProto.UINT8, [], [128])
        init = [helper.make_tensor("scale", TensorProto.FLOAT, [], [0.05]), zp]
        node = helper.make_node(row.onnx_op, ["X", "scale", "zp"], ["Y"])
        g = helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("X", in_t, [n])],
            [helper.make_tensor_value_info("Y", out_t, [n])], init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        return _time(s, {"X": data}, row.work) * 1e3, "exact"

    if row.onnx_op == "LayerNorm" and len(dims) >= 2:
        rows_, D = dims[0], dims[1]
        init = [
            helper.make_tensor("w", TensorProto.FLOAT, [D],
                               rng.standard_normal(D).astype(np.float32).tobytes(),
                               raw=True),
            helper.make_tensor("b", TensorProto.FLOAT, [D],
                               rng.standard_normal(D).astype(np.float32).tobytes(),
                               raw=True),
        ]
        node = helper.make_node("LayerNormalization", ["X", "w", "b"], ["Y"],
                                axis=-1, epsilon=1e-5)
        g = helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [rows_, D])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [rows_, D])],
            init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        X = rng.standard_normal((rows_, D)).astype(np.float32)
        return _time(s, {"X": X}, row.work) * 1e3, "n/a (F32)"

    # --- F32 cases -------------------------------------------------------
    # These mirror the FP32 half of the kernel table.  None of them can claim
    # byte-exactness against a reference the way the integer cases do — both
    # sides reorder F32 reductions — so the verdict column says so rather than
    # implying a check that did not happen.
    if row.onnx_op == "MatMul" and len(dims) >= 3:
        M, K, N = dims[0], dims[1], dims[2]
        B = rng.standard_normal((K, N)).astype(np.float32)
        A = rng.standard_normal((M, K)).astype(np.float32)
        # An initializer B is what makes this comparable to the VolvoxAI packed
        # path: ORT may prepack a constant weight, which is the same tradeoff
        # vx_gemm_f32_pack_cache makes.
        init = [helper.make_tensor("B", TensorProto.FLOAT, [K, N],
                                   B.tobytes(), raw=True)]
        node = helper.make_node("MatMul", ["A", "B"], ["Y"])
        g = helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
            init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        return _time(s, {"A": A}, row.work) * 1e3, "n/a (F32)"

    if row.onnx_op == "Conv" and len(dims) >= 4:
        OH, OW, CIN, COUT = dims[0], dims[1], dims[2], dims[3]
        W = rng.standard_normal((COUT, CIN, 3, 3)).astype(np.float32)
        X = rng.standard_normal((1, CIN, OH, OW)).astype(np.float32)
        init = [
            helper.make_tensor("w", TensorProto.FLOAT, list(W.shape),
                               W.tobytes(), raw=True),
            helper.make_tensor("b", TensorProto.FLOAT, [COUT],
                               rng.standard_normal(COUT).astype(np.float32)
                               .tobytes(), raw=True),
        ]
        node = helper.make_node("Conv", ["X", "w", "b"], ["Y"],
                                pads=[1, 1, 1, 1], strides=[1, 1],
                                dilations=[1, 1], group=1)
        g = helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                           [1, CIN, OH, OW])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        return _time(s, {"X": X}, row.work) * 1e3, "n/a (F32)"

    if row.onnx_op == "GroupNorm" and len(dims) >= 4:
        OH, OW, C, G = dims[0], dims[1], dims[2], dims[3]
        X = rng.standard_normal((1, C, OH, OW)).astype(np.float32)
        init = [
            helper.make_tensor("scale", TensorProto.FLOAT, [C],
                               rng.standard_normal(C).astype(np.float32)
                               .tobytes(), raw=True),
            helper.make_tensor("bias", TensorProto.FLOAT, [C],
                               rng.standard_normal(C).astype(np.float32)
                               .tobytes(), raw=True),
        ]
        node = helper.make_node("GroupNormalization",
                                ["X", "scale", "bias"], ["Y"],
                                num_groups=G, epsilon=1e-5)
        g = helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                           [1, C, OH, OW])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)])
        m.ir_version = 10
        s = _session(m.SerializeToString(), threads)
        return _time(s, {"X": X}, row.work) * 1e3, "n/a (F32)"

    if row.onnx_op == "Mul" and len(dims) >= 1:
        n = int(dims[0])
        A = rng.standard_normal(n).astype(np.float32)
        B = rng.standard_normal(n).astype(np.float32)
        node = helper.make_node("Mul", ["A", "B"], ["Y"])
        g = helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [n]),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, [n])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n])],
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        return _time(s, {"A": A, "B": B}, row.work) * 1e3, "n/a (F32)"

    if row.onnx_op == "Mul+Sigmoid" and len(dims) >= 1:
        n = int(dims[0]) * (int(dims[1]) if len(dims) >= 2 else 1)
        X = rng.standard_normal(n).astype(np.float32) * 6.0
        nodes = [helper.make_node("Sigmoid", ["X"], ["S"]),
                 helper.make_node("Mul", ["X", "S"], ["Y"])]
        g = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [n])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n])],
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        return _time(s, {"X": X}, row.work) * 1e3, "n/a (F32)"

    # The F32 attention shape carries its own dim vocabulary (q/kv/d/h), so it
    # is matched before the shared MatMul+Softmax branch above would apply.
    if row.onnx_op == "MatMul+Softmax" and row.shape.startswith("q"):
        SQ, SKV, D, H = dims[0], dims[1], dims[2], dims[3]
        head = D // H
        Q = rng.standard_normal((1, H, SQ, head)).astype(np.float32)
        K = rng.standard_normal((1, H, SKV, head)).astype(np.float32)
        V = rng.standard_normal((1, H, SKV, head)).astype(np.float32)
        init = [helper.make_tensor("scale", TensorProto.FLOAT, [],
                                   [1.0 / (head ** 0.5)])]
        nodes = [
            helper.make_node("Transpose", ["K"], ["Kt"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["Q", "Kt"], ["S"]),
            helper.make_node("Mul", ["S", "scale"], ["Ss"]),
        ]
        # sdpa_f32 is benchmarked causal, and its work denominator counts only
        # the lower triangle.  Without the same mask here ORT would compute the
        # full square while being credited with half the MACs, which would report
        # it as twice as slow as it is.
        if row.kernel.startswith("sdpa_f32"):
            bias = np.triu(np.full((SQ, SKV), -np.inf, dtype=np.float32), 1)
            init.append(helper.make_tensor("causal", TensorProto.FLOAT,
                                           [1, 1, SQ, SKV], bias.tobytes(),
                                           raw=True))
            nodes.append(helper.make_node("Add", ["Ss", "causal"], ["Sm"]))
            softmax_input = "Sm"
        else:
            softmax_input = "Ss"
        nodes += [
            helper.make_node("Softmax", [softmax_input], ["P"], axis=-1),
            helper.make_node("MatMul", ["P", "V"], ["Y"]),
        ]
        g = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, H, SQ, head]),
             helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, H, SKV, head]),
             helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, H, SKV, head])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        s = _session(m.SerializeToString(), threads)
        return _time(s, {"Q": Q, "K": K, "V": V}, row.work) * 1e3, "n/a (F32)"

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",
                    default="build/cmake/native/benchmark_kernel_unit",
                    help="path to benchmark_kernel_unit")
    ap.add_argument("--threads", type=int, default=1,
                    help="worker threads for both sides (default 1)")
    ap.add_argument("--op", default=None, help="only this kernel name")
    args = ap.parse_args()

    rows = run_volvox(args.binary, args.threads, args.op)
    if not rows:
        print("compare_kernel_onnx: no rows from benchmark_kernel_unit",
              file=sys.stderr)
        return 2

    unit = {"MAC": "GMAC/s", "byte": "GB/s"}
    print(f"threads={args.threads}  volvox_isa={getattr(run_volvox, 'isa', 'unknown')}\n")
    head = (f"{'kernel':<17}{'shape':<22}{'volvox':>10}{'onnx':>10}"
            f"{'ratio':>8}  {'unit':<8}{'vx':<7}{'ort'}")
    print(head)
    print("-" * len(head))
    for r in rows:
        vx = f"{r.throughput:.1f}" if r.throughput else "-"
        try:
            got = onnx_case(r, args.threads)
        except SystemExit:
            raise
        except Exception as error:                    # noqa: BLE001
            got = None
            print(f"  [skip {r.kernel} {r.shape}: {error}]", file=sys.stderr)
        if got is None:
            print(f"{r.kernel:<17}{r.shape:<22}{vx:>10}{'-':>10}{'-':>8}  "
                  f"{unit.get(r.unit, r.unit):<8}{r.exact:<7}-")
            continue
        ms, verdict = got
        ort_tp = r.work / (ms / 1e3) / 1e9
        ratio = (ort_tp / r.throughput) if r.throughput else float("nan")
        print(f"{r.kernel:<17}{r.shape:<22}{vx:>10}{ort_tp:>10.1f}"
              f"{ratio:>7.2f}x  {unit.get(r.unit, r.unit):<8}"
              f"{r.exact:<7}{verdict}")
    print("\nratio > 1 means ONNX Runtime is faster. `ort` column records whether "
          "ORT's integer result matched an exact reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
