"""File-oriented quantization over the generated C plan lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from google.protobuf.json_format import ParseDict, ParseError
import volvoxai_pb2 as pb

from . import FfiError, VolvoxAIError
from ._quantization import WEIGHT_ALGORITHMS, quantize
from ._ptq_sensitivity import ptq_sensitivity


def _batches(paths, label="Calibrated"):
    for index, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as batch:
            if len(batch.files) != len(set(batch.files)):
                raise ValueError(f"{path.name}: duplicate input names")
            yield {name: batch[name] for name in batch.files}
        print(f"{label} {index}/{len(paths)} batches", file=sys.stderr)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    sensitivity = bool(arguments and arguments[0] == "sensitivity")
    if sensitivity:
        arguments.pop(0)
    parser = argparse.ArgumentParser(
        prog="volvoxai ptq sensitivity" if sensitivity else "volvoxai ptq",
        description="Measure quantization sensitivity." if sensitivity else "Quantize model weight storage.")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--weights", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="New JSON report" if sensitivity else "New output package directory")
    parser.add_argument("--weight-bits", type=int, choices=(2, 3, 4, 8), default=8)
    parser.add_argument("--group-size", type=int, choices=(0, 32, 64, 128, 256), default=128)
    parser.add_argument("--algorithm", choices=tuple(WEIGHT_ALGORITHMS), default="rtn")
    parser.add_argument("--static-activations", choices=("i8", "u8"))
    parser.add_argument("--affine-scheme", choices=("symmetric", "asymmetric"))
    parser.add_argument("--reduce-weight-range", action="store_true")
    parser.add_argument("--rules", type=Path, help="JSON array of generated QuantizationRule messages")
    parser.add_argument("--calibration", type=Path, help="Directory of representative .npz input batches")
    parser.add_argument("--samples-per-batch", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--calibration-backend", choices=("cpu", "cuda"), default="cpu",
                        help="Exact observation backend; exported model evaluation is separate")
    if sensitivity:
        parser.add_argument("--evaluation", type=Path, required=True)
        parser.add_argument("--mode", action="append", choices=("isolated", "leave-one-float", "leave-one-higher"))
        parser.add_argument("--higher-weight-bits", type=int, choices=(3, 4, 8), default=4)
        parser.add_argument("--output", action="append", help="F32 graph output to compare")
        parser.add_argument("--packed-arithmetic", choices=("dynamic-i8", "f16", "f32", "f32-fast"), default="f32")
    args = parser.parse_args(arguments)
    if args.samples_per_batch < 1:
        parser.error("--samples-per-batch must be positive")
    for path in (args.graph, *args.weights):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    if args.static_activations is None and (args.affine_scheme or args.reduce_weight_range):
        parser.error("--affine-scheme and --reduce-weight-range require --static-activations")
    calibration = sorted(args.calibration.glob("*.npz")) if args.calibration is not None else None
    if args.calibration is not None and (not args.calibration.is_dir() or not calibration):
        parser.error("--calibration must contain at least one .npz batch")
    if sensitivity:
        evaluation = sorted(args.evaluation.glob("*.npz"))
        if not args.evaluation.is_dir() or not evaluation:
            parser.error("--evaluation must contain at least one .npz batch")
        if args.out.exists():
            parser.error(f"report destination already exists: {args.out}")
    try:
        rules = []
        if args.rules is not None:
            values = json.loads(args.rules.read_text())
            if not isinstance(values, list):
                raise ValueError("--rules must contain a JSON array")
            rules = [ParseDict(value, pb.QuantizationRule()) for value in values]
        static = (pb.StaticActivations(
            dtype=pb.DataType.Value("DATA_TYPE_" + args.static_activations.upper()),
            scheme=pb.AffineScheme.Value("AFFINE_SCHEME_" + (args.affine_scheme or "symmetric").upper()),
            reduce_weight_range=args.reduce_weight_range) if args.static_activations else None)
        options = dict(weights=args.weights, weight_bits=args.weight_bits, group_size=args.group_size,
            algorithm=args.algorithm, rules=rules, static_activations=static,
            samples_per_batch=args.samples_per_batch, cpu_threads=args.cpu_threads,
            calibration_backend=args.calibration_backend,
            calibration_data=_batches(calibration) if calibration is not None else None)
        if sensitivity:
            report = ptq_sensitivity(args.graph, evaluation_data=_batches(evaluation, "Read evaluation"),
                output_names=args.output, modes=args.mode or ("isolated", "leave-one-float"),
                packed_arithmetic=args.packed_arithmetic, higher_weight_bits=args.higher_weight_bits, **options)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as file:
                json.dump(report, file, indent=2, allow_nan=False)
                file.write("\n")
            print(json.dumps({"report": str(args.out), "mostSensitive": report["mostSensitive"],
                              "bestToKeepFloat": report["bestToKeepFloat"]}))
        else:
            result = quantize(args.graph, output=args.out, **options)
            print(json.dumps({"graph": str(result.graph_path), "weights": str(result.weight_paths[0]),
                "quantized_nodes": result.quantized_nodes, "retained_float_nodes": result.retained_float_nodes,
                "requires_calibration": result.requires_calibration,
                "calibration_batches": result.calibration_batches, "calibration_samples": result.calibration_samples,
                "calibration_backend": result.calibration_backend}))
    except (ValueError, TypeError, OSError, ParseError, FfiError, VolvoxAIError) as error:
        parser.error(str(error))
    return 0
