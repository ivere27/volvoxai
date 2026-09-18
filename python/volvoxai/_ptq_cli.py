"""File-oriented PTQ workflow; all quantization runs in generated C dispatch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import FfiError, VolvoxAIError
from ._quantization import quantize


def _batches(paths):
    for index, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as batch:
            if len(batch.files) != len(set(batch.files)):
                raise ValueError(f"{path.name}: duplicate input names")
            yield {name: batch[name] for name in batch.files}
        print(f"Calibrated {index}/{len(paths)} batches", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="volvoxai ptq",
        description=(
            "Quantize an exported FP32 package using representative model inputs. "
            "Each .npz file is one complete inference batch, keyed by input name."
        ),
    )
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--weights", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Directory of .npz calibration batches")
    parser.add_argument("--out", type=Path, required=True,
                        help="Destination directory for graph.json and model.safetensors")
    parser.add_argument("--samples-per-batch", type=int, default=1,
                        help="Logical examples represented by each .npz file (default: 1)")
    parser.add_argument("--activation-dtype", choices=("int8", "uint8"), default="int8")
    parser.add_argument("--activation-scheme", choices=("symmetric", "asymmetric"),
                        default="symmetric")
    parser.add_argument("--float-operator", action="append", default=[])
    parser.add_argument("--float-node", action="append", default=[])
    parser.add_argument("--select-node", action="append", default=[])
    parser.add_argument("--reduce-range", action="store_true")
    args = parser.parse_args(argv)
    if args.samples_per_batch < 1:
        parser.error("--samples-per-batch must be positive")
    for path in (args.graph, *args.weights):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    batches = sorted(args.calibration.glob("*.npz"))
    if not args.calibration.is_dir() or not batches:
        parser.error("--calibration must contain at least one .npz batch")

    try:
        result = quantize(args.graph, weights=args.weights,
            calibration_data=_batches(batches), output=args.out,
            samples_per_batch=args.samples_per_batch, activation_dtype=args.activation_dtype,
            activation_scheme=args.activation_scheme, float_operators=args.float_operator,
            float_nodes=args.float_node, selected_nodes=args.select_node,
            reduce_range=args.reduce_range)
        print(json.dumps({
            "graph": str(result.graph_path), "weights": str(result.weight_paths[0]),
            "quantized_nodes": result.quantized_nodes,
            "retained_float_nodes": result.retained_float_nodes,
            "calibration_batches": result.calibration_batches,
            "calibration_samples": result.calibration_samples,
        }))
        return 0
    except (OSError, TypeError, ValueError, FfiError, VolvoxAIError) as error:
        print(f"PTQ failed: {error}", file=sys.stderr)
        return 1
