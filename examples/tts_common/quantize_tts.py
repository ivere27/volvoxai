#!/usr/bin/env python3
"""Prepare explicitly scoped quantized variants of an existing complete TTS package.

No model downloads or original exports are repeated. Static calibration uses the
prepared requests, including live Volvox stage outputs, only during authoring.
The resulting manifest still executes every stage on every inference request.
Calibration and evaluation share these few inputs: this is a performance and
numerical diagnostic, not representative speech-quality qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
from examples.tts_common.artifacts import sha256
from examples.tts_common.benchmark_native import NativePipeline, load_npz


def absolute_manifest(manifest, root):
    """Keep shared input/reference/model files in place when writing a variant."""
    result = deepcopy(manifest)
    result["onnx"]["path"] = str((root / result["onnx"]["path"]).resolve())
    if "external_data" in result["onnx"]:
        result["onnx"]["external_data"] = [str((root / name).resolve())
                                             for name in result["onnx"]["external_data"]]
    for case in result["cases"]:
        for key in ("inputs", "reference"):
            case[key] = str((root / case[key]).resolve())
        for key in ("input_files", "output_files"):
            for item in case.get(key, []):
                item["path"] = str((root / item["path"]).resolve())
    for stage in result.get("volvox", {}).get("pipeline", []):
        stage["manifest"] = str((root / stage["manifest"]).resolve())
    return result


def onnx_inventory(model_path):
    import onnx
    model = onnx.load(str(model_path), load_external_data=False)
    types = Counter(onnx.TensorProto.DataType.Name(tensor.data_type)
                    for tensor in model.graph.initializer)
    elements = Counter()
    for tensor in model.graph.initializer:
        elements[onnx.TensorProto.DataType.Name(tensor.data_type)] += int(np.prod(tensor.dims))
    return {"path": str(model_path), "sha256": sha256(model_path),
            "bytes": model_path.stat().st_size,
            "operators": dict(Counter(f"{node.domain or 'ai.onnx'}:{node.op_type}"
                                      for node in model.graph.node)),
            "initializer_types": dict(types), "initializer_elements": dict(elements)}


def weight_inventory(paths, names=None):
    """Count physical payload bytes by storage dtype, without loading weights."""
    counts = Counter()
    for path in paths:
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            tensors = json.loads(stream.read(length))
        for name, tensor in tensors.items():
            if name != "__metadata__" and (names is None or name in names):
                start, end = tensor["data_offsets"]
                counts[tensor["dtype"]] += end - start
    return dict(counts)


def prepare_onnx(manifest, destination, scheme, include_lstm, existing_model, node_prefixes):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
        QuantFormat, QuantType, quantize_dynamic, quantize_static)

    source = Path(manifest["onnx"]["path"])
    before = onnx_inventory(source)
    output = destination / "model.onnx"
    if existing_model is not None:
        output = existing_model.resolve()
        policy = {"scheme": "published", "source": str(output),
                  "note": "Published artifact; inventory determines its actual operator coverage."}
    elif scheme == "dynamic-int8":
        operators = ["MatMul", *(["LSTM"] if include_lstm else [])]
        quantize_dynamic(source, output, op_types_to_quantize=operators,
                         weight_type=QuantType.QInt8, per_channel=True,
                         extra_options={"MatMulConstBOnly": True})
        policy = {"scheme": scheme, "operators": operators, "weights": "S8 per channel",
                  "activations": "dynamic U8", "calibration": "none"}
    else:
        class PreparedRequests(CalibrationDataReader):
            def __init__(self):
                self.requests = iter(manifest["cases"])

            def get_next(self):
                case = next(self.requests, None)
                if case is None:
                    return None
                values = load_npz(case["inputs"])
                return {manifest["onnx"]["inputs"][name]: value for name, value in values.items()}

        operators = ["Conv", "MatMul", "Gemm"]
        selected = None
        if node_prefixes:
            graph = onnx.load(str(source), load_external_data=False).graph
            selected = [node.name for node in graph.node if node.op_type in operators
                        and node.name.startswith(tuple(node_prefixes))]
            if not selected:
                raise ValueError("--onnx-node-prefix selected no eligible node")
        quantize_static(source, output, PreparedRequests(), quant_format=QuantFormat.QDQ,
                        op_types_to_quantize=operators, per_channel=True,
                        nodes_to_quantize=selected,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax,
                        calibration_providers=["CPUExecutionProvider"],
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                       "CalibTensorRangeSymmetric": True})
        policy = {"scheme": scheme, "operators": operators, "weights": "S8 per channel",
                  "activations": "static S8 symmetric min/max", "calibration": "prepared requests",
                  "calibration_cases": len(manifest["cases"]),
                  "node_prefixes": node_prefixes, "selected_nodes": selected}
    manifest["onnx"]["path"] = str(output.resolve())
    manifest["onnx"].pop("external_data", None)
    # Only the ORT artifact changed; do not imply the inherited Volvox package
    # represents the same quantized operator graph.
    manifest["volvox"] = {"status": "unsupported",
                           "reason": "ORT quantized variant was not exported to VolvoxAI"}
    return {"engine": "ort", "version": ort.__version__, "policy": policy,
            "source": before, "result": onnx_inventory(output)}


def prepare_volvox(manifest, destination, scheme, threads, library, selected_stages):
    import volvoxai as vx
    from google.protobuf.json_format import MessageToDict

    if library:
        os.environ["VOLVOXAI_LIBRARY"] = str(library.resolve())
    stages = manifest["volvox"]["pipeline"]
    if selected_stages and not set(selected_stages).issubset(entry["name"] for entry in stages):
        raise ValueError("--stages must name existing pipeline stages")
    samples = {entry["name"]: [] for entry in stages}
    if scheme == "static-int8":
        runner = NativePipeline(destination, manifest["volvox"], "cpu", threads, library, 300)
        try:
            for case in manifest["cases"]:
                values = load_npz(case["inputs"])
                for name, engine in runner.stages:
                    engine.prepare({key: values[key] for key in engine.contract["inputs"]})
                    samples[name].append({key: value.copy() for key, value in engine.arrays.items()})
                    outputs, _ = engine.run()
                    values.update(outputs)
        finally:
            runner.close()

    rows = []
    for entry in stages:
        source_manifest = Path(entry["manifest"])
        stage = json.loads(source_manifest.read_text())
        contract = stage["volvox"]
        graph_path = (source_manifest.parent / contract["graph"]).resolve()
        weight_paths = [(source_manifest.parent / name).resolve() for name in contract["weights"]]
        graph = json.loads(graph_path.read_text())
        counts = Counter(node["opType"] for node in graph["nodes"])
        eligible = ("Linear", "MatMul", "Conv2D") if scheme == "static-int8" else ("Linear", "MatMul")
        row = {"name": entry["name"], "source_operators": dict(counts),
               "source_weight_bytes": sum(path.stat().st_size for path in weight_paths),
               "source_storage_bytes": weight_inventory(weight_paths),
               "source_graph_sha256": sha256(graph_path),
               "source_weights_sha256": [sha256(path) for path in weight_paths]}
        if (selected_stages and entry["name"] not in selected_stages) or not any(counts[op] for op in eligible):
            row.update(status="retained_float", reason="stage not selected or no eligible immutable dense weights",
                       result_weight_bytes=row["source_weight_bytes"])
            rows.append(row)
            continue
        options = {"weight_bits": 8, "group_size": 128, "algorithm": "rtn"}
        if scheme == "static-int8":
            policy = vx.pb.WeightQuantization(
                format=vx.pb.WeightFormat(bits=8, group_size=0),
                algorithm=vx.pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN)
            unsupported = [node for node in graph["nodes"] if node["opType"] == "Conv2D"
                           and node.get("params", {}).get("weight_layout") not in ("HWIO", "OHWI")]
            row["retained_conv_layouts"] = [{"node": node["id"],
                "layout": node.get("params", {}).get("weight_layout")} for node in unsupported]
            rules = [vx.pb.QuantizationRule(node_id=node["id"], keep_float=vx.pb.Empty())
                     for node in unsupported]
            rules += [vx.pb.QuantizationRule(operator_name=op, weights=policy)
                     for op in eligible if counts[op]]
            rules.append(vx.pb.QuantizationRule(node_glob="*", keep_float=vx.pb.Empty()))
            options.update(group_size=0, rules=rules,
                static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8,
                    scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC),
                calibration_data=samples[entry["name"]], cpu_threads=threads)
        output = destination / entry["name"]
        package = vx.quantize(graph_path, weights=weight_paths, output=output, **options)
        result_graph = json.loads(package.graph_path.read_text())
        quantized = [node for node in result_graph["nodes"]
                     if node["opType"] in ("PackedLinear", "QLinear", "QMatMul", "QConv2D")]
        quantized_ids = {node["id"] for node in quantized}
        original_weights = {node["inputs"]["weight"] for node in graph["nodes"]
                            if node["id"] in quantized_ids and "weight" in node["inputs"]}
        packed_weights = {node["inputs"]["weight"] for node in quantized if "weight" in node["inputs"]}
        contract["graph"] = str(package.graph_path.resolve())
        contract["weights"] = [str(path.resolve()) for path in package.weight_paths]
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(stage, indent=2) + "\n")
        entry["manifest"] = str(manifest_path.resolve())
        row.update(status="quantized", quantized_nodes=package.quantized_nodes,
                   retained_float_nodes=package.retained_float_nodes,
                   result_operators=dict(Counter(node["opType"] for node in result_graph["nodes"])),
                   result_weight_bytes=sum(path.stat().st_size for path in package.weight_paths),
                   result_storage_bytes=weight_inventory(package.weight_paths),
                   quantized_original_weight_bytes=sum(weight_inventory(weight_paths, original_weights).values()),
                   quantized_code_bytes=sum(weight_inventory(package.weight_paths, packed_weights).values()),
                   result_weights_sha256=[sha256(path) for path in package.weight_paths],
                   calibration_batches=package.calibration_batches,
                   calibration_backend=package.calibration_backend,
                   report=MessageToDict(package.report))
        rows.append(row)
    return {"engine": "volvox", "version": vx.__version__, "stages": rows,
            "policy": {"scheme": scheme, "weight_bits": 8,
                "group_size": 0 if scheme == "static-int8" else 128,
                "activations": "static S8 symmetric min/max" if scheme == "static-int8" else "F32",
                "packed_arithmetic": "not_applicable" if scheme == "static-int8" else "F32",
                "calibration_cases": len(manifest["cases"]) if scheme == "static-int8" else 0,
                "selected_stages": selected_stages or [entry["name"] for entry in stages],
                "LSTM": "retained FP32", "all_stages_execute_each_request": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=("ort", "volvox"), required=True)
    parser.add_argument("--scheme", choices=("dynamic-int8", "static-int8", "packed-int8", "published"), required=True)
    parser.add_argument("--include-lstm", action="store_true")
    parser.add_argument("--onnx-model", type=Path, help="Use an existing published ONNX quantized artifact")
    parser.add_argument("--onnx-node-prefix", action="append", default=[],
                        help="Restrict ORT static quantization by original ONNX node-name prefix")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--stages", nargs="*", help="Volvox stages to quantize; all other stages remain live FP32")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0 or (args.engine == "ort" and args.scheme == "packed-int8") or (
            args.engine == "volvox" and args.scheme in ("dynamic-int8", "published")):
        parser.error("positive threads and an engine-compatible quantization scheme are required")
    if bool(args.onnx_model) != (args.scheme == "published"):
        parser.error("--onnx-model is required exactly for --scheme published")
    if args.onnx_node_prefix and (args.engine != "ort" or args.scheme != "static-int8"):
        parser.error("--onnx-node-prefix applies only to ORT static INT8")
    if (args.include_lstm and (args.engine != "ort" or args.scheme != "dynamic-int8")) or (
            args.onnx_model and args.engine != "ort") or (args.stages and args.engine != "volvox"):
        parser.error("LSTM dynamic quantization and published ONNX artifacts are ORT-only")
    destination = args.out_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = args.manifest.resolve()
    manifest = absolute_manifest(json.loads(source.read_text()), source.parent)
    report = {"format": "volvoxai-tts-quantization/v1", "status": "failed",
              "source_manifest": str(source), "source_manifest_sha256": sha256(source),
              "calibration_limit": "Prepared evaluation requests only; no held-out quality claim",
              "shape_contract": "Original admitted frame extent retained; changed duration must be reported as refused"}
    try:
        if args.engine == "ort":
            report.update(prepare_onnx(manifest, destination, args.scheme, args.include_lstm,
                                       args.onnx_model, args.onnx_node_prefix))
        else:
            report.update(prepare_volvox(manifest, destination, args.scheme, args.threads, args.library, args.stages))
        manifest["quantization"] = report["policy"]
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report["status"] = "prepared"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (destination / "quantization.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
