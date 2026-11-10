#!/usr/bin/env python3
"""Compare complete-request profiles without equating differently fused nodes.

This is offline analysis. It never runs a model or changes graph/weight files.
ORT *_kernel_time events are host intervals; Volvox device program intervals
remain a separate measurement domain. Source-region attribution requires an
exporter's node_sources report for the exact prepared graph.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path)
    return {"file": path.name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def runtime_source_fingerprint(path):
    # The trace protocol's source fingerprint (public_api.c vx_revision_update).
    # SHA256 receipts are recorded separately; this is an identity join only.
    value = 1469598103934665603
    for byte in Path(path).read_bytes():
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"volvox-graph-source/v1:{value or 1:016x}"


def aggregate_volvox(events, domain, layer):
    rows = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    for event in events:
        kind = "program" if "program" in event else "node" if "node" in event else "operation"
        if domain not in event or kind != layer:
            continue
        key = (event.get("name", ""), event.get("activity", ""))
        row = rows[key]
        row["count"] += 1
        row["total_ms"] += int(event[domain].get("elapsedNs" if domain == "device" else "durationNs", 0)) / 1e6
    return sorted(({"name": key[0], "activity": key[1], **row} for key, row in rows.items()),
                  key=lambda row: -row["total_ms"])


def summarize_ort_events(events):
    """Summarize raw ORT node host intervals, excluding fences and sessions."""
    rows = defaultdict(lambda: {"count": 0, "total_host_ms": 0.0})
    gpu_events = []
    for event in events:
        if event.get("cat") == "Kernel":
            gpu_events.append(event)
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        args = event.get("args", {})
        key = (args.get("provider", "unknown"), args.get("op_name", "unknown"))
        rows[key]["count"] += 1
        rows[key]["total_host_ms"] += float(event.get("dur", 0)) / 1000
    return {
        "operator_provider_host_intervals": sorted(
            ({"provider": key[0], "operator": key[1], **row} for key, row in rows.items()),
            key=lambda row: -row["total_host_ms"]),
        "raw_kernel_category_event_count": len(gpu_events),
        "timing_domain": "host wall intervals; CUDA provider intervals are not GPU kernel elapsed time",
        "run_scope": "ALL raw runs, including first and warmups; use profile_ort.py summaries for measured-only comparison",
        "raw_kernel_category_note": "Kernel-category records require separate capture-specific clock/device validation before device-time comparison.",
    }


def match_ort_regions(node_rows, source_regions):
    """Join retained source names exactly; never guess through ORT fusions."""
    result = []
    for region, names in source_regions.items():
        if not names:
            continue
        matched = [row for row in node_rows if row.get("node") in names]
        observed = {row["node"] for row in matched}
        providers = defaultdict(list)
        for row in matched:
            providers[row["provider"]].append(row)
        provider_timings = []
        for provider, rows in providers.items():
            lengths = {len(row["samples_per_run_host_us"]) for row in rows}
            if len(lengths) != 1:
                raise ValueError("ORT node rows do not share the same measured-run population")
            samples = [sum(row["samples_per_run_host_us"][index] for row in rows) / 1000
                       for index in range(next(iter(lengths)))]
            provider_timings.append({"provider": provider, "samples_per_run_host_ms": samples,
                                     "mean_host_ms": statistics.mean(samples) if samples else None,
                                     "median_host_ms": statistics.median(samples) if samples else None,
                                     "operators": dict(Counter(row["op"] for row in rows))})
        result.append({"region": region, "expected_source_names": len(names),
                       "matched_source_names": len(observed), "matched_node_rows": len(matched),
                       "unmatched_source_names": sorted(set(names) - observed),
                       "providers": provider_timings,
                       "warning": "Exact retained names only. Optimized fused/renamed nodes are not guessed; missing names do not mean zero work. Timings include each matched ORT kernel's full work, which can include fused operations."})
    return result


def centered_normalization_regions(model):
    """Recognize connected centered-variance cores without relying on names.

    Affine scale/bias and surrounding layouts are excluded because they can
    branch and be shared. This is attribution, not a fusion or graph rewrite.
    """
    import onnx
    import numpy as np
    nodes = list(model.graph.node)
    producer = {name: node for node in nodes for name in node.output}
    constants = {tensor.name: tensor for tensor in model.graph.initializer}
    names = {id(node): node.name or f"{node.op_type}[{index}]" for index, node in enumerate(nodes)}

    def scalar(name):
        tensor = constants.get(name)
        if tensor is None or any(d != 1 for d in tensor.dims):
            return None
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            return None
        return float(np.asarray(onnx.numpy_helper.to_array(tensor)).reshape(-1)[0])

    def reduction_key(node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        axes = attrs.get("axes")
        if len(node.input) > 1:
            tensor = constants.get(node.input[1])
            if tensor is None or tensor.data_location == onnx.TensorProto.EXTERNAL:
                return None
            axes = onnx.numpy_helper.to_array(tensor).tolist()
        return tuple(axes) if axes is not None else None, attrs.get("keepdims", 1)

    regions = []
    for div in nodes:
        if div.op_type != "Div":
            continue
        sub, sqrt = (producer.get(name) for name in div.input)
        if sub is None or sub.op_type != "Sub" or sqrt is None or sqrt.op_type != "Sqrt":
            continue
        mean = producer.get(sub.input[1])
        add = producer.get(sqrt.input[0])
        if mean is None or mean.op_type != "ReduceMean" or mean.input[0] != sub.input[0] or add is None or add.op_type != "Add":
            continue
        variance, epsilon = None, None
        for value, constant in [add.input, reversed(add.input)]:
            candidate = producer.get(value)
            if candidate is not None and candidate.op_type == "ReduceMean":
                variance, epsilon = candidate, scalar(constant)
                break
        if variance is None or epsilon is None or not np.isfinite(epsilon) or epsilon < 0 or reduction_key(mean) is None or reduction_key(mean) != reduction_key(variance):
            continue
        square = producer.get(variance.input[0])
        if square is None or not ((square.op_type == "Mul" and list(square.input) == [sub.output[0]] * 2) or
            (square.op_type == "Pow" and square.input[0] == sub.output[0] and scalar(square.input[1]) == 2)):
            continue
        core = [mean, sub, square, variance, add, sqrt, div]
        regions.append({"output": div.output[0], "epsilon": epsilon,
                        "source_nodes": [names[id(node)] for node in core],
                        "operators": dict(Counter(node.op_type for node in core))})
    return regions


def source_inventory(path, reference_geometry=()):
    import onnx
    model = onnx.load(path, load_external_data=False)
    shapes = {value.name: [d.dim_value if d.HasField("dim_value") else d.dim_param
                          for d in value.type.tensor_type.shape.dim]
              for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    shapes.update({value.name: list(value.dims) for value in model.graph.initializer})
    rows = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Conv", "ConvTranspose", "LSTM", "MatMul", "Gemm",
                                "InstanceNormalization", "LayerNormalization", "STFT"}:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        attrs = {k: v.decode() if isinstance(v, bytes) else v for k, v in attrs.items()
                 if k in {"group", "strides", "pads", "dilations", "kernel_shape", "hidden_size",
                          "direction", "layout", "transA", "transB", "axis", "epsilon"}}
        row = {"name": node.name or f"{node.op_type}[{index}]", "operator": node.op_type,
               "input_names": list(node.input), "output_names": list(node.output),
               "inputs": [shapes.get(name) for name in node.input],
               "outputs": [shapes.get(name) for name in node.output], "attributes": attrs}
        # This counts the dense, untrimmed transposed-convolution contribution
        # extent, not actual executed multiply instructions or elapsed time.
        if node.op_type == "ConvTranspose":
            x, weight = shapes.get(node.input[0]), shapes.get(node.input[1])
            if x and weight and len(x) == len(weight) == 3 and all(isinstance(v, int) and v > 0 for v in [*x, *weight]):
                row["dense_untrimmed_macs"] = x[0] * x[2] * weight[0] * weight[1] * weight[2]
        rows.append(row)
    producers = {name: (index, node) for index, node in enumerate(model.graph.node) for name in node.output}
    transpose_regions = []
    for reference in reference_geometry:
        if reference["operator"] != "ConvTranspose":
            continue
        found = [producers[name] for name in reference["output_names"] if name in producers]
        if len(found) != 1:
            continue
        index, node = found[0]
        members = [(index, node)]
        kind = "direct"
        if node.op_type == "Conv" and reference["attributes"].get("group", 1) > 1:
            # The application's grouped zero-insert lowering retains the
            # original output and is Gather -> Mul(mask) -> depthwise Conv.
            mul = producers.get(node.input[0])
            gather = producers.get(mul[1].input[0]) if mul and mul[1].op_type == "Mul" else None
            if not gather or gather[1].op_type != "Gather":
                continue
            members = [gather, mul, (index, node)]
            kind = "grouped_zero_insert"
        elif node.op_type != "ConvTranspose":
            continue
        transpose_regions.append({"reference_node": reference["name"], "lowering": kind,
                                  "source_nodes": [n.name or f"{n.op_type}[{i}]" for i, n in members]})
    return {"artifact": receipt(path), "operators": dict(Counter(n.op_type for n in model.graph.node)),
            "selected_geometry": rows, "centered_normalization_cores": centered_normalization_regions(model),
            "reference_convtranspose_regions": transpose_regions}


def region_attribution(stage, graph, provenance, named_source_regions=None):
    sources = {row["id"]: row for row in provenance.get("node_sources", [])}
    runtime = {node["id"]: node for node in graph["nodes"]}
    grouped = defaultdict(list)
    selected_ops = {"Conv", "ConvTranspose", "LSTM", "MatMul", "Gemm",
                    "InstanceNormalization", "LayerNormalization"}
    for node_id, row in sources.items():
        if node_id in runtime and row["source_op"] in selected_ops:
            grouped[row["source_op"]].append(node_id)
    for name, source_names in (named_source_regions or {}).items():
        grouped[name] = [node_id for node_id, row in sources.items()
                         if node_id in runtime and row["source_node"] in source_names]
    result = []
    for source_op, ids in grouped.items():
        if not ids:
            continue
        selected_ids = set(ids)
        steps = [step for plan in stage.get("plans", []) for step in plan.get("steps", [])
                 if selected_ids.intersection(step.get("sourceNodeIds", []))]
        indices = {int(step.get("scheduleIndex", 0)) for step in steps}
        events = [event for event in stage.get("events", []) if "node" in event
                  and int(event["node"].get("scheduleIndex", 0)) in indices]
        timings = {domain + "_" + layer: aggregate_volvox(events, domain, layer)
                   for domain, layer in [("host", "node"), ("device", "program")]}
        host_indices = {int(event["node"].get("scheduleIndex", 0)) for event in events
                        if "host" in event and "program" not in event}
        device_indices = {int(event["node"].get("scheduleIndex", 0)) for event in events
                          if "device" in event and "program" in event}
        result.append({"source_operator": source_op,
            "source_node_count": len({sources[node_id]["source_node"] for node_id in ids}),
            "runtime_node_count": len(ids),
            "runtime_operators": dict(Counter(runtime[node_id]["opType"] for node_id in ids)),
            "non_skipped_plan_steps": sum(not step.get("skipped") for step in steps),
            "host_observed_schedule_steps": len(host_indices),
            "device_program_observed_schedule_steps": len(device_indices),
            "declared_multiply_accumulates": sum(int(step.get("cost", {}).get("multiplyAccumulates", 0))
                for step in steps if not step.get("skipped")),
            "declared_output_bytes": sum(int(step.get("cost", {}).get("outputBytes", 0))
                for step in steps if not step.get("skipped")),
            "timings": timings,
            "total_host_node_ms": sum(row["total_ms"] for row in timings["host_node"]) if timings["host_node"] else None,
            "total_device_program_ms": sum(row["total_ms"] for row in timings["device_program"]) if timings["device_program"] else None,
            "warning": "Source region includes layout/broadcast/crop operations inserted by lowering. Missing device records are unknown, not zero. Static MAC/byte extents are not measured instructions, traffic or speedups."})
    return result


def compare(manifest_path, volvox_paths, ort_paths=(), provenance_paths=None):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    reference_inventory = source_inventory(manifest_path.parent / manifest["onnx"]["path"])
    packages = {}
    for entry in manifest["volvox"]["pipeline"]:
        stage_path = manifest_path.parent / entry["manifest"]
        stage = read_json(stage_path)
        graph_path = stage_path.parent / stage["volvox"]["graph"]
        graph = read_json(graph_path)
        provenance_path = (provenance_paths or {}).get(entry["name"], stage_path.parent / "export-report.json")
        provenance = read_json(provenance_path) if Path(provenance_path).exists() else {}
        named_regions = {}
        if "publication_report" in provenance:
            if not provenance.get("graph_verified_equal") or provenance.get("graph_sha256") != receipt(graph_path)["sha256"]:
                raise ValueError(f"Provenance does not identify {entry['name']} prepared graph")
            if provenance.get("stft_source_nodes"):
                named_regions["application_STFT_lowering"] = set(provenance["stft_source_nodes"])
            provenance = provenance["publication_report"]
        source = source_inventory(stage_path.parent / stage["onnx"]["path"], reference_inventory["selected_geometry"])
        named_regions["centered_normalization_core"] = {
            name for region in source["centered_normalization_cores"] for name in region["source_nodes"]}
        excluded = set().union(*named_regions.values())
        elementary = {"Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log", "Sin", "Cos", "Tanh", "Sigmoid", "Relu", "LeakyRelu", "Erf", "Softplus", "Neg", "Abs"}
        named_regions["remaining_elementwise"] = {
            row["source_node"] for row in provenance.get("node_sources", [])
            if row["source_op"] in elementary and row["source_node"] not in excluded}
        named_regions["reference_Conv"] = {node["name"] for node in reference_inventory["selected_geometry"]
                                           if node["operator"] == "Conv"}
        named_regions["reference_ConvTranspose"] = {
            name for region in source["reference_convtranspose_regions"] for name in region["source_nodes"]}
        packages[entry["name"]] = {"graph": graph, "provenance": provenance,
            "graph_artifact": receipt(graph_path), "graph_fingerprint": runtime_source_fingerprint(graph_path),
            "source": source, "named_regions": named_regions}
    out = {"format": "volvoxai-tts-profile-comparison/v1", "manifest": receipt(manifest_path),
           "model": manifest["model"], "volvox": [], "ort": [],
           "whole_reference_inventory": reference_inventory,
           "source_inventories": {name: package["source"] for name, package in packages.items()},
           "interpretation": [
               "Compare complete prepared-input to host-output latency in the ordinary benchmark; profiling changes the workload.",
               "ORT optimized nodes and Volvox lowered nodes are different computations at the kernel boundary. Counts explain structure, not speedup.",
               "ORT Node kernel_time is host wall time, while Volvox device programs are timestamp/event elapsed time. These domains are never divided or added together.",
               "Mixed ORT providers remain separated. A CUDA label does not establish zero CPU fallback.",
               "Volvox host nodes, device nodes/programs and whole passes overlap. Dropped timestamps make detailed device attribution partial.",
               "Named source regions can overlap operator classifications; do not sum them as a partition of the request.",
               "Prepared source ONNX graphs include application geometry lowering and can differ from the original ORT benchmark graph; region names identify that boundary."]}
    ort_source_regions = defaultdict(set)
    for package in packages.values():
        for name, names in package["named_regions"].items():
            if name in {"centered_normalization_core", "remaining_elementwise"}:
                ort_source_regions[name].update(names)
    for node in out["whole_reference_inventory"]["selected_geometry"]:
        ort_source_regions[node["operator"]].add(node["name"])
    for path in volvox_paths:
        report = read_json(path)
        if report.get("manifest_sha256") != out["manifest"]["sha256"]:
            raise ValueError(f"Volvox profile does not identify this manifest: {path}")
        result = {"artifact": receipt(path), "backend": report["backend"], "stages": {},
                  "diagnostic_request_ms": report.get("untraced_ms", report.get("timings_ms", {}).get("request_ms")),
                  "device_api": report.get("untraced_api_counts"),
                  "validation": report.get("validation"), "route": report.get("route")}
        for name, stage in report["stages"].items():
            package = packages.get(name.removesuffix("_basic"))
            if package is None:
                raise ValueError(f"Unknown Volvox stage: {name}")
            for plan in stage.get("plans", []):
                if plan.get("graphFingerprint") != package["graph_fingerprint"]:
                    raise ValueError(f"Trace graph differs from prepared graph: {path} {name}")
            result["stages"][name] = {
                "prepared_graph": package["graph_artifact"],
                "plan_identity": [{key: plan.get(key) for key in
                                   ("graphFingerprint", "shapeSignature", "sourceMappingComplete")}
                                  for plan in stage.get("plans", [])],
                "coverage": {key: stage["info"].get(key) for key in ("events", "devices", "plans")},
                "host_nodes": aggregate_volvox(stage["events"], "host", "node"),
                "device_programs": aggregate_volvox(stage["events"], "device", "program"),
                "device_passes_and_copies": aggregate_volvox(stage["events"], "device", "operation"),
                "source_regions": region_attribution(stage, package["graph"], package["provenance"], package["named_regions"]),
                "provenance_available": bool(package["provenance"])}
        out["volvox"].append(result)
    for path in ort_paths:
        report = read_json(path)
        if isinstance(report, list):
            out["ort"].append({"artifact": receipt(path), **summarize_ort_events(report)})
        else:
            if report.get("format") != "volvoxai-tts-ort-profile/v1":
                raise ValueError(f"Unsupported ORT profile format: {path}")
            if report.get("manifest_sha256") != out["manifest"]["sha256"]:
                raise ValueError(f"ORT profile does not identify this manifest: {path}")
            trace = report.get("trace", {})
            summaries = trace.get("summaries", {})
            out["ort"].append({"artifact": receipt(path),
                **{key: report.get(key) for key in ("format", "variant", "model_artifacts", "settings", "runtime", "benchmark", "memory", "validation", "comparison_to_own_fp32")},
                "trace": {key: value for key, value in trace.items() if key != "summaries"},
                "operators": summaries.get("operators", []),
                "providers": summaries.get("providers", []),
                "top_host_nodes": summaries.get("nodes", [])[:50],
                "source_name_regions": match_ort_regions(summaries.get("nodes", []), ort_source_regions),
                "graphs": {name: {key: value for key, value in inventory.items()
                                  if key not in {"nodes", "initializers"}}
                           for name, inventory in report.get("graphs", {}).items() if isinstance(inventory, dict)},
                "timing_domain": "ORT node host intervals; validated device intervals, if any, must be separately identified"})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--volvox", type=Path, action="append", required=True)
    parser.add_argument("--ort", type=Path, action="append", default=[])
    parser.add_argument("--provenance", action="append", default=[], metavar="STAGE=FILE")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    provenance = {}
    for binding in args.provenance:
        name, separator, path = binding.partition("=")
        if not separator or not name or not path or name in provenance:
            parser.error("provenance requires unique STAGE=FILE bindings")
        provenance[name] = Path(path)
    result = compare(args.manifest, args.volvox, args.ort, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
