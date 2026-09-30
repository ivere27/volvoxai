"""Offline, layer-wise comparison of ``volvoxai-debug/v1`` captures.

The artifact and report contracts match ``tools/debug_report.mjs``. Matching
uses source tensor and node identities, never session-local snapshot IDs;
quantized storage is dequantized before measuring error. This module needs no
engine, training service or exporter.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

_UNDEFINED = object()
_DTYPES = {
    "DATA_TYPE_F32": "f", "DATA_TYPE_F64": "d", "DATA_TYPE_F16": "e",
    "DATA_TYPE_BF16": "H", "DATA_TYPE_I64": "q", "DATA_TYPE_U64": "Q",
    "DATA_TYPE_I32": "i", "DATA_TYPE_U32": "I", "DATA_TYPE_I16": "h",
    "DATA_TYPE_U16": "H", "DATA_TYPE_I8": "b", "DATA_TYPE_U8": "B",
    "DATA_TYPE_BOOL": "B",
}
_CONSTANT_ROLES = {"MEMORY_RESOURCE_ROLE_WEIGHTS", "MEMORY_RESOURCE_ROLE_PACKED_WEIGHTS"}
_EPSILON = 2.220446049250313e-16


def _artifact(value):
    if isinstance(value, (str, os.PathLike)):
        with Path(value).open(encoding="utf-8") as stream:
            value = json.load(stream)
    if not isinstance(value, Mapping) or value.get("format") != "volvoxai-debug/v1":
        raise ValueError("expected volvoxai-debug/v1")
    return value


def _ptq_activation_name(source):
    digest = hashlib.sha256(f"volvox-typed-ptq/v1\0{source}\0activation".encode()).hexdigest()[:20]
    return f"__ptq__.{digest}.activation"


def _js_string(value):
    # Keys are built from protobuf JSON scalar values just as in the JS tool.
    if value is _UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _after(point):
    return point == "DEBUG_POINT_AFTER" or point == 2


def _available(status):
    return status == "DEBUG_TENSOR_STATUS_AVAILABLE" or status == 1


@dataclass(eq=False)
class _Entry:
    nodes: list
    tensor: object
    source: str
    point: object
    step: int
    kind: str
    snapshot: Mapping


def _key(source, point, node):
    return f"{source}\0{_js_string(point)}\0{_js_string(node)}"


def _index(artifact, aliases):
    plan = artifact.get("plan") or {}
    tensors, steps = plan.get("tensors", []), plan.get("steps", [])
    constants = {_js_string(item.get("allocationId", _UNDEFINED))
                 for item in plan.get("allocations", []) if item.get("role") in _CONSTANT_ROLES}
    produced = {int(tensor_id) for step in steps for tensor_id in step.get("outputs", [])}
    entries, by_node, by_output = [], {}, {}
    aliased = 0
    for event in artifact.get("events", []):
        step_id = event.get("step", 0)
        step = steps[int(step_id)] if 0 <= int(step_id) < len(steps) else {}
        for snapshot in event.get("snapshots", []):
            tensor_id = snapshot.get("tensorId", 0)
            tensor = tensors[int(tensor_id)] if 0 <= int(tensor_id) < len(tensors) else {}
            name = tensor.get("name", _UNDEFINED)
            source = ""
            if name is not _UNDEFINED and name:
                if tensor.get("sourceTensorName"):
                    source = tensor["sourceTensorName"]
                    aliased += 1
                else:
                    stem = re.sub(r"(\.activation)\.\d+$", r"\1", name)
                    source = aliases.get(stem, name)
                    aliased += stem in aliases
            kind = ("activation" if tensor_id in produced else
                    "constant" if _js_string(tensor.get("allocationId", _UNDEFINED)) in constants else
                    "external")
            entry = _Entry(step.get("sourceNodeIds", []), name, source,
                           event.get("point", _UNDEFINED), step_id, kind, snapshot)
            entries.append(entry)
            for node in entry.nodes:
                by_node.setdefault(_key(source, entry.point, node), entry)
            if _after(entry.point):
                by_output.setdefault(source, entry)
    return entries, by_node, by_output, aliased


def _counterpart(entry, by_node, by_output):
    for node in entry.nodes:
        match = by_node.get(_key(entry.source, entry.point, node))
        if match is not None:
            return match, node
    if _after(entry.point) and entry.source in by_output:
        return by_output[entry.source], entry.nodes[0] if entry.nodes else _UNDEFINED
    return None


def _affine(snapshot):
    q = snapshot.get("quantization") or {}
    if q.get("perTensor") is not None:
        affine = q["perTensor"]
        return lambda i: (affine.get("scale", 0), affine.get("zeroPoint", 0))
    if q.get("perAxis") is None:
        return None
    affine, shape = q["perAxis"], [int(size) for size in snapshot.get("shape", [])]
    axis = affine.get("axis", 0)
    inner = math.prod(shape[axis + 1:])
    scales, zeros = affine.get("scales", []), affine.get("zeroPoints", [])

    def parameters(index):
        channel = index // inner % shape[axis]
        return scales[channel], zeros[channel] if channel < len(zeros) else 0

    return parameters


def _values(snapshot):
    encoded, dtype = snapshot.get("valuesBase64"), snapshot.get("dtype")
    if not encoded or dtype not in _DTYPES:
        return None
    data = base64.b64decode(encoded)
    try:
        values = [float(value[0]) for value in struct.iter_unpack("<" + _DTYPES[dtype], data)]
    except struct.error as error:
        raise ValueError("debug tensor bytes do not match dtype width") from error
    if dtype == "DATA_TYPE_BF16":
        values = [struct.unpack("<f", struct.pack("<I", int(value) << 16))[0] for value in values]
    elif dtype == "DATA_TYPE_BOOL":
        values = [1.0 if value else 0.0 for value in values]
    affine = _affine(snapshot)
    if affine is not None:
        values = [scale * (value - zero) for index, value in enumerate(values)
                  for scale, zero in [affine(index)]]
    return values


def _nonfinite(snapshot):
    statistics = snapshot.get("realStatistics")
    if statistics is None:
        statistics = snapshot.get("statistics") or {}
    return sum(float(statistics.get(field, 0)) for field in
               ("nanCount", "positiveInfinityCount", "negativeInfinityCount"))


def _divide(numerator, denominator):
    # JS floating-point arithmetic preserves overflow/underflow evidence until
    # final JSON serialization, whereas Python raises for a zero denominator.
    if denominator == 0:
        return math.nan if numerator == 0 else math.copysign(math.inf, numerator)
    return numerator / denominator


def _metrics(x, y, affine, atol, rtol):
    count = mismatches = 0
    maximum = relative = absolute = squared = signal = candidate_signal = dot = scaled = 0.0
    for index, (left, right) in enumerate(zip(x, y)):
        difference = abs(left - right)
        if not math.isfinite(difference):
            continue
        count += 1
        maximum = max(maximum, difference)
        relative = max(relative, difference / max(abs(left), _EPSILON))
        absolute += difference
        squared += difference * difference
        signal += left * left
        candidate_signal += right * right
        dot += left * right
        if affine is not None:
            scale, _ = affine(index)
            normalized = difference / scale if scale > 0 else 0
            scaled += normalized * normalized
        mismatches += difference > atol + rtol * abs(left)
    ratio = signal / squared if squared > 0 and signal > 0 else None
    result = {
        "elements": len(x), "finitePairs": count, "maxAbsError": maximum, "maxRelError": relative,
        "meanAbsError": absolute / count if count else None,
        "rmse": math.sqrt(squared / count) if count else None,
        "identical": count > 0 and squared == 0, "mismatches": mismatches,
        "sqnrDb": (10 * math.log10(ratio) if ratio != 0 else -math.inf) if ratio is not None else None,
        "cosine": _divide(dot, math.sqrt(signal * candidate_signal)) if signal > 0 and candidate_signal > 0 else None,
        "relativeL2": math.sqrt(squared / signal) if signal > 0 else None,
    }
    if affine is not None and count:
        result["rmseOverScale"] = math.sqrt(scaled / count)
    return result


def _json_safe(value):
    # JSON.stringify omits undefined object members and turns non-finite metric
    # numbers into null. Produce the same JSON-compatible report in Python.
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items() if item is not _UNDEFINED}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def debug_compare(reference, candidate, *, mode="auto", atol=1e-5, rtol=1e-3, min_sqnr_db=20):
    """Compare captures or JSON file paths without loading a native library.

    ``auto`` selects ``quantization`` when an aligned pair changes dtype or
    contains affine quantization; otherwise it selects ``exact``. Exact mode
    flags elements outside ``atol + rtol * abs(reference)``. Quantization mode
    flags SQNR below ``min_sqnr_db`` and ranks the ten worst activation outputs.
    The JSON-safe result uses the JS report's camel-case field names. A result
    with ``comparable=False`` carries no numerical comparison evidence.
    """
    reference, candidate = _artifact(reference), _artifact(candidate)
    if mode not in ("auto", "exact", "quantization"):
        raise ValueError(f"unknown comparison mode {mode}")
    aliases = {_ptq_activation_name(tensor["name"]): tensor["name"]
               for artifact in (reference, candidate)
               for tensor in (artifact.get("plan") or {}).get("tensors", [])
               if tensor.get("name") and not tensor["name"].startswith("__ptq__.")}
    left, _, _, left_aliased = _index(reference, aliases)
    right, by_node, by_output, right_aliased = _index(candidate, aliases)
    rows, unmatched, used = [], [], set()
    activations = matched_activations = 0
    for entry in left:
        activations += entry.kind == "activation"
        found = _counterpart(entry, by_node, by_output)
        if found is None:
            unmatched.append({"nodes": entry.nodes, "tensor": entry.tensor, "point": entry.point,
                              "step": entry.step, "kind": entry.kind})
            continue
        match, node = found
        used.add(match)
        matched_activations += entry.kind == "activation"
        a, b = entry.snapshot, match.snapshot
        row = {"node": node, "nodes": entry.nodes, "tensor": entry.tensor, "candidateTensor": match.tensor,
               "point": entry.point, "kind": entry.kind, "referenceStep": entry.step, "candidateStep": match.step,
               "referenceStatus": a.get("status", _UNDEFINED), "candidateStatus": b.get("status", _UNDEFINED),
               "referenceDtype": a.get("dtype", _UNDEFINED), "candidateDtype": b.get("dtype", _UNDEFINED),
               "quantized": a.get("quantization") is not None or b.get("quantization") is not None,
               "referenceNonfinite": _nonfinite(a), "candidateNonfinite": _nonfinite(b)}
        x = _values(a) if _available(a.get("status")) else None
        y = _values(b) if _available(b.get("status")) else None
        if x is not None and y is not None and len(x) == len(y):
            row.update(_metrics(x, y, _affine(b) or _affine(a), atol, rtol))
        elif x is not None and y is not None:
            row["comparison"] = "shape differs"
        elif not _available(a.get("status")) or not _available(b.get("status")):
            row["comparison"] = "snapshot has no values"
        elif a.get("dtype") not in _DTYPES or b.get("dtype") not in _DTYPES:
            row["comparison"] = "unsupported dtype"
        else:
            row["comparison"] = "values not captured in both sessions"
        rows.append(row)
    effective = (mode if mode != "auto" else "quantization" if any(
        row["quantized"] or row["referenceDtype"] != row["candidateDtype"] for row in rows) else "exact")
    for row in rows:
        numerical = (row["mismatches"] > 0 if effective == "exact" else
                     row["sqnrDb"] is not None and row["sqnrDb"] < min_sqnr_db) if "elements" in row else False
        row["mismatch"] = row["referenceNonfinite"] != row["candidateNonfinite"] or numerical
    rows.sort(key=lambda row: (row["referenceStep"], 0 if row["point"] in ("DEBUG_POINT_BEFORE", 1) else 1))
    compared = sum("elements" in row for row in rows)
    worst = []
    if effective == "quantization":
        worst_rows = sorted((row for row in rows if _after(row["point"]) and row["kind"] == "activation"
                             and row.get("sqnrDb") is not None), key=lambda row: row["sqnrDb"])[:10]
        worst = [{key: row.get(key, _UNDEFINED) for key in
                  ("node", "tensor", "point", "sqnrDb", "cosine", "relativeL2", "rmseOverScale")}
                 for row in worst_rows]
    interpretation = "The first mismatch is where a difference became observable, not proof of its cause."
    if effective == "quantization":
        interpretation = ("Errors accumulate through the graph: a low-SQNR layer can inherit error from earlier layers. "
                          + interpretation)
    return _json_safe({
        "format": "volvoxai-debug-comparison/v2", "mode": effective,
        "tolerance": {"atol": atol, "rtol": rtol, "minSqnrDb": min_sqnr_db}, "comparable": compared > 0,
        "reason": _UNDEFINED if compared else
            "aligned observations carry no comparable values; capture with values: true" if rows else
            "no observation aligned by tensor, point and source node",
        "matched": len(rows), "compared": compared, "unmatched": len(unmatched),
        "candidateOnly": sum(entry not in used for entry in right),
        "activationCoverage": {"matched": matched_activations, "total": activations,
                               "ratio": matched_activations / activations if activations else None},
        "aliases": {"ptqActivations": left_aliased + right_aliased},
        "firstMismatch": next((row for row in rows if row["mismatch"]), None) if compared else None,
        "worst": worst, "rows": rows, "unmatchedObservations": unmatched, "interpretation": interpretation,
    })
