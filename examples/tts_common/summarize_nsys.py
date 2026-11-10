#!/usr/bin/env python3
"""Summarize warmed complete-request NVTX ranges in an Nsight SQLite export.

Host API, kernel, copy and memset intervals are separate timing domains.
Only events fully contained in the named request and its process qualify.
The caller must close each range after receiving the completed host output.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def union_ns(intervals):
    total = 0
    previous_end = None
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end if previous_end is not None else start))
        previous_end = max(end, previous_end if previous_end is not None else end)
    return total


def process_id(value):
    return (int(value) >> 24) & 0xFFFFFF


def summarize_database(path, prefix="tts-measured-run-"):
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def rows(table):
            if table not in names:
                return []
            # Table names below are fixed strings, never supplied by the caller.
            return [dict(row) for row in db.execute(f'SELECT * FROM "{table}"')]

        strings = {row["id"]: row["value"] for row in rows("StringIds")}
        ranges = []
        for row in rows("NVTX_EVENTS"):
            label = row.get("text") or strings.get(row.get("textId"), "")
            if not isinstance(label, str) or not label.startswith(prefix):
                continue
            if row.get("end") is None or row["end"] <= row["start"]:
                raise ValueError(f"Incomplete request range: {label}")
            if row.get("globalTid") is None:
                raise ValueError(f"Request range lacks process identity: {label}")
            ranges.append({"name": label, "start": row["start"], "end": row["end"],
                           "pid": process_id(row["globalTid"])})
        ranges.sort(key=lambda row: row["start"])
        if not ranges:
            raise ValueError(f"No complete NVTX request ranges matching {prefix!r}")
        for i, current in enumerate(ranges):
            if any(prior["pid"] == current["pid"] and prior["end"] > current["start"]
                   for prior in ranges[:i]):
                raise ValueError("Overlapping request ranges would double-count GPU work")

        domains = {
            "kernels": "CUPTI_ACTIVITY_KIND_KERNEL",
            "copies": "CUPTI_ACTIVITY_KIND_MEMCPY",
            "memsets": "CUPTI_ACTIVITY_KIND_MEMSET",
            "runtime_api": "CUPTI_ACTIVITY_KIND_RUNTIME",
            "driver_api": "CUPTI_ACTIVITY_KIND_DRIVER",
        }
        events = {name: rows(table) for name, table in domains.items()}
        copy_names = {row["id"]: row.get("label", row.get("name", str(row["id"])))
                      for row in rows("ENUM_CUDA_MEMCPY_OPER")}
        coverage = {name: {"table_present": table in names, "all_events": len(events[name]),
                          "events_without_process_identity": sum(
                              event.get("globalPid") is None and event.get("globalTid") is None
                              for event in events[name])}
                    for name, table in domains.items()}

        def selected(event, request):
            identifier = event.get("globalPid") or event.get("globalTid")
            return (identifier is not None and process_id(identifier) == request["pid"]
                    and event.get("start") is not None and event.get("end") is not None
                    and request["start"] <= event["start"] <= event["end"] <= request["end"])

        def label(event, domain):
            if domain == "kernels":
                return strings.get(event.get("demangledName"),
                                   strings.get(event.get("shortName"), "unknown"))
            if domain == "copies":
                return copy_names.get(event.get("copyKind"), f"copyKind={event.get('copyKind')}")
            if domain.endswith("_api"):
                return strings.get(event.get("nameId"), "unknown")
            return "memset"

        requests = []
        totals = {name: defaultdict(lambda: {"calls": 0, "ns": 0, "bytes": 0,
                                             "request_ns": [0] * len(ranges), "launches": set()})
                  for name in domains}
        for index, request in enumerate(ranges):
            result = {"name": request["name"], "host_range_ms": (request["end"] - request["start"]) / 1e6}
            gpu_intervals = []
            for domain, entries in events.items():
                matched = [event for event in entries if selected(event, request)]
                intervals = [(event["start"], event["end"]) for event in matched]
                result[domain] = {"calls": len(matched),
                    "elapsed_sum_ms": sum(end - start for start, end in intervals) / 1e6,
                    "elapsed_union_ms": union_ns(intervals) / 1e6,
                    "bytes": sum(event.get("bytes", 0) or 0 for event in matched),
                    "boundary_crossing_events_excluded": sum(
                        (identifier := event.get("globalPid") or event.get("globalTid")) is not None
                        and process_id(identifier) == request["pid"]
                        and event.get("start") is not None and event.get("end") is not None
                        and event["start"] < request["end"] and event["end"] > request["start"]
                        and not selected(event, request) for event in entries)}
                if domain in ("kernels", "copies", "memsets"):
                    gpu_intervals.extend(intervals)
                for event in matched:
                    bucket = totals[domain][label(event, domain)]
                    duration = event["end"] - event["start"]
                    bucket["calls"] += 1
                    bucket["ns"] += duration
                    bucket["bytes"] += event.get("bytes", 0) or 0
                    bucket["request_ns"][index] += duration
                    if domain == "kernels":
                        bucket["launches"].add(tuple(event.get(key) for key in
                            ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ")))
            result["device_activity_union_ms"] = union_ns(gpu_intervals) / 1e6
            requests.append(result)

        groups = {}
        for domain, entries in totals.items():
            groups[domain] = []
            for name, values in sorted(entries.items(), key=lambda pair: pair[1]["ns"], reverse=True):
                group = {"name": name, "calls": values["calls"],
                         "calls_per_request": values["calls"] / len(ranges),
                         "elapsed_sum_ms": values["ns"] / 1e6,
                         "median_request_ms": statistics.median(values["request_ns"]) / 1e6,
                         "bytes": values["bytes"], "bytes_per_request": values["bytes"] / len(ranges)}
                if values["launches"]:
                    group["grid_xyz_block_xyz"] = sorted(values["launches"], key=str)
                groups[domain].append(group)
        have_kernels = any(row["kernels"]["calls"] for row in requests)
        return {"format": "volvoxai-tts-nsys-summary/v1", "sqlite_file": path.name,
                "sqlite_sha256": _sha256(path),
                "range_prefix": prefix, "request_count": len(requests),
                "gpu_timing_available": have_kernels, "coverage": coverage,
                "requests": requests, "groups": groups,
                "median": {"host_range_ms": statistics.median(r["host_range_ms"] for r in requests),
                    "kernel_sum_ms": statistics.median(r["kernels"]["elapsed_sum_ms"] for r in requests) if have_kernels else None,
                    "device_activity_union_ms": statistics.median(r["device_activity_union_ms"] for r in requests) if have_kernels else None},
                "notes": ["Nanosecond CUDA activities are device timestamps; API ranges are host timestamps.",
                    "Only fully contained events from the request process qualify; first calls, warmups and unrelated processes are excluded.",
                    "Runtime/driver APIs can nest; neither is added to device intervals or to the other API total.",
                    "Device interval union describes observed activity intervals, not hardware occupancy or SM utilization.",
                    "Per-kernel attribution to a model operator is not inferred from a name or asynchronous host overlap.",
                    "Missing kernel activity is unavailable timing, not zero GPU computation."]}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--range-prefix", default="tts-measured-run-")
    args = parser.parse_args()
    if args.sqlite.resolve() == args.out.resolve():
        parser.error("output must differ from the source SQLite export")
    result = summarize_database(args.sqlite, args.range_prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
