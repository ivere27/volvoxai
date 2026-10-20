#!/usr/bin/env python3
"""Aggregate current-artifact observations without inventing regression budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from measure_native_baseline import WORKLOADS, distribution, identity, read_policy, require


def call_throughput(samples):
    """Combine raw ID counts and completed-call durations, not percentiles."""
    require(samples and all(count > 0 and duration > 0 for count, duration in samples),
            "throughput needs positive completed-call counts and durations")
    count = sum(count for count, _ in samples)
    duration = sum(duration for _, duration in samples)
    rates = distribution([count * 1_000_000_000 / duration for count, duration in samples])
    rates["unit"] = "selected_ids_per_second"
    return {"completed_calls": len(samples), "selected_ids": count, "sum_call_ns": duration,
            "aggregate_selected_ids_per_second": count * 1_000_000_000 / duration,
            "per_call_rates": rates,
            "scope": "Completed public calls only, including requested output serialization; context creation/retirement, tokenizer, loading and setup are excluded. Cached prefill/decode counts one selected ID per call; Generate counts returned generated IDs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--workload", choices=WORKLOADS, default="cached_execute")
    parser.add_argument("--source-storage", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "refusing to overwrite evidence")
    policy = read_policy(args.policy, args.workload)
    workload = policy["workload"]
    inputs, groups = {}, {}
    for path in sorted(args.results.rglob("*.json")):
        value = json.loads(path.read_text())
        if value.get("format") != "volvox-native-llm-baseline-observation/v1" or value.get("workload") != workload:
            continue
        require(value["status"] == "observed", f"failed worker: {path}")
        require(not value["private_tmp_remaining"], f"retained private source snapshot: {path}")
        require(value["identities_before"] == value["identities_after"], f"artifact changed: {path}")
        require(value["identities_before"]["policy"]["sha256"] == identity(args.policy)["sha256"], "policy differs")
        if value["mode"] == "trace":
            for segment in value["trace"]["segments"]:
                raw = segment["raw_events_and_snapshots_gzip"]
                require(identity(raw["path"]) == raw, "raw trace sidecar identity differs")
        key = (value["model"], value["profile"], value["backend"])
        groups.setdefault(key, {}).setdefault(value["mode"], []).append(value)
        inputs[str(path.resolve())] = identity(path)
    cells = []
    complete = True
    for model in policy["models"]:
        for cell in policy["cells"]:
            key = (model, cell["profile"], cell["backend"])
            modes = groups.get(key, {})
            expected = {"timing": policy["cold_timing_processes_per_cell"], "memory": 1, "trace": 1}
            counts = {mode: len(modes.get(mode, [])) for mode in expected}
            qualified = counts == expected
            complete &= qualified
            row = {"model": model, **cell, "worker_counts": counts, "expected_worker_counts": expected,
                   "planned_observations_complete": qualified}
            values = [v for entries in modes.values() for v in entries]
            if values:
                require(len({v["host"] for v in values}) == 1, "cell spans different hosts")
                require(len({v["identities_before"]["library"]["sha256"] for v in values}) == 1, "cell spans different libraries")
                require(len({v["source_closure_sha256"] for v in values}) == 1, "cell spans different sources")
                for name in values[0]["identities_before"]:
                    require(len({(v["identities_before"][name]["sha256"], v["identities_before"][name]["bytes"])
                                 for v in values}) == 1, f"cell spans different {name} identities")
                row["host"] = values[0]["host"]
                row["source_closure_sha256"] = values[0]["source_closure_sha256"]
                row["artifacts"] = {k: values[0]["identities_before"][k] for k in ("library", "bridge", "graph", "weights", "reference", "harness", "policy")}
                row["source_storage"] = values[0]["prepared_storage_inventory"]
            timings = modes.get("timing", [])
            if timings:
                for value in timings:
                    require(len(value["cycles"]) == 6 * policy["measured_cycles_per_case"], "timing case/repeat coverage differs")
                cold = {}
                for name in timings[0]["cold"]:
                    samples = [v["cold"][name] for v in timings]
                    if isinstance(samples[0], list):
                        cold[name] = [distribution([v[i] for v in samples]) for i in range(len(samples[0]))]
                    else:
                        cold[name] = distribution(samples)
                cases = sorted({c["case"] for v in timings for c in v["cycles"]})
                phases = sorted({s["phase"] for v in timings for c in v["cycles"] for s in c["steps"]})
                row["cold"] = cold
                row["warm"] = {case: {phase: distribution([s["call_ns"] for v in timings for c in v["cycles"]
                    if c["case"] == case for s in c["steps"] if s["phase"] == phase]) for phase in phases} for case in cases}
                row["warm_call_throughput"] = {case: {phase: call_throughput([
                    (s.get("generated_ids", 1), s["call_ns"]) for v in timings for c in v["cycles"]
                    if c["case"] == case for s in c["steps"] if s["phase"] == phase])
                    for phase in phases} for case in cases}
                row["warm_context"] = {case: {name: distribution([c[name] for v in timings for c in v["cycles"]
                    if c["case"] == case]) for name in ("context_create_ns", "context_release_ns")}
                    for case in cases}
            if modes.get("memory"):
                require(len(modes["memory"]) == 1, "memory worker count differs")
                value = modes["memory"][0]
                row["memory"] = {"linux_process_lifetime_VmHWM_bytes": value["process_lifetime_final"]["VmHWM"],
                    "sampled_process_RSS_max_bytes": value["rss_sampled_max_bytes"],
                    "sampled_stage_RSS_max_bytes": value["rss_sampled_max_by_phase"],
                    "sampled_private_snapshot_logical_max_bytes": max(s["private_snapshot_logical_bytes"] for s in value["rss_samples"]),
                    "sampled_private_snapshot_blocks_max_bytes": max(s["private_snapshot_blocks_bytes"] for s in value["rss_samples"]),
                    "sampled_process_swap_max_bytes": max(s["VmSwap"] for s in value["rss_samples"]),
                    "public_resource_snapshot_count": len(value["resource_snapshots"]),
                    "known_ownership_cleanup": value["known_ownership_cleanup"],
                    "private_snapshot_files_after_retirement": value["private_tmp_remaining"],
                    "system_start": value["system_before_library"], "system_end": value["system_final"]}
                for name in ("nvml_worker_sampled_max_bytes", "nvml_before_library", "nvml_after_library_closed", "nvml_interval_sample_scope"):
                    if name in value:
                        row["memory"][name] = value[name]
            if modes.get("trace"):
                require(len(modes["trace"]) == 1, "trace worker count differs")
                value = modes["trace"][0]
                segments = value["trace"]["segments"]
                row["trace"] = {"segments": segments, "segment_count": len(segments),
                    "copy_wait_events": sum(s["copy_wait_events"] for s in segments),
                    "tracked_allocator_scope_only": True,
                    "peak_combination": "No sum of segment or allocator peaks; each public allocator row retains its seeded visited-owner scope. Retirement-only missing allocators remain unvisited/unknown."}
            cells.append(row)
    result = {"format": "volvox-native-llm-baseline-summary/v1", "status": "observed" if complete else "partial",
              "workload": workload, "policy": identity(args.policy), "workers": inputs, "cells": cells,
              "warm_scope": policy["warm_scope"],
              "claims": policy["interpretation"], "service_actions": 0,
              "performance_nonregression_qualified": False, "whole_memory_nonregression_qualified": False,
              "device_interval_peak_exact": False, "matching_baseline": None, "regression_allowance": None}
    if args.source_storage:
        result["source_storage_receipt"] = identity(args.source_storage)
        result["source_storage_and_artifact_sizes"] = json.loads(args.source_storage.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "cells": len(cells), "workers": len(inputs)}))


if __name__ == "__main__":
    main()
