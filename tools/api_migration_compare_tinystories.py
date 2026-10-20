#!/usr/bin/env python3
"""Compare frozen legacy TinyStories captures with unified API captures.

Opaque handles and transport reports differ across runs. Compare token decisions,
logical cache state, every vocabulary logit and every captured KV element instead.
Timing is deliberately excluded; the independent ABBA gate qualifies performance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(baseline: Path, candidate: Path) -> dict:
    old = json.loads((baseline / "report.json").read_text())
    new = json.loads((candidate / "report.json").read_text())
    failures, arrays, modes = [], [], []

    def equal(label, a, b):
        if a != b:
            failures.append({"field": label, "baseline": a, "candidate": b})

    def tensor(label, a, b):
        equal(label + ".shape", a["shape"], b["shape"])
        equal(label + ".dtype", a["dtype"], b["dtype"])
        values = []
        for directory, descriptor in ((baseline, a), (candidate, b)):
            raw = (directory / descriptor["file"]).read_bytes()
            equal(label + ".sha256:" + str(directory), descriptor["sha256"],
                  hashlib.sha256(raw).hexdigest())
            if descriptor["dtype"] not in ("float32", 18):
                raise ValueError(f"{label}: unsupported capture dtype")
            data = np.frombuffer(raw, dtype="<f4")
            equal(label + ".elements", int(np.prod([int(n) for n in descriptor["shape"]])),
                  int(data.size))
            values.append(data)
        a_values, b_values = values
        if a_values.shape != b_values.shape:
            return
        finite = bool(np.isfinite(a_values).all() and np.isfinite(b_values).all())
        error = float(np.max(np.abs(a_values.astype(np.float64) - b_values), initial=0))
        passed = finite and error <= 1e-4
        arrays.append({"name": label, "elements": int(a_values.size),
                       "max_abs_error": error, "finite": finite, "pass": passed})
        if not passed:
            failures.append({"field": label, "max_abs_error": error, "finite": finite})

    for field in ("schema", "profile", "backend"):
        equal(field, old[field], new[field])
    equal("api_generation", "unified", new.get("api_generation"))
    equal("workloads", [w["name"] for w in old["workloads"]],
          [w["name"] for w in new["workloads"]])
    for a, b in zip(old["workloads"], new["workloads"]):
        label = a["name"]
        equal(label + ".status", "executed", b["status"])
        equal(label + ".settings", a["settings"], b["settings"])
        for field in ("promptTokenIds", "requestedNewTokens", "seed", "temperature", "topK",
                      "sequenceCapacity", "promptLength", "generationLimit",
                      "generatedTokenIds", "stopReason", "text"):
            equal(label + ".run." + field, a["run"][field], b["run"][field])
        equal(label + ".steps", len(a["steps"]), len(b["steps"]))
        for x, y in zip(a["steps"], b["steps"]):
            step = f"{label}.step{x['index']}"
            for field in ("index", "position", "phase", "token_id"):
                equal(step + "." + field, x[field], y[field])
            tensor(step + ".logits", x["logits"], y["logits"])
            for field in ("slots", "activeLengths", "empty"):
                equal(step + ".state." + field, x["state"][field], y["state"][field])
            equal(step + ".initialized", x["state"]["prefilled"], y["state"]["initialized"])
            equal(step + ".ready", 1, y["state"]["state"])
            equal(step + ".bindings", True, y["state"]["bindingsAvailable"])
            # Legacy AUTO fell back from ROW(3) to DEPENDENCY(4). The new
            # explicit row contract admits the existing causal SDPA/ArgMax route.
            modes.append({"step": step, "baseline": x["state"]["mode"],
                          "candidate": y["state"]["mode"]})
            expected_mode = 4 if new["backend"] == "webgpu" else 3
            equal(step + ".execution_mode", expected_mode, y["state"]["mode"])
            equal(step + ".generation_stable", True,
                  y["state"]["cacheGeneration"] == b["steps"][0]["state"]["cacheGeneration"])
        old_caches, new_caches = a.get("debug_caches", []), b.get("debug_caches", [])
        equal(label + ".debug_predictions", len(old_caches), len(new_caches))
        for x, y in zip(old_caches, new_caches):
            prefix = f"{label}.debug{x['prediction']}"
            equal(prefix + ".prediction", x["prediction"], y["prediction"])
            equal(prefix + ".slots", x["state"]["slots"], y["state"]["slots"])
            key = lambda item: (item["descriptor"]["attentionStep"], item["descriptor"]["role"])
            old_by_key = {key(c): c for c in x["caches"]}
            new_by_key = {key(c): c for c in y["caches"]}
            equal(prefix + ".cache_identity", sorted(old_by_key), sorted(new_by_key))
            for identity in old_by_key.keys() & new_by_key.keys():
                u, v = old_by_key[identity], new_by_key[identity]
                for field in ("tokenCapacity", "tokenShape", "pageTokens", "writerStep", "written"):
                    equal(prefix + f".cache{identity}." + field,
                          u["descriptor"][field], v["descriptor"][field])
                tensor(prefix + f".cache{identity}", u, v)
        if a["settings"]["temperature"] == 0:
            chunks = b.get("engine_chunks", [])
            refusal = b.get("engine_generation", {})
            if new["backend"] == "webgpu" and refusal.get("status") == "unsupported":
                equal(label + ".Generate.admission", -7, refusal["admission"]["status"])
                equal(label + ".chunk_cases", 0, len(chunks))
            else:
                equal(label + ".chunk_cases", 4, len(chunks))
            for result in chunks:
                ids = [token for chunk in result["chunks"] for token in chunk["ids"]]
                equal(label + ".Generate.ids", a["run"]["generatedTokenIds"], ids)
                equal(label + ".Generate.completed", 4, result["state"]["state"])
                equal(label + ".Generate.steps", str(len(ids)), result["state"]["totalSteps"])
                equal(label + ".Generate.activeLengths", [a["run"]["promptLength"] + len(ids) - 1],
                      result["state"]["activeLengths"])
    return {"schema": "volvoxai.api-migration-tinystories-comparison/v1",
            "baseline": str(baseline), "candidate": str(candidate),
            "tolerance": {"max_abs_error": 1e-4}, "performance_qualified": False,
            "pass": not failures, "failures": failures, "arrays": arrays,
            "mode_observations": modes,
            "max_abs_error": max((a["max_abs_error"] for a in arrays), default=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("arrays", "mode_observations")}))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
