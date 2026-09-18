#!/usr/bin/env python3
"""Replace each example's latest accuracy and complete-request latency records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
BEGIN = "<!-- BEGIN CURRENT PROTO BENCHMARK -->"
END = "<!-- END CURRENT PROTO BENCHMARK -->"
VARIANTS = ["fp32", "int8", "ptq", "ptq-wasm"]
BACKENDS = ["cpu", "cuda", "vulkan", "opengl", "wasm", "webgpu"]
DOCUMENTS = [
    ("digit", "BENCHMARK.md", "receipt_digit_reader"),
    ("vqa", "BENCHMARK.md", "tiny_receipt_vqa"),
]


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_files(spec, model):
    paths = {c[k] for v in spec["models"][model]["variants"].values()
             for c in v.values() for k in ["graph", "weights"]}
    paths.update(p for v in spec["models"][model]["onnx"].values() for p in v.values())
    return {path: value for path, value in spec["file_hashes"].items() if path in paths}


def sample_stats(values):
    ordered = sorted(values)
    at = (len(ordered) - 1) * .95
    lo = int(at)
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms": ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (at - lo),
            "mean_ms": statistics.mean(values), "min_ms": min(values), "max_ms": max(values)}


def publish_accuracy(args, spec, accuracy):
    begin = "<!-- BEGIN CURRENT PROTO VALIDATION -->"
    end = "<!-- END CURRENT PROTO VALIDATION -->"
    for model, document, example in DOCUMENTS:
        measured = [r for r in accuracy["matrix"] if r["model"] == model]
        assert len(measured) == 44 and all(r["samples"] == 2000 for r in measured)
        report_name = "validation"
        report_path = ROOT / "examples" / example / "reports" / (report_name + ".json")
        report = {k: accuracy[k] for k in ["base_head", "working_tree_fixes_applied", "updated_utc", "artifacts", "libraries", "corpora"]}
        report.update(schema="volvoxai.receipt-proto-validation-publication/v1", model=model,
                      source_summary_sha256=digest(args.accuracy), cells=44, samples_per_cell=2000,
                      model_files=model_files(spec, model),
                      matrix=[{k: v for k, v in r.items() if k != "seconds"} for r in measured])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        relative_report = f"reports/{report_name}.json"
        lines = [begin, f"## Accuracy — {accuracy['updated_utc'][:10]}", "",
                 f"The latest audit uses HEAD `{spec['base_head']}` plus the staged PTQ/runtime fixes,",
                 f"with exact deployed artifact hashes in the [44-row validation record]({relative_report}).",
                 "Every backend/variant/profile cell below completed **all 2,000 held-out cases**.",
                 "Native CPU, CUDA, Vulkan, OpenGL and WASM passed in both inference and full profiles;",
                 "WebGPU passed in full. Their paired profiles produce identical digit logits or VQA",
                 "tokens, answers and router choices across all 2,000 cases. Final routes are attested",
                 "with no operator fallback. This is 44 completed cells per model, 88 for both models.", "",
                 "Imported INT8 uses the producer's quantized ONNX. Native C PTQ and WASM C PTQ",
                 "are separately calibrated VolvoxAI packages made from the FP32 source.", "",
                 ("Accuracy is strict `target_exact`: every phone and street digit must be correct on the receipt."
                  if model == "digit" else "Accuracy is exact agreement between the extracted answer and the held-out answer label."), "",
                 "| Backend | FP32 | Imported INT8 | Native C PTQ | WASM C PTQ |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for backend in BACKENDS:
            values = []
            for variant in VARIANTS:
                rows = [r for r in measured if r["backend"] == backend and r["variant"] == variant]
                scores = [round(r["score"]["target_exact"] * 2000) if model == "digit"
                          else r["score"]["answer_exact"] for r in rows]
                assert len(set(scores)) == 1
                values.append(f"{scores[0]}/2000 ({scores[0] / 20:.2f}%)")
            lines.append("| " + " | ".join([backend, *values]) + " |")
        lines += ["", "The WebGPU result records successful serial execution with a patched Deno 2.9.6",
                  "host and a physical RTX 3090. Browser products and simultaneous WebGPU jobs",
                  "were not qualified by this audit."]
        if model == "vqa":
            row = next(r for r in measured if (r["backend"], r["profile"], r["variant"]) == ("cpu", "inference", "int8"))
            agreement = row["source_ort_agreement"]
            lines += ["", f"For imported INT8 on native CPU, whole-token agreement with source ORT is {agreement['whole_tokens_equal']}/2000",
                      f"({agreement['whole_tokens_equal'] / 20:.1f}%), while extracted-answer agreement is {agreement['answer_equal']}/2000",
                      f"({agreement['answer_equal'] / 20:.1f}%). These are output-equivalence metrics, separate from label accuracy.",
                      "Differences in the generated `<value>` field can lower whole-sequence agreement while",
                      "leaving the final answer correct. INT8 import changes bias folding and quantized arithmetic;",
                      "their separate numerical contributions have not been isolated by this audit."]
        lines += ["", "Accuracy evaluation wall time is not inference latency. The separate speed",
                  "experiment retains sessions and repeats fixed inputs under the timing contract above.",
                  "See the [benchmark workflow](benchmarks/README.md) for reproduction.", "", end]
        doc = ROOT / "examples" / example / document
        previous = doc.read_text()
        if begin in previous:
            before, rest = previous.split(begin, 1)
            _, after = rest.split(end, 1)
            updated = before + "\n".join(lines) + after
        else:
            title, rest = previous.split("\n", 1)
            updated = title + "\n\n" + "\n".join(lines) + "\n" + rest
        doc.write_text(updated)


CPU_BACKENDS = ["ort", "cpu", "wasm"]
GPU_BACKENDS = ["cuda", "vulkan", "opengl", "webgpu"]
HOST_FIELDS = {"host_label", "pinned_cpu", "max_foreign_cpu_cores", "deno_sha256", "node_sha256",
               "allow_busy_host", "ort_reference_sha256", "controller_sha256", "spec_sha256"}
LABELS = {"ort": "ORT CPU / Python", "cpu": "Native CPU / Python → C", "wasm": "WASM / Node 20.11.1",
          "cuda": "CUDA / Python → C", "vulkan": "Vulkan / Python → C", "opengl": "OpenGL / Python → C",
          "webgpu": "WebGPU / Deno 2.9.6"}


def read_rows(directory, spec_path):
    spec = load(spec_path)
    schema = "volvoxai.receipt-proto-latency/v1"
    rows = {}
    for path in directory.glob("*.json"):
        row = load(path)
        if row.get("schema") != schema:
            continue
        assert row["measurement_completed"] and row["conditions"] and row["host"]
        assert row["profile"] == ("full" if row["backend"] == "webgpu" else "inference")
        assert row["spec_sha256"] == digest(spec_path) and row["repeated_outputs_identical"]
        key = "/".join(row[k] for k in ["model", "backend", "profile", "variant"])
        assert row["case_indices"] == [c["index"] for c in spec["cases"]]
        for index, case in enumerate(spec["cases"]):
            expected = spec["expected"][key][str(case["index"])]
            signature = hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest()
            assert row["output_sha256"][str(index)] == signature
        assert len(row["samples"]) == row["repeat_per_case"] * len(spec["cases"])
        for sample in row["samples"]:
            assert all(math.isfinite(v) and v >= 0 for v in sample.values())
            assert sample["request_ms"] >= sample["component_ms"] > 0
        for metric, values in row["metrics"].items():
            actual = sample_stats([s[metric] for s in row["samples"]])
            assert all(math.isclose(values[k], v, rel_tol=1e-9) for k, v in actual.items())
        row["by_case"] = {}
        for index in row["case_indices"]:
            samples = [s for s in row["samples"] if s["case_index"] == index]
            assert len(samples) == row["repeat_per_case"]
            row["by_case"][str(index)] = {m: sample_stats([s[m] for s in samples]) for m in row["metrics"]}
            if row["model"] == "vqa":
                lengths = {s["decoder_steps"] for s in samples}
                assert len(lengths) == 1
                row["by_case"][str(index)]["decoder_steps"] = lengths.pop()
        if row["backend"] == "ort":
            assert row["onnxruntime"] == "1.23.2"
        elif row["backend"] in ["wasm", "webgpu"]:
            assert all(spec["artifacts"][k] == v for k, v in row["artifacts"].items())
        else:
            assert row["library_sha256"] == spec["libraries"][row["profile"]]
        row["output_matches_benchmark_reference"] = row.get(
            "output_matches_benchmark_reference", row["output_matches_2000_reference"])
        assert row["output_matches_benchmark_reference"]
        assert row["output_matches_2000_reference"]
        index = (row["model"], row["backend"], row["variant"])
        assert index not in rows
        row["host"] = {k: v for k, v in row["host"].items() if k in HOST_FIELDS}
        row["metadata_redacted"] = True
        rows[index] = row
    return rows



def set_block(path, begin, end, lines):
    previous = path.read_text()
    block = "\n".join([begin, *lines, end])
    if begin in previous:
        before, rest = previous.split(begin, 1)
        _, after = rest.split(end, 1)
        updated = before + block + after
    else:
        title, rest = previous.split("\n", 1)
        updated = title + "\n\n" + block + "\n" + rest
    path.write_text(updated)


def latest_rows(directories, spec_path):
    rows = {}
    for directory in directories:
        for key, row in read_rows(directory, spec_path).items():
            previous = rows.get(key)
            ended = datetime.fromisoformat(row["ended_utc"])
            previous_ended = datetime.fromisoformat(previous["ended_utc"]) if previous else None
            if previous is None or ended > previous_ended:
                rows[key] = row
            elif ended == previous_ended:
                assert row == previous, f"conflicting measurements at the same time: {key}"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--local-results", type=Path, nargs="+")
    parser.add_argument("--gpu-results", type=Path, nargs="+",
                        help="measurement directories; newest completed row wins per model/backend/package")
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()
    spec, accuracy = load(args.spec), load(args.accuracy)
    assert accuracy["matrix_complete"] and accuracy["complete_cells"] == 88 and not accuracy["invalid"]
    assert digest(args.accuracy) == spec["validation_sha256"]
    if args.accuracy_only:
        publish_accuracy(args, spec, accuracy)
        return
    assert args.local_results and args.gpu_results
    local, gpu = latest_rows(args.local_results, args.spec), latest_rows(args.gpu_results, args.spec)
    expected_local = {(m, b, v) for m, _, _ in DOCUMENTS for b in CPU_BACKENDS
                      for v in (["fp32", "int8"] if b == "ort" else VARIANTS)}
    expected_gpu = {(m, b, v) for m, _, _ in DOCUMENTS for b in GPU_BACKENDS for v in VARIANTS}
    assert set(local) == expected_local, "CPU/WASM publication requires exactly the 20 local rows"
    assert set(gpu) == expected_gpu, "GPU publication requires exactly the 32 GPU rows; remote CPU/WASM is excluded"
    for row in local.values():
        assert row["host"]["host_label"] == "cpu-wasm-host"
        if row["backend"] == "wasm":
            assert row["runtime"]["node"] == "v20.11.1"
    for row in gpu.values():
        assert row["host"]["host_label"] == "gpu-host"
        assert row["host_quiet_verified"], "contended GPU latency must not be published"
        assert not row["host"]["allow_busy_host"]
    rows = local | gpu
    assert all(r["cpu_threads"] == 1 and r["host"]["pinned_cpu"] == 0 for r in rows.values())
    counts = {(r["warmup_per_case"], r["repeat_per_case"], tuple(r["case_indices"])) for r in rows.values()}
    assert len(counts) == 1
    warmup, repeat, cases = counts.pop()
    publish_accuracy(args, spec, accuracy)
    for model, document, example in DOCUMENTS:
        measured = [rows[model, b, v] for b in [*CPU_BACKENDS, *GPU_BACKENDS]
                    for v in (["fp32", "int8"] if b == "ort" else VARIANTS)]
        report_path = ROOT / "examples" / example / "reports/benchmark.json"
        report = {"schema": "volvoxai.receipt-proto-benchmark-publication/v1",
                  "published_utc": datetime.now(timezone.utc).isoformat(),
                  "base_head": spec["base_head"], "working_tree_ptq_runtime_fixes": True,
                  "measurement_scope": "local-cpu-wasm-and-idle-remote-gpu",
                  "host_quiet_verified": all(r["host_quiet_verified"] for r in measured),
                  "execution_hosts": {"cpu-wasm-host": CPU_BACKENDS, "gpu-host": GPU_BACKENDS},
                  "metadata_redacted": True,
                  "accuracy_validation_sha256": digest(args.accuracy), "benchmark_spec_sha256": digest(args.spec),
                  "case_selection": spec["case_selection"],
                  "inputs": [{"index": c["index"], "normalized_pixels_sha256": c["sha256"],
                              "question_tokens": len(c["question_ids"])} for c in spec["cases"]],
                  "artifacts": spec["artifacts"], "native_libraries": spec["libraries"],
                  "model_files": model_files(spec, model),
                  "accuracy": [{k: v for k, v in r.items() if k != "seconds"}
                               for r in accuracy["matrix"] if r["model"] == model], "measurements": measured}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        relative_report = "reports/benchmark.json"
        local_rows = [r for r in measured if r["backend"] in CPU_BACKENDS]
        foreign = [r["conditions"]["during_foreign_cpu_mean"] for r in local_rows]
        measured_date = max(r["ended_utc"] for r in measured)[:10]
        lines = [f"## Latency — {measured_date}", "",
                 f"Measured from HEAD `{spec['base_head']}` plus the staged PTQ/runtime fixes used by the completed",
                 f"2,000-case audit. [Raw samples, output and artifact hashes]({relative_report}) identify the measured bytes.", "",
                 "CPU, WASM and ORT CPU run on the local CPU/WASM host.",
                 "CUDA, Vulkan, OpenGL and WebGPU run on the remote GPU host with an RTX 3090",
                 "and driver 535.309.01. Every retained GPU row passed the before/during/after idle gates.",
                 "Measurement tools record host roles and settings without collecting CPU hardware identities.", "",
                 "Local desktop background activity remained; the local rows retain `host_quiet_verified: false`.",
                 f"Unrelated CPU activity averaged {min(foreign):.2f}–{max(foreign):.2f} cores across these routes.",
                 "The two host groups are separate measurement environments; do not interpret the table as a",
                 "same-machine CPU-versus-GPU comparison. Every route requests one numerical thread and is pinned to CPU 0.", "",
                 f"Held-out inputs {', '.join(map(str, cases))} each receive {warmup} warmups and {repeat} measured requests",
                 f"({repeat * len(cases)} observations per row). All outputs reproduce their saved 2,000-case references",
                 "and remain identical across repetitions. The complete 2,000-case accuracy audit is separate",
                 "from this fixed-input latency experiment.", "",
                 "**Timing:** each retained session starts with preprocessed pixels and question token IDs.",
                 "The request timer includes proto input construction, dispatch, synchronization, owned output reads",
                 "and release. VQA includes the encoder and complete explicit-KV greedy decoding through EOS.",
                 "Loading, compilation, context creation, preprocessing, validation and report I/O are excluded.",
                 "Native uses a generated Python client calling C; WASM/WebGPU use the generated JavaScript client",
                 "and released C/WASM owner. ORT uses its Python CPU session API. This run does not separately time a C-only caller.", "",
                 "WASM uses Node 20.11.1. WebGPU uses the pinned patched",
                 "Deno 2.9.6 executable (Cargo optimization level 1, no LTO). Executable hashes are recorded;",
                 "these WebGPU timings do not qualify a browser product.", "",
                 "### Complete-request latency per item", "",
                 "Each cell is **median / p95 in milliseconds**. Native/WASM use inference; WebGPU uses full.",
                 ("One item is one receipt image." if model == "digit" else "One item is one image/question pair through EOS."), "",
                 "| Host | Backend / client | FP32 | Imported INT8 | Native C PTQ | WASM C PTQ |",
                 "| --- | --- | ---: | ---: | ---: | ---: |"]
        for backend in [*CPU_BACKENDS, *GPU_BACKENDS]:
            values = []
            for variant in VARIANTS:
                row = rows.get((model, backend, variant))
                stats = row["metrics"]["request_ms"] if row else None
                values.append(f"{stats['median_ms']:.2f} / {stats['p95_ms']:.2f}" if stats else "—")
            lines.append("| " + " | ".join(["Local" if backend in CPU_BACKENDS else "GPU host", LABELS[backend], *values]) + " |")
        lines += ["", "The JSON contains the latest qualified measurement for each backend/package cell,",
                  "including per-input statistics, component times, serial requests/s and every observation.",
                  "Each row retains its own measured timestamp and caller hash."]
        if model == "vqa":
            steps = {case: sorted({r["by_case"][str(case)]["decoder_steps"] for r in measured}) for case in cases}
            lines += [f"Decoder calls including EOS: {', '.join(f'input {case}: {values}' for case, values in steps.items())}.",
                      "VolvoxAI uses 20–36 calls; original ORT INT8 on the local host uses 38/20 and matches its saved reference.",
                      "The report also records TTFT and generated tokens/s. Use `by_case` for a fixed input and length."]
            cuda_rows = [r for r in measured if r["backend"] == "cuda"]
            if all("execute_ms" in r["metrics"] for r in cuda_rows):
                lines += ["", "### CUDA execution and output reads", "",
                          "Median milliseconds for one complete answer (encoder plus all decoder steps):", "",
                          "| Package | Execute through READY | ReadOutput | Complete request |",
                          "| --- | ---: | ---: | ---: |"]
                for row in cuda_rows:
                    m = row["metrics"]
                    lines.append(f"| {row['variant']} | {m['execute_ms']['median_ms']:.3f} | "
                                 f"{m['read_output_ms']['median_ms']:.3f} | {m['request_ms']['median_ms']:.3f} |")
                lines += ["", "CUDA uses native BufferView inputs and output destinations. Intermediate steps",
                          "read logits, eight K/V tensors and a mask for the next step; EOS reads only logits.",
                          "READY results need no GetResult. Execute timing excludes ReadOutput and ReleaseResult,",
                          "but includes public dispatch, synchronization and runtime output snapshots. It is not",
                          "GPU kernel-only time. Complete-request timing includes every needed read and cache handoff.",
                          "Other rows retain the caller and timing contract identified by their recorded hashes."]
        lines += ["", "Follow the [benchmark workflow](benchmarks/README.md) to prepare inputs,",
                  "measure the two host groups and publish only the approved results.", ""]
        set_block(ROOT / "examples" / example / document, BEGIN, END, lines)
    print("Published 20 local CPU/WASM rows and 32 idle GPU rows; accuracy remains 88 × 2,000 cases.")


if __name__ == "__main__":
    main()
