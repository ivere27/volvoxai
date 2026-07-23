#!/usr/bin/env python3
"""Profile TinyReceipt split packages with the dedicated native C host.

Every invocation uses the same strict deployment path: native CPU, incremental
decode, required row execution, and application-only timing without per-node
runtime tracing. Accuracy runs always use the complete 191-token generation
budget. Shorter runs are explicitly latency-only and never publish a
ground-truth score.

The JSON report intentionally contains content identities and package-relative
asset names, not machine-local dataset, package, or executable paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


REPORT_FORMAT = "volvoxai.tiny-receipt-native-heldout-profile/v1"
PACKAGE_FORMAT = "volvoxai-tiny-receipt-vqa-split-onnx-package-v1"
FULL_MAX_NEW = 191
DEFAULT_COUNT = 10
DEFAULT_BINARY = Path("examples/target/bin/tiny_receipt_split_w8a8")

_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NUMBER = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
_ROUTER_RE = re.compile(
    rf"^\[debug\] tinyreceipt split router=([A-Za-z0-9_]+) "
    rf"selected=([A-Za-z0-9_]+)(?: \((?:requested|verified)\))? "
    rf"encoder=({_NUMBER}) ms$"
)
_GENERATION_RE = re.compile(
    rf"^\[debug\] tinyreceipt split family=([A-Za-z0-9_]+) "
    rf"tokens=([0-9]+) total=({_NUMBER}) ms tok/s=({_NUMBER}) "
    rf"\(incremental retained execution context\)$"
)
_TIMING_RE = re.compile(
    rf"^\[debug\] tinyreceipt split timing encoder=({_NUMBER}) ms "
    rf"(first|seed)=({_NUMBER}) ms steady_steps=([0-9]+) "
    rf"steady_mean=({_NUMBER}) ms steady_tok/s=({_NUMBER})$"
)
_STDOUT_PREFIX = "VolvoxAI Native Runtime\nBackend policy: cpu\n"


class ProfileError(RuntimeError):
    """The dataset, package, native execution, or evidence was invalid."""


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True)
class FileIdentity:
    path: str
    bytes: int
    sha256: str

    def public(self) -> dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PackageSpec:
    label: str
    root: Path
    identity: Mapping[str, object]


@dataclass(frozen=True)
class HeldoutCase:
    case_id: str
    question: str
    truth: str
    image: Path
    annotation_identity: Mapping[str, object]
    image_identity: Mapping[str, object]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ProfileError(f"could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProfileError(f"{label} must contain a JSON object")
    return value


def _read_file(path: Path, label: str) -> bytes:
    try:
        if not path.is_file():
            raise OSError("not a regular file")
        return path.read_bytes()
    except OSError as error:
        raise ProfileError(f"could not read {label}: {error}") from error


def _identity(data: bytes, relative_path: str) -> FileIdentity:
    return FileIdentity(
        path=relative_path,
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProfileError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ProfileError(f"{label} must stay inside its package")
    return path.as_posix()


def _descriptor_identity(
    root: Path, descriptor: Any, label: str
) -> FileIdentity:
    if not isinstance(descriptor, dict):
        raise ProfileError(f"{label} must be an asset descriptor")
    relative = _safe_relative_path(descriptor.get("path"), f"{label}.path")
    declared_bytes = descriptor.get("bytes")
    declared_sha256 = descriptor.get("sha256")
    if (
        isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes < 0
    ):
        raise ProfileError(f"{label}.bytes must be a non-negative integer")
    if not isinstance(declared_sha256, str) or not _SHA256_RE.fullmatch(
        declared_sha256
    ):
        raise ProfileError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ProfileError(f"{label}.path escapes its package") from error
    data = _read_file(resolved, label)
    actual = _identity(data, relative)
    if actual.bytes != declared_bytes or actual.sha256 != declared_sha256:
        raise ProfileError(f"{label} does not match its package manifest identity")
    return actual


def _load_package(argument: str) -> PackageSpec:
    explicit = "=" in argument
    if explicit:
        label, raw_path = argument.split("=", 1)
    else:
        raw_path = argument
        label = Path(raw_path).name
    if not _LABEL_RE.fullmatch(label):
        raise ProfileError(
            f"package label {label!r} must match {_LABEL_RE.pattern!r}"
        )
    root = Path(raw_path)
    if not root.is_dir():
        raise ProfileError(f"package {label!r} is not a directory")
    manifest_data = _read_file(root / "package_manifest.json", f"{label} manifest")
    manifest = _read_json_bytes(manifest_data, f"{label} manifest")
    if manifest.get("format") != PACKAGE_FORMAT:
        raise ProfileError(f"package {label!r} has an unsupported manifest format")
    graphs = manifest.get("graphs")
    assets = manifest.get("assets")
    if not isinstance(graphs, dict) or not isinstance(assets, dict):
        raise ProfileError(f"package {label!r} must declare graphs and assets")

    public_graphs: dict[str, object] = {}
    for graph_name in ("encoder", "decoder"):
        graph = graphs.get(graph_name)
        if not isinstance(graph, dict):
            raise ProfileError(f"package {label!r} is missing graph {graph_name!r}")
        public_graphs[graph_name] = {
            "graph": _descriptor_identity(
                root, graph.get("graph"), f"{label}.{graph_name}.graph"
            ).public(),
            "weights": _descriptor_identity(
                root, graph.get("weights"), f"{label}.{graph_name}.weights"
            ).public(),
            "export_report": _descriptor_identity(
                root,
                graph.get("export_report"),
                f"{label}.{graph_name}.export_report",
            ).public(),
        }

    public_assets = {
        name: _descriptor_identity(root, assets.get(name), f"{label}.{name}").public()
        for name in ("config", "vocab")
    }
    variant = manifest.get("variant")
    routing = manifest.get("routing")
    generation = manifest.get("generation")
    public_variant: dict[str, object] = {}
    if isinstance(variant, dict):
        for key in (
            "requested",
            "producer_key",
            "export_quant_mode",
            "complete_w8a8_fusion",
        ):
            value = variant.get(key)
            if isinstance(value, (str, bool, int, float)):
                public_variant[key] = value
    routing_mode = routing.get("mode") if isinstance(routing, dict) else None
    if not isinstance(routing_mode, str):
        routing_mode = None
    decoder_output = (
        generation.get("decoder_output") if isinstance(generation, dict) else None
    )
    if not isinstance(decoder_output, str):
        decoder_output = None
    identity: dict[str, object] = {
        "label": label,
        "format": PACKAGE_FORMAT,
        "manifest": _identity(manifest_data, "package_manifest.json").public(),
        "assets": public_assets,
        "graphs": public_graphs,
        "routing_mode": routing_mode,
        "decoder_output": decoder_output,
        "variant": public_variant,
    }
    return PackageSpec(label=label, root=root, identity=identity)


def _load_executable_identity(binary: Path) -> Mapping[str, object]:
    data = _read_file(binary, "native split executable")
    if not os.access(binary, os.X_OK):
        raise ProfileError("native split executable is not executable")
    result = _identity(data, binary.name).public()
    result["name"] = result.pop("path")
    return result


def _answer_string(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        raise ProfileError(f"{label} must be a string or finite number")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    raise ProfileError(f"{label} must be a string or finite number")


def _load_case(annotation_path: Path, eval_root: Path) -> HeldoutCase:
    annotation_data = _read_file(annotation_path, f"annotation {annotation_path.name}")
    record = _read_json_bytes(annotation_data, f"annotation {annotation_path.name}")
    case_id = record.get("id")
    question = record.get("question")
    if not isinstance(case_id, str) or not _ID_RE.fullmatch(case_id):
        raise ProfileError(f"annotation {annotation_path.name} has an invalid id")
    if annotation_path.stem != case_id:
        raise ProfileError(
            f"annotation {annotation_path.name} id does not match its filename"
        )
    if not isinstance(question, str) or not question or "\0" in question:
        raise ProfileError(f"annotation {case_id} has an invalid question")
    truth = _answer_string(record.get("answer"), f"annotation {case_id}.answer")
    image = eval_root / "images" / f"{case_id}.jpg"
    image_data = _read_file(image, f"heldout image {case_id}")
    return HeldoutCase(
        case_id=case_id,
        question=question,
        truth=truth,
        image=image,
        annotation_identity=_identity(
            annotation_data, f"annotations/{annotation_path.name}"
        ).public(),
        image_identity=_identity(image_data, f"images/{image.name}").public(),
    )


def load_cases(
    eval_root: Path, count: int | None, requested_ids: Sequence[str] | None
) -> list[HeldoutCase]:
    annotations = eval_root / "annotations"
    if not annotations.is_dir():
        raise ProfileError("heldout root must contain an annotations directory")
    names = sorted(path for path in annotations.iterdir() if path.suffix == ".json")
    by_id = {path.stem: path for path in names}
    if len(by_id) != len(names):
        raise ProfileError("heldout annotations contain duplicate filename identities")

    if requested_ids is not None:
        if not requested_ids:
            raise ProfileError("--ids must contain at least one heldout ID")
        if len(set(requested_ids)) != len(requested_ids):
            raise ProfileError("--ids may not contain duplicates")
        for case_id in requested_ids:
            if not _ID_RE.fullmatch(case_id):
                raise ProfileError(f"invalid heldout ID {case_id!r}")
        missing = [case_id for case_id in requested_ids if case_id not in by_id]
        if missing:
            raise ProfileError(
                f"requested heldout IDs not found: {', '.join(missing)}"
            )
        selected = [by_id[case_id] for case_id in requested_ids]
    else:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ProfileError("--count must be a positive integer")
        selected = names[:count]
        if len(selected) < count:
            raise ProfileError(
                f"requested {count} heldout cases but found only {len(selected)}"
            )
    return [_load_case(path, eval_root) for path in selected]


def validate_generation_settings(max_new: int, latency_only: bool) -> None:
    if isinstance(max_new, bool) or not isinstance(max_new, int):
        raise ProfileError("--max-new must be an integer")
    if max_new < 1 or max_new > FULL_MAX_NEW:
        raise ProfileError(f"--max-new must be in [1, {FULL_MAX_NEW}]")
    if max_new != FULL_MAX_NEW and not latency_only:
        raise ProfileError(
            f"accuracy profiling requires --max-new {FULL_MAX_NEW}; "
            "pass --latency-only for a shorter generation"
        )


def extract_structured_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if len(matches) != 1 or "<" in matches[0]:
        return None
    return matches[0].strip()


def _one_match(pattern: re.Pattern[str], lines: Sequence[str], label: str) -> re.Match[str]:
    matches = [match for line in lines if (match := pattern.fullmatch(line))]
    if len(matches) != 1:
        raise ProfileError(f"native debug output must contain exactly one {label} line")
    return matches[0]


def parse_native_output(
    stdout: bytes, stderr: bytes, max_new: int, latency_only: bool
) -> Mapping[str, object]:
    try:
        stdout_text = stdout.decode("utf-8")
        stderr_text = stderr.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProfileError("native output was not valid UTF-8") from error
    if not stdout_text.startswith(_STDOUT_PREFIX) or not stdout_text.endswith("\n"):
        raise ProfileError("native stdout did not match the TinyReceipt CPU banner contract")
    generated_text = stdout_text[len(_STDOUT_PREFIX) : -1]
    lines = stderr_text.splitlines()
    router = _one_match(_ROUTER_RE, lines, "router timing")
    generation = _one_match(_GENERATION_RE, lines, "generation timing")
    timing = _one_match(_TIMING_RE, lines, "encoder/decoder timing")

    router_family, selected_family = router.group(1), router.group(2)
    generation_family = generation.group(1)
    generated_tokens = int(generation.group(2))
    steady_steps = int(timing.group(4))
    if generation_family != selected_family:
        raise ProfileError("native generation family differs from the selected family")
    if float(router.group(3)) != float(timing.group(1)):
        raise ProfileError("native encoder timing lines disagree")
    if generated_tokens > max_new:
        raise ProfileError("native generation exceeded --max-new")
    if generated_tokens not in (steady_steps, steady_steps + 1):
        raise ProfileError("native token and steady-step counts are inconsistent")
    limit_exhausted = generated_tokens == max_new
    if limit_exhausted and not latency_only:
        raise ProfileError(
            "native generation exhausted all 191 tokens without EOS; "
            "accuracy would be truncated, so rerun only with --latency-only"
        )

    return {
        "generated_text": generated_text,
        "answer": extract_structured_answer(generated_text),
        "router_family": router_family,
        "selected_family": selected_family,
        "generated_tokens": generated_tokens,
        "limit_exhausted": limit_exhausted,
        "generation_total_ms": float(generation.group(3)),
        "generation_tokens_per_second": float(generation.group(4)),
        "encoder_ms": float(timing.group(1)),
        "first_kind": timing.group(2),
        "first_ms": float(timing.group(3)),
        "steady_steps": steady_steps,
        "steady_mean_ms": float(timing.group(5)),
        "steady_tokens_per_second": float(timing.group(6)),
    }


def _decode_failure_output(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")[-2000:].strip()


def run_native_once(
    binary: Path,
    package: PackageSpec,
    test_case: HeldoutCase,
    max_new: int,
    latency_only: bool,
    timeout_seconds: float,
    threads: int = 1,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
    clock_ns: Callable[[], int] | None = None,
) -> Mapping[str, object]:
    runner = subprocess.run if runner is None else runner
    clock_ns = time.perf_counter_ns if clock_ns is None else clock_ns
    command = [
        str(binary),
        str(package.root),
        "--image",
        str(test_case.image),
        "--prompt",
        test_case.question,
        "--family",
        "auto",
        "--max-new",
        str(max_new),
        "--incremental",
        "--cpu",
        "--threads",
        str(threads),
        "--require-row",
        "--timing",
    ]
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    started = clock_ns()
    try:
        completed = runner(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise ProfileError(
            f"native run timed out for case {test_case.case_id}, package {package.label}"
        ) from error
    except OSError as error:
        raise ProfileError(f"could not execute the native split application: {error}") from error
    wall_ms = round((clock_ns() - started) / 1_000_000.0, 3)
    stdout = completed.stdout
    stderr = completed.stderr
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        raise ProfileError("native subprocess runner must return byte output")
    if completed.returncode != 0:
        detail = _decode_failure_output(stderr)
        suffix = f": {detail}" if detail else ""
        raise ProfileError(
            f"native run failed for case {test_case.case_id}, "
            f"package {package.label} (exit {completed.returncode}){suffix}"
        )
    parsed = dict(parse_native_output(stdout, stderr, max_new, latency_only))
    parsed["wall_ms"] = wall_ms
    return parsed


def _summary(values: Sequence[float]) -> Mapping[str, float | int]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ProfileError("timing samples must be finite non-negative numbers")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[math.floor((len(ordered) - 1) * fraction)]

    return {
        "samples": len(values),
        "min": min(values),
        "p10": percentile(0.10),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(0.90),
        "max": max(values),
    }


def _package_timing_summary(
    case_results: Sequence[Mapping[str, object]], label: str
) -> Mapping[str, object]:
    metric_names = (
        "wall_ms",
        "encoder_ms",
        "first_ms",
        "steady_mean_ms",
        "generation_total_ms",
        "generation_tokens_per_second",
        "steady_tokens_per_second",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for result in case_results:
        artifact = result["artifacts"][label]  # type: ignore[index]
        for run in artifact["runs"]:  # type: ignore[index]
            for name in metric_names:
                values[name].append(float(run[name]))
    return {name: _summary(samples) for name, samples in values.items()}


def build_report(
    *,
    eval_root: Path,
    package_arguments: Sequence[str],
    binary: Path,
    count: int | None = DEFAULT_COUNT,
    requested_ids: Sequence[str] | None = None,
    max_new: int = FULL_MAX_NEW,
    latency_only: bool = False,
    repeat: int = 1,
    threads: int = 1,
    timeout_seconds: float = 600.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
    clock_ns: Callable[[], int] | None = None,
) -> Mapping[str, object]:
    validate_generation_settings(max_new, latency_only)
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ProfileError("--repeat must be a positive integer")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ProfileError("--threads must be a positive integer")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ProfileError("--timeout-seconds must be a finite positive number")
    if not package_arguments:
        raise ProfileError("at least one --package is required")
    packages = [_load_package(argument) for argument in package_arguments]
    labels = [package.label for package in packages]
    if len(set(labels)) != len(labels):
        raise ProfileError("--package labels must be unique")
    executable_identity = _load_executable_identity(binary)
    cases = load_cases(eval_root, count, requested_ids)

    case_results: list[dict[str, object]] = []
    for test_case in cases:
        artifacts: dict[str, object] = {}
        for package in packages:
            runs: list[Mapping[str, object]] = []
            for repeat_index in range(repeat):
                run = dict(
                    run_native_once(
                        binary,
                        package,
                        test_case,
                        max_new,
                        latency_only,
                        timeout_seconds,
                        threads,
                        runner=runner,
                        clock_ns=clock_ns,
                    )
                )
                run["repeat"] = repeat_index + 1
                runs.append(run)
            texts = {str(run["generated_text"]) for run in runs}
            if len(texts) != 1:
                raise ProfileError(
                    f"package {package.label} produced nondeterministic text for "
                    f"case {test_case.case_id}"
                )
            families = {str(run["selected_family"]) for run in runs}
            if len(families) != 1:
                raise ProfileError(
                    f"package {package.label} selected an unstable family for "
                    f"case {test_case.case_id}"
                )
            generated_text = str(runs[0]["generated_text"])
            answer = runs[0]["answer"]
            artifacts[package.label] = {
                "generated_text": generated_text,
                "answer": answer,
                "exact_match": None if latency_only else answer == test_case.truth,
                "selected_family": runs[0]["selected_family"],
                "runs": runs,
            }

        answers = [artifacts[label]["answer"] for label in labels]  # type: ignore[index]
        texts = [artifacts[label]["generated_text"] for label in labels]  # type: ignore[index]
        answer_agreement = answers[0] is not None and all(
            answer == answers[0] for answer in answers[1:]
        )
        structured_agreement = all(text == texts[0] for text in texts[1:])
        case_results.append(
            {
                "id": test_case.case_id,
                "question": test_case.question,
                "truth": test_case.truth,
                "data": {
                    "annotation": test_case.annotation_identity,
                    "image": test_case.image_identity,
                },
                "artifacts": artifacts,
                "cross_artifact_agreement": {
                    "answer": answer_agreement,
                    "structured_text": structured_agreement,
                },
            }
        )

    package_summaries: dict[str, object] = {}
    for package in packages:
        correct = sum(
            result["artifacts"][package.label]["exact_match"] is True  # type: ignore[index]
            for result in case_results
        )
        package_summaries[package.label] = {
            "exact_match": (
                None
                if latency_only
                else {
                    "correct": correct,
                    "total": len(case_results),
                    "rate": correct / len(case_results),
                }
            ),
            "timing": _package_timing_summary(case_results, package.label),
        }

    answer_agreement_count = sum(
        result["cross_artifact_agreement"]["answer"] is True  # type: ignore[index]
        for result in case_results
    )
    structured_agreement_count = sum(
        result["cross_artifact_agreement"]["structured_text"] is True  # type: ignore[index]
        for result in case_results
    )
    return {
        "format": REPORT_FORMAT,
        "settings": {
            "backend": "cpu",
            "cpu_threads": threads,
            "decode": "incremental-row-required",
            "application_timing": True,
            "runtime_node_trace": False,
            "case_selection": (
                "explicit-ids" if requested_ids is not None else "sorted-annotations"
            ),
            "case_count": len(cases),
            "repeat": repeat,
            "max_new": max_new,
            "accuracy_mode": "latency-only" if latency_only else "full-generation",
            "package_order": labels,
        },
        "provenance": {
            "executable": executable_identity,
            "packages": [package.identity for package in packages],
            "heldout_cases": [
                {
                    "id": test_case.case_id,
                    "annotation": test_case.annotation_identity,
                    "image": test_case.image_identity,
                }
                for test_case in cases
            ],
        },
        "summary": {
            "packages": package_summaries,
            "cross_artifact_agreement": {
                "answer": {
                    "agree": answer_agreement_count,
                    "total": len(case_results),
                },
                "structured_text": {
                    "agree": structured_agreement_count,
                    "total": len(case_results),
                },
            },
        },
        "cases": case_results,
    }


def _parse_ids(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("must contain at least one comma-separated ID")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        required=True,
        type=Path,
        help="heldout root containing annotations/ and images/",
    )
    parser.add_argument(
        "--package",
        action="append",
        required=True,
        metavar="[LABEL=]DIR",
        help="split package to profile; repeat for artifact comparison",
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=_positive_int)
    selection.add_argument("--ids", type=_parse_ids)
    parser.add_argument("--max-new", type=_positive_int, default=FULL_MAX_NEW)
    parser.add_argument("--repeat", type=_positive_int, default=1)
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=1,
        help="native CPU worker count (default: 1)",
    )
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=_positive_float, default=600.0)
    parser.add_argument("--report", type=Path, help="write JSON here; default stdout")
    return parser


def _print_human_summary(report: Mapping[str, object]) -> None:
    summary = report["summary"]
    settings = report["settings"]
    package_summaries = summary["packages"]  # type: ignore[index]
    for label in settings["package_order"]:  # type: ignore[index]
        package = package_summaries[label]
        wall = package["timing"]["wall_ms"]  # type: ignore[index]
        accuracy = package["exact_match"]  # type: ignore[index]
        accuracy_text = "latency-only"
        if accuracy is not None:
            accuracy_text = f"EM {accuracy['correct']}/{accuracy['total']}"
        print(
            f"{label}: wall median {wall['median']:.3f} ms, {accuracy_text}",
            file=sys.stderr,
        )
    agreement = summary["cross_artifact_agreement"]  # type: ignore[index]
    print(
        "cross-artifact: answers "
        f"{agreement['answer']['agree']}/{agreement['answer']['total']}, "
        "structured text "
        f"{agreement['structured_text']['agree']}/"
        f"{agreement['structured_text']['total']}",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    count = arguments.count if arguments.count is not None else DEFAULT_COUNT
    if arguments.ids is not None:
        count = None
    try:
        report = build_report(
            eval_root=arguments.eval,
            package_arguments=arguments.package,
            binary=arguments.binary,
            count=count,
            requested_ids=arguments.ids,
            max_new=arguments.max_new,
            latency_only=arguments.latency_only,
            repeat=arguments.repeat,
            threads=arguments.threads,
            timeout_seconds=arguments.timeout_seconds,
        )
        encoded = json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        if arguments.report is None:
            sys.stdout.write(encoded)
        else:
            arguments.report.parent.mkdir(parents=True, exist_ok=True)
            arguments.report.write_text(encoded, encoding="utf-8")
        _print_human_summary(report)
    except (ProfileError, OSError, UnicodeError) as error:
        print(f"[tinyreceipt-native-profile] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
