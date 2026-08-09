#!/usr/bin/env python3
"""Reproducible explicit-KV benchmark matrix for TinyReceipt v1 packages.

The harness compares the producer ONNX Runtime CPU path, native C CPU, and
VolvoxAI WASM for both FP32 and static INT8.  Every measured
sample is an isolated first request.  Timings cover the encoder execution and
individual one-token decoder executions; model loading, compilation, image
decode, tokenization, and process startup are excluded from the cross-tier
decoder medians.

The default image is generated from a fixed grayscale integer formula at the
model's exact 672x320 input size.  All six matrix entries must emit the same
normalized F32 input hash, family, and greedy token IDs and must prove the same
masked P=1 sentinel followed by P -> P+1 cache transitions before a report is
published.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import platform
import re
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_FORMAT = "volvoxai.tiny-receipt-explicit-kv-cpu-benchmark/v1"
SAMPLE_SCHEMA = "volvoxai.tiny-receipt-explicit-kv-runtime-sample/v1"
SAMPLE_PREFIX = "TINYRECEIPT_EXPLICIT_KV_SAMPLE "
DYNAMIC_QUALIFICATION_SCHEMA = (
    "volvoxai.tiny-receipt-dynamic-shape-qualification/v1"
)
DYNAMIC_QUALIFICATION_PREFIX = "TINYRECEIPT_DYNAMIC_REBIND_QUALIFICATION "
CANONICAL_MAX_NEW = 4
PACKAGE_FORMAT = "volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v1"
SOURCE_FORMAT = "tiny_receipt_vqa_split_kv_onnx_v1"
IMAGE_WIDTH = 672
IMAGE_HEIGHT = 320
MAXIMUM_NEW_TOKENS = 191
MAXIMUM_EXECUTION_WARMUP_RUNS = 20
DEFAULT_PROMPT = "phone number last one"
DEFAULT_FAMILY = "phone"
FAMILY_NAMES = (
    "phone",
    "address",
    "store",
    "item_row",
    "item_math",
    "item_lookup",
    "math",
    "other",
)
CROSS_NAMES = tuple(
    name for layer in range(4) for name in (f"cross_k_{layer}", f"cross_v_{layer}")
)
PAST_NAMES = tuple(
    name for layer in range(4) for name in (f"past_k_{layer}", f"past_v_{layer}")
)
PRESENT_NAMES = tuple(name.replace("past_", "present_") for name in PAST_NAMES)
PUBLICATION_OUTPUT_PATHS = (
    "docs/tiny-receipt-vqa-bpe1536-benchmark.md",
    "examples/tiny_receipt_vqa/README.md",
    "examples/tiny_receipt_vqa/reports/explicit_kv_v1_cuda_matrix.json",
    "examples/tiny_receipt_vqa/reports/explicit_kv_v1_gpu_matrix.json",
    "examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix.json",
    "examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix_6c.json",
)
USER_OWNED_PROVENANCE_EXCLUSIONS = (
    "PLAN.md",
    "docs/frontier-llm-support-plan.md",
    "docs/tiny-receipt-vqa-wasm-int8-plan.md",
    "docs/wip-tinyreceipt-optimization.md",
)
WORKTREE_EXECUTION_SOURCE_SNAPSHOT_FORMAT = (
    "volvoxai.git-worktree-execution-source-content-snapshot/v1"
)
COMMON_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SAMPLE_ISOLATION_SCOPE = (
    "fresh ONNX Runtime encoder/decoder sessions in the persistent Python "
    "harness process; fresh child process and runtime for each native C and "
    "VolvoxAI WASM request"
)
WARMUP_ISOLATION_SCOPE = (
    "discarded full matrices; ONNX Runtime sessions and native C/WASM child "
    "processes are not reused, while the Python harness process, loaded "
    "ONNX Runtime library/global state, OS page cache, and host "
    "thermal/frequency state persist"
)

_ABI_RE = re.compile(
    r"^\[debug\] tinyreceipt split ABI=explicit-kv-v1 routing=(runtime|specialized) "
    r"decoder_output=f32_logits shape_mode=active argmax=host-first-index$",
    re.MULTILINE,
)
_ROUTER_RE = re.compile(
    r"^\[debug\] tinyreceipt split router=([A-Za-z0-9_]+) "
    r"selected=([A-Za-z0-9_]+)(?: \((?:requested|verified)\))? "
    r"encoder=([0-9]+(?:\.[0-9]+)?) ms$",
    re.MULTILINE,
)
_STEP_RE = re.compile(
    r"^\[debug\] tinyreceipt explicit-kv family=([A-Za-z0-9_]+) "
    r"step=([0-9]+) P=([0-9]+) R=([0-9]+) token=([0-9]+) "
    r"time=([0-9]+(?:\.[0-9]+)?) ms$",
    re.MULTILINE,
)
_TOTAL_RE = re.compile(
    r"^\[debug\] tinyreceipt explicit-kv family=([A-Za-z0-9_]+) "
    r"tokens=([0-9]+) total=([0-9]+(?:\.[0-9]+)?) ms$",
    re.MULTILINE,
)
_TOKENS_RE = re.compile(
    r"^\[debug\] tinyreceipt split emitted_token_ids="
    r"(none|[0-9]+(?:,[0-9]+)*)$",
    re.MULTILINE,
)
_QUESTION_TOKENS_RE = re.compile(
    r"^\[debug\] tinyreceipt split question_token_ids="
    r"([0-9]+(?:,[0-9]+)*)$",
    re.MULTILINE,
)
_SHAPE_RE = re.compile(
    r"^\[debug\] tinyreceipt split shape mode=active "
    r"logical_Q=([0-9]+) bound_Q=([0-9]+) "
    r"logical_M=([0-9]+) bound_M=([0-9]+) seed_P=1 maximum_R=([0-9]+)$",
    re.MULTILINE,
)
_INPUT_RE = re.compile(
    r"^\[debug\] tinyreceipt split input_f32_sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)
_WARMUP_RESULT_RE = re.compile(
    r"^WARMUP_RESULT status=pass count=([0-9]+) warmup_timed=0 measured_runs=1 "
    r"same_runtime=1 same_encoder_context=1 same_decoder_context=1 "
    r"strict_no_fallback=1 token_parity=1 cache_parity=1 cache_reset=1 "
    r"family_id=([0-9]+) tokens=([0-9]+) token_digest=([0-9a-f]{16}) "
    r"token_ids=(none|[0-9]+(?:,[0-9]+)*) seed_P=([0-9]+) "
    r"seed_R=([0-9]+) last_P=([0-9]+) last_R=([0-9]+) "
    r"cache_preserved=([01])$",
    re.MULTILINE,
)
_ENCODER_ROUTE_RE = re.compile(
    r"^\[debug\] tinyreceipt split encoder shape (.+)$", re.MULTILINE
)
_DECODER_ROUTE_RE = re.compile(
    r"^\[debug\] tinyreceipt split decoder shape (.+)$", re.MULTILINE
)
_CUDA_DEVICE_RE = re.compile(
    r"^\[CUDA\] device ([0-9]+): (.+) \(compute ([0-9]+)\.([0-9]+)\)$",
    re.MULTILINE,
)
_VULKAN_DEVICE_RE = re.compile(
    r"^\[VolvoxAI GPU\] Vulkan Compute initialized successfully! Device: "
    r"(.+); packed INT8 dot: (enabled|unavailable)$",
    re.MULTILINE,
)
_OPENGL_DEVICE_RE = re.compile(
    r"^\[VolvoxAI GPU\] OpenGL Compute initialized: (.+) / (.+) / (.+)$",
    re.MULTILINE,
)
_ORT_STRICT_CPU_FALLBACK_ERROR_CLAUSES = (
    "assigned to the default CPU EP",
    "fallback to CPU EP has been explicitly disabled",
)
_DYNAMIC_NATIVE_RUN_RE = re.compile(
    r"^DYNAMIC_REBIND_RUN index=([0-9]+) "
    r"mode=(active|maximum-padded) logical_Q=([0-9]+) bound_Q=([0-9]+) "
    r"logical_M=([0-9]+) bound_M=([0-9]+) family_id=([0-9]+) "
    r"tokens=([0-9]+) token_digest=([0-9a-f]{16}) "
    r"token_ids=([0-9]+(?:,[0-9]+)*) seed_P=([0-9]+) seed_R=([0-9]+) "
    r"step_P=([0-9]+) step_R=([0-9]+) cache_preserved=([01])$",
    re.MULTILINE,
)
_DYNAMIC_NATIVE_RESULT_RE = re.compile(
    r"^DYNAMIC_REBIND_RESULT status=pass backend=([a-z0-9_-]+) timed=0 "
    r"same_runtime=1 same_encoder_context=1 same_decoder_context=1 "
    r"strict_no_fallback=1 cpu_threads=([0-9]+)$",
    re.MULTILINE,
)
_DYNAMIC_RUN_LABELS = (
    ("short-before", "active"),
    ("representative-active", "active"),
    ("representative-maximum-padded", "maximum-padded"),
    ("short-after", "active"),
)
_DYNAMIC_CHECKS = {
    "exactOutputShapes",
    "finiteOutputs",
    "cachePrefixPreserved",
    "appendedCacheRowFinite",
    "selectedFamilyParity",
    "tokenParity",
    "encoderGrowShrink",
    "decoderGrowShrink",
}
_NATIVE_DYNAMIC_BACKENDS = ("cpu", "vulkan", "opengl", "cuda")
_SOFTWARE_GPU_RE = re.compile(
    r"\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software renderer|"
    r"software rasterizer|microsoft basic render|cpu)\b",
    re.IGNORECASE,
)


def _token_id_digest(tokens: Sequence[int]) -> int:
    """Match the native example's 64-bit emitted-token sequence digest."""

    value = 1_469_598_103_934_665_603
    for token in tokens:
        value ^= token & 0xFFFFFFFF
        value = (value * 1_099_511_628_211) & 0xFFFFFFFFFFFFFFFF
    return value


def _validate_execution_warmup(value: int, runtime: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAXIMUM_EXECUTION_WARMUP_RUNS
    ):
        raise BenchmarkFailure(
            f"{runtime} execution warmup must be an integer in "
            f"[0, {MAXIMUM_EXECUTION_WARMUP_RUNS}]"
        )


class BenchmarkFailure(RuntimeError):
    """The benchmark contract, execution evidence, or parity check failed."""


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _maximum_encoder_binding(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"B", "Q", "M"}
        and all(_plain_int(value.get(name)) for name in ("B", "Q", "M"))
        and (value["B"], value["Q"], value["M"]) == (1, 192, 402)
    )


def _validate_dynamic_runs(
    runs: Any,
    *,
    max_new: int,
    require_labels: bool,
    expected_family: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize the canonical short/grow/max-padded/shrink qualification."""

    if not isinstance(runs, list) or len(runs) != len(_DYNAMIC_RUN_LABELS):
        raise BenchmarkFailure("dynamic qualification must contain exactly four runs")
    normalized: list[dict[str, Any]] = []
    for index, (raw, (label, mode)) in enumerate(
        zip(runs, _DYNAMIC_RUN_LABELS, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise BenchmarkFailure(f"dynamic qualification run {index} is invalid")
        if require_labels and raw.get("label") != label:
            raise BenchmarkFailure(
                f"dynamic qualification run {index} has the wrong label"
            )
        shape_mode = raw.get("shapeMode")
        active = raw.get("activeShape")
        logical = raw.get("logicalShape")
        tokens = raw.get("tokenIds")
        cache = raw.get("cache")
        family_id = raw.get("familyId")
        question_ids = raw.get("questionTokenIds")
        if "questionTokenIds" in raw and (
            not isinstance(question_ids, list)
            or not question_ids
            or question_ids[-1] != 2
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= 1536
                for token in question_ids
            )
        ):
            raise BenchmarkFailure(
                f"dynamic qualification run {index} has invalid question tokens"
            )
        token_digest = raw.get("tokenDigest")
        if "tokenDigest" in raw and (
            not isinstance(token_digest, str)
            or re.fullmatch(r"[0-9a-f]{16}", token_digest) is None
        ):
            raise BenchmarkFailure(
                f"dynamic qualification run {index} has an invalid token digest"
            )
        if (
            shape_mode != mode
            or not isinstance(active, Mapping)
            or not isinstance(logical, Mapping)
            or not _plain_int(logical.get("B"))
            or logical["B"] != 1
            or not _plain_int(logical.get("Q"))
            or logical["Q"] < 1
            or logical["Q"] > 192
            or not _plain_int(logical.get("M"))
            or logical.get("M") != logical["Q"] + 210
            or not _plain_int(logical.get("T"))
            or logical.get("T") != max_new + 1
            or not all(_plain_int(active.get(name)) for name in ("B", "Q", "M", "T"))
            or active["B"] != 1
            or active.get("T") != max_new + 1
            or not _plain_int(family_id)
            or family_id < 0
            or family_id >= len(FAMILY_NAMES)
            or not isinstance(tokens, list)
            or len(tokens) < 2
            or len(tokens) > max_new
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            )
            or not isinstance(cache, Mapping)
        ):
            raise BenchmarkFailure(
                f"dynamic qualification run {index} lacks bounded shape/token evidence"
            )
        expected_bound = (
            (192, 402) if mode == "maximum-padded"
            else (logical["Q"], logical["M"])
        )
        if (active.get("Q"), active.get("M")) != expected_bound:
            raise BenchmarkFailure(
                f"dynamic qualification run {index} has the wrong active binding"
            )
        transitions = cache.get("transitions")
        if (
            not _plain_int(cache.get("initialPastLength"))
            or cache.get("initialPastLength") != 1
            or not _plain_int(cache.get("finalPastLength"))
            or cache.get("finalPastLength") != len(tokens) + 1
            or not _plain_int(cache.get("sentinelMaskValue"))
            or cache.get("sentinelMaskValue") != 1
            or not isinstance(transitions, list)
            or len(transitions) != len(tokens)
            or any(
                not isinstance(step, Mapping)
                or not all(
                    _plain_int(step.get(name))
                    for name in ("position", "pastLength", "presentLength")
                )
                or step.get("position") != step_index
                or step.get("pastLength") != step_index + 1
                or step.get("presentLength") != step_index + 2
                for step_index, step in enumerate(transitions)
            )
        ):
            raise BenchmarkFailure(
                f"dynamic qualification run {index} has an invalid cache lifecycle"
            )
        normalized.append({
            "label": label,
            "shapeMode": mode,
            "familyId": family_id,
            "tokenIds": list(tokens),
            "activeShape": dict(active),
            "logicalShape": dict(logical),
            "cache": {
                "initialPastLength": 1,
                "finalPastLength": len(tokens) + 1,
                "sentinelMaskValue": 1,
                "transitions": [dict(step) for step in transitions],
            },
            **({"family": raw.get("family")} if "family" in raw else {}),
            **({"requestedFamily": raw.get("requestedFamily")}
               if "requestedFamily" in raw else {}),
            **({"questionTokenIds": list(question_ids)}
               if "questionTokenIds" in raw else {}),
            **({"tokenDigest": token_digest} if "tokenDigest" in raw else {}),
        })
    short_before, representative, maximum_padded, short_after = normalized
    if short_before["logicalShape"]["Q"] >= representative["logicalShape"]["Q"]:
        raise BenchmarkFailure("dynamic qualification did not grow the encoder shape")
    if (
        short_before["logicalShape"] != short_after["logicalShape"]
        or short_before["familyId"] != short_after["familyId"]
        or short_before["tokenIds"] != short_after["tokenIds"]
        or short_before.get("tokenDigest") != short_after.get("tokenDigest")
        or representative["logicalShape"] != maximum_padded["logicalShape"]
        or representative["familyId"] != maximum_padded["familyId"]
        or representative["tokenIds"] != maximum_padded["tokenIds"]
        or representative.get("tokenDigest") != maximum_padded.get("tokenDigest")
    ):
        raise BenchmarkFailure("dynamic qualification grow/shrink parity failed")
    for field in ("family", "requestedFamily", "questionTokenIds"):
        present = [field in run for run in normalized]
        if any(present) and not all(present):
            raise BenchmarkFailure(
                f"dynamic qualification incompletely reports {field}"
            )
    if "questionTokenIds" in short_before:
        if any(
            len(run["questionTokenIds"]) != run["logicalShape"]["Q"]
            for run in normalized
        ):
            raise BenchmarkFailure(
                "dynamic qualification question evidence is inconsistent"
            )
        if (
            short_before["questionTokenIds"] != short_after["questionTokenIds"]
            or representative["questionTokenIds"]
            != maximum_padded["questionTokenIds"]
        ):
            raise BenchmarkFailure("dynamic qualification question parity failed")
    if "family" in short_before:
        if any(
            run["family"] != FAMILY_NAMES[run["familyId"]]
            or run["requestedFamily"] != expected_family
            for run in normalized
        ):
            raise BenchmarkFailure(
                "dynamic qualification family evidence is inconsistent"
            )
    elif expected_family is not None:
        raise BenchmarkFailure("dynamic qualification lacks family evidence")
    return normalized


def _physical_native_dynamic_device(
    stdout: str, stderr: str, backend: str
) -> dict[str, Any] | None:
    """Parse and reject ambiguous/software devices for native GPU proofs."""

    if backend == "cpu":
        return None
    device: dict[str, Any] | None = None
    if backend == "cuda":
        matches = list(dict.fromkeys(_CUDA_DEVICE_RE.findall(stderr)))
        if len(matches) > 1:
            raise BenchmarkFailure(
                "native cuda dynamic qualification reports conflicting devices"
            )
        if len(matches) == 1:
            device = {
                "index": int(matches[0][0]),
                "name": matches[0][1],
                "computeCapability": f"{matches[0][2]}.{matches[0][3]}",
            }
    elif backend == "vulkan":
        matches = list(dict.fromkeys(_VULKAN_DEVICE_RE.findall(stdout)))
        if len(matches) > 1:
            raise BenchmarkFailure(
                "native vulkan dynamic qualification reports conflicting devices"
            )
        if len(matches) == 1:
            device = {
                "name": matches[0][0],
                "packedInt8Dot": matches[0][1] == "enabled",
            }
    elif backend == "opengl":
        matches = list(dict.fromkeys(_OPENGL_DEVICE_RE.findall(stdout)))
        if len(matches) > 1:
            raise BenchmarkFailure(
                "native opengl dynamic qualification reports conflicting devices"
            )
        if len(matches) == 1:
            device = {
                "vendor": matches[0][0],
                "renderer": matches[0][1],
                "version": matches[0][2],
            }
    if device is None:
        raise BenchmarkFailure(
            f"native {backend} dynamic qualification lacks a unique device identity"
        )
    description = " ".join(
        value.strip() for value in device.values()
        if isinstance(value, str) and value.strip()
    )
    if not description or _SOFTWARE_GPU_RE.search(description):
        raise BenchmarkFailure(
            f"native {backend} software adapter is not a GPU benchmark: {description}"
        )
    return device


def parse_native_dynamic_qualification(
    stdout: bytes,
    stderr: bytes,
    *,
    precision: str,
    backend: str,
    max_new: int,
    cpu_threads: int = 1,
) -> dict[str, Any]:
    """Validate one untimed, fresh-process native dynamic-shape proof."""

    if (
        precision not in ("fp32", "int8")
        or backend not in _NATIVE_DYNAMIC_BACKENDS
        or not _plain_int(cpu_threads)
        or cpu_threads <= 0
    ):
        raise BenchmarkFailure("native dynamic qualification identity is invalid")
    try:
        output = stdout.decode("utf-8", errors="strict")
        debug = stderr.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise BenchmarkFailure(
            f"native {backend}/{precision} dynamic qualification is not UTF-8"
        ) from error
    machine_lines = [
        line for line in output.splitlines()
        if line.startswith("DYNAMIC_REBIND_")
    ]
    matches = _DYNAMIC_NATIVE_RUN_RE.findall(output)
    results = _DYNAMIC_NATIVE_RESULT_RE.findall(output)
    if (
        len(machine_lines) != len(_DYNAMIC_RUN_LABELS) + 1
        or len(matches) != len(_DYNAMIC_RUN_LABELS)
        or len(results) != 1
        or results[0] != (backend, str(cpu_threads))
    ):
        raise BenchmarkFailure(
            f"native {backend}/{precision} dynamic qualification lacks unique evidence"
        )
    raw_runs: list[dict[str, Any]] = []
    for expected_index, match in enumerate(matches):
        (
            raw_index,
            mode,
            raw_logical_q,
            raw_bound_q,
            raw_logical_m,
            raw_bound_m,
            raw_family_id,
            raw_token_count,
            token_digest,
            raw_token_ids,
            raw_seed_p,
            raw_seed_r,
            raw_step_p,
            raw_step_r,
            raw_cache_preserved,
        ) = match
        token_ids = [int(value) for value in raw_token_ids.split(",")]
        token_count = int(raw_token_count)
        if (
            int(raw_index) != expected_index
            or token_count != len(token_ids)
            or token_digest != f"{_token_id_digest(token_ids):016x}"
            or int(raw_seed_p) != 1
            or int(raw_seed_r) != 2
            or int(raw_step_p) != token_count
            or int(raw_step_r) != token_count + 1
            or raw_cache_preserved != "1"
        ):
            raise BenchmarkFailure(
                f"native dynamic qualification run {expected_index} is inconsistent"
            )
        logical_q = int(raw_logical_q)
        logical_m = int(raw_logical_m)
        raw_runs.append({
            "label": _DYNAMIC_RUN_LABELS[expected_index][0],
            "shapeMode": mode,
            "familyId": int(raw_family_id),
            "tokenIds": token_ids,
            "tokenDigest": token_digest,
            "activeShape": {
                "B": 1,
                "Q": int(raw_bound_q),
                "M": int(raw_bound_m),
                "T": max_new + 1,
            },
            "logicalShape": {
                "B": 1,
                "Q": logical_q,
                "M": logical_m,
                "T": max_new + 1,
            },
            "cache": {
                "initialPastLength": 1,
                "finalPastLength": token_count + 1,
                "sentinelMaskValue": 1,
                "transitions": [
                    {
                        "position": index,
                        "pastLength": index + 1,
                        "presentLength": index + 2,
                    }
                    for index in range(token_count)
                ],
            },
        })
    runs = _validate_dynamic_runs(
        raw_runs, max_new=max_new, require_labels=True
    )
    encoder_routes = _ENCODER_ROUTE_RE.findall(debug)
    decoder_routes = _DECODER_ROUTE_RE.findall(debug)
    input_hashes = _INPUT_RE.findall(debug)
    question_token_lines = _QUESTION_TOKENS_RE.findall(debug)
    decoder_steps = _STEP_RE.findall(debug)
    expected_decoder_executions = sum(len(run["tokenIds"]) for run in runs)
    if (
        len(encoder_routes) != len(runs)
        or len(decoder_routes) != expected_decoder_executions
        or len(input_hashes) != len(runs)
        or len(set(input_hashes)) != 1
        or len(question_token_lines) != len(runs)
        or len(decoder_steps) != expected_decoder_executions
    ):
        raise BenchmarkFailure(
            f"native {backend}/{precision} qualification lacks input/step/route "
            "evidence for every execution"
        )
    for run_index, (run, encoded_tokens) in enumerate(
        zip(runs, question_token_lines, strict=True)
    ):
        question_token_ids = [int(value) for value in encoded_tokens.split(",")]
        if (
            not question_token_ids
            or len(question_token_ids) != run["logicalShape"]["Q"]
            or any(token < 0 or token >= 1536 for token in question_token_ids)
        ):
            raise BenchmarkFailure(
                f"native dynamic qualification run {run_index} has invalid "
                "question-token evidence"
            )
        run["questionTokenIds"] = question_token_ids
    if (
        runs[0]["questionTokenIds"] != runs[3]["questionTokenIds"]
        or runs[1]["questionTokenIds"] != runs[2]["questionTokenIds"]
    ):
        raise BenchmarkFailure(
            "native dynamic qualification question-token parity failed"
        )
    step_cursor = 0
    parsed_steps: list[list[dict[str, int]]] = []
    for run_index, run in enumerate(runs):
        family_name = FAMILY_NAMES[run["familyId"]]
        run_steps: list[dict[str, int]] = []
        for expected_step, expected_token in enumerate(run["tokenIds"]):
            (
                step_family,
                raw_step,
                raw_past,
                raw_present,
                raw_token,
                _raw_ms,
            ) = decoder_steps[step_cursor]
            step_cursor += 1
            if (
                step_family != family_name
                or int(raw_step) != expected_step
                or int(raw_past) != expected_step + 1
                or int(raw_present) != expected_step + 2
                or int(raw_token) != expected_token
            ):
                raise BenchmarkFailure(
                    f"native dynamic qualification run {run_index} decoder step "
                    f"{expected_step} contradicts its machine evidence"
                )
            run_steps.append({
                "position": expected_step,
                "pastLength": expected_step + 1,
                "presentLength": expected_step + 2,
                "tokenId": expected_token,
            })
        parsed_steps.append(run_steps)
    required_provider = f"provider=builtin:{backend};"
    routes = [*encoder_routes, *decoder_routes]
    if any(
        required_provider not in route
        or ";fallback=0;missing=0;" not in route
        for route in routes
    ):
        raise BenchmarkFailure(
            f"native {backend}/{precision} qualification does not prove strict fallback 0"
        )
    device = _physical_native_dynamic_device(output, debug, backend)
    runtime_evidence: dict[str, Any] = {"cpuThreads": cpu_threads}
    if device is not None:
        runtime_evidence["device"] = device
    return {
        "schema": DYNAMIC_QUALIFICATION_SCHEMA,
        "engine": "native-c",
        "backend": backend,
        "precision": precision,
        "timed": False,
        "freshProcess": True,
        "sameSession": True,
        "sameRuntime": True,
        "sameEncoderContext": True,
        "sameDecoderContext": True,
        "strictNoFallback": True,
        "boundedDecoderMaximumNewTokens": max_new,
        "maximumLegalEncoderBinding": {"B": 1, "Q": 192, "M": 402},
        "inputTensorSha256": input_hashes[0],
        "checks": {name: True for name in sorted(_DYNAMIC_CHECKS)},
        "runs": runs,
        "runtime": runtime_evidence,
        "routeEvidence": {
            "encoder": encoder_routes,
            "decoder": decoder_routes,
            "decoderSteps": parsed_steps,
        },
    }


def run_native_dynamic_qualification(
    *,
    binary: Path,
    package: Path,
    precision: str,
    image: Path,
    backend: str,
    max_new: int,
    threads: int,
    timeout: float,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Run one canonical native proof in its own process."""

    if not _plain_int(threads) or threads <= 0:
        raise BenchmarkFailure("native dynamic qualification threads must be positive")
    command = [
        str(binary),
        str(package),
        "--image",
        str(image),
        "--qualify-dynamic",
        f"--{backend}",
        "--threads",
        str(threads),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=dict(environment),
    )
    if completed.returncode != 0:
        raise BenchmarkFailure(
            f"native {backend}/{precision} dynamic qualification failed with "
            f"exit {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return parse_native_dynamic_qualification(
        completed.stdout,
        completed.stderr,
        precision=precision,
        backend=backend,
        max_new=max_new,
        cpu_threads=threads,
    )


def _strict_dynamic_route(route: Any, label: str) -> None:
    operator = route.get("operator") if isinstance(route, Mapping) else None
    if (
        not isinstance(operator, Mapping)
        or route.get("tierFallback") is not False
        or operator.get("attestation") != "none"
        or operator.get("used") is not False
        or operator.get("offendingNode") is not None
    ):
        raise BenchmarkFailure(f"{label} does not attest strict fallback 0")


def _strict_dynamic_compilation(report: Any, backend: str, label: str) -> None:
    policy = report.get("requestedPolicy") if isinstance(report, Mapping) else None
    candidates = report.get("candidates") if isinstance(report, Mapping) else None
    if (
        not isinstance(policy, Mapping)
        or policy.get("mode") != "require"
        or policy.get("backend") != backend
        or policy.get("operatorFallback") != "forbid"
        or report.get("selectedBackend") != backend
        or not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], Mapping)
        or candidates[0].get("backend") != backend
        or candidates[0].get("outcome") != "selected"
    ):
        raise BenchmarkFailure(
            f"{label} compilation did not strictly select {backend}"
        )
    _strict_dynamic_route(report.get("routeEvidence"), f"{label} compilation")
    _strict_dynamic_route(
        candidates[0].get("routeEvidence"), f"{label} compilation candidate"
    )


def _strict_dynamic_execution(report: Any, backend: str, label: str) -> None:
    if (
        not isinstance(report, Mapping)
        or report.get("backend") != backend
        or report.get("outcome") != "success"
        or report.get("operatorFallback") != "none"
    ):
        raise BenchmarkFailure(f"{label} did not execute on strict {backend}")
    _strict_dynamic_route(report.get("routeEvidence"), label)


def parse_wasm_dynamic_qualification(
    stdout: bytes,
    *,
    precision: str,
    family: str,
    max_new: int,
    node_version: str,
) -> dict[str, Any]:
    """Independently validate the framed WASM same-context proof."""

    if precision not in ("fp32", "int8") or family != DEFAULT_FAMILY:
        raise BenchmarkFailure("WASM dynamic qualification identity is invalid")
    try:
        lines = stdout.decode("utf-8", errors="strict").splitlines()
        encoded = [
            line[len(DYNAMIC_QUALIFICATION_PREFIX):]
            for line in lines
            if line.startswith(DYNAMIC_QUALIFICATION_PREFIX)
        ]
        if len(encoded) != 1:
            raise ValueError(f"expected one framed proof, received {len(encoded)}")
        value = json.loads(encoded[0])
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BenchmarkFailure(
            f"VolvoxAI wasm/{precision} did not emit one dynamic proof: {error}"
        ) from error
    input_hash = value.get("inputTensorSha256") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != DYNAMIC_QUALIFICATION_SCHEMA
        or value.get("engine") != "volvoxai-wasm"
        or value.get("backend") != "wasm"
        or value.get("precision") != precision
        or value.get("prompt") != DEFAULT_PROMPT
        or value.get("family") != family
        or value.get("timed") is not False
        or value.get("freshProcess") is not True
        or value.get("sameSession") is not True
        or value.get("sameRuntime") is not True
        or value.get("sameEncoderContext") is not True
        or value.get("sameDecoderContext") is not True
        or value.get("strictNoFallback") is not True
        or value.get("boundedDecoderMaximumNewTokens") != max_new
        or not _maximum_encoder_binding(value.get("maximumLegalEncoderBinding"))
        or not isinstance(input_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", input_hash) is None
        or "timing" in value
    ):
        raise BenchmarkFailure(
            f"VolvoxAI wasm/{precision} dynamic qualification identity is invalid"
        )
    checks = value.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _DYNAMIC_CHECKS
        or any(checks.get(name) is not True for name in _DYNAMIC_CHECKS)
    ):
        raise BenchmarkFailure(
            f"VolvoxAI wasm/{precision} lacks complete dynamic correctness checks"
        )
    raw_runs = value.get("runs")
    runs = _validate_dynamic_runs(
        raw_runs,
        max_new=max_new,
        require_labels=True,
        expected_family=family,
    )
    route_evidence = value.get("routeEvidence")
    compilations = (
        route_evidence.get("compilation")
        if isinstance(route_evidence, Mapping) else None
    )
    executions = (
        route_evidence.get("execution")
        if isinstance(route_evidence, Mapping) else None
    )
    if not isinstance(compilations, list) or len(compilations) != 2:
        raise BenchmarkFailure("WASM dynamic proof must contain two compilations")
    _strict_dynamic_compilation(compilations[0], "wasm", "WASM encoder")
    _strict_dynamic_compilation(compilations[1], "wasm", "WASM decoder")
    if not isinstance(executions, list) or len(executions) != len(runs):
        raise BenchmarkFailure("WASM dynamic proof lacks per-run executions")
    encoder_context: Any = None
    decoder_context: Any = None
    for run_index, (run, attestation) in enumerate(
        zip(runs, executions, strict=True)
    ):
        if not isinstance(attestation, Mapping):
            raise BenchmarkFailure(f"WASM dynamic run {run_index} attestation is invalid")
        encoder = attestation.get("encoder")
        decoder = attestation.get("decoder")
        if not isinstance(decoder, list) or len(decoder) != len(run["tokenIds"]):
            raise BenchmarkFailure(
                f"WASM dynamic run {run_index} lacks decoder executions"
            )
        _strict_dynamic_execution(encoder, "wasm", f"WASM run {run_index} encoder")
        for step, report in enumerate(decoder):
            _strict_dynamic_execution(
                report, "wasm", f"WASM run {run_index} decoder step {step}"
            )
        current_encoder = encoder.get("contextId")
        decoder_contexts = {report.get("contextId") for report in decoder}
        if (
            current_encoder is None
            or len(decoder_contexts) != 1
            or None in decoder_contexts
            or current_encoder in decoder_contexts
        ):
            raise BenchmarkFailure(
                f"WASM dynamic run {run_index} lacks distinct context identities"
            )
        current_decoder = next(iter(decoder_contexts))
        if run_index == 0:
            encoder_context = current_encoder
            decoder_context = current_decoder
        elif current_encoder != encoder_context or current_decoder != decoder_context:
            raise BenchmarkFailure("WASM dynamic proof did not reuse both contexts")
    return {
        "schema": DYNAMIC_QUALIFICATION_SCHEMA,
        "engine": "volvoxai-wasm",
        "backend": "wasm",
        "precision": precision,
        "timed": False,
        "freshProcess": True,
        "sameSession": True,
        "sameRuntime": True,
        "sameEncoderContext": True,
        "sameDecoderContext": True,
        "strictNoFallback": True,
        "boundedDecoderMaximumNewTokens": max_new,
        "maximumLegalEncoderBinding": {"B": 1, "Q": 192, "M": 402},
        "inputTensorSha256": input_hash,
        "checks": {name: True for name in sorted(_DYNAMIC_CHECKS)},
        "runs": runs,
        "runtime": {
            "name": "node",
            "version": node_version,
            "executionThreads": 1,
            "workerThreads": 0,
        },
        "routeEvidence": {
            "compilation": compilations,
            "execution": executions,
        },
        **({"artifact": dict(value["artifact"])}
           if isinstance(value.get("artifact"), Mapping) else {}),
    }


def run_wasm_dynamic_qualification(
    *,
    node: str,
    runner: Path,
    api: Path,
    wasm: Path,
    package: Path,
    precision: str,
    image: Path,
    timeout: float,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        node,
        str(runner),
        "--package",
        str(package),
        "--backend",
        "wasm",
        "--image",
        str(image),
        "--qualification",
        "dynamic-rebind",
        "--family",
        DEFAULT_FAMILY,
        "--max-new",
        str(CANONICAL_MAX_NEW),
        "--api",
        str(api),
        "--wasm",
        str(wasm),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=dict(environment),
    )
    if completed.returncode != 0:
        raise BenchmarkFailure(
            f"VolvoxAI wasm/{precision} dynamic qualification failed with exit "
            f"{completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return parse_wasm_dynamic_qualification(
        completed.stdout,
        precision=precision,
        family=DEFAULT_FAMILY,
        max_new=CANONICAL_MAX_NEW,
        node_version=_command_version([node, "--version"]),
    )


def _qualification_key(value: Mapping[str, Any]) -> str:
    return f"{value.get('engine')}/{value.get('backend')}/{value.get('precision')}"


def prove_dynamic_qualification_coverage(
    entries: Sequence[Mapping[str, Any]],
    *,
    native_backends: Sequence[str],
    include_webgpu: bool,
    include_wasm: bool = False,
    expected_question_token_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Prove exact coverage and parity across every requested engine/backend."""

    expected = {
        f"native-c/{backend}/{precision}"
        for precision in ("fp32", "int8")
        for backend in native_backends
    }
    if include_webgpu:
        expected.update(
            f"volvoxai-webgpu/webgpu/{precision}"
            for precision in ("fp32", "int8")
        )
    if include_wasm:
        expected.update(
            f"volvoxai-wasm/wasm/{precision}"
            for precision in ("fp32", "int8")
        )
    canonical_questions = [list(values) for values in expected_question_token_ids]
    if (
        len(canonical_questions) != len(_DYNAMIC_RUN_LABELS)
        or any(
            not values
            or values[-1] != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= 1536
                for value in values
            )
            for values in canonical_questions
        )
        or canonical_questions[0] != canonical_questions[3]
        or canonical_questions[1] != canonical_questions[2]
    ):
        raise BenchmarkFailure(
            "dynamic qualification expected question tokens are not canonical"
        )
    counts = Counter(_qualification_key(entry) for entry in entries)
    if set(counts) != expected or any(counts[key] != 1 for key in expected):
        raise BenchmarkFailure(
            "dynamic qualification coverage is incomplete: "
            f"expected {sorted(expected)}, got {dict(counts)}"
        )
    for entry in entries:
        checks = entry.get("checks")
        if (
            entry.get("schema") != DYNAMIC_QUALIFICATION_SCHEMA
            or entry.get("timed") is not False
            or entry.get("freshProcess") is not True
            or entry.get("sameSession") is not True
            or entry.get("sameRuntime") is not True
            or entry.get("sameEncoderContext") is not True
            or entry.get("sameDecoderContext") is not True
            or entry.get("strictNoFallback") is not True
            or entry.get("boundedDecoderMaximumNewTokens") != CANONICAL_MAX_NEW
            or not _maximum_encoder_binding(entry.get("maximumLegalEncoderBinding"))
            or not isinstance(entry.get("inputTensorSha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["inputTensorSha256"]) is None
            or not isinstance(checks, Mapping)
            or set(checks) != _DYNAMIC_CHECKS
            or any(checks.get(name) is not True for name in _DYNAMIC_CHECKS)
            or "timing" in entry
        ):
            raise BenchmarkFailure(
                f"dynamic qualification {_qualification_key(entry)} is not an "
                "untimed same-context strict proof"
            )
        if entry.get("engine") == "volvoxai-webgpu" and (
            entry.get("deviceResidentKVQualified") is not True
            or any(
                not isinstance(run, Mapping)
                or run.get("gpuResidentKv", {}).get("mode") != "device-qualified"
                or run.get("gpuResidentKv", {}).get("cacheReadbackFree") is not False
                for run in entry.get("runs", [])
            )
        ):
            raise BenchmarkFailure(
                f"dynamic qualification {_qualification_key(entry)} lacks "
                "device-resident KV correctness evidence"
            )
        runs = entry.get("runs")
        if not isinstance(runs, list):
            raise BenchmarkFailure(
                f"dynamic qualification {_qualification_key(entry)} lacks runs"
            )
        _validate_dynamic_runs(
            runs,
            max_new=CANONICAL_MAX_NEW,
            require_labels=True,
            expected_family=(
                DEFAULT_FAMILY if entry.get("engine") != "native-c" else None
            ),
        )
        if entry.get("engine") == "native-c" and (
            not isinstance(entry.get("runtime"), Mapping)
            or not _plain_int(entry["runtime"].get("cpuThreads"))
            or entry["runtime"]["cpuThreads"] <= 0
        ):
            raise BenchmarkFailure(
                f"dynamic qualification {_qualification_key(entry)} lacks "
                "native thread-limit evidence"
            )
        for run_index, run in enumerate(runs):
            question_ids = run.get("questionTokenIds") if isinstance(run, Mapping) else None
            logical_shape = run.get("logicalShape") if isinstance(run, Mapping) else None
            if (
                not isinstance(question_ids, list)
                or not question_ids
                or question_ids != canonical_questions[run_index]
                or question_ids[-1] != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    or value >= 1536
                    for value in question_ids
                )
                or not isinstance(logical_shape, Mapping)
                or len(question_ids) != logical_shape.get("Q")
            ):
                raise BenchmarkFailure(
                    f"dynamic qualification {_qualification_key(entry)} lacks "
                    "exact question-token evidence"
                )
    reference = entries[0]
    if any(
        entry["inputTensorSha256"] != reference["inputTensorSha256"]
        for entry in entries[1:]
    ):
        raise BenchmarkFailure("dynamic qualification input hash parity failed")
    signature = [
        (
            run.get("label"),
            run.get("shapeMode"),
            run.get("familyId"),
            run.get("tokenIds"),
            run.get("activeShape"),
            run.get("logicalShape"),
            run.get("cache"),
            run.get("questionTokenIds"),
        )
        for run in reference["runs"]
    ]
    for entry in entries[1:]:
        candidate = [
            (
                run.get("label"),
                run.get("shapeMode"),
                run.get("familyId"),
                run.get("tokenIds"),
                run.get("activeShape"),
                run.get("logicalShape"),
                run.get("cache"),
                run.get("questionTokenIds"),
            )
            for run in entry["runs"]
        ]
        if candidate != signature:
            raise BenchmarkFailure(
                "dynamic qualification parity failed for "
                f"{_qualification_key(entry)}"
            )
    return {
        "status": "pass",
        "timed": False,
        "executionOrder": "once-before-warmup",
        "sampleIsolation": "fresh process/runtime/session per entry",
        "expectedEntries": sorted(expected),
        "checks": {
            "coverage": True,
            "sameContextGrowShrink": True,
            "strictFallbackZero": True,
            "inputHashParity": True,
            "crossBackendPrecisionParity": True,
        },
        "entries": list(entries),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, name: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkFailure(f"required artifact is not a regular file: {path}")
    return {
        "name": name or path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkFailure(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise BenchmarkFailure(f"{label} must contain a JSON object")
    return value


def _safe_asset(root: Path, record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
        raise BenchmarkFailure(f"{label} must be an exact asset descriptor")
    relative = record.get("path")
    byte_count = record.get("bytes")
    digest = record.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(relative).parts)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise BenchmarkFailure(f"{label} is not a safe complete asset descriptor")
    path = root.joinpath(*Path(relative).parts)
    actual = _file_identity(path, name=relative)
    if actual["bytes"] != byte_count or actual["sha256"] != digest:
        raise BenchmarkFailure(f"{label} does not match its declared identity")
    return actual


def _volvox_graph_summary(path: Path) -> dict[str, Any]:
    graph = _read_json(path, "VolvoxAI graph")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise BenchmarkFailure("VolvoxAI graph nodes must be an array")
    counts: Counter[str] = Counter()
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("opType"), str):
            raise BenchmarkFailure("VolvoxAI graph node lacks a string opType")
        counts[node["opType"]] += 1
    return {
        "nodeCount": len(nodes),
        "operatorCounts": dict(sorted(counts.items())),
    }


def load_package_identity(
    root: Path,
    precision: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise BenchmarkFailure(f"{precision} package must be a real directory")
    manifest_path = root / "package_manifest.json"
    manifest = _read_json(manifest_path, f"{precision} package manifest")
    if manifest.get("format") != PACKAGE_FORMAT:
        raise BenchmarkFailure(
            f"{precision} package must use explicit-KV format {PACKAGE_FORMAT!r}"
        )
    expected_variant = "fp32" if precision == "fp32" else "int8-w8a8"
    package_source = manifest.get("source")
    if (
        not isinstance(package_source, Mapping)
        or package_source.get("format") != SOURCE_FORMAT
        or package_source.get("variant") != expected_variant
    ):
        raise BenchmarkFailure(f"{precision} package source variant is inconsistent")
    selected_keys = source["selected_file_keys"]
    expected = {
        "manifest": source["hashes"]["manifest"],
        "encoder_onnx": source["hashes"][selected_keys["encoder"]],
        "decoder_onnx": source["hashes"][selected_keys["decoder"]],
    }
    for name, digest in expected.items():
        record = package_source.get(name)
        if not isinstance(record, Mapping) or record.get("sha256") != digest:
            raise BenchmarkFailure(f"{precision} package source.{name} hash is inconsistent")
    graphs = manifest.get("graphs")
    assets = manifest.get("assets")
    if not isinstance(graphs, Mapping) or set(graphs) != {"encoder", "decoder"}:
        raise BenchmarkFailure(f"{precision} package must contain two graphs")
    if not isinstance(assets, Mapping):
        raise BenchmarkFailure(f"{precision} package assets must be an object")
    graph_identities: dict[str, Any] = {}
    for kind in ("encoder", "decoder"):
        graph = graphs.get(kind)
        if not isinstance(graph, Mapping):
            raise BenchmarkFailure(f"{precision} package graph {kind} is invalid")
        identities = {
            field: _safe_asset(root, graph.get(field), f"{precision}.{kind}.{field}")
            for field in ("graph", "weights", "export_report")
        }
        identities["graph"].update(
            _volvox_graph_summary(root / identities["graph"]["name"])
        )
        graph_identities[kind] = identities
    asset_identities = {
        field: _safe_asset(root, assets.get(field), f"{precision}.{field}")
        for field in ("config", "vocab")
    }
    for field, identity in asset_identities.items():
        source_identity = _file_identity(Path(source["paths"][field]))
        if (
            identity["bytes"] != source_identity["bytes"]
            or identity["sha256"] != source_identity["sha256"]
        ):
            raise BenchmarkFailure(
                f"{precision} package {field} differs from the producer input"
            )
    return {
        "format": PACKAGE_FORMAT,
        "precision": precision,
        "sourceVariant": expected_variant,
        "manifest": _file_identity(manifest_path, name="package_manifest.json"),
        "graphs": graph_identities,
        "assets": asset_identities,
    }


def _validated_source_identities(
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Identify the exact producer files consumed by all selected variants."""

    if set(sources) != {"fp32", "int8"}:
        raise BenchmarkFailure("benchmark source identities require FP32 and INT8 variants")
    common: dict[str, Any] = {}
    for name in ("manifest", "config", "vocab"):
        paths = [Path(sources[precision]["paths"][name]) for precision in ("fp32", "int8")]
        if paths[0].resolve() != paths[1].resolve():
            raise BenchmarkFailure(f"FP32 and INT8 source {name} paths differ")
        identity = _file_identity(paths[0], name=paths[0].name)
        if any(sources[precision]["hashes"].get(name) != identity["sha256"]
               for precision in ("fp32", "int8")):
            raise BenchmarkFailure(f"validated source {name} changed before benchmarking")
        common[name] = identity

    variants: dict[str, Any] = {}
    for precision in ("fp32", "int8"):
        source = sources[precision]
        selected_keys = source["selected_file_keys"]
        variants[precision] = {}
        for role in ("encoder", "decoder"):
            key = selected_keys[role]
            path = Path(source["selected_paths"][role])
            identity = _file_identity(path, name=path.name)
            if source["hashes"].get(key) != identity["sha256"]:
                raise BenchmarkFailure(
                    f"validated {precision} source {role} changed before benchmarking"
                )
            variants[precision][role] = identity
    return {
        "format": SOURCE_FORMAT,
        "common": common,
        "variants": variants,
    }


def _synthetic_pixels() -> Any:
    import numpy as np

    y = np.arange(IMAGE_HEIGHT, dtype=np.int64)[:, None]
    x = np.arange(IMAGE_WIDTH, dtype=np.int64)[None, :]
    return ((17 * x + 29 * y + 7 * np.bitwise_xor(x, y)) & 255).astype(np.uint8)


def create_synthetic_image(path: Path) -> dict[str, Any]:
    from PIL import Image

    pixels = _synthetic_pixels()
    Image.fromarray(pixels, mode="L").save(path, format="PNG", compress_level=9)
    return {
        "kind": "generated-grayscale-png-v1",
        "generator": "q=(17*x+29*y+7*(x^y))&255",
        "shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "pixelsSha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "file": _file_identity(path, name="synthetic-receipt.png"),
    }


def create_source_tokenizer(source: Mapping[str, Any], tokenizer_type: Any) -> Any:
    tokenizer_document = source.get("tokenizer")
    vocabulary = source.get("vocab")
    if not isinstance(tokenizer_document, Mapping) or not isinstance(vocabulary, Mapping):
        raise BenchmarkFailure("validated source lacks tokenizer documents")
    try:
        tokenizer = tokenizer_type.from_documents(tokenizer_document, vocabulary)
        probe = tokenizer.encode(DEFAULT_PROMPT, add_eos=True, max_len=192)
    except Exception as error:
        raise BenchmarkFailure(f"could not instantiate the canonical source tokenizer: {error}") from error
    if not isinstance(probe, list) or not probe or probe[-1] != 2:
        raise BenchmarkFailure("source tokenizer does not emit the canonical EOS-terminated IDs")
    return tokenizer


def _preprocess_image(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("L").resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(image, dtype=np.float32).copy()
    pixels /= np.float32(255.0)
    pixels *= np.float32(2.0)
    pixels -= np.float32(1.0)
    return np.ascontiguousarray(pixels[None, None, :, :])


def _float32_tensor_sha256(value: Any) -> str:
    import numpy as np

    little_endian = np.ascontiguousarray(
        np.asarray(value, dtype=np.float32).astype("<f4", copy=False)
    )
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _named_outputs(session: Any, feeds: Mapping[str, Any]) -> dict[str, Any]:
    values = session.run(None, dict(feeds))
    names = [output.name for output in session.get_outputs()]
    if len(names) != len(values) or len(names) != len(set(names)):
        raise BenchmarkFailure("ONNX Runtime returned an invalid output signature")
    return dict(zip(names, values, strict=True))


def _timing_summary(step_ms: Sequence[float]) -> dict[str, Any]:
    if len(step_ms) < 2 or any(not math.isfinite(value) or value < 0 for value in step_ms):
        raise BenchmarkFailure("a KV benchmark requires at least two finite decoder steps")
    steady = list(step_ms[1:])
    steady_total = sum(steady)
    return {
        "decoderSeedMs": step_ms[0],
        "decoderSteadySteps": len(steady),
        "decoderSteadyTotalMs": steady_total,
        "decoderSteadyMeanMs": steady_total / len(steady),
        "decoderSteadyTokensPerSecond": (
            1000.0 * len(steady) / steady_total if steady_total > 0 else None
        ),
        "decoderExecutionTotalMs": sum(step_ms),
        "decoderStepMs": list(step_ms),
    }


def _ort_provider_registration(
    provider: str,
    provider_options: Mapping[str, str],
    allow_cpu_fallback: bool,
) -> list[Any]:
    if allow_cpu_fallback and provider == "CPUExecutionProvider":
        raise BenchmarkFailure(
            "ONNX Runtime CPU fallback requires a non-CPU primary provider"
        )
    primary: Any = (
        (provider, dict(provider_options)) if provider_options else provider
    )
    return [
        primary,
        *(["CPUExecutionProvider"] if allow_cpu_fallback else []),
    ]


def _ort_session_options(
    ort: Any,
    threads: int,
    *,
    disable_cpu_fallback: bool = False,
    profile_prefix: Path | None = None,
) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if disable_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options


def _parse_ort_profile_placement(
    path: Path,
    *,
    role: str,
    allowed_providers: Sequence[str],
) -> dict[str, Any]:
    """Reduce one untimed ORT profile to exact executed-node placement."""

    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkFailure(f"ONNX {role} profile is not valid JSON: {error}") from error
    if not isinstance(events, list):
        raise BenchmarkFailure(f"ONNX {role} profile root is not an event array")

    allowed = set(allowed_providers)
    assignments: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    provider_counts: Counter[str] = Counter()
    operator_counts: dict[str, Counter[str]] = {}
    for event in events:
        if not isinstance(event, Mapping) or event.get("cat") != "Node":
            continue
        arguments = event.get("args")
        provider = arguments.get("provider") if isinstance(arguments, Mapping) else None
        raw_index = arguments.get("node_index") if isinstance(arguments, Mapping) else None
        operator = arguments.get("op_name") if isinstance(arguments, Mapping) else None
        name = event.get("name")
        try:
            if isinstance(raw_index, int) and not isinstance(raw_index, bool):
                node_index = raw_index
            elif isinstance(raw_index, str) and re.fullmatch(r"[0-9]+", raw_index):
                node_index = int(raw_index)
            else:
                raise ValueError("node_index is not a non-negative integer")
        except ValueError as error:
            raise BenchmarkFailure(
                f"ONNX {role} profile node has an invalid node_index"
            ) from error
        if (
            node_index < 0
            or not isinstance(provider, str)
            or provider not in allowed
            or not isinstance(operator, str)
            or not operator
            or not isinstance(name, str)
            or not name
            or node_index in seen_indices
        ):
            raise BenchmarkFailure(
                f"ONNX {role} profile has invalid or duplicate node placement"
            )
        seen_indices.add(node_index)
        provider_counts[provider] += 1
        operator_counts.setdefault(provider, Counter())[operator] += 1
        assignments.append({
            "nodeIndex": node_index,
            "name": name,
            "operator": operator,
            "provider": provider,
        })
    if not assignments:
        raise BenchmarkFailure(f"ONNX {role} profile contains no executed nodes")
    assignments.sort(key=lambda value: value["nodeIndex"])
    canonical = json.dumps(
        assignments, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return {
        "executedNodeCount": len(assignments),
        "providers": dict(sorted(provider_counts.items())),
        "operatorCountsByProvider": {
            provider: dict(sorted(counts.items()))
            for provider, counts in sorted(operator_counts.items())
        },
        "nodeAssignmentSha256": hashlib.sha256(canonical).hexdigest(),
    }


def _ort_strict_session_probe(
    ort: Any,
    model_path: Path,
    *,
    role: str,
    threads: int,
    provider: str,
    provider_options: Mapping[str, str],
) -> dict[str, Any]:
    options = _ort_session_options(
        ort, threads, disable_cpu_fallback=True,
    )
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=_ort_provider_registration(provider, provider_options, False),
        )
    except Exception as error:  # ORT exposes provider failures as RuntimeError subclasses.
        message = str(error)
        if any(clause not in message for clause in _ORT_STRICT_CPU_FALLBACK_ERROR_CLAUSES):
            raise BenchmarkFailure(
                f"ONNX {role} strict probe failed for a reason other than CPU partitioning: "
                f"{message}"
            ) from error
        return {
            "stage": "session-create",
            "outcome": "rejected",
            "cpuEpFallbackDisabled": True,
            "errorType": type(error).__name__,
            "error": message,
        }
    return {
        "stage": "session-create",
        "outcome": "accepted",
        "cpuEpFallbackDisabled": True,
        "registeredProviders": list(session.get_providers()),
        "errorType": None,
        "error": None,
    }


def _collect_ort_accelerator_evidence(
    ort: Any,
    source: Mapping[str, Any],
    *,
    image: Any,
    question_ids: Any,
    requested_family_ids: Any,
    threads: int,
    provider: str,
    provider_options: Mapping[str, str],
) -> dict[str, Any]:
    """Collect placement after timing in disposable profiling/probe sessions."""

    import numpy as np

    provider_order = [provider, "CPUExecutionProvider"]
    placements: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="tinyreceipt-ort-profile-") as directory:
        profile_root = Path(directory)
        encoder_options = _ort_session_options(
            ort, threads, profile_prefix=profile_root / "encoder",
        )
        encoder = ort.InferenceSession(
            str(source["selected_paths"]["encoder"]),
            sess_options=encoder_options,
            providers=_ort_provider_registration(provider, provider_options, True),
        )
        if encoder.get_providers() != provider_order:
            raise BenchmarkFailure(
                "ONNX encoder profiling session changed the accelerator-first provider order"
            )
        encoder_profile: Path | None = None
        try:
            encoded = _named_outputs(encoder, {
                "image": image,
                "question_ids": question_ids,
                "family_ids": requested_family_ids,
            })
        finally:
            encoder_profile = Path(encoder.end_profiling())
        placements["encoder"] = _parse_ort_profile_placement(
            encoder_profile, role="encoder", allowed_providers=provider_order,
        )

        selected_ids = encoded.get("selected_family_ids")
        memory_mask = encoded.get("memory_padding_mask")
        if (
            selected_ids is None
            or memory_mask is None
            or any(name not in encoded for name in CROSS_NAMES)
        ):
            raise BenchmarkFailure(
                "ONNX profiling encoder does not expose the explicit cross-cache ABI"
            )
        decoder_options = _ort_session_options(
            ort, threads, profile_prefix=profile_root / "decoder",
        )
        decoder = ort.InferenceSession(
            str(source["selected_paths"]["decoder"]),
            sess_options=decoder_options,
            providers=_ort_provider_registration(provider, provider_options, True),
        )
        if decoder.get_providers() != provider_order:
            raise BenchmarkFailure(
                "ONNX decoder profiling session changed the accelerator-first provider order"
            )
        decoder_profile: Path | None = None
        try:
            _named_outputs(decoder, {
                "decoder_input_ids": np.asarray([[1]], dtype=np.int64),
                "position_ids": np.asarray([0], dtype=np.int64),
                "family_ids": selected_ids.astype(np.int64, copy=False),
                "memory_padding_mask": memory_mask,
                "past_padding_mask": np.ones((1, 1), dtype=np.bool_),
                **{name: encoded[name] for name in CROSS_NAMES},
                **{
                    name: np.zeros((1, 8, 1, 40), dtype=np.float32)
                    for name in PAST_NAMES
                },
            })
        finally:
            decoder_profile = Path(decoder.end_profiling())
        placements["decoder"] = _parse_ort_profile_placement(
            decoder_profile, role="decoder", allowed_providers=provider_order,
        )

    strict_probe = {
        role: _ort_strict_session_probe(
            ort,
            Path(source["selected_paths"][role]),
            role=role,
            threads=threads,
            provider=provider,
            provider_options=provider_options,
        )
        for role in ("encoder", "decoder")
    }
    for role in ("encoder", "decoder"):
        cpu_nodes = placements[role]["providers"].get("CPUExecutionProvider", 0)
        outcome = strict_probe[role]["outcome"]
        if (cpu_nodes > 0) != (outcome == "rejected"):
            raise BenchmarkFailure(
                f"ONNX {role} profiling placement contradicts its strict fallback probe"
            )
        if placements[role]["providers"].get(provider, 0) <= 0:
            raise BenchmarkFailure(
                f"ONNX {role} profiling did not execute any node on {provider}"
            )

    return {
        "source": "separate-untimed-profiling-sessions",
        "executionOrder": "after-measured-request",
        "sameMeasuredSessions": False,
        "measuredSessionsProfiled": False,
        "sessionConfigurationInvariant": (
            "same model, provider order/options, thread counts, sequential execution, "
            "and ORT_ENABLE_ALL; profiling is the only session-option difference"
        ),
        "requests": {
            "encoder": "same canonical image/question/family request",
            "decoder": "same encoder outputs and blocked P=1 seed request",
        },
        "roles": placements,
        "strictProbe": {
            "source": "separate-untimed-session-create-probes",
            "executionOrder": "after-measured-request",
            "roles": strict_probe,
        },
    }


def run_ort_sample(
    source: Mapping[str, Any],
    precision: str,
    image_path: Path,
    prompt: str,
    family: str,
    max_new: int,
    threads: int,
    provider: str = "CPUExecutionProvider",
    provider_options: Mapping[str, str] | None = None,
    allow_cpu_fallback: bool = False,
    execution_warmup: int = 0,
) -> dict[str, Any]:
    _validate_execution_warmup(execution_warmup, "ONNX")

    import numpy as np
    import onnxruntime as ort

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*ExecutionProvider", provider):
        raise BenchmarkFailure("ONNX Runtime provider has an invalid name")
    if provider not in ort.get_available_providers():
        raise BenchmarkFailure(
            f"ONNX Runtime provider {provider!r} is not available; "
            f"available providers: {ort.get_available_providers()}"
        )
    normalized_provider_options = {
        str(name): str(value) for name, value in (provider_options or {}).items()
    }
    strict_primary = provider != "CPUExecutionProvider" and not allow_cpu_fallback
    # ORT otherwise registers its CPU EP as a silent final partition even when
    # only one accelerator provider was requested. A strict GPU reference must
    # fail session construction if any node needs CPU.
    session_options = _ort_session_options(
        ort, threads, disable_cpu_fallback=strict_primary,
    )
    providers = _ort_provider_registration(
        provider, normalized_provider_options, allow_cpu_fallback
    )
    encoder = ort.InferenceSession(
        str(source["selected_paths"]["encoder"]),
        sess_options=session_options,
        providers=providers,
    )
    decoder = ort.InferenceSession(
        str(source["selected_paths"]["decoder"]),
        sess_options=session_options,
        providers=providers,
    )
    encoder_providers = encoder.get_providers()
    decoder_providers = decoder.get_providers()
    expected_registered = (
        [provider, "CPUExecutionProvider"]
        if allow_cpu_fallback else None
    )
    if (
        not encoder_providers
        or not decoder_providers
        or encoder_providers[0] != provider
        or decoder_providers[0] != provider
        or (
            expected_registered is not None
            and (
                encoder_providers != expected_registered
                or decoder_providers != expected_registered
            )
        )
        or (
            provider == "CPUExecutionProvider"
            and (encoder_providers != [provider] or decoder_providers != [provider])
        )
    ):
        raise BenchmarkFailure(
            f"ONNX Runtime did not register the requested provider policy for {provider}: "
            f"encoder={encoder_providers}, decoder={decoder_providers}"
        )
    tokenizer = source["tokenizer_runtime"]
    normalized_prompt = unicodedata.normalize("NFC", prompt)
    question_ids = np.asarray(
        [tokenizer.encode(normalized_prompt, add_eos=True, max_len=192)],
        dtype=np.int64,
    )
    family_id = -1 if family == "auto" else FAMILY_NAMES.index(family)
    requested_family_ids = np.asarray([family_id], dtype=np.int64)
    image = _preprocess_image(image_path)
    input_tensor_sha256 = _float32_tensor_sha256(image)

    def execute_request(*, measured: bool) -> dict[str, Any]:
        encoder_started = time.perf_counter_ns()
        encoder_outputs = _named_outputs(
            encoder,
            {
                "image": image,
                "question_ids": question_ids,
                "family_ids": requested_family_ids,
            },
        )
        encoder_ms = (time.perf_counter_ns() - encoder_started) / 1_000_000.0
        required_encoder = {
            "memory_padding_mask",
            "router_logits",
            "selected_family_ids",
            *CROSS_NAMES,
        }
        if not required_encoder.issubset(encoder_outputs):
            raise BenchmarkFailure("ONNX encoder does not expose the explicit cross-cache ABI")
        memory_mask = encoder_outputs["memory_padding_mask"]
        selected_ids = encoder_outputs["selected_family_ids"]
        selected = int(selected_ids[0])
        if selected < 0 or selected >= len(FAMILY_NAMES):
            raise BenchmarkFailure("ONNX encoder selected an invalid family")
        if family_id >= 0 and selected != family_id:
            raise BenchmarkFailure("ONNX encoder did not preserve the requested family")
        if memory_mask.dtype != np.bool_ or memory_mask.ndim != 2:
            raise BenchmarkFailure("ONNX encoder memory mask is not BOOL [B,M]")
        crosses = {name: encoder_outputs[name] for name in CROSS_NAMES}
        memory_length = int(memory_mask.shape[1])
        for name, value in crosses.items():
            if value.shape != (1, 8, memory_length, 40) or not np.all(np.isfinite(value)):
                raise BenchmarkFailure(f"ONNX encoder {name} has an invalid cache tensor")

        past_length = 1
        past_mask = np.ones((1, 1), dtype=np.bool_)
        past = {
            name: np.zeros((1, 8, 1, 40), dtype=np.float32) for name in PAST_NAMES
        }
        current = 1
        tokens: list[int] = []
        transitions: list[dict[str, int]] = []
        decoder_step_ms: list[float] = []
        for position in range(max_new):
            feed: dict[str, Any] = {
                "decoder_input_ids": np.asarray([[current]], dtype=np.int64),
                "position_ids": np.asarray([position], dtype=np.int64),
                "family_ids": selected_ids.astype(np.int64, copy=False),
                "memory_padding_mask": memory_mask,
                "past_padding_mask": past_mask,
                **crosses,
                **past,
            }
            started = time.perf_counter_ns()
            outputs = _named_outputs(decoder, feed)
            decoder_step_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
            present_length = past_length + 1
            if set(outputs) != {"logits", "present_padding_mask", *PRESENT_NAMES}:
                raise BenchmarkFailure("ONNX decoder output signature is not explicit-KV v1")
            logits = outputs["logits"]
            next_token = int(np.argmax(logits[0, 0, :]))
            present_mask = outputs["present_padding_mask"]
            if (
                logits.shape != (1, 1, 1536)
                or not np.all(np.isfinite(logits))
                or present_mask.shape != (1, present_length)
                or present_mask.dtype != np.bool_
                or not np.array_equal(present_mask[:, :past_length], past_mask)
                or bool(present_mask[0, past_length]) != (current == 0)
            ):
                raise BenchmarkFailure("ONNX decoder mask/logits transition is invalid")
            next_past: dict[str, Any] = {}
            for past_name, present_name in zip(PAST_NAMES, PRESENT_NAMES, strict=True):
                value = outputs[present_name]
                if (
                    value.shape != (1, 8, present_length, 40)
                    or not np.all(np.isfinite(value))
                    or not np.array_equal(value[:, :, :past_length, :], past[past_name])
                    or not np.array_equal(
                        value[:, :, 0, :], np.zeros((1, 8, 40), np.float32)
                    )
                ):
                    raise BenchmarkFailure(
                        f"ONNX decoder {present_name} corrupted its cache prefix"
                    )
                next_past[past_name] = value
            transitions.append({
                "position": position,
                "pastLength": past_length,
                "presentLength": present_length,
            })
            tokens.append(next_token)
            past = next_past
            past_mask = present_mask
            past_length = present_length
            if next_token == 2:
                break
            current = next_token
        return {
            "family": FAMILY_NAMES[selected],
            "familyId": selected,
            "tokens": tokens,
            "memoryLength": memory_length,
            "pastLength": past_length,
            "transitions": transitions,
            **({
                "encoderMs": encoder_ms,
                "timing": _timing_summary(decoder_step_ms),
            } if measured else {}),
        }

    warmups = [execute_request(measured=False) for _ in range(execution_warmup)]
    measured = execute_request(measured=True)
    warmup_signature = [
        (run["familyId"], run["tokens"], run["memoryLength"], run["transitions"])
        for run in warmups
    ]
    measured_signature = (
        measured["familyId"], measured["tokens"], measured["memoryLength"],
        measured["transitions"],
    )
    if any(value != measured_signature for value in warmup_signature):
        raise BenchmarkFailure("ONNX warmup token/cache transition parity failed")
    selected = measured["familyId"]
    tokens = measured["tokens"]
    memory_length = measured["memoryLength"]
    past_length = measured["pastLength"]
    transitions = measured["transitions"]
    timing = measured["timing"]
    accelerator_evidence = None
    if provider != "CPUExecutionProvider" and allow_cpu_fallback:
        # Collect execution placement and strict-fallback evidence only after
        # the measured request. ORT profiling is a per-session option, so using
        # disposable sessions is the only way to keep profiling overhead out of
        # the measured sessions and their same-session warmup lifecycle.
        accelerator_evidence = _collect_ort_accelerator_evidence(
            ort,
            source,
            image=image,
            question_ids=question_ids,
            requested_family_ids=requested_family_ids,
            threads=threads,
            provider=provider,
            provider_options=normalized_provider_options,
        )
    return {
        "schema": SAMPLE_SCHEMA,
        "engine": "onnxruntime",
        "backend": (
            "cpu" if provider == "CPUExecutionProvider"
            else provider.removesuffix("ExecutionProvider").lower()
            + ("-cpu-fallback" if allow_cpu_fallback else "")
        ),
        "precision": precision,
        "provider": provider,
        "strictNoFallback": not allow_cpu_fallback,
        "family": FAMILY_NAMES[selected],
        "familyId": selected,
        "requestedFamily": family,
        "inputTensorSha256": input_tensor_sha256,
        "questionTokenIds": [int(value) for value in question_ids[0]],
        "tokenIds": tokens,
        "stoppedAtEos": tokens[-1] == 2,
        "shape": {
            "mode": "active",
            "B": 1,
            "Q": int(question_ids.shape[1]),
            "M": memory_length,
            "T": max_new + 1,
        },
        "cache": {
            "initialPastLength": 1,
            "finalPastLength": past_length,
            "sentinelMaskValue": 1,
            "transitions": transitions,
        },
        "timing": {"encoderExecutionMs": measured["encoderMs"], **timing},
        "runtime": {
            "name": "onnxruntime",
            "version": ort.__version__,
            "intraOpThreads": threads,
            "interOpThreads": 1,
            "executionMode": "sequential",
            "graphOptimizationLevel": "ORT_ENABLE_ALL",
            "registeredEncoderProviders": encoder_providers,
            "registeredDecoderProviders": decoder_providers,
            "providerOptions": normalized_provider_options,
            "providerOrder": [
                provider,
                *(["CPUExecutionProvider"] if allow_cpu_fallback else []),
            ],
            "providerMode": (
                "accelerator-with-cpu-fallback"
                if allow_cpu_fallback else "strict-single-provider"
            ),
            "cpuEpFallbackAllowed": allow_cpu_fallback,
            "cpuEpFallbackUsage": (
                "exact-executed-node-placement-attested"
                if allow_cpu_fallback
                else "disabled" if provider != "CPUExecutionProvider"
                else "not-applicable-cpu-primary"
            ),
            "cpuEpFallbackDisabled": (
                provider != "CPUExecutionProvider" and not allow_cpu_fallback
            ),
            "executionWarmup": {
                "runs": execution_warmup,
                "sameSessions": True,
                "stateResetToSentinel": True,
                "tokenCacheTransitionParity": True,
            },
            **({
                "providerPlacement": accelerator_evidence["roles"],
                "providerPlacementEvidence": {
                    name: value
                    for name, value in accelerator_evidence.items()
                    if name not in ("roles", "strictProbe")
                },
                "strictProbe": accelerator_evidence["strictProbe"],
            } if accelerator_evidence is not None else {}),
        },
    }


def _parse_json_sample(stdout: bytes, label: str) -> dict[str, Any]:
    try:
        lines = stdout.decode("utf-8").splitlines()
        samples = [line[len(SAMPLE_PREFIX) :] for line in lines if line.startswith(SAMPLE_PREFIX)]
        if len(samples) != 1:
            raise ValueError(f"expected one framed sample, received {len(samples)}")
        value = json.loads(samples[0])
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BenchmarkFailure(f"{label} did not emit one JSON sample: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != SAMPLE_SCHEMA:
        raise BenchmarkFailure(f"{label} emitted an unsupported sample schema")
    return value


def run_volvox_wasm_sample(
    *,
    node: str,
    runner: Path,
    api: Path,
    wasm: Path,
    package: Path,
    precision: str,
    image: Path,
    prompt: str,
    family: str,
    max_new: int,
    timeout: float,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        node,
        str(runner),
        "--package",
        str(package),
        "--backend",
        "wasm",
        "--image",
        str(image),
        "--prompt",
        prompt,
        "--family",
        family,
        "--max-new",
        str(max_new),
        "--api",
        str(api),
        "--wasm",
        str(wasm),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=dict(environment),
    )
    if completed.returncode != 0:
        raise BenchmarkFailure(
            f"VolvoxAI wasm/{precision} failed with exit {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    sample = _parse_json_sample(completed.stdout, f"VolvoxAI wasm/{precision}")
    if sample.get("engine") != "volvoxai-wasm" or sample.get("precision") != precision:
        raise BenchmarkFailure(f"VolvoxAI wasm/{precision} mislabeled its sample")
    sample["runtime"] = {
        "name": "node",
        "version": _command_version([node, "--version"]),
        "executionThreads": 1,
        "workerThreads": 0,
        "threadCountScope": "VolvoxAI WASM engine only",
        "hostRuntimeAuxiliaryThreads": (
            "Node/V8 auxiliary threads are not counted or limited by the harness; "
            "they share the process's inherited CPU affinity"
        ),
    }
    return sample


def parse_native_sample(
    stdout: bytes,
    stderr: bytes,
    *,
    precision: str,
    family: str,
    max_new: int,
    threads: int,
    process_wall_ms: float,
    backend: str = "cpu",
    execution_warmup: int = 0,
) -> dict[str, Any]:
    _validate_execution_warmup(execution_warmup, "native")

    output = stdout.decode("utf-8", errors="strict")
    debug = stderr.decode("utf-8", errors="strict")
    if not output.startswith(
        f"VolvoxAI Native Runtime\nBackend policy: {backend}\n"
    ):
        raise BenchmarkFailure(
            f"native output does not prove the required {backend} policy"
        )
    abi = _ABI_RE.findall(debug)
    routers = _ROUTER_RE.findall(debug)
    totals = _TOTAL_RE.findall(debug)
    token_lines = _TOKENS_RE.findall(debug)
    question_token_lines = _QUESTION_TOKENS_RE.findall(debug)
    shapes = _SHAPE_RE.findall(debug)
    input_hashes = _INPUT_RE.findall(debug)
    steps = _STEP_RE.findall(debug)
    encoder_routes = _ENCODER_ROUTE_RE.findall(debug)
    decoder_routes = _DECODER_ROUTE_RE.findall(debug)
    warmup_results = _WARMUP_RESULT_RE.findall(output)
    warmup_lines = [
        line for line in output.splitlines()
        if line.startswith("WARMUP_RESULT ")
    ]
    device: dict[str, Any] | None = None
    if backend == "cuda":
        matches = _CUDA_DEVICE_RE.findall(debug)
        unique_matches = set(matches)
        if len(unique_matches) > 1:
            raise BenchmarkFailure(
                "native CUDA output reports conflicting device identities"
            )
        if unique_matches:
            match = next(iter(unique_matches))
            device = {
                "index": int(match[0]),
                "name": match[1],
                "computeCapability": f"{match[2]}.{match[3]}",
            }
    elif backend == "vulkan":
        matches = _VULKAN_DEVICE_RE.findall(output)
        if len(matches) == 1:
            device = {
                "name": matches[0][0],
                "packedInt8Dot": matches[0][1] == "enabled",
            }
    elif backend == "opengl":
        matches = _OPENGL_DEVICE_RE.findall(output)
        if len(matches) == 1:
            device = {
                "vendor": matches[0][0],
                "renderer": matches[0][1],
                "version": matches[0][2],
            }
    if not (
        len(abi) == len(routers) == len(totals) == len(token_lines)
        == len(question_token_lines) == len(shapes) == len(input_hashes) == 1
    ):
        raise BenchmarkFailure("native output lacks unique explicit-KV ABI/timing evidence")
    if backend != "cpu":
        routes = [*encoder_routes, *decoder_routes]
        required = f"provider=builtin:{backend};"
        if len(encoder_routes) != 1 or len(decoder_routes) != len(steps):
            raise BenchmarkFailure("native GPU output lacks route evidence for every execution")
        if any(
            required not in route
            or ";fallback=0;missing=0;" not in route
            for route in routes
        ):
            raise BenchmarkFailure(
                f"native GPU output does not prove strict {backend} fallback 0"
            )
        # The native dynamic-plan cache currently retains four decoder shapes.
        # A longer explicit-KV decode can legitimately evict an early P shape
        # before the measured request begins, so only claim all-hit evidence
        # when the measured request fits that cache.
        if execution_warmup > 0 and len(decoder_routes) <= 4 and any(
            re.search(r"(?:^|;)shape_plan=hit(?:;|$)", route) is None
            for route in routes
        ):
            raise BenchmarkFailure(
                f"native GPU output does not prove every warmed {backend} shape plan hit"
            )
        if device is None:
            raise BenchmarkFailure(
                f"native GPU output does not identify its {backend} device"
            )
    selected = routers[0][1]
    if selected != family and family != "auto":
        raise BenchmarkFailure("native selected family differs from the requested family")
    tokens = [] if token_lines[0] == "none" else [int(value) for value in token_lines[0].split(",")]
    question_tokens = [int(value) for value in question_token_lines[0].split(",")]
    if (
        not question_tokens
        or len(question_tokens) != int(shapes[0][1])
        or question_tokens[-1] != 2
        or any(value < 0 or value >= 1536 for value in question_tokens)
    ):
        raise BenchmarkFailure("native output has invalid question token IDs")
    if len(tokens) < 2 or len(tokens) > max_new or len(steps) != len(tokens):
        raise BenchmarkFailure("native output has an invalid decoder token/step count")
    if execution_warmup == 0:
        if warmup_lines or warmup_results:
            raise BenchmarkFailure("native output contains unexpected warmup evidence")
    else:
        if len(warmup_lines) != 1 or len(warmup_results) != 1:
            raise BenchmarkFailure("native output lacks unique same-context warmup evidence")
        (
            raw_count, raw_family_id, raw_token_count, raw_token_digest,
            raw_token_ids, raw_seed_p, raw_seed_r, raw_last_p, raw_last_r,
            raw_cache_preserved,
        ) = warmup_results[0]
        warmup_tokens = [] if raw_token_ids == "none" else [
            int(value) for value in raw_token_ids.split(",")
        ]
        if (
            int(raw_count) != execution_warmup
            or int(raw_family_id) != FAMILY_NAMES.index(selected)
            or int(raw_token_count) != len(tokens)
            or warmup_tokens != tokens
            or int(raw_token_digest, 16) != _token_id_digest(tokens)
            or int(raw_seed_p) != 1
            or int(raw_seed_r) != 2
            or int(raw_last_p) != len(tokens)
            or int(raw_last_r) != len(tokens) + 1
            or raw_cache_preserved != "1"
        ):
            raise BenchmarkFailure("native same-context warmup evidence is inconsistent")
    transitions: list[dict[str, int]] = []
    step_ms: list[float] = []
    for index, (step_family, raw_step, raw_past, raw_present, raw_token, raw_ms) in enumerate(steps):
        past_length = int(raw_past)
        present_length = int(raw_present)
        if (
            step_family != selected
            or int(raw_step) != index
            or past_length != index + 1
            or present_length != past_length + 1
            or int(raw_token) != tokens[index]
        ):
            raise BenchmarkFailure(f"native cache transition {index} is inconsistent")
        transitions.append(
            {"position": index, "pastLength": past_length, "presentLength": present_length}
        )
        step_ms.append(float(raw_ms))
    if (
        totals[0][0] != selected
        or int(totals[0][1]) != len(tokens)
        or int(shapes[0][0]) != int(shapes[0][1])
        or int(shapes[0][2]) != int(shapes[0][3])
        or int(shapes[0][2]) != int(shapes[0][0]) + 210
        or int(shapes[0][4]) != max_new + 1
    ):
        raise BenchmarkFailure("native aggregate/shape evidence is inconsistent")
    timing = _timing_summary(step_ms)
    return {
        "schema": SAMPLE_SCHEMA,
        "engine": "native-c",
        "backend": backend,
        "precision": precision,
        "provider": backend,
        "strictNoFallback": True,
        "family": selected,
        "familyId": FAMILY_NAMES.index(selected),
        "requestedFamily": family,
        "inputTensorSha256": input_hashes[0],
        "questionTokenIds": question_tokens,
        "tokenIds": tokens,
        "stoppedAtEos": tokens[-1] == 2,
        "shape": {
            "mode": "active",
            "B": 1,
            "Q": int(shapes[0][1]),
            "M": int(shapes[0][3]),
            "T": int(shapes[0][4]),
        },
        "cache": {
            "initialPastLength": 1,
            "finalPastLength": len(tokens) + 1,
            "sentinelMaskValue": 1,
            "transitions": transitions,
        },
        "timing": {
            "encoderExecutionMs": float(routers[0][2]),
            **timing,
            "decoderApplicationTotalMs": float(totals[0][2]),
            "processWallMs": process_wall_ms,
        },
        "runtime": {
            "name": "volvoxai-native",
            "cpuThreads": threads,
            "banner": "VolvoxAI Native Runtime",
            "executionWarmup": {
                "runs": execution_warmup,
                "sameContexts": True,
                "sameRuntime": True,
                "sameEncoderContext": True,
                "sameDecoderContext": True,
                "stateResetToSentinel": True,
                "tokenCacheTransitionParity": True,
            },
            **({"device": device} if device is not None else {}),
        },
        **({
            "routeEvidence": {
                "encoder": encoder_routes[0],
                "decoder": decoder_routes,
            },
        } if backend != "cpu" else {}),
    }


def run_native_sample(
    *,
    binary: Path,
    package: Path,
    precision: str,
    image: Path,
    prompt: str,
    family: str,
    max_new: int,
    threads: int,
    timeout: float,
    environment: Mapping[str, str],
    backend: str = "cpu",
    execution_warmup: int = 0,
) -> dict[str, Any]:
    _validate_execution_warmup(execution_warmup, "native")

    backend_flags = {
        "cpu": "--cpu",
        "vulkan": "--vulkan",
        "opengl": "--opengl",
        "cuda": "--cuda",
    }
    if backend not in backend_flags:
        raise BenchmarkFailure(f"unsupported native benchmark backend {backend!r}")
    command = [
        str(binary),
        str(package),
        "--image",
        str(image),
        "--prompt",
        prompt,
        "--family",
        family,
        "--max-new",
        str(max_new),
        "--shape-mode",
        "active",
        backend_flags[backend],
        "--timing",
    ]
    command.extend(["--threads", str(threads)])
    command.extend(["--warmup", str(execution_warmup)])
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=dict(environment),
    )
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if completed.returncode != 0:
        raise BenchmarkFailure(
            f"native/{precision} failed with exit {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return parse_native_sample(
        completed.stdout,
        completed.stderr,
        precision=precision,
        family=family,
        max_new=max_new,
        threads=threads,
        process_wall_ms=wall_ms,
        backend=backend,
        execution_warmup=execution_warmup,
    )


def validate_sample(
    sample: Mapping[str, Any], *, max_new: int,
    expected_strict_no_fallback: bool = True,
) -> None:
    tokens = sample.get("tokenIds")
    shape = sample.get("shape")
    cache = sample.get("cache")
    timing = sample.get("timing")
    if (
        sample.get("schema") != SAMPLE_SCHEMA
        or sample.get("strictNoFallback") is not expected_strict_no_fallback
        or not isinstance(sample.get("inputTensorSha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", sample["inputTensorSha256"])
        or not isinstance(tokens, list)
        or not 2 <= len(tokens) <= max_new
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in tokens)
        or not isinstance(shape, Mapping)
        or set(shape) != {"mode", "B", "Q", "M", "T"}
        or shape.get("mode") != "active"
        or shape.get("B") != 1
        or isinstance(shape.get("Q"), bool)
        or not isinstance(shape.get("Q"), int)
        or not 1 <= shape["Q"] <= 192
        or shape.get("M") != shape["Q"] + 210
        or shape.get("T") != max_new + 1
        or not isinstance(cache, Mapping)
        or cache.get("initialPastLength") != 1
        or cache.get("finalPastLength") != len(tokens) + 1
        or cache.get("sentinelMaskValue") != 1
        or not isinstance(cache.get("transitions"), list)
        or len(cache["transitions"]) != len(tokens)
        or not isinstance(timing, Mapping)
    ):
        raise BenchmarkFailure("runtime sample does not satisfy the explicit-KV contract")
    for index, transition in enumerate(cache["transitions"]):
        if transition != {
            "position": index,
            "pastLength": index + 1,
            "presentLength": index + 2,
        }:
            raise BenchmarkFailure(f"runtime sample cache transition {index} is invalid")
    for field in (
        "encoderExecutionMs",
        "decoderSeedMs",
        "decoderSteadyTotalMs",
        "decoderSteadyMeanMs",
        "decoderExecutionTotalMs",
    ):
        value = timing.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise BenchmarkFailure(f"runtime sample timing.{field} is invalid")


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise BenchmarkFailure("cannot summarize an empty or non-finite timing distribution")
    return {
        "count": len(values),
        "minimumMs": min(values),
        "medianMs": statistics.median(values),
        "meanMs": statistics.fmean(values),
        "maximumMs": max(values),
    }


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault((str(sample["engine"]), str(sample["precision"])), []).append(sample)
    tiers: dict[str, Any] = {}
    for (engine, precision), values in sorted(grouped.items()):
        timings = [value["timing"] for value in values]
        tier = {
            "engine": engine,
            "precision": precision,
            "samples": len(values),
            "family": values[0]["family"],
            "inputTensorSha256": values[0]["inputTensorSha256"],
            "tokenIds": values[0]["tokenIds"],
            "encoder": _distribution([float(value["encoderExecutionMs"]) for value in timings]),
            "decoderSeed": _distribution([float(value["decoderSeedMs"]) for value in timings]),
            "decoderSteadyMean": _distribution(
                [float(value["decoderSteadyMeanMs"]) for value in timings]
            ),
            "decoderExecutionTotal": _distribution(
                [float(value["decoderExecutionTotalMs"]) for value in timings]
            ),
        }
        phases = [value.get("executionPhases") for value in values]
        if all(isinstance(value, Mapping) for value in phases):
            decoder_phases = [value["decoder"] for value in phases]
            if any(not isinstance(value, list) or len(value) < 2 for value in decoder_phases):
                raise BenchmarkFailure("execution phase evidence has no steady decoder step")
            tier["executionPhases"] = {
                "encoderShapeBind": _distribution([
                    float(value["encoder"]["shapeBindMs"]) for value in phases
                ]),
                "encoderProvider": _distribution([
                    float(value["encoder"]["providerMs"]) for value in phases
                ]),
                "decoderSeedShapeBind": _distribution([
                    float(value[0]["shapeBindMs"]) for value in decoder_phases
                ]),
                "decoderSeedProvider": _distribution([
                    float(value[0]["providerMs"]) for value in decoder_phases
                ]),
                "decoderSteadyShapeBindMean": _distribution([
                    statistics.fmean(float(step["shapeBindMs"]) for step in value[1:])
                    for value in decoder_phases
                ]),
                "decoderSteadyProviderMean": _distribution([
                    statistics.fmean(float(step["providerMs"]) for step in value[1:])
                    for value in decoder_phases
                ]),
                "decoderSpecializationCacheHits": sum(
                    step.get("specializationCacheHit") is True
                    for value in decoder_phases for step in value
                ),
            }
        tiers[f"{engine}/{precision}"] = tier
    return tiers


def prove_parity(samples: Sequence[Mapping[str, Any]], repeat: int) -> dict[str, Any]:
    if len(samples) != 6 * repeat:
        raise BenchmarkFailure("benchmark matrix is incomplete")
    reference = samples[0]
    signature = (
        reference["family"], reference["familyId"],
        reference["inputTensorSha256"], reference["tokenIds"],
        reference["shape"], reference["cache"],
    )
    for sample in samples[1:]:
        if (
            sample["family"], sample["familyId"],
            sample["inputTensorSha256"], sample["tokenIds"],
            sample["shape"], sample["cache"],
        ) != signature:
            raise BenchmarkFailure(
                f"input/token/family parity failed for "
                f"{sample['engine']}/{sample['precision']}"
            )
    return {
        "exact": True,
        "family": reference["family"],
        "familyId": reference["familyId"],
        "inputTensorSha256": reference["inputTensorSha256"],
        "tokenIds": reference["tokenIds"],
        "shape": reference["shape"],
        "cache": reference["cache"],
        "matrixEntries": 6,
        "measuredSamples": len(samples),
    }


def _command_version(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output = (completed.stdout or completed.stderr).decode("utf-8", errors="replace").strip()
    if completed.returncode != 0 or not output:
        raise BenchmarkFailure(f"could not query runtime version: {' '.join(command)}")
    return output.splitlines()[0]


def _onnx_graph_summary(onnx_module: Any, path: Path) -> dict[str, Any]:
    model = onnx_module.load(str(path), load_external_data=False)
    counts = Counter(node.op_type for node in model.graph.node)
    return {
        "nodeCount": len(model.graph.node),
        "operatorCounts": dict(sorted(counts.items())),
    }


def _inspect_ort_optimized_graph(
    *,
    ort: Any,
    onnx_module: Any,
    source: Path,
    output: Path,
    logical_name: str,
    threads: int,
) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.optimized_model_filepath = str(output)
    session = ort.InferenceSession(
        str(source), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise BenchmarkFailure("ONNX Runtime graph inspection selected another provider")
    del session
    return {
        "source": _onnx_graph_summary(onnx_module, source),
        "optimized": {
            **_file_identity(output, name=logical_name),
            **_onnx_graph_summary(onnx_module, output),
        },
        "provider": "CPUExecutionProvider",
        "graphOptimizationLevel": "ORT_ENABLE_ALL",
        "intraOpThreadLimit": threads,
        "interOpThreadLimit": 1,
        "executionMode": "sequential",
    }


def _git_file_paths(repository: Path, arguments: Sequence[str], label: str) -> set[bytes]:
    completed = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "-z", *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise BenchmarkFailure(
            f"could not enumerate {label} worktree paths: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return {path for path in completed.stdout.split(b"\0") if path}


def _snapshot_field(digest: Any, domain: bytes, value: bytes) -> None:
    """Append one unambiguous, domain-separated byte field."""

    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _display_git_path(value: bytes) -> str:
    return value.decode("utf-8", errors="backslashreplace")


def _worktree_execution_source_snapshot(
    repository: Path, excluded_paths: Sequence[str]
) -> dict[str, Any]:
    """Hash every tracked and nonignored-untracked execution-source byte.

    The canonical identity is independent of directory enumeration order and
    records path, executable mode, file bytes, or symlink target with separate
    domains. Publication outputs and the four named user planning documents are
    the only omitted nonignored paths.
    """

    tracked = _git_file_paths(repository, ("--cached",), "tracked")
    untracked = _git_file_paths(
        repository, ("--others", "--exclude-standard"), "nonignored untracked"
    )
    overlap = tracked & untracked
    if overlap:
        raise BenchmarkFailure("git classified a worktree path as tracked and untracked")

    publication_exclusions = {
        Path(value).as_posix().encode("utf-8")
        for value in excluded_paths
        if value and not Path(value).is_absolute()
    }
    user_exclusions = {
        value.encode("utf-8") for value in USER_OWNED_PROVENANCE_EXCLUSIONS
    }
    excluded = publication_exclusions | user_exclusions
    digest = hashlib.sha256()
    _snapshot_field(
        digest, b"snapshot-format", WORKTREE_EXECUTION_SOURCE_SNAPSHOT_FORMAT.encode("ascii")
    )
    for relative in sorted(excluded):
        _snapshot_field(digest, b"excluded-path", relative)

    included_tracked = sorted(tracked - excluded)
    included_untracked = sorted(untracked - excluded)
    regular_file_count = 0
    symlink_count = 0
    missing_path_count = 0
    for relative in sorted((*included_tracked, *included_untracked)):
        _snapshot_field(digest, b"entry-path", relative)
        path = repository / os.fsdecode(relative)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            _snapshot_field(digest, b"entry-kind", b"missing")
            missing_path_count += 1
            continue
        if stat.S_ISREG(metadata.st_mode):
            mode = b"100755" if metadata.st_mode & 0o111 else b"100644"
            _snapshot_field(digest, b"entry-kind", b"regular-file")
            _snapshot_field(digest, b"entry-mode", mode)
            _snapshot_field(digest, b"entry-size", metadata.st_size.to_bytes(8, "big"))
            file_digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    file_digest.update(chunk)
            _snapshot_field(digest, b"entry-bytes-sha256", file_digest.digest())
            regular_file_count += 1
            continue
        if stat.S_ISLNK(metadata.st_mode):
            target = os.fsencode(os.readlink(path))
            _snapshot_field(digest, b"entry-kind", b"symbolic-link")
            _snapshot_field(digest, b"entry-mode", b"120000")
            _snapshot_field(digest, b"entry-link-target", target)
            symlink_count += 1
            continue
        raise BenchmarkFailure(f"worktree path has an unsupported file type: {path}")

    excluded_tracked = sorted(tracked & excluded)
    excluded_untracked = sorted(untracked & excluded)
    return {
        "format": WORKTREE_EXECUTION_SOURCE_SNAPSHOT_FORMAT,
        "sha256": digest.hexdigest(),
        "pathCount": len(included_tracked) + len(included_untracked),
        "trackedPathCount": len(included_tracked),
        "nonignoredUntrackedPathCount": len(included_untracked),
        "regularFileCount": regular_file_count,
        "symlinkCount": symlink_count,
        "missingPathCount": missing_path_count,
        "includedTrackedPaths": [
            _display_git_path(path) for path in included_tracked
        ],
        "includedNonignoredUntrackedPaths": [
            _display_git_path(path) for path in included_untracked
        ],
        "excludedPublicationPaths": sorted(
            _display_git_path(path) for path in publication_exclusions
        ),
        "excludedUserOwnedPlanningPaths": list(USER_OWNED_PROVENANCE_EXCLUSIONS),
        "excludedTrackedPaths": [_display_git_path(path) for path in excluded_tracked],
        "excludedNonignoredUntrackedPaths": [
            _display_git_path(path) for path in excluded_untracked
        ],
    }


def _git_provenance(
    repository: Path, excluded_paths: Sequence[str] = ()
) -> dict[str, Any]:
    def run(arguments: Sequence[str]) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise BenchmarkFailure(
                f"could not capture git provenance: "
                f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
            )
        return completed.stdout

    publication_exclusions = {
        Path(value).as_posix()
        for value in excluded_paths
        if value and not Path(value).is_absolute()
    }
    excluded = sorted(publication_exclusions | set(USER_OWNED_PROVENANCE_EXCLUSIONS))
    head = run(("rev-parse", "HEAD")).decode("ascii").strip()
    status = run((
        "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".",
        *(f":(top,exclude,literal){path}" for path in excluded),
    ))
    return {
        "head": head,
        "dirty": bool(status),
        "statusEntries": len([entry for entry in status.split(b"\0") if entry]),
        "statusSha256": hashlib.sha256(status).hexdigest(),
        "excludedStatusPaths": excluded,
        "excludedPublicationStatusPaths": sorted(publication_exclusions),
        "excludedUserOwnedPlanningStatusPaths": list(
            USER_OWNED_PROVENANCE_EXCLUSIONS
        ),
        "worktreeExecutionSource": _worktree_execution_source_snapshot(
            repository, sorted(publication_exclusions)
        ),
    }


def _assert_execution_inputs_stable(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    changed = sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )
    if changed:
        raise BenchmarkFailure(
            "execution inputs changed while the benchmark was running: "
            + ", ".join(changed)
        )


def _benchmark_child_environment(
    inherited: Mapping[str, str], threads: int
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return one privacy-safe, deterministic environment for benchmark children."""

    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise BenchmarkFailure("benchmark child thread limit must be a positive integer")
    invalid = [
        name for name, value in inherited.items()
        if not isinstance(name, str) or not isinstance(value, str) or "\0" in name + value
    ]
    if invalid:
        raise BenchmarkFailure("benchmark child environment must contain plain strings")

    removed = sorted(name for name in inherited if name.startswith("VOLVOX"))
    environment = {
        name: value for name, value in inherited.items() if name not in removed
    }
    forced_threads = {
        name: str(threads) for name in COMMON_THREAD_ENVIRONMENT_VARIABLES
    }
    environment.update(forced_threads)
    return environment, {
        "policy": (
            "inherit non-Volvox variables, remove every uppercase VOLVOX* runtime "
            "override, then force the common CPU thread-limit variables"
        ),
        "removedRuntimeOverrideNames": removed,
        "removedRuntimeOverrideValuesRecorded": False,
        "forcedThreadEnvironment": forced_threads,
        "appliedToEveryBenchmarkChildProcess": True,
    }


def _counterbalanced_tier_order(
    expected_tiers: Sequence[str], matrix_index: int
) -> list[str]:
    """Rotate deterministic forward/reverse pairs across matrix executions."""

    tiers = list(expected_tiers)
    if (
        not tiers
        or len(set(tiers)) != len(tiers)
        or isinstance(matrix_index, bool)
        or not isinstance(matrix_index, int)
        or matrix_index < 0
    ):
        raise BenchmarkFailure(
            "counterbalanced tier order requires unique tiers and a nonnegative index"
        )
    offset = (matrix_index // 2) % len(tiers)
    rotated = tiers[offset:] + tiers[:offset]
    return list(reversed(rotated)) if matrix_index % 2 else rotated


def _parse_cpuinfo(value: str) -> tuple[str | None, list[str]]:
    model = None
    processor_fallback = None
    features: set[str] = set()
    for raw_line in value.splitlines():
        key, separator, raw_value = raw_line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        parsed = raw_value.strip()
        if model is None and key in {"model name", "hardware"} and parsed:
            model = parsed
        elif (
            processor_fallback is None
            and key == "processor"
            and parsed
            and not parsed.isdecimal()
        ):
            processor_fallback = parsed
        if key in {"flags", "features"}:
            features.update(parsed.split())
    return model or processor_fallback, sorted(features)


def _allowed_cpu_topology(allowed_cpu_ids: Sequence[int]) -> dict[str, Any]:
    logical_cpus = []
    physical: dict[tuple[int, int], list[int]] = {}
    for cpu_id in allowed_cpu_ids:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
        try:
            package_id = int(
                (topology / "physical_package_id").read_text(encoding="ascii").strip()
            )
            core_id = int((topology / "core_id").read_text(encoding="ascii").strip())
        except (OSError, UnicodeError, ValueError):
            return {
                "available": False,
                "logicalCpus": [],
                "physicalCores": [],
                "physicalCoreCount": None,
                "oneLogicalCpuPerPhysicalCore": None,
            }
        logical_cpus.append(
            {"logicalCpuId": cpu_id, "packageId": package_id, "coreId": core_id}
        )
        physical.setdefault((package_id, core_id), []).append(cpu_id)
    physical_cores = [
        {
            "packageId": package_id,
            "coreId": core_id,
            "logicalCpuIds": sorted(logical_ids),
        }
        for (package_id, core_id), logical_ids in sorted(physical.items())
    ]
    return {
        "available": True,
        "logicalCpus": logical_cpus,
        "physicalCores": physical_cores,
        "physicalCoreCount": len(physical_cores),
        "oneLogicalCpuPerPhysicalCore": (
            bool(allowed_cpu_ids)
            and len(physical_cores) == len(allowed_cpu_ids)
            and all(len(value["logicalCpuIds"]) == 1 for value in physical_cores)
        ),
    }


def _host_cpu_identity() -> dict[str, Any]:
    model = platform.processor() or None
    features: list[str] = []
    cpuinfo = Path("/proc/cpuinfo")
    try:
        parsed_model, features = _parse_cpuinfo(cpuinfo.read_text(encoding="utf-8"))
        model = parsed_model or model
    except (OSError, UnicodeError):
        pass
    affinity_source = "os.cpu_count"
    allowed_cpu_ids = list(range(os.cpu_count() or 0))
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            allowed_cpu_ids = sorted(get_affinity(0))
            affinity_source = "sched_getaffinity"
        except OSError:
            pass
    topology = _allowed_cpu_topology(allowed_cpu_ids)
    logical_cpu_count = os.cpu_count()
    return {
        "model": model,
        "features": features,
        "allowedCpuIds": allowed_cpu_ids,
        "allowedLogicalCpuCount": len(allowed_cpu_ids),
        "allowedAffinitySource": affinity_source,
        "inheritedAffinityConstrained": (
            logical_cpu_count is not None
            and len(allowed_cpu_ids) < logical_cpu_count
        ),
        "allowedCpuTopology": topology,
    }


def _threading_settings(
    requested_cpu_threads: int, cpu_identity: Mapping[str, Any]
) -> dict[str, Any]:
    topology = cpu_identity["allowedCpuTopology"]
    return {
        "affinityPinnedByHarness": False,
        "inheritedAffinity": {
            "source": cpu_identity["allowedAffinitySource"],
            "allowedCpuIds": cpu_identity["allowedCpuIds"],
            "allowedLogicalCpuCount": cpu_identity["allowedLogicalCpuCount"],
            "constrained": cpu_identity["inheritedAffinityConstrained"],
            "allowedPhysicalCoreCount": topology["physicalCoreCount"],
            "oneLogicalCpuPerPhysicalCore": topology[
                "oneLogicalCpuPerPhysicalCore"
            ],
        },
        "onnxruntime": {
            "intraOpThreadLimit": requested_cpu_threads,
            "interOpThreadLimit": 1,
            "executionMode": "sequential",
            "graphOptimizationLevel": "ORT_ENABLE_ALL",
            "affinityPolicy": "inherited; not pinned by benchmark harness",
        },
        "nativeC": {
            "cpuThreadLimit": requested_cpu_threads,
            "affinityPolicy": "inherited; not pinned by benchmark harness",
        },
        "volvoxaiWasm": {
            "executionThreads": 1,
            "workerThreads": 0,
            "cpuThreadLimitApplied": False,
            "threadCountScope": "VolvoxAI WASM engine only",
            "hostRuntimeAuxiliaryThreads": (
                "Node/V8 auxiliary threads are not counted or limited by the "
                "harness; they share the process's inherited CPU affinity"
            ),
        },
    }


def _atomic_write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o644)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import PIL
    except ImportError as error:
        raise BenchmarkFailure(
            "benchmark requires NumPy, Pillow, ONNX, and ONNX Runtime"
        ) from error
    try:
        from .import_hf_split_onnx import validate_source
        from .tiny_receipt_tokenizer import TinyReceiptTokenizer
    except ImportError:  # Direct script execution keeps the tools directory on sys.path.
        from import_hf_split_onnx import validate_source
        from tiny_receipt_tokenizer import TinyReceiptTokenizer

    repository = Path(__file__).resolve().parents[3]
    publication_outputs = set(PUBLICATION_OUTPUT_PATHS)
    if args.report is not None:
        try:
            publication_outputs.add(
                args.report.resolve().relative_to(repository.resolve()).as_posix()
            )
        except ValueError:
            pass

    source_root = args.source.resolve(strict=True)
    sources = {
        "fp32": dict(validate_source(source_root, variant="fp32")),
        "int8": dict(validate_source(source_root, variant="int8-w8a8")),
    }
    for source in sources.values():
        source["tokenizer_runtime"] = create_source_tokenizer(
            source, TinyReceiptTokenizer
        )
    packages = {
        "fp32": args.fp32_package.resolve(strict=True),
        "int8": args.int8_package.resolve(strict=True),
    }
    package_identities = {
        precision: load_package_identity(packages[precision], precision, sources[precision])
        for precision in ("fp32", "int8")
    }
    binary = args.native_binary.resolve(strict=True)
    runner = args.runtime_runner.resolve(strict=True)
    api = args.api.resolve(strict=True)
    wasm = args.wasm.resolve(strict=True)
    input_image = args.image.resolve(strict=True) if args.image is not None else None
    for executable, label in ((binary, "native binary"),):
        if executable.is_symlink() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise BenchmarkFailure(f"{label} must be a real executable file")
    prompt = unicodedata.normalize("NFC", args.prompt)
    if not prompt or "\0" in prompt:
        raise BenchmarkFailure("--prompt must be non-empty NFC text")
    if args.family not in ("auto", *FAMILY_NAMES):
        raise BenchmarkFailure("--family is not a canonical TinyReceipt family")

    source_identities = _validated_source_identities(sources)
    generator_identity = _file_identity(
        Path(__file__).resolve(), name="benchmark_explicit_kv.py"
    )
    execution_inputs_before = {
        "repository": _git_provenance(repository, sorted(publication_outputs)),
        "generator": generator_identity,
        "source": source_identities,
        "packages": package_identities,
        "nativeBinary": _file_identity(binary),
        "volvoxaiApi": _file_identity(api),
        "volvoxaiWasm": _file_identity(wasm),
        "runtimeRunner": _file_identity(runner),
        **({"inputImage": _file_identity(input_image)} if input_image is not None else {}),
    }

    child_environment, child_environment_policy = _benchmark_child_environment(
        os.environ, args.threads
    )
    expected_tiers = [
        f"{engine}/{precision}"
        for precision in ("fp32", "int8")
        for engine in ("onnxruntime", "native-c", "volvoxai-wasm")
    ]

    with tempfile.TemporaryDirectory(prefix="volvox-tinyreceipt-kv-bench-") as directory:
        ort_graph_inspection = {
            precision: {
                role: _inspect_ort_optimized_graph(
                    ort=ort,
                    onnx_module=onnx,
                    source=Path(sources[precision]["selected_paths"][role]),
                    output=Path(directory) / f"ort-{precision}-{role}.onnx",
                    logical_name=f"{precision}/{role}/optimized.onnx",
                    threads=args.threads,
                )
                for role in ("encoder", "decoder")
            }
            for precision in ("fp32", "int8")
        }
        if input_image is None:
            image = Path(directory) / "synthetic-receipt.png"
            image_identity = create_synthetic_image(image)
        else:
            image = input_image
            image_identity = {
                "kind": "user-supplied-image",
                "file": _file_identity(image),
            }

        qualification_run_order = [
            f"{engine}/{precision}"
            for precision in ("fp32", "int8")
            for engine in ("native-c/cpu", "volvoxai-wasm/wasm")
        ]
        qualification_entries: list[dict[str, Any]] = []
        for precision in ("fp32", "int8"):
            qualification_entries.append(run_native_dynamic_qualification(
                binary=binary,
                package=packages[precision],
                precision=precision,
                image=image,
                backend="cpu",
                max_new=CANONICAL_MAX_NEW,
                threads=args.threads,
                timeout=args.timeout,
                environment=child_environment,
            ))
            qualification_entries.append(run_wasm_dynamic_qualification(
                node=args.node,
                runner=runner,
                api=api,
                wasm=wasm,
                package=packages[precision],
                precision=precision,
                image=image,
                timeout=args.timeout,
                environment=child_environment,
            ))
        dynamic_qualification = prove_dynamic_qualification_coverage(
            qualification_entries,
            native_backends=("cpu",),
            include_webgpu=False,
            include_wasm=True,
            expected_question_token_ids=(
                sources["fp32"]["tokenizer_runtime"].encode(
                    "x", add_eos=True, max_len=192
                ),
                sources["fp32"]["tokenizer_runtime"].encode(
                    DEFAULT_PROMPT, add_eos=True, max_len=192
                ),
                sources["fp32"]["tokenizer_runtime"].encode(
                    DEFAULT_PROMPT, add_eos=True, max_len=192
                ),
                sources["fp32"]["tokenizer_runtime"].encode(
                    "x", add_eos=True, max_len=192
                ),
            ),
        )

        def matrix_once(run_order: Sequence[str]) -> list[dict[str, Any]]:
            runners: dict[str, Any] = {}
            for precision in ("fp32", "int8"):
                runners[f"onnxruntime/{precision}"] = (
                    lambda precision=precision: run_ort_sample(
                        sources[precision], precision, image, prompt,
                        args.family, args.max_new, args.threads,
                    )
                )
                runners[f"native-c/{precision}"] = (
                    lambda precision=precision: run_native_sample(
                        binary=binary,
                        package=packages[precision],
                        precision=precision,
                        image=image,
                        prompt=prompt,
                        family=args.family,
                        max_new=args.max_new,
                        threads=args.threads,
                        timeout=args.timeout,
                        environment=child_environment,
                    )
                )
                runners[f"volvoxai-wasm/{precision}"] = (
                    lambda precision=precision: run_volvox_wasm_sample(
                        node=args.node,
                        runner=runner,
                        api=api,
                        wasm=wasm,
                        package=packages[precision],
                        precision=precision,
                        image=image,
                        prompt=prompt,
                        family=args.family,
                        max_new=args.max_new,
                        timeout=args.timeout,
                        environment=child_environment,
                    )
                )
            if (
                set(runners) != set(expected_tiers)
                or list(run_order) != list(dict.fromkeys(run_order))
                or set(run_order) != set(expected_tiers)
            ):
                raise BenchmarkFailure(
                    "CPU matrix run order does not cover each tier exactly once"
                )
            values = [runners[tier]() for tier in run_order]
            for value in values:
                validate_sample(value, max_new=args.max_new)
            prove_parity(values, 1)
            return values

        run_orders: list[dict[str, Any]] = []
        for warmup_index in range(args.warmup):
            run_order = _counterbalanced_tier_order(
                expected_tiers, warmup_index
            )
            run_orders.append({
                "matrixIndex": warmup_index,
                "phase": "warmup",
                "phaseIndex": warmup_index,
                "tiers": run_order,
            })
            matrix_once(run_order)
        samples: list[dict[str, Any]] = []
        for sample_index in range(args.repeat):
            matrix_index = args.warmup + sample_index
            run_order = _counterbalanced_tier_order(expected_tiers, matrix_index)
            run_orders.append({
                "matrixIndex": matrix_index,
                "phase": "measured",
                "phaseIndex": sample_index,
                "tiers": run_order,
            })
            for sample in matrix_once(run_order):
                sample["sampleIndex"] = sample_index
                samples.append(sample)
        parity = prove_parity(samples, args.repeat)

        execution_inputs_after = {
            "repository": _git_provenance(repository, sorted(publication_outputs)),
            "generator": _file_identity(
                Path(__file__).resolve(), name="benchmark_explicit_kv.py"
            ),
            "source": _validated_source_identities(sources),
            "packages": {
                precision: load_package_identity(
                    packages[precision], precision, sources[precision]
                )
                for precision in ("fp32", "int8")
            },
            "nativeBinary": _file_identity(binary),
            "volvoxaiApi": _file_identity(api),
            "volvoxaiWasm": _file_identity(wasm),
            "runtimeRunner": _file_identity(runner),
            **({"inputImage": _file_identity(input_image)}
               if input_image is not None else {}),
        }
        _assert_execution_inputs_stable(
            execution_inputs_before, execution_inputs_after
        )

    cpu_identity = _host_cpu_identity()
    return {
        "format": REPORT_FORMAT,
        "status": "pass",
        "provenance": {
            "repository": execution_inputs_before["repository"],
            "generator": generator_identity,
            "executionInputStability": {
                "checkedBeforeAndAfterExecution": True,
                "unchanged": True,
            },
        },
        "workload": {
            "prompt": prompt,
            "questionTokenIds": sources["fp32"]["tokenizer_runtime"].encode(
                prompt, add_eos=True, max_len=192
            ),
            "family": args.family,
            "maxNewTokens": args.max_new,
            "image": image_identity,
            "decode": {
                "strategy": "greedy-first-index-explicit-kv",
                "initialPastLength": 1,
                "sentinelMaskValue": 1,
                "presentLengthRelation": "R=P+1",
                "minimumMeasuredDecoderSteps": 2,
            },
        },
        "settings": {
            "warmupMatrices": args.warmup,
            "measuredRepeatsPerTier": args.repeat,
            "cpuThreads": args.threads,
            "cpuThreadsMeaning": (
                "requested ONNX Runtime intra-op and native C CPU thread limit; "
                "not applied to single-threaded VolvoxAI WASM"
            ),
            "threading": _threading_settings(args.threads, cpu_identity),
            "childProcessEnvironment": {
                **child_environment_policy,
                "scope": (
                    "all native C and VolvoxAI WASM qualification, warmup, and "
                    "measured children"
                ),
            },
            "dynamicShapeQualification": (
                "untimed canonical short/grow/maximum-padded/shrink sequence; "
                "one fresh process per native CPU/WASM precision entry, each "
                "reusing one runtime/session and the same encoder/decoder contexts"
            ),
            "dynamicShapeQualificationRunOrder": qualification_run_order,
            "sampleIsolation": SAMPLE_ISOLATION_SCOPE,
            "graphInspectionScope": (
                "one non-executing ORT CPU session per source graph serializes the "
                "optimized graph before warmup; inspection is outside all timings"
            ),
            "warmupScope": WARMUP_ISOLATION_SCOPE,
            "timingScope": (
                "encoder and one-token decoder execution only; package/model loading, "
                "compilation, preprocessing, tokenization, and process startup excluded"
            ),
            "runOrderPolicy": (
                "deterministic forward/reverse pairs, rotating the first tier by "
                "one position after every pair; warmup and measured matrices share "
                "one continuous matrix index"
            ),
            "runOrderPerMatrix": run_orders,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "logicalCpuCount": os.cpu_count(),
            "cpu": cpu_identity,
            "python": platform.python_version(),
            "node": _command_version([args.node, "--version"]),
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "artifacts": {
            "source": source_identities,
            "onnxRuntimeGraphInspection": ort_graph_inspection,
            "packages": package_identities,
            "nativeBinary": execution_inputs_before["nativeBinary"],
            "volvoxaiApi": execution_inputs_before["volvoxaiApi"],
            "volvoxaiWasm": execution_inputs_before["volvoxaiWasm"],
            "benchmarkHarness": generator_identity,
            "runtimeRunner": execution_inputs_before["runtimeRunner"],
        },
        "dynamicShapeQualification": dynamic_qualification,
        "parity": parity,
        "summary": {"tiers": summarize_samples(samples)},
        "samples": samples,
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Benchmark the six-entry TinyReceipt explicit-KV v1 matrix."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fp32-package", type=Path, required=True)
    parser.add_argument("--int8-package", type=Path, required=True)
    parser.add_argument(
        "--native-binary",
        type=Path,
        default=repository / "examples/target/bin/tiny_receipt_split_w8a8",
    )
    parser.add_argument(
        "--api", type=Path, default=repository / "dist/0.4.0/volvoxai.js"
    )
    parser.add_argument(
        "--wasm", type=Path, default=repository / "dist/0.4.0/volvoxai.wasm"
    )
    parser.add_argument(
        "--runtime-runner",
        type=Path,
        default=Path(__file__).with_name("benchmark_explicit_kv_runtime.mjs"),
    )
    parser.add_argument("--node", default="node")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=("auto", *FAMILY_NAMES))
    parser.add_argument("--max-new", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not 2 <= args.max_new <= MAXIMUM_NEW_TOKENS:
        parser.error(f"--max-new must be in [2, {MAXIMUM_NEW_TOKENS}]")
    if not 0 <= args.warmup <= MAXIMUM_EXECUTION_WARMUP_RUNS \
            or args.repeat < 1 or args.threads < 1 or args.timeout <= 0:
        parser.error(
            f"--warmup must be in [0, {MAXIMUM_EXECUTION_WARMUP_RUNS}], "
            "--repeat/--threads >= 1, and --timeout > 0"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        report = build_report(args)
        if args.report is not None:
            _atomic_write(args.report.resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (BenchmarkFailure, OSError, subprocess.SubprocessError) as error:
        print(f"benchmark_explicit_kv: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
