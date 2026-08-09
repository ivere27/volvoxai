#!/usr/bin/env python3
"""Fresh-runtime TinyReceipt explicit-KV GPU benchmark matrix.

This is intentionally separate from ``benchmark_explicit_kv.py`` so the
existing CPU/Native/WASM matrix and its report format remain unchanged.  A
matrix contains an ONNX Runtime provider reference, strict native GPU
backends, and, unless disabled, a strict physical WebGPU browser for FP32 and
W8A8. Python ONNX Runtime is strict by default; accelerator-first CPU fallback
is an explicit non-strict opt-in. The pinned ORT WebGPU reference is necessarily
partitioned with its CPU EP and records optimized-session partition counts plus
a rejected strict probe. Every sample owns a fresh process (and, for WebGPU, a
fresh browser/profile).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .benchmark_explicit_kv import (
        DEFAULT_FAMILY,
        DEFAULT_PROMPT,
        CANONICAL_MAX_NEW,
        DYNAMIC_QUALIFICATION_SCHEMA,
        FAMILY_NAMES,
        MAXIMUM_EXECUTION_WARMUP_RUNS,
        PUBLICATION_OUTPUT_PATHS,
        SAMPLE_PREFIX,
        _DYNAMIC_CHECKS,
        _DYNAMIC_RUN_LABELS,
        BenchmarkFailure,
        _assert_execution_inputs_stable,
        _atomic_write,
        _benchmark_child_environment,
        _command_version,
        _counterbalanced_tier_order,
        _distribution,
        _file_identity,
        _git_provenance,
        _host_cpu_identity,
        _parse_json_sample,
        _maximum_encoder_binding,
        _plain_int,
        _validate_dynamic_runs,
        _validated_source_identities,
        create_source_tokenizer,
        create_synthetic_image,
        load_package_identity,
        parse_native_dynamic_qualification,
        prove_dynamic_qualification_coverage,
        run_native_dynamic_qualification,
        run_native_sample,
        run_ort_sample,
        validate_sample,
    )
except ImportError:  # Direct script execution keeps the tools directory on sys.path.
    from benchmark_explicit_kv import (
        DEFAULT_FAMILY,
        DEFAULT_PROMPT,
        CANONICAL_MAX_NEW,
        DYNAMIC_QUALIFICATION_SCHEMA,
        FAMILY_NAMES,
        MAXIMUM_EXECUTION_WARMUP_RUNS,
        PUBLICATION_OUTPUT_PATHS,
        SAMPLE_PREFIX,
        _DYNAMIC_CHECKS,
        _DYNAMIC_RUN_LABELS,
        BenchmarkFailure,
        _assert_execution_inputs_stable,
        _atomic_write,
        _benchmark_child_environment,
        _command_version,
        _counterbalanced_tier_order,
        _distribution,
        _file_identity,
        _git_provenance,
        _host_cpu_identity,
        _parse_json_sample,
        _maximum_encoder_binding,
        _plain_int,
        _validate_dynamic_runs,
        _validated_source_identities,
        create_source_tokenizer,
        create_synthetic_image,
        load_package_identity,
        parse_native_dynamic_qualification,
        prove_dynamic_qualification_coverage,
        run_native_dynamic_qualification,
        run_native_sample,
        run_ort_sample,
        validate_sample,
    )


REPORT_FORMAT = "volvoxai.tiny-receipt-explicit-kv-gpu-benchmark/v1"
NATIVE_GPU_BACKENDS = ("vulkan", "opengl", "cuda")
WEBGPU_RESULT_PREFIX = "RESULT_JSON "
ORT_WEBGPU_VERSION = "1.27.0"
ORT_WEBGPU_ASSETS = (
    "ort.webgpu.min.mjs",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
)
ORT_WEBGPU_ASSET_SHA256 = {
    "ort.webgpu.min.mjs":
        "46988a5a025f49449850f39f95eb0d21e40e67b3beb13a0b54efd3ab5d83f60e",
    "ort-wasm-simd-threaded.asyncify.mjs":
        "7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655",
    "ort-wasm-simd-threaded.asyncify.wasm":
        "7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a",
}

ORT_WEBGPU_PINNED_MODELS = {
    "fp32": {
        "encoder": {
            "sha256": "0e2206c54756b15d697d2bbd8d22bb68d0d2475d0323965ca03a14fb186e6eb0",
            "providers": {"CPUExecutionProvider": 70, "WebGpuExecutionProvider": 504},
        },
        "decoder": {
            "sha256": "b918a9de531bad205990acff28a2a7e2244b14229c312ed6b2f75d0540d6a6a2",
            "providers": {"CPUExecutionProvider": 3, "WebGpuExecutionProvider": 244},
        },
    },
    "int8": {
        "encoder": {
            "sha256": "a5be30f7507f8e2988c4cf735a80fb6c09a4f1e93c9759b1fc5e12d04c4da45c",
            "providers": {"CPUExecutionProvider": 282, "WebGpuExecutionProvider": 1223},
        },
        "decoder": {
            "sha256": "75437beed7cdac93b4a704e9a4ffbefbe0199855344a57cb8f78ae105e915419",
            "providers": {"CPUExecutionProvider": 133, "WebGpuExecutionProvider": 676},
        },
    },
}
ORT_WEBGPU_STRICT_ERROR_CLAUSES = (
    "assigned to the default CPU EP",
    "fallback to CPU EP has been explicitly disabled",
)
SOFTWARE_ADAPTER_RE = re.compile(
    r"\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software renderer|"
    r"software rasterizer|microsoft basic render|cpu)\b",
    re.IGNORECASE,
)
def _strict_route(route: Any, label: str) -> None:
    operator = route.get("operator") if isinstance(route, Mapping) else None
    if (
        not isinstance(operator, Mapping)
        or route.get("tierFallback") is not False
        or operator.get("attestation") != "none"
        or operator.get("used") is not False
        or operator.get("offendingNode") is not None
    ):
        raise BenchmarkFailure(f"{label} does not attest strict fallback 0")


def _strict_webgpu_compilation(value: Any, expected_label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("label") != expected_label:
        raise BenchmarkFailure(f"WebGPU result lacks its {expected_label} compilation")
    report = value.get("report")
    if not isinstance(report, Mapping):
        raise BenchmarkFailure(f"WebGPU {expected_label} compilation report is invalid")
    policy = report.get("requestedPolicy")
    candidates = report.get("candidates")
    if (
        not isinstance(policy, Mapping)
        or policy.get("mode") != "require"
        or policy.get("backend") != "webgpu"
        or policy.get("operatorFallback") != "forbid"
        or report.get("selectedBackend") != "webgpu"
        or not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], Mapping)
        or candidates[0].get("backend") != "webgpu"
        or candidates[0].get("outcome") != "selected"
    ):
        raise BenchmarkFailure(
            f"WebGPU {expected_label} compilation did not strictly select WebGPU"
        )
    _strict_route(report.get("routeEvidence"), f"WebGPU {expected_label} compilation")
    _strict_route(
        candidates[0].get("routeEvidence"),
        f"WebGPU {expected_label} compilation candidate",
    )
    return report


def _strict_webgpu_execution(value: Any, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("backend") != "webgpu"
        or value.get("outcome") != "success"
        or value.get("operatorFallback") != "none"
    ):
        raise BenchmarkFailure(f"{label} did not execute on strict WebGPU")
    _strict_route(value.get("routeEvidence"), label)


def _physical_webgpu_adapter(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise BenchmarkFailure(
            "WebGPU adapter identity is unavailable; refusing a physical-GPU label"
        )
    identity = {
        str(name): item.strip()
        for name, item in value.items()
        if isinstance(item, str) and item.strip()
    }
    description = " ".join(identity.values())
    if not description:
        raise BenchmarkFailure("WebGPU adapter identity is empty")
    if SOFTWARE_ADAPTER_RE.search(description):
        raise BenchmarkFailure(f"WebGPU software adapter is not a GPU benchmark: {description}")
    return identity


def _canonical_webgpu_adapter_class(value: Any) -> dict[str, str]:
    """Return a conservative adapter-class identity, never a physical-device ID."""

    identity = _physical_webgpu_adapter(value)
    canonical = {
        name: " ".join(unicodedata.normalize("NFKC", identity[name]).split()).casefold()
        for name in ("vendor", "architecture")
        if name in identity
    }
    if set(canonical) != {"vendor", "architecture"}:
        raise BenchmarkFailure(
            "WebGPU adapter class requires non-empty vendor and architecture"
        )
    return canonical


def load_ort_webgpu_identity(root: Path) -> tuple[Path, dict[str, Any]]:
    """Validate the exact external ORT Web distribution used by Chrome."""

    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise BenchmarkFailure("--ort-web-root must be a directory")
    package_path = resolved / "package.json"
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkFailure(
            "--ort-web-root/package.json is not valid UTF-8 JSON"
        ) from error
    if (
        not isinstance(package, Mapping)
        or package.get("name") != "onnxruntime-web"
        or package.get("version") != ORT_WEBGPU_VERSION
    ):
        raise BenchmarkFailure(
            f"--ort-web-root must be onnxruntime-web@{ORT_WEBGPU_VERSION}"
        )
    assets: dict[str, Any] = {}
    for name in ORT_WEBGPU_ASSETS:
        identity = _file_identity(resolved / "dist" / name)
        if identity["sha256"] != ORT_WEBGPU_ASSET_SHA256[name]:
            raise BenchmarkFailure(
                f"onnxruntime-web@{ORT_WEBGPU_VERSION} asset digest mismatch: {name}"
            )
        assets[name] = identity
    return resolved, {
        "name": "onnxruntime-web",
        "version": ORT_WEBGPU_VERSION,
        "package": _file_identity(package_path),
        "assets": assets,
    }


def _physical_native_gpu_device(sample: Any) -> dict[str, Any]:
    """Require an identified physical device for one native GPU sample."""

    if not isinstance(sample, Mapping) or sample.get("engine") != "native-c":
        raise BenchmarkFailure("native GPU sample identity is unavailable")
    backend = sample.get("backend")
    if backend not in NATIVE_GPU_BACKENDS:
        raise BenchmarkFailure(f"native GPU sample has invalid backend {backend!r}")
    runtime = sample.get("runtime")
    device = runtime.get("device") if isinstance(runtime, Mapping) else None
    if not isinstance(device, Mapping):
        raise BenchmarkFailure(
            f"native {backend} device identity is unavailable; "
            "refusing a physical-GPU label"
        )
    required_fields = {
        "vulkan": ("name",),
        "opengl": ("vendor", "renderer"),
        "cuda": ("name",),
    }[backend]
    identity_parts: list[str] = []
    for field in required_fields:
        value = device.get(field)
        if not isinstance(value, str) or not value.strip():
            raise BenchmarkFailure(
                f"native {backend} device identity lacks non-empty {field!r}"
            )
        identity_parts.append(value.strip())
    description = " ".join(identity_parts)
    if SOFTWARE_ADAPTER_RE.search(description):
        raise BenchmarkFailure(
            f"native {backend} software adapter is not a GPU benchmark: {description}"
        )
    return dict(device)


def _specialization_writes(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise BenchmarkFailure(f"{label} lacks its execution-phase evidence")
    evidence = value.get("specializationWrites")
    fields = ("writeCount", "writeBytes", "skipCount", "skipBytes", "contentBytes")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != set(fields)
        or any(
            isinstance(evidence.get(name), bool)
            or not isinstance(evidence.get(name), int)
            or evidence[name] < 0
            for name in fields
        )
    ):
        raise BenchmarkFailure(
            f"{label} lacks complete non-negative specialization-write counters"
        )
    return {name: int(evidence[name]) for name in fields}


def _volvox_webgpu_kv_evidence(
    value: Any,
    *,
    token_count: int,
    qualified: bool,
    label: str,
) -> dict[str, Any]:
    """Require exact same-device explicit-KV handoff/readback evidence."""

    expected_mode = "device-qualified" if qualified else "device-resident"
    if (
        not isinstance(value, Mapping)
        or value.get("enabled") is not True
        or value.get("mode") != expected_mode
        or value.get("crossCacheOutputs") != 8
        or value.get("presentCacheOutputsPerStep") != 8
        or value.get("encoderCrossCacheHandoffs") != token_count * 8
        or value.get("decoderCacheHandoffs") != max(0, token_count - 1) * 8
        or value.get("runtimeValidatedDeviceInputs") is not True
        or value.get("encoderResultRetainedThroughDecode") is not True
        or value.get("decoderResultRetainedUntilSuccessorExecution") is not True
        or value.get("encoderMemoryReadback") is not qualified
        or value.get("encoderMemoryReadbackValidated") is not qualified
        or value.get("encoderCrossCacheReadbackValidated") is not qualified
        or value.get("cachePrefixReadbackValidated") is not qualified
        or value.get("appendedCacheReadbackValidated") is not qualified
        or value.get("cacheReadbackFree") is not (not qualified)
    ):
        raise BenchmarkFailure(
            f"{label} lacks exact same-device explicit-KV handoff/readback evidence"
        )
    return dict(value)


def _parse_webgpu_result(stdout: bytes, label: str) -> Mapping[str, Any]:
    try:
        lines = stdout.decode("utf-8").splitlines()
        encoded = [
            line[len(WEBGPU_RESULT_PREFIX) :]
            for line in lines
            if line.startswith(WEBGPU_RESULT_PREFIX)
        ]
        if len(encoded) != 1:
            raise ValueError(f"expected one result, received {len(encoded)}")
        value = json.loads(encoded[0])
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BenchmarkFailure(f"{label} did not emit one WebGPU JSON result: {error}") from error
    if not isinstance(value, Mapping) or value.get("status") != "pass":
        detail = value.get("error") if isinstance(value, Mapping) else None
        raise BenchmarkFailure(f"{label} failed: {detail or 'invalid result'}")
    return value


def parse_webgpu_dynamic_qualification(
    stdout: bytes,
    *,
    precision: str,
    prompt: str,
    family: str,
    max_new: int,
) -> dict[str, Any]:
    """Validate one untimed, fresh-browser WebGPU dynamic-shape proof."""

    result = _parse_webgpu_result(stdout, f"WebGPU/{precision} qualification")
    input_hash = result.get("inputTensorSha256")
    if (
        result.get("mode") != "dynamic-rebind-qualification"
        or result.get("timed") is not False
        or result.get("precision") != precision
        or result.get("prompt") != prompt
        or result.get("family") != family
        or result.get("maxNewTokens") != max_new
        or not isinstance(input_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", input_hash) is None
    ):
        raise BenchmarkFailure(
            "WebGPU dynamic qualification does not identify the canonical workload"
        )
    compiles = result.get("compiles")
    if not isinstance(compiles, list) or len(compiles) != 2:
        raise BenchmarkFailure(
            "WebGPU dynamic qualification must contain two strict compilations"
        )
    encoder_compile = _strict_webgpu_compilation(compiles[0], "encoder")
    decoder_compile = _strict_webgpu_compilation(compiles[1], "decoder")
    adapter = _physical_webgpu_adapter(result.get("adapter"))
    if (
        encoder_compile.get("selectedDevice") != result.get("adapter")
        or decoder_compile.get("selectedDevice") != result.get("adapter")
    ):
        raise BenchmarkFailure(
            "WebGPU dynamic qualification compiled on different adapters"
        )
    qualification = result.get("qualification")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("schema") != DYNAMIC_QUALIFICATION_SCHEMA
        or qualification.get("timed") is not False
        or qualification.get("boundedDecoderMaximumNewTokens") != max_new
        or not _maximum_encoder_binding(
            qualification.get("maximumLegalEncoderBinding")
        )
        or qualification.get("sameSession") is not True
        or qualification.get("sameRuntime") is not True
        or qualification.get("sameEncoderContext") is not True
        or qualification.get("sameDecoderContext") is not True
        or qualification.get("strictNoFallback") is not True
        or qualification.get("deviceResidentKVQualified") is not True
    ):
        raise BenchmarkFailure(
            "WebGPU dynamic qualification lacks same-context strict evidence"
        )
    checks = qualification.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _DYNAMIC_CHECKS
        or any(checks.get(name) is not True for name in _DYNAMIC_CHECKS)
    ):
        raise BenchmarkFailure(
            "WebGPU dynamic qualification lacks complete correctness checks"
        )
    raw_runs = qualification.get("runs")
    runs = _validate_dynamic_runs(
        raw_runs,
        max_new=max_new,
        require_labels=True,
        expected_family=family,
    )
    for index, (raw, run) in enumerate(zip(raw_runs, runs, strict=True)):
        run["gpuResidentKv"] = _volvox_webgpu_kv_evidence(
            raw.get("gpuResidentKv") if isinstance(raw, Mapping) else None,
            token_count=len(run["tokenIds"]),
            qualified=True,
            label=f"WebGPU qualification run {index}",
        )
    attestations = qualification.get("executionAttestations")
    if not isinstance(attestations, list) or len(attestations) != len(runs):
        raise BenchmarkFailure(
            "WebGPU dynamic qualification lacks per-run execution attestations"
        )
    for index, (run, attestation) in enumerate(
        zip(runs, attestations, strict=True)
    ):
        if not isinstance(attestation, Mapping):
            raise BenchmarkFailure(
                f"WebGPU qualification run {index} attestation is invalid"
            )
        _strict_webgpu_execution(
            attestation.get("encoder"),
            f"WebGPU qualification run {index} encoder",
        )
        decoder = attestation.get("decoder")
        if not isinstance(decoder, list) or len(decoder) != len(run["tokenIds"]):
            raise BenchmarkFailure(
                f"WebGPU qualification run {index} lacks decoder attestations"
            )
        for step, value in enumerate(decoder):
            _strict_webgpu_execution(
                value, f"WebGPU qualification run {index} decoder step {step}"
            )
    return {
        "schema": DYNAMIC_QUALIFICATION_SCHEMA,
        "engine": "volvoxai-webgpu",
        "backend": "webgpu",
        "precision": precision,
        "timed": False,
        "freshProcess": True,
        "freshBrowser": True,
        "sameSession": True,
        "sameRuntime": True,
        "sameEncoderContext": True,
        "sameDecoderContext": True,
        "strictNoFallback": True,
        "deviceResidentKVQualified": True,
        "boundedDecoderMaximumNewTokens": max_new,
        "maximumLegalEncoderBinding": {"B": 1, "Q": 192, "M": 402},
        "inputTensorSha256": input_hash,
        "checks": {name: True for name in sorted(_DYNAMIC_CHECKS)},
        "runs": runs,
        "runtime": {
            "adapter": adapter,
            "packedDot4": result.get("packedDot4") is True,
            "gpuResidentKv": {
                "mode": "device-qualified",
                "sameDeviceHandoff": True,
                "cacheCorrectnessReadback": True,
            },
        },
        "routeEvidence": {
            "compilation": compiles,
            "execution": attestations,
        },
    }


def parse_webgpu_sample(
    stdout: bytes,
    *,
    precision: str,
    prompt: str,
    family: str,
    max_new: int,
    process_wall_ms: float,
    execution_warmup: int = 1,
) -> dict[str, Any]:
    """Validate one fresh-browser result and convert it to the shared schema."""

    if not math.isfinite(process_wall_ms) or process_wall_ms < 0:
        raise BenchmarkFailure("WebGPU process wall time is invalid")
    result = _parse_webgpu_result(stdout, f"WebGPU/{precision}")
    if (
        result.get("prompt") != prompt
        or result.get("family") != family
        or result.get("precision") != precision
        or result.get("maxNewTokens") != max_new
        or result.get("warmup") != execution_warmup
        or result.get("iterations") != 1
    ):
        raise BenchmarkFailure("WebGPU result does not use the canonical one-sample workload")
    input_hash = result.get("inputTensorSha256")
    if not isinstance(input_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", input_hash):
        raise BenchmarkFailure("WebGPU result lacks the normalized F32 input hash")

    compiles = result.get("compiles")
    if not isinstance(compiles, list) or len(compiles) != 2:
        raise BenchmarkFailure("WebGPU result must contain two strict compilations")
    encoder_compile = _strict_webgpu_compilation(compiles[0], "encoder")
    decoder_compile = _strict_webgpu_compilation(compiles[1], "decoder")
    adapter = _physical_webgpu_adapter(result.get("adapter"))
    if (
        encoder_compile.get("selectedDevice") != result.get("adapter")
        or decoder_compile.get("selectedDevice") != result.get("adapter")
    ):
        raise BenchmarkFailure("WebGPU encoder/decoder selected different adapters")

    runs = result.get("runs")
    warmups = result.get("warmupRuns")
    context_reuse = result.get("contextReuse")
    if (
        not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(warmups, list)
        or len(warmups) != execution_warmup
    ):
        raise BenchmarkFailure(
            "WebGPU child browser did not contain the requested same-session "
            "warmup and exactly one measured run"
        )
    if (
        not isinstance(context_reuse, Mapping)
        or context_reuse.get("sameRuntime") is not True
        or context_reuse.get("sameSession") is not True
        or context_reuse.get("encoderContextCount") != 1
        or context_reuse.get("decoderContextCount") != 1
    ):
        raise BenchmarkFailure(
            "WebGPU child browser did not prove one reused encoder/decoder context"
        )
    run = runs[0]
    if not isinstance(run, Mapping):
        raise BenchmarkFailure("WebGPU measured run is invalid")
    tokens = run.get("tokenIds")
    question_tokens = run.get("questionTokenIds")
    shape = run.get("shape")
    cache = run.get("cache")
    timing = run.get("timing")
    attestations = run.get("executionAttestations")
    if (
        run.get("name") != "measured-0"
        or run.get("family") != family
        or run.get("familyId") != FAMILY_NAMES.index(family)
        or run.get("requestedFamily") != family
        or not isinstance(tokens, list)
        or not isinstance(shape, Mapping)
        or not isinstance(question_tokens, list)
        or len(question_tokens) != shape.get("Q")
        or any(
            not _plain_int(value) or value < 0 or value >= 1536
            for value in question_tokens
        )
        or not question_tokens
        or question_tokens[-1] != 2
        or not isinstance(cache, Mapping)
        or not isinstance(timing, Mapping)
        or not isinstance(attestations, Mapping)
    ):
        raise BenchmarkFailure("WebGPU measured run lacks family/token/cache evidence")
    if (
        timing.get("synchronization") != "required-small-output-readback"
        or timing.get("applicationValidationIncluded") is not False
        or timing.get("cacheQualificationReadbackIncluded") is not False
    ):
        raise BenchmarkFailure(
            "WebGPU measured timing is not bounded by required small-output readback"
        )
    resident = _volvox_webgpu_kv_evidence(
        run.get("gpuResidentKv"),
        token_count=len(tokens),
        qualified=False,
        label="WebGPU measured run",
    )
    parity_signature = (
        run.get("family"), run.get("familyId"), run.get("requestedFamily"),
        question_tokens, tokens, shape, cache,
    )
    for index, warmup_run in enumerate(warmups):
        if not isinstance(warmup_run, Mapping):
            raise BenchmarkFailure(f"WebGPU warmup run {index} is invalid")
        warmup_signature = (
            warmup_run.get("family"), warmup_run.get("familyId"),
            warmup_run.get("requestedFamily"),
            warmup_run.get("questionTokenIds"), warmup_run.get("tokenIds"),
            warmup_run.get("shape"), warmup_run.get("cache"),
        )
        if warmup_signature != parity_signature:
            raise BenchmarkFailure(
                f"WebGPU warmup run {index} failed token/cache parity"
            )
        _volvox_webgpu_kv_evidence(
            warmup_run.get("gpuResidentKv"),
            token_count=len(warmup_run.get("tokenIds", [])),
            qualified=index == 0,
            label=f"WebGPU warmup run {index}",
        )
        warmup_attestations = warmup_run.get("executionAttestations")
        warmup_tokens = warmup_run.get("tokenIds")
        if not isinstance(warmup_attestations, Mapping):
            raise BenchmarkFailure(
                f"WebGPU warmup run {index} lacks execution attestations"
            )
        _strict_webgpu_execution(
            warmup_attestations.get("encoder"),
            f"WebGPU warmup run {index} encoder",
        )
        warmup_decoder = warmup_attestations.get("decoder")
        if (
            not isinstance(warmup_decoder, list)
            or not isinstance(warmup_tokens, list)
            or len(warmup_decoder) != len(warmup_tokens)
        ):
            raise BenchmarkFailure(
                f"WebGPU warmup run {index} lacks one decoder attestation per token"
            )
        for step, attestation in enumerate(warmup_decoder):
            _strict_webgpu_execution(
                attestation, f"WebGPU warmup run {index} decoder step {step}"
            )
    generation_ms = run.get("generationMs")
    if (
        isinstance(generation_ms, bool)
        or not isinstance(generation_ms, (int, float))
        or not math.isfinite(generation_ms)
        or generation_ms < 0
    ):
        raise BenchmarkFailure("WebGPU measured run has invalid end-to-end timing")
    qualification = result.get("cacheQualification")
    expected_qualification_source = (
        "warmup-0" if execution_warmup > 0 else "untimed-correctness-request"
    )
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("source") != expected_qualification_source
        or qualification.get("encoderMemoryReadbackValidated") is not True
        or qualification.get("encoderCrossCacheReadbackValidated") is not True
        or qualification.get("cachePrefixReadbackValidated") is not True
        or qualification.get("appendedCacheReadbackValidated") is not True
        or qualification.get("tokenCacheTransitionParity") is not True
    ):
        raise BenchmarkFailure(
            "WebGPU lacks untimed device-cache correctness qualification"
        )
    decoder_attestations = attestations.get("decoder")
    _strict_webgpu_execution(attestations.get("encoder"), "WebGPU encoder")
    if not isinstance(decoder_attestations, list) or len(decoder_attestations) != len(tokens):
        raise BenchmarkFailure("WebGPU result lacks one decoder attestation per token")
    for index, attestation in enumerate(decoder_attestations):
        _strict_webgpu_execution(attestation, f"WebGPU decoder step {index}")

    phases = run.get("executionPhases")
    if not isinstance(phases, Mapping):
        raise BenchmarkFailure("WebGPU result lacks execution-phase evidence")
    decoder_phases = phases.get("decoder")
    if not isinstance(decoder_phases, list) or len(decoder_phases) != len(tokens):
        raise BenchmarkFailure(
            "WebGPU result lacks one decoder execution phase per token"
        )
    encoder_specialization = _specialization_writes(
        phases.get("encoder"), "WebGPU encoder"
    )
    decoder_specialization = [
        _specialization_writes(value, f"WebGPU decoder step {index}")
        for index, value in enumerate(decoder_phases)
    ]
    if execution_warmup > 0 and (
        phases.get("encoder", {}).get("specializationCacheHit") is not True
        or any(value.get("specializationCacheHit") is not True for value in decoder_phases)
    ):
        raise BenchmarkFailure(
            "WebGPU measured request did not hit every warmed shape specialization"
        )
    for previous, current in zip(
        decoder_specialization, decoder_specialization[1:], strict=False
    ):
        if any(current[name] < previous[name] for name in (
            "writeCount", "writeBytes", "skipCount", "skipBytes"
        )):
            raise BenchmarkFailure(
                "WebGPU decoder specialization-write counters are not cumulative"
            )

    sample = {
        "schema": "volvoxai.tiny-receipt-explicit-kv-runtime-sample/v1",
        "engine": "volvoxai-webgpu",
        "backend": "webgpu",
        "precision": precision,
        "provider": "webgpu",
        "strictNoFallback": True,
        "family": run["family"],
        "familyId": run["familyId"],
        "requestedFamily": run["requestedFamily"],
        "inputTensorSha256": input_hash,
        "questionTokenIds": list(question_tokens),
        "tokenIds": tokens,
        "stoppedAtEos": run.get("stoppedAtEos") is True,
        "shape": dict(shape),
        "cache": dict(cache),
        "timing": {
            **dict(timing),
            "endToEndInferenceMs": generation_ms,
            "processWallMs": process_wall_ms,
        },
        "executionPhases": phases,
        "specializationWrites": {
            "encoder": encoder_specialization,
            "decoder": decoder_specialization,
        },
        "runtime": {
            "name": "browser-webgpu",
            "adapter": adapter,
            "executionWarmup": {
                "runs": execution_warmup,
                "sameSession": True,
                "sameRuntime": True,
                "sameEncoderContext": True,
                "sameDecoderContext": True,
                "stateResetToSentinel": True,
                "tokenCacheTransitionParity": True,
            },
            "packedDot4": result.get("packedDot4") is True,
            "freshBrowser": True,
            "runtimeInitMs": result.get("runtimeInitMs"),
            "sessionLoadMs": result.get("sessionLoadMs"),
            "preprocessMs": result.get("preprocessMs"),
            "gpuResidentKv": resident,
            "cacheQualification": dict(result.get("cacheQualification", {})),
        },
        "routeEvidence": {
            "compilation": compiles,
            "execution": attestations,
        },
        "tactics": run.get("tactics"),
    }
    validate_sample(sample, max_new=max_new)
    return sample


def _expected_ort_webgpu_cache_readbacks(
    token_count: int, *, qualifying: bool,
) -> dict[str, Any]:
    cross_count = 8 if qualifying else 0
    present_count = 8 * token_count if qualifying else 0
    return {
        "scope": "application-gpu-buffer-to-host-cache-validation",
        "crossCacheTensorReadbacks": cross_count,
        "presentCacheTensorReadbacks": present_count,
        "totalCacheTensorReadbacks": cross_count + present_count,
        "includedInExecutionTiming": False,
        "includedInRequestEndToEndInferenceMs": qualifying,
    }


def _validate_ort_webgpu_cache_readbacks(
    resident: Any,
    *,
    token_count: int,
    qualifying: bool,
    label: str,
) -> dict[str, Any]:
    expected = _expected_ort_webgpu_cache_readbacks(
        token_count, qualifying=qualifying,
    )
    evidence = (
        resident.get("harnessCacheReadbacks")
        if isinstance(resident, Mapping) else None
    )
    count_fields = (
        "crossCacheTensorReadbacks",
        "presentCacheTensorReadbacks",
        "totalCacheTensorReadbacks",
    )
    if (
        not isinstance(resident, Mapping)
        or resident.get("crossCacheOutputsOnGpuBuffer") != 8
        or resident.get("presentCacheOutputsOnGpuBufferPerStep") != 8
        or resident.get("encoderCrossCacheTensorObjectHandoffs")
        != token_count * 8
        or resident.get("decoderPresentCacheTensorObjectHandoffs")
        != max(0, token_count - 1) * 8
        or resident.get("totalDirectTensorObjectHandoffs")
        != token_count * 8 + max(0, token_count - 1) * 8
        or resident.get("preferredOutputLocation") != "gpu-buffer"
        or resident.get("crossCacheReadbackValidated") is not qualifying
        or resident.get("cachePrefixReadbackValidated") is not qualifying
        or resident.get("appendedCacheReadbackValidated") is not qualifying
        or resident.get("measuredHarnessCacheReadbacks") != 0
        or resident.get("internalEpTransfersAttested") is not False
        or not isinstance(evidence, Mapping)
        or set(evidence) != set(expected)
        or any(not _plain_int(evidence.get(field)) for field in count_fields)
        or evidence.get("scope") != expected["scope"]
        or any(evidence.get(field) != expected[field] for field in count_fields)
        or evidence.get("includedInExecutionTiming") is not False
        or evidence.get("includedInRequestEndToEndInferenceMs") is not qualifying
    ):
        raise BenchmarkFailure(
            f"ORT WebGPU {label} public gpu-buffer handoff/readback evidence is inconsistent"
        )
    return dict(evidence)


def parse_ort_webgpu_sample(
    stdout: bytes,
    *,
    precision: str,
    prompt: str,
    family: str,
    max_new: int,
    process_wall_ms: float,
    execution_warmup: int = 1,
    runtime_identity: Mapping[str, Any] | None = None,
    expected_model_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate one ORT WebGPU/CPU-partitioned fresh-browser sample."""

    label = f"ONNX Runtime WebGPU/{precision}"
    if precision not in ("fp32", "int8"):
        raise BenchmarkFailure("ORT WebGPU precision is invalid")
    if not math.isfinite(process_wall_ms) or process_wall_ms < 0:
        raise BenchmarkFailure("ORT WebGPU process wall time is invalid")
    result = _parse_webgpu_result(stdout, label)
    runtime = result.get("runtime")
    if (
        result.get("backend") != "webgpu-cpu-fallback"
        or result.get("strictNoFallback") is not False
        or not isinstance(runtime, Mapping)
        or runtime.get("name") != "onnxruntime-web"
        or runtime.get("version") != ORT_WEBGPU_VERSION
        or result.get("precision") != precision
        or result.get("prompt") != prompt
        or result.get("family") != family
        or result.get("maxNewTokens") != max_new
        or result.get("warmup") != execution_warmup
        or result.get("iterations") != 1
    ):
        raise BenchmarkFailure(
            "ORT WebGPU result does not use the pinned canonical one-sample workload"
        )
    input_hash = result.get("inputTensorSha256")
    if not isinstance(input_hash, str) or re.fullmatch(r"[0-9a-f]{64}", input_hash) is None:
        raise BenchmarkFailure("ORT WebGPU result lacks the normalized F32 input hash")
    pinned_models = ORT_WEBGPU_PINNED_MODELS[precision]
    expected_model_hashes = {
        role: pinned_models[role]["sha256"] for role in ("encoder", "decoder")
    } if expected_model_sha256 is None else dict(expected_model_sha256)
    if (
        set(expected_model_hashes) != {"encoder", "decoder"}
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in expected_model_hashes.values()
        )
    ):
        raise BenchmarkFailure("ORT WebGPU expected model identities are invalid")
    for role in ("encoder", "decoder"):
        if expected_model_hashes[role] != pinned_models[role]["sha256"]:
            raise BenchmarkFailure(
                f"ORT WebGPU {role} producer model SHA has no pinned partition evidence"
            )
    reported_model_hashes = result.get("modelSha256")
    if (
        not isinstance(reported_model_hashes, Mapping)
        or dict(reported_model_hashes) != expected_model_hashes
    ):
        raise BenchmarkFailure(
            "ORT WebGPU measured/strict session bytes differ from producer model SHA"
        )
    adapter = _physical_webgpu_adapter(result.get("adapter"))
    preflight = _physical_webgpu_adapter(result.get("adapterPreflight"))
    if (
        adapter != preflight
        or result.get("actualAdapterObjectAttested") is not True
    ):
        raise BenchmarkFailure(
            "ORT WebGPU result did not attest the preflight adapter as its actual adapter"
        )

    placement = result.get("placement")
    strict_probe = result.get("strictProbe")
    for role in ("encoder", "decoder"):
        placed = placement.get(role) if isinstance(placement, Mapping) else None
        providers = placed.get("providers") if isinstance(placed, Mapping) else None
        records = placed.get("records") if isinstance(placed, Mapping) else None
        unsupported = (
            placed.get("unsupportedKernelEvents")
            if isinstance(placed, Mapping) else None
        )
        expected = pinned_models[role]["providers"]
        record_map = {
            record.get("provider"): record.get("nodes")
            for record in records
        } if isinstance(records, list) and all(
            isinstance(record, Mapping) for record in records
        ) else {}
        if (
            providers != expected
            or not isinstance(records, list)
            or len(records) != 2
            or record_map != expected
            or any(
                not _plain_int(record.get("nodes")) or record["nodes"] <= 0
                for record in records
            )
            or not isinstance(unsupported, Mapping)
        ):
            raise BenchmarkFailure(
                f"ORT WebGPU {role} placement differs from the pinned graph partition"
            )
        probe = strict_probe.get(role) if isinstance(strict_probe, Mapping) else None
        error = probe.get("error") if isinstance(probe, Mapping) else None
        if (
            probe.get("rejected") is not True
            if isinstance(probe, Mapping) else True
        ) or not isinstance(error, str) or any(
            clause not in error for clause in ORT_WEBGPU_STRICT_ERROR_CLAUSES
        ):
            raise BenchmarkFailure(
                f"ORT WebGPU {role} lacks the exact CPU-fallback-disabled strict probe"
            )

    session_reuse = result.get("sessionReuse")
    warmups = result.get("warmupRuns")
    runs = result.get("runs")
    if (
        not isinstance(session_reuse, Mapping)
        or session_reuse.get("sameRuntime") is not True
        or session_reuse.get("sameSessions") is not True
        or session_reuse.get("encoderSessionCount") != 1
        or session_reuse.get("decoderSessionCount") != 1
        or not isinstance(warmups, list)
        or len(warmups) != execution_warmup
        or not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(runs[0], Mapping)
    ):
        raise BenchmarkFailure(
            "ORT WebGPU did not prove one fresh browser with reused sessions"
        )
    run = runs[0]
    tokens = run.get("tokenIds")
    question_tokens = run.get("questionTokenIds")
    shape = run.get("shape")
    cache = run.get("cache")
    timing = run.get("timing")
    resident = run.get("publicKvBufferHandoff")
    if (
        run.get("measured") is not True
        or run.get("family") != family
        or run.get("familyId") != FAMILY_NAMES.index(family)
        or run.get("requestedFamily") != family
        or not isinstance(tokens, list)
        or not isinstance(question_tokens, list)
        or not isinstance(shape, Mapping)
        or len(question_tokens) != shape.get("Q")
        or not question_tokens
        or question_tokens[-1] != 2
        or any(
            not _plain_int(value) or value < 0 or value >= 1536
            for value in question_tokens
        )
        or not isinstance(cache, Mapping)
        or not isinstance(timing, Mapping)
        or timing.get("synchronization")
        != "session-run-required-cpu-output-materialization"
        or timing.get("applicationValidationIncluded") is not False
        or timing.get("cacheQualificationReadbackIncluded") is not False
    ):
        raise BenchmarkFailure(
            "ORT WebGPU measured run lacks public gpu-buffer explicit-KV handoff evidence"
        )
    _validate_ort_webgpu_cache_readbacks(
        resident,
        token_count=len(tokens),
        qualifying=False,
        label="measured run",
    )
    signature = (
        run.get("family"), run.get("familyId"), run.get("requestedFamily"),
        question_tokens, tokens, shape, cache,
    )
    for index, warmup_run in enumerate(warmups):
        if (
            not isinstance(warmup_run, Mapping)
            or warmup_run.get("measured") is not False
            or (
            warmup_run.get("family"), warmup_run.get("familyId"),
            warmup_run.get("requestedFamily"),
            warmup_run.get("questionTokenIds"), warmup_run.get("tokenIds"),
            warmup_run.get("shape"), warmup_run.get("cache"),
            ) != signature
        ):
            raise BenchmarkFailure(
                f"ORT WebGPU warmup run {index} failed token/cache transition parity"
            )
        _validate_ort_webgpu_cache_readbacks(
            warmup_run.get("publicKvBufferHandoff"),
            token_count=len(tokens),
            qualifying=index == 0,
            label=f"warmup run {index}",
        )
    qualification = result.get("cacheQualification")
    expected_qualification_source = (
        "warmup-0" if execution_warmup > 0 else "untimed-correctness-request"
    )
    qualification_readbacks = (
        qualification.get("harnessCacheReadbacks")
        if isinstance(qualification, Mapping) else None
    )
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("source") != expected_qualification_source
        or qualification.get("crossCacheReadbackValidated") is not True
        or qualification.get("cachePrefixReadbackValidated") is not True
        or qualification.get("appendedCacheReadbackValidated") is not True
        or not isinstance(qualification_readbacks, Mapping)
        or any(
            not _plain_int(qualification_readbacks.get(field))
            for field in (
                "crossCacheTensorReadbacks",
                "presentCacheTensorReadbacks",
                "totalCacheTensorReadbacks",
            )
        )
        or dict(qualification_readbacks)
        != _expected_ort_webgpu_cache_readbacks(
            len(tokens), qualifying=True,
        )
    ):
        raise BenchmarkFailure(
            "ORT WebGPU lacks untimed cache-prefix/readback correctness evidence"
        )
    if qualification.get("tokenCacheTransitionParity") is not True:
        raise BenchmarkFailure(
            "ORT WebGPU cache qualification does not match the measured token/cache transition"
        )
    end_to_end_ms = run.get("endToEndInferenceMs")
    if (
        isinstance(end_to_end_ms, bool)
        or not isinstance(end_to_end_ms, (int, float))
        or not math.isfinite(end_to_end_ms)
        or end_to_end_ms < 0
    ):
        raise BenchmarkFailure("ORT WebGPU measured run has invalid end-to-end timing")

    sample = {
        "schema": "volvoxai.tiny-receipt-explicit-kv-runtime-sample/v1",
        "engine": "onnxruntime-web",
        "backend": "webgpu-cpu-fallback",
        "precision": precision,
        "provider": "WebGpuExecutionProvider",
        "strictNoFallback": False,
        "family": run["family"],
        "familyId": run["familyId"],
        "requestedFamily": run["requestedFamily"],
        "inputTensorSha256": input_hash,
        "questionTokenIds": list(question_tokens),
        "tokenIds": list(tokens),
        "stoppedAtEos": run.get("stoppedAtEos") is True,
        "shape": dict(shape),
        "cache": dict(cache),
        "timing": {
            **dict(timing),
            "endToEndInferenceMs": float(end_to_end_ms),
            "processWallMs": process_wall_ms,
        },
        "runtime": {
            "name": "onnxruntime-web",
            "version": ORT_WEBGPU_VERSION,
            "adapter": adapter,
            "freshBrowser": True,
            "providerMode": "webgpu-with-cpu-fallback",
            "providerOrder": ["WebGpuExecutionProvider", "CPUExecutionProvider"],
            "cpuEpFallbackAllowed": True,
            "cpuEpFallbackUsage": "optimized-session-partition-count-attested",
            "cpuEpFallbackDisabled": False,
            "placement": {role: dict(placement[role]) for role in ("encoder", "decoder")},
            "placementEvidence": {
                "source": "session-creation-optimized-partition-diagnostics",
                "scope": "partition-counts-not-executed-node-placement",
                "modelSha256": expected_model_hashes,
                "measuredAndStrictSessionsUseSameFetchedBytes": True,
            },
            "strictProbe": {role: dict(strict_probe[role]) for role in ("encoder", "decoder")},
            "sessionLoadMs": result.get("sessionLoadMs"),
            "preprocessMs": result.get("preprocessMs"),
            "publicKvBufferHandoff": dict(resident),
            "cacheQualification": dict(qualification),
            "executionWarmup": {
                "runs": execution_warmup,
                "sameSessions": True,
                "stateResetToSentinel": True,
                "tokenCacheTransitionParity": True,
            },
            **({"distribution": dict(runtime_identity)}
               if runtime_identity is not None else {}),
        },
    }
    validate_sample(
        sample, max_new=max_new, expected_strict_no_fallback=False,
    )
    return sample


def _repository_relative(path: Path, repository: Path, label: str) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(repository.resolve())
    except ValueError as error:
        raise BenchmarkFailure(
            f"{label} must be inside the repository so the local browser server can serve it"
        ) from error
    return relative.as_posix()


def run_webgpu_sample(
    *,
    node: str,
    chrome: str,
    runner: Path,
    repository: Path,
    package: Path,
    precision: str,
    image: Path,
    prompt: str,
    family: str,
    max_new: int,
    timeout: float,
    environment: Mapping[str, str],
    dynamic_qualification: bool = False,
    execution_warmup: int = 1,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tinyreceipt-webgpu-supervisor-") as value:
        chrome_pid_file = Path(value) / "chrome.pid"
        command = [
            node,
            "--experimental-websocket",
            str(runner),
            f"--package={_repository_relative(package, repository, 'WebGPU package')}",
            f"--image={_repository_relative(image, repository, 'WebGPU image')}",
            f"--prompt={prompt}",
            f"--family={family}",
            f"--tokens={max_new}",
            f"--warmup={0 if dynamic_qualification else execution_warmup}",
            "--iterations=1",
            "--adapter=hardware",
            f"--timeout={int(timeout * 1000)}",
            f"--chrome-pid-file={chrome_pid_file}",
        ]
        if dynamic_qualification:
            command.append("--qualification=dynamic-rebind")
        child_environment = dict(environment)
        child_environment["CHROME"] = chrome
        started = time.perf_counter_ns()
        child = subprocess.Popen(
            command,
            cwd=repository,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = child.communicate(timeout=timeout + 30.0)
        except subprocess.TimeoutExpired as error:
            # Signal Node first so its handler can kill Chrome's detached
            # process group. The pid file is an independent cleanup path if
            # Node itself is stuck before its handler completes.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                stdout, stderr = child.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate()
            try:
                chrome_pid = int(chrome_pid_file.read_text(encoding="ascii").strip())
                if chrome_pid > 1:
                    os.killpg(chrome_pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
            raise BenchmarkFailure(
                f"WebGPU/{precision} exceeded {timeout + 30.0:.1f}s supervisor deadline"
            ) from error
    process_wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if child.returncode != 0:
        action = "dynamic qualification" if dynamic_qualification else "sample"
        raise BenchmarkFailure(
            f"WebGPU/{precision} {action} failed with exit {child.returncode}: "
            f"{stderr.decode('utf-8', errors='replace').strip()}\n"
            f"{stdout.decode('utf-8', errors='replace')[-2000:]}"
        )
    if dynamic_qualification:
        return parse_webgpu_dynamic_qualification(
            stdout,
            precision=precision,
            prompt=prompt,
            family=family,
            max_new=max_new,
        )
    return parse_webgpu_sample(
        stdout,
        precision=precision,
        prompt=prompt,
        family=family,
        max_new=max_new,
        process_wall_ms=process_wall_ms,
        execution_warmup=execution_warmup,
    )


def run_ort_webgpu_sample(
    *,
    node: str,
    chrome: str,
    runner: Path,
    repository: Path,
    source: Path,
    ort_web_root: Path,
    runtime_identity: Mapping[str, Any],
    precision: str,
    image: Path,
    prompt: str,
    family: str,
    max_new: int,
    timeout: float,
    environment: Mapping[str, str],
    expected_model_sha256: Mapping[str, str],
    execution_warmup: int = 1,
) -> dict[str, Any]:
    """Run one pinned ORT WebGPU tier in an isolated browser process."""

    with tempfile.TemporaryDirectory(prefix="tinyreceipt-ort-webgpu-supervisor-") as value:
        chrome_pid_file = Path(value) / "chrome.pid"
        command = [
            node,
            "--experimental-websocket",
            str(runner),
            f"--source={source}",
            f"--ort-web-root={ort_web_root}",
            f"--image={_repository_relative(image, repository, 'ORT WebGPU image')}",
            f"--precision={precision}",
            f"--prompt={prompt}",
            f"--family={family}",
            f"--tokens={max_new}",
            f"--warmup={execution_warmup}",
            f"--timeout={int(timeout * 1000)}",
            f"--chrome-pid-file={chrome_pid_file}",
        ]
        child_environment = dict(environment)
        child_environment["CHROME"] = chrome
        started = time.perf_counter_ns()
        child = subprocess.Popen(
            command,
            cwd=repository,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = child.communicate(timeout=timeout + 30.0)
        except subprocess.TimeoutExpired as error:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                stdout, stderr = child.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate()
            try:
                chrome_pid = int(chrome_pid_file.read_text(encoding="ascii").strip())
                if chrome_pid > 1:
                    os.killpg(chrome_pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
            raise BenchmarkFailure(
                f"ORT WebGPU/{precision} exceeded {timeout + 30.0:.1f}s supervisor deadline"
            ) from error
    process_wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if child.returncode != 0:
        raise BenchmarkFailure(
            f"ORT WebGPU/{precision} failed with exit {child.returncode}: "
            f"{stderr.decode('utf-8', errors='replace').strip()}\n"
            f"{stdout.decode('utf-8', errors='replace')[-2000:]}"
        )
    return parse_ort_webgpu_sample(
        stdout,
        precision=precision,
        prompt=prompt,
        family=family,
        max_new=max_new,
        process_wall_ms=process_wall_ms,
        execution_warmup=execution_warmup,
        runtime_identity=runtime_identity,
        expected_model_sha256=expected_model_sha256,
    )


def _provider_options(provider: str, entries: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise BenchmarkFailure("--ort-provider-option must be NAME=VALUE")
        name, value = entry.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or not value:
            raise BenchmarkFailure("--ort-provider-option must be a non-empty NAME=VALUE")
        if name in result:
            raise BenchmarkFailure(f"duplicate ONNX Runtime provider option {name!r}")
        result[name] = value
    if provider == "CUDAExecutionProvider":
        result.setdefault("device_id", "0")
        result.setdefault("use_tf32", "0")
    return result


def _run_ort_worker(
    *,
    script: Path,
    source: Path,
    precision: str,
    image: Path,
    prompt: str,
    family: str,
    max_new: int,
    threads: int,
    provider: str,
    provider_options: Mapping[str, str],
    allow_cpu_fallback: bool,
    execution_warmup: int,
    timeout: float,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "__ort_sample__",
        "--source",
        str(source),
        "--precision",
        precision,
        "--image",
        str(image),
        "--prompt",
        prompt,
        "--family",
        family,
        "--max-new",
        str(max_new),
        "--threads",
        str(threads),
        "--provider",
        provider,
        "--execution-warmup",
        str(execution_warmup),
    ]
    if allow_cpu_fallback:
        command.append("--allow-cpu-fallback")
    for name, value in sorted(provider_options.items()):
        command.extend(["--provider-option", f"{name}={value}"])
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=script.parents[3],
        env=dict(environment),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    process_wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if completed.returncode != 0:
        raise BenchmarkFailure(
            f"ONNX Runtime {provider}/{precision} failed with exit {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    sample = _parse_json_sample(
        completed.stdout, f"ONNX Runtime {provider}/{precision}"
    )
    sample["timing"]["processWallMs"] = process_wall_ms
    sample["runtime"]["freshProcess"] = True
    return sample


def _tier_key(sample: Mapping[str, Any]) -> str:
    return f"{sample['engine']}/{sample['backend']}/{sample['precision']}"


def _parity_signature(sample: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        sample.get("family"),
        sample.get("familyId"),
        sample.get("requestedFamily"),
        sample.get("inputTensorSha256"),
        sample.get("questionTokenIds"),
        sample.get("tokenIds"),
        sample.get("shape"),
        sample.get("cache"),
    )


def _stable_webgpu_adapter_class(
    samples: Sequence[Mapping[str, Any]], label: str,
) -> dict[str, str]:
    identities = {
        tuple(sorted(_canonical_webgpu_adapter_class(sample["runtime"]["adapter"]).items()))
        for sample in samples
    }
    if len(identities) != 1:
        raise BenchmarkFailure(f"{label} does not report one stable WebGPU adapter class")
    return dict(next(iter(identities)))


def _build_gpu_comparisons(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build only like-for-like GPU comparisons; never substitute an ORT peer."""

    precisions = sorted({str(sample.get("precision")) for sample in samples})
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for precision in precisions:
        current = [sample for sample in samples if sample.get("precision") == precision]
        ort_web = [sample for sample in current if sample.get("engine") == "onnxruntime-web"]
        volvox_web = [
            sample for sample in current
            if sample.get("engine") == "volvoxai-webgpu"
            and sample.get("backend") == "webgpu"
        ]
        if ort_web and volvox_web:
            reference_class = _stable_webgpu_adapter_class(
                ort_web, f"ORT WebGPU/{precision}",
            )
            candidate_class = _stable_webgpu_adapter_class(
                volvox_web, f"VolvoxAI WebGPU/{precision}",
            )
            if reference_class != candidate_class:
                raise BenchmarkFailure(
                    f"invalid WebGPU pair for {precision}: adapter classes differ"
                )
            webgpu = {
                "status": "direct",
                "referenceTier": _tier_key(ort_web[0]),
                "candidateTier": _tier_key(volvox_web[0]),
                "backendClass": "webgpu",
                "adapterClassIdentity": reference_class,
                "adapterClassIdentityMatch": True,
                "samePhysicalAdapterAttested": False,
                "identityScope": (
                    "normalized WebGPU vendor+architecture class across isolated "
                    "browser processes; no stable physical adapter identifier is exposed"
                ),
            }
        else:
            webgpu = {
                "status": "not-measured",
                "referenceTier": _tier_key(ort_web[0]) if ort_web else "N/A",
                "candidateTier": _tier_key(volvox_web[0]) if volvox_web else "N/A",
                "backendClass": "webgpu",
                "reason": (
                    "direct comparison requires both ONNX Runtime Web WebGPU and "
                    "VolvoxAI WebGPU samples"
                ),
                "samePhysicalAdapterAttested": False,
            }

        ort_cuda = [
            sample for sample in current
            if sample.get("engine") == "onnxruntime"
            and sample.get("provider") == "CUDAExecutionProvider"
            and str(sample.get("backend", "")).startswith("cuda")
        ]
        native_cuda = [
            sample for sample in current
            if sample.get("engine") == "native-c"
            and sample.get("backend") == "cuda"
        ]
        if ort_cuda and native_cuda:
            ort_ordinals: set[int] = set()
            for sample in ort_cuda:
                runtime = sample.get("runtime")
                options = runtime.get("providerOptions") if isinstance(runtime, Mapping) else None
                raw_ordinal = options.get("device_id") if isinstance(options, Mapping) else None
                try:
                    ordinal = int(raw_ordinal)
                except (TypeError, ValueError) as error:
                    raise BenchmarkFailure(
                        f"invalid CUDA pair for {precision}: ORT lacks a device ordinal"
                    ) from error
                if str(ordinal) != str(raw_ordinal):
                    raise BenchmarkFailure(
                        f"invalid CUDA pair for {precision}: ORT device ordinal is ambiguous"
                    )
                ort_ordinals.add(ordinal)
            native_ordinals = {
                int(_physical_native_gpu_device(sample)["index"])
                for sample in native_cuda
            }
            if len(ort_ordinals) != 1 or ort_ordinals != native_ordinals:
                raise BenchmarkFailure(
                    f"invalid CUDA pair for {precision}: selected ordinals differ"
                )
            cuda = {
                "status": "direct",
                "referenceTier": _tier_key(ort_cuda[0]),
                "candidateTier": _tier_key(native_cuda[0]),
                "backendClass": "cuda",
                "visibleDeviceOrdinal": next(iter(ort_ordinals)),
                "deviceSelectionIdentityMatch": True,
                "samePhysicalDeviceAttested": False,
                "physicalDeviceUuidAttested": False,
                "identityScope": (
                    "matching CUDA ordinal under the inherited process environment; "
                    "neither sample reports a comparable physical-device UUID"
                ),
            }
        else:
            cuda = {
                "status": "not-measured",
                "referenceTier": _tier_key(ort_cuda[0]) if ort_cuda else "N/A",
                "candidateTier": _tier_key(native_cuda[0]) if native_cuda else "N/A",
                "backendClass": "cuda",
                "reason": (
                    "direct comparison requires both ONNX Runtime CUDA and native "
                    "VolvoxAI CUDA samples"
                ),
                "samePhysicalDeviceAttested": False,
                "physicalDeviceUuidAttested": False,
            }

        result[precision] = {
            "webgpu": webgpu,
            "cuda": cuda,
            "vulkan": {
                "status": "not-available",
                "referenceTier": "N/A",
                "candidateTier": (
                    f"native-c/vulkan/{precision}"
                    if any(sample.get("backend") == "vulkan" for sample in current)
                    else "N/A"
                ),
                "backendClass": "vulkan",
                "reason": "this harness has no ONNX Runtime Vulkan execution provider peer",
            },
            "opengl": {
                "status": "not-available",
                "referenceTier": "N/A",
                "candidateTier": (
                    f"native-c/opengl/{precision}"
                    if any(sample.get("backend") == "opengl" for sample in current)
                    else "N/A"
                ),
                "backendClass": "opengl",
                "reason": "this harness has no ONNX Runtime OpenGL execution provider peer",
            },
        }
    return result


def _measurement_boundaries(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    present = {str(sample.get("engine")) for sample in samples}
    definitions = {
        "onnxruntime": {
            "timer": "host session.run(None, feeds) call",
            "synchronization": "all requested public outputs materialized as host NumPy arrays",
            "kvHandoff": "host output arrays are supplied as the next decoder feeds",
            "applicationValidationIncluded": False,
        },
        "native-c": {
            "timer": "public vx_execution_context_execute call/report execution time",
            "synchronization": "execute completion for the selected strict backend",
            "kvHandoff": (
                "application vx_result_read and host KV validation/copy occur after the "
                "reported execute interval"
            ),
            "applicationValidationIncluded": False,
        },
        "volvoxai-webgpu": {
            "timer": (
                "host wall time from context execute through the required "
                "small-output GPU readback"
            ),
            "synchronization": (
                "the small control-output readback waits for prior queued GPU work; "
                "execution-phase provider time records host enqueue, not GPU completion"
            ),
            "kvHandoff": (
                "public device tensors are handed directly to the next context execution; "
                "cache correctness readback is untimed"
            ),
            "applicationValidationIncluded": False,
            "internalProviderTransfersAttested": True,
        },
        "onnxruntime-web": {
            "timer": "await InferenceSession.run(feeds)",
            "synchronization": "required CPU public outputs materialized by session.run",
            "kvHandoff": (
                "public cross/present KV gpu-buffer Tensor objects are reused directly; "
                "the measured harness performs zero KV readbacks"
            ),
            "applicationValidationIncluded": False,
            "internalExecutionProviderTransfersAttested": False,
        },
    }
    return {engine: definitions[engine] for engine in sorted(present) if engine in definitions}


def _validate_ort_accelerator_evidence(
    runtime: Mapping[str, Any], provider: str,
) -> None:
    placement = runtime.get("providerPlacement")
    placement_evidence = runtime.get("providerPlacementEvidence")
    strict = runtime.get("strictProbe")
    if (
        not isinstance(placement, Mapping)
        or not isinstance(placement_evidence, Mapping)
        or placement_evidence.get("source")
        != "separate-untimed-profiling-sessions"
        or placement_evidence.get("executionOrder") != "after-measured-request"
        or placement_evidence.get("sameMeasuredSessions") is not False
        or placement_evidence.get("measuredSessionsProfiled") is not False
        or not isinstance(placement_evidence.get("sessionConfigurationInvariant"), str)
        or not isinstance(placement_evidence.get("requests"), Mapping)
        or not isinstance(strict, Mapping)
        or strict.get("source") != "separate-untimed-session-create-probes"
        or strict.get("executionOrder") != "after-measured-request"
        or not isinstance(strict.get("roles"), Mapping)
    ):
        raise BenchmarkFailure(
            "ONNX Runtime accelerator sample lacks isolated placement/strict-probe evidence"
        )

    allowed = {provider, "CPUExecutionProvider"}
    for role in ("encoder", "decoder"):
        role_placement = placement.get(role)
        providers = (
            role_placement.get("providers")
            if isinstance(role_placement, Mapping) else None
        )
        operators = (
            role_placement.get("operatorCountsByProvider")
            if isinstance(role_placement, Mapping) else None
        )
        if (
            not isinstance(providers, Mapping)
            or set(providers) - allowed
            or not _plain_int(providers.get(provider))
            or providers[provider] <= 0
            or any(not _plain_int(value) or value <= 0 for value in providers.values())
            or not _plain_int(role_placement.get("executedNodeCount"))
            or role_placement["executedNodeCount"] != sum(providers.values())
            or not isinstance(operators, Mapping)
            or set(operators) != set(providers)
            or not isinstance(role_placement.get("nodeAssignmentSha256"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}", role_placement["nodeAssignmentSha256"]
            ) is None
        ):
            raise BenchmarkFailure(
                f"ONNX Runtime {role} lacks exact executed-node placement"
            )
        for placed_provider, counts in operators.items():
            if (
                not isinstance(counts, Mapping)
                or not counts
                or any(
                    not isinstance(name, str)
                    or not name
                    or not _plain_int(count)
                    or count <= 0
                    for name, count in counts.items()
                )
                or sum(counts.values()) != providers[placed_provider]
            ):
                raise BenchmarkFailure(
                    f"ONNX Runtime {role} operator placement counts are inconsistent"
                )

        role_probe = strict["roles"].get(role)
        outcome = role_probe.get("outcome") if isinstance(role_probe, Mapping) else None
        cpu_nodes = providers.get("CPUExecutionProvider", 0)
        if (
            not isinstance(role_probe, Mapping)
            or role_probe.get("stage") != "session-create"
            or role_probe.get("cpuEpFallbackDisabled") is not True
            or outcome not in ("accepted", "rejected")
            or (cpu_nodes > 0) != (outcome == "rejected")
        ):
            raise BenchmarkFailure(
                f"ONNX Runtime {role} strict probe contradicts executed placement"
            )
        error = role_probe.get("error")
        if outcome == "rejected":
            if (
                not isinstance(role_probe.get("errorType"), str)
                or not isinstance(error, str)
                or any(clause not in error for clause in ORT_WEBGPU_STRICT_ERROR_CLAUSES)
            ):
                raise BenchmarkFailure(
                    f"ONNX Runtime {role} strict probe lacks exact CPU-partition rejection"
                )
        elif (
            role_probe.get("errorType") is not None
            or error is not None
            or not isinstance(role_probe.get("registeredProviders"), list)
            or not role_probe["registeredProviders"]
            or role_probe["registeredProviders"][0] != provider
        ):
            raise BenchmarkFailure(
                f"ONNX Runtime {role} strict probe lacks exact acceptance evidence"
            )


def prove_gpu_parity(
    samples: Sequence[Mapping[str, Any]],
    *,
    expected_tiers: Sequence[str],
    repeat: int,
    max_new: int,
    allow_ort_cpu_fallback: bool = False,
    expected_execution_warmup: int | None = None,
    expected_question_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(expected_question_token_ids, Sequence)
        or isinstance(expected_question_token_ids, (str, bytes))
        or not expected_question_token_ids
        or expected_question_token_ids[-1] != 2
        or any(
            not _plain_int(value) or value < 0 or value >= 1536
            for value in expected_question_token_ids
        )
    ):
        raise BenchmarkFailure("canonical tokenizer question token IDs are required")
    canonical_question_tokens = list(expected_question_token_ids)
    counts = Counter(_tier_key(sample) for sample in samples)
    expected = set(expected_tiers)
    if set(counts) != expected or any(counts[tier] != repeat for tier in expected):
        raise BenchmarkFailure(
            f"GPU benchmark matrix is incomplete: expected {sorted(expected)}, got {dict(counts)}"
        )
    for sample in samples:
        is_python_ort = sample.get("engine") == "onnxruntime"
        is_ort_webgpu = sample.get("engine") == "onnxruntime-web"
        is_ort = is_python_ort or is_ort_webgpu
        expected_strict = not (
            (allow_ort_cpu_fallback and is_python_ort) or is_ort_webgpu
        )
        validate_sample(
            sample,
            max_new=max_new,
            expected_strict_no_fallback=expected_strict,
        )
        runtime = sample.get("runtime")
        execution_warmup_evidence = (
            runtime.get("executionWarmup") if isinstance(runtime, Mapping) else None
        )
        same_context_key = (
            "sameSessions" if is_ort else "sameSession"
            if sample.get("engine") == "volvoxai-webgpu" else "sameContexts"
        )
        if (
            not isinstance(execution_warmup_evidence, Mapping)
            or not _plain_int(execution_warmup_evidence.get("runs"))
            or execution_warmup_evidence["runs"] < 0
            or (
                expected_execution_warmup is not None
                and execution_warmup_evidence["runs"] != expected_execution_warmup
            )
            or execution_warmup_evidence.get(same_context_key) is not True
            or execution_warmup_evidence.get("stateResetToSentinel") is not True
            or execution_warmup_evidence.get("tokenCacheTransitionParity") is not True
        ):
            raise BenchmarkFailure(
                f"{_tier_key(sample)} lacks same-context execution-warmup evidence"
            )
        if sample.get("engine") == "native-c":
            _physical_native_gpu_device(sample)
            if (
                not isinstance(runtime, Mapping)
                or runtime.get("cpuThreads") != 1
            ):
                raise BenchmarkFailure(
                    f"{_tier_key(sample)} lacks the measured native thread limit"
                )
        if allow_ort_cpu_fallback and is_python_ort:
            provider = sample.get("provider")
            expected_order = [provider, "CPUExecutionProvider"]
            if (
                not isinstance(provider, str)
                or provider == "CPUExecutionProvider"
                or not isinstance(runtime, Mapping)
                or sample.get("backend") != _ort_backend_label(provider, True)
                or runtime.get("providerMode")
                != "accelerator-with-cpu-fallback"
                or runtime.get("providerOrder") != expected_order
                or runtime.get("registeredEncoderProviders") != expected_order
                or runtime.get("registeredDecoderProviders") != expected_order
                or runtime.get("cpuEpFallbackAllowed") is not True
                or runtime.get("cpuEpFallbackUsage")
                != "exact-executed-node-placement-attested"
                or runtime.get("cpuEpFallbackDisabled") is not False
            ):
                raise BenchmarkFailure(
                    "ONNX Runtime CPU fallback sample lacks an honest accelerator-first attestation"
                )
            _validate_ort_accelerator_evidence(runtime, provider)
        if is_ort_webgpu:
            if (
                not isinstance(runtime, Mapping)
                or sample.get("backend") != "webgpu-cpu-fallback"
                or sample.get("provider") != "WebGpuExecutionProvider"
                or runtime.get("providerMode") != "webgpu-with-cpu-fallback"
                or runtime.get("providerOrder")
                != ["WebGpuExecutionProvider", "CPUExecutionProvider"]
                or runtime.get("cpuEpFallbackAllowed") is not True
                or runtime.get("cpuEpFallbackUsage")
                != "optimized-session-partition-count-attested"
                or runtime.get("cpuEpFallbackDisabled") is not False
            ):
                raise BenchmarkFailure(
                    "ORT WebGPU sample lacks honest CPU-partition placement evidence"
                )
    reference = samples[0]
    for sample in samples:
        if sample.get("questionTokenIds") != canonical_question_tokens:
            raise BenchmarkFailure(
                f"canonical question-token parity failed for {_tier_key(sample)}"
            )
    signature = _parity_signature(reference)
    for sample in samples[1:]:
        candidate = _parity_signature(sample)
        if candidate != signature:
            raise BenchmarkFailure(f"token/hash/cache parity failed for {_tier_key(sample)}")
    comparisons = _build_gpu_comparisons(samples)
    return {
        "exact": True,
        "exactScope": [
            "normalized-f32-input-bytes",
            "canonical-tokenizer-question-token-ids-consumed-by-every-runtime",
            "active-shape-binding",
            "selected-family",
            "greedy-token-ids",
            "explicit-kv-shape-transitions",
        ],
        "accuracyEvidence": (
            "single-workload exact greedy-output agreement; this is not a "
            "corpus-level accuracy metric"
        ),
        "numericTensorParity": (
            "not compared across runtimes; each runtime independently checks "
            "finite logits/caches and exact preservation of its own cache prefix"
        ),
        "executionWarmupRunsPerSample": expected_execution_warmup,
        "family": reference["family"],
        "familyId": reference["familyId"],
        "inputTensorSha256": reference["inputTensorSha256"],
        "questionTokenIds": canonical_question_tokens,
        "questionTokenEvidence": {
            "source": "parent validate_source tokenizer documents",
            "allMeasuredRuntimesAttestedConsumedIds": True,
        },
        "tokenIds": reference["tokenIds"],
        "shape": reference["shape"],
        "cache": reference["cache"],
        "matrixEntries": len(expected),
        "measuredSamples": len(samples),
        "comparisons": comparisons,
    }


def summarize_gpu_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(_tier_key(sample), []).append(sample)
    result: dict[str, Any] = {}
    for tier, values in sorted(grouped.items()):
        timings = [value["timing"] for value in values]
        summary = {
            "engine": values[0]["engine"],
            "backend": values[0]["backend"],
            "provider": values[0]["provider"],
            "precision": values[0]["precision"],
            "strictNoFallback": values[0]["strictNoFallback"],
            "samples": len(values),
            "encoder": _distribution(
                [float(value["encoderExecutionMs"]) for value in timings]
            ),
            "decoderSeed": _distribution(
                [float(value["decoderSeedMs"]) for value in timings]
            ),
            "decoderSteadyMean": _distribution(
                [float(value["decoderSteadyMeanMs"]) for value in timings]
            ),
            "decoderExecutionTotal": _distribution(
                [float(value["decoderExecutionTotalMs"]) for value in timings]
            ),
            "componentTotal": _distribution([
                float(value["encoderExecutionMs"])
                + float(value["decoderExecutionTotalMs"])
                for value in timings
            ]),
            "processWall": _distribution(
                [float(value["processWallMs"]) for value in timings]
            ),
        }
        result[tier] = summary
    return result


def _ort_backend_label(provider: str, allow_cpu_fallback: bool) -> str:
    backend = (
        "cpu" if provider == "CPUExecutionProvider"
        else provider.removesuffix("ExecutionProvider").lower()
    )
    return backend + ("-cpu-fallback" if allow_cpu_fallback else "")


def _expected_tiers(
    provider: str,
    native_backends: Sequence[str],
    *,
    include_webgpu: bool = True,
    allow_ort_cpu_fallback: bool = False,
    use_ort_webgpu: bool = False,
) -> list[str]:
    ort_backend = _ort_backend_label(provider, allow_ort_cpu_fallback)
    tiers: list[str] = []
    for precision in ("fp32", "int8"):
        tiers.append(
            f"onnxruntime-web/webgpu-cpu-fallback/{precision}"
            if use_ort_webgpu else f"onnxruntime/{ort_backend}/{precision}"
        )
        tiers.extend(
            f"native-c/{backend}/{precision}" for backend in native_backends
        )
        if include_webgpu:
            tiers.append(f"volvoxai-webgpu/webgpu/{precision}")
    return tiers


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import onnxruntime as ort
        from .import_hf_split_onnx import validate_source
        from .tiny_receipt_tokenizer import TinyReceiptTokenizer
    except ImportError:
        try:
            import onnxruntime as ort
            from import_hf_split_onnx import validate_source
            from tiny_receipt_tokenizer import TinyReceiptTokenizer
        except ImportError as error:
            raise BenchmarkFailure(
                "GPU benchmark requires ONNX Runtime and TinyReceipt exporter helpers"
            ) from error

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
    packages = {
        "fp32": args.fp32_package.resolve(strict=True),
        "int8": args.int8_package.resolve(strict=True),
    }
    binary = args.native_binary.resolve(strict=True)
    ort_web_root: Path | None = None
    ort_web_identity: dict[str, Any] | None = None
    ort_web_runner: Path | None = None
    ort_web_page: Path | None = None
    if args.ort_web_root is not None:
        ort_web_root, ort_web_identity = load_ort_webgpu_identity(args.ort_web_root)
        ort_web_runner = args.ort_webgpu_runner.resolve(strict=True)
        ort_web_page = (
            repository
            / "examples/tiny_receipt_vqa/tools/ort_webgpu_benchmark_tinyreceipt.html"
        ).resolve(strict=True)
        for path, label in (
            (ort_web_runner, "ORT WebGPU runner"),
            (ort_web_page, "ORT WebGPU page"),
        ):
            if path.is_symlink() or not path.is_file():
                raise BenchmarkFailure(f"{label} must be a real file")
    runner: Path | None = None
    api: Path | None = None
    page: Path | None = None
    if not args.no_webgpu:
        runner = args.webgpu_runner.resolve(strict=True)
        api = (repository / "dist/0.4.0/volvoxai.js").resolve(strict=True)
        page = (
            repository
            / "examples/tiny_receipt_vqa/tools/webgpu_w8a8_benchmark_tinyreceipt.html"
        ).resolve(strict=True)
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise BenchmarkFailure("--native-binary must be a real executable")
    if not args.no_webgpu:
        assert runner is not None and api is not None and page is not None
        for path, label in (
            (runner, "WebGPU runner"),
            (api, "Web API"),
            (page, "WebGPU page"),
        ):
            if path.is_symlink() or not path.is_file():
                raise BenchmarkFailure(f"{label} must be a real file")
    if args.allow_ort_cpu_fallback and args.ort_provider == "CPUExecutionProvider":
        raise BenchmarkFailure(
            "--allow-ort-cpu-fallback requires a non-CPU --ort-provider"
        )
    if args.ort_web_root is None and args.ort_provider not in ort.get_available_providers():
        raise BenchmarkFailure(
            f"ONNX Runtime provider {args.ort_provider!r} is unavailable: "
            f"{ort.get_available_providers()}"
        )

    sources = {
        precision: dict(validate_source(
            source_root,
            variant="fp32" if precision == "fp32" else "int8-w8a8",
        ))
        for precision in ("fp32", "int8")
    }
    # Parent-side construction proves the same tokenizer documents before any
    # timed child is admitted. Each runtime must attest the exact IDs produced
    # here; each ORT child still constructs its own independent tokenizer.
    source_tokenizers = {
        precision: create_source_tokenizer(source, TinyReceiptTokenizer)
        for precision, source in sources.items()
    }
    package_identities = {
        precision: load_package_identity(
            packages[precision], precision, sources[precision]
        )
        for precision in ("fp32", "int8")
    }
    source_identities = _validated_source_identities(sources)
    generator_identity = _file_identity(Path(__file__).resolve())
    web_artifacts: dict[str, Any] | None = None
    if not args.no_webgpu:
        assert runner is not None and api is not None and page is not None
        web_artifacts = {
            "api": _file_identity(api),
            "runner": _file_identity(runner),
            "page": _file_identity(page),
            "contract": _file_identity(
                page.parent / "benchmark_explicit_kv_contract.mjs"
            ),
            "dynamicQualification": _file_identity(
                page.parent / "qualify_dynamic_rebind.mjs"
            ),
            "childProcessLifecycle": _file_identity(
                page.parent / "child_process_lifecycle.mjs"
            ),
        }
    ort_web_artifacts: dict[str, Any] | None = None
    if ort_web_root is not None:
        assert (
            ort_web_identity is not None
            and ort_web_runner is not None
            and ort_web_page is not None
        )
        ort_web_artifacts = {
            "distribution": ort_web_identity,
            "runner": _file_identity(ort_web_runner),
            "page": _file_identity(ort_web_page),
        }
    execution_inputs_before = {
        "repository": _git_provenance(repository, sorted(publication_outputs)),
        "generator": generator_identity,
        "source": source_identities,
        "packages": package_identities,
        "nativeBinary": _file_identity(binary),
        **({"web": web_artifacts} if web_artifacts is not None else {}),
        **({"onnxRuntimeWeb": ort_web_artifacts}
           if ort_web_artifacts is not None else {}),
    }

    prompt = unicodedata.normalize("NFC", DEFAULT_PROMPT)
    family = DEFAULT_FAMILY
    max_new = CANONICAL_MAX_NEW
    encoded_questions = {
        precision: list(tokenizer.encode(prompt, add_eos=True, max_len=192))
        for precision, tokenizer in source_tokenizers.items()
    }
    if encoded_questions["fp32"] != encoded_questions["int8"]:
        raise BenchmarkFailure("FP32 and INT8 producer tokenizers encode different questions")
    canonical_question_token_ids = encoded_questions["fp32"]
    expected_model_hashes = {
        precision: {
            role: sources[precision]["hashes"][
                sources[precision]["selected_file_keys"][role]
            ]
            for role in ("encoder", "decoder")
        }
        for precision in ("fp32", "int8")
    }
    provider_options = _provider_options(
        args.ort_provider, args.ort_provider_option
    )
    child_environment, child_environment_policy = _benchmark_child_environment(
        os.environ, args.ort_threads
    )

    build_root = repository / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    expected_tiers = _expected_tiers(
        args.ort_provider,
        args.native_backend,
        include_webgpu=not args.no_webgpu,
        allow_ort_cpu_fallback=args.allow_ort_cpu_fallback,
        use_ort_webgpu=ort_web_root is not None,
    )
    with tempfile.TemporaryDirectory(
        prefix="tinyreceipt-gpu-benchmark-", dir=build_root
    ) as directory:
        image = Path(directory) / "synthetic-receipt.png"
        image_identity = create_synthetic_image(image)

        qualification_entries: list[dict[str, Any]] = []
        # Qualification is intentionally outside matrix_once: each selected
        # physical VolvoxAI backend receives one fresh untimed process/session
        # before any per-sample execution warmup or measured timing can begin.
        for precision in ("fp32", "int8"):
            for backend in args.native_backend:
                qualification_entries.append(run_native_dynamic_qualification(
                    binary=binary,
                    package=packages[precision],
                    precision=precision,
                    image=image,
                    backend=backend,
                    max_new=max_new,
                    threads=1,
                    timeout=args.timeout,
                    environment=child_environment,
                ))
            if not args.no_webgpu:
                assert runner is not None
                qualification_entries.append(run_webgpu_sample(
                    node=args.node,
                    chrome=args.chrome,
                    runner=runner,
                    repository=repository,
                    package=packages[precision],
                    precision=precision,
                    image=image,
                    prompt=prompt,
                    family=family,
                    max_new=max_new,
                    timeout=args.timeout,
                    environment=child_environment,
                    dynamic_qualification=True,
                ))
        dynamic_qualification = prove_dynamic_qualification_coverage(
            qualification_entries,
            native_backends=args.native_backend,
            include_webgpu=not args.no_webgpu,
            expected_question_token_ids=(
                source_tokenizers["fp32"].encode("x", add_eos=True, max_len=192),
                canonical_question_token_ids,
                canonical_question_token_ids,
                source_tokenizers["fp32"].encode("x", add_eos=True, max_len=192),
            ),
        )
        qualification_run_order = [
            _tier_key(entry) for entry in qualification_entries
        ]

        def matrix_once(run_order: Sequence[str]) -> list[dict[str, Any]]:
            runners: dict[str, Any] = {}
            for precision in ("fp32", "int8"):
                ort_tier = (
                    f"onnxruntime-web/webgpu-cpu-fallback/{precision}"
                    if ort_web_root is not None else
                    f"onnxruntime/{_ort_backend_label(args.ort_provider, args.allow_ort_cpu_fallback)}/{precision}"
                )
                if ort_web_root is None:
                    runners[ort_tier] = lambda precision=precision: _run_ort_worker(
                            script=Path(__file__).resolve(),
                            source=source_root,
                            precision=precision,
                            image=image,
                            prompt=prompt,
                            family=family,
                            max_new=max_new,
                            threads=args.ort_threads,
                            provider=args.ort_provider,
                            provider_options=provider_options,
                            allow_cpu_fallback=args.allow_ort_cpu_fallback,
                            execution_warmup=args.warmup,
                            timeout=args.timeout,
                            environment=child_environment,
                        )
                else:
                    assert ort_web_runner is not None and ort_web_identity is not None
                    runners[ort_tier] = lambda precision=precision: run_ort_webgpu_sample(
                            node=args.node,
                            chrome=args.chrome,
                            runner=ort_web_runner,
                            repository=repository,
                            source=source_root,
                            ort_web_root=ort_web_root,
                            runtime_identity=ort_web_identity,
                            precision=precision,
                            image=image,
                            prompt=prompt,
                            family=family,
                            max_new=max_new,
                            timeout=args.timeout,
                            environment=child_environment,
                            expected_model_sha256=expected_model_hashes[precision],
                            execution_warmup=args.warmup,
                        )
                for backend in args.native_backend:
                    tier = f"native-c/{backend}/{precision}"
                    runners[tier] = lambda precision=precision, backend=backend: run_native_sample(
                            binary=binary,
                            package=packages[precision],
                            precision=precision,
                            image=image,
                            prompt=prompt,
                            family=family,
                            max_new=max_new,
                            threads=1,
                            timeout=args.timeout,
                            environment=child_environment,
                            backend=backend,
                            execution_warmup=args.warmup,
                        )
                if not args.no_webgpu:
                    assert runner is not None
                    tier = f"volvoxai-webgpu/webgpu/{precision}"
                    runners[tier] = lambda precision=precision: run_webgpu_sample(
                            node=args.node,
                            chrome=args.chrome,
                            runner=runner,
                            repository=repository,
                            package=packages[precision],
                            precision=precision,
                            image=image,
                            prompt=prompt,
                            family=family,
                            max_new=max_new,
                            timeout=args.timeout,
                            environment=child_environment,
                            execution_warmup=args.warmup,
                        )
            if set(runners) != set(expected_tiers) or list(run_order) != list(
                dict.fromkeys(run_order)
            ) or set(run_order) != set(expected_tiers):
                raise BenchmarkFailure("GPU matrix run order does not cover each tier once")
            values = [runners[tier]() for tier in run_order]
            prove_gpu_parity(
                values,
                expected_tiers=expected_tiers,
                repeat=1,
                max_new=max_new,
                allow_ort_cpu_fallback=args.allow_ort_cpu_fallback,
                expected_execution_warmup=args.warmup,
                expected_question_token_ids=canonical_question_token_ids,
            )
            return values

        samples: list[dict[str, Any]] = []
        run_orders: list[dict[str, Any]] = []
        for sample_index in range(args.repeat):
            run_order = _counterbalanced_tier_order(expected_tiers, sample_index)
            run_orders.append({"sampleIndex": sample_index, "tiers": run_order})
            for sample in matrix_once(run_order):
                sample["sampleIndex"] = sample_index
                samples.append(sample)
        parity = prove_gpu_parity(
            samples,
            expected_tiers=expected_tiers,
            repeat=args.repeat,
            max_new=max_new,
            allow_ort_cpu_fallback=args.allow_ort_cpu_fallback,
            expected_execution_warmup=args.warmup,
            expected_question_token_ids=canonical_question_token_ids,
        )
        comparisons = parity.pop("comparisons")
        measurement_boundaries = _measurement_boundaries(samples)
        web_artifacts_after: dict[str, Any] | None = None
        if not args.no_webgpu:
            assert runner is not None and api is not None and page is not None
            web_artifacts_after = {
                "api": _file_identity(api),
                "runner": _file_identity(runner),
                "page": _file_identity(page),
                "contract": _file_identity(
                    page.parent / "benchmark_explicit_kv_contract.mjs"
                ),
                "dynamicQualification": _file_identity(
                    page.parent / "qualify_dynamic_rebind.mjs"
                ),
                "childProcessLifecycle": _file_identity(
                    page.parent / "child_process_lifecycle.mjs"
                ),
            }
        ort_web_artifacts_after: dict[str, Any] | None = None
        if ort_web_root is not None:
            assert ort_web_runner is not None and ort_web_page is not None
            resolved_ort_web_root, distribution_after = load_ort_webgpu_identity(
                ort_web_root
            )
            if resolved_ort_web_root != ort_web_root:
                raise BenchmarkFailure(
                    "ONNX Runtime Web distribution path changed while benchmarking"
                )
            ort_web_artifacts_after = {
                "distribution": distribution_after,
                "runner": _file_identity(ort_web_runner),
                "page": _file_identity(ort_web_page),
            }
        execution_inputs_after = {
            "repository": _git_provenance(repository, sorted(publication_outputs)),
            "generator": _file_identity(Path(__file__).resolve()),
            "source": _validated_source_identities(sources),
            "packages": {
                precision: load_package_identity(
                    packages[precision], precision, sources[precision]
                )
                for precision in ("fp32", "int8")
            },
            "nativeBinary": _file_identity(binary),
            **({"web": web_artifacts_after}
               if web_artifacts_after is not None else {}),
            **({"onnxRuntimeWeb": ort_web_artifacts_after}
               if ort_web_artifacts_after is not None else {}),
        }
        _assert_execution_inputs_stable(
            execution_inputs_before, execution_inputs_after
        )

    cpu_identity = _host_cpu_identity()
    return {
        "format": REPORT_FORMAT,
        "status": "pass",
        "workload": {
            "image": image_identity,
            "prompt": prompt,
            "family": family,
            "maxNewTokens": max_new,
            "shapeMode": "active",
            "decoder": "greedy-explicit-kv",
            "questionTokenIds": canonical_question_token_ids,
        },
        "settings": {
            "executionWarmupRunsPerSample": args.warmup,
            "executionWarmupScope": (
                "untimed full encoder and explicit-KV decode on the same runtime, "
                "encoder context, and decoder context immediately before each "
                "measured request; state resets to the blocked P=1 sentinel"
            ),
            "measuredRepeatsPerTier": args.repeat,
            "runOrderPolicy": (
                "deterministic forward/reverse pairs, rotating the first tier by "
                "one position after every pair"
            ),
            "runOrderPerMatrix": run_orders,
            "sampleIsolation": (
                (
                    "fresh Node process, Chrome process/profile, pinned ONNX Runtime "
                    "WebGPU runtime, and encoder/decoder sessions"
                    if ort_web_root is not None else
                    "fresh Python process and fresh ONNX Runtime encoder/decoder sessions"
                )
                + "; fresh native child process/runtime"
                + (
                    "; fresh Node process, Chrome process/profile, VolvoxAI WebGPU "
                    "runtime, and encoder/decoder contexts"
                    if not args.no_webgpu else ""
                )
            ),
            "measurementBoundaries": measurement_boundaries,
            "childProcessEnvironment": {
                **child_environment_policy,
                "scope": (
                    "all ONNX Runtime, native C, ORT WebGPU, VolvoxAI WebGPU, "
                    "and dynamic-qualification child processes"
                ),
            },
            "commonTimingExclusions": (
                "process startup, model load/compile, preprocessing, tokenization, "
                "same-context warmup, and application correctness validation"
            ),
            "onnxRuntime": {
                "implementation": (
                    "onnxruntime-web" if ort_web_root is not None else "onnxruntime-python"
                ),
                "provider": (
                    "WebGpuExecutionProvider" if ort_web_root is not None
                    else args.ort_provider
                ),
                "providerOptions": {} if ort_web_root is not None else provider_options,
                "providerOrder": (
                    ["WebGpuExecutionProvider", "CPUExecutionProvider"]
                    if ort_web_root is not None else [
                        args.ort_provider,
                        *(["CPUExecutionProvider"] if args.allow_ort_cpu_fallback else []),
                    ]
                ),
                "strictNoFallback": (
                    False if ort_web_root is not None else not args.allow_ort_cpu_fallback
                ),
                "comparisonLabel": (
                    "optimized-session WebGPU/CPU partition counts; strict probe rejected"
                    if ort_web_root is not None else
                    "non-strict accelerator-first with exact executed-node placement"
                    if args.allow_ort_cpu_fallback else "strict provider"
                ),
                "cpuEpFallbackUsage": (
                    "optimized-session-partition-count-attested" if ort_web_root is not None else
                    "exact-executed-node-placement-attested"
                    if args.allow_ort_cpu_fallback
                    else "disabled"
                    if args.ort_provider != "CPUExecutionProvider"
                    else "not-applicable-cpu-primary"
                ),
                "cpuEpFallbackDisabled": (
                    False if ort_web_root is not None else
                    args.ort_provider != "CPUExecutionProvider"
                    and not args.allow_ort_cpu_fallback
                ),
                "threads": (
                    {
                        "wasmCpuEpThreadLimit": 1,
                        "scope": (
                            "ONNX Runtime Web WASM CPU execution-provider partitions "
                            "only; WebGPU device scheduling and Node/Chrome auxiliary "
                            "threads are outside this limit"
                        ),
                    }
                    if ort_web_root is not None
                    else {
                        "intraOpThreadLimit": args.ort_threads,
                        "interOpThreadLimit": 1,
                        "executionMode": "sequential",
                        "scope": (
                            "ONNX Runtime host CPU execution-provider work only; "
                            "accelerator device scheduling is outside this limit"
                        ),
                    }
                ),
                **({
                    "placementEvidence": (
                        "separate untimed profiling sessions and strict "
                        "CPU-EP-disabled session-create probes execute after each "
                        "measured request; measured sessions are not profiled"
                    ),
                } if args.allow_ort_cpu_fallback else {}),
                **({"version": ORT_WEBGPU_VERSION} if ort_web_root is not None else {}),
            },
            "nativeBackends": list(args.native_backend),
            "nativeStrictNoFallback": True,
            "dynamicShapeQualification": (
                "one untimed fresh process/runtime/session per precision and "
                "selected VolvoxAI GPU backend before per-sample execution warmup"
            ),
            "dynamicShapeQualificationRunOrder": qualification_run_order,
            "webgpu": {
                "enabled": not args.no_webgpu,
                **({
                    "adapterPolicy": "physical-hardware-only",
                    "timingSynchronization": "required-small-output-readback",
                    "measuredExplicitKv": (
                        "public device-buffer handoff with zero harness KV readbacks; "
                        "internal provider transfers are engine-specific"
                    ),
                    "cacheCorrectness": "untimed-device-qualified-request",
                }
                   if not args.no_webgpu else {}),
            },
        },
        "coverage": {
            "expectedTiers": expected_tiers,
            "nativeBackendsAvailableInHarness": list(NATIVE_GPU_BACKENDS),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "logicalCpuCount": os.cpu_count(),
            "cpu": cpu_identity,
            "python": platform.python_version(),
        },
        "dynamicShapeQualification": dynamic_qualification,
        "parity": parity,
        "comparisons": comparisons,
        "tiers": summarize_gpu_samples(samples),
        "samples": samples,
        "artifacts": {
            "source": source_identities,
            "packages": package_identities,
            "nativeBinary": execution_inputs_before["nativeBinary"],
            **({"web": web_artifacts} if web_artifacts is not None else {}),
            **({"onnxRuntimeWeb": ort_web_artifacts}
               if ort_web_artifacts is not None else {}),
        },
        "runtimeVersions": {
            "python": sys.version.splitlines()[0],
            **({
                "onnxruntimeWeb": ORT_WEBGPU_VERSION,
                "onnxruntimePythonValidator": ort.__version__,
            } if ort_web_root is not None else {
                "onnxruntime": ort.__version__,
            }),
            **({
                "node": _command_version([args.node, "--version"]),
                "chrome": _command_version([args.chrome, "--version"]),
            } if not args.no_webgpu or ort_web_root is not None else {}),
        },
        "provenance": {
            "repository": execution_inputs_before["repository"],
            "generator": generator_identity,
            "executionInputStability": {
                "checkedBeforeAndAfterExecution": True,
                "unchanged": True,
            },
        },
    }


def _parse_public_arguments(argv: Sequence[str]) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fp32-package", type=Path, required=True)
    parser.add_argument("--int8-package", type=Path, required=True)
    parser.add_argument(
        "--native-binary",
        type=Path,
        default=root / "examples/target/bin/tiny_receipt_split_w8a8",
    )
    parser.add_argument(
        "--native-backend",
        action="append",
        choices=NATIVE_GPU_BACKENDS,
        dest="native_backend",
        help="repeat to select a subset; default: vulkan, opengl, cuda",
    )
    parser.add_argument("--ort-provider", default="CPUExecutionProvider")
    parser.add_argument(
        "--ort-provider-option", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument(
        "--allow-ort-cpu-fallback",
        action="store_true",
        help=(
            "allow a non-CPU ORT provider to partition unsupported nodes onto "
            "CPU; results are labeled non-strict"
        ),
    )
    parser.add_argument(
        "--ort-web-root",
        type=Path,
        help=(
            f"use external pinned onnxruntime-web@{ORT_WEBGPU_VERSION} WebGPU/CPU "
            "partition baseline instead of the Python ORT reference"
        ),
    )
    parser.add_argument("--ort-threads", type=int, default=1)
    parser.add_argument("--node", default="node")
    parser.add_argument("--chrome", default=os.environ.get("CHROME", "google-chrome"))
    parser.add_argument(
        "--webgpu-runner",
        type=Path,
        default=root / "examples/tiny_receipt_vqa/tools/run_webgpu_tinyreceipt.mjs",
    )
    parser.add_argument(
        "--ort-webgpu-runner",
        type=Path,
        default=(
            root / "examples/tiny_receipt_vqa/tools/run_ort_webgpu_tinyreceipt.mjs"
        ),
    )
    parser.add_argument(
        "--no-webgpu",
        action="store_true",
        help="omit the browser WebGPU tiers and their tool requirements",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.native_backend is None:
        args.native_backend = list(NATIVE_GPU_BACKENDS)
    elif len(set(args.native_backend)) != len(args.native_backend):
        parser.error("--native-backend values must be unique")
    if args.allow_ort_cpu_fallback and args.ort_provider == "CPUExecutionProvider":
        parser.error("--allow-ort-cpu-fallback requires a non-CPU --ort-provider")
    if args.ort_web_root is not None and (
        args.allow_ort_cpu_fallback
        or args.ort_provider != "CPUExecutionProvider"
        or bool(args.ort_provider_option)
    ):
        parser.error(
            "--ort-web-root replaces the Python ORT reference and cannot be "
            "combined with Python ORT provider options"
        )
    if args.ort_web_root is not None and args.ort_threads != 1:
        parser.error(
            "--ort-web-root requires --ort-threads 1 because the browser page "
            "fixes the ONNX Runtime Web WASM CPU EP to one thread"
        )
    if not 0 <= args.warmup <= MAXIMUM_EXECUTION_WARMUP_RUNS \
            or args.repeat < 1 or args.ort_threads < 1 or args.timeout <= 0:
        parser.error(
            f"--warmup must be in [0, {MAXIMUM_EXECUTION_WARMUP_RUNS}], "
            "--repeat/--ort-threads >= 1, timeout > 0"
        )
    return args


def _ort_worker(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "int8"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--family", choices=FAMILY_NAMES, required=True)
    parser.add_argument("--max-new", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--provider-option", action="append", default=[])
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--execution-warmup", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.execution_warmup <= MAXIMUM_EXECUTION_WARMUP_RUNS:
        parser.error(
            "--execution-warmup must be in "
            f"[0, {MAXIMUM_EXECUTION_WARMUP_RUNS}]"
        )
    try:
        try:
            from .import_hf_split_onnx import validate_source
            from .tiny_receipt_tokenizer import TinyReceiptTokenizer
        except ImportError:
            from import_hf_split_onnx import validate_source
            from tiny_receipt_tokenizer import TinyReceiptTokenizer
        source = dict(validate_source(
            args.source.resolve(strict=True),
            variant="fp32" if args.precision == "fp32" else "int8-w8a8",
        ))
        source["tokenizer_runtime"] = create_source_tokenizer(
            source, TinyReceiptTokenizer
        )
        sample = run_ort_sample(
            source,
            args.precision,
            args.image.resolve(strict=True),
            args.prompt,
            args.family,
            args.max_new,
            args.threads,
            provider=args.provider,
            provider_options=_provider_options(args.provider, args.provider_option),
            allow_cpu_fallback=args.allow_cpu_fallback,
            execution_warmup=args.execution_warmup,
        )
        validate_sample(
            sample,
            max_new=args.max_new,
            expected_strict_no_fallback=not args.allow_cpu_fallback,
        )
        sys.stdout.write(f"{SAMPLE_PREFIX}{json.dumps(sample, separators=(',', ':'))}\n")
        return 0
    except (BenchmarkFailure, OSError, RuntimeError, ValueError) as error:
        print(f"GPU benchmark ORT worker failed: {error}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values[:1] == ["__ort_sample__"]:
        return _ort_worker(values[1:])
    try:
        args = _parse_public_arguments(values)
        report = build_report(args)
        if args.report is None:
            json.dump(report, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _atomic_write(args.report, report)
            print(f"wrote {args.report}")
        return 0
    except (BenchmarkFailure, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"GPU benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
