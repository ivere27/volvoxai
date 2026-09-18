#!/usr/bin/env python3
"""Run and canonically publish the current native GPU batching benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import statistics
import subprocess
import sys
import platform
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "volvoxai.native-dynamic-batch-matrix-input"
OUTPUT_SCHEMA = "volvoxai.native-dynamic-batch-benchmark"
CASE_RECORD = "native-dynamic-batch-case"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")
FORBIDDEN_INHERITED_ENV = (
    "VOLVOXAI_SHADER_DIR",
    "VOLVOXAI_NATIVE_SHADER_COMPILER",
    "VOLVOXAI_CPU_ISA",
    "VOLVOXAI_CUDA_PROFILE_PATH",
    "VOLVOXAI_DISABLE_GPU_CPU_ROW",
    "VOLVOXAI_DISABLE_GPU_DEVICE_ROW",
    "VOLVOXAI_ROW_DEBUG",
    "VOLVOXAI_THREADS",
    "VOLVOX_ARENA",
    "VOLVOX_F32_AVX512",
    "VOLVOX_F32_GEMM_PACKED",
    "VOLVOX_GL_PROFILE_SYNC",
    "VOLVOX_OPENGL_DISABLE_TILED_QCONV",
    "VOLVOX_PW_GEMM",
    "VOLVOX_VULKAN_DISABLE_DOT",
    "VOLVOX_DISABLE_OPERATOR_FUSION",
    "VOLVOX_ENABLE_CHAINED_ADD_FUSE",
    "VOLVOX_ENABLE_DW_PW_FUSE",
    "VOLVOX_VULKAN_MB",
    "VOLVOXAI_CUDA_DEVICE",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
)
SOURCE_SNAPSHOT_INCLUDED = (
    "CMakeLists.txt",
    "Makefile",
    "package.json",
    "package-lock.json",
    "native/**",
    "proto/**",
    "shaders/**",
    "tools/**",
    "examples/native_dynamic_batch_benchmark/**",
)
SOURCE_SNAPSHOT_EXCLUDED = (
    "docs/**",
    "examples/receipt_digit_reader/reports/**",
    "examples/tiny_receipt_vqa/reports/**",
)
SOURCE_SNAPSHOT_GIT_PATHS = (
    "CMakeLists.txt",
    "Makefile",
    "package.json",
    "package-lock.json",
    "native",
    "proto",
    "shaders",
    "tools",
    "examples/native_dynamic_batch_benchmark",
)

CONFIG_KEYS = {
    "schema",
    "suite",
    "harness",
    "measuredAtUtc",
    "device",
    "build",
    "warmupGroups",
    "repeatGroups",
    "maxBatchDelayMs",
    "vulkanArenaMB",
    "cases",
}
DEVICE_KEYS = {
    "expectedName",
    "expectedDriverVersion",
    "expectedTotalMemoryMiB",
}
BUILD_KEYS = {
    "sourceRoot",
    "buildDirectory",
    "nativeShaderCompiler",
    "naga",
}
CASE_CONFIG_KEYS = {
    "id",
    "artifactRole",
    "precision",
    "backend",
    "inputMode",
    "batchSize",
    "manifest",
    "graph",
    "weights",
    "laneFiles",
}
CASE_RESULT_KEYS = {
    "record",
    "caseId",
    "backend",
    "inputMode",
    "status",
    "logicalLanes",
    "uniqueInputClasses",
    "fixtureReuse",
    "syntheticFixture",
    "resolvedInputs",
    "runtime",
    "batchContract",
    "strictRoute",
    "measurement",
    "direct",
    "scheduled",
    "parity",
    "publicEvidence",
}
SYNTHETIC_RECEIPT_KEYS = {"kind", "seed"}
SYNTHETIC_ENCODER_KEYS = {"kind", "seed", "Q", "M"}
SYNTHETIC_DECODER_KEYS = {"kind", "seed", "M", "P", "R"}
RESOLVED_INPUT_KEYS = {"name", "dtype", "shape"}
RUNTIME_KEYS = {
    "executionMode",
    "cpuThreads",
    "freshness",
    "maxScheduledRequests",
    "maxScheduledInputBytes",
    "maxUnconsumedResults",
    "maxUnconsumedResultBytes",
    "requestedMaxBatchDelayMs",
    "effectiveMaxBatchDelayMs",
    "vulkanArenaMB",
}
STRICT_ROUTE_KEYS = {
    "routeAttested",
    "provider",
    "nodes",
    "selected",
    "fallback",
    "missing",
    "operatorFallbackEvidence",
}
BATCH_CONTRACT_KEYS = {
    "protocol",
    "axis",
    "symbol",
    "min",
    "max",
    "multiple",
    "proofIdentity",
}
MEASUREMENT_KEYS = {
    "warmupGroups",
    "repeatGroups",
    "order",
    "clock",
    "timingBoundary",
}
DIRECT_KEYS = {
    "logicalLanes",
    "physicalBatchSize",
    "trueBackendInvocationsPerGroup",
    "samplesMs",
    "medianMs",
}
SCHEDULED_KEYS = DIRECT_KEYS | {
    "allLanesSharedPhysicalExecutionId",
    "physicalExecutionIdChangesAcrossGroups",
    "physicalExecutionIdScope",
    "physicalExecutionIds",
}
PARITY_KEYS = {
    "status",
    "absoluteTolerance",
    "relativeTolerance",
    "combinedAllclose",
    "nonfiniteRejected",
    "integerOutputsByteExact",
    "receiptDecodedEquality",
    "comparedElements",
    "outputTensors",
    "maxAbs",
    "maxRel",
}
PUBLIC_EVIDENCE_KEYS = {
    "source",
    "compiledContractExact",
    "scheduledLaneTokensExact",
    "directBatchTokensAbsent",
    "backendExecutionProofExact",
    "vulkanComputeDeviceLocal",
    "vulkanArenaBytes",
    "vulkanStagingBytes",
    "vulkanStagingHostCoherent",
    "cudaGraphReplayMeasured",
}


class ValidationError(ValueError):
    pass


def verify_clean_environment() -> None:
    present = [name for name in FORBIDDEN_INHERITED_ENV if name in os.environ]
    if present:
        raise ValidationError(
            "performance/device override environment must be unset: "
            + ",".join(present)
        )
    if not hasattr(os, "sched_getaffinity") or 0 not in os.sched_getaffinity(0):
        raise ValidationError("official benchmark requires logical CPU 0")


def _pin_cpu_zero() -> None:
    os.sched_setaffinity(0, {0})


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValidationError(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _required_string(value: Any, label: str, *, token: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a non-empty string")
    if token and not SAFE_TOKEN_RE.fullmatch(value):
        raise ValidationError(f"{label} is not a safe token")
    return value


def _required_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _existing_file(value: Any, label: str) -> Path:
    path = Path(_required_string(value, label))
    if not path.is_file():
        raise ValidationError(f"{label} is not a file")
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_lane_identity(config: dict[str, Any]) -> tuple[tuple[int, str], ...]:
    if config["suite"] != "receipt-digit-reader":
        return ()
    lanes = config["cases"][0]["laneFiles"]
    identities: list[tuple[int, str]] = []
    if not lanes:
        raise ValidationError("receipt lane identity requires private classes")
    for lane in lanes:
        path = Path(lane).resolve()
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ValidationError("receipt lane bytes changed while hashing")
        identities.append((after.st_size, digest))
    return tuple(identities)


def _require_receipt_lane_identity(
    config: dict[str, Any], expected: tuple[tuple[int, str], ...]
) -> None:
    if config["suite"] == "receipt-digit-reader" and (
        not expected or _receipt_lane_identity(config) != expected
    ):
        raise ValidationError("private receipt lane bytes changed across cases")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{label} cannot be read: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    return value


def _verify_graph_batch_domain(
    graph: dict[str, Any], expected_max: int, *, require_multiple: bool
) -> str:
    dimensions = graph.get("dimensions")
    inputs = graph.get("inputs")
    if (
        graph.get("format") != "volvox-graph/v1"
        or not isinstance(dimensions, dict)
        or not isinstance(inputs, dict)
        or not inputs
    ):
        raise ValidationError("graph format/dimensions/inputs are invalid")
    symbols: set[str] = set()
    for descriptor in inputs.values():
        shape = descriptor.get("shape") if isinstance(descriptor, dict) else None
        if (
            not isinstance(shape, list)
            or not shape
            or not isinstance(shape[0], str)
            or not shape[0]
        ):
            raise ValidationError("graph input lacks a symbolic leading batch")
        symbols.add(shape[0])
    if len(symbols) != 1:
        raise ValidationError("graph inputs do not share one leading batch symbol")
    symbol = next(iter(symbols))
    expected_domain = {"min": 1, "max": expected_max}
    if require_multiple:
        expected_domain["multiple_of"] = 1
    if dimensions.get(symbol) != expected_domain:
        raise ValidationError("graph batch domain differs from the exact package")
    return symbol


def _package_declared_batch_max(
    suite: str, case: dict[str, Any], manifest_path: Path, graph_path: Path,
) -> int:
    """Return the package's exact leading-B maximum without hashing bytes.

    This is used while validating the declarative matrix.  Full package ABI,
    convention, and digest checks still run immediately before and after each
    benchmark case in ``verify_manifest``.
    """
    manifest = _load_json(manifest_path, "package manifest")
    graph = _load_json(graph_path, "selected graph")
    if suite == "receipt-digit-reader":
        abi = manifest.get("abi")
        batch = abi.get("batch") if isinstance(abi, dict) else None
        declared_max = batch.get("max") if isinstance(batch, dict) else None
    else:
        shape_contract = manifest.get("shape_contract")
        dimensions = (
            shape_contract.get("dimensions")
            if isinstance(shape_contract, dict) else None
        )
        batch = dimensions.get("B") if isinstance(dimensions, dict) else None
        declared_max = batch.get("max") if isinstance(batch, dict) else None
    declared_max = _required_int(
        declared_max, f"{case['id']} package batch maximum", 2
    )
    if declared_max < case["batchSize"]:
        raise ValidationError(
            f"{case['id']} executes B{case['batchSize']} outside package "
            f"maxB{declared_max}"
        )
    _verify_graph_batch_domain(
        graph, declared_max,
        require_multiple=suite == "receipt-digit-reader",
    )
    return declared_max


def verify_manifest(
    suite: str, case: dict[str, Any], manifest_path: Path,
    graph_path: Path, weight_paths: list[Path],
) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "package manifest")
    graph = _load_json(graph_path, "selected graph")
    graph_digest = _sha256(graph_path)
    weight_digests = [_sha256(path) for path in weight_paths]
    if suite == "receipt-digit-reader":
        if (
            manifest_path.name != "manifest.json"
            or graph_path != manifest_path.parent / "graph.json"
            or len(weight_paths) != 1
            or weight_paths[0] != manifest_path.parent / "model.safetensors"
        ):
            raise ValidationError(
                "receipt package must select exact sibling graph/weights"
            )
        expected_variant = case["precision"]
        abi = manifest.get("abi")
        batch = abi.get("batch") if isinstance(abi, dict) else None
        declared_batch_max = (
            batch.get("max") if isinstance(batch, dict) else None
        )
        expected_decode = {
            "slots": 16,
            "phone_slots": 12,
            "street_slots": 4,
            "blank_class": 10,
            "num_classes": 11,
        }
        graph_inputs = graph.get("inputs")
        graph_outputs = graph.get("outputs")
        output_descriptors = []
        graph_nodes = graph.get("nodes")
        for node in graph_nodes if isinstance(graph_nodes, list) else []:
            if not isinstance(node, dict):
                continue
            for descriptor in (
                node.get("outputs", {}).values()
                if isinstance(node.get("outputs"), dict) else ()
            ):
                if (
                    isinstance(descriptor, dict)
                    and descriptor.get("tensor") == "slot_logits"
                ):
                    output_descriptors.append(descriptor)
        if (
            manifest.get("format") !=
                "volvoxai-receipt-digit-reader-onnx-package-v1"
            or manifest.get("source_format") != "receipt_digit_reader_onnx_v1"
            or manifest.get("variant") != expected_variant
            or not isinstance(abi, dict)
            or abi.get("input") != {
                "name": "input0", "shape": [1, 1, 320, 672]
            }
            or abi.get("output") != "slot_logits"
            or manifest.get("decode") != expected_decode
            or not isinstance(batch, dict)
            or set(batch) != {
                "per_request", "symbol", "min", "max", "multiple_of"
            }
            or batch["per_request"] != 1
            or not isinstance(batch["symbol"], str)
            or not batch["symbol"]
            or batch["min"] != 1
            or isinstance(declared_batch_max, bool)
            or not isinstance(declared_batch_max, int)
            or declared_batch_max < case["batchSize"]
            or batch["multiple_of"] != 1
        ):
            raise ValidationError("receipt manifest format/variant/batch ABI differs")
        symbol = _verify_graph_batch_domain(
            graph, declared_batch_max, require_multiple=True
        )
        if (
            symbol != batch["symbol"]
            or not isinstance(graph_inputs, dict)
            or set(graph_inputs) != {"input0"}
            or graph_inputs["input0"] != {
                "shape": [symbol, 1, 320, 672],
                "dtype": "float32",
            }
            or graph_outputs != ["slot_logits"]
            or len(output_descriptors) != 1
            or output_descriptors[0].get("dtype") != "float32"
            or output_descriptors[0].get("shape") != [symbol, 16, 11]
        ):
            raise ValidationError("receipt manifest/graph batch symbol differs")
        verification = "package-convention+manifest-abi+graph-domain"
    else:
        if manifest_path.name != "package_manifest.json":
            raise ValidationError("VQA manifest filename is not canonical")
        expected_variant = (
            "fp32" if case["precision"] == "fp32" else "int8-w8a8"
        )
        component = case["inputMode"].removeprefix("vqa-")
        source = manifest.get("source")
        graphs = manifest.get("graphs")
        selected = graphs.get(component) if isinstance(graphs, dict) else None
        shape_contract = manifest.get("shape_contract")
        shape_dimensions = (
            shape_contract.get("dimensions")
            if isinstance(shape_contract, dict) else None
        )
        declared_batch = (
            shape_dimensions.get("B")
            if isinstance(shape_dimensions, dict) else None
        )
        declared_batch_max = (
            declared_batch.get("max")
            if isinstance(declared_batch, dict) else None
        )
        if (
            not isinstance(declared_batch, dict)
            or set(declared_batch) != {"min", "max"}
            or declared_batch.get("min") != 1
            or isinstance(declared_batch_max, bool)
            or not isinstance(declared_batch_max, int)
            or declared_batch_max < case["batchSize"]
        ):
            raise ValidationError("VQA manifest batch domain differs")
        expected_shape_dimensions = {
            "B": {"min": 1, "max": declared_batch_max},
            "Q": {"min": 1, "max": 192},
            "M": {"min": 211, "max": 402},
            "P": {"min": 1, "max": 191},
            "R": {"min": 2, "max": 192},
        }
        if component == "encoder":
            semantic_inputs = {
                "image": "input0",
                "question_ids": "input1",
                "family_ids": "input2",
                "question_position_ids": "input3",
            }
            graph_inputs_expected = {
                "input0": {
                    "shape": ["B", 1, 320, 672], "dtype": "float32"
                },
                "input1": {"shape": ["B", "Q"], "dtype": "int32"},
                "input2": {"shape": ["B"], "dtype": "int32"},
                "input3": {"shape": ["B", "Q"], "dtype": "int32"},
            }
            graph_dimensions_expected = {
                "B": {"min": 1, "max": declared_batch_max},
                "Q": {"min": 1, "max": 192},
                "M": {"min": 211, "max": 402},
            }
            graph_output_specs = {
                "memory": (["B", "M", 320], "float32"),
                "memory_padding_mask": (["B", "M"], "int32"),
                "router_logits": (["B", 8], "float32"),
                "selected_family_ids": (["B"], "int32"),
                **{
                    f"cross_{kind}_{layer}":
                        (["B", 8, "M", 40], "float32")
                    for layer in range(4) for kind in ("k", "v")
                },
            }
        else:
            semantic_names = [
                "decoder_input_ids", "position_ids", "family_ids",
                "memory_padding_mask", "past_padding_mask",
                *(f"cross_{kind}_{layer}" for layer in range(4)
                  for kind in ("k", "v")),
                *(f"past_{kind}_{layer}" for layer in range(4)
                  for kind in ("k", "v")),
            ]
            semantic_inputs = {
                name: f"input{index}"
                for index, name in enumerate(semantic_names)
            }
            graph_inputs_expected = {
                "input0": {"shape": ["B", 1], "dtype": "int32"},
                "input1": {"shape": ["B"], "dtype": "int32"},
                "input2": {"shape": ["B"], "dtype": "int32"},
                "input3": {"shape": ["B", "M"], "dtype": "int32"},
                "input4": {"shape": ["B", "P"], "dtype": "int32"},
                **{
                    f"input{index}": {
                        "shape": ["B", 8, "M", 40], "dtype": "float32"
                    }
                    for index in range(5, 13)
                },
                **{
                    f"input{index}": {
                        "shape": ["B", 8, "P", 40], "dtype": "float32"
                    }
                    for index in range(13, 21)
                },
            }
            graph_dimensions_expected = {
                "B": {"min": 1, "max": declared_batch_max},
                "M": {"min": 211, "max": 402},
                "P": {"min": 1, "max": 191},
                "R": {"min": 2, "max": 192},
            }
            graph_output_specs = {
                "logits": (["B", 1, 1536], "float32"),
                "present_padding_mask": (["B", "R"], "int32"),
                **{
                    f"present_{kind}_{layer}":
                        (["B", 8, "R", 40], "float32")
                    for layer in range(4) for kind in ("k", "v")
                },
            }
        selected_inputs = selected.get("inputs") if isinstance(selected, dict) else None
        selected_outputs = (
            selected.get("outputs") if isinstance(selected, dict) else None
        )
        graph_inputs = graph.get("inputs")
        raw_graph_dimensions = graph.get("dimensions")
        graph_dimensions = (
            {
                name: value for name, value in raw_graph_dimensions.items()
                if isinstance(name, str) and not name.startswith("bank_")
            }
            if isinstance(raw_graph_dimensions, dict) else None
        )
        expected_output_names = list(graph_output_specs)
        output_descriptors: dict[str, list[dict[str, Any]]] = {
            name: [] for name in expected_output_names
        }
        nodes = graph.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                outputs = node.get("outputs") if isinstance(node, dict) else None
                if not isinstance(outputs, dict):
                    continue
                for descriptor in outputs.values():
                    tensor = (
                        descriptor.get("tensor")
                        if isinstance(descriptor, dict) else None
                    )
                    if tensor in output_descriptors:
                        output_descriptors[tensor].append(descriptor)
        if (
            manifest.get("format") !=
                "volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2"
            or not isinstance(source, dict)
            or source.get("format") !=
                "tiny_receipt_vqa_split_kv_onnx_v2"
            or source.get("variant") != expected_variant
            or not isinstance(graphs, dict)
            or set(graphs) != {"encoder", "decoder"}
            or not isinstance(selected, dict)
            or shape_dimensions != expected_shape_dimensions
            or selected_inputs != semantic_inputs
            or selected_outputs != {
                name: name for name in expected_output_names
            }
            or graph_inputs != graph_inputs_expected
            or graph_dimensions != graph_dimensions_expected
            or graph.get("outputs") != expected_output_names
            or any(
                descriptors != [
                    {
                        "tensor": name,
                        "shape": graph_output_specs[name][0],
                        "dtype": graph_output_specs[name][1],
                    }
                ]
                for name, descriptors in output_descriptors.items()
            )
        ):
            raise ValidationError("VQA manifest format/variant/component differs")
        for field, path, digest in (
            ("graph", graph_path, graph_digest),
            ("weights", weight_paths[0] if len(weight_paths) == 1 else Path(),
             weight_digests[0] if len(weight_digests) == 1 else ""),
        ):
            record = selected.get(field)
            expected_relative_path = (
                f"{component}/graph.json" if field == "graph"
                else f"{component}/model.safetensors"
            )
            if (
                len(weight_paths) != 1
                or not isinstance(record, dict)
                or set(record) != {"path", "bytes", "sha256"}
                or not isinstance(record.get("path"), str)
                or not record["path"]
                or record["path"] != expected_relative_path
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
                or not isinstance(record.get("sha256"), str)
                or not SHA256_RE.fullmatch(record["sha256"])
                or (manifest_path.parent / record["path"]).resolve() !=
                    path.resolve()
                or record["bytes"] != path.stat().st_size
                or record["sha256"] != digest
            ):
                raise ValidationError(
                    f"VQA manifest does not bind selected {field} bytes"
                )
        _verify_graph_batch_domain(
            graph, declared_batch_max, require_multiple=False
        )
        verification = "manifest-selected-path+bytes+sha256+graph-domain"
    return {
        "manifestSha256": _sha256(manifest_path),
        "graphSha256": graph_digest,
        "weightSha256": weight_digests,
        "packageBatchMax": declared_batch_max,
        "packageBindingVerification": verification,
    }


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"config cannot be read: {error}") from error
    config = _exact_keys(raw, CONFIG_KEYS, "config")
    if config["schema"] != INPUT_SCHEMA:
        raise ValidationError("unsupported input schema")
    if config["suite"] not in {"receipt-digit-reader", "tiny-receipt-vqa"}:
        raise ValidationError("unsupported suite")
    _existing_file(config["harness"], "harness")
    measured = _required_string(config["measuredAtUtc"], "measuredAtUtc")
    try:
        parsed = dt.datetime.fromisoformat(measured.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValidationError("measuredAtUtc must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValidationError("measuredAtUtc must include a timezone")
    device = _exact_keys(config["device"], DEVICE_KEYS, "device")
    _required_string(device["expectedName"], "device.expectedName")
    _required_string(
        device["expectedDriverVersion"], "device.expectedDriverVersion", token=True
    )
    _required_int(device["expectedTotalMemoryMiB"],
                  "device.expectedTotalMemoryMiB", 1)
    build = _exact_keys(config["build"], BUILD_KEYS, "build")
    source_root = Path(_required_string(build["sourceRoot"],
                                        "build.sourceRoot")).resolve()
    build_directory = Path(_required_string(build["buildDirectory"],
                                             "build.buildDirectory")).resolve()
    native_shader_compiler = _existing_file(
        build["nativeShaderCompiler"], "build.nativeShaderCompiler"
    )
    naga = _existing_file(build["naga"], "build.naga")
    if (
        not os.access(native_shader_compiler, os.X_OK)
        or not os.access(naga, os.X_OK)
    ):
        raise ValidationError("shader toolchain inputs must be executable")
    if source_root != Path(__file__).resolve().parents[2]:
        raise ValidationError("build.sourceRoot must own this exact wrapper")
    if not (source_root / ".git").exists():
        raise ValidationError("build.sourceRoot is not a git worktree")
    if not (build_directory / "CMakeCache.txt").is_file() or not (
        build_directory / "compile_commands.json"
    ).is_file():
        raise ValidationError(
            "buildDirectory requires CMakeCache.txt and compile_commands.json"
        )
    if _existing_file(config["harness"], "harness") != (
        build_directory / "native" / "native_dynamic_batch_benchmark"
    ):
        raise ValidationError("harness must be the selected build target")
    warmup = _required_int(config["warmupGroups"], "warmupGroups", 1)
    repeat = _required_int(config["repeatGroups"], "repeatGroups", 1)
    if warmup > 1000 or repeat > 1000:
        raise ValidationError("warmupGroups/repeatGroups must be <= 1000")
    delay = _required_int(config["maxBatchDelayMs"], "maxBatchDelayMs", 1)
    arena = _required_int(config["vulkanArenaMB"], "vulkanArenaMB", 1)
    if delay > 0xFFFFFFFF or arena > 0xFFFFFFFF:
        raise ValidationError(
            "maxBatchDelayMs/vulkanArenaMB must fit the public uint32 ABI"
        )
    if not isinstance(config["cases"], list) or not config["cases"]:
        raise ValidationError("cases must be a non-empty array")
    seen: set[str] = set()
    declared_max_by_case: dict[str, int] = {}
    declared_max_by_selection: dict[tuple[str, str], int] = {}
    for index, value in enumerate(config["cases"]):
        case = _exact_keys(value, CASE_CONFIG_KEYS, f"cases[{index}]")
        case_id = _required_string(case["id"], f"cases[{index}].id", token=True)
        if case_id in seen:
            raise ValidationError(f"duplicate case id {case_id}")
        seen.add(case_id)
        _required_string(case["artifactRole"], f"cases[{index}].artifactRole", token=True)
        if case["precision"] not in {"fp32", "int8"}:
            raise ValidationError(f"cases[{index}].precision is unsupported")
        if case["backend"] not in {"vulkan", "opengl", "cuda"}:
            raise ValidationError(f"cases[{index}].backend is unsupported")
        if case["inputMode"] not in {"receipt", "vqa-encoder", "vqa-decoder"}:
            raise ValidationError(f"cases[{index}].inputMode is unsupported")
        batch = _required_int(case["batchSize"], f"cases[{index}].batchSize", 2)
        _existing_file(case["manifest"], f"cases[{index}].manifest")
        _existing_file(case["graph"], f"cases[{index}].graph")
        if not isinstance(case["weights"], list) or not case["weights"]:
            raise ValidationError(f"cases[{index}].weights must be non-empty")
        for weight_index, weight in enumerate(case["weights"]):
            _existing_file(weight, f"cases[{index}].weights[{weight_index}]")
        if not isinstance(case["laneFiles"], list):
            raise ValidationError(f"cases[{index}].laneFiles must be an array")
        for lane_index, lane in enumerate(case["laneFiles"]):
            _existing_file(lane, f"cases[{index}].laneFiles[{lane_index}]")
        if case["inputMode"] == "receipt":
            if config["suite"] != "receipt-digit-reader":
                raise ValidationError("receipt mode belongs to receipt suite")
            if not case["laneFiles"] or len(case["laneFiles"]) > batch:
                raise ValidationError(
                    "receipt lane class count must be nonzero and <= batchSize"
                )
        else:
            if config["suite"] != "tiny-receipt-vqa":
                raise ValidationError("VQA modes belong to VQA suite")
            if case["laneFiles"]:
                raise ValidationError("VQA synthetic fixtures take no lane files")
        selection_key = (
            str(Path(case["manifest"]).resolve()),
            str(Path(case["graph"]).resolve()),
        )
        declared_max = declared_max_by_selection.get(selection_key)
        if declared_max is None:
            declared_max = _package_declared_batch_max(
                config["suite"], case,
                Path(case["manifest"]).resolve(),
                Path(case["graph"]).resolve(),
            )
            declared_max_by_selection[selection_key] = declared_max
        elif declared_max < batch:
            raise ValidationError(
                f"{case_id} executes B{batch} outside package maxB{declared_max}"
            )
        declared_max_by_case[case_id] = declared_max

    actual_combinations = {
        (case["backend"], case["precision"], case["inputMode"],
         case["batchSize"])
        for case in config["cases"]
    }
    required_workloads = (
        {"receipt"}
        if config["suite"] == "receipt-digit-reader"
        else {"vqa-encoder", "vqa-decoder"}
    )
    actual_workloads = {case["inputMode"] for case in config["cases"]}
    if actual_workloads != required_workloads:
        raise ValidationError("cases do not contain every suite workload")
    workload_batches = {
        workload: {
            case["batchSize"] for case in config["cases"]
            if case["inputMode"] == workload
        }
        for workload in required_workloads
    }
    expected_combinations = {
        (backend, precision, workload, batch)
        for workload, batches in workload_batches.items()
        for batch in batches
        for backend in ("vulkan", "opengl", "cuda")
        for precision in ("fp32", "int8")
    }
    if (
        actual_combinations != expected_combinations
        or len(config["cases"]) != len(expected_combinations)
    ):
        raise ValidationError(
            "cases do not form each workload's full "
            "Vulkan/OpenGL/CUDA x FP32/INT8 batch cross-product"
        )
    for case in config["cases"]:
        declared_max = declared_max_by_case[case["id"]]
        expected_role = (
            f"receipt-{case['precision']}-maxB{declared_max}"
            if config["suite"] == "receipt-digit-reader"
            else f"vqa-{case['precision']}-"
                 f"{case['inputMode'].removeprefix('vqa-')}-"
                 f"maxB{declared_max}"
        )
        if case["artifactRole"] != expected_role:
            raise ValidationError("artifactRole is not the exact matrix role")
        expected_id = (
            f"receipt-{case['precision']}-b{case['batchSize']}-"
            f"{case['backend']}"
            if config["suite"] == "receipt-digit-reader"
            else f"vqa-{case['precision']}-"
                 f"{case['inputMode'].removeprefix('vqa-')}-"
                 f"b{case['batchSize']}-{case['backend']}"
        )
        if case["id"] != expected_id:
            raise ValidationError("case id is not the exact matrix identifier")
    backend_order = {"vulkan": 0, "opengl": 1, "cuda": 2}
    precision_order = {"fp32": 0, "int8": 1}
    input_order = {"receipt": 0, "vqa-encoder": 0, "vqa-decoder": 1}
    config["cases"].sort(
        key=lambda case: (
            backend_order[case["backend"]],
            precision_order[case["precision"]],
            input_order[case["inputMode"]],
            case["batchSize"],
        )
    )
    role_artifacts: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    workload_artifacts: dict[
        tuple[str, str],
        tuple[str, tuple[str, str, tuple[str, ...]], int],
    ] = {}
    for case in config["cases"]:
        selection = (
            str(Path(case["manifest"]).resolve()),
            str(Path(case["graph"]).resolve()),
            tuple(str(Path(path).resolve()) for path in case["weights"]),
        )
        previous = role_artifacts.setdefault(case["artifactRole"], selection)
        if previous != selection:
            raise ValidationError(
                "one artifactRole selected different bytes across backends"
            )
        workload_key = (case["precision"], case["inputMode"])
        workload_selection = (
            case["artifactRole"], selection,
            declared_max_by_case[case["id"]],
        )
        previous_workload = workload_artifacts.setdefault(
            workload_key, workload_selection
        )
        if previous_workload != workload_selection:
            raise ValidationError(
                "every batch row in one precision/workload must share one "
                "artifactRole, package maximum, and exact package selection"
            )
    if len(set(role_artifacts.values())) != len(role_artifacts):
        raise ValidationError("different artifact roles alias the same package")
    if config["suite"] == "receipt-digit-reader":
        lane_sets = {
            tuple(str(Path(path).resolve()) for path in case["laneFiles"])
            for case in config["cases"]
        }
        if len(lane_sets) != 1:
            raise ValidationError(
                "receipt matrix must reuse the same private lane classes"
            )
    else:
        component_artifacts: dict[
            tuple[str, int], dict[str, tuple[str, str, tuple[str, ...]]]
        ] = {}
        for case in config["cases"]:
            key = (case["precision"], declared_max_by_case[case["id"]])
            component = case["inputMode"].removeprefix("vqa-")
            component_artifacts.setdefault(key, {})[component] = (
                role_artifacts[case["artifactRole"]]
            )
        for components in component_artifacts.values():
            if set(components) != {"encoder", "decoder"}:
                raise ValidationError(
                    "each VQA precision/package max needs encoder and decoder"
                )
            encoder = components["encoder"]
            decoder = components["decoder"]
            if (
                encoder[0] != decoder[0]
                or encoder[1] == decoder[1]
                or encoder[2] == decoder[2]
            ):
                raise ValidationError(
                    "VQA precision/package max must share one manifest and "
                    "select different encoder/decoder graph+weights"
                )
    return config


def query_device(device: dict[str, Any]) -> dict[str, Any]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0 or process.stderr.strip():
        raise ValidationError("device query failed or wrote diagnostics")
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    fields = [part.strip() for part in lines[0].split(",")] if len(lines) == 1 else []
    if len(fields) != 3:
        raise ValidationError(
            "device query must return exactly 'name, driver, memory MiB'"
        )
    name, driver, memory_text = fields
    try:
        memory_mib = int(memory_text)
    except ValueError as error:
        raise ValidationError("device memory is not an integer MiB value") from error
    if (
        name != device["expectedName"]
        or driver != device["expectedDriverVersion"]
        or memory_mib != device["expectedTotalMemoryMiB"]
    ):
        raise ValidationError(
            "observed device name/driver/memory differs from expected"
        )
    lowered = name.lower()
    if any(token in lowered for token in ("llvmpipe", "swiftshader", "software")):
        raise ValidationError("software device is not publishable")
    return {
        "name": name,
        "driverVersion": driver,
        "totalMemoryMiB": memory_mib,
        "ordinal": 0,
    }


def _run_evidence(command: list[str], label: str) -> str:
    process = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=30
    )
    if process.returncode != 0:
        raise ValidationError(f"{label} query failed")
    output = (process.stdout + "\n" + process.stderr).strip()
    if not output:
        raise ValidationError(f"{label} query returned no evidence")
    return output


def _cmake_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        values[key] = value
    return values


def _source_snapshot(source_root: Path) -> tuple[str, bool]:
    list_process = subprocess.run(
        [
            "git", "-C", str(source_root), "ls-files", "-co",
            "--exclude-standard", "-z", "--", *SOURCE_SNAPSHOT_GIT_PATHS,
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    status_process = subprocess.run(
        [
            "git", "-C", str(source_root), "status", "--porcelain=v1",
            "--untracked-files=all", "--", *SOURCE_SNAPSHOT_GIT_PATHS,
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if list_process.returncode or status_process.returncode:
        raise ValidationError("source-scope snapshot queries failed")
    entries = sorted(
        set(item for item in list_process.stdout.split(b"\0") if item)
    )
    digest = hashlib.sha256()
    digest.update(b"volvoxai-native-benchmark-source-scope\0")
    for scope in SOURCE_SNAPSHOT_INCLUDED:
        digest.update(scope.encode("utf-8") + b"\0")
    for scope in SOURCE_SNAPSHOT_EXCLUDED:
        digest.update(b"exclude\0" + scope.encode("utf-8") + b"\0")
    for relative_bytes in entries:
        try:
            relative = relative_bytes.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise ValidationError("source scope has a non-UTF-8 path") from error
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValidationError("source scope returned an unsafe path")
        path = source_root / relative_path
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            digest.update(b"missing\0" + relative_bytes + b"\0")
            continue
        executable = b"x" if metadata.st_mode & 0o111 else b"-"
        if stat.S_ISREG(metadata.st_mode):
            payload_digest = bytes.fromhex(_sha256(path))
            kind = b"file"
        elif stat.S_ISLNK(metadata.st_mode):
            payload_digest = hashlib.sha256(
                os.readlink(path).encode("utf-8", "strict")
            ).digest()
            kind = b"symlink"
        else:
            raise ValidationError("source scope contains a non-file entry")
        digest.update(kind + b"\0" + executable + b"\0")
        digest.update(relative_bytes + b"\0" + payload_digest)
    return digest.hexdigest(), bool(status_process.stdout)


def _shader_compiler_source_evidence(source_root: Path) -> dict[str, Any]:
    compiler_root = source_root / "tools/metal_shader_compiler"
    cargo_toml = compiler_root / "Cargo.toml"
    cargo_lock = compiler_root / "Cargo.lock"
    source_files = sorted((compiler_root / "src").rglob("*.rs"))
    if not cargo_toml.is_file() or not cargo_lock.is_file() or not source_files:
        raise ValidationError("native shader compiler source is incomplete")
    manifest_text = cargo_toml.read_text(encoding="utf-8")
    lock_text = cargo_lock.read_text(encoding="utf-8")
    package = re.search(
        r"(?ms)^\[package\]\s*(.*?)(?=^\[|\Z)", manifest_text
    )
    version = (
        re.search(r'^version\s*=\s*"([^"]+)"\s*$', package.group(1), re.M)
        if package else None
    )
    naga_dependency = re.search(
        r'(?m)^naga\s*=\s*\{[^}]*version\s*=\s*"=([^"]+)"[^}]*\}\s*$',
        manifest_text,
    )
    locked_naga_versions = []
    for package_block in re.findall(
        r"(?ms)^\[\[package\]\]\s*(.*?)(?=^\[\[package\]\]|\Z)",
        lock_text,
    ):
        package_name = re.search(
            r'(?m)^name\s*=\s*"([^"]+)"\s*$', package_block
        )
        package_version = re.search(
            r'(?m)^version\s*=\s*"([^"]+)"\s*$', package_block
        )
        if package_name and package_name.group(1) == "naga" and package_version:
            locked_naga_versions.append(package_version.group(1))
    if (
        not version
        or not naga_dependency
        or locked_naga_versions != [naga_dependency.group(1)]
    ):
        raise ValidationError("shader compiler Cargo contract is malformed")
    digest = hashlib.sha256()
    for path in [cargo_toml, cargo_lock, *source_files]:
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return {
        "source": "current-locked-cargo-and-rust-source",
        "version": version.group(1),
        "nagaDependencyVersion": naga_dependency.group(1),
        "cargoLockSha256": _sha256(cargo_lock),
        "sourceSha256": digest.hexdigest(),
    }


def collect_host_evidence() -> dict[str, Any]:
    affinity_process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os;print(','.join(map(str,sorted(os.sched_getaffinity(0)))))",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        preexec_fn=_pin_cpu_zero,
        timeout=30,
    )
    if affinity_process.returncode != 0 or affinity_process.stdout.strip() != "0":
        raise ValidationError("child CPU affinity evidence is not exactly {0}")
    return {
        "operatingSystem": platform.system(),
        "benchmarkChildAffinity": [0],
    }


def collect_build_evidence(config: dict[str, Any]) -> dict[str, Any]:
    source_root = Path(config["build"]["sourceRoot"]).resolve()
    build_root = Path(config["build"]["buildDirectory"]).resolve()
    harness = Path(config["harness"]).resolve()
    native_shader_compiler = Path(
        config["build"]["nativeShaderCompiler"]
    ).resolve()
    naga = Path(config["build"]["naga"]).resolve()
    resolved_naga = shutil.which("naga")
    if not resolved_naga or Path(resolved_naga).resolve() != naga:
        raise ValidationError("configured Naga is not the active PATH executable")
    native_shader_compiler_sha256 = _sha256(native_shader_compiler)
    naga_sha256 = _sha256(naga)
    shader_source = _shader_compiler_source_evidence(source_root)
    cache_path = build_root / "CMakeCache.txt"
    commands_path = build_root / "compile_commands.json"
    cache = _cmake_cache(cache_path)
    required_cache = {
        "CMAKE_BUILD_TYPE": "Release",
        "VOLVOXAI_ENABLE_CUDA": "ON",
        "VOLVOXAI_ENABLE_VULKAN": "ON",
        "VOLVOXAI_ENABLE_OPENGL": "ON",
        "VOLVOXAI_CUDA_ARCH": "86",
        "VOLVOXAI_CUDA_FAST_FP32": "OFF",
    }
    if any(cache.get(key) != value for key, value in required_cache.items()):
        raise ValidationError("CMake cache is not the official Release GPU build")
    cmake_home = cache.get("CMAKE_HOME_DIRECTORY")
    if not cmake_home or Path(cmake_home).resolve() != source_root:
        raise ValidationError("CMake build is not bound to the selected sourceRoot")
    if harness != build_root / "native" / "native_dynamic_batch_benchmark":
        raise ValidationError("harness is not the exact native build target")
    try:
        commands = json.loads(commands_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"compile commands cannot be read: {error}") from error
    matches = [
        entry for entry in commands
        if isinstance(entry, dict)
        and isinstance(entry.get("file"), str)
        and Path(entry["file"]).resolve() ==
            source_root / "examples/native_dynamic_batch_benchmark/main.c"
        and isinstance(entry.get("directory"), str)
        and Path(entry["directory"]).resolve() == build_root / "native"
    ]
    if len(matches) != 1:
        raise ValidationError("harness compile command is absent or ambiguous")
    command_value = matches[0].get("arguments", matches[0].get("command"))
    if isinstance(command_value, list):
        compile_tokens = [str(token) for token in command_value]
    elif isinstance(command_value, str):
        import shlex
        compile_tokens = shlex.split(command_value)
    else:
        raise ValidationError("harness compile command has no argv")
    compiler = Path(cache.get("CMAKE_C_COMPILER", ""))
    nvcc = Path(cache.get("VOLVOXAI_NVCC_EXECUTABLE", ""))
    if not compiler.is_file() or not nvcc.is_file():
        raise ValidationError("configured C compiler/NVCC is unavailable")
    if (
        not compile_tokens
        or not Path(compile_tokens[0]).is_file()
        or Path(compile_tokens[0]).resolve() != compiler.resolve()
    ):
        raise ValidationError(
            "harness compile argv[0] differs from configured C compiler"
        )
    if "-O3" not in compile_tokens or "-DNDEBUG" not in compile_tokens:
        raise ValidationError("harness was not compiled as optimized NDEBUG")
    fp_contract = next(
        (
            token.split("=", 1)[1]
            for token in compile_tokens
            if token.startswith("-ffp-contract=")
        ),
        "toolchain-default",
    )
    fast_math = "-ffast-math" in compile_tokens
    shader_rule_path = build_root / "build.ninja"
    if not shader_rule_path.is_file():
        shader_rule_path = (
            build_root / "native/CMakeFiles/volvoxai_compile_shaders.dir/build.make"
        )
    if not shader_rule_path.is_file():
        raise ValidationError("generated shader target rule is unavailable")
    shader_rule_text = shader_rule_path.read_text(
        encoding="utf-8", errors="replace"
    )
    generated_build_files = [build_root / "build.ninja"]
    generated_build_files.extend(
        path for path in (build_root / "native").rglob("build.make")
    )
    generated_text = ""
    for path in generated_build_files:
        if path.is_file():
            generated_text += path.read_text(encoding="utf-8", errors="replace")
    configured_shader_compiler = re.compile(
        rf"(?:^|\s)VOLVOXAI_NATIVE_SHADER_COMPILER="
        rf"{re.escape(str(native_shader_compiler))}(?:\s|$)"
    )
    configured_shader_script = re.compile(
        rf"(?:^|\s)"
        rf"{re.escape(str(source_root / 'tools/compile_shaders.sh'))}"
        rf"(?:\s|$)"
    )
    shader_compile_commands = [
        line for line in shader_rule_text.splitlines()
        if configured_shader_script.search(line) is not None
        and configured_shader_compiler.search(line) is not None
    ]
    if not shader_compile_commands:
        raise ValidationError(
            "generated shader command did not use the configured compiler"
        )
    cuda_forward_tokens = (
        "native/src/backends/cuda_kernels.cu",
        "--ptx",
        "-O3",
        "--fmad=false",
        "--gpu-architecture=compute_86",
    )
    configured_nvcc = re.compile(
        rf"(?:^|\s){re.escape(str(nvcc))}(?:\s|$)"
    )

    def exact_command_token(line: str, token: str) -> bool:
        return re.search(
            rf"(?:^|\s){re.escape(token)}(?:\s|$)", line
        ) is not None

    cuda_forward_commands = [
        line for line in generated_text.splitlines()
        if configured_nvcc.search(line) is not None
        and all(exact_command_token(line, token)
                for token in cuda_forward_tokens)
    ]
    if (
        not cuda_forward_commands
        or any(exact_command_token(line, "--fmad=true")
               for line in generated_text.splitlines()
               if exact_command_token(
                   line, "native/src/backends/cuda_kernels.cu"))
    ):
        raise ValidationError(
            "generated CUDA forward command does not attest strict sm86 PTX"
        )
    compiler_version = _run_evidence(
        [str(compiler), "--version"], "C compiler"
    ).splitlines()[0]
    cmake_version = _run_evidence(["cmake", "--version"], "CMake").splitlines()[0]
    nvcc_output = _run_evidence([str(nvcc), "--version"], "NVCC")
    nvcc_lines = [line.strip() for line in nvcc_output.splitlines() if line.strip()]
    cuda_toolkit = next(
        (line for line in nvcc_lines if "release " in line), nvcc_lines[-1]
    )
    naga_version = _run_evidence([str(naga), "--version"], "Naga")
    if (
        len(naga_version.splitlines()) != 1
        or naga_version.strip() != shader_source["nagaDependencyVersion"]
    ):
        raise ValidationError("standalone Naga differs from the locked version")
    commit = _run_evidence(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        "source commit",
    ).splitlines()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValidationError("source commit is not a full SHA-1")
    source_snapshot, source_dirty = _source_snapshot(source_root)
    build_contract = {
        "configuration": "Release",
        "cudaArch": "sm86",
        "cudaFp32Contract": "strict-no-fma",
        "nvccFmad": False,
        "nvccFmadEvidence": "generated-build-command---fmad=false",
        "backends": ["cuda", "opengl", "vulkan"],
        "optimization": "O3",
        "ndebug": True,
        "hostFpContract": fp_contract,
        "fastMath": fast_math,
    }
    build_contract_sha256 = hashlib.sha256(
        json.dumps(
            build_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "configuration": "Release",
        "sourceCommit": commit,
        "sourceSnapshotSha256": source_snapshot,
        "sourceSnapshotScope": {
            "included": list(SOURCE_SNAPSHOT_INCLUDED),
            "excluded": list(SOURCE_SNAPSHOT_EXCLUDED),
            "algorithm": (
                "sha256-sorted-relative-path-executable-bit-content"
            ),
        },
        "sourceDirtyWithinSnapshotScope": source_dirty,
        "compilerVersion": compiler_version,
        "cmakeVersion": cmake_version,
        "cudaToolkitVersion": cuda_toolkit,
        "cudaArch": build_contract["cudaArch"],
        "cudaFp32Contract": build_contract["cudaFp32Contract"],
        "nvccFmad": build_contract["nvccFmad"],
        "nvccFmadEvidence": build_contract["nvccFmadEvidence"],
        "backends": build_contract["backends"],
        "optimization": build_contract["optimization"],
        "assertionsEnabled": False,
        "hostFpContract": fp_contract,
        "fastMath": fast_math,
        "buildContractSha256": build_contract_sha256,
        "harnessSha256": _sha256(harness),
        "shaderToolchain": {
            "nativeShaderCompiler": {
                "binarySha256": native_shader_compiler_sha256,
                "source": shader_source["source"],
                "version": shader_source["version"],
                "cargoLockSha256": shader_source["cargoLockSha256"],
                "sourceSha256": shader_source["sourceSha256"],
            },
            "naga": {
                "version": naga_version.strip(),
                "binarySha256": naga_sha256,
            },
            "cmakeCommandUsedConfiguredNativeCompiler": True,
        },
    }


def _parse_backend_identity(
    backend: str, stdout: str, stderr: str, expected_name: str,
    expected_driver: str,
) -> dict[str, Any]:
    combined = stdout + "\n" + stderr
    lowered = combined.lower()
    if any(token in lowered for token in ("llvmpipe", "swiftshader")):
        raise ValidationError("backend initialized a software renderer")
    if backend == "vulkan":
        marker = "Vulkan Compute initialized successfully! Device:"
        matches = re.findall(
            r"Vulkan Compute initialized successfully! Device: ([^;\r\n]+); "
            r"packed INT8 dot: (enabled|unavailable)",
            combined,
        )
        observations = [(name.strip(), packed) for name, packed in matches]
        if (
            not observations
            or combined.count(marker) != len(observations)
            or len(set(observations)) != 1
            or observations[0][0] != expected_name
        ):
            raise ValidationError("Vulkan init identity is absent or ambiguous")
        return {
            "api": "vulkan",
            "deviceName": expected_name,
            "packedInt8Dot": observations[0][1] == "enabled",
            "observationCount": len(observations),
        }
    elif backend == "cuda":
        marker = "[CUDA] device "
        matches = re.findall(
            r"\[CUDA\] device ([0-9]+): (.+?) "
            r"\(compute ([0-9]+\.[0-9]+)\)",
            combined,
        )
        observations = [
            (ordinal, name.strip(), capability)
            for ordinal, name, capability in matches
        ]
        if (
            not observations
            or combined.count(marker) != len(observations)
            or len(set(observations)) != 1
            or observations[0][0] != "0"
            or observations[0][1] != expected_name
        ):
            raise ValidationError("CUDA init identity is absent or ambiguous")
        if observations[0][2] != "8.6":
            raise ValidationError("CUDA selected device is not compute 8.6")
        return {
            "api": "cuda",
            "ordinal": 0,
            "deviceName": expected_name,
            "computeCapability": "8.6",
            "observationCount": len(observations),
        }
    else:
        marker = "OpenGL Compute initialized:"
        matches = re.findall(
            r"OpenGL Compute initialized: ([^/\r\n]+) / (.+?) / ([^\r\n]+)",
            combined,
        )
        observations = [tuple(item.strip() for item in match) for match in matches]
        if (
            not observations
            or combined.count(marker) != len(observations)
            or len(set(observations)) != 1
        ):
            raise ValidationError("OpenGL init identity is absent or ambiguous")
        vendor, renderer, version = observations[0]
        if vendor != "NVIDIA Corporation" or expected_name not in renderer:
            raise ValidationError("OpenGL renderer differs from expected device")
        if expected_driver not in version:
            raise ValidationError("OpenGL version omits the expected driver")
        api_version = version.split()[0]
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", api_version):
            raise ValidationError("OpenGL API version is malformed")
        return {
            "api": "opengl",
            "vendor": "NVIDIA Corporation",
            "deviceName": expected_name,
            "apiVersion": api_version,
            "driverVersion": expected_driver,
            "observationCount": len(observations),
        }


def _number_list(value: Any, label: str, count: int) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not (0 < float(item) < float("inf"))
            for item in value
        )
    ):
        raise ValidationError(f"{label} must contain {count} finite samples")
    return [float(item) for item in value]


def _timing_summary(
    samples: list[float], logical_lanes: int, physical_per_group: int
) -> dict[str, Any]:
    ordered = sorted(samples)
    count = len(ordered)
    total = sum(ordered)
    p95_index = max(0, (95 * count + 99) // 100 - 1)
    logical_requests = logical_lanes * count
    if total <= 0:
        raise ValidationError("timing sum must be positive")
    return {
        "minMs": ordered[0],
        "medianMs": statistics.median(ordered),
        "p95Ms": ordered[p95_index],
        "p95Definition": "nearest-rank-ceil(0.95*N)",
        "maxMs": ordered[-1],
        "sumMs": total,
        "logicalRequests": logical_requests,
        "physicalInvocations": physical_per_group * count,
        "aggregateLogicalRequestsPerSecond":
            logical_requests * 1000.0 / total,
    }


def add_derived_metrics(result: dict[str, Any]) -> None:
    lanes = result["logicalLanes"]
    direct_samples = [float(value) for value in result["direct"]["samplesMs"]]
    scheduled_samples = [
        float(value) for value in result["scheduled"]["samplesMs"]
    ]
    direct = _timing_summary(direct_samples, lanes, lanes)
    scheduled = _timing_summary(scheduled_samples, lanes, 1)
    if abs(direct["medianMs"] - float(result["direct"]["medianMs"])) > 1e-6:
        raise ValidationError("C direct median differs from raw samples")
    if abs(
        scheduled["medianMs"] - float(result["scheduled"]["medianMs"])
    ) > 1e-6:
        raise ValidationError("C scheduled median differs from raw samples")
    result["derived"] = {
        "direct": direct,
        "scheduled": scheduled,
        "sameBackendMedianSpeedup":
            direct["medianMs"] / scheduled["medianMs"]
            if scheduled["medianMs"] > 0 else None,
    }


def _artifact_role_batch_max(role: Any) -> int:
    role = _required_string(role, "artifactRole", token=True)
    match = re.search(r"-maxB([1-9][0-9]*)$", role)
    if not match:
        raise ValidationError("artifactRole lacks its declared maxB suffix")
    return int(match.group(1))


def validate_case_result(
    raw: Any, configured: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    result = _exact_keys(raw, CASE_RESULT_KEYS, "harness result")
    if (
        result["record"] != CASE_RECORD
        or result["caseId"] != configured["id"]
        or result["backend"] != configured["backend"]
        or result["inputMode"] != configured["inputMode"]
        or result["status"] != "pass"
    ):
        raise ValidationError("harness identity/status differs from case")
    lanes = configured["batchSize"]
    package_batch_max = _artifact_role_batch_max(configured["artifactRole"])
    if result["logicalLanes"] != lanes:
        raise ValidationError("harness logical lane count differs")
    if configured["inputMode"] == "receipt":
        lane_classes = len(configured["laneFiles"])
        if (
            result["uniqueInputClasses"] != lane_classes
            or result["fixtureReuse"] != (
                "none" if lanes == lane_classes else "cyclic"
            )
        ):
            raise ValidationError("receipt class reuse contract differs")
        synthetic_keys = SYNTHETIC_RECEIPT_KEYS
    elif configured["inputMode"] == "vqa-encoder":
        if (
            result["uniqueInputClasses"] != lanes
            or result["fixtureReuse"] != "none"
        ):
            raise ValidationError("encoder lanes are not distinct")
        synthetic_keys = SYNTHETIC_ENCODER_KEYS
    else:
        if (
            result["uniqueInputClasses"] != lanes
            or result["fixtureReuse"] != "none"
        ):
            raise ValidationError("decoder lanes are not distinct")
        synthetic_keys = SYNTHETIC_DECODER_KEYS
    synthetic = _exact_keys(result["syntheticFixture"], synthetic_keys, "syntheticFixture")
    if configured["inputMode"] == "receipt":
        if synthetic != {"kind": "private-f32-lanes", "seed": None}:
            raise ValidationError("receipt fixture descriptor differs")
    else:
        if (
            synthetic["kind"] != "component-only-graph-domain-minimum"
            or synthetic["seed"] != 20260821
            or synthetic["M"] != 211
        ):
            raise ValidationError("VQA graph-domain fixture descriptor differs")
        if configured["inputMode"] == "vqa-encoder" and synthetic["Q"] != 1:
            raise ValidationError("encoder graph-domain minimum must be Q=1")
        if configured["inputMode"] == "vqa-decoder" and (
            synthetic["P"] != 1 or synthetic["R"] != 2
        ):
            raise ValidationError("decoder graph-domain minimum must be P=1/R=2")
    if not isinstance(result["resolvedInputs"], list) or not result["resolvedInputs"]:
        raise ValidationError("resolvedInputs must be non-empty")
    for index, item in enumerate(result["resolvedInputs"]):
        resolved = _exact_keys(item, RESOLVED_INPUT_KEYS, f"resolvedInputs[{index}]")
        _required_string(resolved["name"], f"resolvedInputs[{index}].name")
        _required_int(resolved["dtype"], f"resolvedInputs[{index}].dtype", 1)
        if (
            not isinstance(resolved["shape"], list)
            or not resolved["shape"]
            or resolved["shape"][0] != 1
            or any(_required_int(dim, "resolved shape extent", 1) < 1 for dim in resolved["shape"])
        ):
            raise ValidationError("resolved input is not concrete B1")
    if configured["inputMode"] == "receipt":
        expected_resolved_inputs = [
            {"name": "input0", "dtype": 18, "shape": [1, 1, 320, 672]}
        ]
    elif configured["inputMode"] == "vqa-encoder":
        expected_resolved_inputs = [
            {"name": "input0", "dtype": 18, "shape": [1, 1, 320, 672]},
            {"name": "input1", "dtype": 16, "shape": [1, 1]},
            {"name": "input2", "dtype": 16, "shape": [1]},
            {"name": "input3", "dtype": 16, "shape": [1, 1]},
        ]
    else:
        expected_resolved_inputs = [
            {"name": "input0", "dtype": 16, "shape": [1, 1]},
            {"name": "input1", "dtype": 16, "shape": [1]},
            {"name": "input2", "dtype": 16, "shape": [1]},
            {"name": "input3", "dtype": 16, "shape": [1, 211]},
            {"name": "input4", "dtype": 16, "shape": [1, 1]},
            *(
                {"name": f"input{index}", "dtype": 18,
                 "shape": [1, 8, 211, 40]}
                for index in range(5, 13)
            ),
            *(
                {"name": f"input{index}", "dtype": 18,
                 "shape": [1, 8, 1, 40]}
                for index in range(13, 21)
            ),
        ]
    if result["resolvedInputs"] != expected_resolved_inputs:
        raise ValidationError("resolved input ABI differs from the exact fixture")
    runtime = _exact_keys(result["runtime"], RUNTIME_KEYS, "runtime")
    expected_arena = config["vulkanArenaMB"] if configured["backend"] == "vulkan" else None
    if runtime != {
        "executionMode": "scheduled",
        "cpuThreads": 1,
        "freshness": "ALL",
        "maxScheduledRequests": lanes,
        "maxScheduledInputBytes": 2 * 1024 * 1024 * 1024,
        "maxUnconsumedResults": lanes * 2,
        "maxUnconsumedResultBytes": 2 * 1024 * 1024 * 1024,
        "requestedMaxBatchDelayMs": config["maxBatchDelayMs"],
        "effectiveMaxBatchDelayMs": config["maxBatchDelayMs"],
        "vulkanArenaMB": expected_arena,
    }:
        raise ValidationError("runtime scheduling/arena contract differs")
    contract = _exact_keys(result["batchContract"], BATCH_CONTRACT_KEYS, "batchContract")
    if (
        contract["protocol"] != "provider-batch-contract/v1"
        or contract["axis"] != 0
        or not isinstance(contract["symbol"], str)
        or not contract["symbol"]
        or contract["min"] != 1
        or contract["max"] != package_batch_max
        or isinstance(contract["multiple"], bool)
        or not isinstance(contract["multiple"], int)
        or contract["multiple"] < 1
        or lanes % contract["multiple"]
        or not isinstance(contract["proofIdentity"], str)
        or not contract["proofIdentity"].startswith("typed-independent-batch-proof/v1")
    ):
        raise ValidationError("compiled batch contract differs")
    strict = _exact_keys(result["strictRoute"], STRICT_ROUTE_KEYS, "strictRoute")
    if (
        strict["routeAttested"] is not True
        or strict["provider"] != f"builtin:{configured['backend']}"
        or _required_int(strict["nodes"], "strictRoute.nodes", 1) < 1
        or strict["selected"] != strict["nodes"]
        or strict["fallback"] != 0
        or strict["missing"] != 0
        or strict["operatorFallbackEvidence"] != "operator=none"
    ):
        raise ValidationError("strict route evidence differs")
    measurement = _exact_keys(result["measurement"], MEASUREMENT_KEYS, "measurement")
    if (
        measurement["warmupGroups"] != config["warmupGroups"]
        or measurement["repeatGroups"] != config["repeatGroups"]
        or measurement["order"] != "alternating-direct-scheduled"
        or measurement["clock"] != "CLOCK_MONOTONIC"
        or "read-parity-release-untimed" not in measurement["timingBoundary"]
    ):
        raise ValidationError("measurement boundary differs")
    direct = _exact_keys(result["direct"], DIRECT_KEYS, "direct")
    scheduled = _exact_keys(result["scheduled"], SCHEDULED_KEYS, "scheduled")
    _number_list(direct["samplesMs"], "direct.samplesMs", config["repeatGroups"])
    _number_list(scheduled["samplesMs"], "scheduled.samplesMs", config["repeatGroups"])
    if (
        direct["logicalLanes"] != lanes
        or direct["physicalBatchSize"] != 1
        or direct["trueBackendInvocationsPerGroup"] != lanes
        or scheduled["logicalLanes"] != lanes
        or scheduled["physicalBatchSize"] != lanes
        or scheduled["trueBackendInvocationsPerGroup"] != 1
        or scheduled["allLanesSharedPhysicalExecutionId"] is not True
        or scheduled["physicalExecutionIdChangesAcrossGroups"] is not True
        or scheduled["physicalExecutionIdScope"] !=
            "process-local-runtime-counter"
    ):
        raise ValidationError("direct/scheduled physical execution proof differs")
    physical_ids = scheduled["physicalExecutionIds"]
    if (
        not isinstance(physical_ids, list)
        or len(physical_ids) != config["repeatGroups"]
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in physical_ids
        )
        or len(set(physical_ids)) != len(physical_ids)
    ):
        raise ValidationError("measured physical execution IDs are invalid")
    for label, value in (("direct.medianMs", direct["medianMs"]), ("scheduled.medianMs", scheduled["medianMs"])):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 < float(value) < float("inf")):
            raise ValidationError(f"{label} must be finite and positive")
    parity = _exact_keys(result["parity"], PARITY_KEYS, "parity")
    outputs_per_lane = (
        1 if configured["inputMode"] == "receipt"
        else 12 if configured["inputMode"] == "vqa-encoder"
        else 10
    )
    elements_per_lane = (
        176 if configured["inputMode"] == "receipt"
        else 607900 if configured["inputMode"] == "vqa-encoder"
        else 6658
    )
    expected_output_tensors = (
        outputs_per_lane * lanes *
        (config["warmupGroups"] + config["repeatGroups"])
    )
    expected_compared_elements = (
        elements_per_lane * lanes *
        (config["warmupGroups"] + config["repeatGroups"])
    )
    if (
        parity["status"] != "pass"
        or parity["absoluteTolerance"] != 0.0001
        or parity["relativeTolerance"] != 0.0001
        or parity["combinedAllclose"] is not True
        or parity["nonfiniteRejected"] is not True
        or parity["integerOutputsByteExact"] is not True
        or parity["receiptDecodedEquality"] is not (
            True if configured["inputMode"] == "receipt" else None
        )
        or parity["comparedElements"] != expected_compared_elements
        or parity["outputTensors"] != expected_output_tensors
    ):
        raise ValidationError("lane parity contract differs")
    for label in ("maxAbs", "maxRel"):
        value = parity[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 <= float(value) < float("inf")):
            raise ValidationError(f"parity.{label} must be finite")
    evidence = _exact_keys(result["publicEvidence"], PUBLIC_EVIDENCE_KEYS, "publicEvidence")
    expected_vulkan = configured["backend"] == "vulkan"
    expected_cuda = configured["backend"] == "cuda"
    if {
        key: evidence[key]
        for key in (
            "source",
            "compiledContractExact",
            "scheduledLaneTokensExact",
            "directBatchTokensAbsent",
            "backendExecutionProofExact",
        )
    } != {
        "source": "VxReport.route_evidence",
        "compiledContractExact": True,
        "scheduledLaneTokensExact": True,
        "directBatchTokensAbsent": True,
        "backendExecutionProofExact": True,
    }:
        raise ValidationError("public evidence gate differs")
    expected_device_local = True if expected_vulkan else None
    expected_arena_bytes = (
        config["vulkanArenaMB"] * 1024 * 1024 if expected_vulkan else None
    )
    expected_staging_bytes = 32 * 1024 * 1024 if expected_vulkan else None
    expected_cuda_replay = True if expected_cuda else None
    if (
        evidence["vulkanComputeDeviceLocal"] is not expected_device_local
        or evidence["vulkanArenaBytes"] != expected_arena_bytes
        or evidence["vulkanStagingBytes"] != expected_staging_bytes
        or (
            expected_vulkan and
            not isinstance(evidence["vulkanStagingHostCoherent"], bool)
        )
        or (
            not expected_vulkan and
            evidence["vulkanStagingHostCoherent"] is not None
        )
        or evidence["cudaGraphReplayMeasured"] is not expected_cuda_replay
    ):
        raise ValidationError("backend execution evidence differs")
    return result


def run_case(
    config: dict[str, Any], case: dict[str, Any], observed_device: dict[str, Any]
) -> dict[str, Any]:
    harness = str(Path(config["harness"]).resolve())
    artifact_evidence = verify_manifest(
        config["suite"],
        case,
        Path(case["manifest"]).resolve(),
        Path(case["graph"]).resolve(),
        [Path(path).resolve() for path in case["weights"]],
    )
    if artifact_evidence["packageBatchMax"] != _artifact_role_batch_max(
        case["artifactRole"]
    ):
        raise ValidationError(
            f"case {case['id']} artifactRole/package max changed"
        )
    command = [
        harness,
        "--case-id",
        case["id"],
        "--graph",
        str(Path(case["graph"]).resolve()),
    ]
    for weight in case["weights"]:
        command.extend(("--weight", str(Path(weight).resolve())))
    command.extend(
        (
            "--backend",
            case["backend"],
            "--input-mode",
            case["inputMode"],
            "--batch-size",
            str(case["batchSize"]),
            "--warmup",
            str(config["warmupGroups"]),
            "--repeat",
            str(config["repeatGroups"]),
            "--max-batch-delay-ms",
            str(config["maxBatchDelayMs"]),
        )
    )
    for lane in case["laneFiles"]:
        command.extend(("--lane-file", str(Path(lane).resolve())))
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    if case["backend"] == "vulkan":
        environment["VOLVOX_VULKAN_MB"] = str(config["vulkanArenaMB"])
        command.extend(("--vulkan-arena-mb", str(config["vulkanArenaMB"])))
    if case["backend"] == "cuda":
        configured_device = environment.get("VOLVOXAI_CUDA_DEVICE")
        if configured_device not in (None, "0"):
            raise ValidationError(
                "VOLVOXAI_CUDA_DEVICE must be unset or exactly 0"
            )
        environment["VOLVOXAI_CUDA_DEVICE"] = "0"
        command.extend(("--cuda-device", "0"))
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        preexec_fn=_pin_cpu_zero,
        timeout=60 * 60,
    )
    if process.returncode != 0:
        raise ValidationError(
            f"case {case['id']} failed with exit {process.returncode}: "
            f"{process.stderr[-1000:]}"
        )
    artifact_evidence_after = verify_manifest(
        config["suite"],
        case,
        Path(case["manifest"]).resolve(),
        Path(case["graph"]).resolve(),
        [Path(path).resolve() for path in case["weights"]],
    )
    if artifact_evidence_after != artifact_evidence:
        raise ValidationError(
            f"case {case['id']} package bytes changed during execution"
        )
    json_lines = [
        line.strip() for line in process.stdout.splitlines()
        if line.lstrip().startswith("{")
    ]
    if len(json_lines) != 1:
        raise ValidationError(f"case {case['id']} emitted ambiguous JSON")
    backend_identity = _parse_backend_identity(
        case["backend"],
        process.stdout,
        process.stderr,
        observed_device["name"],
        observed_device["driverVersion"],
    )
    try:
        raw = json.loads(json_lines[0])
    except json.JSONDecodeError as error:
        raise ValidationError(f"case {case['id']} emitted malformed JSON") from error
    result = validate_case_result(raw, case, config)
    result.pop("record")
    add_derived_metrics(result)
    result["artifactRole"] = case["artifactRole"]
    result["precision"] = case["precision"]
    result["artifacts"] = artifact_evidence
    result["deviceEvidence"] = {
        "backendInitMatchedExpectedDevice": True,
        "nvidiaSmiOrdinal0NameDriverMemoryMatchedExpected": True,
        "backendIdentity": backend_identity,
    }
    return result


def privacy_check(report: dict[str, Any], config: dict[str, Any]) -> None:
    serialized = json.dumps(report, sort_keys=True, separators=(",", ":"))
    private_values = [
        str(Path(config["harness"]).resolve()),
        str(Path(config["build"]["sourceRoot"]).resolve()),
        str(Path(config["build"]["buildDirectory"]).resolve()),
        str(Path(config["build"]["nativeShaderCompiler"]).resolve()),
        str(Path(config["build"]["naga"]).resolve()),
        *(
            str(Path(case[field]).resolve())
            for case in config["cases"]
            for field in ("manifest", "graph")
        ),
        *(
            str(Path(path).resolve())
            for case in config["cases"]
            for path in [*case["weights"], *case["laneFiles"]]
        ),
    ]
    if any(value and value in serialized for value in private_values):
        raise ValidationError("report contains a private filesystem path")
    pending: list[Any] = [report]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            if re.search(r"(?:^|[\s=:])/(?!/)[^\s,;]+", value) or re.search(
                r"(?:^|[\s=:])[A-Za-z]:[\\/][^\s,;]+", value
            ):
                raise ValidationError("report contains a filesystem path")
            if re.search(
                r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@"
                r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])",
                value,
            ):
                raise ValidationError("report contains an email address")
    lowered = serialized.lower()
    for forbidden in (
        '"uuid"',
        '"pci"',
        "pcie",
        '"phone"',
        '"street"',
        '"decodedvalue"',
        '"inputdigest"',
        '"outputdigest"',
    ):
        if forbidden in lowered:
            raise ValidationError(f"report contains forbidden field/value {forbidden}")


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    verify_clean_environment()
    host_evidence = collect_host_evidence()
    observed_device = query_device(config["device"])
    build_evidence = collect_build_evidence(config)
    receipt_lane_identity = _receipt_lane_identity(config)
    cases = []
    for configured_case in config["cases"]:
        _require_receipt_lane_identity(config, receipt_lane_identity)
        cases.append(run_case(config, configured_case, observed_device))
        _require_receipt_lane_identity(config, receipt_lane_identity)
    source_snapshot_after, _ = _source_snapshot(
        Path(config["build"]["sourceRoot"]).resolve()
    )
    if source_snapshot_after != build_evidence["sourceSnapshotSha256"]:
        raise ValidationError("build-relevant source changed during measurement")
    if _sha256(Path(config["harness"]).resolve()) != build_evidence[
        "harnessSha256"
    ]:
        raise ValidationError("benchmark harness binary changed during measurement")
    shader_toolchain = build_evidence["shaderToolchain"]
    if (
        _sha256(Path(config["build"]["nativeShaderCompiler"]).resolve())
        != shader_toolchain["nativeShaderCompiler"]["binarySha256"]
        or _sha256(Path(config["build"]["naga"]).resolve())
        != shader_toolchain["naga"]["binarySha256"]
    ):
        raise ValidationError("shader toolchain binary changed during measurement")
    role_artifacts: dict[str, dict[str, Any]] = {}
    for case in cases:
        previous = role_artifacts.setdefault(
            case["artifactRole"], case["artifacts"]
        )
        if previous != case["artifacts"]:
            raise ValidationError(
                "one artifact role changed bytes across backend cases"
            )
    report = {
        "schema": OUTPUT_SCHEMA,
        "suite": config["suite"],
        "measuredAtUtc": config["measuredAtUtc"],
        "canonicalJson": True,
        "comparisonPolicy": "same-package-dynamic-batch-sweep-only",
        "build": build_evidence,
        "device": {
            **observed_device,
            "identityEvidence":
                "backend-init-log+nvidia-smi-ordinal0-name-driver-memory",
        },
        "host": host_evidence,
        "matrix": {
            "batchSizes": sorted({
                case["batchSize"] for case in config["cases"]
            }),
            "warmupGroups": config["warmupGroups"],
            "repeatGroups": config["repeatGroups"],
            "maxBatchDelayMs": config["maxBatchDelayMs"],
            "vulkanArenaMB": config["vulkanArenaMB"],
            "cpuAffinity": [0],
            "childEnvironment": {
                "LC_ALL": "C",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "performanceOverrides": "unset",
                "cudaDevice": 0,
            },
            "privateReceiptLaneBytesStableAcrossMatrix": (
                True if config["suite"] == "receipt-digit-reader" else None
            ),
        },
        "privacy": {
            "privatePathsPresent": False,
            "rawBackendLogsPresent": False,
            "receiptInputDigestsPresent": False,
            "decodedReceiptValuesPresent": False,
        },
        "cases": cases,
    }
    privacy_check(report, config)
    return report


def write_canonical(path: Path, report: dict[str, Any]) -> None:
    encoded = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = build_report(config)
        write_canonical(args.output, report)
    except (OSError, subprocess.SubprocessError, ValidationError) as error:
        print(f"native dynamic-batch matrix: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
