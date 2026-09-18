#!/usr/bin/env python3
"""Generate dependency-free projections of public volvoxai.proto enums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYNURANG_MANIFESTS = (
    ROOT / "runtime/generated/c/.synurang-c.manifest.json",
    ROOT / "runtime/generated/c/inference/.synurang-c-inference.manifest.json",
    ROOT / "runtime/generated/typescript/.synurang-typescript.manifest.json",
    ROOT / "runtime/generated/typescript/inference/.synurang-typescript-inference.manifest.json",
)


@dataclass(frozen=True)
class EnumSpec:
    proto_name: str
    proto_prefix: str
    c_type: str
    c_prefix: str
    profile: str


SPECS = (
    EnumSpec("BufferAccessMode", "BUFFER_ACCESS_MODE_", "VxBufferAccessMode", "VX_BUFFER_ACCESS_", "base"),
    EnumSpec("ParameterExportMode", "PARAMETER_EXPORT_MODE_", "VxParameterExportMode", "VX_PARAMETER_EXPORT_", "full"),
    EnumSpec("NativeResourceKind", "NATIVE_RESOURCE_KIND_", "VxNativeBufferKind", "VX_NATIVE_BUFFER_", "base"),
    EnumSpec("ApiEffect", "API_EFFECT_", "VxApiEffect", "VX_API_EFFECT_", "base"),
    EnumSpec("ApiRuleKind", "API_RULE_KIND_", "VxApiRuleKind", "VX_API_RULE_", "base"),
    EnumSpec("InputValidationCode", "INPUT_VALIDATION_CODE_", "VxInputValidationCode", "VX_INPUT_", "base"),
    EnumSpec("BatchWorkKind", "BATCH_WORK_KIND_", "VxBatchWorkKind", "VX_BATCH_WORK_KIND_", "base"),
    EnumSpec("BatchWorkState", "BATCH_WORK_STATE_", "VxBatchWorkState", "VX_BATCH_WORK_STATE_", "base"),
    EnumSpec("BatchQueuePolicy", "BATCH_QUEUE_POLICY_", "VxBatchQueuePolicy", "VX_BATCH_QUEUE_POLICY_", "base"),
    EnumSpec("DecodeCachePolicy", "DECODE_CACHE_POLICY_", "VxDecodeCachePolicy", "VX_DECODE_CACHE_POLICY_", "base"),
    EnumSpec("TokenizationMode", "TOKENIZATION_MODE_", "VxTokenizationMode", "VX_TOKENIZATION_", "base"),
    EnumSpec("NativeStatus", "NATIVE_STATUS_", "VxStatus", "VX_STATUS_", "mixed"),
    EnumSpec("OperationCode", "OPERATION_CODE_", "VxOperationCode", "VX_CODE_", "mixed"),
    EnumSpec("OperationStage", "OPERATION_STAGE_", "VxStage", "VX_STAGE_", "mixed"),
    EnumSpec("DataType", "DATA_TYPE_", "VxDataType", "VX_DTYPE_", "base"),
    EnumSpec("ResultState", "RESULT_STATE_", "VxResultState", "VX_RESULT_STATE_", "base"),
    EnumSpec(
        "BackendPolicyMode",
        "BACKEND_POLICY_MODE_",
        "VxBackendPolicyMode",
        "VX_BACKEND_",
        "base",
    ),
    EnumSpec(
        "OperatorFallback",
        "OPERATOR_FALLBACK_",
        "VxOperatorFallback",
        "VX_OPERATOR_FALLBACK_",
        "base",
    ),
    EnumSpec(
        "MemoryLocation",
        "MEMORY_LOCATION_",
        "VxMemoryLocation",
        "VX_MEMORY_",
        "base",
    ),
    EnumSpec(
        "DecodeRowMode",
        "DECODE_ROW_MODE_",
        "VxDecodeRowMode",
        "VX_DECODE_ROW_",
        "base",
    ),
    EnumSpec(
        "ExecutionMode",
        "EXECUTION_MODE_",
        "VxExecutionMode",
        "VX_EXECUTION_MODE_",
        "base",
    ),
    EnumSpec(
        "MemorySpace",
        "MEMORY_SPACE_",
        "VxMemorySpace",
        "VX_MEMORY_SPACE_",
        "base",
    ),
    EnumSpec(
        "MemoryOwnerKind",
        "MEMORY_OWNER_KIND_",
        "VxMemoryOwnerKind",
        "VX_MEMORY_OWNER_",
        "mixed",
    ),
    EnumSpec(
        "MemoryResourceRole",
        "MEMORY_RESOURCE_ROLE_",
        "VxMemoryResourceRole",
        "VX_MEMORY_RESOURCE_ROLE_",
        "mixed",
    ),
    EnumSpec(
        "MemoryBackingRelation",
        "MEMORY_BACKING_RELATION_",
        "VxMemoryBackingRelation",
        "VX_MEMORY_BACKING_RELATION_",
        "base",
    ),
    EnumSpec(
        "MemoryBoundKind",
        "MEMORY_BOUND_KIND_",
        "VxMemoryBoundKind",
        "VX_MEMORY_BOUND_",
        "base",
    ),
    EnumSpec(
        "MemorySnapshotPoint",
        "MEMORY_SNAPSHOT_POINT_",
        "VxMemorySnapshotPoint",
        "VX_MEMORY_SNAPSHOT_",
        "base",
    ),
    EnumSpec(
        "MemoryMetric",
        "MEMORY_METRIC_",
        "VxMemoryMetric",
        "VX_MEMORY_METRIC_",
        "base",
    ),
    EnumSpec(
        "MemoryEvidenceSource",
        "MEMORY_EVIDENCE_SOURCE_",
        "VxMemoryEvidenceSource",
        "VX_MEMORY_EVIDENCE_SOURCE_",
        "base",
    ),
    EnumSpec(
        "MemoryValueRelation",
        "MEMORY_VALUE_RELATION_",
        "VxMemoryValueRelation",
        "VX_MEMORY_VALUE_RELATION_",
        "base",
    ),
    EnumSpec(
        "MemoryTemporalCoverage",
        "MEMORY_TEMPORAL_COVERAGE_",
        "VxMemoryTemporalCoverage",
        "VX_MEMORY_TEMPORAL_COVERAGE_",
        "base",
    ),
    EnumSpec(
        "MemoryEnvelopeKind",
        "MEMORY_ENVELOPE_KIND_",
        "VxMemoryEnvelopeKind",
        "VX_MEMORY_ENVELOPE_",
        "base",
    ),
    EnumSpec(
        "MemoryInventoryKind",
        "MEMORY_INVENTORY_KIND_",
        "VxMemoryInventoryKind",
        "VX_MEMORY_INVENTORY_",
        "base",
    ),
    EnumSpec(
        "BuildProfile",
        "BUILD_PROFILE_",
        "VxBuildProfile",
        "VX_BUILD_PROFILE_",
        "base",
    ),
    EnumSpec(
        "TransportProfile",
        "TRANSPORT_PROFILE_",
        "VxTransportProfile",
        "VX_TRANSPORT_PROFILE_",
        "base",
    ),
    EnumSpec(
        "DimensionKind",
        "DIMENSION_KIND_",
        "VxDimensionKind",
        "VX_DIMENSION_",
        "base",
    ),
    EnumSpec(
        "CandidateOutcome",
        "CANDIDATE_OUTCOME_",
        "VxCandidateOutcome",
        "VX_CANDIDATE_OUTCOME_",
        "base",
    ),
    EnumSpec(
        "ShapePlanOrigin",
        "SHAPE_PLAN_ORIGIN_",
        "VxShapePlanOrigin",
        "VX_SHAPE_PLAN_ORIGIN_",
        "base",
    ),
    EnumSpec(
        "DecodeMode",
        "DECODE_MODE_",
        "VxDecodeMode",
        "VX_DECODE_MODE_",
        "base",
    ),
    EnumSpec(
        "RequestState",
        "REQUEST_STATE_",
        "VxRequestState",
        "VX_REQUEST_STATE_",
        "base",
    ),
    EnumSpec(
        "RequestFreshness",
        "REQUEST_FRESHNESS_",
        "VxRequestFreshness",
        "VX_REQUEST_FRESHNESS_",
        "base",
    ),
    EnumSpec(
        "GraphPlanSourceKind",
        "GRAPH_PLAN_SOURCE_KIND_",
        "VxGraphPlanSourceKind",
        "VX_GRAPH_PLAN_SOURCE_",
        "base",
    ),
    EnumSpec(
        "GraphTensorKind",
        "GRAPH_TENSOR_KIND_",
        "VxGraphTensorKind",
        "VX_GRAPH_TENSOR_",
        "base",
    ),
    EnumSpec(
        "ShapeDiagnosticCode",
        "SHAPE_DIAGNOSTIC_CODE_",
        "VxShapeDiagnosticCode",
        "VX_SHAPE_DIAGNOSTIC_",
        "base",
    ),
    EnumSpec(
        "ShapeDomainProofKind",
        "SHAPE_DOMAIN_PROOF_KIND_",
        "VxShapeDomainProofKind",
        "VX_SHAPE_DOMAIN_PROOF_",
        "base",
    ),
    EnumSpec(
        "ShapeDomainFact",
        "SHAPE_DOMAIN_FACT_",
        "VxShapeDomainFact",
        "VX_SHAPE_DOMAIN_FACT_",
        "base",
    ),
    EnumSpec(
        "IndependentBatchRefusalReason",
        "INDEPENDENT_BATCH_REFUSAL_REASON_",
        "VxIndependentBatchRefusalReason",
        "VX_INDEPENDENT_BATCH_REFUSAL_",
        "base",
    ),
    EnumSpec(
        "CallIntentKind",
        "CALL_INTENT_KIND_",
        "VxCallIntentKind",
        "VX_CALL_INTENT_",
        "base",
    ),
    EnumSpec(
        "NodeSelection",
        "NODE_SELECTION_",
        "VxNodeSelection",
        "VX_NODE_SELECTION_",
        "base",
    ),
    EnumSpec(
        "SafetensorsDiagnosticCode",
        "SAFETENSORS_DIAGNOSTIC_CODE_",
        "VxSafetensorsDiagnosticCode",
        "VX_SAFETENSORS_DIAGNOSTIC_",
        "base",
    ),
    EnumSpec(
        "SafetensorsDiagnosticSection",
        "SAFETENSORS_DIAGNOSTIC_SECTION_",
        "VxSafetensorsDiagnosticSection",
        "VX_SAFETENSORS_DIAGNOSTIC_SECTION_",
        "base",
    ),
    EnumSpec(
        "TrainingOptimizerKind",
        "TRAINING_OPTIMIZER_KIND_",
        "VxOptimizerKind",
        "VX_OPTIMIZER_",
        "full",
    ),
    EnumSpec("PtqMode", "PTQ_MODE_", "VxPTQMode", "VX_PTQ_MODE_", "full"),
    EnumSpec("PtqScheme", "PTQ_SCHEME_", "VxPTQScheme", "VX_PTQ_SCHEME_", "full"),
    EnumSpec(
        "PtqLayerKind",
        "PTQ_LAYER_KIND_",
        "VxPTQLayerKind",
        "VX_PTQ_LAYER_",
        "full",
    ),
)

# The one profile boundary. The inference profile consumes models; it does not
# author them, train them or quantize them. Every generator and checker that
# projects a profile reads this, so the boundary cannot be spelled twice.
INFERENCE_SERVICES = (
    "VxPlatformService",
    "VxTextService",
    "VxInferenceService",
    "VxBufferService",
    "VxSchedulerService",
)
FULL_ONLY_SERVICES = frozenset((
    "VxPlanningService",
    "VxTrainingService",
    "VxQuantizationService",
))

FULL_ONLY_OPERATION_CODES = frozenset((
    "OPERATION_CODE_ACCUMULATION_PENDING",
    "OPERATION_CODE_CALIBRATION_EXECUTION_FAILED",
    "OPERATION_CODE_CHECKPOINT_EXPORT_FAILED",
    "OPERATION_CODE_COMMIT_FAILED",
    "OPERATION_CODE_DUPLICATE_EXPORT_PATH",
    "OPERATION_CODE_EXPORT_FAILED",
    "OPERATION_CODE_EXPORT_MISMATCH",
    "OPERATION_CODE_INPUT_QUERY_FAILED",
    "OPERATION_CODE_INVALID_CALIBRATION_SAMPLE",
    "OPERATION_CODE_INVALID_CHECKPOINT",
    "OPERATION_CODE_INVALID_COMMIT",
    "OPERATION_CODE_INVALID_EXPORT",
    "OPERATION_CODE_INVALID_INPUT_QUERY",
    "OPERATION_CODE_INVALID_PTQ_COVERAGE_QUERY",
    "OPERATION_CODE_INVALID_PTQ_INFO",
    "OPERATION_CODE_INVALID_PTQ_OUTPUT_PATHS",
    "OPERATION_CODE_INVALID_PTQ_PLAN",
    "OPERATION_CODE_INVALID_PTQ_PROFILE_QUERY",
    "OPERATION_CODE_INVALID_PTQ_SPEC",
    "OPERATION_CODE_INVALID_PTQ_TENSOR_QUERY",
    "OPERATION_CODE_INVALID_SHAPE_OPTIONS",
    "OPERATION_CODE_INVALID_TEMPLATE_GRAPH",
    "OPERATION_CODE_INVALID_TRAINER_OPTIONS",
    "OPERATION_CODE_INVALID_TRAINING_BACKEND",
    "OPERATION_CODE_INVALID_TRAIN_STEP",
    "OPERATION_CODE_NO_APPLIED_UPDATE",
    "OPERATION_CODE_PENDING",
    "OPERATION_CODE_PTQ_COVERAGE_BUFFER_TOO_SMALL",
    "OPERATION_CODE_PTQ_COVERAGE_QUERY_FAILED",
    "OPERATION_CODE_PTQ_INSPECT_FAILED",
    "OPERATION_CODE_PTQ_PACKAGE_WRITE_FAILED",
    "OPERATION_CODE_PTQ_PLAN_CREATE_FAILED",
    "OPERATION_CODE_PTQ_PROFILE_COVERAGE_REQUIRED",
    "OPERATION_CODE_PTQ_PROFILE_NOT_FOUND",
    "OPERATION_CODE_PTQ_PROFILE_QUERY_FAILED",
    "OPERATION_CODE_PTQ_SINGLE_SHARD_REQUIRED",
    "OPERATION_CODE_PTQ_TENSOR_NOT_FOUND",
    "OPERATION_CODE_PTQ_TENSOR_QUERY_FAILED",
    "OPERATION_CODE_PTQ_WRITE_FAILED",
    "OPERATION_CODE_ROLLBACK_FAILED",
    "OPERATION_CODE_TEMPLATE_SNAPSHOT_FAILED",
    "OPERATION_CODE_TRAINER_CREATE_FAILED",
    "OPERATION_CODE_TRAINER_POISONED",
    "OPERATION_CODE_TRAINER_STATE_FAILED",
    "OPERATION_CODE_TRAINING_BACKEND_UNAVAILABLE",
    "OPERATION_CODE_TRAINING_BACKEND_UNSUPPORTED",
    "OPERATION_CODE_TRAINING_GRAPH_UNSUPPORTED",
    "OPERATION_CODE_TRAINING_PREFLIGHT_FAILED",
    "OPERATION_CODE_TRAIN_STEP_FAILED",
    "OPERATION_CODE_TRAIN_STEP_NOT_FOUND",
))

RUNTIME_DTYPE_PROTO_NAMES = (
    "DATA_TYPE_F32",
    "DATA_TYPE_I8",
    "DATA_TYPE_U8",
    "DATA_TYPE_I32",
)

def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def parse_enums(source: str) -> dict[str, tuple[tuple[str, int], ...]]:
    clean = strip_comments(source)
    declared = set(re.findall(r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", clean))
    classified = {spec.proto_name for spec in SPECS}
    unclassified = sorted(declared - classified)
    if unclassified:
        raise ValueError(
            "unclassified protobuf enum(s): " + ", ".join(unclassified)
        )
    missing = sorted(classified - declared)
    if missing:
        raise ValueError("missing enum(s): " + ", ".join(missing))
    parsed: dict[str, tuple[tuple[str, int], ...]] = {}
    for spec in SPECS:
        match = re.search(
            rf"\benum\s+{re.escape(spec.proto_name)}\s*\{{(.*?)\}}",
            clean,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(f"missing enum {spec.proto_name}")
        entries: list[tuple[str, int]] = []
        for statement in match.group(1).split(";"):
            statement = statement.strip()
            if not statement:
                continue
            entry = re.fullmatch(r"([A-Z][A-Z0-9_]*)\s*=\s*(-?[0-9]+)", statement)
            if entry is None:
                raise ValueError(
                    f"unsupported declaration in enum {spec.proto_name}: {statement!r}"
                )
            name = entry.group(1)
            if not name.startswith(spec.proto_prefix):
                raise ValueError(
                    f"{spec.proto_name} value {name} must start with {spec.proto_prefix}"
                )
            entries.append((name, int(entry.group(2))))
        if not entries or entries[0][1] != 0:
            raise ValueError(f"enum {spec.proto_name} must have a real zero-valued default")
        if len({name for name, _ in entries}) != len(entries):
            raise ValueError(f"enum {spec.proto_name} has duplicate names")
        if len({value for _, value in entries}) != len(entries):
            raise ValueError(f"enum {spec.proto_name} has duplicate numeric values")
        parsed[spec.proto_name] = tuple(entries)
    return parsed


def camel_case(suffix: str) -> str:
    parts = suffix.lower().split("_")
    return parts[0].capitalize() + "".join(part.capitalize() for part in parts[1:])


def c_entries(
    spec: EnumSpec,
    entries: tuple[tuple[str, int], ...],
    predicate,
) -> list[str]:
    selected = [
        (spec.c_prefix + name.removeprefix(spec.proto_prefix), value)
        for name, value in entries
        if predicate(name)
    ]
    return [
        f"    {name} = {value}{',' if index + 1 < len(selected) else ''}"
        for index, (name, value) in enumerate(selected)
    ]


def generated_banner(proto_hash: str, marker: str) -> str:
    return (
        f"{marker} DO NOT EDIT: generated by tools/generate_proto_enums.py.\n"
        f"{marker} proto/volvoxai.proto SHA-256: {proto_hash}\n"
    )


def is_full_only_entry(spec: EnumSpec, name: str) -> bool:
    if spec.profile == "full":
        return True
    if spec.profile == "base":
        return False
    if spec.proto_name == "OperationCode":
        return name in FULL_ONLY_OPERATION_CODES
    if spec.proto_name == "NativeStatus":
        return False
    if spec.proto_name == "OperationStage":
        return "_TRAINER_" in name or "_PTQ_" in name
    if spec.proto_name == "MemoryOwnerKind":
        return name.endswith("_TRAINER") or name.endswith("_PTQ_PLAN")
    if spec.proto_name == "MemoryResourceRole":
        return "_TRAINING_" in name or "_PTQ_" in name
    raise ValueError(f"mixed enum {spec.proto_name} has no value classifier")


def render_c_base(
    enums: dict[str, tuple[tuple[str, int], ...]], proto_hash: str
) -> bytes:
    sections: list[str] = []
    for spec in SPECS:
        if spec.profile == "full":
            continue
        predicate = lambda name, current=spec: not is_full_only_entry(current, name)
        entries = c_entries(spec, enums[spec.proto_name], predicate)
        sections.append(
            f"typedef int32_t {spec.c_type};\n"
            "enum {\n"
            + "\n".join(entries)
            + "\n};"
        )
    content = (
        generated_banner(proto_hash, "/*").replace("\n", " */\n")
        + "#ifndef VOLVOXAI_ENUMS_H\n#define VOLVOXAI_ENUMS_H\n\n"
        + "#include <stdint.h>\n\n"
        + "\n\n".join(sections)
        + "\n\n#endif /* VOLVOXAI_ENUMS_H */\n"
    )
    return content.encode()


def render_c_full(
    enums: dict[str, tuple[tuple[str, int], ...]], proto_hash: str
) -> bytes:
    sections: list[str] = []
    for spec in SPECS:
        selected = tuple(
            entry
            for entry in enums[spec.proto_name]
            if is_full_only_entry(spec, entry[0])
        )
        if not selected:
            continue
        declaration = f"typedef int32_t {spec.c_type};\n" if spec.profile == "full" else ""
        sections.append(
            declaration
            + "enum {\n"
            + "\n".join(c_entries(spec, selected, lambda _name: True))
            + "\n};"
        )
    content = (
        generated_banner(proto_hash, "/*").replace("\n", " */\n")
        + "#ifndef VOLVOXAI_FULL_ENUMS_H\n#define VOLVOXAI_FULL_ENUMS_H\n\n"
        + '#include "volvoxai_enums.h"\n\n'
        + "\n\n".join(sections)
        + "\n\n#endif /* VOLVOXAI_FULL_ENUMS_H */\n"
    )
    return content.encode()


def semantic_values(
    entries: tuple[tuple[str, int], ...], prefix: str, mapping=None
) -> list[str]:
    values = []
    for name, _ in entries:
        suffix = name.removeprefix(prefix).lower()
        if callable(mapping):
            values.append(mapping(suffix))
        else:
            values.append(mapping.get(suffix, suffix) if mapping else suffix)
    return values


def render_ts_enum(
    spec: EnumSpec,
    entries: tuple[tuple[str, int], ...],
    exported_name: str | None = None,
) -> str:
    lines = []
    for name, value in entries:
        suffix = name.removeprefix(spec.proto_prefix)
        lines.append(f"  {camel_case(suffix)} = {value},")
    return (
        f"export enum {exported_name or spec.proto_name} {{\n"
        + "\n".join(lines)
        + "\n}"
    )


def render_ts_base(
    enums: dict[str, tuple[tuple[str, int], ...]], proto_hash: str
) -> bytes:
    sections: list[str] = []
    for spec in SPECS:
        if spec.profile == "full":
            continue
        entries = tuple(
            entry
            for entry in enums[spec.proto_name]
            if not is_full_only_entry(spec, entry[0])
        )
        sections.append(render_ts_enum(spec, entries))

    # Both hosts use one error adapter. Erased numeric unions accept either
    # profile's reports without pulling full-only enum names into inference.
    for enum_name in ("OperationCode", "OperationStage"):
        sections.append(
            f"export type {enum_name}Number = "
            + " | ".join(str(value) for _, value in enums[enum_name]) + ";"
        )

    semantic_specs = (
        (
            "NativeStatusCode",
            "nativeStatusCodes",
            "NativeStatus",
            lambda suffix: suffix.upper(),
        ),
        (
            "OperationStageValue",
            "operationStages",
            "OperationStage",
            lambda suffix: suffix.replace("_", "-"),
        ),
        (
            "RuntimeDType",
            "runtimeDTypes",
            "DataType",
            {"f32": "float32", "i32": "int32", "i8": "int8", "u8": "uint8"},
        ),
        ("MemoryLocationValue", "memoryLocations", "MemoryLocation", None),
        ("BackendPolicyModeValue", "backendPolicyModes", "BackendPolicyMode", None),
        ("OperatorFallbackValue", "operatorFallbackValues", "OperatorFallback", None),
        ("DecodeRowModeValue", "decodeRowModes", "DecodeRowMode", None),
        ("ExecutionModeValue", "executionModes", "ExecutionMode", None),
    )
    by_spec = {spec.proto_name: spec for spec in SPECS}
    for type_name, const_name, enum_name, mapping in semantic_specs:
        spec = by_spec[enum_name]
        entries = tuple(
            entry
            for entry in enums[enum_name]
            if not is_full_only_entry(spec, entry[0])
        )
        if type_name == "RuntimeDType":
            entries_by_name = dict(entries)
            missing = [
                name for name in RUNTIME_DTYPE_PROTO_NAMES
                if name not in entries_by_name
            ]
            if missing:
                raise ValueError(
                    "missing runtime dtype enum value(s): " + ", ".join(missing)
                )
            entries = tuple(
                (name, entries_by_name[name]) for name in RUNTIME_DTYPE_PROTO_NAMES
            )
        values = semantic_values(
            entries, spec.proto_prefix, mapping=mapping
        )
        rendered = ", ".join(repr(value) for value in values)
        sections.append(
            f"export const {const_name} = Object.freeze([{rendered}] as const);\n"
            f"export type {type_name} = (typeof {const_name})[number];"
        )
    sections.append(
        "export type NativeFailureCode = Exclude<NativeStatusCode, 'OK'>;"
    )
    content = generated_banner(proto_hash, "//") + "\n\n".join(sections) + "\n"
    return content.encode()


def render_ts_full(
    enums: dict[str, tuple[tuple[str, int], ...]], proto_hash: str
) -> bytes:
    sections: list[str] = []
    for spec in SPECS:
        entries = tuple(
            entry
            for entry in enums[spec.proto_name]
            if is_full_only_entry(spec, entry[0])
        )
        if not entries:
            continue
        exported_name = (
            f"Full{spec.proto_name}" if spec.profile == "mixed" else spec.proto_name
        )
        sections.append(render_ts_enum(spec, entries, exported_name))

    semantic_specs = (
        (
            "FullNativeStatusCode",
            "fullNativeStatusCodes",
            "NativeStatus",
            lambda suffix: suffix.upper(),
        ),
        (
            "FullOperationStageValue",
            "fullOperationStages",
            "OperationStage",
            lambda suffix: suffix.replace("_", "-"),
        ),
        (
            "TrainingOptimizerKindValue",
            "trainingOptimizerKinds",
            "TrainingOptimizerKind",
            None,
        ),
        ("PtqModeValue", "ptqModes", "PtqMode", None),
        ("PtqSchemeValue", "ptqSchemes", "PtqScheme", None),
        ("PtqLayerKindValue", "ptqLayerKinds", "PtqLayerKind", None),
    )
    by_spec = {spec.proto_name: spec for spec in SPECS}
    for type_name, const_name, enum_name, mapping in semantic_specs:
        spec = by_spec[enum_name]
        entries = tuple(
            entry
            for entry in enums[enum_name]
            if is_full_only_entry(spec, entry[0])
        )
        if not entries:
            continue
        values = semantic_values(entries, spec.proto_prefix, mapping=mapping)
        rendered = ", ".join(repr(value) for value in values)
        sections.append(
            f"export const {const_name} = Object.freeze([{rendered}] as const);\n"
            f"export type {type_name} = (typeof {const_name})[number];"
        )

    content = generated_banner(proto_hash, "//") + "\n\n".join(sections) + "\n"
    return content.encode()


def render_python(
    enums: dict[str, tuple[tuple[str, int], ...]], proto_hash: str
) -> bytes:
    """Render the lite Python enum contract used by the offline toolchain.

    Every public proto enum becomes a plain ``IntEnum`` with no protobuf runtime
    dependency. The implementation-only operator vocabulary has its own
    generated module.
    """
    sections: list[str] = []
    for spec in SPECS:
        lines = [f"class {spec.proto_name}(IntEnum):"]
        for name, value in enums[spec.proto_name]:
            lines.append(f"    {name.removeprefix(spec.proto_prefix)} = {value}")
        sections.append("\n".join(lines))
    content = (
        generated_banner(proto_hash, "#")
        + "\nfrom enum import IntEnum\n\n\n"
        + "\n\n\n".join(sections)
        + "\n"
    )
    return content.encode()


def render_methods(source: str, proto_hash: str) -> tuple[bytes, bytes, bytes]:
    """Project service and RPC spellings from the same public schema."""
    clean = strip_comments(source)
    package = re.search(r"\bpackage\s+([\w.]+)\s*;", clean)
    if package is None:
        raise ValueError("public API must declare its package")
    services = re.findall(r"\bservice\s+(\w+)\s*\{(.*?)\}", clean, re.DOTALL)
    c = [generated_banner(proto_hash, "/*").replace("\n", " */\n").rstrip(),
         "#ifndef VOLVOXAI_PROTO_METHODS_H", "#define VOLVOXAI_PROTO_METHODS_H"]
    ts = [generated_banner(proto_hash, "//").rstrip(),
          f"export const PROTO_PACKAGE = {package[1]!r};", "export enum ProtoService {"]
    for service, body in services:
        ts.append(f"  {service} = {service!r},")
        for method in re.findall(r"\brpc\s+(\w+)\s*\(", body):
            token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", service + "_" + method).upper()
            c.append(f'#define VX_RPC_{token} "/{package[1]}.{service}/{method}"')
    c.extend(["#endif", ""])
    ts.extend(["}", "export const PROTO_METHOD_RESPONSES = Object.freeze({"])
    full = [generated_banner(proto_hash, "//").rstrip(),
            "import { PROTO_METHOD_RESPONSES as common } from './protoMethods.js';",
            "export const PROTO_METHOD_RESPONSES = Object.freeze({", "  ...common,"]
    for service, body in services:
        target = full if service in FULL_ONLY_SERVICES else ts
        for method, response in re.findall(
                r"\brpc\s+(\w+)\s*\([^)]*\)\s*returns\s*\(\s*(\w+)\s*\)", body):
            target.append(f"  '/{package[1]}.{service}/{method}': '{response}',")
    ts.extend(["} as const);", ""])
    full.extend(["} as const);", ""])
    return "\n".join(c).encode(), "\n".join(ts).encode(), "\n".join(full).encode()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def synurang_drift_errors(proto_hash: str) -> list[str]:
    errors: list[str] = []
    for manifest_path in SYNURANG_MANIFESTS:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"invalid Synurang manifest {manifest_path}: {error}")
            continue
        if manifest.get("proto_sha256") != proto_hash:
            errors.append(
                f"stale Synurang manifest {manifest_path}: "
                f"expected proto SHA-256 {proto_hash}"
            )
        files = manifest.get("files")
        if not isinstance(files, list) or not files or not all(
            isinstance(filename, str) and filename for filename in files
        ):
            errors.append(f"invalid Synurang file list in {manifest_path}")
            continue
        for filename in files:
            generated = manifest_path.parent / filename
            try:
                contents = generated.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                errors.append(f"missing or invalid Synurang output {generated}: {error}")
                continue
            # Runtime support and licenses come from the pinned source snapshot
            # and do not depend on VolvoxAI's schema. The codegen gate verifies
            # those files byte-for-byte; only schema projections carry this hash.
            if generated.name.startswith("volvoxai_") and proto_hash not in contents[:4096]:
                errors.append(
                    f"stale Synurang provenance in {generated}: "
                    f"expected proto SHA-256 {proto_hash}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--proto", type=Path, default=ROOT / "proto/volvoxai.proto")
    parser.add_argument(
        "--native-out",
        type=Path,
        default=ROOT / "native/include/volvoxai_enums.h",
    )
    parser.add_argument(
        "--native-full-out",
        type=Path,
        default=ROOT / "native/include/volvoxai_full_enums.h",
    )
    parser.add_argument(
        "--typescript-out",
        type=Path,
        default=ROOT / "ts/generated/volvoxaiEnums.ts",
    )
    parser.add_argument(
        "--typescript-full-out",
        type=Path,
        default=ROOT / "tools/generated/volvoxaiFullEnums.ts",
    )
    parser.add_argument(
        "--python-out",
        type=Path,
        default=ROOT / "tools/exporter/generated/volvox_enums.py",
    )
    args = parser.parse_args()

    proto = args.proto.resolve()
    source_bytes = proto.read_bytes()
    source = source_bytes.decode("utf-8")
    proto_hash = hashlib.sha256(source_bytes).hexdigest()
    try:
        enums = parse_enums(source)
    except ValueError as error:
        print(f"enum generation failed: {error}", file=sys.stderr)
        return 1
    c_methods, ts_methods, ts_full_methods = render_methods(source, proto_hash)
    outputs = {
        args.native_out.resolve(): render_c_base(enums, proto_hash),
        args.native_full_out.resolve(): render_c_full(enums, proto_hash),
        args.typescript_out.resolve(): render_ts_base(enums, proto_hash),
        args.typescript_full_out.resolve(): render_ts_full(enums, proto_hash),
        args.python_out.resolve(): render_python(enums, proto_hash),
        args.native_out.resolve().parent.parent / "src/generated/proto_methods.h": c_methods,
        args.typescript_out.resolve().with_name("protoMethods.ts"): ts_methods,
        args.typescript_out.resolve().with_name("protoMethodsFull.ts"): ts_full_methods,
    }
    stale = [path for path, data in outputs.items() if not path.is_file() or path.read_bytes() != data]
    if args.check:
        manifest_errors = synurang_drift_errors(proto_hash)
        if stale or manifest_errors:
            for path in stale:
                print(f"stale generated enum contract: {path}", file=sys.stderr)
            for error in manifest_errors:
                print(error, file=sys.stderr)
            print("run make proto_codegen", file=sys.stderr)
            return 1
        print(
            f"Verified {len(outputs)} generated enum contract files and "
            f"{len(SYNURANG_MANIFESTS)} Synurang manifests"
        )
        return 0
    for path, data in outputs.items():
        write_atomic(path, data)
        print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
