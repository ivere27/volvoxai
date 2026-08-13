#!/usr/bin/env python3
"""Generate the small native/TypeScript enum contract from volvoxai.proto."""

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
    ROOT / "runtime/generated/c/.synurang-c-lite.manifest.json",
    ROOT / "runtime/generated/c/.synurang-c-native.manifest.json",
    ROOT / "runtime/generated/typescript/.synurang-typescript.manifest.json",
    ROOT / "runtime/src/gen/.synurang-rust.manifest.json",
)


@dataclass(frozen=True)
class EnumSpec:
    proto_name: str
    proto_prefix: str
    c_type: str
    c_prefix: str
    profile: str


SPECS = (
    EnumSpec("NativeStatus", "NATIVE_STATUS_", "VxStatus", "VX_STATUS_", "mixed"),
    EnumSpec("OperationStage", "OPERATION_STAGE_", "VxStage", "VX_STAGE_", "mixed"),
    EnumSpec("DataType", "DATA_TYPE_", "VxDataType", "VX_DTYPE_", "base"),
    EnumSpec(
        "OperatorKind",
        "OPERATOR_KIND_",
        "VxOperatorKind",
        "VX_OP_",
        "base",
    ),
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

RUNTIME_DTYPE_PROTO_NAMES = (
    "DATA_TYPE_F32",
    "DATA_TYPE_I8",
    "DATA_TYPE_U8",
    "DATA_TYPE_I32",
)

_OPERATOR_GRAPH_TOKEN_CASE = {
    "Q": "Q",
    "MATMUL": "MatMul",
    "SDPA": "SDPA",
    "RMS": "RMS",
    "SSM": "SSM",
    "MOE": "MoE",
    "ROPE": "RoPE",
    "RELU": "ReLU",
    "GELU": "GELU",
    "SILU": "SiLU",
    "PRELU": "PReLU",
    "1D": "1D",
    "2D": "2D",
}
def operator_graph_name(proto_name: str) -> str:
    """Return the sole current volvox-graph/v1 spelling for one enum value."""

    suffix = proto_name.removeprefix("OPERATOR_KIND_")
    return "".join(
        _OPERATOR_GRAPH_TOKEN_CASE.get(token, token.lower().capitalize())
        for token in suffix.split("_")
    )


def operator_graph_entries(
    enums: dict[str, tuple[tuple[str, int], ...]],
) -> tuple[tuple[str, int, str], ...]:
    result: list[tuple[str, int, str]] = []
    for proto_name, value in enums["OperatorKind"]:
        if value == 0:
            continue
        result.append((proto_name, value, operator_graph_name(proto_name)))
    names = [name for _, _, name in result]
    if len(names) != len(set(names)):
        raise ValueError("OperatorKind graph spellings must be unique")
    return tuple(result)


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
    runtime_operator_names = [name for _, _, name in operator_graph_entries(enums)]
    runtime_operator_lines = [
        f"        {json.dumps(name)}," for name in runtime_operator_names
    ]
    sections.append(
        "static inline int vx_graph_runtime_operator_name_is_current(\n"
        "        const char* name) {\n"
        "    static const char* const names[] = {\n"
        + "\n".join(runtime_operator_lines)
        + "\n    };\n"
        "    if (!name) return 0;\n"
        "    for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); "
        "index++) {\n"
        "        const char* actual = name;\n"
        "        const char* expected = names[index];\n"
        "        while (*actual && *actual == *expected) {\n"
        "            actual++;\n"
        "            expected++;\n"
        "        }\n"
        "        if (*actual == *expected) return 1;\n"
        "    }\n"
        "    return 0;\n"
        "}"
    )
    content = (
        generated_banner(proto_hash, "/*").replace("\n", " */\n")
        + "#ifndef VOLVOXAI_ENUMS_H\n#define VOLVOXAI_ENUMS_H\n\n"
        + "#include <stddef.h>\n#include <stdint.h>\n\n"
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
    operator_entries = operator_graph_entries(enums)
    runtime_names = [name for _, _, name in operator_entries]
    sections.append(
        "export const runtimeOperatorNames = Object.freeze(["
        + ", ".join(repr(name) for name in runtime_names)
        + "] as const);\n"
        "export type RuntimeOperatorName = (typeof runtimeOperatorNames)[number];"
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


def render_js_operator_names(
    enums: dict[str, tuple[tuple[str, int], ...]], proto_hash: str
) -> bytes:
    entries = operator_graph_entries(enums)
    runtime = [name for _, _, name in entries]
    content = (
        generated_banner(proto_hash, "//")
        + "\nexport const runtimeOperatorNames = Object.freeze("
        + json.dumps(runtime, separators=(",", ":"))
        + ");\n"
    )
    return content.encode()


def render_python(
    enums: dict[str, tuple[tuple[str, int], ...]], proto_hash: str
) -> bytes:
    """Render the lite Python enum contract used by the offline toolchain.

    Every proto enum becomes a plain ``IntEnum`` with no protobuf runtime
    dependency, so the exporter/optimizer can reference the canonical dtype and
    operator vocabulary by the same numbers the C and TypeScript contracts use.
    """
    sections: list[str] = []
    for spec in SPECS:
        lines = [f"class {spec.proto_name}(IntEnum):"]
        for name, value in enums[spec.proto_name]:
            lines.append(f"    {name.removeprefix(spec.proto_prefix)} = {value}")
        sections.append("\n".join(lines))
    operator_entries = operator_graph_entries(enums)
    operator_name_lines = [
        f"    OperatorKind.{proto_name.removeprefix('OPERATOR_KIND_')}: {name!r},"
        for proto_name, _, name in operator_entries
    ]
    runtime_name_lines = [
        f"    {name!r},"
        for _, _, name in operator_entries
    ]
    sections.append(
        "OPERATOR_GRAPH_NAMES = {\n"
        + "\n".join(operator_name_lines)
        + "\n}\n\n"
        "RUNTIME_OPERATOR_NAMES = (\n"
        + "\n".join(runtime_name_lines)
        + "\n)"
    )
    content = (
        generated_banner(proto_hash, "#")
        + "\nfrom enum import IntEnum\n\n\n"
        + "\n\n\n".join(sections)
        + "\n"
    )
    return content.encode()


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
            if proto_hash not in contents[:4096]:
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
        default=ROOT / "ts/generated/volvoxaiFullEnums.ts",
    )
    parser.add_argument(
        "--python-out",
        type=Path,
        default=ROOT / "tools/exporter/generated/volvox_enums.py",
    )
    parser.add_argument(
        "--javascript-out",
        type=Path,
        default=ROOT / "tools/generated/volvoxaiGraphOperators.mjs",
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
    outputs = {
        args.native_out.resolve(): render_c_base(enums, proto_hash),
        args.native_full_out.resolve(): render_c_full(enums, proto_hash),
        args.typescript_out.resolve(): render_ts_base(enums, proto_hash),
        args.typescript_full_out.resolve(): render_ts_full(enums, proto_hash),
        args.python_out.resolve(): render_python(enums, proto_hash),
        args.javascript_out.resolve(): render_js_operator_names(enums, proto_hash),
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
