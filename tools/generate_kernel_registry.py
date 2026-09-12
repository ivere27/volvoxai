#!/usr/bin/env python3
"""Generate dependency-free views of the internal protobuf kernel registry.

The normal and ``--check`` paths use only the Python standard library.  A
strict parser reads the protobuf aggregate assigned to the custom FileOptions
extension; ``--protoc-check`` additionally asks protoc to validate the complete
schema when that tool is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from generate_operator_vocabulary import (
        operator_graph_entries,
        parse_operator_vocabulary,
    )
except ImportError:  # Imported as tools.generate_kernel_registry in tests.
    from tools.generate_operator_vocabulary import (
        operator_graph_entries,
        parse_operator_vocabulary,
    )


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_OPTION = "volvoxai.kernel_registry.registry"
SHAPE_FUNCTION_ID_RE = re.compile(
    r"volvox\.shape\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*\.v[1-9][0-9]*"
)
SHAPE_CLASSIFICATION_PREFIX = "SHAPE_CONTRACT_CLASSIFICATION_"
BROWSER_RUNTIME_BACKEND_IDS = ("wasm", "webgpu")


@dataclass(frozen=True)
class Token:
    value: str
    line: int
    kind: str


@dataclass(frozen=True)
class Route:
    operator: str
    route: str
    predicate_id: str


@dataclass(frozen=True)
class Backend:
    kind: str
    kind_value: int
    runtime_id: str
    provider_name: str
    exporter_target: str
    support_mode: str
    default_route: str
    default_predicate_id: str
    runtime_operators: tuple[str, ...]
    routes: tuple[Route, ...]
    qualified_operators: tuple[str, ...]

    def route_for(self, operator: str) -> Route:
        for route in self.routes:
            if route.operator == operator:
                return route
        return Route(operator, self.default_route, self.default_predicate_id)


@dataclass(frozen=True)
class Variant:
    id: str
    backend: str
    operators: tuple[str, ...]
    profile: str
    phase: str
    entrypoint_id: str
    predicate_id: str
    required_features: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class ShapeContract:
    operator: str
    shape_function_id: str
    classification: str


@dataclass(frozen=True)
class Registry:
    schema_version: int
    backends: tuple[Backend, ...]
    target_profiles: tuple[tuple[str, tuple[str, ...]], ...]
    variants: tuple[Variant, ...]
    shape_contracts: tuple[ShapeContract, ...]
    operator_numbers: dict[str, int]
    graph_names: dict[str, str]
    shape_classification_numbers: dict[str, int]
    registry_hash: str
    vocabulary_hash: str


TOKEN_RE = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<line_comment>//[^\n]*)"
    r"|(?P<block_comment>/\*.*?\*/)"
    r'|(?P<string>"(?:\\.|[^"\\])*")'
    r"|(?P<number>-?[0-9]+)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_.]*)"
    r"|(?P<punct>[{}():=;,])"
    r"|(?P<invalid>.)",
    re.DOTALL,
)


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    line = 1
    offset = 0
    while offset < len(source):
        match = TOKEN_RE.match(source, offset)
        assert match is not None
        value = match.group(0)
        kind = match.lastgroup or "invalid"
        if kind == "invalid":
            raise ValueError(f"unexpected character {value!r} at line {line}")
        if kind not in {"space", "line_comment", "block_comment"}:
            tokens.append(Token(value, line, kind))
        line += value.count("\n")
        offset = match.end()
    return tokens


class AggregateParser:
    def __init__(self, tokens: list[Token], index: int):
        self.tokens = tokens
        self.index = index

    def peek(self) -> Token:
        if self.index >= len(self.tokens):
            raise ValueError("unexpected end of protobuf registry option")
        return self.tokens[self.index]

    def take(self, expected: str | None = None) -> Token:
        token = self.peek()
        if expected is not None and token.value != expected:
            raise ValueError(
                f"expected {expected!r}, got {token.value!r} at line {token.line}"
            )
        self.index += 1
        return token

    def message(self) -> dict[str, list[Any]]:
        self.take("{")
        result: dict[str, list[Any]] = {}
        while self.peek().value != "}":
            name = self.take()
            if name.kind != "ident" or "." in name.value:
                raise ValueError(f"invalid field name {name.value!r} at line {name.line}")
            self.take(":")
            value = self.value()
            result.setdefault(name.value, []).append(value)
            while self.peek().value in {",", ";"}:
                self.take()
        self.take("}")
        return result

    def value(self) -> Any:
        token = self.peek()
        if token.value == "{":
            return self.message()
        self.take()
        if token.kind == "string":
            return json.loads(token.value)
        if token.kind == "number":
            return int(token.value)
        if token.kind == "ident":
            if token.value == "true":
                return True
            if token.value == "false":
                return False
            return token.value
        raise ValueError(f"invalid value {token.value!r} at line {token.line}")


def parse_registry_option(source: str) -> dict[str, list[Any]]:
    tokens = tokenize(source)
    for index in range(len(tokens) - 5):
        values = [token.value for token in tokens[index:index + 6]]
        if values == ["option", "(", REGISTRY_OPTION, ")", "=", "{"]:
            parser = AggregateParser(tokens, index + 5)
            result = parser.message()
            parser.take(";")
            return result
    raise ValueError(f"missing option ({REGISTRY_OPTION})")


def parse_declared_enum(source: str, name: str) -> dict[str, int]:
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    match = re.search(rf"\benum\s+{re.escape(name)}\s*\{{(.*?)\}}", clean, re.DOTALL)
    if match is None:
        raise ValueError(f"missing enum {name}")
    result: dict[str, int] = {}
    for statement in match.group(1).split(";"):
        statement = statement.strip()
        if not statement:
            continue
        if statement.startswith("reserved "):
            continue
        entry = re.fullmatch(r"([A-Z][A-Z0-9_]*)\s*=\s*(-?[0-9]+)", statement)
        if entry is None:
            raise ValueError(f"unsupported {name} declaration: {statement!r}")
        key, value = entry.group(1), int(entry.group(2))
        if key in result or value in result.values():
            raise ValueError(f"duplicate {name} enum entry {key}")
        result[key] = value
    return result


def reject_unknown(message: dict[str, list[Any]], allowed: set[str], where: str) -> None:
    unknown = sorted(set(message) - allowed)
    if unknown:
        raise ValueError(f"{where} has unknown field(s): {', '.join(unknown)}")


def values(message: dict[str, list[Any]], field: str, kind: type, where: str) -> list[Any]:
    result = message.get(field, [])
    if not all(isinstance(value, kind) and not (kind is int and isinstance(value, bool)) for value in result):
        raise ValueError(f"{where}.{field} has the wrong protobuf aggregate type")
    return result


def scalar(
    message: dict[str, list[Any]], field: str, kind: type, where: str,
    *, required: bool = True, default: Any = None,
) -> Any:
    result = values(message, field, kind, where)
    if len(result) > 1:
        raise ValueError(f"{where}.{field} must not be repeated")
    if not result:
        if required:
            raise ValueError(f"{where}.{field} is required")
        return default
    return result[0]


def unique(items: Iterable[str], where: str) -> tuple[str, ...]:
    result = tuple(items)
    if len(result) != len(set(result)):
        duplicate = next(item for index, item in enumerate(result) if item in result[:index])
        raise ValueError(f"duplicate {where}: {duplicate}")
    return result


def expand_operator_refs(
    message: dict[str, list[Any]], direct_field: str, set_field: str,
    operator_sets: dict[str, tuple[str, ...]], where: str,
) -> tuple[str, ...]:
    result: list[str] = []
    for set_name in values(message, set_field, str, where):
        if set_name not in operator_sets:
            raise ValueError(f"{where} references unknown operator set {set_name!r}")
        result.extend(operator_sets[set_name])
    result.extend(values(message, direct_field, str, where))
    return unique(result, f"expanded operator in {where}")


def load_registry(registry_proto: Path, operator_vocabulary_proto: Path) -> Registry:
    registry_bytes = registry_proto.read_bytes()
    vocabulary_bytes = operator_vocabulary_proto.read_bytes()
    source = registry_bytes.decode("utf-8")
    vocabulary_source = vocabulary_bytes.decode("utf-8")
    root = parse_registry_option(source)
    reject_unknown(
        root,
        {
            "schema_version", "backend", "target_profile", "variant",
            "operator_set", "shape_contract",
        },
        "registry",
    )

    schema_version = scalar(root, "schema_version", int, "registry")
    if schema_version != 1:
        raise ValueError(f"unsupported kernel registry schema_version {schema_version}")

    operator_enums = parse_operator_vocabulary(vocabulary_source)
    operator_entries = operator_graph_entries(operator_enums)
    operator_numbers = {name: number for name, number, _graph in operator_entries}
    graph_names = {name: graph for name, _number, graph in operator_entries}
    runtime_operator_names = tuple(name for name, _number, _graph in operator_entries)
    allowed_operators = set(operator_numbers)

    backend_kinds = parse_declared_enum(source, "BackendKind")
    profiles = parse_declared_enum(source, "KernelProfile")
    phases = parse_declared_enum(source, "KernelPhase")
    modes = parse_declared_enum(source, "RuntimeSupportMode")
    shape_classifications = parse_declared_enum(source, "ShapeContractClassification")

    shape_contracts: list[ShapeContract] = []
    shape_contract_operators: list[str] = []
    classification_by_shape_function: dict[str, str] = {}
    shape_contract_fields = {"operator", "shape_function_id", "classification"}
    for index, item in enumerate(values(root, "shape_contract", dict, "registry")):
        where = f"shape_contract[{index}]"
        reject_unknown(item, shape_contract_fields, where)
        operator = scalar(item, "operator", str, where)
        shape_function_id = scalar(item, "shape_function_id", str, where)
        classification = scalar(item, "classification", str, where)
        if operator not in allowed_operators:
            raise ValueError(f"{where} has unknown operator {operator!r}")
        if SHAPE_FUNCTION_ID_RE.fullmatch(shape_function_id) is None:
            raise ValueError(
                f"{where}.shape_function_id has invalid stable ID {shape_function_id!r}"
            )
        if (
            classification not in shape_classifications
            or shape_classifications[classification] == 0
        ):
            raise ValueError(f"{where} has invalid shape-contract classification {classification!r}")
        previous_classification = classification_by_shape_function.setdefault(
            shape_function_id, classification
        )
        if previous_classification != classification:
            raise ValueError(
                f"shape-function ID {shape_function_id!r} has conflicting classifications"
            )
        shape_contract_operators.append(operator)
        shape_contracts.append(ShapeContract(operator, shape_function_id, classification))
    unique(shape_contract_operators, "shape-contract operator")
    shape_contract_operator_set = set(shape_contract_operators)
    if shape_contract_operator_set != allowed_operators:
        missing = sorted(allowed_operators - shape_contract_operator_set)
        extra = sorted(shape_contract_operator_set - allowed_operators)
        raise ValueError(
            "shape contracts must exactly cover OperatorKind; "
            f"missing={missing}, extra={extra}"
        )
    shape_contracts.sort(key=lambda item: operator_numbers[item.operator])

    operator_sets: dict[str, tuple[str, ...]] = {}
    for index, item in enumerate(values(root, "operator_set", dict, "registry")):
        where = f"operator_set[{index}]"
        reject_unknown(item, {"name", "operator"}, where)
        name = scalar(item, "name", str, where)
        if not name or name in operator_sets:
            raise ValueError(f"duplicate or empty operator set name {name!r}")
        operators = unique(values(item, "operator", str, where), f"operator in {where}")
        invalid = [operator for operator in operators if operator not in allowed_operators]
        if invalid:
            raise ValueError(f"{where} has unknown operator(s): {', '.join(invalid)}")
        operator_sets[name] = operators
    if set(operator_sets.get("runtime-all", ())) != set(runtime_operator_names):
        missing = sorted(set(runtime_operator_names) - set(operator_sets.get("runtime-all", ())))
        extra = sorted(set(operator_sets.get("runtime-all", ())) - set(runtime_operator_names))
        raise ValueError(f"runtime-all must exactly cover OperatorKind; missing={missing}, extra={extra}")

    backends: list[Backend] = []
    seen_kinds: set[str] = set()
    seen_runtime_ids: set[str] = set()
    seen_targets: set[str] = set()
    backend_fields = {
        "kind", "runtime_id", "provider_name", "exporter_target", "support_mode",
        "default_route", "default_predicate_id", "runtime_operator_set",
        "exporter_qualified_operator_set", "runtime_operator", "route",
        "exporter_qualified_operator",
    }
    for index, item in enumerate(values(root, "backend", dict, "registry")):
        where = f"backend[{index}]"
        reject_unknown(item, backend_fields, where)
        kind = scalar(item, "kind", str, where)
        if kind not in backend_kinds or backend_kinds[kind] == 0:
            raise ValueError(f"{where} has invalid backend kind {kind!r}")
        runtime_id = scalar(item, "runtime_id", str, where)
        provider_name = scalar(item, "provider_name", str, where)
        target = scalar(item, "exporter_target", str, where, required=False, default="")
        mode = scalar(item, "support_mode", str, where)
        if mode not in modes or modes[mode] == 0:
            raise ValueError(f"{where} has invalid support mode {mode!r}")
        default_route = scalar(item, "default_route", str, where, required=False, default="")
        default_predicate = scalar(item, "default_predicate_id", str, where, required=False, default="")
        if kind in seen_kinds or runtime_id in seen_runtime_ids or (target and target in seen_targets):
            raise ValueError(f"duplicate backend kind/id/target at {where}")
        if not re.fullmatch(r"[a-z][a-z0-9._-]{0,62}", provider_name):
            raise ValueError(f"{where}.provider_name is invalid")
        if not runtime_id:
            raise ValueError(f"{where}.runtime_id must be non-empty")
        seen_kinds.add(kind)
        seen_runtime_ids.add(runtime_id)
        if target:
            seen_targets.add(target)

        runtime_operators = expand_operator_refs(
            item, "runtime_operator", "runtime_operator_set", operator_sets, where
        )
        qualified = expand_operator_refs(
            item, "exporter_qualified_operator",
            "exporter_qualified_operator_set", operator_sets, where,
        )
        invalid_runtime = [operator for operator in runtime_operators if operator not in allowed_operators]
        invalid_qualified = [operator for operator in qualified if operator not in allowed_operators]
        if invalid_runtime or invalid_qualified:
            raise ValueError(
                f"{where} has unknown operator(s): {invalid_runtime + invalid_qualified}"
            )
        if not set(qualified) <= set(runtime_operators):
            raise ValueError(f"{where} qualifies an operator absent from runtime registration")
        if qualified and not target:
            raise ValueError(f"{where} has qualification without exporter_target")
        routes: list[Route] = []
        route_fields = {"operator", "route", "predicate_id"}
        for route_index, route_item in enumerate(values(item, "route", dict, where)):
            route_where = f"{where}.route[{route_index}]"
            reject_unknown(route_item, route_fields, route_where)
            operator = scalar(route_item, "operator", str, route_where)
            route_name = scalar(route_item, "route", str, route_where)
            predicate = scalar(route_item, "predicate_id", str, route_where)
            if operator not in runtime_operators:
                raise ValueError(f"{route_where} overrides unregistered operator {operator}")
            if not route_name or not predicate:
                raise ValueError(f"{route_where} route and predicate_id must be non-empty")
            routes.append(Route(operator, route_name, predicate))
        unique((route.operator for route in routes), f"route operator in {where}")
        if bool(default_route) != bool(default_predicate):
            raise ValueError(f"{where} must set default_route and default_predicate_id together")
        if not default_route and set(route.operator for route in routes) != set(runtime_operators):
            missing = sorted(set(runtime_operators) - {route.operator for route in routes})
            raise ValueError(f"{where} has runtime operators without routes: {missing}")
        backends.append(Backend(
            kind, backend_kinds[kind], runtime_id, provider_name, target, mode,
            default_route, default_predicate, runtime_operators,
            tuple(routes), qualified,
        ))

    backends.sort(key=lambda backend: backend.kind_value)
    unique((backend.provider_name for backend in backends), "provider name")
    declared_backend_kinds = {name for name, value in backend_kinds.items() if value != 0}
    if seen_kinds != declared_backend_kinds:
        raise ValueError(f"backend inventory must cover BackendKind exactly: missing={sorted(declared_backend_kinds - seen_kinds)}")

    target_profiles: list[tuple[str, tuple[str, ...]]] = []
    profile_names: set[str] = set()
    for index, item in enumerate(values(root, "target_profile", dict, "registry")):
        where = f"target_profile[{index}]"
        reject_unknown(item, {"name", "member_target"}, where)
        name = scalar(item, "name", str, where)
        members = unique(values(item, "member_target", str, where), f"member in {where}")
        if not name or name in profile_names or name in seen_targets:
            raise ValueError(f"duplicate or invalid target profile {name!r}")
        invalid = [member for member in members if member not in seen_targets]
        if invalid:
            raise ValueError(f"{where} has unknown member target(s): {', '.join(invalid)}")
        profile_names.add(name)
        target_profiles.append((name, members))
    if profile_names != {"portable", "browser"}:
        raise ValueError("target profiles must define exactly portable and browser")

    variants: list[Variant] = []
    variant_ids: set[str] = set()
    backend_by_kind = {backend.kind: backend for backend in backends}
    variant_fields = {
        "id", "backend", "operator", "profile", "phase", "entrypoint_id",
        "predicate_id", "required_feature", "priority",
    }
    for index, item in enumerate(values(root, "variant", dict, "registry")):
        where = f"variant[{index}]"
        reject_unknown(item, variant_fields, where)
        variant_id = scalar(item, "id", str, where)
        backend_kind = scalar(item, "backend", str, where)
        operators = unique(values(item, "operator", str, where), f"operator in {where}")
        profile = scalar(item, "profile", str, where)
        phase = scalar(item, "phase", str, where)
        entrypoint = scalar(item, "entrypoint_id", str, where)
        predicate = scalar(item, "predicate_id", str, where)
        features = unique(values(item, "required_feature", str, where), f"feature in {where}")
        priority = scalar(item, "priority", int, where)
        if not variant_id or variant_id in variant_ids:
            raise ValueError(f"duplicate or empty kernel variant id {variant_id!r}")
        if backend_kind not in backend_by_kind:
            raise ValueError(f"{where} has unknown backend {backend_kind!r}")
        if not operators:
            raise ValueError(f"{where} must name at least one operator")
        invalid = [operator for operator in operators if operator not in backend_by_kind[backend_kind].runtime_operators]
        if invalid:
            raise ValueError(f"{where} uses operator(s) absent from backend: {invalid}")
        if profile not in profiles or profiles[profile] == 0:
            raise ValueError(f"{where} has invalid profile {profile!r}")
        if phase not in phases or phases[phase] == 0:
            raise ValueError(f"{where} has invalid phase {phase!r}")
        if not entrypoint or not predicate or priority <= 0 or any(not feature for feature in features):
            raise ValueError(f"{where} requires entrypoint, predicate, positive priority, and non-empty features")
        if profile == "KERNEL_PROFILE_INFERENCE":
            if phase != "KERNEL_PHASE_FORWARD":
                raise ValueError(f"{where} leaks non-forward phase into inference profile")
            leakage_text = " ".join((variant_id, entrypoint, predicate)).lower()
            if any(term in leakage_text for term in ("training", "backward", "optimizer", "ptq")):
                raise ValueError(f"{where} leaks a training/full identifier into inference profile")
        variant_ids.add(variant_id)
        variants.append(Variant(
            variant_id, backend_kind, operators, profile, phase,
            entrypoint, predicate, features, priority,
        ))

    return Registry(
        schema_version,
        tuple(backends),
        tuple(target_profiles),
        tuple(variants),
        tuple(shape_contracts),
        operator_numbers,
        graph_names,
        shape_classifications,
        hashlib.sha256(registry_bytes).hexdigest(),
        hashlib.sha256(vocabulary_bytes).hexdigest(),
    )


def banner(registry: Registry, marker: str) -> str:
    return (
        f"{marker} DO NOT EDIT: generated by tools/generate_kernel_registry.py.\n"
        f"{marker} proto/kernel_registry.proto SHA-256: {registry.registry_hash}\n"
        f"{marker} proto/operator_vocabulary.proto SHA-256: {registry.vocabulary_hash}\n"
    )


def ordered_operators(registry: Registry, operators: Iterable[str]) -> list[str]:
    return sorted(operators, key=registry.operator_numbers.__getitem__)


def graph_names(registry: Registry, operators: Iterable[str]) -> list[str]:
    return [registry.graph_names[operator] for operator in ordered_operators(registry, operators)]


def shape_classification_label(classification: str) -> str:
    if not classification.startswith(SHAPE_CLASSIFICATION_PREFIX):
        raise ValueError(f"invalid shape-contract classification {classification!r}")
    return classification.removeprefix(SHAPE_CLASSIFICATION_PREFIX).lower().replace("_", "-")


def render_python(registry: Registry) -> bytes:
    atomic_targets = [backend.exporter_target for backend in registry.backends if backend.exporter_target]
    profiles = {name: members for name, members in registry.target_profiles}
    lines = [
        banner(registry, "#"),
        "from types import MappingProxyType",
        "",
        f"SCHEMA_VERSION = {registry.schema_version}",
    ]
    lines.append(f"ATOMIC_TARGETS = {tuple(atomic_targets)!r}")
    lines.append(f"PROFILE_MEMBERS = MappingProxyType({profiles!r})")
    lines.append("TARGETS = (*PROFILE_MEMBERS, *ATOMIC_TARGETS)")
    classifications = tuple(
        shape_classification_label(name)
        for name, number in sorted(
            registry.shape_classification_numbers.items(), key=lambda item: item[1]
        )
        if number != 0
    )
    lines.append(f"SHAPE_CONTRACT_CLASSIFICATIONS = frozenset({classifications!r})")
    lines.append("OPERATOR_SHAPE_CONTRACTS = MappingProxyType({")
    for contract in registry.shape_contracts:
        operator = registry.graph_names[contract.operator]
        route = {
            "shape_function_id": contract.shape_function_id,
            "classification": shape_classification_label(contract.classification),
        }
        lines.append(f"    {operator!r}: MappingProxyType({route!r}),")
    lines.append("})")
    lines.append("SHAPE_FUNCTION_ID_BY_OPERATOR = MappingProxyType({")
    for contract in registry.shape_contracts:
        operator = registry.graph_names[contract.operator]
        lines.append(f"    {operator!r}: {contract.shape_function_id!r},")
    lines.append("})")
    lines.append("RUNTIME_OPERATORS_BY_BACKEND = MappingProxyType({")
    for backend in registry.backends:
        lines.append(f"    {backend.runtime_id!r}: frozenset({graph_names(registry, backend.runtime_operators)!r}),")
    lines.append("})")
    lines.append("KERNEL_ROUTES_BY_BACKEND = MappingProxyType({")
    for backend in registry.backends:
        route_map = {
            registry.graph_names[operator]: backend.route_for(operator).route
            for operator in ordered_operators(registry, backend.runtime_operators)
        }
        lines.append(f"    {backend.runtime_id!r}: MappingProxyType({route_map!r}),")
    lines.append("})")
    lines.append("RUNTIME_SUPPORT_MODE_BY_BACKEND = MappingProxyType({")
    for backend in registry.backends:
        mode = backend.support_mode.removeprefix("RUNTIME_SUPPORT_MODE_").lower()
        lines.append(f"    {backend.runtime_id!r}: {mode!r},")
    lines.append("})")
    lines.append("OPS_BY_TARGET = MappingProxyType({")
    for backend in registry.backends:
        if backend.exporter_target:
            lines.append(f"    {backend.exporter_target!r}: frozenset({graph_names(registry, backend.qualified_operators)!r}),")
    lines.append("})")
    lines.append("KERNEL_VARIANTS = (")
    for variant in registry.variants:
        if variant.profile != "KERNEL_PROFILE_INFERENCE":
            continue
        backend = next(item for item in registry.backends if item.kind == variant.backend)
        item = {
            "id": variant.id,
            "backend": backend.runtime_id,
            "operators": tuple(graph_names(registry, variant.operators)),
            "phase": "forward",
            "entrypoint_id": variant.entrypoint_id,
            "predicate_id": variant.predicate_id,
            "required_features": variant.required_features,
            "priority": variant.priority,
        }
        lines.append(f"    MappingProxyType({item!r}),")
    lines.append(")")
    return ("\n".join(lines) + "\n").encode()


def render_python_full(registry: Registry) -> bytes:
    variants = []
    for variant in registry.variants:
        if variant.profile != "KERNEL_PROFILE_FULL":
            continue
        backend = next(item for item in registry.backends if item.kind == variant.backend)
        variants.append({
            "id": variant.id,
            "backend": backend.runtime_id,
            "operators": tuple(graph_names(registry, variant.operators)),
            "phase": variant.phase.removeprefix("KERNEL_PHASE_").lower(),
            "entrypoint_id": variant.entrypoint_id,
            "predicate_id": variant.predicate_id,
            "required_features": variant.required_features,
            "priority": variant.priority,
        })
    lines = [banner(registry, "#"), "from types import MappingProxyType", "", "FULL_KERNEL_VARIANTS = ("]
    for variant in variants:
        lines.append(f"    MappingProxyType({variant!r}),")
    lines.append(")")
    return ("\n".join(lines) + "\n").encode()


def ts_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def ts_frozen_array(values_: Iterable[Any]) -> str:
    return f"Object.freeze({ts_literal(list(values_))})"


def ts_frozen_array_map(mapping: dict[str, Iterable[Any]]) -> str:
    fields = ",".join(
        f"{ts_literal(key)}:{ts_frozen_array(value)}"
        for key, value in mapping.items()
    )
    return f"Object.freeze({{{fields}}})"


def ts_frozen_record_map(mapping: dict[str, dict[str, str]]) -> str:
    fields = ",".join(
        f"{ts_literal(key)}:Object.freeze({ts_literal(value)})"
        for key, value in mapping.items()
    )
    return f"Object.freeze({{{fields}}})"


def ts_frozen_shape_contract_map(registry: Registry) -> str:
    fields = ",".join(
        f"{ts_literal(registry.graph_names[contract.operator])}:Object.freeze({{"
        f"shapeFunctionId:{ts_literal(contract.shape_function_id)},"
        f"classification:{ts_literal(shape_classification_label(contract.classification))}"
        "})"
        for contract in registry.shape_contracts
    )
    return f"Object.freeze({{{fields}}})"


def ts_frozen_operator_kind_map(registry: Registry) -> str:
    fields = ",".join(
        f"{ts_literal(registry.graph_names[contract.operator])}:"
        f"{registry.operator_numbers[contract.operator]}"
        for contract in registry.shape_contracts
    )
    return f"Object.freeze({{{fields}}})"


def ts_frozen_variants(variants: list[dict[str, Any]]) -> str:
    items: list[str] = []
    for variant in variants:
        fields: list[str] = []
        for key, value in variant.items():
            rendered = ts_frozen_array(value) if key in {"operators", "requiredFeatures"} else ts_literal(value)
            fields.append(f"{key}:{rendered}")
        items.append("Object.freeze({" + ",".join(fields) + "})")
    return "Object.freeze([" + ",".join(items) + "])"


def render_ts(registry: Registry) -> bytes:
    backend_ids = [backend.runtime_id for backend in registry.backends]
    runtime_ops = {backend.runtime_id: graph_names(registry, backend.runtime_operators) for backend in registry.backends}
    routes = {
        backend.runtime_id: {
            registry.graph_names[operator]: backend.route_for(operator).route
            for operator in ordered_operators(registry, backend.runtime_operators)
        }
        for backend in registry.backends
    }
    targets = {
        backend.exporter_target: graph_names(registry, backend.qualified_operators)
        for backend in registry.backends if backend.exporter_target
    }
    profiles = {name: list(members) for name, members in registry.target_profiles}
    modes = {
        backend.runtime_id: backend.support_mode.removeprefix("RUNTIME_SUPPORT_MODE_").lower()
        for backend in registry.backends
    }
    variants = []
    for variant in registry.variants:
        if variant.profile != "KERNEL_PROFILE_INFERENCE":
            continue
        backend = next(item for item in registry.backends if item.kind == variant.backend)
        variants.append({
            "id": variant.id,
            "backend": backend.runtime_id,
            "operators": graph_names(registry, variant.operators),
            "phase": "forward",
            "entrypointId": variant.entrypoint_id,
            "predicateId": variant.predicate_id,
            "requiredFeatures": list(variant.required_features),
            "priority": variant.priority,
        })
    runtime_ops_literal = ts_frozen_array_map(runtime_ops)
    routes_literal = ts_frozen_record_map(routes)
    modes_literal = f"Object.freeze({ts_literal(modes)})"
    targets_literal = ts_frozen_array_map(targets)
    profiles_literal = ts_frozen_array_map(profiles)
    variants_literal = ts_frozen_variants(variants)
    shape_contracts_literal = ts_frozen_shape_contract_map(registry)
    shape_function_ids = {
        registry.graph_names[contract.operator]: contract.shape_function_id
        for contract in registry.shape_contracts
    }
    shape_function_ids_literal = (
        f"Object.freeze({ts_literal(shape_function_ids)} as const)"
    )
    shape_classification_values = [
        shape_classification_label(name)
        for name, number in sorted(
            registry.shape_classification_numbers.items(), key=lambda item: item[1]
        )
        if number != 0
    ]
    shape_classifications_literal = (
        f"Object.freeze({ts_literal(shape_classification_values)} as const)"
    )
    content = banner(registry, "//") + f"""
export const shapeContractClassifications = {shape_classifications_literal};
export type ShapeContractClassification = (typeof shapeContractClassifications)[number];

export interface GeneratedOperatorShapeContract {{
  readonly shapeFunctionId: string;
  readonly classification: ShapeContractClassification;
}}

export const operatorShapeContracts = {shape_contracts_literal} as
  Readonly<Record<string, GeneratedOperatorShapeContract>>;
export const operatorShapeFunctionIds = {shape_function_ids_literal};

export function operatorShapeContract(operator: string): GeneratedOperatorShapeContract | null {{
  return operatorShapeContracts[operator] ?? null;
}}

export const kernelBackends = Object.freeze({ts_literal(backend_ids)} as const);
export type KernelBackend = (typeof kernelBackends)[number];

export const runtimeOperatorsByBackend = {runtime_ops_literal} as
  Readonly<Record<KernelBackend, readonly string[]>>;
export const kernelRoutesByBackend = {routes_literal} as
  Readonly<Record<KernelBackend, Readonly<Record<string, string>>>>;
export const runtimeSupportModeByBackend = {modes_literal} as
  Readonly<Record<KernelBackend, 'direct' | 'dynamic'>>;
export const exporterQualifiedOperators = {targets_literal} as
  Readonly<Record<string, readonly string[]>>;
export const targetProfiles = {profiles_literal} as
  Readonly<Record<string, readonly string[]>>;

export const WASM_KERNEL_ROUTES: Readonly<Record<string, string>> =
  kernelRoutesByBackend.wasm;

export function runtimeSupportsOperator(backend: KernelBackend, operator: string): boolean {{
  return Object.prototype.hasOwnProperty.call(kernelRoutesByBackend[backend], operator);
}}

export function kernelRoute(backend: KernelBackend, operator: string): string | null {{
  return kernelRoutesByBackend[backend][operator] ?? null;
}}

export function exporterTargetSupports(target: string, operator: string): boolean {{
  return exporterQualifiedOperators[target]?.includes(operator) === true;
}}

export interface KernelVariant {{
  readonly id: string;
  readonly backend: KernelBackend;
  readonly operators: readonly string[];
  readonly phase: 'forward';
  readonly entrypointId: string;
  readonly predicateId: string;
  readonly requiredFeatures: readonly string[];
  readonly priority: number;
}}

export const kernelVariants = {variants_literal} as
  readonly KernelVariant[];
"""
    return content.encode()


def render_ts_runtime(registry: Registry) -> bytes:
    """Render the browser runtime's dependency-minimal registry projection.

    The general TypeScript projection remains the complete generated contract
    for tooling and inspection. Browser execution only consumes shared shape
    contracts plus the WASM/WebGPU runtime inventories and routes, so keeping
    native/exporter/variant tables out of this module lets esbuild omit them
    from the inference and full browser bundles.
    """
    backend_by_runtime_id = {
        backend.runtime_id: backend for backend in registry.backends
    }
    missing = [
        backend_id for backend_id in BROWSER_RUNTIME_BACKEND_IDS
        if backend_id not in backend_by_runtime_id
    ]
    if missing:
        raise ValueError(
            "browser runtime projection is missing backend(s): " + ", ".join(missing)
        )
    runtime_backends = [
        backend_by_runtime_id[backend_id]
        for backend_id in BROWSER_RUNTIME_BACKEND_IDS
    ]
    runtime_ops = {
        backend.runtime_id: graph_names(registry, backend.runtime_operators)
        for backend in runtime_backends
    }
    routes = {
        backend.runtime_id: {
            registry.graph_names[operator]: backend.route_for(operator).route
            for operator in ordered_operators(registry, backend.runtime_operators)
        }
        for backend in runtime_backends
    }
    shape_contracts_literal = ts_frozen_shape_contract_map(registry)
    operator_kinds_literal = ts_frozen_operator_kind_map(registry)
    shape_function_ids = {
        registry.graph_names[contract.operator]: contract.shape_function_id
        for contract in registry.shape_contracts
    }
    shape_function_ids_literal = (
        f"Object.freeze({ts_literal(shape_function_ids)} as const)"
    )
    shape_classification_values = [
        shape_classification_label(name)
        for name, number in sorted(
            registry.shape_classification_numbers.items(), key=lambda item: item[1]
        )
        if number != 0
    ]
    shape_classification_type = " | ".join(
        ts_literal(value) for value in shape_classification_values
    )
    runtime_ops_literal = ts_frozen_array_map(runtime_ops)
    routes_literal = ts_frozen_record_map(routes)
    content = banner(registry, "//") + f"""
export type ShapeContractClassification = {shape_classification_type};

export interface GeneratedOperatorShapeContract {{
  readonly shapeFunctionId: string;
  readonly classification: ShapeContractClassification;
}}

export const operatorShapeContracts = {shape_contracts_literal} as
  Readonly<Record<string, GeneratedOperatorShapeContract>>;
export const operatorShapeFunctionIds = {shape_function_ids_literal};
export const operatorKindByName = {operator_kinds_literal} as
  Readonly<Record<string, number>>;

export function operatorShapeContract(operator: string): GeneratedOperatorShapeContract | null {{
  return operatorShapeContracts[operator] ?? null;
}}

export type RuntimeKernelBackend = {" | ".join(ts_literal(value) for value in BROWSER_RUNTIME_BACKEND_IDS)};

export const runtimeOperatorsByBackend = {runtime_ops_literal} as
  Readonly<Record<RuntimeKernelBackend, readonly string[]>>;
const kernelRoutesByBackend = {routes_literal} as
  Readonly<Record<RuntimeKernelBackend, Readonly<Record<string, string>>>>;

export function kernelRoute(backend: RuntimeKernelBackend, operator: string): string | null {{
  return kernelRoutesByBackend[backend][operator] ?? null;
}}
"""
    return content.encode()


def render_ts_full(registry: Registry) -> bytes:
    variants = []
    for variant in registry.variants:
        if variant.profile != "KERNEL_PROFILE_FULL":
            continue
        backend = next(item for item in registry.backends if item.kind == variant.backend)
        variants.append({
            "id": variant.id,
            "backend": backend.runtime_id,
            "operators": graph_names(registry, variant.operators),
            "phase": variant.phase.removeprefix("KERNEL_PHASE_").lower(),
            "entrypointId": variant.entrypoint_id,
            "predicateId": variant.predicate_id,
            "requiredFeatures": list(variant.required_features),
            "priority": variant.priority,
        })
    return (banner(registry, "//") +
            "export const fullKernelVariants = " + ts_frozen_variants(variants) + ";\n").encode()


def c_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render_backend_vocabulary(registry: Registry) -> tuple[bytes, str]:
    lines = [banner(registry, "/*").replace("\n", " */\n").rstrip(),
        "#ifndef VOLVOXAI_BACKEND_VOCABULARY_H", "#define VOLVOXAI_BACKEND_VOCABULARY_H",
        "#include <stddef.h>", "#include <stdint.h>",
        "typedef int32_t VxBackendKind;", "enum {", "    VX_BACKEND_KIND_UNSPECIFIED = 0,"]
    lines.extend(f"    VX_{b.kind} = {b.kind_value}," for b in registry.backends)
    lines.append("};")
    lines.extend(f"#define VX_BACKEND_NAME_{b.kind.removeprefix('BACKEND_KIND_')} {c_string(b.provider_name)}" for b in registry.backends)
    lines.extend(["#if defined(__wasm__)",
        "#define VX_PORTABLE_BACKEND_KIND VX_BACKEND_KIND_WASM",
        "#define VX_PORTABLE_BACKEND_NAME VX_BACKEND_NAME_WASM", "#else",
        "#define VX_PORTABLE_BACKEND_KIND VX_BACKEND_KIND_NATIVE_CPU",
        "#define VX_PORTABLE_BACKEND_NAME VX_BACKEND_NAME_NATIVE_CPU", "#endif",
        "#ifdef __cplusplus", 'extern "C" {', "#endif",
        "VxBackendKind vx_backend_kind_from_name(const char* name);",
        "VxBackendKind vx_backend_kind_from_bytes(const void* data, size_t length);",
        "const char* vx_backend_kind_name(VxBackendKind kind);",
        "const char* vx_backend_kind_provider_id(VxBackendKind kind);",
        "#ifdef __cplusplus", "}", "#endif", "#endif", ""])
    rows = [f"    {{VX_{b.kind}, {c_string(b.provider_name)}, {len(b.provider_name)}, "
            f"{c_string('builtin:' + b.provider_name)}}}," for b in registry.backends]
    source = "\n".join([
        "", "static const struct {", "    VxBackendKind kind; const char* name; size_t length; const char* provider_id;",
        "} vx_backend_names[] = {", *rows, "};", "",
        "VxBackendKind vx_backend_kind_from_bytes(const void* data, size_t length) {",
        "    if (!data) return VX_BACKEND_KIND_UNSPECIFIED;",
        "    for (size_t i = 0; i < sizeof(vx_backend_names) / sizeof(vx_backend_names[0]); i++)",
        "        if (vx_backend_names[i].length == length && !memcmp(data, vx_backend_names[i].name, length))",
        "            return vx_backend_names[i].kind;",
        "    return VX_BACKEND_KIND_UNSPECIFIED;", "}", "",
        "VxBackendKind vx_backend_kind_from_name(const char* name) {",
        "    return name ? vx_backend_kind_from_bytes(name, strlen(name)) : VX_BACKEND_KIND_UNSPECIFIED;", "}", "",
        "const char* vx_backend_kind_name(VxBackendKind kind) {",
        "    for (size_t i = 0; i < sizeof(vx_backend_names) / sizeof(vx_backend_names[0]); i++)",
        "        if (vx_backend_names[i].kind == kind) return vx_backend_names[i].name;",
        "    return NULL;", "}", "",
        "const char* vx_backend_kind_provider_id(VxBackendKind kind) {",
        "    for (size_t i = 0; i < sizeof(vx_backend_names) / sizeof(vx_backend_names[0]); i++)",
        "        if (vx_backend_names[i].kind == kind) return vx_backend_names[i].provider_id;",
        "    return NULL;", "}", ""])
    return "\n".join(lines).encode(), source


def render_c(registry: Registry) -> tuple[bytes, bytes]:
    shape_classifications = [
        (name, number)
        for name, number in sorted(
            registry.shape_classification_numbers.items(), key=lambda item: item[1]
        )
        if number != 0
    ]
    registration_count = sum(len(backend.runtime_operators) for backend in registry.backends)
    variant_count = sum(len(variant.operators) for variant in registry.variants
                        if variant.profile == "KERNEL_PROFILE_INFERENCE")
    definitions = [
        banner(registry, "/*").replace("\n", " */\n").rstrip(),
        '#include "kernel_registry.h"',
        "",
        "const int32_t vx_generated_operator_kinds[] = {",
    ]
    lines = [
        banner(registry, "/*").replace("\n", " */\n").rstrip(),
        "#ifndef VOLVOXAI_INTERNAL_KERNEL_REGISTRY_H",
        "#define VOLVOXAI_INTERNAL_KERNEL_REGISTRY_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <string.h>",
        '#include "operator_vocabulary.h"',
        '#include "backend_vocabulary.h"',
        "",
        '#ifdef __cplusplus',
        'extern "C" {',
        '#endif',
        "",
        "typedef enum VxShapeContractClassification {",
    ]
    for classification, number in shape_classifications:
        lines.append(f"    VX_{classification} = {number},")
    lines.extend([
        "} VxShapeContractClassification;",
        "",
        f"extern const int32_t vx_generated_operator_kinds[{len(registry.shape_contracts)}];",
    ])
    for contract in registry.shape_contracts:
        definitions.append(f"    {registry.operator_numbers[contract.operator]},")
    definitions.extend([
        "};", "",
        "const VxGeneratedShapeContractRoute vx_generated_shape_contract_routes[] = {",
    ])
    lines.extend([
        "static const size_t vx_generated_operator_kind_count =",
        "    sizeof(vx_generated_operator_kinds) / sizeof(vx_generated_operator_kinds[0]);",
        "",
        "static inline int vx_generated_operator_kind_valid(int32_t operator_kind) {",
        "    size_t index;",
        "    for (index = 0; index < vx_generated_operator_kind_count; ++index) {",
        "        if (vx_generated_operator_kinds[index] == operator_kind) return 1;",
        "    }",
        "    return 0;",
        "}",
        "",
        "typedef struct VxGeneratedShapeContractRoute {",
        "    VxOperatorKind operator_kind; const char* operator_name;",
        "    const char* shape_function_id; VxShapeContractClassification classification;",
        "} VxGeneratedShapeContractRoute;",
        "",
        "extern const VxGeneratedShapeContractRoute "
        f"vx_generated_shape_contract_routes[{len(registry.shape_contracts)}];",
    ])
    for contract in registry.shape_contracts:
        definitions.append("    {" + ", ".join((
            str(registry.operator_numbers[contract.operator]),
            c_string(registry.graph_names[contract.operator]),
            c_string(contract.shape_function_id),
            f"VX_{contract.classification}",
        )) + "},")
    definitions.extend([
        "};", "",
        "const VxKernelRegistration vx_kernel_registrations[] = {",
    ])
    lines.extend([
        "static const size_t vx_generated_shape_contract_route_count =",
        "    sizeof(vx_generated_shape_contract_routes) / sizeof(vx_generated_shape_contract_routes[0]);",
        "",
        "static inline const VxGeneratedShapeContractRoute* vx_kernel_shape_contract_find(",
        "        VxOperatorKind operator_kind) {",
        "    size_t index;",
        "    if (operator_kind == VX_OP_UNSPECIFIED) return NULL;",
        "    for (index = 0; index < vx_generated_shape_contract_route_count; ++index) {",
        "        const VxGeneratedShapeContractRoute* item = &vx_generated_shape_contract_routes[index];",
        "        if (item->operator_kind == operator_kind) return item;",
        "    }",
        "    return NULL;",
        "}",
        "",
        "typedef struct VxKernelRegistration {",
        "    VxBackendKind backend; VxOperatorKind operator_kind; const char* operator_name;",
        "    const char* route; const char* predicate_id; int dynamic; int exporter_qualified;",
        "} VxKernelRegistration;",
        "",
        f"extern const VxKernelRegistration vx_kernel_registrations[{registration_count}];",
    ])
    for backend in registry.backends:
        qualified = set(backend.qualified_operators)
        dynamic = int(backend.support_mode == "RUNTIME_SUPPORT_MODE_DYNAMIC")
        for operator in ordered_operators(registry, backend.runtime_operators):
            route = backend.route_for(operator)
            definitions.append(
                "    {" + ", ".join((
                    "VX_" + backend.kind,
                    str(registry.operator_numbers[operator]),
                    c_string(registry.graph_names[operator]),
                    c_string(route.route),
                    c_string(route.predicate_id),
                    str(dynamic),
                    str(int(operator in qualified)),
                )) + "},"
            )
    definitions.extend([
        "};", "",
        "const VxKernelVariantRegistration vx_kernel_variants[] = {",
    ])
    lines.extend([
        "static const size_t vx_kernel_registration_count =",
        "    sizeof(vx_kernel_registrations) / sizeof(vx_kernel_registrations[0]);",
        "",
        "static inline const VxKernelRegistration* vx_kernel_registry_find(",
        "        VxBackendKind backend, VxOperatorKind operator_kind) {",
        "    size_t index;",
        "    if (!backend || operator_kind == VX_OP_UNSPECIFIED) return NULL;",
        "    for (index = 0; index < vx_kernel_registration_count; ++index) {",
        "        const VxKernelRegistration* item = &vx_kernel_registrations[index];",
        "        if (item->backend == backend && item->operator_kind == operator_kind) return item;",
        "    }",
        "    return NULL;",
        "}",
        "",
        "typedef struct VxKernelVariantRegistration {",
        "    const char* id; VxBackendKind backend; VxOperatorKind operator_kind;",
        "    const char* entrypoint_id; const char* predicate_id;",
        "    const char* required_features; int priority;",
        "} VxKernelVariantRegistration;",
        "",
        "extern const VxKernelVariantRegistration "
        f"vx_kernel_variants[{max(1, variant_count)}];",
    ])
    backend_by_kind = {backend.kind: backend for backend in registry.backends}
    for variant in registry.variants:
        if variant.profile != "KERNEL_PROFILE_INFERENCE":
            continue
        backend = backend_by_kind[variant.backend]
        features = ",".join(variant.required_features)
        for operator in ordered_operators(registry, variant.operators):
            definitions.append("    {" + ", ".join((
                c_string(variant.id), "VX_" + backend.kind,
                str(registry.operator_numbers[operator]), c_string(variant.entrypoint_id),
                c_string(variant.predicate_id), c_string(features), str(variant.priority),
            )) + "},")
    if not variant_count:
        definitions.append("    {NULL, VX_BACKEND_KIND_UNSPECIFIED, 0, NULL, NULL, NULL, 0},")
    definitions.extend(["};", ""])
    lines.extend([
        f"static const size_t vx_kernel_variant_count = {variant_count};",
        "",
        '#ifdef __cplusplus',
        '}',
        '#endif',
        "",
        "#endif /* VOLVOXAI_INTERNAL_KERNEL_REGISTRY_H */",
        "",
    ])
    return "\n".join(lines).encode(), "\n".join(definitions).encode()


def render_c_full(registry: Registry) -> tuple[bytes, bytes]:
    variants = [variant for variant in registry.variants if variant.profile == "KERNEL_PROFILE_FULL"]
    count = sum(len(variant.operators) for variant in variants)
    definitions = [
        banner(registry, "/*").replace("\n", " */\n").rstrip(),
        '#include "kernel_registry_full.h"',
        "",
        "const VxKernelVariantRegistration vx_full_kernel_variants[] = {",
    ]
    lines = [
        banner(registry, "/*").replace("\n", " */\n").rstrip(),
        "#ifndef VOLVOXAI_INTERNAL_KERNEL_REGISTRY_FULL_H",
        "#define VOLVOXAI_INTERNAL_KERNEL_REGISTRY_FULL_H",
        "",
        '#include "kernel_registry.h"',
        "",
        '#ifdef __cplusplus',
        'extern "C" {',
        '#endif',
        "",
        "extern const VxKernelVariantRegistration "
        f"vx_full_kernel_variants[{max(1, count)}];",
    ]
    backend_by_kind = {backend.kind: backend for backend in registry.backends}
    if variants:
        for variant in variants:
            backend = backend_by_kind[variant.backend]
            features = ",".join(variant.required_features)
            for operator in ordered_operators(registry, variant.operators):
                definitions.append("    {" + ", ".join((
                    c_string(variant.id), "VX_" + backend.kind,
                    str(registry.operator_numbers[operator]), c_string(variant.entrypoint_id),
                    c_string(variant.predicate_id), c_string(features), str(variant.priority),
                )) + "},")
    else:
        definitions.append("    {NULL, VX_BACKEND_KIND_UNSPECIFIED, 0, NULL, NULL, NULL, 0},")
    definitions.extend(["};", ""])
    lines.extend([
        f"static const size_t vx_full_kernel_variant_count = {count};",
        "",
        '#ifdef __cplusplus',
        '}',
        '#endif',
        "",
        "#endif /* VOLVOXAI_INTERNAL_KERNEL_REGISTRY_FULL_H */",
        "",
    ])
    return "\n".join(lines).encode(), "\n".join(definitions).encode()


def render_docs(registry: Registry) -> bytes:
    lines = [
        "<!-- DO NOT EDIT: generated by tools/generate_kernel_registry.py. -->",
        f"<!-- proto/kernel_registry.proto SHA-256: {registry.registry_hash} -->",
        f"<!-- proto/operator_vocabulary.proto SHA-256: {registry.vocabulary_hash} -->",
        "",
        "# Kernel registry",
        "",
        "`proto/kernel_registry.proto` is the internal source of truth for logical backend",
        "operator registrations, route IDs, strict exporter qualification, and the physical",
        "variants whose selection metadata has been migrated into the registry.",
        "",
        "A runtime registration is a route candidate, not a promise that every dtype, shape,",
        "attribute, or device is legal. `dynamic` routes may decline and permit runtime fallback.",
        "Exporter qualification is separate and requires correctness plus no-fallback evidence.",
        "Consequently, an empty accelerator qualification set is intentional.",
        "",
        "Physical-variant rows are canonical where present. Absence means physical selection",
        "is still owned by the named logical route; it must not be read as proof that only one",
        "machine kernel exists. Schema version 1 currently registers forward/inference variants",
        "only; profile-separated generated files prevent future full-only entries from leaking",
        "into inference artifacts.",
        "",
        "## Shape-contract routes",
        "",
        "Every runtime `OperatorKind` has exactly one stable logical shape-function route.",
        "`canonical` selects the authoritative contract for the current implementation tranche;",
        "it does not attest cross-language coverage, backend proof, or release qualification.",
        "`deferred` reserves an ID for a later tranche. `bounded-value-dependent` marks the",
        "fixed-capacity plus valid-count/padding convention, never compact data-dependent shape.",
        "",
        "| Operator | Shape-function ID | Classification |",
        "| --- | --- | --- |",
    ]
    for contract in registry.shape_contracts:
        lines.append(
            f"| `{registry.graph_names[contract.operator]}` | "
            f"`{contract.shape_function_id}` | "
            f"{shape_classification_label(contract.classification)} |"
        )
    lines.extend([
        "",
        "## Backends",
        "",
        "| Backend | Runtime mode | Registered operators | Exporter target | Qualified operators | Physical variants |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ])
    for backend in registry.backends:
        mode = backend.support_mode.removeprefix("RUNTIME_SUPPORT_MODE_").lower()
        variant_count = sum(1 for variant in registry.variants if variant.backend == backend.kind)
        lines.append(
            f"| `{backend.runtime_id}` | {mode} | {len(backend.runtime_operators)} | "
            f"`{backend.exporter_target}` | {len(backend.qualified_operators)} | {variant_count} |"
            if backend.exporter_target else
            f"| `{backend.runtime_id}` | {mode} | {len(backend.runtime_operators)} | — | 0 | {variant_count} |"
        )
    lines.extend(["", "## Exporter profiles", ""])
    for name, members in registry.target_profiles:
        lines.append(f"- `{name}`: " + ", ".join(f"`{member}`" for member in members))
    lines.extend([
        "",
        "## Operator matrix",
        "",
        "`R` is a direct registration, `D` is a dynamic/fallback-capable registration,",
        "and `Q` additionally marks strict exporter qualification.",
        "",
        "| Operator | " + " | ".join(f"`{backend.runtime_id}`" for backend in registry.backends) + " |",
        "| --- | " + " | ".join("---:" for _ in registry.backends) + " |",
    ])
    all_operators = ordered_operators(
        registry,
        set().union(*(set(backend.runtime_operators) for backend in registry.backends)),
    )
    for operator in all_operators:
        cells = []
        for backend in registry.backends:
            if operator not in backend.runtime_operators:
                cells.append("—")
            else:
                marker = "D" if backend.support_mode == "RUNTIME_SUPPORT_MODE_DYNAMIC" else "R"
                if operator in backend.qualified_operators:
                    marker += "+Q"
                cells.append(marker)
        lines.append(f"| `{registry.graph_names[operator]}` | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Logical routes",
        "",
    ])
    for backend in registry.backends:
        lines.extend([f"### `{backend.runtime_id}`", ""])
        if backend.default_route:
            lines.append(
                f"Default route: `{backend.default_route}`; predicate: "
                f"`{backend.default_predicate_id}`."
            )
        else:
            lines.append("Every registered operator has an explicit route.")
        if backend.routes:
            lines.extend([
                "",
                "| Operator | Route | Predicate |",
                "| --- | --- | --- |",
            ])
            for route in sorted(
                backend.routes,
                key=lambda item: registry.operator_numbers[item.operator],
            ):
                lines.append(
                    f"| `{registry.graph_names[route.operator]}` | `{route.route}` | "
                    f"`{route.predicate_id}` |"
                )
        lines.append("")
    lines.extend([
        "## Physical variants",
        "",
        "| ID | Backend | Operators | Entrypoint | Predicate | Features | Priority |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ])
    backend_by_kind = {backend.kind: backend for backend in registry.backends}
    for variant in sorted(registry.variants, key=lambda item: (backend_by_kind[item.backend].runtime_id, -item.priority, item.id)):
        ops = ", ".join(f"`{name}`" for name in graph_names(registry, variant.operators))
        features = ", ".join(f"`{feature}`" for feature in variant.required_features) or "—"
        lines.append(
            f"| `{variant.id}` | `{backend_by_kind[variant.backend].runtime_id}` | {ops} | "
            f"`{variant.entrypoint_id}` | `{variant.predicate_id}` | {features} | {variant.priority} |"
        )
    lines.append("")
    return "\n".join(lines).encode()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def protoc_check(registry_proto: Path, proto_root: Path, protoc: str) -> None:
    executable = shutil.which(protoc) if not Path(protoc).is_file() else protoc
    if not executable:
        raise ValueError(f"protoc executable not found: {protoc}")
    with tempfile.TemporaryDirectory(prefix="volvoxai-kernel-registry-") as directory:
        result = subprocess.run(
            [executable, f"--proto_path={proto_root}",
             f"--descriptor_set_out={Path(directory) / 'registry.pb'}",
             str(registry_proto)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    if result.returncode:
        raise ValueError("protoc validation failed:\n" + result.stderr.rstrip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--protoc-check", action="store_true")
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument("--registry-proto", type=Path, default=ROOT / "proto/kernel_registry.proto")
    parser.add_argument(
        "--operator-vocabulary-proto", type=Path,
        default=ROOT / "proto/operator_vocabulary.proto",
    )
    parser.add_argument("--python-out", type=Path, default=ROOT / "tools/exporter/generated/kernel_registry.py")
    parser.add_argument("--python-full-out", type=Path, default=ROOT / "tools/exporter/generated/kernel_registry_full.py")
    parser.add_argument("--typescript-out", type=Path, default=ROOT / "tools/generated/kernelRegistry.ts")
    parser.add_argument("--typescript-runtime-out", type=Path, default=ROOT / "tools/generated/kernelRegistryRuntime.ts")
    parser.add_argument("--typescript-full-out", type=Path, default=ROOT / "tools/generated/kernelRegistryFull.ts")
    parser.add_argument("--native-out", type=Path, default=ROOT / "native/src/generated/kernel_registry.h")
    parser.add_argument("--native-source-out", type=Path, default=ROOT / "native/src/generated/kernel_registry.c")
    parser.add_argument("--native-full-out", type=Path, default=ROOT / "native/src/generated/kernel_registry_full.h")
    parser.add_argument("--native-full-source-out", type=Path, default=ROOT / "native/src/generated/kernel_registry_full.c")
    parser.add_argument("--docs-out", type=Path, default=ROOT / "docs/generated/kernel-registry.md")
    args = parser.parse_args()

    try:
        registry_proto = args.registry_proto.resolve()
        operator_vocabulary_proto = args.operator_vocabulary_proto.resolve()
        registry = load_registry(registry_proto, operator_vocabulary_proto)
        if args.protoc_check:
            protoc_check(registry_proto, registry_proto.parent, args.protoc)
        backend_header, backend_source = render_backend_vocabulary(registry)
        native_header, native_source = render_c(registry)
        native_source += backend_source.encode()
        native_full_header, native_full_source = render_c_full(registry)
        outputs = {
            args.python_out.resolve(): render_python(registry),
            args.python_full_out.resolve(): render_python_full(registry),
            args.typescript_out.resolve(): render_ts(registry),
            args.typescript_runtime_out.resolve(): render_ts_runtime(registry),
            args.typescript_full_out.resolve(): render_ts_full(registry),
            args.native_out.resolve().with_name("backend_vocabulary.h"): backend_header,
            args.native_out.resolve(): native_header,
            args.native_source_out.resolve(): native_source,
            args.native_full_out.resolve(): native_full_header,
            args.native_full_source_out.resolve(): native_full_source,
            args.docs_out.resolve(): render_docs(registry),
        }
        inference_variant_tokens = {
            token
            for variant in registry.variants
            if variant.profile == "KERNEL_PROFILE_INFERENCE"
            for token in (variant.id, variant.entrypoint_id, variant.predicate_id)
        }
        full_only_variant_tokens = {
            token
            for variant in registry.variants
            if variant.profile == "KERNEL_PROFILE_FULL"
            for token in (variant.id, variant.entrypoint_id, variant.predicate_id)
        } - inference_variant_tokens
        inference_outputs = (
            outputs[args.python_out.resolve()] + outputs[args.typescript_out.resolve()] +
            outputs[args.typescript_runtime_out.resolve()] +
            outputs[args.native_out.resolve()] + outputs[args.native_source_out.resolve()]
        )
        for token in full_only_variant_tokens:
            if token.encode() in inference_outputs:
                raise ValueError(f"full-only variant metadata leaked into inference output: {token}")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"kernel registry generation failed: {error}", file=sys.stderr)
        return 1

    stale = [path for path, data in outputs.items() if not path.is_file() or path.read_bytes() != data]
    if args.check:
        if stale:
            for path in stale:
                print(f"stale generated kernel registry: {path}", file=sys.stderr)
            print("run python3 tools/generate_kernel_registry.py", file=sys.stderr)
            return 1
        print(f"Verified {len(outputs)} generated kernel registry files")
        return 0
    for path, data in outputs.items():
        write_atomic(path, data)
        print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
