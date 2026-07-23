#!/usr/bin/env python3
"""Generate checked views of the internal protobuf optimizer registry.

The normal and ``--check`` paths use only the Python standard library.  The
authored protobuf file keeps schema and inventory together in a custom file
option; executable pass matchers remain ordinary code referenced by stable
implementation IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from generate_kernel_registry import (
        AggregateParser,
        ROOT,
        load_registry as load_kernel_registry,
        parse_declared_enum,
        reject_unknown,
        scalar,
        tokenize,
        unique,
        values,
    )
except ImportError:  # Imported as tools.generate_optimizer_registry in tests.
    from tools.generate_kernel_registry import (
        AggregateParser,
        ROOT,
        load_registry as load_kernel_registry,
        parse_declared_enum,
        reject_unknown,
        scalar,
        tokenize,
        unique,
        values,
    )


REGISTRY_OPTION = "volvoxai.optimizer_registry.registry"


@dataclass(frozen=True)
class Feature:
    id: str
    summary: str


@dataclass(frozen=True)
class TargetRule:
    backends: tuple[str, ...]
    required_operators: tuple[str, ...]
    required_kernel_variants: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerPass:
    id: str
    version: int
    implementation_id: str
    summary: str
    scope: str
    stage: str
    input_dialects: tuple[str, ...]
    output_dialect: str
    repeatable: bool
    semantics: str
    requires_calibration: bool
    changes_public_abi: bool
    changes_precision: bool
    matched_operators: tuple[str, ...]
    emitted_operators: tuple[str, ...]
    must_run_after: tuple[str, ...]
    must_run_before: tuple[str, ...]
    conflicts_with: tuple[str, ...]
    target_rules: tuple[TargetRule, ...]


@dataclass(frozen=True)
class PassGroup:
    id: str
    summary: str
    pass_ids: tuple[str, ...]
    fixed_point: bool
    max_iterations: int


@dataclass(frozen=True)
class PipelineGroupOverlay:
    group_id: str
    required_features: tuple[str, ...]
    forbidden_features: tuple[str, ...]


@dataclass(frozen=True)
class PipelineRecipe:
    id: str
    version: int
    summary: str
    group_overlays: tuple[PipelineGroupOverlay, ...]
    targets: tuple[str, ...]
    allowed_semantics: tuple[str, ...]
    allow_calibration: bool
    allow_public_abi_change: bool
    required_features: tuple[str, ...]
    supported_features: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerRegistry:
    schema_version: int
    features: tuple[Feature, ...]
    passes: tuple[OptimizerPass, ...]
    groups: tuple[PassGroup, ...]
    pipelines: tuple[PipelineRecipe, ...]
    operator_numbers: dict[str, int]
    operator_graph_names: dict[str, str]
    backend_runtime_ids: dict[str, str]
    enum_numbers: dict[str, dict[str, int]]
    registry_hash: str
    public_hash: str
    kernel_registry_hash: str


def parse_registry_option(source: str) -> dict[str, list[Any]]:
    tokens = tokenize(source)
    for index in range(len(tokens) - 5):
        prefix = [token.value for token in tokens[index:index + 6]]
        if prefix == ["option", "(", REGISTRY_OPTION, ")", "=", "{"]:
            parser = AggregateParser(tokens, index + 5)
            result = parser.message()
            parser.take(";")
            return result
    raise ValueError(f"missing option ({REGISTRY_OPTION})")


def _enum(
    token: str,
    declared: dict[str, int],
    where: str,
) -> str:
    if token not in declared or declared[token] == 0:
        raise ValueError(f"{where} has invalid enum value {token!r}")
    return token


def _nonempty_unique_strings(
    message: dict[str, list[Any]], field: str, where: str,
) -> tuple[str, ...]:
    result = unique(values(message, field, str, where), f"{field} in {where}")
    if any(not value for value in result):
        raise ValueError(f"{where}.{field} contains an empty value")
    return result


def _check_dependency_cycles(passes: tuple[OptimizerPass, ...]) -> None:
    edges: dict[str, set[str]] = {item.id: set() for item in passes}
    for item in passes:
        for dependency in item.must_run_after:
            edges[dependency].add(item.id)
        for successor in item.must_run_before:
            edges[item.id].add(successor)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(pass_id: str) -> None:
        if pass_id in visiting:
            raise ValueError(f"optimizer pass ordering contains a cycle at {pass_id!r}")
        if pass_id in visited:
            return
        visiting.add(pass_id)
        for successor in edges[pass_id]:
            visit(successor)
        visiting.remove(pass_id)
        visited.add(pass_id)

    for pass_id in edges:
        visit(pass_id)


def load_registry(
    registry_proto: Path,
    public_proto: Path,
    kernel_registry_proto: Path,
) -> OptimizerRegistry:
    registry_bytes = registry_proto.read_bytes()
    public_bytes = public_proto.read_bytes()
    source = registry_bytes.decode("utf-8")
    root = parse_registry_option(source)
    reject_unknown(
        root,
        {"schema_version", "feature", "pass", "group", "pipeline"},
        "registry",
    )

    schema_version = scalar(root, "schema_version", int, "registry")
    if schema_version != 1:
        raise ValueError(f"unsupported optimizer registry schema_version {schema_version}")

    kernel = load_kernel_registry(kernel_registry_proto, public_proto)
    operator_numbers = kernel.operator_numbers
    graph_names = kernel.graph_names
    backend_by_kind = {backend.kind: backend for backend in kernel.backends}
    variant_by_id = {variant.id: variant for variant in kernel.variants}

    enum_numbers = {
        name: parse_declared_enum(source, name)
        for name in (
            "OptimizerPassScope",
            "OptimizerDialect",
            "OptimizerPassStage",
            "RewriteSemantics",
        )
    }

    features: list[Feature] = []
    feature_ids: set[str] = set()
    for index, item in enumerate(values(root, "feature", dict, "registry")):
        where = f"feature[{index}]"
        reject_unknown(item, {"id", "summary"}, where)
        feature_id = scalar(item, "id", str, where)
        summary = scalar(item, "summary", str, where)
        if not feature_id or not summary or feature_id in feature_ids:
            raise ValueError(f"duplicate or empty feature declaration at {where}")
        feature_ids.add(feature_id)
        features.append(Feature(feature_id, summary))
    if not features:
        raise ValueError("optimizer registry must declare at least one feature")

    pass_fields = {
        "id", "version", "implementation_id", "summary", "scope", "stage",
        "input_dialect", "output_dialect", "repeatable", "semantics",
        "requires_calibration", "changes_public_abi", "changes_precision",
        "matched_operator", "emitted_operator",
        "must_run_after", "must_run_before", "conflicts_with", "target",
    }
    target_fields = {
        "backend", "required_operator", "required_kernel_variant_id",
    }
    passes: list[OptimizerPass] = []
    pass_ids: set[str] = set()
    implementation_ids: set[str] = set()
    for index, item in enumerate(values(root, "pass", dict, "registry")):
        where = f"pass[{index}]"
        reject_unknown(item, pass_fields, where)
        pass_id = scalar(item, "id", str, where)
        version = scalar(item, "version", int, where)
        if version != 1:
            raise ValueError(f"{where} requires optimizer pass version 1")
        implementation_id = scalar(item, "implementation_id", str, where)
        summary = scalar(item, "summary", str, where)
        scope = _enum(
            scalar(item, "scope", str, where),
            enum_numbers["OptimizerPassScope"], f"{where}.scope",
        )
        stage = _enum(
            scalar(item, "stage", str, where),
            enum_numbers["OptimizerPassStage"], f"{where}.stage",
        )
        input_dialects = unique(
            (
                _enum(value, enum_numbers["OptimizerDialect"], f"{where}.input_dialect")
                for value in values(item, "input_dialect", str, where)
            ),
            f"input dialect in {where}",
        )
        output_dialect = _enum(
            scalar(item, "output_dialect", str, where),
            enum_numbers["OptimizerDialect"], f"{where}.output_dialect",
        )
        repeatable = scalar(item, "repeatable", bool, where, required=False, default=False)
        semantics = _enum(
            scalar(item, "semantics", str, where),
            enum_numbers["RewriteSemantics"], f"{where}.semantics",
        )
        requires_calibration = scalar(
            item, "requires_calibration", bool, where, required=False, default=False,
        )
        changes_public_abi = scalar(
            item, "changes_public_abi", bool, where, required=False, default=False,
        )
        changes_precision = scalar(
            item, "changes_precision", bool, where, required=False, default=False,
        )
        matched = _nonempty_unique_strings(item, "matched_operator", where)
        emitted = _nonempty_unique_strings(item, "emitted_operator", where)
        invalid_operators = [
            operator for operator in (*matched, *emitted)
            if operator not in operator_numbers
        ]
        if invalid_operators:
            raise ValueError(f"{where} references unknown operator(s): {invalid_operators}")

        must_after = _nonempty_unique_strings(item, "must_run_after", where)
        must_before = _nonempty_unique_strings(item, "must_run_before", where)
        conflicts = _nonempty_unique_strings(item, "conflicts_with", where)

        target_rules: list[TargetRule] = []
        targeted_backends: set[str] = set()
        for target_index, target in enumerate(values(item, "target", dict, where)):
            target_where = f"{where}.target[{target_index}]"
            reject_unknown(target, target_fields, target_where)
            backends = unique(
                (
                    _enum(value, {key: backend.kind_value for key, backend in backend_by_kind.items()} | {"BACKEND_KIND_UNSPECIFIED": 0}, f"{target_where}.backend")
                    for value in values(target, "backend", str, target_where)
                ),
                f"backend in {target_where}",
            )
            required_operators = _nonempty_unique_strings(
                target, "required_operator", target_where,
            )
            required_variants = _nonempty_unique_strings(
                target, "required_kernel_variant_id", target_where,
            )
            if not backends:
                raise ValueError(f"{target_where} requires backend IDs")
            repeated = targeted_backends & set(backends)
            if repeated:
                raise ValueError(f"{where} has duplicate target backend rules: {sorted(repeated)}")
            targeted_backends.update(backends)
            unknown_operators = [
                operator for operator in required_operators
                if operator not in operator_numbers
            ]
            if unknown_operators:
                raise ValueError(
                    f"{target_where} references unknown operator(s): {unknown_operators}"
                )
            if emitted and not set(emitted) <= set(required_operators):
                raise ValueError(
                    f"{target_where} must require every emitted operator; "
                    f"missing={sorted(set(emitted) - set(required_operators))}"
                )
            for backend_kind in backends:
                backend = backend_by_kind[backend_kind]
                admitted = (
                    backend.qualified_operators
                    if scope == "OPTIMIZER_PASS_SCOPE_PORTABLE_PACKAGE"
                    else backend.runtime_operators
                )
                unsupported = sorted(set(required_operators) - set(admitted))
                if unsupported:
                    support = "exporter qualification" if scope.endswith("PORTABLE_PACKAGE") else "runtime inventory"
                    raise ValueError(
                        f"{target_where} requires {unsupported} absent from "
                        f"{backend.runtime_id} {support}"
                    )
            for variant_id in required_variants:
                variant = variant_by_id.get(variant_id)
                if variant is None:
                    raise ValueError(f"{target_where} references unknown kernel variant {variant_id!r}")
                if variant.backend not in backends:
                    raise ValueError(
                        f"{target_where} variant {variant_id!r} belongs to "
                        f"{variant.backend}, outside its backend list"
                    )
            target_rules.append(TargetRule(
                backends, required_operators, required_variants,
            ))

        if (
            not pass_id or not implementation_id or not summary
            or not input_dialects
        ):
            raise ValueError(f"{where} requires non-empty IDs, summary, and dialects")
        if pass_id in pass_ids:
            raise ValueError(f"duplicate optimizer pass id {pass_id!r}")
        if implementation_id in implementation_ids:
            raise ValueError(f"duplicate optimizer implementation id {implementation_id!r}")
        pass_ids.add(pass_id)
        implementation_ids.add(implementation_id)
        passes.append(OptimizerPass(
            pass_id, version, implementation_id, summary, scope, stage,
            input_dialects, output_dialect, repeatable, semantics,
            requires_calibration, changes_public_abi, changes_precision,
            matched, emitted, must_after, must_before, conflicts,
            tuple(target_rules),
        ))

    if not passes:
        raise ValueError("optimizer registry must declare at least one pass")
    pass_by_id = {item.id: item for item in passes}
    for item in passes:
        references = set(item.must_run_after) | set(item.must_run_before) | set(item.conflicts_with)
        unknown = sorted(references - pass_ids)
        if unknown:
            raise ValueError(f"pass {item.id!r} references unknown passes: {unknown}")
        if item.id in references:
            raise ValueError(f"pass {item.id!r} has a self dependency or conflict")
        contradictory = set(item.must_run_after) & set(item.must_run_before)
        if contradictory:
            raise ValueError(f"pass {item.id!r} has contradictory ordering: {sorted(contradictory)}")
    _check_dependency_cycles(tuple(passes))

    groups: list[PassGroup] = []
    group_ids: set[str] = set()
    for index, item in enumerate(values(root, "group", dict, "registry")):
        where = f"group[{index}]"
        reject_unknown(item, {"id", "summary", "pass_id", "fixed_point", "max_iterations"}, where)
        group_id = scalar(item, "id", str, where)
        summary = scalar(item, "summary", str, where)
        selected = _nonempty_unique_strings(item, "pass_id", where)
        fixed_point = scalar(item, "fixed_point", bool, where, required=False, default=False)
        max_iterations = scalar(item, "max_iterations", int, where)
        unknown = sorted(set(selected) - pass_ids)
        if unknown:
            raise ValueError(f"{where} references unknown passes: {unknown}")
        if not group_id or not summary or not selected or group_id in group_ids:
            raise ValueError(f"duplicate or empty pass group at {where}")
        if max_iterations <= 0 or (not fixed_point and max_iterations != 1):
            raise ValueError(f"{where} has invalid fixed-point iteration policy")
        nonrepeatable = [pass_id for pass_id in selected if not pass_by_id[pass_id].repeatable]
        if fixed_point and nonrepeatable:
            raise ValueError(f"{where} repeats non-repeatable passes: {nonrepeatable}")
        group_ids.add(group_id)
        groups.append(PassGroup(group_id, summary, selected, fixed_point, max_iterations))
    if not groups:
        raise ValueError("optimizer registry must declare at least one pass group")
    group_by_id = {item.id: item for item in groups}

    semantics_enum = enum_numbers["RewriteSemantics"]
    backend_enum = {
        key: backend.kind_value for key, backend in backend_by_kind.items()
    } | {"BACKEND_KIND_UNSPECIFIED": 0}
    pipelines: list[PipelineRecipe] = []
    pipeline_ids: set[str] = set()

    def overlays_can_co_select(
        recipe_required: frozenset[str],
        left: PipelineGroupOverlay,
        right: PipelineGroupOverlay,
    ) -> bool:
        required = recipe_required | frozenset(left.required_features) | frozenset(
            right.required_features
        )
        forbidden = frozenset(left.forbidden_features) | frozenset(
            right.forbidden_features
        )
        return not required & forbidden

    for index, item in enumerate(values(root, "pipeline", dict, "registry")):
        where = f"pipeline[{index}]"
        reject_unknown(
            item,
            {
                "id", "version", "summary", "group_overlay", "target",
                "allowed_semantics", "allow_calibration",
                "allow_public_abi_change", "required_feature", "supported_feature",
            },
            where,
        )
        pipeline_id = scalar(item, "id", str, where)
        version = scalar(item, "version", int, where)
        if version != 1:
            raise ValueError(
                f"{where} requires optimizer pipeline recipe version 1"
            )
        summary = scalar(item, "summary", str, where)
        targets = unique(
            (
                _enum(value, backend_enum, f"{where}.target")
                for value in values(item, "target", str, where)
            ),
            f"target in {where}",
        )
        allowed_semantics = unique(
            (
                _enum(value, semantics_enum, f"{where}.allowed_semantics")
                for value in values(item, "allowed_semantics", str, where)
            ),
            f"allowed semantics in {where}",
        )
        allow_calibration = scalar(
            item, "allow_calibration", bool, where, required=False, default=False,
        )
        allow_abi = scalar(
            item, "allow_public_abi_change", bool, where,
            required=False, default=False,
        )
        required_features = _nonempty_unique_strings(
            item, "required_feature", where,
        )
        supported_features = _nonempty_unique_strings(
            item, "supported_feature", where,
        )
        required_feature_set = frozenset(required_features)
        supported_feature_set = frozenset(supported_features)
        overlapping_domain = sorted(required_feature_set & supported_feature_set)
        if overlapping_domain:
            raise ValueError(
                f"{where} both requires and optionally supports features: "
                f"{overlapping_domain}"
            )
        unknown_recipe_features = sorted(
            (required_feature_set | supported_feature_set) - feature_ids
        )
        if unknown_recipe_features:
            raise ValueError(
                f"{where} references unknown features: {unknown_recipe_features}"
            )
        admitted_feature_set = required_feature_set | supported_feature_set

        overlays: list[PipelineGroupOverlay] = []
        for overlay_index, raw_overlay in enumerate(
            values(item, "group_overlay", dict, where)
        ):
            overlay_where = f"{where}.group_overlay[{overlay_index}]"
            reject_unknown(
                raw_overlay,
                {"group_id", "required_feature", "forbidden_feature"},
                overlay_where,
            )
            group_id = scalar(raw_overlay, "group_id", str, overlay_where)
            overlay_required = _nonempty_unique_strings(
                raw_overlay, "required_feature", overlay_where,
            )
            overlay_forbidden = _nonempty_unique_strings(
                raw_overlay, "forbidden_feature", overlay_where,
            )
            overlap = sorted(set(overlay_required) & set(overlay_forbidden))
            if overlap:
                raise ValueError(
                    f"{overlay_where} both requires and forbids features: {overlap}"
                )
            unknown_features = sorted(
                (set(overlay_required) | set(overlay_forbidden))
                - admitted_feature_set
            )
            if unknown_features:
                raise ValueError(
                    f"{overlay_where} references unsupported features: "
                    f"{unknown_features}"
                )
            if group_id not in group_by_id:
                raise ValueError(
                    f"{overlay_where} references unknown group {group_id!r}"
                )
            if (required_feature_set | set(overlay_required)) & set(
                overlay_forbidden
            ):
                raise ValueError(f"{overlay_where} can never be selected")
            overlays.append(PipelineGroupOverlay(
                group_id,
                overlay_required,
                overlay_forbidden,
            ))

        if (
            not pipeline_id or pipeline_id in pipeline_ids
            or not summary or not overlays or not allowed_semantics
        ):
            raise ValueError(f"duplicate or incomplete pipeline recipe at {where}")

        referenced_features = {
            feature
            for overlay in overlays
            for feature in (*overlay.required_features, *overlay.forbidden_features)
        }
        unused_optional_features = sorted(
            supported_feature_set - referenced_features
        )
        if unused_optional_features:
            raise ValueError(
                f"{where} supports optional features with no group overlay: "
                f"{unused_optional_features}"
            )

        occurrences: list[tuple[int, str, PipelineGroupOverlay]] = []
        position = 0
        for overlay in overlays:
            for pass_id in group_by_id[overlay.group_id].pass_ids:
                descriptor = pass_by_id[pass_id]
                occurrences.append((position, pass_id, overlay))
                position += 1
                if (
                    descriptor.semantics != "REWRITE_SEMANTICS_EXACT"
                    or descriptor.requires_calibration
                    or descriptor.changes_public_abi
                ) and not required_feature_set and not overlay.required_features:
                    raise ValueError(
                        f"{where} exposes policy-sensitive pass {pass_id!r} "
                        "without a required recipe or group feature"
                    )
                if descriptor.semantics not in allowed_semantics:
                    raise ValueError(
                        f"{where} does not allow semantics of pass {pass_id!r}"
                    )
                if descriptor.requires_calibration and not allow_calibration:
                    raise ValueError(
                        f"{where} forbids calibration required by {pass_id!r}"
                    )
                if descriptor.changes_public_abi and not allow_abi:
                    raise ValueError(
                        f"{where} forbids public ABI change by {pass_id!r}"
                    )
                if targets and descriptor.target_rules:
                    supported_targets = {
                        backend
                        for rule in descriptor.target_rules
                        for backend in rule.backends
                    }
                    missing_targets = sorted(set(targets) - supported_targets)
                    if missing_targets:
                        raise ValueError(
                            f"{where} targets {missing_targets} unsupported by "
                            f"pass {pass_id!r}"
                        )

        for left_index, (left_position, left_id, left_overlay) in enumerate(
            occurrences
        ):
            left = pass_by_id[left_id]
            for right_position, right_id, right_overlay in occurrences[
                left_index + 1:
            ]:
                if not overlays_can_co_select(
                    required_feature_set,
                    left_overlay,
                    right_overlay,
                ):
                    continue
                if left_id == right_id:
                    raise ValueError(
                        f"{where} can select pass {left_id!r} more than once"
                    )
                right = pass_by_id[right_id]
                if right_id in left.conflicts_with or left_id in right.conflicts_with:
                    raise ValueError(
                        f"{where} can select conflicting passes {left_id!r} and "
                        f"{right_id!r}"
                    )
                if right_id in left.must_run_after:
                    raise ValueError(
                        f"{where} orders {left_id!r} before required predecessor "
                        f"{right_id!r}"
                    )
                if left_id in right.must_run_before:
                    raise ValueError(
                        f"{where} orders {right_id!r} after required successor "
                        f"{left_id!r}"
                    )
                assert left_position < right_position

        pipeline_ids.add(pipeline_id)
        pipelines.append(PipelineRecipe(
            pipeline_id,
            version,
            summary,
            tuple(overlays),
            targets,
            allowed_semantics,
            allow_calibration,
            allow_abi,
            required_features,
            supported_features,
        ))
    if not pipelines:
        raise ValueError("optimizer registry must declare at least one pipeline recipe")
    used_feature_ids = {
        feature_id
        for pipeline in pipelines
        for feature_id in (
            *pipeline.required_features,
            *pipeline.supported_features,
        )
    }
    unused_feature_ids = sorted(feature_ids - used_feature_ids)
    if unused_feature_ids:
        raise ValueError(
            "optimizer registry declares features unused by every pipeline recipe: "
            f"{unused_feature_ids}"
        )
    for left_index, left in enumerate(pipelines):
        for right in pipelines[left_index + 1:]:
            required = set(left.required_features) | set(right.required_features)
            left_domain = set(left.required_features) | set(left.supported_features)
            right_domain = set(right.required_features) | set(
                right.supported_features
            )
            common_support = left_domain & right_domain
            if required <= common_support:
                raise ValueError(
                    f"pipeline recipes {left.id!r} and {right.id!r} ambiguously "
                    f"match features {sorted(required)!r}"
                )

    return OptimizerRegistry(
        schema_version=schema_version,
        features=tuple(features),
        passes=tuple(passes),
        groups=tuple(groups),
        pipelines=tuple(pipelines),
        operator_numbers=operator_numbers,
        operator_graph_names=graph_names,
        backend_runtime_ids={kind: backend.runtime_id for kind, backend in backend_by_kind.items()},
        enum_numbers=enum_numbers,
        registry_hash=hashlib.sha256(registry_bytes).hexdigest(),
        public_hash=hashlib.sha256(public_bytes).hexdigest(),
        kernel_registry_hash=kernel.registry_hash,
    )


def _banner(registry: OptimizerRegistry, marker: str) -> str:
    return (
        f"{marker} DO NOT EDIT: generated by tools/generate_optimizer_registry.py.\n"
        f"{marker} proto/optimizer_registry.proto SHA-256: {registry.registry_hash}\n"
        f"{marker} proto/volvoxai.proto SHA-256: {registry.public_hash}\n"
        f"{marker} proto/kernel_registry.proto SHA-256: {registry.kernel_registry_hash}\n"
    )


def _short(token: str, prefix: str) -> str:
    return token.removeprefix(prefix).lower().replace("_", "-")


def _operators(registry: OptimizerRegistry, values_: Iterable[str]) -> tuple[str, ...]:
    ordered = sorted(values_, key=registry.operator_numbers.__getitem__)
    return tuple(registry.operator_graph_names[value] for value in ordered)


def _python_expression(value: Any) -> str:
    if isinstance(value, dict):
        body = ", ".join(
            f"{key!r}: {_python_expression(nested)}"
            for key, nested in value.items()
        )
        return f"MappingProxyType({{{body}}})"
    if isinstance(value, tuple):
        if not value:
            return "()"
        body = ", ".join(_python_expression(item) for item in value)
        suffix = "," if len(value) == 1 else ""
        return f"({body}{suffix})"
    return repr(value)


def _target_rule_dict(
    registry: OptimizerRegistry, rule: TargetRule,
) -> dict[str, Any]:
    return {
        "backends": tuple(registry.backend_runtime_ids[value] for value in rule.backends),
        "required_operators": _operators(registry, rule.required_operators),
        "required_kernel_variant_ids": rule.required_kernel_variants,
    }


def render_python(registry: OptimizerRegistry) -> bytes:
    features = {item.id: item.summary for item in registry.features}
    passes: dict[str, Any] = {}
    for item in registry.passes:
        passes[item.id] = {
            "version": item.version,
            "implementation_id": item.implementation_id,
            "summary": item.summary,
            "scope": _short(item.scope, "OPTIMIZER_PASS_SCOPE_"),
            "stage": _short(item.stage, "OPTIMIZER_PASS_STAGE_"),
            "input_dialects": tuple(
                _short(value, "OPTIMIZER_DIALECT_") for value in item.input_dialects
            ),
            "output_dialect": _short(item.output_dialect, "OPTIMIZER_DIALECT_"),
            "repeatable": item.repeatable,
            "semantics": _short(item.semantics, "REWRITE_SEMANTICS_"),
            "requires_calibration": item.requires_calibration,
            "changes_public_abi": item.changes_public_abi,
            "changes_precision": item.changes_precision,
            "matched_operators": _operators(registry, item.matched_operators),
            "emitted_operators": _operators(registry, item.emitted_operators),
            "must_run_after": item.must_run_after,
            "must_run_before": item.must_run_before,
            "conflicts_with": item.conflicts_with,
            "target_rules": tuple(
                _target_rule_dict(registry, rule) for rule in item.target_rules
            ),
        }
    groups = {
        item.id: {
            "summary": item.summary,
            "passes": item.pass_ids,
            "fixed_point": item.fixed_point,
            "max_iterations": item.max_iterations,
        }
        for item in registry.groups
    }
    pipelines = {
        item.id: {
            "version": item.version,
            "summary": item.summary,
            "group_overlays": tuple({
                "group": overlay.group_id,
                "required_features": overlay.required_features,
                "forbidden_features": overlay.forbidden_features,
            } for overlay in item.group_overlays),
            "targets": tuple(
                registry.backend_runtime_ids[value] for value in item.targets
            ),
            "allowed_semantics": tuple(
                _short(value, "REWRITE_SEMANTICS_")
                for value in item.allowed_semantics
            ),
            "allow_calibration": item.allow_calibration,
            "allow_public_abi_change": item.allow_public_abi_change,
            "required_features": item.required_features,
            "supported_features": item.supported_features,
        }
        for item in registry.pipelines
    }
    lines = [
        _banner(registry, "#"),
        "from types import MappingProxyType",
        "",
        f"SCHEMA_VERSION = {registry.schema_version}",
        f"REGISTRY_SHA256 = {registry.registry_hash!r}",
        f"PUBLIC_PROTO_SHA256 = {registry.public_hash!r}",
        f"KERNEL_REGISTRY_SHA256 = {registry.kernel_registry_hash!r}",
        f"FEATURES = {_python_expression(features)}",
        f"PASSES = {_python_expression(passes)}",
        f"PASS_GROUPS = {_python_expression(groups)}",
        f"PIPELINE_RECIPES = {_python_expression(pipelines)}",
    ]
    return ("\n".join(lines) + "\n").encode()


def _markdown_list(values_: Iterable[str]) -> str:
    values = tuple(values_)
    return ", ".join(f"`{value}`" for value in values) if values else "—"


def render_docs(registry: OptimizerRegistry) -> bytes:
    lines = [
        "<!-- DO NOT EDIT: generated by tools/generate_optimizer_registry.py. -->",
        f"<!-- proto/optimizer_registry.proto SHA-256: {registry.registry_hash} -->",
        "",
        "# Optimizer registry",
        "",
        "`proto/optimizer_registry.proto` is the internal source of truth for typed pass",
        "metadata and deterministic recipes. Each implementation ID resolves to the executable",
        "transactional IRPass implementation. Target and recipe applicability are evaluated",
        "directly from typed generated fields. Backend rules are checked against",
        "`proto/kernel_registry.proto`; this inventory is not a benchmark result or a public",
        "runtime RPC.",
    ]
    lines.extend([
        "",
        "## Features",
        "",
        "| ID | Explicit caller contract |",
        "| --- | --- |",
    ])
    for item in registry.features:
        lines.append(f"| `{item.id}` | {item.summary} |")

    lines.extend([
        "",
        "## Passes",
        "",
        "| Pass | Scope / stage | Semantics | Dialect | Effects | Match → emit | Targets | Executable implementation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for item in registry.passes:
        scope = _short(item.scope, "OPTIMIZER_PASS_SCOPE_")
        stage = _short(item.stage, "OPTIMIZER_PASS_STAGE_")
        semantics = _short(item.semantics, "REWRITE_SEMANTICS_")
        dialect = (
            "/".join(_short(value, "OPTIMIZER_DIALECT_") for value in item.input_dialects)
            + " → " + _short(item.output_dialect, "OPTIMIZER_DIALECT_")
        )
        effects = []
        if item.repeatable:
            effects.append("repeatable")
        if item.requires_calibration:
            effects.append("calibration")
        if item.changes_public_abi:
            effects.append("ABI")
        if item.changes_precision:
            effects.append("precision")
        match_emit = (
            _markdown_list(_operators(registry, item.matched_operators))
            + " → " + _markdown_list(_operators(registry, item.emitted_operators))
        )
        target_rules = [
            _markdown_list(
                registry.backend_runtime_ids[backend] for backend in rule.backends
            )
            for rule in item.target_rules
        ]
        targets = "; ".join(target_rules) if target_rules else "—"
        lines.append(
            f"| `{item.id}` v{item.version} | `{scope}` / `{stage}` | `{semantics}` | "
            f"`{dialect}` | {', '.join(effects) or '—'} | {match_emit} | {targets} | "
            f"`{item.implementation_id}` |"
        )
        lines.append(f"|  | {item.summary} |  |  |  |  |  |  |")

    lines.extend([
        "",
        "## Pass groups",
        "",
        "| Group | Pass order | Iteration |",
        "| --- | --- | --- |",
    ])
    for item in registry.groups:
        iteration = f"fixed point, max {item.max_iterations}" if item.fixed_point else "once"
        lines.append(f"| `{item.id}` | {_markdown_list(item.pass_ids)} | {iteration} |")

    lines.extend([
        "",
        "## Pipeline recipes",
        "",
        "| Recipe | Required / supported features | Ordered group overlays | Targets | Allowed semantics | Policy |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for item in registry.pipelines:
        targets = _markdown_list(registry.backend_runtime_ids[value] for value in item.targets)
        semantics = _markdown_list(
            _short(value, "REWRITE_SEMANTICS_") for value in item.allowed_semantics
        )
        overlays = []
        for overlay in item.group_overlays:
            gates = []
            if overlay.required_features:
                gates.append("requires " + "+".join(overlay.required_features))
            if overlay.forbidden_features:
                gates.append("forbids " + "+".join(overlay.forbidden_features))
            overlays.append(
                f"`{overlay.group_id}`"
                + (f" ({'; '.join(gates)})" if gates else "")
            )
        policy = []
        if item.allow_calibration:
            policy.append("calibration")
        if item.allow_public_abi_change:
            policy.append("ABI change")
        lines.append(
            f"| `{item.id}` v{item.version} | requires "
            f"{_markdown_list(item.required_features)}; supports "
            f"{_markdown_list(item.supported_features)} | {' → '.join(overlays)} | "
            f"{targets} | {semantics} | "
            f"{', '.join(policy) or 'no extra effects'} |"
        )
        lines.append(f"|  | {item.summary} |  |  |  |  |")
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
    with tempfile.TemporaryDirectory(prefix="volvoxai-optimizer-registry-") as directory:
        result = subprocess.run(
            [
                executable,
                f"--proto_path={proto_root}",
                f"--descriptor_set_out={Path(directory) / 'registry.pb'}",
                str(registry_proto),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    if result.returncode:
        raise ValueError("protoc validation failed:\n" + result.stderr.rstrip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--protoc-check", action="store_true")
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument(
        "--registry-proto", type=Path,
        default=ROOT / "proto/optimizer_registry.proto",
    )
    parser.add_argument(
        "--public-proto", type=Path, default=ROOT / "proto/volvoxai.proto",
    )
    parser.add_argument(
        "--kernel-registry-proto", type=Path,
        default=ROOT / "proto/kernel_registry.proto",
    )
    parser.add_argument(
        "--python-out", type=Path,
        default=ROOT / "tools/exporter/generated/optimizer_registry.py",
    )
    parser.add_argument(
        "--docs-out", type=Path,
        default=ROOT / "docs/generated/optimizer-registry.md",
    )
    args = parser.parse_args()

    try:
        registry_proto = args.registry_proto.resolve()
        public_proto = args.public_proto.resolve()
        kernel_proto = args.kernel_registry_proto.resolve()
        registry = load_registry(registry_proto, public_proto, kernel_proto)
        if args.protoc_check:
            protoc_check(registry_proto, registry_proto.parent, args.protoc)
        outputs = {
            args.python_out.resolve(): render_python(registry),
            args.docs_out.resolve(): render_docs(registry),
        }
    except (OSError, UnicodeError, ValueError) as error:
        print(f"optimizer registry generation failed: {error}", file=sys.stderr)
        return 1

    stale = [
        path for path, data in outputs.items()
        if not path.is_file() or path.read_bytes() != data
    ]
    if args.check:
        if stale:
            for path in stale:
                print(f"stale generated optimizer registry: {path}", file=sys.stderr)
            print("run python3 tools/generate_optimizer_registry.py", file=sys.stderr)
            return 1
        print(f"Verified {len(outputs)} generated optimizer registry files")
        return 0

    for path, data in outputs.items():
        write_atomic(path, data)
        print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
