#!/usr/bin/env python3
"""Generate dependency-free views of the internal operator-param registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from generate_kernel_registry import (
        AggregateParser,
        parse_declared_enum,
        protoc_check,
        reject_unknown,
        scalar,
        tokenize,
        unique,
        values,
    )
    from generate_operator_vocabulary import (
        operator_graph_entries,
        parse_operator_vocabulary,
    )
except ImportError:  # Imported as tools.generate_operator_param_registry.
    from tools.generate_kernel_registry import (
        AggregateParser,
        parse_declared_enum,
        protoc_check,
        reject_unknown,
        scalar,
        tokenize,
        unique,
        values,
    )
    from tools.generate_operator_vocabulary import (
        operator_graph_entries,
        parse_operator_vocabulary,
    )


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PROTO = ROOT / "proto/operator_param_registry.proto"
OPERATOR_VOCABULARY_PROTO = ROOT / "proto/operator_vocabulary.proto"
REGISTRY_OPTION = "volvoxai.operator_param_registry.registry"


@dataclass(frozen=True)
class ParameterDefinition:
    key: str
    key_value: int
    json_name: str
    json_form: str
    json_form_value: int
    profile: str
    profile_value: int


@dataclass(frozen=True)
class SymbolDefinition:
    symbol: str
    symbol_value: int
    json_value: str


@dataclass(frozen=True)
class OperatorContract:
    operator: str
    operator_value: int
    operator_name: str
    keys: tuple[str, ...]
    full_keys: tuple[str, ...]
    nullable_keys: tuple[str, ...]


@dataclass(frozen=True)
class AliasGroup:
    id: str
    profile: str
    operators: tuple[str, ...]
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Registry:
    parameters: tuple[ParameterDefinition, ...]
    symbols: tuple[SymbolDefinition, ...]
    operators: tuple[OperatorContract, ...]
    parameter_free_operators: tuple[str, ...]
    aliases: tuple[AliasGroup, ...]
    key_numbers: dict[str, int]
    value_kind_numbers: dict[str, int]
    json_form_numbers: dict[str, int]
    symbol_numbers: dict[str, int]
    profile_numbers: dict[str, int]
    quantization_numbers: dict[str, int]
    operator_numbers: dict[str, int]
    graph_names: dict[str, str]
    registry_hash: str
    vocabulary_hash: str


def parse_registry_option(source: str) -> dict[str, list[object]]:
    tokens = tokenize(source)
    for index in range(len(tokens) - 5):
        actual = [token.value for token in tokens[index:index + 6]]
        if actual == ["option", "(", REGISTRY_OPTION, ")", "=", "{"]:
            parser = AggregateParser(tokens, index + 5)
            result = parser.message()
            parser.take(";")
            return result
    raise ValueError(f"missing option ({REGISTRY_OPTION})")


def _declared_enum(
    source: str,
    name: str,
    prefix: str,
    *,
    dense: bool = True,
) -> dict[str, int]:
    result = parse_declared_enum(source, name)
    if not result:
        raise ValueError(f"{name} must not be empty")
    invalid = [item for item in result if not item.startswith(prefix)]
    if invalid:
        raise ValueError(f"{name} has invalid prefix: {', '.join(invalid)}")
    ordered_values = list(result.values())
    if ordered_values[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if dense and ordered_values != list(range(len(ordered_values))):
        raise ValueError(f"{name} values must be dense, ordered cache indices")
    return result


def _enum_ref(value: str, numbers: dict[str, int], where: str) -> str:
    if value not in numbers:
        raise ValueError(f"{where} has unknown enum value {value!r}")
    return value


def load_registry(
    registry_proto: Path = REGISTRY_PROTO,
    operator_vocabulary_proto: Path = OPERATOR_VOCABULARY_PROTO,
) -> Registry:
    registry_proto = registry_proto.resolve()
    operator_vocabulary_proto = operator_vocabulary_proto.resolve()
    registry_bytes = registry_proto.read_bytes()
    vocabulary_bytes = operator_vocabulary_proto.read_bytes()
    source = registry_bytes.decode("utf-8")
    vocabulary_source = vocabulary_bytes.decode("utf-8")
    root = parse_registry_option(source)
    reject_unknown(
        root,
        {
            "schema_version", "parameter", "symbol", "operator",
            "parameter_free_operator", "alias_group",
        },
        "registry",
    )
    if scalar(root, "schema_version", int, "registry") != 1:
        raise ValueError("registry.schema_version must be 1")

    key_numbers = _declared_enum(
        source, "NodeParameterKey", "NODE_PARAMETER_KEY_"
    )
    if next(iter(key_numbers)) != "NODE_PARAMETER_KEY_AXIS":
        raise ValueError("NodeParameterKey must preserve AXIS = 0")
    value_kind_numbers = _declared_enum(
        source, "NodeParameterValueKind", "NODE_PARAMETER_VALUE_KIND_"
    )
    if next(iter(value_kind_numbers)) != "NODE_PARAMETER_VALUE_KIND_ABSENT":
        raise ValueError("NodeParameterValueKind must preserve ABSENT = 0")
    json_form_numbers = _declared_enum(
        source, "NodeParameterJsonForm", "NODE_PARAMETER_JSON_FORM_"
    )
    symbol_numbers = _declared_enum(
        source, "NodeParameterSymbol", "NODE_PARAMETER_SYMBOL_"
    )
    if next(iter(symbol_numbers)) != "NODE_PARAMETER_SYMBOL_INVALID":
        raise ValueError("NodeParameterSymbol must preserve INVALID = 0")
    profile_numbers = _declared_enum(
        source, "NodeParameterProfile", "NODE_PARAMETER_PROFILE_"
    )
    required_profiles = {
        "NODE_PARAMETER_PROFILE_UNSPECIFIED",
        "NODE_PARAMETER_PROFILE_INFERENCE",
        "NODE_PARAMETER_PROFILE_FULL",
    }
    if set(profile_numbers) != required_profiles:
        raise ValueError("NodeParameterProfile must define exactly unspecified/inference/full")

    quantization_numbers = _declared_enum(source, "TensorQuantizationKind", "TENSOR_QUANTIZATION_KIND_")

    operator_enums = parse_operator_vocabulary(vocabulary_source)
    operator_entries = operator_graph_entries(operator_enums)
    operator_numbers = {name: value for name, value, _ in operator_entries}
    graph_names = {name: graph_name for name, _, graph_name in operator_entries}

    parameters: list[ParameterDefinition] = []
    seen_keys: set[str] = set()
    seen_names: set[str] = set()
    for index, item in enumerate(values(root, "parameter", dict, "registry")):
        where = f"parameter[{index}]"
        reject_unknown(item, {"key", "json_name", "json_form", "profile"}, where)
        key = _enum_ref(scalar(item, "key", str, where), key_numbers, where)
        json_name = scalar(item, "json_name", str, where)
        json_form = _enum_ref(
            scalar(item, "json_form", str, where), json_form_numbers, where
        )
        profile = _enum_ref(
            scalar(item, "profile", str, where), profile_numbers, where
        )
        if not json_name or "\x00" in json_name:
            raise ValueError(f"{where}.json_name must be non-empty text")
        if json_form == "NODE_PARAMETER_JSON_FORM_UNSPECIFIED":
            raise ValueError(f"{where}.json_form must be concrete")
        if profile == "NODE_PARAMETER_PROFILE_UNSPECIFIED":
            raise ValueError(f"{where}.profile must be concrete")
        if key in seen_keys:
            raise ValueError(f"duplicate parameter key {key}")
        if json_name in seen_names:
            raise ValueError(f"duplicate parameter JSON name {json_name!r}")
        seen_keys.add(key)
        seen_names.add(json_name)
        parameters.append(ParameterDefinition(
            key, key_numbers[key], json_name,
            json_form, json_form_numbers[json_form],
            profile, profile_numbers[profile],
        ))
    if seen_keys != set(key_numbers):
        missing = sorted(set(key_numbers) - seen_keys)
        extra = sorted(seen_keys - set(key_numbers))
        raise ValueError(
            "parameter definitions must cover every NodeParameterKey; "
            f"missing={missing}, extra={extra}"
        )
    parameters.sort(key=lambda item: item.key_value)
    parameter_by_key = {item.key: item for item in parameters}

    symbols: list[SymbolDefinition] = []
    seen_symbols: set[str] = set()
    seen_symbol_values: set[str] = set()
    for index, item in enumerate(values(root, "symbol", dict, "registry")):
        where = f"symbol[{index}]"
        reject_unknown(item, {"symbol", "json_value"}, where)
        symbol = _enum_ref(
            scalar(item, "symbol", str, where), symbol_numbers, where
        )
        json_value = scalar(item, "json_value", str, where)
        if symbol == "NODE_PARAMETER_SYMBOL_INVALID":
            raise ValueError("the invalid symbol must not have a JSON spelling")
        if "\x00" in json_value:
            raise ValueError(f"{where}.json_value contains NUL")
        if symbol in seen_symbols:
            raise ValueError(f"duplicate symbol definition {symbol}")
        if json_value in seen_symbol_values:
            raise ValueError(f"duplicate symbol JSON value {json_value!r}")
        seen_symbols.add(symbol)
        seen_symbol_values.add(json_value)
        symbols.append(SymbolDefinition(symbol, symbol_numbers[symbol], json_value))
    expected_symbols = set(symbol_numbers) - {"NODE_PARAMETER_SYMBOL_INVALID"}
    if seen_symbols != expected_symbols:
        raise ValueError("symbol definitions must cover every non-invalid symbol")
    symbols.sort(key=lambda item: item.symbol_value)

    operators: list[OperatorContract] = []
    seen_operators: set[str] = set()
    for index, item in enumerate(values(root, "operator", dict, "registry")):
        where = f"operator[{index}]"
        reject_unknown(item, {"operator", "key", "full_key", "nullable_key"}, where)
        operator = _enum_ref(
            scalar(item, "operator", str, where), operator_numbers, where
        )
        if operator in seen_operators:
            raise ValueError(f"duplicate operator contract {operator}")
        keys = unique(
            (_enum_ref(key, key_numbers, where) for key in values(item, "key", str, where)),
            f"{where}.key",
        )
        full_keys = unique(
            (_enum_ref(key, key_numbers, where) for key in values(item, "full_key", str, where)),
            f"{where}.full_key",
        )
        nullable_keys = unique(
            (_enum_ref(key, key_numbers, where) for key in values(item, "nullable_key", str, where)),
            f"{where}.nullable_key",
        )
        if not keys and not full_keys:
            raise ValueError(f"{where} must allow at least one parameter")
        if set(keys) & set(full_keys):
            raise ValueError(f"{where}.key and full_key must be disjoint")
        for key in keys:
            if parameter_by_key[key].profile != "NODE_PARAMETER_PROFILE_INFERENCE":
                raise ValueError(f"{where}.key references full-only {key}")
        if not set(nullable_keys) <= set(keys):
            raise ValueError(
                f"{where}.nullable_key must be an inference-profile key allowed "
                "by the operator"
            )
        seen_operators.add(operator)
        operators.append(OperatorContract(
            operator, operator_numbers[operator], graph_names[operator],
            keys, full_keys, nullable_keys,
        ))
    operators.sort(key=lambda item: item.operator_value)

    parameter_free_operators = unique(
        (
            _enum_ref(item, operator_numbers, "parameter_free_operator")
            for item in values(root, "parameter_free_operator", str, "registry")
        ),
        "parameter_free_operator",
    )
    free = set(parameter_free_operators)
    if free & seen_operators:
        raise ValueError("an operator cannot be both parameterized and parameter-free")
    covered = free | seen_operators
    if covered != set(operator_numbers):
        missing = sorted(set(operator_numbers) - covered)
        raise ValueError(f"operator parameter coverage is incomplete: {missing}")
    parameter_free_operators = tuple(
        sorted(parameter_free_operators, key=operator_numbers.__getitem__)
    )

    contract_by_operator = {item.operator: item for item in operators}
    aliases: list[AliasGroup] = []
    seen_alias_ids: set[str] = set()
    seen_alias_memberships: set[tuple[str, str]] = set()
    for index, item in enumerate(values(root, "alias_group", dict, "registry")):
        where = f"alias_group[{index}]"
        reject_unknown(item, {"id", "profile", "operator", "key"}, where)
        group_id = scalar(item, "id", str, where)
        profile = _enum_ref(
            scalar(item, "profile", str, where), profile_numbers, where
        )
        group_operators = unique(
            (_enum_ref(op, operator_numbers, where) for op in values(item, "operator", str, where)),
            f"{where}.operator",
        )
        keys = unique(
            (_enum_ref(key, key_numbers, where) for key in values(item, "key", str, where)),
            f"{where}.key",
        )
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", group_id):
            raise ValueError(f"{where}.id is invalid")
        if group_id in seen_alias_ids:
            raise ValueError(f"duplicate alias group id {group_id!r}")
        if profile == "NODE_PARAMETER_PROFILE_UNSPECIFIED":
            raise ValueError(f"{where}.profile must be concrete")
        if not group_operators or len(keys) < 2:
            raise ValueError(f"{where} needs operators and at least two keys")
        for operator in group_operators:
            contract = contract_by_operator.get(operator)
            if contract is None:
                raise ValueError(f"{where} references parameter-free {operator}")
            allowed = set(contract.keys)
            if profile == "NODE_PARAMETER_PROFILE_FULL":
                allowed.update(contract.full_keys)
            if not set(keys) <= allowed:
                raise ValueError(f"{where} contains a key not allowed for {operator}")
            for key in keys:
                membership = (operator, key)
                if membership in seen_alias_memberships:
                    raise ValueError(f"duplicate alias membership {membership}")
                seen_alias_memberships.add(membership)
        if profile == "NODE_PARAMETER_PROFILE_INFERENCE":
            full_only = [key for key in keys if parameter_by_key[key].profile == "NODE_PARAMETER_PROFILE_FULL"]
            if full_only:
                raise ValueError(f"{where} leaks full-only keys: {full_only}")
        else:
            full_only = [
                key for key in keys
                if parameter_by_key[key].profile == "NODE_PARAMETER_PROFILE_FULL"
            ]
            if not full_only:
                raise ValueError(f"{where} full-profile alias has no full-only key")
        seen_alias_ids.add(group_id)
        aliases.append(AliasGroup(group_id, profile, group_operators, keys))

    return Registry(
        tuple(parameters), tuple(symbols), tuple(operators),
        parameter_free_operators, tuple(aliases), key_numbers,
        value_kind_numbers, json_form_numbers, symbol_numbers, profile_numbers, quantization_numbers,
        operator_numbers, graph_names,
        hashlib.sha256(registry_bytes).hexdigest(),
        hashlib.sha256(vocabulary_bytes).hexdigest(),
    )


def node_parameter_name_map(
    registry: Registry, *, include_full: bool
) -> dict[str, int]:
    return {
        item.json_name: item.key_value
        for item in registry.parameters
        if include_full or item.profile == "NODE_PARAMETER_PROFILE_INFERENCE"
    }


def _banner(registry: Registry, marker: str) -> str:
    return (
        f"{marker} DO NOT EDIT: generated by tools/generate_operator_param_registry.py.\n"
        f"{marker} proto/operator_param_registry.proto SHA-256: {registry.registry_hash}\n"
        f"{marker} proto/operator_vocabulary.proto SHA-256: {registry.vocabulary_hash}\n"
    )


def _c_name(name: str, prefix: str, replacement: str) -> str:
    return replacement + name.removeprefix(prefix)


def _c_enum_lines(numbers: dict[str, int], prefix: str, replacement: str) -> list[str]:
    items = list(numbers.items())
    return [
        f"    {_c_name(name, prefix, replacement)} = {value},"
        for name, value in items
    ]


def _c_parameter_key(key: str) -> str:
    return _c_name(key, "NODE_PARAMETER_KEY_", "VX_NODE_PARAM_")


def _c_json_form(form: str) -> str:
    return _c_name(form, "NODE_PARAMETER_JSON_FORM_", "VX_NODE_PARAM_JSON_")


def _c_profile(profile: str) -> str:
    return _c_name(profile, "NODE_PARAMETER_PROFILE_", "VX_NODE_PARAM_PROFILE_")


def _c_table(type_name: str, name: str, count_name: str, rows: list[str]) -> list[str]:
    if not rows:
        return [
            f"static const {type_name}* const {name} = NULL;",
            f"static const size_t {count_name} = 0;",
        ]
    return [
        f"static const {type_name} {name}[] = {{",
        *rows,
        "};",
        f"static const size_t {count_name} =",
        f"    sizeof({name}) / sizeof({name}[0]);",
    ]


def _c_definitions(registry: Registry, profile: str, prefix: str) -> list[str]:
    lines = []
    for item in registry.parameters:
        if item.profile != profile:
            continue
        lines.append(
            "    {" + ", ".join((
                json.dumps(item.json_name), _c_parameter_key(item.key),
                _c_json_form(item.json_form), _c_profile(item.profile),
            )) + "},"
        )
    return _c_table("VxGeneratedNodeParamDefinition", f"{prefix}_definitions",
                    f"{prefix}_definition_count", lines)


def _c_operator_rules(registry: Registry, *, full: bool, prefix: str) -> list[str]:
    lines = []
    for contract in registry.operators:
        keys = contract.full_keys if full else contract.keys
        for key in keys:
            lines.append(
                f"    {{{_c_name(contract.operator, 'OPERATOR_KIND_', 'VX_OP_')}, {_c_parameter_key(key)}}},"
            )
    return _c_table("VxGeneratedNodeParamOperatorRule", f"{prefix}_operator_rules",
                    f"{prefix}_operator_rule_count", lines)


def _c_alias_members(registry: Registry, profile: str, prefix: str) -> list[str]:
    lines = []
    for group in registry.aliases:
        if group.profile != profile:
            continue
        for operator in group.operators:
            operator_kind = _c_name(operator, 'OPERATOR_KIND_', 'VX_OP_')
            for key in group.keys:
                lines.append(
                    "    {" + ", ".join((
                        operator_kind, json.dumps(group.id),
                        _c_parameter_key(key), _c_profile(group.profile),
                    )) + "},"
                )
    return _c_table("VxGeneratedNodeParamAliasMember", f"{prefix}_alias_members",
                    f"{prefix}_alias_member_count", lines)


def render_c_ids(registry: Registry) -> bytes:
    lines = [
        _banner(registry, "/*").replace("\n", " */\n").rstrip(),
        "#ifndef VOLVOXAI_INTERNAL_OPERATOR_PARAM_IDS_H",
        "#define VOLVOXAI_INTERNAL_OPERATOR_PARAM_IDS_H",
        "",
        "#include <stdint.h>",
        "",
        "typedef int32_t VxTensorQuantizationKind;", "enum {",
        *_c_enum_lines(registry.quantization_numbers, "TENSOR_QUANTIZATION_KIND_", "VX_QUANTIZATION_"),
        "};", "",
        "typedef int32_t VxNodeParamKey;",
        "enum {",
        *_c_enum_lines(registry.key_numbers, "NODE_PARAMETER_KEY_", "VX_NODE_PARAM_"),
        f"    VX_NODE_PARAM_COUNT = {len(registry.key_numbers)}",
        "};",
        "",
        "typedef int32_t VxNodeParamValueKind;",
        "enum {",
        *_c_enum_lines(
            registry.value_kind_numbers,
            "NODE_PARAMETER_VALUE_KIND_", "VX_NODE_PARAM_",
        ),
        "};",
        "",
        "typedef int32_t VxNodeParamSymbol;",
        "enum {",
        *_c_enum_lines(registry.symbol_numbers, "NODE_PARAMETER_SYMBOL_", "VX_NODE_SYMBOL_"),
        "};",
        "",
        "typedef int32_t VxNodeParamJsonForm;",
        "enum {",
        *_c_enum_lines(registry.json_form_numbers, "NODE_PARAMETER_JSON_FORM_", "VX_NODE_PARAM_JSON_"),
        "};",
        "",
        "typedef int32_t VxNodeParamProfile;",
        "enum {",
        *_c_enum_lines(registry.profile_numbers, "NODE_PARAMETER_PROFILE_", "VX_NODE_PARAM_PROFILE_"),
        "};",
        "",
        "#if defined(__cplusplus)",
        "#define VX_NODE_PARAM_STATIC_ASSERT(condition_, message_) static_assert(condition_, message_)",
        "#else",
        "#define VX_NODE_PARAM_STATIC_ASSERT(condition_, message_) _Static_assert(condition_, message_)",
        "#endif",
        "VX_NODE_PARAM_STATIC_ASSERT(VX_NODE_PARAM_AXIS == 0, \"AXIS cache index must remain zero\");",
        "VX_NODE_PARAM_STATIC_ASSERT(VX_NODE_PARAM_SCORE_THRESHOLD + 1 == VX_NODE_PARAM_COUNT, \"parameter keys must remain dense\");",
        "VX_NODE_PARAM_STATIC_ASSERT(VX_NODE_PARAM_COUNT <= UINT8_MAX, \"parameter key/count storage overflow\");",
        "VX_NODE_PARAM_STATIC_ASSERT(VX_NODE_PARAM_SHAPE_ARRAY <= UINT8_MAX, \"parameter kind storage overflow\");",
        "VX_NODE_PARAM_STATIC_ASSERT(VX_NODE_SYMBOL_EMPTY <= UINT16_MAX, \"parameter symbol storage overflow\");",
        "#undef VX_NODE_PARAM_STATIC_ASSERT",
        "",
        "#endif /* VOLVOXAI_INTERNAL_OPERATOR_PARAM_IDS_H */",
        "",
    ]
    return "\n".join(lines).encode()


def render_c_base(registry: Registry) -> bytes:
    lines = [
        _banner(registry, "/*").replace("\n", " */\n").rstrip(),
        "#ifndef VOLVOXAI_INTERNAL_OPERATOR_PARAM_REGISTRY_H",
        "#define VOLVOXAI_INTERNAL_OPERATOR_PARAM_REGISTRY_H",
        "",
        "#include <stddef.h>",
        "#include <string.h>",
        "",
        '#include "operator_param_ids.h"',
        '#include "operator_vocabulary.h"',
        "",
        "typedef struct VxGeneratedNodeParamDefinition {",
        "    const char* name;",
        "    VxNodeParamKey key;",
        "    VxNodeParamJsonForm json_form;",
        "    VxNodeParamProfile profile;",
        "} VxGeneratedNodeParamDefinition;",
        "",
        "typedef struct VxGeneratedNodeParamSymbolDefinition {",
        "    const char* value;",
        "    VxNodeParamSymbol symbol;",
        "} VxGeneratedNodeParamSymbolDefinition;",
        "",
        "typedef struct VxGeneratedNodeParamOperatorRule {",
        "    VxOperatorKind operator_kind;",
        "    VxNodeParamKey key;",
        "} VxGeneratedNodeParamOperatorRule;",
        "",
        "typedef struct VxGeneratedNodeParamNullableRule {",
        "    VxOperatorKind operator_kind;",
        "    VxNodeParamKey key;",
        "} VxGeneratedNodeParamNullableRule;",
        "",
        "typedef struct VxGeneratedNodeParamAliasMember {",
        "    VxOperatorKind operator_kind;",
        "    const char* group_id;",
        "    VxNodeParamKey key;",
        "    VxNodeParamProfile profile;",
        "} VxGeneratedNodeParamAliasMember;",
        "",
        *_c_definitions(
            registry, "NODE_PARAMETER_PROFILE_INFERENCE",
            "vx_generated_node_param",
        ),
        "",
        "static const VxGeneratedNodeParamSymbolDefinition vx_generated_node_param_symbol_definitions[] = {",
    ]
    for item in registry.symbols:
        lines.append(
            f"    {{{json.dumps(item.json_value)}, "
            f"{_c_name(item.symbol, 'NODE_PARAMETER_SYMBOL_', 'VX_NODE_SYMBOL_')}}},"
        )
    lines.extend([
        "};",
        "static const size_t vx_generated_node_param_symbol_definition_count =",
        "    sizeof(vx_generated_node_param_symbol_definitions) / sizeof(vx_generated_node_param_symbol_definitions[0]);",
        "",
        *_c_operator_rules(
            registry, full=False, prefix="vx_generated_node_param"
        ),
        "",
        "static const VxGeneratedNodeParamNullableRule vx_generated_node_param_nullable_rules[] = {",
    ])
    for contract in registry.operators:
        for key in contract.nullable_keys:
            lines.append(
                f"    {{{_c_name(contract.operator, 'OPERATOR_KIND_', 'VX_OP_')}, {_c_parameter_key(key)}}},"
            )
    lines.extend([
        "};",
        "static const size_t vx_generated_node_param_nullable_rule_count =",
        "    sizeof(vx_generated_node_param_nullable_rules) / sizeof(vx_generated_node_param_nullable_rules[0]);",
        "",
        *_c_alias_members(
            registry, "NODE_PARAMETER_PROFILE_INFERENCE",
            "vx_generated_node_param",
        ),
        "",
        "static const VxOperatorKind vx_generated_node_param_parameter_free_operators[] = {",
    ])
    for operator in registry.parameter_free_operators:
        lines.append(f"    {_c_name(operator, 'OPERATOR_KIND_', 'VX_OP_')},")
    lines.extend([
        "};",
        "static const size_t vx_generated_node_param_parameter_free_operator_count =",
        "    sizeof(vx_generated_node_param_parameter_free_operators) / sizeof(vx_generated_node_param_parameter_free_operators[0]);",
        "",
        "static inline const VxGeneratedNodeParamDefinition*",
        "vx_generated_node_param_definition_find(const char* name) {",
        "    size_t index;",
        "    if (!name) return NULL;",
        "    for (index = 0; index < vx_generated_node_param_definition_count; ++index)",
        "        if (!strcmp(name, vx_generated_node_param_definitions[index].name))",
        "            return &vx_generated_node_param_definitions[index];",
        "    return NULL;",
        "}",
        "",
        "static inline const char* vx_generated_node_param_name(VxNodeParamKey key) {",
        "    for (size_t i = 0; i < vx_generated_node_param_definition_count; i++)",
        "        if (vx_generated_node_param_definitions[i].key == key)",
        "            return vx_generated_node_param_definitions[i].name;",
        "    return NULL;", "}", "",
        "static inline VxNodeParamSymbol",
        "vx_generated_node_param_symbol_from_bytes(const void* data, size_t length) {",
        "    if (!data) return VX_NODE_SYMBOL_INVALID;",
        "    for (size_t i = 0; i < vx_generated_node_param_symbol_definition_count; i++) {",
        "        const VxGeneratedNodeParamSymbolDefinition* item = &vx_generated_node_param_symbol_definitions[i];",
        "        if (strlen(item->value) == length && !memcmp(data, item->value, length)) return item->symbol;",
        "    }", "    return VX_NODE_SYMBOL_INVALID;", "}", "",
        "static inline VxNodeParamSymbol",
        "vx_generated_node_param_symbol_from_json(const char* value) {",
        "    return value ? vx_generated_node_param_symbol_from_bytes(value, strlen(value)) : VX_NODE_SYMBOL_INVALID;",
        "}", "",
        "static inline VxTensorQuantizationKind vx_generated_quantization_kind(const char* value) {",
        "    if (!value) return VX_QUANTIZATION_NONE;",
        *(f"    if (!strcmp(value, {json.dumps(name.removeprefix('TENSOR_QUANTIZATION_KIND_').lower())})) return VX_QUANTIZATION_{name.removeprefix('TENSOR_QUANTIZATION_KIND_')};"
          for name, value in registry.quantization_numbers.items() if value),
        "    return VX_QUANTIZATION_NONE;", "}", "",
        "static inline int vx_generated_node_param_allowed(",
        "        VxOperatorKind operator_kind, VxNodeParamKey key) {",
        "    size_t index;",
        "    if (operator_kind == VX_OP_UNSPECIFIED || key < 0 || key >= VX_NODE_PARAM_COUNT) return 0;",
        "    for (index = 0; index < vx_generated_node_param_operator_rule_count; ++index) {",
        "        const VxGeneratedNodeParamOperatorRule* rule =",
        "            &vx_generated_node_param_operator_rules[index];",
        "        if (rule->key == key && operator_kind == rule->operator_kind) return 1;",
        "    }",
        "    return 0;",
        "}",
        "",
        "static inline int vx_generated_node_param_allows_null(",
        "        VxOperatorKind operator_kind, VxNodeParamKey key) {",
        "    size_t index;",
        "    if (operator_kind == VX_OP_UNSPECIFIED || key < 0 || key >= VX_NODE_PARAM_COUNT) return 0;",
        "    for (index = 0; index < vx_generated_node_param_nullable_rule_count; ++index) {",
        "        const VxGeneratedNodeParamNullableRule* rule =",
        "            &vx_generated_node_param_nullable_rules[index];",
        "        if (rule->key == key && operator_kind == rule->operator_kind) return 1;",
        "    }",
        "    return 0;",
        "}",
        "",
        "#endif /* VOLVOXAI_INTERNAL_OPERATOR_PARAM_REGISTRY_H */",
        "",
    ])
    return "\n".join(lines).encode()


def render_c_full(registry: Registry) -> bytes:
    lines = [
        _banner(registry, "/*").replace("\n", " */\n").rstrip(),
        "#ifndef VOLVOXAI_INTERNAL_OPERATOR_PARAM_REGISTRY_FULL_H",
        "#define VOLVOXAI_INTERNAL_OPERATOR_PARAM_REGISTRY_FULL_H",
        "",
        '#include "operator_param_registry.h"',
        "",
        *_c_definitions(
            registry, "NODE_PARAMETER_PROFILE_FULL",
            "vx_generated_full_node_param",
        ),
        "",
        *_c_operator_rules(
            registry, full=True, prefix="vx_generated_full_node_param"
        ),
        "",
        *_c_alias_members(
            registry, "NODE_PARAMETER_PROFILE_FULL",
            "vx_generated_full_node_param",
        ),
        "",
        "static inline const VxGeneratedNodeParamDefinition*",
        "vx_generated_full_node_param_definition_find(const char* name) {",
        "    size_t index;",
        "    const VxGeneratedNodeParamDefinition* base;",
        "    if (!name) return NULL;",
        "    for (index = 0; index < vx_generated_full_node_param_definition_count; ++index)",
        "        if (!strcmp(name, vx_generated_full_node_param_definitions[index].name))",
        "            return &vx_generated_full_node_param_definitions[index];",
        "    base = vx_generated_node_param_definition_find(name);",
        "    return base;",
        "}",
        "",
        "static inline int vx_generated_full_node_param_allowed(",
        "        VxOperatorKind operator_kind, VxNodeParamKey key) {",
        "    size_t index;",
        "    if (vx_generated_node_param_allowed(operator_kind, key)) return 1;",
        "    if (operator_kind == VX_OP_UNSPECIFIED || key < 0 || key >= VX_NODE_PARAM_COUNT) return 0;",
        "    for (index = 0; index < vx_generated_full_node_param_operator_rule_count; ++index) {",
        "        const VxGeneratedNodeParamOperatorRule* rule =",
        "            &vx_generated_full_node_param_operator_rules[index];",
        "        if (rule->key == key && operator_kind == rule->operator_kind) return 1;",
        "    }",
        "    return 0;",
        "}",
        "",
        "static inline int vx_generated_full_node_param_allows_null(",
        "        VxOperatorKind operator_kind, VxNodeParamKey key) {",
        "    return vx_generated_node_param_allows_null(operator_kind, key);",
        "}",
        "",
        "#endif /* VOLVOXAI_INTERNAL_OPERATOR_PARAM_REGISTRY_FULL_H */",
        "",
    ]
    return "\n".join(lines).encode()


def _member(name: str, prefix: str) -> str:
    suffix = name.removeprefix(prefix)
    return "_" + suffix if suffix[0].isdigit() else suffix


def _ts_enum(name: str, numbers: dict[str, int], prefix: str) -> list[str]:
    lines = [f"export enum {name} {{"]
    lines.extend(
        f"  {_member(item, prefix)} = {value}," for item, value in numbers.items()
    )
    lines.append("}")
    return lines


def _ts_key_ref(registry: Registry, key: str, full_file: bool) -> str:
    parameter = next(item for item in registry.parameters if item.key == key)
    enum_name = (
        "FullNodeParameterKey"
        if full_file and parameter.profile == "NODE_PARAMETER_PROFILE_FULL"
        else "NodeParameterKey"
    )
    return f"{enum_name}.{_member(key, 'NODE_PARAMETER_KEY_')}"


def _ts_form_ref(form: str) -> str:
    return f"NodeParameterJsonForm.{_member(form, 'NODE_PARAMETER_JSON_FORM_')}"


def _ts_profile_ref(profile: str) -> str:
    return f"NodeParameterProfile.{_member(profile, 'NODE_PARAMETER_PROFILE_')}"


def _ts_flat_rules(registry: Registry, *, full: bool) -> list[str]:
    rows: list[str] = []
    for contract in registry.operators:
        for key in contract.full_keys if full else contract.keys:
            rows.append(
                "  Object.freeze({ operator: " + json.dumps(contract.operator_name)
                + ", key: " + _ts_key_ref(registry, key, full) + " }),"
            )
    return rows


def render_ts_base(registry: Registry) -> bytes:
    base_parameters = tuple(
        item for item in registry.parameters
        if item.profile == "NODE_PARAMETER_PROFILE_INFERENCE"
    )
    base_keys = {item.key: item.key_value for item in base_parameters}
    lines = [_banner(registry, "//").rstrip(), ""]
    lines.extend(_ts_enum("NodeParameterKey", base_keys, "NODE_PARAMETER_KEY_"))
    lines.extend(_ts_enum("TensorQuantizationKind", registry.quantization_numbers, "TENSOR_QUANTIZATION_KIND_"))
    lines.extend([f"export const NODE_PARAMETER_KEY_COUNT = {len(registry.key_numbers)};", ""])
    lines.extend(_ts_enum(
        "NodeParameterValueKind", registry.value_kind_numbers,
        "NODE_PARAMETER_VALUE_KIND_",
    ))
    lines.append("")
    lines.extend(_ts_enum(
        "NodeParameterJsonForm", registry.json_form_numbers,
        "NODE_PARAMETER_JSON_FORM_",
    ))
    lines.append("")
    lines.extend(_ts_enum(
        "NodeParameterSymbol", registry.symbol_numbers,
        "NODE_PARAMETER_SYMBOL_",
    ))
    lines.append("")
    lines.extend(_ts_enum(
        "NodeParameterProfile", registry.profile_numbers,
        "NODE_PARAMETER_PROFILE_",
    ))
    lines.extend([
        "",
        "export const nodeParameterDefinitions = Object.freeze([",
    ])
    for item in base_parameters:
        lines.append(
            "  Object.freeze({ name: " + json.dumps(item.json_name)
            + ", key: " + _ts_key_ref(registry, item.key, False)
            + ", jsonForm: " + _ts_form_ref(item.json_form)
            + ", profile: " + _ts_profile_ref(item.profile) + " }),"
        )
    lines.extend([
        "] as const);",
        "export type NodeParameterName = (typeof nodeParameterDefinitions)[number]['name'];",
        "",
        "export const nodeParameterKeyByName = Object.freeze({",
    ])
    for item in base_parameters:
        lines.append(
            f"  {json.dumps(item.json_name)}: {_ts_key_ref(registry, item.key, False)},"
        )
    lines.extend([
        "} as const);",
        "",
        "export const nodeParameterSymbolDefinitions = Object.freeze([",
    ])
    for item in registry.symbols:
        lines.append(
            "  Object.freeze({ value: " + json.dumps(item.json_value)
            + ", symbol: NodeParameterSymbol."
            + _member(item.symbol, "NODE_PARAMETER_SYMBOL_") + " }),"
        )
    lines.extend([
        "] as const);",
        "",
        "export const operatorParameterRules = Object.freeze([",
        *_ts_flat_rules(registry, full=False),
        "] as const);",
        "",
        "export const nullableOperatorParameterRules = Object.freeze([",
    ])
    for contract in registry.operators:
        for key in contract.nullable_keys:
            lines.append(
                "  Object.freeze({ operator: " + json.dumps(contract.operator_name)
                + ", key: " + _ts_key_ref(registry, key, False) + " }),"
            )
    lines.extend([
        "] as const);",
        "",
        "export const nodeParameterAliasMembers = Object.freeze([",
    ])
    for group in registry.aliases:
        if group.profile != "NODE_PARAMETER_PROFILE_INFERENCE":
            continue
        for operator in group.operators:
            for key in group.keys:
                lines.append(
                    "  Object.freeze({ operator: " + json.dumps(registry.graph_names[operator])
                    + ", group: " + json.dumps(group.id)
                    + ", key: " + _ts_key_ref(registry, key, False)
                    + ", profile: " + _ts_profile_ref(group.profile) + " }),"
                )
    lines.extend([
        "] as const);",
        "",
        "export const parameterFreeOperators = Object.freeze([",
    ])
    for operator in registry.parameter_free_operators:
        lines.append(f"  {json.dumps(registry.graph_names[operator])},")
    lines.extend([
        "] as const);",
        "",
    ])
    return "\n".join(lines).encode()


def render_ts_full(registry: Registry) -> bytes:
    full_parameters = tuple(
        item for item in registry.parameters
        if item.profile == "NODE_PARAMETER_PROFILE_FULL"
    )
    full_keys = {item.key: item.key_value for item in full_parameters}
    lines = [
        _banner(registry, "//").rstrip(),
        "",
        "import {",
        "  NodeParameterJsonForm, NodeParameterKey, NodeParameterProfile,",
        "  nodeParameterDefinitions, nodeParameterKeyByName, operatorParameterRules,",
        "  nodeParameterAliasMembers,",
        "} from './operatorParamRegistry.js';",
        "",
    ]
    lines.extend(_ts_enum("FullNodeParameterKey", full_keys, "NODE_PARAMETER_KEY_"))
    lines.extend([
        "export type FullNodeParameterKeyValue = NodeParameterKey | FullNodeParameterKey;",
        "",
        "export const fullNodeParameterDefinitions = Object.freeze([",
    ])
    for item in full_parameters:
        lines.append(
            "  Object.freeze({ name: " + json.dumps(item.json_name)
            + ", key: " + _ts_key_ref(registry, item.key, True)
            + ", jsonForm: " + _ts_form_ref(item.json_form)
            + ", profile: " + _ts_profile_ref(item.profile) + " }),"
        )
    lines.extend([
        "] as const);",
        "export type FullOnlyNodeParameterName = (typeof fullNodeParameterDefinitions)[number]['name'];",
        "",
        "export const fullNodeParameterKeyByName = Object.freeze({",
    ])
    for item in full_parameters:
        lines.append(
            f"  {json.dumps(item.json_name)}: {_ts_key_ref(registry, item.key, True)},"
        )
    lines.extend([
        "} as const);",
        "",
        "export const allNodeParameterDefinitions = Object.freeze([",
        "  ...nodeParameterDefinitions, ...fullNodeParameterDefinitions,",
        "] as const);",
        "export const allNodeParameterKeysByName = Object.freeze({",
        "  ...nodeParameterKeyByName, ...fullNodeParameterKeyByName,",
        "} as const);",
        "",
        "export const fullOperatorParameterRules = Object.freeze([",
        *_ts_flat_rules(registry, full=True),
        "] as const);",
        "export const allOperatorParameterRules = Object.freeze([",
        "  ...operatorParameterRules, ...fullOperatorParameterRules,",
        "] as const);",
        "",
        "export const fullNodeParameterAliasMembers = Object.freeze([",
    ])
    for group in registry.aliases:
        if group.profile != "NODE_PARAMETER_PROFILE_FULL":
            continue
        for operator in group.operators:
            for key in group.keys:
                lines.append(
                    "  Object.freeze({ operator: " + json.dumps(registry.graph_names[operator])
                    + ", group: " + json.dumps(group.id)
                    + ", key: " + _ts_key_ref(registry, key, True)
                    + ", profile: " + _ts_profile_ref(group.profile) + " }),"
                )
    lines.extend([
        "] as const);",
        "export const allNodeParameterAliasMembers = Object.freeze([",
        "  ...nodeParameterAliasMembers, ...fullNodeParameterAliasMembers,",
        "] as const);",
        "",
    ])
    return "\n".join(lines).encode()


def _python_enum(name: str, numbers: dict[str, int], prefix: str) -> list[str]:
    lines = [f"class {name}(IntEnum):"]
    lines.extend(
        f"    {_member(item, prefix)} = {value}" for item, value in numbers.items()
    )
    if not numbers:
        lines.append("    pass")
    return lines


def _python_key_ref(registry: Registry, key: str, full_file: bool) -> str:
    parameter = next(item for item in registry.parameters if item.key == key)
    enum_name = (
        "FullNodeParameterKey"
        if full_file and parameter.profile == "NODE_PARAMETER_PROFILE_FULL"
        else "NodeParameterKey"
    )
    return f"{enum_name}.{_member(key, 'NODE_PARAMETER_KEY_')}"


def render_python_base(registry: Registry) -> bytes:
    base_parameters = tuple(
        item for item in registry.parameters
        if item.profile == "NODE_PARAMETER_PROFILE_INFERENCE"
    )
    base_keys = {item.key: item.key_value for item in base_parameters}
    lines = [
        _banner(registry, "#").rstrip(),
        "",
        "from collections import namedtuple",
        "from enum import IntEnum",
        "from types import MappingProxyType",
        "",
    ]
    lines.extend(_python_enum("NodeParameterKey", base_keys, "NODE_PARAMETER_KEY_"))
    lines.extend(_python_enum("TensorQuantizationKind", registry.quantization_numbers, "TENSOR_QUANTIZATION_KIND_"))
    lines.extend(["", f"NODE_PARAMETER_KEY_COUNT = {len(registry.key_numbers)}", ""])
    lines.extend(_python_enum(
        "NodeParameterValueKind", registry.value_kind_numbers,
        "NODE_PARAMETER_VALUE_KIND_",
    ))
    lines.append("")
    lines.extend(_python_enum(
        "NodeParameterJsonForm", registry.json_form_numbers,
        "NODE_PARAMETER_JSON_FORM_",
    ))
    lines.append("")
    lines.extend(_python_enum(
        "NodeParameterSymbol", registry.symbol_numbers,
        "NODE_PARAMETER_SYMBOL_",
    ))
    lines.append("")
    lines.extend(_python_enum(
        "NodeParameterProfile", registry.profile_numbers,
        "NODE_PARAMETER_PROFILE_",
    ))
    lines.extend([
        "",
        "ParameterDefinition = namedtuple('ParameterDefinition', 'name key json_form profile')",
        "OperatorParameterRule = namedtuple('OperatorParameterRule', 'operator key')",
        "AliasMember = namedtuple('AliasMember', 'operator group key profile')",
        "SymbolDefinition = namedtuple('SymbolDefinition', 'value symbol')",
        "",
        "NODE_PARAMETER_DEFINITIONS = (",
    ])
    for item in base_parameters:
        lines.append(
            f"    ParameterDefinition({item.json_name!r}, {_python_key_ref(registry, item.key, False)}, "
            f"NodeParameterJsonForm.{_member(item.json_form, 'NODE_PARAMETER_JSON_FORM_')}, "
            f"NodeParameterProfile.{_member(item.profile, 'NODE_PARAMETER_PROFILE_')}),"
        )
    lines.extend([
        ")",
        "NODE_PARAMETER_KEY_BY_NAME = MappingProxyType({item.name: item.key for item in NODE_PARAMETER_DEFINITIONS})",
        "",
        "NODE_PARAMETER_SYMBOL_DEFINITIONS = (",
    ])
    for item in registry.symbols:
        lines.append(
            f"    SymbolDefinition({item.json_value!r}, "
            f"NodeParameterSymbol.{_member(item.symbol, 'NODE_PARAMETER_SYMBOL_')}),"
        )
    lines.extend([
        ")",
        "",
        "OPERATOR_PARAMETER_RULES = (",
    ])
    for contract in registry.operators:
        for key in contract.keys:
            lines.append(
                f"    OperatorParameterRule({contract.operator_name!r}, "
                f"{_python_key_ref(registry, key, False)}),"
            )
    lines.extend([
        ")",
        "NULLABLE_OPERATOR_PARAMETER_RULES = (",
    ])
    for contract in registry.operators:
        for key in contract.nullable_keys:
            lines.append(
                f"    OperatorParameterRule({contract.operator_name!r}, "
                f"{_python_key_ref(registry, key, False)}),"
            )
    lines.extend([
        ")",
        "NODE_PARAMETER_ALIAS_MEMBERS = (",
    ])
    for group in registry.aliases:
        if group.profile != "NODE_PARAMETER_PROFILE_INFERENCE":
            continue
        for operator in group.operators:
            for key in group.keys:
                lines.append(
                    f"    AliasMember({registry.graph_names[operator]!r}, {group.id!r}, "
                    f"{_python_key_ref(registry, key, False)}, "
                    f"NodeParameterProfile.{_member(group.profile, 'NODE_PARAMETER_PROFILE_')}),"
                )
    lines.extend([
        ")",
        "PARAMETER_FREE_OPERATORS = (",
    ])
    for operator in registry.parameter_free_operators:
        lines.append(f"    {registry.graph_names[operator]!r},")
    lines.extend([
        ")",
        "",
    ])
    return "\n".join(lines).encode()


def render_python_full(registry: Registry) -> bytes:
    full_parameters = tuple(
        item for item in registry.parameters
        if item.profile == "NODE_PARAMETER_PROFILE_FULL"
    )
    full_keys = {item.key: item.key_value for item in full_parameters}
    lines = [
        _banner(registry, "#").rstrip(),
        "",
        "from enum import IntEnum",
        "from types import MappingProxyType",
        "",
        "from .operator_param_registry import (",
        "    AliasMember, NodeParameterJsonForm, NodeParameterKey, NodeParameterProfile,",
        "    ParameterDefinition, OperatorParameterRule, NODE_PARAMETER_ALIAS_MEMBERS,",
        "    NODE_PARAMETER_DEFINITIONS, NODE_PARAMETER_KEY_BY_NAME, OPERATOR_PARAMETER_RULES,",
        ")",
        "",
    ]
    lines.extend(_python_enum("FullNodeParameterKey", full_keys, "NODE_PARAMETER_KEY_"))
    lines.extend(["", "FULL_NODE_PARAMETER_DEFINITIONS = ("])
    for item in full_parameters:
        lines.append(
            f"    ParameterDefinition({item.json_name!r}, {_python_key_ref(registry, item.key, True)}, "
            f"NodeParameterJsonForm.{_member(item.json_form, 'NODE_PARAMETER_JSON_FORM_')}, "
            f"NodeParameterProfile.{_member(item.profile, 'NODE_PARAMETER_PROFILE_')}),"
        )
    lines.extend([
        ")",
        "FULL_NODE_PARAMETER_KEY_BY_NAME = MappingProxyType({item.name: item.key for item in FULL_NODE_PARAMETER_DEFINITIONS})",
        "ALL_NODE_PARAMETER_DEFINITIONS = NODE_PARAMETER_DEFINITIONS + FULL_NODE_PARAMETER_DEFINITIONS",
        "ALL_NODE_PARAMETER_KEY_BY_NAME = MappingProxyType({**NODE_PARAMETER_KEY_BY_NAME, **FULL_NODE_PARAMETER_KEY_BY_NAME})",
        "",
        "FULL_OPERATOR_PARAMETER_RULES = (",
    ])
    for contract in registry.operators:
        for key in contract.full_keys:
            lines.append(
                f"    OperatorParameterRule({contract.operator_name!r}, "
                f"{_python_key_ref(registry, key, True)}),"
            )
    lines.extend([
        ")",
        "ALL_OPERATOR_PARAMETER_RULES = OPERATOR_PARAMETER_RULES + FULL_OPERATOR_PARAMETER_RULES",
        "FULL_NODE_PARAMETER_ALIAS_MEMBERS = (",
    ])
    for group in registry.aliases:
        if group.profile != "NODE_PARAMETER_PROFILE_FULL":
            continue
        for operator in group.operators:
            for key in group.keys:
                lines.append(
                    f"    AliasMember({registry.graph_names[operator]!r}, {group.id!r}, "
                    f"{_python_key_ref(registry, key, True)}, "
                    f"NodeParameterProfile.{_member(group.profile, 'NODE_PARAMETER_PROFILE_')}),"
                )
    lines.extend([
        ")",
        "ALL_NODE_PARAMETER_ALIAS_MEMBERS = NODE_PARAMETER_ALIAS_MEMBERS + FULL_NODE_PARAMETER_ALIAS_MEMBERS",
        "",
    ])
    return "\n".join(lines).encode()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--protoc-check", action="store_true")
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument("--registry-proto", type=Path, default=REGISTRY_PROTO)
    parser.add_argument(
        "--operator-vocabulary-proto", type=Path,
        default=OPERATOR_VOCABULARY_PROTO,
    )
    parser.add_argument(
        "--native-ids-out", type=Path,
        default=ROOT / "native/src/generated/operator_param_ids.h",
    )
    parser.add_argument(
        "--native-out", type=Path,
        default=ROOT / "native/src/generated/operator_param_registry.h",
    )
    parser.add_argument(
        "--native-full-out", type=Path,
        default=ROOT / "native/src/generated/operator_param_registry_full.h",
    )
    parser.add_argument(
        "--typescript-out", type=Path,
        default=ROOT / "tools/generated/operatorParamRegistry.ts",
    )
    parser.add_argument(
        "--typescript-full-out", type=Path,
        default=ROOT / "tools/generated/operatorParamRegistryFull.ts",
    )
    parser.add_argument(
        "--python-out", type=Path,
        default=ROOT / "tools/exporter/generated/operator_param_registry.py",
    )
    parser.add_argument(
        "--python-full-out", type=Path,
        default=ROOT / "tools/exporter/generated/operator_param_registry_full.py",
    )
    args = parser.parse_args()
    try:
        registry = load_registry(
            args.registry_proto, args.operator_vocabulary_proto
        )
        if args.protoc_check:
            protoc_check(args.registry_proto.resolve(), ROOT / "proto", args.protoc)
        outputs = {
            args.native_ids_out.resolve(): render_c_ids(registry),
            args.native_out.resolve(): render_c_base(registry),
            args.native_full_out.resolve(): render_c_full(registry),
            args.typescript_out.resolve(): render_ts_base(registry),
            args.typescript_full_out.resolve(): render_ts_full(registry),
            args.python_out.resolve(): render_python_base(registry),
            args.python_full_out.resolve(): render_python_full(registry),
        }
        full_only_tokens = tuple(
            item.json_name.encode() for item in registry.parameters
            if item.profile == "NODE_PARAMETER_PROFILE_FULL"
        )
        base_paths = {
            args.native_ids_out.resolve(), args.native_out.resolve(), args.typescript_out.resolve(),
            args.python_out.resolve(),
        }
        for path in base_paths:
            leaked = [token.decode() for token in full_only_tokens if token in outputs[path]]
            if leaked:
                raise ValueError(f"full-only parameter strings leaked into {path}: {leaked}")
        for path in set(outputs) - base_paths:
            missing = [token.decode() for token in full_only_tokens if token not in outputs[path]]
            if missing:
                raise ValueError(f"full projection {path} lacks full-only strings: {missing}")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        print(f"operator-param registry generation failed: {error}", file=sys.stderr)
        return 1

    stale = [
        path for path, data in outputs.items()
        if not path.is_file() or path.read_bytes() != data
    ]
    if args.check:
        if stale:
            for path in stale:
                print(f"stale generated operator-param registry: {path}", file=sys.stderr)
            print("run make operator_param_registry_codegen", file=sys.stderr)
            return 1
        print(f"Verified {len(outputs)} generated operator-param registry files")
        return 0
    for path, data in outputs.items():
        write_atomic(path, data)
        print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
