#!/usr/bin/env python3
"""Run Synurang after retaining one service and message dependency profile.

protoc plugins receive a serialized ``CodeGeneratorRequest`` on stdin and
return a serialized ``CodeGeneratorResponse`` on stdout.  Synurang's pinned
generator can filter a single service itself, but its comma-delimited plugin
parameter parser cannot express VolvoxAI's inference service profile.
This wrapper filters the descriptor request instead, retains the transitive
message/enum closure of those services, then delegates generation to the exact
upstream executable selected by ``tools/generate_proto.py``.  Filtering the
types as well as the dispatch is important for static libraries: otherwise the
lite codec would still publish functions for services omitted by the profile.

Configuration is deliberately environment-only because protoc launches plugin
executables without a portable argument vector:

``VOLVOXAI_SYNURANG_GENERATOR``
    Absolute path to the pinned upstream generator.
``VOLVOXAI_SYNURANG_SERVICES``
    Comma-separated fully-qualified service names to retain.
``VOLVOXAI_SYNURANG_OMIT_ENUM_VALUES``
    Optional comma-separated fully-qualified enum values to omit from the
    generated profile.  The numeric wire vocabulary remains owned by the
    source schema; this removes names for capabilities the profile cannot
    expose from its language projection.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    from google.protobuf import descriptor_pb2
    from google.protobuf.compiler import plugin_pb2
    from google.protobuf.message import DecodeError
except ImportError as error:  # pragma: no cover - exercised in dependency-poor hosts
    print(
        "profile-filter codegen requires the Python protobuf package; "
        "install tools/requirements-codegen.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from error


GENERATOR_ENV = "VOLVOXAI_SYNURANG_GENERATOR"
SERVICES_ENV = "VOLVOXAI_SYNURANG_SERVICES"
OMIT_ENUM_VALUES_ENV = "VOLVOXAI_SYNURANG_OMIT_ENUM_VALUES"
MESSAGE_FIELD_NUMBER = 4
ENUM_FIELD_NUMBER = 5
SERVICE_FIELD_NUMBER = 6
ENUM_VALUE_FIELD_NUMBER = 2
GENERATED_RESPONSE_SUFFIXES = {
    "c": ("_lite.c", "_lite.h", "_ffi.c", "_ffi.h"),
    "typescript": ("_lite.ts", "_ffi.ts", "synurang_runtime.ts"),
}


class FilterError(RuntimeError):
    """A deterministic profile-filter configuration or protocol failure."""


def configured_generator() -> Path:
    raw = os.environ.get(GENERATOR_ENV, "").strip()
    if not raw:
        raise FilterError(f"{GENERATOR_ENV} is required")
    generator = Path(raw).expanduser().resolve()
    if not generator.is_file() or not os.access(generator, os.X_OK):
        raise FilterError(f"upstream generator is not executable: {generator}")
    if generator == Path(__file__).resolve():
        raise FilterError("upstream generator resolves to the profile wrapper itself")
    return generator


def configured_services() -> tuple[str, ...]:
    raw = os.environ.get(SERVICES_ENV, "")
    services = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not services:
        raise FilterError(f"{SERVICES_ENV} must name at least one service")
    if len(set(services)) != len(services):
        raise FilterError(f"{SERVICES_ENV} contains a duplicate service")
    for service in services:
        if service.startswith(".") or "." not in service:
            raise FilterError(
                f"service must be a fully-qualified protobuf name: {service!r}"
            )
    return services


def configured_omitted_enum_values() -> tuple[str, ...]:
    raw = os.environ.get(OMIT_ENUM_VALUES_ENV, "")
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(set(values)) != len(values):
        raise FilterError(f"{OMIT_ENUM_VALUES_ENV} contains a duplicate value")
    for value in values:
        if value.startswith(".") or value.count(".") < 2:
            raise FilterError(
                "enum values must be fully-qualified protobuf names: "
                f"{value!r}"
            )
    return values


def service_name(package: str, name: str) -> str:
    return f"{package}.{name}" if package else name


def filter_source_locations(
    descriptor,
    index_maps: dict[int, dict[int, int]],
    enum_value_index_maps: dict[int, dict[int, int]],
) -> None:
    """Drop/remap SourceCodeInfo paths for filtered top-level declarations."""

    retained = []
    for location in descriptor.source_code_info.location:
        clone = type(location)()
        clone.CopyFrom(location)
        original_top_level_index = clone.path[1] if len(clone.path) >= 2 else None
        index_map = index_maps.get(clone.path[0]) if len(clone.path) >= 2 else None
        if index_map is not None:
            mapped_index = index_map.get(clone.path[1])
            if mapped_index is None:
                continue
            clone.path[1] = mapped_index
        if (
            len(clone.path) >= 4
            and clone.path[0] == ENUM_FIELD_NUMBER
            and clone.path[2] == ENUM_VALUE_FIELD_NUMBER
            and original_top_level_index is not None
        ):
            value_index_map = enum_value_index_maps.get(original_top_level_index)
            if value_index_map is not None:
                mapped_value_index = value_index_map.get(clone.path[3])
                if mapped_value_index is None:
                    continue
                clone.path[3] = mapped_value_index
        retained.append(clone)
    descriptor.source_code_info.ClearField("location")
    descriptor.source_code_info.location.extend(retained)


def nested_messages(message):
    """Yields one message and all messages nested below it."""

    yield message
    for child in message.nested_type:
        yield from nested_messages(child)


def filter_generated_descriptor(
    descriptor,
    allowed: set[str],
    omitted_enum_values: set[str],
) -> tuple[dict[str, str], set[str]]:
    """Retains selected services and every local type they transitively use."""

    retained_services = []
    service_index_map: dict[int, int] = {}
    found: dict[str, str] = {}
    required_types: set[str] = set()
    for original_index, service in enumerate(descriptor.service):
        qualified_name = service_name(descriptor.package, service.name)
        if qualified_name not in allowed:
            continue
        clone = type(service)()
        clone.CopyFrom(service)
        service_index_map[original_index] = len(retained_services)
        retained_services.append(clone)
        found[qualified_name] = descriptor.name
        for method in service.method:
            required_types.add(method.input_type)
            required_types.add(method.output_type)

    package_prefix = f".{descriptor.package}." if descriptor.package else "."
    message_owners: dict[str, int] = {}
    nested_enum_owners: dict[str, int] = {}
    top_level_enums: dict[str, int] = {}

    def register_message(message, qualified_name: str, owner: int) -> None:
        message_owners[qualified_name] = owner
        for enum in message.enum_type:
            nested_enum_owners[f"{qualified_name}.{enum.name}"] = owner
        for child in message.nested_type:
            register_message(child, f"{qualified_name}.{child.name}", owner)

    for index, message in enumerate(descriptor.message_type):
        register_message(message, f"{package_prefix}{message.name}", index)
    for index, enum in enumerate(descriptor.enum_type):
        top_level_enums[f"{package_prefix}{enum.name}"] = index

    retained_message_indexes: set[int] = set()
    retained_enum_indexes: set[int] = set()
    pending = list(required_types)
    visited: set[str] = set()
    while pending:
        type_name = pending.pop()
        if type_name in visited:
            continue
        visited.add(type_name)

        message_index = message_owners.get(type_name)
        if message_index is not None:
            if message_index in retained_message_indexes:
                continue
            retained_message_indexes.add(message_index)
            owner = descriptor.message_type[message_index]
            for message in nested_messages(owner):
                for field in message.field:
                    if field.type in {
                        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
                        descriptor_pb2.FieldDescriptorProto.TYPE_GROUP,
                        descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
                    }:
                        pending.append(field.type_name)
            continue

        nested_owner = nested_enum_owners.get(type_name)
        if nested_owner is not None:
            pending.append(
                f"{package_prefix}{descriptor.message_type[nested_owner].name}"
            )
            continue

        enum_index = top_level_enums.get(type_name)
        if enum_index is not None:
            retained_enum_indexes.add(enum_index)
            continue
        # Imported types are deliberately left to their unmodified descriptor.

    retained_messages = []
    message_index_map: dict[int, int] = {}
    for original_index, message in enumerate(descriptor.message_type):
        if original_index not in retained_message_indexes:
            continue
        clone = type(message)()
        clone.CopyFrom(message)
        message_index_map[original_index] = len(retained_messages)
        retained_messages.append(clone)

    retained_enums = []
    enum_index_map: dict[int, int] = {}
    enum_value_index_maps: dict[int, dict[int, int]] = {}
    found_omitted_values: set[str] = set()
    for original_index, enum in enumerate(descriptor.enum_type):
        if original_index not in retained_enum_indexes:
            continue
        clone = type(enum)()
        clone.CopyFrom(enum)
        qualified_enum = service_name(descriptor.package, enum.name)
        retained_values = []
        value_index_map: dict[int, int] = {}
        for original_value_index, value in enumerate(enum.value):
            qualified_value = f"{qualified_enum}.{value.name}"
            if qualified_value in omitted_enum_values:
                if original_value_index == 0:
                    raise FilterError(
                        f"cannot omit the zero/default enum value {qualified_value}"
                    )
                found_omitted_values.add(qualified_value)
                continue
            value_clone = type(value)()
            value_clone.CopyFrom(value)
            value_index_map[original_value_index] = len(retained_values)
            retained_values.append(value_clone)
        clone.ClearField("value")
        clone.value.extend(retained_values)
        enum_index_map[original_index] = len(retained_enums)
        enum_value_index_maps[original_index] = value_index_map
        retained_enums.append(clone)

    descriptor.ClearField("service")
    descriptor.service.extend(retained_services)
    descriptor.ClearField("message_type")
    descriptor.message_type.extend(retained_messages)
    descriptor.ClearField("enum_type")
    descriptor.enum_type.extend(retained_enums)
    filter_source_locations(
        descriptor,
        {
            MESSAGE_FIELD_NUMBER: message_index_map,
            ENUM_FIELD_NUMBER: enum_index_map,
            SERVICE_FIELD_NUMBER: service_index_map,
        },
        enum_value_index_maps,
    )
    return found, found_omitted_values


def filter_services(
    request: plugin_pb2.CodeGeneratorRequest,
    allowed_services: tuple[str, ...],
    omitted_enum_values: tuple[str, ...],
) -> None:
    allowed = set(allowed_services)
    omitted = set(omitted_enum_values)
    generated_files = set(request.file_to_generate)
    found: dict[str, str] = {}
    found_omitted_values: set[str] = set()

    for descriptor in request.proto_file:
        if descriptor.name not in generated_files:
            continue
        descriptor_found, descriptor_omitted = filter_generated_descriptor(
            descriptor, allowed, omitted
        )
        duplicates = sorted(set(found) & set(descriptor_found))
        if duplicates:
            raise FilterError(
                "service appears in more than one descriptor: " + ", ".join(duplicates)
            )
        found.update(descriptor_found)
        found_omitted_values.update(descriptor_omitted)

    missing = sorted(allowed - set(found))
    if missing:
        raise FilterError(f"requested service was not found: {', '.join(missing)}")
    imported = sorted(
        name for name, filename in found.items() if filename not in generated_files
    )
    if imported:
        raise FilterError(
            "requested service belongs to an imported file: " + ", ".join(imported)
        )
    missing_omissions = sorted(omitted - found_omitted_values)
    if missing_omissions:
        raise FilterError(
            "requested enum value was not retained by the profile: "
            + ", ".join(missing_omissions)
        )


def requested_language(parameter: str) -> str:
    options = dict(
        part.split("=", 1)
        for part in parameter.split(",")
        if "=" in part
    )
    language = options.get("lang", "").strip()
    if language not in GENERATED_RESPONSE_SUFFIXES:
        raise FilterError(f"profile filtering is not configured for lang={language!r}")
    return language


def validate_response(data: bytes, language: str) -> bytes:
    response = plugin_pb2.CodeGeneratorResponse()
    try:
        response.ParseFromString(data)
    except DecodeError as error:
        raise FilterError("upstream generator returned an invalid response") from error
    if response.error:
        return data

    names = {generated_file.name for generated_file in response.file}
    missing = [
        suffix
        for suffix in GENERATED_RESPONSE_SUFFIXES[language]
        if not any(name.endswith(suffix) for name in names)
    ]
    if missing:
        raise FilterError(
            "upstream generator omitted profile output: " + ", ".join(missing)
        )
    return data


def main() -> int:
    try:
        generator = configured_generator()
        services = configured_services()
        omitted_enum_values = configured_omitted_enum_values()
        request = plugin_pb2.CodeGeneratorRequest()
        try:
            request.ParseFromString(sys.stdin.buffer.read())
        except DecodeError as error:
            raise FilterError("stdin is not a valid CodeGeneratorRequest") from error
        if not request.file_to_generate:
            raise FilterError("CodeGeneratorRequest has no file_to_generate")

        language = requested_language(request.parameter)
        filter_services(request, services, omitted_enum_values)
        result = subprocess.run(
            [str(generator)],
            input=request.SerializeToString(),
            stdout=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
        sys.stdout.buffer.write(validate_response(result.stdout, language))
        return 0
    except (FilterError, OSError) as error:
        print(f"profile-filter codegen failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
