#!/usr/bin/env python3
"""Split, strip, and prove a Linux native release executable.

The release ELF keeps its loadable bytes and dynamic-loader contract. A GNU
debuglink points at a build-id-matched sidecar containing the optimized symbol
table and DWARF line information. The adjacent JSON file records the pre/post
comparison so size reports do not have to trust an opaque post-build command.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Sequence


FORMAT = "volvoxai-native-release-finalization/v1"
AARCH64_ELF_MACHINE = 183
PUBLISH_LOCK = "native/.release-publish.lock"
RUNTIME_SNAPSHOT_KEYS = (
    "buildId",
    "loader",
    "elfHeader",
    "programHeaders",
    "loadSegments",
    "loadableSectionsSha256",
    "version",
)
AARCH64_ADDITIONAL_SYMBOLS = (
    (
        "vx_qlinear_i8u8_arm_dotprod_try",
        "native/src/kernels/qlinear_w8a8_arm_dotprod.c",
    ),
    (
        "vx_qlinear_i8u8_arm_i8mm_packed_try",
        "native/src/kernels/qlinear_w8a8_arm_i8mm.c",
    ),
)


class FinalizationError(RuntimeError):
    """A native release could not be finalized without changing its contract."""


def command(arguments: Sequence[str], *, stderr: bool = True) -> str:
    result = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise FinalizationError(
            f"command failed ({result.returncode}): {' '.join(arguments)}"
            + (f"\n{detail}" if detail else "")
        )
    if stderr and result.stderr:
        sys.stderr.write(result.stderr)
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path, logical_path: str) -> dict[str, object]:
    if not path.is_file():
        raise FinalizationError(f"required release input does not exist: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise FinalizationError(f"required release input is empty: {path}")
    return {
        "path": logical_path,
        "rawBytes": size,
        "sha256": sha256(path),
    }


def release_link_manifest(
    repository_root: Path,
    build_directory: Path,
    link_objects: Sequence[Path],
) -> dict[str, object]:
    """Fingerprint the exact ordered object graph that produced an artifact."""
    if not link_objects:
        raise FinalizationError("native release has no ordered object inputs")
    objects: list[dict[str, object]] = []
    seen: set[Path] = set()
    for object_path in link_objects:
        if not object_path.is_absolute():
            raise FinalizationError(
                f"native link object path is not absolute: {object_path}"
            )
        object_path = build_private_path(
            build_directory, object_path.resolve(), "native link object"
        )
        if object_path in seen:
            raise FinalizationError(
                f"native link command repeats object input: {object_path}"
            )
        seen.add(object_path)
        objects.append(
            fingerprint(
                object_path, repository_relative(repository_root, object_path)
            )
        )
    return {"objects": objects}


def repository_path(repository_root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise FinalizationError(
            f"{label} must be inside repository root {repository_root}: {resolved}"
        ) from error
    return resolved


def build_private_path(build_directory: Path, path: Path, label: str) -> Path:
    try:
        path.relative_to(build_directory)
    except ValueError as error:
        raise FinalizationError(
            f"{label} must be inside build directory {build_directory}: {path}"
        ) from error
    return path


def repository_relative(repository_root: Path, path: Path) -> str:
    return Path(os.path.relpath(path, repository_root)).as_posix()


def logical_release_path(value: str, label: str) -> str:
    if "\\" in value:
        raise FinalizationError(f"{label} must use repository-relative POSIX syntax")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.as_posix() != value
    ):
        raise FinalizationError(
            f"{label} must be a canonical repository-relative path: {value!r}"
        )
    return value


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.{os.getpid()}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size or sha256(temporary) != sha256(
            source
        ):
            raise FinalizationError(
                f"atomic publish copy changed release bytes: {source} -> {destination}"
            )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_release_bundle(
    repository_root: Path,
    artifact: Path,
    debug: Path,
    evidence: Path,
    publish_artifact: Path,
    publish_debug: Path,
    publish_evidence: Path,
) -> None:
    lock_path = repository_root / PUBLISH_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # The lock inode carries no data. Open it read-only so a host build can
    # reuse a readable 0644 lock created by a root-run release container.
    # Linux flock permits an exclusive advisory lock on a read-only descriptor.
    descriptor = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            atomic_copy(debug, publish_debug)
            atomic_copy(evidence, publish_evidence)
            atomic_copy(artifact, publish_artifact)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def tool_version(tool: str) -> str:
    return command((tool, "--version"), stderr=False).splitlines()[0].strip()


def readelf_output(readelf: str, arguments: Sequence[str], path: Path) -> str:
    return command((readelf, *arguments, str(path)), stderr=False)


def build_id(readelf: str, path: Path) -> str:
    output = readelf_output(readelf, ("-nW",), path)
    match = re.search(r"Build ID:\s*([0-9A-Fa-f]+)", output)
    if match is None:
        raise FinalizationError(f"{path} has no GNU build ID")
    return match.group(1).lower()


def sections(readelf: str, path: Path) -> set[str]:
    output = readelf_output(readelf, ("-SW",), path)
    return {
        match.group(1)
        for match in re.finditer(r"^\s*\[\s*\d+\]\s+(\S+)", output, re.MULTILINE)
    }


def loader_contract(readelf: str, path: Path) -> dict[str, object]:
    program_headers = readelf_output(readelf, ("-lW",), path)
    interpreter_match = re.search(
        r"Requesting program interpreter:\s*([^\]]+)\]", program_headers
    )
    dynamic = readelf_output(readelf, ("-dW",), path)
    values: dict[str, list[str]] = {
        "needed": [],
        "rpath": [],
        "runpath": [],
        "soname": [],
    }
    for match in re.finditer(
        r"\((NEEDED|RPATH|RUNPATH|SONAME)\).*?\[([^\]]*)\]", dynamic
    ):
        values[match.group(1).lower()].append(match.group(2))
    return {
        "interpreter": interpreter_match.group(1) if interpreter_match else None,
        **{key: sorted(value) for key, value in values.items()},
    }


def inspect_elf_image(
    path: Path, allocated_literal: bytes | None = None
) -> tuple[dict[str, object], int | None]:
    data = path.read_bytes()
    if len(data) < 16:
        raise FinalizationError(f"{path} has a truncated ELF identification header")
    if data[:4] != b"\x7fELF":
        raise FinalizationError(f"{path} is not an ELF artifact")
    elf_class = data[4]
    byte_order = data[5]
    if elf_class not in (1, 2) or byte_order not in (1, 2):
        raise FinalizationError(f"{path} has an unsupported ELF class or byte order")
    endian = "<" if byte_order == 1 else ">"
    if elf_class == 2:
        header_format = endian + "HHIQQQIHHHHHH"
        program_format = endian + "IIQQQQQQ"
        section_format = endian + "IIQQQQIIQQ"
    else:
        header_format = endian + "HHIIIIIHHHHHH"
        program_format = endian + "IIIIIIII"
        section_format = endian + "IIIIIIIIII"

    header_size = struct.calcsize(header_format)
    if len(data) < 16 + header_size:
        raise FinalizationError(f"{path} has a truncated ELF header")
    (
        file_type,
        machine,
        elf_version,
        entry_point,
        program_offset,
        section_offset,
        elf_flags,
        elf_header_size,
        program_entry_size,
        program_entry_count,
        section_entry_size,
        section_entry_count,
        section_names_index,
    ) = struct.unpack_from(header_format, data, 16)
    if elf_header_size < 16 + header_size:
        raise FinalizationError(f"{path} has an undersized ELF header")

    elf_header = {
        "classBits": 64 if elf_class == 2 else 32,
        "byteOrder": "little" if byte_order == 1 else "big",
        "identVersion": data[6],
        "osAbi": data[7],
        "abiVersion": data[8],
        "type": file_type,
        "machine": machine,
        "version": elf_version,
        "entryPoint": entry_point,
        "programHeaderOffset": program_offset,
        "flags": elf_flags,
        "headerBytes": elf_header_size,
        "programHeaderEntryBytes": program_entry_size,
        "programHeaderCount": program_entry_count,
    }

    program_size = struct.calcsize(program_format)
    if program_entry_size < program_size:
        raise FinalizationError(f"{path} has undersized program-header entries")
    if program_offset + program_entry_count * program_entry_size > len(data):
        raise FinalizationError(f"{path} has a truncated program-header table")
    program_headers: list[dict[str, int]] = []
    load_segments: list[dict[str, int]] = []
    for index in range(program_entry_count):
        entry = program_offset + index * program_entry_size
        values = struct.unpack_from(program_format, data, entry)
        if elf_class == 2:
            (
                program_type,
                flags,
                offset,
                address,
                physical,
                file_size,
                memory_size,
                align,
            ) = values
        else:
            (
                program_type,
                offset,
                address,
                physical,
                file_size,
                memory_size,
                flags,
                align,
            ) = values
        if offset + file_size > len(data):
            raise FinalizationError(
                f"{path} has a truncated program segment at index {index}"
            )
        program_headers.append(
            {
                "type": program_type,
                "flags": flags,
                "offset": offset,
                "virtualAddress": address,
                "physicalAddress": physical,
                "fileBytes": file_size,
                "memoryBytes": memory_size,
                "alignment": align,
            }
        )
        if program_type != 1:  # PT_LOAD
            continue
        load_segments.append(
            {
                "offset": offset,
                "address": address,
                "physicalAddress": physical,
                "fileBytes": file_size,
                "memoryBytes": memory_size,
                "flags": flags,
                "alignment": align,
            }
        )
    if not load_segments:
        raise FinalizationError(f"{path} has no PT_LOAD segments")

    section_size = struct.calcsize(section_format)
    if section_entry_size < section_size or section_entry_count == 0:
        raise FinalizationError(f"{path} has no supported section-header table")
    if section_offset + section_entry_count * section_entry_size > len(data):
        raise FinalizationError(f"{path} has a truncated section-header table")
    section_headers = [
        struct.unpack_from(
            section_format, data, section_offset + index * section_entry_size
        )
        for index in range(section_entry_count)
    ]
    if section_names_index >= len(section_headers):
        raise FinalizationError(f"{path} has an invalid section-name table index")
    names_header = section_headers[section_names_index]
    names_offset, names_size = names_header[4], names_header[5]
    if names_offset + names_size > len(data):
        raise FinalizationError(f"{path} has a truncated section-name table")
    names = data[names_offset : names_offset + names_size]

    def section_name(offset: int) -> str:
        if offset >= len(names):
            raise FinalizationError(f"{path} has an invalid section name offset")
        end = names.find(b"\0", offset)
        if end < 0:
            raise FinalizationError(f"{path} has an unterminated section name")
        return names[offset:end].decode("utf-8", errors="strict")

    # Strip legitimately moves the non-loadable section table and therefore
    # changes e_shoff in the mapped ELF header. Hash the actual SHF_ALLOC
    # section identity/content and compare the PT_LOAD mapping contract instead
    # of treating that non-runtime metadata pointer as executable code.
    digest = hashlib.sha256()
    allocated_count = 0
    allocated_literal_matches = 0 if allocated_literal is not None else None
    for header in section_headers:
        name_offset, section_type, flags, address, offset, size = header[:6]
        if not flags & 0x2:  # SHF_ALLOC
            continue
        name = section_name(name_offset)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(struct.pack(">QQQQ", section_type, flags, address, size))
        if section_type != 8:  # SHT_NOBITS
            if offset + size > len(data):
                raise FinalizationError(f"{path} has a truncated {name} section")
            contents = data[offset : offset + size]
            digest.update(contents)
            if allocated_literal is not None:
                assert allocated_literal_matches is not None
                allocated_literal_matches += contents.count(allocated_literal)
        allocated_count += 1
    if allocated_count == 0:
        raise FinalizationError(f"{path} has no SHF_ALLOC sections")
    return (
        {
            "elfHeader": elf_header,
            "programHeaders": program_headers,
            "loadSegments": load_segments,
            "loadableSectionsSha256": digest.hexdigest(),
        },
        allocated_literal_matches,
    )


def loadable_image(path: Path) -> dict[str, object]:
    image, _ = inspect_elf_image(path)
    return image


def version_prefix(expected_version: str) -> str:
    return f"VolvoxAI Native Engine {expected_version} ("


def version_output(
    path: Path,
    enabled: bool,
    expected_version: str,
    allocated_prefix_matches: int | None = None,
) -> str:
    prefix = version_prefix(expected_version)
    if not enabled:
        if allocated_prefix_matches != 1:
            raise FinalizationError(
                f"{path} must contain the exact version prefix {prefix!r} once in "
                f"SHF_ALLOC bytes; found {allocated_prefix_matches or 0}"
            )
        return prefix
    output = command((str(path), "--version"), stderr=False).strip()
    pattern = re.compile(
        rf"{re.escape(prefix)}commit (?P<commit>[^,\s]+), "
        rf"built (?P<build_date>[^)\r\n]*\S)\)"
    )
    if pattern.fullmatch(output) is None:
        raise FinalizationError(
            f"{path} --version is not a complete provenance record for "
            f"version {expected_version!r}"
        )
    return output


def defined_symbols(nm: str, path: Path) -> tuple[set[str], dict[str, str]]:
    output = command((nm, "-n", "--defined-only", str(path)), stderr=False)
    names: set[str] = set()
    addresses: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3 or not re.fullmatch(r"[0-9A-Fa-f]+", fields[0]):
            continue
        name = fields[-1]
        names.add(name)
        addresses.setdefault(name, fields[0])
    return names, addresses


def defined_dynamic_symbols(readelf: str, path: Path) -> set[str]:
    output = readelf_output(readelf, ("--dyn-syms", "-W"), path)
    names: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if (
            len(fields) >= 8
            and fields[0].endswith(":")
            and fields[4] in {"GLOBAL", "WEAK"}
            and fields[6] != "UND"
        ):
            names.add(fields[7])
    return names


def is_expected_symbolized_source(reported: str, expected: str) -> bool:
    """Match a prefix-mapped DWARF source by its repository-relative identity.

    When an in-repository build directory is also below the source directory,
    GCC and Clang can choose different matches from the overlapping
    ``-fdebug-prefix-map`` options.  The resulting DWARF may combine a remapped
    compilation directory such as ``./build/release/native`` with a remapped
    file name such as ``./native/cli/main.c``.  That combined path is not a
    filesystem path, but its canonical repository-relative suffix is stable.
    """
    expected_parts = PurePosixPath(expected).parts
    reported_parts = PurePosixPath(reported).parts
    return (
        bool(expected_parts)
        and len(reported_parts) >= len(expected_parts)
        and reported_parts[-len(expected_parts) :] == expected_parts
    )


def require_debug_symbolization(
    addr2line: str,
    artifact: Path,
    debug: Path,
    address: str,
    repository_root: Path,
    symbol: str = "main",
    source_path: str = "native/cli/main.c",
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    expected_source = (repository_root / source_path).resolve()
    for label, target in (("release", artifact), ("debug", debug)):
        lines = command(
            (addr2line, "-f", "-C", "-e", str(target), address), stderr=False
        ).strip().splitlines()
        location = (
            re.fullmatch(r"(.+):([0-9]+)(?::([0-9]+))?", lines[1])
            if len(lines) >= 2
            else None
        )
        if len(lines) < 2 or lines[0] != symbol or location is None:
            raise FinalizationError(
                f"{target} cannot symbolize {symbol} through the separated DWARF image"
            )
        if not expected_source.is_file() or not is_expected_symbolized_source(
            location.group(1), source_path
        ):
            raise FinalizationError(
                f"{target} resolves {symbol} to unexpected source '{lines[1]}'"
            )
        suffix = f":{location.group(3)}" if location.group(3) is not None else ""
        resolved[label] = f"{source_path}:{location.group(2)}{suffix}"
    return {
        "symbol": symbol,
        "address": address,
        **resolved,
    }


def require_additional_symbolizations(
    machine: int,
    addr2line: str,
    artifact: Path,
    debug: Path,
    debug_symbols: set[str],
    debug_addresses: dict[str, str],
    repository_root: Path,
) -> list[dict[str, str]]:
    if machine != AARCH64_ELF_MACHINE:
        return []
    missing = [
        symbol for symbol, _ in AARCH64_ADDITIONAL_SYMBOLS if symbol not in debug_symbols
    ]
    if missing:
        raise FinalizationError(
            "AArch64 debug artifact is missing required hot-kernel symbols: "
            + ", ".join(missing)
        )
    return [
        require_debug_symbolization(
            addr2line,
            artifact,
            debug,
            debug_addresses[symbol],
            repository_root,
            symbol,
            source_path,
        )
        for symbol, source_path in AARCH64_ADDITIONAL_SYMBOLS
    ]


def snapshot(
    artifact: Path, readelf: str, run_version: bool, expected_version: str
) -> dict[str, object]:
    prefix = None if run_version else version_prefix(expected_version).encode("utf-8")
    image, allocated_prefix_matches = inspect_elf_image(artifact, prefix)
    return {
        "buildId": build_id(readelf, artifact),
        "loader": loader_contract(readelf, artifact),
        **image,
        "version": version_output(
            artifact,
            run_version,
            expected_version,
            allocated_prefix_matches,
        ),
    }


def require_unchanged_runtime_snapshot(
    before: dict[str, object], after: dict[str, object]
) -> None:
    for key in RUNTIME_SNAPSHOT_KEYS:
        if before[key] != after[key]:
            raise FinalizationError(
                f"native release finalization changed {key}: "
                f"{before[key]!r} != {after[key]!r}"
            )


def write_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--link-map", type=Path, required=True)
    parser.add_argument("--link-objects", type=Path, nargs="+", required=True)
    parser.add_argument("--logical-artifact", required=True)
    parser.add_argument("--logical-debug", required=True)
    parser.add_argument("--publish-artifact", type=Path, required=True)
    parser.add_argument("--publish-debug", type=Path, required=True)
    parser.add_argument("--publish-evidence", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--objcopy", required=True)
    parser.add_argument("--strip", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--addr2line", required=True)
    parser.add_argument("--run-version", action="store_true")
    return parser.parse_args()


def main() -> None:
    options = arguments()
    repository_root = options.repository_root.resolve()
    if not repository_root.is_dir():
        raise FinalizationError(
            f"repository root does not exist: {repository_root}"
        )
    build_directory = options.build_directory.resolve()
    if not build_directory.is_dir():
        raise FinalizationError(
            f"native build directory does not exist: {build_directory}"
        )
    artifact = build_private_path(
        build_directory,
        options.artifact.resolve(),
        "native artifact",
    )
    debug = build_private_path(
        build_directory,
        options.debug.resolve(),
        "native debug sidecar",
    )
    evidence = build_private_path(
        build_directory,
        options.evidence.resolve(),
        "native evidence",
    )
    link_map = build_private_path(
        build_directory,
        options.link_map.resolve(),
        "native linker map",
    )
    publish_artifact = repository_path(
        repository_root, options.publish_artifact, "published native artifact"
    )
    publish_debug = repository_path(
        repository_root, options.publish_debug, "published native debug sidecar"
    )
    publish_evidence = repository_path(
        repository_root, options.publish_evidence, "published native evidence"
    )
    logical_artifact = logical_release_path(
        options.logical_artifact, "logical native artifact"
    )
    logical_debug = logical_release_path(
        options.logical_debug, "logical native debug sidecar"
    )
    if repository_relative(repository_root, publish_artifact) != logical_artifact:
        raise FinalizationError(
            "published native artifact path does not match --logical-artifact"
        )
    if repository_relative(repository_root, publish_debug) != logical_debug:
        raise FinalizationError(
            "published native debug path does not match --logical-debug"
        )
    if repository_relative(repository_root, publish_evidence) != f"{logical_debug}.json":
        raise FinalizationError(
            "published native evidence path must be --logical-debug plus '.json'"
        )
    if not artifact.is_file():
        raise FinalizationError(f"native release artifact does not exist: {artifact}")
    build_directory_logical = repository_relative(repository_root, build_directory)
    artifact_logical = repository_relative(repository_root, artifact)
    link_map_logical = repository_relative(repository_root, link_map)
    link_map_before = fingerprint(link_map, link_map_logical)
    link_manifest_before = release_link_manifest(
        repository_root,
        build_directory,
        options.link_objects,
    )
    debug.parent.mkdir(parents=True, exist_ok=True)
    evidence.parent.mkdir(parents=True, exist_ok=True)

    pre_sections = sections(options.readelf, artifact)
    for required in (".symtab", ".strtab", ".debug_info"):
        if required not in pre_sections:
            raise FinalizationError(
                f"unstripped native release {artifact} is missing {required}"
            )
    pre_symbols, _ = defined_symbols(options.nm, artifact)
    for required in ("main", "Synurang_GetApi"):
        if required not in pre_symbols:
            raise FinalizationError(
                f"unstripped native release {artifact} is missing {required}"
            )
    before = snapshot(
        artifact, options.readelf, options.run_version, options.expected_version
    )
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{artifact.name}.finalize-", dir=artifact.parent)
    )
    staging_artifact = staging_root / artifact.name
    staging_debug = staging_root / ".debug" / debug.name
    staging_evidence = staging_root / evidence.name
    staging_debug.parent.mkdir(parents=True)
    try:
        # Never mutate the linker output until every structural, runtime and
        # symbolization check has passed. A failed post-build action therefore
        # leaves a complete unstripped executable that CMake can safely retry.
        shutil.copy2(artifact, staging_artifact)
        command(
            (
                options.objcopy,
                "--only-keep-debug",
                str(staging_artifact),
                str(staging_debug),
            )
        )
        command((options.strip, "--strip-all", str(staging_artifact)))
        command(
            (
                options.objcopy,
                f"--add-gnu-debuglink={staging_debug}",
                str(staging_artifact),
            )
        )

        after = snapshot(
            staging_artifact,
            options.readelf,
            options.run_version,
            options.expected_version,
        )
        require_unchanged_runtime_snapshot(before, after)

        release_sections = sections(options.readelf, staging_artifact)
        forbidden_release_sections = sorted(
            section
            for section in release_sections
            if section in {".symtab", ".strtab"} or section.startswith(".debug")
        )
        if forbidden_release_sections:
            raise FinalizationError(
                "stripped native release retained debug/symbol sections: "
                + ", ".join(forbidden_release_sections)
            )
        for required in (".dynsym", ".dynstr", ".gnu_debuglink"):
            if required not in release_sections:
                raise FinalizationError(
                    f"stripped native release is missing required {required}"
                )
        dynamic_exports = sorted(
            defined_dynamic_symbols(options.readelf, staging_artifact)
        )
        if dynamic_exports:
            raise FinalizationError(
                "release executable exposes defined dynamic symbols: "
                + ", ".join(dynamic_exports)
            )
        debuglink = readelf_output(
            options.readelf,
            ("--string-dump=.gnu_debuglink",),
            staging_artifact,
        )
        if debug.name not in debuglink:
            raise FinalizationError(
                f"native release debuglink does not name {debug.name}"
            )

        debug_sections = sections(options.readelf, staging_debug)
        for required in (".symtab", ".strtab", ".debug_info"):
            if required not in debug_sections:
                raise FinalizationError(
                    f"debug artifact {staging_debug} is missing {required}"
                )
        if build_id(options.readelf, staging_debug) != after["buildId"]:
            raise FinalizationError("release and debug artifact build IDs differ")
        debug_symbols, debug_addresses = defined_symbols(options.nm, staging_debug)
        for required in ("main", "Synurang_GetApi"):
            if required not in debug_symbols:
                raise FinalizationError(f"debug artifact is missing {required}")
        symbolization = require_debug_symbolization(
            options.addr2line,
            staging_artifact,
            staging_debug,
            debug_addresses["main"],
            repository_root,
        )
        additional_symbolization = require_additional_symbolizations(
            int(after["elfHeader"]["machine"]),
            options.addr2line,
            staging_artifact,
            staging_debug,
            debug_symbols,
            debug_addresses,
            repository_root,
        )

        link_map_after = fingerprint(link_map, link_map_logical)
        if link_map_after != link_map_before:
            raise FinalizationError(
                "native linker map changed while the release was finalized"
            )
        link_manifest_after = release_link_manifest(
            repository_root,
            build_directory,
            options.link_objects,
        )
        if link_manifest_after != link_manifest_before:
            raise FinalizationError(
                "native link object inputs changed while the release was finalized"
            )
        build_artifact_fingerprint = fingerprint(
            staging_artifact, artifact_logical
        )
        debug_fingerprint = fingerprint(staging_debug, logical_debug)

        document: dict[str, object] = {
            "format": FORMAT,
            "artifact": logical_artifact,
            "debugArtifact": logical_debug,
            "build": {
                "directory": build_directory_logical,
                "artifact": build_artifact_fingerprint,
                "linkMap": link_map_after,
                "link": link_manifest_after,
            },
            "before": before,
            "after": after,
            "release": {
                "rawBytes": build_artifact_fingerprint["rawBytes"],
                "sha256": build_artifact_fingerprint["sha256"],
                "hasGnuDebuglink": True,
                "definedDynamicExports": dynamic_exports,
            },
            "debug": {
                "rawBytes": debug_fingerprint["rawBytes"],
                "sha256": debug_fingerprint["sha256"],
                "buildId": after["buildId"],
            },
            "symbolization": symbolization,
            "additionalSymbolization": additional_symbolization,
            "tools": {
                "objcopy": tool_version(options.objcopy),
                "strip": tool_version(options.strip),
                "readelf": tool_version(options.readelf),
                "nm": tool_version(options.nm),
                "addr2line": tool_version(options.addr2line),
            },
        }
        write_atomic(staging_evidence, document)

        # All three outputs were fully validated before these same-filesystem
        # atomic replacements. Install the release executable last so a reader
        # never sees it before its matching sidecar exists.
        os.replace(staging_debug, debug)
        os.replace(staging_evidence, evidence)
        os.replace(staging_artifact, artifact)
        if fingerprint(artifact, artifact_logical) != build_artifact_fingerprint:
            raise FinalizationError(
                "installed build-private artifact differs from finalization evidence"
            )
        if fingerprint(debug, logical_debug) != debug_fingerprint:
            raise FinalizationError(
                "installed build-private debug sidecar differs from finalization evidence"
            )
        publish_release_bundle(
            repository_root,
            artifact,
            debug,
            evidence,
            publish_artifact,
            publish_debug,
            publish_evidence,
        )
        print(
            f"Finalized {artifact_logical} and published {logical_artifact} "
            f"(build ID {after['buildId']})."
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except (FinalizationError, OSError, ValueError) as error:
        print(f"native release finalization failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
