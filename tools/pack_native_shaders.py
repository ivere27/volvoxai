#!/usr/bin/env python3
"""Generate deterministic C data for VolvoxAI's embedded native shaders.

The generated files contain data and metadata only.  Runtime decompression and
lookup deliberately live elsewhere so this build tool does not impose a native
decoder implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import lzma
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


PACK_FORMAT_VERSION = 1
ALIGNMENT = 4
XZ_DICTIONARY_SIZE = 1 << 20
MAX_U32 = (1 << 32) - 1
BROWSER_ONLY_MARKER = b"// @volvoxai-browser-only"
NATIVE_SPV_ONLY_MARKER = b"// @volvoxai-native-spv-only"


@dataclass(frozen=True)
class Backend:
    name: str
    suffix: str
    c_name: str


BACKENDS = (
    Backend("spv", ".spv", "VOLVOXAI_SHADER_BACKEND_SPV"),
    Backend("glsl", ".comp", "VOLVOXAI_SHADER_BACKEND_GLSL"),
    Backend("gles", ".comp", "VOLVOXAI_SHADER_BACKEND_GLES"),
    Backend("metal", ".metal", "VOLVOXAI_SHADER_BACKEND_METAL"),
)

SCOPE_INFERENCE = "inference"
SCOPE_TRAINING = "training"
SCOPES = (SCOPE_INFERENCE, SCOPE_TRAINING)
SCOPE_C_NAMES = {
    SCOPE_INFERENCE: "VOLVOXAI_SHADER_SCOPE_INFERENCE",
    SCOPE_TRAINING: "VOLVOXAI_SHADER_SCOPE_TRAINING",
}


class PackError(ValueError):
    """Raised when shader inputs cannot form an unambiguous pack."""


@dataclass(frozen=True)
class SourceShader:
    path: str
    stem: str
    scope: str
    data: bytes
    backend_names: tuple[str, ...] | None


@dataclass(frozen=True)
class CompiledShader:
    path: str
    backend: Backend
    scope: str
    data: bytes


@dataclass(frozen=True)
class ShaderRecord:
    path: str
    block: int
    offset: int
    size: int


@dataclass(frozen=True)
class ShaderBlock:
    backend: Backend
    scope: str
    compressed_data: bytes
    uncompressed_size: int


@dataclass(frozen=True)
class ShaderPack:
    profile: str
    input_hash: str
    blocks: tuple[ShaderBlock, ...]
    records: tuple[ShaderRecord, ...]


def _read_training_list(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None

    entries: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        entry = raw_line.split("#", 1)[0].strip().replace("\\", "/")
        if not entry:
            continue
        while entry.startswith("./"):
            entry = entry[2:]
        pure = PurePosixPath(entry)
        if pure.is_absolute() or ".." in pure.parts:
            raise PackError(
                f"{path}:{line_number}: training shader paths must be relative "
                "to --shader-source-dir"
            )
        if entry.endswith(".wgsl"):
            entry = entry[: -len(".wgsl")]
        entries.append(entry)
    return tuple(entries)


def _resolve_training_entries(
    unscoped_paths: Sequence[str], entries: Sequence[str]
) -> set[str]:
    by_relative_stem: dict[str, str] = {}
    by_basename_stem: dict[str, list[str]] = {}
    for path in unscoped_paths:
        relative_stem = path[: -len(".wgsl")]
        by_relative_stem[relative_stem] = path
        by_basename_stem.setdefault(PurePosixPath(path).stem, []).append(path)

    resolved: set[str] = set()
    for entry in entries:
        path = by_relative_stem.get(entry)
        if path is None and "/" not in entry:
            matches = by_basename_stem.get(entry, [])
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                formatted = ", ".join(sorted(matches))
                raise PackError(
                    f"training shader {entry!r} is ambiguous; use one of: {formatted}"
                )
        if path is None:
            raise PackError(
                f"training shader {entry!r} does not name an unscoped WGSL input"
            )
        resolved.add(path)
    return resolved


def discover_source_shaders(
    shader_source_dir: Path, training_list: Path | None = None
) -> tuple[SourceShader, ...]:
    if not shader_source_dir.is_dir():
        raise PackError(f"shader source directory does not exist: {shader_source_dir}")

    source_paths = sorted(
        (
            path
            for path in shader_source_dir.rglob("*.wgsl")
            if path.is_file()
            and path.read_bytes().splitlines()[:1] != [BROWSER_ONLY_MARKER]
        ),
        key=lambda path: path.relative_to(shader_source_dir).as_posix(),
    )
    if not source_paths:
        raise PackError(f"no WGSL sources found under {shader_source_dir}")

    relative_paths = [path.relative_to(shader_source_dir).as_posix() for path in source_paths]
    unscoped_paths = [
        path
        for path in relative_paths
        if PurePosixPath(path).parts[0] not in SCOPES
    ]
    training_entries = _read_training_list(training_list)
    if unscoped_paths and training_entries is None:
        raise PackError(
            "unscoped WGSL sources require --training-list; place sources under "
            "inference/ and training/ to make their ownership self-describing"
        )
    if not unscoped_paths and training_entries:
        raise PackError(
            "--training-list contains entries, but all WGSL sources already use "
            "inference/ or training/ directories"
        )

    training_paths = _resolve_training_entries(unscoped_paths, training_entries or ())
    sources: list[SourceShader] = []
    for absolute_path, relative_path in zip(source_paths, relative_paths):
        first_part = PurePosixPath(relative_path).parts[0]
        if first_part in SCOPES:
            scope = first_part
        else:
            scope = SCOPE_TRAINING if relative_path in training_paths else SCOPE_INFERENCE
        data = absolute_path.read_bytes()
        first_line = data.splitlines()[:1]
        sources.append(
            SourceShader(
                path=relative_path,
                stem=PurePosixPath(relative_path).stem,
                scope=scope,
                data=data,
                backend_names=("spv",) if first_line == [NATIVE_SPV_ONLY_MARKER]
                else None,
            )
        )

    by_stem: dict[str, list[str]] = {}
    for source in sources:
        by_stem.setdefault(source.stem, []).append(source.path)
    duplicates = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(
            f"{stem}: {', '.join(sorted(paths))}"
            for stem, paths in sorted(duplicates.items())
        )
        raise PackError(
            "WGSL basenames must be unique because compiled shader names are flat; " + details
        )

    return tuple(sources)


def _source_for_artifact(
    artifact_stem: str, sources: Sequence[SourceShader]
) -> SourceShader:
    matches = [
        source
        for source in sources
        if artifact_stem == source.stem or artifact_stem.startswith(source.stem + "_")
    ]
    if not matches:
        raise PackError(
            f"compiled shader {artifact_stem!r} has no matching WGSL source"
        )
    longest = max(len(source.stem) for source in matches)
    matches = [source for source in matches if len(source.stem) == longest]
    if len(matches) != 1:
        formatted = ", ".join(sorted(source.path for source in matches))
        raise PackError(
            f"compiled shader {artifact_stem!r} matches multiple WGSL sources: {formatted}"
        )
    return matches[0]


def discover_compiled_shaders(
    compiled_dir: Path,
    sources: Sequence[SourceShader],
    profile: str,
    backends: Sequence[Backend] = BACKENDS,
) -> tuple[CompiledShader, ...]:
    if not compiled_dir.is_dir():
        raise PackError(f"compiled shader directory does not exist: {compiled_dir}")

    included_scopes = (
        {SCOPE_INFERENCE}
        if profile == SCOPE_INFERENCE
        else {SCOPE_INFERENCE, SCOPE_TRAINING}
    )
    artifacts: list[CompiledShader] = []
    for backend in backends:
        backend_dir = compiled_dir / backend.name
        if not backend_dir.is_dir():
            raise PackError(f"compiled backend directory does not exist: {backend_dir}")
        backend_paths = sorted(
            (
                path
                for path in backend_dir.rglob(f"*{backend.suffix}")
                if path.is_file()
            ),
            key=lambda path: path.relative_to(compiled_dir).as_posix(),
        )
        for path in backend_paths:
            source = _source_for_artifact(path.stem, sources)
            if source.scope not in included_scopes:
                continue
            if source.backend_names is not None and backend.name not in source.backend_names:
                continue
            artifacts.append(
                CompiledShader(
                    path=path.relative_to(compiled_dir).as_posix(),
                    backend=backend,
                    scope=source.scope,
                    data=path.read_bytes(),
                )
            )

    if backends and not artifacts:
        raise PackError(f"no compiled shaders selected for the {profile!r} profile")
    return tuple(sorted(artifacts, key=lambda artifact: artifact.path))


def _hash_field(digest: "hashlib._Hash", label: str, data: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(struct.pack(">I", len(label_bytes)))
    digest.update(label_bytes)
    digest.update(struct.pack(">Q", len(data)))
    digest.update(data)


def calculate_input_hash(
    profile: str,
    sources: Sequence[SourceShader],
    artifacts: Sequence[CompiledShader],
    backends: Sequence[Backend] = BACKENDS,
) -> str:
    included_scopes = (
        {SCOPE_INFERENCE}
        if profile == SCOPE_INFERENCE
        else {SCOPE_INFERENCE, SCOPE_TRAINING}
    )
    digest = hashlib.sha256()
    _hash_field(
        digest,
        "format",
        f"volvoxai-native-shader-pack-v{PACK_FORMAT_VERSION}".encode("ascii"),
    )
    _hash_field(digest, "profile", profile.encode("ascii"))
    _hash_field(
        digest,
        "backends",
        ",".join(backend.name for backend in backends).encode("ascii"),
    )
    for source in sorted(sources, key=lambda item: item.path):
        if source.scope not in included_scopes:
            continue
        if source.backend_names is not None and not any(
            backend.name in source.backend_names for backend in backends
        ):
            continue
        _hash_field(
            digest,
            f"source:{source.scope}:{source.path}",
            source.data,
        )
    for artifact in sorted(artifacts, key=lambda item: item.path):
        _hash_field(
            digest,
            f"compiled:{artifact.scope}:{artifact.path}",
            artifact.data,
        )
    return digest.hexdigest()


def _check_u32(value: int, description: str) -> None:
    if not 0 <= value <= MAX_U32:
        raise PackError(f"{description} exceeds the generated format's uint32_t limit")


def build_pack(
    compiled_dir: Path | str,
    shader_source_dir: Path | str,
    profile: str,
    training_list: Path | str | None = None,
    backend_names: Sequence[str] | None = None,
) -> ShaderPack:
    if profile not in (SCOPE_INFERENCE, "full"):
        raise PackError("profile must be 'inference' or 'full'")

    compiled_path = Path(compiled_dir)
    source_path = Path(shader_source_dir)
    training_list_path = Path(training_list) if training_list is not None else None
    backend_by_name = {backend.name: backend for backend in BACKENDS}
    if backend_names is None:
        selected_backends = BACKENDS
    else:
        unknown = sorted(set(backend_names) - set(backend_by_name))
        if unknown:
            raise PackError(f"unknown shader backend(s): {', '.join(unknown)}")
        if len(set(backend_names)) != len(backend_names):
            raise PackError("shader backend names must be unique")
        selected = set(backend_names)
        selected_backends = tuple(
            backend for backend in BACKENDS if backend.name in selected
        )
    sources = discover_source_shaders(source_path, training_list_path)
    artifacts = discover_compiled_shaders(
        compiled_path, sources, profile, selected_backends
    )

    scopes = (SCOPE_INFERENCE,) if profile == SCOPE_INFERENCE else SCOPES
    records: list[ShaderRecord] = []
    blocks: list[ShaderBlock] = []
    for backend in selected_backends:
        for scope in scopes:
            block_index = len(blocks)
            payload = bytearray()
            block_artifacts = sorted(
                (
                    artifact
                    for artifact in artifacts
                    if artifact.backend == backend and artifact.scope == scope
                ),
                key=lambda artifact: artifact.path,
            )
            for artifact in block_artifacts:
                offset = len(payload)
                _check_u32(offset, f"offset for {artifact.path}")
                _check_u32(len(artifact.data), f"size for {artifact.path}")
                records.append(
                    ShaderRecord(
                        path=artifact.path,
                        block=block_index,
                        offset=offset,
                        size=len(artifact.data),
                    )
                )
                payload.extend(artifact.data)
                payload.extend(b"\x00" * (-len(payload) % ALIGNMENT))

            _check_u32(len(payload), f"uncompressed {backend.name}/{scope} block")
            compressed = lzma.compress(
                bytes(payload),
                format=lzma.FORMAT_XZ,
                check=lzma.CHECK_CRC32,
                filters=[
                    {
                        "id": lzma.FILTER_LZMA2,
                        "preset": 9 | lzma.PRESET_EXTREME,
                        # A fixed 1 MiB dictionary keeps native decoder memory
                        # bounded while covering every current shader block.
                        "dict_size": XZ_DICTIONARY_SIZE,
                    }
                ],
            )
            _check_u32(len(compressed), f"compressed {backend.name}/{scope} block")
            blocks.append(
                ShaderBlock(
                    backend=backend,
                    scope=scope,
                    compressed_data=compressed,
                    uncompressed_size=len(payload),
                )
            )

    input_hash = calculate_input_hash(profile, sources, artifacts, selected_backends)
    return ShaderPack(
        profile=profile,
        input_hash=input_hash,
        blocks=tuple(blocks),
        records=tuple(sorted(records, key=lambda record: record.path)),
    )


def _c_string(value: str) -> str:
    encoded = value.encode("utf-8")
    parts: list[str] = ['"']
    for byte in encoded:
        if byte == ord('"'):
            parts.append('\\"')
        elif byte == ord("\\"):
            parts.append("\\\\")
        elif 0x20 <= byte <= 0x7E:
            parts.append(chr(byte))
        else:
            # Three-digit octal escapes cannot absorb following path characters.
            parts.append(f"\\{byte:03o}")
    parts.append('"')
    return "".join(parts)


def _format_byte_array(data: bytes) -> str:
    lines: list[str] = []
    for start in range(0, len(data), 12):
        chunk = data[start : start + 12]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    return "\n".join(lines)


def _generated_banner(input_hash: str) -> str:
    return (
        "/* DO NOT EDIT: generated by tools/pack_native_shaders.py. */\n"
        f"/* Input SHA-256: {input_hash} */\n"
    )


def render_header(pack: ShaderPack) -> str:
    return _generated_banner(pack.input_hash) + f"""
#ifndef VOLVOXAI_EMBEDDED_SHADERS_GENERATED_H
#define VOLVOXAI_EMBEDDED_SHADERS_GENERATED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern \"C\" {{
#endif

#define VOLVOXAI_EMBEDDED_SHADER_PACK_VERSION UINT32_C({PACK_FORMAT_VERSION})
#define VOLVOXAI_EMBEDDED_SHADER_ALIGNMENT UINT32_C({ALIGNMENT})
#define VOLVOXAI_EMBEDDED_SHADER_XZ_DICTIONARY_SIZE UINT32_C({XZ_DICTIONARY_SIZE})

typedef enum VolvoxAIShaderBackend {{
    VOLVOXAI_SHADER_BACKEND_SPV = 0,
    VOLVOXAI_SHADER_BACKEND_GLSL = 1,
    VOLVOXAI_SHADER_BACKEND_GLES = 2,
    VOLVOXAI_SHADER_BACKEND_METAL = 3
}} VolvoxAIShaderBackend;

typedef enum VolvoxAIShaderScope {{
    VOLVOXAI_SHADER_SCOPE_INFERENCE = 0,
    VOLVOXAI_SHADER_SCOPE_TRAINING = 1
}} VolvoxAIShaderScope;

typedef struct VolvoxAIEmbeddedShaderBlock {{
    VolvoxAIShaderBackend backend;
    VolvoxAIShaderScope scope;
    /* An XZ stream using LZMA2 and a CRC32 integrity check. */
    const uint8_t *compressed_data;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
}} VolvoxAIEmbeddedShaderBlock;

typedef struct VolvoxAIEmbeddedShaderRecord {{
    /* Compiled-root-relative path. The generated table is sorted by this key. */
    const char *path;
    /* Index into volvoxai_embedded_shader_blocks. */
    uint32_t block;
    /* offset and size address bytes in the decompressed block. */
    uint32_t offset;
    uint32_t size;
}} VolvoxAIEmbeddedShaderRecord;

extern const char volvoxai_embedded_shader_input_hash[65];
extern const char volvoxai_embedded_shader_profile[];
extern const VolvoxAIEmbeddedShaderBlock volvoxai_embedded_shader_blocks[];
extern const size_t volvoxai_embedded_shader_block_count;
extern const VolvoxAIEmbeddedShaderRecord volvoxai_embedded_shader_records[];
extern const size_t volvoxai_embedded_shader_record_count;

#ifdef __cplusplus
}}
#endif

#endif
"""


def render_c(pack: ShaderPack, header_name: str) -> str:
    sections = [
        _generated_banner(pack.input_hash),
        f"#include {_c_string(header_name)}\n\n",
    ]
    for index, block in enumerate(pack.blocks):
        sections.append(
            f"static const uint8_t volvoxai_embedded_shader_block_{index}[] = {{\n"
            f"{_format_byte_array(block.compressed_data)}\n"
            "};\n\n"
        )

    sections.append(
        "const char volvoxai_embedded_shader_input_hash[65] =\n"
        f"    {_c_string(pack.input_hash)};\n"
        "const char volvoxai_embedded_shader_profile[] =\n"
        f"    {_c_string(pack.profile)};\n\n"
        "const VolvoxAIEmbeddedShaderBlock volvoxai_embedded_shader_blocks[] = {\n"
    )
    for index, block in enumerate(pack.blocks):
        sections.append(
            "    { "
            f"{block.backend.c_name}, {SCOPE_C_NAMES[block.scope]}, "
            f"volvoxai_embedded_shader_block_{index}, "
            f"UINT32_C({len(block.compressed_data)}), "
            f"UINT32_C({block.uncompressed_size})"
            " },\n"
        )
    if not pack.blocks:
        sections.append(
            "    { VOLVOXAI_SHADER_BACKEND_SPV, VOLVOXAI_SHADER_SCOPE_INFERENCE, "
            "NULL, UINT32_C(0), UINT32_C(0) },\n"
        )
    sections.append(
        "};\n"
        "const size_t volvoxai_embedded_shader_block_count = "
        f"{len(pack.blocks)};\n\n"
        "const VolvoxAIEmbeddedShaderRecord volvoxai_embedded_shader_records[] = {\n"
    )
    for record in pack.records:
        sections.append(
            "    { "
            f"{_c_string(record.path)}, UINT32_C({record.block}), "
            f"UINT32_C({record.offset}), UINT32_C({record.size})"
            " },\n"
        )
    if not pack.records:
        sections.append("    { NULL, UINT32_C(0), UINT32_C(0), UINT32_C(0) },\n")
    sections.append(
        "};\n"
        "const size_t volvoxai_embedded_shader_record_count = "
        f"{len(pack.records)};\n"
    )
    return "".join(sections)


def _write_if_changed(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    try:
        if path.read_bytes() == encoded:
            return
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_pack(pack: ShaderPack, output_c: Path | str, output_h: Path | str) -> None:
    c_path = Path(output_c)
    h_path = Path(output_h)
    _write_if_changed(h_path, render_header(pack))
    _write_if_changed(c_path, render_c(pack, h_path.name))


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pack compiled VolvoxAI shaders into deterministic, backend/scope XZ "
            "blocks and generate their C metadata."
        )
    )
    parser.add_argument("--compiled-dir", required=True, type=Path)
    parser.add_argument("--shader-source-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("inference", "full"))
    parser.add_argument("--output-c", required=True, type=Path)
    parser.add_argument("--output-h", required=True, type=Path)
    parser.add_argument(
        "--backend",
        action="append",
        choices=tuple(backend.name for backend in BACKENDS) + ("none",),
        help=(
            "Compiled backend format to include; repeat for multiple formats. "
            "Omit for all formats, or pass 'none' by itself for an empty pack."
        ),
    )
    parser.add_argument(
        "--training-list",
        type=Path,
        help=(
            "UTF-8 file with one training WGSL path or basename per line. Required "
            "for the current flat shader source layout; unnecessary once sources "
            "live under inference/ and training/."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    backend_names = args.backend
    if backend_names and "none" in backend_names:
        if len(backend_names) != 1:
            parser.error("--backend none cannot be combined with another backend")
        backend_names = ()
    try:
        pack = build_pack(
            compiled_dir=args.compiled_dir,
            shader_source_dir=args.shader_source_dir,
            profile=args.profile,
            training_list=args.training_list,
            backend_names=backend_names,
        )
        write_pack(pack, args.output_c, args.output_h)
    except (OSError, PackError, lzma.LZMAError) as error:
        parser.error(str(error))

    compressed_size = sum(len(block.compressed_data) for block in pack.blocks)
    uncompressed_size = sum(block.uncompressed_size for block in pack.blocks)
    print(
        f"Packed {len(pack.records)} shaders into {len(pack.blocks)} blocks "
        f"({uncompressed_size} -> {compressed_size} bytes), input "
        f"{pack.input_hash}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
