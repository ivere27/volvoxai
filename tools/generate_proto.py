#!/usr/bin/env python3
"""Generate VolvoxAI bindings with the pinned Synurang generator.

Download the official, digest-verified GitHub release generator and matching
runtime sources into ``build/``. Normal builds use committed projections and
need no network or adjacent checkout. Generated sources and vendored runtimes
carry deterministic manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNURANG_VERSION = "0.8.0"
GENERATOR_NAME = "protoc-gen-synurang-ffi"
SYNURANG_REVISION = "53180b484cf7ca07a1e7d6f24e58b8a19a2dcfa8"
SYNURANG_RELEASE_URL = f"https://github.com/ivere27/synurang/releases/tag/v{SYNURANG_VERSION}"
SYNURANG_SOURCE_URL = (
    f"https://codeload.github.com/ivere27/synurang/tar.gz/refs/tags/v{SYNURANG_VERSION}"
)
SYNURANG_SOURCE_SHA256 = "c0c0615a824565fa4ee581ae49993a361b65d342bd334039f1411cfe7bdbd4f6"
# Published SHA256SUMS, also recorded in GitHub release asset metadata.
GENERATOR_ARCHIVES = {
    "x86_64-unknown-linux-musl": (
        "tar.gz", "759148a4a1568f73c39d8fa127c71204226ef56a1cdab53ec0778b69e6f9db0b",
    ),
    "aarch64-unknown-linux-musl": (
        "tar.gz", "7f5196ce1eb0818d7b6993084922bb038c07d9468b931ee3ce6d0b49dffe294d",
    ),
    "x86_64-pc-windows-gnu": (
        "zip", "07226a91ba8cbaf7cf996824612231d5d818967d974bd0700e760dc5e50617d9",
    ),
}
SYNURANG_LABEL = f"v{SYNURANG_VERSION} ({SYNURANG_REVISION[:12]})"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_GENERATOR_BYTES = 32 * 1024 * 1024
PYTHON_RUNTIME_FILES = ("errors.py", "module.py", "protolite.py", "proto.py")
NATIVE_RUNTIME_FILES = (
    "LICENSE",
    "include/synurang/c_runtime.h",
    "include/synurang/call.h",
    "include/synurang/module_host.h",
    "src/c_runtime.c",
    "src/call.c",
    "src/wasm.c",
    "src/module_host.c",
)
PROFILE_FILTER = REPOSITORY_ROOT / "tools" / "filter_codegen_request.py"
try:
    from .generate_proto_enums import (
        FULL_ONLY_OPERATION_CODES, INFERENCE_SERVICES as INFERENCE_SERVICE_NAMES)
except ImportError:
    from generate_proto_enums import (
        FULL_ONLY_OPERATION_CODES, INFERENCE_SERVICES as INFERENCE_SERVICE_NAMES)

# Fully qualified for the descriptor filter, from the one boundary definition.
INFERENCE_SERVICES = tuple(
    f"volvoxai.v1.{name}" for name in INFERENCE_SERVICE_NAMES)

INFERENCE_OMITTED_ENUM_VALUES = (
    *("volvoxai.v1.OperationCode." + name for name in sorted(FULL_ONLY_OPERATION_CODES)),
    "volvoxai.v1.OperationStage.OPERATION_STAGE_TRAINER_CREATE",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_TRAINER_INPUT",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_TRAINER_STEP",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_TRAINER_COMMIT",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_TRAINER_ROLLBACK",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_TRAINER_EXPORT",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_PTQ_CREATE",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_PTQ_CALIBRATE",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_PTQ_INSPECT",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_PTQ_WRITE",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_PTQ_CLOSE",
    "volvoxai.v1.OperationStage.OPERATION_STAGE_PTQ_AUTHOR",
    "volvoxai.v1.MemoryOwnerKind.MEMORY_OWNER_KIND_TRAINER",
    "volvoxai.v1.MemoryOwnerKind.MEMORY_OWNER_KIND_PTQ_PLAN",
    "volvoxai.v1.MemoryResourceRole.MEMORY_RESOURCE_ROLE_TRAINING_WORKING_WEIGHTS",
    "volvoxai.v1.MemoryResourceRole.MEMORY_RESOURCE_ROLE_TRAINING_GRADIENTS",
    "volvoxai.v1.MemoryResourceRole.MEMORY_RESOURCE_ROLE_TRAINING_OPTIMIZER_SLOTS",
    "volvoxai.v1.MemoryResourceRole.MEMORY_RESOURCE_ROLE_TRAINING_ACCUMULATION",
    "volvoxai.v1.MemoryResourceRole.MEMORY_RESOURCE_ROLE_PTQ_OBSERVER_STATE",
)


@dataclass(frozen=True)
class GenerationSpec:
    option: str
    destination: str
    subdirectory: str = ""
    services: tuple[str, ...] = ()
    omitted_enum_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    files: dict[Path, bytes]
    mode_files: dict[str, tuple[Path, ...]]
    mode_roots: dict[str, Path]


GENERATION_SPECS = {
    "c": GenerationSpec(
        # Module generation includes the message codec and module dispatch.
        # Short enum names avoid repeating the protobuf enum prefix in C.
        option="lang=c,mode=module,enum_names=short",
        destination="c",
    ),
    "c-inference": GenerationSpec(
        option="lang=c,mode=module,enum_names=short",
        destination="c",
        subdirectory="inference",
        services=INFERENCE_SERVICES,
        omitted_enum_values=INFERENCE_OMITTED_ENUM_VALUES,
    ),
    "typescript": GenerationSpec(
        option="lang=typescript,mode=client",
        destination="typescript",
    ),
    "typescript-inference": GenerationSpec(
        option="lang=typescript,mode=client",
        destination="typescript",
        subdirectory="inference",
        services=INFERENCE_SERVICES,
        omitted_enum_values=INFERENCE_OMITTED_ENUM_VALUES,
    ),
    "python": GenerationSpec(
        option="lang=python,mode=client",
        destination="python",
    ),
}
GENERATION_ORDER = tuple(GENERATION_SPECS)


class CodegenError(RuntimeError):
    """A user-facing deterministic generation failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive_bytes(data: bytes, expected: str, source: str) -> None:
    if len(data) > MAX_ARCHIVE_BYTES:
        raise CodegenError(f"Synurang archive exceeds the {MAX_ARCHIVE_BYTES}-byte limit: {source}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise CodegenError(
            f"Synurang archive SHA-256 mismatch for {source}: "
            f"expected {expected}, got {actual}"
        )


def cached_archive(
    cache_root: Path, filename: str, url: str, expected: str, offline: bool,
) -> bytes:
    path = cache_root / f"v{SYNURANG_VERSION}" / "archives" / filename
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink():
            raise CodegenError(f"cached archive is not a regular file: {path}")
        try:
            with path.open("rb") as handle:
                data = handle.read(MAX_ARCHIVE_BYTES + 1)
        except OSError as error:
            raise CodegenError(f"could not read cached archive {path}: {error}") from error
        verify_archive_bytes(data, expected, str(path))
        return data
    if offline:
        raise CodegenError(f"offline cache is missing {path}; run --fetch-only online first")
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "volvoxai-codegen"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_ARCHIVE_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise CodegenError(f"could not download {url}: {error}") from error
    verify_archive_bytes(data, expected, url)
    atomic_write(path, data)
    return data


def source_archive_bytes(cache_root: Path, offline: bool) -> bytes:
    return cached_archive(
        cache_root, f"synurang-v{SYNURANG_VERSION}.tar.gz",
        SYNURANG_SOURCE_URL, SYNURANG_SOURCE_SHA256, offline,
    )


def source_files(archive_data: bytes) -> dict[str, bytes]:
    prefix = f"synurang-{SYNURANG_VERSION}/"
    files: dict[str, bytes] = {}
    total_size = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.name.startswith(prefix):
                    raise CodegenError(f"invalid Synurang source path: {member.name}")
                relative = Path(member.name[len(prefix):])
                if (not member.isfile() or relative.is_absolute() or
                        ".." in relative.parts or member.size > MAX_ARCHIVE_BYTES):
                    raise CodegenError(f"invalid Synurang source path: {member.name}")
                total_size += member.size
                if total_size > MAX_ARCHIVE_BYTES:
                    raise CodegenError("expanded Synurang source exceeds the size limit")
                name = relative.as_posix()
                if name in files:
                    raise CodegenError(f"duplicate Synurang source path: {member.name}")
                with archive.extractfile(member) as source:
                    files[name] = source.read()
    except (OSError, EOFError, tarfile.TarError) as error:
        raise CodegenError(f"could not extract Synurang source: {error}") from error
    return files


def generator_target() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        machine = "x86_64"
    elif machine in {"arm64", "aarch64"}:
        machine = "aarch64"
    targets = {
        ("linux", "x86_64"): "x86_64-unknown-linux-musl",
        ("linux", "aarch64"): "aarch64-unknown-linux-musl",
        ("windows", "x86_64"): "x86_64-pc-windows-gnu",
    }
    target = targets.get((system, machine))
    if target is None:
        raise CodegenError(
            f"Synurang {SYNURANG_LABEL} has no release generator for {system}/{machine}; "
            "use --generator to select an existing executable"
        )
    return target


def generator_archive_name(target: str) -> str:
    extension, _ = GENERATOR_ARCHIVES[target]
    return f"{GENERATOR_NAME}-{SYNURANG_VERSION}-{target}.{extension}"


def extract_generator(archive_data: bytes, target: str) -> tuple[str, bytes]:
    name = GENERATOR_NAME + (".exe" if "windows" in target else "")
    member_name = f"{GENERATOR_NAME}-{SYNURANG_VERSION}-{target}/{name}"
    try:
        if GENERATOR_ARCHIVES[target][0] == "zip":
            with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
                member = archive.getinfo(member_name)
                if not 0 < member.file_size <= MAX_GENERATOR_BYTES:
                    raise CodegenError("release generator has an invalid size")
                data = archive.read(member)
        else:
            with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as archive:
                member = archive.getmember(member_name)
                if not member.isfile() or not 0 < member.size <= MAX_GENERATOR_BYTES:
                    raise CodegenError("release generator is not a regular file with a valid size")
                with archive.extractfile(member) as source:
                    data = source.read()
    except (OSError, EOFError, KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise CodegenError(f"could not extract release generator: {error}") from error
    return name, data


def release_generator(cache_root: Path, offline: bool) -> Path:
    target = generator_target()
    filename = generator_archive_name(target)
    url = f"https://github.com/ivere27/synurang/releases/download/v{SYNURANG_VERSION}/{filename}"
    archive = cached_archive(cache_root, filename, url, GENERATOR_ARCHIVES[target][1], offline)
    name, binary = extract_generator(archive, target)
    executable = cache_root / f"v{SYNURANG_VERSION}" / target / name
    # Compare against the verified archive on every use; no mutable receipt is
    # trusted as an independent source of executable identity.
    expected = hashlib.sha256(binary).hexdigest()
    if (not executable.is_file() or executable.is_symlink() or
            executable.stat().st_size != len(binary) or
            sha256_file(executable) != expected or not os.access(executable, os.X_OK)):
        atomic_write(executable, binary, mode=0o755)
    print(f"Using Synurang {SYNURANG_LABEL} release generator {executable}")
    return executable.resolve()


def resolve_generator(args: argparse.Namespace) -> Path:
    if args.generator:
        executable = Path(args.generator).expanduser().resolve()
        if not executable.is_file():
            raise CodegenError(f"generator override is not a file: {executable}")
        if not os.access(executable, os.X_OK):
            raise CodegenError(f"generator override is not executable: {executable}")
        print(
            f"Using generator override {executable} "
            f"(expected Synurang {SYNURANG_LABEL})"
        )
        return executable
    return release_generator(args.cache_dir, args.offline)


def resolve_executable(command: str, label: str) -> str:
    if os.sep in command or (os.altsep and os.altsep in command):
        candidate = Path(command).expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise CodegenError(f"{label} is not executable: {candidate}")
        return str(candidate)
    found = shutil.which(command)
    if found is None:
        raise CodegenError(f"{label} was not found on PATH: {command}")
    return found


def parse_languages(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return GENERATION_ORDER
    selected: set[str] = set()
    for value in values:
        for language in value.split(","):
            language = language.strip()
            if language == "all":
                return GENERATION_ORDER
            if language not in GENERATION_SPECS:
                expected = ", ".join(("all", *GENERATION_ORDER))
                raise CodegenError(
                    f"unknown language {language!r}; expected one of: {expected}"
                )
            selected.add(language)
    return tuple(language for language in GENERATION_ORDER if language in selected)


def provenance_line(suffix: str, proto_sha256: str) -> str:
    content = (
        f"Synurang generator: {SYNURANG_VERSION}; revision: {SYNURANG_REVISION}; "
        f"proto SHA-256: {proto_sha256}"
    )
    if suffix in {".c", ".h"}:
        return f"/* {content} */\n"
    if suffix == ".py":
        return f"# {content}\n"
    return f"// {content}\n"


def add_provenance(data: bytes, suffix: str, proto_sha256: str) -> bytes:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CodegenError("generator emitted a non-UTF-8 source file") from error
    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines else 0
    lines.insert(insert_at, provenance_line(suffix, proto_sha256))
    return "".join(lines).encode("utf-8")


def generated_bytes(
    path: Path,
    proto_sha256: str,
) -> bytes:
    data = path.read_bytes()
    if path.name.endswith("_lite.ts"):
        data = typescript_bulk_writer(data)
    if path.name == "synurang_runtime.ts":
        # New TypeScript typed-array generics infer ArrayBuffer for defaults.
        # These parameters also accept views backed by ArrayBufferLike. Keep
        # the upstream behavior and add only the two missing type annotations.
        original = b"details = new Uint8Array()"
        if data.count(original) != 2:
            raise CodegenError("Synurang runtime byte-array annotations need review")
        data = data.replace(original, b"details: Uint8Array = new Uint8Array()")
    data = add_provenance(data, path.suffix, proto_sha256)
    return data.rstrip(b"\n") + b"\n"


def typescript_bulk_writer(data: bytes) -> bytes:
    """Keep the release codec's wire format while copying byte fields in bulk.

    The pinned writer expands every tensor byte into a JavaScript number array
    at each enclosing message. A growing typed buffer preserves immediate-copy
    ownership and independent returned snapshots without that amplification.
    Review the transformation if the pinned generator changes its writer.
    """
    replacements = (
        (b'''  private readonly bytes: number[] = [];

  toUint8Array(): Uint8Array {
    return new Uint8Array(this.bytes);
  }
''', b'''  private buffer = new Uint8Array(128);
  private length = 0;

  toUint8Array(): Uint8Array {
    return this.buffer.slice(0, this.length);
  }

  private reserve(extra: number): void {
    const required = this.length + extra;
    if (required <= this.buffer.length) return;
    const grown = new Uint8Array(Math.max(required, this.buffer.length * 2));
    grown.set(this.buffer.subarray(0, this.length));
    this.buffer = grown;
  }

  private writeByte(value: number): void {
    this.reserve(1);
    this.buffer[this.length++] = value;
  }
'''),
        (b'''  private writeRaw(data: Uint8Array): void {
    for (const byte of data) {
      this.bytes.push(byte);
    }
  }
''', b'''  private writeRaw(data: Uint8Array): void {
    this.reserve(data.byteLength);
    this.buffer.set(data, this.length);
    this.length += data.byteLength;
  }
'''),
        (b"this.bytes.push(Number((v & 0x7fn) | 0x80n));",
         b"this.writeByte(Number((v & 0x7fn) | 0x80n));"),
        (b"this.bytes.push(Number(v));", b"this.writeByte(Number(v));"),
    )
    for before, after in replacements:
        if data.count(before) != 1:
            raise CodegenError("Synurang TypeScript bulk writer needs review")
        data = data.replace(before, after)
    return data


def python_runtime_init() -> bytes:
    # The module client needs neither the old plugin ABI nor gRPC transports.
    # Keep its upstream implementation files exact and project only the package
    # exports, so importing synurang never imports an unused transport stack.
    return f'''# Code generated by tools/generate_proto.py. DO NOT EDIT.
"""Synurang module runtime and protobuf codec support for VolvoxAI."""

from .errors import FfiError, PluginClosedError
from .protolite import DecodeError, EncodeError, Field, ProtoError, ProtoMessage
from .module import (ModuleHost, ModuleCall, AsyncModuleHost, AsyncModuleCall,
                     TypedModuleCall, TypedAsyncModuleCall, RequestClosedError)

__all__ = [
    "ModuleHost", "ModuleCall", "AsyncModuleHost", "AsyncModuleCall",
    "TypedModuleCall", "TypedAsyncModuleCall", "RequestClosedError",
    "FfiError", "PluginClosedError", "DecodeError", "EncodeError", "Field",
    "ProtoError", "ProtoMessage",
]

__version__ = "{SYNURANG_VERSION}"
'''.encode("utf-8")


def manifest_path(root: Path, mode: str) -> Path:
    return root / f".synurang-{mode}.manifest.json"


def manifest_bytes(
    mode: str,
    root: Path,
    files: Sequence[Path],
    proto_sha256: str,
    services: Sequence[str],
    omitted_enum_values: Sequence[str],
) -> bytes:
    relative_files = sorted(path.relative_to(root).as_posix() for path in files)
    manifest = {
        "generator": GENERATOR_NAME,
        "generator_version": SYNURANG_VERSION,
        "generator_revision": SYNURANG_REVISION,
        "source_sha256": SYNURANG_SOURCE_SHA256,
        "source_url": SYNURANG_SOURCE_URL,
        "generator_release": SYNURANG_RELEASE_URL,
        "generator_archives": {generator_archive_name(target): digest
                               for target, (_, digest) in GENERATOR_ARCHIVES.items()},
        "generator_options": GENERATION_SPECS[mode].option,
        "mode": mode,
        "proto_sha256": proto_sha256,
        "files": relative_files,
    }
    if services:
        manifest["services"] = list(services)
        manifest["descriptor_filter"] = "transitive-service-type-closure"
    if mode.startswith("typescript"):
        manifest["runtime_type_annotations"] = "explicit-Uint8Array-details"
        manifest["codec_byte_writer"] = "growing-Uint8Array-with-snapshot-copy-v1"
    if mode == "python":
        manifest["runtime_package_exports"] = "module-and-protobuf-only"
    if omitted_enum_values:
        manifest["omitted_enum_values"] = list(omitted_enum_values)
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def protoc_command(
    protoc: str,
    generator: Path,
    proto: Path,
    proto_root: Path,
    includes: Iterable[Path],
    output: Path,
    option: str,
) -> list[str]:
    try:
        proto_argument = proto.relative_to(proto_root).as_posix()
    except ValueError as error:
        raise CodegenError(
            f"proto {proto} is not under --proto-root {proto_root}"
        ) from error

    command = [
        protoc,
        f"--proto_path={proto_root}",
    ]
    seen = {proto_root}
    for include in includes:
        include = include.resolve()
        if include in seen:
            continue
        if not include.is_dir():
            raise CodegenError(f"protobuf include directory does not exist: {include}")
        seen.add(include)
        command.append(f"--proto_path={include}")
    command.extend(
        [
            "--experimental_allow_proto3_optional",
            f"--plugin=protoc-gen-synurang-ffi={generator}",
            f"--synurang-ffi_out={output}",
            f"--synurang-ffi_opt={option}",
            proto_argument,
        ]
    )
    return command


def collect_generation(
    args: argparse.Namespace,
    generator: Path,
    languages: Sequence[str],
) -> GenerationResult:
    upstream = source_files(source_archive_bytes(args.cache_dir, args.offline))
    protoc = resolve_executable(args.protoc, "protoc")
    proto = args.proto.resolve()
    proto_root = (args.proto_root or proto.parent).resolve()
    if not proto.is_file():
        raise CodegenError(f"proto source does not exist: {proto}")
    proto_sha256 = sha256_file(proto)
    destinations = {
        "c": args.c_out.resolve(),
        "typescript": args.typescript_out.resolve(),
        "python": args.python_out.resolve(),
    }
    includes = list(args.proto_path)
    if Path("/usr/include").is_dir():
        includes.append(Path("/usr/include"))

    generated: dict[Path, bytes] = {}
    mode_files: dict[str, tuple[Path, ...]] = {}
    mode_roots: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="volvoxai-synurang-") as temporary:
        temporary_root = Path(temporary)
        def generate_mode(language: str) -> tuple[str, Path]:
            spec = GENERATION_SPECS[language]
            output = temporary_root / language
            output.mkdir(parents=True)
            plugin = generator
            environment = None
            if spec.services:
                if not PROFILE_FILTER.is_file() or not os.access(PROFILE_FILTER, os.X_OK):
                    raise CodegenError(
                        f"profile filter is not executable: {PROFILE_FILTER}"
                    )
                plugin = PROFILE_FILTER
                environment = os.environ.copy()
                environment["VOLVOXAI_SYNURANG_GENERATOR"] = str(generator)
                environment["VOLVOXAI_SYNURANG_SERVICES"] = ",".join(spec.services)
                if spec.omitted_enum_values:
                    environment["VOLVOXAI_SYNURANG_OMIT_ENUM_VALUES"] = ",".join(
                        spec.omitted_enum_values
                    )
            command = protoc_command(
                protoc=protoc,
                generator=plugin,
                proto=proto,
                proto_root=proto_root,
                includes=includes,
                output=output,
                option=spec.option,
            )
            print(f"+ {shlex.join(command)}", flush=True)
            result = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
            )
            if result.returncode != 0:
                raise CodegenError(
                    f"protoc failed for {language} with exit code {result.returncode}"
                )

            return language, output

        # Each protoc invocation writes its own temporary directory. Validate
        # and install their results in declaration order only after all finish.
        with ThreadPoolExecutor(max_workers=min(4, len(languages))) as executor:
            outputs = list(executor.map(generate_mode, languages))
        for language, output in outputs:
            spec = GENERATION_SPECS[language]
            files = sorted(path for path in output.rglob("*") if path.is_file())
            if not files:
                raise CodegenError(f"Synurang emitted no files for {language}")
            root = destinations[spec.destination] / spec.subdirectory
            destinations_for_mode: list[Path] = []
            for path in files:
                relative = path.relative_to(output)
                destination = root / relative
                content = generated_bytes(path, proto_sha256)
                generated[destination] = content
                destinations_for_mode.append(destination)
            # Transport implementations are copied unchanged from the same
            # release as the generator, not independently versioned packages.
            runtime_sources: dict[str, str] = {}
            if spec.destination == "typescript":
                runtime_sources["typescript/src/synurang_wasm.ts"] = "synurang_wasm.ts"
                runtime_sources["LICENSE"] = "SYNURANG-LICENSE"
            elif spec.destination == "python":
                runtime_sources = {
                    f"python/synurang/{name}": f"synurang/{name}"
                    for name in PYTHON_RUNTIME_FILES
                }
                runtime_sources["LICENSE"] = "SYNURANG-LICENSE"
            for source, relative in runtime_sources.items():
                destination = root / relative
                generated[destination] = upstream[source]
                destinations_for_mode.append(destination)
            if spec.destination == "python":
                destination = root / "synurang/__init__.py"
                generated[destination] = python_runtime_init()
                destinations_for_mode.append(destination)
            if spec.destination == "typescript" and not spec.subdirectory:
                # Both profiles use one runtime identity. The implementation
                # lives with inference and has no application service imports.
                for filename in ("synurang_runtime.ts", "synurang_wasm.ts"):
                    module = filename.removesuffix(".ts") + ".js"
                    generated[root / filename] = (
                        "// Code generated by tools/generate_proto.py. DO NOT EDIT.\n"
                        + provenance_line(".ts", proto_sha256)
                        + f"export * from './inference/{module}';\n"
                    ).encode("utf-8")
            mode_files[language] = tuple(destinations_for_mode)
            mode_roots[language] = root
            generated[manifest_path(root, language)] = manifest_bytes(
                language,
                root,
                destinations_for_mode,
                proto_sha256,
                spec.services,
                spec.omitted_enum_values,
            )
    if any(GENERATION_SPECS[language].destination == "c" for language in languages):
        root = REPOSITORY_ROOT / "native/third_party/synurang"
        for name in NATIVE_RUNTIME_FILES:
            generated[root / name] = upstream[name]
        generated[manifest_path(root, "runtime")] = (json.dumps({
            "version": SYNURANG_VERSION,
            "revision": SYNURANG_REVISION,
            "source_sha256": SYNURANG_SOURCE_SHA256,
            "source_url": SYNURANG_SOURCE_URL,
            "files": {name: hashlib.sha256(upstream[name]).hexdigest()
                      for name in NATIVE_RUNTIME_FILES},
        }, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return GenerationResult(
        files=generated,
        mode_files=mode_files,
        mode_roots=mode_roots,
    )


def atomic_write(path: Path, content: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        temporary.chmod(mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def is_synurang_generated(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        prefix = path.read_bytes()[:512]
    except OSError:
        return False
    return (b"Code generated by protoc-gen-synurang-ffi" in prefix or
            b"Generated by protoc-gen-synurang-ffi" in prefix)


def paths_from_previous_manifest(root: Path, mode: str) -> set[Path]:
    path = manifest_path(root, mode)
    if not path.is_file():
        return set()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CodegenError(f"could not read generated-file manifest {path}: {error}") from error
    files = manifest.get("files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        raise CodegenError(f"generated-file manifest has an invalid files list: {path}")

    root = root.resolve()
    result: set[Path] = set()
    for item in files:
        candidate = root / item
        try:
            candidate.resolve().relative_to(root)
        except ValueError as error:
            raise CodegenError(
                f"generated-file manifest path escapes its output root: {item!r}"
            ) from error
        # Delete a manifest-owned symlink itself, never its in-root target.
        result.add(candidate)
    return result


def obsolete_generated_paths(
    result: GenerationResult,
    languages: Sequence[str],
) -> list[Path]:
    expected_sources = {
        path.resolve()
        for files in result.mode_files.values()
        for path in files
    }
    obsolete: set[Path] = set()

    # A subset invocation owns only the modes it is replacing. The preceding
    # manifest records files that mode emitted before a generator/proto change.
    for language in languages:
        root = result.mode_roots[language]
        for previous in paths_from_previous_manifest(root, language):
            if previous not in expected_sources and previous.is_file():
                obsolete.add(previous)

    if tuple(languages) == GENERATION_ORDER:
        # A full pass owns every dedicated output root, so it also catches
        # artifacts from generators that predate these manifests.
        expected_manifests = {
            manifest_path(result.mode_roots[language], language).resolve()
            for language in languages
        }
        for root in {path.resolve() for path in result.mode_roots.values()}:
            if not root.is_dir():
                continue
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                candidate_path = candidate.absolute()
                if candidate_path in expected_sources or candidate_path in expected_manifests:
                    continue
                if (
                    candidate.name.startswith(".synurang-")
                    and candidate.name.endswith(".manifest.json")
                ) or is_synurang_generated(candidate):
                    # Keep the lexical path so unlinking a generated symlink
                    # removes the link, never a target outside the output root.
                    obsolete.add(candidate_path)

    return sorted(obsolete)


def remove_empty_parents(path: Path, roots: set[Path]) -> None:
    parent = path.parent
    while parent not in roots:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def install_or_check(
    result: GenerationResult,
    languages: Sequence[str],
    check: bool,
) -> bool:
    generated = result.files
    stale: list[Path] = []
    for destination in sorted(generated):
        expected = generated[destination]
        if not destination.is_file() or destination.read_bytes() != expected:
            stale.append(destination)
    obsolete = obsolete_generated_paths(result, languages)

    if check:
        if stale or obsolete:
            for destination in stale:
                state = "stale" if destination.exists() else "missing"
                print(f"{state}: {destination}", file=sys.stderr)
            for destination in obsolete:
                print(f"unexpected generated file: {destination}", file=sys.stderr)
            print(
                "generated protobuf bindings are not current; run make proto_codegen",
                file=sys.stderr,
            )
            return False
        print(f"Verified {len(generated)} generated file(s) are current")
        return True

    for destination in sorted(generated):
        if destination in stale:
            atomic_write(destination, generated[destination])
            print(f"Generated {destination}")
        else:
            print(f"Unchanged {destination}")
    roots = {path.resolve() for path in result.mode_roots.values()}
    for destination in obsolete:
        destination.unlink()
        remove_empty_parents(destination, roots)
        print(f"Removed obsolete generated file {destination}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Download the official Synurang {SYNURANG_LABEL} release generator "
            "and generate reproducible C, TypeScript, and Python bindings."
        )
    )
    parser.add_argument(
        "--generator",
        default=os.environ.get("PROTOC_GEN_SYNURANG_FFI"),
        help=(
            "explicit protoc-gen-synurang-ffi executable; "
            f"defaults to the verified GitHub Synurang {SYNURANG_LABEL} release"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPOSITORY_ROOT / "build" / "cache" / "synurang-codegen",
        help="verified release archive/binary cache (default: build/cache/synurang-codegen)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="disable downloads; require verified generator and runtime source archives in cache",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="download and verify generator/runtime archives without running protoc",
    )
    parser.add_argument(
        "--language",
        action="append",
        help=(
            "generation selection; repeat or comma-separate all, c, "
            "c-inference, typescript, typescript-inference, "
            "python (default: all)"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in a temporary directory and fail on missing/stale output",
    )
    parser.add_argument(
        "--protoc",
        default=os.environ.get("PROTOC", "protoc"),
        help="protoc executable (default: $PROTOC or protoc)",
    )
    parser.add_argument(
        "--proto",
        type=Path,
        default=REPOSITORY_ROOT / "proto" / "volvoxai.proto",
        help="source proto (default: proto/volvoxai.proto)",
    )
    parser.add_argument(
        "--proto-root",
        type=Path,
        help="root used for the proto's protoc-relative path (default: proto parent)",
    )
    parser.add_argument(
        "--proto-path",
        type=Path,
        action="append",
        default=[],
        help="additional protobuf include directory; may be repeated",
    )
    parser.add_argument(
        "--c-out",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "generated" / "c",
        help="C module/codec output directory",
    )
    parser.add_argument(
        "--typescript-out",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "generated" / "typescript",
        help="TypeScript output directory",
    )
    parser.add_argument(
        "--python-out",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "generated" / "python",
        help="Python output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.fetch_only and args.check:
        parser.error("--fetch-only and --check cannot be combined")
    try:
        languages = parse_languages(args.language)
        generator = resolve_generator(args)
        print(f"Generator: {generator}")
        if args.fetch_only:
            source_archive_bytes(args.cache_dir, args.offline)
            return 0
        generated = collect_generation(args, generator, languages)
        return 0 if install_or_check(generated, languages, args.check) else 1
    except CodegenError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
