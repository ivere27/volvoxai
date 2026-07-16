#!/usr/bin/env python3
"""Generate VolvoxAI bindings with the pinned Synurang release binary.

The generator is downloaded into ``build/`` (which is ignored by Git), verified
against the release checksum, and invoked through protoc. Generated sources
carry only deterministic provenance, the imports that the Rust plugin-server
template needs in this repository, and one canonical final newline. Per-mode
manifests make stale generated-file removal deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNURANG_VERSION = "0.7.2"
RELEASE_BASE_URL = (
    "https://github.com/ivere27/synurang/releases/download/"
    f"v{SYNURANG_VERSION}"
)
GENERATOR_NAME = "protoc-gen-synurang-ffi"
CHECKSUM_MANIFEST_URL = f"{RELEASE_BASE_URL}/SHA256SUMS"


@dataclass(frozen=True)
class ReleaseAsset:
    target: str
    archive: str
    sha256: str

    @property
    def bundle_directory(self) -> str:
        return self.archive.removesuffix(".tar.gz")


# Pinned from the v0.7.2 SHA256SUMS release asset.  Keep the manifest URL next
# to the values so a version update necessarily reviews both source and digest.
RELEASE_ASSETS = {
    "x86_64-unknown-linux-musl": ReleaseAsset(
        target="x86_64-unknown-linux-musl",
        archive=(
            "protoc-gen-synurang-ffi-0.7.2-"
            "x86_64-unknown-linux-musl.tar.gz"
        ),
        sha256="b5d5e030dc6ab58c0a8ec1db61b46bb865a8894ba4265c2fe30d173a60b87cbc",
    ),
    "aarch64-unknown-linux-musl": ReleaseAsset(
        target="aarch64-unknown-linux-musl",
        archive=(
            "protoc-gen-synurang-ffi-0.7.2-"
            "aarch64-unknown-linux-musl.tar.gz"
        ),
        sha256="3b7ead54bb9a7b13c79e071595d36bf063596c9ffcfb9afbcfdc867b43fee610",
    ),
}


@dataclass(frozen=True)
class GenerationSpec:
    name: str
    option: str
    destination: str
    rust_imports: bool = False


@dataclass(frozen=True)
class GenerationResult:
    files: dict[Path, bytes]
    mode_files: dict[str, tuple[Path, ...]]
    mode_roots: dict[str, Path]


GENERATION_SPECS = {
    "c-lite": GenerationSpec(
        name="c-lite",
        option="lang=c,mode=lite",
        destination="c",
    ),
    "c-native": GenerationSpec(
        name="c-native",
        option="lang=c,mode=native",
        destination="c",
    ),
    "typescript": GenerationSpec(
        name="typescript",
        option="lang=typescript",
        destination="typescript",
    ),
    "rust": GenerationSpec(
        name="rust",
        option="lang=rust,mode=plugin_server",
        destination="rust",
        rust_imports=True,
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


def detect_host_target() -> str:
    if platform.system() != "Linux":
        raise CodegenError(
            "Synurang v0.7.2 publishes no prebuilt generator for "
            f"{platform.system()}; pass --generator with a compatible executable"
        )
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64-unknown-linux-musl"
    if machine in {"aarch64", "arm64"}:
        return "aarch64-unknown-linux-musl"
    raise CodegenError(
        f"Synurang v0.7.2 has no prebuilt Linux generator for {machine}; "
        "pass --generator with a compatible executable"
    )


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.download"
    )
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "volvoxai-synurang-codegen/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        os.replace(temporary, destination)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise CodegenError(f"failed to download {url}: {error}") from error


def verified_archive(asset: ReleaseAsset, cache_root: Path) -> Path:
    release_directory = cache_root / f"v{SYNURANG_VERSION}"
    archive = release_directory / asset.archive
    if archive.is_file():
        actual = sha256_file(archive)
        if actual == asset.sha256:
            return archive
        print(
            f"Discarding corrupt cached archive {archive} "
            f"(expected {asset.sha256}, got {actual})",
            file=sys.stderr,
        )
        archive.unlink()

    url = f"{RELEASE_BASE_URL}/{asset.archive}"
    print(f"Downloading Synurang v{SYNURANG_VERSION}: {url}")
    download_file(url, archive)
    actual = sha256_file(archive)
    if actual != asset.sha256:
        archive.unlink(missing_ok=True)
        raise CodegenError(
            f"checksum mismatch for {asset.archive}: expected {asset.sha256}, "
            f"got {actual}; published manifest: {CHECKSUM_MANIFEST_URL}"
        )
    print(f"Verified SHA-256 {actual}")
    return archive


def extract_generator(
    archive: Path,
    asset: ReleaseAsset,
    cache_root: Path,
) -> Path:
    install_directory = (
        cache_root / f"v{SYNURANG_VERSION}" / asset.bundle_directory
    )
    executable = install_directory / GENERATOR_NAME
    install_directory.mkdir(parents=True, exist_ok=True)
    temporary = executable.with_name(f".{GENERATOR_NAME}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)

    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            candidates = [
                member
                for member in bundle.getmembers()
                if member.isfile()
                and PurePosixPath(member.name).name == GENERATOR_NAME
            ]
            if len(candidates) != 1:
                raise CodegenError(
                    f"expected one {GENERATOR_NAME} in {archive}, "
                    f"found {len(candidates)}"
                )
            source = bundle.extractfile(candidates[0])
            if source is None:
                raise CodegenError(
                    f"could not read {candidates[0].name} from {archive}"
                )
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
        temporary.chmod(
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
        os.replace(temporary, executable)
    except (OSError, tarfile.TarError) as error:
        temporary.unlink(missing_ok=True)
        raise CodegenError(f"failed to extract {archive}: {error}") from error
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
            f"(expected Synurang v{SYNURANG_VERSION})"
        )
        return executable

    target = args.target or detect_host_target()
    asset = RELEASE_ASSETS[target]
    if not args.fetch_only and target != detect_host_target():
        raise CodegenError(
            f"cannot execute {target} generator on {detect_host_target()}; "
            "use the non-host target only with --fetch-only"
        )
    archive = verified_archive(asset, args.cache_dir)
    return extract_generator(archive, asset, args.cache_dir)


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
        f"Synurang generator: v{SYNURANG_VERSION}; "
        f"proto SHA-256: {proto_sha256}"
    )
    if suffix in {".c", ".h"}:
        return f"/* {content} */\n"
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


def add_rust_imports(data: bytes) -> bytes:
    text = data.decode("utf-8")
    lines = text.splitlines(keepends=True)
    required = ("use super::pb::*;", "use std::boxed::Box;")
    missing = [statement for statement in required if statement not in text]
    if not missing:
        return data
    inner_attributes = [
        index for index, line in enumerate(lines) if line.lstrip().startswith("#![")
    ]
    if not inner_attributes:
        raise CodegenError(
            "Rust plugin output has no inner attributes; refusing to guess import placement"
        )
    insert_at = inner_attributes[-1] + 1
    block = [f"{statement}\n" for statement in missing]
    lines[insert_at:insert_at] = block
    return "".join(lines).encode("utf-8")


def generated_bytes(
    path: Path,
    proto_sha256: str,
    rust_imports: bool,
) -> bytes:
    data = add_provenance(path.read_bytes(), path.suffix, proto_sha256)
    if rust_imports:
        data = add_rust_imports(data)
    return data.rstrip(b"\n") + b"\n"


def manifest_path(root: Path, mode: str) -> Path:
    return root / f".synurang-{mode}.manifest.json"


def manifest_bytes(
    mode: str,
    root: Path,
    files: Sequence[Path],
    proto_sha256: str,
) -> bytes:
    relative_files = sorted(path.relative_to(root).as_posix() for path in files)
    manifest = {
        "generator": GENERATOR_NAME,
        "generator_version": SYNURANG_VERSION,
        "mode": mode,
        "proto_sha256": proto_sha256,
        "files": relative_files,
    }
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
    protoc = resolve_executable(args.protoc, "protoc")
    proto = args.proto.resolve()
    proto_root = (args.proto_root or proto.parent).resolve()
    if not proto.is_file():
        raise CodegenError(f"proto source does not exist: {proto}")
    proto_sha256 = sha256_file(proto)
    destinations = {
        "c": args.c_out.resolve(),
        "typescript": args.typescript_out.resolve(),
        "rust": args.rust_out.resolve(),
    }
    includes = list(args.proto_path)
    if Path("/usr/include").is_dir():
        includes.append(Path("/usr/include"))

    generated: dict[Path, bytes] = {}
    mode_files: dict[str, tuple[Path, ...]] = {}
    mode_roots: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="volvoxai-synurang-") as temporary:
        temporary_root = Path(temporary)
        for language in languages:
            spec = GENERATION_SPECS[language]
            output = temporary_root / language
            output.mkdir(parents=True)
            command = protoc_command(
                protoc=protoc,
                generator=generator,
                proto=proto,
                proto_root=proto_root,
                includes=includes,
                output=output,
                option=spec.option,
            )
            print(f"+ {shlex.join(command)}")
            result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
            if result.returncode != 0:
                raise CodegenError(
                    f"protoc failed for {language} with exit code {result.returncode}"
                )

            files = sorted(path for path in output.rglob("*") if path.is_file())
            if not files:
                raise CodegenError(f"Synurang emitted no files for {language}")
            destinations_for_mode: list[Path] = []
            for path in files:
                relative = path.relative_to(output)
                destination = destinations[spec.destination] / relative
                content = generated_bytes(path, proto_sha256, spec.rust_imports)
                previous = generated.get(destination)
                if previous is not None and previous != content:
                    raise CodegenError(
                        f"multiple modes emitted different content for {destination}"
                    )
                generated[destination] = content
                destinations_for_mode.append(destination)
            root = destinations[spec.destination]
            mode_files[language] = tuple(destinations_for_mode)
            mode_roots[language] = root
            generated[manifest_path(root, language)] = manifest_bytes(
                language,
                root,
                destinations_for_mode,
                proto_sha256,
            )
    return GenerationResult(
        files=generated,
        mode_files=mode_files,
        mode_roots=mode_roots,
    )


def atomic_write(path: Path, content: bytes) -> None:
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
        temporary.chmod(0o644)
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
    return b"Code generated by protoc-gen-synurang-ffi" in prefix


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
        candidate = (root / item).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise CodegenError(
                f"generated-file manifest path escapes its output root: {item!r}"
            ) from error
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
            if previous not in expected_sources and is_synurang_generated(previous):
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
            "Fetch the pinned Synurang code generator and generate reproducible "
            "C, TypeScript, and Rust bindings."
        )
    )
    parser.add_argument(
        "--generator",
        default=os.environ.get("PROTOC_GEN_SYNURANG_FFI"),
        help=(
            "explicit protoc-gen-synurang-ffi executable; defaults to the "
            "verified v0.7.2 prebuilt for this host"
        ),
    )
    parser.add_argument(
        "--target",
        choices=tuple(RELEASE_ASSETS),
        help="prebuilt target to fetch (non-host targets require --fetch-only)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPOSITORY_ROOT / "build" / "cache" / "synurang",
        help="download/extraction cache (default: build/cache/synurang)",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="verify and extract the generator without running protoc",
    )
    parser.add_argument(
        "--language",
        action="append",
        help=(
            "generation selection; repeat or comma-separate all, c-lite, "
            "c-native, typescript, rust (default: all)"
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
        help="C lite/native output directory",
    )
    parser.add_argument(
        "--typescript-out",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "generated" / "typescript",
        help="TypeScript output directory",
    )
    parser.add_argument(
        "--rust-out",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "src" / "gen",
        help="Rust plugin-server output directory",
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
            return 0
        generated = collect_generation(args, generator, languages)
        return 0 if install_or_check(generated, languages, args.check) else 1
    except CodegenError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
