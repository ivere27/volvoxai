#!/usr/bin/env python3
"""Verify exact shared exports and physical native library profile boundaries."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_FFI_HEADER = (
    REPOSITORY_ROOT / "runtime" / "generated" / "c" / "inference"
    / "volvoxai_ffi.h"
)
FULL_FFI_HEADER = (
    REPOSITORY_ROOT / "runtime" / "generated" / "c" / "volvoxai_ffi.h"
)
INFERENCE_LITE_HEADER = (
    REPOSITORY_ROOT / "runtime" / "generated" / "c" / "inference"
    / "volvoxai_lite.h"
)
FULL_LITE_HEADER = (
    REPOSITORY_ROOT / "runtime" / "generated" / "c" / "volvoxai_lite.h"
)
BACKEND_HEADER = REPOSITORY_ROOT / "native" / "include" / "volvoxai_backend.h"

PROTOTYPE_PATTERN = re.compile(
    r"^[ \t]*(?!#)[A-Za-z_][A-Za-z0-9_ \t*]*[ \t*]"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*;",
    re.MULTILINE,
)
FFI_PREFIXES = (
    "Synurang_Invoke_",
    "Synurang_Stream_",
    "vx_platform_",
    "vx_profiling_",
    "vx_inference_",
    "vx_scheduler_",
    "vx_planning_",
    "vx_text_",
    "vx_buffer_",
    "vx_training_",
    "vx_quantization_",
)
PROFILE_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"Synurang_(?:Invoke|Stream)_Vx(?:Training|Quantization)Service"
    r"|vx_(?:training|quantization)_"
    r"|vx_model_create_(?:trainer|ptq_plan)"
    r"|vx_(?:trainer|ptq_plan)_"
    r"|volvoxai_engine_train_"
    r"|volvoxai_training_"
    r"|volvoxai_autograd_"
    r"|vx_dynamic_autograd"
    r"|vx_training_control_"
    r"|optimizer_state_for"
    r"|volvoxai_(?:engine_)?ptq_"
    r")"
)
GPU_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"volvoxai_cuda_ptx"
    r"|cuda_(?:init|cleanup)"
    r"|vk_(?:init|cleanup)"
    r"|opengl_(?:init|cleanup)"
    r"|metal_(?:init|cleanup)"
    r")"
)
IGNORED_NM_SYMBOLS = {"VOLVOXAI_1"}
FULL_ONLY_CODEC_ANCHORS = {
    "volvoxai_v1_author_ptq_template_request_init",
    "volvoxai_v1_create_trainer_request_init",
    "volvoxai_v1_ptq_plan_handle_init",
    "volvoxai_v1_train_step_request_init",
}
INTERNAL_VOCABULARY_CODEC_ANCHORS = {
    "volvoxai_v1_operator_kind_name",
    "volvoxai_v1_operator_kind_parse",
}


class CheckError(RuntimeError):
    """A deterministic library-boundary violation."""


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def declared_functions(path: Path, prefixes: tuple[str, ...]) -> set[str]:
    if not path.is_file():
        raise CheckError(f"public header does not exist: {path}")
    source = strip_comments(path.read_text(encoding="utf-8"))
    return {
        name
        for name in PROTOTYPE_PATTERN.findall(source)
        if name.startswith(prefixes)
    }


def run_nm(nm: str, arguments: Sequence[str], artifact: Path) -> str:
    if not artifact.is_file():
        raise CheckError(f"native library artifact does not exist: {artifact}")
    result = subprocess.run(
        [nm, *arguments, str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CheckError(
            f"nm failed for {artifact} with exit code {result.returncode}: {detail}"
        )
    return result.stdout


def run_tool(tool: str, arguments: Sequence[str], artifact: Path) -> str:
    if not artifact.is_file():
        raise CheckError(f"native library artifact does not exist: {artifact}")
    result = subprocess.run(
        [tool, *arguments, str(artifact)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CheckError(
            f"{tool} failed for {artifact} with exit code "
            f"{result.returncode}: {detail}"
        )
    return result.stdout


def nm_arguments(system_name: str, inventory: str) -> tuple[str, ...]:
    if system_name == "Darwin":
        arguments = {
            "shared_exports": ("-gU",),
            "all_defined": ("-U",),
            "static_globals": ("-gU",),
            "static_members": ("-A", "-gU"),
        }
    elif system_name in {
        "Linux",
        "Android",
        "FreeBSD",
        "NetBSD",
        "OpenBSD",
        "SunOS",
    }:
        arguments = {
            "shared_exports": ("-D", "--defined-only"),
            "all_defined": ("--defined-only",),
            "static_globals": ("-g", "--defined-only"),
            "static_members": ("-A", "-g", "--defined-only"),
        }
    else:
        raise CheckError(
            f"library symbol inspection is not implemented for {system_name}; "
            "Windows requires a PE export checker and profile .def files"
        )
    return arguments[inventory]


def parsed_symbols(output: str, system_name: str, member: str | None = None) -> set[str]:
    symbols: set[str] = set()
    for line in output.splitlines():
        if member is not None and member not in line:
            continue
        fields = line.split()
        if len(fields) < 2 or not re.fullmatch(r"[A-Za-z?]", fields[-2]):
            continue
        symbol = fields[-1]
        if system_name == "Darwin" and symbol.startswith("_"):
            symbol = symbol[1:]
        symbol = symbol.split("@", 1)[0]
        if symbol and symbol not in IGNORED_NM_SYMBOLS:
            symbols.add(symbol)
    return symbols


def elf_build_id(readelf: str, target: Path) -> str | None:
    try:
        notes = run_tool(readelf, ("-n",), target)
        match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", notes)
        if match:
            return match.group(1).lower()
    except Exception:
        pass
    return None


def resolve_symbol_target(
    nm: str,
    system_name: str,
    artifact: Path,
    inventory: str,
    readelf: str | None = None,
) -> Path:
    if inventory != "all_defined" or system_name == "Darwin":
        return artifact

    # If the artifact itself still contains defined symbols in .symtab, use it directly.
    try:
        probe = subprocess.run(
            [nm, "--defined-only", str(artifact)],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return artifact
    except Exception:
        pass

    # The library is stripped; resolve its companion debug sidecar in .debug/.
    candidates = [
        artifact.parent / ".debug" / f"{artifact.name}.debug",
        artifact.parent / ".debug" / f"{artifact.stem}.debug",
    ]
    if artifact.is_symlink():
        try:
            real_target = artifact.resolve()
            candidates.insert(
                0,
                real_target.parent / ".debug" / f"{real_target.name}.debug",
            )
        except Exception:
            pass

    for candidate in candidates:
        if candidate.is_file():
            if readelf:
                bin_id = elf_build_id(readelf, artifact)
                dbg_id = elf_build_id(readelf, candidate)
                if bin_id and dbg_id and bin_id != dbg_id:
                    raise CheckError(
                        f"build ID mismatch between {artifact} ({bin_id}) and {candidate} ({dbg_id})"
                    )
            return candidate

    return artifact


def symbols(
    nm: str,
    system_name: str,
    artifact: Path,
    inventory: str,
    member: str | None = None,
    readelf: str | None = None,
) -> set[str]:
    target = resolve_symbol_target(nm, system_name, artifact, inventory, readelf)
    output = run_nm(nm, nm_arguments(system_name, inventory), target)
    return parsed_symbols(output, system_name, member)


def compare_exact(label: str, actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return
    details = [f"{label} does not match its header-derived contract"]
    if missing:
        details.append("  missing: " + ", ".join(missing))
    if extra:
        details.append("  extra: " + ", ".join(extra))
    raise CheckError("\n".join(details))


def require_symbols(label: str, actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    if missing:
        raise CheckError(f"{label} is missing: {', '.join(missing)}")


def reject_profile_symbols(label: str, actual: set[str]) -> None:
    forbidden = sorted(symbol for symbol in actual if PROFILE_ONLY_PATTERN.match(symbol))
    if forbidden:
        raise CheckError(
            f"{label} physically contains full-profile symbols: "
            + ", ".join(forbidden)
        )


def reject_gpu_symbols(label: str, actual: set[str]) -> None:
    forbidden = sorted(symbol for symbol in actual if GPU_ONLY_PATTERN.match(symbol))
    if forbidden:
        raise CheckError(
            f"{label} physically contains GPU backend symbols: "
            + ", ".join(forbidden)
        )


def expected_exports(ffi_header: Path, lite_header: Path) -> tuple[set[str], set[str]]:
    ffi = declared_functions(ffi_header, FFI_PREFIXES)
    # Shared lite helpers are header-only; generated message codecs are the
    # callable symbols in the library.
    codecs = declared_functions(lite_header, ("volvoxai_v1_",))
    backend = declared_functions(BACKEND_HEADER, ("vx_backend_register_provider",))
    expected = {"Synurang_GetApi"} | codecs | backend
    required_anchors = {
        "Synurang_GetApi",
        "volvoxai_v1_create_runtime_request_init",
        "volvoxai_v1_operation_report_decode",
        "vx_backend_register_provider",
    }
    require_symbols(f"contract derived from {ffi_header}", expected, required_anchors)
    return ffi, expected


def check_elf_shared_contract(
    label: str,
    readelf: str,
    shared: Path,
    expected_soname: str,
    expected_exports: set[str],
) -> None:
    dynamic = run_tool(readelf, ("-dW",), shared)
    sonames = re.findall(r"\(SONAME\).*?\[([^\]]+)\]", dynamic)
    if sonames != [expected_soname]:
        raise CheckError(
            f"{label} SONAME is {sonames!r}; expected [{expected_soname!r}]"
        )
    dynamic_symbols = run_tool(readelf, ("--dyn-syms", "-W"), shared)
    versioned_exports: set[str] = set()
    for line in dynamic_symbols.splitlines():
        fields = line.split()
        if (
            len(fields) >= 8
            and fields[0].endswith(":")
            and fields[4] == "GLOBAL"
            and fields[6] != "UND"
        ):
            versioned_exports.add(fields[7])
    missing_versions = sorted(
        symbol
        for symbol in expected_exports
        if f"{symbol}@@VOLVOXAI_1" not in versioned_exports
    )
    if missing_versions:
        raise CheckError(
            f"{label} exports are not bound to @@VOLVOXAI_1: "
            + ", ".join(missing_versions)
        )
    sections = run_tool(readelf, ("-SW",), shared)
    if ".symtab" not in sections:
        if ".gnu_debuglink" not in sections:
            raise CheckError(
                f"{label} is stripped but missing .gnu_debuglink section"
            )


def check_profile(
    label: str,
    nm: str,
    readelf: str,
    system_name: str,
    shared: Path,
    static: Path,
    ffi_header: Path,
    lite_header: Path,
    runtime_header: Path,
    inference: bool,
) -> int:
    ffi_expected, shared_expected = expected_exports(ffi_header, lite_header)
    lite_expected = declared_functions(lite_header, ("volvoxai_v1_",))
    shared_exports = symbols(nm, system_name, shared, "shared_exports")
    compare_exact(f"{label} shared exports", shared_exports, shared_expected)
    if system_name != "Darwin":
        check_elf_shared_contract(
            label,
            readelf,
            shared,
            "libvolvoxai-lite.so.1" if inference else "libvolvoxai.so.1",
            shared_expected,
        )

    static_globals = symbols(nm, system_name, static, "static_globals")
    require_symbols(
        f"{label} static archive",
        static_globals,
        shared_expected,
    )
    ffi_member = symbols(
        nm,
        system_name,
        static,
        "static_members",
        member="volvoxai_ffi.c.o",
    )
    if not ffi_member:
        raise CheckError(f"{label} static archive has no volvoxai_ffi.c.o member")
    compare_exact(f"{label} static dispatch object", ffi_member, ffi_expected)
    lite_member = symbols(
        nm,
        system_name,
        static,
        "static_members",
        member="volvoxai_lite.c.o",
    )
    if not lite_member:
        raise CheckError(f"{label} static archive has no volvoxai_lite.c.o member")
    compare_exact(f"{label} static lite codec object", lite_member, lite_expected)

    if inference:
        reject_profile_symbols(
            "inference shared library",
            symbols(nm, system_name, shared, "all_defined", readelf=readelf),
        )
        reject_profile_symbols(
            "inference static archive",
            symbols(nm, system_name, static, "all_defined", readelf=readelf),
        )
        reject_gpu_symbols(
            "inference shared library",
            symbols(nm, system_name, shared, "all_defined", readelf=readelf),
        )
        reject_gpu_symbols(
            "inference static archive",
            symbols(nm, system_name, static, "all_defined", readelf=readelf),
        )
    return len(shared_expected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--synurang-header", type=Path, required=True)
    parser.add_argument("--inference-shared", type=Path, required=True)
    parser.add_argument("--inference-static", type=Path, required=True)
    parser.add_argument("--full-shared", type=Path, required=True)
    parser.add_argument("--full-static", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inference_codec = declared_functions(INFERENCE_LITE_HEADER, ("volvoxai_v1_",))
        full_codec = declared_functions(FULL_LITE_HEADER, ("volvoxai_v1_",))
        require_symbols("full lite codec", full_codec, FULL_ONLY_CODEC_ANCHORS)
        leaked_internal = sorted(
            INTERNAL_VOCABULARY_CODEC_ANCHORS & (inference_codec | full_codec)
        )
        if leaked_internal:
            raise CheckError(
                "public lite codecs expose the internal operator vocabulary: "
                + ", ".join(leaked_internal)
            )
        leaked_codec = sorted(FULL_ONLY_CODEC_ANCHORS & inference_codec)
        if leaked_codec:
            raise CheckError(
                "inference lite codec exposes full-profile symbols: "
                + ", ".join(leaked_codec)
            )
        inference_count = check_profile(
            "inference",
            args.nm,
            args.readelf,
            args.system_name,
            args.inference_shared.resolve(),
            args.inference_static.resolve(),
            INFERENCE_FFI_HEADER,
            INFERENCE_LITE_HEADER,
            args.synurang_header.resolve(),
            True,
        )
        full_count = check_profile(
            "full",
            args.nm,
            args.readelf,
            args.system_name,
            args.full_shared.resolve(),
            args.full_static.resolve(),
            FULL_FFI_HEADER,
            FULL_LITE_HEADER,
            args.synurang_header.resolve(),
            False,
        )
        print(
            "Verified exact native library profile boundaries "
            f"(inference={inference_count}, full={full_count} exports)."
        )
        return 0
    except (CheckError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
