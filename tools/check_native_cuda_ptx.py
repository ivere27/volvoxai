#!/usr/bin/env python3
"""Check that a real CUDA build exports exactly the registered PTX kernels."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORWARD_REGISTRY = (
    REPOSITORY_ROOT
    / "native/src/backends/cuda/host/cuda_forward_function_registry_host.inc"
)
TRAINING_REGISTRY = (
    REPOSITORY_ROOT
    / "native/src/backends/cuda/host/cuda_training_function_registry_host.inc"
)

FORWARD_PATTERN = re.compile(
    r"VOLVOXAI_CUDA_FORWARD_FUNCTION\(\s*REQUIRED\s*,\s*[^,]+\s*,\s*"
    r'"([^"]+)"\s*\)',
    re.DOTALL,
)
TRAINING_PATTERN = re.compile(
    r"VOLVOXAI_CUDA_TRAINING_FUNCTION\(\s*REQUIRED\s*,\s*[^,]+\s*,\s*"
    r'[^,]+\s*,\s*"([^"]+)"',
    re.DOTALL,
)
PTX_ENTRY_PATTERN = re.compile(
    r"^\s*(?:\.visible\s+)?\.entry\s+([^\s(]+)\s*\(", re.MULTILINE
)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error


def _unique_symbols(symbols: list[str], source: Path) -> set[str]:
    if not symbols:
        raise RuntimeError(f"no required CUDA symbols found in {source}")
    unique = set(symbols)
    if len(unique) != len(symbols):
        duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
        raise RuntimeError(f"duplicate CUDA symbols in {source}: {', '.join(duplicates)}")
    return unique


def _registry_symbols(path: Path, pattern: re.Pattern[str]) -> set[str]:
    return _unique_symbols(pattern.findall(_read(path)), path)


def _ptx_symbols(path: Path) -> set[str]:
    return _unique_symbols(PTX_ENTRY_PATTERN.findall(_read(path)), path)


def _cache_bool(cache: str, name: str) -> bool:
    match = re.search(rf"^{re.escape(name)}:BOOL=(ON|OFF)$", cache, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{name} is missing from CMakeCache.txt")
    return match.group(1) == "ON"


def _assert_exact(label: str, expected: set[str], actual: set[str]) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if not missing and not unexpected:
        return
    details: list[str] = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unexpected:
        details.append("unregistered: " + ", ".join(unexpected))
    raise RuntimeError(f"{label} PTX inventory mismatch ({'; '.join(details)})")


def check(build_dir: Path) -> None:
    cache_path = build_dir / "CMakeCache.txt"
    cache = _read(cache_path)
    if not _cache_bool(cache, "VOLVOXAI_ENABLE_CUDA"):
        raise RuntimeError(f"CUDA is disabled in {cache_path}")

    contract = (
        "fast-fma"
        if _cache_bool(cache, "VOLVOXAI_CUDA_FAST_FP32")
        else "strict-no-fma"
    )
    generated = build_dir / "gen" / "cuda" / contract
    forward_ptx = generated / "cuda_kernels.ptx"
    training_ptx = generated / "cuda_training_kernels.ptx"

    registered_forward = _registry_symbols(FORWARD_REGISTRY, FORWARD_PATTERN)
    registered_training = _registry_symbols(TRAINING_REGISTRY, TRAINING_PATTERN)
    if overlap := registered_forward & registered_training:
        raise RuntimeError(
            "forward and training registries overlap: " + ", ".join(sorted(overlap))
        )

    actual_forward = _ptx_symbols(forward_ptx)
    actual_training = _ptx_symbols(training_ptx)
    _assert_exact("forward", registered_forward, actual_forward)
    _assert_exact("training", registered_training, actual_training)
    if leaked := actual_forward & registered_training:
        raise RuntimeError(
            "training entries leaked into inference PTX: " + ", ".join(sorted(leaked))
        )

    print(
        "CUDA PTX inventory OK: "
        f"{len(actual_forward)} forward, {len(actual_training)} training "
        f"({contract})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="configured CMake build directory containing generated PTX",
    )
    arguments = parser.parse_args()
    try:
        check(arguments.build_dir.resolve())
    except RuntimeError as error:
        print(f"CUDA PTX inventory failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
