"""Find and load a source-built libvolvoxai.

The fixed release does not include native libraries. A repository build can
produce two shared-library profiles whose application surface is generated
from ``proto/volvoxai.proto``:

===================  =========================================================
``libvolvoxai``      inference — Platform, Text, Planning, Inference, Scheduler
``libvolvoxai-full`` full — the above plus Training and Quantization
===================  =========================================================

The inference profile physically omits training and quantization handlers and
symbols. Select the full profile before constructing those generated clients.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Iterator
from pathlib import Path

from synurang import ModuleHost

from .errors import VolvoxAIError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

PROFILES = ("inference", "full")


def library_filename(profile: str = "inference") -> str:
    """The platform's file name for one profile's library."""

    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    stem = "volvoxai" if profile == "inference" else "volvoxai-full"
    system = platform.system()
    if system == "Windows":
        return f"{stem}.dll"
    if system == "Darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def _candidates(profile: str) -> Iterator[Path]:
    """Where to look, nearest first."""

    filename = library_filename(profile)
    override = os.environ.get("VOLVOXAI_LIBRARY")
    if override:
        # An explicit path is a decision, not a hint: use it for either
        # profile and let the open fail loudly if it is wrong.
        yield Path(override)
    override_dir = os.environ.get("VOLVOXAI_LIBRARY_DIR")
    if override_dir:
        yield Path(override_dir) / filename
    # Installed alongside this package, then the in-repository build.
    yield Path(__file__).resolve().parent / filename
    yield REPOSITORY_ROOT / "native" / filename


def find_library(profile: str = "inference") -> Path:
    """Locate one profile's library, or say where it was looked for."""

    searched: list[Path] = []
    for candidate in _candidates(profile):
        if candidate.is_file():
            return candidate
        searched.append(candidate)
    locations = "\n  ".join(str(path) for path in searched)
    raise VolvoxAIError(
        f"{library_filename(profile)} not found. Build the source libraries with\n"
        f"  make build_native_libraries\n"
        f"or set VOLVOXAI_LIBRARY. Looked in:\n  {locations}"
    )


def _loader_filename() -> str:
    system = platform.system()
    if system == "Windows":
        return "synurang_module_host.dll"
    if system == "Darwin":
        return "libsynurang_module_host.dylib"
    return "libsynurang_module_host.so"


def open_library(
    profile: str = "inference",
    path: str | os.PathLike[str] | None = None,
    *,
    loader: str | os.PathLike[str] | None = None,
) -> ModuleHost:
    """Create one isolated module instance; close it to retire its resources.

    Generated clients receive this host directly. The source-library build also
    provides Synurang's loader beside the engine libraries. ``loader`` or
    ``SYNURANG_MODULE_HOST_LIBRARY`` selects another explicit loader location.
    """

    resolved = Path(path) if path is not None else find_library(profile)
    selected_loader = loader or os.environ.get("SYNURANG_MODULE_HOST_LIBRARY")
    if selected_loader is None:
        candidates = (resolved.parent / _loader_filename(),
                      REPOSITORY_ROOT / "native" / _loader_filename())
        selected_loader = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected_loader is None:
            raise VolvoxAIError(
                f"{_loader_filename()} not found. Build the module and host libraries with\n"
                "  make build_native_libraries\n"
                "or pass loader= to open_library."
            )
    return ModuleHost.load(resolved, loader=selected_loader)
