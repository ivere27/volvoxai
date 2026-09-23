"""Find and load a bundled or source-built libvolvoxai.

Python ships one library. ``libvolvoxai`` carries the complete application
surface generated from ``proto/volvoxai.proto``, including profiling and
memory snapshots.

The repository also builds a smaller inference-only ``libvolvoxai-lite`` for
the native and browser releases. It is not installed by the Python package and
is reachable only through an explicit ``VOLVOXAI_LIBRARY`` or ``path``; the
generated Training and Quantization clients do not work against it.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Iterator
from pathlib import Path

from synurang import ModuleHost

from .errors import VolvoxAIError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def library_filename() -> str:
    """The platform's file name for the engine library."""

    system = platform.system()
    if system == "Windows":
        return "volvoxai.dll"
    if system == "Darwin":
        return "libvolvoxai.dylib"
    return "libvolvoxai.so"


def _directories() -> Iterator[Path]:
    """Where to look, nearest first."""

    override_dir = os.environ.get("VOLVOXAI_LIBRARY_DIR")
    if override_dir:
        yield Path(override_dir).expanduser()
    # Installed alongside this package, then the in-repository build.
    yield Path(__file__).resolve().parent
    yield REPOSITORY_ROOT / "native"


def find_library() -> Path:
    """Find the installed library, then the in-repository build.

    VOLVOXAI_LIBRARY is an authoritative file override: a missing file is an
    error, not a fallback.
    """

    filename = library_filename()
    override = os.environ.get("VOLVOXAI_LIBRARY")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise VolvoxAIError(f"VOLVOXAI_LIBRARY does not name a file: {candidate}")
        return candidate
    searched: list[Path] = []
    for directory in _directories():
        candidate = directory / filename
        if candidate.is_file():
            return candidate
        searched.append(candidate)
    locations = "\n  ".join(str(path) for path in searched)
    raise VolvoxAIError(
        f"{filename} not found. Build the source libraries with\n"
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
    path: str | os.PathLike[str] | None = None,
    *,
    loader: str | os.PathLike[str] | None = None,
) -> ModuleHost:
    """Create one isolated module instance; close it to retire its resources.

    Generated clients receive this host directly. The source-library build also
    provides Synurang's loader beside the engine libraries. ``loader`` or
    ``SYNURANG_MODULE_HOST_LIBRARY`` selects another explicit loader location.
    """

    resolved = Path(path).expanduser() if path is not None else find_library()
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
