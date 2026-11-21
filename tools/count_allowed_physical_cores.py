#!/usr/bin/env python3
"""Count distinct topology-proven physical CPU cores allowed to this process."""
from __future__ import annotations

from pathlib import Path
import sys


def parse_cpu_list(value: str) -> set[int]:
    """Expand a Linux CPU list such as '0-3,8,10-11' into CPU IDs."""
    text = value.strip()
    if not text:
        raise ValueError("CPU list must not be empty")
    cpus: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"invalid CPU-list component: {part!r}")
        bounds = token.split("-")
        if len(bounds) == 1:
            if not bounds[0].isdigit():
                raise ValueError(f"invalid CPU ID: {token!r}")
            cpus.add(int(bounds[0]))
        elif len(bounds) == 2:
            if not (bounds[0].isdigit() and bounds[1].isdigit()):
                raise ValueError(f"invalid CPU range: {token!r}")
            start, end = int(bounds[0]), int(bounds[1])
            if end < start:
                raise ValueError(f"inverted CPU range: {token!r}")
            cpus.update(range(start, end + 1))
        else:
            raise ValueError(f"invalid CPU range: {token!r}")
    return cpus


def read_allowed_cpus(status_path: Path = Path("/proc/self/status")) -> set[int] | None:
    """Return allowed CPU IDs from /proc/self/status, or None if unavailable/empty."""
    if not status_path.is_file():
        return None
    status = status_path.read_text(encoding="utf-8")
    raw = [
        line.split(":", 1)[1].strip()
        for line in status.splitlines()
        if line.startswith("Cpus_allowed_list:")
    ]
    if raw and raw[0]:
        return parse_cpu_list(raw[0])
    return None


def count_allowed_physical_cores(
    status_path: Path = Path("/proc/self/status"),
    cpu_root: Path = Path("/sys/devices/system/cpu"),
) -> int:
    """Count unique (physical_package_id, core_id) pairs among allowed CPUs."""
    allowed = read_allowed_cpus(status_path)
    if not cpu_root.is_dir():
        raise RuntimeError(f"CPU topology root not found: {cpu_root}")
    cores: set[tuple[str, str]] = set()
    for entry in cpu_root.glob("cpu[0-9]*"):
        suffix = entry.name[3:]
        if not suffix.isdigit():
            continue
        cpu_id = int(suffix)
        if allowed is not None and cpu_id not in allowed:
            continue
        pkg_path = entry / "topology" / "physical_package_id"
        core_path = entry / "topology" / "core_id"
        if not (pkg_path.is_file() and core_path.is_file()):
            continue
        pkg_id = pkg_path.read_text(encoding="utf-8").strip()
        core_id = core_path.read_text(encoding="utf-8").strip()
        if pkg_id and core_id:
            cores.add((pkg_id, core_id))
    if not cores:
        raise RuntimeError("No topology-proven allowed physical cores detected")
    return len(cores)


def main() -> int:
    try:
        print(count_allowed_physical_cores())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
