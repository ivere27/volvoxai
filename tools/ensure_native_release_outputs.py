#!/usr/bin/env python3
"""Force a native release relink when its recoverable outputs are stale.

CMake's executable is the primary output of a link rule. Linker maps and
POST_BUILD split-debug files are byproducts, so Unix Makefiles do not relink
merely because one of those files was deleted. This guard runs before the
executable target and touches an explicit LINK_DEPENDS stamp whenever the
build-private bundle, map, or published bundle is incomplete or inconsistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


FORMAT = "volvoxai-native-release-finalization/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"empty file: {path}")
    return size, sha256(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--published-artifact", type=Path, required=True)
    parser.add_argument("--stamp", type=Path, required=True)
    parser.add_argument("--build-directory")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--debug", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--published-debug", type=Path)
    parser.add_argument("--published-evidence", type=Path)
    return parser.parse_args()


def validate_linux_bundle(options: argparse.Namespace) -> None:
    required = (
        options.build_directory,
        options.repository_root,
        options.map,
        options.debug,
        options.evidence,
        options.published_debug,
        options.published_evidence,
    )
    if any(value is None for value in required):
        raise ValueError("Linux bundle guard arguments are incomplete")

    repository_root = options.repository_root.resolve()
    selected_build_directory = (repository_root / options.build_directory).resolve()

    artifact_size, artifact_sha = fingerprint(options.artifact)
    debug_size, debug_sha = fingerprint(options.debug)
    map_size, map_sha = fingerprint(options.map)
    evidence_bytes = options.evidence.read_bytes()
    if not evidence_bytes:
        raise ValueError(f"empty file: {options.evidence}")
    try:
        evidence = json.loads(evidence_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evidence JSON: {options.evidence}: {error}") from error

    if evidence.get("format") != FORMAT:
        raise ValueError(f"unexpected evidence format: {options.evidence}")
    build = evidence.get("build")
    if not isinstance(build, dict) or build.get("directory") != options.build_directory:
        raise ValueError("evidence belongs to a different CMake build directory")
    expected_artifact = build.get("artifact")
    if not isinstance(expected_artifact, dict):
        raise ValueError("evidence has no build artifact fingerprint")
    if (expected_artifact.get("rawBytes"), expected_artifact.get("sha256")) != (
        artifact_size,
        artifact_sha,
    ):
        raise ValueError("build-private release artifact differs from evidence")
    expected_map = build.get("linkMap")
    if not isinstance(expected_map, dict):
        raise ValueError("evidence has no linker-map fingerprint")
    if (expected_map.get("rawBytes"), expected_map.get("sha256")) != (
        map_size,
        map_sha,
    ):
        raise ValueError("native release linker map differs from evidence")

    link = build.get("link")
    link_objects = link.get("objects") if isinstance(link, dict) else None
    if not isinstance(link_objects, list) or not link_objects:
        raise ValueError("evidence has no ordered native link inputs")
    seen: set[Path] = set()
    for index, expected in enumerate(link_objects):
        if not isinstance(expected, dict) or not isinstance(expected.get("path"), str):
            raise ValueError(f"invalid native link input evidence at index {index}")
        object_path = (repository_root / expected["path"]).resolve()
        try:
            object_path.relative_to(selected_build_directory)
        except ValueError as error:
            raise ValueError(
                f"native link input {index} is outside the selected build tree"
            ) from error
        if object_path in seen:
            raise ValueError(f"native link input {index} is duplicated")
        seen.add(object_path)
        object_size, object_sha = fingerprint(object_path)
        if (expected.get("rawBytes"), expected.get("sha256")) != (
            object_size,
            object_sha,
        ):
            raise ValueError(f"native link input {index} differs from evidence")

    release = evidence.get("release")
    debug = evidence.get("debug")
    if not isinstance(release, dict) or (
        release.get("rawBytes"), release.get("sha256")
    ) != (artifact_size, artifact_sha):
        raise ValueError("release fingerprint differs from finalization evidence")
    if not isinstance(debug, dict) or (
        debug.get("rawBytes"), debug.get("sha256")
    ) != (debug_size, debug_sha):
        raise ValueError("debug fingerprint differs from finalization evidence")

    published_size, published_sha = fingerprint(options.published_artifact)
    if (published_size, published_sha) != (artifact_size, artifact_sha):
        raise ValueError("published release differs from this build tree")
    published_debug_size, published_debug_sha = fingerprint(options.published_debug)
    if (published_debug_size, published_debug_sha) != (debug_size, debug_sha):
        raise ValueError("published debug sidecar differs from this build tree")
    if options.published_evidence.read_bytes() != evidence_bytes:
        raise ValueError("published evidence differs from this build tree")


def validate_plain_artifact(options: argparse.Namespace) -> None:
    artifact = fingerprint(options.artifact)
    published = fingerprint(options.published_artifact)
    if artifact != published:
        raise ValueError("published release differs from this build tree")


def main() -> None:
    options = arguments()
    linux_bundle = any(
        value is not None
        for value in (
            options.map,
            options.debug,
            options.evidence,
            options.published_debug,
            options.published_evidence,
        )
    )
    try:
        if linux_bundle:
            validate_linux_bundle(options)
        else:
            validate_plain_artifact(options)
        return
    except (OSError, ValueError) as error:
        reason = str(error)

    options.stamp.parent.mkdir(parents=True, exist_ok=True)
    options.stamp.touch(exist_ok=True)
    # LINK_DEPENDS must be newer than an existing target even when a build and
    # recovery request happen within one filesystem timestamp tick.
    artifact_mtime = (
        options.artifact.stat().st_mtime_ns if options.artifact.exists() else 0
    )
    now = time.time_ns()
    os.utime(options.stamp, ns=(now, max(now, artifact_mtime + 1)))
    print(f"Native release outputs require relink: {reason}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"native release output guard failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
