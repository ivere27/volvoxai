#!/usr/bin/env python3
"""Build an isolated, pinned Deno with the Vulkan device-cache fixes."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCES = [
    ("deno-2.9.6", "deno-v2.9.6.tar.gz",
     "https://codeload.github.com/denoland/deno/tar.gz/refs/tags/v2.9.6",
     "d27f0ec13979c5dc76c7a0f20d5a389041746c005025102c1e220fa034674bbb"),
    ("wgpu-core-29.0.1", "wgpu-core-29.0.1.crate",
     "https://static.crates.io/crates/wgpu-core/wgpu-core-29.0.1.crate",
     "1e80ac6cf1895df6342f87d975162108f9d98772a0d74bc404ab7304ac29469e"),
    ("wgpu-hal-29.0.3", "wgpu-hal-29.0.3.crate",
     "https://static.crates.io/crates/wgpu-hal/wgpu-hal-29.0.3.crate",
     "31f8e1a9e7a8512f276f7c62e018c7fa8d60954303fed2e5750114332049193f"),
    ("gpu-allocator-0.28.0", "gpu-allocator-0.28.0.crate",
     "https://static.crates.io/crates/gpu-allocator/gpu-allocator-0.28.0.crate",
     "51255ea7cfaadb6c5f1528d43e92a82acb2b96c43365989a28b2d44ee38f8795"),
]
PATCHES = [HERE / "wgpu-memory-budget.patch", HERE / "wgpu-device-cache.patch"]
WORKSPACE_CONFIG = '''
[patch.crates-io]
wgpu-core = { path = "../wgpu-core-29.0.1" }
wgpu-hal = { path = "../wgpu-hal-29.0.3" }
gpu-allocator = { path = "../gpu-allocator-0.28.0" }

[profile.webgpu-fix]
inherits = "release"
opt-level = 1
codegen-units = 128
lto = false
debug = false
strip = "debuginfo"
'''


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def fetch(cache: Path, filename: str, url: str, expected: str, offline: bool) -> Path:
    archive = cache / filename
    if not archive.exists():
        if offline:
            raise RuntimeError(f"Missing verified source archive: {archive}")
        partial = archive.with_suffix(archive.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if digest(partial) != expected:
            raise RuntimeError(f"Source digest mismatch: {url}")
        partial.rename(archive)
    if digest(archive) != expected:
        raise RuntimeError(f"Cached source digest mismatch: {archive}")
    return archive


def prepare(cache: Path, offline: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=cache) as temporary:
        source = Path(temporary)
        for _, filename, url, expected in SOURCES:
            archive = fetch(cache, filename, url, expected, offline)
            with tarfile.open(archive) as packed:
                packed.extractall(source, filter="data")
        for patch in PATCHES:
            subprocess.run(["patch", "-p1", "--batch", "--fuzz=0", "-i", str(patch)],
                           cwd=source, check=True)
        deno = source / SOURCES[0][0]
        manifest = deno / "Cargo.toml"
        manifest.write_text(manifest.read_text() + WORKSPACE_CONFIG)

        # Only the three pinned registry sources become local patched sources.
        # Keep every version and dependency in the upstream lock file unchanged.
        lock = deno / "Cargo.lock"
        lock_text = lock.read_text()
        for name in ("wgpu-core", "wgpu-hal", "gpu-allocator"):
            pattern = rf'(?ms)^\[\[package\]\]\nname = "{name}"\n.*?(?=^\[\[package\]\]|\Z)'
            def local_source(match: re.Match) -> str:
                return re.sub(r'(?m)^(source|checksum) = .*\n', '', match.group())
            lock_text, count = re.subn(pattern, local_source, lock_text)
            if count != 1:
                raise RuntimeError(f"Expected one locked version of {name}")
        lock.write_text(lock_text)

        # Verify an existing source tree instead of overwriting local edits.
        # Build outputs go in target/, outside these trees.
        source_hashes = {}
        for directory, *_ in SOURCES:
            prepared = source / directory
            destination = cache / directory
            existing = destination.exists()
            for path in sorted(prepared.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source)
                expected = digest(path)
                if existing and (not (cache / relative).is_file()
                                 or digest(cache / relative) != expected):
                    raise RuntimeError(f"Source differs from pinned recipe: {cache / relative}. "
                                       "Use a new --build-dir for a changed recipe.")
                source_hashes[str(relative)] = expected
            if not existing:
                shutil.move(str(prepared), destination)
        return {
            "sources": {name: {"url": url, "sha256": sha} for name, _, url, sha in SOURCES},
            "patches": {patch.name: digest(patch) for patch in PATCHES},
            "prepared_sources_sha256": hashlib.sha256(
                json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
            "cargo_lock_sha256": digest(cache / SOURCES[0][0] / "Cargo.lock"),
            "recipe_sha256": digest(Path(__file__)),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/deno")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cache = args.build_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = prepare(cache, args.offline)
        (cache / "sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if args.prepare_only:
            print(f"Verified patched source: {cache}")
            return
        env = os.environ.copy()
        env.setdefault("RUSTUP_TOOLCHAIN", "stable")
        env["CARGO_HOME"] = str(cache / "cargo-home")
        env["CARGO_TARGET_DIR"] = str(cache / "target")
        rustc = subprocess.check_output(["rustc", "--version"], env=env, text=True).strip()
        if not rustc.startswith("rustc 1.97.0 "):
            raise RuntimeError("This runner is qualified with Rust 1.97.0; select it with "
                               f"RUSTUP_TOOLCHAIN (found {rustc}).")
        command = ["cargo", "build", "--locked", "-p", "deno", "--bin", "deno",
                   "--profile", "webgpu-fix", "-j", str(args.jobs)]
        if args.offline:
            command.append("--offline")
        subprocess.run(command, cwd=cache / SOURCES[0][0], env=env, check=True)
        executable = cache / "target/webgpu-fix/deno"
        receipt.update({
            "rustc": rustc,
            "cargo": subprocess.check_output(["cargo", "--version"], env=env, text=True).strip(),
            "command": command,
            "binary_sha256": digest(executable),
            "binary_bytes": executable.stat().st_size,
            "version": subprocess.check_output([str(executable), "--version"], text=True).strip(),
        })
        (cache / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(executable)


if __name__ == "__main__":
    main()
