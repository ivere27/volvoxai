#!/usr/bin/env python3
"""Build and qualify a Linux wheel inside python/Dockerfile.wheel."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSIONS = ("310", "311", "312", "313", "314")


@contextmanager
def working_directory(path):
    if path is None:
        with tempfile.TemporaryDirectory(prefix="volvoxai-wheel-") as temporary:
            yield Path(temporary)
    else:
        path.mkdir(parents=True, exist_ok=True)
        yield path.resolve()


def run(command, *, cwd, env=None):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def git(*arguments):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT}", *arguments], cwd=ROOT,
    ).decode().strip()


def copy_sources(destination):
    # Include working changes, including new packaging files, without copying
    # ignored models/builds or unrelated untracked notes into the build context.
    names = set(git("ls-files", "-z").split("\0"))
    names.update(git("ls-files", "-z", "--others", "--exclude-standard", "--",
                     "python", "native/src", "native/tests").split("\0"))
    names = {name for name in names if name and (ROOT / name).is_file()}
    manifest = destination.parent / "source-files.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else []
    for name in set(previous) - names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid cached source path")
        (destination / relative).unlink(missing_ok=True)
    digest = hashlib.sha256()
    for name in sorted(names):
        source = ROOT / name
        if not name or not source.is_file():
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid source path: {name}")
        content = source.read_bytes()
        digest.update(name.encode() + b"\0" + hashlib.sha256(content).digest())
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest.write_text(json.dumps(sorted(names)) + "\n")
    return digest.hexdigest()


def stage_package(source, stage, provenance):
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    for name in ("pyproject.toml", "setup.py", "README.md"):
        shutil.copy2(source / "python" / name, stage / name)
    (stage / "VERSION").write_text(provenance["version"] + "\n")
    shutil.copy2(source / "LICENSE", stage / "LICENSE")
    generated = source / "runtime/generated/python"
    shutil.copy2(generated / "SYNURANG-LICENSE", stage / "SYNURANG-LICENSE")
    shutil.copytree(source / "python/third_party", stage / "third_party")
    shutil.copy2(source / "python/third_party/dlpack/LICENSE", stage / "DLPACK-LICENSE")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.so")
    shutil.copytree(source / "python/volvoxai", stage / "volvoxai", ignore=ignore)
    shutil.copytree(source / "tools/exporter", stage / "volvoxai/exporter", ignore=ignore)
    shutil.copy2(source / "tools/export_safetensors.py", stage / "volvoxai/_export.py")
    shutil.copytree(generated / "synurang", stage / "synurang", ignore=ignore)
    for name in ("volvoxai_lite.py", "volvoxai_client.py"):
        shutil.copy2(generated / name, stage / name)
    (stage / "volvoxai/schema").mkdir()
    shutil.copy2(source / "proto/volvoxai.proto", stage / "volvoxai/schema/volvoxai.proto")
    # Python ships the full engine only; lite stays a native/browser release profile.
    for name in ("libvolvoxai.so", "libsynurang_module_host.so"):
        target = stage / "volvoxai" / name
        shutil.copy2(source / "native" / name, target)
        run(["strip", "--strip-unneeded", target], cwd=stage)
    (stage / "volvoxai/bin").mkdir()
    shutil.copy2(source / "native/volvoxai", stage / "volvoxai/bin/volvoxai")
    (stage / "volvoxai/BUILD-INFO.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


def qualify(wheel, source, workspace, environment):
    # No checkout paths, library overrides, or system site packages are admitted.
    clean = dict(environment)
    for name in tuple(clean):
        if name.startswith(("VOLVOXAI_", "SYNURANG_")) or name in ("PYTHONPATH", "PYTHONHOME"):
            clean.pop(name)
    clean["PIP_CACHE_DIR"] = str(workspace / "pip-cache")
    test = workspace / "test_installed_wheel.py"
    shutil.copy2(source / "python/tests/test_installed_wheel.py", test)
    error_test = workspace / "test_client_errors.py"
    shutil.copy2(source / "python/tests/test_client_errors.py", error_test)
    session_test = workspace / "test_inference_session.py"
    shutil.copy2(source / "python/tests/test_inference_session.py", session_test)
    workflows_test = workspace / "test_python_workflows.py"
    shutil.copy2(source / "python/tests/test_python_workflows.py", workflows_test)
    tensor_test = workspace / "test_tensor_interop.py"
    shutil.copy2(source / "python/tests/test_tensor_interop.py", tensor_test)
    qualification = workspace / "qualification"
    (qualification / "python/tests").mkdir(parents=True, exist_ok=True)
    (qualification / "tools").mkdir(exist_ok=True)
    training_test = qualification / "python/tests/test_training_end_to_end.py"
    shutil.copy2(source / "python/tests/test_training_end_to_end.py", training_test)
    shutil.copy2(source / "tools/test_native_gpu_training_smoke.py",
                 qualification / "tools/test_native_gpu_training_smoke.py")
    tested = []
    for version in PYTHON_VERSIONS:
        interpreter = Path(f"/opt/python/cp{version}-cp{version}/bin/python")
        if not interpreter.is_file():
            raise RuntimeError(f"missing qualification interpreter: {interpreter}")
        venv = workspace / f"venv-{version}"
        if venv.exists():
            shutil.rmtree(venv)
        run([interpreter, "-m", "venv", venv], cwd=workspace, env=clean)
        python = venv / "bin/python"
        run([python, "-m", "pip", "install", "--only-binary=:all:", wheel],
            cwd=workspace, env=clean)
        run([python, "-m", "pip", "check"], cwd=workspace, env=clean)
        run([python, "-I", test], cwd=workspace, env=clean)
        run([python, "-I", error_test], cwd=workspace, env=clean)
        run([python, "-I", session_test], cwd=workspace, env=clean)
        run([python, "-I", workflows_test], cwd=workspace, env=clean)
        run([python, "-I", tensor_test], cwd=workspace, env=clean)
        run([python, "-I", training_test], cwd=qualification, env=clean)
        tested.append(f"3.{version[1:]}")
    return tested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Parent output directory; a package-version directory is added")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--work-dir", type=Path,
                        help="Private reusable build directory; installation tests remain clean")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("use the Linux x86_64 wheel Docker image")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    version = json.loads((ROOT / "package.json").read_text())["version"]
    output = args.out.resolve() / version
    output.mkdir(parents=True, exist_ok=True)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", git("log", "-1", "--format=%ct")))
    environment = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch))
    with working_directory(args.work_dir) as workspace:
        source = workspace / "source"
        image_id = os.environ.get("VOLVOXAI_WHEEL_IMAGE_ID", "")
        # BuildKit can change an attestation/index ID on a fully cached build.
        # Cache by filesystem layers and execution config, retaining the exact
        # image ID separately as provenance.
        image_key = os.environ.get("VOLVOXAI_WHEEL_IMAGE_KEY", image_id)
        if args.work_dir is not None and not image_id:
            parser.error("a reusable --work-dir requires VOLVOXAI_WHEEL_IMAGE_ID; use make build_wheel")
        image_record = workspace / "builder-image.txt"
        if (source.exists() and (not image_record.is_file()
                or image_record.read_text().strip() not in (image_key, image_id))):
            shutil.rmtree(source)
        image_record.write_text(image_key + "\n")
        source.mkdir(exist_ok=True)
        provenance = {
            "version": version,
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "source_sha256": copy_sources(source),
            "source_date_epoch": epoch,
            "platform": "manylinux_2_28_x86_64",
            "backends": ["cpu", "cuda", "vulkan", "opengl"],
            "profiles": ["full"],
            "synurang_version": "0.8.0",
            "builder_image": image_id,
            "builder_fingerprint": image_key,
        }
        build = source / "build/wheel-native"
        # Keep libc data references in the GOT. Copy relocations would expose
        # stderr as a defined executable export with newer Clang toolchains.
        flags = (
            f"-ffile-prefix-map={source}=/volvoxai -fdebug-prefix-map={source}=/volvoxai "
            "-fno-direct-access-external-data"
        )
        run([
            "cmake", "-S", source, "-B", build,
            "-DCMAKE_C_COMPILER=clang", f"-DPython3_EXECUTABLE={sys.executable}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", f"-DCMAKE_C_FLAGS={flags}",
            "-DVOLVOXAI_CPU_TARGET=baseline",
            "-DVOLVOXAI_ENABLE_VULKAN=ON", "-DVOLVOXAI_ENABLE_OPENGL=ON",
            "-DVOLVOXAI_ENABLE_CUDA=ON", "-DVOLVOXAI_ENABLE_METAL=OFF",
            "-DVOLVOXAI_CUDA_ARCH=75",
            "-DVOLVOXAI_BUILD_DATE=" + datetime.fromtimestamp(epoch, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        ], cwd=source, env=environment)
        run(["cmake", "--build", build, "--parallel", args.jobs, "--target",
             "test_shared_library_abi", "volvoxai", "volvoxai-lite"],
            cwd=source, env=environment)
        run(["ctest", "--test-dir", build, "--output-on-failure", "-R",
             "^(test_shared_library_abi(_full)?|check_native_library_profile_boundaries)$"],
            cwd=source, env=environment)
        run([source / "native/cmake/check_profile_boundaries.sh",
             source / "native/volvoxai-lite", source / "native/volvoxai"],
            cwd=source, env=environment)
        run([sys.executable, source / "tools/check_native_cuda_ptx.py", "--build-dir", build],
            cwd=source, env=environment)
        stage = workspace / "package"
        stage_package(source, stage, provenance)
        raw = workspace / "raw"
        if raw.exists():
            shutil.rmtree(raw)
        run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", raw],
            cwd=stage, env=environment)
        built, = raw.glob("*.whl")
        repaired = workspace / "repaired"
        if repaired.exists():
            shutil.rmtree(repaired)
        run(["auditwheel", "show", built], cwd=workspace, env=environment)
        run(["auditwheel", "repair", "--plat", "manylinux_2_28_x86_64",
             "--only-plat", "--wheel-dir", repaired, built], cwd=workspace, env=environment)
        wheel, = repaired.glob("*.whl")
        run([sys.executable, "-m", "twine", "check", "--strict", wheel],
            cwd=workspace, env=environment)
        tested = qualify(wheel, source, workspace, environment)
        report = dict(provenance, python_versions_tested=tested,
                      qualification_backend="cpu",
                      qualification_tests=["test_tensor_interop.py", "test_installed_wheel.py", "test_client_errors.py",
                                           "test_inference_session.py", "test_python_workflows.py",
                                           "test_training_end_to_end.py"],
                      wheel=wheel.name, bytes=wheel.stat().st_size,
                      sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
        destination = output / wheel.name
        shutil.copy2(wheel, destination.with_suffix(".whl.tmp"))
        destination.with_suffix(".whl.tmp").replace(destination)
        # ABI tag changes replace the previous candidate of this same version,
        # so a wildcard PyPI upload cannot publish an obsolete API alongside it.
        for previous in output.glob(f"volvoxai-{version}-*.whl"):
            if previous != destination:
                previous.unlink()
        (output / "SHA256SUMS").write_text(f"{report['sha256']}  {wheel.name}\n")
        (output / "wheel-validation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
