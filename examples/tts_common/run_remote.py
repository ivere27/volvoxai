#!/usr/bin/env python3
"""Transfer prepared examples to an authorized SSH target and collect benchmarks.

Uses existing Python dependencies on the target. Every transfer, cache and
report lives in a unique temporary directory; finally collects and removes it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = "reports/.collection-manifest.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def error_summary(error):
    if isinstance(error, subprocess.TimeoutExpired):
        return f"operation exceeded {error.timeout} seconds"
    if isinstance(error, subprocess.CalledProcessError):
        return f"remote operation exited {error.returncode}"
    return f"{type(error).__name__}: {error}"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


REMOTE_COMMON = '''
import hashlib,json,os,pathlib,shutil,signal,time
def verified_task():
 if not p.exists(): return None
 assert p.parent==pathlib.Path('/tmp') and p.name=='volvox-tts-examples-'+token
 assert not p.is_symlink() and p.resolve()==p and p.stat().st_uid==os.getuid()
 assert (p/'.owner-token').is_file() and not (p/'.owner-token').is_symlink()
 assert (p/'.owner-token').read_text()==token
 return p
def identity(pid):
 try:
  entry=pathlib.Path('/proc')/str(pid)
  if entry.stat().st_uid!=os.getuid(): return None
  cwd=(entry/'cwd').resolve(strict=True)
  if cwd!=p and p not in cwd.parents: return None
  fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
  if fields[0]=='Z': return None
  return (pid,fields[19])
 except (OSError,RuntimeError,IndexError): return None
def active():
 return [value for entry in pathlib.Path('/proc').iterdir()
         if entry.name.isdigit() and int(entry.name)!=os.getpid()
         and (value:=identity(int(entry.name))) is not None]
def send(value,signum):
 if identity(value[0])!=value: return False
 try: os.kill(value[0],signum); return True
 except ProcessLookupError: return False
def file_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def report_files():
 result=[]
 for item in sorted((p/'reports').rglob('*')):
  assert not item.is_symlink(), 'report symlink refused'
  if item.is_dir(): continue
  assert item.is_file(), 'non-regular report refused'
  name=item.relative_to(p).as_posix()
  if name=='reports/.collection-manifest.json': continue
  result.append({'path':name,'bytes':item.stat().st_size,'sha256':file_hash(item)})
 return result
'''


def remote_script(owned, token, body, operation):
    if not re.fullmatch(r"[0-9a-f]{32}", token) or owned != f"/tmp/volvox-tts-examples-{token}":
        raise ValueError("invalid task ownership identity")
    return f"# tts-controller:{operation}\n" + REMOTE_COMMON + f"\np=pathlib.Path({owned!r})\ntoken={token!r}\n" + body


def stop_code(owned, token):
    return remote_script(owned, token, '''
if verified_task() is None:
 print(json.dumps({'task_absent':True,'survivors':[]}))
else:
 terminated=[]; killed=[]
 for value in active():
  if send(value,signal.SIGTERM): terminated.append(value[0])
 deadline=time.monotonic()+5
 while active() and time.monotonic()<deadline: time.sleep(.1)
 deadline=time.monotonic()+5
 while True:
  remaining=active()
  if not remaining or time.monotonic()>=deadline: break
  for value in remaining:
   if send(value,signal.SIGKILL) and value[0] not in killed: killed.append(value[0])
  time.sleep(.1)
 result={'task_absent':False,'term_pids':terminated,'kill_pids':killed,
         'survivors':[value[0] for value in active()]}
 (p/'reports').mkdir(exist_ok=True)
 audit=p/'reports/controller-stop.json'
 if not audit.exists() or result['survivors']: audit.write_text(json.dumps(result,indent=2)+'\\n')
 print(json.dumps(result))
''', "stop")


def seal_code(owned, token, execution):
    return remote_script(owned, token, f'''
assert verified_task() is not None, 'task disappeared before collection'
assert not active(), 'task processes are still running'
(p/'reports/controller-execution.json').write_text(json.dumps({execution!r},indent=2)+'\\n')
files=report_files()
manifest=p/'reports/.collection-manifest.json'
manifest.write_text(json.dumps({{'format':'volvoxai-remote-reports/v1','files':files}},indent=2)+'\\n')
print(json.dumps({{'manifest_sha256':file_hash(manifest),'manifest_bytes':manifest.stat().st_size,
                  'file_count':len(files),'total_bytes':sum(item['bytes'] for item in files)}}))
''', "seal")


def cleanup_code(owned, token, *, manifest_hash=None, preserve_reports=False):
    return remote_script(owned, token, f'''
if verified_task() is None:
 print(json.dumps({{'task_removed':True,'already_absent':True,'recovered_bytes':None,
                   'available_bytes_after':shutil.disk_usage('/tmp').free}}))
else:
 assert not active(), 'refusing cleanup with live task processes'
 preserve={preserve_reports!r}
 if not preserve:
  manifest=p/'reports/.collection-manifest.json'
  assert file_hash(manifest)=={manifest_hash!r}, 'report manifest changed after collection'
  assert json.loads(manifest.read_text())['files']==report_files(), 'reports changed after collection'
 before=shutil.disk_usage('/tmp').free
 if preserve:
  for child in p.iterdir():
   if child.name in ('.owner-token','reports'): continue
   if child.is_dir() and not child.is_symlink(): shutil.rmtree(child)
   else: child.unlink()
 else: shutil.rmtree(p)
 after=shutil.disk_usage('/tmp').free
 print(json.dumps({{'recovered_bytes':after-before,'available_bytes_after':after,
                   'task_removed':not p.exists(),'reports_preserved':preserve}}))
''', "prune" if preserve_reports else "cleanup")


def safe_report_path(name):
    path = Path(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.parts[0] != "reports":
        raise ValueError("unsafe report archive path")
    return path


def extract_verified(archive_path, local, sealed):
    """Publish only a complete archive matching independently sealed hashes."""
    with tempfile.TemporaryDirectory(prefix=".collect-", dir=local) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive_path) as archive:
            members, names = archive.getmembers(), set()
            for item in members:
                normalized = safe_report_path(item.name).as_posix()
                if normalized in names or not (item.isdir() or item.isfile()):
                    raise ValueError("duplicate or non-regular report archive member")
                names.add(normalized)
            entry = archive.getmember(MANIFEST)
            if not entry.isfile() or entry.size != sealed["manifest_bytes"]:
                raise ValueError("report manifest size mismatch")
            content = archive.extractfile(entry).read()
            if hashlib.sha256(content).hexdigest() != sealed["manifest_sha256"]:
                raise ValueError("report manifest hash mismatch")
            manifest = json.loads(content)
            if manifest.get("format") != "volvoxai-remote-reports/v1":
                raise ValueError("unknown report manifest format")
            entries = {}
            for entry in manifest["files"]:
                name = safe_report_path(entry["path"]).as_posix()
                if name in entries or name == MANIFEST or not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
                    raise ValueError("invalid report manifest entry")
                entries[name] = entry
            if len(entries) != sealed["file_count"] or sum(e["bytes"] for e in entries.values()) != sealed["total_bytes"]:
                raise ValueError("report manifest totals mismatch")
            if {safe_report_path(m.name).as_posix() for m in members if m.isfile()} != {*entries, MANIFEST}:
                raise ValueError("report archive inventory mismatch")
            for name, entry in entries.items():
                if archive.getmember(name).size != entry["bytes"]:
                    raise ValueError("report archive file size mismatch")
            if shutil.disk_usage(local).free < sealed["total_bytes"] + 8 * 1024 * 1024:
                raise OSError("not enough local space to retain reports")
            archive.extractall(staging)
        for name, entry in entries.items():
            if digest(staging / name) != entry["sha256"]:
                raise ValueError("report file hash mismatch")
        destination = local / "reports"
        if destination.exists():
            if digest(local / MANIFEST) != sealed["manifest_sha256"] or any(
                not (local / name).is_file() or digest(local / name) != entry["sha256"]
                for name, entry in entries.items()
            ):
                raise ValueError("existing local reports differ from sealed remote reports")
        else:
            (staging / "reports").replace(destination)


def retry_json(python, code, *, timeout, attempts=3, delay=time.sleep):
    for attempt in range(attempts):
        try:
            response = python(code, capture_output=True, text=True, timeout=timeout)
            return json.loads(response.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            if attempt + 1 == attempts:
                raise
            delay(min(attempt + 1, 3))


def finalize_task(remote, python, local, owned, token, execution, *, delay=time.sleep):
    local = Path(local)
    try:
        stopped = retry_json(python, stop_code(owned, token), timeout=20, delay=delay)
        write_json(local / "stop.json", stopped)
        if stopped.get("survivors"):
            raise RuntimeError("task processes survived SIGKILL; remote files retained")
        if stopped.get("task_absent"):
            write_json(local / "cleanup.json", {"task_removed": True, "already_absent": True})
            return True
    except Exception as error:
        write_json(local / "cleanup.json", {"task_removed": False, "reports_preserved": True,
            "error": error_summary(error), "reason": "process shutdown or ownership was not verified"})
        return False
    attempts, sealed = [], None
    for attempt in range(1, 4):
        archive = local / f".reports-{attempt}.tar"
        try:
            sealed = retry_json(python, seal_code(owned, token, execution), timeout=30, attempts=1, delay=delay)
            with archive.open("wb") as stream:
                remote("tar -C " + shlex.quote(owned) + " -cf - reports", stdout=stream, timeout=90)
            extract_verified(archive, local, sealed)
            attempts.append({"attempt": attempt, "verified": True, "archive_sha256": digest(archive)})
            write_json(local / "collection.json", {"verified": True, **sealed, "attempts": attempts})
            break
        except Exception as error:
            attempts.append({"attempt": attempt, "verified": False, "error": error_summary(error)})
            write_json(local / "collection.json", {"verified": False, "attempts": attempts})
            if attempt < 3:
                delay(attempt)
        finally:
            archive.unlink(missing_ok=True)
    else:
        try:
            result = retry_json(python, cleanup_code(owned, token, preserve_reports=True), timeout=90, delay=delay)
            result["reason"] = "report collection failed; compact reports preserved for --recover"
        except Exception as error:
            result = {"task_removed": False, "reports_preserved": True, "error": error_summary(error)}
        write_json(local / "cleanup.json", result)
        return False
    try:
        result = retry_json(python, cleanup_code(owned, token, manifest_hash=sealed["manifest_sha256"]),
                            timeout=90, delay=delay)
        write_json(local / "cleanup.json", result)
        return bool(result.get("task_removed"))
    except Exception as error:
        write_json(local / "cleanup.json", {"task_removed": False, "reports_verified_locally": True,
            "error": error_summary(error)})
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--python", default="python3", help="Existing remote Python with numpy, protobuf and onnxruntime")
    parser.add_argument("--deno", default="deno", help="Remote Deno with physical WebGPU support")
    parser.add_argument("--mode", choices=("full", "all", "wasm", "components"), default="full")
    parser.add_argument("--backend", default=None, help="Filter to run only a single backend (e.g. cuda)")
    parser.add_argument("--recover", type=Path, help="Retry collection/cleanup for a prior local run directory")
    args = parser.parse_args()
    token = uuid.uuid4().hex
    owned = f"/tmp/volvox-tts-examples-{token}"
    local = args.recover.resolve() if args.recover else ROOT / "build/tts-examples/remote" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + token[:8])
    if not args.recover:
        local.mkdir(parents=True)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", args.target]
    def remote(code, **kwargs):
        return subprocess.run([*ssh, code], check=True, **kwargs)
    def python(code, **kwargs):
        return remote("python3 -c " + shlex.quote(code), **kwargs)

    if args.recover:
        local = args.recover.resolve()
        recovery = json.loads((local / "recovery.json").read_text())
        if recovery["target"] != args.target:
            raise ValueError("recovery target differs from original authorized target")
        ok = finalize_task(remote, python, local, recovery["owned"], recovery["token"],
                           json.loads((local / "execution.json").read_text()))
        print("Reports:", local, "Cleanup verified:", ok, flush=True)
        raise SystemExit(0 if ok else 1)

    write_json(local / "recovery.json", {"target": args.target, "owned": owned, "token": token})
    execution = {"mode": args.mode, "exit_code": None}
    write_json(local / "execution.json", execution)

    version = json.loads((ROOT / "package.json").read_text())["version"]
    roots = ["examples/tts_common", "examples/common/RuntimeRelease.js", "python/volvoxai", "runtime/generated/python",
             "package.json", f"native/libvolvoxai.so.{version}", f"native/libvolvoxai-lite.so.{version}",
             "native/libsynurang_module_host.so", f"dist/{version}"]
    if args.mode == "full":
        # Transfer only the artifacts consumed by whole-model benchmarking.
        # Source checkpoints and intermediate ONNX exports remain local.
        for directory in ("piper-full", "kokoro-complete"):
            manifest_path = ROOT / "build/tts-examples" / directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            model_root = manifest_path.parent
            required = {manifest_path, model_root / manifest["onnx"]["path"]}
            required.update(model_root / path for path in manifest["onnx"].get("external_data", []))
            for case in manifest["cases"]:
                required.update((model_root / case["inputs"], model_root / case["reference"]))
                required.update(model_root / desc["path"] for desc in (*case["input_files"], *case["output_files"]))
            for entry in manifest["volvox"]["pipeline"]:
                stage_path = model_root / entry["manifest"]
                stage = json.loads(stage_path.read_text())
                required.add(stage_path)
                required.add(stage_path.parent / stage["volvox"]["graph"])
                required.update(stage_path.parent / path for path in stage["volvox"].get("weights", []))
            roots.extend(str(path.resolve().relative_to(ROOT)) for path in sorted(required))
    else:
        roots.extend(["build/tts-examples/assets", "build/tts-examples/piper", "build/tts-examples/kokoro"] if args.mode == "all"
            else ["build/tts-examples/piper/vocoder", "build/tts-examples/kokoro/spectral-head", "build/tts-examples/kokoro/generator-stage2"])
    paths = []
    for name in roots:
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"Prepare and build this artifact before remote execution: {name}")
        paths.extend(p for p in (path.rglob("*") if path.is_dir() else [path]) if p.is_file() and
                     "__pycache__" not in p.parts and p.suffix not in {".pyc", ".log"} and p.name != "frontend.onnx")
    payload = local / "payload.tar"
    created = False
    result = None
    cleanup_ok = False
    try:
        with tarfile.open(payload, "w") as archive:
            for path in paths:
                archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        value = hashlib.sha256()
        with payload.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        digest = value.hexdigest()
        execution["payload_sha256"] = digest
        # A lost setup response may still have created the owned directory.
        created = True
        setup = python("\n".join(["import pathlib,shutil,json", f"p=pathlib.Path({owned!r})",
            "free=shutil.disk_usage('/tmp').free", f"assert free>{payload.stat().st_size * 2 + 1073741824}",
            "p.mkdir(mode=0o700)", f"(p/'.owner-token').write_text({token!r})",
            "(p/'reports').mkdir()",
            "print(json.dumps({'available_bytes_before':free}))"]), capture_output=True, text=True, timeout=30)
        (local / "setup.json").write_text(setup.stdout)
        transfer_timeout = max(1800, int(payload.stat().st_size / (100 * 1024)))
        print(f"Transferring payload ({payload.stat().st_size / 1024 / 1024:.1f} MB) with timeout {transfer_timeout}s...", flush=True)
        transfer_ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=0", "-o", "TCPKeepAlive=yes", args.target]
        with payload.open("rb") as stream:
            subprocess.run([*transfer_ssh, "cat > " + shlex.quote(owned + "/payload.tar")], stdin=stream, check=True, timeout=transfer_timeout)
        print("Payload transfer complete. Verifying and running benchmark...", flush=True)
        verify = "\n".join(["import pathlib,hashlib,tarfile", "p=pathlib.Path('payload.tar')", "h=hashlib.sha256()",
            "with p.open('rb') as f:", " for b in iter(lambda:f.read(1048576),b''): h.update(b)",
            f"assert h.hexdigest()=={digest!r}", "with tarfile.open(p) as t: t.extractall('.')", "p.unlink()"])
        command = " && ".join(["cd " + shlex.quote(owned), "python3 -c " + shlex.quote(verify),
            "mkdir -p tmp/xdg", "chmod 700 tmp/xdg",
            "env TMPDIR=" + shlex.quote(owned + "/tmp") + " XDG_RUNTIME_DIR=" + shlex.quote(owned + "/tmp/xdg") +
            " XDG_CACHE_HOME=" + shlex.quote(owned + "/tmp/cache") + " DENO_DIR=" + shlex.quote(owned + "/tmp/deno") +
            " DENO_WEBGPU_BACKEND=vulkan EGL_PLATFORM=surfaceless timeout --kill-after=15s 1800s " +
            shlex.quote(args.python) + " examples/tts_common/remote_runner.py --out reports --mode " + args.mode +
            (" --backend " + shlex.quote(args.backend) if args.backend else "") + " --deno " + shlex.quote(args.deno)])
        result = subprocess.run([*ssh, command], timeout=1860)
        execution["exit_code"] = result.returncode
    except BaseException as error:
        execution["error"] = error_summary(error)
        raise
    finally:
        write_json(local / "execution.json", execution)
        try:
            if created:
                cleanup_ok = finalize_task(remote, python, local, owned, token, execution)
                print("Reports:", local.relative_to(ROOT), flush=True)
                print("Cleanup:", (local / "cleanup.json").read_text().strip(), flush=True)
                if not cleanup_ok:
                    print("Retry with --target and --recover", local.relative_to(ROOT), flush=True)
        finally:
            payload.unlink(missing_ok=True)
    if result is None or result.returncode or not cleanup_ok:
        raise SystemExit(result.returncode if result is not None and result.returncode else 1)


if __name__ == "__main__":
    main()
