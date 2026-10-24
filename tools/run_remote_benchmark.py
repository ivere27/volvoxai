#!/usr/bin/env python3
"""Run a prepared benchmark bundle in a disposable, owned SSH directory.

The spec names a payload directory, read-only shared or local weight files,
and ordered commands. Results are recovered even on failure; no shared cache
is written. Every remote deletion verifies a unique ownership token. Local
payload and weight paths are relative to the spec file. Remote shared weight
paths must be absolute. The remote root defaults to the SSH user's home.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import signal
import subprocess
import tarfile
import time
import uuid


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--host', required=True, help='SSH destination, such as user@gpu-host')
    parser.add_argument('--remote-root', help='Existing absolute directory on the remote host (default: remote home)')
    args = parser.parse_args(argv)
    if args.remote_root is not None and not PurePosixPath(args.remote_root).is_absolute():
        parser.error('--remote-root must be an absolute remote path')
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    payload = (spec_path.parent / spec['payload_dir']).resolve()
    if not payload.is_dir():
        raise NotADirectoryError(payload)
    reserved = {'.owner', '.job.json', '.supervisor.json', 'results',
                'run-spec.json', 'supervise.py', 'payload.tar.gz'}
    if reserved.intersection(item.name for item in payload.iterdir()):
        raise ValueError('payload contains reserved controller files or results directory')
    for weight in spec['weights']:
        destination = PurePosixPath(weight['destination'])
        if destination.is_absolute() or '..' in destination.parts or not destination.parts:
            raise ValueError(f'invalid weight destination: {destination}')
        if destination.parts[0] in reserved:
            raise ValueError(f'weight destination is reserved: {destination}')
        if ('shared' in weight) == ('local' in weight):
            raise ValueError('each weight must specify exactly one shared or local source')
        if 'shared' in weight:
            if not PurePosixPath(weight['shared']).is_absolute():
                raise ValueError('shared weight paths must be absolute remote paths')
        else:
            weight['local'] = str((spec_path.parent / weight['local']).resolve())
            if not Path(weight['local']).is_file():
                raise FileNotFoundError(weight['local'])
    out = args.out_dir.resolve()
    if out.is_relative_to(payload):
        raise ValueError('--out-dir must be outside payload_dir')
    out.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host]
    token = uuid.uuid4().hex
    receipt = {'status': 'RUNNING', 'host': args.host, 'token': token,
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'spec_sha256': sha256(args.spec)}
    remote = None
    guard = None
    archive = out / 'payload.tar.gz'

    def save():
        (out / 'remote-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def remote_python(code, timeout=120):
        r = subprocess.run(ssh + ['python3 -'], input=code, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        if r.returncode:
            raise RuntimeError(r.stdout[-8000:])
        return r.stdout

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        remote_root = json.loads(remote_python(f"""
import json, pathlib
root = pathlib.Path({args.remote_root!r}) if {args.remote_root is not None!r} else pathlib.Path.home()
root = root.resolve(strict=True)
if not root.is_dir():
    raise NotADirectoryError(root)
print(json.dumps(str(root)))
"""))
        remote = str(PurePosixPath(remote_root) / ('volvox-model-benchmark.' + token))
        guard = f"""
import json, pathlib, shutil, os, signal, time
p = pathlib.Path({remote!r})
if p.parent != pathlib.Path({remote_root!r}) or p.name != {'volvox-model-benchmark.' + token!r}:
    raise RuntimeError('owned directory path mismatch')
if p.is_symlink():
    raise RuntimeError('owned directory must not be a symlink')
if not p.exists():
    print(json.dumps({{'absent': True}}))
    raise SystemExit(0)
if (p / '.owner').is_symlink() or (p / '.owner').read_text() != {token!r}:
    raise RuntimeError('ownership mismatch')
"""
        receipt['remote_root'] = remote_root
        receipt['remote_dir'] = remote
        save()
        setup = remote_python(f"""
import json, pathlib, shutil
free = shutil.disk_usage({remote_root!r}).free
if free < {spec.get('minimum_free_bytes', 8 * 1024**3)}:
    raise RuntimeError('insufficient remote free space')
p = pathlib.Path({remote!r})
p.mkdir(mode=0o700)
(p / '.owner').write_text({token!r})
(p / 'results').mkdir()
print(json.dumps({{'disk_available_before': free}}))
""")
        receipt.update(json.loads(setup))
        save()
        print(f'Owned remote directory: {remote}', flush=True)
        with tarfile.open(archive, 'w:gz') as tar:
            for item in sorted(payload.iterdir()):
                tar.add(item, arcname=item.name)
        archive_hash = sha256(archive)
        receipt['payload_sha256'] = archive_hash
        save()
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(archive),
                        f'{args.host}:{remote}/payload.tar.gz'], check=True, timeout=180)
        remote_python(guard + f"""
import tarfile, hashlib
a = p / 'payload.tar.gz'
if hashlib.sha256(a.read_bytes()).hexdigest() != {archive_hash!r}:
    raise RuntimeError('payload hash mismatch')
with tarfile.open(a) as tar:
    tar.extractall(p)
a.unlink()
""")
        archive.unlink()
        receipt['weights'] = []
        for weight in spec['weights']:
            destination = weight['destination']
            remote_python(guard + f"""
destination = p / {destination!r}
if not destination.resolve().is_relative_to(p):
    raise RuntimeError('weight destination escapes owned directory')
if destination.exists() or destination.is_symlink():
    raise FileExistsError(destination)
destination.parent.mkdir(parents=True, exist_ok=True)
""")
            if 'shared' in weight:
                remote_python(guard + f"""
src = pathlib.Path({weight['shared']!r})
if not src.is_file():
    raise FileNotFoundError(src)
(p / {destination!r}).symlink_to(src)
""")
            else:
                print(f'Transferring {destination} into owned directory', flush=True)
                subprocess.run(['rsync', '-s', '-L', '--partial', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                                weight['local'], f'{args.host}:{remote}/{destination}'],
                               check=True, timeout=1200)
            actual = json.loads(remote_python(guard + f"""
import hashlib
f = p / {destination!r}
h = hashlib.sha256()
with f.open('rb') as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(chunk)
print(json.dumps({{'destination': {destination!r}, 'sha256': h.hexdigest(), 'bytes': f.stat().st_size,
                  'shared': f.is_symlink()}}))
""", timeout=180))
            if actual['sha256'] != weight['sha256']:
                raise RuntimeError(f'weight hash mismatch: {destination}')
            receipt['weights'].append(actual)
            save()
        run_spec = {'steps': spec['steps'], 'environment': spec.get('environment', {})}
        remote_python(guard + f"(p / 'run-spec.json').write_text({json.dumps(run_spec)!r})\n")
        supervisor = r'''
import json, os, pathlib, signal, subprocess, time
p = pathlib.Path.cwd()
(p / '.supervisor.json').write_text(json.dumps({'pid': os.getpid(), 'starttime': pathlib.Path('/proc/self/stat').read_text().split()[21]}))
def interrupted(signum, frame):
    raise KeyboardInterrupt('supervisor signal ' + str(signum))
signal.signal(signal.SIGTERM, interrupted)
spec = json.loads((p / 'run-spec.json').read_text())
env = dict(os.environ)
env.update(spec['environment'])
env['VOLVOXAI_LIBRARY'] = str(p / 'native/libvolvoxai.so')
env['PYTHONPATH'] = str(p / 'python') + ':' + str(p / 'runtime/generated/python')
env['LD_LIBRARY_PATH'] = str(p / 'native') + ':' + env.get('LD_LIBRARY_PATH', '')
report = {'status': 'RUNNING', 'steps': []}
try:
    for step in spec['steps']:
        name = step['name']
        print('START ' + name, flush=True)
        started = time.time()
        with (p / 'results' / (name + '.log')).open('w') as log:
            child = subprocess.Popen(step['argv'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = pathlib.Path('/proc', str(child.pid), 'stat').read_text().split()[21]
            (p / '.job.json').write_text(json.dumps({'pid': child.pid, 'starttime': identity}))
            try:
                code = child.wait(timeout=step.get('timeout_seconds', 3600))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                (p / '.job.json').unlink(missing_ok=True)
        report['steps'].append({'name': name, 'exit_code': code, 'elapsed_seconds': time.time() - started})
        (p / 'results/supervisor.json').write_text(json.dumps(report, indent=2))
        print('FINISH ' + name + ' exit=' + str(code), flush=True)
        if code:
            raise RuntimeError('step failed: ' + name)
    report['status'] = 'COMPLETED'
except BaseException as exc:
    report['status'] = 'FAILED'
    report['error'] = repr(exc)
    raise
finally:
    (p / 'results/supervisor.json').write_text(json.dumps(report, indent=2))
'''
        remote_python(guard + f"(p / 'supervise.py').write_text({supervisor!r})\n")
        command = f'cd {shlex.quote(remote)} && python3 -u supervise.py'
        subprocess.run(ssh + [command], check=True,
                       timeout=sum(step.get('timeout_seconds', 3600) for step in spec['steps']) + 120)
        receipt['status'] = 'COMPLETED'
    except BaseException as exc:
        receipt['status'] = 'FAILED'
        receipt['error'] = repr(exc)
        raise
    finally:
        archive.unlink(missing_ok=True)
        if remote:
            try:
                # A timed-out/disconnected SSH session may leave its owned child.
                stopped = json.loads(remote_python(guard + """
def stop(identity_file, group, wait_seconds):
    try:
        job = json.loads(identity_file.read_text())
        proc = pathlib.Path('/proc', str(job['pid']))
        def same():
            try:
                return (proc / 'stat').read_text().split()[21] == job['starttime']
            except FileNotFoundError:
                return False
        if not same():
            return
        if pathlib.Path(os.readlink(proc / 'cwd')) != p:
            raise RuntimeError('job does not belong to owned directory')
        kill = os.killpg if group else os.kill
        kill(job['pid'], signal.SIGTERM)
        for _ in range(int(wait_seconds * 10)):
            if not same():
                return
            time.sleep(.1)
        if same():
            kill(job['pid'], signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        pass
stop(p / '.supervisor.json', False, 20)
stop(p / '.job.json', True, 3)
print(json.dumps({'absent': False}))
"""))
                if stopped['absent']:
                    receipt['cleanup'] = stopped
                else:
                    recovered = out / 'results'
                    recovered.mkdir(exist_ok=True)
                    subprocess.run(['rsync', '-s', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                                    f'{args.host}:{remote}/results/', str(recovered) + '/'], check=True, timeout=600)
                    cleanup = json.loads(remote_python(guard + """
size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
before = shutil.disk_usage(p.parent).free
shutil.rmtree(p)
after = shutil.disk_usage(p.parent).free
print(json.dumps({'removed': str(p), 'owned_file_bytes': size, 'recovered_bytes': after - before,
                  'disk_available_after': after}))
"""))
                    receipt['cleanup'] = cleanup
                    print('Cleaned owned directory: ' + json.dumps(cleanup), flush=True)
            except BaseException as cleanup_error:
                receipt['cleanup_error'] = repr(cleanup_error)
                receipt['status'] = 'CLEANUP_FAILED'
                receipt['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                save()
                print('Cleanup/recovery needs attention: ' + repr(cleanup_error), flush=True)
                signal.signal(signal.SIGTERM, previous_sigterm)
                raise
        receipt['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
