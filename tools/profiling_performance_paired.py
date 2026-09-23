#!/usr/bin/env python3
"""Compare warmed native/WASM releases, alternating 50-call measurement blocks.

The archive contains the release files, package.json, python/volvoxai and
runtime/generated/python from before the change. Finish builds before running.
Use --cpu on Linux to keep all worker processes on one logical CPU.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--output', type=Path, default=root / 'build/profiling-performance-paired.json')
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--cpu', type=int)
parser.add_argument('--backend', action='append', choices=['wasm', 'native', 'cuda', 'opengl', 'vulkan', 'webgpu'])
parser.add_argument('--deno', type=Path, default=root / 'build/deno/target/webgpu-fix/deno')
parser.add_argument('--webgpu-sequential', action='store_true',
                    help='measure WebGPU case 0 in isolated processes when two live devices cannot coexist')
args = parser.parse_args()
if args.rounds < 1:
    parser.error('--rounds must be positive')
archive = args.baseline.resolve()
versions = {release: json.loads((release / 'package.json').read_text())['version']
            for release in [archive, root]}
args.output.parent.mkdir(parents=True, exist_ok=True)
prefix = ['taskset', '-c', str(args.cpu)] if args.cpu is not None else []
commands = {
    'wasm': [
        ['node', 'tools/profiling_performance.mjs',
         str(release / 'dist' / versions[release] / 'volvoxai.lite.js'), '--paired']
        for release in [archive, root]
    ],
    'native': [
        [sys.executable, 'tools/profiling_performance.py', str(release),
         str(release / 'native' / f'libvolvoxai-lite.so.{versions[release]}'), '--paired']
        for release in [archive, root]
    ],
}
reports = []


def observation(worker):
    while True:
        line = worker.stdout.readline()
        if line.startswith('[VolvoxAI GPU]'):
            print(line.rstrip(), file=sys.stderr)
            continue
        if not line:
            raise RuntimeError(f'benchmark worker exited before its response: {worker.poll()}')
        return json.loads(line)


for backend in ['cuda', 'opengl', 'vulkan']:
    commands[backend] = [
        [sys.executable, 'tools/profiling_performance.py', str(release),
         str(release / 'native' / f'libvolvoxai.so.{versions[release]}'), '--backend', backend,
         '--warmup', '300', '--batch', '20', '--paired']
        for release in [archive, root]
    ]
commands['webgpu'] = [
    [str(args.deno), 'run', '--no-config', '--unstable-webgpu', '--allow-read', '--allow-env', '--allow-ffi',
     'tools/profiling_performance.mjs', str(release / 'dist' / versions[release] / 'volvoxai.js'), '--paired', '--webgpu']
    for release in [archive, root]
]
for backend in args.backend or ['wasm', 'native']:
    variants = commands[backend]
    for round_index in range(args.rounds):
        if backend == 'webgpu' and args.webgpu_sequential:
            pair = {}
            for index in [0, 1] if round_index % 2 == 0 else [1, 0]:
                command = [v for v in variants[index] if v != '--paired'] + ['--case=0']
                completed = subprocess.run([*prefix, *command], cwd=root, text=True,
                                           stdout=subprocess.PIPE, check=True, timeout=300)
                payload = json.loads(completed.stdout.strip())
                pair['baseline' if index == 0 else 'current'] = payload['cases'][0]
            row = {'backend': backend, 'round': round_index, 'method': 'sequential-process-pair',
                   'nodes': pair['current']['nodes'], 'width': pair['current']['width'], 'measurements': pair}
            for metric in ['engine', 'wall']:
                row[metric + 'Ms'] = {side: pair[side][metric]['p50Ms'] for side in ['baseline', 'current']}
                row[metric + 'Ms']['changePercent'] = (
                    row[metric + 'Ms']['current'] / row[metric + 'Ms']['baseline'] - 1) * 100
            reports.append(row)
            args.output.write_text(json.dumps(reports, indent=2))
            print(backend, round_index, row['engineMs'], row['wallMs'], flush=True)
            continue
        workers, ready = [], []
        try:
            # Keep GPU warmups as well as measurement blocks sequential.
            for command in variants:
                worker = subprocess.Popen([*prefix, *command], cwd=root,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                workers.append(worker)
                ready.append(observation(worker))
            for case in range(3):
                assert ready[0] == ready[1] and 'ready' in ready[0], ready
                samples = []
                for sample in range(21):
                    pair = {}
                    for index in [0, 1] if sample % 2 == 0 else [1, 0]:
                        worker = workers[index]
                        worker.stdin.write('sample\n')
                        worker.stdin.flush()
                        value = observation(worker)
                        assert value['sample'] == sample
                        pair['baseline' if index == 0 else 'current'] = value
                    samples.append(pair)
                row = {'backend': backend, 'round': round_index,
                       **ready[0]['ready'], 'samples': samples}
                for metric in ['engineMs', 'wallMs']:
                    row[metric] = {
                        'baseline': statistics.median(s['baseline'][metric] for s in samples),
                        'current': statistics.median(s['current'][metric] for s in samples),
                        'pairedChangePercent': statistics.median(
                            (s['current'][metric] / s['baseline'][metric] - 1) * 100
                            for s in samples),
                    }
                reports.append(row)
                args.output.write_text(json.dumps(reports, indent=2))
                print(backend, round_index, row['nodes'], row['width'],
                      row['engineMs'], row['wallMs'], flush=True)
                ready = []
                for worker in workers:
                    worker.stdin.write('next\n')
                    worker.stdin.flush()
                    ready.append(observation(worker))
            for worker in workers:
                assert worker.wait() == 0
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait()
