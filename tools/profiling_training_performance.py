#!/usr/bin/env python3
"""Measure disabled TrainStep latency against an archived matching release.

Uses one 2x2 Linear, cross entropy and SGD. Every worker checks a closed-form
first update before warming. Native/WASM alternate blocks; WebGPU processes
run sequentially because the qualification runner cannot host two GPU owners.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--backend', action='append', choices=['native', 'wasm', 'cuda', 'opengl', 'vulkan', 'webgpu'])
parser.add_argument('--output', type=Path)
parser.add_argument('--cpu', type=int)
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--worker', action='store_true')
args = parser.parse_args()


def native_worker():
    release = args.baseline.resolve()
    version = json.loads((release / 'package.json').read_text())['version']
    sys.path.insert(0, str(release / 'python'))
    import volvoxai as vx
    p = vx.pb
    backend = args.backend[0]
    if backend == 'native':
        backend = 'cpu'
    graph = dict(format='volvox-graph/v1', dimensions={}, inputs={'x': dict(dtype='float32', shape=[1, 2])},
        nodes=[dict(id='linear', opType='Linear', inputs={'input': 'x', 'weight': 'w'},
                    outputs={'out': dict(tensor='logits', dtype='float32', shape=[1, 2])},
                    params={'weight_layout': 'din_dout'})], outputs=['logits'])
    header = json.dumps({'w': dict(dtype='F32', shape=[2, 2], data_offsets=[0, 16])}).encode()
    header += b' ' * (-len(header) % 8)
    weights = struct.pack('<Q', len(header)) + header + struct.pack('<4f', .2, -.4, .1, .3)
    with vx.open_library(release / 'native' / f'libvolvoxai.so.{version}',
                         loader=ROOT / 'native/libsynurang_module_host.so') as host:
        inference, training = vx.VxInferenceServiceClient(host), vx.VxTrainingServiceClient(host)
        runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
        model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=p.ModelPackage(graph_document=json.dumps(graph).encode(), weight_shards=[weights])))
        trainer = training.create_trainer(p.CreateTrainerRequest(model_id=model.model_id, backend=backend))
        request = p.TrainStepRequest(trainer_id=trainer.trainer_id,
            inputs=[p.Tensor(name='x', shape=[1, 2], dtype=p.DataType.DATA_TYPE_F32, inline=struct.pack('<2f', 1, 0))],
            losses=[p.CrossEntropyLoss(name='ce', logits_name='logits', targets=p.Tensor(
                shape=[1], dtype=p.DataType.DATA_TYPE_I32, inline=struct.pack('<i', 0)))], trainable_names=['w'],
            optimizer=p.TrainerOptimizerOptions(kind=p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD, learning_rate=1e-5))
        def run():
            result = training.train_step(request)
            assert result.update_applied and result.backend == backend and math.isfinite(result.loss)
            return result
        first = run()
        probability = math.exp(.2) / (math.exp(.2) + math.exp(-.4))
        assert abs(first.loss + math.log(probability)) < 1e-6
        shard = training.export_trainer_weights(p.ExportTrainerWeightsRequest(trainer_id=trainer.trainer_id)).shards[0]
        length, = struct.unpack_from('<Q', shard)
        offset = json.loads(shard[8:8 + length])['w']['data_offsets'][0]
        actual = struct.unpack_from('<4f', shard, 8 + length + offset)
        expected = [.2 + 1e-5 * (1 - probability), -.4 - 1e-5 * (1 - probability), .1, .3]
        assert max(abs(a-b) for a,b in zip(actual, expected)) < 1e-6
        for _ in range(1000): run()
        print(json.dumps({'ready': True}), flush=True)
        for sample in range(21):
            assert input() == 'sample'
            start = time.perf_counter_ns()
            for _ in range(20): run()
            print(json.dumps({'sample': sample, 'wallMs': (time.perf_counter_ns() - start) / 20e6}), flush=True)
        assert input() == 'done'


def read(worker):
    while True:
        line = worker.stdout.readline()
        if line.startswith('[VolvoxAI GPU]'): continue
        if not line: raise RuntimeError(f'worker exited: {worker.poll()}')
        return json.loads(line)


if args.worker:
    native_worker()
    sys.exit(0)
assert args.output and args.rounds > 0
args.output.parent.mkdir(parents=True, exist_ok=True)
prefix = ['taskset', '-c', str(args.cpu)] if args.cpu is not None else []
rows = []
for backend in args.backend or ['native', 'wasm']:
    for round_index in range(args.rounds):
        def start_worker(release):
            if backend in ('wasm', 'webgpu'):
                runner = ['node'] if backend == 'wasm' else [str(ROOT / 'build/deno/target/webgpu-fix/deno'),
                    'run', '--no-config', '--unstable-webgpu', '--allow-read', '--allow-env', '--allow-ffi']
                command = runner + ['tools/profiling_training_performance.mjs', str(release), backend]
            else:
                command = [sys.executable, __file__, '--worker', '--baseline', str(release), '--backend', backend]
            worker = subprocess.Popen(prefix + command, cwd=ROOT, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            assert read(worker) == {'ready': True}
            return worker
        workers, samples = [], [[], []]
        order = [round_index % 2, 1 - round_index % 2]
        releases = [args.baseline.resolve(), ROOT]
        try:
            if backend == 'webgpu':
                for index in order:
                    worker = start_worker(releases[index]); workers.append(worker)
                    for _ in range(21):
                        worker.stdin.write('sample\n'); worker.stdin.flush(); samples[index].append(read(worker)['wallMs'])
                    worker.stdin.write('done\n'); worker.stdin.flush(); assert worker.wait() == 0
            else:
                workers = [start_worker(release) for release in releases]
                for sample in range(21):
                    for index in ([0, 1] if (sample + round_index) % 2 == 0 else [1, 0]):
                        workers[index].stdin.write('sample\n'); workers[index].stdin.flush()
                        samples[index].append(read(workers[index])['wallMs'])
                for worker in workers:
                    worker.stdin.write('done\n'); worker.stdin.flush(); assert worker.wait() == 0
        finally:
            for worker in workers:
                if worker.poll() is None: worker.kill(); worker.wait()
        row = dict(backend=backend, round=round_index, warmup=1000, batch=20, samples=samples,
            method='sequential-process-pair' if backend == 'webgpu' else 'alternating-blocks',
            changePercent=statistics.median((b/a-1)*100 for a,b in zip(*samples)))
        rows.append(row); args.output.write_text(json.dumps(rows, indent=2) + '\n')
        print(backend, round_index, row['changePercent'], flush=True)
