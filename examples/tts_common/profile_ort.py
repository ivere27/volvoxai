#!/usr/bin/env python3
"""Profile complete ONNX TTS requests independently of Volvox execution.

Ordinary latency uses a session with profiling disabled. A separate session
retains its untouched ORT trace; model_run intervals identify first execution,
warmups and measured requests. ORT Node *_kernel_time events are host intervals,
including CUDA provider nodes. Only explicit Kernel events are device intervals.
Use --external-capture for a separate Nsight run bounded by cudaProfilerApi.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tts_common.artifacts import sha256
from examples.tts_common.benchmark_native import checked_cast, compare_outputs, load_npz, summarize


def memory_observation():
    resident, peak = None, None
    if sys.platform.startswith('linux'):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                resident = int(line.split()[1]) * 1024
            elif line.startswith('VmHWM:'):
                peak = int(line.split()[1]) * 1024
    method = '/proc/self/status VmHWM' if peak is not None else 'getrusage(RUSAGE_SELF).ru_maxrss'
    if peak is None:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)
    return {'resident_bytes': resident,
            'process_peak_rss_bytes': peak, 'peak_method': method,
            'scope': 'Entire process, including sessions, Python, graph inspection and profiling; not device allocation.'}


def validate_outputs(actual, expected, waveform=None):
    """Retain duration changes without truncation, alignment or a false pass."""
    if actual.keys() != expected.keys():
        raise ValueError('Output names differ')
    differences = {}
    equal_shapes = {}
    for name, value in actual.items():
        reference = expected[name]
        if value.dtype != reference.dtype or not value.size or not np.isfinite(value).all():
            raise ValueError(f'{name}: invalid output dtype, size or finiteness')
        if value.shape != reference.shape:
            differences[name] = {'numerical_parity': False, 'criterion': 'shape',
                'reason': 'output_shape_changed', 'shape': list(value.shape),
                'reference_shape': list(reference.shape), 'dtype': value.dtype.name,
                'finite': True, 'samples': value.size, 'reference_samples': reference.size,
                'elementwise_comparison': 'not_performed; no truncation or time alignment'}
        else:
            equal_shapes[name] = value
    if equal_shapes:
        differences.update(compare_outputs(equal_shapes, {k: expected[k] for k in equal_shapes},
            deterministic=True, audio_output='audio', rtol=1e-4, atol=1e-5,
            waveform=waveform, report_waveform_differences=True))
    return differences


def summarize_trace(events, warmups=2, repeats=5):
    """Partition by completed model_run, preserving host/device distinctions."""
    windows = sorted((e for e in events if e.get('name') == 'model_run' and
                      e.get('ph') == 'X'), key=lambda e: e['ts'])
    expected = 1 + warmups + repeats
    if len(windows) != expected:
        raise ValueError(f'Expected {expected} model_run intervals, found {len(windows)}')
    for a, b in zip(windows, windows[1:]):
        if a['ts'] + a['dur'] > b['ts']:
            raise ValueError('Overlapping model_run intervals cannot identify serial requests')
    groups = {key: {} for key in ('operators', 'providers', 'nodes', 'shapes')}
    runs, device_events, unassigned = [], [], 0
    for index, window in enumerate(windows):
        start, end = window['ts'], window['ts'] + window['dur']
        kind = 'first' if index == 0 else 'warmup' if index <= warmups else 'measurement'
        nodes = [e for e in events if e.get('cat') == 'Node' and e.get('ph') == 'X' and
                 e.get('name', '').endswith('_kernel_time') and
                 start <= e.get('ts', -1) and e.get('ts', 0) + e.get('dur', 0) <= end]
        runs.append({'index': index, 'kind': kind, 'ts_us': start, 'duration_us': window['dur'],
                     'host_node_events': len(nodes)})
        if kind != 'measurement':
            continue
        sample = index - warmups - 1
        for event in nodes:
            args = event.get('args', {})
            provider, op = args.get('provider', 'unknown'), args.get('op_name', 'unknown')
            name = event['name'].removesuffix('_kernel_time')
            shapes = {'inputs': args.get('input_type_shape'), 'outputs': args.get('output_type_shape')}
            identities = {'operators': {'provider': provider, 'op': op},
                'providers': {'provider': provider},
                'nodes': {'provider': provider, 'op': op, 'node': name, 'node_index': args.get('node_index')},
                'shapes': {'provider': provider, 'op': op, **shapes}}
            for category, identity in identities.items():
                key = json.dumps(identity, sort_keys=True)
                group = groups[category].setdefault(key, {**identity, 'event_count': 0,
                    'samples_per_run_host_us': [0.0] * repeats})
                group['event_count'] += 1
                group['samples_per_run_host_us'][sample] += event.get('dur', 0)
    for event in events:
        if event.get('cat') == 'Kernel' and event.get('ph') == 'X':
            device_events.append(event)
        if event.get('cat') == 'Node' and event.get('name', '').endswith('_kernel_time'):
            if not any(w['ts'] <= event.get('ts', -1) and
                       event.get('ts', 0) + event.get('dur', 0) <= w['ts'] + w['dur'] for w in windows):
                unassigned += 1
    result = {}
    for category, mapping in groups.items():
        result[category] = []
        for group in mapping.values():
            values = group['samples_per_run_host_us']
            result[category].append({**group, 'total_host_us': sum(values),
                'mean_per_run_host_us': statistics.mean(values),
                'median_per_run_host_us': statistics.median(values)})
        result[category].sort(key=lambda row: row['total_host_us'], reverse=True)
    return {'runs': runs, 'summaries': result, 'unassigned_host_node_events': unassigned,
        'explicit_device_kernel_events': len(device_events),
        'device_timing_available': bool(device_events),
        'timing_semantics': 'Node *_kernel_time is a host-side operator interval, even for CUDA. '
            'It can include submission, synchronization or waiting; it is not GPU kernel duration. '
            'Explicit Kernel events are counted separately. Real CUDA timelines require CUPTI/Nsight.'}


def graph_inventory(path):
    import onnx
    model = onnx.load_model(str(path), load_external_data=False)
    counts = collections.Counter((node.domain or 'ai.onnx', node.op_type) for node in model.graph.node)
    nodes = []
    for index, node in enumerate(model.graph.node):
        attrs = {}
        for attr in node.attribute:
            value = onnx.helper.get_attribute_value(attr)
            if isinstance(value, bytes): value = value.decode('utf8', errors='replace')
            if isinstance(value, (str, int, float)) or (isinstance(value, list) and
                    all(isinstance(v, (int, float, str)) for v in value)):
                attrs[attr.name] = value
            elif attr.type == onnx.AttributeProto.TENSOR:
                attrs[attr.name] = {'dtype': onnx.TensorProto.DataType.Name(value.data_type), 'shape': list(value.dims)}
            else:
                attrs[attr.name] = {'attribute_type': onnx.AttributeProto.AttributeType.Name(attr.type)}
        nodes.append({'index': index, 'name': node.name, 'domain': node.domain or 'ai.onnx',
                      'op': node.op_type, 'inputs': list(node.input), 'outputs': list(node.output), 'attributes': attrs})
    initializers = []
    for value in model.graph.initializer:
        external = {entry.key: entry.value for entry in value.external_data}
        initializers.append({'name': value.name, 'dtype': onnx.TensorProto.DataType.Name(value.data_type),
            'shape': list(value.dims), 'elements': math.prod(value.dims), 'raw_bytes': len(value.raw_data),
            'external': external})
    return {'sha256': sha256(path), 'bytes': path.stat().st_size,
        'node_count': len(nodes), 'operators': [{'domain': d, 'op': op, 'count': count}
            for (d, op), count in sorted(counts.items())], 'nodes': nodes, 'initializers': initializers,
        'opsets': [{'domain': item.domain or 'ai.onnx', 'version': item.version} for item in model.opset_import],
        'inputs': [v.name for v in model.graph.input], 'outputs': [v.name for v in model.graph.output]}


class CudaCapture:
    def __init__(self):
        self.cuda = ctypes.CDLL('libcudart.so.12')
        self.nvtx = ctypes.CDLL('libnvToolsExt.so.1')
        self.nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]

    def start(self):
        if self.cuda.cudaProfilerStart() != 0: raise RuntimeError('cudaProfilerStart failed')

    def stop(self):
        if self.cuda.cudaProfilerStop() != 0: raise RuntimeError('cudaProfilerStop failed')

    def push(self, name): self.nvtx.nvtxRangePushA(name.encode())
    def pop(self): self.nvtx.nvtxRangePop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, help='Whole-model ONNX override, e.g. a quantized checkpoint')
    parser.add_argument('--variant', default='fp32')
    parser.add_argument('--case-id')
    parser.add_argument('--provider', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--optimization-level', choices=['all', 'disabled'], default='all')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--external-capture', action='store_true', help='Separate instrumented Nsight trial, not ordinary latency')
    parser.add_argument('--compare-outputs', type=Path, help='Own-engine FP32 output NPZ; mismatched lengths remain visible')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.warmups < 0 or args.repeats < 1 or args.out.suffix != '.json':
        parser.error('positive threads/repeats, nonnegative warmups and .json output are required')
    if args.external_capture and args.provider != 'cuda':
        parser.error('external CUDA capture requires --provider cuda')
    import onnxruntime as ort
    provider = 'CUDAExecutionProvider' if args.provider == 'cuda' else 'CPUExecutionProvider'
    if provider not in ort.get_available_providers(): raise RuntimeError(f'{provider} is unavailable')
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    case = next((c for c in manifest['cases'] if args.case_id is None or c['id'] == args.case_id), None)
    if case is None: raise ValueError('Requested case is absent')
    model = (args.onnx or root / manifest['onnx']['path']).resolve()
    original_model = (root / manifest['onnx']['path']).resolve()
    inputs, reference = load_npz(root / case['inputs']), load_npz(root / case['reference'])
    own_reference = load_npz(args.compare_outputs) if args.compare_outputs else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'format': 'volvoxai-tts-ort-profile/v1', 'status': 'failed', 'variant': args.variant,
        'model': manifest['model'], 'case': case['id'], 'manifest_sha256': sha256(args.manifest),
        'caller_sha256': sha256(Path(__file__)), 'model_artifacts': {
            'executed': {'sha256': sha256(model), 'bytes': model.stat().st_size},
            'original_fp32': {'sha256': sha256(original_model), 'bytes': original_model.stat().st_size},
            'inputs_sha256': sha256(root / case['inputs']), 'reference_sha256': sha256(root / case['reference'])},
        'settings': {'threads': args.threads, 'warmups': args.warmups, 'repeats': args.repeats,
            'optimization_level': args.optimization_level, 'tf32': False, 'external_capture': args.external_capture},
        'runtime': {'ort_version': ort.__version__, 'providers_available': ort.get_available_providers(),
            'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(),
            'libraries': {p.name: sha256(p) for p in sorted(Path(ort.__file__).parent.glob('capi/*.so*'))}},
        'memory': {'before_session': memory_observation()}}
    if own_reference is not None: report['own_fp32_reference_sha256'] = sha256(args.compare_outputs)
    options = {'use_tf32': '0'} if args.provider == 'cuda' else {}
    providers = [(provider, options)] + (['CPUExecutionProvider'] if args.provider == 'cuda' else [])

    def session(profile=False, optimized=None, prefix=None):
        config = ort.SessionOptions()
        config.intra_op_num_threads = config.inter_op_num_threads = args.threads
        config.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        config.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if
            args.optimization_level == 'all' else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        config.enable_profiling = profile
        if optimized is not None: config.optimized_model_filepath = str(optimized)
        if prefix is not None: config.profile_file_prefix = str(prefix)
        started = time.perf_counter_ns()
        owner = ort.InferenceSession(str(model), config, providers=providers)
        owner.disable_fallback()
        if provider not in owner.get_providers():
            raise RuntimeError(f'Requested provider failed to register: {owner.get_providers()}')
        setup_ms = (time.perf_counter_ns() - started) / 1e6
        names = {s.name: s.type for s in owner.get_inputs()}
        types = {'tensor(float)': np.float32, 'tensor(double)': np.float64,
                 'tensor(int64)': np.int64, 'tensor(int32)': np.int32, 'tensor(bool)': np.bool_}
        if set(names) != set(manifest['onnx']['inputs'].values()):
            raise ValueError('Whole-model override changes public input names')
        feed = {name: checked_cast(inputs[key], types[names[name]]) for key, name in manifest['onnx']['inputs'].items()}
        return owner, feed, setup_ms

    def run_session(owner, feed, external=False):
        values, evidence, capture = [], [], CudaCapture() if external else None
        active = False
        try:
            for index in range(1 + args.warmups + args.repeats):
                measured = index > args.warmups
                if capture and measured and not active: capture.start(); active = True
                if capture and measured: capture.push(f'tts-measured-run-{index-args.warmups-1}')
                try:
                    start = time.perf_counter_ns()
                    outputs = owner.run(list(manifest['onnx']['outputs'].values()), feed)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                finally:
                    if capture and measured: capture.pop()
                actual = dict(zip(manifest['onnx']['outputs'], outputs))
                comparison = validate_outputs(actual, reference, case.get('validation', {}).get('waveform'))
                if any(not v['numerical_parity'] for v in comparison.values()): report['numerical_difference_seen'] = True
                evidence.append({'index': index, 'kind': 'first' if index == 0 else 'warmup' if not measured else 'measurement',
                    'output_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in actual.items()},
                    'comparison_to_original': comparison,
                    **({'comparison_to_own_fp32': validate_outputs(actual, own_reference,
                        case.get('validation', {}).get('waveform'))} if own_reference is not None else {})})
                values.append(elapsed)
            return actual, values, evidence
        finally:
            if active: capture.stop()

    try:
        owner, feed, setup = session()
        registered = {'registered_providers': owner.get_providers(), 'provider_options': owner.get_provider_options()}
        report['runtime'].update(registered)
        actual, elapsed, evidence = run_session(owner, feed, args.external_capture)
        report['benchmark'] = {'setup_ms': setup, 'first_ms': elapsed[0],
            'warmup_ms': elapsed[1:1+args.warmups], 'request_ms': summarize(elapsed[1+args.warmups:]),
            'instrumented_external': args.external_capture, 'requests': evidence,
            'unique_output_hashes': {name: sorted({row['output_sha256'][name] for row in evidence}) for name in actual},
            'maximum_errors_all_requests': {name: {key: max(row['comparison_to_original'][name].get(key, 0)
                for row in evidence) for key in ['max_absolute_error', 'rmse', 'strict_mismatches']}
                for name in actual if all('max_absolute_error' in row['comparison_to_original'][name] for row in evidence)},
            'scope': 'prepared host inputs through completed host outputs; loading, assertions and saving excluded'}
        report['validation'] = validate_outputs(actual, reference, case.get('validation', {}).get('waveform'))
        if own_reference is not None:
            report['comparison_to_own_fp32'] = validate_outputs(actual, own_reference, case.get('validation', {}).get('waveform'))
        audio = actual.get('audio')
        if audio is not None:
            report['audio'] = {'shape': list(audio.shape), 'samples': audio.size,
                'seconds': audio.shape[-1] / case.get('sample_rate', manifest['sample_rate']),
                'sha256': hashlib.sha256(audio.tobytes()).hexdigest()}
        np.savez(args.out.with_suffix('.outputs.npz'), **actual)
        report['memory']['after_benchmark'] = memory_observation()
        del owner, feed
        gc.collect()
        if not args.benchmark_only and not args.external_capture:
            with tempfile.TemporaryDirectory(prefix='ort-graph-', dir=args.out.parent) as temporary:
                optimized = Path(temporary) / 'optimized.onnx'
                owner, feed, setup = session(True, optimized, args.out.with_suffix('.ort'))
                if registered != {'registered_providers': owner.get_providers(), 'provider_options': owner.get_provider_options()}:
                    raise RuntimeError('Benchmark and profile session provider configurations differ')
                _, traced_elapsed, traced_evidence = run_session(owner, feed)
                trace_source = Path(owner.end_profiling())
                trace_path = args.out.with_suffix('.trace.json')
                trace_source.replace(trace_path)
                del owner, feed
                gc.collect()
                events = json.loads(trace_path.read_text())
                report['trace'] = {'raw_file': trace_path.name, 'sha256': sha256(trace_path),
                    'setup_ms': setup, 'instrumented_request_ms': traced_elapsed, 'requests': traced_evidence,
                    **summarize_trace(events, args.warmups, args.repeats)}
                placed = {row['provider'] for row in report['trace']['summaries']['providers']}
                if provider not in placed or not placed <= {provider, 'CPUExecutionProvider'}:
                    raise RuntimeError(f'Unexpected actual node placement: {placed}')
                report['memory']['after_profile'] = memory_observation()
                report['graphs'] = {'original': graph_inventory(model), 'optimized': graph_inventory(optimized),
                    'optimized_weights_retained': False}
                report['memory']['after_graph_inspection'] = memory_observation()
        report['status'] = 'numerical_difference' if report.get('numerical_difference_seen') else 'passed'
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'provider': provider, 'variant': args.variant,
                      'median_ms': report['benchmark']['request_ms']['median']}), flush=True)
    return 3 if report['status'] == 'numerical_difference' else 0


if __name__ == '__main__':
    raise SystemExit(main())
