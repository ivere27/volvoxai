#!/usr/bin/env python3
"""Compare native PackedLinear fixtures on generated or replayed model cases.

Build each library from packed_row_tiles.c and test_packed_linear_i8.c with
-shared -fPIC -O3 -ffp-contract=off and the matching private tile defines.
Pass --variant NAME=/absolute/path/library.so repeatedly, baseline first.
Use repeated --graph FILE to derive workload shapes; no graphs generates only
boundary/control cases. --cases FILE replays a saved suite or benchmark report.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

if __package__:
    from .packed_linear_cases import load_case_suite
else:
    from packed_linear_cases import load_case_suite

ROOT = Path(__file__).resolve().parents[1]
TESTS = ['test_packed_linear_i3_unpack', 'test_packed_linear_i8', 'test_packed_linear_i8_tiles',
         'test_packed_linear_exact_bounds', 'test_packed_linear_lowbit_arithmetic', 'test_packed_linear_row_tiles']


def bind_fixture(path):
    lib = ctypes.CDLL(str(path))
    lib.tile_bench_output_ptr.restype = ctypes.c_void_p
    for name in ('tile_bench_output_bytes', 'tile_bench_scratch_bytes', 'tile_bench_hash'):
        getattr(lib, name).restype = ctypes.c_uint32
    lib.tile_bench_init.argtypes = [ctypes.c_int] * 5
    lib.tile_bench_run.argtypes = [ctypes.c_int]
    modern = hasattr(lib, 'tile_bench_required_bytes')
    if modern:
        for name in ('tile_bench_required_bytes', 'tile_bench_memory_limit', 'tile_bench_all_bits', 'tile_bench_release'):
            if not hasattr(lib, name):
                raise ValueError('Incomplete bounded-allocation fixture API: ' + str(path))
        lib.tile_bench_required_bytes.argtypes = [ctypes.c_int] * 5
        lib.tile_bench_required_bytes.restype = ctypes.c_uint32
        lib.tile_bench_memory_limit.restype = ctypes.c_uint32
        lib.tile_bench_release.restype = None
    return lib, modern


def admit_cases(lib, modern, cases, name):
    """Reject unsupported shapes before any timed sample; never drop a case."""
    largest = 0
    for case in cases:
        shape = tuple(case[key] for key in ('rows', 'k', 'n', 'bits', 'group'))
        if any(type(value) is not int or not 0 < value <= 2147483647 for value in shape):
            raise ValueError(f'{name}: dimensions exceed the fixture integer ABI: {shape}')
        if modern:
            required = lib.tile_bench_required_bytes(*shape)
            if not required or required > lib.tile_bench_memory_limit():
                raise ValueError(f'{name}: shape {shape} exceeds the {lib.tile_bench_memory_limit()}-byte fixture budget')
            largest = max(largest, required)
        elif case['rows'] > 1608 or case['k'] > 1283 or case['n'] > 1536:
            raise ValueError(f'{name}: shape {shape} exceeds legacy fixture limits M1608/K1283/N1536; rebuild the fixture')
    return ({'storage': 'per-case-bounded-allocation', 'memory_limit_bytes': lib.tile_bench_memory_limit(),
             'max_case_required_bytes': largest} if modern else
            {'storage': 'legacy-static-v1', 'limits': {'rows': 1608, 'k': 1283, 'n': 1536}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--work-per-sample', type=int, default=100_000_000)
    parser.add_argument('--graph', type=Path, action='append', default=[])
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--family', action='append', default=[])
    parser.add_argument('--no-boundaries', action='store_true')
    parser.add_argument('--dim', action='append', default=[], metavar='NAME=VALUE')
    args = parser.parse_args()
    if args.cycles < 1 or args.work_per_sample < 1 or args.out.exists():
        raise ValueError('Positive cycles/work and a new output file are required')
    dimensions = {}
    for item in args.dim:
        name, value = item.split('=', 1)
        if name in dimensions:
            raise ValueError('Repeated dimension binding: ' + name)
        dimensions[name] = int(value)
    families = [name for value in args.family for name in value.split(',')]
    loader = Path(__file__).with_name('packed_linear_cases.py')
    loader_hash = hashlib.sha256(loader.read_bytes()).hexdigest()
    suite = load_case_suite(args.graph, cases_path=args.cases, families=families,
                            boundaries=not args.no_boundaries, dimensions=dimensions)
    if hashlib.sha256(loader.read_bytes()).hexdigest() != loader_hash:
        raise ValueError('Case loader changed during extraction')
    cases = suite['cases']
    variants, provenance = {}, {}
    for item in args.variant:
        name, raw = item.split('=', 1)
        path = Path(raw).resolve()
        if not name or name in variants:
            raise ValueError('Empty or repeated variant')
        lib, modern = bind_fixture(path)
        variants[name] = (lib, modern)
        provenance[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'rows': lib.tile_bench_rows(), 'tile2': lib.tile_bench_two(),
                            'all_bits': lib.tile_bench_all_bits() if modern else None}
    report = {'format': 'volvoxai-packed-linear-tiles/v2', 'status': 'running', 'backend': 'native',
              'scope': 'Standalone dynamic-I8 kernel; synthetic tensors derived from shapes; allocation/dispatch/model I/O excluded',
              'cpu_threads': 1, 'cycles': args.cycles, 'work_per_sample': args.work_per_sample,
              'warmups': 2, 'min_sample_ms': 1.0, 'max_repeats': 10000, 'order': 'Forward/reverse variant order each cycle', 'host_quiet_verified': False,
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'case_loader_sha256': loader_hash,
              'case_suite': suite, 'variants': provenance, 'admission': {}, 'tests': {}, 'samples': []}
    try:
        for name, (lib, modern) in variants.items():
            report['admission'][name] = admit_cases(lib, modern, cases, name)
            checks = {test: getattr(lib, test)() for test in TESTS}
            report['tests'][name] = checks
            if any(checks.values()):
                raise RuntimeError(f'{name}: numerical tests failed {checks}')
        for case in cases:
            rows, k, n, bits, group = (case[key] for key in ('rows', 'k', 'n', 'bits', 'group'))
            times = {name: [] for name in variants}
            reference, hashes, scratch = None, {}, {}
            repeats = max(1, min(1000, args.work_per_sample // (rows * k * n)))
            for name, (lib, _) in variants.items():
                if lib.tile_bench_init(rows, k, n, bits, group) or lib.tile_bench_run(2):
                    raise RuntimeError(f'{name}: setup failed for {case}')
                raw = ctypes.string_at(lib.tile_bench_output_ptr(), lib.tile_bench_output_bytes())
                if reference is None:
                    reference = raw
                if raw != reference or lib.tile_bench_check_scratch():
                    raise RuntimeError(f'Output/scratch mismatch {name}')
                hashes[name] = lib.tile_bench_hash()
                scratch[name] = lib.tile_bench_scratch_bytes()
            def run_timed(name, count):
                began = time.perf_counter_ns()
                code = variants[name][0].tile_bench_run(count)
                elapsed = (time.perf_counter_ns() - began) / 1e6
                if code or not math.isfinite(elapsed) or elapsed < 0:
                    raise RuntimeError(f'{name}: kernel failed or invalid timer')
                return elapsed

            # Use one common repeat count so small boundary cases remain
            # measurable. Calibration is outside the measured cycles.
            calibration = []
            while True:
                elapsed = {name: run_timed(name, repeats) for name in variants}
                calibration.append({'repeats': repeats, 'elapsed_ms': elapsed})
                fastest = min(elapsed.values())
                if fastest >= report['min_sample_ms'] or repeats == report['max_repeats']:
                    if fastest <= 0:
                        raise RuntimeError('Timer did not advance at the fixture repeat limit')
                    break
                repeats = min(report['max_repeats'], max(repeats * 2,
                    math.ceil(repeats * report['min_sample_ms'] / fastest * 1.1) if fastest > 0 else repeats * 10))
            for cycle in range(args.cycles):
                for name in list(variants)[::(-1 if cycle % 2 else 1)]:
                    elapsed = run_timed(name, repeats)
                    if elapsed <= 0:
                        raise RuntimeError(f'{name}: timer did not advance during a measured sample')
                    times[name].append(elapsed / repeats)
            for name, (lib, _) in variants.items():
                raw = ctypes.string_at(lib.tile_bench_output_ptr(), lib.tile_bench_output_bytes())
                if raw != reference or lib.tile_bench_check_scratch():
                    raise RuntimeError(f'{name}: repeated output or scratch changed')
            medians = {name: statistics.median(value) for name, value in times.items()}
            baseline = next(iter(medians.values()))
            report['samples'].append({**case, 'repeats': repeats, 'calibration': calibration, 'timing_ms': times, 'median_ms': medians,
                                      'speedup': {name: baseline / value for name, value in medians.items()},
                                      'scratch_required_bytes': scratch, 'output_fnv1a32': hashes,
                                      'all_outputs_byte_equal': True})
            print(f'TILE_NATIVE {case["family"]} {rows}x{k}x{n} W{bits}/G{group}', flush=True)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        for lib, modern in variants.values():
            if modern:
                lib.tile_bench_release()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
