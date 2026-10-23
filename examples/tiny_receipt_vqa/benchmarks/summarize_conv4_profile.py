#!/usr/bin/env python3
"""Validate and summarize the Conv4/PackedLinear receipt comparison matrix.

Reads before/runtime/after cases 0 and 1, plus controls case 0. It does not
run inference or alter source measurements. Storage controls are checked from
actual package bytes, independently of the collector's conversion functions.
"""
import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from summarize_extension_ptq_profile import item_summary, read, sha


MODELS = ('digit', 'vqa')
BACKENDS = ('native', 'chrome')
MATRIX = tuple((label, case) for case in (0, 1)
               for label in ('before', 'runtime', 'after')) + (('controls', 0),)
VARIANTS = {'before': ('w8a8', 'w4a8'), 'runtime': ('w4a8',),
            'after': ('w8a8', 'w4a8', 'w4a8-conv4'),
            'controls': ('w4-codes-i8', 'conv4-codes-i8')}
EXPAND_LINEAR = tuple(bytes(((byte & 15) + 120, (byte >> 4) + 120)) for byte in range(256))
EXPAND_CONV = tuple(bytes((((byte & 15) - 8) & 255, ((byte >> 4) - 8) & 255)) for byte in range(256))


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=None)
def package(path):
    path = Path(path)
    graph_bytes = (path / 'graph.json').read_bytes()
    data = (path / 'model.safetensors').read_bytes()
    require(len(data) >= 8, f'{path}: truncated safetensors')
    size = int.from_bytes(data[:8], 'little')
    require(8 + size <= len(data), f'{path}: truncated safetensors header')
    header = json.loads(data[8:8 + size])
    payload = memoryview(data)[8 + size:]
    tensors = {}
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        start, end = entry['data_offsets']
        require(0 <= start <= end <= len(payload), f'{path}: invalid {name} span')
        raw = payload[start:end]
        tensors[name] = {'dtype': entry['dtype'], 'shape': entry['shape'],
                         'raw': raw, 'sha256': hashlib.sha256(raw).hexdigest()}
    return {'graph': json.loads(graph_bytes), 'tensors': tensors,
            'package_bytes': len(graph_bytes) + len(data),
            'sha256': {'graph.json': hashlib.sha256(graph_bytes).hexdigest(),
                       'model.safetensors': hashlib.sha256(data).hexdigest()}}


def descriptor(tensor):
    return {key: tensor[key] for key in ('dtype', 'shape', 'sha256')}


def tensor_shapes(graph):
    shapes = {name: item['shape'] for name, item in graph['inputs'].items()}
    for node in graph['nodes']:
        shapes.update({item['tensor']: item['shape'] for item in node['outputs'].values()})
    return shapes


def verify_unchanged_tensors(a, b, replaced):
    require(a.keys() == b.keys(), 'storage control changed tensor inventory')
    for name in a.keys() - replaced:
        require(descriptor(a[name]) == descriptor(b[name]),
                f'storage control changed unrelated tensor/affine {name}')


@lru_cache(maxsize=None)
def validate_linear_storage(base_path, control_path):
    """Only I4 weight storage becomes I8; every scale and other tensor stays exact."""
    base, control = package(str(base_path)), package(str(control_path))
    graph = deepcopy(base['graph'])
    shapes = tensor_shapes(graph)
    a, b = base['tensors'], control['tensors']
    converted = set()
    nodes = 0
    for node in graph['nodes']:
        if node['opType'] != 'PackedLinear':
            continue
        require(node['params']['weight_bits'] == 4, 'Linear baseline is not I4')
        node['params']['weight_bits'] = 8
        name = node['inputs']['weight']
        width = shapes[node['inputs']['input']][-1]
        require(isinstance(width, int) and width > 0, 'Linear K must be fixed')
        nodes += 1
        x, y = a[name], b[name]
        require(x['dtype'] == y['dtype'] == 'I32' and len(x['shape']) == 2,
                f'{name}: wrong PackedLinear physical dtype/rank')
        rows = x['shape'][0]
        old_row, new_row = 4 * ((width + 7) // 8), 4 * ((width + 3) // 4)
        require(x['shape'] == [rows, old_row // 4] and y['shape'] == [rows, new_row // 4],
                f'{name}: wrong PackedLinear storage shape')
        require(len(x['raw']) == rows * old_row and len(y['raw']) == rows * new_row,
                f'{name}: wrong PackedLinear storage byte count')
        if name not in converted:
            for row in range(rows):
                codes = b''.join(EXPAND_LINEAR[value] for value in x['raw'][row * old_row:(row + 1) * old_row])
                require(codes[width:] == bytes([128]) * (len(codes) - width),
                        f'{name}: noncanonical I4 padding')
                expected = codes[:width] + bytes([128]) * (new_row - width)
                require(y['raw'][row * new_row:(row + 1) * new_row] == expected,
                        f'{name}: I8 storage changed logical Linear codes')
            converted.add(name)
    require(graph == control['graph'], 'Linear control changed graph beyond weight_bits')
    verify_unchanged_tensors(a, b, converted)
    return {'converted_nodes': nodes, 'converted_weight_banks': len(converted),
            'logical_codes_equal': True, 'canonical_padding': True,
            'other_graph_fields_identical': True, 'all_scales_bias_and_other_tensors_identical': True}


@lru_cache(maxsize=None)
def validate_conv_storage(base_path, control_path):
    """Only Conv4 codes become signed I8 OHWI and their offset zero points become 0."""
    base, control = package(str(base_path)), package(str(control_path))
    graph = deepcopy(base['graph'])
    shapes = tensor_shapes(graph)
    a, b = base['tensors'], control['tensors']
    converted, zero_banks = set(), set()
    nodes = 0
    for node in graph['nodes']:
        if node['opType'] != 'QConv2D' or node['params'].get('weight_bits', 8) != 4:
            continue
        kh, kw = node['params']['kernel_size']
        groups = node['params'].get('groups', 1)
        channels = shapes[node['inputs']['input']][-1]
        require(isinstance(channels, int) and channels > 0 and groups > 0 and channels % groups == 0,
                'Conv input channels/groups must be fixed and divisible')
        width = kh * kw * (channels // groups)
        name = node['inputs']['weight']
        zero_name = graph['quantization']['tensors'][name]['zero_point_tensor']
        x, y = a[name], b[name]
        require(x['dtype'] == 'U8' and y['dtype'] == 'I8' and len(x['shape']) == 2,
                f'{name}: wrong Conv storage dtype/rank')
        rows, row_bytes = x['shape']
        require(row_bytes == (width + 1) // 2 and y['shape'] == [rows, kh, kw, channels // groups],
                f'{name}: wrong Conv storage shape')
        require(len(x['raw']) == rows * row_bytes and len(y['raw']) == rows * width,
                f'{name}: wrong Conv storage byte count')
        if name not in converted:
            for row in range(rows):
                codes = b''.join(EXPAND_CONV[value] for value in x['raw'][row * row_bytes:(row + 1) * row_bytes])
                require(codes[width:] == bytes(len(codes) - width), f'{name}: nonzero unused I4 nibble')
                require(y['raw'][row * width:(row + 1) * width] == codes[:width],
                        f'{name}: I8 storage changed logical Conv codes')
            converted.add(name)
        z0, z1 = a[zero_name], b[zero_name]
        require(z0['dtype'] == 'U8' and z1['dtype'] == 'I8' and z0['shape'] == z1['shape'] == [rows],
                f'{zero_name}: wrong Conv zero-point descriptor')
        require(z0['raw'] == bytes([8]) * rows and z1['raw'] == bytes(rows),
                f'{zero_name}: Conv offset-signed transform is not exact')
        zero_banks.add(zero_name)
        del node['params']['weight_bits'], node['params']['kernel_size']
        nodes += 1
    require(graph == control['graph'], 'Conv control changed graph beyond storage parameters')
    for name, affine in graph.get('quantization', {}).get('tensors', {}).items():
        if name not in converted:
            require(affine.get('scale_tensor') not in zero_banks and affine.get('zero_point_tensor') not in zero_banks,
                    f'Conv zero-point bank is shared with unrelated affine {name}')
    verify_unchanged_tensors(a, b, converted | zero_banks)
    return {'converted_nodes': nodes, 'converted_weight_banks': len(converted),
            'converted_zero_point_banks': len(zero_banks), 'logical_codes_equal': True,
            'canonical_padding': True, 'other_graph_fields_identical': True,
            'all_scales_bias_and_other_tensors_identical': True}


def input_identity(spec, model):
    return [(entry['name'], entry['shape'], entry['dtype'], entry['sha256'])
            for entry in spec['inputs'][model]]


def model_identity(spec, model, variant):
    row = spec['models'][model][variant]
    return {'arithmetic': row['arithmetic'], 'roles': {
        role: {'inputs': entry['inputs'], 'sha256': entry['inventory']['sha256']}
        for role, entry in row['roles'].items()}}


def compare(a, b):
    first, second = a['warm_total']['median_ms'], b['warm_total']['median_ms']
    return {'before_ms': first, 'after_ms': second, 'speedup': first / second,
            'before_bytes': a['package_bytes'], 'after_bytes': b['package_bytes'],
            'before_correct': a['correct'], 'after_correct': b['correct']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--task-jobs-paused', action='store_true')
    parser.add_argument('--evidence', type=Path, action='append', default=[],
                        help='Bind an existing validation/codegen/release report by path and SHA256')
    args = parser.parse_args()
    root = args.directory.resolve()
    summary = {
        'format': 'receipt-conv4-profile-comparison/v1', 'host_quiet_verified': False,
        'task_owned_jobs_paused_during_measurements': args.task_jobs_paused, 'backend_threads': 1,
        'scope': 'Two public example inputs, first VQA question per input; storage controls use case 0. Accuracy qualification of the full heldout set is separate.',
        'timing': 'Fresh host/context setup and first request separate; 2 warmups then 5 samples. Native CPU one thread or isolated Chrome WASM; no extension UI, tabs or shared-worker queue.',
        'limitations': [
            'Desktop/background activity is not excluded; whole-host quiet is unverified.',
            'Cold means a fresh host/context and its first request; operating-system file caches are not flushed.',
            'Native includes caller-owned NumPy/result copies; Chrome includes inline protobuf transport.',
            'p95 is nearest rank and equals the maximum of five samples; this does not establish production-tail latency.',
            'Node/allocator/API probes are separate from warm timing. PackedLinear includes activation preparation, integer dots and group scaling.',
            'Allocator coverage is partial; absent events do not imply allocation-free execution.',
            'Lossless storage controls use identical numerical codes and scales; they do not measure a separately fitted INT8 model.',
            'Regenerated-model comparisons may include recipe and calibration changes; only runtime-only and byte-verified storage controls isolate their stated changes.',
        ], 'runs': {}, 'primary': [], 'current_modes': [], 'controls': {},
        'artifact_sha256': {}, 'runtime_hashes': {}, 'input_hashes': {}, 'model_hashes': {},
        'validation_evidence': {str(path): sha(path) for path in args.evidence},
    }
    raw, specs = {}, {}
    expected_affinity, expected_browser = None, None
    for label, case in MATRIX:
        directory = root / f'{label}-case{case}'
        spec_path = directory / 'spec.json'
        spec = specs[label, case] = read(spec_path)
        require(spec['label'] == label and spec['case_index'] == case, 'spec label/case mismatch')
        require(spec['repeat'] == 5 and spec['warmup'] == 2 and spec['feedback'], 'matrix measurement policy changed')
        summary['artifact_sha256'][str(spec_path.relative_to(root))] = sha(spec_path)
        summary['input_hashes'][f'{label}/case{case}'] = spec['inputs']
        for model in MODELS:
            for entry in spec['inputs'][model]:
                require(sha(Path(entry['file'])) == entry['sha256'], f'input bytes changed: {entry["file"]}')
            require(set(spec['models'][model]) == set(VARIANTS[label]), 'unexpected variant inventory')
            for variant, contract in spec['models'][model].items():
                summary['model_hashes'][f'{label}/case{case}/{model}/{variant}'] = model_identity(spec, model, variant)
                for role in contract['roles'].values():
                    actual = package(role['path'])
                    require(actual['sha256'] == role['inventory']['sha256'], f'package changed since measurement: {role["path"]}')
                    require(actual['package_bytes'] == role['inventory']['package_bytes'], 'package size mismatch')
        host = directory / 'host.txt'
        if host.is_file():
            summary.setdefault('host_observations', {})[f'{label}/case{case}'] = host.read_text()
            summary['artifact_sha256'][str(host.relative_to(root))] = sha(host)
        for backend in BACKENDS:
            path = directory / f'{label}-{backend}.json'
            report = raw[label, case, backend] = read(path)
            require(report['label'] == label and report['case_index'] == case and report['feedback'], 'report label/case/feedback mismatch')
            require(report['cpu_threads'] == 1, 'report thread count changed')
            if backend == 'native':
                require(report['spec_sha256'] == sha(spec_path), 'native report does not match its spec')
                require(report.get('cpu_isa_override') is None, 'ISA override invalidates the ordinary-runtime comparison')
                expected_affinity = report['affinity'] if expected_affinity is None else expected_affinity
                require(expected_affinity == report['affinity'], 'native CPU affinity changed')
                runtime = report['library_sha256']
            else:
                expected_browser = report['browser'] if expected_browser is None else expected_browser
                require(expected_browser == report['browser'], 'Chrome version changed')
                runtime = report['runtime_sha256']
            runtime_key = f'{label}/{backend}'
            require(runtime_key not in summary['runtime_hashes'] or summary['runtime_hashes'][runtime_key] == runtime,
                    'runtime changed between public cases')
            summary['runtime_hashes'][runtime_key] = runtime
            summary['artifact_sha256'][str(path.relative_to(root))] = sha(path)
            rows = summary['runs'][f'{label}/case{case}/{backend}'] = {}
            for model in MODELS:
                require(set(report['models'][model]) == set(VARIANTS[label]), 'report variant inventory mismatch')
                rows[model] = {}
                for variant, item in report['models'][model].items():
                    require(len(item['warm']) == 5, 'wrong number of warm samples')
                    record = rows[model][variant] = item_summary(model, item, backend)
                    roles = spec['models'][model][variant]['roles']
                    record['storage'] = {role: entry['inventory'] for role, entry in roles.items()}
                    record['package_bytes'] = sum(entry['inventory']['package_bytes'] for entry in roles.values())
                    record['output_sha256'] = item['output_sha256']
                    record['arithmetic'] = spec['models'][model][variant]['arithmetic']
    summary['cpu_affinity'], summary['chrome_version'] = expected_affinity, expected_browser
    for label in ('before', 'runtime', 'after'):
        for model in MODELS:
            for variant in VARIANTS[label]:
                require(model_identity(specs[label, 0], model, variant) == model_identity(specs[label, 1], model, variant),
                        f'{label}/{model}/{variant}: model changed between cases')
    for label, case in MATRIX:
        for model in MODELS:
            require(input_identity(specs[label, case], model) == input_identity(specs['before', case], model),
                    f'{label}/case{case}/{model}: input changed')
            require(specs[label, case]['expected'][model] == specs['before', case]['expected'][model], 'expected answer changed')
        require(specs[label, case].get('question') == specs['before', case].get('question'), 'VQA question changed')
    for backend in BACKENDS:
        for label in ('after', 'controls'):
            require(summary['runtime_hashes'][f'{label}/{backend}'] == summary['runtime_hashes'][f'runtime/{backend}'],
                    'new-runtime measurements used different binaries')
        controls = summary['controls'][backend] = {'runtime_only': [], 'regenerated_w4': [], 'storage': {}}
        for case in (0, 1):
            before = summary['runs'][f'before/case{case}/{backend}']
            runtime = summary['runs'][f'runtime/case{case}/{backend}']
            after = summary['runs'][f'after/case{case}/{backend}']
            for model in MODELS:
                require(model_identity(specs['before', case], model, 'w4a8') == model_identity(specs['runtime', case], model, 'w4a8'),
                        'runtime-only control changed model/arithmetic/bindings')
                a, b = raw['before', case, backend]['models'][model]['w4a8'], raw['runtime', case, backend]['models'][model]['w4a8']
                require(a['output_sha256'] == b['output_sha256'] and a['cold']['signature'] == b['cold']['signature'],
                        'runtime-only control changed numerical outputs')
                controls['runtime_only'].append({'case': case, 'model': model, 'input_model_and_output_bytes_equal': True,
                    **compare(before[model]['w4a8'], runtime[model]['w4a8'])})
                controls['regenerated_w4'].append({'case': case, 'model': model,
                    'model_bytes_equal': model_identity(specs['runtime', case], model, 'w4a8') == model_identity(specs['after', case], model, 'w4a8'),
                    'output_bytes_equal': runtime[model]['w4a8']['output_sha256'] == after[model]['w4a8']['output_sha256'],
                    **compare(runtime[model]['w4a8'], after[model]['w4a8'])})
                for variant in VARIANTS['after']:
                    baseline = 'w4a8' if variant == 'w4a8-conv4' else variant
                    summary['primary'].append({'case': case, 'backend': backend, 'model': model,
                        'baseline_variant': baseline, 'candidate_variant': variant,
                        **compare(before[model][baseline], after[model][variant])})
                    summary['current_modes'].append({'case': case, 'backend': backend, 'model': model,
                        'baseline_variant': 'w8a8', 'candidate_variant': variant,
                        **compare(after[model]['w8a8'], after[model][variant])})
        for model in MODELS:
            controls['storage'][model] = {}
            for variant, baseline, validator in (
                ('w4-codes-i8', 'w4a8', validate_linear_storage),
                ('conv4-codes-i8', 'w4a8-conv4', validate_conv_storage),
            ):
                base = raw['after', 0, backend]['models'][model][baseline]
                candidate = raw['controls', 0, backend]['models'][model][variant]
                require(base['output_sha256'] == candidate['output_sha256'] and base['cold']['signature'] == candidate['cold']['signature'],
                        f'{variant}: lossless storage changed outputs')
                source = specs['after', 0]['models'][model][baseline]
                target = specs['controls', 0]['models'][model][variant]
                require(source['arithmetic'] == target['arithmetic'] == 'dynamic-i8', 'storage control changed arithmetic')
                require(source['roles'].keys() == target['roles'].keys(), 'storage control changed roles')
                checks = {}
                for role, contract in source['roles'].items():
                    require(contract['inputs'] == target['roles'][role]['inputs'], 'storage control changed input binding')
                    checks[role] = validator(contract['path'], target['roles'][role]['path'])
                require(sum(check['converted_nodes'] for check in checks.values()) > 0, 'empty storage control')
                controls['storage'][model][variant] = {'baseline_variant': baseline, 'output_bytes_equal': True,
                    'signature_equal': True, 'storage_checks': checks,
                    **compare(summary['runs'][f'after/case0/{backend}'][model][baseline],
                              summary['runs'][f'controls/case0/{backend}'][model][variant])}
    caller_names = ('extension_ptq_profile.py', 'extension_ptq_profile.mjs',
                    'summarize_extension_ptq_profile.py', Path(__file__).name)
    summary['caller_sha256'] = {name: sha(Path(__file__).with_name(name)) for name in caller_names}
    summary['timing_caller_sha256'] = {name: sha(root / 'timing-callers' / name) for name in caller_names
        if (root / 'timing-callers' / name).is_file()}
    summary['timing_caller_snapshot_present'] = all(name in summary['timing_caller_sha256'] for name in caller_names[:2])
    repository = Path(__file__).resolve().parents[3]
    contract_paths = ('proto/volvoxai.proto', 'proto/operator_param_registry.proto',
        'native/src/generated/operator_param_ids.h', 'native/src/generated/operator_param_registry.h',
        'native/src/generated/operator_param_registry_full.h', 'tools/generated/operatorParamRegistry.ts',
        'tools/generated/operatorParamRegistryFull.ts', 'tools/exporter/generated/operator_param_registry.py',
        'tools/exporter/generated/operator_param_registry_full.py')
    summary['codegen_scope'] = {
        'description': 'Existing public PTQ lifecycle; Conv4 weight_bits/kernel_size extend QConv2D parameter vocabulary and generated registries. Hashes describe source projections at summarization time; validation evidence is separate.',
        'current_source_sha256': {name: sha(repository / name) for name in contract_paths},
        'validation_claim': 'No generation/check success is inferred from these hashes. See bound validation_evidence reports.',
    }
    temporary = root / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(root / 'summary.json')
    for row in summary['primary']:
        print(row)


if __name__ == '__main__':
    main()
