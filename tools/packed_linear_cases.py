#!/usr/bin/env python3
"""Generate reproducible PackedLinear benchmark workloads from graph metadata.

The benchmark always requests dynamic-I8 arithmetic; model weights and input
values are not loaded. Occurrences count graph nodes, not autoregressive steps.
Symbolic dimensions need explicit --dim bindings (or equal declared bounds).
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

FORMAT = 'volvoxai-packed-linear-cases/v2'
GENERATOR_VERSION = 1
TILES = (4, 8, 16, 32, 64)
GROUPS = (32, 64, 128, 256)
BITS = (2, 3, 4, 8)
INT_MAX = (1 << 31) - 1
FAMILIES = ('model', 'row_boundaries', 'tails', 'bit_controls', 'group_controls')
CASE_KEYS = {'family', 'rows', 'k', 'n', 'bits', 'group', 'arithmetic', 'occurrences', 'origins'}
SUITE_KEYS = {'format', 'generator', 'sources', 'dimensions', 'boundaries', 'boundary_rule',
              'cases', 'cases_sha256', 'suite_sha256'}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label, *, minimum=1):
    _require(type(value) is int and minimum <= value <= INT_MAX,
             f'{label} must be an integer in {minimum}..{INT_MAX}')
    return value


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _key(case):
    return tuple(case[name] for name in ('family', 'rows', 'k', 'n', 'bits', 'group'))


def _case(family, rows, k, n, bits=4, group=32):
    return dict(family=family, rows=rows, k=k, n=n, bits=bits, group=group,
                arithmetic='dynamic-i8', occurrences=0, origins=[])


def generated_boundaries():
    """Small generated controls, independent of any receipt/model dimensions."""
    cases = []
    rows = {1, 2, *(tile + offset for tile in TILES for offset in (-1, 0, 1, 2, 3))}
    cases.extend(_case('row_boundaries', row, 2 * GROUPS[0], 8) for row in sorted(rows))
    # Include full four-column panels and both sides of their eligibility edge.
    for row in (2, 5):
        for k in (GROUPS[0] - 1, GROUPS[0], GROUPS[0] + 1):
            for n in (3, 4, 5):
                cases.append(_case('tails', row, k, n))
    for bits in BITS:
        for row in (2, 8):
            cases.append(_case('bit_controls', row, 2 * GROUPS[0], 8, bits))
    for group in GROUPS:
        for k in (group - 1, group, group + 1):
            for row in (2, 17):
                cases.append(_case('group_controls', row, k, 8, group=group))
    return sorted(cases, key=_key)


def _constraint(name, constraints, value):
    _require(name in constraints, f'Undeclared symbolic dimension {name!r}')
    spec = constraints[name]
    _require(isinstance(spec, dict), f'Invalid dimension constraint for {name!r}')
    lo = _integer(spec.get('min'), f'{name}.min', minimum=0)
    hi = _integer(spec.get('max'), f'{name}.max', minimum=0)
    multiple = _integer(spec.get('multiple_of', 1), f'{name}.multiple_of')
    _require(lo <= hi, f'Inverted bounds for dimension {name!r}')
    if value is None:
        _require(lo == hi, f'Unresolved dimension {name!r}; supply --dim {name}=VALUE')
        value = lo
    value = _integer(value, f'Dimension {name}', minimum=0)
    _require(lo <= value <= hi and value % multiple == 0,
             f'Dimension {name}={value} violates its bounds or multiple_of')
    return value


def _axis(axis, constraints, bindings):
    if type(axis) is int:
        return _integer(axis, 'Shape axis')
    _require(isinstance(axis, str) and axis, 'Shape axes must be positive integers or dimension expressions')
    if axis in constraints:
        return _integer(_constraint(axis, constraints, bindings.get(axis)), 'Resolved shape axis')
    try:
        expression = ast.parse(axis, mode='eval').body
    except (SyntaxError, ValueError) as error:
        raise ValueError(f'Unsupported shape expression {axis!r}') from error

    def evaluate(node):
        if isinstance(node, ast.Name):
            return _constraint(node.id, constraints, bindings.get(node.id))
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return _integer(node.value, 'Expression constant', minimum=0)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                _require(right > 0, 'Shape expression divides by zero')
                if isinstance(node.op, ast.Div):
                    _require(left % right == 0, 'Shape expression division is not integral')
                value = left // right
            return _integer(value, 'Shape expression result', minimum=0)
        raise ValueError(f'Unsupported shape expression {axis!r}')

    return _integer(evaluate(expression), 'Resolved shape axis')


def _shape(descriptor, constraints, bindings, label):
    _require(isinstance(descriptor, dict), f'Missing tensor descriptor for {label}')
    raw = descriptor.get('shape')
    _require(isinstance(raw, list) and 1 <= len(raw) <= 8, f'{label} needs a rank-1..8 shape')
    _require(descriptor.get('dtype', 'float32') == 'float32', f'{label} must have float32 graph boundary dtype')
    return [_axis(axis, constraints, bindings) for axis in raw]


def _graph_cases(path, bindings):
    raw = path.read_bytes()
    graph_hash = hashlib.sha256(raw).hexdigest()
    graph = json.loads(raw)
    _require(isinstance(graph, dict) and graph.get('format') == 'volvox-graph/v1',
             f'{path.name}: expected a volvox-graph/v1 graph')
    constraints = graph.get('dimensions', {})
    _require(isinstance(constraints, dict), f'{path.name}: invalid dimension declarations')
    inputs, nodes = graph.get('inputs'), graph.get('nodes')
    _require(isinstance(inputs, dict) and isinstance(nodes, list), f'{path.name}: inputs/nodes are missing')
    tensors = dict(inputs)
    node_ids = set()
    for node in nodes:
        _require(isinstance(node, dict) and isinstance(node.get('id'), str) and node['id'], 'Graph node needs an id')
        _require(node['id'] not in node_ids, f'Duplicate node id {node["id"]!r}')
        node_ids.add(node['id'])
        outputs = node.get('outputs')
        _require(isinstance(outputs, dict), f'{node["id"]}: output descriptors are missing')
        for descriptor in outputs.values():
            _require(isinstance(descriptor, dict) and isinstance(descriptor.get('tensor'), str)
                     and descriptor['tensor'], f'{node["id"]}: invalid output tensor descriptor')
            name = descriptor['tensor']
            _require(name not in tensors, f'Duplicate tensor producer for {name!r}')
            tensors[name] = descriptor
    cases = []
    for node in nodes:
        if node.get('opType') != 'PackedLinear':
            continue
        try:
            source = node['inputs']['input']
            _require(source in tensors, f'Input tensor {source!r} has no shape descriptor')
            _require(len(node['outputs']) == 1, 'PackedLinear needs exactly one output')
            before = _shape(tensors[source], constraints, bindings, 'PackedLinear input')
            after = _shape(next(iter(node['outputs'].values())), constraints, bindings, 'PackedLinear output')
            _require(before[:-1] == after[:-1], 'PackedLinear input/output leading shapes differ')
            params = node.get('params', {})
            _require(isinstance(params, dict), 'Invalid PackedLinear parameters')
            bits, group = params.get('weight_bits', 4), params.get('group_size')
            _require(type(bits) is int and bits in BITS, 'Unsupported PackedLinear weight_bits')
            _require(type(group) is int and group in GROUPS, 'Missing or unsupported PackedLinear group_size')
            rows = _integer(math.prod(before[:-1]), 'Flattened PackedLinear rows')
            case = _case('model', rows, before[-1], after[-1], bits, group)
            case['occurrences'] = 1
            case['origins'] = [{'source_sha256': graph_hash, 'node_id': node['id']}]
            cases.append(case)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{path.name}/{node["id"]}: {error}') from error
    _require(cases, f'{path.name}: no PackedLinear nodes found')
    for name, value in bindings.items():
        if name in constraints:
            _constraint(name, constraints, value)
    source = {'name': path.name, 'sha256': graph_hash, 'packed_nodes': len(cases)}
    return cases, source, set(constraints)


def _seal(suite):
    suite['cases_sha256'] = digest(suite['cases'])
    suite['suite_sha256'] = digest({key: value for key, value in suite.items() if key != 'suite_sha256'})
    return suite


def validate_case_suite(suite):
    """Validate a self-contained generated workload, including replay hashes."""
    _require(isinstance(suite, dict) and set(suite) == SUITE_KEYS and suite['format'] == FORMAT,
             'Expected a generated PackedLinear cases/v2 suite')
    generator = suite['generator']
    _require(isinstance(generator, dict) and set(generator) == {'version', 'sha256'}
             and type(generator['version']) is int and generator['version'] >= 1 and _hash(generator['sha256']),
             'Invalid generator identity')
    _require(type(suite['boundaries']) is bool and suite['boundary_rule'] == 'tile-group-neighbours/v1',
             'Invalid boundary generation policy')
    dimensions = suite['dimensions']
    _require(isinstance(dimensions, dict), 'Invalid stored dimension bindings')
    for name, value in dimensions.items():
        _require(isinstance(name, str) and re.fullmatch(r'[A-Za-z_]\w*', name), 'Invalid dimension name')
        _integer(value, 'Stored dimension binding', minimum=0)
    sources = suite['sources']
    _require(isinstance(sources, list), 'Missing graph identities')
    source_counts = {}
    for source in sources:
        _require(isinstance(source, dict) and set(source) == {'name', 'sha256', 'packed_nodes'}
                 and isinstance(source['name'], str) and source['name']
                 and '/' not in source['name'] and '\\' not in source['name']
                 and _hash(source['sha256']), 'Invalid graph source identity')
        _integer(source['packed_nodes'], 'Source PackedLinear count')
        _require(source['sha256'] not in source_counts, 'Repeated graph source')
        source_counts[source['sha256']] = source['packed_nodes']
    _require(sources == sorted(sources, key=lambda item: item['sha256']), 'Graph sources are not ordered')
    cases = suite['cases']
    _require(isinstance(cases, list) and cases, 'No benchmark cases selected')
    keys, all_origins, observed = set(), set(), Counter()
    for case in cases:
        _require(isinstance(case, dict) and set(case) == CASE_KEYS, 'Invalid benchmark case fields')
        _require(case['family'] in FAMILIES and case['arithmetic'] == 'dynamic-i8', 'Unsupported case family/arithmetic')
        for field in ('rows', 'k', 'n'):
            _integer(case[field], 'Case ' + field)
        _require(type(case['bits']) is int and case['bits'] in BITS
                 and type(case['group']) is int and case['group'] in GROUPS, 'Unsupported case weight format')
        _integer(case['occurrences'], 'Case occurrences', minimum=0)
        key = _key(case)
        _require(key not in keys, 'Repeated benchmark case')
        keys.add(key)
        origins = case['origins']
        _require(isinstance(origins, list), 'Invalid case origins')
        for origin in origins:
            _require(isinstance(origin, dict) and set(origin) == {'source_sha256', 'node_id'}
                     and _hash(origin['source_sha256']) and origin['source_sha256'] in source_counts
                     and isinstance(origin['node_id'], str) and origin['node_id'], 'Invalid graph node origin')
            identity = (origin['source_sha256'], origin['node_id'])
            _require(identity not in all_origins, 'Graph node counted more than once')
            all_origins.add(identity)
            observed[identity[0]] += 1
        _require(origins == sorted(origins, key=lambda item: (item['source_sha256'], item['node_id'])),
                 'Graph origins are not ordered')
        if case['family'] == 'model':
            _require(case['occurrences'] == len(origins) > 0, 'Model case occurrence count differs from origins')
        else:
            _require(suite['boundaries'] and case['occurrences'] == 0 and not origins,
                     'Synthetic boundary case has graph occurrences or boundaries are disabled')
    if any(case['family'] == 'model' for case in cases):
        _require(dict(observed) == source_counts, 'Graph PackedLinear node coverage is incomplete')
    _require(cases == sorted(cases, key=_key), 'Benchmark cases are not ordered')
    _require(suite['cases_sha256'] == digest(cases), 'Benchmark case hash differs')
    _require(suite['suite_sha256'] == digest({key: value for key, value in suite.items() if key != 'suite_sha256'}),
             'Benchmark suite hash differs')
    return suite


def load_case_suite(graph_paths=(), *, cases_path=None, families=(), boundaries=True, dimensions=None):
    """Return generated cases or a validated replay suite; never read weights."""
    names = set()
    for name in families:
        _require(isinstance(name, str) and name and all(name.split(',')), 'Empty case family')
        names.update(name.split(','))
    _require(names <= set(FAMILIES), 'Unknown case family: ' + ', '.join(sorted(names - set(FAMILIES))))
    bindings = dict(dimensions or {})
    for name, value in bindings.items():
        _require(isinstance(name, str) and re.fullmatch(r'[A-Za-z_]\w*', name), 'Invalid dimension binding name')
        _integer(value, f'Dimension {name}', minimum=0)
    paths = [Path(path) for path in graph_paths]
    if cases_path is not None:
        _require(not paths and not bindings and not names and boundaries,
                 '--cases replays an exact suite; do not combine it with graphs, dimensions, families or --no-boundaries')
        payload = json.loads(Path(cases_path).read_bytes())
        suite = payload.get('case_suite', payload) if isinstance(payload, dict) else payload
        return copy.deepcopy(validate_case_suite(suite))
    sources, model_cases, declared, seen = [], {}, set(), set()
    for path in paths:
        cases, source, dimension_names = _graph_cases(path, bindings)
        _require(source['sha256'] not in seen, 'Repeated graph content; supply each graph once')
        seen.add(source['sha256'])
        sources.append(source)
        declared.update(dimension_names)
        for case in cases:
            key = _key(case)
            if key not in model_cases:
                model_cases[key] = case
            else:
                model_cases[key]['occurrences'] += 1
                model_cases[key]['origins'].extend(case['origins'])
    _require(set(bindings) <= declared, 'Dimension binding is not declared by any supplied graph')
    cases = [*model_cases.values(), *(generated_boundaries() if boundaries else [])]
    if names:
        cases = [case for case in cases if case['family'] in names]
    for case in cases:
        case['origins'].sort(key=lambda item: (item['source_sha256'], item['node_id']))
    suite = {'format': FORMAT,
             'generator': {'version': GENERATOR_VERSION, 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
             'sources': sorted(sources, key=lambda item: item['sha256']),
             'dimensions': dict(sorted(bindings.items())), 'boundaries': boundaries,
             'boundary_rule': 'tile-group-neighbours/v1', 'cases': sorted(cases, key=_key)}
    return validate_case_suite(_seal(suite))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', action='append', type=Path, default=[], help='Graph JSON; repeat for multiple graphs')
    parser.add_argument('--cases', type=Path, help='Replay an exact generated suite or report containing case_suite')
    parser.add_argument('--family', action='append', default=[], help='Case family; repeat or use comma-separated names')
    parser.add_argument('--no-boundaries', action='store_true', help='Only cases extracted from graph nodes')
    parser.add_argument('--dim', action='append', default=[], metavar='NAME=VALUE', help='Concrete symbolic dimension binding')
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument('--out', type=Path, help='Write a new generated suite file')
    destination.add_argument('--stdout', action='store_true', help='Write the suite to stdout (the default)')
    args = parser.parse_args(argv)
    try:
        bindings = {}
        for item in args.dim:
            name, value = item.split('=', 1)
            _require(name not in bindings, 'Repeated dimension binding: ' + name)
            bindings[name] = int(value)
        suite = load_case_suite(args.graph, cases_path=args.cases, families=args.family,
                                boundaries=not args.no_boundaries, dimensions=bindings)
        text = json.dumps(suite, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('x', encoding='utf-8') as output:
                output.write(text)
        else:
            print(text, end='')
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
