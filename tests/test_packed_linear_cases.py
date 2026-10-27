"""Graph coverage, bounded shape resolution, and portable workload replay."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from packed_linear_cases import (BITS, GROUPS, TILES, generated_boundaries,
                                 load_case_suite, validate_case_suite)


def graph(shape=(2, 3, 32), width=8):
    """Two independent projections share one workload shape."""
    return {'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'input0': {'shape': list(shape), 'dtype': 'float32'}},
            'nodes': [{'id': 'projection_' + str(i), 'opType': 'PackedLinear',
                       'inputs': {'input': 'input0', 'weight': 'weight_' + str(i)},
                       'outputs': {'out': {'tensor': 'output_' + str(i),
                                           'shape': [*shape[:-1], width], 'dtype': 'float32'}},
                       'params': {'group_size': 32, 'weight_bits': 4}}
                      for i in range(2)]}


def reseal(suite):
    """Construct checksum-valid malformed inputs to exercise semantic guards."""
    def hashed(value):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()
    suite['cases_sha256'] = hashed(suite['cases'])
    suite['suite_sha256'] = hashed({key: value for key, value in suite.items() if key != 'suite_sha256'})
    return suite


class PackedLinearCasesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value, ensure_ascii=False))
        return path

    def test_all_graph_nodes_counted_and_identical_workloads_aggregated(self):
        value = graph()
        value['nodes'].append({'id': 'router', 'opType': 'PackedLinear',
                               'inputs': {'input': 'output_0', 'weight': 'router_weight'},
                               'outputs': {'out': {'tensor': 'routed', 'shape': [2, 3, 4], 'dtype': 'float32'}},
                               'params': {'group_size': 64}})
        path = self.write('모델 graph.json', value)
        suite = load_case_suite([path], boundaries=False)
        self.assertEqual(sum(case['occurrences'] for case in suite['cases']), 3)
        self.assertEqual({(c['rows'], c['k'], c['n'], c['occurrences']) for c in suite['cases']},
                         {(6, 32, 8, 2), (6, 8, 4, 1)})
        self.assertEqual(suite['sources'][0]['packed_nodes'], 3)
        self.assertEqual(suite['sources'][0]['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(suite['sources'][0]['name'], path.name)
        self.assertNotIn(str(self.directory), json.dumps(suite))

    def test_sources_have_stable_order_and_duplicate_graphs_are_rejected(self):
        a, b = self.write('a.json', graph()), self.write('b.json', graph((4, 32)))
        self.assertEqual(load_case_suite([a, b]), load_case_suite([b, a]))
        duplicate = self.directory / 'copy.json'
        duplicate.write_bytes(a.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Repeated graph content'):
            load_case_suite([a, duplicate])

    def test_unrelated_symbolic_tensors_do_not_require_bindings(self):
        value = graph()
        value['dimensions'] = {'P': {'min': 1, 'max': 191}}
        value['inputs']['unused_cache'] = {'shape': [1, 'P', 8], 'dtype': 'float32'}
        suite = load_case_suite([self.write('static.json', value)], boundaries=False)
        self.assertEqual(suite['cases'][0]['rows'], 6)

    def test_dynamic_shapes_require_explicit_legal_bindings(self):
        value = graph((2, 'tokens', 32))
        value['dimensions'] = {'tokens': {'min': 2, 'max': 8, 'multiple_of': 2}}
        path = self.write('dynamic.json', value)
        with self.assertRaisesRegex(ValueError, 'Unresolved dimension'):
            load_case_suite([path])
        suite = load_case_suite([path], dimensions={'tokens': 4}, boundaries=False)
        self.assertEqual(suite['cases'][0]['rows'], 8)
        self.assertEqual(suite['dimensions'], {'tokens': 4})
        for value in [1, 3, 10, True]:
            with self.subTest(binding=value), self.assertRaises(ValueError):
                load_case_suite([path], dimensions={'tokens': value})
        with self.assertRaisesRegex(ValueError, 'not declared'):
            load_case_suite([path], dimensions={'tokens': 4, 'typo': 1})

    def test_fixed_dimension_bounds_and_safe_shape_formulas(self):
        value = graph((2, 'tokens*2+1', 32))
        value['dimensions'] = {'tokens': {'min': 3, 'max': 3}}
        suite = load_case_suite([self.write('formula.json', value)], boundaries=False)
        self.assertEqual(suite['cases'][0]['rows'], 14)
        for expression in ['__import__("os").getpid()', '2**10000', 'tokens/2', 'tokens//0', 'tokens-3']:
            bad = graph((2, expression, 32)); bad['dimensions'] = value['dimensions']
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                load_case_suite([self.write('bad-formula.json', bad)])

    def test_zero_binding_is_valid_inside_positive_resolved_axis(self):
        value = graph(('P + 1', 32))
        value['dimensions'] = {'P': {'min': 0, 'max': 8}}
        suite = load_case_suite([self.write('empty-cache.json', value)],
                               dimensions={'P': 0}, boundaries=False)
        self.assertEqual(suite['cases'][0]['rows'], 1)
        self.assertEqual(suite['dimensions'], {'P': 0})
        validate_case_suite(suite)
        direct = graph(('P', 32)); direct['dimensions'] = value['dimensions']
        with self.assertRaisesRegex(ValueError, 'Resolved shape axis'):
            load_case_suite([self.write('empty-rows.json', direct)], dimensions={'P': 0})

    def test_invalid_graph_shapes_formats_and_coverage_fail(self):
        def mismatch(g):
            g['nodes'][0]['outputs']['out']['shape'][0] = 1
        def no_group(g):
            del g['nodes'][0]['params']['group_size']
        def bad_bits(g):
            g['nodes'][0]['params']['weight_bits'] = True
        def missing_input(g):
            g['nodes'][0]['inputs']['input'] = 'missing_tensor'
        def repeated_node(g):
            g['nodes'][1]['id'] = g['nodes'][0]['id']
        def repeated_tensor(g):
            g['nodes'][1]['outputs']['out']['tensor'] = 'output_0'
        def no_packed(g):
            g['nodes'] = []
        def wrong_dtype(g):
            g['inputs']['input0']['dtype'] = 'float16'
        for change in [mismatch, no_group, bad_bits, missing_input, repeated_node, repeated_tensor, no_packed, wrong_dtype]:
            value = graph(); change(value)
            with self.subTest(change=change.__name__), self.assertRaises(ValueError):
                load_case_suite([self.write('invalid.json', value)])
        for shape in [(True, 32), (0, 32), (2147483647, 2, 32), (1,) * 9]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                load_case_suite([self.write('invalid-shape.json', graph(shape))])

    def test_generated_controls_cover_tiles_groups_bits_and_panel_edges(self):
        cases = generated_boundaries()
        rows = {c['rows'] for c in cases if c['family'] == 'row_boundaries'}
        self.assertTrue({1, 2} <= rows)
        for tile in TILES:
            self.assertTrue({tile - 1, tile, tile + 1, tile + 2, tile + 3} <= rows)
        groups = {(c['group'], c['k'], c['n']) for c in cases if c['family'] == 'group_controls'}
        for group in GROUPS:
            self.assertTrue({(group, group + delta, 8) for delta in (-1, 0, 1)} <= groups)
        self.assertEqual({c['bits'] for c in cases if c['family'] == 'bit_controls'}, set(BITS))
        self.assertEqual({c['n'] for c in cases if c['family'] == 'tails'}, {3, 4, 5})
        self.assertTrue(all(c['occurrences'] == 0 and c['origins'] == [] for c in cases))
        self.assertLessEqual(len(cases), 100)

    def test_replay_remains_exact_without_original_graphs(self):
        graph_path = self.write('original.json', graph())
        suite = load_case_suite([graph_path])
        report = self.write('report.json', {'format': 'volvoxai-packed-linear-tiles/v2', 'case_suite': suite})
        graph_path.unlink()
        self.assertEqual(load_case_suite(cases_path=report), suite)
        for extra in [dict(families=['model']), dict(boundaries=False), dict(dimensions={'tokens': 2}), dict(graph_paths=[graph_path])]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'exact suite'):
                load_case_suite(cases_path=report, **extra)

    def test_replay_rejects_tampering_and_checksum_valid_invalid_cases(self):
        suite = load_case_suite([self.write('graph.json', graph())])
        changed = copy.deepcopy(suite); changed['cases'][0]['k'] += 1
        with self.assertRaises(ValueError):
            validate_case_suite(changed)
        changes = [lambda s: s['cases'][0].update(bits=5),
                   lambda s: s['cases'][0].update(rows=True),
                   lambda s: s['cases'].append(copy.deepcopy(s['cases'][0])),
                   lambda s: s.update(unexpected='metadata')]
        for change in changes:
            changed = copy.deepcopy(suite); change(changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_case_suite(reseal(changed))
        changed = copy.deepcopy(suite)
        model = next(c for c in changed['cases'] if c['family'] == 'model')
        model['origins'].pop(); model['occurrences'] -= 1
        with self.assertRaisesRegex(ValueError, 'coverage is incomplete'):
            validate_case_suite(reseal(changed))

    def test_family_selection_has_new_identity_and_no_empty_silent_run(self):
        suite = load_case_suite()
        subset = load_case_suite(families=['row_boundaries,tails'])
        self.assertEqual({c['family'] for c in subset['cases']}, {'row_boundaries', 'tails'})
        self.assertNotEqual(subset['suite_sha256'], suite['suite_sha256'])
        for kwargs in [dict(boundaries=False), dict(families=['unknown']), dict(families=['model']), dict(families=[''])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load_case_suite(**kwargs)

    def test_cli_output_and_no_overwrite(self):
        path = self.write('입력 graph.json', graph())
        command = [sys.executable, str(ROOT / 'tools/packed_linear_cases.py'), '--graph', str(path), '--no-boundaries']
        result = subprocess.run([*command, '--stdout'], capture_output=True, text=True, check=True)
        suite = json.loads(result.stdout)
        self.assertEqual(len(suite['cases']), 1)
        output = self.directory / 'cases.json'
        subprocess.run([*command, '--out', str(output)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(output.read_text()), suite)
        before = output.read_bytes()
        rejected = subprocess.run([*command, '--out', str(output)], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
