#!/usr/bin/env python3
"""Compare every frozen VQA prediction across both native and WASM profiles.

Incomplete audits are reported as incomplete and never qualify. Every token,
family selection and EOS flag must match; aggregate scores cannot hide a changed
prediction. Reports are fingerprinted so the comparison identifies its inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/api-migration'))
    parser.add_argument('--candidate-stage', choices=('candidate', 'candidate-final'), default='candidate-final')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for backend in ('native', 'wasm'):
        for profile in ('inference', 'full'):
            baseline = args.root / 'baseline' / f'audit-{backend}' / (
                f'{profile}/report.json' if backend == 'native' else f'resumed/{profile}.json')
            candidate = args.root / args.candidate_stage / f'audit-{backend}' / f'{profile}.json'
            before, before_hash = read(baseline)
            after, after_hash = read(candidate)
            for variant in ('fp32', 'int8'):
                left = (before or {}).get('variants', {}).get(variant, {})
                right = (after or {}).get('variants', {}).get(variant, {})
                old, new = left.get('predictions', []), right.get('predictions', [])
                common = min(len(old), len(new))
                mismatches = [i for i in range(common) if old[i] != new[i]]
                complete = (left.get('status') == right.get('status') == 'executed'
                            and len(old) == len(new) == 2000)
                identity = bool(before and after and before['corpus_sha256'] == after['corpus_sha256']
                                and before['profile'] == after['profile'] == profile
                                and before['backend'] == after['backend'] == ('cpu' if backend == 'native' else 'wasm'))
                rows.append({'backend': backend, 'profile': profile, 'variant': variant,
                             'baseline_report_sha256': before_hash, 'candidate_report_sha256': after_hash,
                             'baseline_count': len(old), 'candidate_count': len(new),
                             'compared_count': common, 'complete': complete, 'identity_matches': identity,
                             'mismatched_case_indices': mismatches,
                             'baseline_score': left.get('score'), 'candidate_score': right.get('score'),
                             'pass': complete and identity and not mismatches})
    result = {'schema': 'volvoxai.api-migration-audit-comparison/v1',
              'candidate_stage': args.candidate_stage,
              'required_rows': 8, 'cases_per_row': 2000, 'rows': rows,
              'complete': all(row['complete'] for row in rows),
              'pass': all(row['pass'] for row in rows), 'performance_qualified': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.out)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))
    return 0 if result['pass'] else 1 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
