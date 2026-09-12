#!/usr/bin/env python3
"""Build isolated instrumentation from the recorded inference recipe.

Run inside the build image after make build_wasm, then run the companion .mjs.
The baseline reparses the same retained graph bytes at lowering; it deliberately
excludes the old file read. Production sources and release artifacts are untouched.
"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'build/graph-parse-probe'
DEST.mkdir(parents=True, exist_ok=True)
evidence = json.loads((ROOT / 'build/wasm/provenance/volvoxai.wasm.build.json').read_text())
component = evidence['components'][0]
objects = component['orderedObjects']
control = next(item for item in objects if item['id'] == 'portable-control')
compiler = evidence['toolchain']['compiler']['command']
linker = evidence['toolchain']['linker']['command']
source = subprocess.check_output([compiler, *control['flags'], '-E', control['source']], cwd=ROOT, text=True)
prefix = '''
static unsigned long long vx_probe_counts[4];
static const char* vx_probe_graph_source;
__attribute__((export_name("probe_reset"))) void probe_reset(void) {
  for(int i=0;i<4;i++) vx_probe_counts[i]=0;
}
__attribute__((export_name("probe_count"))) unsigned long long probe_count(int i) {return vx_probe_counts[i];}
__attribute__((export_name("probe_source"))) void probe_source(const char* text) {vx_probe_graph_source=text;}
'''
patterns = [
    (r'(cJSON_ParseWithLengthOpts\([^;{}]+\)\s*\{)', 'vx_probe_counts[0]++;'),
    (r'(cJSON_New_Item\([^;{}]+\)\s*\{)', 'vx_probe_counts[1]++;'),
    (r'(void\* malloc\(size_t size\)\s*\{)', 'vx_probe_counts[2]++; vx_probe_counts[3]+=size;'),
]
for pattern, statement in patterns:
    source, count = re.subn(pattern, lambda match: match[1] + statement, source)
    if count != 1:
        raise RuntimeError(f'instrumentation matched {count} definitions: {pattern}')
source = prefix + source
needle = 'root = cJSON_Duplicate(logical_graph, 1);'
if source.count(needle) != 1:
    raise RuntimeError('expected one canonical graph lowering clone')
for variant in ('shared', 'reparse'):
    text = source if variant == 'shared' else source.replace(needle, 'root = cJSON_Parse(vx_probe_graph_source);')
    path = DEST / f'{variant}.c'
    path.write_text(text)
    output = DEST / f'{variant}.o'
    subprocess.run([compiler, *control['flags'], control['optimization'], '-c', str(path), '-o', str(output)], cwd=ROOT, check=True)
    paths = [str(output) if item['id'] == control['id'] else
             str(ROOT / 'build/wasm/release/objects/inference/parent/objects' / f"{item['index']:03}-{item['id']}.o")
             for item in objects]
    subprocess.run([linker, *component['link']['flags'], *paths, '-o', str(DEST / f'{variant}.wasm')], cwd=ROOT, check=True)
print(DEST)
