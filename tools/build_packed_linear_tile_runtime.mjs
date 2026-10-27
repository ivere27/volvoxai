#!/usr/bin/env node
// Private experiment builds; the release recipe remains authoritative.
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {buildWasmProfile} from './build_wasm_release.mjs';
import {wasmProfile} from './release_profiles.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const arg = (name, fallback) => process.argv.includes(name)
  ? process.argv[process.argv.indexOf(name) + 1] : fallback;
const rows = Number(arg('--rows', '4')), tile2 = Number(arg('--tile2', '0'));
const out = arg('--out');
if (!out || ![4, 8, 16, 32, 64].includes(rows) || ![0, 1].includes(tile2))
  throw new Error('Usage: --rows 4|8|16|32|64 --tile2 0|1 --out NEW_DIR');
const directory = path.resolve(root, out);
await mkdir(path.dirname(directory), {recursive: true});
await mkdir(directory, {recursive: false});
const base = wasmProfile('inference', root);
const defines = [`-DVX_PACKED_LINEAR_TILE_ROWS=${rows}`, `-DVX_PACKED_LINEAR_TILE2=${tile2}`,
  '-DVX_PACKED_LINEAR_TILE_ALL_BITS=0'];
const profile = {...base, recipe: {...base.recipe, parent: {...base.recipe.parent,
  objects: base.recipe.parent.objects.map(object => ({...object, flags: [...object.flags, ...defines]})),
}}};
const result = await buildWasmProfile({repositoryRoot: root, profile,
  outputDirectory: path.join(directory, 'runtime'), buildRoot: path.join(directory, 'objects'),
  evidenceDirectory: path.join(directory, 'evidence')});
const bytes = await readFile(path.join(directory, 'runtime', base.filename));
await writeFile(path.join(directory, 'experiment.json'), JSON.stringify({
  format: 'volvoxai-packed-linear-tile-build/v1', rows, tile2, all_bits: 0, defines,
  wasm_sha256: createHash('sha256').update(bytes).digest('hex'), wasm_bytes: bytes.length,
  note: 'Private tuning build; use matching recipe evidence, not default-release provenance.',
}, null, 2) + '\n');
console.log(`Built PackedLinear MR=${rows}, two-row=${tile2}: ${bytes.length} bytes`);
