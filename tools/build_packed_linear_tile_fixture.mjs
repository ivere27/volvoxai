#!/usr/bin/env node
// Build standalone kernel + independent numerical oracles with release flags.
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {buildWasmComponent} from './build_wasm_release.mjs';
import {wasmKernelFixtureRecipe} from './build_wasm_kernel_fixture.mjs';
import {sha256} from './release_profiles.mjs';

const root=fileURLToPath(new URL('../',import.meta.url));
const option=(key,fallback)=>process.argv.includes(key)?process.argv[process.argv.indexOf(key)+1]:fallback;
const rows=Number(option('--rows','4')),tile2=Number(option('--tile2','0'));
const allBits=Number(option('--all-bits','0'));
assert([4,8,16,32,64].includes(rows)&&[0,1].includes(tile2)&&[0,1].includes(allBits)&&option('--out'),
  'Usage: --rows 4|8|16|32|64 --tile2 0|1 [--all-bits 0|1] --out NEW_DIR');
const directory=path.resolve(root,option('--out'));
await mkdir(directory,{recursive:false});
const tests=['test_packed_linear_i3_unpack','test_packed_linear_i8','test_packed_linear_i8_tiles',
  'test_packed_linear_exact_bounds','test_packed_linear_lowbit_arithmetic','test_packed_linear_row_tiles'];
const exports=[...tests,'tile_bench_init','tile_bench_run','tile_bench_output_ptr','tile_bench_output_bytes',
  'tile_bench_hash','tile_bench_scratch_bytes','tile_bench_check_scratch','tile_bench_rows','tile_bench_two',
  'tile_bench_all_bits','tile_bench_required_bytes','tile_bench_memory_limit','tile_bench_release'];
const sources=['examples/tiny_receipt_vqa/benchmarks/packed_row_tiles.c','native/tests/test_packed_linear_i8.c',
  'native/src/runtime/wasm_libc.c','native/src/kernels/wasm_heap_arena.c'];
const base=wasmKernelFixtureRecipe(root),parent=base.recipe.parent;
const kernel=parent.objects.find(item=>item.id==='kernels');
const defines=[`-DVX_PACKED_LINEAR_TILE_ROWS=${rows}`,`-DVX_PACKED_LINEAR_TILE2=${tile2}`,
  `-DVX_PACKED_LINEAR_TILE_ALL_BITS=${allBits}`];
const profile={...base,id:`packed-row-${rows}-${tile2}`,filename:'fixture.wasm',recipe:{...base.recipe,
  parent:{...parent,objects:sources.map((source,index)=>({...kernel,id:`source-${index}`,source,
    flags:[...kernel.flags,'-ffp-contract=off',...defines]})),
    linkFlags:[...parent.linkFlags.filter(flag=>!flag.startsWith('--export')&&!flag.startsWith('--initial-memory')),
      '--export-memory','--initial-memory=67108864',...exports.map(name=>`--export=${name}`)]}}};
const result=await buildWasmComponent({repositoryRoot:root,profile,kind:'parent',
  output:path.join(directory,'fixture.wasm'),buildRoot:path.join(directory,'objects'),
  evidence:path.join(directory,'fixture.build.json')});
const module=new WebAssembly.Module(result.artifactContents);
assert.deepEqual(WebAssembly.Module.imports(module),[]);
await writeFile(path.join(directory,'fixture.json'),JSON.stringify({format:'volvoxai-packed-row-fixture/v2',
  rows,tile2,all_bits:allBits,storage:'per-case-bounded-allocation',
  fixture_source_sha256:sha256(await readFile(path.join(root,sources[0]))),
  defines,tests,wasm_sha256:sha256(result.artifactContents),wasm_bytes:result.artifactContents.length},null,2)+'\n');
console.log(`Built standalone MR=${rows}, tile2=${tile2}: ${result.artifactContents.length} bytes`);
