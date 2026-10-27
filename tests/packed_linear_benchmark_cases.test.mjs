import assert from 'node:assert/strict';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {loadCaseSuite} from '../tools/benchmark_packed_linear_tiles.mjs';

function graph(id) {
  return {format:'volvox-graph/v1',dimensions:{sequence:{min:1,max:8}},
    inputs:{x:{shape:[2,'sequence',32],dtype:'float32'}},outputs:['y'],banks:{},
    nodes:[{id,opType:'PackedLinear',inputs:{input:'x',weight:'w',scale:'s'},
      outputs:{out:{tensor:'y',shape:[2,'sequence',8],dtype:'float32'}},
      params:{weight_bits:4,group_size:32}}]};
}
async function directory(t) {
  const value=await mkdtemp(path.join(os.tmpdir(),'volvoxai-case-bridge-'));
  t.after(()=>rm(value,{recursive:true,force:true}));return value;
}

test('Node bridge resolves repeated graph paths and symbolic dimensions through the shared generator',async t=>{
  const dir=await directory(t),files=[path.join(dir,'first graph.json'),path.join(dir,'두번째 graph.json')];
  await Promise.all(files.map((file,i)=>writeFile(file,JSON.stringify(graph(`projection_${i}`)))));
  const suite=await loadCaseSuite({graphs:files,dimensions:['sequence=3'],families:['model'],noBoundaries:true});
  assert.equal(suite.format,'volvoxai-packed-linear-cases/v2');
  assert.equal(suite.sources.length,2);
  assert.equal(suite.cases.length,1);
  const c=suite.cases[0];
  assert.deepEqual([c.rows,c.k,c.n,c.bits,c.group,c.arithmetic],[6,32,8,4,32,'dynamic-i8']);
  assert.equal(c.occurrences,2);
  assert.equal(c.origins.length,2);
  assert.match(suite.cases_sha256,/^[a-f0-9]{64}$/);
  assert.match(suite.suite_sha256,/^[a-f0-9]{64}$/);
});

test('A saved benchmark report replays its exact suite without the original model graphs',async t=>{
  const dir=await directory(t),file=path.join(dir,'source.json');
  await writeFile(file,JSON.stringify(graph('projection')));
  const original=await loadCaseSuite({graphs:[file],dimensions:['sequence=2'],noBoundaries:true});
  const report=path.join(dir,'previous benchmark.json');
  await writeFile(report,JSON.stringify({format:'volvoxai-packed-linear-tiles/v2',case_suite:original}));
  await rm(file);
  const replayed=await loadCaseSuite({cases:report});
  assert.deepEqual(replayed,original);
});

test('Unresolved model dimensions and invalid replay integrity fail through the Node bridge',async t=>{
  const dir=await directory(t),file=path.join(dir,'dynamic.json');
  await writeFile(file,JSON.stringify(graph('projection')));
  await assert.rejects(loadCaseSuite({graphs:[file],noBoundaries:true}),/case generation failed.*sequence/s);
  const suite=await loadCaseSuite({graphs:[file],dimensions:['sequence=2'],noBoundaries:true});
  suite.cases[0].rows+=1;
  const tampered=path.join(dir,'tampered.json');await writeFile(tampered,JSON.stringify(suite));
  await assert.rejects(loadCaseSuite({cases:tampered}),/case generation failed/i);
});
