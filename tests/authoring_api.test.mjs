import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {initializeTensor} from './parity/reference/Initializers.ts';
import {qualifyAuthoring} from './parity/external/authoring_api.mjs';
const {version}=JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
// VxPlanningService is a full-profile surface, so authoring qualifies only
// the full bundles. Inference consumes models and never authors them.
for(const filename of ['volvoxai.js','volvoxai.min.js']) {
  test(`${filename} authoring transactions and generic storage execute in C`,async()=>{
    const api=await import(new URL(`../dist/${version}/${filename}`,import.meta.url));
    await qualifyAuthoring(api,initializeTensor,new URL(`../dist/${version}/${'volvoxai.wasm'}`,import.meta.url));
  });
}
