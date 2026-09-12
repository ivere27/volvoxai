import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {qualifyLora} from './parity/external/lora_api.mjs';
const {version}=JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
for(const file of ['volvoxai.full.js','volvoxai.full.min.js'])test(`${file} C-authored LoRA and weight snapshots`,async()=>{
  const api=await import(new URL(`../dist/${version}/${file}`,import.meta.url));
  await qualifyLora(api,new URL(`../dist/${version}/volvoxai.full.wasm`,import.meta.url));
});
