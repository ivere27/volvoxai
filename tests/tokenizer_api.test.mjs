import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { Tokenizer } from './parity/reference/Tokenizer.ts';
import { qualifyTokenizer } from './parity/external/tokenizer_api.mjs';
const {version}=JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
for(const filename of ['volvoxai.js','volvoxai.min.js','volvoxai.full.js','volvoxai.full.min.js']) {
  test(`${filename} matches the previous Tokenizer through generated C dispatch`,async()=>{
    const api=await import(new URL(`../dist/${version}/${filename}`,import.meta.url));
    await qualifyTokenizer(api,Tokenizer,new URL(`../dist/${version}/${filename.includes('.full')?'volvoxai.full.wasm':'volvoxai.wasm'}`,import.meta.url));
  });
}
