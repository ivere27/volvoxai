import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { Tokenizer } from './parity/reference/Tokenizer.ts';
import { qualifyTokenizer } from './parity/external/tokenizer_api.mjs';
import { qualifyTokenizerBytes } from './parity/external/tokenizer_bytes.mjs';
const {version}=JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
for(const filename of ['volvoxai.lite.js','volvoxai.lite.min.js','volvoxai.js','volvoxai.min.js']) {
  test(`${filename} preserves previous encoding and joins decoded bytes through generated C dispatch`,async()=>{
    const api=await import(new URL(`../dist/${version}/${filename}`,import.meta.url));
    await qualifyTokenizer(api,Tokenizer,new URL(`../dist/${version}/${filename.includes('.lite')?'volvoxai.lite.wasm':'volvoxai.wasm'}`,import.meta.url));
  });
}
for(const filename of ['volvoxai.lite.js','volvoxai.js']){
  test(`${filename} ByteLevel token bytes use generated C dispatch`,async()=>{
    const api=await import(new URL(`../dist/${version}/${filename}`,import.meta.url));
    await qualifyTokenizerBytes(api,new URL(`../dist/${version}/${filename.includes('.lite')?'volvoxai.lite.wasm':'volvoxai.wasm'}`,import.meta.url));
  });
}

