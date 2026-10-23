/** Direct Chrome SIMD128 projection probe; no UI, worker or RPC timing.
 * --before/--after are freestanding builds of packed_i8_tiling.c; --test is a
 * freestanding build of native/tests/test_packed_linear_i8.c. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const options={};
for(let i=2;i<process.argv.length;i+=2)options[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
for(const key of ['before','after','test','puppeteer-root','out'])assert(options[key],`--${key} required`);
const blobs=await Promise.all(['before','after','test'].map(async key=>{
 const data=await readFile(options[key]);return {key,bytes:[...data],sha256:createHash('sha256').update(data).digest('hex')};
}));
const {default:puppeteer}=await import(pathToFileURL(path.join(options['puppeteer-root'],'node_modules/puppeteer/lib/puppeteer/puppeteer.js')));
const browser=await puppeteer.launch({headless:true,protocolTimeout:600000,args:['--no-sandbox','--disable-dev-shm-usage']});
try {
 const page=await browser.newPage();
 const result=await page.evaluate(async blobs=>{
  const modules={};
  for(const {key,bytes} of blobs)modules[key]=(await WebAssembly.instantiate(new Uint8Array(bytes))).instance.exports;
  const tests={};
  for(const name of ['test_packed_linear_i8','test_packed_linear_i8_tiles']) {
   tests[name]=modules.test[name]();if(tests[name])throw Error(`${name} failed at C line ${tests[name]}`);
  }
  const samples=[];
  for(const [rows,k,n] of [[1,320,1280],[4,320,1280],[8,320,1280],[402,320,960],[402,320,1280],[402,1280,320],[9,257,13]])for(const bits of [4,8]) {
   const timing={before:[],after:[]},repeats=rows>100?2:100;
   for(const key of ['before','after']) {
    if(modules[key].packed_bench_init(rows,k,n,bits)||modules[key].packed_bench_run(2))throw Error(`setup ${key}`);
   }
   for(let trial=0;trial<5;trial++)for(const key of trial%2?['after','before']:['before','after']) {
    const start=performance.now();const status=modules[key].packed_bench_run(repeats);
    timing[key].push((performance.now()-start)/repeats);if(status)throw Error(`run ${key}`);
   }
   const hashes=Object.fromEntries(['before','after'].map(key=>[key,modules[key].packed_bench_hash()>>>0]));
   if(hashes.before!==hashes.after)throw Error(`output mismatch ${rows}/${k}/${n}/I${bits}`);
   const median=key=>[...timing[key]].sort((a,b)=>a-b)[2];
   samples.push({rows,k,n,bits,hash:hashes.after,before_ms:median('before'),after_ms:median('after'),speedup:median('before')/median('after'),timing});
  }
  return {tests,samples};
 },blobs);
 await writeFile(options.out,JSON.stringify({boundary:'isolated direct Chrome WASM projection; group32; 2 warmups; 5 alternating batches; no UI/worker/RPC',host_quiet_verified:false,chrome:await browser.version(),sha256:Object.fromEntries(blobs.map(({key,sha256})=>[key,sha256])),...result},null,2)+'\n');
 console.log(JSON.stringify(result,null,2));
} finally {await browser.close();}
