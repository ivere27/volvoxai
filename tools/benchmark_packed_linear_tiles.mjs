#!/usr/bin/env node
// Standalone WASM kernels in Node or real Chrome; every output byte is checked.
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {readFile,writeFile,mkdir,access} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parseArgs,promisify} from 'node:util';

const root=fileURLToPath(new URL('../',import.meta.url));
const executeFile=promisify(execFile);
const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
const usage=`Usage: node tools/benchmark_packed_linear_tiles.mjs
  --variant NAME=FIXTURE_DIR (repeat; baseline first) --out NEW_FILE
  [--graph GRAPH_JSON ...] [--dim NAME=VALUE ...] [--no-boundaries]
  [--cases PREVIOUS_SUITE_OR_REPORT] [--family NAME ...]
  [--browser node|chrome] [--cycles 5] [--work-per-sample 100000000]
  [--python python3] [--puppeteer-root DIR]
Graph shapes are extracted automatically; no graphs selects generated boundary
and format controls. Reports retain the exact generated case suite for --cases replay.`;

// Use the same stdlib generator as the native benchmark. No graph/shape
// inference is duplicated here, and Python never runs inside the timed region.
export async function loadCaseSuite({graphs=[],cases,families=[],dimensions=[],
  noBoundaries=false,python=process.env.PYTHON||'python3'}={}) {
  const args=[path.join(root,'tools/packed_linear_cases.py'),'--stdout'];
  for(const file of graphs)args.push('--graph',path.resolve(file));
  if(cases)args.push('--cases',path.resolve(cases));
  for(const name of families)args.push('--family',name);
  for(const value of dimensions)args.push('--dim',value);
  if(noBoundaries)args.push('--no-boundaries');
  let stdout;
  try {
    ({stdout}=await executeFile(python,args,{encoding:'utf8',maxBuffer:32*1024*1024,timeout:120000}));
  } catch(error) {
    throw new Error(`PackedLinear case generation failed: ${error.stderr?.trim()||error.message}`,{cause:error});
  }
  const suite=JSON.parse(stdout);
  assert.equal(suite.format,'volvoxai-packed-linear-cases/v2','Unexpected case generator format');
  assert(Array.isArray(suite.cases)&&suite.cases.length,'The case suite is empty');
  for(const item of suite.cases) {
    assert(['rows','k','n'].every(name=>Number.isSafeInteger(item[name])&&item[name]>0&&item[name]<=0x7fffffff),
      'Case dimensions must be positive signed-I32 integers');
    assert([2,3,4,8].includes(item.bits)&&[32,64,128,256].includes(item.group),
      'Unsupported PackedLinear storage format');
    assert.equal(item.arithmetic,'dynamic-i8','This fixture measures dynamic-I8 arithmetic');
  }
  return suite;
}

async function benchmark(spec){
  const modules=Object.create(null),tests=Object.create(null),samples=[],admission=Object.create(null);
  try{
  for(const item of spec.variants){
    const module=await WebAssembly.compile(new Uint8Array(item.bytes));
    if(WebAssembly.Module.imports(module).length)throw Error('Unexpected imports');
    const value=modules[item.name]=(await WebAssembly.instantiate(module)).exports;
    if(value.tile_bench_rows()!==item.metadata.rows||value.tile_bench_two()!==item.metadata.tile2)throw Error('Tuning metadata mismatch');
    const dynamic=item.metadata.format==='volvoxai-packed-row-fixture/v2';
    if(dynamic){
      for(const name of ['tile_bench_required_bytes','tile_bench_memory_limit','tile_bench_release','tile_bench_all_bits'])
        if(typeof value[name]!=='function')throw Error(`${item.name}: missing bounded fixture export ${name}`);
      if(value.tile_bench_all_bits()!==item.metadata.all_bits)throw Error('Weight-format tuning metadata mismatch');
      const limit=value.tile_bench_memory_limit()>>>0;
      admission[item.name]={capacity_mode:'bounded-dynamic',memory_limit_bytes:limit,required_bytes:[]};
      for(const c of spec.cases){
        const bytes=value.tile_bench_required_bytes(c.rows,c.k,c.n,c.bits,c.group)>>>0;
        if(!bytes||bytes>limit)throw Error(`${item.name}: ${c.rows}x${c.k}x${c.n} W${c.bits}/G${c.group} exceeds fixture admission (limit ${limit} bytes)`);
        admission[item.name].required_bytes.push(bytes);
      }
    }else{
      if(item.metadata.format!=='volvoxai-packed-row-fixture/v1')throw Error(`${item.name}: unknown fixture format`);
      const limits={rows:1608,k:1283,n:1536};
      admission[item.name]={capacity_mode:'legacy-fixed',limits};
      for(const c of spec.cases)if(Object.entries(limits).some(([key,limit])=>c[key]>limit))
        throw Error(`${item.name}: ${c.rows}x${c.k}x${c.n} exceeds legacy fixture limits; rebuild the fixture`);
    }
    tests[item.name]=Object.create(null);
    for(const name of item.metadata.tests){
      const result=tests[item.name][name]=value[name]();if(result)throw Error(`${item.name}/${name}: native test line ${result}`);
    }
  }
  const names=Object.keys(modules),median=values=>{
    const sorted=[...values].sort((a,b)=>a-b),middle=Math.floor(sorted.length/2);
    return sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2;
  };
  for(const item of spec.cases){
    const {rows,k,n,bits,group}=item;
    let repeats=Math.max(1,Math.min(1000,Math.floor(spec.workPerSample/(rows*k*n))));
    const timing=Object.create(null),scratch=Object.create(null),hashes=Object.create(null);let reference;
    const bytes=value=>new Uint8Array(value.memory.buffer,value.tile_bench_output_ptr(),value.tile_bench_output_bytes());
    const verify=(name,value)=>{
      const actual=bytes(value);if(!reference)reference=actual.slice();
      if(actual.length!==reference.length||actual.some((byte,index)=>byte!==reference[index])||value.tile_bench_check_scratch())
        throw Error(`Output/scratch mismatch ${name} ${JSON.stringify(item)}`);
    };
    for(const name of names){
      const value=modules[name];timing[name]=[];
      const status=value.tile_bench_init(rows,k,n,bits,group);
      if(status)throw Error(`${name}: fixture initialization failed for ${rows}x${k}x${n} W${bits}/G${group} (code ${status})`);
      if(value.tile_bench_run(2))throw Error(`${name}: warmup failed for ${JSON.stringify(item)}`);
      verify(name,value);scratch[name]=value.tile_bench_scratch_bytes();hashes[name]=value.tile_bench_hash()>>>0;
    }
    const runTimed=(name,count)=>{
      const began=performance.now(),code=modules[name].tile_bench_run(count),elapsed=performance.now()-began;
      if(code)throw Error(`${name}: kernel failed for ${JSON.stringify(item)}`);
      if(!Number.isFinite(elapsed)||elapsed<0)throw Error(`${name}: invalid elapsed time ${elapsed}`);
      return elapsed;
    };
    // Generated boundary cases can take less than one browser clock tick.
    // Calibrate one common repeat count outside the measured cycles, using
    // the fastest variant, so every reported denominator is observable.
    const calibration=[];
    for(;;){
      const elapsed=Object.fromEntries(names.map(name=>[name,runTimed(name,repeats)]));
      calibration.push({repeats,elapsed_ms:elapsed});
      const fastest=Math.min(...Object.values(elapsed));
      if(fastest>=spec.minSampleMs||repeats===spec.maxRepeats){
        if(fastest<=0)throw Error('Timer did not advance at the fixture repeat limit; use a higher-resolution timing environment');
        break;
      }
      repeats=Math.min(spec.maxRepeats,Math.max(repeats*2,
        fastest>0?Math.ceil(repeats*spec.minSampleMs/fastest*1.1):repeats*10));
    }
    for(let cycle=0;cycle<spec.cycles;cycle++)for(const name of cycle%2?[...names].reverse():names){
      const elapsed=runTimed(name,repeats);
      if(elapsed<=0)throw Error(`${name}: timer did not advance during a measured sample; no latency or speedup is reported`);
      timing[name].push(elapsed/repeats);
    }
    for(const name of names)verify(name,modules[name]);
    const medians=Object.fromEntries(names.map(name=>[name,median(timing[name])])),baseline=medians[names[0]];
    samples.push({...item,repeats,calibration,timing_ms:timing,median_ms:medians,
      speedup:Object.fromEntries(names.map(name=>[name,baseline/medians[name]])),
      scratch_required_bytes:scratch,output_fnv1a32:hashes,all_outputs_byte_equal:true});
    console.log(`TILE_WASM ${item.family} ${rows}x${k}x${n} W${bits}/G${group}`);
  }
  return {tests,samples,admission};
  }finally{
    for(const value of Object.values(modules))value.tile_bench_release?.();
  }
}

export async function main(args=process.argv.slice(2)) {
  const {values}=parseArgs({args,options:{
    variant:{type:'string',multiple:true},out:{type:'string'},browser:{type:'string',default:'node'},
    cycles:{type:'string',default:'5'},'work-per-sample':{type:'string',default:'100000000'},
    'puppeteer-root':{type:'string'},python:{type:'string'},
    graph:{type:'string',multiple:true},cases:{type:'string'},family:{type:'string',multiple:true},
    dim:{type:'string',multiple:true},'no-boundaries':{type:'boolean',default:false},help:{type:'boolean'},
  }});
  if(values.help){console.log(usage);return;}
  const specifications=values.variant??[],cycles=Number(values.cycles),backend=values.browser,out=values.out;
  assert(specifications.length&&out&&Number.isInteger(cycles)&&cycles>0&&['node','chrome'].includes(backend),usage);
  const workPerSample=Number(values['work-per-sample']);
  assert(Number.isSafeInteger(workPerSample)&&workPerSample>0,'--work-per-sample must be a positive safe integer');
  await assert.rejects(access(out),{code:'ENOENT'});
  const loader=path.join(root,'tools/packed_linear_cases.py'),loaderHash=sha256(await readFile(loader));
  const suite=await loadCaseSuite({graphs:values.graph,cases:values.cases,families:values.family,
    dimensions:values.dim,noBoundaries:values['no-boundaries'],python:values.python});
  assert.equal(sha256(await readFile(loader)),loaderHash,'Case loader changed during extraction');
  const variants=[];
  for(const item of specifications){
    const separator=item.indexOf('='),name=item.slice(0,separator),directory=path.resolve(item.slice(separator+1));
    assert(separator>0&&!variants.some(value=>value.name===name),'Invalid or duplicate variant');
    const [bytes,raw]=await Promise.all([readFile(path.join(directory,'fixture.wasm')),readFile(path.join(directory,'fixture.json'),'utf8')]);
    const metadata=JSON.parse(raw);
    assert.equal(sha256(bytes),metadata.wasm_sha256);
    assert(Array.isArray(metadata.tests)&&metadata.tests.length&&metadata.tests.every(name=>typeof name==='string'),
      'Fixture metadata must include its numerical checks');
    variants.push({name,bytes:[...bytes],metadata});
  }
  const spec={variants,cases:suite.cases,cycles,workPerSample,minSampleMs:1,maxRepeats:10000};
  const report={format:'volvoxai-packed-linear-tiles/v2',status:'running',backend,
    runner_sha256:sha256(await readFile(fileURLToPath(import.meta.url))),case_loader_sha256:loaderHash,
    scope:'Standalone WASM kernel; preallocated inputs and scratch; no allocation/dispatch/model I/O timed',
    cpu_threads:1,cycles,warmups:2,work_per_sample:workPerSample,
    min_sample_ms:spec.minSampleMs,max_repeats:spec.maxRepeats,
    order:'Forward/reverse variant order each cycle',host_quiet_verified:false,
    case_suite:suite,variants:Object.fromEntries(variants.map(({name,metadata})=>[name,metadata]))};
  let browser;
  try{
    let result;
    if(backend==='node'){report.runtime=process.version;result=await benchmark(spec);}
    else{
      const packageRoot=path.resolve(values['puppeteer-root']??path.join(root,'../unaveraged'));
      const {default:puppeteer}=await import(pathToFileURL(path.join(packageRoot,'node_modules/puppeteer/lib/puppeteer/puppeteer.js')));
      browser=await puppeteer.launch({headless:true,protocolTimeout:3600000,args:['--no-sandbox','--disable-dev-shm-usage']});
      report.runtime=await browser.version();const page=await browser.newPage();
      page.on('console',message=>console.log(message.text()));result=await page.evaluate(benchmark,spec);
    }
    Object.assign(report,result,{status:'passed'});
  }catch(error){report.status='failed';report.error=String(error);throw error;}
  finally{await browser?.close();await mkdir(path.dirname(out),{recursive:true});await writeFile(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});}
}

if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href)await main();
