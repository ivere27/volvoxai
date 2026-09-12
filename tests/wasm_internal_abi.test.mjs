import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { WASM_INTERNAL_ABI_MANIFEST as manifest } from '../tools/wasm_internal_abi_manifest.mjs';
import { validateReleaseDeclarations, wasmProfile } from '../tools/release_profiles.mjs';
import { WASM_PROFILE_EXPORTS, WASM_PROFILE_IMPORTS, WASM_PROFILE_EXPORT_KINDS,
  WASM_INTERNAL_ABI_MANIFEST_SHA256 } from '../tools/generated/wasmInternalAbi.mjs';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PROFILES=['inference','full'];

test('release WASM omits debug names and retains license and build evidence',()=>{
  const {version}=JSON.parse(fs.readFileSync(path.join(ROOT,'package.json'),'utf8'));
  for(const id of PROFILES) {
    const {filename}=wasmProfile(id);
    const module=new WebAssembly.Module(fs.readFileSync(path.join(ROOT,'dist',version,filename)));
    assert.equal(WebAssembly.Module.customSections(module,'name').length,0,filename);
    assert.equal(WebAssembly.Module.customSections(module,'license.unicode').length,1,filename);
    assert.equal(WebAssembly.Module.customSections(module,'volvoxai.release.provenance.v1').length,1,filename);
  }
});

test('the release contains one C owner per profile and no embedded execution children',()=>{
  for(const id of PROFILES) {
    const profile=wasmProfile(id), {parent,relaxed,ptqAuthoring}=profile.recipe;
    assert.equal(relaxed,null); assert.equal(ptqAuthoring,null);
    const sources=parent.objects.map(o=>o.source);
    assert.equal(new Set(sources).size,sources.length);
    assert.deepEqual(profile.sourceEntries.filter(p=>p.endsWith('.c')),sources);
    assert.ok(sources.includes('native/src/runtime/portable_control_wasm.c'));
    assert.ok(sources.includes('native/src/runtime/wasm_math.c'));
    if(id==='inference') assert.ok(!sources.some(p=>/training|ptq/.test(p)));
    else for(const file of ['trainer_api','ptq_api','quantization_runtime','quantization_package','ptq_authoring','training_control'])
      assert.ok(sources.includes(`native/src/training/${file}.c`),file);
    assert.ok(!sources.some(p=>p.endsWith('training_control_wasm.c')));
  }
});

test('C inference and full keep distinct codec, WebGPU and training compile closures',()=>{
  for(const id of PROFILES) {
    const objects=wasmProfile(id).recipe.parent.objects;
    for(const object of objects) {
      assert.equal(object.optimization,/^numerical-/.test(object.category)?'-O3':'-Oz');
      assert.ok(object.flags.includes('-msimd128'));
      assert.ok(object.flags.includes('-DVOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC=1'));
      assert.ok(!object.flags.includes('-mrelaxed-simd'));
      assert.equal(object.flags.includes('-DVOLVOXAI_ENABLE_TRAINING=1'),id==='full');
      assert.ok(object.flags.includes(`-DVOLVOXAI_ENABLE_WEBGPU=${id==='full'?1:0}`));
      assert.ok(object.flags.includes(id==='full'?'-Iruntime/generated/c':'-Iruntime/generated/c/inference'));
    }
  }
});

test('compiler, linker and runtime libraries are pinned by their content',()=>{
  for(const id of PROFILES) {
    const t=wasmProfile(id).recipe.toolchain;
    assert.equal(t.compiler,'clang-17'); assert.equal(t.linker,'wasm-ld-17');
    assert.equal(t.compilerSha256,'6f673bb8b459659cdefed06f2348b3ad47a13fc84e5736df38a83ffc62532e70');
    assert.equal(t.linkerSha256,'aa716ebc4baaeb6a1bcf6cf0f875f59ad93d49590757a1c1c8a5acfbffa41702');
    for(const file of [...t.compilerRuntimeDependencies,...t.linkerRuntimeDependencies]) {
      assert.match(file.sha256,/^[0-9a-f]{64}$/); assert.ok(file.rawBytes>0); assert.ok(file.path.startsWith('/'));
    }
  }
});

test('inference imports clock, entropy and call wakeup; full also imports the device bridge',()=>{
  assert.deepEqual(WASM_PROFILE_IMPORTS.inference, [
    'host.vx_host_monotonic_micros_v1:function',
    'host.vx_host_random_u64_v1:function',
    'synurang.wakeup:function',
  ]);
  assert.equal(WASM_PROFILE_IMPORTS.full.length,14);
  assert.ok(WASM_PROFILE_IMPORTS.full.every(s=>s.startsWith('host.')||s.startsWith('gpu.')||s==='synurang.wakeup:function'));
  assert.match(WASM_INTERNAL_ABI_MANIFEST_SHA256,/^[0-9a-f]{64}$/);
  for(const section of [manifest.parentImports,manifest.parentExports]) {
    assert.equal(new Set(section.map(e=>e.name)).size,section.length);
    for(const entry of section) for(const field of ['name','owner','consumer','group','kind','profiles','tsSignature'])
      assert.ok(entry[field],`${entry.name}: ${field}`);
  }
});

test('generated ABI projections and link roots agree exactly',()=>{
  assert.equal(WASM_PROFILE_EXPORTS.inference.length,143);
  assert.equal(WASM_PROFILE_EXPORTS.full.length,239);
  for(const id of PROFILES) {
    const entries=manifest.parentExports.filter(e=>e.profiles.includes(id));
    assert.deepEqual([...WASM_PROFILE_EXPORTS[id]].sort(),entries.map(e=>e.name).sort());
    const roots=wasmProfile(id).recipe.parent.linkFlags.filter(s=>s.startsWith('--export'));
    assert.deepEqual(roots,WASM_PROFILE_EXPORTS[id].map(name=>name==='memory'?'--export-memory':`--export=${name}`));
    for(const entry of entries) assert.equal(WASM_PROFILE_EXPORT_KINDS[id][entry.name],entry.kind);
    for(const name of ['vx_wasm_mount_file_begin','vx_wasm_mount_file_write','vx_wasm_mount_file_finish','vx_wasm_mount_file_abort','vx_wasm_unmount_file'])
      assert.ok(WASM_PROFILE_EXPORTS[id].includes(name));
  }
  assert.ok(WASM_PROFILE_EXPORTS.inference.every(n=>WASM_PROFILE_EXPORTS.full.includes(n)));
  assert.ok(!WASM_PROFILE_EXPORTS.inference.some(n=>/training|ptq|gpu/.test(n)));
  assert.ok(WASM_PROFILE_EXPORTS.full.includes('vx_wasm_prepare_gpu_v1'));
  assert.ok(!WASM_PROFILE_EXPORTS.full.some(n=>n.startsWith('vx_training_control_')));
});

test('the transport consumes generated ABI declarations',()=>{
  const control=fs.readFileSync(path.join(ROOT,'ts/core/ModelControlWasm.ts'),'utf8');
  assert.match(control,/WASM_PROFILE_IMPORTS/);
  assert.match(control,/MODEL_CONTROL_WASM_REQUIRED_EXPORTS/);
  const source=fs.readFileSync(path.join(ROOT,'ts/generated/wasmInternalAbi.ts'),'utf8');
  assert.match(source,/export interface WasmInferenceParentExports/);
  assert.doesNotMatch(source,/export const WASM_PROFILE_EXPORTS/);
});

test('Makefile and package declarations preserve the central release composition',async()=>{
  const {errors}=await validateReleaseDeclarations(ROOT); assert.deepEqual(errors,[]);
});

test('generated ABI and freestanding math constants are fresh',()=>{
  for(const [program,args] of [[process.execPath,['tools/generate_wasm_internal_abi.mjs','--check']],['python3',['tools/generate_wasm_math.py','--check']]]) {
    const result=spawnSync(program,args,{cwd:ROOT,encoding:'utf8'});
    assert.equal(result.status,0,`${result.stdout}\n${result.stderr}`);
  }
});
