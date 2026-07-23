// Native-GPU producer for the L1 op / L2 graph matrices. Runs each authored
// package through the native binary with a backend flag (--vulkan/--opengl/--metal)
// and writes a signature per case in the format the matrices consume:
//   <sigDir>/<id>.<tier>.json.  Run on a box with the matching GPU loader.
//
//   node tests/parity/native_matrix.mjs <packagesDir> <sigDir> <tier> <flag>
//   e.g.  ... out/opcases   out/ops    native-vulkan --vulkan
//         ... out/graphcases out/graphsigs native-opengl --opengl
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { signature } from './lib/extract.mjs';
import { validateNativeRuntimeEvidence } from './lib/artifact.mjs';
import { requirePhysicalNativeGpuLog } from './lib/backend.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const [pkgDirArg, sigDirArg, tier, flag] = process.argv.slice(2);
if (!pkgDirArg || !sigDirArg || !tier || !flag) {
  console.error('usage: native_matrix.mjs <packagesDir> <sigDir> <tier> <flag(--vulkan|--opengl|--metal)>');
  process.exit(2);
}
const pkgDir = path.isAbsolute(pkgDirArg) ? pkgDirArg : path.join(ROOT, pkgDirArg);
const sigDir = path.isAbsolute(sigDirArg) ? sigDirArg : path.join(ROOT, sigDirArg);
const nativeBin = path.join(ROOT, 'native', 'volvoxai');
if (!fs.existsSync(nativeBin)) { console.error(`no native binary at ${nativeBin}`); process.exit(2); }
fs.mkdirSync(sigDir, { recursive: true });

const ROUTES = {
  'native-vulkan': { flag: '--vulkan', reported: 'vulkan' },
  'native-opengl': { flag: '--opengl', reported: 'opengl' },
  'native-metal': { flag: '--metal', reported: 'metal' },
};
const route = ROUTES[tier];
if (!route || flag !== route.flag) {
  console.error(`tier/flag must be one of: ${Object.entries(ROUTES).map(([name, value]) => `${name} ${value.flag}`).join(', ')}`);
  process.exit(2);
}

const DTYPES = {
  float32: { Array: Float32Array, bytes: 4, ext: 'f32', finite: true },
  int32: { Array: Int32Array, bytes: 4, ext: 'i32', finite: false },
  int8: { Array: Int8Array, bytes: 1, ext: 'i8', finite: false },
  uint8: { Array: Uint8Array, bytes: 1, ext: 'u8', finite: false },
};

function atomicWriteJson(file, value) {
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(value)}\n`, { flag: 'wx' });
    fs.renameSync(temporary, file);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}

function assertReportedBackend(stdout) {
  const matches = [...stdout.matchAll(/^Backend: ([^\r\n]+)\r?$/gm)].map((match) => match[1]);
  if (matches.length !== 1 || matches[0] !== route.reported) {
    throw new Error(`native CLI reported backend ${JSON.stringify(matches)}, expected '${route.reported}'`);
  }
  if (['vulkan', 'opengl'].includes(route.reported)) {
    return requirePhysicalNativeGpuLog(stdout, route.reported, `${tier} matrix`);
  }
  return null;
}

function outputDescriptor(meta) {
  const dtype = DTYPES[meta.outputDtype];
  if (!dtype || !Array.isArray(meta.outputShape) ||
      meta.outputShape.some((dim) => !Number.isSafeInteger(dim) || dim <= 0)) {
    throw new Error('meta.json must declare a valid outputShape and supported outputDtype');
  }
  const elements = meta.outputShape.reduce((count, dim) => count * dim, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) throw new Error('declared output element count is invalid');
  return { ...dtype, elements, sizeBytes: elements * dtype.bytes };
}

// Inputs were materialized as inputs/<name>.<ext>; find whichever dtype exists.
function inputFile(pkg, name) {
  for (const ext of ['f32', 'i32', 'i8', 'u8']) {
    const f = path.join(pkg, 'inputs', `${name}.${ext}`);
    if (fs.existsSync(f)) return f;
  }
  return null;
}

const ids = fs.readdirSync(pkgDir)
  .filter((d) => fs.existsSync(path.join(pkgDir, d, 'meta.json')))
  .sort();
for (const id of ids) {
  const pkg = path.join(pkgDir, id);
  fs.rmSync(path.join(sigDir, `${id}.${tier}.json`), { force: true });
  for (const ext of ['f32', 'i32', 'i8', 'u8']) {
    fs.rmSync(path.join(pkg, `${tier}_y.${ext}`), { force: true });
  }
}

let ok = 0, skipped = 0, failed = 0;
for (const id of ids) {
  const pkg = path.join(pkgDir, id);
  let stagedOutput = null;
  let stagedEvidence = null;
  try {
    const meta = JSON.parse(fs.readFileSync(path.join(pkg, 'meta.json'), 'utf8'));
    if (meta.id !== id) throw new Error(`meta id '${meta.id}' does not match package '${id}'`);
    if (meta.skip?.includes(tier)) {
      console.log(`  ${tier} ${id}: declared skip`);
      skipped++;
      continue;
    }
    const output = outputDescriptor(meta);
    const args = ['run', pkg];
    for (const name of meta.inputs) {
      const f = inputFile(pkg, name);
      if (!f) throw new Error(`no input file for ${name}`);
      args.push('--input', `${name}=${f}`);
    }
    const outf = path.join(pkg, `${tier}_y.${output.ext}`);
    stagedOutput = path.join(pkg, `.${tier}_y.${process.pid}.${randomUUID()}.tmp.${output.ext}`);
    stagedEvidence = path.join(
      pkg,
      `.${tier}_runtime.${process.pid}.${randomUUID()}.tmp.json`,
    );
    args.push(
      '--output', `y=${stagedOutput}`,
      flag,
      '--report-json', stagedEvidence,
    );
    const stdout = execFileSync(nativeBin, args, { encoding: 'utf8' });
    const adapterInfo = assertReportedBackend(stdout);
    const runtimeEvidence = JSON.parse(fs.readFileSync(stagedEvidence, 'utf8'));
    validateNativeRuntimeEvidence(runtimeEvidence, route.reported);
    const b = fs.readFileSync(stagedOutput);
    if (b.byteLength !== output.sizeBytes) {
      throw new Error(`output has ${b.byteLength} bytes, expected ${output.sizeBytes}`);
    }
    const y = new output.Array(b.buffer, b.byteOffset, output.elements);
    if (output.finite) {
      const bad = y.findIndex((value) => !Number.isFinite(value));
      if (bad !== -1) throw new Error(`output is non-finite at index ${bad}`);
    }
    fs.renameSync(stagedOutput, outf);
    stagedOutput = null;
    atomicWriteJson(path.join(sigDir, `${id}.${tier}.json`), {
      id,
      tier,
      backend: route.reported,
      ...(adapterInfo ? { adapterInfo } : {}),
      sig: signature(y, { topk: 0, shape: meta.outputShape }),
      runtimeEvidence,
    });
    fs.rmSync(stagedEvidence, { force: true });
    stagedEvidence = null;
    ok++;
  } catch (error) {
    if (stagedOutput) fs.rmSync(stagedOutput, { force: true });
    if (stagedEvidence) fs.rmSync(stagedEvidence, { force: true });
    failed++;
    console.error(`  ${tier} ${id}: ERROR ${String(error.message || error).split('\n')[0].slice(0, 240)}`);
  }
}
console.log(`${tier} matrix: ${ok} ok, ${skipped} declared skip, ${failed} failed -> ${path.relative(ROOT, sigDir)}`);
if (ok > 0) {
  console.log(`  route evidence: strict '${route.reported}' tier and no operator fallback attested`);
} else if (failed > 0) {
  console.error(`  no successful '${route.reported}' execution produced route evidence`);
}
if (failed > 0) process.exit(1);
