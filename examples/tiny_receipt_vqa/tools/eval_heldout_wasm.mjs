#!/usr/bin/env node
/**
 * Held-out answer accuracy for the WASM INT8 deployment path.
 *
 * The kernel work is mostly bit-exact, but "bit-exact on a synthetic buffer" is
 * not the same claim as "this model still answers correctly". This runs real
 * held-out records through the shipping WASM backend and scores the extracted
 * answer against the annotation, so a change that silently degrades quality is
 * caught by the same measure the release reports.
 *
 * It is the gate any change that alters numerics by design — a QDQ fusion, a
 * kernel rewrite — has to clear before it lands.
 *
 *   node --import tsx examples/tiny_receipt_vqa/tools/eval_heldout_wasm.mjs \
 *     --package /tmp/volvoxai-bpe1536-int8-canonical-root \
 *     --eval "$RECEIPT_VQA_DATA_ROOT/eval/heldout" --count 200
 *
 * `--out results.json` writes per-record answers so two builds can be diffed
 * exactly rather than compared only on their aggregate score.
 */
import { execFileSync } from 'node:child_process';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

const repository = fileURLToPath(new URL('../../../', import.meta.url));

function parseArgs(argv) {
  const options = {
    package: '/tmp/volvoxai-bpe1536-int8-canonical-root',
    eval: process.env.RECEIPT_VQA_DATA_ROOT
      ? join(process.env.RECEIPT_VQA_DATA_ROOT, 'eval', 'heldout') : null,
    wasm: `${repository}dist/0.4.0/volvoxai.wasm`,
    count: 200, tokens: 191, out: null, start: 0,
  };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index], value = argv[index + 1];
    if (flag === '--package') { options.package = value; index++; }
    else if (flag === '--eval') { options.eval = value; index++; }
    else if (flag === '--wasm') { options.wasm = value; index++; }
    else if (flag === '--count') { options.count = Number(value); index++; }
    else if (flag === '--start') { options.start = Number(value); index++; }
    else if (flag === '--tokens') { options.tokens = Number(value); index++; }
    else if (flag === '--out') { options.out = value; index++; }
    else throw new Error(`unknown argument '${flag}'`);
  }
  if (!options.eval) {
    throw new Error('pass --eval <dir> or set RECEIPT_VQA_DATA_ROOT');
  }
  return options;
}

const options = parseArgs(process.argv.slice(2));

// `ReadOnlySafetensorsCache` is not part of the published entrypoint, so the
// shared-weight fast path is unavailable here; each graph loads its own store.
const { Model, ModelLoader, VolvoxAI } =
  await import(`${repository}ts/index.js`);
const { TinyReceiptSplitSession } =
  await import(`${repository}examples/tiny_receipt_vqa/TinyReceiptSplitSession.js`);
const { preprocessTinyReceiptImage } =
  await import(`${repository}examples/tiny_receipt_vqa/TinyReceiptW8A8Session.js`);

async function fileFetch(input) {
  const url = input instanceof URL ? input : new URL(String(input));
  try {
    return new Response(await readFile(url), { status: 200, statusText: 'OK' });
  } catch (error) {
    return new Response(String(error), { status: 404, statusText: 'Not Found' });
  }
}

/* Same Pillow RGB decode boundary the release harness uses. */
const decodeScript = [
  'from PIL import Image',
  'import struct,sys',
  "im=Image.open(sys.argv[1]).convert('RGB')",
  "sys.stdout.buffer.write(struct.pack('<II', *im.size))",
  'sys.stdout.buffer.write(im.tobytes())',
].join('\n');
function decodeImage(path) {
  const raw = execFileSync('python3', ['-c', decodeScript, path],
    { maxBuffer: 64 * 1024 * 1024 });
  return {
    data: new Uint8Array(raw.buffer, raw.byteOffset + 8, raw.byteLength - 8),
    width: raw.readUInt32LE(0), height: raw.readUInt32LE(4), channels: 3,
  };
}

const REFERENCE_FAMILY = { phone: 'phone_number' };

const answerPattern = /<answer>([\s\S]*?)<\/answer>/;
const extractAnswer = (text) => (answerPattern.exec(text ?? '')?.[1] ?? '').trim();

const runtime = await VolvoxAI.createRuntime({
  backends: ['wasm'], wasmUrl: pathToFileURL(options.wasm),
});
const session = await TinyReceiptSplitSession.load({
  runtime,
  packageUrl: pathToFileURL(`${options.package}/package_manifest.json`),
  fetch: fileFetch, decodePolicy: 'required',
  compileOptions: {
    backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
  },
  snapshotLoader: async ({ weightsUrl, graphUrl, fetch }) => Model.capture(
    await ModelLoader.load(weightsUrl, { graphUrl, fetch }),
  ),
});

const names = (await readdir(join(options.eval, 'annotations'))).sort();
const selected = names.slice(options.start, options.start + options.count);
const byFamily = new Map();
const records = [];
let correct = 0;
const started = performance.now();

for (let index = 0; index < selected.length; index++) {
  const annotation = JSON.parse(
    await readFile(join(options.eval, 'annotations', selected[index]), 'utf8'));
  const imagePath = join(options.eval, 'images', `${annotation.id}.jpg`);
  const image = await preprocessTinyReceiptImage(
    decodeImage(imagePath), session.package.preprocessing);
  const result = await session.generate({
    image, prompt: annotation.question, family: 'auto',
    preprocessed: true, maxNewTokens: options.tokens,
  });
  const predicted = extractAnswer(result.text);
  const expected = String(annotation.answer ?? '').trim();
  const exact = predicted === expected;
  correct += exact ? 1 : 0;

  /* The reference report names this bucket `phone_number`; the router emits
     `phone`. Normalise so the two summaries can be compared field by field. */
  const family = REFERENCE_FAMILY[result.family] ?? result.family ?? 'unknown';
  const bucket = byFamily.get(family) ?? { n: 0, correct: 0 };
  bucket.n++; bucket.correct += exact ? 1 : 0;
  byFamily.set(family, bucket);

  records.push({ id: annotation.id, family, expected, predicted, exact, text: result.text });
  if ((index + 1) % 25 === 0 || index + 1 === selected.length) {
    process.stderr.write(
      `${index + 1}/${selected.length}  answer_exact ${(correct / (index + 1)).toFixed(4)}\n`);
  }
}

const seconds = (performance.now() - started) / 1000;
const summary = {
  format: 'tiny_receipt_vqa_wasm_heldout_eval_v1',
  package: options.package, records: selected.length, tokens: options.tokens,
  wall_seconds: seconds, seconds_per_record: seconds / selected.length,
  overall: { n: selected.length, answer_exact: correct / selected.length },
  by_family: Object.fromEntries([...byFamily].map(([family, bucket]) =>
    [family, { n: bucket.n, answer_exact: bucket.correct / bucket.n }])),
};
console.log(JSON.stringify(summary, null, 2));
if (options.out) {
  await writeFile(options.out, JSON.stringify({ ...summary, records }, null, 2));
  process.stderr.write(`wrote ${options.out}\n`);
}
