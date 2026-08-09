#!/usr/bin/env node
/**
 * One record, one definition of "total inference time", every JS-reachable backend.
 *
 * Cross-backend numbers elsewhere in this repository were collected by different
 * harnesses with different lifecycles, so they cannot be compared. This measures
 * exactly one heldout record on each backend with one clock and one boundary:
 *
 *   total = encoder execution + every decoder step to EOS
 *
 * Model load and graph compilation are reported separately and excluded from the
 * total, because they are amortized once per process rather than per request.
 * The generated text is reported too: a latency number for a wrong answer is not
 * comparable to one for a right answer.
 *
 *   node --import tsx examples/tiny_receipt_vqa/tools/measure_backend_latency.mjs \
 *     --package build/tiny-receipt-f32-from-f32 \
 *     --eval "$RECEIPT_VQA_DATA_ROOT/eval/heldout" --record 00000 \
 *     --backend cpu --backend wasm --out build/backend-latency.json
 */

import { execFileSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

const repository = fileURLToPath(new URL('../../../', import.meta.url));

function parseArguments(values) {
  const options = {
    record: '00000', tokens: 191, backends: [], out: null,
    wasm: `${repository}dist/0.4.0/volvoxai.wasm`, repeat: 1,
  };
  for (let index = 0; index < values.length; index++) {
    const flag = values[index];
    const value = values[index + 1];
    if (flag === '--package') { options.package = value; index++; }
    else if (flag === '--eval') { options.eval = value; index++; }
    else if (flag === '--record') { options.record = value; index++; }
    else if (flag === '--tokens') { options.tokens = Number(value); index++; }
    else if (flag === '--backend') { options.backends.push(value); index++; }
    else if (flag === '--wasm') { options.wasm = value; index++; }
    else if (flag === '--repeat') { options.repeat = Number(value); index++; }
    else if (flag === '--out') { options.out = value; index++; }
    else throw new Error(`unknown option '${flag}'`);
  }
  if (!options.package || !options.eval) {
    throw new Error('pass --package <dir> --eval <heldout dir>');
  }
  if (options.backends.length === 0) options.backends = ['cpu', 'wasm'];
  return options;
}

const { Model, ModelLoader, VolvoxAI } = await import(`${repository}ts/index.js`);
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

const answerPattern = /<answer>([\s\S]*?)<\/answer>/;
const extractAnswer = (text) => (answerPattern.exec(text ?? '')?.[1] ?? '').trim();

async function measure(options, backend, annotation, imagePath) {
  const runtime = await VolvoxAI.createRuntime(
    backend === 'wasm'
      ? { backends: ['wasm'], wasmUrl: pathToFileURL(options.wasm) }
      : { backends: [backend] },
  );
  const loadStarted = performance.now();
  const session = await TinyReceiptSplitSession.load({
    runtime,
    packageUrl: pathToFileURL(`${options.package}/package_manifest.json`),
    fetch: fileFetch, decodePolicy: 'required',
    compileOptions: {
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    },
    snapshotLoader: async ({ graphUrl, weightsUrl, fetch }) => Model.capture(
      await ModelLoader.load(weightsUrl, { graphUrl, fetch }),
    ),
  });
  const loadMs = performance.now() - loadStarted;

  const image = await preprocessTinyReceiptImage(
    decodeImage(imagePath), session.package.preprocessing);

  const runs = [];
  for (let repeat = 0; repeat < options.repeat; repeat++) {
    const started = performance.now();
    const result = await session.generate({
      image, prompt: annotation.question, family: 'auto',
      preprocessed: true, maxNewTokens: options.tokens,
    });
    const totalMs = performance.now() - started;
    runs.push({
      repeat,
      total_ms: totalMs,
      tokens: result.tokens?.length ?? null,
      family: result.family ?? null,
      text: result.text,
      answer: extractAnswer(result.text),
    });
    process.stderr.write(
      `${backend} repeat ${repeat + 1}/${options.repeat}: `
      + `${totalMs.toFixed(1)} ms -> ${extractAnswer(result.text)}\n`,
    );
  }
  await session.close?.();
  await runtime.close();
  return { backend, load_and_compile_ms: loadMs, runs };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const annotation = JSON.parse(await readFile(
    join(options.eval, 'annotations', `${options.record}.json`), 'utf8'));
  const imagePath = join(options.eval, 'images', `${options.record}.jpg`);

  const results = [];
  for (const backend of options.backends) {
    try {
      results.push(await measure(options, backend, annotation, imagePath));
    } catch (error) {
      results.push({ backend, error: error?.message ?? String(error) });
      process.stderr.write(`${backend}: FAILED ${error?.message ?? error}\n`);
    }
  }

  const report = {
    schema: 'volvoxai.tiny-receipt-backend-latency/v1',
    boundary: 'encoder execution plus every decoder step to EOS; '
      + 'model load and graph compilation excluded',
    package: options.package,
    record: options.record,
    question: annotation.question,
    truth: String(annotation.answer ?? '').trim(),
    max_new_tokens: options.tokens,
    results,
  };
  const text = `${JSON.stringify(report, null, 2)}\n`;
  if (options.out) await writeFile(options.out, text);
  else process.stdout.write(text);
  for (const item of results) {
    if (item.error) { console.log(`${item.backend}: ERROR ${item.error}`); continue; }
    const median = [...item.runs].sort((a, b) => a.total_ms - b.total_ms)[
      Math.floor(item.runs.length / 2)];
    console.log(
      `${item.backend}: total ${median.total_ms.toFixed(1)} ms, `
      + `${median.tokens} tokens, answer ${JSON.stringify(median.answer)}, `
      + `load+compile ${item.load_and_compile_ms.toFixed(1)} ms`,
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
