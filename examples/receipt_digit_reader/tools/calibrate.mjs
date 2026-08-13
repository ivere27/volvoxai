/*
 * Produce a calibration profile for the receipt digit reader's PTQ variant.
 *
 * Weights quantize data-free; activation affines can only come from running
 * the graph on real receipts and watching the ranges. VolvoxAI owns the
 * observation mechanism (tools/calibration/activation_observer.mjs); this tool
 * owns the model-specific part -- which images, how they are preprocessed, and
 * which package revision the profile is bound to.
 *
 * The emitted profile carries the exact sample count and a SHA-256 digest of
 * the preprocessed batch. `plan_runtime_ptq` rejects a profile whose graph
 * fingerprint or counts do not match, so the profile is only valid for the
 * prepared package it was measured on.
 *
 *   node examples/receipt_digit_reader/tools/calibrate.mjs \
 *     --package build/receipt-digit-reader-ptq/.prepared \
 *     --images calibration/*.jpg --out calibration.json
 */

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';

import { ModelLoader, VolvoxAI } from '../../../ts/index.js';
import {
  observationBatches,
  observeActivations,
  observeInputs,
} from '../../../tools/calibration/activation_observer.mjs';
import { decodeGrayscaleImageFile, normalizeReceiptPixels } from '../ReceiptDigitInput.js';

function fail(message) {
  throw new Error(`[receipt_digit_reader/calibrate] ${message}`);
}

function parseArguments(argv) {
  const options = {
    package: null, out: null, images: [], raw: null,
    backend: 'wasm', batch: 12, wasmUrl: null,
  };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    const value = () => {
      const next = argv[++index];
      if (next === undefined) fail(`${flag} needs a value`);
      return next;
    };
    if (flag === '--package') options.package = value();
    else if (flag === '--out') options.out = value();
    else if (flag === '--raw') options.raw = value();
    else if (flag === '--backend') options.backend = value();
    else if (flag === '--wasm-url') options.wasmUrl = value();
    else if (flag === '--batch') options.batch = Number(value());
    else if (flag === '--images') {
      while (index + 1 < argv.length && !argv[index + 1].startsWith('--')) {
        options.images.push(argv[++index]);
      }
    } else fail(`unknown option ${flag}`);
  }
  if (!options.package || !options.out) fail('--package and --out are required');
  if ((options.images.length === 0) === (options.raw === null)) {
    fail('supply exactly one of --images or --raw');
  }
  if (!Number.isInteger(options.batch) || options.batch <= 0) {
    fail('--batch must be a positive integer');
  }
  return options;
}

const fileFetch = async (source) => {
  const path = source.startsWith('file:') ? fileURLToPath(source) : source;
  const buffer = readFileSync(path);
  return {
    ok: true,
    text: async () => buffer.toString('utf8'),
    json: async () => JSON.parse(buffer.toString('utf8')),
    arrayBuffer: async () => buffer.buffer.slice(
      buffer.byteOffset, buffer.byteOffset + buffer.byteLength,
    ),
  };
};

export async function calibrate(options) {
  const document = JSON.parse(readFileSync(`${options.package}/graph.json`, 'utf8'));
  const inputName = Object.keys(document.inputs)[0];
  const inputShape = document.inputs[inputName].shape;
  if (inputShape.length !== 4 || inputShape[0] !== 1) {
    fail(`prepared package input must be [1, C, H, W]; found [${inputShape}]`);
  }
  const [, channels, height, width] = inputShape;

  const perSample = channels * height * width;
  const digest = createHash('sha256');
  const inputSets = [];
  const accept = (data, label) => {
    if (data.length !== perSample) {
      fail(`${label} produced ${data.length} values, expected ${perSample}`);
    }
    digest.update(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
    inputSets.push({ [inputName]: { data, shape: [...inputShape] } });
  };
  if (options.raw !== null) {
    // A producer pipeline that already owns the exact preprocessing can hand
    // over the normalized batch directly, which keeps this tool free of an
    // image-decoder dependency and removes one source of drift.
    const buffer = readFileSync(options.raw);
    if (buffer.byteLength % (perSample * Float32Array.BYTES_PER_ELEMENT) !== 0) {
      fail(`--raw payload is not a whole number of ${perSample}-value samples`);
    }
    const batch = new Float32Array(
      buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
    );
    for (let offset = 0; offset < batch.length; offset += perSample) {
      accept(batch.slice(offset, offset + perSample), `${options.raw}[${offset / perSample}]`);
    }
  }
  for (const path of options.images) {
    const pixels = await decodeGrayscaleImageFile(path, { width, height });
    accept(normalizeReceiptPixels(pixels), path);
  }
  if (inputSets.length === 0) fail('no calibration samples were produced');

  const runtime = await VolvoxAI.createRuntime({
    backends: [options.backend],
    ...(options.wasmUrl ? { wasmUrl: pathToFileURL(options.wasmUrl).href } : {}),
  });
  let observations = {};
  try {
    const logicalPackage = await ModelLoader.load(
      pathToFileURL(`${options.package}/model.safetensors`).href,
      { graphUrl: pathToFileURL(`${options.package}/graph.json`).href, fetch: fileFetch },
    );
    for (const set of inputSets) observeInputs(document, set, observations);
    // Promoting every intermediate at once would keep the whole graph live, so
    // the sweep runs in batches and merges ranges across them.
    const names = [...new Set([
      ...(document.outputs || []),
      ...(document.nodes || []).flatMap(
        (node) => Object.values(node.outputs || {}).map((port) => port?.tensor),
      ),
    ])].filter((name) => typeof name === 'string');
    for (const batch of observationBatches(names, options.batch)) {
      await observeActivations({
        logicalPackage, document, runtime,
        backend: options.backend, inputSets, names: batch, observations,
      });
    }
  } finally {
    await runtime.close();
  }

  return {
    format: 'volvoxai-receipt-digit-reader-calibration-v1',
    samples: inputSets.length,
    digest: digest.digest('hex'),
    backend: options.backend,
    ranges: observations,
  };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const options = parseArguments(process.argv.slice(2));
  const profile = await calibrate(options);
  writeFileSync(options.out, `${JSON.stringify(profile, null, 1)}\n`);
  process.stdout.write(
    `calibrated ${Object.keys(profile.ranges).length} tensors over ${profile.samples} images\n`,
  );
}
