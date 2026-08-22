#!/usr/bin/env node

import { createHash } from 'node:crypto';
import {
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import {
  basename,
  dirname,
  extname,
  join,
  resolve,
} from 'node:path';
import {
  fileURLToPath,
  pathToFileURL,
} from 'node:url';

import {
  createTinyReceiptSplitE2EBindings,
  createTinyReceiptSplitE2EFixtureManifest,
  runTinyReceiptSplitE2E,
  tinyReceiptSplitE2ERawBytes,
} from '../TinyReceiptSplitE2E.js';
import {
  TinyReceiptByteFallbackBPEVocab,
  TINY_RECEIPT_SPLIT_KV_PACKAGE_FORMAT,
} from '../TinyReceiptSplitSession.js';

const repository = fileURLToPath(new URL('../../../', import.meta.url));

function fail(message) {
  throw new Error(`[run_split_e2e] ${message}`);
}

function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index++) {
    const argument = values[index];
    if (!argument.startsWith('--')) fail(`unexpected positional argument '${argument}'.`);
    const equals = argument.indexOf('=');
    if (equals >= 0) {
      const name = argument.slice(2, equals);
      if (name.length === 0 || Object.hasOwn(result, name)) fail(`duplicate option '${name}'.`);
      result[name] = argument.slice(equals + 1);
      continue;
    }
    const name = argument.slice(2);
    if (name.length === 0 || Object.hasOwn(result, name)) fail(`duplicate option '${name}'.`);
    if (index + 1 < values.length && !values[index + 1].startsWith('--')) {
      result[name] = values[++index];
    } else {
      result[name] = true;
    }
  }
  const allowed = new Set([
    'backend', 'package', 'reference',
    'fixtures-dir', 'fixtures-only', 'wasm', 'out', 'no-reference',
  ]);
  for (const name of Object.keys(result)) {
    if (!allowed.has(name)) fail(`unknown option '--${name}'.`);
  }
  return result;
}

function requiredString(value, label) {
  if (typeof value !== 'string' || value.length === 0) fail(`${label} is required.`);
  return value;
}

function manifestFilename(value) {
  const source = requiredString(value, '--package');
  if (/^[a-z][a-z0-9+.-]*:/i.test(source)) {
    const url = new URL(source);
    return url.pathname.endsWith('/') ? new URL('package_manifest.json', url) : url;
  }
  const filename = resolve(source);
  return pathToFileURL(extname(filename) === '.json'
    ? filename
    : join(filename, 'package_manifest.json'));
}

function localFilename(value) {
  const source = requiredString(value, 'filename');
  if (/^[a-z][a-z0-9+.-]*:/i.test(source)) {
    const url = new URL(source);
    if (url.protocol !== 'file:') fail('only local file URLs are supported.');
    return fileURLToPath(url);
  }
  return resolve(source);
}

async function fileFetch(input) {
  try {
    const source = input instanceof Request ? input.url : String(input);
    const url = new URL(source);
    if (url.protocol !== 'file:') {
      return new Response('Only local package assets are accepted.', {
        status: 403,
        statusText: 'Forbidden',
      });
    }
    const bytes = await readFile(fileURLToPath(url));
    return new Response(bytes, { status: 200, statusText: 'OK' });
  } catch (error) {
    return new Response(String(error?.message || error), {
      status: 404,
      statusText: 'Not Found',
    });
  }
}

async function readJson(filename, label) {
  try {
    return JSON.parse(await readFile(filename, 'utf8'));
  } catch (error) {
    fail(`${label} is not readable JSON: ${error?.message || error}`);
  }
}

function assetPath(value, label) {
  if (typeof value !== 'string' || value.length === 0 ||
      value.includes('\\') || value.startsWith('/') ||
      value.split('/').some((part) => part === '' || part === '.' || part === '..')) {
    fail(`${label} must be a package-relative asset path.`);
  }
  return value;
}

function hasExactKeys(value, expected) {
  return value && typeof value === 'object' && !Array.isArray(value) &&
    Object.keys(value).sort().join('\0') === [...expected].sort().join('\0');
}

async function packageVocabulary(packageManifestUrl) {
  if (packageManifestUrl.protocol !== 'file:') fail('fixture emission requires a local package.');
  const filename = fileURLToPath(packageManifestUrl);
  const manifest = await readJson(filename, 'package manifest');
  if (manifest?.format !== TINY_RECEIPT_SPLIT_KV_PACKAGE_FORMAT) {
    fail('fixture emission requires an explicit-KV v2 package.');
  }
  const vocabPath = assetPath(manifest?.assets?.vocab?.path, 'manifest assets.vocab.path');
  const value = await readJson(join(dirname(filename), vocabPath), 'vocab');
  const tokenizer = manifest?.tokenizer;
  const tokenIds = tokenizer?.token_ids;
  if (!hasExactKeys(tokenizer, [
    'type', 'version', 'vocab_size', 'normalization', 'tokenizer_hash',
    'itos_key', 'merges_key', 'token_ids',
  ]) || tokenizer.type !== 'byte_fallback_bpe' || tokenizer.version !== 1 ||
      !hasExactKeys(tokenIds, ['pad', 'bos', 'eos', 'unk'])) {
    fail('package tokenizer must use the canonical byte_fallback_bpe v1 contract.');
  }
  return TinyReceiptByteFallbackBPEVocab.fromJSON(value, tokenIds, {
    vocabSize: tokenizer.vocab_size,
    normalization: tokenizer.normalization,
    tokenizerHash: tokenizer.tokenizer_hash,
  });
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function emitFixtures(directory, packageManifestUrl) {
  const vocab = await packageVocabulary(packageManifestUrl);
  const bindings = createTinyReceiptSplitE2EBindings({ vocab });
  const raw = tinyReceiptSplitE2ERawBytes(bindings);
  const names = {
    image: 'image.f32',
    question_ids: 'question_ids.i32',
    question_position_ids: 'question_position_ids.i32',
    family_ids: 'family_ids.i32',
    decoder_input_ids: 'decoder_input_ids.i32',
    position_ids: 'position_ids.i32',
    past_padding_mask: 'past_padding_mask.i32',
    ...Object.fromEntries(
      [...Array(4).keys()].flatMap((layer) => [
        [`past_k_${layer}`, `past_k_${layer}.f32`],
        [`past_v_${layer}`, `past_v_${layer}.f32`],
      ]),
    ),
  };
  const output = resolve(directory);
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output);
  try {
    const records = {};
    for (const [name, bytes] of Object.entries(raw)) {
      const path = names[name];
      await writeFile(join(output, path), bytes, { flag: 'wx' });
      records[name] = { path, bytes: bytes.byteLength, sha256: sha256(bytes) };
    }
    const manifest = await createTinyReceiptSplitE2EFixtureManifest(bindings, records);
    await writeFile(
      join(output, 'fixture_manifest.json'),
      `${JSON.stringify(manifest, null, 2)}\n`,
      { flag: 'wx' },
    );
    return manifest;
  } catch (error) {
    await rm(output, { recursive: true, force: true });
    throw error;
  }
}

async function emitReport(value, output) {
  const serialized = `${JSON.stringify(value, null, 2)}\n`;
  if (output) {
    const filename = localFilename(output);
    const directory = dirname(filename);
    const temporary = join(
      directory,
      `.${basename(filename)}.tmp-${process.pid}-${Date.now()}`,
    );
    await mkdir(directory, { recursive: true });
    try {
      await writeFile(temporary, serialized, { flag: 'wx' });
      await rename(temporary, filename);
    } finally {
      await rm(temporary, { force: true });
    }
  }
  process.stdout.write(serialized);
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const backend = options.backend || 'cpu-js';
  if (!['cpu-js', 'wasm'].includes(backend)) {
    fail("--backend must be 'cpu-js' or 'wasm'; use the Deno runner for physical WebGPU.");
  }
  const packageManifestUrl = manifestFilename(options.package);
  if (options['fixtures-only'] && !options['fixtures-dir']) {
    fail('--fixtures-only requires --fixtures-dir.');
  }
  const fixture = options['fixtures-dir']
    ? await emitFixtures(requiredString(options['fixtures-dir'], '--fixtures-dir'), packageManifestUrl)
    : null;
  if (options['fixtures-only']) {
    await emitReport({
      schema: fixture.schema,
      status: 'pass',
      emitted: true,
      files: Object.keys(fixture.tensors).map((name) => basename(fixture.tensors[name].path)),
      manifest: 'fixture_manifest.json',
    }, options.out);
    return;
  }
  if (Boolean(options.reference) === Boolean(options['no-reference'])) {
    fail('pass exactly one of --reference <v1-oracle> or --no-reference.');
  }
  const reference = options['no-reference']
    ? null
    : await readJson(localFilename(options.reference), 'reference');

  const packageMetadata = JSON.parse(await readFile(join(repository, 'package.json'), 'utf8'));
  const api = await import(pathToFileURL(
    join(repository, 'dist', packageMetadata.version, 'volvoxai.js'),
  ).href);
  const wasmUrl = pathToFileURL(options.wasm
    ? localFilename(options.wasm)
    : join(repository, 'dist', packageMetadata.version, 'volvoxai.wasm'));

  const report = await runTinyReceiptSplitE2E({
    api,
    backend,
    packageUrl: packageManifestUrl,
    fetch: fileFetch,
    wasmUrl,
    reference,
  });

  await emitReport({
    ...report,
    fixture: fixture == null
      ? null
      : {
        schema: fixture.schema,
        emitted: true,
        files: Object.keys(fixture.tensors).map((name) => basename(fixture.tensors[name].path)),
      },
  }, options.out);
}

main().catch((error) => {
  console.error(JSON.stringify({
    schema: 'volvoxai.tiny-receipt-split-e2e-result/v1',
    status: 'fail',
    error: {
      name: error?.name || 'Error',
      message: error?.message || String(error),
      report: error?.report || null,
    },
  }, null, 2));
  process.exitCode = 1;
});
