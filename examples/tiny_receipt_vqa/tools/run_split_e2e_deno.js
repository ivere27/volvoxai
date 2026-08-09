#!/usr/bin/env -S deno run --allow-read --allow-write --unstable-webgpu

import {
  ModelLoader,
  Model,
  VolvoxAI,
} from '../../../dist/0.4.0/volvoxai.js';
import {
  requireTinyReceiptPhysicalAdapterIdentity,
  runTinyReceiptSplitE2E,
} from '../TinyReceiptSplitE2E.js';

function fail(message) {
  throw new Error(`[run_split_e2e_deno] ${message}`);
}

function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index++) {
    const argument = values[index];
    if (!argument.startsWith('--')) fail(`unexpected positional argument '${argument}'.`);
    const equals = argument.indexOf('=');
    if (equals >= 0) {
      const name = argument.slice(2, equals);
      if (!name || Object.hasOwn(result, name)) fail(`duplicate option '${name}'.`);
      result[name] = argument.slice(equals + 1);
      continue;
    }
    const name = argument.slice(2);
    if (!name || Object.hasOwn(result, name)) fail(`duplicate option '${name}'.`);
    if (index + 1 < values.length && !values[index + 1].startsWith('--')) {
      result[name] = values[++index];
    } else {
      result[name] = true;
    }
  }
  for (const name of Object.keys(result)) {
    if (![
      'backend', 'package', 'reference', 'no-reference', 'out', 'adapter', 'require-adapter',
    ].includes(name)) {
      fail(`unknown option '--${name}'.`);
    }
  }
  return result;
}

function requiredString(value, label) {
  if (typeof value !== 'string' || value.length === 0) fail(`${label} is required.`);
  return value;
}

function fileUrl(value, label) {
  const source = requiredString(value, label);
  if (/^[a-z][a-z0-9+.-]*:/i.test(source)) {
    const result = new URL(source);
    if (result.protocol !== 'file:') fail(`${label} must identify a local file.`);
    return result;
  }
  const filename = source.startsWith('/') ? source : `${Deno.cwd()}/${source}`;
  const result = new URL('file:///');
  result.pathname = filename;
  return result;
}

function packageUrl(value) {
  const source = fileUrl(value, '--package');
  return source.pathname.endsWith('/')
    ? new URL('package_manifest.json', source)
    : source.pathname.endsWith('.json')
      ? source
      : new URL(`${source.href.replace(/\/?$/, '/')}package_manifest.json`);
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
    return new Response(await Deno.readFile(url), { status: 200, statusText: 'OK' });
  } catch (error) {
    return new Response(String(error?.message || error), {
      status: 404,
      statusText: 'Not Found',
    });
  }
}

async function emitReport(report, output) {
  const serialized = `${JSON.stringify(report, null, 2)}\n`;
  if (output) {
    const filename = fileUrl(output, '--out');
    const directory = new URL('.', filename);
    const leaf = filename.pathname.slice(filename.pathname.lastIndexOf('/') + 1);
    const temporary = new URL(
      `.${leaf}.tmp-${Deno.pid}-${crypto.randomUUID()}`,
      directory,
    );
    await Deno.mkdir(directory, { recursive: true });
    try {
      await Deno.writeTextFile(temporary, serialized, { createNew: true });
      await Deno.rename(temporary, filename);
    } finally {
      try {
        await Deno.remove(temporary);
      } catch (error) {
        if (!(error instanceof Deno.errors.NotFound)) throw error;
      }
    }
  }
  await Deno.stdout.write(new TextEncoder().encode(serialized));
}

async function main() {
  const options = parseArguments(Deno.args);
  const backend = options.backend || 'webgpu';
  if (backend !== 'webgpu') fail("--backend must be 'webgpu'.");
  const adapterPreference = options.adapter || 'high-performance';
  if (!['default', 'high-performance', 'low-power'].includes(adapterPreference)) {
    fail("--adapter must be 'default', 'high-performance', or 'low-power'.");
  }
  const requiredAdapter = requiredString(options['require-adapter'], '--require-adapter');
  const manifest = packageUrl(options.package);
  if (Boolean(options.reference) === Boolean(options['no-reference'])) {
    fail('pass exactly one of --reference <v1-oracle> or --no-reference.');
  }
  const reference = options['no-reference']
    ? null
    : JSON.parse(await Deno.readTextFile(fileUrl(options.reference, '--reference')));
  const gpu = navigator.gpu;
  if (!gpu) fail('WebGPU is unavailable.');
  const originalRequestAdapter = gpu.requestAdapter;
  if (adapterPreference !== 'default') {
    const bound = originalRequestAdapter.bind(gpu);
    gpu.requestAdapter = (request = {}) => bound({
      ...request,
      powerPreference: adapterPreference,
    });
  }
  let report;
  try {
    report = await runTinyReceiptSplitE2E({
      api: { ModelLoader, Model, VolvoxAI },
      backend,
      packageUrl: manifest,
      fetch: fileFetch,
      reference,
    });
  } finally {
    gpu.requestAdapter = originalRequestAdapter;
  }
  if (!Array.isArray(report.provider.devices) || report.provider.devices.length !== 1) {
    fail('strict WebGPU run must report one identical encoder/decoder adapter identity.');
  }
  requireTinyReceiptPhysicalAdapterIdentity(report.provider.devices[0], requiredAdapter);
  await emitReport(report, options.out);
}

try {
  await main();
} catch (error) {
  console.error(JSON.stringify({
    schema: 'volvoxai.tiny-receipt-split-e2e-result/v1',
    status: 'fail',
    error: {
      name: error?.name || 'Error',
      message: error?.message || String(error),
      report: error?.report || null,
    },
  }, null, 2));
  Deno.exitCode = 1;
}
