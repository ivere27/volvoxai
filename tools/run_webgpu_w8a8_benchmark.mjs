// Zero-dependency driver for tools/webgpu_w8a8_benchmark.html.
//
// Hardware (default):
//   node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs
// SwiftShader correctness fallback:
//   node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs --adapter=swiftshader
// Require the packed integer-dot WGSL feature:
//   node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs --require-dot

import { spawn } from 'node:child_process';
import { accessSync, constants, readFileSync, rmSync, statSync } from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, '..');
const rawArguments = process.argv.slice(2);

function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; ++index) {
    const argument = values[index];
    if (!argument.startsWith('--')) throw new Error(`unexpected argument: ${argument}`);
    const equals = argument.indexOf('=');
    if (equals >= 0) {
      result[argument.slice(2, equals)] = argument.slice(equals + 1);
      continue;
    }
    const name = argument.slice(2);
    if (index + 1 < values.length && !values[index + 1].startsWith('--')) result[name] = values[++index];
    else result[name] = true;
  }
  return result;
}

const options = parseArguments(rawArguments);
const adapterMode = options.adapter || 'hardware';
if (!['hardware', 'swiftshader'].includes(adapterMode)) {
  throw new Error(`--adapter must be hardware or swiftshader, got ${adapterMode}`);
}

function integerOption(name, fallback, minimum, maximum) {
  if (options[name] == null) return fallback;
  const value = Number(options[name]);
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer in [${minimum}, ${maximum}]`);
  }
  return value;
}

const timeoutMs = integerOption('timeout-ms', 180000, 1000, 1800000);
const warmup = integerOption('warmup', 1, 0, 20);
const iterations = integerOption('iterations', 20, 1, 50);
const requestedServerPort = integerOption('port', 0, 0, 65535);
const requestedDebugPort = options['debug-port'] == null ? null : integerOption('debug-port', 0, 1, 65535);
const requireDot = Boolean(options['require-dot']);
const modelDirectory = options['model-dir'] ? resolve(options['model-dir']) : null;
const imageFilename = options.image ? resolve(options.image) : null;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.wasm': 'application/wasm',
  '.wgsl': 'text/plain; charset=utf-8',
};

function serve(root, port) {
  const server = http.createServer((request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      let allowedRoot = root;
      let filename;
      if (pathname.startsWith('/__benchmark_model__/') && modelDirectory) {
        allowedRoot = modelDirectory;
        filename = resolve(modelDirectory, pathname.slice('/__benchmark_model__/'.length));
      } else if (pathname === '/__benchmark_image__' && imageFilename) {
        allowedRoot = dirname(imageFilename);
        filename = imageFilename;
      } else {
        filename = resolve(root, pathname.replace(/^\/+/, ''));
      }
      if (filename !== allowedRoot && !filename.startsWith(allowedRoot + sep)) {
        response.writeHead(403).end('403');
        return;
      }
      if (statSync(filename).isDirectory()) filename = resolve(filename, 'index.html');
      response.writeHead(200, {
        'content-type': MIME[extname(filename)] || 'application/octet-stream',
        'cache-control': 'no-store',
      });
      response.end(readFileSync(filename));
    } catch {
      response.writeHead(404).end('404');
    }
  });
  return new Promise((accept, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () => accept(server));
  });
}

function reservePort() {
  return new Promise((accept, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      server.close((error) => error ? reject(error) : accept(port));
    });
  });
}

function findChrome() {
  const explicit = options.chrome || process.env.CHROME_BIN;
  if (explicit) return explicit;
  const absoluteCandidates = [
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ];
  for (const candidate of absoluteCandidates) {
    try {
      accessSync(candidate, constants.X_OK);
      return candidate;
    } catch { /* Try the next known path. */ }
  }
  return 'google-chrome';
}

function delay(milliseconds) {
  return new Promise((accept) => setTimeout(accept, milliseconds));
}

async function cdpTargets(hostAndPort) {
  const response = await fetch(`http://${hostAndPort}/json`, { signal: AbortSignal.timeout(1500) });
  if (!response.ok) throw new Error(`CDP target list returned HTTP ${response.status}`);
  return response.json();
}

async function waitForPage(hostAndPort, chrome, chromeError) {
  const deadline = Date.now() + Math.min(timeoutMs, 30000);
  while (Date.now() < deadline) {
    if (chromeError.value) throw chromeError.value;
    if (chrome.exitCode != null) throw new Error(`Chrome exited early with code ${chrome.exitCode}`);
    try {
      const targets = await cdpTargets(hostAndPort);
      const page = targets.find((target) => target.type === 'page' && /webgpu_w8a8_benchmark/.test(target.url));
      if (page?.webSocketDebuggerUrl) return page;
    } catch { /* Chrome may not have opened its debugging endpoint yet. */ }
    await delay(250);
  }
  throw new Error('timed out waiting for the benchmark page target');
}

async function pollResult(webSocketUrl) {
  if (typeof WebSocket === 'undefined') {
    throw new Error('global WebSocket is unavailable; rerun Node with --experimental-websocket or use Node 22+');
  }
  const socket = new WebSocket(webSocketUrl);
  await new Promise((accept, reject) => {
    socket.onopen = accept;
    socket.onerror = () => reject(new Error('failed to open the Chrome DevTools WebSocket'));
  });
  let messageId = 0;
  const pending = new Map();
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    const request = pending.get(message.id);
    if (request) {
      pending.delete(message.id);
      request.accept(message);
    }
  };
  socket.onclose = () => {
    for (const request of pending.values()) request.reject(new Error('Chrome DevTools WebSocket closed'));
    pending.clear();
  };
  const send = (method, params = {}) => new Promise((accept, reject) => {
    if (socket.readyState !== WebSocket.OPEN) {
      reject(new Error('Chrome DevTools WebSocket is not open'));
      return;
    }
    const id = ++messageId;
    pending.set(id, { accept, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  await send('Runtime.enable');
  const expression = `(() => ({
    result: window.__webgpuW8A8BenchmarkResult || null,
    text: document.getElementById('out')?.innerText || ''
  }))()`;
  const deadline = Date.now() + timeoutMs;
  let lastValue = { result: null, text: '' };
  try {
    while (Date.now() < deadline) {
      const response = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
      if (response.result?.exceptionDetails) {
        throw new Error(response.result.exceptionDetails.text || 'Runtime.evaluate failed');
      }
      lastValue = response.result?.result?.value || lastValue;
      if (lastValue.result?.status === 'pass' || lastValue.result?.status === 'fail') return lastValue;
      await delay(300);
    }
  } finally {
    socket.close();
  }
  throw new Error(`benchmark timed out after ${timeoutMs} ms\n${lastValue.text}`);
}

async function stopChrome(chrome) {
  if (chrome.exitCode != null) return;
  chrome.kill('SIGTERM');
  await Promise.race([
    new Promise((accept) => chrome.once('exit', accept)),
    delay(1000),
  ]);
  if (chrome.exitCode == null) chrome.kill('SIGKILL');
}

async function main() {
  const server = await serve(repositoryRoot, requestedServerPort);
  const serverPort = server.address().port;
  const debugPort = requestedDebugPort || await reservePort();
  const serverBase = `http://127.0.0.1:${serverPort}`;
  const pageUrl = new URL(options.url || '/tools/webgpu_w8a8_benchmark.html', serverBase);
  if (!pageUrl.searchParams.has('warmup')) pageUrl.searchParams.set('warmup', String(warmup));
  if (!pageUrl.searchParams.has('iterations')) pageUrl.searchParams.set('iterations', String(iterations));
  if (requireDot) pageUrl.searchParams.set('requireDot', '1');

  const profile = `/tmp/volvoxai-webgpu-w8a8-${process.pid}`;
  const chromeArguments = [
    '--headless=new',
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--no-sandbox',
    '--disable-gpu-sandbox',
    `--user-data-dir=${profile}`,
    `--remote-debugging-port=${debugPort}`,
  ];
  if (adapterMode === 'swiftshader') chromeArguments.push('--use-webgpu-adapter=swiftshader');
  else chromeArguments.push('--use-angle=vulkan', '--disable-vulkan-surface');
  chromeArguments.push(pageUrl.href);

  const chromePath = findChrome();
  console.log(`Adapter mode: ${adapterMode}`);
  console.log(`Chrome: ${chromePath}`);
  console.log(`Page: ${pageUrl.href}`);
  const chrome = spawn(chromePath, chromeArguments, { stdio: ['ignore', 'ignore', 'pipe'] });
  const chromeError = { value: null };
  let chromeStderr = '';
  chrome.on('error', (error) => { chromeError.value = error; });
  chrome.stderr.on('data', (chunk) => {
    chromeStderr = (chromeStderr + chunk.toString()).slice(-12000);
  });

  try {
    const page = await waitForPage(`127.0.0.1:${debugPort}`, chrome, chromeError);
    const value = await pollResult(page.webSocketDebuggerUrl);
    console.log('\n' + value.text);
    if (adapterMode === 'hardware' && /swiftshader/i.test(value.result?.adapter || '')) {
      console.warn('\nWARNING: hardware mode was requested, but Chrome selected SwiftShader; do not treat these timings as hardware results.');
    }
    console.log('\nRESULT_JSON ' + JSON.stringify(value.result));
    process.exitCode = value.result.status === 'pass' ? 0 : 1;
  } catch (error) {
    if (chromeStderr) console.error(`Chrome stderr (tail):\n${chromeStderr}`);
    throw error;
  } finally {
    await stopChrome(chrome);
    await new Promise((accept) => server.close(accept));
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* Best-effort temporary cleanup. */ }
  }
}

main().catch((error) => {
  console.error(error?.stack || error);
  process.exitCode = 2;
});
