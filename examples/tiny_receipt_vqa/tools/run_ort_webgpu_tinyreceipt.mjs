// Drive the pinned ONNX Runtime Web WebGPU TinyReceipt benchmark in a fresh
// hardware-backed Chrome process. The runtime package stays external to this
// repository and is mounted read-only by the local HTTP server.

import http from 'node:http';
import { spawn } from 'node:child_process';
import {
  mkdtempSync, readFileSync, realpathSync, rmSync, statSync, unlinkSync, writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join, resolve, sep } from 'node:path';

import {
  childProcessHasSettled,
  observeChildProcessExit,
} from './child_process_lifecycle.mjs';
import { cdpReplyTimeoutMs } from './cdp_reply_timeout.mjs';

const args = Object.fromEntries(process.argv.slice(2).map((argument) => {
  const match = argument.match(/^--([^=]+)=?(.*)$/);
  return match ? [match[1], match[2]] : [argument, ''];
}));

const EXPECTED_ORT_WEB_VERSION = '1.27.0';
const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.json': 'application/json',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
};

function requiredArgument(name) {
  const value = args[name];
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`--${name} is required`);
  }
  return value;
}

function realDirectory(value, label) {
  const path = realpathSync(value);
  if (!statSync(path).isDirectory()) throw new Error(`${label} must be a directory`);
  return path;
}

function resolveInside(root, relative, label) {
  const path = resolve(root, relative);
  if (path !== root && !path.startsWith(`${root}${sep}`)) {
    throw new Error(`${label} escapes its mounted root`);
  }
  return path;
}

function mountedFile(pathname, mounts) {
  for (const [prefix, root] of mounts) {
    if (pathname.startsWith(prefix)) {
      return resolveInside(root, pathname.slice(prefix.length), pathname);
    }
  }
  throw new Error(`unmounted URL path: ${pathname}`);
}

function serve(mounts) {
  const server = http.createServer((request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      if (url.pathname === '/favicon.ico') {
        response.writeHead(204);
        response.end();
        return;
      }
      const path = mountedFile(decodeURIComponent(url.pathname), mounts);
      if (!statSync(path).isFile()) throw new Error('not a file');
      response.writeHead(200, {
        'cache-control': 'no-store',
        'content-type': MIME[extname(path).toLowerCase()] || 'application/octet-stream',
      });
      response.end(readFileSync(path));
    } catch (error) {
      response.writeHead(404, { 'content-type': 'text/plain' });
      response.end(`404: ${error?.message || error}`);
    }
  });
  return new Promise((resolveListen) => {
    server.listen(0, '127.0.0.1', () => resolveListen(server));
  });
}

function deadlinePromise(timeoutMs, label) {
  return new Promise((_, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out`)), timeoutMs);
    timer.unref?.();
  });
}

async function cdpTargets(hostport) {
  return Promise.race([
    fetch(`http://${hostport}/json`).then((response) => response.json()),
    deadlinePromise(5000, 'CDP target discovery'),
  ]);
}

async function evalUntil(wsUrl, expression, timeoutMs) {
  const socket = new WebSocket(wsUrl);
  const deadline = Date.now() + timeoutMs;
  await Promise.race([
    new Promise((resolveOpen, reject) => {
      socket.onopen = resolveOpen;
      socket.onerror = () => reject(new Error('CDP WebSocket open failed'));
    }),
    deadlinePromise(Math.min(timeoutMs, 5000), 'CDP WebSocket open'),
  ]);
  let id = 0;
  const pending = new Map();
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const request = pending.get(message.id);
      clearTimeout(request.timer);
      request.resolve(message);
      pending.delete(message.id);
    }
  };
  const rejectPending = (reason) => {
    for (const request of pending.values()) {
      clearTimeout(request.timer);
      request.reject(reason);
    }
    pending.clear();
  };
  socket.addEventListener('error', () => rejectPending(new Error('CDP WebSocket failed')));
  socket.addEventListener('close', () => rejectPending(new Error('CDP WebSocket closed')));
  const send = (method, params) => new Promise((resolveReply, reject) => {
    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      reject(new Error(`CDP ${method} exceeded benchmark deadline`));
      return;
    }
    const next = ++id;
    // Runtime.evaluate runs on the renderer main thread. Model/session graph
    // compilation may legitimately occupy it for more than five seconds, so
    // use the existing overall benchmark deadline for evaluate replies while
    // retaining the short watchdog for CDP control commands.
    const replyTimeoutMs = cdpReplyTimeoutMs(method, remaining);
    const timer = setTimeout(() => {
      pending.delete(next);
      reject(new Error(`CDP ${method} timed out`));
    }, replyTimeoutMs);
    pending.set(next, { resolve: resolveReply, reject, timer });
    try {
      socket.send(JSON.stringify({ id: next, method, params }));
    } catch (error) {
      clearTimeout(timer);
      pending.delete(next);
      reject(error);
    }
  });
  await send('Runtime.enable', {});
  await send('Log.enable', {});
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') {
      const detail = message.params?.exceptionDetails;
      console.error('[page exception] '
        + (detail?.exception?.description || detail?.text || JSON.stringify(detail)));
    } else if (message.method === 'Log.entryAdded') {
      const entry = message.params?.entry;
      if (entry?.level === 'error' || entry?.level === 'warning') {
        console.error(`[page ${entry.level}] ${entry.text} ${entry.url || ''}`);
      }
    }
  });
  let last = null;
  while (Date.now() < deadline) {
    const reply = await send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: false,
    });
    last = reply.result?.result?.value ?? null;
    if (last) break;
    await new Promise((resolveWait) => setTimeout(resolveWait, 1000));
  }
  rejectPending(new Error('CDP evaluation finished'));
  socket.close();
  return last;
}

const repository = realDirectory(process.cwd(), 'repository');
const source = realDirectory(requiredArgument('source'), 'source');
const ortWebRoot = realDirectory(requiredArgument('ort-web-root'), 'ORT Web root');
const packageDocument = JSON.parse(readFileSync(join(ortWebRoot, 'package.json'), 'utf8'));
if (packageDocument.name !== 'onnxruntime-web'
    || packageDocument.version !== EXPECTED_ORT_WEB_VERSION) {
  throw new Error(
    `--ort-web-root must be onnxruntime-web@${EXPECTED_ORT_WEB_VERSION}`,
  );
}
const ortDist = realDirectory(join(ortWebRoot, 'dist'), 'ORT Web dist');
for (const asset of [
  'ort.webgpu.min.mjs',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
]) {
  if (!statSync(join(ortDist, asset)).isFile()) throw new Error(`missing ORT Web asset ${asset}`);
}

const imageArgument = requiredArgument('image');
const image = resolveInside(repository, imageArgument, 'image');
if (!statSync(image).isFile()) throw new Error('--image must name a repository file');
const imageRelative = image.slice(repository.length + 1).split(sep).join('/');
const query = new URLSearchParams({
  image: `/repo/${imageRelative}`,
  precision: requiredArgument('precision'),
  prompt: args.prompt || 'phone number last one',
  family: args.family || 'phone',
  maxNewTokens: args.tokens || '4',
  warmup: args.warmup || '1',
});
const server = await serve(new Map([
  ['/repo/', repository],
  ['/source/', source],
  ['/ort/', ortDist],
]));
const address = server.address();
if (typeof address !== 'object' || address == null) {
  throw new Error('benchmark HTTP server has no TCP address');
}
const url = `http://127.0.0.1:${address.port}/repo/examples/tiny_receipt_vqa/tools/`
  + `ort_webgpu_benchmark_tinyreceipt.html?${query}`;

const profile = mkdtempSync(join(tmpdir(), 'ort-webgpu-tr-'));
const chromeArguments = [
  '--headless=new',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--no-sandbox',
  '--disable-gpu-sandbox',
  '--use-angle=vulkan',
  '--disable-vulkan-surface',
  `--user-data-dir=${profile}`,
  '--remote-debugging-port=0',
  url,
];
console.log(`runtime=onnxruntime-web@${EXPECTED_ORT_WEB_VERSION}\nurl=${url}`);
const chrome = spawn(process.env.CHROME || 'google-chrome', chromeArguments, {
  stdio: 'ignore',
  detached: true,
});
const chromeExit = observeChildProcessExit(chrome);
if (args['chrome-pid-file']) {
  writeFileSync(args['chrome-pid-file'], `${chrome.pid}\n`, { encoding: 'ascii' });
}
const killChrome = () => {
  if (childProcessHasSettled(chrome)) return;
  try {
    process.kill(-chrome.pid, 'SIGKILL');
  } catch {
    chrome.kill('SIGKILL');
  }
};
for (const [signal, number] of [['SIGTERM', 15], ['SIGINT', 2], ['SIGHUP', 1]]) {
  process.once(signal, () => {
    killChrome();
    process.exit(128 + number);
  });
}

try {
  let debugPort = null;
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const value = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8')
        .split(/\r?\n/, 1)[0];
      const candidate = Number(value);
      if (Number.isInteger(candidate) && candidate > 0 && candidate <= 65535) {
        debugPort = candidate;
        break;
      }
    } catch { /* Chrome has not published its endpoint yet. */ }
    await new Promise((resolveWait) => setTimeout(resolveWait, 250));
  }
  if (debugPort == null) throw new Error('Chrome debugging endpoint not found');
  let targets = [];
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      targets = await cdpTargets(`127.0.0.1:${debugPort}`);
      if (targets.some((target) => target.type === 'page')) break;
    } catch { /* Chrome is not ready yet. */ }
    await new Promise((resolveWait) => setTimeout(resolveWait, 250));
  }
  const page = targets.find((target) => target.type === 'page');
  if (!page) throw new Error('Chrome page target not found');
  const result = await evalUntil(
    page.webSocketDebuggerUrl,
    '(window.__ortWebgpuBenchmarkResult '
      + "&& window.__ortWebgpuBenchmarkResult.status !== 'running') "
      + '? JSON.stringify(window.__ortWebgpuBenchmarkResult) : null',
    Number(args.timeout || 600000),
  );
  if (!result) {
    console.error('ORT WebGPU page did not publish a structured result');
    process.exitCode = 2;
  } else {
    const parsed = JSON.parse(result);
    console.log(`RESULT_JSON ${JSON.stringify(parsed)}`);
    process.exitCode = parsed.status === 'pass' ? 0 : 1;
  }
} finally {
  if (!childProcessHasSettled(chrome)) killChrome();
  await chromeExit;
  await new Promise((resolveClose) => server.close(resolveClose));
  try {
    rmSync(profile, { recursive: true, force: true, maxRetries: 50, retryDelay: 100 });
  } catch (error) {
    console.error(`warning: could not remove Chrome profile ${profile}: ${error}`);
  }
  if (args['chrome-pid-file']) {
    try {
      unlinkSync(args['chrome-pid-file']);
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  }
}
