// Drive the TinyReceipt WebGPU harness in headless Chrome and print its result.
//
// tools/run_webgpu_tests.mjs waits for the operator-test page's SUMMARY marker;
// the TinyReceipt page instead publishes a structured result object, so this
// polls for that object and prints it verbatim.
//
//   node --experimental-websocket <this> --package build/... --image build/x.jpg
//     [--adapter=swiftshader|hardware] [--prompt "..."] [--tokens 8]

import http from 'node:http';
import { spawn } from 'node:child_process';
import {
  mkdtempSync, readFileSync, rmSync, statSync, unlinkSync, writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join } from 'node:path';

import {
  childProcessHasSettled,
  observeChildProcessExit,
} from './child_process_lifecycle.mjs';
import { cdpReplyTimeoutMs } from './cdp_reply_timeout.mjs';

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.match(/^--([^=]+)=?(.*)$/); return m ? [m[1], m[2]] : [a, ''];
}));

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.json': 'application/json',
  '.jpg': 'image/jpeg', '.png': 'image/png', '.safetensors': 'application/octet-stream',
};

function serve(root, port) {
  const server = http.createServer((request, response) => {
    try {
      if (request.url.split('?')[0] === '/favicon.ico') {
        response.writeHead(204);
        response.end();
        return;
      }
      let path = join(root, decodeURIComponent(request.url.split('?')[0]));
      if (statSync(path).isDirectory()) path = join(path, 'index.html');
      const body = readFileSync(path);
      response.writeHead(200, {
        // Every benchmark owns a fresh origin and Chrome profile, and each
        // immutable artifact is consumed only once.  Caching the large model
        // stores adds avoidable disk-cache traffic and can overlap the later
        // lazy decoder load with an already-live GPU working set.
        'cache-control': 'no-store',
        'content-length': body.byteLength,
        'content-type': MIME[extname(path)] || 'application/octet-stream',
      });
      response.end(body);
    } catch { response.writeHead(404); response.end('404'); }
  });
  return new Promise((resolve) =>
    server.listen(port, '127.0.0.1', () => resolve(server)));
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
    new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = () => reject(new Error('ws open failed'));
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
  const send = (method, params) => new Promise((resolve, reject) => {
    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      reject(new Error(`CDP ${method} exceeded benchmark deadline`));
      return;
    }
    const next = ++id;
    // Runtime.evaluate is delivered on the renderer main thread. A legitimate
    // synchronous graph compile can keep that thread busy for much longer than
    // the short control-command watchdog, so bind it to the benchmark's
    // already-bounded overall deadline instead of introducing a false 5 s
    // failure. Runtime.enable/Log.enable should still answer promptly.
    const replyTimeoutMs = cdpReplyTimeoutMs(method, remaining);
    const timer = setTimeout(() => {
      pending.delete(next);
      reject(new Error(`CDP ${method} timed out`));
    }, replyTimeoutMs);
    pending.set(next, { resolve, reject, timer });
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
  // Surface page-side failures: a module that fails to import never reaches the
  // harness's own try/catch, so the page would otherwise just sit at 'Starting'.
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') {
      const detail = message.params?.exceptionDetails;
      console.log('[page exception] '
        + (detail?.exception?.description || detail?.text || JSON.stringify(detail)));
    } else if (message.method === 'Log.entryAdded') {
      const entry = message.params?.entry;
      if (entry?.level === 'error' || entry?.level === 'warning') {
        console.log(`[page ${entry.level}] ${entry.text} ${entry.url || ''}`);
      }
    } else if (message.method === 'Runtime.consoleAPICalled') {
      const text = (message.params?.args || [])
        .map((a) => a.value ?? a.description ?? '').join(' ');
      if (text) console.log('[page console] ' + text);
    }
  });
  let last = null;
  while (Date.now() < deadline) {
    const reply = await send('Runtime.evaluate', {
      expression, returnByValue: true, awaitPromise: false,
    });
    last = reply.result?.result?.value ?? null;
    if (last) break;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  rejectPending(new Error('CDP evaluation finished'));
  socket.close();
  return last;
}

const adapter = args.adapter || 'swiftshader';
const query = new URLSearchParams({
  model: `/${args.package}/package_manifest.json`,
  image: `/${args.image}`,
  prompt: args.prompt || 'phone number last one',
  family: args.family || 'phone',
  maxNewTokens: args.tokens || '4',
  warmup: args.warmup || '1',
  iterations: args.iterations || '5',
  ...(args.qualification ? { qualification: args.qualification } : {}),
});
const server = await serve(process.cwd(), 0);
const address = server.address();
if (typeof address !== 'object' || address == null) {
  throw new Error('benchmark HTTP server has no TCP address');
}
const url = `http://127.0.0.1:${address.port}/examples/tiny_receipt_vqa/tools/`
  + `webgpu_w8a8_benchmark_tinyreceipt.html?${query}`;

// A newly created directory plus Chrome's dynamic debugging port makes every
// child independent of stale profiles and concurrent benchmark processes.
const profile = mkdtempSync(join(tmpdir(), 'volvox-chrome-tr-'));
const chromeArguments = [
  '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan',
  '--no-sandbox', '--disable-gpu-sandbox',
  `--user-data-dir=${profile}`, '--remote-debugging-port=0',
];
if (adapter === 'hardware') {
  chromeArguments.push('--use-angle=vulkan', '--disable-vulkan-surface');
} else {
  chromeArguments.push('--use-webgpu-adapter=swiftshader');
}
chromeArguments.push(url);

console.log(`adapter=${adapter}\nurl=${url}`);
const chrome = spawn(process.env.CHROME || 'google-chrome', chromeArguments,
  { stdio: 'ignore', detached: true });
// Observe the terminal event immediately. Waiting on a listener installed only
// during cleanup can miss an earlier signal exit because signalCode is set while
// exitCode remains null.
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
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  if (debugPort == null) throw new Error('chrome debugging endpoint not found');
  let targets = [];
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      targets = await cdpTargets(`127.0.0.1:${debugPort}`);
      if (targets.some((target) => target.type === 'page')) break;
    } catch { /* chrome not up yet */ }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  const page = targets.find((target) => target.type === 'page');
  if (!page) throw new Error('chrome page target not found');
  const result = await evalUntil(
    page.webSocketDebuggerUrl,
    '(window.__webgpuW8A8BenchmarkResult && '
    + "window.__webgpuW8A8BenchmarkResult.status !== 'running') "
    + '? JSON.stringify(window.__webgpuW8A8BenchmarkResult) : null',
    Number(args.timeout || 600000),
  );
  if (!result) {
    const text = await evalUntil(
      page.webSocketDebuggerUrl,
      "document.getElementById('out') ? document.getElementById('out').innerText : null",
      5000,
    );
    console.log('no structured result; page text:\n' + (text || '(empty)'));
    process.exitCode = 2;
  } else {
    const parsed = JSON.parse(result);
    console.log('RESULT_JSON ' + JSON.stringify(parsed));
    process.exitCode = parsed.status === 'pass' ? 0 : 1;
  }
} finally {
  // Chrome spawns GPU/utility children. Kill the isolated process group so
  // they cannot keep the temporary profile or benchmark process alive. The
  // observer was installed at spawn time, so an already-signaled child cannot
  // leave cleanup waiting for an event that already happened.
  if (!childProcessHasSettled(chrome)) killChrome();
  await chromeExit;
  await new Promise((resolve) => server.close(resolve));
  // Chrome's utility children can release profile files just after the main
  // process exits.  Let Node retry the recursive removal instead of turning a
  // valid measured result into a cleanup-race failure.
  try {
    rmSync(profile, {
      recursive: true,
      force: true,
      maxRetries: 50,
      retryDelay: 100,
    });
  } catch (error) {
    // The result is already complete and Chrome used a unique profile. A
    // lingering utility child must not falsify the measured inference result;
    // report the best-effort cleanup failure for the caller to diagnose.
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
