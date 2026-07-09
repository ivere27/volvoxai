// Drives tools/webgpu_op_tests.html in a Chrome instance and prints the results.
// Works against local headless Chrome (SwiftShader) or a remote Chrome DevTools
// endpoint (e.g. an Android device via `adb forward`).
//
//   node --experimental-websocket tools/run_webgpu_tests.mjs [--cdp=host:port] [--url=URL]
//
// With no --cdp it launches local headless Chrome with SwiftShader WebGPU and
// serves the repo over http. Requires: google-chrome on PATH (local mode).
import http from 'node:http';
import { spawn } from 'node:child_process';
import { readFileSync, statSync } from 'node:fs';
import { extname, join } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.match(/^--([^=]+)=?(.*)$/); return [m[1], m[2]];
}));

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };

function serve(root, port) {
  const srv = http.createServer((req, res) => {
    try {
      let p = join(root, decodeURIComponent(req.url.split('?')[0]));
      if (statSync(p).isDirectory()) p = join(p, 'index.html');
      res.writeHead(200, { 'content-type': MIME[extname(p)] || 'application/octet-stream' });
      res.end(readFileSync(p));
    } catch { res.writeHead(404); res.end('404'); }
  });
  return new Promise((r) => srv.listen(port, () => r(srv)));
}

async function cdpFetch(hostport) {
  // The /json/list endpoint returns the debuggable targets.
  const res = await fetch(`http://${hostport}/json`);
  return res.json();
}

async function evalUntil(wsUrl, expr, done, timeoutMs = 90000) {
  const ws = new WebSocket(wsUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = () => j(new Error('ws open failed')); });
  let id = 0; const pending = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
  const send = (method, params) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  await send('Runtime.enable', {});
  const start = Date.now();
  let last = '';
  while (Date.now() - start < timeoutMs) {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true });
    last = r.result?.result?.value ?? '';
    if (done(last)) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  ws.close();
  return last;
}

async function main() {
  const expr = "document.getElementById('out') ? document.getElementById('out').innerText : ''";
  const done = (t) => /SUMMARY|FATAL/.test(t);

  if (args.cdp) {
    // Remote/pre-launched Chrome (e.g. Android via adb forward). Assumes the page
    // is already open at the debugging endpoint.
    const targets = await cdpFetch(args.cdp);
    const page = targets.find((t) => t.type === 'page' && /webgpu_op_tests/.test(t.url)) || targets.find((t) => t.type === 'page');
    if (!page) throw new Error('no page target; open the harness URL on the device first');
    const text = await evalUntil(page.webSocketDebuggerUrl, expr, done);
    console.log(text);
    process.exit(/ALL_PASS/.test(text) ? 0 : 1);
  }

  // Local headless Chrome + SwiftShader.
  const port = 8091;
  const srv = await serve(process.cwd(), port);
  const url = args.url || `http://localhost:${port}/tools/webgpu_op_tests.html`;
  const dbgPort = 9333;
  const profile = `/tmp/volvox-chrome-${process.pid}`;
  const chrome = spawn('google-chrome', [
    '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader',
    '--no-sandbox', '--disable-gpu-sandbox', `--user-data-dir=${profile}`, `--remote-debugging-port=${dbgPort}`, url,
  ], { stdio: 'ignore' });
  try {
    // Wait for the debugging endpoint, then find our page target.
    let targets = [];
    for (let i = 0; i < 40; i++) {
      try { targets = await cdpFetch(`localhost:${dbgPort}`); if (targets.some((t) => t.type === 'page')) break; } catch {}
      await new Promise((r) => setTimeout(r, 250));
    }
    const page = targets.find((t) => t.type === 'page');
    if (!page) throw new Error('chrome page target not found');
    const text = await evalUntil(page.webSocketDebuggerUrl, expr, done);
    console.log(text);
    process.exitCode = /ALL_PASS/.test(text) ? 0 : 1;
  } finally {
    chrome.kill('SIGKILL');
    srv.close();
  }
}

main().catch((e) => { console.error(e); process.exit(2); });
