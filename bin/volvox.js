#!/usr/bin/env node

import { Command } from 'commander';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const packageVersion = JSON.parse(
  fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
).version;
const {
  ModelLoader,
  Model,
  VolvoxAI,
} = await import(
  new URL(`../dist/${packageVersion}/volvoxai.js`, import.meta.url)
);

const program = new Command();

program
  .name('volvox')
  .description('Volvox AI CLI for running local inference')
  .version(packageVersion);

program
  .command('run')
  .description('Run inference with a given model and input')
  .requiredOption('-m, --model <path>', 'Path to the safetensors model file')
  .option('-b, --backend <type>', 'Backend to use (wasm, cpu)', 'wasm')
  .action(async (options) => {
    let runtime = null;
    let compiled = null;
    try {
      installFileFetchShim();
      console.log(`[Volvox CLI] Initializing runtime (backend: ${options.backend})...`);
      const cliDir = path.dirname(fileURLToPath(import.meta.url));
      const wasmPath = path.resolve(cliDir, '..', 'dist', packageVersion, 'volvoxai.wasm');
      runtime = await VolvoxAI.createRuntime({
        backends: [options.backend],
        wasmUrl: pathToFileURL(wasmPath),
      });

      console.log(`[Volvox CLI] Loading logical model snapshot: ${options.model}`);
      if (!fs.existsSync(options.model)) {
        throw new Error(`Model file not found at ${options.model}`);
      }

      const modelUrl = pathToFileURL(path.resolve(options.model));
      const snapshot = Model.capture(
        await ModelLoader.load(modelUrl.href),
      );

      console.log('[Volvox CLI] Compiling model...');
      compiled = await runtime.compile(snapshot, {
        backend: {
          mode: 'require',
          backend: options.backend,
          operatorFallback: 'forbid',
        },
      });
      console.log(`[Volvox CLI] Model ready on ${compiled.backend}.`);
    } catch (err) {
      console.error('[Volvox CLI] Error during execution:', err);
      process.exitCode = 1;
    } finally {
      await compiled?.close().catch((error) => {
        console.error('[Volvox CLI] Failed to close compiled model:', error);
        process.exitCode = 1;
      });
      await runtime?.close().catch((error) => {
        console.error('[Volvox CLI] Failed to close runtime:', error);
        process.exitCode = 1;
      });
    }
  });

program
  .command('info')
  .description('Print system and engine info')
  .action(async () => {
    console.log('Volvox AI Engine CLI');
    console.log('Node.js:', process.version);
    console.log('Available backends: wasm, cpu');
  });

await program.parseAsync();

function installFileFetchShim() {
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const href = typeof url === 'string' ? url : url?.url;
    if (href && href.startsWith('file://')) {
      const data = await fs.promises.readFile(fileURLToPath(href));
      return new Response(data, { status: 200 });
    }
    return nativeFetch(url, init);
  };
}
