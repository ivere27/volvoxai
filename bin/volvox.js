#!/usr/bin/env node

import { Command } from 'commander';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const packageVersion = JSON.parse(
  fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
).version;
const { VolvoxAI } = await import(
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
    try {
      installFileFetchShim();
      console.log(`[Volvox CLI] Initializing Engine (Backend: ${options.backend})...`);
      const cliDir = path.dirname(fileURLToPath(import.meta.url));
      const wasmPath = path.resolve(cliDir, '..', 'dist', packageVersion, 'volvoxai.wasm');
      const engine = await VolvoxAI.init(options.backend, wasmPath);

      console.log(`[Volvox CLI] Loading Model: ${options.model}`);
      if (!fs.existsSync(options.model)) {
        console.error(`Error: Model file not found at ${options.model}`);
        process.exit(1);
      }
      
      const modelUrl = 'file://' + path.resolve(options.model);
      const graph = await engine.loadGraph(modelUrl);
      
      console.log(`[Volvox CLI] Compiling Model...`);
      const executor = await engine.compile(graph, modelUrl);

      console.log(`[Volvox CLI] Model Ready! Provide input data via scripts for full inference.`);
      // TODO: In a real CLI, we would parse JSON inputs or specific tensors,
      // and call executor.execute(inputs).
      
    } catch (err) {
      console.error('[Volvox CLI] Error during execution:', err);
      process.exit(1);
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

program.parse();

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
