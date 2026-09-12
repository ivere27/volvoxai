#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import {
  createReleaseSizeDocument,
  renderReleaseSizeMarkdown,
  stablePrettyJSON} from './release_size.mjs';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function usage() {
  return `Usage:
  node tools/report_release_size.mjs report [options]
  node tools/report_release_size.mjs check [options]

Options:
  --web-only            Inspect the six dist artifacts plus required WASM build/object
                        evidence, and skip native ELF tools.
  --budgets <path>      Budget JSON (default: tools/release_size_budgets.json).
  --output-dir <path>   Report directory (default: build/size-reports).
  --json <path>         JSON report path (report command only).
  --markdown <path>     Markdown report path (report command only).
  --help                Show this help.
`;
}

function parseArguments(argv) {
  const [command, ...arguments_] = argv;
  if (command === '--help' || command === '-h') return { help: true };
  if (command !== 'report' && command !== 'check') {
    throw new Error("First argument must be 'report' or 'check'.");
  }
  const options = {
    command,
    webOnly: false,
    budgetPath: path.join(repositoryRoot, 'tools', 'release_size_budgets.json'),
    outputDirectory: path.join(repositoryRoot, 'build', 'size-reports'),
    jsonPath: null,
    markdownPath: null,
  };
  for (let index = 0; index < arguments_.length; index++) {
    const argument = arguments_[index];
    if (argument === '--help' || argument === '-h') return { help: true };
    if (argument === '--web-only') {
      options.webOnly = true;
      continue;
    }
    const valueOptions = new Map([
      ['--budgets', 'budgetPath'],
      ['--output-dir', 'outputDirectory'],
      ['--json', 'jsonPath'],
      ['--markdown', 'markdownPath'],
    ]);
    const property = valueOptions.get(argument);
    if (property === undefined) throw new Error(`Unknown option '${argument}'.`);
    if (index + 1 >= arguments_.length) throw new Error(`Missing value for '${argument}'.`);
    options[property] = path.resolve(repositoryRoot, arguments_[++index]);
  }
  if (command === 'check' && (options.jsonPath !== null || options.markdownPath !== null)) {
    throw new Error("The check command is read-only; use 'report' to write JSON or Markdown.");
  }
  return options;
}

async function writeAtomic(destination, contents) {
  await fs.mkdir(path.dirname(destination), { recursive: true });
  const temporary = path.join(
    path.dirname(destination),
    `.${path.basename(destination)}.${process.pid}.tmp`,
  );
  try {
    await fs.writeFile(temporary, contents);
    await fs.rename(temporary, destination);
  } finally {
    await fs.rm(temporary, { force: true });
  }
}

function printGateResult(document) {
  const evaluation = document.budgetEvaluation;
  if (evaluation.passed) {
    console.log(
      `Release size gate passed for ${document.fixedArtifactCount} fixed artifacts ` +
      `(${document.scope}).`,
    );
    return;
  }
  console.error(`Release size gate failed with ${evaluation.failures.length} regression(s):`);
  for (const failure of evaluation.failures) {
    if (failure.comparison === 'exact') {
      console.error(
        `  ${failure.id} ${failure.metric}: ${failure.actualValue ?? 'missing'} != ` +
        `${failure.baselineValue} (exact linked-WASM contract)`,
      );
      continue;
    }
    console.error(
      `  ${failure.id} ${failure.metric}: ${failure.actualBytes ?? 'missing'} > ` +
      `${failure.limitBytes}`,
    );
  }
  process.exitCode = 1;
}

async function main() {
  let options;
  try {
    options = parseArguments(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    console.error(usage());
    process.exitCode = 2;
    return;
  }
  if (options.help) {
    console.log(usage());
    return;
  }

  const document = await createReleaseSizeDocument({
    repositoryRoot,
    budgetPath: options.budgetPath,
    webOnly: options.webOnly,
  });
  if (options.command === 'report') {
    const jsonPath = options.jsonPath ?? path.join(options.outputDirectory, 'release-size.json');
    const markdownPath = options.markdownPath ??
      path.join(options.outputDirectory, 'release-size.md');
    await Promise.all([
      writeAtomic(jsonPath, stablePrettyJSON(document)),
      writeAtomic(markdownPath, renderReleaseSizeMarkdown(document)),
    ]);
    console.log(`Wrote ${path.relative(repositoryRoot, jsonPath)}.`);
    console.log(`Wrote ${path.relative(repositoryRoot, markdownPath)}.`);
  }
  printGateResult(document);
}

try {
  await main();
} catch (error) {
  console.error(`Release size inspection failed: ${error.message}`);
  process.exitCode = 1;
}
