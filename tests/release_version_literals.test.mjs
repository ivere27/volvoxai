import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sourceDirectories = ['tools', 'tests', 'examples', 'docs', 'python'];
const excludedDirectories = new Set([
  'generated', 'third_party', 'assets', 'target', 'node_modules', 'build',
  'venv', '.venv', '__pycache__', 'reports', 'test-results',
]);
const textExtensions = new Set([
  '.mjs', '.js', '.cjs', '.ts', '.tsx', '.py', '.sh', '.bash',
  '.c', '.h', '.cc', '.cpp', '.inc', '.cmake', '.toml', '.json',
  '.md', '.rst', '.txt', '.html', '.yaml', '.yml',
]);
// These literals exercise version changes in a temporary package fixture.
const fixtureExemptions = new Set(['tests/release_version.test.mjs']);
const releasePath = /\bdist\/(?:python\/)?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)*/g;

function* authoredFiles(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) =>
    a.name.localeCompare(b.name))) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (!excludedDirectories.has(entry.name)) yield* authoredFiles(filename);
    } else if (entry.isFile() && (textExtensions.has(path.extname(entry.name)) ||
      entry.name === 'Makefile' || entry.name === 'CMakeLists.txt')) {
      yield filename;
    }
  }
}

test('authored tools and documentation derive release paths from the package version', () => {
  const files = [path.join(repositoryRoot, 'README.md')];
  for (const directory of sourceDirectories) {
    files.push(...authoredFiles(path.join(repositoryRoot, directory)));
  }
  const violations = [];
  for (const filename of files) {
    const relative = path.relative(repositoryRoot, filename).split(path.sep).join('/');
    if (fixtureExemptions.has(relative)) continue;
    const lines = readFileSync(filename, 'utf8').split(/\r?\n/);
    for (let index = 0; index < lines.length; index++) {
      for (const match of lines[index].matchAll(releasePath)) {
        violations.push(`${relative}:${index + 1}: ${match[0]}`);
      }
    }
  }
  assert.deepEqual(violations, [],
    "Read release paths from package.json using the workflow's existing runtime " +
    'or the shared browser release loader. Hardcoded release paths:\n' +
    violations.join('\n'));
});
