import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const DOCUMENTS = [
  '../README.md',
  '../docs/quickstart.md',
  '../docs/model_builder_training.md',
  '../docs/quantization.md',
  '../docs/training-ptq-runtime-matrix.md',
  '../docs/browser-runtime.md',
  '../docs/textbook/05-optimizer-and-training-loop.md',
  '../docs/textbook/ko/05-optimizer-and-training-loop.md',
  '../examples/seq2seq_training/README.md',
  './parity/README.md',
  './parity/TODO.md',
];

const LIFECYCLE_DOCUMENTS = [
  '../README.md',
  '../docs/browser-runtime.md',
  '../docs/quickstart.md',
  '../docs/textbook/01-foundations.md',
  '../docs/textbook/08-inside-the-engine.md',
  '../docs/textbook/11-glossary-and-next-steps.md',
  '../docs/textbook/ko/01-foundations.md',
  '../docs/textbook/ko/08-inside-the-engine.md',
  '../docs/textbook/ko/11-glossary-and-next-steps.md',
];

function javascriptFences(markdown) {
  return [...markdown.matchAll(/(?:```|~~~)(?:js|javascript)\s*\n([\s\S]*?)(?:```|~~~)/g)]
    .map((match) => match[1])
    .join('\n');
}

test('public training snippets use logical snapshots and never revive concrete Model training', () => {
  const forbidden = [
    /\bnew\s+RuntimeGraphBuilder\s*\(/,
    /\bRuntimeGraphBuilder\s*,/,
    /\bruntime\.createModel\s*\(/,
    /\b(?:model|runtimeModel|trainingModel)\.compile\s*\(/,
    /\bcreateTrainer\s*\(\s*(?:model|runtimeModel|trainingModel|graph)\b/,
    /\bimportModelCheckpoint\([^\n]*\)\.graph\b/,
    /\brestored\.graph\b/,
  ];
  for (const relative of DOCUMENTS) {
    const markdown = readFileSync(new URL(relative, import.meta.url), 'utf8');
    const snippets = javascriptFences(markdown);
    for (const pattern of forbidden) {
      assert.doesNotMatch(snippets, pattern, `${relative} contains a retired public training path`);
    }
  }
});

test('primary v1 training guides show shaped input views and successor snapshots', () => {
  for (const relative of [
    '../README.md',
    '../docs/quickstart.md',
    '../docs/model_builder_training.md',
    '../examples/seq2seq_training/README.md',
  ]) {
    const markdown = readFileSync(new URL(relative, import.meta.url), 'utf8');
    assert.match(markdown, /\bModel\b/, `${relative} must name the public model type`);
    assert.match(markdown, /\bdata\s*:[\s\S]{0,240}?\bshape\s*:/,
      `${relative} must show an explicit shaped input`);
    assert.match(markdown, /\bsuccessor\w*\s*=\s*await\s+trainer\.commit\(\)/,
      `${relative} must retain the immutable commit successor`);
  }
});

test('public lifecycle chapters do not teach retired Model factories or compilation', () => {
  for (const relative of LIFECYCLE_DOCUMENTS) {
    const markdown = readFileSync(new URL(relative, import.meta.url), 'utf8');
    assert.doesNotMatch(markdown, /\b(?:Runtime|runtime)\.loadModel\s*\(/, relative);
    assert.doesNotMatch(markdown, /\bruntime\.createModel\s*\(/, relative);
    assert.doesNotMatch(markdown, /\bModel\.compile\s*\(/, relative);
  }
});
