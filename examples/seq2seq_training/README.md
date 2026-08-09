# Encoder-decoder training example

The public v1 training boundary is an immutable bounded
`Model`. The application owns tokenization, padding, and teacher
forcing; a Trainer owns only its private parameter, optimizer, and accumulation
state. Batch size and sequence lengths may change between steps when every
concrete shape remains inside the logical graph's declared domain.

The `Seq2SeqBuilder.js` file in this directory remains a repository-internal
fixed-shape numerical fixture. It is not exported from `volvoxai` or
`volvoxai/full` and is not a public authoring API. Production model-family
exporters should emit `volvox-graph/v1` plus fixed safetensors weights.

The public load/train/checkpoint flow is:

```js
import {
  Model,
  Trainer,
  importModelCheckpoint,
} from 'volvoxai/full';

const source = await Model.load('./seq2seq/model.safetensors');
const trainer = await Trainer.create(source, { backend: 'cpu' });

// Teacher forcing is application policy. Each input is a shaped view; these
// example arrays represent one B=2, source-length=4, target-length=3 batch.
await trainer.trainStep({
  inputs: {
    source_tokens: {
      data: Int32Array.of(12, 7, 4, 0, 3, 9, 0, 0),
      shape: [2, 4],
    },
    decoder_tokens: {
      data: Int32Array.of(1, 5, 8, 1, 6, 0),
      shape: [2, 3],
    },
    source_mask: {
      data: Int32Array.of(1, 1, 1, 0, 1, 1, 0, 0),
      shape: [2, 4],
    },
    target_mask: {
      data: Int32Array.of(1, 1, 1, 1, 1, 0),
      shape: [2, 3],
    },
  },
  logitsTensor: 'logits',
  targets: Int32Array.of(5, 8, 2, 6, 2, 0),
  lossMask: Float32Array.of(1, 1, 1, 1, 1, 0),
  trainableTensors: source.weightNames,
  updateMode: 'adamw',
  optimizer: { learningRate: 3e-4, maxGradNorm: 1 },
});

const checkpoint = await trainer.exportCheckpoint({
  metadata: { epoch: 1 },
});
const successor = await trainer.commit();
await trainer.close();

// Resume exact optimizer state. Import exposes a snapshot, never a mutable
// concrete graph.
const restored = importModelCheckpoint(checkpoint);
const resumed = await Trainer.create(restored.snapshot, {
  backend: 'cpu',
  checkpoint,
});
await resumed.close();
```

The exact input set is model-defined; position IDs, feature tensors, and other
masks must also be supplied as shaped views when present. `commit()` returns a
new immutable snapshot. It does not change `source`, and a checkpoint may also
be exported before commit to resume the Trainer's private working revision.
