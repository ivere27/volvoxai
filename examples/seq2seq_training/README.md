# Encoder-decoder training example

This directory is a model-specific training fixture. The application owns
tokenization, padding, teacher forcing, masks, and target construction.
The Trainer keeps private weights, gradients, and optimizer state while the
application prepares each teacher-forcing batch.

`Seq2SeqBuilder.js` remains a repository-internal fixed-shape numerical
fixture. It is not exported by a runtime entry and is not a public authoring
API. Production exporters should emit a bounded `volvox-graph/v1` graph plus
SafeTensors weights.

Use `FullEngineHost` and the generated inference/training clients, as shown in
the [training guide](../../docs/model_builder_training.md). The
sequence is:

1. Create a Runtime and load the package through `LoadModel`.
2. Create a Trainer for the returned model ID.
3. Send shaped `Tensor` inputs, typed losses, trainable names, and optimizer
   options through `TrainStep`.
4. Publish the private update with `CommitTrainer`, or discard it with
   `RollbackTrainer`.
5. Release the Trainer, Model, and Runtime handles.

A teacher-forcing step for this fixture conceptually supplies:

```javascript
const step = await training.trainStep(new pb.TrainStepRequest({
  trainerId: trainer.trainerId,
  inputs: [
    tensor('source_tokens', Int32Array.of(12, 7, 4, 0), [1n, 4n]),
    tensor('decoder_tokens', Int32Array.of(1, 5, 8), [1n, 3n]),
    tensor('source_mask', Int32Array.of(1, 1, 1, 0), [1n, 4n]),
    tensor('target_mask', Int32Array.of(1, 1, 1), [1n, 3n]),
  ],
  losses: [new pb.CrossEntropyLoss({
    name: 'tokens',
    logitsName: 'logits',
    targets: [5, 8, 2],
  })],
  trainableNames,
  optimizer: new pb.TrainerOptimizerOptions({
    kind: pb.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW,
    learningRate: 3e-4,
  }),
}));
```

Here `tensor` is application code that maps a typed array to a generated
`pb.Tensor` with the matching `DataType`, explicit shape, and byte view. The
exact input set is model-defined; position IDs, features, and additional masks
must be supplied when declared.

Batch and sequence extents may change between steps only inside the graph's
bounded domain. Existing compiled inference models remain pinned to their old
weight revision after commit. Save `ExportTrainerCheckpoint` bytes to resume
private weights, optimizer moments/settings, and RNG state through
`CreateTrainer.checkpoint`. `ExportTrainerWeights` returns SafeTensors bytes
for inference deployment. Browser applications choose how to save those bytes;
native full can also write weight shards to filesystem paths.
