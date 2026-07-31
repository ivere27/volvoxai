# Encoder-decoder training example

`Seq2SeqBuilder.js` is an application-level composition of VolvoxAI's generic
graph and training APIs. It builds a fixed-shape pre-norm Transformer
encoder-decoder and owns the associated token, padding, and teacher-forcing
policy. It is deliberately not exported from `volvoxai` or `volvoxai/full`.

```js
import { ModelBuilder, VolvoxAI } from "volvoxai/full";
import {
  buildEncoderDecoderTransformer,
  createTeacherForcingBatchForGraph,
} from "./Seq2SeqBuilder.js";

const model = new ModelBuilder();
const seq2seq = buildEncoderDecoderTransformer(model, {
  sourceLength: 32,
  targetLength: 24,
  vocabSize: 8000,
  dModel: 256,
  numHeads: 8,
});
const graph = model.build();
const batch = seq2seq.teacherForcing(sourceIds, targetIds);
const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
const runtimeModel = runtime.createModel(graph);
const trainer = await VolvoxAI.createTrainer(runtimeModel, { backend: 'cpu' });

await trainer.trainStep(batch);
await trainer.commit();

// After a generic checkpoint import:
const resumed = createTeacherForcingBatchForGraph(graph, sourceIds, targetIds);

await trainer.close();
await runtimeModel.close();
await runtime.close();
```

`trainStep()` changes only the Trainer's private working revision. Call
`commit()` before compiling inference against the update. Call `rollback()` to
discard uncommitted work and restore the last committed baseline.
