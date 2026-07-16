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
await new VolvoxAI().trainStep(graph, batch);

// After a generic checkpoint import:
const resumed = createTeacherForcingBatchForGraph(graph, sourceIds, targetIds);
```
