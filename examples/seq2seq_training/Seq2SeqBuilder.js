/* Example-owned encoder/decoder authoring and teacher-forcing policy.
 * The VolvoxAI package exposes generic graph/training primitives only. */

function positiveInteger(name, value) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive safe integer.`);
  return value;
}

function flattenIds(value, label) {
  const output = [];
  const visit = (entry) => {
    if (ArrayBuffer.isView(entry)) {
      for (const item of entry) visit(item);
    } else if (Array.isArray(entry)) {
      for (const item of entry) visit(item);
    } else {
      if (!Number.isInteger(entry)) throw new Error(`${label} must contain integer token ids.`);
      output.push(entry);
    }
  };
  visit(value);
  return output;
}

function checkedIds(value, expected, label, vocabularySize = null, batchSize = null, rowLength = null) {
  if (Array.isArray(value) && value.some((entry) => Array.isArray(entry) || ArrayBuffer.isView(entry))) {
    if (value.length !== batchSize || value.some((row) =>
      !(Array.isArray(row) || (ArrayBuffer.isView(row) && !(row instanceof DataView))) ||
      flattenIds(row, label).length !== rowLength)) {
      throw new Error(`${label} must be a rectangular [${batchSize},${rowLength}] batch.`);
    }
  }
  const values = flattenIds(value, label);
  if (values.length !== expected) throw new Error(`${label} contains ${values.length} ids; expected ${expected}.`);
  if (vocabularySize != null) {
    for (const token of values) {
      if (token < 0 || token >= vocabularySize) {
        throw new Error(`${label} token id ${token} is outside vocabulary size ${vocabularySize}.`);
      }
    }
  }
  return values;
}

function positions(batch, length) {
  const output = new Int32Array(batch * length);
  for (let row = 0; row < batch; row++) {
    for (let position = 0; position < length; position++) output[row * length + position] = position;
  }
  return output;
}

function optionValue(options, name, fallback) {
  return Object.prototype.hasOwnProperty.call(options, name) ? options[name] : fallback;
}

function validateSpecialTokenIds(padTokenId, bosTokenId, sourceVocabSize, targetVocabSize) {
  if (padTokenId !== null && (!Number.isInteger(padTokenId) || padTokenId < 0 ||
      padTokenId >= sourceVocabSize || padTokenId >= targetVocabSize)) {
    throw new Error(
      `padTokenId ${padTokenId} must be null or an integer inside both source (${sourceVocabSize}) ` +
      `and target (${targetVocabSize}) vocabularies.`,
    );
  }
  if (!Number.isInteger(bosTokenId) || bosTokenId < 0 || bosTokenId >= targetVocabSize) {
    throw new Error(`bosTokenId ${bosTokenId} is outside target vocabulary size ${targetVocabSize}.`);
  }
  if (padTokenId !== null && padTokenId === bosTokenId) {
    throw new Error("padTokenId and bosTokenId must be different.");
  }
}

function sameShape(left, right) {
  return Array.isArray(left) && left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

/** Validate checkpointed metadata before it can select inputs or trainable weights. */
export function validateEncoderDecoderTrainingMetadata(graph, metadata = graph?.trainingMetadata) {
  if (!graph || metadata?.kind !== "encoder-decoder-transformer" ||
      !metadata.teacherForcingSpec || !Array.isArray(metadata.trainableTensors)) {
    throw new Error("Graph does not contain valid encoder-decoder teacher-forcing metadata.");
  }
  const spec = metadata.teacherForcingSpec;
  const batchSize = positiveInteger("teacherForcingSpec.batchSize", spec.batchSize);
  const sourceLength = positiveInteger("teacherForcingSpec.sourceLength", spec.sourceLength);
  const sourceFeatureLength = spec.sourceFeatureLength == null
    ? 0
    : Number.isSafeInteger(spec.sourceFeatureLength) && spec.sourceFeatureLength >= 0
      ? spec.sourceFeatureLength
      : -1;
  if (sourceFeatureLength < 0) {
    throw new Error("teacherForcingSpec.sourceFeatureLength must be a non-negative safe integer.");
  }
  const memoryLength = sourceLength + sourceFeatureLength;
  if (!Number.isSafeInteger(memoryLength)) {
    throw new Error("teacherForcingSpec memory length exceeds the safe integer range.");
  }
  if (spec.memoryLength != null && spec.memoryLength !== memoryLength) {
    throw new Error("teacherForcingSpec.memoryLength does not match its token and feature lengths.");
  }
  const targetLength = positiveInteger("teacherForcingSpec.targetLength", spec.targetLength);
  const sourceVocabSize = positiveInteger("teacherForcingSpec.sourceVocabSize", spec.sourceVocabSize);
  const targetVocabSize = positiveInteger("teacherForcingSpec.targetVocabSize", spec.targetVocabSize);
  validateSpecialTokenIds(spec.padTokenId, spec.bosTokenId, sourceVocabSize, targetVocabSize);

  const inputRequirements = [
    ["sourceTokensName", [batchSize, sourceLength]],
    ["sourcePositionsName", [batchSize, sourceLength]],
    ["sourceMaskName", [batchSize, memoryLength]],
    ["decoderTokensName", [batchSize, targetLength]],
    ["targetPositionsName", [batchSize, targetLength]],
    ["targetMaskName", [batchSize, targetLength]],
  ];
  const inputNames = new Set();
  for (const [field, shape] of inputRequirements) {
    const name = spec[field];
    if (typeof name !== "string" || !name || inputNames.has(name)) {
      throw new Error(`teacherForcingSpec.${field} must name a unique input tensor.`);
    }
    const tensor = graph.getTensor(name);
    if (!tensor?.isInput || tensor.dtype !== "int32" || !sameShape(tensor.shape, shape)) {
      throw new Error(`teacherForcingSpec.${field} must reference an int32 input shaped [${shape}].`);
    }
    inputNames.add(name);
  }
  const logits = graph.getTensor(spec.logitsName);
  if (typeof spec.logitsName !== "string" || !logits || logits.dtype !== "float32" ||
      !sameShape(logits.shape, [batchSize, targetLength, targetVocabSize]) ||
      !graph.outputNames?.includes(spec.logitsName)) {
    throw new Error(
      `teacherForcingSpec.logitsName must reference a selected float32 output shaped ` +
      `[${batchSize},${targetLength},${targetVocabSize}].`,
    );
  }
  if (metadata.trainableTensors.length === 0 ||
      new Set(metadata.trainableTensors).size !== metadata.trainableTensors.length) {
    throw new Error("Encoder-decoder trainableTensors must be a non-empty unique list.");
  }
  for (const name of metadata.trainableTensors) {
    const tensor = typeof name === "string" ? graph.getTensor(name) : null;
    if (!tensor?.isWeight || tensor.dtype !== "float32" || !(tensor.buffer instanceof Float32Array)) {
      throw new Error(`Encoder-decoder trainable tensor '${String(name)}' must be an initialized float32 weight.`);
    }
  }
  return metadata;
}

/** Create a teacher-forcing batch for a descriptor returned by the builder. */
export function createTeacherForcingBatch(descriptor, sourceIds, targetIds, options = {}) {
  if (!descriptor?._teacherForcingSpec) throw new Error("Teacher forcing requires an encoder-decoder descriptor.");
  const spec = descriptor._teacherForcingSpec;
  const source = checkedIds(
    sourceIds,
    spec.batchSize * spec.sourceLength,
    "sourceIds",
    spec.sourceVocabSize,
    spec.batchSize,
    spec.sourceLength,
  );
  const targets = checkedIds(
    targetIds,
    spec.batchSize * spec.targetLength,
    "targetIds",
    spec.targetVocabSize,
    spec.batchSize,
    spec.targetLength,
  );
  const padTokenId = optionValue(options, "padTokenId", spec.padTokenId);
  const bosTokenId = optionValue(options, "bosTokenId", spec.bosTokenId);
  validateSpecialTokenIds(padTokenId, bosTokenId, spec.sourceVocabSize, spec.targetVocabSize);

  const decoder = new Int32Array(targets.length);
  for (let batchIndex = 0; batchIndex < spec.batchSize; batchIndex++) {
    const base = batchIndex * spec.targetLength;
    decoder[base] = bosTokenId;
    for (let position = 1; position < spec.targetLength; position++) {
      decoder[base + position] = targets[base + position - 1];
    }
  }
  const sourceArray = Int32Array.from(source);
  const targetArray = Int32Array.from(targets);
  const sourceFeatureLength = spec.sourceFeatureLength ?? 0;
  const memoryLength = spec.sourceLength + sourceFeatureLength;
  const sourceMask = new Int32Array(spec.batchSize * memoryLength);
  const targetMask = new Int32Array(decoder.length);
  const lossMask = new Float32Array(targetArray.length);
  for (let batchIndex = 0; batchIndex < spec.batchSize; batchIndex++) {
    const sourceBase = batchIndex * spec.sourceLength;
    const memoryBase = batchIndex * memoryLength;
    sourceMask.fill(1, memoryBase, memoryBase + sourceFeatureLength);
    for (let position = 0; position < spec.sourceLength; position++) {
      const token = sourceArray[sourceBase + position];
      sourceMask[memoryBase + sourceFeatureLength + position] =
        padTokenId == null || token !== padTokenId ? 1 : 0;
    }
  }
  for (let index = 0; index < decoder.length; index++) {
    targetMask[index] = padTokenId == null || decoder[index] !== padTokenId ? 1 : 0;
    lossMask[index] = padTokenId == null || targetArray[index] !== padTokenId ? 1 : 0;
  }
  if (options.lossMask != null) {
    const explicitMask = Array.from(options.lossMask);
    if (explicitMask.length !== lossMask.length) {
      throw new Error(`lossMask contains ${explicitMask.length} values; expected ${lossMask.length}.`);
    }
    for (let index = 0; index < lossMask.length; index++) {
      const value = explicitMask[index];
      if (value !== true && value !== false && value !== 0 && value !== 1) {
        throw new Error("lossMask values must be boolean or 0/1.");
      }
      lossMask[index] *= value ? 1 : 0;
    }
  }

  return {
    inputs: {
      [spec.sourceTokensName]: sourceArray,
      [spec.sourcePositionsName]: positions(spec.batchSize, spec.sourceLength),
      [spec.sourceMaskName]: sourceMask,
      [spec.decoderTokensName]: decoder,
      [spec.targetPositionsName]: positions(spec.batchSize, spec.targetLength),
      [spec.targetMaskName]: targetMask,
    },
    targets: targetArray,
    lossMask,
    ...(padTokenId == null ? {} : { ignoreIndex: padTokenId }),
    logitsTensor: spec.logitsName,
    trainableTensors: [...descriptor.trainableTensors],
  };
}

/** Recreate a teacher-forcing batch after importing a full model checkpoint. */
export function createTeacherForcingBatchForGraph(graph, sourceIds, targetIds, options = {}) {
  const metadata = graph?.trainingMetadata;
  validateEncoderDecoderTrainingMetadata(graph, metadata);
  return createTeacherForcingBatch({
    _teacherForcingSpec: metadata.teacherForcingSpec,
    trainableTensors: metadata.trainableTensors,
  }, sourceIds, targetIds, options);
}

/** Build a fixed-shape pre-norm Transformer encoder-decoder in a ModelBuilder. */
function buildEncoderDecoderTransformerUnsafe(builder, options = {}) {
  const name = options.name || "seq2seq";
  const batchSize = positiveInteger("batchSize", options.batchSize ?? 1);
  const sourceLength = positiveInteger("sourceLength", options.sourceLength);
  const targetLength = positiveInteger("targetLength", options.targetLength);
  const sourceVocabSize = positiveInteger(
    "sourceVocabSize",
    options.sourceVocabSize ?? options.vocabSize,
  );
  const targetVocabSize = positiveInteger(
    "targetVocabSize",
    options.targetVocabSize ?? options.vocabSize,
  );
  const dModel = positiveInteger("dModel", options.dModel);
  const numHeads = positiveInteger("numHeads", options.numHeads);
  const dFF = positiveInteger("dFF", options.dFF ?? dModel * 4);
  const encoderLayers = positiveInteger("encoderLayers", options.encoderLayers ?? 1);
  const decoderLayers = positiveInteger("decoderLayers", options.decoderLayers ?? 1);
  if (dModel % numHeads !== 0) throw new Error("dModel must be divisible by numHeads.");
  const sourceFeatures = options.sourceFeatures ?? null;
  let sourceFeatureLength = 0;
  if (sourceFeatures != null) {
    if (sourceFeatures.dtype !== "float32" || !Array.isArray(sourceFeatures.shape) ||
        sourceFeatures.shape.length !== 3 || sourceFeatures.shape[0] !== batchSize ||
        sourceFeatures.shape[2] !== dModel || !builder.graph.getTensor(sourceFeatures.name)) {
      throw new Error(
        `sourceFeatures must be an existing float32 tensor shaped [${batchSize},featureLength,${dModel}].`,
      );
    }
    sourceFeatureLength = positiveInteger("sourceFeatures featureLength", sourceFeatures.shape[1]);
  }
  const memoryLength = sourceLength + sourceFeatureLength;
  if (!Number.isSafeInteger(memoryLength)) throw new Error("combined source memory length is too large.");
  const padTokenId = optionValue(options, "padTokenId", 0);
  const bosTokenId = optionValue(options, "bosTokenId", 1);
  validateSpecialTokenIds(padTokenId, bosTokenId, sourceVocabSize, targetVocabSize);
  const epsilon = options.epsilon ?? 1e-5;
  if (!Number.isFinite(epsilon) || epsilon <= 0) throw new Error("epsilon must be positive and finite.");
  const dropoutProbability = options.dropout ?? 0;
  if (typeof dropoutProbability !== "number" || !Number.isFinite(dropoutProbability) ||
      dropoutProbability < 0 || dropoutProbability >= 1) {
    throw new Error("dropout must be finite and in [0, 1).");
  }
  const attentionDropoutProbability = options.attentionDropout ?? dropoutProbability;
  if (typeof attentionDropoutProbability !== "number" ||
      !Number.isFinite(attentionDropoutProbability) || attentionDropoutProbability < 0 ||
      attentionDropoutProbability >= 1) {
    throw new Error("attentionDropout must be finite and in [0, 1).");
  }
  const dropoutSeed = options.dropoutSeed ?? options.seed ?? 0;
  if (!Number.isSafeInteger(dropoutSeed) || dropoutSeed < 0) {
    throw new Error("dropoutSeed must be a non-negative safe integer.");
  }

  let seed = Number.isFinite(options.seed) ? Math.trunc(options.seed) : 0;
  const trainableTensors = [];
  const trainableNames = new Set();
  const registerTrainable = (reference) => {
    const tensor = typeof reference === "string" ? builder.getTensor(reference) : reference;
    if (!tensor?.isWeight || tensor.dtype !== "float32" || !builder.graph.getTensor(tensor.name)) {
      throw new Error("additional trainable tensors must reference existing float32 graph weights.");
    }
    if (!trainableNames.has(tensor.name)) {
      trainableNames.add(tensor.name);
      trainableTensors.push(tensor.name);
    }
    return tensor;
  };
  if (options.additionalTrainableTensors != null) {
    if (!Array.isArray(options.additionalTrainableTensors)) {
      throw new Error("additionalTrainableTensors must be an array.");
    }
    for (const tensor of options.additionalTrainableTensors) registerTrainable(tensor);
  }
  const weight = (suffix, shape, initializer) => {
    const tensor = builder.weight(`${name}.${suffix}`, shape, "float32", {
      initializer: { ...(typeof initializer === "string" ? { type: initializer } : initializer), seed: seed++ },
    });
    registerTrainable(tensor);
    return tensor;
  };
  const output = (suffix, shape) => ({ name: `${name}.${suffix}`, shape });
  const add = (a, b, suffix) => builder.addOp(
    "Add",
    { a, b },
    { out: output(suffix, [...a.shape]) },
    {},
    { id: `${name}.${suffix}` },
  ).out;
  const embedding = (tokens, table, suffix, shape) => builder.addOp(
    "Embedding",
    { input: tokens, weight: table },
    { out: output(suffix, shape) },
    {},
    { id: `${name}.${suffix}` },
  ).out;
  let dropoutIndex = 0;
  const dropout = (input, suffix) => {
    if (dropoutProbability === 0) return input;
    return builder.addOp(
      "Dropout",
      { input },
      { out: output(suffix, [...input.shape]) },
      { ratio: dropoutProbability, seed: (dropoutSeed + dropoutIndex++) >>> 0 },
      { id: `${name}.${suffix}` },
    ).out;
  };
  const attentionDropoutParams = () => attentionDropoutProbability === 0
    ? {}
    : {
      dropout: attentionDropoutProbability,
      dropout_seed: (dropoutSeed + dropoutIndex++) >>> 0,
    };
  const norm = (input, suffix) => {
    const scale = weight(`${suffix}.scale`, [dModel], "ones");
    const bias = weight(`${suffix}.bias`, [dModel], "zeros");
    return builder.addOp(
      "LayerNorm",
      { input, weight: scale, bias },
      { out: output(`${suffix}.out`, [...input.shape]) },
      { d_model: dModel, eps: epsilon },
      { id: `${name}.${suffix}` },
    ).out;
  };
  const linear = (input, outFeatures, suffix) => {
    const inFeatures = input.shape.at(-1);
    const matrix = weight(`${suffix}.weight`, [inFeatures, outFeatures], "xavierUniform");
    const bias = weight(`${suffix}.bias`, [outFeatures], "zeros");
    return builder.addOp(
      "Linear",
      { input, weight: matrix, bias },
      { out: output(`${suffix}.out`, [...input.shape.slice(0, -1), outFeatures]) },
      { weight_layout: "din_dout" },
      { id: `${name}.${suffix}`, wLayout: "din" },
    ).out;
  };
  const feedForward = (input, prefix) => {
    const normalized = norm(input, `${prefix}.norm`);
    const expanded = linear(normalized, dFF, `${prefix}.in`);
    let activated = builder.addOp(
      "GELU",
      { input: expanded },
      { out: output(`${prefix}.gelu`, [...expanded.shape]) },
      {},
      { id: `${name}.${prefix}.gelu` },
    ).out;
    activated = dropout(activated, `${prefix}.activation_dropout`);
    const projected = dropout(
      linear(activated, dModel, `${prefix}.out`),
      `${prefix}.output_dropout`,
    );
    return add(input, projected, `${prefix}.residual`);
  };

  const sourceTokens = builder.input(`${name}.source_tokens`, [batchSize, sourceLength], "int32");
  const sourcePositions = builder.input(`${name}.source_positions`, [batchSize, sourceLength], "int32");
  const sourceMask = builder.input(`${name}.source_mask`, [batchSize, memoryLength], "int32");
  const decoderTokens = builder.input(`${name}.decoder_tokens`, [batchSize, targetLength], "int32");
  const targetPositions = builder.input(`${name}.target_positions`, [batchSize, targetLength], "int32");
  const targetMask = builder.input(`${name}.target_mask`, [batchSize, targetLength], "int32");

  const sourceTokenTable = weight("source_embedding.weight", [sourceVocabSize, dModel], {
    type: "normal", stddev: options.embeddingStddev ?? 0.02,
  });
  const sourcePositionTable = weight("source_position_embedding.weight", [sourceLength, dModel], {
    type: "normal", stddev: options.embeddingStddev ?? 0.02,
  });
  const sourceTokenEmbedding = add(
    embedding(sourceTokens, sourceTokenTable, "source_embedding", [batchSize, sourceLength, dModel]),
    embedding(sourcePositions, sourcePositionTable, "source_position_embedding", [batchSize, sourceLength, dModel]),
    "source_embedding_sum",
  );
  let memory = sourceTokenEmbedding;
  if (sourceFeatures) {
    memory = builder.addOp(
      "Concat",
      { a: sourceFeatures, b: sourceTokenEmbedding },
      { out: output("source_feature_concat", [batchSize, memoryLength, dModel]) },
      { axis: 1 },
      { id: `${name}.source_feature_concat` },
    ).out;
  }

  for (let layer = 0; layer < encoderLayers; layer++) {
    const prefix = `encoder.${layer}`;
    const normalized = norm(memory, `${prefix}.self_attention.norm`);
    const qkv = linear(normalized, dModel * 3, `${prefix}.self_attention.qkv`);
    const attention = builder.addOp(
      "SDPA",
      { qkv, mask: sourceMask },
      { out: output(`${prefix}.self_attention.attention`, [...memory.shape]) },
      { heads: numHeads, causal: false, ...attentionDropoutParams() },
      { id: `${name}.${prefix}.self_attention.attention` },
    ).out;
    memory = add(
      memory,
      dropout(
        linear(attention, dModel, `${prefix}.self_attention.output`),
        `${prefix}.self_attention.output_dropout`,
      ),
      `${prefix}.self_attention.residual`,
    );
    memory = feedForward(memory, `${prefix}.feed_forward`);
  }
  if (options.finalEncoderNorm !== false) memory = norm(memory, "encoder.final_norm");
  if (options.transformMemory != null) {
    if (typeof options.transformMemory !== "function") {
      throw new Error("transformMemory must be a function.");
    }
    const transformed = options.transformMemory(memory, Object.freeze({
      builder,
      sourceMask,
      registerTrainable,
      name,
    }));
    if (!transformed || transformed.dtype !== "float32" ||
        !sameShape(transformed.shape, memory.shape) || !builder.graph.getTensor(transformed.name)) {
      throw new Error(`transformMemory must return an existing float32 tensor shaped [${memory.shape}].`);
    }
    memory = transformed;
  }

  if (options.tieSourceTargetEmbeddings === true && sourceVocabSize !== targetVocabSize) {
    throw new Error("tieSourceTargetEmbeddings requires equal source and target vocabulary sizes.");
  }
  const targetTokenTable = options.tieSourceTargetEmbeddings === true
    ? sourceTokenTable
    : weight("target_embedding.weight", [targetVocabSize, dModel], {
      type: "normal", stddev: options.embeddingStddev ?? 0.02,
    });
  const targetPositionTable = weight("target_position_embedding.weight", [targetLength, dModel], {
    type: "normal", stddev: options.embeddingStddev ?? 0.02,
  });
  let hidden = add(
    embedding(decoderTokens, targetTokenTable, "target_embedding", [batchSize, targetLength, dModel]),
    embedding(targetPositions, targetPositionTable, "target_position_embedding", [batchSize, targetLength, dModel]),
    "target_embedding_sum",
  );

  for (let layer = 0; layer < decoderLayers; layer++) {
    const prefix = `decoder.${layer}`;
    let normalized = norm(hidden, `${prefix}.self_attention.norm`);
    const qkv = linear(normalized, dModel * 3, `${prefix}.self_attention.qkv`);
    let attention = builder.addOp(
      "SDPA",
      { qkv, mask: targetMask },
      { out: output(`${prefix}.self_attention.attention`, [...hidden.shape]) },
      { heads: numHeads, causal: true, ...attentionDropoutParams() },
      { id: `${name}.${prefix}.self_attention.attention` },
    ).out;
    hidden = add(
      hidden,
      dropout(
        linear(attention, dModel, `${prefix}.self_attention.output`),
        `${prefix}.self_attention.output_dropout`,
      ),
      `${prefix}.self_attention.residual`,
    );

    normalized = norm(hidden, `${prefix}.cross_attention.norm`);
    const query = linear(normalized, dModel, `${prefix}.cross_attention.query`);
    const key = linear(memory, dModel, `${prefix}.cross_attention.key`);
    const value = linear(memory, dModel, `${prefix}.cross_attention.value`);
    attention = builder.addOp(
      "CrossSDPA",
      { q: query, k: key, v: value, mask: sourceMask },
      { out: output(`${prefix}.cross_attention.attention`, [...hidden.shape]) },
      { heads: numHeads, causal: false, ...attentionDropoutParams() },
      { id: `${name}.${prefix}.cross_attention.attention` },
    ).out;
    hidden = add(
      hidden,
      dropout(
        linear(attention, dModel, `${prefix}.cross_attention.output`),
        `${prefix}.cross_attention.output_dropout`,
      ),
      `${prefix}.cross_attention.residual`,
    );
    hidden = feedForward(hidden, `${prefix}.feed_forward`);
  }
  if (options.transformDecoderOutput != null) {
    if (typeof options.transformDecoderOutput !== "function") {
      throw new Error("transformDecoderOutput must be a function.");
    }
    const transformed = options.transformDecoderOutput(hidden, Object.freeze({
      builder,
      sourceMask,
      targetMask,
      registerTrainable,
      name,
    }));
    if (!transformed || transformed.dtype !== "float32" ||
        !sameShape(transformed.shape, hidden.shape) || !builder.graph.getTensor(transformed.name)) {
      throw new Error(`transformDecoderOutput must return an existing float32 tensor shaped [${hidden.shape}].`);
    }
    hidden = transformed;
  }
  if (options.finalDecoderNorm !== false) hidden = norm(hidden, "decoder.final_norm");

  let logits;
  if (options.tieTargetEmbeddings === true) {
    const projectionSuffix = options.lmHeadBias === true ? "lm_head.projection" : "logits";
    logits = builder.addOp(
      "Linear",
      { input: hidden, weight: targetTokenTable },
      { out: output(projectionSuffix, [batchSize, targetLength, targetVocabSize]) },
      { weight_layout: "dout_din" },
      { id: `${name}.lm_head`, wLayout: "dout" },
    ).out;
    if (options.lmHeadBias === true) {
      logits = add(logits, weight("lm_head.bias", [targetVocabSize], "zeros"), "logits");
    }
  } else {
    logits = linear(hidden, targetVocabSize, "lm_head");
    builder.renameTensor(logits.name, `${name}.logits`);
    logits = builder.getTensor(`${name}.logits`);
  }
  builder.outputs(logits);

  const descriptor = {
    graph: builder.graph,
    logits,
    memory,
    trainableTensors: Object.freeze([...trainableTensors]),
    inputs: Object.freeze({
      sourceTokens,
      sourcePositions,
      sourceMask,
      ...(sourceFeatures ? { sourceFeatures } : {}),
      decoderTokens,
      targetPositions,
      targetMask,
    }),
    _teacherForcingSpec: Object.freeze({
      batchSize,
      sourceLength,
      sourceFeatureLength,
      memoryLength,
      targetLength,
      sourceVocabSize,
      targetVocabSize,
      padTokenId,
      bosTokenId,
      sourceTokensName: sourceTokens.name,
      sourcePositionsName: sourcePositions.name,
      sourceMaskName: sourceMask.name,
      decoderTokensName: decoderTokens.name,
      targetPositionsName: targetPositions.name,
      targetMaskName: targetMask.name,
      logitsName: logits.name,
    }),
  };
  descriptor.teacherForcing = (sourceIds, targetIds, batchOptions = {}) =>
    createTeacherForcingBatch(descriptor, sourceIds, targetIds, batchOptions);
  const result = Object.freeze(descriptor);
  builder.graph.trainingMetadata = {
    kind: "encoder-decoder-transformer",
    teacherForcingSpec: { ...descriptor._teacherForcingSpec },
    trainableTensors: [...descriptor.trainableTensors],
  };
  return result;
}

/** Build atomically: a failed high-level build restores the exact prior graph. */
export function buildEncoderDecoderTransformer(builder, options = {}) {
  if (!builder?.graph || typeof builder.topologyTransaction !== "function") {
    throw new Error("buildEncoderDecoderTransformer expects a ModelBuilder.");
  }
  return builder.topologyTransaction(() => buildEncoderDecoderTransformerUnsafe(builder, options));
}
