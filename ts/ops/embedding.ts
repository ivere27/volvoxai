export function _cpuEmbedding(node) {
  const tokenTensor = node.inputs.input;
  const weight = node.inputs.weight;
  const output = node.outputs.out;
  if (tokenTensor?.dtype !== "int32" || !(tokenTensor.buffer instanceof Int32Array)) {
    throw new Error(`Embedding node ${node.id} token IDs must use int32 storage.`);
  }
  if (weight?.dtype !== "float32" || !(weight.buffer instanceof Float32Array) ||
      output?.dtype !== "float32" || !(output.buffer instanceof Float32Array) ||
      weight.shape.length !== 2) {
    throw new Error(`Embedding node ${node.id} requires a rank-2 float32 table and float32 output.`);
  }
  const tokens = tokenTensor.buffer;
  const dModel = output.shape[output.shape.length - 1];
  const vocabularySize = weight.shape[0];
  if (weight.shape[1] !== dModel || output.buffer.length !== tokens.length * dModel) {
    throw new Error(`Embedding node ${node.id} shapes are incompatible.`);
  }
  for (let i = 0; i < tokens.length; i++) {
    const tokenId = tokens[i];
    if (tokenId < 0 || tokenId >= vocabularySize) {
      throw new Error(`Embedding node ${node.id} token id ${tokenId} is outside vocabulary size ${vocabularySize}.`);
    }
    for (let j = 0; j < dModel; j++) {
      output.buffer[i * dModel + j] = weight.buffer[tokenId * dModel + j];
    }
  }
}
