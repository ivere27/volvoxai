/*
 * Example typed host session for a materialized TinyReceiptVQA W8A8 package.
 *
 * A materialized package exposes two ordinary Volvox graphs: an I32/QArgMax
 * router and one I32/QArgMax explicit-family graph.  Keeping the sequencing
 * here makes the model-format contract equally usable by the JS CPU, WASM,
 * and WebGPU graph executors without slipping back to an F32-logit host path.
 */

const PACKAGE_FORMAT = 'volvoxai-tiny-receipt-vqa-w8a8-materialized-package-v1';
const FAMILY_ORDER = Object.freeze([
  'phone', 'address', 'store', 'item_row', 'item_math', 'item_lookup', 'math', 'other',
]);

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function fail(message) {
  throw new Error(`[TinyReceiptW8A8Session] ${message}`);
}

function isReadOnlySafetensorsCache(value) {
  return !!value && typeof value === 'object' &&
    typeof value.load === 'function' && typeof value.clear === 'function' &&
    Number.isInteger(value.size) && value.size >= 0;
}

function nonEmptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) fail(`${label} must be a non-empty string.`);
  return value;
}

function nonNegativeInteger(value, label) {
  if (!Number.isInteger(value) || value < 0) fail(`${label} must be a non-negative integer.`);
  return value;
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function assetPath(value, label) {
  const path = nonEmptyString(value, label);
  if (path.startsWith('/') || path.startsWith('\\') || path.includes('\\') ||
      path.includes('?') || path.includes('#') || path.split('/').some((part) => part === '.' || part === '..' || part === '')) {
    fail(`${label} must be a package-relative asset path.`);
  }
  return path;
}

function manifestUrl(value) {
  if (value instanceof URL) return value;
  const source = nonEmptyString(value, 'packageUrl');
  const base = globalThis.location?.href || import.meta.url;
  const url = new URL(source, base);
  return url.pathname.endsWith('/') ? new URL('package_manifest.json', url) : url;
}

function assetUrl(manifest, asset, label) {
  return new URL(assetPath(asset, label), new URL('.', manifest)).toString();
}

function inputName(value, label) {
  return nonEmptyString(value, label);
}

function inputTensor(graph, name, label) {
  const tensor = graph?.getTensor?.(name) || graph?.tensors?.get?.(name);
  if (!tensor || tensor.isInput !== true) fail(`${label} '${name}' must be a graph input.`);
  return tensor;
}

function tensor(graph, name, label) {
  const result = graph?.getTensor?.(name) || graph?.tensors?.get?.(name);
  if (!result) fail(`${label} '${name}' does not exist in the graph.`);
  return result;
}

function requireI32Sequence(tensorValue, label) {
  if (tensorValue.dtype !== 'int32' || tensorValue.quantization != null ||
      !sameShape(tensorValue.shape, [1, tensorValue.shape?.[1]]) ||
      !Number.isInteger(tensorValue.shape?.[1]) || tensorValue.shape[1] <= 0) {
    fail(`${label} must be an unquantized I32 [1,S] graph input.`);
  }
  return tensorValue.shape[1];
}

function requireImageTensor(tensorValue, width, height, label) {
  if (tensorValue.dtype !== 'float32' || tensorValue.quantization != null ||
      !sameShape(tensorValue.shape, [1, height, width, 1])) {
    fail(`${label} must be an unquantized F32 [1,${height},${width},1] graph input.`);
  }
}

function assertQArgMaxOutput(graph, name, shape, label) {
  const result = tensor(graph, name, label);
  if (result.dtype !== 'int32' || result.quantization != null || !sameShape(result.shape, shape)) {
    fail(`${label} '${name}' must be an unquantized I32 [${shape.join(',')}] tensor.`);
  }
  if (Array.isArray(graph?.outputNames) && !graph.outputNames.includes(name)) {
    fail(`${label} '${name}' must be declared in the graph document outputs list.`);
  }
  const producer = graph?.nodes?.find((node) => Object.values(node.outputs || {}).some((output) => output?.name === name));
  if (!producer || producer.opType !== 'QArgMax') {
    fail(`${label} '${name}' must be produced directly by a terminal QArgMax node.`);
  }
  return result;
}

function validateRouterGraph(graph, router) {
  const qLength = requireI32Sequence(inputTensor(graph, router.inputs.qIds, 'router q_ids input'), 'router q_ids input');
  const keepLength = requireI32Sequence(inputTensor(graph, router.inputs.routerKeep, 'router router_keep input'), 'router router_keep input');
  if (qLength !== keepLength) fail('router q_ids and router_keep sequence lengths must match.');
  assertQArgMaxOutput(graph, router.outputName, [1], 'router output');
  return { qLength };
}

function validateFamilyGraph(graph, family, preprocessing, expectedQLength) {
  const inputs = family.inputs;
  requireImageTensor(inputTensor(graph, inputs.image, 'family image input'), preprocessing.width, preprocessing.height, 'family image input');
  const qLength = requireI32Sequence(inputTensor(graph, inputs.qIds, 'family q_ids input'), 'family q_ids input');
  const keepLength = requireI32Sequence(inputTensor(graph, inputs.routerKeep, 'family router_keep input'), 'family router_keep input');
  const memoryLength = requireI32Sequence(inputTensor(graph, inputs.memoryKeep, 'family memory_keep input'), 'family memory_keep input');
  const decoderLength = requireI32Sequence(inputTensor(graph, inputs.yIds, 'family y_ids input'), 'family y_ids input');
  const decoderKeepLength = requireI32Sequence(inputTensor(graph, inputs.yKeep, 'family y_keep input'), 'family y_keep input');
  if (qLength !== expectedQLength || keepLength !== qLength || decoderKeepLength !== decoderLength || memoryLength < qLength) {
    fail('family graph does not match the router typed sequence ABI.');
  }
  assertQArgMaxOutput(graph, family.outputName, [1, decoderLength], 'family output');
  return { qLength, memoryLength, decoderLength };
}

function cleanText(value) {
  // Normalize canonically, then apply the source
  // `re.sub(r"\\s+", " ", str(value).replace("\\n", " ")).strip()` policy.
  // `for...of` below walks Unicode code points, matching Python's CharVocab
  // character indexing rather than UTF-8 bytes or the generic BPE tokenizer.
  return String(value ?? '').normalize('NFC')
    .replace(/\n/g, ' ').replace(/\s+/gu, ' ').trim();
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function rawPixelSource(image) {
  if (!isRecord(image) || !ArrayBuffer.isView(image.data) || image.data instanceof DataView ||
      !Number.isInteger(image.width) || image.width <= 0 || !Number.isInteger(image.height) || image.height <= 0) {
    return null;
  }
  const pixels = image.width * image.height;
  const channels = image.channels ?? image.data.length / pixels;
  if (!Number.isInteger(channels) || ![1, 3, 4].includes(channels) || image.data.length !== pixels * channels) {
    fail('raw image data must use exactly 1, 3, or 4 tightly packed channels.');
  }
  return { data: image.data, width: image.width, height: image.height, channels };
}

async function drawablePixelSource(image) {
  const width = image?.naturalWidth || image?.videoWidth || image?.width;
  const height = image?.naturalHeight || image?.videoHeight || image?.height;
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    fail('image must be raw { data, width, height }, an ImageData-like value, or a drawable browser image.');
  }
  let canvas = null;
  if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(width, height);
  } else if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
    canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
  }
  if (!canvas) {
    fail('drawable image decoding is unavailable here; Node callers must supply raw pixels or preprocessed F32 NHWC data.');
  }
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) fail('could not create a 2D canvas context for image preprocessing.');
  context.drawImage(image, 0, 0, width, height);
  const imageData = context.getImageData(0, 0, width, height);
  return { data: imageData.data, width, height, channels: 4 };
}

function grayscalePlane(source) {
  const gray = new Float32Array(source.width * source.height);
  for (let index = 0, pixel = 0; pixel < gray.length; pixel++, index += source.channels) {
    let value;
    if (source.channels === 1) {
      value = Number(source.data[index]);
    } else {
      // PIL's `convert("L")` (the source evaluator contract) produces an
      // 8-bit luminance image before it is resized.  Preserve that discrete
      // grayscale boundary rather than interpolating independent RGB planes.
      value = Math.round(0.299 * Number(source.data[index]) +
        0.587 * Number(source.data[index + 1]) + 0.114 * Number(source.data[index + 2]));
    }
    if (!Number.isFinite(value)) fail('raw image pixels must be finite numbers.');
    gray[pixel] = clamp(value, 0, 255);
  }
  return gray;
}

function resizeAndNormalize(gray, inputWidth, inputHeight, outputWidth, outputHeight) {
  const output = new Float32Array(outputWidth * outputHeight);
  // PIL BILINEAR widens its triangle filter while downsampling.  The source
  // evaluator first converts to an 8-bit L plane, resizes that plane, then
  // converts its rounded 8-bit result to F32.  Keep that order rather than
  // resizing RGB planes or exporting an unrounded interpolation result.
  const axisKernel = (inputSize, outputSize, destination) => {
    const scale = inputSize / outputSize;
    const filterScale = Math.max(scale, 1);
    const center = (destination + 0.5) * scale;
    let first = Math.ceil(center - filterScale - 0.5);
    let last = Math.floor(center + filterScale - 0.5);
    first = clamp(first, 0, inputSize - 1);
    last = clamp(last, 0, inputSize - 1);
    if (first > last) first = last = clamp(Math.floor(center), 0, inputSize - 1);
    const entries = [];
    let total = 0;
    for (let source = first; source <= last; source++) {
      const weight = Math.max(0, 1 - Math.abs((source + 0.5 - center) / filterScale));
      if (weight !== 0) {
        entries.push([source, weight]);
        total += weight;
      }
    }
    if (!(total > 0)) fail('could not construct a bilinear image-resize kernel.');
    for (const entry of entries) entry[1] /= total;
    return entries;
  };
  const xKernels = Array.from({ length: outputWidth }, (_, x) => axisKernel(inputWidth, outputWidth, x));
  const yKernels = Array.from({ length: outputHeight }, (_, y) => axisKernel(inputHeight, outputHeight, y));
  const horizontal = new Float32Array(inputHeight * outputWidth);
  for (let y = 0; y < inputHeight; y++) {
    for (let x = 0; x < outputWidth; x++) {
      let pixel = 0;
      for (const [sourceX, weight] of xKernels[x]) pixel += gray[y * inputWidth + sourceX] * weight;
      horizontal[y * outputWidth + x] = pixel;
    }
  }
  for (let y = 0; y < outputHeight; y++) {
    for (let x = 0; x < outputWidth; x++) {
      let pixel = 0;
      for (const [sourceY, weight] of yKernels[y]) pixel += horizontal[sourceY * outputWidth + x] * weight;
      // `(pixel / 255.0 - 0.5) / 0.5` is recorded verbatim in the package.
      const rounded = Math.round(clamp(pixel, 0, 255));
      output[y * outputWidth + x] = (rounded / 255.0 - 0.5) / 0.5;
    }
  }
  return output;
}

/**
 * Convert raw pixels to the materialized package's F32 NHWC image tensor.
 * The returned flat `Float32Array` has shape `[1,height,width,1]`.
 *
 * Browser callers may pass ImageData, ImageBitmap, HTMLImageElement, or a
 * canvas. Node callers intentionally supply `{ data, width, height, channels
 * }`, avoiding a hidden image-decoder dependency in the inference entry.
 */
export async function preprocessTinyReceiptImage(image, preprocessing) {
  if (!preprocessing || !Number.isInteger(preprocessing.width) || !Number.isInteger(preprocessing.height)) {
    fail('preprocessing requires positive width and height.');
  }
  const source = rawPixelSource(image) || await drawablePixelSource(image);
  return resizeAndNormalize(
    grayscalePlane(source), source.width, source.height, preprocessing.width, preprocessing.height,
  );
}

export class TinyReceiptCharVocab {
  constructor(itos, tokenIds) {
    if (!Array.isArray(itos) || itos.length === 0 || itos.some((value) => typeof value !== 'string')) {
      fail('vocab.json must contain a non-empty string itos array.');
    }
    this.itos = Object.freeze([...itos]);
    this.stoi = new Map();
    for (let index = 0; index < this.itos.length; index++) this.stoi.set(this.itos[index], index);
    this.pad = nonNegativeInteger(tokenIds.pad, 'vocab.token_ids.pad');
    this.bos = nonNegativeInteger(tokenIds.bos, 'vocab.token_ids.bos');
    this.eos = nonNegativeInteger(tokenIds.eos, 'vocab.token_ids.eos');
    this.unk = nonNegativeInteger(tokenIds.unk, 'vocab.token_ids.unk');
    for (const [name, id, literal] of [
      ['pad', this.pad, '<pad>'], ['bos', this.bos, '<bos>'], ['eos', this.eos, '<eos>'], ['unk', this.unk, '<unk>'],
    ]) {
      if (id >= this.itos.length || this.itos[id] !== literal) {
        fail(`vocab.token_ids.${name} must identify ${literal} in vocab.json.`);
      }
    }
  }

  encodeQuestion(value, capacity) {
    if (!Number.isInteger(capacity) || capacity <= 0) fail('question capacity must be a positive integer.');
    const ids = [];
    for (const character of cleanText(value)) ids.push(this.stoi.get(character) ?? this.unk);
    ids.push(this.eos);
    if (ids.length > capacity) {
      ids.length = capacity;
      ids[capacity - 1] = this.eos;
    }
    return ids;
  }

  decode(tokenIds) {
    let text = '';
    for (const id of tokenIds) {
      if (id === this.eos) break;
      if (id === this.pad || id === this.bos) continue;
      if (Number.isInteger(id) && id >= 0 && id < this.itos.length) text += this.itos[id];
    }
    return text;
  }
}

function normalizeManifest(manifest, url) {
  if (!isRecord(manifest) || manifest.format !== PACKAGE_FORMAT) {
    fail(`package manifest must use format '${PACKAGE_FORMAT}'.`);
  }
  if (!isRecord(manifest.weights)) fail('package manifest requires a weights object.');
  const normalized = {
    url: url.toString(),
    weightsUrl: assetUrl(url, manifest.weights.file, 'weights.file'),
    preprocessing: null,
    router: null,
    families: new Map(),
    familyOrder: null,
    vocabUrl: null,
    tokenIds: null,
  };
  if (!isRecord(manifest.preprocessing) || !isRecord(manifest.preprocessing.resize) ||
      manifest.preprocessing.color_space !== 'grayscale' ||
      manifest.preprocessing.resize.resample !== 'bilinear' ||
      manifest.preprocessing.normalization !== 'minus-one-one' ||
      manifest.preprocessing.model_input_layout !== 'NHWC' ||
      manifest.preprocessing.model_input_dtype !== 'float32') {
    fail('package preprocessing must be grayscale + bilinear + minus-one-one F32 NHWC.');
  }
  const width = manifest.preprocessing.resize.width;
  const height = manifest.preprocessing.resize.height;
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    fail('package preprocessing resize width and height must be positive integers.');
  }
  normalized.preprocessing = Object.freeze({ width, height });

  if (!isRecord(manifest.router) || !isRecord(manifest.router.inputs)) fail('package manifest requires router.inputs.');
  normalized.router = Object.freeze({
    graphUrl: assetUrl(url, manifest.router.graph, 'router.graph'),
    outputName: nonEmptyString(manifest.router.output_name, 'router.output_name'),
    inputs: Object.freeze({
      qIds: inputName(manifest.router.inputs.q_ids, 'router.inputs.q_ids'),
      routerKeep: inputName(manifest.router.inputs.router_keep, 'router.inputs.router_keep'),
    }),
  });

  if (!isRecord(manifest.explicit_families)) fail('package manifest requires explicit_families.');
  const declaredOrder = manifest.family_order;
  if (!Array.isArray(declaredOrder) || declaredOrder.length !== FAMILY_ORDER.length ||
      declaredOrder.some((name, index) => name !== FAMILY_ORDER[index])) {
    fail(`package family_order must be ${JSON.stringify(FAMILY_ORDER)}.`);
  }
  normalized.familyOrder = FAMILY_ORDER;
  const keys = Object.keys(manifest.explicit_families);
  if (keys.length !== FAMILY_ORDER.length || keys.some((name) => !FAMILY_ORDER.includes(name))) {
    fail('package explicit_families must contain exactly the eight declared router families.');
  }
  for (const name of FAMILY_ORDER) {
    const definition = manifest.explicit_families[name];
    if (!isRecord(definition) || !isRecord(definition.interface) || !isRecord(definition.interface.inputs)) {
      fail(`explicit_families.${name} requires graph and interface.inputs.`);
    }
    const inputs = definition.interface.inputs;
    normalized.families.set(name, Object.freeze({
      graphUrl: assetUrl(url, definition.graph, `explicit_families.${name}.graph`),
      outputName: nonEmptyString(definition.interface.output_name, `explicit_families.${name}.interface.output_name`),
      inputs: Object.freeze({
        image: inputName(inputs.image, `explicit_families.${name}.interface.inputs.image`),
        qIds: inputName(inputs.q_ids, `explicit_families.${name}.interface.inputs.q_ids`),
        routerKeep: inputName(inputs.router_keep, `explicit_families.${name}.interface.inputs.router_keep`),
        memoryKeep: inputName(inputs.memory_keep, `explicit_families.${name}.interface.inputs.memory_keep`),
        yIds: inputName(inputs.y_ids, `explicit_families.${name}.interface.inputs.y_ids`),
        yKeep: inputName(inputs.y_keep, `explicit_families.${name}.interface.inputs.y_keep`),
      }),
    }));
  }
  if (!isRecord(manifest.vocab) || !isRecord(manifest.vocab.token_ids)) fail('package manifest requires vocab.file and vocab.token_ids.');
  normalized.vocabUrl = assetUrl(url, manifest.vocab.file, 'vocab.file');
  normalized.tokenIds = Object.freeze({
    pad: nonNegativeInteger(manifest.vocab.token_ids.pad, 'vocab.token_ids.pad'),
    bos: nonNegativeInteger(manifest.vocab.token_ids.bos, 'vocab.token_ids.bos'),
    eos: nonNegativeInteger(manifest.vocab.token_ids.eos, 'vocab.token_ids.eos'),
    unk: nonNegativeInteger(manifest.vocab.token_ids.unk, 'vocab.token_ids.unk'),
  });
  return Object.freeze(normalized);
}

async function fetchJson(fetchImpl, url, label) {
  if (typeof fetchImpl !== 'function') fail(`no fetch implementation is available to load ${label}.`);
  let response;
  try {
    response = await fetchImpl(url);
  } catch (error) {
    throw new Error(`[TinyReceiptW8A8Session] could not load ${label}: ${error.message || error}`);
  }
  if (!response || response.ok === false) {
    fail(`could not load ${label}: ${response?.statusText || response?.status || 'request failed'}.`);
  }
  try {
    return await response.json();
  } catch (error) {
    throw new Error(`[TinyReceiptW8A8Session] could not parse ${label}: ${error.message || error}`);
  }
}

/**
 * Host-side session for a materialized TinyReceiptVQA W8A8 package.
 *
 * ```js
 * const runtime = await VolvoxAI.createRuntime({ backends: ['webgpu', 'wasm', 'cpu'] });
 * const graphLoader = ({ weightsUrl, graphUrl, fetch, safetensorsCache }) =>
 *   GraphLoader.load(new Graph(), weightsUrl, { graphUrl, fetch, safetensorsCache });
 * const session = await TinyReceiptW8A8Session.load({ runtime,
 *   graphLoader,
 *   packageUrl: '/models/tiny-receipt/package_manifest.json' });
 * const result = await session.generate({ image: imageData, prompt: 'phone number?' });
 * ```
 *
 * `runtime` is a Runtime handle. A custom `graphLoader` and `fetch` can adapt
 * package I/O without importing training code or a Node-only image dependency.
 */
export class TinyReceiptW8A8Session {
  static async load(options = {}) {
    const runtime = options.runtime;
    if (!runtime || typeof runtime.createModel !== 'function') {
      fail('load requires a Runtime handle with createModel().');
    }
    const url = manifestUrl(options.packageUrl);
    const fetchImpl = options.fetch || globalThis.fetch?.bind(globalThis);
    const rawManifest = await fetchJson(fetchImpl, url.toString(), 'package_manifest.json');
    const packageInfo = normalizeManifest(rawManifest, url);
    const rawVocab = await fetchJson(fetchImpl, packageInfo.vocabUrl, 'vocab.json');
    const vocab = new TinyReceiptCharVocab(rawVocab?.itos, packageInfo.tokenIds);
    const safetensorsCache = options.safetensorsCache ?? null;
    if (safetensorsCache != null && !isReadOnlySafetensorsCache(safetensorsCache)) {
      fail('safetensorsCache must be a ReadOnlySafetensorsCache when provided.');
    }
    const graphLoader = options.graphLoader;
    if (typeof graphLoader !== 'function') fail('load requires a graphLoader function.');
    return new TinyReceiptW8A8Session({
      runtime,
      packageInfo,
      vocab,
      graphLoader,
      compileOptions: options.compileOptions,
      fetchImpl,
      safetensorsCache,
    });
  }

  constructor({ runtime, packageInfo, vocab, graphLoader, compileOptions, fetchImpl, safetensorsCache }) {
    this.runtime = runtime;
    this.package = packageInfo;
    this.vocab = vocab;
    this._fetch = fetchImpl;
    this._safetensorsCache = safetensorsCache;
    this._graphLoader = graphLoader;
    this._compileOptions = compileOptions || {};
    this._graphs = new Map();
    this._contexts = new Map();
    this._tail = Promise.resolve();
    this._closed = false;
    this._closePromise = null;
  }

  _exclusive(task) {
    const previous = this._tail;
    let release;
    this._tail = new Promise((resolve) => { release = resolve; });
    return previous.catch(() => undefined).then(task).finally(release);
  }

  async _loadGraph(key, graphUrl, kind, family = null) {
    if (this._closed) fail('session is closed.');
    let graph = this._graphs.get(key);
    if (!graph) {
      graph = await this._graphLoader({
        weightsUrl: this.package.weightsUrl,
        graphUrl,
        kind,
        family,
        fetch: this._fetch,
        safetensorsCache: this._safetensorsCache,
      });
      if (!graph || !graph.tensors || !Array.isArray(graph.nodes)) fail(`${kind} graph loader returned an invalid graph.`);
      this._graphs.set(key, graph);
    }
    return graph;
  }

  async _contextFor(key, graph, changedInputs = []) {
    const cached = this._contexts.get(key);
    if (cached?.graph === graph) return cached.context;
    const model = this.runtime.createModel(graph);
    let compiled;
    let context;
    try {
      compiled = await model.compile(this._compileOptions);
      context = await compiled.createContext({
        decode: { changedInputs, rowMode: 'auto' },
      });
    } catch (error) {
      await context?.close();
      await compiled?.close();
      await model.close();
      throw error;
    }
    this._contexts.set(key, { graph, model, compiled, context });
    return context;
  }

  async _executeOutput(context, outputName, inputs, options = undefined) {
    const executionResult = await context.execute(inputs, options);
    return this._outputFromExecution(outputName, executionResult);
  }

  async _outputFromExecution(outputName, executionResult,
    { elementOffset = null, elementCount = null } = {}) {
    try {
      let output = await executionResult.output(outputName).read();
      if (!(output instanceof Int32Array)) {
        fail(`graph output '${outputName}' must be read as raw Int32 QArgMax IDs.`);
      }
      if (elementOffset != null || elementCount != null) {
        if (!Number.isInteger(elementOffset) || elementOffset < 0 ||
            !Number.isInteger(elementCount) || elementCount <= 0 ||
            elementOffset + elementCount > output.length) {
          fail(`graph output '${outputName}' requested an invalid I32 element range.`);
        }
        output = output.slice(elementOffset, elementOffset + elementCount);
      }
      return output;
    } finally {
      await executionResult.close();
    }
  }

  async _routeImpl(prompt) {
    const router = this.package.router;
    const graph = await this._loadGraph('router', router.graphUrl, 'router');
    const abi = validateRouterGraph(graph, router);
    const encoded = this.vocab.encodeQuestion(prompt, abi.qLength);
    const qIds = new Int32Array(abi.qLength);
    qIds.fill(this.vocab.pad);
    qIds.set(encoded);
    const routerKeep = new Int32Array(abi.qLength);
    for (let index = 0; index < encoded.length; index++) routerKeep[index] = 1;
    const context = await this._contextFor('router', graph);
    const result = await this._executeOutput(context, router.outputName, {
      [router.inputs.qIds]: qIds,
      [router.inputs.routerKeep]: routerKeep,
    });
    if (result.length !== 1 || !Number.isInteger(result[0]) || result[0] < 0 || result[0] >= FAMILY_ORDER.length) {
      fail(`router output '${router.outputName}' must contain one valid I32 family ID.`);
    }
    const familyId = result[0];
    return Object.freeze({
      family: FAMILY_ORDER[familyId],
      familyId,
      qIds,
      routerKeep,
      questionTokenIds: Object.freeze([...encoded]),
    });
  }

  /** Run only the hard I32/QArgMax router for a question. */
  async route(prompt) {
    return this._exclusive(() => this._routeImpl(prompt));
  }

  async _preloadImpl({ families = [] } = {}) {
    const selected = families === 'all' ? [...FAMILY_ORDER] : families;
    if (!Array.isArray(selected) || selected.some((family) => !FAMILY_ORDER.includes(family))) {
      fail(`preload families must be 'all' or an array drawn from ${FAMILY_ORDER.join(', ')}.`);
    }
    const uniqueFamilies = [...new Set(selected)];
    const router = this.package.router;
    const routerGraph = await this._loadGraph('router', router.graphUrl, 'router');
    const routerAbi = validateRouterGraph(routerGraph, router);
    for (const family of uniqueFamilies) {
      const definition = this.package.families.get(family);
      const graph = await this._loadGraph(
        `family:${family}`, definition.graphUrl, 'explicit-family', family,
      );
      validateFamilyGraph(graph, definition, this.package.preprocessing, routerAbi.qLength);
    }
    return Object.freeze({
      router: true,
      families: Object.freeze(uniqueFamilies),
    });
  }

  /**
   * Fetch and assemble the router plus selected family graphs ahead of the
   * first request. Model compilation remains lazy until the graph is executed.
   */
  async preload(options = {}) {
    return this._exclusive(() => this._preloadImpl(options));
  }

  async _prepareImage(image, preprocessed) {
    const expected = this.package.preprocessing.width * this.package.preprocessing.height;
    let result = null;
    if (preprocessed === true || image instanceof Float32Array || image?.preprocessed === true) {
      result = image instanceof Float32Array ? image : image?.data;
      if (!(result instanceof Float32Array) || result.length !== expected) {
        fail(`preprocessed image must be F32 NHWC [1,${this.package.preprocessing.height},${this.package.preprocessing.width},1].`);
      }
      return result;
    }
    result = await preprocessTinyReceiptImage(image, this.package.preprocessing);
    if (!(result instanceof Float32Array) || result.length !== expected) fail('image preprocessing returned an invalid F32 NHWC tensor.');
    return result;
  }

  async _generateImpl({ image, prompt, family = 'auto', maxNewTokens = null,
    preprocessed = false, incremental = false } = {}) {
    const routed = await this._routeImpl(prompt);
    if (family !== 'auto' && !FAMILY_ORDER.includes(family)) {
      fail(`family must be 'auto' or one of ${FAMILY_ORDER.join(', ')}.`);
    }
    const selectedFamily = family === 'auto' ? routed.family : family;
    const definition = this.package.families.get(selectedFamily);
    const graph = await this._loadGraph(`family:${selectedFamily}`, definition.graphUrl, 'explicit-family', selectedFamily);
    const abi = validateFamilyGraph(graph, definition, this.package.preprocessing, routed.qIds.length);
    const imageInput = await this._prepareImage(image, preprocessed);
    const memoryKeep = new Int32Array(abi.memoryLength);
    memoryKeep.fill(1, 0, abi.memoryLength - abi.qLength);
    memoryKeep.set(routed.routerKeep, abi.memoryLength - abi.qLength);
    const yIds = new Int32Array(abi.decoderLength);
    yIds.fill(this.vocab.pad);
    yIds[0] = this.vocab.bos;
    const yKeep = new Int32Array(abi.decoderLength);
    yKeep[0] = 1;
    const limit = maxNewTokens == null ? abi.decoderLength : maxNewTokens;
    if (!Number.isInteger(limit) || limit < 0) fail('maxNewTokens must be a non-negative integer.');
    const context = await this._contextFor(
      `family:${selectedFamily}`,
      graph,
      [definition.inputs.yIds, definition.inputs.yKeep],
    );
    const incrementalRequested = incremental === true;
    const tokenIds = [];
    let stoppedAtEos = false;
    const tokenLimit = Math.min(limit, abi.decoderLength);
    const executionInputs = {
      [definition.inputs.image]: imageInput,
      [definition.inputs.qIds]: routed.qIds,
      [definition.inputs.routerKeep]: routed.routerKeep,
      [definition.inputs.memoryKeep]: memoryKeep,
      [definition.inputs.yIds]: yIds,
      [definition.inputs.yKeep]: yKeep,
    };
    // The package ABI remains fixed-size I32 plus terminal QArgMax. A decode
    // context owns all backend cache state; each result owns its output snapshot.
    if (incrementalRequested) await context.decode.reset();
    for (let step = 0; step < tokenLimit; step++) {
      const executionResult = incrementalRequested
        ? step === 0
          ? await context.decode.seed(executionInputs)
          : await context.decode.step(executionInputs, { position: step })
        : await context.execute(executionInputs);
      const output = await this._outputFromExecution(
        definition.outputName,
        executionResult,
      );
      if (output.length !== 1 && output.length !== abi.decoderLength) {
        fail(`family output '${definition.outputName}' length changed during execution.`);
      }
      const next = output.length === 1 ? output[0] : output[step];
      if (!Number.isInteger(next) || next < 0 || next >= this.vocab.itos.length) {
        fail(`terminal QArgMax emitted out-of-vocabulary token ${next}.`);
      }
      tokenIds.push(next);
      if (next === this.vocab.eos) {
        stoppedAtEos = true;
        break;
      }
      if (step + 1 < abi.decoderLength) {
        yIds[step + 1] = next;
        yKeep[step + 1] = 1;
      }
    }
    return Object.freeze({
      family: selectedFamily,
      familyId: FAMILY_ORDER.indexOf(selectedFamily),
      routerFamily: routed.family,
      routerFamilyId: routed.familyId,
      questionTokenIds: routed.questionTokenIds,
      tokenIds: Object.freeze([...tokenIds]),
      text: this.vocab.decode(tokenIds),
      stoppedAtEos,
      execution: incrementalRequested ? 'context-decode' : 'ordinary-forward',
    });
  }

  /**
   * Route once, select one explicit family graph, and autoregress using raw
   * terminal QArgMax token IDs. The session serializes calls because an
   * ExecutionContext owns mutable decode state.
   */
  async generate(options = {}) {
    return this._exclusive(() => this._generateImpl(options));
  }

  close() {
    if (this._closePromise) return this._closePromise;
    this._closePromise = this._exclusive(async () => {
      this._closed = true;
      const records = [...this._contexts.values()];
      for (const { context } of records) await context.close();
      for (const { compiled } of records) await compiled.close();
      for (const { model } of records) await model.close();
      this._contexts.clear();
      this._graphs.clear();
    });
    return this._closePromise;
  }

}

export { FAMILY_ORDER as TINY_RECEIPT_W8A8_FAMILY_ORDER, PACKAGE_FORMAT as TINY_RECEIPT_W8A8_PACKAGE_FORMAT };
