const SOFTWARE_ADAPTER = /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software rasterizer|microsoft basic render|cpu)\b/i;

function configuredAdapterRequirement() {
  try {
    const value = globalThis.Deno?.env?.get?.('VOLVOXAI_PARITY_GPU_ADAPTER') ??
      globalThis.process?.env?.VOLVOXAI_PARITY_GPU_ADAPTER ??
      globalThis.Deno?.env?.get?.('VOLVOXAI_PARITY_WEBGPU_ADAPTER') ??
      globalThis.process?.env?.VOLVOXAI_PARITY_WEBGPU_ADAPTER;
    return typeof value === 'string' ? value.trim() : '';
  } catch {
    return '';
  }
}

export function requirePhysicalAdapterIdentity(info, label = 'WebGPU parity', {
  requiredIdentity = configuredAdapterRequirement(),
} = {}) {
  if (info == null || typeof info !== 'object' || Array.isArray(info)) {
    throw new Error(`${label}: adapter identity is unavailable; refusing to label the run physical-GPU`);
  }
  const identity = Object.fromEntries(Object.entries(info)
    .filter(([, value]) => typeof value === 'string' && value.trim())
    .map(([key, value]) => [key, value.trim()]));
  const description = Object.values(identity).join(' ');
  if (!description) {
    throw new Error(`${label}: adapter identity is empty; refusing to label the run physical-GPU`);
  }
  if (SOFTWARE_ADAPTER.test(description)) {
    throw new Error(`${label}: software adapter rejected (${description})`);
  }
  if (requiredIdentity && !description.toLowerCase().includes(requiredIdentity.toLowerCase())) {
    throw new Error(`${label}: adapter '${description}' does not match required identity '${requiredIdentity}'`);
  }
  return Object.freeze(identity);
}

export function requirePhysicalWebGPU(compiled, label = 'WebGPU parity', options) {
  if (compiled?.backend !== 'webgpu') {
    throw new Error(`${label}: requested WebGPU compiled on '${compiled?.backend || 'unknown'}'`);
  }
  return requirePhysicalAdapterIdentity(compiled.report?.selectedDevice, label, options);
}

export function requirePhysicalNativeAdapterIdentity(
  info,
  expectedBackend,
  label = `native ${expectedBackend} parity`,
  options,
) {
  if (!['vulkan', 'opengl'].includes(expectedBackend)) {
    throw new Error(`${label}: unsupported native physical-GPU backend '${expectedBackend}'`);
  }
  if (info == null || typeof info !== 'object' || Array.isArray(info)) {
    throw new Error(`${label}: native adapter identity is unavailable`);
  }
  if (info?.backend !== expectedBackend) {
    throw new Error(`${label}: native adapter backend is '${info?.backend || 'missing'}', expected '${expectedBackend}'`);
  }
  if (typeof info.device !== 'string' || !info.device.trim() || /^unknown$/i.test(info.device.trim())) {
    throw new Error(`${label}: native device identity is unavailable`);
  }
  return requirePhysicalAdapterIdentity(info, label, options);
}

export function requirePhysicalNativeGpuLog(
  output,
  expectedBackend,
  label = `native ${expectedBackend} parity`,
  options,
) {
  if (!['vulkan', 'opengl'].includes(expectedBackend)) {
    throw new Error(`${label}: unsupported native physical-GPU backend '${expectedBackend}'`);
  }
  const text = String(output);
  const routes = [...text.matchAll(/^Backend: ([^\r\n]+)\r?$/gm)].map((match) => match[1]);
  if (routes.length !== 1 || routes[0] !== expectedBackend) {
    throw new Error(`${label}: native CLI reported ${JSON.stringify(routes)}, expected ${expectedBackend}`);
  }

  let matches;
  let identity;
  if (expectedBackend === 'vulkan') {
    matches = [...text.matchAll(
      /^\[VolvoxAI GPU\] Vulkan Compute initialized successfully! Device: ([^;\r\n]+); packed INT8 dot: (enabled|unavailable)\r?$/gm,
    )];
    if (matches.length === 1) {
      identity = {
        backend: 'vulkan',
        device: matches[0][1],
        packedInt8Dot: matches[0][2],
      };
    }
  } else {
    matches = [...text.matchAll(
      /^\[VolvoxAI GPU\] OpenGL Compute initialized: (.*?) \/ (.*?) \/ ([^\r\n]+)\r?$/gm,
    )];
    if (matches.length === 1) {
      identity = {
        backend: 'opengl',
        vendor: matches[0][1],
        device: matches[0][2],
        version: matches[0][3],
      };
    }
  }
  if (matches.length !== 1) {
    const displayName = expectedBackend === 'vulkan' ? 'Vulkan' : 'OpenGL';
    throw new Error(`${label}: expected exactly one ${displayName} device initialization, found ${matches.length}`);
  }
  return requirePhysicalNativeAdapterIdentity(identity, expectedBackend, label, options);
}

export function formatAdapterIdentity(info) {
  return Object.entries(info).map(([key, value]) => `${key}=${value}`).join(', ');
}
