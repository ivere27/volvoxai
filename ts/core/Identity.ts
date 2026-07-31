function randomSuffix(): string {
  const crypto = globalThis.crypto;
  if (typeof crypto?.randomUUID === 'function') return crypto.randomUUID();
  if (typeof crypto?.getRandomValues === 'function') {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
  }
  // Supported browser and Node runtimes provide Web Crypto. This fallback
  // keeps unusual embedders usable without introducing a process-owned counter.
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/** Create an opaque report identity without process-global mutable state. */
export function runtimeIdentity(prefix: string): string {
  if (typeof prefix !== 'string' || !/^[a-z][a-z0-9-]*$/.test(prefix)) {
    throw new Error('Runtime identity prefixes must be canonical lowercase names.');
  }
  return `${prefix}-${randomSuffix()}`;
}
