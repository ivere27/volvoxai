// Examples always load the JavaScript and WASM from the same package release.
const packageUrl = new URL('../../package.json', import.meta.url);
let packageDocument;
if (packageUrl.protocol === 'file:') {
  const { readFile } = await import('node:fs/promises');
  packageDocument = JSON.parse(await readFile(packageUrl, 'utf8'));
} else {
  const response = await fetch(packageUrl);
  if (!response.ok) throw new Error(`Cannot load package version: ${response.status}`);
  packageDocument = await response.json();
}
if (typeof packageDocument.version !== 'string' || !packageDocument.version) {
  throw new Error('package.json must declare a release version');
}

export function releaseUrls(profile = 'full') {
  if (profile !== 'full' && profile !== 'inference') throw new Error(`Unknown release profile: ${profile}`);
  const stem = profile === 'inference' ? 'volvoxai.lite' : 'volvoxai';
  const directory = new URL(`../../dist/${packageDocument.version}/`, import.meta.url);
  return { jsUrl: new URL(`${stem}.js`, directory), wasmUrl: new URL(`${stem}.wasm`, directory) };
}

export async function loadRuntimeRelease(profile = 'full') {
  const { jsUrl, wasmUrl } = releaseUrls(profile);
  return { api: await import(jsUrl.href), wasmUrl: wasmUrl.href };
}
