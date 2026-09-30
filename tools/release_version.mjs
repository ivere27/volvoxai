import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function validatePackageVersion(version) {
  const match = typeof version === 'string' &&
    /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?$/.exec(version);
  const invalidIdentifiers = (value, prerelease) => value?.split('.').some(identifier =>
    !/^[0-9A-Za-z-]+$/.test(identifier) ||
    (prerelease && /^\d+$/.test(identifier) && /^0\d/.test(identifier)));
  if (!match || invalidIdentifiers(match[4], true) || invalidIdentifiers(match[5], false)) {
    throw new Error(`Invalid package release version: ${JSON.stringify(version)}.`);
  }
  return version;
}

/** package.json is the single release-version authority for repository tools. */
export function readPackageVersion(root = repositoryRoot) {
  const pkg = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
  return validatePackageVersion(pkg.version);
}
