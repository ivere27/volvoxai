import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('package_release', ROOT / 'tools/package_release.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class DeterministicPackageTests(unittest.TestCase):
    def test_profile_inventory_integrity_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('package.json', 'proto/volvoxai.proto', 'proto/kernel_registry.proto',
                         'tools/generated/wasmInternalAbi.mjs', 'ts/generated/gpuBridge.ts',
                         'ts/generated/shaderCatalog.ts'):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            version = json.loads((root / 'package.json').read_text())['version']
            for profile, stem, native in [('inference', 'volvoxai.lite', 'volvoxai-lite'),
                                          ('full', 'volvoxai', 'volvoxai')]:
                names = {f'dist/{version}/{stem}{suffix}' for suffix in ('.js', '.min.js', '.wasm')}
                names.add('native/' + native)
                for name in names:
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(('fixture ' + name).encode())
                files, manifest = package.package_files(root, profile)
                self.assertEqual(set(manifest['files']), names)
                self.assertEqual(manifest['profile'], profile)
                for key in ('gpu_bridge_abi_sha256', 'shader_catalogue_sha256'):
                    self.assertEqual(key in manifest['contract'], profile == 'full')
                first, second = root / 'first.zip', root / 'second.zip'
                package.write_archive(first, files)
                package.write_archive(second, files)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                with zipfile.ZipFile(first) as archive:
                    self.assertEqual(set(archive.namelist()), names | {'manifest.json'})
                    for name, info in manifest['files'].items():
                        payload = archive.read(name)
                        self.assertEqual(len(payload), info['bytes'])
                        self.assertEqual(hashlib.sha256(payload).hexdigest(), info['sha256'])
            with self.assertRaisesRegex(ValueError, 'inference or full'):
                package.package_files(root, 'model')


if __name__ == '__main__':
    unittest.main()
