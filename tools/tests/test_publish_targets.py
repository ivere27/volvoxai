"""Exercise the real Make publish recipes with local tool stubs, without uploads."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
STUB = r'''
import json
import os
from pathlib import Path
import subprocess
import sys

name, args = Path(sys.argv[0]).name, sys.argv[1:]

def event(kind):
    with open(os.environ['PUBLISH_TEST_LOG'], 'a') as stream:
        stream.write(json.dumps({'event': kind, 'args': args}) + '\n')
    if os.environ.get('PUBLISH_TEST_FAIL') == kind:
        raise SystemExit(7)

if name == 'node':
    if args != ['tools/check_release_artifacts.mjs', '--web-only']:
        raise SystemExit('unexpected node command')
    event('release-check')
elif name == 'npm':
    kinds = {
        ('pack', '--dry-run'): 'npm-pack',
        ('whoami',): 'npm-auth',
        ('publish', '--access', 'public'): 'npm-upload',
        ('install',): 'npm-install',
        ('run', 'build:all'): 'web-build',
        ('run', 'fixture:wasm'): 'wasm-build',
    }
    event(kinds[tuple(args)])
elif name == 'python':
    if args[:2] == ['-m', 'twine']:
        event('twine-' + args[2])
    elif args[0] == 'python/build_wheel.py':
        event('wheel-build')
    else:
        raise SystemExit('unexpected python command')
elif name == 'docker':
    if args[:2] == ['image', 'inspect']:
        print('fixture-image')
    elif args[0] == 'build':
        event('image-build')
    elif args[0] == 'run':
        mounts, workdir, i = {}, None, 1
        while args[i].startswith('-'):
            option = args[i]
            if option in ['-v', '-w', '-e', '--platform', '--user']:
                value = args[i + 1]
                if option == '-v':
                    host, guest, *_ = value.split(':')
                    mounts[guest] = host
                elif option == '-w':
                    workdir = value
                i += 2
            elif option in ['--rm', '-i', '-t'] or option.startswith('--cpus='):
                i += 1
            else:
                raise SystemExit('unexpected docker option: ' + option)
        event('container')
        command = args[i + 1:]
        if command[0] == 'chown':
            raise SystemExit(0)
        if command[0] not in ['sh', 'python', 'npm', 'make']:
            raise SystemExit('unexpected container command')
        raise SystemExit(subprocess.call(command, cwd=mounts.get(workdir)))
    else:
        raise SystemExit('unexpected docker command')
else:
    raise SystemExit('unexpected stub executable')
'''


class PublishTargetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='volvoxai-publish-test-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        shutil.copy2(ROOT / 'Makefile', self.directory / 'Makefile')
        package = json.loads((ROOT / 'package.json').read_text())
        (self.directory / 'package.json').write_text(json.dumps(package))
        self.wheels = self.directory / 'dist' / 'python' / package['version']
        self.wheels.mkdir(parents=True)
        wheel = self.wheels / 'volvoxai-fixture.whl'
        wheel.write_bytes(b'fixture wheel')
        (self.wheels / 'SHA256SUMS').write_text(
            f'{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n'
        )
        binaries = self.directory / 'bin'
        binaries.mkdir()
        for name in ['node', 'npm', 'python', 'docker']:
            executable = binaries / name
            executable.write_text(f'#!{sys.executable}\n' + STUB)
            executable.chmod(0o755)
        self.log = self.directory / 'calls.jsonl'
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith('TWINE_') and key not in ['DRY_RUN', 'MAKEFLAGS', 'MFLAGS']
        }
        self.environment.update(
            PATH=f'{binaries}{os.pathsep}{os.environ["PATH"]}',
            PUBLISH_TEST_LOG=str(self.log),
        )

    def run_make(self, target, dry_run=None, fail=None):
        self.log.unlink(missing_ok=True)
        args = ['make', '--no-print-directory', target]
        if dry_run is not None:
            args.append(f'DRY_RUN={dry_run}')
        environment = dict(self.environment)
        if fail:
            environment['PUBLISH_TEST_FAIL'] = fail
        result = subprocess.run(
            args, cwd=self.directory, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20,
        )
        self.calls = [json.loads(line) for line in self.log.read_text().splitlines()] \
            if self.log.exists() else []
        self.events = [call['event'] for call in self.calls]
        return result

    def test_npm_dry_run_checks_package_without_auth_or_upload(self):
        result = self.run_make('publish_npm', '1')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events, ['release-check', 'npm-pack'])

    def test_npm_default_upload_follows_validation(self):
        result = self.run_make('publish_npm')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('npm-auth', self.events)
        self.assertLess(self.events.index('release-check'), self.events.index('npm-upload'))
        self.assertLess(self.events.index('npm-pack'), self.events.index('npm-upload'))

    def test_npm_validation_failure_prevents_upload(self):
        for failure in ['release-check', 'npm-pack']:
            with self.subTest(failure=failure):
                result = self.run_make('publish_npm', fail=failure)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn('npm-upload', self.events)

    def test_pypi_dry_run_checks_existing_wheel_without_build_or_upload(self):
        result = self.run_make('publish_pypi', '1')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('twine-check', self.events)
        self.assertNotIn('wheel-build', self.events)
        self.assertNotIn('image-build', self.events)
        self.assertNotIn('twine-upload', self.events)
        self.assertFalse(any(
            arg.startswith('TWINE_')
            for call in self.calls if call['event'] == 'container'
            for arg in call['args']
        ))

    def test_pypi_default_builds_and_checks_before_upload(self):
        result = self.run_make('publish_pypi')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertLess(self.events.index('wheel-build'), self.events.index('twine-check'))
        self.assertLess(self.events.index('twine-check'), self.events.index('twine-upload'))

    def test_pypi_validation_failure_prevents_upload(self):
        result = self.run_make('publish_pypi', fail='twine-check')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn('twine-upload', self.events)

    def test_pypi_checksum_failure_prevents_metadata_check_and_upload(self):
        (self.wheels / 'volvoxai-fixture.whl').write_bytes(b'changed wheel')
        result = self.run_make('publish_pypi', '1')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn('twine-check', self.events)
        self.assertNotIn('twine-upload', self.events)

    def test_build_wheel_finishes_with_validation_without_upload(self):
        result = self.run_make('build_wheel')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events.count('wheel-build'), 1)
        self.assertLess(self.events.index('wheel-build'), self.events.index('twine-check'))
        self.assertNotIn('twine-upload', self.events)

    def test_build_web_finishes_with_package_validation_and_propagates_failure(self):
        # Numerical compilation is outside this packaging test. Keep the actual
        # build_web recipe and recursive make, replacing only its WASM work.
        with (self.directory / 'Makefile').open('a') as stream:
            stream.write(
                '\nbuild_wasm: ; @npm run fixture:wasm\n'
                'wasm_abi_codegen_check shader_catalog_codegen_check: ;\n'
            )
        for failure in [None, 'npm-pack']:
            with self.subTest(failure=failure):
                result = self.run_make('build_web', fail=failure)
                if failure:
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout)
                self.assertLess(self.events.index('web-build'), self.events.index('wasm-build'))
                self.assertLess(self.events.index('wasm-build'), self.events.index('npm-pack'))
                self.assertNotIn('npm-auth', self.events)
                self.assertNotIn('npm-upload', self.events)

    def test_invalid_dry_run_is_rejected(self):
        for target in ['publish_npm', 'publish_pypi']:
            with self.subTest(target=target):
                result = self.run_make(target, 'invalid')
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn('npm-upload', self.events)
                self.assertNotIn('twine-upload', self.events)


if __name__ == '__main__':
    unittest.main()
