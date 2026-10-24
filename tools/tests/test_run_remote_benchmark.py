"""Exercise the SSH controller locally without contacting a remote host."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_remote_benchmark as runner


class RemoteBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.remote_home = self.base / 'remote-home'
        self.remote_home.mkdir()
        self.spec_dir = self.base / 'spec'
        self.spec_dir.mkdir()
        self.payload = self.spec_dir / 'payload'
        self.payload.mkdir()
        (self.payload / 'run.py').write_text(
            "from pathlib import Path\n"
            "assert Path('models/local.bin').read_bytes() == b'local weights'\n"
            "assert Path('models/shared.bin').read_bytes() == b'shared weights'\n"
            "Path('results/value.json').write_text('{\"value\": 42}')\n"
        )
        self.local_weight = self.spec_dir / 'weights.bin'
        self.local_weight.write_bytes(b'local weights')
        self.shared_weight = self.remote_home / 'shared.bin'
        self.shared_weight.write_bytes(b'shared weights')
        self.spec = {
            'payload_dir': 'payload',
            'minimum_free_bytes': 0,
            'weights': [
                {'destination': 'models/local.bin', 'local': 'weights.bin',
                 'sha256': runner.sha256(self.local_weight)},
                {'destination': 'models/shared.bin', 'shared': str(self.shared_weight),
                 'sha256': runner.sha256(self.shared_weight)},
            ],
            'steps': [{'name': 'test', 'argv': [sys.executable, 'run.py'], 'timeout_seconds': 5}],
        }
        self.spec_path = self.spec_dir / 'spec.json'
        self.out = self.base / 'output'
        self.calls = []
        self.real_run = subprocess.run

    def fake_ssh(self, command, **kwargs):
        self.calls.append(command)
        if command[0] == 'ssh':
            # Execute the actual generated remote Python, but in a local temp home.
            env = dict(os.environ, HOME=str(self.remote_home))
            if command[-1] == 'python3 -':
                return self.real_run([sys.executable, '-'], env=env, **kwargs)
            return self.real_run(['bash', '-c', command[-1]], env=env, **kwargs)
        if command[0] not in {'rsync', 'scp'}:
            raise AssertionError(command)
        source, destination = command[-2:]
        source = source.removeprefix('test-host:')
        destination = destination.removeprefix('test-host:')
        if Path(source).is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, destination)
        return subprocess.CompletedProcess(command, 0)

    def run_controller(self, *extra):
        self.spec_path.write_text(json.dumps(self.spec))
        argv = ['--spec', str(self.spec_path), '--out-dir', str(self.out),
                '--host', 'test-host', *extra]
        with patch.object(runner.subprocess, 'run', side_effect=self.fake_ssh), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.main(argv)

    def receipt(self):
        return json.loads((self.out / 'remote-receipt.json').read_text())

    def assert_cleaned(self):
        receipt = self.receipt()
        self.assertFalse(Path(receipt['remote_dir']).exists())
        self.assertFalse((self.out / 'payload.tar.gz').exists())
        self.assertEqual(self.shared_weight.read_bytes(), b'shared weights')
        return receipt

    def test_default_remote_home_and_spec_relative_paths(self):
        self.run_controller()
        receipt = self.assert_cleaned()
        self.assertEqual(receipt['status'], 'COMPLETED')
        self.assertEqual(receipt['remote_root'], str(self.remote_home))
        self.assertEqual([weight['shared'] for weight in receipt['weights']], [False, True])
        self.assertEqual(json.loads((self.out / 'results/value.json').read_text()), {'value': 42})

    def test_explicit_remote_root_is_resolved_before_ownership_checks(self):
        alias = self.base / 'remote-alias'
        alias.symlink_to(self.remote_home, target_is_directory=True)
        self.run_controller('--remote-root', str(alias))
        receipt = self.assert_cleaned()
        self.assertEqual(receipt['remote_root'], str(self.remote_home.resolve()))

    def test_failed_setup_does_not_transfer_or_recover_absent_directory(self):
        self.spec['minimum_free_bytes'] = 10**30
        with self.assertRaisesRegex(RuntimeError, 'insufficient remote free space'):
            self.run_controller()
        receipt = self.assert_cleaned()
        self.assertEqual(receipt['status'], 'FAILED')
        self.assertEqual(receipt['cleanup'], {'absent': True})
        self.assertFalse(any(command[0] in {'rsync', 'scp'} for command in self.calls))

    def test_failed_step_recovers_results_and_cleans_owned_directory(self):
        with (self.payload / 'run.py').open('a') as stream:
            stream.write('raise SystemExit(7)\n')
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_controller()
        receipt = self.assert_cleaned()
        self.assertEqual(receipt['status'], 'FAILED')
        report = json.loads((self.out / 'results/supervisor.json').read_text())
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['steps'][0]['exit_code'], 7)
        self.assertTrue((self.out / 'results/value.json').is_file())

    def test_existing_unowned_directory_is_not_copied_or_removed(self):
        token = 'a' * 32
        unowned = self.remote_home / ('volvox-model-benchmark.' + token)
        unowned.mkdir()
        (unowned / '.owner').write_text('someone else')
        (unowned / 'keep').write_text('preserve')
        with patch.object(runner.uuid, 'uuid4', return_value=SimpleNamespace(hex=token)):
            with self.assertRaisesRegex(RuntimeError, 'ownership mismatch'):
                self.run_controller()
        self.assertEqual((unowned / 'keep').read_text(), 'preserve')
        self.assertEqual(self.receipt()['status'], 'CLEANUP_FAILED')
        self.assertFalse(any(command[0] in {'rsync', 'scp'} for command in self.calls))

    def test_explicit_host_and_absolute_remote_root_are_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main(['--spec', 'unused', '--out-dir', 'unused'])
            with self.assertRaises(SystemExit):
                runner.main(['--spec', 'unused', '--out-dir', 'unused',
                             '--host', 'test-host', '--remote-root', 'relative'])


if __name__ == '__main__':
    unittest.main()
