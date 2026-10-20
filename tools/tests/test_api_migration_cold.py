import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location('migration_cold', Path(__file__).parents[1] / 'api_migration_cold.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Client:
    def create_runtime(self): return 'runtime'
    def load_model(self): return 'model'
    def compile_model(self): return 'compiled'
    def create_execution_context(self): return 'context'
    def release_execution_context(self): return None


class ColdCaptureTests(unittest.TestCase):
    def test_saved_cleanup_callbacks_are_observed_only_during_retirement(self):
        client = Client()
        with module.capture_cold(Client) as setup:
            session = SimpleNamespace(_engine=client,
                _resources=[(client.release_execution_context, None)])
        original = session._resources[0]
        self.assertIs(original[0].__func__, Client.release_execution_context)
        original[0]()
        self.assertEqual(setup['calls'], {})
        with module.capture_cold(Client, retirement=True, sessions=[session]) as retirement:
            session._resources[0][0]()
        self.assertIs(session._resources[0], original)
        self.assertEqual(retirement['calls']['release_execution_context']['count'], 1)
        original[0]()
        self.assertEqual(retirement['calls']['release_execution_context']['count'], 1)
        self.assertEqual(setup['calls'], {})

    def test_observers_end_before_hot_calls_and_restore_on_failure(self):
        original_method, original_read = Client.load_model, Path.read_bytes
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'model.safetensors'
            weights.write_bytes(b'weights')
            with module.capture_cold(Client) as cold:
                self.assertEqual(weights.read_bytes(), b'weights')
                self.assertEqual(client.load_model(), 'model')
                client.compile_model()
            self.assertIs(Client.load_model, original_method)
            self.assertIs(Path.read_bytes, original_read)
            self.assertEqual(cold['asset_bytes'], 7)
            self.assertEqual(cold['calls']['load_model']['count'], 1)
            client.load_model()
            self.assertEqual(cold['calls']['load_model']['count'], 1)
            with self.assertRaisesRegex(RuntimeError, 'construction failed'):
                with module.capture_cold(Client):
                    raise RuntimeError('construction failed')
            self.assertIs(Client.load_model, original_method)
            self.assertIs(Path.read_bytes, original_read)


if __name__ == '__main__':
    unittest.main()
