import subprocess
import sys
import unittest


class CompletePipelineTests(unittest.TestCase):
    def test_synthesis_import_does_not_load_oracle(self):
        # A fresh process prevents another numerical test's ORT session from
        # hiding an accidental import in the production synthesis modules.
        source = """
import importlib.abc
import sys
class RejectOracle(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'onnxruntime' or fullname.startswith('onnxruntime.'):
            raise AssertionError('synthesis imported the comparison oracle')
sys.meta_path.insert(0, RejectOracle())
from examples.kokoro.complete import run
from examples.kokoro.synth import main
from examples.tts_common.benchmark_native import NativePipeline
assert 'onnxruntime' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
