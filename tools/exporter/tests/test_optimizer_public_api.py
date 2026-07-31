from __future__ import annotations

import unittest

from tools.exporter import optimizer
from tools.exporter.optimizer import compiled_model, profiling, workspace


class OptimizerPublicApiTests(unittest.TestCase):
    def test_stable_deployment_modules_are_published_at_package_root(self):
        for module in (compiled_model, profiling, workspace):
            for name in module.__all__:
                with self.subTest(module=module.__name__, name=name):
                    self.assertIn(name, optimizer.__all__)
                    self.assertIs(getattr(optimizer, name), getattr(module, name))


if __name__ == "__main__":
    unittest.main()
