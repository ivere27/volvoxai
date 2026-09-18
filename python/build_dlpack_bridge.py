"""Build the capsule adapter for source-tree tests; wheels build it in Docker."""
from pathlib import Path

from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


def main():
    root = Path(__file__).resolve().parents[1]
    distribution = Distribution({"ext_modules": [Extension(
        "volvoxai._dlpack", [str(root / "python/volvoxai/_dlpack.c")],
        include_dirs=[str(root / "python/third_party/dlpack")],
        py_limited_api=True)]})
    command = build_ext(distribution)
    command.build_lib = str(root / "python")
    command.build_temp = str(root / "build/python-dlpack-bridge")
    command.ensure_finalized()
    command.run()


if __name__ == "__main__":
    main()
