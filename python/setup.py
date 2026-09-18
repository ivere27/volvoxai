"""Wheel metadata for the source tree staged by build_wheel.py."""

from pathlib import Path

from setuptools import Distribution, Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        # The engine uses ctypes; DLPack capsules use a small stable-ABI bridge.
        return True


class NativeWheel(bdist_wheel):
    def run(self):
        for name in (
            "volvoxai/libvolvoxai.so",
            "volvoxai/libsynurang_module_host.so",
            "volvoxai/bin/volvoxai",
            "volvoxai/exporter/__init__.py",
            "volvoxai_client.py",
            "volvoxai_lite.py",
            "synurang/module.py",
        ):
            if not Path(name).is_file():
                raise RuntimeError(
                    f"Missing staged wheel input {name}; use make build_wheel."
                )
        super().run()

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "cp310", "abi3", platform


setup(distclass=NativeDistribution, cmdclass={"bdist_wheel": NativeWheel},
      ext_modules=[Extension("volvoxai._dlpack", ["volvoxai/_dlpack.c"],
                            include_dirs=["third_party/dlpack"], py_limited_api=True)])
