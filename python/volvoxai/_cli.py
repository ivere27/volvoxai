"""Command-line workflows composed from the exporter and generated API."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import __version__


def _export_module():
    if (Path(__file__).parent / "_export.py").is_file():
        return importlib.import_module("volvoxai._export")
    # Source checkout: use the same exporter without copying generated files.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("tools.export_safetensors")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="volvoxai",
        description=(
            "Model conversion and PTQ workflows for VolvoxAI. Import volvoxai "
            "for the complete native inference, training, text and scheduling API."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=("export", "ptq"))
    if arguments and arguments[0] == "export":
        module = _export_module()
        return module._exporter_module("cli").main(
            module.export_model, arguments[1:]
        )
    if arguments and arguments[0] == "ptq":
        from ._ptq_cli import main as ptq_main
        return ptq_main(arguments[1:])
    parser.parse_args(arguments)
    return 0
