"""Fresh package publication shared by PTQ and training workflows."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

from ._model import ModelPackage


@contextmanager
def package_stage(output, weight_name="model.safetensors"):
    # Import publication tooling only when exporting, never during inference.
    from ._cli import _export_module
    exporter = _export_module()
    publication = exporter._exporter_module("publication")
    exporter_error = exporter._exporter_module("errors").ExporterError
    try:
        with publication.PackageStage(Path(output).expanduser() / weight_name) as stage:
            yield stage
    except exporter_error as error:
        raise ValueError(str(error)) from error


def write_checkpoint(path, payload):
    # Link a complete file into a fresh pathname. Never truncate a checkpoint
    # that another process or an earlier run has already published.
    path = Path(os.path.abspath(Path(path).expanduser()))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path
