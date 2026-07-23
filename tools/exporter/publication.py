"""Staged package publication with rollback of pre-existing outputs."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

from .errors import Diagnostic, ExporterError


class PackageStage:
    """Build a package in a sibling directory and publish validated files."""

    def __init__(self, output: Path):
        self.output = Path(output).resolve()
        self.destination = self.output.parent
        self.destination.mkdir(parents=True, exist_ok=True)
        self._temporary = Path(tempfile.mkdtemp(
            prefix=f".{self.output.stem}.export-",
            dir=str(self.destination),
        ))
        self.staged_output = self._temporary / self.output.name

    @property
    def directory(self) -> Path:
        return self._temporary

    def cleanup(self) -> None:
        shutil.rmtree(self._temporary, ignore_errors=True)

    def publish(self, filenames: Iterable[str] = ("graph.json",)) -> None:
        names = [self.output.name, *filenames]
        missing = [name for name in names if not (self._temporary / name).is_file()]
        if missing:
            raise ExporterError(Diagnostic(
                code="VXPUB001",
                message=f"staged package is missing: {', '.join(missing)}",
                stage="publication",
                constraint="complete staged package",
            ))

        backup_dir = self._temporary / ".rollback"
        backup_dir.mkdir()
        replaced: list[str] = []
        try:
            for name in names:
                destination = self.destination / name
                if destination.exists():
                    os.replace(destination, backup_dir / name)
                os.replace(self._temporary / name, destination)
                replaced.append(name)
        except Exception as error:
            for name in reversed(replaced):
                destination = self.destination / name
                if destination.exists():
                    os.replace(destination, self._temporary / name)
            for name in names:
                backup = backup_dir / name
                if backup.exists():
                    os.replace(backup, self.destination / name)
            raise ExporterError(Diagnostic(
                code="VXPUB002",
                message=f"failed to publish staged package: {error}",
                stage="publication",
            )) from error
        finally:
            self.cleanup()

    def __enter__(self) -> "PackageStage":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()


def _absolute(path: Path) -> Path:
    """Normalize ``path`` without following a destination symlink."""

    return Path(os.path.abspath(os.fspath(path)))


class ArtifactTransaction:
    """Stage and rollback a small set of package files.

    All staged files live beside the sentinel destination and every target
    parent must be on that same filesystem. Existing destinations are copied
    into the private staging directory before the first replacement. Non-
    sentinel artifacts publish first and the graph sentinel publishes last.
    """

    def __init__(
        self,
        destinations: Mapping[str, Path],
        *,
        sentinel: str,
    ) -> None:
        if not destinations or sentinel not in destinations:
            raise ValueError("artifact transaction requires its sentinel destination")
        if any(not isinstance(name, str) or not name for name in destinations):
            raise ValueError("artifact transaction names must be non-empty strings")

        self.destinations = {
            name: _absolute(Path(path)) for name, path in destinations.items()
        }
        resolved = [path.resolve(strict=False) for path in self.destinations.values()]
        if len(set(resolved)) != len(resolved):
            raise ExporterError(Diagnostic(
                code="VXPUB003",
                message="package artifact destinations must be distinct",
                stage="publication",
                constraint="distinct package artifact paths",
            ))
        self.sentinel = sentinel
        self._temporary: Path | None = None
        self._staged: dict[str, Path] = {}

    @property
    def directory(self) -> Path:
        if self._temporary is None:
            raise RuntimeError("artifact transaction has not been entered")
        return self._temporary

    def staged_path(self, name: str) -> Path:
        try:
            return self._staged[name]
        except KeyError as error:
            raise KeyError(f"unknown staged artifact {name!r}") from error

    def _prepare(self) -> None:
        parents = {path.parent for path in self.destinations.values()}
        for path in self.destinations.values():
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ExporterError(Diagnostic(
                    code="VXPUB003",
                    message=f"package artifact destination is not a regular file: {path}",
                    stage="publication",
                    constraint="regular package artifact destinations",
                ))
        for parent in parents:
            parent.mkdir(parents=True, exist_ok=True)
        devices = {parent.stat().st_dev for parent in parents}
        if len(devices) != 1:
            raise ExporterError(Diagnostic(
                code="VXPUB003",
                message="package artifacts must publish on one destination filesystem",
                stage="publication",
                constraint="same-filesystem staged publication",
            ))

        sentinel_path = self.destinations[self.sentinel]
        self._temporary = Path(tempfile.mkdtemp(
            prefix=f".{sentinel_path.stem}.publish-",
            dir=str(sentinel_path.parent),
        ))
        if self._temporary.stat().st_dev not in devices:
            self.cleanup()
            raise ExporterError(Diagnostic(
                code="VXPUB003",
                message="staging directory is not on the destination filesystem",
                stage="publication",
                constraint="same-filesystem staged publication",
            ))
        self._staged = {
            name: self._temporary / f"{index:02d}-{destination.name}"
            for index, (name, destination) in enumerate(self.destinations.items())
        }

    def require_complete(self) -> None:
        missing = [
            name for name, path in self._staged.items()
            if path.is_symlink() or not path.is_file()
        ]
        if missing:
            raise ExporterError(Diagnostic(
                code="VXPUB001",
                message=f"staged package is missing: {', '.join(missing)}",
                stage="publication",
                constraint="complete staged package",
            ))

    def _restore(
        self,
        order: list[str],
        backups: Mapping[str, Path],
        existed: Mapping[str, bool],
    ) -> list[str]:
        errors: list[str] = []
        for name in order:
            destination = self.destinations[name]
            try:
                if existed[name]:
                    os.replace(backups[name], destination)
                elif destination.exists() or destination.is_symlink():
                    destination.unlink()
            except Exception as error:  # pragma: no cover - catastrophic I/O
                errors.append(f"{name}: {error}")
        return errors

    def publish(self) -> None:
        self.require_complete()
        order = [name for name in self.destinations if name != self.sentinel]
        order.append(self.sentinel)
        rollback = self.directory / ".rollback"
        rollback.mkdir()
        existed = {
            name: destination.exists()
            for name, destination in self.destinations.items()
        }
        backups = {name: rollback / f"{index:02d}.backup"
                   for index, name in enumerate(self.destinations)}

        try:
            for name, destination in self.destinations.items():
                if existed[name]:
                    shutil.copyfile(destination, backups[name])
        except Exception as error:
            self.cleanup()
            raise ExporterError(Diagnostic(
                code="VXPUB002",
                message=f"failed to snapshot existing package: {error}",
                stage="publication",
                constraint="rollback snapshot before publication",
            )) from error

        try:
            for name in order:
                os.replace(self._staged[name], self.destinations[name])
        except Exception as error:
            rollback_errors = self._restore(order, backups, existed)
            self.cleanup()
            detail = (
                f"; rollback also failed: {', '.join(rollback_errors)}"
                if rollback_errors else ""
            )
            raise ExporterError(Diagnostic(
                code="VXPUB002",
                message=f"failed to publish staged package: {error}{detail}",
                stage="publication",
                constraint="rollback-safe package publication",
            )) from error
        self.cleanup()

    def cleanup(self) -> None:
        if self._temporary is not None:
            shutil.rmtree(self._temporary, ignore_errors=True)
        self._temporary = None
        self._staged = {}

    def __enter__(self) -> "ArtifactTransaction":
        try:
            self._prepare()
        except Exception:
            self.cleanup()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()


class DirectoryPackageStage:
    """Copy a source package privately and expose it with one final rename."""

    def __init__(self, source: Path, output: Path) -> None:
        self.source = Path(source).resolve()
        self.output = _absolute(Path(output))
        self._temporary: Path | None = None
        self.staged_output: Path | None = None
        resolved_output = self.output.resolve(strict=False)
        if (
            not self.source.is_dir()
            or not self.output.name
            or self.output == self.output.parent
            or self.source == resolved_output
            or resolved_output.is_relative_to(self.source)
            or self.source.is_relative_to(resolved_output)
        ):
            raise ExporterError(Diagnostic(
                code="VXPUB005",
                message=(
                    "split-package source and destination must be distinct, "
                    "non-nested directories"
                ),
                stage="publication",
                constraint="safe split-package destination",
            ))
        if self.output.exists() or self.output.is_symlink():
            raise ExporterError(Diagnostic(
                code="VXPUB005",
                message=f"refusing to overwrite existing destination {self.output}",
                stage="publication",
                constraint="new split-package destination",
            ))

    @property
    def directory(self) -> Path:
        if self._temporary is None:
            raise RuntimeError("directory package stage has not been entered")
        return self._temporary

    def _prepare(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = Path(tempfile.mkdtemp(
            prefix=f".{self.output.name}.publish-",
            dir=str(self.output.parent),
        ))
        self.staged_output = self._temporary
        try:
            shutil.copytree(self.source, self.staged_output, dirs_exist_ok=True)
        except Exception as error:
            self.cleanup()
            raise ExporterError(Diagnostic(
                code="VXPUB006",
                message=f"failed to stage split package: {error}",
                stage="publication",
                constraint="complete private split-package copy",
            )) from error

    def publish(self) -> None:
        if self.staged_output is None or not self.staged_output.is_dir():
            raise ExporterError(Diagnostic(
                code="VXPUB001",
                message="staged split package is missing",
                stage="publication",
                constraint="complete staged package",
            ))
        if self.output.exists() or self.output.is_symlink():
            raise ExporterError(Diagnostic(
                code="VXPUB005",
                message=f"refusing to overwrite existing destination {self.output}",
                stage="publication",
                constraint="new split-package destination",
            ))
        try:
            os.replace(self.staged_output, self.output)
        except Exception as error:
            raise ExporterError(Diagnostic(
                code="VXPUB007",
                message=f"failed to publish staged split package: {error}",
                stage="publication",
                constraint="single-rename split-package publication",
            )) from error
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self._temporary is not None:
            shutil.rmtree(self._temporary, ignore_errors=True)
        self._temporary = None
        self.staged_output = None

    def __enter__(self) -> "DirectoryPackageStage":
        try:
            self._prepare()
        except Exception:
            self.cleanup()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()
