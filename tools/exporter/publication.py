"""Staged publication for fresh, graph-sentinel package destinations.

Publishing several pathnames cannot make them appear simultaneously.  These
helpers therefore expose a narrower reader-atomic contract: package readers
must open the graph sentinel before any payload, and publishers never replace
an already visible package.  Payloads are linked into fresh destinations
first and the graph is linked last.  A missing graph means "not committed".
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .errors import Diagnostic, ExporterError


def _absolute(path: Path) -> Path:
    """Normalize ``path`` without following a destination symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _exists(path: Path) -> bool:
    """Return true for every occupied pathname, including dangling symlinks."""

    return path.exists() or path.is_symlink()


def _lock_path(sentinel: Path) -> Path:
    return sentinel.with_name(f".{sentinel.name}.publish.lock")


@contextmanager
def _publication_lock(sentinel: Path):
    """Serialize cooperating publishers and fail closed on a stale lock."""

    lock = _lock_path(sentinel)
    try:
        descriptor = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise ExporterError(Diagnostic(
            code="VXPUB004",
            message=f"another publication owns or left the lock {lock}",
            stage="publication",
            constraint="exclusive fresh-package publication",
        )) from error
    except OSError as error:
        raise ExporterError(Diagnostic(
            code="VXPUB002",
            message=f"failed to acquire publication lock {lock}: {error}",
            stage="publication",
            constraint="exclusive fresh-package publication",
        )) from error
    os.close(descriptor)
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _remove_linked_destinations(
    names: Iterable[str],
    destinations: Mapping[str, Path],
) -> list[str]:
    errors: list[str] = []
    for name in reversed(list(names)):
        try:
            destinations[name].unlink()
        except FileNotFoundError:
            pass
        except OSError as error:  # pragma: no cover - catastrophic I/O
            errors.append(f"{name}: {error}")
    return errors


def _publish_fresh_files(
    staged: Mapping[str, Path],
    destinations: Mapping[str, Path],
    *,
    sentinel: str,
) -> None:
    """Hard-link complete files into fresh paths, with ``sentinel`` last.

    ``os.link`` supplies the portable no-clobber property that ``os.replace``
    lacks.  Unsupported filesystems fail before the sentinel becomes visible.
    """

    occupied = [name for name, path in destinations.items() if _exists(path)]
    if occupied:
        raise ExporterError(Diagnostic(
            code="VXPUB004",
            message=(
                "refusing to replace an existing package artifact: "
                f"{', '.join(occupied)}"
            ),
            stage="publication",
            constraint="fresh package artifact destinations",
        ))

    order = [name for name in destinations if name != sentinel]
    order.append(sentinel)
    linked: list[str] = []
    try:
        for name in order:
            os.link(staged[name], destinations[name])
            linked.append(name)
    except FileExistsError as error:
        rollback_errors = _remove_linked_destinations(linked, destinations)
        detail = (
            f"; cleanup also failed: {', '.join(rollback_errors)}"
            if rollback_errors else ""
        )
        raise ExporterError(Diagnostic(
            code="VXPUB004",
            message=f"package destination became occupied during publication{detail}",
            stage="publication",
            constraint="fresh package artifact destinations",
        )) from error
    except OSError as error:
        rollback_errors = _remove_linked_destinations(linked, destinations)
        detail = (
            f"; cleanup also failed: {', '.join(rollback_errors)}"
            if rollback_errors else ""
        )
        raise ExporterError(Diagnostic(
            code="VXPUB002",
            message=f"failed to publish fresh staged package: {error}{detail}",
            stage="publication",
            constraint="fresh-path graph-sentinel publication",
        )) from error


class PackageStage:
    """Build privately and publish a new package with ``graph.json`` last.

    Existing artifacts are never replaced.  A conforming reader opens
    ``graph.json`` first and opens the payload only after that succeeds.
    """

    def __init__(self, output: Path):
        self.output = _absolute(Path(output))
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

    def publish(
        self,
        filenames: Iterable[str] = ("graph.json",),
        *,
        sentinel: str = "graph.json",
    ) -> None:
        names = [self.output.name, *filenames]
        if (
            sentinel not in names
            or len(names) != len(set(names))
            or any(
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                for name in names
            )
        ):
            raise ExporterError(Diagnostic(
                code="VXPUB003",
                message=(
                    "package artifact names must be distinct basenames "
                    "containing the sentinel"
                ),
                stage="publication",
                constraint="safe package artifact names",
            ))
        staged = {name: self._temporary / name for name in names}
        missing = [
            name for name, path in staged.items()
            if path.is_symlink() or not path.is_file()
        ]
        if missing:
            raise ExporterError(Diagnostic(
                code="VXPUB001",
                message=f"staged package is missing: {', '.join(missing)}",
                stage="publication",
                constraint="complete staged package",
            ))

        destinations = {name: self.destination / name for name in names}
        try:
            with _publication_lock(destinations[sentinel]):
                _publish_fresh_files(
                    staged,
                    destinations,
                    sentinel=sentinel,
                )
        finally:
            self.cleanup()

    def __enter__(self) -> "PackageStage":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()


class ArtifactTransaction:
    """Stage a small set of files for fresh graph-sentinel publication.

    All staged files live beside the sentinel destination and every target
    parent must be on that same filesystem. Existing destinations are rejected.
    Non-sentinel artifacts publish first and the graph sentinel publishes last;
    readers must open the sentinel before opening any payload.
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

    def publish(self) -> None:
        self.require_complete()
        try:
            with _publication_lock(self.destinations[self.sentinel]):
                _publish_fresh_files(
                    self._staged,
                    self.destinations,
                    sentinel=self.sentinel,
                )
        finally:
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
        try:
            with _publication_lock(self.output):
                if _exists(self.output):
                    raise ExporterError(Diagnostic(
                        code="VXPUB005",
                        message=(
                            "refusing to overwrite existing destination "
                            f"{self.output}"
                        ),
                        stage="publication",
                        constraint="new split-package destination",
                    ))
                os.replace(self.staged_output, self.output)
        except ExporterError:
            raise
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
