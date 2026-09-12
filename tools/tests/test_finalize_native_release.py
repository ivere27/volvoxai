from __future__ import annotations

import fcntl
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

from tools import finalize_native_release as finalizer


PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def _gnu_stack_flags(data: bytearray) -> tuple[str, int, int]:
    if data[:4] != b"\x7fELF":
        raise AssertionError("the Python test interpreter is not an ELF executable")
    elf_class = data[4]
    byte_order = data[5]
    endian = "<" if byte_order == 1 else ">"
    if elf_class == 2:
        program_offset = struct.unpack_from(endian + "Q", data, 32)[0]
        program_entry_size = struct.unpack_from(endian + "H", data, 54)[0]
        program_entry_count = struct.unpack_from(endian + "H", data, 56)[0]
        flags_field_offset = 4
    elif elf_class == 1:
        program_offset = struct.unpack_from(endian + "I", data, 28)[0]
        program_entry_size = struct.unpack_from(endian + "H", data, 42)[0]
        program_entry_count = struct.unpack_from(endian + "H", data, 44)[0]
        flags_field_offset = 24
    else:
        raise AssertionError(f"unsupported ELF class {elf_class}")

    for index in range(program_entry_count):
        entry = program_offset + index * program_entry_size
        program_type = struct.unpack_from(endian + "I", data, entry)[0]
        if program_type == PT_GNU_STACK:
            flags_offset = entry + flags_field_offset
            flags = struct.unpack_from(endian + "I", data, flags_offset)[0]
            return endian, flags_offset, flags
    raise AssertionError("the Python test interpreter has no PT_GNU_STACK header")


def _allocated_payload_offset(data: bytearray, minimum_size: int) -> int:
    elf_class = data[4]
    byte_order = data[5]
    endian = "<" if byte_order == 1 else ">"
    if elf_class == 2:
        section_offset = struct.unpack_from(endian + "Q", data, 40)[0]
        section_entry_size = struct.unpack_from(endian + "H", data, 58)[0]
        section_entry_count = struct.unpack_from(endian + "H", data, 60)[0]
        section_format = endian + "IIQQQQIIQQ"
    elif elf_class == 1:
        section_offset = struct.unpack_from(endian + "I", data, 32)[0]
        section_entry_size = struct.unpack_from(endian + "H", data, 46)[0]
        section_entry_count = struct.unpack_from(endian + "H", data, 48)[0]
        section_format = endian + "IIIIIIIIII"
    else:
        raise AssertionError(f"unsupported ELF class {elf_class}")

    for index in range(section_entry_count):
        entry = section_offset + index * section_entry_size
        header = struct.unpack_from(section_format, data, entry)
        section_type, flags, offset, size = header[1], header[2], header[4], header[5]
        if flags & 0x2 and section_type != 8 and size >= minimum_size:
            return offset
    raise AssertionError("the Python test interpreter has no large SHF_ALLOC section")


class NativeReleaseFinalizationTests(unittest.TestCase):
    def test_release_link_manifest_attests_ordered_object_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build = root / "build"
            working = build / "native"
            first = working / "CMakeFiles" / "first.dir" / "first.c.o"
            second = working / "CMakeFiles" / "second dir" / "second.c.o"
            for path in (first, second):
                path.parent.mkdir(parents=True, exist_ok=True)
            first.write_bytes(b"first object")
            second.write_bytes(b"second object")

            manifest = finalizer.release_link_manifest(
                root, build, [first, second]
            )

            self.assertEqual(
                [entry["path"] for entry in manifest["objects"]],
                [
                    "build/native/CMakeFiles/first.dir/first.c.o",
                    "build/native/CMakeFiles/second dir/second.c.o",
                ],
            )
            self.assertEqual(manifest["objects"][0]["rawBytes"], len(b"first object"))

            with self.assertRaisesRegex(
                finalizer.FinalizationError, r"repeats object input"
            ):
                finalizer.release_link_manifest(
                    root, build, [first, first]
                )

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "ELF finalization is Linux-only"
    )
    def test_runtime_snapshot_rejects_executable_stack_mutation(self) -> None:
        original = bytearray(Path(sys.executable).resolve().read_bytes())
        endian, flags_offset, stack_flags = _gnu_stack_flags(original)
        self.assertEqual(stack_flags & PF_X, 0, "test interpreter already has RWE stack")
        mutated = bytearray(original)
        struct.pack_into(endian + "I", mutated, flags_offset, stack_flags | PF_X)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_path = root / "original"
            mutated_path = root / "rwe-stack"
            original_path.write_bytes(original)
            mutated_path.write_bytes(mutated)
            before_image = finalizer.loadable_image(original_path)
            after_image = finalizer.loadable_image(mutated_path)

        # The mutation is outside the old PT_LOAD/allocated-section evidence.
        self.assertEqual(before_image["elfHeader"], after_image["elfHeader"])
        self.assertEqual(before_image["loadSegments"], after_image["loadSegments"])
        self.assertEqual(
            before_image["loadableSectionsSha256"],
            after_image["loadableSectionsSha256"],
        )
        self.assertNotEqual(
            before_image["programHeaders"], after_image["programHeaders"]
        )

        stable = {"buildId": "unchanged", "loader": {}, "version": None}
        before = {**stable, **before_image}
        after = {**stable, **after_image}
        with self.assertRaisesRegex(
            finalizer.FinalizationError,
            r"changed programHeaders",
        ):
            finalizer.require_unchanged_runtime_snapshot(before, after)

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "ELF finalization is Linux-only"
    )
    def test_cross_version_requires_one_allocated_literal_prefix(self) -> None:
        version = "9.8.7-test"
        prefix = finalizer.version_prefix(version).encode("utf-8")
        original = bytearray(Path(sys.executable).resolve().read_bytes())
        offset = _allocated_payload_offset(original, len(prefix))
        mutated = bytearray(original)
        mutated[offset : offset + len(prefix)] = prefix

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_path = root / "missing-prefix"
            mutated_path = root / "exact-prefix"
            original_path.write_bytes(original)
            mutated_path.write_bytes(mutated)
            _, missing = finalizer.inspect_elf_image(original_path, prefix)
            _, exact = finalizer.inspect_elf_image(mutated_path, prefix)

            self.assertEqual(missing, 0)
            with self.assertRaisesRegex(
                finalizer.FinalizationError, r"found 0"
            ):
                finalizer.version_output(original_path, False, version, missing)
            self.assertEqual(exact, 1)
            self.assertEqual(
                finalizer.version_output(mutated_path, False, version, exact),
                prefix.decode("utf-8"),
            )

    def test_host_version_requires_complete_nonempty_provenance(self) -> None:
        artifact = Path("/tmp/volvoxai-version-test")
        valid = (
            "VolvoxAI Native Engine 9.8.7-test "
            "(commit 0123456789ab-dirty, built 2026-09-02T01:02:03Z)\n"
        )
        with mock.patch.object(finalizer, "command", return_value=valid):
            self.assertEqual(
                finalizer.version_output(artifact, True, "9.8.7-test"),
                valid.strip(),
            )

        invalid = (
            "VolvoxAI Native Engine 9.8.7-test "
            "(commit , built 2026-09-02T01:02:03Z)\n"
        )
        with mock.patch.object(finalizer, "command", return_value=invalid):
            with self.assertRaisesRegex(
                finalizer.FinalizationError, r"complete provenance record"
            ):
                finalizer.version_output(artifact, True, "9.8.7-test")

    def test_external_build_paths_stay_inside_the_selected_build_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            root = temporary / "repository"
            build = temporary / "external-build"
            root.mkdir()
            build.mkdir()
            artifact = build / "release"
            self.assertEqual(
                finalizer.build_private_path(
                    build,
                    artifact,
                    "artifact",
                ),
                artifact,
            )
            self.assertEqual(
                finalizer.repository_relative(root, artifact),
                "../external-build/release",
            )
            with self.assertRaisesRegex(
                finalizer.FinalizationError, r"inside repository root"
            ):
                finalizer.repository_path(root, artifact, "published artifact")
            with self.assertRaisesRegex(
                finalizer.FinalizationError, r"inside build directory"
            ):
                finalizer.build_private_path(build, root / "native/release", "artifact")
            with self.assertRaisesRegex(
                finalizer.FinalizationError, r"canonical repository-relative"
            ):
                finalizer.logical_release_path("native/../release", "artifact")

    def test_publish_bundle_is_locked_and_installed_in_safe_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "build"
            sources.mkdir()
            artifact = sources / "volvoxai"
            debug = sources / "volvoxai.debug"
            evidence = sources / "volvoxai.debug.json"
            artifact.write_bytes(b"artifact")
            artifact.chmod(0o755)
            debug.write_bytes(b"debug")
            evidence.write_bytes(b"evidence")
            publish_artifact = root / "native/volvoxai"
            publish_debug = root / "native/.debug/volvoxai.debug"
            publish_evidence = root / "native/.debug/volvoxai.debug.json"
            lock = root / finalizer.PUBLISH_LOCK
            lock.parent.mkdir(parents=True)
            lock.touch()
            lock.chmod(0o444)
            destinations: list[Path] = []
            lock_opens: list[tuple[int, int]] = []
            atomic_copy = finalizer.atomic_copy
            real_os_open = os.open

            def record_open(
                path: os.PathLike[str] | str,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                if Path(path) == lock:
                    lock_opens.append((flags, mode))
                options = {} if dir_fd is None else {"dir_fd": dir_fd}
                return real_os_open(path, flags, mode, **options)

            def record_copy(source: Path, destination: Path) -> None:
                probe = real_os_open(lock, os.O_RDONLY)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(probe)
                destinations.append(destination)
                atomic_copy(source, destination)

            with mock.patch.object(
                finalizer.os, "open", side_effect=record_open
            ), mock.patch.object(finalizer, "atomic_copy", side_effect=record_copy):
                finalizer.publish_release_bundle(
                    root,
                    artifact,
                    debug,
                    evidence,
                    publish_artifact,
                    publish_debug,
                    publish_evidence,
                )

            self.assertEqual(
                destinations,
                [publish_debug, publish_evidence, publish_artifact],
            )
            self.assertEqual(publish_debug.read_bytes(), b"debug")
            self.assertEqual(publish_evidence.read_bytes(), b"evidence")
            self.assertEqual(publish_artifact.read_bytes(), b"artifact")
            self.assertEqual(len(lock_opens), 1)
            lock_flags, lock_mode = lock_opens[0]
            self.assertEqual(lock_flags & os.O_ACCMODE, os.O_RDONLY)
            self.assertNotEqual(lock_flags & os.O_CREAT, 0)
            self.assertEqual(lock_mode, 0o666)
            self.assertEqual(lock.stat().st_mode & 0o777, 0o444)

    def test_symbolization_accepts_prefix_mapped_source_identity(self) -> None:
        locations = (
            "././native/cli/main.c:1971",
            "./build/m1-shards/native/./native/cli/main.c:1971",
            "../external-build/native/./native/cli/main.c:1971",
            "/workspace/volvoxai/native/cli/main.c:1971",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native/cli/main.c"
            source.parent.mkdir(parents=True)
            source.write_text("int main(void) { return 0; }\n", encoding="utf-8")

            for location in locations:
                with self.subTest(location=location), mock.patch.object(
                    finalizer,
                    "command",
                    return_value=f"main\n{location}\n",
                ):
                    evidence = finalizer.require_debug_symbolization(
                        "addr2line",
                        Path("release"),
                        Path("debug"),
                        "1234",
                        root,
                    )
                self.assertEqual(evidence["release"], "native/cli/main.c:1971")
                self.assertEqual(evidence["debug"], "native/cli/main.c:1971")

            with mock.patch.object(
                finalizer,
                "command",
                return_value="main\n/workspace/volvoxai/src/cli/main.c:1971\n",
            ), self.assertRaisesRegex(
                finalizer.FinalizationError, r"unexpected source"
            ):
                finalizer.require_debug_symbolization(
                    "addr2line",
                    Path("release"),
                    Path("debug"),
                    "1234",
                    root,
                )

    def test_aarch64_requires_both_hot_kernel_symbolizations(self) -> None:
        symbols = {symbol for symbol, _ in finalizer.AARCH64_ADDITIONAL_SYMBOLS}
        addresses = {symbol: f"{index + 1:x}" for index, symbol in enumerate(symbols)}
        with mock.patch.object(
            finalizer,
            "require_debug_symbolization",
            side_effect=lambda *arguments: {"symbol": arguments[5]},
        ) as symbolize:
            evidence = finalizer.require_additional_symbolizations(
                finalizer.AARCH64_ELF_MACHINE,
                "addr2line",
                Path("release"),
                Path("debug"),
                symbols,
                addresses,
                Path("repository"),
            )
        self.assertEqual(
            [entry["symbol"] for entry in evidence],
            [symbol for symbol, _ in finalizer.AARCH64_ADDITIONAL_SYMBOLS],
        )
        self.assertEqual(symbolize.call_count, 2)

        with self.assertRaisesRegex(
            finalizer.FinalizationError, r"missing required hot-kernel symbols"
        ):
            finalizer.require_additional_symbolizations(
                finalizer.AARCH64_ELF_MACHINE,
                "addr2line",
                Path("release"),
                Path("debug"),
                set(),
                {},
                Path("repository"),
            )


if __name__ == "__main__":
    unittest.main()
