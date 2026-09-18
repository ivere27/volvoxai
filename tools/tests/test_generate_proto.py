import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_proto as codegen


def tar_archive(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return data.getvalue()


class GeneratorReleaseTests(unittest.TestCase):
    target = "x86_64-unknown-linux-musl"

    def test_changed_typescript_writer_requires_review(self):
        with self.assertRaisesRegex(codegen.CodegenError, "bulk writer needs review"):
            codegen.typescript_bulk_writer(b"class ProtoWriter { /* unfamiliar upstream codec */ }")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        self.binary = b"verified generator fixture"
        self.archive = tar_archive({
            f"{codegen.GENERATOR_NAME}-{codegen.SYNURANG_VERSION}-{self.target}/"
            f"{codegen.GENERATOR_NAME}": self.binary,
        })
        digest = hashlib.sha256(self.archive).hexdigest()
        patches = [
            mock.patch.object(codegen, "generator_target", return_value=self.target),
            mock.patch.dict(codegen.GENERATOR_ARCHIVES, {self.target: ("tar.gz", digest)}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.archive_path = (self.cache / f"v{codegen.SYNURANG_VERSION}" / "archives"
                             / codegen.generator_archive_name(self.target))
        self.archive_path.parent.mkdir(parents=True)
        self.archive_path.write_bytes(self.archive)

    def test_offline_extracts_verified_release_without_network(self):
        with mock.patch.object(codegen.urllib.request, "urlopen") as network:
            executable = codegen.release_generator(self.cache, True)
            self.assertEqual(executable.read_bytes(), self.binary)
            self.assertEqual(codegen.release_generator(self.cache, True), executable)
        network.assert_not_called()

    def test_modified_binary_is_restored_from_verified_archive(self):
        executable = codegen.release_generator(self.cache, True)
        executable.write_bytes(b"x" * len(self.binary))
        self.assertEqual(codegen.release_generator(self.cache, True).read_bytes(), self.binary)

    def test_modified_archive_is_rejected_even_with_cached_binary(self):
        codegen.release_generator(self.cache, True)
        self.archive_path.write_bytes(b"modified release")
        with self.assertRaisesRegex(codegen.CodegenError, "SHA-256 mismatch"):
            codegen.release_generator(self.cache, True)

    def test_missing_offline_archive_does_not_download(self):
        self.archive_path.unlink()
        with mock.patch.object(codegen.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(codegen.CodegenError, "offline cache is missing"):
                codegen.release_generator(self.cache, True)
        network.assert_not_called()

    def test_symlink_cache_is_rejected(self):
        source = self.cache / "release.tar.gz"
        self.archive_path.rename(source)
        self.archive_path.symlink_to(source)
        with self.assertRaisesRegex(codegen.CodegenError, "not a regular file"):
            codegen.release_generator(self.cache, True)

    def test_download_verified_before_cache_write(self):
        self.archive_path.unlink()
        with mock.patch.object(codegen.urllib.request, "urlopen",
                               return_value=io.BytesIO(b"corrupt download")):
            with self.assertRaisesRegex(codegen.CodegenError, "SHA-256 mismatch"):
                codegen.release_generator(self.cache, False)
        self.assertFalse(self.archive_path.exists())

    def test_download_can_be_reused_offline(self):
        self.archive_path.unlink()
        with mock.patch.object(codegen.urllib.request, "urlopen",
                               return_value=io.BytesIO(self.archive)) as network:
            executable = codegen.release_generator(self.cache, False)
        self.assertEqual(network.call_count, 1)
        self.assertEqual(codegen.release_generator(self.cache, True), executable)

    def test_fetch_only_also_populates_runtime_source_cache(self):
        with mock.patch.object(codegen, "resolve_generator", return_value=Path("fixture")):
            with mock.patch.object(codegen, "source_archive_bytes") as source:
                self.assertEqual(codegen.main([
                    "--fetch-only", "--offline", "--cache-dir", str(self.cache),
                ]), 0)
        source.assert_called_once_with(self.cache.resolve(), True)


class GeneratorArchiveTests(unittest.TestCase):
    def test_selects_only_published_platform_assets(self):
        for system, machine, expected in [
            ("Linux", "AMD64", "x86_64-unknown-linux-musl"),
            ("Linux", "arm64", "aarch64-unknown-linux-musl"),
            ("Windows", "AMD64", "x86_64-pc-windows-gnu"),
        ]:
            with self.subTest(system=system, machine=machine):
                with mock.patch.object(codegen.platform, "system", return_value=system):
                    with mock.patch.object(codegen.platform, "machine", return_value=machine):
                        self.assertEqual(codegen.generator_target(), expected)
        with mock.patch.object(codegen.platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(codegen.CodegenError, "use --generator"):
                codegen.generator_target()

    def test_extracts_windows_release_executable(self):
        target = "x86_64-pc-windows-gnu"
        data = io.BytesIO()
        with zipfile.ZipFile(data, mode="w") as archive:
            archive.writestr(
                f"{codegen.GENERATOR_NAME}-{codegen.SYNURANG_VERSION}-{target}/"
                f"{codegen.GENERATOR_NAME}.exe", b"Windows generator fixture",
            )
        name, binary = codegen.extract_generator(data.getvalue(), target)
        self.assertEqual(name, codegen.GENERATOR_NAME + ".exe")
        self.assertEqual(binary, b"Windows generator fixture")

    def test_source_paths_cannot_escape_tag_root(self):
        prefix = f"synurang-{codegen.SYNURANG_VERSION}/"
        for path in (prefix + "../outside", "other/source.c"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(codegen.CodegenError, "invalid Synurang source path"):
                    codegen.source_files(tar_archive({path: b"fixture"}))

    def test_source_archive_keeps_upstream_bytes(self):
        prefix = f"synurang-{codegen.SYNURANG_VERSION}/"
        source = b"/* upstream runtime */\n"
        self.assertEqual(codegen.source_files(tar_archive({prefix + "src/call.c": source})),
                         {"src/call.c": source})

    def test_manifest_owned_legacy_runtime_files_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "synurang/plugin.py"
            legacy.parent.mkdir()
            legacy.write_text("# old upstream runtime, without a generated marker\n")
            codegen.manifest_path(root, "python").write_text(json.dumps({
                "files": ["synurang/plugin.py"],
            }))
            result = codegen.GenerationResult(
                files={}, mode_files={"python": ()}, mode_roots={"python": root},
            )
            self.assertEqual(codegen.obsolete_generated_paths(result, ("python",)), [legacy])

    def test_obsolete_symlink_does_not_delete_its_unowned_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "user.py"
            target.write_text("# user file\n")
            legacy = root / "plugin.py"
            legacy.symlink_to(target)
            codegen.manifest_path(root, "python").write_text(json.dumps({
                "files": ["plugin.py"],
            }))
            result = codegen.GenerationResult(
                files={}, mode_files={"python": ()}, mode_roots={"python": root},
            )
            codegen.install_or_check(result, ("python",), False)
            self.assertFalse(legacy.exists())
            self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main()
