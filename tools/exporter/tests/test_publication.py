from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.exporter import publication
from tools.exporter.errors import ExporterError
from tools.exporter.publication import (
    ArtifactTransaction,
    DirectoryPackageStage,
    PackageStage,
)


class PackageStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvox-exporter-publication-")
        self.root = Path(self.temporary.name)
        self.weights = self.root / "model.safetensors"
        self.graph = self.root / "graph.json"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_stage(stage: PackageStage, weights: bytes = b"new weights", graph: str = "new graph") -> None:
        stage.staged_output.write_bytes(weights)
        (stage.directory / "graph.json").write_text(graph, encoding="utf-8")

    def assert_no_stage_directories(self):
        self.assertEqual(list(self.root.glob(".model.export-*")), [])

    def test_success_publishes_weights_then_graph_and_removes_stage(self):
        self.weights.write_bytes(b"old weights")
        self.graph.write_text("old graph", encoding="utf-8")
        real_replace = os.replace
        replacements: list[tuple[Path, Path]] = []

        with PackageStage(self.weights) as stage:
            stage_directory = stage.directory
            self.write_stage(stage)
            staged_weights = stage.staged_output
            staged_graph = stage.directory / "graph.json"

            def record_replace(source, destination):
                replacements.append((Path(source), Path(destination)))
                return real_replace(source, destination)

            with patch.object(publication.os, "replace", side_effect=record_replace):
                stage.publish()

        self.assertEqual(self.weights.read_bytes(), b"new weights")
        self.assertEqual(self.graph.read_text(encoding="utf-8"), "new graph")
        self.assertLess(
            replacements.index((staged_weights, self.weights)),
            replacements.index((staged_graph, self.graph)),
        )
        self.assertFalse(stage_directory.exists())
        self.assert_no_stage_directories()


class ArtifactTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="volvox-optimizer-publication-"
        )
        self.root = Path(self.temporary.name)
        self.graph = self.root / "optimized.graph.json"
        self.weights = self.root / "optimized.safetensors"

    def tearDown(self):
        self.temporary.cleanup()

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".optimized.graph.publish-*")), [])

    def test_graph_sentinel_publishes_last(self):
        self.graph.write_bytes(b"old graph")
        self.weights.write_bytes(b"old weights")
        real_replace = os.replace
        replacements: list[tuple[Path, Path]] = []

        with ArtifactTransaction(
            {"graph": self.graph, "weights": self.weights}, sentinel="graph"
        ) as transaction:
            staged_graph = transaction.staged_path("graph")
            staged_weights = transaction.staged_path("weights")
            staged_graph.write_bytes(b"new graph")
            staged_weights.write_bytes(b"new weights")

            def record_replace(source, destination):
                replacements.append((Path(source), Path(destination)))
                return real_replace(source, destination)

            with patch.object(publication.os, "replace", side_effect=record_replace):
                transaction.publish()

        self.assertEqual(self.graph.read_bytes(), b"new graph")
        self.assertEqual(self.weights.read_bytes(), b"new weights")
        self.assertLess(
            replacements.index((staged_weights, self.weights)),
            replacements.index((staged_graph, self.graph)),
        )
        self.assert_no_staging()

    def test_graph_replace_failure_restores_both_outputs_byte_exact(self):
        old_graph = b"old graph\x00payload"
        old_weights = b"old weights\xffpayload"
        self.graph.write_bytes(old_graph)
        self.weights.write_bytes(old_weights)
        real_replace = os.replace

        with ArtifactTransaction(
            {"graph": self.graph, "weights": self.weights}, sentinel="graph"
        ) as transaction:
            staged_graph = transaction.staged_path("graph")
            transaction.staged_path("weights").write_bytes(b"new weights")
            staged_graph.write_bytes(b"new graph")

            def fail_sentinel(source, destination):
                if Path(source) == staged_graph and Path(destination) == self.graph:
                    raise OSError("injected sentinel replace failure")
                return real_replace(source, destination)

            with patch.object(publication.os, "replace", side_effect=fail_sentinel):
                with self.assertRaises(ExporterError) as caught:
                    transaction.publish()
            self.assertEqual(caught.exception.diagnostic.code, "VXPUB002")

        self.assertEqual(self.graph.read_bytes(), old_graph)
        self.assertEqual(self.weights.read_bytes(), old_weights)
        self.assert_no_staging()

    def test_incomplete_stage_never_changes_destinations(self):
        self.graph.write_bytes(b"old graph")
        self.weights.write_bytes(b"old weights")
        with ArtifactTransaction(
            {"graph": self.graph, "weights": self.weights}, sentinel="graph"
        ) as transaction:
            transaction.staged_path("weights").write_bytes(b"new weights")
            with self.assertRaises(ExporterError) as caught:
                transaction.publish()
            self.assertEqual(caught.exception.diagnostic.code, "VXPUB001")
        self.assertEqual(self.graph.read_bytes(), b"old graph")
        self.assertEqual(self.weights.read_bytes(), b"old weights")
        self.assert_no_staging()


class DirectoryPackageStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="volvox-split-publication-"
        )
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "package_manifest.json").write_bytes(b"source")
        self.output = self.root / "optimized"

    def tearDown(self):
        self.temporary.cleanup()

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".optimized.publish-*")), [])

    def test_one_final_rename_exposes_complete_directory(self):
        real_replace = os.replace
        replacements: list[tuple[Path, Path]] = []
        with DirectoryPackageStage(self.source, self.output) as stage:
            staged = stage.staged_output
            assert staged is not None
            (staged / "package_manifest.json").write_bytes(b"optimized")

            def record_replace(source, destination):
                replacements.append((Path(source), Path(destination)))
                return real_replace(source, destination)

            with patch.object(publication.os, "replace", side_effect=record_replace):
                stage.publish()

        self.assertEqual(replacements, [(staged, self.output)])
        self.assertEqual(
            (self.output / "package_manifest.json").read_bytes(), b"optimized"
        )
        self.assert_no_staging()

    def test_failed_final_rename_leaves_no_visible_destination(self):
        real_replace = os.replace
        with DirectoryPackageStage(self.source, self.output) as stage:
            staged = stage.staged_output
            assert staged is not None

            def fail_replace(source, destination):
                if Path(source) == staged and Path(destination) == self.output:
                    raise OSError("injected directory rename failure")
                return real_replace(source, destination)

            with patch.object(publication.os, "replace", side_effect=fail_replace):
                with self.assertRaises(ExporterError) as caught:
                    stage.publish()
            self.assertEqual(caught.exception.diagnostic.code, "VXPUB007")

        self.assertFalse(self.output.exists())
        self.assert_no_staging()

    def test_nested_destination_is_rejected_before_staging(self):
        nested = self.source / "optimized"
        with self.assertRaises(ExporterError) as caught:
            DirectoryPackageStage(self.source, nested)
        self.assertEqual(caught.exception.diagnostic.code, "VXPUB005")
        self.assertFalse(nested.exists())


class PackageStageFailureTests(PackageStageTests):

    def test_missing_staged_file_keeps_existing_package_unchanged(self):
        self.weights.write_bytes(b"old weights")
        self.graph.write_text("old graph", encoding="utf-8")

        with PackageStage(self.weights) as stage:
            stage.staged_output.write_bytes(b"new weights")
            with self.assertRaises(ExporterError) as caught:
                stage.publish()
            self.assertEqual(caught.exception.diagnostic.code, "VXPUB001")

        self.assertEqual(self.weights.read_bytes(), b"old weights")
        self.assertEqual(self.graph.read_text(encoding="utf-8"), "old graph")
        self.assert_no_stage_directories()

    def test_failed_replace_restores_preexisting_package(self):
        self.weights.write_bytes(b"old weights")
        self.graph.write_text("old graph", encoding="utf-8")
        real_replace = os.replace

        with PackageStage(self.weights) as stage:
            self.write_stage(stage)
            staged_graph = stage.directory / "graph.json"

            def fail_graph_replace(source, destination):
                if Path(source) == staged_graph and Path(destination) == self.graph:
                    raise OSError("injected graph publication failure")
                return real_replace(source, destination)

            with patch.object(publication.os, "replace", side_effect=fail_graph_replace):
                with self.assertRaises(ExporterError) as caught:
                    stage.publish()
            self.assertEqual(caught.exception.diagnostic.code, "VXPUB002")

        self.assertEqual(self.weights.read_bytes(), b"old weights")
        self.assertEqual(self.graph.read_text(encoding="utf-8"), "old graph")
        self.assert_no_stage_directories()

    def test_failed_first_publication_leaves_no_partial_package(self):
        real_replace = os.replace

        with PackageStage(self.weights) as stage:
            self.write_stage(stage)
            staged_graph = stage.directory / "graph.json"

            def fail_graph_replace(source, destination):
                if Path(source) == staged_graph and Path(destination) == self.graph:
                    raise OSError("injected graph publication failure")
                return real_replace(source, destination)

            with patch.object(publication.os, "replace", side_effect=fail_graph_replace):
                with self.assertRaises(ExporterError) as caught:
                    stage.publish()
            self.assertEqual(caught.exception.diagnostic.code, "VXPUB002")

        self.assertFalse(self.weights.exists())
        self.assertFalse(self.graph.exists())
        self.assert_no_stage_directories()


if __name__ == "__main__":
    unittest.main()
