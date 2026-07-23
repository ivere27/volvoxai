from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import unittest

from tools.exporter.optimizer.profiling import (
    CompilerIdentity,
    DeviceIdentity,
    MeasuredBackend,
    ProfileMetrics,
    ProfileRecord,
    TimingAggregate,
    TimingCacheKey,
    content_digest,
)
from tools.exporter.optimizer.candidate import RewriteSemantics
from tools.exporter.optimizer.workspace import (
    BackendProfile,
    OptimizationCandidate,
    OptimizationWorkspace,
    PrecisionContract,
    TransformRecord,
    optimizer_implementation_hash,
)


def _digest(label: str) -> str:
    return content_digest(label)


_SOURCE_GRAPH_HASH = _digest("source graph")
_SOURCE_WEIGHTS_HASH = _digest("source weights")
_OPTIMIZER_HASH = _digest("optimizer")
_KERNEL_REGISTRY_HASH = _digest("kernel-registry")


def _backend_profile() -> BackendProfile:
    return BackendProfile(
        "portable",
        required_features=("quantized-linear", "static-shape"),
    )


def _precision(name: str = "w4a32") -> PrecisionContract:
    return PrecisionContract(
        name=name,
        activation_dtype="float32",
        weight_dtype="s4-packed-u8",
        accumulator_dtype="float32",
        output_dtype="float32",
        scheme="symmetric-per-group",
        group_size=64,
        parameters=(("nibble_order", "low-even"), ("tail", "zero-pad")),
    )


def _candidate(
    name: str = "candidate", *, parent: str | None = None,
) -> OptimizationCandidate:
    return OptimizationCandidate(
        graph_hash=(
            _SOURCE_GRAPH_HASH if parent is None else _digest(f"{name}-graph")
        ),
        weights_hash=(
            _SOURCE_WEIGHTS_HASH if parent is None else _digest(f"{name}-weights")
        ),
        optimizer_hash=_OPTIMIZER_HASH,
        kernel_registry_hash=_KERNEL_REGISTRY_HASH,
        backend_profile=_backend_profile(),
        precision=_precision(),
        transforms=(
            TransformRecord(
                pass_id="runtime-silu-fusion",
                implementation_hash=optimizer_implementation_hash(
                    "runtime-silu-fusion"
                ),
                config_hash=_digest("pass config"),
                semantic_mode=RewriteSemantics.NUMERICAL_MIGRATION,
            ),
        ),
        parent_candidate_id=parent,
    )


def _measured_backend() -> MeasuredBackend:
    return MeasuredBackend(
        backend_id="wasm",
        runtime_version="0.9.0",
        compiler=CompilerIdentity("emcc", "4.0.1", _digest("emcc build")),
        device=DeviceIdentity(
            "browser:wasm",
            _digest("browser cpu"),
            "browser-131",
            ("wasm.simd128",),
        ),
    )


class OptimizerWorkspaceTests(unittest.TestCase):
    def workspace(self) -> OptimizationWorkspace:
        return OptimizationWorkspace(
            source_graph_hash=_SOURCE_GRAPH_HASH,
            source_weights_hash=_SOURCE_WEIGHTS_HASH,
            optimizer_hash=_OPTIMIZER_HASH,
            kernel_registry_hash=_KERNEL_REGISTRY_HASH,
        )

    def test_candidate_is_immutable_content_addressed_and_round_trips(self):
        candidate = _candidate()
        encoded = candidate.to_json()
        restored = OptimizationCandidate.from_json(encoded)

        self.assertEqual(restored, candidate)
        self.assertEqual(restored.to_json(), encoded)
        self.assertTrue(candidate.candidate_id.startswith("sha256:"))
        self.assertEqual(
            candidate.backend_profile.member_targets,
            ("cpu-js", "wasm", "webgpu", "native-cpu"),
        )
        with self.assertRaises(FrozenInstanceError):
            candidate.graph_hash = _digest("mutated")

        changed = replace(candidate, weights_hash=_digest("changed weights"))
        self.assertNotEqual(changed.candidate_id, candidate.candidate_id)

    def test_transform_order_is_part_of_candidate_identity(self):
        first = _candidate()
        second_transform = TransformRecord(
            "runtime-canonicalize",
            optimizer_implementation_hash("runtime-canonicalize"),
            _digest("fuse config"),
        )
        forward = replace(first, transforms=(*first.transforms, second_transform))
        reverse = replace(first, transforms=(second_transform, *first.transforms))
        self.assertNotEqual(forward.candidate_id, reverse.candidate_id)

    def test_backend_profile_members_are_owned_by_kernel_registry(self):
        self.assertEqual(
            BackendProfile("native-cpu").member_targets,
            ("native-cpu",),
        )
        with self.assertRaisesRegex(ValueError, "members are generated"):
            BackendProfile("portable", member_targets=("wasm",))
        with self.assertRaisesRegex(ValueError, "unknown generated"):
            BackendProfile("private-target")

    def test_transform_semantics_use_the_typed_optimizer_vocabulary(self):
        authored = TransformRecord(
            "runtime-ptq-authoring",
            optimizer_implementation_hash("runtime-ptq-authoring"),
            _digest("ptq config"),
            RewriteSemantics.QUANTIZATION_AUTHORING,
        )
        self.assertEqual(
            authored.to_dict()["semantic_mode"],
            "quantization-authoring",
        )
        with self.assertRaisesRegex(TypeError, "RewriteSemantics"):
            TransformRecord(
                "bad",
                _digest("bad implementation"),
                _digest("bad config"),
                "abi-specialization",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "unknown generated optimizer pass"):
            TransformRecord(
                "private-pass",
                _digest("private implementation"),
                _digest("private config"),
                RewriteSemantics.EXACT,
            )
        with self.assertRaisesRegex(ValueError, "requires semantic_mode"):
            TransformRecord(
                "runtime-silu-fusion",
                _digest("silu implementation"),
                _digest("silu config"),
                RewriteSemantics.EXACT,
            )
        with self.assertRaisesRegex(ValueError, "active executable source"):
            TransformRecord(
                "runtime-canonicalize",
                _digest("stale implementation"),
                _digest("canonicalize config"),
                RewriteSemantics.EXACT,
            )

    def test_workspace_updates_are_immutable_and_content_addressed(self):
        empty = self.workspace()
        baseline = _candidate("baseline")
        one = empty.with_candidate(baseline)
        optimized = _candidate("optimized", parent=baseline.candidate_id)
        two = one.with_candidate(optimized)

        self.assertEqual(empty.candidates, ())
        self.assertEqual(len(one.candidates), 1)
        self.assertEqual(len(two.candidates), 2)
        self.assertNotEqual(empty.workspace_id, one.workspace_id)
        self.assertNotEqual(one.workspace_id, two.workspace_id)
        self.assertEqual(empty.workspace_lineage_id, two.workspace_lineage_id)
        self.assertIs(two.with_candidate(optimized), two)

        restored = OptimizationWorkspace.from_json(two.to_json())
        self.assertEqual(restored, two)
        self.assertEqual(restored.to_json(), two.to_json())

        with self.assertRaisesRegex(ValueError, "parent is not present"):
            empty.with_candidate(optimized)

        foreign_root = replace(
            baseline,
            graph_hash=_digest("foreign source graph"),
        )
        with self.assertRaisesRegex(ValueError, "rooted in the workspace"):
            empty.with_candidate(foreign_root)

    def test_workspace_binds_cache_key_to_lineage_plan_candidate_and_precision(self):
        candidate = _candidate()
        workspace = self.workspace().with_candidate(candidate)
        measured = _measured_backend()
        key = TimingCacheKey(
            workspace_lineage_id=workspace.workspace_lineage_id,
            graph_hash=candidate.graph_hash,
            weights_hash=candidate.weights_hash,
            optimizer_hash=candidate.optimizer_hash,
            kernel_registry_hash=candidate.kernel_registry_hash,
            compiled_plan_id=_digest("compiled model plan"),
            backend_profile=candidate.backend_profile.name,
            measured_backend=measured,
            workload_hash=_digest("decode-row workload"),
            precision_hash=candidate.precision.precision_id,
            candidate_id=candidate.candidate_id,
        )
        profile = ProfileRecord(
            workspace_lineage_id=workspace.workspace_lineage_id,
            candidate_id=candidate.candidate_id,
            compiled_plan_id=key.compiled_plan_id,
            backend_profile=candidate.backend_profile.name,
            measured_backend=measured,
            workload_hash=key.workload_hash,
            precision_hash=candidate.precision.precision_id,
            metrics=ProfileMetrics(
                execution=TimingAggregate.from_samples((2.0, 2.2, 2.1)),
                compile_timing=TimingAggregate.from_samples((20.0,)),
                pack_timing=TimingAggregate.from_samples((5.0,)),
                peak_scratch_bytes=4096,
            ),
        )
        measured_workspace = workspace.with_profile(key, profile)

        self.assertIsNone(workspace.timing_cache.lookup(key))
        self.assertEqual(measured_workspace.timing_cache.lookup(key), profile)
        self.assertNotEqual(workspace.workspace_id, measured_workspace.workspace_id)
        self.assertEqual(
            OptimizationWorkspace.from_json(measured_workspace.to_json()),
            measured_workspace,
        )

        with self.assertRaisesRegex(ValueError, "precision_hash"):
            workspace.with_profile(
                replace(key, precision_hash=_digest("wrong precision")),
                replace(profile, precision_hash=_digest("wrong precision")),
            )

        wrong_lineage = _digest("wrong workspace lineage")
        with self.assertRaisesRegex(ValueError, "workspace_lineage_id"):
            workspace.with_profile(
                replace(key, workspace_lineage_id=wrong_lineage),
                replace(profile, workspace_lineage_id=wrong_lineage),
            )

    def test_workspace_rejects_tampered_content_address(self):
        workspace = self.workspace().with_candidate(_candidate())
        payload = json.loads(workspace.to_json())
        payload["workspace_id"] = _digest("tampered")
        with self.assertRaisesRegex(ValueError, "does not match"):
            OptimizationWorkspace.from_dict(payload)

        payload = json.loads(workspace.to_json())
        payload["workspace_lineage_id"] = _digest("foreign lineage")
        with self.assertRaisesRegex(ValueError, "workspace_lineage_id does not match"):
            OptimizationWorkspace.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
