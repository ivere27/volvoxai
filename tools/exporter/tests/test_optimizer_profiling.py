from __future__ import annotations

from dataclasses import replace
import json
import unittest

from tools.exporter.optimizer.profiling import (
    CompilerIdentity,
    DeviceIdentity,
    MeasuredBackend,
    ParetoConstraints,
    ProfileMetrics,
    ProfileRecord,
    ScopedTiming,
    TimingAggregate,
    TimingCache,
    TimingCacheKey,
    content_digest,
    pareto_frontier,
)


def _digest(label: str) -> str:
    return content_digest(label)


def _measured_backend(suffix: str = "") -> MeasuredBackend:
    return MeasuredBackend(
        backend_id="native-cpu",
        runtime_version="1.2.3",
        compiler=CompilerIdentity(
            compiler_id="clang",
            version="19.1",
            build_hash=_digest(f"compiler{suffix}"),
        ),
        device=DeviceIdentity(
            device_id="cpu:0",
            fingerprint=_digest(f"device{suffix}"),
            driver_version="microcode-7",
            features=("x86.avx2", "x86.avx-vnni"),
        ),
    )


def _cache_key(suffix: str = "") -> TimingCacheKey:
    return TimingCacheKey(
        workspace_lineage_id=_digest(f"lineage{suffix}"),
        graph_hash=_digest(f"graph{suffix}"),
        weights_hash=_digest(f"weights{suffix}"),
        optimizer_hash=_digest(f"optimizer{suffix}"),
        kernel_registry_hash=_digest(f"registry{suffix}"),
        compiled_plan_id=_digest(f"compiled-plan{suffix}"),
        backend_profile="portable",
        measured_backend=_measured_backend(suffix),
        workload_hash=_digest(f"workload{suffix}"),
        precision_hash=_digest(f"precision{suffix}"),
        candidate_id=_digest(f"candidate{suffix}"),
    )


def _profile(
    key: TimingCacheKey,
    *,
    execution: tuple[float, ...] = (1.0, 1.2, 1.4),
    scratch: int = 64,
    quality_loss: float | None = 0.0,
) -> ProfileRecord:
    return ProfileRecord(
        workspace_lineage_id=key.workspace_lineage_id,
        candidate_id=key.candidate_id,
        compiled_plan_id=key.compiled_plan_id,
        backend_profile=key.backend_profile,
        measured_backend=key.measured_backend,
        workload_hash=key.workload_hash,
        precision_hash=key.precision_hash,
        metrics=ProfileMetrics(
            execution=TimingAggregate.from_samples(execution),
            compile_timing=TimingAggregate.from_samples((4.0, 5.0)),
            pack_timing=TimingAggregate.from_samples((2.0, 3.0)),
            peak_scratch_bytes=scratch,
            artifact_bytes=1024,
            quality_loss=quality_loss,
            node_timings=(
                ScopedTiming("node-b", TimingAggregate.from_samples((0.4, 0.5)), "Add"),
                ScopedTiming("node-a", TimingAggregate.from_samples((0.2, 0.3)), "QLinear"),
            ),
            region_timings=(
                ScopedTiming("decoder.block.0", TimingAggregate.from_samples((0.8, 0.9))),
            ),
        ),
    )


class OptimizerProfilingTests(unittest.TestCase):
    def test_timing_aggregates_and_scopes_are_deterministic(self):
        timing = TimingAggregate.from_samples((4.0, 1.0, 3.0, 2.0))
        self.assertEqual(timing.sample_count, 4)
        self.assertEqual(timing.minimum_ms, 1.0)
        self.assertEqual(timing.maximum_ms, 4.0)
        self.assertEqual(timing.mean_ms, 2.5)
        self.assertEqual(timing.p50_ms, 2.5)
        self.assertAlmostEqual(timing.p95_ms, 3.85)

        profile = _profile(_cache_key())
        self.assertEqual(
            tuple(item.scope_id for item in profile.metrics.node_timings),
            ("node-a", "node-b"),
        )
        with self.assertRaisesRegex(ValueError, "scope IDs must be unique"):
            ProfileMetrics(
                execution=timing,
                node_timings=(
                    ScopedTiming("same", timing),
                    ScopedTiming("same", timing),
                ),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TimingAggregate.from_samples((1.0, -1.0))

    def test_cache_key_covers_every_codegen_and_measurement_dimension(self):
        base = _cache_key()
        changed_backend = replace(
            base.measured_backend,
            compiler=replace(
                base.measured_backend.compiler,
                build_hash=_digest("another compiler"),
            ),
        )
        changed_device = replace(
            base.measured_backend,
            device=replace(
                base.measured_backend.device,
                fingerprint=_digest("another device"),
            ),
        )
        variants = (
            replace(base, workspace_lineage_id=_digest("another lineage")),
            replace(base, graph_hash=_digest("another graph")),
            replace(base, weights_hash=_digest("another weights")),
            replace(base, optimizer_hash=_digest("another optimizer")),
            replace(base, kernel_registry_hash=_digest("another registry")),
            replace(base, compiled_plan_id=_digest("another compiled plan")),
            replace(base, backend_profile="native-cpu"),
            replace(base, measured_backend=changed_backend),
            replace(base, measured_backend=changed_device),
            replace(base, workload_hash=_digest("another workload")),
            replace(base, precision_hash=_digest("another precision")),
            replace(base, candidate_id=_digest("another candidate")),
        )
        identifiers = {base.key_id, *(item.key_id for item in variants)}
        self.assertEqual(len(identifiers), len(variants) + 1)

    def test_profile_keeps_target_profile_separate_from_measured_backend(self):
        key = _cache_key()
        profile = _profile(key)
        payload = profile.to_dict()
        self.assertEqual(payload["backend_profile"], "portable")
        self.assertEqual(payload["measured_backend"]["backend_id"], "native-cpu")
        self.assertNotEqual(
            payload["backend_profile"], payload["measured_backend"]["backend_id"],
        )
        self.assertEqual(ProfileRecord.from_dict(payload), profile)

    def test_timing_cache_round_trip_is_deterministic_and_collision_safe(self):
        key = _cache_key()
        profile = _profile(key)
        cache = TimingCache().with_entry(key, profile)
        encoded = cache.to_json()
        restored = TimingCache.from_json(encoded)

        self.assertEqual(restored, cache)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(restored.lookup(key), profile)
        self.assertIs(cache.with_entry(key, profile), cache)
        with self.assertRaisesRegex(ValueError, "different profile"):
            cache.with_entry(
                key,
                replace(profile, metrics=replace(profile.metrics, peak_scratch_bytes=65)),
            )
        with self.assertRaisesRegex(ValueError, "workspace lineages differ"):
            TimingCache().with_entry(
                key,
                replace(
                    profile,
                    workspace_lineage_id=_digest("another workspace lineage"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "compiled plans differ"):
            TimingCache().with_entry(
                key,
                replace(profile, compiled_plan_id=_digest("another physical plan")),
            )

        tampered = json.loads(encoded)
        tampered["cache_id"] = _digest("tampered")
        with self.assertRaisesRegex(ValueError, "does not match"):
            TimingCache.from_dict(tampered)

    def test_pareto_frontier_applies_constraints_and_removes_dominated_points(self):
        base = _cache_key()

        def point(
            name: str, execution: tuple[float, ...], scratch: int, loss: float,
        ) -> ProfileRecord:
            key = replace(
                base,
                candidate_id=_digest(name),
                compiled_plan_id=_digest(f"compiled-plan-{name}"),
                precision_hash=_digest(f"precision-{name}"),
            )
            return _profile(
                key,
                execution=execution,
                scratch=scratch,
                quality_loss=loss,
            )

        fast = point("fast", (0.8, 1.0), 100, 0.20)
        compact = point("compact", (1.4, 1.6), 40, 0.05)
        dominated = point("dominated", (2.0, 2.2), 160, 0.30)
        inaccurate = point("inaccurate", (0.5, 0.7), 80, 0.60)

        frontier = pareto_frontier(
            (dominated, compact, inaccurate, fast),
            constraints=ParetoConstraints(max_quality_loss=0.25),
            objectives=("execution_p50_ms", "peak_scratch_bytes", "quality_loss"),
        )
        self.assertEqual(
            {item.candidate_id for item in frontier},
            {fast.candidate_id, compact.candidate_id},
        )

        other_workload = replace(compact, workload_hash=_digest("other workload"))
        with self.assertRaisesRegex(ValueError, "one workspace lineage"):
            pareto_frontier((fast, other_workload))

        other_lineage = replace(
            compact,
            workspace_lineage_id=_digest("another workspace lineage"),
        )
        with self.assertRaisesRegex(ValueError, "one workspace lineage"):
            pareto_frontier((fast, other_lineage))


if __name__ == "__main__":
    unittest.main()
