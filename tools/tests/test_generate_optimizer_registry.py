from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.generate_optimizer_registry import (
    ROOT,
    load_registry,
    render_docs,
    render_python,
)


REGISTRY_PROTO = ROOT / "proto/optimizer_registry.proto"
PUBLIC_PROTO = ROOT / "proto/volvoxai.proto"
KERNEL_REGISTRY_PROTO = ROOT / "proto/kernel_registry.proto"


def _active_group_ids(recipe, features=()) -> tuple[str, ...]:
    selected = frozenset(features)
    return tuple(
        overlay.group_id
        for overlay in recipe.group_overlays
        if (
            frozenset(overlay.required_features) <= selected
            and not frozenset(overlay.forbidden_features) & selected
        )
    )


def _active_pass_ids(registry, recipe_id: str, features=()) -> tuple[str, ...]:
    recipes = {item.id: item for item in registry.pipelines}
    groups = {item.id: item for item in registry.groups}
    return tuple(
        pass_id
        for group_id in _active_group_ids(recipes[recipe_id], features)
        for pass_id in groups[group_id].pass_ids
    )


class OptimizerRegistryGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(
            REGISTRY_PROTO, PUBLIC_PROTO, KERNEL_REGISTRY_PROTO,
        )

    def test_v1_inventory_covers_current_typed_runtime_passes(self):
        self.assertEqual(self.registry.schema_version, 1)
        expected = {
            "runtime-vocabulary",
            "runtime-input-specialization",
            "runtime-input-hoisting",
            "runtime-declared-input-pruning",
            "runtime-constant-folding",
            "runtime-output-qargmax",
            "runtime-packed-qlinear-split",
            "runtime-quantized-bias-folding",
            "runtime-affine-reference-canonicalization",
            "runtime-float-attention-fusion",
            "runtime-quantized-attention-fusion",
            "runtime-quantized-attention-layout",
            "runtime-attention-layout",
            "runtime-keep-mask",
            "runtime-singleton-transpose",
            "runtime-qdq-transpose-cancellation",
            "runtime-pointwise-transpose-hoist",
            "runtime-qdq-movement",
            "runtime-elementwise-transpose",
            "runtime-shape-chain",
            "runtime-static-qdq-compute-fusion",
            "runtime-static-qdq-qbatch-matmul-fusion",
            "runtime-static-qdq-groupnorm-silu-fusion",
            "runtime-common-subexpression",
            "runtime-canonicalize",
            "redundant-qdq",
            "runtime-dead-code",
            "runtime-bias-folding",
            "runtime-silu-fusion",
            "runtime-grouped-projection-split",
            "runtime-sequence-layout",
            "runtime-ptq-authoring",
        }
        self.assertEqual({item.id for item in self.registry.passes}, expected)
        self.assertTrue(all(item.version == 1 for item in self.registry.passes))
        self.assertTrue(all(
            item.scope == "OPTIMIZER_PASS_SCOPE_PORTABLE_PACKAGE"
            for item in self.registry.passes
        ))
        self.assertIn(
            "OPTIMIZER_PASS_SCOPE_COMPILED_PREPARATION",
            self.registry.enum_numbers["OptimizerPassScope"],
        )

    def test_every_implementation_id_resolves_to_the_declared_class(self):
        for item in self.registry.passes:
            module_name, separator, class_name = item.implementation_id.partition(":")
            self.assertEqual(separator, ":")
            module = importlib.import_module(module_name)
            implementation = getattr(module, class_name)
            self.assertEqual(implementation.name, item.id)

    def test_feature_inventory_is_the_only_recipe_vocabulary(self):
        self.assertEqual(
            {item.id for item in self.registry.features},
            {
                "output-qargmax",
                "input-specialization",
                "input-hoisting",
                "declared-input-pruning",
                "static-qdq-compute-migration",
                "quantized-bias-folding",
                "static-qdq-qbatch-matmul-migration",
                "static-qdq-groupnorm-silu-migration",
                "defer-static-qdq-layout",
                "float-attention-fusion",
                "quantized-attention-fusion",
                "silu-fusion",
                "exact-common-subexpression",
                "fp32-pre-ptq",
                "ptq-authoring",
            },
        )

    def test_modes_and_conditional_group_overlays_are_composable(self):
        recipes = {item.id: item for item in self.registry.pipelines}
        self.assertEqual(
            set(recipes),
            {"runtime-package", "runtime-fp32-pre-ptq", "runtime-ptq-authoring"},
        )
        self.assertTrue(all(item.version == 1 for item in recipes.values()))
        self.assertEqual(
            recipes["runtime-package"].allowed_semantics,
            (
                "REWRITE_SEMANTICS_EXACT",
                "REWRITE_SEMANTICS_NUMERICAL_MIGRATION",
            ),
        )
        self.assertEqual(recipes["runtime-package"].required_features, ())
        self.assertEqual(
            set(recipes["runtime-package"].supported_features),
            {
                "output-qargmax",
                "input-specialization",
                "input-hoisting",
                "declared-input-pruning",
                "static-qdq-compute-migration",
                "quantized-bias-folding",
                "static-qdq-qbatch-matmul-migration",
                "static-qdq-groupnorm-silu-migration",
                "defer-static-qdq-layout",
                "float-attention-fusion",
                "quantized-attention-fusion",
                "silu-fusion",
                "exact-common-subexpression",
            },
        )
        self.assertTrue(recipes["runtime-package"].allow_public_abi_change)

        safe_passes = _active_pass_ids(self.registry, "runtime-package")
        self.assertIn("runtime-constant-folding", safe_passes)
        self.assertNotIn("runtime-silu-fusion", safe_passes)
        self.assertNotIn("runtime-static-qdq-compute-fusion", safe_passes)
        self.assertNotIn(
            "runtime-static-qdq-qbatch-matmul-fusion", safe_passes,
        )
        self.assertNotIn(
            "runtime-static-qdq-groupnorm-silu-fusion", safe_passes,
        )
        self.assertIn("runtime-quantized-attention-layout", safe_passes)
        self.assertIn("runtime-attention-layout", safe_passes)
        self.assertIn("runtime-keep-mask", safe_passes)

        silu_only = _active_pass_ids(
            self.registry, "runtime-package", ("silu-fusion",),
        )
        self.assertIn("runtime-silu-fusion", silu_only)
        self.assertNotIn("runtime-bias-folding", silu_only)
        self.assertNotIn("runtime-grouped-projection-split", silu_only)
        self.assertNotIn("runtime-sequence-layout", silu_only)

        static_deferred = _active_pass_ids(
            self.registry,
            "runtime-package",
            ("static-qdq-compute-migration", "defer-static-qdq-layout"),
        )
        self.assertIn("runtime-static-qdq-compute-fusion", static_deferred)
        self.assertNotIn("runtime-qdq-movement", static_deferred)
        self.assertNotIn("runtime-quantized-attention-layout", static_deferred)

        qbatch_only = _active_pass_ids(
            self.registry,
            "runtime-package",
            ("static-qdq-qbatch-matmul-migration",),
        )
        self.assertIn(
            "runtime-static-qdq-qbatch-matmul-fusion", qbatch_only,
        )
        self.assertNotIn("runtime-static-qdq-compute-fusion", qbatch_only)

        self.assertEqual(
            recipes["runtime-fp32-pre-ptq"].required_features,
            ("fp32-pre-ptq",),
        )
        pre_ptq_passes = _active_pass_ids(
            self.registry,
            "runtime-fp32-pre-ptq",
            ("fp32-pre-ptq", "float-attention-fusion"),
        )
        self.assertIn("runtime-float-attention-fusion", pre_ptq_passes)
        self.assertLess(
            pre_ptq_passes.index("runtime-bias-folding"),
            pre_ptq_passes.index("runtime-grouped-projection-split"),
        )
        self.assertLess(
            pre_ptq_passes.index("runtime-grouped-projection-split"),
            pre_ptq_passes.index("runtime-sequence-layout"),
        )
        self.assertLess(
            pre_ptq_passes.index("runtime-sequence-layout"),
            pre_ptq_passes.index("runtime-canonicalize"),
        )
        self.assertIn("runtime-silu-fusion", pre_ptq_passes)
        grouped = next(
            item for item in self.registry.passes
            if item.id == "runtime-grouped-projection-split"
        )
        self.assertEqual(
            grouped.semantics,
            "REWRITE_SEMANTICS_NUMERICAL_MIGRATION",
        )
        sequence = next(
            item for item in self.registry.passes
            if item.id == "runtime-sequence-layout"
        )
        self.assertEqual(sequence.input_dialects, ("OPTIMIZER_DIALECT_RUNTIME",))
        self.assertEqual(sequence.output_dialect, "OPTIMIZER_DIALECT_RUNTIME")
        self.assertTrue(sequence.repeatable)
        self.assertEqual(sequence.semantics, "REWRITE_SEMANTICS_EXACT")

    def test_ptq_authoring_has_a_unique_calibrated_precision_recipe(self):
        self.assertIn(
            "REWRITE_SEMANTICS_QUANTIZATION_AUTHORING",
            self.registry.enum_numbers["RewriteSemantics"],
        )
        descriptor = next(
            item for item in self.registry.passes
            if item.id == "runtime-ptq-authoring"
        )
        self.assertEqual(
            descriptor.semantics,
            "REWRITE_SEMANTICS_QUANTIZATION_AUTHORING",
        )
        self.assertFalse(descriptor.repeatable)
        self.assertTrue(descriptor.requires_calibration)
        self.assertTrue(descriptor.changes_precision)
        self.assertEqual(
            set(descriptor.matched_operators),
            {
                "OPERATOR_KIND_LINEAR",
                "OPERATOR_KIND_MATMUL",
                "OPERATOR_KIND_CONV_2D",
                "OPERATOR_KIND_EMBEDDING",
                "OPERATOR_KIND_ADD",
                "OPERATOR_KIND_GELU",
                "OPERATOR_KIND_SILU",
                "OPERATOR_KIND_LAYER_NORM",
                "OPERATOR_KIND_GROUP_NORM",
                "OPERATOR_KIND_BATCH_MATMUL",
                "OPERATOR_KIND_SDPA",
                "OPERATOR_KIND_CROSS_SDPA",
            },
        )
        self.assertEqual(
            set(descriptor.emitted_operators),
            {
                "OPERATOR_KIND_Q_LINEAR",
                "OPERATOR_KIND_Q_CONV_2D",
                "OPERATOR_KIND_Q_EMBEDDING",
                "OPERATOR_KIND_Q_ADD",
                "OPERATOR_KIND_EXPAND",
                "OPERATOR_KIND_Q_GELU",
                "OPERATOR_KIND_Q_SILU",
                "OPERATOR_KIND_Q_LAYER_NORM",
                "OPERATOR_KIND_Q_GROUP_NORM",
                "OPERATOR_KIND_Q_BATCH_MATMUL",
                "OPERATOR_KIND_Q_SDPA",
                "OPERATOR_KIND_QUANTIZE_LINEAR",
                "OPERATOR_KIND_DEQUANTIZE_LINEAR",
                "OPERATOR_KIND_CROSS_SDPA",
            },
        )
        self.assertEqual(len(descriptor.target_rules), 1)
        self.assertEqual(
            set(descriptor.target_rules[0].required_operators),
            set(descriptor.emitted_operators),
        )

        recipe = next(
            item for item in self.registry.pipelines
            if item.id == "runtime-ptq-authoring"
        )
        self.assertEqual(recipe.required_features, ("ptq-authoring",))
        self.assertEqual(recipe.supported_features, ())
        self.assertTrue(recipe.allow_calibration)
        self.assertEqual(
            recipe.allowed_semantics,
            (
                "REWRITE_SEMANTICS_EXACT",
                "REWRITE_SEMANTICS_QUANTIZATION_AUTHORING",
            ),
        )
        self.assertEqual(
            _active_group_ids(recipe, ("ptq-authoring",)),
            (
                "runtime-validation",
                "runtime-ptq-authoring-stage",
                "runtime-ptq-post-cleanup-fixed-point",
            ),
        )
        flattened = tuple(
            pass_id
            for group_id in _active_group_ids(recipe, ("ptq-authoring",))
            for pass_id in next(
                group for group in self.registry.groups
                if group.id == group_id
            ).pass_ids
        )
        self.assertEqual(
            flattened,
            (
                "runtime-vocabulary",
                "runtime-ptq-authoring",
                "runtime-canonicalize",
                "redundant-qdq",
                "runtime-dead-code",
            ),
        )
        docs = render_docs(self.registry).decode("utf-8")
        self.assertIn("`runtime-ptq-authoring` v1", docs)
        self.assertIn("`quantization-authoring`", docs)

    def test_quantized_bias_folding_is_in_every_static_qdq_migration_recipe(self):
        descriptor = next(
            item for item in self.registry.passes
            if item.id == "runtime-quantized-bias-folding"
        )
        self.assertEqual(
            descriptor.semantics,
            "REWRITE_SEMANTICS_NUMERICAL_MIGRATION",
        )
        self.assertTrue(descriptor.repeatable)
        self.assertTrue(descriptor.changes_precision)
        self.assertEqual(
            set(descriptor.matched_operators),
            {
                "OPERATOR_KIND_Q_LINEAR",
                "OPERATOR_KIND_Q_MATMUL",
                "OPERATOR_KIND_Q_GEMM",
                "OPERATOR_KIND_DEQUANTIZE_LINEAR",
                "OPERATOR_KIND_ADD",
            },
        )
        self.assertEqual(
            set(descriptor.emitted_operators),
            {
                "OPERATOR_KIND_Q_LINEAR",
                "OPERATOR_KIND_Q_MATMUL",
                "OPERATOR_KIND_Q_GEMM",
                "OPERATOR_KIND_DEQUANTIZE_LINEAR",
            },
        )
        for features in (
            ("static-qdq-compute-migration",),
            ("static-qdq-compute-migration", "defer-static-qdq-layout"),
        ):
            with self.subTest(features=features):
                pass_ids = _active_pass_ids(
                    self.registry, "runtime-package", features,
                )
                self.assertLess(
                    pass_ids.index("runtime-quantized-bias-folding"),
                    pass_ids.index("runtime-packed-qlinear-split"),
                )
                self.assertLess(
                    pass_ids.index("runtime-quantized-bias-folding"),
                    pass_ids.index("runtime-static-qdq-compute-fusion"),
                )

        self.assertNotIn(
            "runtime-quantized-bias-folding",
            _active_pass_ids(self.registry, "runtime-package"),
        )
        self.assertNotIn(
            "runtime-quantized-bias-folding",
            _active_pass_ids(
                self.registry, "runtime-fp32-pre-ptq", ("fp32-pre-ptq",),
            ),
        )
        self.assertNotIn(
            "runtime-quantized-bias-folding",
            _active_pass_ids(
                self.registry, "runtime-ptq-authoring", ("ptq-authoring",),
            ),
        )

    def test_quantized_fusion_target_rule_is_kernel_qualified(self):
        descriptor = next(
            item for item in self.registry.passes
            if item.id == "runtime-static-qdq-compute-fusion"
        )
        self.assertEqual(len(descriptor.target_rules), 1)
        rule = descriptor.target_rules[0]
        self.assertEqual(
            set(rule.required_operators),
            set(descriptor.emitted_operators),
        )
        self.assertEqual(
            set(rule.backends),
            {
                "BACKEND_KIND_CPU_JS",
                "BACKEND_KIND_WASM",
                "BACKEND_KIND_WEBGPU",
                "BACKEND_KIND_NATIVE_CPU",
            },
        )

    def test_generated_python_catalog_is_recursively_read_only(self):
        namespace: dict[str, object] = {}
        exec(render_python(self.registry), namespace)
        self.assertNotIn("ANALYSES", namespace)
        passes = namespace["PASSES"]
        with self.assertRaises(TypeError):
            passes["new-pass"] = {}  # type: ignore[index]
        descriptor = passes["runtime-vocabulary"]  # type: ignore[index]
        for retired in (
            "required_analyses", "preserved_analyses", "invalidated_analyses",
            "legality_predicate_id",
        ):
            self.assertNotIn(retired, descriptor)
        with self.assertRaises(TypeError):
            descriptor["version"] = 2  # type: ignore[index]
        target_rule = passes["runtime-static-qdq-compute-fusion"][  # type: ignore[index]
            "target_rules"
        ][0]
        self.assertNotIn("applicability_predicate_id", target_rule)
        self.assertNotIn("cost_model_id", target_rule)
        recipes = namespace["PIPELINE_RECIPES"]
        self.assertNotIn(
            "applicability_predicate_id",
            recipes["runtime-package"],  # type: ignore[index]
        )
        features = namespace["FEATURES"]
        with self.assertRaises(TypeError):
            features["new-feature"] = "invalid"  # type: ignore[index]

    def test_rejects_duplicate_pass_id(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated = source.replace(
            'id: "runtime-output-qargmax"',
            'id: "runtime-vocabulary"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate optimizer pass id"):
            self._load_mutated(mutated)

    def test_rejects_non_v1_pass_and_pipeline_recipe_versions(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutations = {
            "pass": (
                source.replace(
                    'id: "runtime-vocabulary"\n    version: 1',
                    'id: "runtime-vocabulary"\n    version: 2',
                    1,
                ),
                "optimizer pass version 1",
            ),
            "pipeline": (
                source.replace(
                    'id: "runtime-package"\n    version: 1',
                    'id: "runtime-package"\n    version: 2',
                    1,
                ),
                "optimizer pipeline recipe version 1",
            ),
        }
        for label, (mutated, message) in mutations.items():
            with self.subTest(label=label):
                self.assertNotEqual(mutated, source)
                with self.assertRaisesRegex(ValueError, message):
                    self._load_mutated(mutated)

    def test_rejects_retired_registry_metadata_fields(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutations = {
            "analysis": source.replace(
                "  feature: {",
                '  analysis: { id: "use-def" summary: "retired" }\n  feature: {',
                1,
            ),
            "required_analysis": source.replace(
                '    must_run_before: "runtime-output-qargmax"',
                '    required_analysis: "use-def"\n'
                '    must_run_before: "runtime-output-qargmax"',
                1,
            ),
            "legality_predicate_id": source.replace(
                '    must_run_before: "runtime-output-qargmax"',
                '    legality_predicate_id: "optimizer.retired.v1"\n'
                '    must_run_before: "runtime-output-qargmax"',
                1,
            ),
            "cost_model_id": source.replace(
                "      required_operator: OPERATOR_KIND_Q_ARG_MAX",
                '      cost_model_id: "optimizer.retired-cost"\n'
                "      required_operator: OPERATOR_KIND_Q_ARG_MAX",
                1,
            ),
            "applicability_predicate_id": source.replace(
                "      required_operator: OPERATOR_KIND_Q_ARG_MAX",
                '      applicability_predicate_id: "optimizer.retired-target"\n'
                "      required_operator: OPERATOR_KIND_Q_ARG_MAX",
                1,
            ),
        }
        for field, mutated in mutations.items():
            with self.subTest(field=field):
                self.assertNotEqual(mutated, source)
                with self.assertRaisesRegex(ValueError, field):
                    self._load_mutated(mutated)

    def test_rejects_backend_without_emitted_operator_qualification(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        needle = (
            "backend: BACKEND_KIND_NATIVE_CPU\n"
            "      required_operator: OPERATOR_KIND_Q_ARG_MAX"
        )
        mutated = source.replace(
            needle,
            "backend: BACKEND_KIND_NATIVE_CPU\n"
            "      backend: BACKEND_KIND_WEBNN\n"
            "      required_operator: OPERATOR_KIND_Q_ARG_MAX",
            1,
        )
        self.assertNotEqual(mutated, source)
        with self.assertRaisesRegex(ValueError, "absent from webnn exporter qualification"):
            self._load_mutated(mutated)

    def test_rejects_recipe_that_omits_selected_pass_semantics(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated = source.replace(
            "    allowed_semantics: REWRITE_SEMANTICS_NUMERICAL_MIGRATION\n",
            "",
            1,
        )
        with self.assertRaisesRegex(ValueError, "does not allow semantics"):
            self._load_mutated(mutated)

    def test_rejects_unknown_and_unused_recipe_features(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        unknown = source.replace(
            'supported_feature: "output-qargmax"',
            'supported_feature: "misspelled-feature"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "unknown features"):
            self._load_mutated(unknown)

        unused = source.replace(
            'supported_feature: "quantized-attention-fusion"',
            'supported_feature: "quantized-attention-fusion"\n'
            '    supported_feature: "ptq-authoring"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "no group overlay"):
            self._load_mutated(unused)

        unused_declaration = source.replace(
            '  feature: {\n    id: "output-qargmax"',
            '  feature: {\n    id: "declared-but-unused"\n'
            '    summary: "Mutation used by the generator test."\n'
            '  }\n  feature: {\n    id: "output-qargmax"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "unused by every pipeline"):
            self._load_mutated(unused_declaration)

    def test_symbolically_validates_conditional_overlay_co_selection(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        validation = '    group_overlay: { group_id: "runtime-validation" }\n'
        duplicate = source.replace(validation, validation + validation, 1)
        with self.assertRaisesRegex(ValueError, "select pass .* more than once"):
            self._load_mutated(duplicate)

        exclusive = source.replace(
            validation,
            '''    group_overlay: {
      group_id: "runtime-validation"
      required_feature: "defer-static-qdq-layout"
    }
    group_overlay: {
      group_id: "runtime-validation"
      forbidden_feature: "defer-static-qdq-layout"
    }
''',
            1,
        )
        registry = self._load_mutated(exclusive)
        recipe = next(
            item for item in registry.pipelines if item.id == "runtime-package"
        )
        self.assertEqual(
            _active_group_ids(recipe).count("runtime-validation"),
            1,
        )
        self.assertEqual(
            _active_group_ids(
                recipe, ("defer-static-qdq-layout",)
            ).count("runtime-validation"),
            1,
        )

        exact_prelude = (
            '    group_overlay: { group_id: "runtime-exact-prelude" }\n'
        )
        misordered = (
            source.replace(validation, "__VALIDATION_OVERLAY__\n", 1)
            .replace(exact_prelude, validation, 1)
            .replace("__VALIDATION_OVERLAY__\n", exact_prelude, 1)
        )
        with self.assertRaisesRegex(ValueError, "required predecessor"):
            self._load_mutated(misordered)

    def test_rejects_unreachable_or_ungated_policy_sensitive_overlay(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        unreachable = source.replace(
            'required_feature: "output-qargmax"',
            'required_feature: "output-qargmax"\n'
            '      forbidden_feature: "output-qargmax"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "both requires and forbids"):
            self._load_mutated(unreachable)

        ungated = source.replace(
            'required_feature: "output-qargmax"',
            'forbidden_feature: "output-qargmax"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "policy-sensitive pass"):
            self._load_mutated(ungated)

    def test_rejects_ambiguous_recipe_feature_domains(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        ambiguous = source.replace(
            'supported_feature: "quantized-attention-fusion"',
            'supported_feature: "quantized-attention-fusion"\n'
            '    supported_feature: "fp32-pre-ptq"',
            1,
        ).replace(
            'group_id: "runtime-safe-fixed-point"\n'
            '      forbidden_feature: "static-qdq-compute-migration"',
            'group_id: "runtime-safe-fixed-point"\n'
            '      forbidden_feature: "fp32-pre-ptq"\n'
            '      forbidden_feature: "static-qdq-compute-migration"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "ambiguously match"):
            self._load_mutated(ambiguous)

    def test_generated_files_are_current_and_protoc_accepts_schema(self):
        command = [
            sys.executable,
            str(ROOT / "tools/generate_optimizer_registry.py"),
            "--check",
        ]
        if shutil.which("protoc"):
            command.append("--protoc-check")
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def _load_mutated(self, source: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer_registry.proto"
            path.write_text(source, encoding="utf-8")
            return load_registry(path, PUBLIC_PROTO, KERNEL_REGISTRY_PROTO)


if __name__ == "__main__":
    unittest.main()
