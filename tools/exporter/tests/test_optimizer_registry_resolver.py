from __future__ import annotations

import json
import unittest

from tools.exporter.generated.kernel_registry import PROFILE_MEMBERS
from tools.exporter.generated.optimizer_registry import (
    KERNEL_REGISTRY_SHA256,
    PASSES,
    PASS_GROUPS,
    PIPELINE_RECIPES,
    REGISTRY_SHA256,
    SCHEMA_VERSION,
)
from tools.exporter.errors import ExporterError
from tools.exporter.optimizer.candidate import RewritePolicy
from tools.exporter.optimizer.registry_resolver import (
    PASS_FACTORIES,
    OptimizerRegistryError,
    RegistryPipelineRequest,
    RuntimePassFactoryContext,
    resolve_runtime_pipeline,
    validate_factory_coverage,
)
from tools.exporter.optimizer.target import TargetEnvironment
from tools.exporter.optimizer.typed_attention import (
    RuntimeAttentionLayoutPass,
    RuntimeFloatAttentionFusionPass,
    RuntimeKeepMaskPass,
)
from tools.exporter.optimizer.typed_passes import OutputArgMaxSpecialization
from tools.exporter.optimizer.typed_quantized_attention import (
    RuntimeQuantizedAttentionFusionPass,
    RuntimeQuantizedAttentionLayoutPass,
)
from tools.exporter.optimizer.typed_quantized_bias_folding import (
    RuntimeQuantizedBiasFoldingPass,
)
from tools.exporter.optimizer.typed_groupnorm_silu_island import (
    RuntimeStaticQDQGroupNormSiLUFusionPass,
)
from tools.exporter.optimizer.typed_specialization import InputHoistingSpec
from tools.exporter.optimizer.typed_pipeline import (
    default_runtime_pipeline,
    optimize_runtime_package,
    serialize_pipeline_report,
)
from tools.exporter.runtime_ir import import_runtime_package


def _target(
    profile: str = "portable",
    *,
    compile_backend: str = "cpu-js",
    tune_backend: str = "wasm",
) -> TargetEnvironment:
    return TargetEnvironment(
        backend_profile=profile,
        compile_backend=compile_backend,
        tune_backend=tune_backend,
    )


def _request(
    *,
    features=(),
    target: TargetEnvironment | None = None,
    policy: RewritePolicy | None = None,
    allow_calibration: bool = False,
    output_argmax=(),
) -> RegistryPipelineRequest:
    return RegistryPipelineRequest(
        factory_context=RuntimePassFactoryContext(
            tensor_data={},
            output_argmax=tuple(output_argmax),
        ),
        target=_target() if target is None else target,
        rewrite_policy=(RewritePolicy.exact() if policy is None else policy),
        allow_calibration=allow_calibration,
        selection_features=frozenset(features),
    )


def _active_group_ids(recipe_id: str, features=()) -> tuple[str, ...]:
    selected = frozenset(features)
    return tuple(
        overlay["group"]
        for overlay in PIPELINE_RECIPES[recipe_id]["group_overlays"]
        if (
            frozenset(overlay["required_features"]) <= selected
            and not frozenset(overlay["forbidden_features"]) & selected
        )
    )


def _flatten(recipe_id: str, features=()) -> tuple[str, ...]:
    return tuple(
        pass_id
        for group_id in _active_group_ids(recipe_id, features)
        for pass_id in PASS_GROUPS[group_id]["passes"]
    )


def _identity_package() -> dict:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {"out": {
                "tensor": "y", "shape": [1], "dtype": "float32",
            }},
            "params": {},
        }],
    }


def _transpose_elementwise_package(operator: str) -> dict:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 2, 3], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [
            {
                "id": "to-moved",
                "opType": "Transpose",
                "inputs": {"input": "x"},
                "outputs": {"out": {
                    "tensor": "moved", "shape": [1, 3, 2],
                    "dtype": "float32",
                }},
                "params": {"perm": [0, 2, 1]},
            },
            {
                "id": "elementwise",
                "opType": operator,
                "inputs": {"input": "moved"},
                "outputs": {"out": {
                    "tensor": "activated", "shape": [1, 3, 2],
                    "dtype": "float32",
                }},
                "params": {},
            },
            {
                "id": "to-source",
                "opType": "Transpose",
                "inputs": {"input": "activated"},
                "outputs": {"out": {
                    "tensor": "y", "shape": [1, 2, 3],
                    "dtype": "float32",
                }},
                "params": {"perm": [0, 2, 1]},
            },
        ],
    }


def _relu_chain_package(count: int, *, stale_package_class: bool = False) -> dict:
    document = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1], "dtype": "float32"}},
        "outputs": [f"value_{count}"],
        "nodes": [
            {
                "id": f"relu_{index}",
                "opType": "ReLU",
                "inputs": {
                    "input": "x" if index == 0 else f"value_{index}"
                },
                "outputs": {"out": {
                    "tensor": f"value_{index + 1}",
                    "shape": [1],
                    "dtype": "float32",
                }},
                "params": {},
            }
            for index in range(count)
        ],
    }
    if stale_package_class:
        document["source"] = {"package_class": "w8a8-v1"}
    return document


class OptimizerRegistryResolverTests(unittest.TestCase):
    def test_factory_inventory_is_exact_and_fail_closed(self):
        validate_factory_coverage()
        expected = {
            descriptor["implementation_id"] for descriptor in PASSES.values()
        }
        self.assertEqual(set(PASS_FACTORIES), expected)

        implementation_id = next(iter(PASS_FACTORIES))
        missing = dict(PASS_FACTORIES)
        missing.pop(implementation_id)
        with self.assertRaisesRegex(
            OptimizerRegistryError, "coverage mismatch.*missing",
        ):
            validate_factory_coverage(missing)

        extra = dict(PASS_FACTORIES)
        extra["invalid.extra:Pass"] = PASS_FACTORIES[implementation_id]
        with self.assertRaisesRegex(
            OptimizerRegistryError, "coverage mismatch.*extra",
        ):
            validate_factory_coverage(extra)

    def test_attention_passes_are_registry_owned_and_factory_bound(self):
        expected = {
            "runtime-float-attention-fusion": (
                "tools.exporter.optimizer.typed_attention:RuntimeFloatAttentionFusionPass",
                RuntimeFloatAttentionFusionPass,
                "numerical-migration",
            ),
            "runtime-quantized-attention-fusion": (
                "tools.exporter.optimizer.typed_quantized_attention:RuntimeQuantizedAttentionFusionPass",
                RuntimeQuantizedAttentionFusionPass,
                "numerical-migration",
            ),
            "runtime-quantized-attention-layout": (
                "tools.exporter.optimizer.typed_quantized_attention:RuntimeQuantizedAttentionLayoutPass",
                RuntimeQuantizedAttentionLayoutPass,
                "exact",
            ),
            "runtime-attention-layout": (
                "tools.exporter.optimizer.typed_attention:RuntimeAttentionLayoutPass",
                RuntimeAttentionLayoutPass,
                "exact",
            ),
            "runtime-keep-mask": (
                "tools.exporter.optimizer.typed_attention:RuntimeKeepMaskPass",
                RuntimeKeepMaskPass,
                "exact",
            ),
        }
        tensor_data = {}
        context = RuntimePassFactoryContext(tensor_data=tensor_data)
        for pass_id, (implementation_id, pass_type, semantics) in expected.items():
            with self.subTest(pass_id=pass_id):
                descriptor = PASSES[pass_id]
                self.assertEqual(descriptor["implementation_id"], implementation_id)
                self.assertEqual(descriptor["semantics"], semantics)
                instance = PASS_FACTORIES[implementation_id].build(context)
                self.assertIsInstance(instance, pass_type)
                if hasattr(instance, "tensor_data"):
                    self.assertIs(instance.tensor_data, tensor_data)

        self.assertEqual(
            PASS_GROUPS["runtime-fp32-attention-migration-prelude"]["passes"],
            ("runtime-float-attention-fusion",),
        )
        self.assertEqual(
            PASS_GROUPS["runtime-quantized-attention-migration-prelude"]["passes"],
            ("runtime-quantized-attention-fusion",),
        )
        self.assertEqual(
            PASS_GROUPS["runtime-quantized-attention-layout-prelude"]["passes"],
            ("runtime-quantized-attention-layout",),
        )
        self.assertEqual(
            PASS_GROUPS["runtime-attention-exact-fixed-point"]["passes"],
            ("runtime-attention-layout", "runtime-keep-mask"),
        )
        self.assertIn(
            "runtime-static-qdq-compute-fusion",
            PASSES["runtime-quantized-attention-fusion"]["must_run_before"],
        )

        pipeline = default_runtime_pipeline(
            tensor_data=tensor_data,
            allow_quantized_attention_numerical_migration=True,
            allow_static_qdq_compute_numerical_migration=True,
        )
        pass_ids = pipeline.metadata.pass_ids
        self.assertLess(
            pass_ids.index("runtime-quantized-attention-fusion"),
            pass_ids.index("runtime-static-qdq-compute-fusion"),
        )
        self.assertLess(
            pass_ids.index("runtime-quantized-attention-fusion"),
            pass_ids.index("runtime-quantized-attention-layout"),
        )

    def test_factory_validation_rejects_name_and_contract_drift(self):
        renamed = dict(PASSES)
        descriptor = dict(renamed.pop("runtime-vocabulary"))
        renamed["runtime-vocabulary-renamed"] = descriptor
        with self.assertRaisesRegex(
            OptimizerRegistryError, "pass name.*does not match registry id",
        ):
            validate_factory_coverage(pass_catalog=renamed)

        contract_drift = dict(PASSES)
        descriptor = dict(contract_drift["runtime-vocabulary"])
        descriptor["repeatable"] = True
        contract_drift["runtime-vocabulary"] = descriptor
        with self.assertRaisesRegex(
            OptimizerRegistryError, "PassContract does not match",
        ):
            validate_factory_coverage(pass_catalog=contract_drift)

    def test_input_abi_factories_require_matching_feature_context(self):
        empty = RuntimePassFactoryContext(tensor_data={})
        for implementation_id in (
            "tools.exporter.optimizer.typed_specialization:RuntimeInputSpecializationPass",
            "tools.exporter.optimizer.typed_specialization:RuntimeInputHoistingPass",
        ):
            with self.subTest(implementation_id=implementation_id):
                with self.assertRaisesRegex(
                    OptimizerRegistryError, "requires explicit",
                ):
                    PASS_FACTORIES[implementation_id].build(empty)

        context = RuntimePassFactoryContext(
            tensor_data={},
            input_specializations={"tokens": [1, 2, 3]},
        )
        request = RegistryPipelineRequest(
            factory_context=context,
            target=_target(),
            rewrite_policy=RewritePolicy.exact(allow_abi_change=True),
        )
        with self.assertRaisesRegex(
            OptimizerRegistryError, "input-specialization selection feature",
        ):
            resolve_runtime_pipeline(request)

        selected = RegistryPipelineRequest(
            factory_context=context,
            target=_target(),
            rewrite_policy=RewritePolicy.exact(allow_abi_change=True),
            selection_features=frozenset({"input-specialization"}),
        )
        pipeline = resolve_runtime_pipeline(selected)
        self.assertEqual(pipeline.metadata.recipe_id, "runtime-package")
        self.assertIn("runtime-input-specialization", pipeline.metadata.pass_ids)

        combined = default_runtime_pipeline(
            tensor_data={},
            input_specializations={"tokens": [1, 2, 3]},
            input_hoistings=(InputHoistingSpec(
                "encoder_hidden", "float32", tensor_name="encoder_hidden",
            ),),
        )
        self.assertLess(
            combined.metadata.pass_ids.index("runtime-input-specialization"),
            combined.metadata.pass_ids.index("runtime-input-hoisting"),
        )
        self.assertLess(
            combined.metadata.pass_ids.index("runtime-constant-folding"),
            combined.metadata.pass_ids.index("runtime-input-hoisting"),
        )
        self.assertEqual(
            combined.metadata.selection_features,
            ("input-hoisting", "input-specialization"),
        )

    def test_generated_recipe_owns_order_groups_and_fixed_points(self):
        pipeline = default_runtime_pipeline(tensor_data={})
        recipe_id = "runtime-package"
        groups = _active_group_ids(recipe_id)

        self.assertEqual(
            tuple(item.name for group in pipeline.groups for item in group.passes),
            _flatten(recipe_id),
        )
        self.assertEqual(
            tuple(group.fixed_point for group in pipeline.groups),
            tuple(
                PASS_GROUPS[group_id]["fixed_point"]
                for group_id in groups
            ),
        )
        self.assertEqual(
            tuple(group.max_iterations for group in pipeline.groups),
            tuple(
                PASS_GROUPS[group_id]["max_iterations"]
                for group_id in groups
            ),
        )
        self.assertNotIn("runtime-bias-folding", _flatten(recipe_id))
        self.assertEqual(pipeline.metadata.recipe_id, recipe_id)

    def test_pre_ptq_factories_share_explicit_tensor_context(self):
        tensors = {}
        pipeline = default_runtime_pipeline(
            tensor_data=tensors,
            enable_fp32_pre_ptq_optimization=True,
        )
        instances = {
            item.name: item
            for group in pipeline.groups
            for item in group.passes
        }
        self.assertIs(instances["runtime-bias-folding"].tensor_data, tensors)
        self.assertIs(
            instances["runtime-grouped-projection-split"].tensor_data,
            tensors,
        )
        self.assertLess(
            pipeline.metadata.pass_ids.index("runtime-bias-folding"),
            pipeline.metadata.pass_ids.index("runtime-grouped-projection-split"),
        )
        self.assertLess(
            pipeline.metadata.pass_ids.index("runtime-grouped-projection-split"),
            pipeline.metadata.pass_ids.index("runtime-sequence-layout"),
        )
        self.assertLess(
            pipeline.metadata.pass_ids.index("runtime-sequence-layout"),
            pipeline.metadata.pass_ids.index("runtime-canonicalize"),
        )
        self.assertNotIn(
            "runtime-sequence-layout",
            _flatten("runtime-package"),
        )

    def test_silu_migration_is_a_narrow_explicit_runtime_package_feature(self):
        default = default_runtime_pipeline(tensor_data={})
        selected = default_runtime_pipeline(
            tensor_data={},
            allow_silu_numerical_migration=True,
        )

        self.assertNotIn("runtime-silu-fusion", default.metadata.pass_ids)
        self.assertEqual(selected.metadata.recipe_id, "runtime-package")
        self.assertEqual(selected.metadata.selection_features, ("silu-fusion",))
        self.assertIn("runtime-silu-migration-prelude", selected.metadata.group_ids)
        self.assertEqual(selected.metadata.pass_ids.count("runtime-silu-fusion"), 1)
        for unrelated in (
            "runtime-bias-folding",
            "runtime-grouped-projection-split",
            "runtime-sequence-layout",
        ):
            self.assertNotIn(unrelated, selected.metadata.pass_ids)

        broader = default_runtime_pipeline(
            tensor_data={},
            allow_silu_numerical_migration=True,
            enable_fp32_pre_ptq_optimization=True,
        )
        self.assertEqual(broader.metadata.recipe_id, "runtime-fp32-pre-ptq")
        self.assertEqual(broader.metadata.pass_ids.count("runtime-silu-fusion"), 1)

    def test_quantized_bias_factory_has_independent_narrow_opt_in(self):
        tensors = {}
        pipeline = default_runtime_pipeline(
            tensor_data=tensors,
            allow_static_qdq_compute_numerical_migration=True,
        )
        instances = {
            item.name: item
            for group in pipeline.groups
            for item in group.passes
        }
        folding = instances["runtime-quantized-bias-folding"]
        self.assertIsInstance(folding, RuntimeQuantizedBiasFoldingPass)
        self.assertIs(folding.tensor_data, tensors)
        pass_ids = pipeline.metadata.pass_ids
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
            _flatten("runtime-package"),
        )
        self.assertNotIn(
            "runtime-quantized-bias-folding",
            _flatten("runtime-ptq-authoring", ("ptq-authoring",)),
        )

        narrow = default_runtime_pipeline(
            tensor_data=tensors,
            allow_quantized_bias_folding_numerical_migration=True,
        )
        self.assertEqual(
            narrow.metadata.selection_features,
            ("quantized-bias-folding",),
        )
        self.assertIn(
            "runtime-quantized-bias-migration-prelude",
            narrow.metadata.group_ids,
        )
        self.assertNotIn(
            "runtime-quantized-migration-prelude",
            narrow.metadata.group_ids,
        )
        self.assertEqual(
            narrow.metadata.pass_ids.count("runtime-quantized-bias-folding"),
            1,
        )
        for unrelated in (
            "runtime-static-qdq-compute-fusion",
            "runtime-bias-folding",
            "runtime-silu-fusion",
            "runtime-grouped-projection-split",
            "runtime-sequence-layout",
        ):
            self.assertNotIn(unrelated, narrow.metadata.pass_ids)

        combined = default_runtime_pipeline(
            tensor_data=tensors,
            allow_static_qdq_compute_numerical_migration=True,
            allow_quantized_bias_folding_numerical_migration=True,
        )
        self.assertEqual(
            combined.metadata.pass_ids.count("runtime-quantized-bias-folding"),
            1,
        )
        self.assertIn(
            "runtime-quantized-migration-prelude",
            combined.metadata.group_ids,
        )
        self.assertNotIn(
            "runtime-quantized-bias-migration-prelude",
            combined.metadata.group_ids,
        )

    def test_groupnorm_silu_factory_has_independent_narrow_opt_in(self):
        tensors = {}
        default = default_runtime_pipeline(tensor_data=tensors)
        selected = default_runtime_pipeline(
            tensor_data=tensors,
            allow_static_qdq_groupnorm_silu_numerical_migration=True,
        )

        self.assertNotIn(
            "runtime-static-qdq-groupnorm-silu-fusion",
            default.metadata.pass_ids,
        )
        self.assertEqual(
            selected.metadata.selection_features,
            ("static-qdq-groupnorm-silu-migration",),
        )
        self.assertIn(
            "runtime-static-qdq-groupnorm-silu-migration",
            selected.metadata.group_ids,
        )
        instances = {
            item.name: item
            for group in selected.groups
            for item in group.passes
        }
        fusion = instances["runtime-static-qdq-groupnorm-silu-fusion"]
        self.assertIsInstance(
            fusion, RuntimeStaticQDQGroupNormSiLUFusionPass,
        )
        self.assertIs(fusion.tensor_data, tensors)
        for unrelated in (
            "runtime-static-qdq-compute-fusion",
            "runtime-static-qdq-qbatch-matmul-fusion",
            "runtime-quantized-bias-folding",
        ):
            self.assertNotIn(unrelated, selected.metadata.pass_ids)

    def test_independent_features_compose_without_cartesian_recipes(self):
        specialization = OutputArgMaxSpecialization("logits", "token")
        for mask in range(8):
            features = set()
            arguments = {}
            if mask & 1:
                features.add("output-qargmax")
                arguments["output_argmax"] = (specialization,)
            if mask & 2:
                features.add("static-qdq-compute-migration")
                arguments["allow_static_qdq_compute_numerical_migration"] = True
            if mask & 4:
                features.add("defer-static-qdq-layout")
                arguments["enable_static_qdq_layout_optimization"] = False
            with self.subTest(features=sorted(features)):
                pipeline = default_runtime_pipeline(tensor_data={}, **arguments)
                self.assertIsNotNone(pipeline.metadata)
                self.assertEqual(pipeline.metadata.recipe_id, "runtime-package")
                self.assertEqual(
                    pipeline.metadata.group_ids,
                    _active_group_ids("runtime-package", features),
                )
                self.assertEqual(
                    pipeline.metadata.pass_ids,
                    _flatten("runtime-package", features),
                )
                self.assertEqual(
                    pipeline.metadata.selection_features,
                    tuple(sorted(features)),
                )

        fp32 = default_runtime_pipeline(
            tensor_data={},
            enable_fp32_pre_ptq_optimization=True,
            allow_float_attention_numerical_migration=True,
        )
        self.assertEqual(fp32.metadata.recipe_id, "runtime-fp32-pre-ptq")
        self.assertEqual(fp32.metadata.recipe_version, 1)
        self.assertIn("runtime-float-attention-fusion", fp32.metadata.pass_ids)

    def test_feature_domain_and_inactive_overlays_fail_closed(self):
        with self.assertRaisesRegex(
            OptimizerRegistryError, "unknown optimizer selection features",
        ):
            resolve_runtime_pipeline(_request(features=("misspelled",)))

        with self.assertRaisesRegex(
            OptimizerRegistryError, "resolve exactly one recipe",
        ):
            resolve_runtime_pipeline(_request(
                features=("fp32-pre-ptq", "static-qdq-compute-migration"),
                policy=RewritePolicy.qualified(),
            ))

        with self.assertRaisesRegex(
            OptimizerRegistryError, "requires features.*supports only",
        ):
            resolve_runtime_pipeline(RegistryPipelineRequest(
                factory_context=RuntimePassFactoryContext(tensor_data={}),
                target=_target(),
                recipe_id="runtime-ptq-authoring",
            ))

        pass_catalog = dict(PASSES)
        restricted = dict(pass_catalog["runtime-output-qargmax"])
        restricted["emitted_operators"] = (
            *restricted["emitted_operators"],
            "GatherElements",
        )
        pass_catalog["runtime-output-qargmax"] = restricted
        safe = resolve_runtime_pipeline(_request(), pass_catalog=pass_catalog)
        self.assertEqual(safe.metadata.recipe_id, "runtime-package")

        specialization = OutputArgMaxSpecialization("logits", "token")
        with self.assertRaisesRegex(
            OptimizerRegistryError, "emits.*profile member 'native-cpu'",
        ):
            resolve_runtime_pipeline(_request(
                features=("output-qargmax",),
                output_argmax=(specialization,),
                policy=RewritePolicy.exact(allow_abi_change=True),
            ), pass_catalog=pass_catalog)

    def test_portable_legality_uses_every_profile_member(self):
        pass_catalog = dict(PASSES)
        descriptor = dict(pass_catalog["runtime-shape-chain"])
        descriptor["emitted_operators"] = (
            *descriptor["emitted_operators"],
            "GatherElements",
        )
        pass_catalog["runtime-shape-chain"] = descriptor

        request = _request(target=_target(
            "portable",
            compile_backend="cpu-js",
            tune_backend="wasm",
        ))
        with self.assertRaisesRegex(
            OptimizerRegistryError, "profile member 'native-cpu'",
        ):
            resolve_runtime_pipeline(request, pass_catalog=pass_catalog)

        browser = resolve_runtime_pipeline(
            _request(target=_target(
                "browser",
                compile_backend="native-cpu",
                tune_backend="native-cpu",
            )),
            pass_catalog=pass_catalog,
        )
        self.assertEqual(
            browser.metadata.backend_profile_members,
            tuple(PROFILE_MEMBERS["browser"]),
        )

    def test_kind_preserving_pass_checks_actual_not_theoretical_operators(self):
        optimized, _, report = optimize_runtime_package(
            _transpose_elementwise_package("ReLU"),
            {},
            target_environment=_target("portable"),
            shape_profile={},
        )
        moved = [
            run for run in report.runs
            if run.name == "runtime-elementwise-transpose"
        ]
        self.assertTrue(any(run.changes for run in moved))
        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["ReLU", "Identity"],
        )

        unsupported = import_runtime_package(
            _transpose_elementwise_package("ReLU"),
            {},
        )
        unsupported.nodes[1].op_type = "Sin"
        with self.assertRaisesRegex(
            ExporterError,
            r"Sin at elementwise is not admitted by target cpu-js",
        ):
            default_runtime_pipeline(
                tensor_data={},
                target_environment=_target("portable"),
            ).run(unsupported)

    def test_graph_legality_validates_exact_descriptors_profile_wide(self):
        oversized = import_runtime_package(_relu_chain_package(1025), {})
        with self.assertRaises(ExporterError) as caught:
            default_runtime_pipeline(
                tensor_data={},
                target_environment=_target(
                    "portable",
                    compile_backend="wasm",
                    tune_backend="cpu-js",
                ),
            ).run(oversized)
        self.assertEqual(caught.exception.diagnostic.code, "VXCAP_NATIVE_NODES")
        self.assertEqual(caught.exception.diagnostic.target, "native-cpu")

        with self.assertRaises(ExporterError) as stale:
            import_runtime_package(
                _relu_chain_package(1, stale_package_class=True),
                {},
            )
        self.assertEqual(stale.exception.diagnostic.code, "VXRTIR038")

    def test_compile_and_tune_backends_are_validated_but_do_not_select_passes(self):
        with self.assertRaisesRegex(
            OptimizerRegistryError, "unknown compile backend 'mystery'",
        ):
            resolve_runtime_pipeline(_request(target=_target(
                compile_backend="mystery",
                tune_backend="wasm",
            )))
        with self.assertRaisesRegex(
            OptimizerRegistryError, "unknown tune backend 'mystery'",
        ):
            resolve_runtime_pipeline(_request(target=_target(
                compile_backend="native-cpu",
                tune_backend="mystery",
            )))

        wasm_compile = _target(
            "portable",
            compile_backend="wasm",
            tune_backend="native-cpu",
        )
        native_compile = _target(
            "portable",
            compile_backend="native-cpu",
            tune_backend="wasm",
        )
        first_graph, first_tensors, first_report = optimize_runtime_package(
            _identity_package(), {}, target_environment=wasm_compile,
        )
        second_graph, second_tensors, second_report = optimize_runtime_package(
            _identity_package(), {}, target_environment=native_compile,
        )
        self.assertEqual(first_graph, second_graph)
        self.assertEqual(first_tensors, second_tensors)
        self.assertEqual(first_report.metadata.pass_ids, second_report.metadata.pass_ids)
        self.assertEqual(
            first_report.metadata.backend_profile_members,
            second_report.metadata.backend_profile_members,
        )

    def test_required_kernel_variants_are_checked_for_every_profile_member(self):
        pass_catalog = dict(PASSES)
        descriptor = dict(pass_catalog["runtime-shape-chain"])
        descriptor["target_rules"] = ({
            "backends": ("cpu-js", "wasm", "webgpu", "native-cpu"),
            "required_operators": ("QLinear",),
            "required_kernel_variant_ids": (
                "cpu-js.qlinear.reference",
                "wasm.qlinear.baseline",
                "webgpu.qlinear.scalar",
            ),
        },)
        pass_catalog["runtime-shape-chain"] = descriptor

        with self.assertRaisesRegex(
            OptimizerRegistryError,
            "no required kernel variant.*profile member 'native-cpu'",
        ):
            resolve_runtime_pipeline(_request(), pass_catalog=pass_catalog)

    def test_semantic_abi_and_calibration_policies_are_explicit(self):
        with self.assertRaisesRegex(
            OptimizerRegistryError, "rewrite policy rejects pass",
        ):
            resolve_runtime_pipeline(_request(
                features=("static-qdq-compute-migration",),
            ))

        with self.assertRaisesRegex(
            OptimizerRegistryError, "rewrite policy rejects pass",
        ):
            resolve_runtime_pipeline(_request(
                features=("quantized-bias-folding",),
            ))

        specialization = OutputArgMaxSpecialization("logits", "token")
        with self.assertRaisesRegex(
            OptimizerRegistryError, "rewrite policy rejects pass",
        ):
            resolve_runtime_pipeline(_request(
                features=("output-qargmax",),
                output_argmax=(specialization,),
            ))

        recipe_catalog = dict(PIPELINE_RECIPES)
        output_recipe = dict(recipe_catalog["runtime-package"])
        output_recipe["allow_public_abi_change"] = False
        recipe_catalog["runtime-package"] = output_recipe
        with self.assertRaisesRegex(
            OptimizerRegistryError, "does not authorize ABI-changing pass",
        ):
            resolve_runtime_pipeline(
                _request(
                    features=("output-qargmax",),
                    output_argmax=(specialization,),
                    policy=RewritePolicy.exact(allow_abi_change=True),
                ),
                recipe_catalog=recipe_catalog,
            )

        recipe_catalog = dict(PIPELINE_RECIPES)
        exact_only = dict(recipe_catalog["runtime-package"])
        exact_only["allowed_semantics"] = ("exact",)
        recipe_catalog["runtime-package"] = exact_only
        with self.assertRaisesRegex(
            OptimizerRegistryError, "does not authorize pass.*semantics",
        ):
            resolve_runtime_pipeline(
                _request(
                    features=("static-qdq-compute-migration",),
                    policy=RewritePolicy.qualified(),
                ),
                recipe_catalog=recipe_catalog,
            )

        pass_catalog = dict(PASSES)
        descriptor = dict(pass_catalog["runtime-shape-chain"])
        descriptor["requires_calibration"] = True
        pass_catalog["runtime-shape-chain"] = descriptor
        with self.assertRaisesRegex(
            OptimizerRegistryError, "requires explicit calibration policy",
        ):
            resolve_runtime_pipeline(
                _request(),
                pass_catalog=pass_catalog,
            )
        with self.assertRaisesRegex(
            OptimizerRegistryError, "recipe.*does not authorize calibrated pass",
        ):
            resolve_runtime_pipeline(
                _request(allow_calibration=True),
                pass_catalog=pass_catalog,
            )

    def test_report_serializes_registry_recipe_and_target_deterministically(self):
        target = _target(
            "portable",
            compile_backend="wasm",
            tune_backend="native-cpu",
        )
        _, _, report = optimize_runtime_package(
            _identity_package(),
            {},
            target_environment=target,
        )
        first = serialize_pipeline_report(report)
        second = serialize_pipeline_report(report)

        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )
        metadata = first["pipeline"]
        self.assertEqual(metadata["registry"], {
            "schema_version": SCHEMA_VERSION,
            "sha256": REGISTRY_SHA256,
            "kernel_registry_sha256": KERNEL_REGISTRY_SHA256,
        })
        self.assertEqual(metadata["recipe"]["id"], "runtime-package")
        self.assertEqual(
            metadata["recipe"]["passes"],
            list(_flatten("runtime-package")),
        )
        self.assertEqual(metadata["recipe"]["required_features"], [])
        self.assertEqual(
            set(metadata["recipe"]["supported_features"]),
            set(PIPELINE_RECIPES["runtime-package"]["supported_features"]),
        )
        self.assertEqual(metadata["backend_profile"], {
            "id": "portable",
            "members": list(PROFILE_MEMBERS["portable"]),
        })
        self.assertEqual(metadata["compile_backend"], "wasm")
        self.assertEqual(metadata["tune_backend"], "native-cpu")


if __name__ == "__main__":
    unittest.main()
