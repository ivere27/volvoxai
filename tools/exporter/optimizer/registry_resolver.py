"""Resolve generated optimizer recipes into verified typed pass pipelines.

The protobuf-generated catalog owns pass identity, ordering, effects, and
target requirements.  This module owns only executable constructor bindings
and fail-closed resolution against one explicit caller context.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, MutableMapping, Optional

from ..generated.kernel_registry import (
    ATOMIC_TARGETS,
    KERNEL_VARIANTS,
    OPS_BY_TARGET,
    PROFILE_MEMBERS,
    RUNTIME_OPERATORS_BY_BACKEND,
)
from ..generated.optimizer_registry import (
    FEATURES,
    KERNEL_REGISTRY_SHA256,
    PASSES,
    PASS_GROUPS,
    PIPELINE_RECIPES,
    REGISTRY_SHA256,
    SCHEMA_VERSION,
)
from ..capabilities import validate_graph
from ..errors import ExporterError
from ..ir import GraphIR
from ..pipeline import (
    ConcretePassPolicy,
    IRPass,
    PassGroup,
    PipelineMetadata,
    VerifiedPipeline,
)
from ..typed_ptq import CalibrationProfile, PTQConfig
from .candidate import RewritePolicy, RewriteSemantics
from .static_qdq_fusion import (
    RuntimeStaticQDQBatchMatMulFusionPass,
    RuntimeStaticQDQComputeFusionPass,
)
from .target import TargetEnvironment
from .typed_affine_canonicalization import (
    RuntimeAffineReferenceCanonicalizationPass,
)
from .typed_attention import (
    RuntimeAttentionLayoutPass,
    RuntimeFloatAttentionFusionPass,
    RuntimeKeepMaskPass,
)
from .typed_bias_folding import RuntimeBiasFoldingPass
from .typed_constant_folding import RuntimeConstantFoldingPass
from .typed_common_subexpression import (
    RuntimeCommonSubexpressionEliminationPass,
)
from .typed_elementwise_transpose import RuntimeElementwiseTransposePass
from .typed_grouped_projection import RuntimeGroupedProjectionSplitPass
from .typed_groupnorm_silu_island import (
    RuntimeStaticQDQGroupNormSiLUFusionPass,
)
from .typed_passes import (
    OutputArgMaxSpecialization,
    RedundantQDQPass,
    RuntimeCanonicalizePass,
    RuntimeDeadCodePass,
    RuntimeOutputArgMaxPass,
    RuntimePackedQLinearSplitPass,
    RuntimeShapeChainPass,
    RuntimeVocabularyPass,
)
from .typed_ptq_authoring import RuntimePTQAuthoringPass
from .typed_quantized_bias_folding import RuntimeQuantizedBiasFoldingPass
from .typed_quantized_attention import (
    RuntimeQuantizedAttentionFusionPass,
    RuntimeQuantizedAttentionLayoutPass,
)
from .typed_qdq_layout import (
    RuntimePointwiseTransposeHoistPass,
    RuntimeQDQMovementPass,
    RuntimeQDQTransposeCancellationPass,
)
from .typed_sequence_layout import RuntimeSequenceLayoutPass
from .typed_singleton_transpose import RuntimeSingletonTransposePass
from .typed_silu_fusion import RuntimeSiluFusionPass
from .typed_specialization import (
    InputHoistingSpec,
    RuntimeDeclaredInputPruningPass,
    RuntimeInputHoistingPass,
    RuntimeInputSpecializationPass,
)
from ..runtime_ir import export_runtime_package


_UNSELECTED_BACKEND = "unselected"


class OptimizerRegistryError(ValueError):
    """Generated registry metadata or one resolution request is inconsistent."""


@dataclass(frozen=True)
class RuntimePassFactoryContext:
    """Every caller-owned value needed to construct current RuntimeIR passes."""

    tensor_data: MutableMapping[str, Any]
    output_argmax: tuple[OutputArgMaxSpecialization, ...] = ()
    monotonic_argmax_inputs: frozenset[str] = field(default_factory=frozenset)
    input_specializations: Mapping[str, Any] = field(default_factory=dict)
    input_hoistings: tuple[InputHoistingSpec, ...] = ()
    causal_mask_inputs: frozenset[str] = field(default_factory=frozenset)
    ptq_calibration: Optional[CalibrationProfile] = None
    ptq_config: PTQConfig = field(default_factory=PTQConfig)
    ptq_selected_nodes: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tensor_data, MutableMapping):
            raise TypeError("runtime pass context requires mutable tensor data")
        object.__setattr__(self, "output_argmax", tuple(self.output_argmax))
        if any(
            not isinstance(item, OutputArgMaxSpecialization)
            for item in self.output_argmax
        ):
            raise TypeError(
                "runtime pass context output_argmax must contain "
                "OutputArgMaxSpecialization values"
            )
        if isinstance(self.monotonic_argmax_inputs, str):
            raise TypeError("monotonic argmax inputs must be a collection")
        monotonic = frozenset(self.monotonic_argmax_inputs)
        if any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in monotonic
        ):
            raise ValueError(
                "monotonic argmax inputs must be non-empty trimmed strings"
            )
        object.__setattr__(self, "monotonic_argmax_inputs", monotonic)
        for attribute, label in (
            ("input_specializations", "input specializations"),
        ):
            value = getattr(self, attribute)
            if not isinstance(value, Mapping):
                raise TypeError(f"{label} must be a mapping")
            object.__setattr__(self, attribute, MappingProxyType(dict(value)))
        object.__setattr__(self, "input_hoistings", tuple(self.input_hoistings))
        if any(
            not isinstance(item, InputHoistingSpec)
            for item in self.input_hoistings
        ):
            raise TypeError("input hoistings must contain InputHoistingSpec values")
        if isinstance(self.causal_mask_inputs, str):
            raise TypeError("causal mask inputs must be a collection")
        causal = frozenset(self.causal_mask_inputs)
        if any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in causal
        ):
            raise ValueError(
                "causal mask inputs must be non-empty trimmed strings"
            )
        object.__setattr__(self, "causal_mask_inputs", causal)
        if self.ptq_calibration is not None and not isinstance(
            self.ptq_calibration, CalibrationProfile,
        ):
            raise TypeError("PTQ calibration must be a CalibrationProfile or None")
        if not isinstance(self.ptq_config, PTQConfig):
            raise TypeError("PTQ config must be a PTQConfig")
        if self.ptq_selected_nodes is not None:
            if isinstance(self.ptq_selected_nodes, (str, bytes)):
                raise TypeError("selected PTQ nodes must be a collection")
            selected = tuple(self.ptq_selected_nodes)
            if any(
                not isinstance(name, str)
                or not name
                or name != name.strip()
                for name in selected
            ):
                raise ValueError(
                    "selected PTQ nodes must contain non-empty trimmed names"
                )
            if len(selected) != len(set(selected)):
                raise ValueError("selected PTQ nodes must not contain duplicates")
            object.__setattr__(self, "ptq_selected_nodes", selected)


@dataclass(frozen=True)
class RegistryPipelineRequest:
    """Explicit policy, target identity, and recipe-selection facts."""

    factory_context: RuntimePassFactoryContext
    target: TargetEnvironment
    rewrite_policy: RewritePolicy = field(default_factory=RewritePolicy.exact)
    allow_calibration: bool = False
    selection_features: frozenset[str] = field(default_factory=frozenset)
    recipe_id: Optional[str] = None
    shape_profile: Optional[Mapping[str, int]] = None
    concrete_pass_policy: ConcretePassPolicy = ConcretePassPolicy.FAIL

    def __post_init__(self) -> None:
        if not isinstance(self.factory_context, RuntimePassFactoryContext):
            raise TypeError("pipeline request requires RuntimePassFactoryContext")
        if not isinstance(self.target, TargetEnvironment):
            raise TypeError("pipeline request requires TargetEnvironment")
        if not isinstance(self.rewrite_policy, RewritePolicy):
            raise TypeError("pipeline request requires RewritePolicy")
        if not isinstance(self.allow_calibration, bool):
            raise TypeError("allow_calibration must be a bool")
        if isinstance(self.selection_features, str):
            raise TypeError("selection features must be a collection")
        features = frozenset(self.selection_features)
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in features
        ):
            raise ValueError("selection features must be non-empty trimmed strings")
        object.__setattr__(self, "selection_features", features)
        if self.recipe_id is not None and (
            not isinstance(self.recipe_id, str)
            or not self.recipe_id
            or self.recipe_id != self.recipe_id.strip()
        ):
            raise ValueError("recipe_id must be None or a non-empty trimmed string")
        if self.shape_profile is not None:
            if not isinstance(self.shape_profile, Mapping) or any(
                not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                for name, value in self.shape_profile.items()
            ):
                raise TypeError(
                    "shape_profile must map symbol names to integer bindings"
                )
            object.__setattr__(
                self,
                "shape_profile",
                MappingProxyType(dict(self.shape_profile)),
            )
        if not isinstance(self.concrete_pass_policy, ConcretePassPolicy):
            raise TypeError(
                "concrete_pass_policy must be a ConcretePassPolicy"
            )


@dataclass(frozen=True)
class RegisteredPassFactory:
    pass_type: type[IRPass]
    build: Callable[[RuntimePassFactoryContext], IRPass]

    def __post_init__(self) -> None:
        if not isinstance(self.pass_type, type) or not issubclass(
            self.pass_type, IRPass,
        ):
            raise TypeError("registered pass type must inherit IRPass")
        if not callable(self.build):
            raise TypeError("registered pass builder must be callable")


def _output_qargmax(context: RuntimePassFactoryContext) -> IRPass:
    if not context.output_argmax:
        raise OptimizerRegistryError(
            "runtime-output-qargmax requires explicit output specializations"
        )
    return RuntimeOutputArgMaxPass(
        context.output_argmax,
        monotonic_byte_tensors=context.monotonic_argmax_inputs,
    )


def _packed_qlinear(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimePackedQLinearSplitPass(context.tensor_data)


def _affine_canonicalization(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeAffineReferenceCanonicalizationPass(context.tensor_data)


def _declared_input_pruning(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeDeclaredInputPruningPass(context.causal_mask_inputs)


def _float_attention_fusion(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeFloatAttentionFusionPass(
        context.tensor_data,
        allow_numerical_migration=True,
        causal_mask_inputs=context.causal_mask_inputs,
    )


def _quantized_attention_fusion(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeQuantizedAttentionFusionPass(
        context.tensor_data,
        allow_numerical_migration=True,
    )


def _quantized_attention_layout(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeQuantizedAttentionLayoutPass(context.tensor_data)


def _keep_mask(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeKeepMaskPass(context.tensor_data)


def _static_qdq_fusion(context: RuntimePassFactoryContext) -> IRPass:
    del context
    return RuntimeStaticQDQComputeFusionPass(allow_numerical_migration=True)


def _static_qdq_qbatch_matmul_fusion(
    context: RuntimePassFactoryContext,
) -> IRPass:
    return RuntimeStaticQDQBatchMatMulFusionPass(
        allow_numerical_migration=True,
        tensor_data=context.tensor_data,
    )


def _static_qdq_groupnorm_silu_fusion(
    context: RuntimePassFactoryContext,
) -> IRPass:
    return RuntimeStaticQDQGroupNormSiLUFusionPass(
        allow_numerical_migration=True,
        tensor_data=context.tensor_data,
    )


def _bias_folding(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeBiasFoldingPass(context.tensor_data)


def _grouped_projection(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeGroupedProjectionSplitPass(context.tensor_data)


def _ptq_authoring(context: RuntimePassFactoryContext) -> IRPass:
    if context.ptq_calibration is None:
        raise OptimizerRegistryError(
            "runtime-ptq-authoring requires an explicit CalibrationProfile"
        )
    return RuntimePTQAuthoringPass(
        context.tensor_data,
        context.ptq_calibration,
        selected_nodes=context.ptq_selected_nodes,
        config=context.ptq_config,
    )


def _quantized_bias_folding(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeQuantizedBiasFoldingPass(
        context.tensor_data,
        allow_numerical_migration=True,
    )


def _constant_folding(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeConstantFoldingPass(context.tensor_data)


def _common_subexpression(context: RuntimePassFactoryContext) -> IRPass:
    return RuntimeCommonSubexpressionEliminationPass(context.tensor_data)


def _input_specialization(context: RuntimePassFactoryContext) -> IRPass:
    if not context.input_specializations:
        raise OptimizerRegistryError(
            "runtime-input-specialization requires explicit bindings"
        )
    return RuntimeInputSpecializationPass(
        context.input_specializations,
        context.tensor_data,
    )


def _input_hoisting(context: RuntimePassFactoryContext) -> IRPass:
    if not context.input_hoistings:
        raise OptimizerRegistryError(
            "runtime-input-hoisting requires explicit tensor contracts"
        )
    return RuntimeInputHoistingPass(context.input_hoistings)


def _without_context(pass_type: type[IRPass]) -> Callable[[RuntimePassFactoryContext], IRPass]:
    def build(context: RuntimePassFactoryContext) -> IRPass:
        del context
        return pass_type()

    return build


PASS_FACTORIES: Mapping[str, RegisteredPassFactory] = MappingProxyType({
    "tools.exporter.optimizer.typed_passes:RuntimeVocabularyPass":
        RegisteredPassFactory(RuntimeVocabularyPass, _without_context(RuntimeVocabularyPass)),
    "tools.exporter.optimizer.typed_specialization:RuntimeInputSpecializationPass":
        RegisteredPassFactory(
            RuntimeInputSpecializationPass,
            _input_specialization,
        ),
    "tools.exporter.optimizer.typed_specialization:RuntimeInputHoistingPass":
        RegisteredPassFactory(RuntimeInputHoistingPass, _input_hoisting),
    "tools.exporter.optimizer.typed_specialization:RuntimeDeclaredInputPruningPass":
        RegisteredPassFactory(
            RuntimeDeclaredInputPruningPass,
            _declared_input_pruning,
        ),
    "tools.exporter.optimizer.typed_constant_folding:RuntimeConstantFoldingPass":
        RegisteredPassFactory(RuntimeConstantFoldingPass, _constant_folding),
    "tools.exporter.optimizer.typed_common_subexpression:RuntimeCommonSubexpressionEliminationPass":
        RegisteredPassFactory(
            RuntimeCommonSubexpressionEliminationPass,
            _common_subexpression,
        ),
    "tools.exporter.optimizer.typed_passes:RuntimeOutputArgMaxPass":
        RegisteredPassFactory(RuntimeOutputArgMaxPass, _output_qargmax),
    "tools.exporter.optimizer.typed_passes:RuntimePackedQLinearSplitPass":
        RegisteredPassFactory(RuntimePackedQLinearSplitPass, _packed_qlinear),
    "tools.exporter.optimizer.typed_affine_canonicalization:RuntimeAffineReferenceCanonicalizationPass":
        RegisteredPassFactory(
            RuntimeAffineReferenceCanonicalizationPass,
            _affine_canonicalization,
        ),
    "tools.exporter.optimizer.typed_attention:RuntimeFloatAttentionFusionPass":
        RegisteredPassFactory(
            RuntimeFloatAttentionFusionPass,
            _float_attention_fusion,
        ),
    "tools.exporter.optimizer.typed_quantized_attention:RuntimeQuantizedAttentionFusionPass":
        RegisteredPassFactory(
            RuntimeQuantizedAttentionFusionPass,
            _quantized_attention_fusion,
        ),
    "tools.exporter.optimizer.typed_quantized_attention:RuntimeQuantizedAttentionLayoutPass":
        RegisteredPassFactory(
            RuntimeQuantizedAttentionLayoutPass,
            _quantized_attention_layout,
        ),
    "tools.exporter.optimizer.typed_attention:RuntimeAttentionLayoutPass":
        RegisteredPassFactory(
            RuntimeAttentionLayoutPass,
            _without_context(RuntimeAttentionLayoutPass),
        ),
    "tools.exporter.optimizer.typed_attention:RuntimeKeepMaskPass":
        RegisteredPassFactory(RuntimeKeepMaskPass, _keep_mask),
    "tools.exporter.optimizer.typed_singleton_transpose:RuntimeSingletonTransposePass":
        RegisteredPassFactory(
            RuntimeSingletonTransposePass,
            _without_context(RuntimeSingletonTransposePass),
        ),
    "tools.exporter.optimizer.typed_qdq_layout:RuntimeQDQTransposeCancellationPass":
        RegisteredPassFactory(
            RuntimeQDQTransposeCancellationPass,
            _without_context(RuntimeQDQTransposeCancellationPass),
        ),
    "tools.exporter.optimizer.typed_qdq_layout:RuntimePointwiseTransposeHoistPass":
        RegisteredPassFactory(
            RuntimePointwiseTransposeHoistPass,
            _without_context(RuntimePointwiseTransposeHoistPass),
        ),
    "tools.exporter.optimizer.typed_qdq_layout:RuntimeQDQMovementPass":
        RegisteredPassFactory(
            RuntimeQDQMovementPass,
            _without_context(RuntimeQDQMovementPass),
        ),
    "tools.exporter.optimizer.typed_elementwise_transpose:RuntimeElementwiseTransposePass":
        RegisteredPassFactory(
            RuntimeElementwiseTransposePass,
            _without_context(RuntimeElementwiseTransposePass),
        ),
    "tools.exporter.optimizer.typed_passes:RuntimeShapeChainPass":
        RegisteredPassFactory(RuntimeShapeChainPass, _without_context(RuntimeShapeChainPass)),
    "tools.exporter.optimizer.static_qdq_fusion:RuntimeStaticQDQComputeFusionPass":
        RegisteredPassFactory(RuntimeStaticQDQComputeFusionPass, _static_qdq_fusion),
    "tools.exporter.optimizer.static_qdq_fusion:RuntimeStaticQDQBatchMatMulFusionPass":
        RegisteredPassFactory(
            RuntimeStaticQDQBatchMatMulFusionPass,
            _static_qdq_qbatch_matmul_fusion,
        ),
    "tools.exporter.optimizer.typed_groupnorm_silu_island:RuntimeStaticQDQGroupNormSiLUFusionPass":
        RegisteredPassFactory(
            RuntimeStaticQDQGroupNormSiLUFusionPass,
            _static_qdq_groupnorm_silu_fusion,
        ),
    "tools.exporter.optimizer.typed_passes:RuntimeCanonicalizePass":
        RegisteredPassFactory(
            RuntimeCanonicalizePass,
            _without_context(RuntimeCanonicalizePass),
        ),
    "tools.exporter.optimizer.typed_passes:RedundantQDQPass":
        RegisteredPassFactory(RedundantQDQPass, _without_context(RedundantQDQPass)),
    "tools.exporter.optimizer.typed_passes:RuntimeDeadCodePass":
        RegisteredPassFactory(RuntimeDeadCodePass, _without_context(RuntimeDeadCodePass)),
    "tools.exporter.optimizer.typed_bias_folding:RuntimeBiasFoldingPass":
        RegisteredPassFactory(RuntimeBiasFoldingPass, _bias_folding),
    "tools.exporter.optimizer.typed_silu_fusion:RuntimeSiluFusionPass":
        RegisteredPassFactory(
            RuntimeSiluFusionPass,
            _without_context(RuntimeSiluFusionPass),
        ),
    "tools.exporter.optimizer.typed_grouped_projection:RuntimeGroupedProjectionSplitPass":
        RegisteredPassFactory(
            RuntimeGroupedProjectionSplitPass,
            _grouped_projection,
        ),
    "tools.exporter.optimizer.typed_sequence_layout:RuntimeSequenceLayoutPass":
        RegisteredPassFactory(
            RuntimeSequenceLayoutPass,
            _without_context(RuntimeSequenceLayoutPass),
        ),
    "tools.exporter.optimizer.typed_ptq_authoring:RuntimePTQAuthoringPass":
        RegisteredPassFactory(
            RuntimePTQAuthoringPass,
            _ptq_authoring,
        ),
    "tools.exporter.optimizer.typed_quantized_bias_folding:RuntimeQuantizedBiasFoldingPass":
        RegisteredPassFactory(
            RuntimeQuantizedBiasFoldingPass,
            _quantized_bias_folding,
        ),
})


def _resolved_implementation(implementation_id: str) -> type[IRPass]:
    module_name, separator, class_name = implementation_id.partition(":")
    if separator != ":" or not module_name or not class_name:
        raise OptimizerRegistryError(
            f"invalid optimizer implementation_id {implementation_id!r}"
        )
    active_exporter_root = __package__.rsplit(".optimizer", 1)[0]
    canonical_exporter_root = "tools.exporter"
    active_module_name = module_name
    if module_name.startswith(canonical_exporter_root + "."):
        active_module_name = (
            active_exporter_root
            + module_name.removeprefix(canonical_exporter_root)
        )
    try:
        implementation = getattr(
            importlib.import_module(active_module_name),
            class_name,
        )
    except (ImportError, AttributeError) as error:
        raise OptimizerRegistryError(
            f"optimizer implementation_id {implementation_id!r} does not resolve"
        ) from error
    if not isinstance(implementation, type) or not issubclass(implementation, IRPass):
        raise OptimizerRegistryError(
            f"optimizer implementation_id {implementation_id!r} is not an IRPass"
        )
    return implementation


def validate_factory_coverage(
    factories: Mapping[str, RegisteredPassFactory] = PASS_FACTORIES,
    pass_catalog: Mapping[str, Mapping[str, Any]] = PASSES,
) -> None:
    """Require an exact one-to-one registry/factory/class contract."""

    implementation_to_pass = {
        str(descriptor["implementation_id"]): pass_id
        for pass_id, descriptor in pass_catalog.items()
    }
    if len(implementation_to_pass) != len(pass_catalog):
        raise OptimizerRegistryError(
            "optimizer registry implementation_id values must be unique"
        )
    expected = set(implementation_to_pass)
    actual = set(factories)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise OptimizerRegistryError(
            f"optimizer pass factory coverage mismatch: missing={missing}, extra={extra}"
        )

    for implementation_id, pass_id in implementation_to_pass.items():
        descriptor = pass_catalog[pass_id]
        factory = factories[implementation_id]
        resolved = _resolved_implementation(implementation_id)
        if factory.pass_type is not resolved:
            raise OptimizerRegistryError(
                f"factory {implementation_id!r} binds {factory.pass_type.__name__}, "
                f"not the registry implementation"
            )
        if factory.pass_type.name != pass_id:
            raise OptimizerRegistryError(
                f"factory {implementation_id!r} pass name {factory.pass_type.name!r} "
                f"does not match registry id {pass_id!r}"
            )
        contract = factory.pass_type.contract
        input_dialects = frozenset(value.value for value in contract.input_dialects)
        expected_inputs = frozenset(str(value) for value in descriptor["input_dialects"])
        if (
            input_dialects != expected_inputs
            or contract.output_dialect.value != descriptor["output_dialect"]
            or contract.repeatable is not descriptor["repeatable"]
        ):
            raise OptimizerRegistryError(
                f"factory {implementation_id!r} PassContract does not match "
                f"registry descriptor {pass_id!r}"
            )


def default_target_environment() -> TargetEnvironment:
    """Return portable legality with no claimed compile or tuning backend."""

    return TargetEnvironment(
        backend_profile="portable",
        compile_backend=_UNSELECTED_BACKEND,
        tune_backend=_UNSELECTED_BACKEND,
    )


def expand_backend_profile(profile: str) -> tuple[str, ...]:
    if profile in PROFILE_MEMBERS:
        return tuple(PROFILE_MEMBERS[profile])
    if profile in ATOMIC_TARGETS:
        return (profile,)
    expected = sorted((*PROFILE_MEMBERS, *ATOMIC_TARGETS))
    raise OptimizerRegistryError(
        f"unknown backend profile {profile!r}; expected one of {expected}"
    )


def _validate_execution_backend(value: str, label: str) -> None:
    if value == _UNSELECTED_BACKEND:
        return
    if value not in RUNTIME_OPERATORS_BY_BACKEND and value not in ATOMIC_TARGETS:
        expected = sorted({*RUNTIME_OPERATORS_BY_BACKEND, *ATOMIC_TARGETS})
        raise OptimizerRegistryError(
            f"unknown {label} {value!r}; expected a generated backend identity "
            f"or {_UNSELECTED_BACKEND!r}: {expected}"
        )


def _runtime_backend_id(target: str) -> str:
    if target in RUNTIME_OPERATORS_BY_BACKEND:
        return target
    if target.startswith("backend:"):
        candidate = target.removeprefix("backend:")
        if candidate in RUNTIME_OPERATORS_BY_BACKEND:
            return candidate
    raise OptimizerRegistryError(
        f"kernel registry target {target!r} has no runtime backend identity"
    )


def _selected_recipe(
    request: RegistryPipelineRequest,
    recipe_catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]]:
    unknown_features = sorted(request.selection_features - FEATURES.keys())
    if unknown_features:
        raise OptimizerRegistryError(
            f"unknown optimizer selection features {unknown_features!r}"
        )
    matches = [
        recipe_id
        for recipe_id, recipe in recipe_catalog.items()
        if (
            frozenset(recipe["required_features"])
            <= request.selection_features
            <= (
                frozenset(recipe["required_features"])
                | frozenset(recipe["supported_features"])
            )
        )
    ]
    if request.recipe_id is not None:
        if request.recipe_id not in recipe_catalog:
            raise OptimizerRegistryError(
                f"unknown optimizer recipe {request.recipe_id!r}"
            )
        if request.recipe_id not in matches:
            recipe = recipe_catalog[request.recipe_id]
            raise OptimizerRegistryError(
                f"optimizer recipe {request.recipe_id!r} requires features "
                f"{tuple(recipe['required_features'])!r} and supports only "
                f"{tuple(recipe['supported_features'])!r}; got "
                f"{sorted(request.selection_features)!r}"
            )
        matches = [request.recipe_id]
    if len(matches) != 1:
        raise OptimizerRegistryError(
            "optimizer selection features must resolve exactly one recipe; "
            f"features={sorted(request.selection_features)}, matches={sorted(matches)}"
        )
    recipe_id = matches[0]
    return recipe_id, recipe_catalog[recipe_id]


def _selected_group_ids(
    recipe: Mapping[str, Any],
    features: frozenset[str],
) -> tuple[str, ...]:
    """Filter authored group overlays without changing their order."""

    selected: list[str] = []
    for overlay in recipe["group_overlays"]:
        required = frozenset(str(value) for value in overlay["required_features"])
        forbidden = frozenset(
            str(value) for value in overlay["forbidden_features"]
        )
        if required <= features and not forbidden & features:
            selected.append(str(overlay["group"]))
    if not selected:
        raise OptimizerRegistryError(
            "optimizer recipe selected no pass groups for features "
            f"{sorted(features)!r}"
        )
    return tuple(selected)


def _semantics(value: str) -> RewriteSemantics:
    try:
        return RewriteSemantics(value)
    except ValueError as error:
        raise OptimizerRegistryError(
            f"optimizer registry has unknown rewrite semantics {value!r}"
        ) from error


def _profile_legality(
    *,
    pass_id: str,
    descriptor: Mapping[str, Any],
    profile_members: tuple[str, ...],
) -> None:
    matched = frozenset(str(value) for value in descriptor["matched_operators"])
    emitted = frozenset(str(value) for value in descriptor["emitted_operators"])
    introduced = emitted - matched
    rules = tuple(descriptor["target_rules"])
    variants = {str(item["id"]): item for item in KERNEL_VARIANTS}

    for member in profile_members:
        admitted = OPS_BY_TARGET.get(member)
        if admitted is None:
            raise OptimizerRegistryError(
                f"backend profile member {member!r} has no exporter qualification"
            )
        unsupported_emitted = sorted(introduced - admitted)
        if unsupported_emitted:
            raise OptimizerRegistryError(
                f"pass {pass_id!r} emits {unsupported_emitted} outside backend "
                f"profile member {member!r}"
            )
        if not rules:
            continue

        runtime_backend = _runtime_backend_id(member)
        matching = [rule for rule in rules if runtime_backend in rule["backends"]]
        if len(matching) != 1:
            raise OptimizerRegistryError(
                f"pass {pass_id!r} has {len(matching)} target rules for backend "
                f"profile member {member!r}; expected exactly one"
            )
        rule = matching[0]
        required_operators = frozenset(rule["required_operators"])
        unsupported_required = sorted(required_operators - admitted)
        if unsupported_required:
            raise OptimizerRegistryError(
                f"pass {pass_id!r} requires {unsupported_required} outside backend "
                f"profile member {member!r}"
            )

        required_variant_ids = tuple(rule["required_kernel_variant_ids"])
        if not required_variant_ids:
            continue
        unknown = sorted(set(required_variant_ids) - variants.keys())
        if unknown:
            raise OptimizerRegistryError(
                f"pass {pass_id!r} requires unknown kernel variants {unknown}"
            )
        member_variants = [
            variants[variant_id]
            for variant_id in required_variant_ids
            if variants[variant_id]["backend"] == runtime_backend
        ]
        if not member_variants:
            raise OptimizerRegistryError(
                f"pass {pass_id!r} has no required kernel variant for backend "
                f"profile member {member!r}"
            )
        covered_operators = frozenset(
            operator
            for variant in member_variants
            for operator in variant["operators"]
        )
        uncovered = sorted(required_operators - covered_operators)
        if uncovered:
            raise OptimizerRegistryError(
                f"pass {pass_id!r} required kernel variants do not cover "
                f"operators {uncovered} for backend profile member {member!r}"
            )


def _graph_profile_legality(
    graph: GraphIR,
    profile_members: tuple[str, ...],
    tensor_data: Mapping[str, Any],
) -> None:
    """Validate exact persisted descriptors against every profile member."""

    document, packaged_tensors = export_runtime_package(graph, tensor_data)
    # The target validator proves every symbolic operator formula and generated
    # route over the complete bounded domain for every profile member, then
    # applies conservative maximum-resource checks.  It never qualifies from a
    # warm profile or a maximum-shape sample.
    validation = validate_graph(
        document,
        profile_members,
        weights=packaged_tensors,
    )
    diagnostics = tuple(
        diagnostic
        for diagnostic in validation.diagnostics
        if diagnostic.code != "VXPKG_CLASS"
    )
    if diagnostics:
        raise ExporterError(diagnostics[0])


def resolve_runtime_pipeline(
    request: RegistryPipelineRequest,
    *,
    factories: Mapping[str, RegisteredPassFactory] = PASS_FACTORIES,
    pass_catalog: Mapping[str, Mapping[str, Any]] = PASSES,
    group_catalog: Mapping[str, Mapping[str, Any]] = PASS_GROUPS,
    recipe_catalog: Mapping[str, Mapping[str, Any]] = PIPELINE_RECIPES,
) -> VerifiedPipeline:
    """Resolve one generated recipe without target or policy inference."""

    validate_factory_coverage(factories, pass_catalog)
    _validate_execution_backend(
        request.target.compile_backend,
        "compile backend",
    )
    _validate_execution_backend(
        request.target.tune_backend,
        "tune backend",
    )
    recipe_id, recipe = _selected_recipe(request, recipe_catalog)
    profile_members = expand_backend_profile(request.target.backend_profile)
    profile_runtime_ids = tuple(_runtime_backend_id(member) for member in profile_members)

    recipe_targets = frozenset(str(value) for value in recipe["targets"])
    if recipe_targets:
        unsupported = sorted(set(profile_runtime_ids) - recipe_targets)
        if unsupported:
            raise OptimizerRegistryError(
                f"recipe {recipe_id!r} does not support every backend profile "
                f"member: {unsupported}"
            )

    recipe_semantics = frozenset(
        _semantics(str(value)) for value in recipe["allowed_semantics"]
    )

    if (
        ("output-qargmax" in request.selection_features)
        != bool(request.factory_context.output_argmax)
    ):
        raise OptimizerRegistryError(
            "output-qargmax selection feature must exactly match explicit "
            "output specializations"
        )
    if (
        ("input-specialization" in request.selection_features)
        != bool(request.factory_context.input_specializations)
    ):
        raise OptimizerRegistryError(
            "input-specialization selection feature must exactly match explicit "
            "input bindings"
        )
    if (
        ("input-hoisting" in request.selection_features)
        != bool(request.factory_context.input_hoistings)
    ):
        raise OptimizerRegistryError(
            "input-hoisting selection feature must exactly match explicit "
            "computed tensor contracts"
        )
    ptq_selected = "ptq-authoring" in request.selection_features
    has_ptq_calibration = request.factory_context.ptq_calibration is not None
    if ptq_selected != has_ptq_calibration:
        raise OptimizerRegistryError(
            "ptq-authoring selection must exactly match an explicit "
            "CalibrationProfile"
        )
    if not ptq_selected and request.factory_context.ptq_selected_nodes is not None:
        raise OptimizerRegistryError(
            "selected PTQ nodes require the ptq-authoring recipe"
        )

    group_ids = _selected_group_ids(recipe, request.selection_features)
    selected_pass_ids: list[str] = []
    groups: list[PassGroup] = []
    for group_id in group_ids:
        group = group_catalog.get(group_id)
        if group is None:
            raise OptimizerRegistryError(
                f"recipe {recipe_id!r} references unknown group {group_id!r}"
            )
        group_passes: list[IRPass] = []
        for pass_id_value in group["passes"]:
            pass_id = str(pass_id_value)
            if pass_id in selected_pass_ids:
                raise OptimizerRegistryError(
                    f"recipe {recipe_id!r} selects pass {pass_id!r} more than once"
                )
            descriptor = pass_catalog.get(pass_id)
            if descriptor is None:
                raise OptimizerRegistryError(
                    f"group {group_id!r} references unknown pass {pass_id!r}"
                )

            pass_semantics = _semantics(str(descriptor["semantics"]))
            if pass_semantics not in recipe_semantics:
                raise OptimizerRegistryError(
                    f"recipe {recipe_id!r} does not authorize pass {pass_id!r} "
                    f"semantics {pass_semantics.value!r}"
                )
            effects = {pass_semantics}
            if descriptor["changes_public_abi"]:
                effects.add(RewriteSemantics.ABI_CHANGE)
            if not request.rewrite_policy.permits(frozenset(effects)):
                rejected = request.rewrite_policy.rejected_effects(
                    frozenset(effects)
                )
                raise OptimizerRegistryError(
                    f"rewrite policy rejects pass {pass_id!r} effects "
                    f"{tuple(value.value for value in rejected)}"
                )
            if descriptor["requires_calibration"] and not request.allow_calibration:
                raise OptimizerRegistryError(
                    f"pass {pass_id!r} requires explicit calibration policy"
                )
            if descriptor["changes_public_abi"] and not recipe["allow_public_abi_change"]:
                raise OptimizerRegistryError(
                    f"recipe {recipe_id!r} does not authorize ABI-changing pass {pass_id!r}"
                )
            if descriptor["requires_calibration"] and not recipe["allow_calibration"]:
                raise OptimizerRegistryError(
                    f"recipe {recipe_id!r} does not authorize calibrated pass {pass_id!r}"
                )

            _profile_legality(
                pass_id=pass_id,
                descriptor=descriptor,
                profile_members=profile_members,
            )
            implementation_id = str(descriptor["implementation_id"])
            factory = factories[implementation_id]
            instance = factory.build(request.factory_context)
            if type(instance) is not factory.pass_type:
                raise OptimizerRegistryError(
                    f"factory {implementation_id!r} returned {type(instance).__name__}, "
                    f"expected {factory.pass_type.__name__}"
                )
            if instance.name != pass_id:
                raise OptimizerRegistryError(
                    f"factory {implementation_id!r} returned pass name "
                    f"{instance.name!r}, expected {pass_id!r}"
                )
            group_passes.append(instance)
            selected_pass_ids.append(pass_id)
        groups.append(PassGroup(
            tuple(group_passes),
            fixed_point=bool(group["fixed_point"]),
            max_iterations=int(group["max_iterations"]),
        ))

    metadata = PipelineMetadata(
        registry_schema_version=SCHEMA_VERSION,
        registry_sha256=REGISTRY_SHA256,
        kernel_registry_sha256=KERNEL_REGISTRY_SHA256,
        recipe_id=recipe_id,
        recipe_version=int(recipe["version"]),
        backend_profile=request.target.backend_profile,
        backend_profile_members=profile_members,
        compile_backend=request.target.compile_backend,
        tune_backend=request.target.tune_backend,
        group_ids=group_ids,
        pass_ids=tuple(selected_pass_ids),
        allowed_semantics=tuple(str(value) for value in recipe["allowed_semantics"]),
        allow_calibration=bool(recipe["allow_calibration"]),
        allow_public_abi_change=bool(recipe["allow_public_abi_change"]),
        recipe_required_features=tuple(
            str(value) for value in recipe["required_features"]
        ),
        recipe_supported_features=tuple(
            str(value) for value in recipe["supported_features"]
        ),
        selection_features=tuple(sorted(request.selection_features)),
    )
    return VerifiedPipeline(
        groups,
        metadata=metadata,
        legality_validator=lambda graph: _graph_profile_legality(
            graph,
            profile_members,
            request.factory_context.tensor_data,
        ),
        mutable_stores=(request.factory_context.tensor_data,),
        shape_profile=request.shape_profile,
        concrete_pass_policy=request.concrete_pass_policy,
    )


validate_factory_coverage()


__all__ = [
    "OptimizerRegistryError",
    "PASS_FACTORIES",
    "RegisteredPassFactory",
    "RegistryPipelineRequest",
    "RuntimePassFactoryContext",
    "default_target_environment",
    "expand_backend_profile",
    "resolve_runtime_pipeline",
    "validate_factory_coverage",
]
