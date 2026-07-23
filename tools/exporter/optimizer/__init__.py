"""Verified typed optimizer and deployment-planning API.

The publication path imports ``volvox-graph/v1`` into typed ``RuntimeIR``,
verifies every pass transactionally, and re-emits a runtime package.
"""

from .analysis import (
    AnalysisError,
    AnalysisManager,
    GraphAnalysis,
    UseDefAnalysis,
)
from .candidate import (
    Candidate,
    CostDirection,
    CostMetric,
    CostOrigin,
    CostVector,
    LegalityProof,
    RewritePolicy,
    RewriteSemantics,
)
from .compiled_model import (
    COMPILED_MODEL_PLAN_FORMAT,
    CompiledModelPlan,
    CompiledModelPlanningError,
    CompiledNodePlan,
    CompiledRegionOpportunity,
    KernelPredicateContext,
    KernelPredicateEvaluator,
    KernelPredicateEvidence,
    KernelVariantCandidate,
    build_compiled_model_plan,
)
from .profiling import (
    CompilerIdentity,
    DeviceIdentity,
    MeasuredBackend,
    ParetoConstraints,
    ProfileMetrics,
    ProfileRecord,
    ScopedTiming,
    TimingAggregate,
    TimingCache,
    TimingCacheEntry,
    TimingCacheKey,
    content_digest,
    pareto_frontier,
)
from .quantized_regions import (
    QuantizedRegionAnalysis,
    QuantizedRegionCandidateAnalysis,
    QuantizedRegionCandidatePlan,
    QuantizedRegionCandidateRecord,
    RegionCandidatePlacement,
)
from .target import TargetEnvironment
from .typed_passes import (
    OutputArgMaxSpecialization,
    RedundantQDQPass,
    RuntimeCanonicalizePass,
    RuntimeDeadCodePass,
    RuntimeOutputArgMaxPass,
    RuntimeShapeChainPass,
    RuntimeVocabularyPass,
)
from .typed_pipeline import (
    author_runtime_ptq_graph,
    author_runtime_ptq_package,
    default_runtime_pipeline,
    optimize_runtime_graph,
    optimize_runtime_package,
    runtime_ptq_authoring_pipeline,
    serialize_pipeline_report,
)
from .static_qdq_fusion import RuntimeStaticQDQComputeFusionPass
from .typed_affine_canonicalization import (
    RuntimeAffineReferenceCanonicalizationPass,
)
from .typed_qdq_layout import (
    RuntimePointwiseTransposeHoistPass,
    RuntimeQDQMovementPass,
    RuntimeQDQTransposeCancellationPass,
)
from .typed_singleton_transpose import RuntimeSingletonTransposePass
from .typed_constant_folding import RuntimeConstantFoldingPass
from .typed_silu_fusion import RuntimeSiluFusionPass
from .typed_attention import (
    RuntimeAttentionLayoutPass,
    RuntimeFloatAttentionFusionPass,
    RuntimeKeepMaskPass,
)
from .typed_quantized_attention import (
    RuntimeQuantizedAttentionFusionPass,
    RuntimeQuantizedAttentionLayoutPass,
    quantized_attention_candidate_count,
)
from .typed_specialization import (
    DerivedValueSelector,
    InputHoistingSpec,
    RuntimeInputHoistingPass,
    RuntimeInputSpecializationPass,
)
from .workspace import (
    BackendProfile,
    OptimizationCandidate,
    OptimizationWorkspace,
    PrecisionContract,
    TransformRecord,
    optimizer_implementation_hash,
)

__all__ = [
    "AnalysisError",
    "AnalysisManager",
    "BackendProfile",
    "COMPILED_MODEL_PLAN_FORMAT",
    "Candidate",
    "CompiledModelPlan",
    "CompiledModelPlanningError",
    "CompiledNodePlan",
    "CompiledRegionOpportunity",
    "CompilerIdentity",
    "CostDirection",
    "CostMetric",
    "CostOrigin",
    "CostVector",
    "DeviceIdentity",
    "DerivedValueSelector",
    "GraphAnalysis",
    "InputHoistingSpec",
    "KernelPredicateContext",
    "KernelPredicateEvaluator",
    "KernelPredicateEvidence",
    "KernelVariantCandidate",
    "LegalityProof",
    "MeasuredBackend",
    "OptimizationCandidate",
    "OptimizationWorkspace",
    "ParetoConstraints",
    "PrecisionContract",
    "ProfileMetrics",
    "ProfileRecord",
    "RedundantQDQPass",
    "OutputArgMaxSpecialization",
    "QuantizedRegionAnalysis",
    "QuantizedRegionCandidateAnalysis",
    "QuantizedRegionCandidatePlan",
    "QuantizedRegionCandidateRecord",
    "RegionCandidatePlacement",
    "RewritePolicy",
    "RewriteSemantics",
    "RuntimeAffineReferenceCanonicalizationPass",
    "RuntimeAttentionLayoutPass",
    "RuntimeCanonicalizePass",
    "RuntimeConstantFoldingPass",
    "RuntimeDeadCodePass",
    "RuntimeFloatAttentionFusionPass",
    "RuntimeInputHoistingPass",
    "RuntimeInputSpecializationPass",
    "RuntimeKeepMaskPass",
    "RuntimeOutputArgMaxPass",
    "RuntimePointwiseTransposeHoistPass",
    "RuntimeQDQMovementPass",
    "RuntimeQDQTransposeCancellationPass",
    "RuntimeQuantizedAttentionFusionPass",
    "RuntimeQuantizedAttentionLayoutPass",
    "RuntimeShapeChainPass",
    "RuntimeSiluFusionPass",
    "RuntimeSingletonTransposePass",
    "RuntimeStaticQDQComputeFusionPass",
    "RuntimeVocabularyPass",
    "ScopedTiming",
    "TargetEnvironment",
    "TimingAggregate",
    "TimingCache",
    "TimingCacheEntry",
    "TimingCacheKey",
    "TransformRecord",
    "UseDefAnalysis",
    "author_runtime_ptq_package",
    "author_runtime_ptq_graph",
    "build_compiled_model_plan",
    "content_digest",
    "default_runtime_pipeline",
    "optimize_runtime_graph",
    "optimize_runtime_package",
    "optimizer_implementation_hash",
    "pareto_frontier",
    "quantized_attention_candidate_count",
    "runtime_ptq_authoring_pipeline",
    "serialize_pipeline_report",
]
