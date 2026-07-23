"""Model-neutral analysis of float regions bounded by static Q/DQ edges.

QuantizeLinear and DequantizeLinear are observable numerical operations, not
casts.  This analysis therefore never rewrites or deletes them.  It discovers
maximal connected float-compute regions, records their producer-authored
affine boundaries and exposes the facts required by later candidate planners:
shared values, broadcasts, escaping tensors and intermediate values that have
no quantized affine of their own.

The result is deliberately independent of ONNX, TensorFlow Lite, model names
and backend policy.  A portable rewrite or a backend ``CompiledModel`` region
fusion may consume the same facts while applying different legality rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from ..ir import AffineQuantization, GraphIR, IRDialect, OpNode
from .analysis import AnalysisManager, GraphAnalysis, UseDefAnalysis
from .candidate import RewritePolicy, RewriteSemantics


_BOUNDARY_OPS = frozenset({
    "QuantizeLinear", "DequantizeLinear", "RequantizeLinear",
})
_BROADCAST_OPS = frozenset({"Add", "Sub", "Mul", "Div"})


@dataclass(frozen=True)
class QuantizationDomain:
    """One immutable logical affine domain attached to a byte tensor."""

    dtype: str
    scheme: str
    scale: str
    zero_point: str
    axis: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "scheme": self.scheme,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "axis": self.axis,
        }

    @classmethod
    def from_affine(
        cls,
        *,
        dtype: str,
        affine: AffineQuantization,
    ) -> "QuantizationDomain":
        return cls(
            dtype=dtype,
            scheme=affine.scheme,
            scale=affine.scale,
            zero_point=affine.zero_point,
            axis=affine.axis,
        )


@dataclass(frozen=True)
class QuantizationBoundary:
    """A Q or DQ node at one edge of a discovered float region."""

    node_index: int
    node_name: str
    kind: str
    byte_tensor: str
    float_tensor: str
    domain: QuantizationDomain


@dataclass(frozen=True)
class BroadcastEdge:
    node_index: int
    node_name: str
    input_port: str
    input_shape: tuple[int | str | None, ...]
    output_shape: tuple[int | str | None, ...]


@dataclass(frozen=True)
class QuantizedRegion:
    """A maximal connected component of float compute around Q/DQ edges."""

    id: str
    node_indices: tuple[int, ...]
    node_names: tuple[str, ...]
    operators: tuple[str, ...]
    input_boundaries: tuple[QuantizationBoundary, ...]
    output_boundaries: tuple[QuantizationBoundary, ...]
    static_parameters: tuple[str, ...]
    auxiliary_inputs: tuple[str, ...]
    open_float_inputs: tuple[str, ...]
    escaping_tensors: tuple[str, ...]
    unquantized_intermediates: tuple[str, ...]
    broadcasts: tuple[BroadcastEdge, ...]
    closed: bool

    @property
    def multi_op(self) -> bool:
        return len(self.node_indices) > 1

    @property
    def producer_affines_complete(self) -> bool:
        """Whether every dynamic float edge is bounded by producer Q/DQ facts."""

        return self.closed


@dataclass(frozen=True)
class QuantizedRegionReport:
    analysis_id: str
    regions: tuple[QuantizedRegion, ...]
    by_node: Mapping[str, str]
    graph_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(
            self,
            "by_node",
            MappingProxyType(dict(sorted(self.by_node.items()))),
        )


class _DisjointSet:
    def __init__(self, values: set[int]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self._parent[value]
        while parent != self._parent[parent]:
            parent = self._parent[parent]
        while value != parent:
            following = self._parent[value]
            self._parent[value] = parent
            value = following
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


_USE_DEF_ANALYSIS = UseDefAnalysis()


class QuantizedRegionAnalysis(GraphAnalysis[QuantizedRegionReport]):
    """Discover maximal Q/DQ-bounded float regions in verified RuntimeIR."""

    name = "quantized-regions"
    id = "quantized-regions/v1"

    def run(
        self,
        graph: GraphIR,
        analyses: AnalysisManager | None = None,
    ) -> QuantizedRegionReport:
        """Return region facts directly or through an :class:`AnalysisManager`.

        ``analyses`` is optional to preserve the original one-argument API.
        Managed execution reuses the registered use/definition analysis and is
        fingerprint-cached by ``AnalysisManager``.
        """

        if analyses is not None and analyses.graph is not graph:
            raise ValueError("quantized-region analysis manager owns another graph")
        graph.verify(IRDialect.RUNTIME)
        fingerprint = graph.fingerprint()
        compute = {
            index for index, node in enumerate(graph.nodes)
            if self._is_float_compute(graph, node)
        }
        if not compute:
            return QuantizedRegionReport(
                self.id, (), MappingProxyType({}), fingerprint,
            )

        index = self._use_def(graph, analyses)
        components = _DisjointSet(compute)
        for tensor_name, tensor in graph.tensors.items():
            if tensor.dtype != "float32" or tensor.initializer:
                continue
            attached: list[int] = []
            definition = index.producers.get(tensor_name)
            if definition is not None and definition.node_index in compute:
                attached.append(definition.node_index)
            attached.extend(
                use.node_index for use in index.consumers.get(tensor_name, ())
                if use.node_index in compute
            )
            if len(attached) > 1:
                anchor = attached[0]
                for other in attached[1:]:
                    components.union(anchor, other)

        grouped: dict[int, list[int]] = {}
        for node_index in sorted(compute):
            grouped.setdefault(components.find(node_index), []).append(node_index)

        regions: list[QuantizedRegion] = []
        for node_indices in sorted(grouped.values(), key=lambda value: value[0]):
            region = self._describe(graph, index, tuple(node_indices))
            if region.input_boundaries or region.output_boundaries:
                regions.append(region)
        by_node = {
            node_name: region.id
            for region in regions
            for node_name in region.node_names
        }
        return QuantizedRegionReport(
            self.id,
            tuple(regions),
            MappingProxyType(by_node),
            fingerprint,
        )

    @staticmethod
    def _use_def(graph: GraphIR, analyses: AnalysisManager | None):
        if analyses is None:
            return graph.use_def()
        try:
            return analyses.get(UseDefAnalysis.name)
        except KeyError:
            return analyses.require(_USE_DEF_ANALYSIS)

    @staticmethod
    def _is_float_compute(graph: GraphIR, node: OpNode) -> bool:
        if node.op_type in _BOUNDARY_OPS:
            return False
        values = [
            port.value for port in (*node.inputs, *node.outputs)
            if port.value is not None
        ]
        return any(
            (tensor := graph.tensors.get(name)) is not None
            and not tensor.initializer
            and tensor.dtype == "float32"
            for name in values
        )

    def _describe(
        self,
        graph: GraphIR,
        index,
        node_indices: tuple[int, ...],
    ) -> QuantizedRegion:
        selected = frozenset(node_indices)
        input_boundaries: dict[int, QuantizationBoundary] = {}
        output_boundaries: dict[int, QuantizationBoundary] = {}
        static_parameters: set[str] = set()
        auxiliary_inputs: set[str] = set()
        open_float_inputs: set[str] = set()
        escaping_tensors: set[str] = set()
        intermediates: set[str] = set()
        broadcasts: list[BroadcastEdge] = []

        for node_index in node_indices:
            node = graph.nodes[node_index]
            output_shapes = [
                graph.tensors[port.value].shape
                for port in node.outputs if port.value is not None
            ]
            primary_output_shape = output_shapes[0] if output_shapes else ()
            for port in node.inputs:
                name = port.value
                if name is None:
                    continue
                tensor = graph.tensors[name]
                if (
                    node.op_type in _BROADCAST_OPS
                    and primary_output_shape
                    and tensor.shape != primary_output_shape
                ):
                    broadcasts.append(BroadcastEdge(
                        node_index=node_index,
                        node_name=node.name,
                        input_port=port.name,
                        input_shape=tensor.shape,
                        output_shape=primary_output_shape,
                    ))
                if tensor.initializer:
                    static_parameters.add(name)
                    continue
                definition = index.producers.get(name)
                if tensor.dtype != "float32":
                    auxiliary_inputs.add(name)
                    continue
                if definition is not None and definition.node_index in selected:
                    continue
                boundary = self._input_boundary(graph, name, definition)
                if boundary is None:
                    open_float_inputs.add(name)
                else:
                    input_boundaries[boundary.node_index] = boundary

            for port in node.outputs:
                name = port.value
                if name is None:
                    continue
                tensor = graph.tensors[name]
                uses = index.consumers.get(name, ())
                internal_consumers = tuple(
                    use for use in uses if use.node_index in selected
                )
                if tensor.dtype == "float32" and internal_consumers:
                    intermediates.add(name)
                external = tuple(
                    use for use in uses if use.node_index not in selected
                )
                for use in external:
                    consumer = graph.nodes[use.node_index]
                    boundary = self._output_boundary(
                        graph, name, use.node_index, consumer,
                    )
                    if boundary is None:
                        escaping_tensors.add(name)
                    else:
                        output_boundaries[boundary.node_index] = boundary
                if tensor.public_output:
                    escaping_tensors.add(name)
                if not uses and not tensor.public_output:
                    escaping_tensors.add(name)
                if tensor.dtype != "float32" and external:
                    escaping_tensors.add(name)

        # An intermediate consumed exclusively inside the component has no
        # producer-authored byte affine.  A planner may keep it in F32 inside a
        # compound schedule, but cannot split it into independent Q* kernels
        # without authoring a new quantization choice.
        unquantized = tuple(sorted(intermediates))
        ordered_inputs = tuple(
            input_boundaries[key] for key in sorted(input_boundaries)
        )
        ordered_outputs = tuple(
            output_boundaries[key] for key in sorted(output_boundaries)
        )
        names = tuple(graph.nodes[value].name for value in node_indices)
        operators = tuple(graph.nodes[value].op_type for value in node_indices)
        closed = bool(ordered_inputs and ordered_outputs) and not (
            open_float_inputs or escaping_tensors
        )
        identifier = self._region_id(
            names=names,
            inputs=ordered_inputs,
            outputs=ordered_outputs,
        )
        return QuantizedRegion(
            id=identifier,
            node_indices=node_indices,
            node_names=names,
            operators=operators,
            input_boundaries=ordered_inputs,
            output_boundaries=ordered_outputs,
            static_parameters=tuple(sorted(static_parameters)),
            auxiliary_inputs=tuple(sorted(auxiliary_inputs)),
            open_float_inputs=tuple(sorted(open_float_inputs)),
            escaping_tensors=tuple(sorted(escaping_tensors)),
            unquantized_intermediates=unquantized,
            broadcasts=tuple(sorted(
                broadcasts,
                key=lambda item: (item.node_index, item.input_port),
            )),
            closed=closed,
        )

    @staticmethod
    def _input_boundary(graph, float_name, definition):
        if definition is None:
            return None
        node = graph.nodes[definition.node_index]
        if node.op_type != "DequantizeLinear":
            return None
        inputs = node.input_map()
        outputs = node.output_map()
        byte_name = inputs.get("input")
        byte = graph.tensors.get(byte_name or "")
        if (
            outputs.get("out") != float_name
            or byte is None
            or byte.dtype not in {"int8", "uint8"}
            or byte.quantization is None
            or inputs.get("scale") != byte.quantization.scale
            or inputs.get("zero_point") != byte.quantization.zero_point
        ):
            return None
        return QuantizationBoundary(
            node_index=definition.node_index,
            node_name=node.name,
            kind="dequantize",
            byte_tensor=byte_name,
            float_tensor=float_name,
            domain=QuantizationDomain.from_affine(
                dtype=byte.dtype, affine=byte.quantization,
            ),
        )

    @staticmethod
    def _output_boundary(graph, float_name, node_index, node):
        if node.op_type != "QuantizeLinear":
            return None
        inputs = node.input_map()
        outputs = node.output_map()
        byte_name = outputs.get("out")
        byte = graph.tensors.get(byte_name or "")
        if (
            inputs.get("input") != float_name
            or byte is None
            or byte.dtype not in {"int8", "uint8"}
            or byte.quantization is None
            or inputs.get("scale") != byte.quantization.scale
            or inputs.get("zero_point") != byte.quantization.zero_point
        ):
            return None
        return QuantizationBoundary(
            node_index=node_index,
            node_name=node.name,
            kind="quantize",
            byte_tensor=byte_name,
            float_tensor=float_name,
            domain=QuantizationDomain.from_affine(
                dtype=byte.dtype, affine=byte.quantization,
            ),
        )

    @staticmethod
    def _region_id(
        *,
        names: tuple[str, ...],
        inputs: tuple[QuantizationBoundary, ...],
        outputs: tuple[QuantizationBoundary, ...],
    ) -> str:
        payload = {
            "nodes": list(names),
            "inputs": [item.byte_tensor for item in inputs],
            "outputs": [item.byte_tensor for item in outputs],
        }
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return f"qdq-region:{digest[:20]}"


class RegionCandidatePlacement(str, Enum):
    """Where a region candidate may materialize."""

    PORTABLE_LOGICAL_REWRITE = "portable-logical-rewrite"
    COMPILED_MODEL_FUSION = "compiled-model-fusion"


@dataclass(frozen=True)
class QuantizedRegionCandidateRecord:
    """Immutable planning record, not an unverified graph rewrite.

    Portable records must be bound to a registered logical rewrite that builds
    a typed ``IRPass`` before they can become a :class:`Candidate`.  Backend
    records intentionally never become graph-rewrite candidates: they are
    consumed while preparing one backend's ``CompiledModel`` and leave the
    portable RuntimeIR unchanged.
    """

    source_fingerprint: str
    region_id: str
    placement: RegionCandidatePlacement
    semantics: frozenset[RewriteSemantics]
    node_names: tuple[str, ...]
    operators: tuple[str, ...]
    input_byte_tensors: tuple[str, ...]
    output_byte_tensors: tuple[str, ...]
    input_domains: tuple[QuantizationDomain, ...]
    output_domains: tuple[QuantizationDomain, ...]
    static_parameters: tuple[str, ...]
    auxiliary_inputs: tuple[str, ...]
    unquantized_intermediates: tuple[str, ...]
    broadcast_edges: tuple[str, ...]
    requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.placement, RegionCandidatePlacement):
            raise TypeError("region candidate placement is invalid")
        semantics = frozenset(self.semantics)
        if any(not isinstance(item, RewriteSemantics) for item in semantics):
            raise TypeError("region candidate semantics are invalid")
        object.__setattr__(self, "semantics", semantics)
        expected = (
            frozenset({RewriteSemantics.NUMERICAL_MIGRATION})
            if self.placement is RegionCandidatePlacement.PORTABLE_LOGICAL_REWRITE
            else frozenset({RewriteSemantics.EXACT})
        )
        if semantics != expected:
            raise ValueError(
                f"{self.placement.value} requires semantics "
                f"{sorted(item.value for item in expected)!r}"
            )
        for value, label in (
            (self.source_fingerprint, "source fingerprint"),
            (self.region_id, "region ID"),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"region candidate {label} must be non-empty and trimmed")
        for field_name in (
            "node_names", "operators", "input_byte_tensors",
            "output_byte_tensors", "static_parameters", "auxiliary_inputs",
            "unquantized_intermediates", "broadcast_edges", "requirements",
        ):
            values = tuple(getattr(self, field_name))
            if any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in values
            ):
                raise ValueError(
                    f"region candidate {field_name} must contain trimmed strings"
                )
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "requirements", tuple(sorted(self.requirements)))
        if not self.node_names or len(self.node_names) != len(self.operators):
            raise ValueError("region candidate needs one operator per node")
        if not self.input_byte_tensors or not self.output_byte_tensors:
            raise ValueError("region candidate requires closed byte boundaries")
        for field_name, boundaries in (
            ("input_domains", self.input_domains),
            ("output_domains", self.output_domains),
        ):
            normalized_domains = tuple(boundaries)
            if any(
                not isinstance(domain, QuantizationDomain)
                for domain in normalized_domains
            ):
                raise TypeError(f"region candidate {field_name} are invalid")
            object.__setattr__(self, field_name, normalized_domains)
        if len(self.input_byte_tensors) != len(self.input_domains):
            raise ValueError("region candidate input tensors/domains differ")
        if len(self.output_byte_tensors) != len(self.output_domains):
            raise ValueError("region candidate output tensors/domains differ")
        if len(set(self.requirements)) != len(self.requirements):
            raise ValueError("region candidate requirements must be unique")

    @property
    def changes_portable_graph(self) -> bool:
        return self.placement is RegionCandidatePlacement.PORTABLE_LOGICAL_REWRITE

    @property
    def compiled_model_only(self) -> bool:
        return self.placement is RegionCandidatePlacement.COMPILED_MODEL_FUSION

    def _payload(self) -> dict[str, object]:
        return {
            "format": "quantized-region-candidate/v1",
            "source_fingerprint": self.source_fingerprint,
            "region_id": self.region_id,
            "placement": self.placement.value,
            "semantics": sorted(item.value for item in self.semantics),
            "node_names": list(self.node_names),
            "operators": list(self.operators),
            "input_byte_tensors": list(self.input_byte_tensors),
            "output_byte_tensors": list(self.output_byte_tensors),
            "input_domains": [domain.to_dict() for domain in self.input_domains],
            "output_domains": [domain.to_dict() for domain in self.output_domains],
            "static_parameters": list(self.static_parameters),
            "auxiliary_inputs": list(self.auxiliary_inputs),
            "unquantized_intermediates": list(self.unquantized_intermediates),
            "broadcast_edges": list(self.broadcast_edges),
            "requirements": list(self.requirements),
        }

    @property
    def candidate_id(self) -> str:
        payload = self._payload()
        digest = hashlib.sha256(json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return f"qdq-candidate:{digest}"

    def to_dict(self) -> dict[str, object]:
        return {"candidate_id": self.candidate_id, **self._payload()}


@dataclass(frozen=True)
class QuantizedRegionCandidatePlan:
    analysis_id: str
    graph_fingerprint: str
    candidates: tuple[QuantizedRegionCandidateRecord, ...]
    by_region: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        for value, label in (
            (self.analysis_id, "analysis ID"),
            (self.graph_fingerprint, "graph fingerprint"),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"region candidate plan {label} is invalid")
        candidates = tuple(self.candidates)
        if any(
            not isinstance(item, QuantizedRegionCandidateRecord)
            for item in candidates
        ):
            raise TypeError("region candidate plan contains an invalid candidate")
        identifiers = tuple(item.candidate_id for item in candidates)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("region candidate IDs must be unique")
        if any(
            item.source_fingerprint != self.graph_fingerprint
            for item in candidates
        ):
            raise ValueError("region candidate belongs to another graph revision")
        normalized = {
            region_id: tuple(candidate_ids)
            for region_id, candidate_ids in sorted(self.by_region.items())
        }
        if any(
            not isinstance(region_id, str)
            or not region_id
            or region_id != region_id.strip()
            for region_id in normalized
        ):
            raise ValueError("region candidate index has an invalid region ID")
        indexed_sequence = tuple(
            identifier
            for values in normalized.values()
            for identifier in values
        )
        if (
            len(indexed_sequence) != len(set(indexed_sequence))
            or set(indexed_sequence) != set(identifiers)
        ):
            raise ValueError("region candidate index does not match candidate records")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "by_region", MappingProxyType(normalized))

    @property
    def portable_rewrites(self) -> tuple[QuantizedRegionCandidateRecord, ...]:
        return tuple(
            item for item in self.candidates
            if item.placement is RegionCandidatePlacement.PORTABLE_LOGICAL_REWRITE
        )

    @property
    def compiled_model_fusions(self) -> tuple[QuantizedRegionCandidateRecord, ...]:
        return tuple(
            item for item in self.candidates
            if item.placement is RegionCandidatePlacement.COMPILED_MODEL_FUSION
        )

    def permitted_by(
        self,
        policy: RewritePolicy,
    ) -> tuple[QuantizedRegionCandidateRecord, ...]:
        """Filter planning records through the shared candidate policy."""

        if not isinstance(policy, RewritePolicy):
            raise TypeError("region candidate policy must be a RewritePolicy")
        return tuple(
            item for item in self.candidates
            if policy.permits(item.semantics)
        )


_REGION_ANALYSIS = QuantizedRegionAnalysis()


class QuantizedRegionCandidateAnalysis(
    GraphAnalysis[QuantizedRegionCandidatePlan]
):
    """Plan two explicitly different opportunities for every closed region."""

    name = "quantized-region-candidates"
    id = "quantized-region-candidates/v1"

    def run(
        self,
        graph: GraphIR,
        analyses: AnalysisManager | None = None,
    ) -> QuantizedRegionCandidatePlan:
        if analyses is not None and analyses.graph is not graph:
            raise ValueError("quantized-region planner owns another graph")
        if analyses is None:
            regions = _REGION_ANALYSIS.run(graph)
        else:
            try:
                regions = analyses.get(QuantizedRegionAnalysis.name)
            except KeyError:
                regions = analyses.require(_REGION_ANALYSIS)
        if not isinstance(regions, QuantizedRegionReport):
            raise TypeError("quantized-region analysis returned an invalid report")
        fingerprint = graph.fingerprint()
        if regions.graph_fingerprint and regions.graph_fingerprint != fingerprint:
            raise ValueError("quantized-region report belongs to another graph revision")

        candidates: list[QuantizedRegionCandidateRecord] = []
        by_region: dict[str, tuple[str, ...]] = {}
        for region in regions.regions:
            if not region.closed:
                continue
            common = {
                "source_fingerprint": fingerprint,
                "region_id": region.id,
                "node_names": region.node_names,
                "operators": region.operators,
                "input_byte_tensors": tuple(
                    boundary.byte_tensor for boundary in region.input_boundaries
                ),
                "output_byte_tensors": tuple(
                    boundary.byte_tensor for boundary in region.output_boundaries
                ),
                "input_domains": tuple(
                    boundary.domain for boundary in region.input_boundaries
                ),
                "output_domains": tuple(
                    boundary.domain for boundary in region.output_boundaries
                ),
                "static_parameters": region.static_parameters,
                "auxiliary_inputs": region.auxiliary_inputs,
                "unquantized_intermediates": region.unquantized_intermediates,
                "broadcast_edges": tuple(
                    f"{edge.node_name}:{edge.input_port}"
                    for edge in region.broadcasts
                ),
            }
            portable = QuantizedRegionCandidateRecord(
                **common,
                placement=RegionCandidatePlacement.PORTABLE_LOGICAL_REWRITE,
                semantics=frozenset({RewriteSemantics.NUMERICAL_MIGRATION}),
                requirements=(
                    "bind-registered-portable-logical-rewrite",
                    "preserve-producer-boundary-affines",
                    "qualify-numerical-migration",
                ),
            )
            backend = QuantizedRegionCandidateRecord(
                **common,
                placement=RegionCandidatePlacement.COMPILED_MODEL_FUSION,
                semantics=frozenset({RewriteSemantics.EXACT}),
                requirements=(
                    "compiled-model-only",
                    "preserve-portable-runtime-graph",
                    "preserve-qdq-rounding-order",
                    "preserve-f32-evaluation-order",
                ),
            )
            candidates.extend((portable, backend))
            by_region[region.id] = (portable.candidate_id, backend.candidate_id)
        return QuantizedRegionCandidatePlan(
            analysis_id=self.id,
            graph_fingerprint=fingerprint,
            candidates=tuple(candidates),
            by_region=MappingProxyType(dict(sorted(by_region.items()))),
        )


__all__ = [
    "BroadcastEdge",
    "QuantizationBoundary",
    "QuantizationDomain",
    "QuantizedRegion",
    "QuantizedRegionAnalysis",
    "QuantizedRegionCandidateAnalysis",
    "QuantizedRegionCandidatePlan",
    "QuantizedRegionCandidateRecord",
    "QuantizedRegionReport",
    "RegionCandidatePlacement",
]
