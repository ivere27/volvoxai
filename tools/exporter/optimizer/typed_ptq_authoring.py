"""Registry-compatible precision authoring for calibrated RuntimeIR graphs.

The numerical work lives in :mod:`tools.exporter.typed_ptq`.  This adapter is
deliberately small: it binds one immutable calibration profile and caller
policy to the normal :class:`IRPass` protocol so PTQ participates in the same
protobuf-selected, verifier-guarded pipeline as every other graph rewrite.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional, Sequence

from ..errors import Diagnostic
from ..ir import IRDialect
from ..pipeline import IRPass, PassContract, PassResult
from ..typed_ptq import (
    CalibrationProfile,
    PTQConfig,
    PTQMaterializationReport,
    PTQPlan,
    quantize_runtime_ptq,
)


class RuntimePTQAuthoringPass(IRPass):
    """Author one calibrated W8A8 RuntimeIR revision transactionally."""

    name = "runtime-ptq-authoring"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=False)

    def __init__(
        self,
        tensor_data: MutableMapping[str, Any],
        calibration: CalibrationProfile,
        *,
        selected_nodes: Optional[Sequence[str]] = None,
        config: PTQConfig = PTQConfig(),
    ) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("PTQ authoring requires mutable tensor data")
        if not isinstance(calibration, CalibrationProfile):
            raise TypeError("PTQ authoring requires an immutable CalibrationProfile")
        if selected_nodes is not None:
            if isinstance(selected_nodes, (str, bytes)):
                raise TypeError("selected PTQ nodes must be a sequence of node names")
            selected = tuple(selected_nodes)
            if any(
                not isinstance(name, str) or not name or name != name.strip()
                for name in selected
            ):
                raise ValueError(
                    "selected PTQ nodes must contain non-empty trimmed names"
                )
            if len(selected) != len(set(selected)):
                raise ValueError("selected PTQ nodes must not contain duplicates")
        else:
            selected = None
        if not isinstance(config, PTQConfig):
            raise TypeError("PTQ authoring config must be a PTQConfig")
        self.tensor_data = tensor_data
        self.calibration = calibration
        self.selected_nodes = selected
        self.config = config
        self.plan: PTQPlan | None = None
        self.report: PTQMaterializationReport | None = None

    def run(self, graph) -> PassResult:
        before = graph.fingerprint()
        plan, report = quantize_runtime_ptq(
            graph,
            self.tensor_data,
            self.calibration,
            selected_nodes=self.selected_nodes,
            config=self.config,
        )
        self.plan = plan
        self.report = report
        changed = graph.fingerprint() != before
        # ``VerifiedPipeline`` treats zero/non-zero as the mutation contract.
        # Count authored compute nodes where possible and retain one structural
        # change for any future plan that only rewrites boundaries.
        changes = max(1, report.nodes_quantized) if changed else 0
        retained_note = (
            "retained F32 instances: " + ", ".join(
                f"{item.node_name}({item.source_op},{item.diagnostic_code})"
                for item in report.retained_nodes
            )
            if report.retained_nodes else "retained F32 instances: none"
        )
        return PassResult(
            changes,
            notes=(
                f"authored {report.nodes_quantized} quantized compute nodes",
                f"inserted {report.quantize_boundaries} quantize and "
                f"{report.dequantize_boundaries} dequantize boundaries",
                f"reused {report.byte_edges_reused} byte-domain edges",
                retained_note,
            ),
            metrics=(
                ("nodes_quantized", report.nodes_quantized),
                ("quantize_boundaries", report.quantize_boundaries),
                ("dequantize_boundaries", report.dequantize_boundaries),
                ("byte_edges_reused", report.byte_edges_reused),
                ("broadcast_expands", report.broadcast_expands),
                ("retained_f32_instances", len(report.retained_nodes)),
            ),
            diagnostics=tuple(
                Diagnostic(
                    code=item.diagnostic_code,
                    message=item.reason,
                    stage="typed-ptq-retained",
                    source_node=item.node_name,
                    source_op=item.source_op,
                    constraint="automatic PTQ retained this valid F32 instance",
                )
                for item in report.retained_nodes
            ),
        )


__all__ = ["RuntimePTQAuthoringPass"]
