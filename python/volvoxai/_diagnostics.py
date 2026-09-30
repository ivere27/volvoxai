"""Profiling and debugging scopes over the generated diagnostic services.

``InferenceSession.trace()`` and ``InferenceSession.debug()`` compose existing
RPCs. The C engine owns collection, stepping and every observation; these
adapters add Python scopes, AIP-158 page iteration and NumPy views only.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import contextlib
import base64
import math
import os

import numpy as np
import volvoxai_lite as pb

from ._clients import VxDebugServiceClient, VxPlatformServiceClient, VxProfilingServiceClient
from ._metadata import NUMPY_DTYPES

_DETAILS = {"basic": pb.TraceDetail.TRACE_DETAIL_BASIC, "nodes": pb.TraceDetail.TRACE_DETAIL_NODES}
_GROUPINGS = {
    "node": pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_NODE,
    "operator": pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_OPERATOR,
    "operation": pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_OPERATION,
    "activity": pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_ACTIVITY,
}


def _proto_json(message) -> dict:
    """Standard protobuf JSON, matching the generated JavaScript ``toJson``."""
    def scalar(kind, value):
        if kind in ("float", "double") and not math.isfinite(value):
            return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return value

    output = {}
    for name, encoded in message.to_dict().items():
        field = message.__fields_by_name__[name]
        value = getattr(message, name)
        # Explicit presence belongs to optional/oneof fields. Plain scalar
        # defaults are omitted by the JavaScript protobuf JSON encoder.
        if not (field.optional or field.oneof or field.repeated or field.is_map
                or field.kind == "message") and value in (0, False, "", b""):
            continue
        key = name.split("_")[0] + "".join(part[:1].upper() + part[1:] for part in name.split("_")[1:])
        if field.repeated:
            encoded = [_proto_json(item) for item in value] if field.kind == "message" else [
                scalar(field.kind, item) for item in encoded]
        elif field.is_map:
            encoded = {str(key): _proto_json(item) if field.map_value_kind == "message"
                       else scalar(field.map_value_kind, encoded[str(key)]) for key, item in value.items()}
        elif field.kind == "message":
            encoded = _proto_json(value)
        else:
            encoded = scalar(field.kind, encoded)
        output[key] = encoded
    return output


def list_all(method, request, field: str) -> list:
    """Collect every page of an AIP-158 ``List*`` method."""
    records = []
    while True:
        page = method(request)
        records.extend(getattr(page, field))
        if not page.next_page_token:
            return records
        request.page_token = page.next_page_token


class Trace:
    """A trace scope, like ``torch.profiler.profile``.

    Collection starts on entry. Leaving the block stops it and waits until
    every observation has drained; ``events``, ``resource_snapshots`` and
    ``plans`` are then populated. Export or release while the object is open.
    """

    def __init__(self, host, runtime_id: int, options: pb.TraceOptions, cleanup: list):
        self._client = VxProfilingServiceClient(host)
        self._platform = VxPlatformServiceClient(host)
        self._runtime_id = runtime_id
        self._options = options
        self._cleanup = cleanup
        self.trace_id = 0
        self.info = None
        self.events: list = []
        self.resource_snapshots: list = []
        self.plans: list = []

    def __enter__(self) -> Trace:
        started = self._client.start_trace(pb.StartTraceRequest(runtime_id=self._runtime_id, options=self._options))
        self.trace_id = started.trace_id
        self.info = started
        self._cleanup.append((self._client.release_trace, pb.TraceRef(trace_id=self.trace_id)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    @contextlib.contextmanager
    def annotate(self, name: str) -> Iterator[None]:
        """Record an application range, like ``torch.profiler.record_function``."""
        start = self._platform.get_monotonic_time(pb.Empty()).nanoseconds
        try:
            yield
        finally:
            end = self._platform.get_monotonic_time(pb.Empty()).nanoseconds
            self._client.annotate_trace(pb.AnnotateTraceRequest(
                trace_id=self.trace_id, name=name, start_ns=start, end_ns=end))

    def stop(self) -> pb.TraceInfo:
        """Stop, wait for READY and read the typed records. Idempotent."""
        if self.info is not None and self.info.state == pb.TraceState.TRACE_STATE_READY:
            return self.info
        self.info = self._client.stop_trace(pb.TraceRef(trace_id=self.trace_id))
        self.events = list_all(self._client.list_trace_events,
                               pb.ListTraceEventsRequest(trace_id=self.trace_id, page_size=4096), "events")
        self.resource_snapshots = list_all(
            self._client.list_trace_resource_snapshots,
            pb.ListTraceResourceSnapshotsRequest(trace_id=self.trace_id, page_size=4096), "resource_snapshots")
        self.plans = [self._client.get_trace_plan(pb.GetTracePlanRequest(trace_id=self.trace_id, plan_id=plan_id)).plan
                      for plan_id in range(1, self.info.plans.count + 1)]
        return self.info

    def export_chrome_trace(self, path: str | os.PathLike[str]) -> None:
        """Write Chrome Trace Event JSON for Perfetto or chrome://tracing."""
        self.stop()
        request = pb.ExportChromeTraceRequest(trace_id=self.trace_id, page_size=1024)
        with open(path, "wb") as file:
            while True:
                page = self._client.export_chrome_trace(request)
                file.write(page.data)
                if not page.next_page_token:
                    return
                request.page_token = page.next_page_token

    def summary(self, group_by: str = "node", *, max_rows: int = 256) -> pb.TraceSummary:
        """Engine-computed aggregates: ``node``, ``operator``, ``operation`` or ``activity``.

        Rows are ordered by total time and never mix host and device time.
        Node rows carry the plan step's cost when ``execution_plans=True``.
        """
        if group_by not in _GROUPINGS:
            raise ValueError("group_by must be 'node', 'operator', 'operation' or 'activity'.")
        self.stop()
        return self._client.get_trace_summary(pb.GetTraceSummaryRequest(
            trace_id=self.trace_id, group_by=_GROUPINGS[group_by], max_rows=max_rows))

    def table(self, group_by: str = "operator", *, row_limit: int = 20) -> str:
        """A text table of ``summary()``, like ``torch.profiler``'s ``key_averages().table()``."""
        summary = self.summary(group_by, max_rows=row_limit)
        domains = {pb.TraceTimeDomain.TRACE_TIME_DOMAIN_HOST: "host",
                   pb.TraceTimeDomain.TRACE_TIME_DOMAIN_DEVICE: "device"}
        header = ("Name", "Backend", "Domain", "Calls", "Total ms", "Share", "Median us", "p95 us", "GFLOP/s")
        lines = []
        for row in summary.rows:
            name = row.name if not row.HasField("schedule_index") else f"{row.schedule_index} {row.name}"
            rate = f"{row.achieved_flops_per_second / 1e9:.3g}" if row.HasField("achieved_flops_per_second") else ""
            lines.append((name, row.backend, domains.get(row.domain, ""), str(row.count),
                          f"{row.total_ns / 1e6:.3f}", f"{row.share:.1%}", f"{row.median_ns / 1e3:.1f}",
                          f"{row.p95_ns / 1e3:.1f}", rate))
        widths = [max(len(item[i]) for item in (header, *lines)) for i in range(len(header))]
        text = [" ".join(value.ljust(widths[i]) for i, value in enumerate(item)).rstrip()
                for item in (header, *lines)]
        text.insert(1, " ".join("-" * width for width in widths))
        if summary.events.dropped:
            text.append(f"{summary.events.dropped} events were lost; totals are partial.")
        return "\n".join(text)

    def release(self) -> None:
        self._client.release_trace(pb.TraceRef(trace_id=self.trace_id))


class DebugSession:
    """One isolated, steppable execution of a compiled model.

    Mirrors the Debug Adapter Protocol: ``step()`` runs one schedule entry,
    ``continue_()`` runs to a breakpoint, a NaN/Inf output or completion. A
    failing model sets ``state`` to FAILED and ``failure``; it does not raise.
    """

    def __init__(self, host, info: pb.DebugSessionInfo, cleanup: list, owner=None, buffers=None,
                 *, compiled_cleanup=None):
        self._client = VxDebugServiceClient(host)
        self._owner, self._buffers = owner, buffers
        self._ref = pb.DebugSessionRef(debug_session_id=info.debug_session_id)
        self._cleanup = cleanup
        debug_cleanup = (self._client.release_debug_session, self._ref)
        self._cleanup_entries = ([compiled_cleanup] if compiled_cleanup else []) + [debug_cleanup]
        cleanup.append(debug_cleanup)
        self.info = info
        self.plan = self._client.get_debug_plan(self._ref).plan

    state = property(lambda self: self.info.state)
    stop_reason = property(lambda self: self.info.stop_reason)
    next_step = property(lambda self: self.info.next_step)

    @property
    def failure(self) -> pb.OperationReport | None:
        return self.info.failure if self.info.HasField("failure") else None

    def step(self) -> pb.DebugSessionInfo:
        self.info = self._client.step_debug_session(pb.StepDebugSessionRequest(
            debug_session_id=self._ref.debug_session_id, expected_revision=self.info.revision))
        return self.info

    def continue_(self, *, break_before: Sequence[str] = (), break_on_nonfinite: bool = False) -> pb.DebugSessionInfo:
        """Run until a listed source node, a non-finite output, or completion."""
        self.info = self._client.continue_debug_session(pb.ContinueDebugSessionRequest(
            debug_session_id=self._ref.debug_session_id, expected_revision=self.info.revision,
            break_before_nodes=list(break_before), break_on_nonfinite=break_on_nonfinite))
        return self.info

    def cancel(self) -> pb.DebugSessionInfo:
        self.info = self._client.cancel_debug_session(self._ref)
        return self.info

    def events(self) -> list:
        """Published events, each with its tensor snapshots."""
        return list_all(self._client.list_debug_events,
                        pb.ListDebugEventsRequest(debug_session_id=self._ref.debug_session_id, page_size=4096),
                        "events")

    def collect(self, *, values: bool = True) -> dict:
        """Read a JSON-compatible ``volvoxai-debug/v1`` capture without stepping.

        Includes the current plan, every event and optional raw tensor bytes
        encoded as base64. Collect while this session is open; the returned
        dictionary remains usable after it closes. ``volvoxai.debug_compare``
        and the JavaScript debug-report tool accept the same capture format.
        """
        self.info = self._client.get_debug_session(self._ref)
        self.plan = self._client.get_debug_plan(self._ref).plan
        artifact = {"format": "volvoxai-debug/v1", "info": _proto_json(self.info),
                    "plan": _proto_json(self.plan), "events": []}
        for event in self.events():
            record = _proto_json(event)
            record["snapshots"] = []
            for snapshot in event.snapshots:
                entry = _proto_json(snapshot)
                if values and snapshot.status == pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE:
                    entry["valuesBase64"] = base64.b64encode(self._snapshot_bytes(snapshot)).decode("ascii")
                record["snapshots"].append(entry)
            artifact["events"].append(record)
        return artifact

    def _snapshot_bytes(self, snapshot: pb.DebugTensorSnapshot) -> bytes:
        data = bytearray()
        request = pb.ReadDebugTensorRequest(debug_session_id=self._ref.debug_session_id,
                                            snapshot_id=snapshot.snapshot_id, read_limit=1 << 20)
        while True:
            chunk = self._client.read_debug_tensor(request)
            if chunk.status != pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE:
                raise ValueError(f"Snapshot {snapshot.snapshot_id} has no values: "
                                 f"{pb.DebugTensorStatus(chunk.status).name}.")
            data += chunk.data
            request.read_offset = len(data)
            if len(data) == chunk.size_bytes:
                return bytes(data)
            if not chunk.data or len(data) > chunk.size_bytes:
                raise ValueError("Debug tensor read did not advance within its declared size.")

    def tensor(self, snapshot: pb.DebugTensorSnapshot) -> np.ndarray:
        """Raw storage values of an AVAILABLE snapshot, never dequantized."""
        dtype = NUMPY_DTYPES.get(snapshot.dtype)
        if dtype is None:
            raise ValueError(f"No NumPy dtype for {pb.DataType(snapshot.dtype).name}.")
        return np.frombuffer(self._snapshot_bytes(snapshot), dtype=dtype).reshape(tuple(snapshot.shape))

    def tensor_buffer(self, snapshot: pb.DebugTensorSnapshot):
        """An AVAILABLE snapshot as a retained ``vx.Tensor`` that supports DLPack
        (``np.from_dlpack``, ``torch.from_dlpack``) and outlives the session."""
        from ._tensor import Tensor
        batch = self._client.export_debug_tensor(pb.ExportDebugTensorRequest(
            debug_session_id=self._ref.debug_session_id, snapshot_id=snapshot.snapshot_id))
        return Tensor._from_handle(self._owner, self._buffers, batch.outputs[0])

    def close(self) -> None:
        for entry in reversed(self._cleanup_entries):
            if entry in self._cleanup:
                release, reference = entry
                release(reference)
                self._cleanup.remove(entry)

    def __enter__(self) -> DebugSession:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def trace_options(*, detail: str = "basic", device_timing: bool = False, memory: bool = False,
                  utilization: bool = False, execution_plans: bool = False,
                  capacity_bytes: int | None = None, external_annotations: bool = False,
                  external_capture: bool = False) -> pb.TraceOptions:
    if detail not in _DETAILS:
        raise ValueError("detail must be 'basic' or 'nodes'.")
    options = pb.TraceOptions(detail=_DETAILS[detail], device_timing=device_timing, memory=memory,
                              utilization=utilization, execution_plans=execution_plans)
    if external_annotations or external_capture:
        options.external = pb.TraceExternalOptions(annotations=external_annotations, capture=external_capture)
    if capacity_bytes is not None:
        options.capacity_bytes = capacity_bytes
    return options
