"""Host-wall setup observations removed before any inference timing begins."""
from contextlib import contextmanager
from pathlib import Path
import time


@contextmanager
def capture_cold(client_type, *, retirement=False, sessions=()):
    """Observe setup only; restore original descriptors when the scope closes.

    Package byte reads and public load/compile/context calls are recorded
    separately. The caller must use this around construction, never a hot loop.
    """
    report = {"asset_read_ms": 0.0, "asset_bytes": 0, "calls": {}}
    originals, callbacks = {}, []
    names = ("release_execution_context",) if retirement else (
        "create_runtime", "load_model", "compile_model", "create_execution_context")
    for name in names:
        original = getattr(client_type, name)
        originals[name] = original

        def observed(*args, _name=name, _original=original, **kwargs):
            before = time.perf_counter_ns()
            try:
                return _original(*args, **kwargs)
            finally:
                item = report["calls"].setdefault(_name, {"count": 0, "wall_ms": 0.0})
                item["count"] += 1
                item["wall_ms"] += (time.perf_counter_ns() - before) / 1e6

        setattr(client_type, name, observed)
    # SDK finalizers retain bound cleanup methods. Only rebind them while the
    # retirement scope is active; setup must never save an observer callback.
    for session in sessions:
        for index, item in enumerate(session._resources):
            release, reference = item
            name = ("release_execution_context" if hasattr(reference, "context_id")
                    else getattr(release, "__name__", ""))
            if name in originals:
                replacement = (getattr(session._engine, name), reference)
                session._resources[index] = replacement
                callbacks.append((session._resources, replacement, item))
    read_bytes = Path.read_bytes

    def read_asset(path):
        before = time.perf_counter_ns()
        data = read_bytes(path)
        if path.name == "graph.json" or path.suffix == ".safetensors":
            report["asset_read_ms"] += (time.perf_counter_ns() - before) / 1e6
            report["asset_bytes"] += len(data)
        return data

    Path.read_bytes = read_asset
    started = time.perf_counter_ns()
    try:
        yield report
    finally:
        report["total_setup_ms"] = (time.perf_counter_ns() - started) / 1e6
        Path.read_bytes = read_bytes
        for name, original in originals.items():
            setattr(client_type, name, original)
        for resources, replacement, original in callbacks:
            if replacement in resources:
                resources[resources.index(replacement)] = original
        report["other_host_setup_ms"] = max(0.0, report["total_setup_ms"] - report["asset_read_ms"]
            - sum(item["wall_ms"] for item in report["calls"].values()))
