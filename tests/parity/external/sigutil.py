"""Versioned parity signatures shared by the external Python oracles.

This is a structural port of tests/parity/lib/extract.mjs signature(). Keep the
schema, validation, sampling order, rounding, and stable top-k ordering aligned.
"""
import json
import math
import os
import tempfile

import numpy as np


SIGNATURE_SCHEMA = "volvoxai.parity.signature"
SIGNATURE_VERSION = 2
SAMPLE_MAX = 512
SAMPLE_STRATEGY = "multiscale-axis-v1"


def write_json_atomic(path, value, *, indent=None):
    """Publish JSON without ever exposing a partial artifact to a comparator."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", delete=False,
        ) as temp:
            temp_path = temp.name
            json.dump(value, temp, indent=indent, allow_nan=False)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
        raise


def sig6(x):
    value = float(f"{float(x):.6g}")
    return 0.0 if value == 0 else value


def _normalize_shape(shape, n):
    if not isinstance(shape, (list, tuple)):
        raise TypeError("signature shape must be an array")
    if len(shape) == 0 and n != 1:
        raise ValueError(f"scalar signature shape [] requires exactly 1 element, got {n}")
    elements = 1
    normalized = []
    for axis, dimension in enumerate(shape):
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)) or int(dimension) <= 0:
            raise ValueError(f"signature shape[{axis}] must be a positive integer, got {dimension}")
        dimension = int(dimension)
        elements *= dimension
        normalized.append(dimension)
    if elements != n:
        raise ValueError(f"signature shape {normalized} has {elements} elements, expected {n}")
    return normalized


def _normalize_axes(sample_axes, rank):
    if not isinstance(sample_axes, (list, tuple)):
        raise TypeError("signature sample_axes must be an array")
    normalized = []
    seen = set()
    for position, axis in enumerate(sample_axes):
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or int(axis) < 0 or int(axis) >= rank:
            raise ValueError(f"signature sample_axes[{position}] must be in [0, {rank}), got {axis}")
        axis = int(axis)
        if axis in seen:
            raise ValueError(f"signature sample_axes contains duplicate axis {axis}")
        seen.add(axis)
        normalized.append(axis)
    return normalized


def _interval_midpoints(length, limit):
    result = []
    queue = [(0, length)]
    head = 0
    while len(result) < limit and head < len(queue):
        lo, hi = queue[head]
        head += 1
        mid = (lo + hi) // 2
        result.append(mid)
        if lo < mid:
            queue.append((lo, mid))
        if mid + 1 < hi:
            queue.append((mid + 1, hi))
    return result


def _flatten_index(coords, shape):
    index = 0
    for coordinate, dimension in zip(coords, shape):
        index = index * dimension + coordinate
    return index


def _sample_indices(shape, axes):
    n = math.prod(shape)
    count = min(n, SAMPLE_MAX)
    if n <= SAMPLE_MAX:
        return list(range(n))

    axis_sequences = [
        _interval_midpoints(dimension, min(dimension, count)) for dimension in shape
    ]
    selected = []
    selected_set = set()

    cursors = [0] * len(axes)
    active = len(axes)
    while len(selected) < count and active > 0:
        active = 0
        for slot, axis in enumerate(axes):
            if len(selected) >= count:
                break
            sequence = axis_sequences[axis]
            cursor = cursors[slot]
            if cursor >= len(sequence):
                continue
            active += 1
            cursors[slot] += 1
            coords = []
            for other_axis, other in enumerate(axis_sequences):
                if other_axis == axis:
                    coords.append(sequence[cursor])
                else:
                    coords.append(other[(cursor + axis * 37 + other_axis * 17) % len(other)])
            index = _flatten_index(coords, shape)
            if index not in selected_set:
                selected_set.add(index)
                selected.append(index)

    for index in _interval_midpoints(n, count):
        if len(selected) >= count:
            break
        if index not in selected_set:
            selected_set.add(index)
            selected.append(index)
    return selected


def signature(flat, topk=10, shape=None, sample_axes=None):
    flat = np.asarray(flat, dtype=np.float64).ravel()
    n = int(flat.size)
    if n == 0:
        raise ValueError("signature input must not be empty")
    if not np.isfinite(flat).all():
        index = int(np.flatnonzero(~np.isfinite(flat))[0])
        raise ValueError(f"signature input contains a non-finite number at flat index {index}")
    if isinstance(topk, (bool, np.bool_)) or not isinstance(topk, (int, np.integer)) or int(topk) < 0:
        raise ValueError(f"signature topk must be a non-negative integer, got {topk}")
    topk = int(topk)

    normalized_shape = _normalize_shape([n] if shape is None else shape, n)
    normalized_axes = _normalize_axes([] if sample_axes is None else sample_axes, len(normalized_shape))
    total = float(flat.sum())
    sumsq = float(np.dot(flat, flat))
    if not math.isfinite(total) or not math.isfinite(sumsq):
        raise ValueError("signature statistics overflowed; input magnitude is too large")

    indices = _sample_indices(normalized_shape, normalized_axes)
    values = [sig6(flat[index]) for index in indices]
    top_count = min(topk, n)
    entries = []
    if top_count > 0:
        top_indices = np.argsort(-flat, kind="stable")[:top_count]
        entries = [{"i": int(index), "v": sig6(flat[index])} for index in top_indices]

    return {
        "schema": SIGNATURE_SCHEMA,
        "version": SIGNATURE_VERSION,
        "n": n,
        "shape": normalized_shape,
        "stats": {
            "min": sig6(flat.min()),
            "max": sig6(flat.max()),
            "mean": sig6(total / n),
            "sumsq": sig6(sumsq),
        },
        "sample": {
            "strategy": SAMPLE_STRATEGY,
            "axes": normalized_axes,
            "count": len(indices),
            "indices": indices,
            "values": values,
        },
        "topk": {"requested": topk, "count": len(entries), "entries": entries},
    }
