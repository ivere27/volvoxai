"""Explicit full-waveform acceptance, expressed in the example's PCM16 units."""
from __future__ import annotations

import math
import numpy as np

WAVEFORM_CRITERIA = {
    "max_absolute_error": 1 / 32767,
    "max_rmse": .1 / 32767,
    "pcm16_scale": 32767,
    "reason": "complete float32 pipeline; independently checked stages retain rtol=1e-4 and atol=1e-5",
}


def compare_waveform(actual, expected, criteria=WAVEFORM_CRITERIA):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not actual.size:
        raise ValueError("waveform shapes must match and be nonempty")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("waveforms must be finite")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    peak = float(np.max(abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    reference_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    mismatches = int(np.count_nonzero(abs(difference) > 1e-5 + 1e-4 * abs(expected)))
    scale = criteria["pcm16_scale"]
    report = {"max_absolute_error": peak, "rmse": rmse,
        "snr_db": 20 * math.log10(reference_rms / rmse) if rmse and reference_rms else None,
        "pcm16_lsb_max": peak * scale, "pcm16_lsb_rmse": rmse * scale,
        "strict_rtol": 1e-4, "strict_atol": 1e-5,
        "strict_mismatches": mismatches, "strict_pass": mismatches == 0,
        "waveform_criteria": dict(criteria),
        "waveform_pass": peak <= criteria["max_absolute_error"] and rmse <= criteria["max_rmse"]}
    if not report["waveform_pass"]:
        raise AssertionError(f"full-waveform acceptance failed: {report}")
    return report
