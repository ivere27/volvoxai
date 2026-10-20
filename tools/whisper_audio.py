"""Caller-owned Whisper log-mel preparation for mono F32 16 kHz PCM.

This NumPy implementation accepts the checkpoint's pinned processor metadata.
It performs no audio file decoding, resampling, language detection or engine
dispatch. Those task choices remain with the application.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def prepare_whisper_features(pcm, processor: Mapping, *, sampling_rate=16000):
    """Return an independent [1,80,3000] F32 log-mel snapshot.

    The initial contract is Whisper-tiny's 30 s, 400-point periodic Hann STFT,
    160-sample hop and pinned Slaney filters. Right padding/truncation is
    explicit; no waveform normalization or dither is applied. Filter precision
    remains supplied by the metadata; rounding follows the NumPy CPU processor.
    """
    expected = {"sampling_rate": 16000, "n_fft": 400, "hop_length": 160,
        "n_samples": 480000, "feature_size": 80, "nb_max_frames": 3000,
        "padding_side": "right", "padding_value": 0.0, "chunk_length": 30}
    if not isinstance(processor, Mapping) or any(processor.get(k) != v for k, v in expected.items()):
        raise ValueError("processor must declare the supported Whisper-tiny feature geometry")
    if sampling_rate != processor["sampling_rate"]:
        raise ValueError("caller must supply 16 kHz PCM; implicit resampling is unavailable")
    value = np.asarray(pcm)
    if value.dtype != np.float32 or value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("PCM must be a nonempty finite mono F32 vector")
    filters = np.asarray(processor.get("mel_filters"), np.float64)
    if filters.shape != (80, 201) or not np.isfinite(filters).all() or (filters < 0).any():
        raise ValueError("processor must supply finite nonnegative pinned Slaney filters [80,201]")
    waveform = np.zeros(480000, np.float64)
    length = min(len(value), len(waveform))
    waveform[:length] = value[:length]
    centered = np.pad(waveform, (200, 200), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(centered, 400)[::160]
    window = np.hanning(401)[:-1]
    # Match the pinned NumPy processor's rounding boundaries: it stores each
    # FFT frame as complex64, computes power/mel in F64, then stores log-mel
    # as F32 before dynamic-range clipping and normalization.
    spectrum = np.fft.rfft(frames * window, axis=-1).astype(np.complex64)
    power = np.abs(spectrum, dtype=np.float64) ** 2
    mel = np.dot(filters, power.T)
    log_mel = np.log10(np.maximum(mel, 1e-10)).astype(np.float32)[:, :-1]
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return np.ascontiguousarray(((log_mel + 4.0) / 4.0)[None], dtype=np.float32)
