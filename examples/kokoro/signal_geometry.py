"""F32 Bluestein STFT graph for Kokoro's phase-sensitive source network.

Twiddles and permutations are geometry constants. Every operation involving
the signal is an ordinary runtime tensor operation. The butterfly ordering
matches the scalar F32 algorithm documented by ONNX Runtime 1.23.2 dft.cc:
https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/providers/cpu/signal/dft.cc
"""
from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
import math
import sys
import numpy as np
from onnx import helper


def _reverse_six_bits(index):
    return int(f"{index:06b}"[::-1], 2)


@lru_cache(maxsize=1)
def _f32_trig_functions():
    """Resolve scalar C99 F32 math, without silently widening to double.

    Geometry is prepared on the host. Its libm can differ in the last bit
    across platforms, just as the source F32 DFT can. NumPy's SIMD ufunc and
    Python's double math functions are different evaluators and must not
    replace the scalar F32 calls used by the source Bluestein algorithm.
    """
    candidates = [None]
    if sys.platform == "win32":
        candidates.append("ucrtbase.dll")
    else:
        candidates.extend(ctypes.util.find_library(name) for name in ("m", "System"))
    for name in dict.fromkeys(candidates):
        try:
            library = ctypes.CDLL(name)
            cosine, sine = library.cosf, library.sinf
        except (OSError, AttributeError):
            continue
        for function in (cosine, sine):
            function.argtypes = [ctypes.c_float]
            function.restype = ctypes.c_float
        # Keep the library alive together with its resolved functions.
        return library, cosine, sine
    raise RuntimeError("Kokoro STFT preparation requires scalar C99 cosf/sinf; "
                       "no F32 math library is available on this platform")


def _f32_sincos(angles):
    _, cosine, sine = _f32_trig_functions()
    angles = np.asarray(angles, dtype=np.float32)
    real = np.array([cosine(float(value)) for value in angles.flat], dtype=np.float32)
    imaginary = np.array([sine(float(value)) for value in angles.flat], dtype=np.float32)
    return real.reshape(angles.shape), imaginary.reshape(angles.shape)


def _roots():
    velocity = np.float32(-np.float32(2) * np.float32(math.pi) / np.float32(64))
    angles = np.arange(64, dtype=np.float32) * velocity
    return _f32_sincos(angles)


def _fft_constants(real, imaginary):
    """Transform only the constant chirp kernel, never an activation."""
    permutation = [_reverse_six_bits(i) for i in range(64)]
    real, imaginary = real[permutation].copy(), imaginary[permutation].copy()
    roots_real, roots_imaginary = _roots()
    for width in (2, 4, 8, 16, 32, 64):
        index = np.arange(64)
        even = (index // width) * width + index % (width // 2)
        odd = even + width // 2
        twiddle = (index % width) * (64 // width)
        product_real = roots_real[twiddle] * real[odd] - roots_imaginary[twiddle] * imaginary[odd]
        product_imaginary = roots_real[twiddle] * imaginary[odd] + roots_imaginary[twiddle] * real[odd]
        real, imaginary = real[even] + product_real, imaginary[even] + product_imaginary
    return real, imaginary


def bluestein_stft_nodes(signal, output, shape, window, stem, constant):
    length, hop, fft_size = 20, 5, 64
    frames = (shape[1] - length) // hop + 1
    count = 0
    nodes = []
    def emit(op, inputs, **attrs):
        nonlocal count
        name = f"{stem}_fft_{count}"
        count += 1
        nodes.append(helper.make_node(op, inputs, [name], **attrs))
        return name
    def value(label, array):
        return constant(stem + "_" + label, np.asarray(array))
    def complex_product(ar, ai, br, bi):
        return (emit("Sub", [emit("Mul", [ar, br]), emit("Mul", [ai, bi])]),
            emit("Add", [emit("Mul", [ar, bi]), emit("Mul", [ai, br])]))
    permutation = value("bit_reverse", np.array([_reverse_six_bits(i) for i in range(64)], dtype=np.int64))
    roots_real, roots_imaginary = _roots()
    def fft(real, imaginary, label):
        real, imaginary = emit("Gather", [real, permutation], axis=2), emit("Gather", [imaginary, permutation], axis=2)
        for width in (2, 4, 8, 16, 32, 64):
            index = np.arange(64)
            even = (index // width) * width + index % (width // 2)
            odd = even + width // 2
            twiddle = (index % width) * (64 // width)
            e = value(f"{label}_{width}_even", even.astype(np.int64))
            o = value(f"{label}_{width}_odd", odd.astype(np.int64))
            er, ei = emit("Gather", [real, e], axis=2), emit("Gather", [imaginary, e], axis=2)
            pr, pi = complex_product(emit("Gather", [real, o], axis=2), emit("Gather", [imaginary, o], axis=2),
                value(f"{label}_{width}_wr", roots_real[twiddle].reshape(1, 1, 64)),
                value(f"{label}_{width}_wi", roots_imaginary[twiddle].reshape(1, 1, 64)))
            real, imaginary = emit("Add", [er, pr]), emit("Add", [ei, pi])
        return real, imaginary
    positions = np.arange(length, dtype=np.float32)
    angles = np.float32(np.float32(np.float32(-np.float32(math.pi) * positions) * positions) / np.float32(length))
    chirp_real, chirp_imaginary = _f32_sincos(angles)
    kernel_real, kernel_imaginary = np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32)
    kernel_real[:20], kernel_imaginary[:20] = chirp_real, -chirp_imaginary
    kernel_real[-19:], kernel_imaginary[-19:] = chirp_real[1:][::-1], -chirp_imaginary[1:][::-1]
    kernel_real, kernel_imaginary = _fft_constants(kernel_real, kernel_imaginary)
    if len(shape) == 3:
        signal = emit("Reshape", [signal, value("real_shape", np.array([1, shape[1]], dtype=np.int64))])
    frame_indices = np.arange(frames, dtype=np.int64)[:, None] * hop + np.arange(length, dtype=np.int64)[None, :]
    samples = emit("Gather", [signal, value("frame_indices", frame_indices)], axis=1)
    samples = emit("Mul", [samples, value("window", window.reshape(1, 1, 20))])
    zero = value("zero", np.array(0, dtype=np.float32))
    real, imaginary = complex_product(samples, zero,
        value("chirp_real", chirp_real.reshape(1, 1, 20)), value("chirp_imaginary", chirp_imaginary.reshape(1, 1, 20)))
    padding = value("padding", np.zeros((1, frames, fft_size - length), dtype=np.float32))
    real, imaginary = emit("Concat", [real, padding], axis=2), emit("Concat", [imaginary, padding], axis=2)
    real, imaginary = fft(real, imaginary, "forward")
    real, imaginary = complex_product(real, imaginary,
        value("kernel_real", kernel_real.reshape(1, 1, 64)), value("kernel_imaginary", kernel_imaginary.reshape(1, 1, 64)))
    # The source reuses its forward twiddle table for the inverse FFT, then
    # reverses the nonzero output indices and divides by 64.
    real, imaginary = fft(real, imaginary, "inverse")
    divisor = value("fft_size", np.array(64, dtype=np.float32))
    real, imaginary = emit("Div", [real, divisor]), emit("Div", [imaginary, divisor])
    selected = value("inverse_indices", np.array([0, *range(63, 53, -1)], dtype=np.int64))
    real, imaginary = complex_product(emit("Gather", [real, selected], axis=2), emit("Gather", [imaginary, selected], axis=2),
        value("output_chirp_real", chirp_real[:11].reshape(1, 1, 11)), value("output_chirp_imaginary", chirp_imaginary[:11].reshape(1, 1, 11)))
    axis = value("complex_axis", np.array([3], dtype=np.int64))
    real, imaginary = emit("Unsqueeze", [real, axis]), emit("Unsqueeze", [imaginary, axis])
    nodes.append(helper.make_node("Concat", [real, imaginary], [output], axis=3))
    return nodes
