"""Speech vocoder spectral loss functions for QAT fine-tuning and validation.

Implements Multi-Resolution STFT Loss (Yamamoto et al., 2020) and Mel-Spectrogram Loss
for speech synthesis vocoders (StyleTTS2, HiFi-GAN, Kokoro, Piper).
Supports both deterministic NumPy evaluation and PyTorch differentiable training modules.
"""
from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Default (n_fft, hop_length, win_length) resolutions for multi-resolution STFT
DEFAULT_RESOLUTIONS: Tuple[Tuple[int, int, int], ...] = (
    (1024, 120, 600),
    (2048, 240, 1200),
    (512, 50, 240),
)


# ---------------------------------------------------------------------------
# Pure NumPy Implementation
# ---------------------------------------------------------------------------

def _hann_window(win_length: int) -> np.ndarray:
    """Periodic Hann window matching torch.hann_window(periodic=True)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)


def stft_complex(
    waveform: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    """Compute complex STFT(x) for 1D or 2D (B, T) waveforms.

    Args:
        waveform: Float32 array of shape (T,) or (B, T).
        n_fft: FFT size.
        hop_length: STFT hop length.
        win_length: Window size (padded to n_fft with zeros if win_length < n_fft).

    Returns:
        Complex spectrogram of shape (B, freq_bins, frames) where freq_bins = n_fft // 2 + 1.
    """
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"waveform must be 1D or 2D, got shape {x.shape}")

    batch_size, _ = x.shape
    pad = n_fft // 2
    x_padded = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")

    window = _hann_window(win_length).astype(np.float32)
    specs = []
    for b in range(batch_size):
        frames = np.lib.stride_tricks.sliding_window_view(x_padded[b], win_length)[::hop_length]
        if win_length < n_fft:
            frames = np.pad(frames, ((0, 0), (0, n_fft - win_length)), mode="constant")
            window_padded = np.pad(window, (0, n_fft - win_length), mode="constant")
        else:
            window_padded = window
        spec = np.fft.rfft(frames * window_padded, n=n_fft, axis=-1).T  # (freq_bins, frames)
        specs.append(spec.astype(np.complex64))

    return np.stack(specs, axis=0)


def stft_magnitude(
    waveform: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    """Compute real STFT magnitude |STFT(x)| for 1D or 2D (B, T) waveforms."""
    return np.abs(stft_complex(waveform, n_fft, hop_length, win_length))


def _anti_wrapping(phase_diff: np.ndarray) -> np.ndarray:
    """Anti-wrapping phase operator mapping phase differences into [-pi, pi]."""
    return np.abs(phase_diff - 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi)))


def instantaneous_phase_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 120,
    win_length: int = 600,
    mag_Floor: float = 1e-3,
) -> float:
    """Compute magnitude-weighted anti-wrapped Instantaneous Phase (IP) error.

    Bins with negligible energy receive smoothly attenuated weight so arbitrary
    numerical phase noise in silent bins does not corrupt true harmonic phase.
    """
    spec_true = stft_complex(y_true, n_fft, hop_length, win_length)
    spec_pred = stft_complex(y_pred, n_fft, hop_length, win_length)

    mag_true = np.abs(spec_true)
    weight = mag_true / (np.mean(mag_true, axis=(-2, -1), keepdims=True) + mag_Floor)
    weight = np.clip(weight, 0.05, 5.0)

    phase_true = np.angle(spec_true)
    phase_pred = np.angle(spec_pred)
    aw_err = _anti_wrapping(phase_true - phase_pred)
    return float(np.mean(weight * aw_err))


def group_delay_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 120,
    win_length: int = 600,
    mag_floor: float = 1e-3,
) -> float:
    """Compute anti-wrapped Group Delay (GD) loss across adjacent frequency bins.

    Group delay tau_g(f, t) = -(phi(f+1, t) - phi(f, t)) governs destructive
    interference between neighboring STFT bins during iSTFT overlap-add.
    """
    spec_true = stft_complex(y_true, n_fft, hop_length, win_length)
    spec_pred = stft_complex(y_pred, n_fft, hop_length, win_length)

    mag_true = np.abs(spec_true)
    weight = 0.5 * (mag_true[:, 1:, :] + mag_true[:, :-1, :])
    weight = weight / (np.mean(weight, axis=(-2, -1), keepdims=True) + mag_floor)
    weight = np.clip(weight, 0.05, 5.0)

    phase_true = np.angle(spec_true)
    phase_pred = np.angle(spec_pred)

    gd_true = -(phase_true[:, 1:, :] - phase_true[:, :-1, :])
    gd_pred = -(phase_pred[:, 1:, :] - phase_pred[:, :-1, :])
    aw_gd_err = _anti_wrapping(gd_true - gd_pred)
    return float(np.mean(weight * aw_gd_err))


def instantaneous_frequency_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 120,
    win_length: int = 600,
    mag_floor: float = 1e-3,
) -> float:
    """Compute anti-wrapped Instantaneous Frequency (temporal phase derivative) loss."""
    spec_true = stft_complex(y_true, n_fft, hop_length, win_length)
    spec_pred = stft_complex(y_pred, n_fft, hop_length, win_length)

    mag_true = np.abs(spec_true)
    weight = 0.5 * (mag_true[:, :, 1:] + mag_true[:, :, :-1])
    weight = weight / (np.mean(weight, axis=(-2, -1), keepdims=True) + mag_floor)
    weight = np.clip(weight, 0.05, 5.0)

    phase_true = np.angle(spec_true)
    phase_pred = np.angle(spec_pred)

    if_true = phase_true[:, :, 1:] - phase_true[:, :, :-1]
    if_pred = phase_pred[:, :, 1:] - phase_pred[:, :, :-1]
    aw_if_err = _anti_wrapping(if_true - if_pred)
    return float(np.mean(weight * aw_if_err))


def phase_aware_stft_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resolutions: Sequence[Tuple[int, int, int]] = DEFAULT_RESOLUTIONS,
    factor_sc: float = 1.0,
    factor_mag: float = 1.0,
    factor_ip: float = 0.5,
    factor_gd: float = 0.5,
    factor_if: float = 0.5,
) -> Tuple[float, dict]:
    """Compute multi-resolution Phase-Aware STFT loss combining magnitude and phase terms.

    Returns:
        Tuple of (total_loss, components_dict).
    """
    sc_total = 0.0
    mag_total = 0.0
    ip_total = 0.0
    gd_total = 0.0
    if_total = 0.0
    for n_fft, hop_length, win_length in resolutions:
        sc_total += spectral_convergence_loss(y_true, y_pred, n_fft, hop_length, win_length)
        mag_total += log_stft_magnitude_loss(y_true, y_pred, n_fft, hop_length, win_length)
        ip_total += instantaneous_phase_loss(y_true, y_pred, n_fft, hop_length, win_length)
        gd_total += group_delay_loss(y_true, y_pred, n_fft, hop_length, win_length)
        if_total += instantaneous_frequency_loss(y_true, y_pred, n_fft, hop_length, win_length)

    n_res = float(len(resolutions))
    comp = {
        "sc_loss": sc_total / n_res,
        "mag_loss": mag_total / n_res,
        "ip_loss": ip_total / n_res,
        "gd_loss": gd_total / n_res,
        "if_loss": if_total / n_res,
    }
    total = (
        factor_sc * comp["sc_loss"]
        + factor_mag * comp["mag_loss"]
        + factor_ip * comp["ip_loss"]
        + factor_gd * comp["gd_loss"]
        + factor_if * comp["if_loss"]
    )
    return float(total), comp


def spectral_convergence_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> float:
    """Compute spectral convergence loss || |Y| - |Y_hat| ||_F / || |Y| ||_F."""
    mag_true = stft_magnitude(y_true, n_fft, hop_length, win_length)
    mag_pred = stft_magnitude(y_pred, n_fft, hop_length, win_length)
    diff_norm = np.linalg.norm(mag_true - mag_pred, ord="fro", axis=(-2, -1))
    true_norm = np.maximum(np.linalg.norm(mag_true, ord="fro", axis=(-2, -1)), 1e-7)
    return float(np.mean(diff_norm / true_norm))


def log_stft_magnitude_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    eps: float = 1e-5,
) -> float:
    """Compute log STFT magnitude loss: mean(|log(|Y| + eps) - log(|Y_hat| + eps)|)."""
    mag_true = stft_magnitude(y_true, n_fft, hop_length, win_length)
    mag_pred = stft_magnitude(y_pred, n_fft, hop_length, win_length)
    log_true = np.log(np.maximum(mag_true, eps))
    log_pred = np.log(np.maximum(mag_pred, eps))
    return float(np.mean(np.abs(log_true - log_pred)))


def multi_resolution_stft_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resolutions: Sequence[Tuple[int, int, int]] = DEFAULT_RESOLUTIONS,
    factor_sc: float = 1.0,
    factor_mag: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute multi-resolution STFT loss over a set of FFT resolutions.

    Returns:
        Tuple of (total_loss, sc_loss, mag_loss).
    """
    sc_total = 0.0
    mag_total = 0.0
    for n_fft, hop_length, win_length in resolutions:
        sc = spectral_convergence_loss(y_true, y_pred, n_fft, hop_length, win_length)
        mag = log_stft_magnitude_loss(y_true, y_pred, n_fft, hop_length, win_length)
        sc_total += sc
        mag_total += mag

    sc_avg = sc_total / len(resolutions)
    mag_avg = mag_total / len(resolutions)
    total = factor_sc * sc_avg + factor_mag * mag_avg
    return total, sc_avg, mag_avg


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Construct triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    if f_max is None:
        f_max = sample_rate / 2.0
    num_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(np.array(f_min))
    mel_max = _hz_to_mel(np.array(f_max))
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)

    fbank = np.zeros((n_mels, num_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus and k < num_bins:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m and k < num_bins:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank


def mel_spectrogram_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float | None = None,
    eps: float = 1e-5,
) -> float:
    """Compute L1 loss on log-mel spectrograms."""
    fbank = build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
    mag_true = stft_magnitude(y_true, n_fft, hop_length, win_length)
    mag_pred = stft_magnitude(y_pred, n_fft, hop_length, win_length)

    # (B, n_mels, frames) = (n_mels, freq_bins) @ (B, freq_bins, frames)
    mel_true = np.matmul(fbank, mag_true)
    mel_pred = np.matmul(fbank, mag_pred)

    log_mel_true = np.log(np.maximum(mel_true, eps))
    log_mel_pred = np.log(np.maximum(mel_pred, eps))
    return float(np.mean(np.abs(log_mel_true - log_mel_pred)))


# ---------------------------------------------------------------------------
# PyTorch Modules for QAT Training
# ---------------------------------------------------------------------------

if HAVE_TORCH:
    class TorchSTFTLoss(nn.Module):
        """Single-resolution STFT loss combining spectral convergence and log STFT magnitude."""
        def __init__(self, n_fft: int, hop_length: int, win_length: int):
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length
            self.register_buffer("window", torch.hann_window(win_length, periodic=True))

        def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if y_true.ndim == 1:
                y_true = y_true.unsqueeze(0)
            if y_pred.ndim == 1:
                y_pred = y_pred.unsqueeze(0)

            # STFT magnitude
            mag_true = torch.stft(
                y_true,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode="reflect",
            ).abs()

            mag_pred = torch.stft(
                y_pred,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode="reflect",
            ).abs()

            # Spectral convergence
            diff_norm = torch.linalg.norm(mag_true - mag_pred, ord="fro", dim=(-2, -1))
            true_norm = torch.linalg.norm(mag_true, ord="fro", dim=(-2, -1)).clamp(min=1e-7)
            sc_loss = (diff_norm / true_norm).mean()

            # Log STFT magnitude
            log_true = torch.log(mag_true.clamp(min=1e-5))
            log_pred = torch.log(mag_pred.clamp(min=1e-5))
            mag_loss = torch.mean(torch.abs(log_true - log_pred))

            return sc_loss, mag_loss


    class TorchMultiResolutionSTFTLoss(nn.Module):
        """Multi-Resolution STFT Loss module for PyTorch vocoder QAT fine-tuning."""
        def __init__(
            self,
            resolutions: Sequence[Tuple[int, int, int]] = DEFAULT_RESOLUTIONS,
            factor_sc: float = 1.0,
            factor_mag: float = 1.0,
        ):
            super().__init__()
            self.factor_sc = factor_sc
            self.factor_mag = factor_mag
            self.losses = nn.ModuleList([
                TorchSTFTLoss(n_fft, hop, win) for n_fft, hop, win in resolutions
            ])

        def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            sc_total = torch.tensor(0.0, device=y_true.device)
            mag_total = torch.tensor(0.0, device=y_true.device)
            for loss_mod in self.losses:
                sc, mag = loss_mod(y_true, y_pred)
                sc_total = sc_total + sc
                mag_total = mag_total + mag

            sc_avg = sc_total / len(self.losses)
            mag_avg = mag_total / len(self.losses)
            return self.factor_sc * sc_avg + self.factor_mag * mag_avg


    class TorchMelSpectrogramLoss(nn.Module):
        """L1 loss between log Mel-spectrograms for speech vocoder training."""
        def __init__(
            self,
            sample_rate: int = 24000,
            n_fft: int = 1024,
            hop_length: int = 256,
            win_length: int = 1024,
            n_mels: int = 80,
            f_min: float = 0.0,
            f_max: float | None = None,
        ):
            super().__init__()
            fbank = build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
            self.register_buffer("fbank", torch.from_numpy(fbank).float())
            self.register_buffer("window", torch.hann_window(win_length, periodic=True))
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length

        def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            if y_true.ndim == 1:
                y_true = y_true.unsqueeze(0)
            if y_pred.ndim == 1:
                y_pred = y_pred.unsqueeze(0)

            mag_true = torch.stft(
                y_true,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode="reflect",
            ).abs()

            mag_pred = torch.stft(
                y_pred,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode="reflect",
            ).abs()

            # (B, n_mels, frames)
            mel_true = torch.matmul(self.fbank, mag_true)
            mel_pred = torch.matmul(self.fbank, mag_pred)

            log_mel_true = torch.log(mel_true.clamp(min=1e-5))
            log_mel_pred = torch.log(mel_pred.clamp(min=1e-5))
            return torch.mean(torch.abs(log_mel_true - log_mel_pred))


    def _torch_anti_wrapping(phase_diff: torch.Tensor) -> torch.Tensor:
        """Differentiable anti-wrapping operator using Straight-Through Estimator for round()."""
        two_pi = 2.0 * torch.pi
        cycles = phase_diff / two_pi
        cycles_ste = cycles - (cycles - torch.round(cycles)).detach()
        return torch.abs(phase_diff - two_pi * cycles_ste)


    class TorchPhaseAwareSTFTLoss(nn.Module):
        """Multi-Resolution Phase-Aware STFT Loss for vocoder QAT fine-tuning.

        Combines:
        - Spectral Convergence & Log-Magnitude Loss
        - Instantaneous Phase (IP) Anti-Wrapping + Cosine Phase Loss
        - Group Delay (GD) Anti-Wrapping Loss (frequency derivative of phase)
        - Instantaneous Frequency (IF) Anti-Wrapping Loss (time derivative of phase)
        - Silence Destructive-Interference Penalty (suppresses hiss in silent frames)
        """
        def __init__(
            self,
            resolutions: Sequence[Tuple[int, int, int]] = DEFAULT_RESOLUTIONS,
            factor_sc: float = 1.0,
            factor_mag: float = 1.0,
            factor_ip: float = 0.5,
            factor_gd: float = 0.5,
            factor_if: float = 0.5,
            factor_silence: float = 2.0,
        ):
            super().__init__()
            self.resolutions = tuple(resolutions)
            self.factor_sc = factor_sc
            self.factor_mag = factor_mag
            self.factor_ip = factor_ip
            self.factor_gd = factor_gd
            self.factor_if = factor_if
            self.factor_silence = factor_silence
            for idx, (_, _, win) in enumerate(self.resolutions):
                self.register_buffer(f"window_{idx}", torch.hann_window(win, periodic=True))

        def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            if y_true.ndim == 1:
                y_true = y_true.unsqueeze(0)
            if y_pred.ndim == 1:
                y_pred = y_pred.unsqueeze(0)

            total = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
            for idx, (n_fft, hop, win) in enumerate(self.resolutions):
                window = getattr(self, f"window_{idx}")
                spec_true = torch.stft(
                    y_true, n_fft=n_fft, hop_length=hop, win_length=win,
                    window=window, return_complex=True, center=True, pad_mode="reflect"
                )
                spec_pred = torch.stft(
                    y_pred, n_fft=n_fft, hop_length=hop, win_length=win,
                    window=window, return_complex=True, center=True, pad_mode="reflect"
                )

                mag_true = spec_true.abs()
                mag_pred = spec_pred.abs()

                # 1. Spectral Convergence & Log-Magnitude
                diff_norm = torch.linalg.norm(mag_true - mag_pred, ord="fro", dim=(-2, -1))
                true_norm = torch.linalg.norm(mag_true, ord="fro", dim=(-2, -1)).clamp(min=1e-7)
                sc_loss = (diff_norm / true_norm).mean()

                log_true = torch.log(mag_true.clamp(min=1e-5))
                log_pred = torch.log(mag_pred.clamp(min=1e-5))
                mag_loss = torch.mean(torch.abs(log_true - log_pred))

                # Stable phase angle with epsilon preventing NaN gradients at 0+0j
                phase_true = torch.atan2(spec_true.imag, spec_true.real + 1e-8)
                phase_pred = torch.atan2(spec_pred.imag, spec_pred.real + 1e-8)

                # Magnitude weighting so active harmonic bins dominate phase alignment
                w_ip = (mag_true / (mag_true.mean(dim=(-2, -1), keepdim=True) + 1e-3)).clamp(0.05, 5.0).detach()
                w_gd = (0.5 * (mag_true[:, 1:, :] + mag_true[:, :-1, :]))
                w_gd = (w_gd / (w_gd.mean(dim=(-2, -1), keepdim=True) + 1e-3)).clamp(0.05, 5.0).detach()
                w_if = (0.5 * (mag_true[:, :, 1:] + mag_true[:, :, :-1]))
                w_if = (w_if / (w_if.mean(dim=(-2, -1), keepdim=True) + 1e-3)).clamp(0.05, 5.0).detach()

                # 2. Instantaneous Phase (IP) Loss (Anti-Wrapping + Circular Cosine)
                delta_phi = phase_true - phase_pred
                ip_loss = torch.mean(w_ip * (_torch_anti_wrapping(delta_phi) + (1.0 - torch.cos(delta_phi))))

                # 3. Group Delay (GD) Loss across frequency bins
                gd_true = -(phase_true[:, 1:, :] - phase_true[:, :-1, :])
                gd_pred = -(phase_pred[:, 1:, :] - phase_pred[:, :-1, :])
                delta_gd = gd_true - gd_pred
                gd_loss = torch.mean(w_gd * (_torch_anti_wrapping(delta_gd) + (1.0 - torch.cos(delta_gd))))

                # 4. Instantaneous Frequency (IF) Loss across time frames
                if_true = phase_true[:, :, 1:] - phase_true[:, :, :-1]
                if_pred = phase_pred[:, :, 1:] - phase_pred[:, :, :-1]
                delta_if = if_true - if_pred
                if_loss = torch.mean(w_if * (_torch_anti_wrapping(delta_if) + (1.0 - torch.cos(delta_if))))

                res_loss = (
                    self.factor_sc * sc_loss
                    + self.factor_mag * mag_loss
                    + self.factor_ip * ip_loss
                    + self.factor_gd * gd_loss
                    + self.factor_if * if_loss
                )
                total = total + res_loss

            total = total / float(len(self.resolutions))

            # 5. Time-domain silence destructive-interference penalty
            if self.factor_silence > 0.0:
                silence_mask = (y_true.abs() < 0.01).float().detach()
                silence_loss = torch.mean(silence_mask * torch.abs(y_pred - y_true))
                total = total + self.factor_silence * silence_loss

            return total
