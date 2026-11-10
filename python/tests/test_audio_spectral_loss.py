"""Unit tests for speech vocoder spectral loss functions."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.audio_spectral_loss import (
    stft_complex,
    stft_magnitude,
    spectral_convergence_loss,
    log_stft_magnitude_loss,
    multi_resolution_stft_loss,
    mel_spectrogram_loss,
    instantaneous_phase_loss,
    group_delay_loss,
    instantaneous_frequency_loss,
    phase_aware_stft_loss,
    HAVE_TORCH,
)

if HAVE_TORCH:
    import torch
    from tools.audio_spectral_loss import (
        TorchMultiResolutionSTFTLoss,
        TorchMelSpectrogramLoss,
        TorchPhaseAwareSTFTLoss,
    )


class TestAudioSpectralLoss(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        # 1 second of synthetic speech-like harmonic signal at 24 kHz
        sr = 24000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # Fundamental + harmonics
        self.y_true = (
            0.5 * np.sin(2 * np.pi * 220 * t)
            + 0.25 * np.sin(2 * np.pi * 440 * t)
            + 0.12 * np.sin(2 * np.pi * 880 * t)
            + 0.05 * rng.standard_normal(sr).astype(np.float32)
        )
        # Predicted audio with slightly perturbed harmonics and high-frequency noise
        self.y_pred = (
            0.48 * np.sin(2 * np.pi * 220 * t + 0.1)
            + 0.26 * np.sin(2 * np.pi * 440 * t)
            + 0.10 * np.sin(2 * np.pi * 880 * t)
            + 0.08 * rng.standard_normal(sr).astype(np.float32)
        )

    def test_zero_loss_on_identical_inputs(self):
        """Identical audio waveforms must produce 0.0 spectral loss."""
        sc = spectral_convergence_loss(self.y_true, self.y_true, 1024, 120, 600)
        mag = log_stft_magnitude_loss(self.y_true, self.y_true, 1024, 120, 600)
        total, sc_avg, mag_avg = multi_resolution_stft_loss(self.y_true, self.y_true)
        mel_loss = mel_spectrogram_loss(self.y_true, self.y_true)

        self.assertAlmostEqual(sc, 0.0, places=5)
        self.assertAlmostEqual(mag, 0.0, places=5)
        self.assertAlmostEqual(total, 0.0, places=5)
        self.assertAlmostEqual(mel_loss, 0.0, places=5)

    def test_positive_loss_on_perturbed_inputs(self):
        """Perturbed waveforms must produce strictly positive losses."""
        sc = spectral_convergence_loss(self.y_true, self.y_pred, 1024, 120, 600)
        mag = log_stft_magnitude_loss(self.y_true, self.y_pred, 1024, 120, 600)
        total, sc_avg, mag_avg = multi_resolution_stft_loss(self.y_true, self.y_pred)
        mel_loss = mel_spectrogram_loss(self.y_true, self.y_pred)

        self.assertGreater(sc, 0.0)
        self.assertGreater(mag, 0.0)
        self.assertGreater(total, 0.0)
        self.assertGreater(mel_loss, 0.0)

    @unittest.skipUnless(HAVE_TORCH, "PyTorch required for Torch module tests")
    def test_pytorch_gradient_backprop(self):
        """TorchMultiResolutionSTFTLoss and TorchMelSpectrogramLoss must backpropagate gradients."""
        t_true = torch.from_numpy(self.y_true).unsqueeze(0)
        t_pred = torch.from_numpy(self.y_pred).unsqueeze(0).requires_grad_(True)

        mrstft = TorchMultiResolutionSTFTLoss()
        loss = mrstft(t_true, t_pred)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertIsNotNone(t_pred.grad)
        self.assertTrue(torch.isfinite(t_pred.grad).all())
        self.assertGreater(torch.linalg.norm(t_pred.grad).item(), 0.0)

        # Mel spectrogram loss backprop
        t_pred2 = torch.from_numpy(self.y_pred).unsqueeze(0).requires_grad_(True)
        mel_mod = TorchMelSpectrogramLoss()
        mel_loss = mel_mod(t_true, t_pred2)
        self.assertTrue(torch.isfinite(mel_loss))
        mel_loss.backward()

        self.assertIsNotNone(t_pred2.grad)
        self.assertTrue(torch.isfinite(t_pred2.grad).all())
        self.assertGreater(torch.linalg.norm(t_pred2.grad).item(), 0.0)

    @unittest.skipUnless(HAVE_TORCH, "PyTorch required for numpy-torch parity test")
    def test_numpy_torch_parity(self):
        """NumPy and PyTorch loss implementations must yield consistent loss values."""
        # Use single resolution for exact parity check
        n_fft, hop, win = 1024, 256, 1024
        sc_np = spectral_convergence_loss(self.y_true, self.y_pred, n_fft, hop, win)

        from tools.audio_spectral_loss import TorchSTFTLoss
        torch_loss = TorchSTFTLoss(n_fft, hop, win)
        with torch.no_grad():
            sc_torch, _ = torch_loss(
                torch.from_numpy(self.y_true).unsqueeze(0),
                torch.from_numpy(self.y_pred).unsqueeze(0),
            )

        np.testing.assert_allclose(sc_np, sc_torch.item(), rtol=1e-3, atol=1e-4)

    def test_phase_aware_losses_detect_pure_phase_distortion(self):
        """Phase-aware losses (IP, GD, IF) must be 0 on identical signals and strictly
        positive when phase is scrambled even if magnitude is preserved."""
        ip_zero = instantaneous_phase_loss(self.y_true, self.y_true)
        gd_zero = group_delay_loss(self.y_true, self.y_true)
        if_zero = instantaneous_frequency_loss(self.y_true, self.y_true)
        self.assertAlmostEqual(ip_zero, 0.0, places=5)
        self.assertAlmostEqual(gd_zero, 0.0, places=5)
        self.assertAlmostEqual(if_zero, 0.0, places=5)

        # Create a signal with identical global FFT magnitude but scrambled phase
        rng = np.random.default_rng(123)
        spec = np.fft.rfft(self.y_true)
        random_phase = rng.uniform(-np.pi, np.pi, size=spec.shape)
        y_phase_scrambled = np.fft.irfft(np.abs(spec) * np.exp(1j * random_phase), n=len(self.y_true)).astype(np.float32)

        ip_err = instantaneous_phase_loss(self.y_true, y_phase_scrambled)
        gd_err = group_delay_loss(self.y_true, y_phase_scrambled)
        if_err = instantaneous_frequency_loss(self.y_true, y_phase_scrambled)
        total_pa, comp = phase_aware_stft_loss(self.y_true, y_phase_scrambled)

        self.assertGreater(ip_err, 0.2)
        self.assertGreater(gd_err, 0.2)
        self.assertGreater(if_err, 0.2)
        self.assertGreater(total_pa, 1.0)
        self.assertIn("gd_loss", comp)
        self.assertIn("ip_loss", comp)

    @unittest.skipUnless(HAVE_TORCH, "PyTorch required for TorchPhaseAwareSTFTLoss test")
    def test_torch_phase_aware_stft_loss_backprop(self):
        """TorchPhaseAwareSTFTLoss must compute finite loss and finite non-zero gradients."""
        t_true = torch.from_numpy(self.y_true).unsqueeze(0)
        t_pred = torch.from_numpy(self.y_pred).unsqueeze(0).requires_grad_(True)

        pa_loss_fn = TorchPhaseAwareSTFTLoss()
        loss = pa_loss_fn(t_true, t_pred)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertIsNotNone(t_pred.grad)
        self.assertTrue(torch.isfinite(t_pred.grad).all())
        self.assertGreater(torch.linalg.norm(t_pred.grad).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
