"""Tests for U-Net speech enhancement model and waveform wrapper.

All tests use synthetic inputs and execute without real dataset dependencies.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.enhancement.model import (
    EnhancementModelConfig,
    WaveformUNetEnhancer,
    count_parameters,
    model_summary,
)
from src.enhancement.unet import UNet2D, UNetConfig
from src.preprocessing.stft import STFTConfig


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestUNetConfig:
    def test_default_config(self):
        cfg = UNetConfig()
        assert cfg.in_channels == 1
        assert cfg.out_channels == 1
        assert cfg.channels == (16, 32, 64, 128)
        assert cfg.norm_type == "batchnorm"
        assert cfg.output_mode == "mask"
        assert cfg.mask_act == "sigmoid"

    def test_invalid_channels_empty(self):
        with pytest.raises(ValueError, match="channels must not be empty"):
            UNetConfig(channels=())

    def test_invalid_in_channels(self):
        with pytest.raises(ValueError, match="in_channels must be >= 1"):
            UNetConfig(in_channels=0)

    def test_invalid_out_channels(self):
        with pytest.raises(ValueError, match="out_channels must be >= 1"):
            UNetConfig(out_channels=0)

    def test_invalid_norm_type(self):
        with pytest.raises(ValueError, match="Unsupported norm_type"):
            UNetConfig(norm_type="layernorm")

    def test_invalid_output_mode(self):
        with pytest.raises(ValueError, match="Unsupported output_mode"):
            UNetConfig(output_mode="additive")

    def test_invalid_mask_act(self):
        with pytest.raises(ValueError, match="Unsupported mask_act"):
            UNetConfig(mask_act="tanh")

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout must be in"):
            UNetConfig(dropout=-0.1)
        with pytest.raises(ValueError, match="dropout must be in"):
            UNetConfig(dropout=1.0)


# ==============================================================================
# 2D U-Net Architecture Tests
# ==============================================================================


class TestUNet2D:
    def test_forward_standard_shape(self):
        """Standard 16 kHz STFT segment: 257 frequency bins, 65 time frames."""
        unet = UNet2D()
        x = torch.rand(2, 1, 257, 65)
        out = unet(x)
        assert out.shape == (2, 1, 257, 65)

    def test_forward_arbitrary_frames(self):
        """Preserves arbitrary time frame counts without dimension mismatch."""
        unet = UNet2D()
        for frames in (17, 32, 45, 64, 80, 103):
            x = torch.rand(1, 1, 257, frames)
            out = unet(x)
            assert out.shape == (1, 1, 257, frames)

    def test_all_frequency_bins_preserved(self):
        """Verify output has exactly the same frequency dimension as input."""
        unet = UNet2D()
        x = torch.rand(1, 1, 257, 65)
        out = unet(x)
        assert out.shape[2] == 257

    def test_output_mode_mask_sigmoid(self):
        """Sigmoid mask outputs should be bounded by input for non-negative magnitudes."""
        cfg = UNetConfig(output_mode="mask", mask_act="sigmoid")
        unet = UNet2D(cfg)
        x = torch.rand(2, 1, 257, 65) + 0.1  # strictly positive
        out = unet(x)
        assert torch.all(out >= 0.0)
        # Because sigmoid is in (0, 1), out = mask * x <= x
        assert torch.all(out <= x + 1e-6)

    def test_output_mode_mask_relu(self):
        """ReLU mask allows amplification beyond 1.0."""
        cfg = UNetConfig(output_mode="mask", mask_act="relu")
        unet = UNet2D(cfg)
        x = torch.rand(2, 1, 257, 65)
        out = unet(x)
        assert torch.all(out >= 0.0)

    def test_output_mode_mask_softplus(self):
        """Softplus mask allows smooth non-negative amplification."""
        cfg = UNetConfig(output_mode="mask", mask_act="softplus")
        unet = UNet2D(cfg)
        x = torch.rand(2, 1, 257, 65)
        out = unet(x)
        assert torch.all(out >= 0.0)

    def test_output_mode_direct(self):
        """Direct magnitude estimation should produce strictly non-negative outputs."""
        cfg = UNetConfig(output_mode="direct")
        unet = UNet2D(cfg)
        x = torch.rand(2, 1, 257, 65)
        out = unet(x)
        assert torch.all(out >= 0.0)

    def test_norm_types(self):
        """Test batchnorm, instancenorm, and none."""
        for norm in ("batchnorm", "instancenorm", "none"):
            cfg = UNetConfig(norm_type=norm)
            unet = UNet2D(cfg)
            x = torch.rand(2, 1, 257, 65)
            out = unet(x)
            assert out.shape == (2, 1, 257, 65)

    def test_custom_channels_and_depth(self):
        """Depth 3 and depth 5 configurations."""
        cfg_d3 = UNetConfig(channels=(16, 32, 64))
        unet_d3 = UNet2D(cfg_d3)
        x = torch.rand(1, 1, 257, 65)
        assert unet_d3(x).shape == (1, 1, 257, 65)

        cfg_d5 = UNetConfig(channels=(16, 32, 64, 128, 256))
        unet_d5 = UNet2D(cfg_d5)
        assert unet_d5(x).shape == (1, 1, 257, 65)

    def test_dropout(self):
        cfg = UNetConfig(dropout=0.2)
        unet = UNet2D(cfg)
        unet.train()
        x = torch.rand(2, 1, 257, 65)
        out_train = unet(x)
        unet.eval()
        out_eval = unet(x)
        assert out_train.shape == (2, 1, 257, 65)
        assert out_eval.shape == (2, 1, 257, 65)

    def test_invalid_input_dim(self):
        unet = UNet2D()
        with pytest.raises(ValueError, match="Expected 4-D input tensor"):
            unet(torch.rand(257, 65))


# ==============================================================================
# WaveformUNetEnhancer Wrapper Tests
# ==============================================================================


class TestWaveformUNetEnhancer:
    def test_waveform_forward_batched(self):
        """Batched 2D waveform (B, T) -> (B, T)."""
        enhancer = WaveformUNetEnhancer()
        wav = torch.randn(2, 16384)
        out = enhancer(wav)
        assert out.shape == (2, 16384)

    def test_waveform_forward_1d(self):
        """Single 1D waveform (T,) -> (T,)."""
        enhancer = WaveformUNetEnhancer()
        wav = torch.randn(16384)
        out = enhancer(wav)
        assert out.shape == (16384,)

    def test_exact_length_preservation(self):
        """Reconstructed waveform must match arbitrary length exactly."""
        enhancer = WaveformUNetEnhancer()
        for length in (8000, 16000, 20480, 32000):
            wav = torch.randn(1, length)
            out = enhancer(wav, length=length)
            assert out.shape == (1, length)

    def test_enhance_spectrogram_2d_and_3d(self):
        """Direct magnitude-to-magnitude mapping."""
        enhancer = WaveformUNetEnhancer()
        # 3D: (B, F, N)
        mag_3d = torch.rand(2, 257, 65)
        out_3d = enhancer.enhance_spectrogram(mag_3d)
        assert out_3d.shape == (2, 257, 65)

        # 2D: (F, N)
        mag_2d = torch.rand(257, 65)
        out_2d = enhancer.enhance_spectrogram(mag_2d)
        assert out_2d.shape == (257, 65)

    def test_end_to_end_gradient_flow(self):
        """Verify backpropagation through STFT -> U-Net -> iSTFT."""
        enhancer = WaveformUNetEnhancer()
        noisy = torch.randn(2, 16384, requires_grad=False)
        target = torch.randn(2, 16384)

        enhanced = enhancer(noisy)
        loss = nn.functional.mse_loss(enhanced, target)
        loss.backward()

        # Verify all parameters in U-Net received finite gradients
        for name, param in enhancer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_custom_config(self):
        stft_cfg = STFTConfig(n_fft=256, hop_length=128, win_length=256)
        unet_cfg = UNetConfig(channels=(8, 16, 32))
        model_cfg = EnhancementModelConfig(stft=stft_cfg, unet=unet_cfg)

        enhancer = WaveformUNetEnhancer(model_cfg)
        wav = torch.randn(2, 8192)
        out = enhancer(wav)
        assert out.shape == (2, 8192)


# ==============================================================================
# Model Utility Tests
# ==============================================================================


class TestModelUtilities:
    def test_count_parameters(self):
        unet = UNet2D(UNetConfig(channels=(16, 32, 64, 128)))
        counts = count_parameters(unet)
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]
        assert counts["non_trainable"] == 0

    def test_parameter_freezing(self):
        unet = UNet2D(UNetConfig(channels=(16, 32, 64, 128)))
        initial_trainable = count_parameters(unet)["trainable"]

        # Freeze initial convolution
        for p in unet.inc.parameters():
            p.requires_grad = False

        frozen_counts = count_parameters(unet)
        assert frozen_counts["trainable"] < initial_trainable
        assert frozen_counts["non_trainable"] > 0
        assert frozen_counts["total"] == initial_trainable

    def test_model_summary(self):
        enhancer = WaveformUNetEnhancer()
        summary = model_summary(enhancer)
        assert "WaveformUNetEnhancer Summary:" in summary
        assert "Total parameters:" in summary
        assert "Trainable parameters:" in summary


# ==============================================================================
# Device Compatibility Tests
# ==============================================================================


class TestDeviceCompatibility:
    def test_cpu_execution(self):
        enhancer = WaveformUNetEnhancer().to("cpu")
        wav = torch.randn(1, 16384, device="cpu")
        out = enhancer(wav)
        assert out.device.type == "cpu"
        assert out.shape == (1, 16384)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self):
        enhancer = WaveformUNetEnhancer().to("cuda")
        wav = torch.randn(2, 16384, device="cuda")
        out = enhancer(wav)
        assert out.device.type == "cuda"
        assert out.shape == (2, 16384)
