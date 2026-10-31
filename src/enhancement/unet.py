"""2D U-Net architecture for magnitude-spectrogram speech enhancement.

Designed for STFT magnitude inputs (e.g. 257 frequency bins at 16 kHz).
Handles down/up-sampling shape mismatches via dynamic internal padding
to the nearest multiple of 2^depth and unpad cropping, ensuring all
frequency bins and time frames are preserved without dimensional errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UNetConfig:
    """Configuration for 2D U-Net speech enhancement model.

    Parameters
    ----------
    in_channels : int
        Number of input spectrogram channels (1 for magnitude).
    out_channels : int
        Number of output channels (1 for single-channel enhancement).
    channels : tuple of int
        Channel counts for each encoder level.  The length determines
        the depth of the U-Net.  Default: (16, 32, 64, 128) -> depth 4,
        lightweight (~0.482M params), well within RTX 3050 6 GB budget.
    kernel_size : int
        Kernel size for 2D convolutions.
    norm_type : str
        Normalization type: 'batchnorm', 'instancenorm', or 'none'.
    dropout : float
        Dropout probability in convolutional blocks.
    output_mode : str
        'mask': predict an attenuation/gain mask applied to input (M * X).
        'direct': predict clean magnitude directly (ReLU output).
    mask_act : str
        Activation for mask output: 'sigmoid' in (0, 1), 'relu' in [0, inf),
        or 'softplus' in (0, inf).  Ignored if output_mode is 'direct'.
    """

    in_channels: int = 1
    out_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64, 128)
    kernel_size: int = 3
    norm_type: str = "batchnorm"
    dropout: float = 0.0
    output_mode: str = "mask"
    mask_act: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if self.out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if not self.channels:
            raise ValueError("channels must not be empty")
        if self.norm_type not in ("batchnorm", "instancenorm", "none"):
            raise ValueError(f"Unsupported norm_type: {self.norm_type!r}")
        if self.output_mode not in ("mask", "direct"):
            raise ValueError(f"Unsupported output_mode: {self.output_mode!r}")
        if self.mask_act not in ("sigmoid", "relu", "softplus"):
            raise ValueError(f"Unsupported mask_act: {self.mask_act!r}")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")


def _make_norm(norm_type: str, channels: int) -> nn.Module:
    """Create normalization layer."""
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(channels)
    if norm_type == "instancenorm":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


class DoubleConv(nn.Module):
    """Two successive [Conv2D -> Norm -> LeakyReLU] layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_type: str = "batchnorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        use_bias = norm_type == "none"

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=use_bias),
            _make_norm(norm_type, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=use_bias),
            _make_norm(norm_type, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool2D -> DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_type: str = "batchnorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, norm_type, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling block: ConvTranspose2D -> Concatenate Skip -> DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_type: str = "batchnorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, norm_type, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle any 1-pixel rounding mismatch if odd dimensions occurred
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, (0, diff_w, 0, diff_h), mode="replicate")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """2D U-Net for magnitude spectrogram enhancement.

    Features:
    - Automatically computes padding to the nearest multiple of 2^depth.
    - Preserves all frequency bins (e.g. 257) and time frames via unpad slice.
    - Supports ratio-masking (sigmoid, relu, softplus) or direct magnitude estimation.
    - Fully configurable channels, depth, norm, and dropout.
    """

    def __init__(self, config: UNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or UNetConfig()
        channels = self.config.channels
        depth = len(channels)
        self.divisor = 2 ** depth

        # Initial double-convolution
        self.inc = DoubleConv(
            self.config.in_channels,
            channels[0],
            self.config.kernel_size,
            self.config.norm_type,
            self.config.dropout,
        )

        # Encoder down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(depth - 1):
            self.down_blocks.append(
                DownBlock(
                    channels[i],
                    channels[i + 1],
                    self.config.kernel_size,
                    self.config.norm_type,
                    self.config.dropout,
                )
            )

        # Decoder up blocks (reversed channels)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_blocks.append(
                UpBlock(
                    channels[i + 1],
                    channels[i],
                    self.config.kernel_size,
                    self.config.norm_type,
                    self.config.dropout,
                )
            )

        # Final 1x1 conv to output channels
        self.out_conv = nn.Conv2d(channels[0], self.config.out_channels, kernel_size=1)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad frequency and time dimensions to multiples of 2^depth.

        Returns (padded_x, pad_freq, pad_time).
        """
        orig_h, orig_w = x.shape[2], x.shape[3]
        pad_h = (self.divisor - (orig_h % self.divisor)) % self.divisor
        pad_w = (self.divisor - (orig_w % self.divisor)) % self.divisor

        if pad_h == 0 and pad_w == 0:
            return x, 0, 0

        # Use replicate padding when dimensions allow, fallback to constant (zero)
        mode = "replicate" if (orig_h > 1 and orig_w > 1) else "constant"
        padded_x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return padded_x, pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Parameters
        ----------
        x : Tensor
            Input spectrogram tensor of shape ``(B, C, F, N)``, typically
            ``(B, 1, 257, N)`` for magnitude spectrograms.

        Returns
        -------
        Tensor
            Enhanced spectrogram of exact shape ``(B, out_channels, F, N)``.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input tensor (B, C, F, N), got {x.dim()}-D")

        orig_f, orig_n = x.shape[2], x.shape[3]
        x_padded, pad_f, pad_n = self._pad(x)

        # Encoder
        skips: list[torch.Tensor] = []
        feat = self.inc(x_padded)
        skips.append(feat)

        for down in self.down_blocks:
            feat = down(feat)
            skips.append(feat)

        # Bottleneck is top of skips stack; remove it for decoder pairing
        feat = skips.pop()

        # Decoder
        for up in self.up_blocks:
            skip = skips.pop()
            feat = up(feat, skip)

        raw_out = self.out_conv(feat)

        # Output activation / masking
        if self.config.output_mode == "mask":
            if self.config.mask_act == "sigmoid":
                mask = torch.sigmoid(raw_out)
            elif self.config.mask_act == "relu":
                mask = F.relu(raw_out)
            elif self.config.mask_act == "softplus":
                mask = F.softplus(raw_out)
            else:
                raise ValueError(f"Unknown mask_act: {self.config.mask_act!r}")
            out_padded = x_padded * mask
        elif self.config.output_mode == "direct":
            out_padded = F.relu(raw_out)
        else:
            raise ValueError(f"Unknown output_mode: {self.config.output_mode!r}")

        # Crop back to original dimensions (unpad)
        return out_padded[:, :, :orig_f, :orig_n]
