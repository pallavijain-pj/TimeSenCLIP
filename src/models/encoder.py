import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class TimeSenCLIPEncoder(nn.Module):
    """
    TimeSenCLIP: Temporal-Spectral Transformer for Remote Sensing Time Series

    Args:
        image_size (int): Spatial size of input images (assumed square).
        dim (int): Embedding dimension.
        depth (int): Number of transformer encoder layers.
        heads (int): Number of attention heads.
        mlp_dim (int): Feedforward network dimension in transformer.
        time_frames (int): Number of temporal frames (e.g., months).
        spectral_bands (int): Number of spectral bands.
        dim_head (int): Dimension per attention head.
        dropout (float): Dropout rate.
        dropout_type (str): Temporal/spectral dropout strategy. One of ['None', 'RandomTS', 'TSMixAug', 'TSMS'].
    """

    def __init__(
        self, image_size=1, dim=512, depth=6, heads=8, mlp_dim=256,
        time_frames=12, spectral_bands=10, dim_head=64,
        dropout=0.1, dropout_type='TSMixAug'
    ):
        super().__init__()

        self.time_frames = time_frames
        self.spectral_bands = spectral_bands
        self.dim = dim
        self.dropout_type = dropout_type

        patch_dim = spectral_bands * image_size * image_size

        self.spectral_embedding = nn.Sequential(
            Rearrange('b t c h w -> b t (c h w)'),
            nn.Linear(patch_dim, dim),
        )

        self.time_pos_encoding = nn.Parameter(torch.zeros(1, time_frames, dim))
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        """
        Forward pass during training.
        Args:
            x (Tensor): Input tensor of shape (B, T, S, H, W)
        Returns:
            Tensor: Output embedding (B, dim)
        """
        B, T, S, H, W = x.shape
        assert T == self.time_frames, f"Expected T={self.time_frames}, but got {T}"

        # Spectral dropout (TSMS only)
        if self.dropout_type == 'TSMS':
            spectral_mask = self._spectral_dropout(S)
            x = x[:, :, spectral_mask]
        else:
            spectral_mask = torch.ones(S, dtype=torch.bool)

        x = self.spectral_embedding(x)  # Shape: (B, T, dim)

        # Temporal dropout
        if self.dropout_type != 'None' and torch.rand(1).item() > 0.5:
            if self.dropout_type == 'TSMixAug':
                x = self._tsmixaug(x)
            elif self.dropout_type in ['RandomTS', 'TSMS']:
                mask = self._random_temporal_dropout(x)
                x = x[:, mask]
        mask = torch.ones(x.size(1), dtype=torch.bool)  # default mask

        # Positional encoding
        time_pos = self.time_pos_encoding.expand(B, -1, -1)
        if self.training:
            if x.size(1) > 1:
                time_pos = time_pos[:, :x.size(1)]
            else:
                time_pos = time_pos.mean(dim=1, keepdim=True)
        x = x + time_pos

        # Add class token
        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([class_token.to(x.dtype), x], dim=1)

        # Transformer
        x = self.transformer(x)

        # Project output
        x = self.to_latent(x[:, 0])
        return self.mlp_head(x)

    def inference(self, x):
        """
        Forward pass without dropout for inference.
        Args:
            x (Tensor): Input tensor of shape (B, T, S, H, W)
        Returns:
            Tensor: Output embedding (B, dim)
        """
        B, T, S, H, W = x.shape
        x = self.spectral_embedding(x)

        b, n, d = x.shape
        time_pos = self.time_pos_encoding.expand(b, -1, -1)

        if n == 1:
            time_pos = time_pos.mean(dim=1, keepdim=True)
        elif n == 4:
            time_pos = time_pos.view(b, 4, 3, d).median(dim=2).values

        x += time_pos

        class_token = self.class_token.expand(b, -1, -1)
        x = torch.cat([class_token.to(x.dtype), x], dim=1)

        x = self.transformer(x)
        x = self.to_latent(x[:, 0])
        return self.mlp_head(x)

    # ------------------------ Augmentation Methods ------------------------

    def _random_temporal_dropout(self, x, min_frames=1, max_frames=12):
        """
        Randomly select a subset of temporal frames to keep.
        """
        total = x.shape[1]
        num_to_keep = random.randint(min_frames, min(max_frames, total))
        selected_indices = torch.randperm(total)[:num_to_keep]
        mask = torch.zeros(total, dtype=torch.bool)
        mask[selected_indices] = True
        return mask

    def _quarter_drop(self, x):
        """
        Drop a random quarter (3 consecutive frames) from the sequence.
        """
        total = x.shape[1]
        quarter = 3
        start = random.randint(0, total - quarter)
        mask = torch.ones(total, dtype=torch.bool)
        mask[start:start + quarter] = False
        return mask

    def _spectral_dropout(self, S, min_bands=3):
        """
        Randomly drop spectral bands, retaining the first 3.
        """
        if torch.rand(1).item() <= 0.3:
            return torch.ones(S, dtype=torch.bool)

        num_to_keep = random.randint(min_bands, S)
        mask = torch.zeros(S, dtype=torch.bool)
        mask[:3] = True  # Always keep first 3
        remaining = torch.randperm(S - 3)[:num_to_keep - 3]
        mask[3:][remaining] = True
        return mask

    def _tsmixaug(self, x):
        """
        Temporal augmentation strategy: pooling or quarter-drop.
        """
        if torch.rand(1).item() > 0.5:
            return torch.median(x, dim=1, keepdim=True).values
        else:
            mask = self._quarter_drop(x)
            return x[:, mask]
