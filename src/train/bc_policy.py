"""Behavioral Cloning policy network.

Simple CNN encoder + MLP head for image-conditioned action prediction.
Designed for LeRobot-format data: multi-camera RGB + proprioceptive state -> action (7D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Lightweight CNN encoder for a single camera image."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(64 * 4 * 4, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class BCPolicy(nn.Module):
    """Behavioral cloning policy: multi-camera images + state -> action.

    Architecture:
        - Per-camera CNN encoders (shared or independent)
        - Concatenate image features + proprioceptive state
        - MLP head outputs action
    """

    def __init__(
        self,
        num_cameras: int = 3,
        image_feature_dim: int = 128,
        state_dim: int = 7,
        action_dim: int = 7,
        hidden_dim: int = 256,
        share_encoder: bool = True,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.share_encoder = share_encoder
        self.action_dim = action_dim

        if share_encoder:
            self.image_encoder = ImageEncoder(in_channels=3, feature_dim=image_feature_dim)
        else:
            self.image_encoders = nn.ModuleList([
                ImageEncoder(in_channels=3, feature_dim=image_feature_dim)
                for _ in range(num_cameras)
            ])

        fused_dim = num_cameras * image_feature_dim + state_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: list of (B, 3, H, W) tensors, one per camera.
            state: (B, state_dim) proprioceptive state.

        Returns:
            action: (B, action_dim) predicted action.
        """
        feats = []
        for i, img in enumerate(images):
            if self.share_encoder:
                feats.append(self.image_encoder(img))
            else:
                feats.append(self.image_encoders[i](img))

        h = torch.cat(feats + [state], dim=-1)
        return self.mlp(h)

    def predict(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """Inference-time prediction (no grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward(images, state)
