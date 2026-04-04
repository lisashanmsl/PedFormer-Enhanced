import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExtractor(nn.Module):
    """從 SAM 分割遮罩中提取 Patch 特徵。

    Pipeline:
        SAM 遮罩 [batch, max_objects, H, W]
        → 乘上原始影像 → 逐物體 masked region
        → CNN 提取局部特徵
        → 全局池化 → Scene Embedding [batch, num_patches, sam_dim]
    """

    def __init__(self, sam_feature_dim: int = 256, num_patches: int = 16):
        super().__init__()
        self.num_patches = num_patches
        self.sam_feature_dim = sam_feature_dim

        # 輕量 CNN 從 masked patch 提取特徵
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Sequential(
            nn.Linear(128, sam_feature_dim),
            nn.LayerNorm(sam_feature_dim),
            nn.GELU(),
        )

    def forward(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: [batch, 3, H, W] — 原始 RGB 影像 (normalized)
            masks:  [batch, num_patches, H, W] — SAM 二值遮罩

        Returns:
            scene_features: [batch, num_patches, sam_feature_dim]
        """
        batch, num_masks, h, w = masks.shape
        num_patches = min(num_masks, self.num_patches)

        # 將影像尺寸與遮罩對齊
        if images.shape[2] != h or images.shape[3] != w:
            images = F.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)

        all_feats = []
        for i in range(num_patches):
            # 將遮罩擴展為 [batch, 1, H, W] 並乘上影像
            mask_i = masks[:, i : i + 1, :, :]  # [batch, 1, H, W]
            masked_img = images * mask_i  # [batch, 3, H, W]

            feat = self.patch_cnn(masked_img).flatten(1)  # [batch, 128]
            feat = self.proj(feat)  # [batch, sam_feature_dim]
            all_feats.append(feat)

        # 若 patches 數量不足，用零向量填充
        while len(all_feats) < self.num_patches:
            all_feats.append(
                torch.zeros(batch, self.sam_feature_dim, device=images.device)
            )

        return torch.stack(all_feats, dim=1)  # [batch, num_patches, sam_feature_dim]
