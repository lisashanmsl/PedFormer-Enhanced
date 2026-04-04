import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class FlowEncoder(nn.Module):
    """將 RAFT 輸出的光流場編碼為固定維度的特徵向量 φ_flow。

    Pipeline:
        光流場 [batch, seq, 2, H, W]
        → Conv 調整通道 (2→3)
        → ResNet-50 backbone (去除最後 fc)
        → 全局平均池化
        → 投影至 flow_feature_dim

    每個時間步獨立編碼，輸出 [batch, seq, flow_feature_dim]。
    """

    def __init__(self, flow_feature_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()

        # 2 channels (u, v) → 3 channels for ResNet
        self.channel_adapter = nn.Conv2d(2, 3, kernel_size=1, bias=False)

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 移除最後全連接層和平均池化
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ResNet-50 最後一層輸出 2048 維
        self.proj = nn.Sequential(
            nn.Linear(2048, flow_feature_dim),
            nn.LayerNorm(flow_feature_dim),
            nn.GELU(),
        )

    def encode_single(self, flow: torch.Tensor) -> torch.Tensor:
        """編碼單一時間步的光流。

        Args:
            flow: [batch, 2, H, W]

        Returns:
            feat: [batch, flow_feature_dim]
        """
        x = self.channel_adapter(flow)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)  # [batch, 2048]
        return self.proj(x)

    def forward(self, flow_sequence: torch.Tensor) -> torch.Tensor:
        """編碼光流序列。

        Args:
            flow_sequence: [batch, seq_len, 2, H, W]

        Returns:
            flow_features: [batch, seq_len, flow_feature_dim]
        """
        batch, seq_len = flow_sequence.shape[:2]
        feats = []
        for t in range(seq_len):
            feat = self.encode_single(flow_sequence[:, t])
            feats.append(feat)
        return torch.stack(feats, dim=1)
