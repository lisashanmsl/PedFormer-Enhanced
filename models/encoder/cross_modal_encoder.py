import torch
import torch.nn as nn

from models.encoder.positional_encoding import SinusoidalPositionalEncoding
from models.encoder.modal_embedding import MultiModalEmbedding


class CrossModalTransformerEncoder(nn.Module):
    """跨模態 Transformer 編碼器。

    保留 PedFormer 的核心設計:
    - 多模態嵌入 (traj, ego, flow, scene)
    - 位置編碼 (Sinusoidal)
    - Multi-Head Self-Attention 在所有模態 token 上進行交互
    - 輸出融合後的時空特徵向量 Z_encoded
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        traj_dim: int = 4,
        ego_dim: int = 2,
        flow_dim: int = 256,
        sam_dim: int = 256,
    ):
        super().__init__()

        self.embedding = MultiModalEmbedding(
            d_model=d_model,
            traj_dim=traj_dim,
            ego_dim=ego_dim,
            flow_dim=flow_dim,
            sam_dim=sam_dim,
            dropout=dropout,
        )

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        traj: torch.Tensor,
        ego: torch.Tensor,
        flow_feat: torch.Tensor,
        scene_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            traj:       [batch, obs_len, 4]
            ego:        [batch, obs_len, 2]
            flow_feat:  [batch, obs_len, flow_dim]
            scene_feat: [batch, num_patches, sam_dim]

        Returns:
            Z_encoded: [batch, total_seq_len, d_model]
        """
        # 多模態嵌入 → 串接
        combined = self.embedding(traj, ego, flow_feat, scene_feat)

        # 加入位置編碼
        combined = self.pos_encoding(combined)

        # Transformer Self-Attention 跨模態交互
        encoded = self.transformer_encoder(combined)

        # 輸出正規化
        encoded = self.output_norm(encoded)

        return encoded
