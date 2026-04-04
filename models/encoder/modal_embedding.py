import torch
import torch.nn as nn


class ModalityEmbedding(nn.Module):
    """將單一模態的原始輸入映射至統一的 d_model 維度空間。

    支援的模態:
      - traj:  行人歷史軌跡 bbox [batch, seq, 4]
      - ego:   自車運動     [batch, seq, 2]
      - flow:  光流特徵     [batch, seq, flow_dim]
      - scene: SAM 語義特徵 [batch, num_patches, sam_dim]
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiModalEmbedding(nn.Module):
    """管理所有模態的 Embedding 層。"""

    def __init__(
        self,
        d_model: int = 128,
        traj_dim: int = 4,
        ego_dim: int = 2,
        flow_dim: int = 256,
        sam_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.traj_emb = ModalityEmbedding(traj_dim, d_model, dropout)
        self.ego_emb = ModalityEmbedding(ego_dim, d_model, dropout)
        self.flow_emb = ModalityEmbedding(flow_dim, d_model, dropout)
        self.scene_emb = ModalityEmbedding(sam_dim, d_model, dropout)

        # 模態類型嵌入 (Modality Type Token)，區分不同來源
        self.modality_tokens = nn.Embedding(4, d_model)  # 0:traj 1:ego 2:flow 3:scene

    def forward(
        self,
        traj: torch.Tensor,
        ego: torch.Tensor,
        flow_feat: torch.Tensor,
        scene_feat: torch.Tensor,
    ) -> torch.Tensor:
        """將所有模態嵌入並在序列維度上串接。

        Returns:
            combined: [batch, total_seq_len, d_model]
        """
        device = traj.device
        e_traj = self.traj_emb(traj) + self.modality_tokens(
            torch.zeros(traj.size(1), dtype=torch.long, device=device)
        )
        e_ego = self.ego_emb(ego) + self.modality_tokens(
            torch.ones(ego.size(1), dtype=torch.long, device=device)
        )
        e_flow = self.flow_emb(flow_feat) + self.modality_tokens(
            torch.full((flow_feat.size(1),), 2, dtype=torch.long, device=device)
        )
        e_scene = self.scene_emb(scene_feat) + self.modality_tokens(
            torch.full((scene_feat.size(1),), 3, dtype=torch.long, device=device)
        )

        combined = torch.cat([e_traj, e_ego, e_flow, e_scene], dim=1)
        return combined
