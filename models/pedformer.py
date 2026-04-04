import torch
import torch.nn as nn

from models.encoder.cross_modal_encoder import CrossModalTransformerEncoder
from models.saim.enhanced_saim import EnhancedSAIM
from models.decoder.trajectory_decoder import TrajectoryDecoder
from models.decoder.intention_decoder import IntentionDecoder
from models.encoder.modal_embedding import ModalityEmbedding


class PedFormerEnhanced(nn.Module):
    """PedFormer-Enhanced: 結合 PedFormer 跨模態編碼 + PTINet 光流/LSTM 解碼。

    整體架構:
        1. 多模態輸入 → Cross-Modal Transformer Encoder
        2. Enhanced SAIM (光流 + SAM 動態語義融合)
        3. 雙流 LSTM 解碼器 (軌跡 + 意圖)

    輸入:
        - traj: [batch, obs_len, 4]  行人歷史 bbox
        - ego:  [batch, obs_len, 2]  自車運動 (速度, 轉向)
        - flow_feat: [batch, obs_len, flow_dim]  RAFT 光流特徵 (預計算)
        - scene_feat: [batch, num_patches, sam_dim]  SAM 場景特徵 (預計算)

    輸出:
        - pred_traj: [batch, pred_len, 2]  未來軌跡
        - step_intents: [batch, pred_len, 1]  逐步穿越機率
        - global_intent: [batch, 1]  全局穿越意圖
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        traj_dim: int = 4,
        ego_dim: int = 2,
        flow_dim: int = 256,
        sam_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        pred_len: int = 45,
        num_patches: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len

        # ========== 1. Cross-Modal Transformer Encoder ==========
        self.encoder = CrossModalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            traj_dim=traj_dim,
            ego_dim=ego_dim,
            flow_dim=flow_dim,
            sam_dim=sam_dim,
        )

        # ========== 2. Enhanced SAIM ==========
        # SAIM 需要各模態的獨立嵌入作為輸入
        self.saim_traj_emb = ModalityEmbedding(traj_dim, d_model, dropout)
        self.saim_ego_emb = ModalityEmbedding(ego_dim, d_model, dropout)
        self.saim_flow_emb = ModalityEmbedding(flow_dim, d_model, dropout)
        self.saim_scene_emb = ModalityEmbedding(sam_dim, d_model, dropout)

        self.enhanced_saim = EnhancedSAIM(
            d_model=d_model, nhead=nhead, dropout=dropout
        )

        # ========== 3. 特徵融合層 ==========
        # 將 Encoder 輸出 + SAIM 輸出融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ========== 4. 雙流 LSTM 解碼器 ==========
        self.trajectory_decoder = TrajectoryDecoder(
            d_model=d_model,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            pred_len=pred_len,
            output_dim=2,
            dropout=dropout,
        )

        self.intention_decoder = IntentionDecoder(
            d_model=d_model,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            pred_len=pred_len,
            dropout=dropout,
        )

    def forward(
        self,
        traj: torch.Tensor,
        ego: torch.Tensor,
        flow_feat: torch.Tensor,
        scene_feat: torch.Tensor,
    ) -> dict:
        """
        Args:
            traj:       [batch, obs_len, 4]
            ego:        [batch, obs_len, 2]
            flow_feat:  [batch, obs_len, flow_dim]
            scene_feat: [batch, num_patches, sam_dim]

        Returns:
            dict with:
                'pred_traj':     [batch, pred_len, 2]
                'step_intents':  [batch, pred_len, 1]
                'global_intent': [batch, 1]
        """
        # ---- 1. Cross-Modal Transformer Encoder ----
        encoder_out = self.encoder(
            traj, ego, flow_feat, scene_feat
        )  # [batch, total_seq, d_model]

        # 全局特徵 (Global Average Pooling)
        encoder_global = encoder_out.mean(dim=1)  # [batch, d_model]

        # ---- 2. Enhanced SAIM ----
        saim_traj = self.saim_traj_emb(traj)
        saim_ego = self.saim_ego_emb(ego)
        saim_flow = self.saim_flow_emb(flow_feat)
        saim_scene = self.saim_scene_emb(scene_feat)

        saim_out = self.enhanced_saim(
            saim_traj, saim_ego, saim_flow, saim_scene
        )  # [batch, obs_len, d_model]

        saim_global = saim_out.mean(dim=1)  # [batch, d_model]

        # ---- 3. 融合 Encoder + SAIM ----
        fused = self.fusion(
            torch.cat([encoder_global, saim_global], dim=-1)
        )  # [batch, d_model]

        # ---- 4. 雙流 LSTM 解碼 ----
        # 最後觀察位置 (取 bbox 中心: (x1+x2)/2, (y1+y2)/2)
        last_bbox = traj[:, -1, :]  # [batch, 4]
        last_pos = torch.stack(
            [(last_bbox[:, 0] + last_bbox[:, 2]) / 2,
             (last_bbox[:, 1] + last_bbox[:, 3]) / 2],
            dim=-1,
        )  # [batch, 2]

        pred_traj = self.trajectory_decoder(
            encoder_feat=fused, last_obs_pos=last_pos, context=fused
        )

        step_intents, global_intent = self.intention_decoder(
            encoder_feat=fused, context=fused
        )

        return {
            "pred_traj": pred_traj,
            "step_intents": step_intents,
            "global_intent": global_intent,
        }


if __name__ == "__main__":
    print("測試 PedFormerEnhanced 模型...")
    model = PedFormerEnhanced(
        d_model=128, nhead=8, num_encoder_layers=4,
        flow_dim=256, sam_dim=256, pred_len=45,
    )

    batch = 2
    obs_len = 16
    dummy_traj = torch.randn(batch, obs_len, 4)
    dummy_ego = torch.randn(batch, obs_len, 2)
    dummy_flow = torch.randn(batch, obs_len, 256)
    dummy_scene = torch.randn(batch, 16, 256)

    output = model(dummy_traj, dummy_ego, dummy_flow, dummy_scene)

    print(f"pred_traj:     {output['pred_traj'].shape}")
    print(f"step_intents:  {output['step_intents'].shape}")
    print(f"global_intent: {output['global_intent'].shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")
