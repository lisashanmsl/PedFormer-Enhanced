import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedSAIM(nn.Module):
    """光流與 SAM 增強型語義注意力交互模組 (Enhanced SAIM)。

    核心改良:
    1. 將行人軌跡 + 自車動態 + 光流特徵編碼為「動態查詢向量」(Dynamic Query)
    2. 將 SAM 分割出的場景特徵作為「鍵值對」(Key-Value)
    3. 透過全域注意力機制進行加權融合

    使 SAIM 同時理解:
    - 「物體是什麼」(SAM 語義)
    - 「物體如何移動」(RAFT 光流)
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, dropout: float = 0.1):
        super().__init__()

        # 動態查詢投影: 將 traj + ego + flow 融合後投影為 query
        self.query_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Cross-Attention: Query=動態特徵, Key/Value=SAM場景特徵
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # 融合後 FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Softmax 門控: 動態決定語義 vs 動態特徵的權重
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        traj_feat: torch.Tensor,
        ego_feat: torch.Tensor,
        flow_feat: torch.Tensor,
        scene_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            traj_feat:  [batch, seq_len, d_model] — 軌跡嵌入
            ego_feat:   [batch, seq_len, d_model] — 自車嵌入
            flow_feat:  [batch, seq_len, d_model] — 光流嵌入
            scene_feat: [batch, num_patches, d_model] — SAM 場景嵌入

        Returns:
            fused: [batch, seq_len, d_model] — 融合後的 SAIM 特徵
        """
        # Step 1: 建立動態查詢 (Dynamic Query)
        # 將 traj, ego, flow 串接後投影
        dynamic_concat = torch.cat([traj_feat, ego_feat, flow_feat], dim=-1)
        dynamic_query = self.query_proj(dynamic_concat)  # [batch, seq, d_model]

        # Step 2: Cross-Attention (Query=動態, Key/Value=場景)
        attn_out, _ = self.cross_attention(
            query=dynamic_query,
            key=scene_feat,
            value=scene_feat,
        )
        attn_out = self.norm1(dynamic_query + attn_out)  # 殘差連接

        # Step 3: 門控融合 (Gated Fusion)
        # 動態決定最終輸出中語義與動態特徵的比例
        gate_input = torch.cat(
            [attn_out, flow_feat], dim=-1
        )  # [batch, seq, d_model*2]
        gate_weight = self.gate(gate_input)  # [batch, seq, d_model], 0~1
        gated_out = gate_weight * attn_out + (1 - gate_weight) * flow_feat

        # Step 4: FFN + 殘差
        output = self.norm2(gated_out + self.ffn(gated_out))

        return output
