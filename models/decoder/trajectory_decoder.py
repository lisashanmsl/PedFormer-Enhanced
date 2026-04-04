import torch
import torch.nn as nn


class TrajectoryDecoder(nn.Module):
    """LSTM 軌跡解碼器 (參考 PTINet 設計)。

    遞歸生成未來 τ 步的 2D 座標位置:
    - 以編碼器融合特徵初始化 LSTM 隱藏狀態
    - 每步接受前一步預測座標作為輸入 (autoregressive)
    - 輸出: [batch, pred_len, 2] — 未來軌跡座標序列
    """

    def __init__(
        self,
        d_model: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        pred_len: int = 45,
        output_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # 將編碼器特徵投影至 LSTM 隱藏狀態維度
        self.hidden_init = nn.Linear(d_model, lstm_hidden_dim * lstm_num_layers)
        self.cell_init = nn.Linear(d_model, lstm_hidden_dim * lstm_num_layers)

        # LSTM 輸入: 前一步座標 (2D) + 編碼器特徵 (d_model)
        self.lstm = nn.LSTM(
            input_size=output_dim + d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # 輸出投影: 隱藏狀態 → 2D 座標
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def _init_hidden(
        self, encoder_feat: torch.Tensor
    ) -> tuple:
        """從編碼器全局特徵初始化 LSTM 隱藏狀態。

        Args:
            encoder_feat: [batch, d_model]

        Returns:
            (h_0, c_0): 各為 [num_layers, batch, hidden_dim]
        """
        batch = encoder_feat.size(0)

        h = self.hidden_init(encoder_feat)  # [batch, hidden * layers]
        c = self.cell_init(encoder_feat)

        h = h.view(batch, self.lstm_num_layers, self.lstm_hidden_dim)
        c = c.view(batch, self.lstm_num_layers, self.lstm_hidden_dim)

        h = h.permute(1, 0, 2).contiguous()  # [layers, batch, hidden]
        c = c.permute(1, 0, 2).contiguous()

        return h, c

    def forward(
        self,
        encoder_feat: torch.Tensor,
        last_obs_pos: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_feat: [batch, d_model] — 全局編碼特徵 (用於初始化隱藏狀態)
            last_obs_pos: [batch, 2] — 最後觀察到的座標位置
            context:      [batch, d_model] — 每步共享的上下文特徵

        Returns:
            pred_traj: [batch, pred_len, 2]
        """
        h, c = self._init_hidden(encoder_feat)

        current_pos = last_obs_pos  # [batch, 2]
        outputs = []

        for _ in range(self.pred_len):
            # LSTM 輸入: 當前座標 + 上下文
            lstm_input = torch.cat([current_pos, context], dim=-1)
            lstm_input = lstm_input.unsqueeze(1)  # [batch, 1, input_dim]

            out, (h, c) = self.lstm(lstm_input, (h, c))

            # 預測下一步座標偏移
            delta = self.output_proj(out.squeeze(1))  # [batch, 2]
            current_pos = current_pos + delta  # 殘差預測
            outputs.append(current_pos)

        return torch.stack(outputs, dim=1)  # [batch, pred_len, 2]
