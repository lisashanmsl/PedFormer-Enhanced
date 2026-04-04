import torch
import torch.nn as nn


class IntentionDecoder(nn.Module):
    """LSTM 意圖解碼器 (參考 PTINet 設計)。

    預測未來每個時間步的穿越機率:
    - 以編碼器融合特徵初始化 LSTM 隱藏狀態
    - 每步輸出一個 [0, 1] 之間的穿越機率
    - 輸出: [batch, pred_len, 1] — 逐步穿越機率

    也支援輸出單一全局意圖: [batch, 1]
    """

    def __init__(
        self,
        d_model: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        pred_len: int = 45,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # 隱藏狀態初始化
        self.hidden_init = nn.Linear(d_model, lstm_hidden_dim * lstm_num_layers)
        self.cell_init = nn.Linear(d_model, lstm_hidden_dim * lstm_num_layers)

        # LSTM 輸入: 前一步意圖 (1) + 上下文 (d_model)
        self.lstm = nn.LSTM(
            input_size=1 + d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # 輸出: 穿越機率
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 全局意圖分類頭 (用於 BCE loss)
        self.global_intent_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _init_hidden(self, encoder_feat: torch.Tensor) -> tuple:
        batch = encoder_feat.size(0)
        h = self.hidden_init(encoder_feat).view(
            batch, self.lstm_num_layers, self.lstm_hidden_dim
        ).permute(1, 0, 2).contiguous()
        c = self.cell_init(encoder_feat).view(
            batch, self.lstm_num_layers, self.lstm_hidden_dim
        ).permute(1, 0, 2).contiguous()
        return h, c

    def forward(
        self,
        encoder_feat: torch.Tensor,
        context: torch.Tensor,
        initial_intent: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            encoder_feat:   [batch, d_model]
            context:        [batch, d_model]
            initial_intent: [batch, 1] 或 None (預設 0.5)

        Returns:
            step_intents: [batch, pred_len, 1] — 逐步穿越機率
            global_intent: [batch, 1] — 全局穿越機率 (0~1)
        """
        h, c = self._init_hidden(encoder_feat)
        batch = encoder_feat.size(0)

        if initial_intent is None:
            current_intent = torch.full((batch, 1), 0.5, device=encoder_feat.device)
        else:
            current_intent = initial_intent

        step_outputs = []
        for _ in range(self.pred_len):
            lstm_input = torch.cat([current_intent, context], dim=-1)
            lstm_input = lstm_input.unsqueeze(1)

            out, (h, c) = self.lstm(lstm_input, (h, c))

            intent_logit = self.output_proj(out.squeeze(1))  # [batch, 1]
            current_intent = torch.sigmoid(intent_logit)
            step_outputs.append(current_intent)

        step_intents = torch.stack(step_outputs, dim=1)  # [batch, pred_len, 1]

        # 全局意圖: 取最後隱藏狀態
        global_logit = self.global_intent_head(h[-1])  # [batch, 1]
        global_intent = torch.sigmoid(global_logit)

        return step_intents, global_intent
