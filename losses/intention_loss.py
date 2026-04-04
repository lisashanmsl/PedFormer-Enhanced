import torch
import torch.nn as nn


class IntentionLoss(nn.Module):
    """意圖預測損失: 結合全局 BCE 與逐步 BCE。

    L_intent = BCE(global_pred, label) + λ_step * BCE(step_preds, label_expanded)
    """

    def __init__(self, step_weight: float = 0.3):
        super().__init__()
        self.bce = nn.BCELoss()
        self.step_weight = step_weight

    def forward(
        self,
        global_intent: torch.Tensor,
        step_intents: torch.Tensor,
        target_intent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            global_intent: [batch, 1] — 全局穿越機率
            step_intents:  [batch, pred_len, 1] — 逐步穿越機率
            target_intent: [batch, 1] — 真實標籤 (0 or 1)
        """
        # 全局意圖 BCE
        loss_global = self.bce(global_intent, target_intent)

        # 逐步意圖 BCE (將標籤擴展至每個時間步)
        pred_len = step_intents.size(1)
        target_expanded = target_intent.unsqueeze(1).expand(-1, pred_len, -1)
        loss_step = self.bce(step_intents, target_expanded)

        return loss_global + self.step_weight * loss_step
