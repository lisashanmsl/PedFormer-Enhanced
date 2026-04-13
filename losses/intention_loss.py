import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentionLoss(nn.Module):
    """意圖預測損失: 結合全局 BCE 與逐步 BCE。

    L_intent = weighted_BCE(global_pred, label) + λ_step * weighted_BCE(step_preds, label_expanded)

    使用 pos_weight 對正類別（crossing）加權，解決類別不平衡問題。
    資料集中約 85% 為 not-crossing，pos_weight=5.0 讓模型更重視 crossing 樣本。
    """

    def __init__(self, step_weight: float = 0.3, pos_weight: float = 5.0):
        super().__init__()
        self.step_weight = step_weight
        self.pos_weight = pos_weight

    def _weighted_bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """對正類別加權的 BCE，輸入已經過 sigmoid。"""
        weight = torch.where(target >= 0.5, self.pos_weight, 1.0)
        return F.binary_cross_entropy(pred, target, weight=weight)

    def forward(
        self,
        global_intent: torch.Tensor,
        step_intents: torch.Tensor,
        target_intent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            global_intent: [batch, 1] — 全局穿越機率 (已過 sigmoid)
            step_intents:  [batch, pred_len, 1] — 逐步穿越機率 (已過 sigmoid)
            target_intent: [batch, 1] — 真實標籤 (0 or 1)
        """
        # 全局意圖 weighted BCE
        loss_global = self._weighted_bce(global_intent, target_intent)

        # 逐步意圖 weighted BCE (將標籤擴展至每個時間步)
        pred_len = step_intents.size(1)
        target_expanded = target_intent.unsqueeze(1).expand(-1, pred_len, -1)
        loss_step = self._weighted_bce(step_intents, target_expanded)

        return loss_global + self.step_weight * loss_step
