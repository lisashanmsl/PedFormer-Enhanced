import torch
import torch.nn as nn

from losses.trajectory_loss import TrajectoryLoss
from losses.intention_loss import IntentionLoss


class MultiTaskLoss(nn.Module):
    """多任務聯合損失函數。

    L_total = w_traj * L_traj + w_intent * L_intent

    支援可學習權重 (Uncertainty Weighting) 或固定權重。
    """

    def __init__(
        self,
        w_traj: float = 1.0,
        w_intent: float = 0.5,
        fde_weight: float = 0.5,
        step_weight: float = 0.3,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.traj_loss = TrajectoryLoss(fde_weight=fde_weight)
        self.intent_loss = IntentionLoss(step_weight=step_weight)

        self.learnable_weights = learnable_weights
        if learnable_weights:
            # 可學習的 log(σ²) 用於 Uncertainty Weighting
            self.log_var_traj = nn.Parameter(torch.zeros(1))
            self.log_var_intent = nn.Parameter(torch.zeros(1))
        else:
            self.w_traj = w_traj
            self.w_intent = w_intent

    def forward(
        self,
        pred_traj: torch.Tensor,
        target_traj: torch.Tensor,
        global_intent: torch.Tensor,
        step_intents: torch.Tensor,
        target_intent: torch.Tensor,
    ) -> dict:
        """
        Returns:
            dict with keys: 'total', 'traj', 'intent'
        """
        l_traj = self.traj_loss(pred_traj, target_traj)
        l_intent = self.intent_loss(global_intent, step_intents, target_intent)

        if self.learnable_weights:
            # Kendall et al. (2018) Multi-Task Learning Using Uncertainty
            precision_traj = torch.exp(-self.log_var_traj)
            precision_intent = torch.exp(-self.log_var_intent)
            total = (
                precision_traj * l_traj
                + self.log_var_traj
                + precision_intent * l_intent
                + self.log_var_intent
            )
        else:
            total = self.w_traj * l_traj + self.w_intent * l_intent

        return {
            "total": total,
            "traj": l_traj.detach(),
            "intent": l_intent.detach(),
        }
