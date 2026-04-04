import torch
import torch.nn as nn


class TrajectoryLoss(nn.Module):
    """軌跡預測損失: 結合 ADE-based MSE 與 FDE 加權。

    L_traj = MSE(pred, target) + λ_fde * MSE(pred[-1], target[-1])
    """

    def __init__(self, fde_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.fde_weight = fde_weight

    def forward(
        self, pred_traj: torch.Tensor, target_traj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_traj:   [batch, pred_len, 2]
            target_traj: [batch, pred_len, 2]
        """
        # 全序列 MSE (ADE-like)
        loss_ade = self.mse(pred_traj, target_traj)

        # 終點 FDE loss
        loss_fde = self.mse(pred_traj[:, -1, :], target_traj[:, -1, :])

        return loss_ade + self.fde_weight * loss_fde
