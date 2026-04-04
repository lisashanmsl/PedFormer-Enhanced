import torch
import torch.nn as nn
import numpy as np

try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    HAS_RAFT_TORCHVISION = True
except ImportError:
    HAS_RAFT_TORCHVISION = False


class RAFTExtractor(nn.Module):
    """使用 RAFT 從連續影像幀計算稠密光流場。

    支援兩種模式:
    1. 即時計算: 輸入 RGB 幀對，輸出光流場
    2. 預計算快取: 直接讀取預先算好的光流 .npy 檔

    光流輸出: [batch, seq_len-1, 2, H, W] (水平+垂直方向)
    """

    def __init__(self, iters: int = 20, pretrained: bool = True):
        super().__init__()
        self.iters = iters

        if HAS_RAFT_TORCHVISION and pretrained:
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights)
            self.transforms = weights.transforms()
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            self.model = None
            self.transforms = None

    @torch.no_grad()
    def compute_flow(
        self, frame1: torch.Tensor, frame2: torch.Tensor
    ) -> torch.Tensor:
        """計算兩幀間的光流。

        Args:
            frame1: [batch, 3, H, W] 前一幀 (0-255 uint8 or 0-1 float)
            frame2: [batch, 3, H, W] 後一幀

        Returns:
            flow: [batch, 2, H, W]
        """
        if self.model is None:
            b, _, h, w = frame1.shape
            return torch.zeros(b, 2, h, w, device=frame1.device)

        if self.transforms is not None:
            frame1, frame2 = self.transforms(frame1, frame2)

        flow_list = self.model(frame1, frame2, num_flow_updates=self.iters)
        return flow_list[-1]  # 取最後一次迭代的結果

    @torch.no_grad()
    def compute_flow_sequence(
        self, frames: torch.Tensor
    ) -> torch.Tensor:
        """計算連續幀序列的光流。

        Args:
            frames: [batch, seq_len, 3, H, W]

        Returns:
            flows: [batch, seq_len-1, 2, H, W]
        """
        batch, seq_len, c, h, w = frames.shape
        flows = []
        for t in range(seq_len - 1):
            flow = self.compute_flow(frames[:, t], frames[:, t + 1])
            flows.append(flow)
        return torch.stack(flows, dim=1)

    @staticmethod
    def load_precomputed_flow(flow_path: str) -> torch.Tensor:
        """讀取預計算的光流 .npy 快取。"""
        flow_np = np.load(flow_path)
        return torch.from_numpy(flow_np).float()

    def forward(
        self, frames: torch.Tensor = None, precomputed_flow: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            frames: [batch, seq_len, 3, H, W] — 即時計算模式
            precomputed_flow: [batch, seq_len-1, 2, H, W] — 快取模式

        Returns:
            flows: [batch, seq_len-1, 2, H, W]
        """
        if precomputed_flow is not None:
            return precomputed_flow
        return self.compute_flow_sequence(frames)
