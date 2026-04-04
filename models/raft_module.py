import torch
import torch.nn as nn

class RAFTFeatureExtractor(nn.Module):
    """
    未來實作：利用 RAFT 提取光流特徵的模組。
    目前僅作為佔位符。
    """
    def __init__(self, output_dim=64):
        super(RAFTFeatureExtractor, self).__init__()
        self.output_dim = output_dim
        # 未來可以載入預訓練的 RAFT 模型或加入卷積層壓縮特徵
        self.dummy_layer = nn.Linear(3, output_dim) # 隨便建一個假層避免報錯

    def forward(self, frames):
        # 尚未實作真正的光流提取
        batch_size = frames.shape[0]
        seq_len = frames.shape[1]
        
        # 回傳全是 0 的假特徵陣列
        return torch.zeros((batch_size, seq_len, self.output_dim), device=frames.device)
