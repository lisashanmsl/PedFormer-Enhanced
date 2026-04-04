import torch
import torch.nn as nn

class SAMFeatureExtractor(nn.Module):
    """
    未來實作：利用 Segment Anything Model (SAM) 提取語意分割特徵的模組。
    目前僅作為佔位符。
    """
    def __init__(self, output_dim=64):
        super(SAMFeatureExtractor, self).__init__()
        self.output_dim = output_dim
        self.dummy_layer = nn.Linear(3, output_dim) 

    def forward(self, frames):
        # 尚未實作真正的 SAM 分割提取
        batch_size = frames.shape[0]
        seq_len = frames.shape[1]
        
        # 回傳全是 0 的假特徵陣列
        return torch.zeros((batch_size, seq_len, self.output_dim), device=frames.device)
