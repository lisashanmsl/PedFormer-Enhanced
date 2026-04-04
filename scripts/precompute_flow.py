"""預計算 RAFT 光流特徵並儲存為 .npy 快取。

使用方式:
    python scripts/precompute_flow.py --data_dir data/PIE --output_dir data/flow_cache
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from models.saim.optical_flow.raft_extractor import RAFTExtractor
from models.saim.optical_flow.flow_encoder import FlowEncoder


def precompute_flow(
    data_dir: str,
    output_dir: str,
    flow_feature_dim: int = 256,
    image_size: tuple = (224, 224),
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 初始化 RAFT + FlowEncoder
    raft = RAFTExtractor(pretrained=True).to(device)
    encoder = FlowEncoder(flow_feature_dim=flow_feature_dim).to(device)
    encoder.eval()

    # 掃描影像目錄
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        print(f"影像目錄不存在: {image_dir}")
        print("請先執行 scripts/split_clips_to_frames.sh 提取影像幀。")
        return

    # 取得所有 set 目錄
    sets = sorted(
        [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    )

    total_computed = 0

    for set_name in sets:
        set_dir = os.path.join(image_dir, set_name)
        videos = sorted(
            [d for d in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, d))]
        )

        for video_name in tqdm(videos, desc=f"Processing {set_name}"):
            video_dir = os.path.join(set_dir, video_name)
            frames = sorted(
                [f for f in os.listdir(video_dir) if f.endswith((".png", ".jpg"))]
            )

            if len(frames) < 2:
                continue

            # 逐對計算光流並編碼
            for i in range(len(frames) - 1):
                output_path = os.path.join(
                    output_dir, f"{set_name}_{video_name}_{i:06d}.npy"
                )
                if os.path.exists(output_path):
                    continue

                img1 = cv2.imread(os.path.join(video_dir, frames[i]))
                img2 = cv2.imread(os.path.join(video_dir, frames[i + 1]))

                if img1 is None or img2 is None:
                    continue

                img1 = cv2.resize(img1, image_size)
                img2 = cv2.resize(img2, image_size)

                # BGR → RGB → tensor [1, 3, H, W]
                t1 = (
                    torch.from_numpy(img1[:, :, ::-1].copy())
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                t2 = (
                    torch.from_numpy(img2[:, :, ::-1].copy())
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.no_grad():
                    flow = raft.compute_flow(t1, t2)  # [1, 2, H, W]
                    feat = encoder.encode_single(flow)  # [1, flow_dim]

                np.save(output_path, feat.cpu().numpy()[0])
                total_computed += 1

    print(f"\n光流特徵計算完成！共 {total_computed} 幀")
    print(f"儲存位置: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/PIE")
    parser.add_argument("--output_dir", default="data/flow_cache")
    parser.add_argument("--flow_dim", type=int, default=256)
    args = parser.parse_args()

    precompute_flow(args.data_dir, args.output_dir, args.flow_dim)
