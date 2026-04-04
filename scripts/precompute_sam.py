"""預計算 SAM 語義分割特徵並儲存為 .npy 快取。

使用方式:
    python scripts/precompute_sam.py --data_dir data/PIE --output_dir data/sam_cache
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

from models.saim.segmentation.sam_wrapper import SAMWrapper
from models.saim.segmentation.patch_extractor import PatchExtractor


def precompute_sam(
    data_dir: str,
    output_dir: str,
    sam_feature_dim: int = 256,
    num_patches: int = 16,
    sam_checkpoint: str = "weights/sam_vit_h_4b8939.pth",
):
    os.makedirs(output_dir, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"使用裝置: {device}")

    # 初始化 SAM + PatchExtractor
    sam = SAMWrapper(
        model_type="vit_h",
        checkpoint=sam_checkpoint,
        device=device_str,
    )

    patch_extractor = PatchExtractor(
        sam_feature_dim=sam_feature_dim, num_patches=num_patches
    ).to(device)
    patch_extractor.eval()

    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        print(f"影像目錄不存在: {image_dir}")
        return

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

            for frame_name in frames:
                output_path = os.path.join(
                    output_dir, f"{set_name}_{video_name}_{frame_name}.npy"
                )
                if os.path.exists(output_path):
                    continue

                img_path = os.path.join(video_dir, frame_name)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # SAM 分割
                mask_tensor = sam(img_rgb, max_objects=num_patches)
                mask_tensor = mask_tensor.unsqueeze(0).to(device)

                # 準備影像 tensor
                img_resized = cv2.resize(img_bgr, (224, 224))
                img_t = (
                    torch.from_numpy(img_resized[:, :, ::-1].copy())
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                    / 255.0
                )

                # 調整 mask 大小
                mask_resized = torch.nn.functional.interpolate(
                    mask_tensor, size=(224, 224), mode="nearest"
                )

                with torch.no_grad():
                    scene_feat = patch_extractor(img_t, mask_resized)

                np.save(output_path, scene_feat.cpu().numpy()[0])
                total_computed += 1

    print(f"\nSAM 特徵計算完成！共 {total_computed} 幀")
    print(f"儲存位置: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/PIE")
    parser.add_argument("--output_dir", default="data/sam_cache")
    parser.add_argument("--sam_dim", type=int, default=256)
    parser.add_argument("--sam_checkpoint", default="weights/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    precompute_sam(args.data_dir, args.output_dir, args.sam_dim)
