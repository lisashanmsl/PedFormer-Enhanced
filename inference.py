"""即時推論入口: 從影像串流進行行人軌跡與意圖預測。

支援模式:
  1. 單張影像推論
  2. 影片檔推論
  3. 即時串流推論 (搭配 hardware/ 模組)
"""

import time
import torch
import yaml
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from models.pedformer import PedFormerEnhanced
from utils.visualization import draw_trajectory_on_frame


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PedFormerInference:
    """PedFormer-Enhanced 推論管線。"""

    def __init__(
        self,
        weights_path: str = "weights/pedformer_best.pth",
        config_path: str = "configs/default.yaml",
        device: str = None,
    ):
        self.cfg = load_config(config_path)
        m = self.cfg["model"]

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = PedFormerEnhanced(
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_encoder_layers=m["num_encoder_layers"],
            dim_feedforward=m["dim_feedforward"],
            dropout=0.0,  # 推論時不使用 dropout
            traj_dim=m["traj_input_dim"],
            ego_dim=m["ego_input_dim"],
            flow_dim=m["flow_feature_dim"],
            sam_dim=m["sam_feature_dim"],
            lstm_hidden_dim=m["lstm_hidden_dim"],
            lstm_num_layers=m["lstm_num_layers"],
            pred_len=self.cfg["data"]["pred_len"],
        ).to(self.device)

        self._load_weights(weights_path)
        self.model.eval()

        self.obs_len = self.cfg["data"]["obs_len"]
        self.safety_radius = self.cfg["inference"]["safety_zone_radius"]
        self.threshold = self.cfg["inference"]["confidence_threshold"]

    def _load_weights(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"模型權重已載入: {path}")

    @torch.no_grad()
    def predict(
        self,
        past_traj: np.ndarray,
        ego: np.ndarray = None,
        flow_feat: np.ndarray = None,
        scene_feat: np.ndarray = None,
    ) -> dict:
        """單次推論。

        Args:
            past_traj: [obs_len, 4] — 行人歷史 bbox
            ego: [obs_len, 2] (可選)
            flow_feat: [obs_len, flow_dim] (可選)
            scene_feat: [num_patches, sam_dim] (可選)

        Returns:
            dict: pred_traj [pred_len, 2], crossing_prob float
        """
        m = self.cfg["model"]
        obs_len = past_traj.shape[0]

        traj_t = torch.from_numpy(past_traj).float().unsqueeze(0).to(self.device)

        if ego is None:
            ego = np.zeros((obs_len, m["ego_input_dim"]), dtype=np.float32)
        ego_t = torch.from_numpy(ego).float().unsqueeze(0).to(self.device)

        if flow_feat is None:
            flow_feat = np.zeros((obs_len, m["flow_feature_dim"]), dtype=np.float32)
        flow_t = torch.from_numpy(flow_feat).float().unsqueeze(0).to(self.device)

        if scene_feat is None:
            scene_feat = np.zeros((16, m["sam_feature_dim"]), dtype=np.float32)
        scene_t = torch.from_numpy(scene_feat).float().unsqueeze(0).to(self.device)

        output = self.model(traj_t, ego_t, flow_t, scene_t)

        pred_traj = output["pred_traj"][0].cpu().numpy()      # [pred_len, 2]
        crossing_prob = output["global_intent"][0, 0].cpu().item()  # float

        return {
            "pred_traj": pred_traj,
            "crossing_prob": crossing_prob,
            "is_crossing": crossing_prob >= self.threshold,
        }

    def run_on_video(self, video_path: str, output_path: str = None):
        """在影片上執行推論並視覺化。"""
        if not HAS_CV2:
            print("需要安裝 opencv-python")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # 簡化演示: 使用假軌跡資料
        # 實際使用時需搭配行人偵測器 (如 YOLOv8) 提供 bbox
        frame_count = 0
        traj_buffer = []

        print(f"處理影片: {video_path} ({w}x{h} @ {fps:.0f}fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 模擬: 使用畫面中心點作為假軌跡
            fake_bbox = np.array(
                [w * 0.4, h * 0.5, w * 0.6, h * 0.7], dtype=np.float32
            )
            traj_buffer.append(fake_bbox)

            if len(traj_buffer) >= self.obs_len:
                past_traj = np.array(traj_buffer[-self.obs_len :])
                result = self.predict(past_traj)

                # 繪製視覺化
                past_centers = np.stack(
                    [
                        (past_traj[:, 0] + past_traj[:, 2]) / 2,
                        (past_traj[:, 1] + past_traj[:, 3]) / 2,
                    ],
                    axis=-1,
                )

                frame = draw_trajectory_on_frame(
                    frame,
                    past_traj=past_centers,
                    pred_traj=result["pred_traj"],
                    crossing_prob=result["crossing_prob"],
                    safety_zone_radius=self.safety_radius,
                )

            if writer:
                writer.write(frame)

            cv2.imshow("PedFormer-Enhanced Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"處理完成: {frame_count} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PedFormer-Enhanced Inference")
    parser.add_argument("--video", type=str, help="影片路徑")
    parser.add_argument("--weights", default="weights/pedformer_best.pth")
    parser.add_argument("--output", type=str, default=None, help="輸出影片路徑")
    args = parser.parse_args()

    engine = PedFormerInference(weights_path=args.weights)

    if args.video:
        engine.run_on_video(args.video, args.output)
    else:
        # 快速測試
        dummy_traj = np.random.randn(16, 4).astype(np.float32) * 100 + 500
        result = engine.predict(dummy_traj)
        print(f"預測軌跡 shape: {result['pred_traj'].shape}")
        print(f"穿越機率: {result['crossing_prob']:.4f}")
        print(f"是否穿越: {result['is_crossing']}")
