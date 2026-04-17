import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from data.pie_data import PIE
from data.jaad_data import JAAD


class PIEPedestrianDataset(Dataset):
    """PIE 資料集的 PyTorch Dataset 封裝。"""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        obs_len: int = 16,
        pred_len: int = 45,
        flow_cache_dir: str = None,
        sam_cache_dir: str = None,
        flow_dim: int = 256,
        sam_dim: int = 256,
        num_patches: int = 16,
    ):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.flow_cache_dir = flow_cache_dir
        self.sam_cache_dir = sam_cache_dir
        self.flow_dim = flow_dim
        self.sam_dim = sam_dim
        self.num_patches = num_patches
        self.dataset_name = "PIE"

        print(f"[PIE] 正在載入資料庫 (路徑: {data_path}, 分割: {split})...")
        self.pie = PIE(data_path=data_path)
        self.db = self.pie.generate_database()

        data_splits = {
            "train": ["set01", "set02", "set03", "set04"],
            "val": ["set05"],
            "test": ["set06"],
        }

        print("[PIE] 正在提取軌跡與意圖標籤...")
        self.raw_data = self.pie.generate_data_trajectory_sequence(
            image_set=split,
            database=self.db,
            seq_type="trajectory",
            data_splits=data_splits,
        )

        self.samples = []
        self._prepare_data(split)

    def _prepare_data(self, split: str):
        bboxes = self.raw_data.get("bbox", [])
        # PIE 返回 'intention_prob'，不是 'intent'
        intents = self.raw_data.get("intention_prob", self.raw_data.get("intent", []))
        images = self.raw_data.get("image", [])
        obd_speeds = self.raw_data.get("obd_speed", [])
        head_angles = self.raw_data.get("heading_angle", [])

        if len(bboxes) == 0:
            print(f"[PIE] 警告：在 {split} 集中找不到任何軌跡資料。")
            return

        for i in range(len(bboxes)):
            bbox_seq = bboxes[i]
            if len(bbox_seq) < self.seq_len:
                continue

            past_traj = np.array(bbox_seq[: self.obs_len], dtype=np.float32)
            future_traj_bbox = np.array(
                bbox_seq[self.obs_len : self.seq_len], dtype=np.float32
            )
            future_traj = np.stack(
                [
                    (future_traj_bbox[:, 0] + future_traj_bbox[:, 2]) / 2,
                    (future_traj_bbox[:, 1] + future_traj_bbox[:, 3]) / 2,
                ],
                axis=-1,
            )

            # intent: PIE 的 intention_prob 是連續機率值 (0~1)
            # 使用平均機率 >= 0.5 作為穿越閾值
            if len(intents) > i and len(intents[i]) > 0:
                intent_vals = [x[0] if isinstance(x, list) else x for x in intents[i]]
                intent_label = 1.0 if np.mean(intent_vals) >= 0.5 else 0.0
            else:
                intent_label = 0.0

            # ego motion: OBD speed + heading angle
            if len(obd_speeds) > i and len(head_angles) > i:
                spd = [x[0] if isinstance(x, list) else x for x in obd_speeds[i][:self.obs_len]]
                ang = [x[0] if isinstance(x, list) else x for x in head_angles[i][:self.obs_len]]
                ego_data = np.array(list(zip(spd, ang)), dtype=np.float32)
                # 補齊長度
                if len(ego_data) < self.obs_len:
                    pad = np.zeros((self.obs_len - len(ego_data), 2), dtype=np.float32)
                    ego_data = np.concatenate([ego_data, pad])
            else:
                ego_data = np.zeros((self.obs_len, 2), dtype=np.float32)

            # 儲存影像路徑（用於查找光流/SAM 快取）
            img_paths = images[i][:self.obs_len] if len(images) > i else []

            self.samples.append({
                "past_traj": past_traj,
                "future_traj": future_traj,
                "intent": np.array([intent_label], dtype=np.float32),
                "ego": ego_data,
                "image_paths": img_paths,
                "sample_idx": i,
                "source": "PIE",
            })

        print(f"[PIE][{split}] 資料準備完成！共 {len(self.samples)} 筆有效樣本。")

    @staticmethod
    def _frame_key(img_path: str) -> str:
        """從影像路徑提取快取 key，例如 'images/set01/video_0001/00015.png' → 'set01_video_0001_00015'"""
        parts = img_path.replace("\\", "/").split("/")
        # 找到 set/video/frame 部分
        for j, p in enumerate(parts):
            if p.startswith("set"):
                vid = parts[j + 1] if j + 1 < len(parts) else ""
                frame = os.path.splitext(parts[j + 2])[0] if j + 2 < len(parts) else ""
                return f"{p}_{vid}_{frame}"
        return ""

    def _load_flow_features(self, idx: int) -> torch.Tensor:
        """載入 obs_len 個光流特徵，按影像路徑查找快取。"""
        sample = self.samples[idx]
        img_paths = sample.get("image_paths", [])

        if self.flow_cache_dir and os.path.exists(self.flow_cache_dir) and len(img_paths) >= 2:
            feats = []
            for i in range(min(len(img_paths) - 1, self.obs_len)):
                key = self._frame_key(img_paths[i])
                flow_path = os.path.join(self.flow_cache_dir, f"{key}.npy")
                if os.path.exists(flow_path):
                    feats.append(torch.from_numpy(np.load(flow_path)).float())
                else:
                    feats.append(torch.zeros(self.flow_dim))
            # 補齊到 obs_len（第一幀或末尾可能不足）
            while len(feats) < self.obs_len:
                feats.append(torch.zeros(self.flow_dim))
            return torch.stack(feats[:self.obs_len])

        return torch.zeros(self.obs_len, self.flow_dim)

    def _load_sam_features(self, idx: int) -> torch.Tensor:
        """載入 SAM 場景特徵，使用最後一個觀察幀。"""
        sample = self.samples[idx]
        img_paths = sample.get("image_paths", [])

        if self.sam_cache_dir and os.path.exists(self.sam_cache_dir) and len(img_paths) > 0:
            # 使用最後一個觀察幀的 SAM 特徵
            key = self._frame_key(img_paths[-1])
            sam_path = os.path.join(self.sam_cache_dir, f"{key}.npy")
            if os.path.exists(sam_path):
                return torch.from_numpy(np.load(sam_path)).float()

        return torch.zeros(self.num_patches, self.sam_dim)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "past_traj": torch.from_numpy(sample["past_traj"]),
            "future_traj": torch.from_numpy(sample["future_traj"]),
            "intent": torch.from_numpy(sample["intent"]),
            "ego": torch.from_numpy(sample["ego"]),
            "flow_feat": self._load_flow_features(idx),
            "scene_feat": self._load_sam_features(idx),
        }


class JAADPedestrianDataset(Dataset):
    """JAAD 資料集的 PyTorch Dataset 封裝。

    JAAD 的 API 與 PIE 相似，均提供 generate_data_trajectory_sequence()，
    返回包含 'bbox', 'intent' 等欄位的字典。
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        obs_len: int = 16,
        pred_len: int = 45,
        flow_cache_dir: str = None,
        sam_cache_dir: str = None,
        flow_dim: int = 256,
        sam_dim: int = 256,
        num_patches: int = 16,
    ):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.flow_cache_dir = flow_cache_dir
        self.sam_cache_dir = sam_cache_dir
        self.flow_dim = flow_dim
        self.sam_dim = sam_dim
        self.num_patches = num_patches
        self.dataset_name = "JAAD"

        print(f"[JAAD] 正在載入資料庫 (路徑: {data_path}, 分割: {split})...")
        self.jaad = JAAD(data_path=data_path)

        print("[JAAD] 正在提取軌跡與意圖標籤...")
        self.raw_data = self.jaad.generate_data_trajectory_sequence(
            image_set=split,
            seq_type="trajectory",
            subset="default",
        )

        self.samples = []
        self._prepare_data(split)

    def _prepare_data(self, split: str):
        bboxes = self.raw_data.get("bbox", [])
        intents = self.raw_data.get("intent", [])
        images = self.raw_data.get("image", [])
        vehicle_acts = self.raw_data.get("vehicle_act", [])

        if len(bboxes) == 0:
            print(f"[JAAD] 警告：在 {split} 集中找不到任何軌跡資料。")
            return

        for i in range(len(bboxes)):
            bbox_seq = bboxes[i]
            if len(bbox_seq) < self.seq_len:
                continue

            past_traj = np.array(bbox_seq[: self.obs_len], dtype=np.float32)
            future_traj_bbox = np.array(
                bbox_seq[self.obs_len : self.seq_len], dtype=np.float32
            )
            future_traj = np.stack(
                [
                    (future_traj_bbox[:, 0] + future_traj_bbox[:, 2]) / 2,
                    (future_traj_bbox[:, 1] + future_traj_bbox[:, 3]) / 2,
                ],
                axis=-1,
            )

            # JAAD intent: list of lists per frame e.g. [[1],[0],[1],...]
            if len(intents) > i and len(intents[i]) > 0:
                intent_vals = [x[0] if isinstance(x, list) else x for x in intents[i]]
                intent_label = 1.0 if sum(intent_vals) > 0 else 0.0
            else:
                intent_label = 0.0

            ego_data = np.zeros((self.obs_len, 2), dtype=np.float32)

            # 儲存影像路徑
            img_paths = images[i][:self.obs_len] if len(images) > i else []

            self.samples.append({
                "past_traj": past_traj,
                "future_traj": future_traj,
                "intent": np.array([intent_label], dtype=np.float32),
                "ego": ego_data,
                "image_paths": img_paths,
                "sample_idx": i,
                "source": "JAAD",
            })

        print(f"[JAAD][{split}] 資料準備完成！共 {len(self.samples)} 筆有效樣本。")

    @staticmethod
    def _frame_key(img_path: str) -> str:
        """從影像路徑提取快取 key，例如 'images/video_0001/00015.png' → 'video_0001_00015'"""
        parts = img_path.replace("\\", "/").split("/")
        for j, p in enumerate(parts):
            if p.startswith("video"):
                frame = os.path.splitext(parts[j + 1])[0] if j + 1 < len(parts) else ""
                return f"{p}_{frame}"
        return ""

    def _load_flow_features(self, idx: int) -> torch.Tensor:
        """載入 obs_len 個光流特徵，按影像路徑查找快取。"""
        sample = self.samples[idx]
        img_paths = sample.get("image_paths", [])

        if self.flow_cache_dir and os.path.exists(self.flow_cache_dir) and len(img_paths) >= 2:
            feats = []
            for i in range(min(len(img_paths) - 1, self.obs_len)):
                key = self._frame_key(img_paths[i])
                flow_path = os.path.join(self.flow_cache_dir, f"{key}.npy")
                if os.path.exists(flow_path):
                    feats.append(torch.from_numpy(np.load(flow_path)).float())
                else:
                    feats.append(torch.zeros(self.flow_dim))
            while len(feats) < self.obs_len:
                feats.append(torch.zeros(self.flow_dim))
            return torch.stack(feats[:self.obs_len])

        return torch.zeros(self.obs_len, self.flow_dim)

    def _load_sam_features(self, idx: int) -> torch.Tensor:
        """載入 SAM 場景特徵，使用最後一個觀察幀。"""
        sample = self.samples[idx]
        img_paths = sample.get("image_paths", [])

        if self.sam_cache_dir and os.path.exists(self.sam_cache_dir) and len(img_paths) > 0:
            key = self._frame_key(img_paths[-1])
            sam_path = os.path.join(self.sam_cache_dir, f"{key}.npy")
            if os.path.exists(sam_path):
                return torch.from_numpy(np.load(sam_path)).float()

        return torch.zeros(self.num_patches, self.sam_dim)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "past_traj": torch.from_numpy(sample["past_traj"]),
            "future_traj": torch.from_numpy(sample["future_traj"]),
            "intent": torch.from_numpy(sample["intent"]),
            "ego": torch.from_numpy(sample["ego"]),
            "flow_feat": self._load_flow_features(idx),
            "scene_feat": self._load_sam_features(idx),
        }


def get_dataloader(
    data_path: str = "data/PIE",
    dataset_name: str = "PIE",
    split: str = "train",
    batch_size: int = 16,
    obs_len: int = 16,
    pred_len: int = 45,
    flow_cache_dir: str = None,
    sam_cache_dir: str = None,
    flow_dim: int = 256,
    sam_dim: int = 256,
    num_patches: int = 16,
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """建立單一資料集的 DataLoader。"""
    if shuffle is None:
        shuffle = split == "train"

    common_args = dict(
        data_path=data_path, split=split, obs_len=obs_len, pred_len=pred_len,
        flow_cache_dir=flow_cache_dir, sam_cache_dir=sam_cache_dir,
        flow_dim=flow_dim, sam_dim=sam_dim, num_patches=num_patches,
    )

    if dataset_name.upper() == "JAAD":
        dataset = JAADPedestrianDataset(**common_args)
    else:
        dataset = PIEPedestrianDataset(**common_args)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        drop_last=(split == "train"),
    )


def get_combined_dataloader(
    pie_path: str = "data/PIE",
    jaad_path: str = "data/JAAD",
    split: str = "train",
    batch_size: int = 16,
    obs_len: int = 16,
    pred_len: int = 45,
    flow_cache_dir: str = None,
    sam_cache_dir: str = None,
    flow_dim: int = 256,
    sam_dim: int = 256,
    num_patches: int = 16,
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """建立 PIE + JAAD 合併的 DataLoader。

    使用 ConcatDataset 將兩個資料集串接，共同訓練。
    """
    if shuffle is None:
        shuffle = split == "train"

    common_args = dict(
        split=split, obs_len=obs_len, pred_len=pred_len,
        flow_cache_dir=flow_cache_dir, sam_cache_dir=sam_cache_dir,
        flow_dim=flow_dim, sam_dim=sam_dim, num_patches=num_patches,
    )

    datasets = []

    # PIE
    try:
        pie_ds = PIEPedestrianDataset(data_path=pie_path, **common_args)
        if len(pie_ds) > 0:
            datasets.append(pie_ds)
    except Exception as e:
        print(f"[PIE] 載入失敗: {e}")

    # JAAD
    try:
        jaad_ds = JAADPedestrianDataset(data_path=jaad_path, **common_args)
        if len(jaad_ds) > 0:
            datasets.append(jaad_ds)
    except Exception as e:
        print(f"[JAAD] 載入失敗: {e}")

    if len(datasets) == 0:
        raise RuntimeError("PIE 與 JAAD 均無法載入，請檢查資料路徑。")

    combined = ConcatDataset(datasets)
    total = len(combined)
    sources = [d.dataset_name for d in datasets]
    sizes = [len(d) for d in datasets]
    print(f"\n合併資料集: {' + '.join(sources)} = {' + '.join(map(str, sizes))} = {total} 筆樣本")

    return DataLoader(
        combined, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        drop_last=(split == "train"),
    )
