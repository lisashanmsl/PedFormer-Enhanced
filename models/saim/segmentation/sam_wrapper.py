import torch
import torch.nn as nn
import numpy as np

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


class SAMWrapper(nn.Module):
    """Segment Anything Model 封裝層。

    使用 SAM 對場景影像進行零樣本分割:
    - 自動偵測並分割場景中所有物體 (行人、車輛、道路等)
    - 輸出每個物體的二值遮罩 (binary mask)
    - 支援預計算快取模式

    輸出: List[dict] — 每個 dict 包含 'segmentation', 'area', 'bbox' 等
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: str = "weights/sam_vit_h_4b8939.pth",
        points_per_side: int = 32,
        device: str = "cuda",
    ):
        super().__init__()
        self.device_str = device

        if HAS_SAM:
            try:
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(device)
                sam.eval()
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=points_per_side,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    min_mask_region_area=100,
                )
                self.sam = sam
            except Exception:
                self.mask_generator = None
                self.sam = None
        else:
            self.mask_generator = None
            self.sam = None

    @torch.no_grad()
    def generate_masks(self, image_np: np.ndarray) -> list:
        """對單張 RGB 影像進行自動分割。

        Args:
            image_np: [H, W, 3] uint8 numpy array (RGB)

        Returns:
            masks: List[dict] with keys 'segmentation', 'area', 'bbox', etc.
        """
        if self.mask_generator is None:
            return []
        return self.mask_generator.generate(image_np)

    @staticmethod
    def masks_to_tensor(
        masks: list, image_size: tuple, max_objects: int = 16
    ) -> torch.Tensor:
        """將 SAM 遮罩列表轉為固定大小的 tensor。

        Args:
            masks: SAM 輸出的遮罩列表
            image_size: (H, W)
            max_objects: 最多保留多少個物體遮罩

        Returns:
            mask_tensor: [max_objects, H, W] binary float tensor
        """
        h, w = image_size
        mask_tensor = torch.zeros(max_objects, h, w)

        # 按面積排序，取前 max_objects 個
        sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        for i, m in enumerate(sorted_masks[:max_objects]):
            seg = m["segmentation"]  # [H, W] bool
            mask_tensor[i] = torch.from_numpy(seg).float()

        return mask_tensor

    def forward(self, image_np: np.ndarray, max_objects: int = 16) -> torch.Tensor:
        """
        Args:
            image_np: [H, W, 3] uint8
            max_objects: 最大物體數

        Returns:
            mask_tensor: [max_objects, H, W]
        """
        masks = self.generate_masks(image_np)
        h, w = image_np.shape[:2]
        return self.masks_to_tensor(masks, (h, w), max_objects)
