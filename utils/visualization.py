import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def draw_trajectory_on_frame(
    frame: np.ndarray,
    past_traj: np.ndarray,
    pred_traj: np.ndarray = None,
    gt_traj: np.ndarray = None,
    crossing_prob: float = None,
    safety_zone_radius: int = 50,
) -> np.ndarray:
    """在影像上繪製軌跡與安全區域。

    Args:
        frame: [H, W, 3] BGR 影像
        past_traj:  [obs_len, 2] 過去軌跡中心點 (cx, cy)
        pred_traj:  [pred_len, 2] 預測軌跡 (可選)
        gt_traj:    [pred_len, 2] 真實軌跡 (可選)
        crossing_prob: 0~1 穿越機率 (可選)
        safety_zone_radius: 安全區域半徑 (pixels)

    Returns:
        annotated_frame: [H, W, 3]
    """
    if not HAS_CV2:
        return frame

    vis = frame.copy()

    # 繪製過去軌跡 (藍色)
    for i in range(len(past_traj) - 1):
        pt1 = tuple(past_traj[i].astype(int))
        pt2 = tuple(past_traj[i + 1].astype(int))
        cv2.line(vis, pt1, pt2, (255, 150, 0), 2)
    for pt in past_traj:
        cv2.circle(vis, tuple(pt.astype(int)), 3, (255, 150, 0), -1)

    # 繪製預測軌跡 (綠色)
    if pred_traj is not None:
        for i in range(len(pred_traj) - 1):
            pt1 = tuple(pred_traj[i].astype(int))
            pt2 = tuple(pred_traj[i + 1].astype(int))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
        for pt in pred_traj:
            cv2.circle(vis, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

    # 繪製真實軌跡 (紅色虛線)
    if gt_traj is not None:
        for i in range(len(gt_traj) - 1):
            pt1 = tuple(gt_traj[i].astype(int))
            pt2 = tuple(gt_traj[i + 1].astype(int))
            cv2.line(vis, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    # 繪製安全區域 (Safety Zone)
    if pred_traj is not None and len(pred_traj) > 0:
        last_pred = tuple(pred_traj[-1].astype(int))

        if crossing_prob is not None and crossing_prob > 0.5:
            # 高風險: 紅色安全區域
            color = (0, 0, 255)
            radius = int(safety_zone_radius * (1 + crossing_prob))
        else:
            # 低風險: 綠色安全區域
            color = (0, 200, 0)
            radius = safety_zone_radius

        overlay = vis.copy()
        cv2.circle(overlay, last_pred, radius, color, -1)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        cv2.circle(vis, last_pred, radius, color, 2)

    # 顯示穿越機率文字
    if crossing_prob is not None:
        text = f"Crossing: {crossing_prob:.1%}"
        text_color = (0, 0, 255) if crossing_prob > 0.5 else (0, 200, 0)
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    return vis


def plot_trajectory_comparison(
    past_traj: np.ndarray,
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    title: str = "Trajectory Comparison",
    save_path: str = None,
):
    """用 matplotlib 繪製軌跡對比圖。"""
    if not HAS_PLT:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(past_traj[:, 0], past_traj[:, 1], "b.-", label="Past", linewidth=2)
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], "r.--", label="Ground Truth", linewidth=1)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "g.-", label="Prediction", linewidth=2)

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    ax.set_aspect("equal")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
