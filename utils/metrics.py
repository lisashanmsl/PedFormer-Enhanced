import numpy as np


def compute_ade(pred_traj: np.ndarray, target_traj: np.ndarray) -> float:
    """Average Displacement Error: 所有時間步的平均 L2 距離。

    Args:
        pred_traj:   [batch, seq_len, 2]
        target_traj: [batch, seq_len, 2]
    """
    err = np.linalg.norm(pred_traj - target_traj, axis=2)
    return float(np.mean(err))


def compute_fde(pred_traj: np.ndarray, target_traj: np.ndarray) -> float:
    """Final Displacement Error: 最終時間步的 L2 距離。"""
    err = np.linalg.norm(pred_traj[:, -1, :] - target_traj[:, -1, :], axis=1)
    return float(np.mean(err))


def compute_intent_accuracy(
    pred_intent: np.ndarray, target_intent: np.ndarray, threshold: float = 0.5
) -> float:
    """意圖預測準確率。

    Args:
        pred_intent:   [batch, 1] — 機率值 (0~1)
        target_intent: [batch, 1] — 真實標籤 (0 or 1)
    """
    pred_binary = (pred_intent >= threshold).astype(float)
    correct = (pred_binary == target_intent).sum()
    return float(correct / len(target_intent))


def compute_f1_score(
    pred_intent: np.ndarray, target_intent: np.ndarray, threshold: float = 0.5
) -> dict:
    """計算意圖預測的 Precision, Recall, F1-score。

    Returns:
        dict with keys: 'precision', 'recall', 'f1'
    """
    pred_binary = (pred_intent >= threshold).astype(float).flatten()
    target = target_intent.flatten()

    tp = ((pred_binary == 1) & (target == 1)).sum()
    fp = ((pred_binary == 1) & (target == 0)).sum()
    fn = ((pred_binary == 0) & (target == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_all_metrics(
    pred_traj: np.ndarray,
    target_traj: np.ndarray,
    pred_intent: np.ndarray,
    target_intent: np.ndarray,
) -> dict:
    """計算所有評估指標。"""
    f1_metrics = compute_f1_score(pred_intent, target_intent)

    return {
        "ADE": compute_ade(pred_traj, target_traj),
        "FDE": compute_fde(pred_traj, target_traj),
        "Intent_Accuracy": compute_intent_accuracy(pred_intent, target_intent),
        "Precision": f1_metrics["precision"],
        "Recall": f1_metrics["recall"],
        "F1": f1_metrics["f1"],
    }
