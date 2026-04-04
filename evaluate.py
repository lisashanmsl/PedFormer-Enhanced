import torch
import yaml
import numpy as np

from models.pedformer import PedFormerEnhanced
from data.dataset import get_dataloader, get_combined_dataloader
from utils.metrics import compute_all_metrics


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataloader(cfg, split):
    m = cfg["model"]
    d = cfg["data"]
    p = cfg["paths"]
    common = dict(
        split=split,
        batch_size=cfg["train"]["batch_size"],
        obs_len=d["obs_len"],
        pred_len=d["pred_len"],
        flow_cache_dir=p.get("flow_cache_dir"),
        sam_cache_dir=p.get("sam_cache_dir"),
        flow_dim=m["flow_feature_dim"],
        sam_dim=m["sam_feature_dim"],
        shuffle=False,
    )

    dataset_choice = d.get("dataset", "PIE").lower()

    if dataset_choice == "combined":
        return get_combined_dataloader(
            pie_path=p["pie_dir"], jaad_path=p["jaad_dir"], **common,
        )
    elif dataset_choice == "jaad":
        return get_dataloader(data_path=p["jaad_dir"], dataset_name="JAAD", **common)
    else:
        return get_dataloader(data_path=p["pie_dir"], dataset_name="PIE", **common)


def evaluate(weights_path: str = "weights/pedformer_best.pth", split: str = "test"):
    cfg = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"使用評測裝置: {device}")

    m = cfg["model"]
    model = PedFormerEnhanced(
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        traj_dim=m["traj_input_dim"],
        ego_dim=m["ego_input_dim"],
        flow_dim=m["flow_feature_dim"],
        sam_dim=m["sam_feature_dim"],
        lstm_hidden_dim=m["lstm_hidden_dim"],
        lstm_num_layers=m["lstm_num_layers"],
        pred_len=cfg["data"]["pred_len"],
    ).to(device)

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"成功載入模型權重 (epoch {checkpoint.get('epoch', '?')})")
        else:
            model.load_state_dict(checkpoint)
            print("成功載入模型權重！")
    except FileNotFoundError:
        print(f"找不到權重檔案: {weights_path}")
        return
    except Exception as e:
        print(f"載入權重失敗: {e}")
        return

    test_loader = build_dataloader(cfg, split)

    model.eval()
    all_pred_traj, all_gt_traj = [], []
    all_pred_intent, all_gt_intent = [], []

    print(f"--- 開始評測 ({split} 集, dataset={cfg['data'].get('dataset')}) ---")

    with torch.no_grad():
        for batch in test_loader:
            past_traj = batch["past_traj"].to(device)
            future_traj = batch["future_traj"].to(device)
            intent = batch["intent"].to(device)
            ego = batch["ego"].to(device)
            flow_feat = batch["flow_feat"].to(device)
            scene_feat = batch["scene_feat"].to(device)

            output = model(past_traj, ego, flow_feat, scene_feat)

            all_pred_traj.append(output["pred_traj"].cpu().numpy())
            all_gt_traj.append(future_traj.cpu().numpy())
            all_pred_intent.append(output["global_intent"].cpu().numpy())
            all_gt_intent.append(intent.cpu().numpy())

    if len(all_pred_traj) == 0:
        print("沒有測試資料！")
        return

    pred_t = np.concatenate(all_pred_traj)
    gt_t = np.concatenate(all_gt_traj)
    pred_i = np.concatenate(all_pred_intent)
    gt_i = np.concatenate(all_gt_intent)

    metrics = compute_all_metrics(pred_t, gt_t, pred_i, gt_i)

    print("\n===== 評測結果 =====")
    print(f"樣本數:           {len(pred_t)}")
    print(f"ADE (平均位移誤差): {metrics['ADE']:.4f}")
    print(f"FDE (終點位移誤差): {metrics['FDE']:.4f}")
    print(f"意圖準確率:         {metrics['Intent_Accuracy']:.2%}")
    print(f"Precision:         {metrics['Precision']:.4f}")
    print(f"Recall:            {metrics['Recall']:.4f}")
    print(f"F1-Score:          {metrics['F1']:.4f}")
    print("====================")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/pedformer_best.pth")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(weights_path=args.weights, split=args.split)
