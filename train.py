import os
import time
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.pedformer import PedFormerEnhanced
from data.dataset import get_dataloader, get_combined_dataloader
from losses.multitask_loss import MultiTaskLoss
from utils.metrics import compute_all_metrics
import numpy as np


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataloader(cfg, split):
    """根據設定建立 DataLoader，支援 PIE / JAAD / combined。"""
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
    )

    dataset_choice = d.get("dataset", "PIE").lower()

    if dataset_choice == "combined":
        return get_combined_dataloader(
            pie_path=p["pie_dir"],
            jaad_path=p["jaad_dir"],
            **common,
        )
    elif dataset_choice == "jaad":
        return get_dataloader(data_path=p["jaad_dir"], dataset_name="JAAD", **common)
    else:
        return get_dataloader(data_path=p["pie_dir"], dataset_name="PIE", **common)


def train():
    cfg = load_config()
    os.makedirs(cfg["paths"]["weights_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)

    # 裝置設定
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"使用訓練裝置: {device}")

    # 初始化模型
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可訓練參數量: {total_params:,}")
    print(f"資料集模式: {cfg['data'].get('dataset', 'PIE')}")

    # 資料載入 (支援 PIE / JAAD / combined)
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    # 損失函數
    criterion = MultiTaskLoss(
        w_traj=cfg["train"]["w_traj"],
        w_intent=cfg["train"]["w_intent"],
    )

    # 優化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # 學習率排程
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["num_epochs"], eta_min=1e-6
    )

    num_epochs = cfg["train"]["num_epochs"]
    patience = cfg["train"]["patience"]
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n--- 開始訓練 PedFormer-Enhanced ({num_epochs} epochs) ---")
    print(f"    Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        # ===== Training =====
        model.train()
        train_loss_sum = 0
        train_traj_loss = 0
        train_intent_loss = 0

        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            past_traj = batch["past_traj"].to(device)
            future_traj = batch["future_traj"].to(device)
            intent = batch["intent"].to(device)
            ego = batch["ego"].to(device)
            flow_feat = batch["flow_feat"].to(device)
            scene_feat = batch["scene_feat"].to(device)

            optimizer.zero_grad()

            output = model(past_traj, ego, flow_feat, scene_feat)

            losses = criterion(
                pred_traj=output["pred_traj"],
                target_traj=future_traj,
                global_intent=output["global_intent"],
                step_intents=output["step_intents"],
                target_intent=intent,
            )

            losses["total"].backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"]["grad_clip"]
            )

            optimizer.step()

            train_loss_sum += losses["total"].item()
            train_traj_loss += losses["traj"].item()
            train_intent_loss += losses["intent"].item()

        scheduler.step()
        elapsed = time.time() - t0
        n = len(train_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {train_loss_sum/n:.4f} "
            f"(traj: {train_traj_loss/n:.4f}, intent: {train_intent_loss/n:.4f}) "
            f"lr: {scheduler.get_last_lr()[0]:.6f} "
            f"time: {elapsed:.1f}s"
        )

        # ===== Validation =====
        model.eval()
        val_loss_sum = 0
        all_pred_traj, all_gt_traj = [], []
        all_pred_intent, all_gt_intent = [], []

        with torch.no_grad():
            for batch in val_loader:
                past_traj = batch["past_traj"].to(device)
                future_traj = batch["future_traj"].to(device)
                intent = batch["intent"].to(device)
                ego = batch["ego"].to(device)
                flow_feat = batch["flow_feat"].to(device)
                scene_feat = batch["scene_feat"].to(device)

                output = model(past_traj, ego, flow_feat, scene_feat)

                losses = criterion(
                    pred_traj=output["pred_traj"],
                    target_traj=future_traj,
                    global_intent=output["global_intent"],
                    step_intents=output["step_intents"],
                    target_intent=intent,
                )
                val_loss_sum += losses["total"].item()

                all_pred_traj.append(output["pred_traj"].cpu().numpy())
                all_gt_traj.append(future_traj.cpu().numpy())
                all_pred_intent.append(output["global_intent"].cpu().numpy())
                all_gt_intent.append(intent.cpu().numpy())

        if len(all_pred_traj) > 0:
            pred_t = np.concatenate(all_pred_traj)
            gt_t = np.concatenate(all_gt_traj)
            pred_i = np.concatenate(all_pred_intent)
            gt_i = np.concatenate(all_gt_intent)

            metrics = compute_all_metrics(pred_t, gt_t, pred_i, gt_i)
            val_loss = val_loss_sum / max(len(val_loader), 1)

            print(
                f"  Val Loss: {val_loss:.4f} | "
                f"ADE: {metrics['ADE']:.4f} | FDE: {metrics['FDE']:.4f} | "
                f"Acc: {metrics['Intent_Accuracy']:.2%} | F1: {metrics['F1']:.4f}"
            )

            # Early Stopping + Save Best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_path = os.path.join(
                    cfg["paths"]["weights_dir"], "pedformer_best.pth"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": val_loss,
                        "metrics": metrics,
                        "config": cfg,
                    },
                    save_path,
                )
                print(f"  -> 最佳模型已儲存至 {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停觸發 (patience={patience})，停止訓練。")
                    break

    # 儲存最終模型
    final_path = os.path.join(cfg["paths"]["weights_dir"], "pedformer_latest.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n--- 訓練完成 ---\n最終模型: {final_path}")


if __name__ == "__main__":
    train()
