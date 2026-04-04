# PedFormer 專題實作

這是一個基於 Transformer、整合 RAFT 光流特徵與 SAM 語意分割特徵的雙任務行人預測模型 (軌跡預測 + 意圖預測)。

## 專案結構
- `configs/`: 存放訓練與模型超參數設定 (`default.yaml`)。
- `data/`: 存放 Dataset、DataLoader 的實作，以及放置 PIE 原生影像與標註資料。
- `models/`: 存放模型架構 (`pedformer.py` 等)。
- `scripts/`: 存放開發與資料處理腳本 (例如：下載影片、抽幀)。
- `utils/`: 存放輔助工具 (評估指標、視覺化等)。
- `weights/`: 存放預訓練模型與訓練好的權重。
- `train.py`: 主要的訓練腳本。
- `evaluate.py`: 評估與測試腳本。

## 快速開始

1. 安裝相依套件：
   ```bash
   pip install -r requirements.txt
   ```
2. 將下載的影片放置在相應的 `data/PIE/PIE_clips/` 目錄下 (例如：`data/PIE/PIE_clips/set01/video_0001.mp4`)。
3. 執行訓練程式：
   ```bash
   python train.py
   ```
