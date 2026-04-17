# PedFormer-Enhanced 技術完整指南

> 運用 Transformer 結合光流預測行人與行車路徑實現用路人安全之研究

---

## 一、專案方向：這個專案在做什麼？

### 1.1 解決的問題

自動駕駛車輛需要「預測行人的未來行為」，才能提前做出安全決策。

本專案同時預測兩件事：

| 任務 | 問題 | 輸出 |
|------|------|------|
| **軌跡預測 (Trajectory Prediction)** | 這個行人接下來會走到哪裡？ | 未來 45 幀的 (x, y) 座標序列 |
| **意圖預測 (Intention Prediction)** | 這個行人會不會穿越馬路？ | 0~1 的穿越機率 |

### 1.2 為什麼需要這個研究？

現有模型的痛點：

```
PedFormer (2023):
  優點: Transformer 跨模態編碼很強
  缺點: 只看靜態語義圖 → 分不出「靜止車輛」vs「高速接近的車輛」

PTINet (2025):
  優點: 用光流捕捉動態 + 雙 LSTM 解碼效果好
  缺點: 依賴精細人工標註 + 特徵融合太簡單 (單純相加)
```

### 1.3 我們的解法

**取兩者精華，補兩者短板：**

```
PedFormer 的 Transformer 編碼器 (強項: 跨模態特徵融合)
  + RAFT 光流 (解決: 靜態語義看不到動態)
  + SAM 零樣本分割 (解決: 傳統分割圖太粗糙)
  + PTINet 的 LSTM 解碼器 (解決: 更好的時序預測)
  = PedFormer-Enhanced
```

---

## 二、系統架構全景圖

### 2.1 整體 Forward Pass（含 Tensor Shape）

```
輸入 (batch=B, obs_len=16)
 │
 ├── traj:       [B, 16, 4]   行人歷史 bbox (x1,y1,x2,y2)
 ├── ego:        [B, 16, 2]   自車運動 (速度, 轉向角)
 ├── flow_feat:  [B, 16, 256] RAFT 光流特徵 (預計算快取)
 └── scene_feat: [B, 16, 256] SAM 場景特徵 (預計算快取)
       │
       ├──────────────────────────────────────────────────┐
       ▼                                                  ▼
 ┌─────────────────────────────┐        ┌────────────────────────────────┐
 │  Cross-Modal Transformer    │        │       Enhanced SAIM             │
 │  Encoder                    │        │                                 │
 │                             │        │  各模態分別嵌入到 128 維         │
 │  各模態嵌入 → 串接 →         │        │  saim_traj_emb:  [B,16,4]→128  │
 │  [B, 64, 128]               │        │  saim_ego_emb:   [B,16,2]→128  │
 │  ↓ 位置編碼                  │        │  saim_flow_emb:  [B,16,256]→128│
 │  ↓ 4層 Pre-LN Transformer   │        │  saim_scene_emb: [B,16,256]→128│
 │  ↓ LayerNorm                │        │  ↓ Cross-Attention + 門控融合   │
 │  encoder_out: [B, 64, 128]  │        │  saim_out: [B, 16, 128]         │
 │  ↓ Global Avg Pool          │        │  ↓ Global Avg Pool              │
 │  encoder_global: [B, 128]   │        │  saim_global: [B, 128]          │
 └────────────┬────────────────┘        └──────────────┬─────────────────┘
              └──────────────┬──────────────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  融合層 (Fusion)              │
              │  concat([128, 128]) → [B,256] │
              │  Linear(256→128)             │
              │  LayerNorm → GELU → Dropout  │
              │  fused: [B, 128]             │
              └──────────────┬───────────────┘
                             │
               ┌─────────────┴────────────┐
               ▼                          ▼
 ┌─────────────────────────┐  ┌──────────────────────────┐
 │  TrajectoryDecoder       │  │  IntentionDecoder         │
 │  (LSTM, Autoregressive)  │  │  (LSTM, Autoregressive)  │
 │                          │  │                          │
 │  初始位置: bbox 中心      │  │  初始意圖: 0.5            │
 │  last_pos: [B, 2]        │  │                          │
 │                          │  │  每步: [前一意圖 + 上下文]│
 │  每步: [當前位置 + 上下文]│  │  → LSTM → sigmoid → 機率 │
 │  → LSTM → 預測Δ(x,y)    │  │                          │
 │  → 殘差加回當前位置       │  │  step_intents: [B,45,1]  │
 │                          │  │  global_intent: [B,1]    │
 │  pred_traj: [B, 45, 2]  │  │  (取最後隱藏狀態)         │
 └─────────────────────────┘  └──────────────────────────┘
```

### 2.2 模型規模

| 項目 | 數值 |
|------|------|
| d_model (Transformer 隱藏維度) | 128 |
| nhead (Multi-Head Attention 頭數) | 8 |
| 每頭維度 | 128 ÷ 8 = **16** |
| Transformer 層數 | 4 |
| FFN 中間層維度 | 512 |
| LSTM 隱藏維度 | 256 |
| LSTM 層數 | 2 |
| 觀察長度 | 16 幀（~0.5 秒） |
| 預測長度 | 45 幀（~1.5 秒） |
| SAM Patch 數 | 16 個物體 |

---

## 三、逐模組技術詳解

### 3.1 多模態嵌入 (ModalityEmbedding)

**檔案:** `models/encoder/modal_embedding.py`

每個模態都有一個獨立的嵌入網路，將不同維度對齊到統一的 128 維：

```python
# 每個模態的嵌入層結構
self.proj = nn.Sequential(
    nn.Linear(input_dim, d_model),   # 對齊維度：4/2/256 → 128
    nn.LayerNorm(d_model),           # 正規化
    nn.GELU(),                       # 非線性激活
    nn.Dropout(dropout),             # 防過擬合
    nn.Linear(d_model, d_model),     # 再投影
)

# 模態類型 Token：告訴模型「這個 token 來自哪種模態」
self.modality_tokens = nn.Embedding(4, d_model)
# 0 = 軌跡, 1 = 自車, 2 = 光流, 3 = 場景
```

**MultiModalEmbedding** 將 4 種模態嵌入後串接成一個長序列：

```
traj:  [B, 16, 4]   → embed → [B, 16, 128]  (加上 modality token 0)
ego:   [B, 16, 2]   → embed → [B, 16, 128]  (加上 modality token 1)
flow:  [B, 16, 256] → embed → [B, 16, 128]  (加上 modality token 2)
scene: [B, 16, 256] → embed → [B, 16, 128]  (加上 modality token 3)
                                    ↓ torch.cat(dim=1)
                         combined: [B, 64, 128]  ← 4×16=64 個 token
```

### 3.2 位置編碼 (Sinusoidal Positional Encoding)

**檔案:** `models/encoder/positional_encoding.py`

```
數學公式:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  pos = 位置索引 (0~63)
  i   = 維度索引 (0~63，對應 128 維)
```

**直覺理解：**
- 每個位置得到獨特的「正弦指紋」
- 相近位置的指紋相似（內積大），遠的差異大
- 模型可學到「第 15 幀比第 1 幀更重要」

### 3.3 Cross-Modal Transformer Encoder（Pre-LN）

**檔案:** `models/encoder/cross_modal_encoder.py`

**重要設計：使用 Pre-LN（norm_first=True）**

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=8,
    dim_feedforward=512,
    dropout=0.1,
    activation="gelu",
    batch_first=True,
    norm_first=True,    # ← Pre-LN：先 LayerNorm 再 Attention
                        #   訓練更穩定，不需要 warmup 也能收斂
)
```

**Pre-LN vs Post-LN 的差異：**

```
Post-LN (原版 Transformer):
  x → Attention → x + Attention(x) → LayerNorm
  問題: 早期訓練梯度不穩定，需要學習率 warmup

Pre-LN (本專案採用):
  x → LayerNorm → Attention → x + Attention(LN(x))
  優點: 梯度更穩定，訓練初期不容易爆炸
```

**完整 Forward 流程：**

```
combined: [B, 64, 128]
    ↓ SinusoidalPositionalEncoding (加入位置資訊)
    ↓ 4層 TransformerEncoderLayer (Pre-LN, GELU, 8-head attention)
    ↓ LayerNorm (最終輸出正規化)
Z_encoded: [B, 64, 128]   ← 64個token，每個都融合了所有模態的資訊
    ↓ .mean(dim=1)
encoder_global: [B, 128]  ← 全局特徵（所有 token 平均）
```

### 3.4 Enhanced SAIM（核心創新）

**檔案:** `models/saim/enhanced_saim.py`

這是本專案最重要的創新點，讓模型同時理解「物體是什麼」和「物體如何移動」。

#### Step 1：建立動態查詢（Dynamic Query）

```python
# 將 traj + ego + flow 的嵌入串接後投影
dynamic_concat = torch.cat([traj_feat, ego_feat, flow_feat], dim=-1)
# [B, 16, 128*3] = [B, 16, 384]

dynamic_query = self.query_proj(dynamic_concat)
# Linear(384→128) → LayerNorm → GELU
# → [B, 16, 128]
```

**語意：** 「行人的位置 + 自車的狀態 + 場景的運動」= 綜合動態情境查詢

#### Step 2：Cross-Attention（跨模態注意力）

```python
attn_out, _ = self.cross_attention(
    query=dynamic_query,   # [B, 16, 128] ← 問題：「當前動態情境下，哪個物體最重要？」
    key=scene_feat,        # [B, 16, 128] ← 場景中有哪些物體？
    value=scene_feat,      # [B, 16, 128] ← 這些物體的特徵內容
)
# MultiheadAttention: embed_dim=128, num_heads=8
attn_out = norm1(dynamic_query + attn_out)  # 殘差 + LayerNorm
# → [B, 16, 128]
```

**語意：** 光流顯示「有東西快速從右邊靠近」→ SAM 告訴我們「那是一輛車」→ 高危險性

#### Step 3：門控融合（Gated Fusion）

```python
gate_input = torch.cat([attn_out, flow_feat], dim=-1)
# [B, 16, 256]

gate_weight = sigmoid(Linear(256 → 128))
# → [B, 16, 128]，每個維度一個 0~1 的門控值

gated_out = gate_weight * attn_out + (1 - gate_weight) * flow_feat
```

**語意：**
- gate → 1：靜態場景，語義更重要（「車站附近，行人可能過馬路」）
- gate → 0：高速動態，光流更重要（「有東西快速靠近，不管是什麼都要閃」）
- 模型自動學習什麼情境下依賴哪個模態

#### Step 4：FFN + 殘差

```python
output = norm2(gated_out + ffn(gated_out))
# ffn: Linear(128→512) → GELU → Dropout → Linear(512→128)
# → [B, 16, 128]
```

### 3.5 融合層（Encoder + SAIM → fused）

**檔案:** `models/pedformer.py`

```python
# encoder_global: [B, 128]  ← Transformer 看全局跨模態關係
# saim_global:    [B, 128]  ← SAIM 看動態語義交互關係

fused = self.fusion(
    torch.cat([encoder_global, saim_global], dim=-1)  # [B, 256]
)
# Linear(256→128) → LayerNorm → GELU → Dropout
# fused: [B, 128]
```

這個融合讓兩條路徑各司其職：
- Transformer：理解多模態間的全局時序關係
- SAIM：理解動態情境與場景物體的交互

### 3.6 RAFT 光流提取

**檔案:** `models/saim/optical_flow/raft_extractor.py`, `flow_encoder.py`

#### 什麼是光流？

```
幀 t:  [車在左邊]     幀 t+1: [車在中間]
              ↓
光流場: 每個像素 (dx, dy) → 向右偏移的大向量
車的區域: 大向量（移動快）
靜止路面: 接近零向量
```

#### RAFT 演算法

- 2020 ECCV 最佳論文（Teed & Deng）
- 使用迭代 refinement，預設 20 次迭代
- 本專案使用 torchvision 內建預訓練權重 `raft_large_C_T_SKHT_V2`

#### 光流編碼 Pipeline（預計算時執行）

```
連續兩幀 (img_t, img_t+1)
    ↓ RAFT（20次迭代）
光流場: [B, 2, H, W]  ← dx, dy 各一個 channel
    ↓ 通道調整 (2→3 channels, 複製第1個channel) 以適配 ResNet
    ↓ ResNet-50（ImageNet 預訓練，特徵提取層凍結）
ResNet 特徵: [B, 2048, 1, 1]
    ↓ Global Average Pooling + Flatten
    ↓ Linear(2048 → 256) + ReLU
光流特徵 φ_flow: [256]  ← 存成 .npy 檔
```

**快取策略：**
每對相鄰幀預計算一次，存為 `flow_cache/set01_video_0001_00015.npy`。
訓練時直接讀取，避免每次重算 RAFT（節省大量 GPU 時間）。

### 3.7 SAM 語義分割

**檔案:** `models/saim/segmentation/sam_wrapper.py`, `patch_extractor.py`

#### SAM v1（vit_h）規格

| 項目 | 數值 |
|------|------|
| 版本 | SAM v1（Meta AI, 2023） |
| 骨幹 | ViT-Huge (vit_h) |
| 權重檔 | `sam_vit_h_4b8939.pth`（2.4 GB） |
| GPU 記憶體需求 | ~9.8 GB |
| 分割方式 | `SamAutomaticMaskGenerator`（自動，無需提示） |

#### SAM 預測流程

```python
SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,        # 在影像上均勻採樣 32×32=1024 個點作為提示
    pred_iou_thresh=0.86,      # 只保留 IoU 預測 > 0.86 的 mask
    stability_score_thresh=0.92, # 只保留穩定性分數 > 0.92 的 mask
    min_mask_region_area=100,  # 過濾掉小於 100 像素的雜訊 mask
)
```

#### Patch 特徵提取 Pipeline（預計算時執行）

```
單張影像 [H, W, 3]
    ↓ SAM → 自動偵測所有物體
    ↓ 按面積排序，取最大的 16 個物體的二值遮罩 [16, H, W]
    ↓
for i in range(16):
    masked_img = image × mask_i      # [3, H, W]，只留第 i 個物體的像素
    ↓ Conv(3→32, k=3, s=2) → BN → ReLU
    ↓ Conv(32→64, k=3, s=2) → BN → ReLU
    ↓ Conv(64→128, k=3, s=2) → BN → ReLU
    ↓ AdaptiveAvgPool2d(1) → Flatten
    feat_i: [128]
    ↓ Linear(128→256) → LayerNorm → GELU
    patch_feat_i: [256]
    ↓
16 個特徵 → stack → scene_feat: [16, 256]  ← 存成 .npy 檔
```

### 3.8 軌跡解碼器（TrajectoryDecoder）

**檔案:** `models/decoder/trajectory_decoder.py`

#### LSTM 隱藏狀態初始化

```python
# 用 fused 特徵初始化 LSTM 隱藏狀態（而非全零初始化）
h = Linear(128 → 256 × 2)(fused)  # 256 hidden × 2 layers
c = Linear(128 → 256 × 2)(fused)
# reshape → [2, B, 256]  ← LSTM 期待的格式 (layers, batch, hidden)
```

#### Autoregressive 軌跡生成（殘差預測）

```
初始位置: last_pos = bbox 中心 = ((x1+x2)/2, (y1+y2)/2)
初始狀態: (h, c) 由 fused 初始化

for step in range(45):
    lstm_input = concat([current_pos, fused])  # [B, 2+128] = [B, 130]
                   ↓ unsqueeze(1) → [B, 1, 130]
    out, (h, c) = LSTM(lstm_input, (h, c))
                   ↓ out: [B, 1, 256]
    delta = Linear(256→64) → ReLU → Linear(64→2)
    # delta: [B, 2]  ← 預測「位移量」而非絕對座標
    current_pos = current_pos + delta    ← 殘差加回
    outputs.append(current_pos)

pred_traj: [B, 45, 2]
```

**為什麼預測位移（殘差）而非絕對座標？**
- 行人通常不會突然跳很遠，位移量在小範圍內
- 預測小數值比預測大絕對值更穩定
- 即使 LSTM 狀態有些誤差，殘差加回機制會自我修正

### 3.9 意圖解碼器（IntentionDecoder）

**檔案:** `models/decoder/intention_decoder.py`

```
初始意圖: current_intent = 0.5  ← 表示「不確定」

for step in range(45):
    lstm_input = concat([current_intent, fused])  # [B, 1+128] = [B, 129]
                   ↓
    out, (h, c) = LSTM(lstm_input, (h, c))
                   ↓
    intent_logit = Linear(256→32) → ReLU → Linear(32→1)
    current_intent = sigmoid(intent_logit)  ← 轉為機率
    step_outputs.append(current_intent)

step_intents: [B, 45, 1]   ← 每個未來幀的穿越機率

# 全局意圖：用最後一層 LSTM 的隱藏狀態
global_logit = Linear(256→32) → ReLU → Linear(32→1)(h[-1])
global_intent = sigmoid(global_logit)  # [B, 1]
```

### 3.10 多任務損失函數

**檔案:** `losses/multitask_loss.py`, `trajectory_loss.py`, `intention_loss.py`

#### 軌跡損失（TrajectoryLoss）

```
L_traj = MSE(pred_traj, target_traj)          ← 全序列 ADE-like
       + 0.5 × MSE(pred_traj[:,-1], target[:,-1])  ← 終點 FDE-like

MSE = (1/N) × Σ ||預測 - 真實||²
```

**為什麼要額外加 FDE 損失？**
純 ADE 損失讓模型可以「前面很準但最後跑偏」。FDE 加權讓模型特別關注終點精度。

#### 意圖損失（IntentionLoss）

```
L_intent = BCE(global_intent, label)              ← 全局判斷
         + 0.3 × BCE_mean(step_intents, label)    ← 每步監督

BCE = -(y × log(p) + (1-y) × log(1-p))
```

**為什麼逐步監督意圖？**
如果只監督全局，模型可能前面亂猜然後最後才改正。
逐步監督讓整條軌跡上的每一幀都往正確方向學習。

**pos_weight 處理類別不平衡：**
PIE/JAAD 中「不穿越」樣本遠多於「穿越」，
IntentionLoss 使用 `pos_weight` 加大穿越樣本的損失權重，避免模型永遠預測 0。

#### 聯合損失

```
L_total = 1.0 × L_traj + 0.5 × L_intent

（支援可學習不確定性加權，Kendall et al. 2018）
若 learnable_weights=True:
  L_total = exp(-log_σ²_traj) × L_traj  + log_σ²_traj
          + exp(-log_σ²_intent) × L_intent + log_σ²_intent
```

---

## 四、資料 Pipeline（從原始影片到訓練）

### 4.1 完整資料流

```
PIE/JAAD 原始 .mp4 影片
    ↓ [Step 1] 用 imageio-ffmpeg 逐幀提取
影像幀 PNG 存到 /mnt/PIE/images/set01/video_0001/00000.png
    ↓ [Step 2] precompute_flow.py（Docker, GPU）
RAFT 光流特徵 → flow_cache/set01_video_0001_00000.npy  [256 floats]
    ↓ [Step 3] precompute_sam.py（Docker, GPU）
SAM 分割特徵 → sam_cache/set01_video_0001_00000.npy    [16×256 floats]
    ↓ [Step 4] Docker 訓練
DataLoader 讀取 → 模型 → 損失 → 更新
```

### 4.2 快取檔案命名規則

| 資料集 | 檔名格式 | 範例 |
|--------|----------|------|
| PIE 光流 | `{set}_{video}_{frame:05d}.npy` | `set01_video_0001_00015.npy` |
| PIE SAM | `{set}_{video}_{frame:05d}.npy` | `set01_video_0001_00015.npy` |
| JAAD 光流 | `{video}_{frame:05d}.npy` | `video_0001_00015.npy` |
| JAAD SAM | `{video}_{frame:05d}.npy` | `video_0001_00015.npy` |

### 4.3 磁碟空間分配

| 資料 | 路徑 | 大小 |
|------|------|------|
| PIE 影片 | `/mnt/PIE/set01~set06/*.mp4` | ~70 GB |
| PIE 影像幀 | `/mnt/PIE/images/` | ~797 GB |
| JAAD 影片 | `/mnt/JAAD/*.mp4` | ~3.3 GB |
| JAAD 影像幀 | `/home/nutn/JAAD_images/`（symlink 至 `/mnt/JAAD/images`） | ~50 GB（估） |
| 光流快取 | `flow_cache/` | ~700 MB（估） |
| SAM 快取 | `sam_cache/` | ~7 GB（估） |

---

## 五、資料集說明

### 5.1 PIE（Pedestrian Intention Estimation）

- 來源：2019 ICCV
- 第一人稱視角車載攝影機
- 6 小時行車影像，1842 位行人
- 軌跡 + 穿越意圖標註
- 6 個場景 set（set01~set06）
- 每部影片：~13,000 幀（約 10 分鐘，30 fps）

### 5.2 JAAD（Joint Attention in Autonomous Driving）

- 來源：2016 arXiv
- 第一人稱視角車載攝影機
- 346 段影片，每段 ~300 幀（約 10 秒）
- 行人行為標註（穿越、注視方向等）
- 官方提供 train/val/test split

### 5.3 為什麼合併兩個資料集？

```
PIE:  場景多樣性高（6 種地點），但行人數量有限
JAAD: 行人行為標註豐富，但場景較單一

合併 → 更多訓練樣本 → 模型泛化更好
     → 兩個資料集互補 → 學到更多行人行為模式
     → 論文中可分別報告 PIE/JAAD/Combined 的結果
```

---

## 六、評估指標

| 指標 | 全名 | 公式 | 意義 |
|------|------|------|------|
| **ADE** | Average Displacement Error | mean(‖pred_t - gt_t‖₂) 對所有 t | 整體軌跡偏差，越小越好 |
| **FDE** | Final Displacement Error | ‖pred_45 - gt_45‖₂ | 終點偏差，越小越好 |
| **Accuracy** | Intent Accuracy | correct / total | 穿越意圖準確率 |
| **F1** | F1-Score | 2×Precision×Recall / (P+R) | 不平衡資料下比 Accuracy 更可靠 |

---

## 七、訓練流程

```
初始化:
  - 裝置: CUDA（RTX 4070 Ti, 12GB）
  - 優化器: AdamW（lr=0.0001, weight_decay=0.0001）
  - 排程器: CosineAnnealingLR（逐漸降低學習率到接近 0）
  - 資料集模式: combined（PIE + JAAD）

每個 Epoch:
  for batch in train_loader:
    1. 前向傳播: (traj, ego, flow, scene) → 模型 → (pred_traj, step_intents, global_intent)
    2. 損失計算: L = 1.0×L_traj + 0.5×L_intent
    3. 反向傳播: L.backward()
    4. 梯度裁剪: clip_grad_norm_(model.parameters(), max_norm=1.0)
    5. 優化器步進: optimizer.step()
    6. 學習率更新: scheduler.step()

每 Epoch 結束:
  → 驗證集評估: ADE / FDE / Accuracy / F1
  → 若 Val Loss 歷史最佳 → 儲存 weights/pedformer_best.pth
  → 若連續 15 Epoch 無改善 → Early Stopping

超參數:
  - Batch size: 16
  - 最大 Epochs: 100
  - Warmup: 5 Epochs
  - 觀察長度: 16 幀（~0.5秒 @ 30fps）
  - 預測長度: 45 幀（~1.5秒 @ 30fps）
  - Early stopping patience: 15
```

---

## 八、Docker 訓練環境

```yaml
# docker-compose.yml 重點
base image: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
python: 3.10
torch: 2.3.1 + CUDA 12.1

volumes:
  /mnt/PIE                → /workspace/PedFormer/data/PIE
  /mnt/JAAD               → /workspace/PedFormer/data/JAAD
  /home/nutn/JAAD_images  → /workspace/PedFormer/data/JAAD/images  (symlink target)
  ./weights               → /workspace/PedFormer/weights
  ./flow_cache            → /workspace/PedFormer/data/flow_cache
  ./sam_cache             → /workspace/PedFormer/data/sam_cache
  ./logs                  → /workspace/PedFormer/logs
```

---

## 九、專案檔案架構與閱讀順序

**建議按數字順序閱讀：**

```
PedFormer-Enhanced/
│
│  ① configs/default.yaml              ← 先看這裡了解所有超參數
│  ② train.py                          ← 訓練主迴圈（整體流程）
│
├── models/
│  ③ pedformer.py                      ← 主模型 forward()，串接所有模組
│  ④ encoder/modal_embedding.py        ← 各模態嵌入層
│  ⑤ encoder/positional_encoding.py    ← 正弦位置編碼
│  ⑥ encoder/cross_modal_encoder.py    ← Pre-LN Transformer 跨模態編碼
│  ⑦ saim/enhanced_saim.py            ← 核心創新：動態查詢 + Cross-Attention + 門控
│  ⑧ saim/optical_flow/raft_extractor.py  ← RAFT 光流估算
│  ⑨ saim/optical_flow/flow_encoder.py    ← ResNet-50 光流特徵編碼
│  ⑩ saim/segmentation/sam_wrapper.py     ← SAM 自動分割
│  ⑪ saim/segmentation/patch_extractor.py ← CNN Patch 特徵提取
│  ⑫ decoder/trajectory_decoder.py     ← LSTM 軌跡解碼（殘差預測）
│  ⑬ decoder/intention_decoder.py      ← LSTM 意圖解碼（逐步 + 全局）
│
├── losses/
│  ⑭ multitask_loss.py                 ← 聯合損失（支援可學習不確定性加權）
│  ⑮ trajectory_loss.py                ← ADE + FDE 損失
│
├── utils/
│  ⑯ metrics.py                        ← ADE / FDE / F1 計算
│
│  ⑰ evaluate.py                       ← 模型評估腳本
│  ⑱ inference.py                      ← 即時推論
│
├── scripts/
│  ⑲ precompute_flow.py               ← 預計算 RAFT 光流
│  ⑳ precompute_sam.py                ← 預計算 SAM 特徵
│  ㉑ split_clips_to_frames.sh         ← PIE 影片抽幀（舊版，用 ffmpeg）
│
└── hardware/
   ㉒ frontend/capture_stream.py       ← Raspberry Pi 擷取串流
   ㉓ backend/inference_server.py      ← GPU 推論伺服器
```

---

## 十、Git 版本管理

```bash
git status            # 查看修改了什麼
git diff              # 查看具體差異
git add <file>        # 加入暫存
git commit -m "..."   # 建立版本

# Commit 訊息慣例:
# feat:     新功能
# fix:      修 bug
# docs:     文件更新
# refactor: 重構
# exp:      實驗調整
```
