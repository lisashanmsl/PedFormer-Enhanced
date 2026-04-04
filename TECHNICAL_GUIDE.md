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

```
輸入 (車載攝影機第一人稱視角)
 │
 ├── 行人歷史軌跡 [16幀 × bbox(x1,y1,x2,y2)]
 ├── 自車運動資料 [16幀 × (速度, 轉向角)]
 ├── RAFT 光流特徵 [16幀 × 256維向量]
 └── SAM 場景特徵 [16個物體 × 256維向量]
       │
       ▼
 ┌─────────────────────────────────────────┐
 │     Enhanced SAIM (語義注意力交互模組)     │
 │                                         │
 │  動態查詢 = [軌跡 + 自車 + 光流] 串接    │
 │  場景特徵 = SAM 分割出的物體             │
 │                                         │
 │  Cross-Attention: 「這些物體對行人        │
 │                    的影響有多大？」        │
 │  門控融合: 動態決定語義 vs 動態的比重      │
 └────────────────┬────────────────────────┘
                  │
                  ▼
 ┌─────────────────────────────────────────┐
 │   Cross-Modal Transformer Encoder       │
 │                                         │
 │  所有模態的 token 互相 attend            │
 │  → 產出融合的時空特徵 Z_encoded          │
 └────────────────┬────────────────────────┘
                  │
          ┌───────┴───────┐
          ▼               ▼
 ┌─────────────┐  ┌──────────────┐
 │ LSTM 軌跡    │  │ LSTM 意圖     │
 │ 解碼器       │  │ 解碼器        │
 │             │  │              │
 │ 逐步生成    │  │ 逐步預測      │
 │ 未來座標    │  │ 穿越機率      │
 └──────┬──────┘  └──────┬───────┘
        │                │
        ▼                ▼
  軌跡 [45, 2]    意圖 [1] (0~1)
```

---

## 三、逐模組技術詳解

### 3.1 多模態嵌入 (Modal Embedding)

**檔案:** `models/encoder/modal_embedding.py`

**什麼是「模態」？**
模態 = 一種資料來源。本系統有 4 種：

| 模態 | 原始維度 | 內容 | 嵌入後維度 |
|------|---------|------|----------|
| traj (軌跡) | 4 | 行人 bbox: x1, y1, x2, y2 | 128 |
| ego (自車) | 2 | 車速, 轉向角 | 128 |
| flow (光流) | 256 | RAFT 編碼的運動特徵 | 128 |
| scene (場景) | 256 | SAM 分割的物體特徵 | 128 |

**為什麼要嵌入？**
- 不同模態的原始維度不同 (4, 2, 256, 256)
- Transformer 需要所有輸入維度一致
- 嵌入層 = 一個小型神經網路，把任意維度映射到統一的 128 維

**程式碼核心邏輯：**
```python
# 每個模態的嵌入: 原始維度 → 128 維
self.proj = nn.Sequential(
    nn.Linear(input_dim, d_model),   # 線性投影
    nn.LayerNorm(d_model),           # 正規化
    nn.GELU(),                       # 非線性激活
    nn.Dropout(dropout),             # 防過擬合
    nn.Linear(d_model, d_model),     # 再一層投影
)

# 模態類型標記: 告訴 Transformer「這個 token 來自哪種模態」
self.modality_tokens = nn.Embedding(4, d_model)
# 0 = 軌跡, 1 = 自車, 2 = 光流, 3 = 場景
```

### 3.2 位置編碼 (Positional Encoding)

**檔案:** `models/encoder/positional_encoding.py`

**為什麼需要？**
- Transformer 不像 LSTM，它不知道 token 的「順序」
- 位置編碼 = 告訴模型「這是第幾個時間步」
- 使用正弦/餘弦函數生成，不需要學習

**數學原理：**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos = 位置 (0, 1, 2, ..., 15)
i   = 維度索引
```

**直覺理解：**
- 每個位置得到一組獨特的「指紋」
- 相近位置的指紋相似，遠的位置差異大
- 模型可以學會「3 幀前的資料」比「10 幀前」更相關

### 3.3 Cross-Modal Transformer Encoder

**檔案:** `models/encoder/cross_modal_encoder.py`

**這是什麼？**
Transformer 的核心 = **Self-Attention (自注意力)**

把所有模態的所有時間步排成一個長序列，讓每個 token 都能「看到」其他所有 token：

```
輸入序列 = [traj_1, traj_2, ..., traj_16,   ← 16 個軌跡 token
            ego_1,  ego_2,  ..., ego_16,    ← 16 個自車 token
            flow_1, flow_2, ..., flow_16,   ← 16 個光流 token
            scene_1, scene_2, ..., scene_16] ← 16 個場景 token

總共 64 個 token，每個都是 128 維
```

**Self-Attention 的直覺：**
```
Q (Query)  = 「我在找什麼資訊？」
K (Key)    = 「我有什麼資訊？」
V (Value)  = 「如果你需要我，這是我的內容」

Attention(Q, K, V) = softmax(Q × K^T / √d) × V

例如: traj_16 (最後一幀軌跡) 在做 attention 時
  → 發現 flow_15 (前一幀光流) 的 K 跟自己的 Q 很相似
  → 給 flow_15 的 V 很高的權重
  → 意思: 「最後觀察到的位置」很依賴「前一刻的場景運動」
```

**Multi-Head Attention (多頭注意力)：**
- 8 個 head，每個關注不同的特徵關係
- Head 1 可能關注「軌跡 ↔ 光流」的關係
- Head 2 可能關注「軌跡 ↔ 場景」的關係
- 最終合併所有 head 的結果

**本專案的設定：**
```yaml
d_model: 128          # 隱藏維度
nhead: 8              # 注意力頭數 (128/8=16 維/頭)
num_encoder_layers: 4  # 堆疊 4 層 Transformer
dim_feedforward: 512   # FFN 中間層維度
```

### 3.4 Enhanced SAIM (核心創新)

**檔案:** `models/saim/enhanced_saim.py`

**這是本專案最重要的創新點。**

原始 PedFormer 的 SAIM 只用靜態語義圖，我們改良為：

```
Step 1: 建立「動態查詢」
  動態查詢 = concat(軌跡特徵, 自車特徵, 光流特徵)
  → 投影到 128 維
  意義: 「結合行人位置 + 車的狀態 + 場景運動」形成一個綜合查詢

Step 2: Cross-Attention (交叉注意力)
  Query = 動態查詢 (「行人在什麼動態情境下？」)
  Key/Value = SAM 場景特徵 (「場景中有哪些物體？」)

  → 模型學會: 「光流顯示有東西快速靠近 + 場景中有一輛車 = 危險！」

Step 3: 門控融合 (Gated Fusion)
  gate = sigmoid(Linear([attention結果, 光流特徵]))
  output = gate × attention結果 + (1-gate) × 光流特徵

  → gate ≈ 1: 信任語義+注意力的結果 (靜態場景下)
  → gate ≈ 0: 更依賴光流動態資訊 (高速場景下)
  → 模型自動學會什麼時候該看語義、什麼時候該看動態
```

**為什麼這是創新？**
- PedFormer: 只有語義 → 看不到「車在動」
- PTINet: 只有光流 → 不知道「那個動的東西是車還是樹影」
- Enhanced SAIM: 同時知道「那是一輛車」(SAM) 且「它在快速靠近」(光流)

### 3.5 RAFT 光流提取

**檔案:** `models/saim/optical_flow/raft_extractor.py`

**什麼是光流 (Optical Flow)？**
```
光流 = 影像中每個像素在連續兩幀間的位移向量

幀 t:   [車在左邊]     幀 t+1: [車移到中間]
         ↓
光流圖:  每個像素有一個 (dx, dy) 向量
         → 車的區域有很大的向右向量
         → 靜止物體的區域向量接近零
```

**RAFT (Recurrent All-Pairs Field Transforms) 演算法：**
- 2020 年 ECCV 最佳論文
- 目前最先進的光流估算方法
- 使用迭代 refinement (預設 20 次迭代)
- 本專案使用 torchvision 內建的預訓練 RAFT

**光流編碼 Pipeline:**

**檔案:** `models/saim/optical_flow/flow_encoder.py`

```
連續兩幀 → RAFT → 光流場 [2, H, W] (dx, dy)
                    ↓
              通道調整 (2→3 channels，適配 ResNet)
                    ↓
              ResNet-50 backbone (預訓練，凍結權重)
                    ↓
              全局平均池化 → 2048 維
                    ↓
              線性投影 → 256 維特徵向量 φ_flow
```

### 3.6 SAM 語義分割

**檔案:** `models/saim/segmentation/sam_wrapper.py`, `patch_extractor.py`

**什麼是 SAM (Segment Anything Model)？**
- Meta AI 的 2023 年基礎模型
- 「零樣本分割」= 不需要針對特定類別訓練，就能分割出所有物體
- 輸入任意影像 → 輸出所有物體的精確遮罩 (mask)

**為什麼用 SAM 取代傳統語義分割？**
```
傳統語義分割 (如 DeepLab):
  - 固定類別 (車、人、路)
  - 邊界粗糙
  - 看不到「不認識的物體」

SAM:
  - 不限類別，能分割任何物體
  - 邊界精確
  - 泛化能力強 (不同城市、天氣都能用)
```

**Patch 特徵提取 Pipeline:**
```
SAM 分割 → 16 個物體遮罩 [16, H, W]
              ↓
        每個遮罩 × 原圖 = 16 個 masked 區域
              ↓
        輕量 CNN (3層卷積) 提取特徵
              ↓
        16 個 256 維向量 → [16, 256] = 場景特徵
```

### 3.7 雙流 LSTM 解碼器

**檔案:** `models/decoder/trajectory_decoder.py`, `intention_decoder.py`

**為什麼用 LSTM 而不是 Transformer 解碼？**
```
Transformer 解碼器:
  - 優點: 並行計算快
  - 缺點: 生成軌跡時，每一步都要看完整序列，計算量大

LSTM 解碼器 (PTINet 驗證過):
  - 優點: 天然的時序遞歸結構，生成軌跡很直覺
  - 優點: 記憶細胞能保持長期依賴
  - 本專案: 前端用 Transformer 編碼，後端用 LSTM 解碼 = 取兩者所長
```

**軌跡解碼器的工作方式 (Autoregressive)：**
```
初始化:
  - 用 Encoder 的全局特徵初始化 LSTM 隱藏狀態
  - 起始位置 = 最後觀察到的行人位置 (bbox 中心)

逐步生成 (共 45 步):
  Step 1: 輸入 = [當前位置, 上下文特徵] → LSTM → 預測位移 Δ(x,y)
          下一位置 = 當前位置 + Δ(x,y)   ← 殘差預測
  Step 2: 輸入 = [Step1的位置, 上下文] → LSTM → 預測 Δ(x,y)
          ...
  Step 45: 最終位置

為什麼用「殘差預測」？
  - 直接預測絕對座標 → 誤差大
  - 預測位移量 (多少偏移) → 更穩定，誤差小
```

**意圖解碼器：**
```
同樣用 LSTM，但每步輸出穿越機率 (0~1)
  - 逐步意圖: [batch, 45, 1] → 未來每一刻的穿越傾向
  - 全局意圖: [batch, 1] → 取最後隱藏狀態 → 最終判定

為什麼需要兩個解碼器而不是一個？
  → PTINet 的實驗證明:
     分開解碼 > 共享解碼，因為:
     - 軌跡是連續數值回歸問題 (座標)
     - 意圖是離散二元分類問題 (過/不過)
     - 兩個任務的數學性質不同，強制共享反而干擾
     - 但它們「共享編碼特徵」，這讓兩個任務互相增強 (多任務學習)
```

### 3.8 多任務損失函數

**檔案:** `losses/multitask_loss.py`

```
L_total = w_traj × L_traj + w_intent × L_intent

L_traj = MSE(預測軌跡, 真實軌跡) + 0.5 × MSE(預測終點, 真實終點)
         ↑ ADE-like (全序列)         ↑ FDE-like (終點加權)

L_intent = BCE(全局意圖, 標籤) + 0.3 × BCE(逐步意圖, 標籤)
           ↑ 二元交叉熵               ↑ 每步都監督

預設權重: w_traj=1.0, w_intent=0.5
```

**什麼是 MSE Loss？**
```
MSE = (1/N) × Σ (預測值 - 真實值)²
→ 預測越接近真實，loss 越小
→ 用於軌跡座標 (連續數值)
```

**什麼是 BCE Loss？**
```
BCE = -(y × log(p) + (1-y) × log(1-p))
→ y=1 (會穿越) 時，p 越接近 1 loss 越小
→ y=0 (不穿越) 時，p 越接近 0 loss 越小
→ 用於意圖分類 (二元)
```

---

## 四、資料集說明

### 4.1 PIE (Pedestrian Intention Estimation)

- 來源: 2019 ICCV
- 第一人稱視角車載攝影機
- 6 小時行車影像
- 1842 位行人的軌跡 + 穿越意圖標註
- 6 個場景 set (set01~set06)

### 4.2 JAAD (Joint Attention in Autonomous Driving)

- 來源: 2016 arXiv
- 第一人稱視角車載攝影機
- 346 段影片
- 行人行為標註 (穿越、注視方向等)
- 官方提供 train/val/test split

### 4.3 為什麼合併兩個資料集？

```
PIE:  場景多樣性高，但行人數量有限
JAAD: 行人行為標註豐富，但場景較單一

合併:
  → 更多訓練樣本 → 模型泛化更好
  → 兩個資料集互補 → 學到更多行人行為模式
  → 論文中比較時可以分別報告 PIE/JAAD/Combined 的結果
```

---

## 五、評估指標

| 指標 | 全名 | 公式 | 意義 |
|------|------|------|------|
| **ADE** | Average Displacement Error | mean(‖pred - gt‖₂) 所有步 | 整體軌跡偏差，越小越好 |
| **FDE** | Final Displacement Error | ‖pred[-1] - gt[-1]‖₂ | 終點偏差，越小越好 |
| **Accuracy** | Intent Accuracy | correct / total | 穿越意圖分對幾成 |
| **F1** | F1-Score | 2×P×R/(P+R) | 精確率與召回率的調和平均 |

---

## 六、訓練流程

```
每個 Epoch:
  1. 從 PIE+JAAD 取一批資料 (batch_size=16)
  2. 前向傳播: 資料 → 模型 → 預測軌跡 + 意圖
  3. 計算損失: L = w_traj × L_traj + w_intent × L_intent
  4. 反向傳播: 計算每個參數的梯度
  5. 梯度裁剪: 防止梯度爆炸 (max_norm=1.0)
  6. 更新權重: AdamW 優化器
  7. 學習率調整: Cosine Annealing (逐漸降低)

每個 Epoch 結束後:
  - 在驗證集上評估 ADE/FDE/F1
  - 如果比歷史最佳好 → 儲存模型
  - 如果連續 15 個 Epoch 沒改善 → 早停

超參數:
  - 學習率: 0.0001 (AdamW)
  - Batch size: 16
  - 最大 Epochs: 100
  - 觀察長度: 16 幀 (~0.5秒)
  - 預測長度: 45 幀 (~1.5秒)
```

---

## 七、專案檔案架構與閱讀順序

**建議按數字順序閱讀：**

```
PedFormer-Enhanced/
│
│  ① configs/default.yaml              ← 先看這裡了解所有參數
│  ② train.py                          ← 看訓練主迴圈理解整體流程
│  ③ data/dataset.py                   ← 理解資料怎麼來的
│
├── models/
│  ④ pedformer.py                      ← 主模型 forward()，串接所有模組
│  ⑤ encoder/modal_embedding.py        ← 各模態嵌入
│  ⑥ encoder/positional_encoding.py    ← 位置編碼
│  ⑦ encoder/cross_modal_encoder.py    ← Transformer 跨模態編碼
│  ⑧ saim/enhanced_saim.py            ← 核心創新模組
│  ⑨ saim/optical_flow/raft_extractor.py  ← RAFT 光流
│  ⑩ saim/optical_flow/flow_encoder.py    ← ResNet-50 光流編碼
│  ⑪ saim/segmentation/sam_wrapper.py     ← SAM 分割
│  ⑫ saim/segmentation/patch_extractor.py ← Patch 特徵
│  ⑬ decoder/trajectory_decoder.py     ← LSTM 軌跡解碼
│  ⑭ decoder/intention_decoder.py      ← LSTM 意圖解碼
│
├── losses/
│  ⑮ multitask_loss.py                 ← 聯合損失函數
│
├── utils/
│  ⑯ metrics.py                        ← 評估指標
│  ⑰ visualization.py                  ← 視覺化
│
│  ⑱ evaluate.py                       ← 模型評估
│  ⑲ inference.py                      ← 即時推論
│
├── hardware/
│  ⑳ frontend/capture_stream.py        ← Raspberry Pi 擷取
│  ㉑ backend/inference_server.py       ← GPU 推論伺服器
│
└── scripts/
   ㉒ precompute_flow.py               ← 預計算光流
   ㉓ precompute_sam.py                ← 預計算 SAM
```

---

## 八、日常 Git 版本管理指令

```bash
# 查看目前修改了什麼
git status

# 查看具體改了哪些程式碼
git diff

# 加入修改的檔案到暫存區
git add models/pedformer.py          # 加入特定檔案
git add .                            # 加入所有修改

# 建立新版本 (commit)
git commit -m "fix: 修正 SAIM 門控融合維度錯誤"

# 推送到 GitHub
git push

# 查看版本歷史
git log --oneline

# Commit 訊息慣例:
#   feat: 新功能
#   fix:  修 bug
#   docs: 文件更新
#   refactor: 重構 (不改功能)
#   exp:  實驗調整
```
