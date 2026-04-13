#!/bin/bash
# 資料準備腳本：提取影像幀 → 預計算光流 → 預計算 SAM
# 在 Docker 容器內執行
set -e

PIE_DIR="${PIE_DIR:-data/PIE}"
JAAD_DIR="${JAAD_DIR:-data/JAAD}"
FLOW_CACHE="data/flow_cache"
SAM_CACHE="data/sam_cache"

echo "============================================"
echo "PedFormer 資料準備腳本"
echo "============================================"

# ============================================
# Step 1: 從影片提取影像幀
# ============================================
echo ""
echo "[Step 1/3] 提取影像幀..."

# --- PIE ---
PIE_IMAGES="${PIE_DIR}/images"
if [ -d "${PIE_IMAGES}" ] && [ "$(ls -A ${PIE_IMAGES} 2>/dev/null)" ]; then
    echo "[PIE] 影像目錄已存在，跳過提取。"
else
    echo "[PIE] 開始提取影像幀..."
    # PIE 影片可能在 PIE_clips/ 或直接在 set01/, set02/...
    for set_dir in set01 set02 set03 set04 set05 set06; do
        # 嘗試 PIE_clips/setXX 和 setXX 兩種路徑
        if [ -d "${PIE_DIR}/PIE_clips/${set_dir}" ]; then
            VIDEO_PATH="${PIE_DIR}/PIE_clips/${set_dir}"
        elif [ -d "${PIE_DIR}/${set_dir}" ]; then
            VIDEO_PATH="${PIE_DIR}/${set_dir}"
        else
            echo "  [PIE] ${set_dir} 影片目錄不存在，跳過。"
            continue
        fi

        for video in ${VIDEO_PATH}/*.mp4; do
            [ -f "$video" ] || continue
            filename=$(basename "$video")
            fname="${filename%.*}"
            out_dir="${PIE_IMAGES}/${set_dir}/${fname}"
            if [ -d "${out_dir}" ] && [ "$(ls -A ${out_dir} 2>/dev/null)" ]; then
                continue  # 已提取
            fi
            mkdir -p "${out_dir}"
            ffmpeg -y -i "$video" -start_number 0 -f image2 -qscale 1 "${out_dir}/%05d.png" -loglevel warning
        done
        echo "  [PIE] ${set_dir} 完成"
    done
fi

# --- JAAD ---
JAAD_IMAGES="${JAAD_DIR}/images"
if [ -d "${JAAD_IMAGES}" ] && [ "$(ls -A ${JAAD_IMAGES} 2>/dev/null)" ]; then
    echo "[JAAD] 影像目錄已存在，跳過提取。"
else
    echo "[JAAD] 開始提取影像幀..."
    JAAD_CLIPS="${JAAD_DIR}/JAAD_clips"
    if [ ! -d "${JAAD_CLIPS}" ]; then
        echo "  [JAAD] JAAD_clips 目錄不存在，跳過。"
    else
        for video in ${JAAD_CLIPS}/*.mp4; do
            [ -f "$video" ] || continue
            filename=$(basename "$video")
            fname="${filename%.*}"
            out_dir="${JAAD_IMAGES}/${fname}"
            if [ -d "${out_dir}" ] && [ "$(ls -A ${out_dir} 2>/dev/null)" ]; then
                continue
            fi
            mkdir -p "${out_dir}"
            ffmpeg -y -i "$video" -start_number 0 -f image2 -qscale 1 "${out_dir}/%05d.png" -loglevel warning
        done
        echo "  [JAAD] 完成"
    fi
fi

# ============================================
# Step 2: 預計算 RAFT 光流特徵
# ============================================
echo ""
echo "[Step 2/3] 預計算 RAFT 光流特徵..."

if [ -d "${PIE_IMAGES}" ]; then
    echo "[PIE] 計算光流..."
    python scripts/precompute_flow.py --data_dir "${PIE_DIR}" --output_dir "${FLOW_CACHE}" --dataset pie
fi

if [ -d "${JAAD_IMAGES}" ]; then
    echo "[JAAD] 計算光流..."
    python scripts/precompute_flow.py --data_dir "${JAAD_DIR}" --output_dir "${FLOW_CACHE}" --dataset jaad
fi

# ============================================
# Step 3: 預計算 SAM 語義分割特徵
# ============================================
echo ""
echo "[Step 3/3] 預計算 SAM 語義分割特徵..."

if [ -d "${PIE_IMAGES}" ]; then
    echo "[PIE] 計算 SAM 特徵..."
    python scripts/precompute_sam.py --data_dir "${PIE_DIR}" --output_dir "${SAM_CACHE}" --dataset pie
fi

if [ -d "${JAAD_IMAGES}" ]; then
    echo "[JAAD] 計算 SAM 特徵..."
    python scripts/precompute_sam.py --data_dir "${JAAD_DIR}" --output_dir "${SAM_CACHE}" --dataset jaad
fi

echo ""
echo "============================================"
echo "資料準備完成！可以開始訓練: python train.py"
echo "============================================"
