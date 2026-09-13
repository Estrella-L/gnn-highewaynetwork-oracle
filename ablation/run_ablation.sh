#!/bin/bash
# EP_high 解剖式消融：A2/A3（单 GNN）先跑，再跑 A1（三段式去欧氏残差，100 轮）
set -u
ROOT=/root/autodl-tmp/gnn-euclidean-local
PY=/root/autodl-tmp/conda/envs/torch_pro6000/bin/python
OFF=$ROOT/dataset/EP_high/EP_high.off
LB=$ROOT/outputs/cache_ep_high/surface/surface_random.csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $ROOT/ablation_ep_high || exit 1
mkdir -p results
LOG=results/ablation.log
echo "[$(date '+%F %H:%M:%S')] === A2 单 GNN + 欧氏残差 ===" >> "$LOG"
$PY -u single_gnn.py --off "$OFF" --labels_csv "$LB" --prediction_mode euclidean_residual \
    --epochs 100 --out_json results/A2_single_gnn_euclid_residual.json >> "$LOG" 2>&1

echo "[$(date '+%F %H:%M:%S')] === A3 单 GNN + 直接预测 ===" >> "$LOG"
$PY -u single_gnn.py --off "$OFF" --labels_csv "$LB" --prediction_mode direct \
    --epochs 100 --out_json results/A3_single_gnn_direct.json >> "$LOG" 2>&1

echo "[$(date '+%F %H:%M:%S')] === A1 三段式（分区+highway）去掉欧氏残差 ===" >> "$LOG"
cd $ROOT || exit 1
CLOUD_LOG=$ROOT/logs/ablation_a1_train.log $PY -u scripts_local/cloud_ep_high_runner.py \
    --off_file dataset/EP_high/EP_high.off --workers 32 \
    --grid_json scripts_local/ep_high_grid_ablation_a1.json --run_tag ep_high_ablation_a1 \
    --only prep,sample,grid --chunk 64 >> "$LOG" 2>&1

echo "[$(date '+%F %H:%M:%S')] ALL DONE" >> "$LOG"
touch $ROOT/ablation_ep_high/results/DONE
