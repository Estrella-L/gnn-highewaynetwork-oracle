#!/bin/bash
# 在 EP_high 上跑三篇 baseline（GeGnn / NeuroGF / LiteGE）× 两种输出参数化
set -u
PY=${PY:-python}
OFF=${OFF:-dataset/EP_high/EP_high.off}
LABELS=${LABELS:-outputs/cache_ep_high/surface/surface_random.csv}
EPOCHS=${EPOCHS:-20}
OUTDIR=${OUTDIR:-results}
BATCH=${BATCH:-8192}
mkdir -p "$OUTDIR"

for arch in gegnn neurogf litege; do
  for mode in native euclidean_residual; do
    echo "=== $arch / $mode ==="
    $PY -u train.py --off "$OFF" --labels_csv "$LABELS" --arch "$arch" --out_mode "$mode" \
      --epochs "$EPOCHS" --batch "$BATCH" \
      --pca_cache "${OFF}.litege_pca200.npz" \
      --out_json "$OUTDIR/${arch}_${mode}.json" 2>&1 | tail -40
  done
done

$PY summarize.py --results "$OUTDIR" --out "$OUTDIR/对比总表.md"
