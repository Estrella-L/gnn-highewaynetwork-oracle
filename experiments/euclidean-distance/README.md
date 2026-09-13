# Euclidean Distance GNN Experiments

This directory contains the experimental extension of the original project.
The original repository files are preserved unchanged. Highway InterGNN
(Inner / Inter / Fusion) remains the model architecture.

Features include Euclidean residual prediction, sampling variants, validation
metric selection, grouped evaluation, aligned baselines, and transit sparsification.
Inference supports residual modes, 3D coordinates, and canonical undirected pairs.

Run commands from this directory. Install the repository dependencies in your
training environment. Large datasets, caches, checkpoints, and local SSH helpers
are not included. Supply an absolute terrain path when running cloud experiments.

```bash
cd experiments/euclidean-distance
python main.py --help
python infer_distance.py --help
python main.py --off_file data/sample_terrain.off --file_folder . --max_depth 2 --capacity 16 --distance_samples 100 --num_epoch 1 --hidden_dim 16 --out_dim 8 --prediction_mode euclidean_residual --disable_highway_distance_feature --device cpu
```

For inference, use the same dimensions, partition settings, prediction mode,
highway_k, transit_k, and highway feature setting as training. Preserve the
generated parameter file alongside each checkpoint.

Historical cloud logs report EP_low test relative error 0.006351 (100,000 pairs,
seed 42) and EP_high 0.02858457 (50,000 pairs, seed 42). These are previous run
results, not results reproduced by this upload. Multi-seed and aligned baseline
evaluation remain necessary for stronger experimental conclusions.
