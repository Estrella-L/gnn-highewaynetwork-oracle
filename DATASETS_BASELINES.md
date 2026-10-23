# Datasets and baseline code

This branch adds the six terrain dataset label/split files and the terrain-adapted GeGnn, NeuroGF and LiteGE baseline implementations. Existing main-branch code and results are preserved.

- [Datasets and protocols](datasets/README.md)
- [Baseline implementations and EP-high run command](baselines/README.md)
- [Input data assets](https://github.com/Estrella-L/gnn-highewaynetwork-oracle/releases/tag/terrain-datasets-20261023)

Large meshes/elevation arrays are stored as gzip Release assets. Decompress them into their dataset directories and verify the original SHA-256 hashes in datasets/manifest.json. Compression is lossless and does not reduce terrain resolution, query counts, or training budgets. LA is the original 4000 x 4000 elevation array; the local archive does not include LA.off.

```bash
python scripts/fetch_terrain_data.py --dataset EP_high
python baselines/train.py --off datasets/EP_high/EP_high.off --labels_csv datasets/EP_high/labels.csv --arch neurogf --out_mode euclidean_residual --epochs 100 --seed 42 --out_json outputs/baselines/EP_high_neurogf.json
```

No training runs were performed during this upload. Complete baseline comparisons across all six terrains are not claimed.
