# Terrain-adapted baseline implementations

GeGnn, NeuroGF and LiteGE implementations used for the historical EP-high experiments. These are our terrain adaptations, not unmodified official repositories. GeGnn omits hierarchical pooling; LiteGE uses terrain patches for the UDF-PCA descriptor. See HISTORICAL_README.md for architecture details.

Labels are approximate surface geodesic lengths (flip-out, with historical heat-method fallback), not certified exact MMP distances. Historical references to exact labels in archived comments/README are superseded by this note.

Run from the repository root after installing requirements.txt and baselines/requirements.txt:

```bash
python baselines/train.py --off datasets/EP_high/EP_high.off --labels_csv datasets/EP_high/labels.csv --arch neurogf --out_mode euclidean_residual --epochs 100 --seed 42 --out_json outputs/baselines/EP_high_neurogf.json
```

The historical runner shuffles labels with seed 42 before the 80/10/10 split. The five new datasets use fixed-order splits; this runner must not be presented as reproducing that protocol without adapting the data loader. Full three-method baseline comparisons are available on EP-high only. BH-high has partial early baseline results; SF-high, USGS-CO, USGS-WA and LA do not have complete baseline comparisons.

`euclidean_residual` uses scale 0.5. Our separate scale-1.0 EP-high result is not a matched scale-0.5 comparison.
