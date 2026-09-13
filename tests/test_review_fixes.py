import csv
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import gnn
from preprocess import load_label_pairs_csv, split_distance_dataset
from euclidean_geodesic.labels.surface_poison_fill import merge


class ReviewFixTests(unittest.TestCase):
    def test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'labels.csv'
            p.write_text('s,t,distance\n0,1,2\n1,0,2\n')
            self.assertEqual(load_label_pairs_csv(p, num_nodes=3),
                             [{'s': 0, 't': 1, 'distance': 2.0}])
            for row in ['0,1,inf', '0,1,nan', '0,1,-1', '0,1,0', '-1,1,2', '0,3,2', 'bad,1,2', '1,0,3']:
                p.write_text('s,t,distance\n0,1,2\n' + row + '\n')
                with self.subTest(row=row), self.assertRaises(ValueError):
                    load_label_pairs_csv(p, num_nodes=3)
            rows = ''.join(f'{i},{i+1},2\n{i+1},{i},2\n' for i in range(20))
            p.write_text('s,t,distance\n' + rows)
            splits = split_distance_dataset(load_label_pairs_csv(p))
            pairs = [{(x['s'], x['t']) for x in part} for part in splits]
            self.assertFalse(pairs[0] & pairs[1] or pairs[0] & pairs[2] or pairs[1] & pairs[2])

    def test_scale_roundtrip_and_legacy(self):
        for cls, kwargs in [(gnn.DistancePredictor, dict(node_feat_dim=4, highway_feat_dim=4, global_feat_dim=2)),
                            (gnn.SingleGNNPredictor, dict(node_feat_dim=4))]:
            with self.subTest(model=cls.__name__):
                with patch.object(gnn, '_RESIDUAL_SCALE', 1.0):
                    original = cls(**kwargs, prediction_mode='euclidean_residual').eval()
                with patch.object(gnn, '_RESIDUAL_SCALE', 0.5):
                    restored = cls(**kwargs, prediction_mode='euclidean_residual').eval()
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'model.pt'
                    torch.save(original.state_dict(), path)
                    restored.load_state_dict(torch.load(path, weights_only=True))
                self.assertEqual(restored.residual_scale.item(), 1.0)
                if cls is gnn.DistancePredictor:
                    x = torch.randn(2, original.fusion_mlp[0].in_features)
                    fn = lambda m: m._predict_from_fusion(x, euclidean_dist_feat=torch.tensor([2., 3.]))
                else:
                    x, y = torch.randn(2, 32), torch.randn(2, 32)
                    fn = lambda m: m.predict_from_embeddings(x, y, torch.tensor([2., 3.]))
                torch.testing.assert_close(fn(original), fn(restored))
                legacy = {k: v for k, v in original.state_dict().items() if k != 'residual_scale'}
                with self.assertWarnsRegex(UserWarning, 'Legacy checkpoint'):
                    restored.load_state_dict(legacy)

    def test_heat_merge_preserves_audit_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'labels.csv'
            p.write_text('s,t,true_distance,graph_distance\n0,1,inf,3.5\n1,2,2,2.4\n')
            Path(str(p) + '.poisonfill0').write_text('s,t,d\n0,1,3\n')
            merge(str(p))
            with p.open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['graph_distance'], '3.5')
            self.assertEqual(rows[0]['label_method'], 'heat')
            self.assertEqual(rows[1]['label_method'], 'unknown')

if __name__ == '__main__':
    unittest.main()
