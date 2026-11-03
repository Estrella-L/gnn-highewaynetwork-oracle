# AccessTopK 流式 topk 视图的回归测试（大图 OOM 修复的正确性验证）
# 用法：在仓库根目录执行  python -u scripts_local/test_accesstopk_regression.py
# 验证目标：AccessTopK（流式滚动 top-16）与原稠密 [N,K] 矩阵在**所有消费口径**下语义一致：
#   [1] precompute_nearest_k 的 top-16 距离序列逐行相等（含 inf/不可达、批大小非约数的 flush 边界）
#   [2] 行视图 getitem/len/enumerate 协议（覆盖 baseline._nearest_indices 的整行扫描兜底）
#   [3] 随 outputs/cache 的 torch.save/load（pickle）往返
#   [4] 真实网格 sample_terrain 端到端：强制稠密 vs 强制 topk 各跑一遍 build_pipeline_inputs，
#       transit 边集完全一致 + topk 距离一致
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_highway as bh
from build_highway import AccessTopK, _merge_access_topk
from preprocess import precompute_nearest_k

rng = np.random.default_rng(0)
N, K, keep = 4000, 120, 16
dense = rng.random((N, K)).astype(np.float32)
dense[rng.random((N, K)) < 0.3] = np.inf   # 部分不可达
dense[0] = np.inf                          # 整行不可达极端行

# ---- [1] 流式滚动 topk vs 稠密（批大小 13 非约数，覆盖末尾 flush）----
td = np.full((N, keep), np.inf, dtype=np.float32)
ti = np.full((N, keep), -1, dtype=np.int32)
pend = []
for j in range(K):
    pend.append((j, dense[:, j].copy()))
    if len(pend) >= 13:
        _merge_access_topk(td, ti, pend, keep)
        pend.clear()
if pend:
    _merge_access_topk(td, ti, pend, keep)
view = AccessTopK(ti, td, K)

ref = precompute_nearest_k(dense, k_max=keep)   # 稠密原逻辑
got = precompute_nearest_k(view, k_max=keep)    # 视图快路径
assert got.shape == ref.shape == (N, keep), (got.shape, ref.shape)
rows = np.arange(N)[:, None]
ref_d = dense[rows, ref.astype(np.int64)]
got_d = np.where(got >= 0, dense[rows, np.clip(got, 0, K - 1).astype(np.int64)], np.inf)
assert np.array_equal(np.sort(ref_d, 1), np.sort(got_d, 1)), "topk 距离序列不一致"
print("[1] 流式 topk vs 稠密: precompute_nearest_k 距离序列逐行一致 OK", flush=True)

# ---- [2] 行视图语义：命中/未命中/len/enumerate ----
r = view[7]
assert len(r) == K
top_set = {int(x) for x in ti[7].tolist()}
for j in (0, 1, 5, 60, 119):
    got_v = float(r[j])
    exp_v = float(dense[7, j]) if j in top_set else float("inf")
    assert got_v == exp_v, (j, got_v, exp_v)
finite_cnt = sum(1 for d in r if d != float("inf"))     # baseline._nearest_indices 式整行枚举
assert finite_cnt == int(np.isfinite(td[7]).sum()), (finite_cnt, int(np.isfinite(td[7]).sum()))
assert float(view[0][0]) == float("inf")                # 全 inf 行
print("[2] 行视图 getitem/len/enumerate 语义 OK", flush=True)

# ---- [3] 随缓存 torch.save/load 往返 ----
p = os.path.join(tempfile.mkdtemp(), "ctx.pt")
torch.save({"access_dist": view}, p, pickle_protocol=4)
v2 = torch.load(p, map_location="cpu", weights_only=False)["access_dist"]
assert isinstance(v2, AccessTopK) and v2.shape == (N, K)
j0 = int(ti[7, 0])
assert float(v2[7][j0]) == float(td[7, 0])
print("[3] torch.save/load pickle 往返 OK", flush=True)

# ---- [4] 端到端：sample_terrain 强制 topk 路径 vs 强制稠密路径 ----
off = r"data\sample_terrain.off"
bh.ACCESS_TOPK_KEEP = 10**9      # 强制稠密
*_, ctx_dense = bh.build_pipeline_inputs(off, 3, 32, True, True, 64, "cpu", transit_k=8)
ad = np.asarray(ctx_dense["access_dist"], dtype=np.float32)
K_real = ad.shape[1]
bh.ACCESS_TOPK_KEEP = 4          # 强制 topk 路径（K_real > 4 时生效）
*_, ctx_topk = bh.build_pipeline_inputs(off, 3, 32, True, True, 64, "cpu", transit_k=8)
at = ctx_topk["access_dist"]
print(f"    sample_terrain: N={ad.shape[0]}, K={K_real}, topk路径类型={type(at).__name__}", flush=True)
# transit 边（高速图边集）两路径必须完全一致
eu_d = ctx_dense["edge_index_highway"].cpu().numpy()
eu_t = ctx_topk["edge_index_highway"].cpu().numpy()
key_d = np.sort(eu_d[0].astype(np.int64) * (K_real + 1) + eu_d[1])
key_t = np.sort(eu_t[0].astype(np.int64) * (K_real + 1) + eu_t[1])
assert eu_d.shape == eu_t.shape and np.array_equal(key_d, key_t), "transit 边不一致"
if isinstance(at, AccessTopK):
    kk = min(4, K_real)
    ref4 = precompute_nearest_k(ad, k_max=kk)
    got4 = precompute_nearest_k(at, k_max=kk)
    rr = np.arange(ad.shape[0])[:, None]
    rd = ad[rr, ref4.astype(np.int64)]
    gd = np.where(got4 >= 0, ad[rr, np.clip(got4, 0, K_real - 1).astype(np.int64)], np.inf)
    assert np.array_equal(np.sort(rd, 1), np.sort(gd, 1)), "端到端 top4 不一致"
    print("[4] 端到端：topk 路径与稠密路径 top-4 距离一致、transit 边集一致 OK", flush=True)
else:
    print("[4] 端到端：sample K<=4，两路径同为稠密，transit 边集一致 OK", flush=True)

print("ALL OK", flush=True)
