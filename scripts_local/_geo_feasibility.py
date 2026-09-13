import sys, time, math
import numpy as np
sys.path.insert(0, '/Users/zhangronghua/Documents/科研小组/.pylibs')
import pygeodesic.geodesic as pg

def load_off(p):
    tok = open(p, encoding='utf-8', errors='ignore').read().split()
    i = 0
    if tok[0].upper().startswith('OFF'): i = 1
    nv = int(tok[i]); nf = int(tok[i+1]); i += 3
    V = np.array(tok[i:i+3*nv], dtype=np.float64).reshape(nv,3); i += 3*nv
    F = []
    for _ in range(nf):
        k = int(tok[i]); i += 1
        F.append([int(tok[i+j]) for j in range(k)]); i += k
    return V, np.asarray(F, dtype=np.int32)

for name, path in [
    ('small_terrain', '/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/small/small_terrain.off'),
    ('EP_low', '/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_low/EP_low.off'),
]:
    V, F = load_off(path)
    print(f'--- {name}: |V|={len(V)} |F|={len(F)}')
    t0 = time.time()
    geo = pg.PyGeodesicAlgorithmExact(V, F)
    t_build = time.time() - t0
    print(f'    build {t_build:.2f}s')
    # 3 个源，SSSD 到全图
    rng = np.random.default_rng(0)
    srcs = rng.choice(len(V), size=3, replace=False)
    tot = 0.0
    for s in srcs:
        t0 = time.time()
        d, bs = geo.geodesicDistances(np.array([s], dtype=np.int32), None)
        dt = time.time() - t0
        tot += dt
        # 与 3D 欧氏对照
        eu = np.linalg.norm(V - V[s], axis=1)
        ok = d > 0
        ratio = (d[ok] / np.maximum(eu[ok], 1e-9))
        finite = np.isfinite(d)
        print(f'    src={s}: {dt:.2f}s  finite={finite.mean()*100:.1f}%  '
              f'ratio(geo/euclid) min={ratio.min():.4f} median={np.median(ratio):.4f} max={ratio.max():.4f}')
    print(f'    avg per-source SSSD = {tot/3:.2f}s  (开曲面边界边存在，未报错)')
