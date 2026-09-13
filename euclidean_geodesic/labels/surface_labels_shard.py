# -*- coding: utf-8 -*-
"""表面测地标签：分片独立进程版（不使用 multiprocessing.Pool）。

关键设计（针对 potpourri3d flip-out 在某些点对上会 SIGSEGV 的问题）：
  - 每个点对在计算【之前】先写入 .tried 日志（含 (s,t)），算完写结果
  - 重启后：.tried 中且已有结果的跳过；.tried 中但无结果的视为“毒点对”，直接跳过
    → 崩溃不会导致死循环，最多损失个别点对
  - 结果逐对写入 <out_csv>.shard<i>（行级 flush），进度不丢

用法:
  python surface_labels_shard.py --off_file X.off --pairs_csv in.csv --out_csv main.csv --shard 0 --shards 6
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build_solver(off_path):
    import potpourri3d as pp3d
    npz = off_path + ".npz"
    if os.path.exists(npz):
        z = np.load(npz)
        V = np.asarray(z["V"], dtype=np.float64)
        F = np.asarray(z["F"], dtype=np.int64)
    else:
        from build_highway import load_off
        V, F = load_off(off_path)
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)
    return pp3d.EdgeFlipGeodesicSolver(V, F)


def read_pairs(path):
    out = set()
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                out.add((int(row["s"]), int(row["t"])))
            except Exception:
                continue
    return out


def compute_isolated(solver, s, t, timeout=180):
    """在 fork 出的子进程里算一个点对：子进程段错误不会带走父进程（solver 只建一次）。

    返回 (距离, 是否崩溃)；崩溃 -> (None, True)
    """
    import signal

    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(rfd)
            try:
                path = solver.find_geodesic_path(int(s), int(t))
                d = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            except Exception:
                d = float("inf")
            try:
                os.write(wfd, ("%.6f" % d).encode())
            except Exception:
                pass
            try:
                os.close(wfd)
            except Exception:
                pass
        finally:
            os._exit(0)

    os.close(wfd)
    buf = b""

    def _alarm(signum, frame):
        raise TimeoutError("child timeout")

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    timed_out = False
    try:
        while True:
            chunk = os.read(rfd, 64)
            if not chunk:
                break
            buf += chunk
        _, status = os.waitpid(pid, 0)
    except TimeoutError:
        timed_out = True
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            os.waitpid(pid, 0)
        except Exception:
            pass
        status = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        try:
            os.close(rfd)
        except Exception:
            pass

    if timed_out or status is None:
        return None, True
    if os.WIFSIGNALED(status):
        return None, True
    if not buf:
        return float("inf"), False
    try:
        return float(buf.decode().strip()), False
    except Exception:
        return float("inf"), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off_file", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--no_fork", action="store_true", help="不用子进程隔离（调试用）")
    args = ap.parse_args()

    rows = []
    with open(args.pairs_csv, encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        has_graph = "distance" in (rdr.fieldnames or [])
        for row in rdr:
            rows.append((int(row["s"]), int(row["t"]),
                         float(row["distance"]) if has_graph else None))

    tried_path = args.out_csv + ".tried%d" % args.shard
    nowork_path = args.out_csv + ".nowork%d" % args.shard
    if os.path.exists(nowork_path):
        try:
            os.remove(nowork_path)
        except OSError:
            pass
    done = read_pairs(args.out_csv)
    tried = read_pairs(tried_path)
    todo = [(s, t, g) for i, (s, t, g) in enumerate(rows)
            if i % args.shards == args.shard and (s, t) not in done and (s, t) not in tried]
    poisoned = [(s, t) for i, (s, t, g) in enumerate(rows)
                if i % args.shards == args.shard and (s, t) in tried and (s, t) not in done]
    print("[shard%d] 待算 %d 对 | 已完成 %d | 毒点对 %d" %
          (args.shard, len(todo), len(done), len(poisoned)), flush=True)

    # 毒点对写 inf 占位（直接写主表，保证最终覆盖 50000 对）
    if poisoned:
        hdr = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
        with open(args.out_csv, "a", encoding="utf-8-sig", newline="", buffering=1) as f0:
            w0 = csv.writer(f0)
            if hdr:
                w0.writerow(["s", "t", "true_distance", "graph_distance"])
            for s, t in poisoned:
                w0.writerow([int(s), int(t), "inf", ""])
            f0.flush()
        print("[shard%d] 毒点对占位 %d 对（待热方法回填）" % (args.shard, len(poisoned)), flush=True)

    if not todo:
        open(nowork_path, "w").close()
        print("[shard%d] 无待算点对" % args.shard, flush=True)
        return

    t0 = time.time()
    solver = build_solver(args.off_file)
    print("[shard%d] solver 就绪 %.1fs" % (args.shard, time.time() - t0), flush=True)

    hdr = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
    tried_new = not os.path.exists(tried_path)
    written = 0
    crashes = 0
    with open(args.out_csv, "a", encoding="utf-8-sig", newline="", buffering=1) as f,             open(tried_path, "a", encoding="utf-8-sig", newline="", buffering=1) as ft:
        w = csv.writer(f)
        wt = csv.writer(ft)
        if hdr:
            w.writerow(["s", "t", "true_distance", "graph_distance"])
        if tried_new:
            wt.writerow(["s", "t"])
        for s, t, g in todo:
            # 先登记再计算：崩溃后该点对按毒点对跳过，不会死循环
            wt.writerow([int(s), int(t)])
            ft.flush()
            if args.no_fork:
                try:
                    path = solver.find_geodesic_path(int(s), int(t))
                    d = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                except Exception:
                    d = float("inf")
            else:
                d, crashed = compute_isolated(solver, s, t)
                if crashed:
                    # 子进程崩溃：记为毒点对（写 inf，交由热方法回填），继续算下一个
                    w.writerow([int(s), int(t), "inf", "" if g is None else "%.6f" % g])
                    f.flush()
                    crashes += 1
                    continue
            w.writerow([int(s), int(t), "%.6f" % d, "" if g is None else "%.6f" % g])
            f.flush()
            written += 1
            if written % 200 == 0:
                el = time.time() - t0
                print("[shard%d] %d/%d 用时%.0fs 速率%.2f对/s" %
                      (args.shard, written, len(todo), el, written / max(el, 1e-9)), flush=True)
    print("[shard%d] 本轮完成 %d 对（隔离崩溃 %d 次）" % (args.shard, written, crashes), flush=True)


if __name__ == "__main__":
    main()
