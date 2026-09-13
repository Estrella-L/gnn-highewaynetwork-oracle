# -*- coding: utf-8 -*-
"""消融实验矩阵驱动器（解剖式 ablation study）。

设计目标：把「总体方法」拆开，逐项拿掉技术组件，用**完全相同的采样、划分与测试集**
测量每项组件的贡献。所有配置都从同一个 main.py 走，保证除了被消融的组件之外，
损失函数、优化器、特征、评估口径完全一致。

消融矩阵（对应组会要求的 4 项解剖实验）：

  M0  full          完整方法（总体效果）      three_stage + euclidean_residual
  A1  no_euclid     去掉欧氏距离残差          three_stage + direct        <- 实验 1
  A2  no_part_hw    去掉地形分区 + highway    single_gnn  + euclidean_residual <- 实验 2
  A3  no_both       上述两者都去掉            single_gnn  + direct        <- 实验 3
  A4  euclidean     纯 3D/2D 欧氏距离（无学习）  baseline.py                 <- 实验 4

另外提供 *_rel 补充变体（direct 模式改用 relative 损失），用于证明
"直接预测绝对距离"的退化不是因为损失函数选择不当（避免"稻草人基线"质疑）。

用法示例：
    python3 scripts_local/run_ablation.py \
        --off_file /path/to/terrain.off --tag bhhmini \
        --seeds 42 43 44 --distance_samples 20000 --num_epoch 100

    # 只跑部分配置 / 断点续跑（已完成的 run 会自动跳过）
    python3 scripts_local/run_ablation.py ... --runs M0_full,A2_no_part_hw
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # gnn-euclidean-local
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)      # 科研小组

# ---------------------------------------------------------------------------
# 消融矩阵定义
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = [
    {
        "id": "M0_full",
        "label": "总体方法（完整）",
        "desc": "四叉树地形分区 + highway 网络 + Inner/Inter GNN + Fusion + 3D 欧氏距离残差",
        "args": {"architecture": "three_stage", "prediction_mode": "euclidean_residual", "loss_type": "huber"},
    },
    {
        "id": "A1_no_euclid",
        "label": "去掉欧氏距离残差",
        "desc": "实验1：保留地形分区 / highway / Inter-GNN / Inner-GNN，但直接回归 geodesic distance（无欧氏锚点）",
        "args": {"architecture": "three_stage", "prediction_mode": "direct", "loss_type": "huber"},
    },
    {
        "id": "A2_no_part_hw",
        "label": "去掉地形分区 + highway 网络",
        "desc": "实验2：只用一个全图 GNN 编码 s/t 直接预测，但保留 3D 欧氏距离残差参数化",
        "args": {"architecture": "single_gnn", "prediction_mode": "euclidean_residual", "loss_type": "huber"},
    },
    {
        "id": "A3_no_both",
        "label": "去掉两者（分区+highway 与欧氏残差）",
        "desc": "实验3：单 GNN 直接回归 geodesic distance，既无分区/highway，也无欧氏残差",
        "args": {"architecture": "single_gnn", "prediction_mode": "direct", "loss_type": "huber"},
    },
    {
        "id": "A1_hwfeat",
        "label": "[补充] 去掉欧氏残差，但开启 highway 分解距离特征",
        "desc": "防守型对照：直接回归的模型仍拿到 access+highway+access 三段分解距离特征，"
                "排除「A1 失败只是因为模型完全没有任何距离输入」这一解释",
        "args": {"architecture": "three_stage", "prediction_mode": "direct", "loss_type": "huber",
                 "keep_highway_distance_feature": True},
    },
    {
        "id": "A2_deep",
        "label": "[补充] 去掉分区+highway，加深单 GNN（6 层）",
        "desc": "防守型对照：把单 GNN 加深到 6 层（感受野 6-hop），排除「消融分支容量/深度不够」的解释",
        "args": {"architecture": "single_gnn", "prediction_mode": "euclidean_residual",
                 "loss_type": "huber", "single_gnn_layers": 6},
    },
    {
        "id": "A3_deep",
        "label": "[补充] 去掉两者，加深单 GNN（6 层）",
        "desc": "防守型对照：单 GNN 6 层 + direct 预测",
        "args": {"architecture": "single_gnn", "prediction_mode": "direct",
                 "loss_type": "huber", "single_gnn_layers": 6},
    },
    {
        "id": "A1_rel",
        "label": "[补充] 去掉欧氏残差 + relative 损失",
        "desc": "公平性对照：A1 改用跨尺度归一化的 relative 损失，排除「损失选得不好」的质疑",
        "args": {"architecture": "three_stage", "prediction_mode": "direct", "loss_type": "relative"},
    },
    {
        "id": "A3_rel",
        "label": "[补充] 去掉两者 + relative 损失",
        "desc": "公平性对照：A3 改用 relative 损失",
        "args": {"architecture": "single_gnn", "prediction_mode": "direct", "loss_type": "relative"},
    },
]

# ---------------------------------------------------------------------------
# 三篇 baseline 论文的方法（严格按其官方仓库实现），跑在同一套标签/划分/测试集上
# ---------------------------------------------------------------------------
def _baseline_cfg(cid, arch, label, mode, desc, loss=None):
    # native 模式用各论文自己的损失函数，避免"用我们的损失去跑别人的方法"造成不公平：
    #   GeGnn  : relative MAE   (源码 GnnDist.py: (|pred-gt|/(gt+1e-3)).mean())
    #   LiteGE : MAPE           (源码 trainUDFPointCloud.py: MAPELoss)
    #   NeuroGF: 统一取 relative MAE
    # 我们的 relative 损失定义就是 mean(|pred-true|/(true+eps))，三者等价。
    if loss is None:
        loss = "relative" if mode == "native" else "huber"
    return {
        "id": cid, "label": label, "desc": desc,
        # 不硬编码 batch/step：统一由命令行 --single_gnn_head_batch / --single_gnn_step_samples 决定，
        # 这样 baseline 与我方单 GNN 分支（A2/A3）拿到**完全相同的优化步数**，
        # 也方便按图规模调整（EP_low 上用小 step 会慢到不可行）。
        "args": {"architecture": arch, "prediction_mode": mode, "loss_type": loss},
    }


BASELINE_CONFIGS = [
    _baseline_cfg("B_gegnn", "gegnn", "baseline: GeGnn（原样输出）", "native",
                  "GeGnn: 整图 GeoConv(max 聚合 + 相对位置/边长) -> 256 维顶点嵌入 -> MLP 解码 (e_i-e_j)^2"),
    _baseline_cfg("B_neurogf", "neurogf", "baseline: NeuroGF（原样输出）", "native",
                  "NeuroGF: 逐点 lifting FC(3->64->128->256) -> |e_s - e_t| -> MLP 回归测地距离"),
    _baseline_cfg("B_litege", "litege", "baseline: LiteGE（原样输出）", "native",
                  "LiteGE: CoordMLP + UDF-PCA 形状描述子融合 -> (e_s - e_t) -> MLP -> abs()"),
    _baseline_cfg("B_gegnn_euc", "gegnn", "baseline: GeGnn + 欧氏残差输出", "euclidean_residual",
                  "消融对照：GeGnn 的网络 + 我们的 d_3D*(1+0.5tanh(f)) 输出参数化"),
    _baseline_cfg("B_neurogf_euc", "neurogf", "baseline: NeuroGF + 欧氏残差输出", "euclidean_residual",
                  "消融对照：NeuroGF 的网络 + 我们的输出参数化"),
    _baseline_cfg("B_litege_euc", "litege", "baseline: LiteGE + 欧氏残差输出", "euclidean_residual",
                  "消融对照：LiteGE 的网络 + 我们的输出参数化"),
]

ALL_CONFIGS = ABLATION_CONFIGS + BASELINE_CONFIGS
CONFIG_BY_ID = {c["id"]: c for c in ALL_CONFIGS}
DEFAULT_RUNS = ["M0_full", "A1_no_euclid", "A2_no_part_hw", "A3_no_both"]
BASELINE_RUNS = [c["id"] for c in BASELINE_CONFIGS]


def build_parser():
    p = argparse.ArgumentParser(description="消融矩阵驱动器")
    p.add_argument("--off_file", required=True, help="输入 .off 地形（绝对路径或相对 data/ 的路径）")
    p.add_argument("--tag", required=True, help="本次消融实验标签，用于命名日志与结果文件")
    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="随机种子列表")
    p.add_argument("--runs", type=str, default=",".join(DEFAULT_RUNS),
                   help="要跑的配置 id，逗号分隔；all 表示全部（含补充变体）")
    p.add_argument("--out_root", type=str, default=os.path.join(WORKSPACE_ROOT, "消融实验"),
                   help="日志与结果输出根目录")
    p.add_argument("--python", type=str, default=sys.executable, help="python 解释器")

    # 训练超参（所有配置共用，保证单变量对照）
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--capacity", type=int, default=512)
    p.add_argument("--distance_samples", type=int, default=20000)
    p.add_argument("--num_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--out_dim", type=int, default=32)
    p.add_argument("--dropout_ratio", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--lr_scheduler", type=str, default="plateau")
    p.add_argument("--early_stop_patience", type=int, default=15)
    p.add_argument("--selection_metric", type=str, default="relative_error")
    p.add_argument("--highway_k", type=int, default=3)
    p.add_argument("--transit_k", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--sample_strategy", type=str, default="random")
    p.add_argument("--single_gnn_layers", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=0.0, help="梯度裁剪阈值；0=关闭")
    p.add_argument("--torch_threads", type=int, default=0,
                   help="PyTorch intra-op 线程数；多任务并行时设小值避免线程超订")
    p.add_argument("--single_gnn_step_samples", type=int, default=0,
                   help="0 = 自动等于 --batch_size（与三段式优化步数一致，消融公平性）")
    p.add_argument("--single_gnn_head_batch", type=int, default=8192)
    p.add_argument("--keep_highway_distance_feature", action="store_true",
                   help="不关闭 highway 分解距离特征（默认关闭，与 EP_low 主结果一致）")
    p.add_argument("--labels_file", type=str, default="",
                   help="外部标签 CSV（精确测地真值）；给出后所有配置都用它当监督标签")
    p.add_argument("--labels_col", type=str, default="exact_geo")
    p.add_argument("--dry_run", action="store_true", help="只打印将要执行的命令")
    p.add_argument("--force", action="store_true", help="重跑已有记录的配置（默认跳过已完成的）")
    return p


def run_name(base_name, tag, cfg_id, seed):
    return f"{base_name}_abl_{tag}_{cfg_id}_s{seed}"


def result_txt_path(project_root, base_name, tag, cfg_id, seed):
    return os.path.join(project_root, "outputs", "results", run_name(base_name, tag, cfg_id, seed) + ".txt")


def test_pairs_path(project_root, base_name, tag, cfg_id, seed):
    return os.path.join(project_root, "outputs", "results",
                        run_name(base_name, tag, cfg_id, seed) + "_test_pairs.csv")


def parse_result_txt(path):
    """解析 main.py 写出的 <run>.txt -> dict（含 test_* 与 group_* 指标）。"""
    out = {}
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            key, val = parts[0], parts[1]
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def parse_baseline_txt(path):
    """解析 baseline.py 的 <out>.txt -> {baseline_name: {mae, rmse, relative_error, mean_bias}}"""
    res = {}
    if not os.path.exists(path):
        return res
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)  # 表头
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                res[parts[0]] = {
                    "mae": float(parts[1]), "rmse": float(parts[2]),
                    "relative_error": float(parts[3]), "mean_bias": float(parts[4]),
                }
    return res


def main():
    args = build_parser().parse_args()
    os.chdir(PROJECT_ROOT)

    if os.path.isabs(args.off_file):
        off_path = args.off_file
    else:
        _direct = os.path.join(PROJECT_ROOT, args.off_file)
        off_path = _direct if os.path.exists(_direct) else os.path.join(PROJECT_ROOT, "data", args.off_file)
    if not os.path.exists(off_path):
        print(f"[ablation] 找不到地形文件：{off_path}")
        return 1
    base_name = os.path.splitext(os.path.basename(off_path))[0]

    if args.runs.strip() == "all":
        run_ids = [c["id"] for c in ALL_CONFIGS]
    elif args.runs.strip() == "baselines":
        run_ids = list(BASELINE_RUNS)
    else:
        run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    for rid in run_ids:
        if rid not in CONFIG_BY_ID:
            print(f"[ablation] 未知配置 id：{rid}；可选：{list(CONFIG_BY_ID)}")
            return 1

    log_dir = os.path.join(args.out_root, "logs", args.tag)
    result_dir = os.path.join(args.out_root, "results")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    jsonl_path = os.path.join(result_dir, f"{args.tag}_results.jsonl")

    # ---- 单实例锁：同一 tag 只允许一个驱动器在跑（避免两个进程写同一批日志/结果）----
    lock_path = os.path.join(result_dir, f"{args.tag}.lock")
    if os.path.exists(lock_path) and not args.force:
        try:
            old_pid = int(open(lock_path, encoding="utf-8").read().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid:
            try:
                os.kill(old_pid, 0)          # 进程还活着
                print(f"[ablation] 已有驱动器在跑（pid={old_pid}，锁文件 {lock_path}）。"
                      f"确认它已退出后可加 --force 覆盖，或手动删除锁文件。")
                return 1
            except OSError:
                pass  # 陈旧锁，继续
    with open(lock_path, "w", encoding="utf-8") as lf:
        lf.write(str(os.getpid()))
    import atexit
    atexit.register(lambda: os.path.exists(lock_path) and os.remove(lock_path))

    records = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[(rec["run_id"], rec["seed"])] = rec

    common = [
        "--off_file", off_path,
        "--max_depth", str(args.max_depth),
        "--capacity", str(args.capacity),
        "--distance_samples", str(args.distance_samples),
        "--num_epoch", str(args.num_epoch),
        "--batch_size", str(args.batch_size),
        "--hidden_dim", str(args.hidden_dim),
        "--out_dim", str(args.out_dim),
        "--dropout_ratio", str(args.dropout_ratio),
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", args.lr_scheduler,
        "--early_stop_patience", str(args.early_stop_patience),
        "--selection_metric", args.selection_metric,
        "--highway_k", str(args.highway_k),
        "--transit_k", str(args.transit_k),
        "--sample_strategy", args.sample_strategy,
        "--single_gnn_layers", str(args.single_gnn_layers),
        "--single_gnn_step_samples", str(args.single_gnn_step_samples),
        "--single_gnn_head_batch", str(args.single_gnn_head_batch),
        "--device", args.device,
        "--torch_threads", str(args.torch_threads),
    ]
    if not args.keep_highway_distance_feature:
        common.append("--disable_highway_distance_feature")
    if args.labels_file:
        common.extend(["--labels_file", args.labels_file, "--labels_col", args.labels_col])

    t_all = time.time()
    def common_for(cfg):
        """把配置专属的覆盖项（single_gnn_layers / highway 距离特征开关）替换进公共参数表。"""
        out = []
        skip_next = False
        for tok in common:
            if skip_next:
                skip_next = False
                continue
            if tok == "--single_gnn_layers":
                out.extend([tok, str(cfg["args"].get("single_gnn_layers", args.single_gnn_layers))])
                skip_next = True
            elif tok == "--single_gnn_head_batch":
                out.extend([tok, str(cfg["args"].get("single_gnn_head_batch", args.single_gnn_head_batch))])
                skip_next = True
            elif tok == "--single_gnn_step_samples":
                out.extend([tok, str(cfg["args"].get("single_gnn_step_samples", args.single_gnn_step_samples))])
                skip_next = True
            elif tok == "--disable_highway_distance_feature" and cfg["args"].get("keep_highway_distance_feature"):
                continue  # 该配置保留 highway 分解距离特征
            else:
                out.append(tok)
        return out

    for seed in args.seeds:
        for rid in run_ids:
            cfg = CONFIG_BY_ID[rid]
            key = (rid, seed)
            if (not args.force) and key in records and records[key].get("status") == "ok":
                print(f"[ablation] 已完成，跳过 {rid} seed={seed}")
                continue
            rname = run_name(base_name, args.tag, rid, seed)
            cmd = [args.python, "-u", "main.py"] + common_for(cfg) + [
                "--seed", str(seed),
                "--architecture", cfg["args"]["architecture"],
                "--prediction_mode", cfg["args"]["prediction_mode"],
                "--loss_type", cfg["args"]["loss_type"],
                "--run_tag", f"abl_{args.tag}_{rid}_s{seed}",
            ]
            log_path = os.path.join(log_dir, f"{rid}_seed{seed}.log")
            print(f"\n[ablation] === {rid} seed={seed} -> {log_path}")
            print("[ablation] " + " ".join(cmd))
            if args.dry_run:
                continue
            t0 = time.time()
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                      env={**os.environ, "PYTHONUNBUFFERED": "1"})
            elapsed = time.time() - t0
            parsed = parse_result_txt(result_txt_path(PROJECT_ROOT, base_name, args.tag, rid, seed))
            rec = {
                "run_id": rid, "label": cfg["label"], "seed": int(seed),
                "architecture": cfg["args"]["architecture"],
                "prediction_mode": cfg["args"]["prediction_mode"],
                "loss_type": cfg["args"]["loss_type"],
                "single_gnn_layers": cfg["args"].get("single_gnn_layers", args.single_gnn_layers),
                "returncode": proc.returncode,
                "wall_seconds": round(elapsed, 1),
                "log": os.path.relpath(log_path, WORKSPACE_ROOT),
                "status": "ok" if (proc.returncode == 0 and parsed) else "failed",
            }
            if parsed:
                rec.update({k: v for k, v in parsed.items() if isinstance(v, (int, float))})
            records[key] = rec
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in records.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if rec["status"] != "ok":
                print(f"[ablation] !! {rid} seed={seed} 失败，返回码 {proc.returncode}，见 {log_path}")

    if args.dry_run:
        return 0

    # ---- 一致性校验：同一 seed 下所有配置必须落在同一批 test 点对上 -------------
    import hashlib

    def _md5(path):
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    print("\n[ablation] === test 集一致性校验（同一 seed 下各配置必须完全一致）===")
    consistency_ok = True
    for seed in args.seeds:
        digests = {}
        for rid in run_ids:
            p = test_pairs_path(PROJECT_ROOT, base_name, args.tag, rid, seed)
            if os.path.exists(p):
                digests[rid] = _md5(p)
        if len(set(digests.values())) > 1:
            consistency_ok = False
            print(f"[ablation] !! seed={seed} test 集不一致：{digests}")
        elif digests:
            print(f"[ablation] seed={seed} OK: {len(digests)} 个配置共享同一 test 集 "
                  f"(md5={list(digests.values())[0][:12]}, {len(digests)} configs)")
    with open(os.path.join(result_dir, f"{args.tag}_consistency.txt"), "w", encoding="utf-8") as fh:
        fh.write("all_configs_share_same_test_set " + str(consistency_ok) + "\n")

    # ---- 实验 4：纯欧氏距离（无学习）基线，在同一批 test 点对上评估 -------------
    baseline_records = {}
    for seed in args.seeds:
        ref = records.get(("M0_full", seed))
        pairs_file = test_pairs_path(PROJECT_ROOT, base_name, args.tag, "M0_full", seed)
        if not os.path.exists(pairs_file):
            for rid in run_ids:
                alt = test_pairs_path(PROJECT_ROOT, base_name, args.tag, rid, seed)
                if os.path.exists(alt):
                    pairs_file = alt
                    break
        if not os.path.exists(pairs_file):
            print(f"[ablation] seed={seed} 找不到 test 点对文件，跳过欧氏基线")
            continue
        out_txt = os.path.join(result_dir, f"{args.tag}_baseline_seed{seed}.txt")
        cmd = [
            args.python, "-u", "baseline.py", "--off_file", off_path,
            "--test_pairs_file", pairs_file, "--out_file", out_txt,
            "--max_depth", str(args.max_depth), "--capacity", str(args.capacity),
            "--highway_k", str(args.highway_k), "--transit_k", str(args.transit_k),
        ]
        log_path = os.path.join(log_dir, f"A4_euclidean_seed{seed}.log")
        print(f"\n[ablation] === A4 无学习基线 seed={seed} -> {log_path}")
        with open(log_path, "w", encoding="utf-8") as logf:
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        bl = parse_baseline_txt(out_txt)
        for name, m in bl.items():
            rid = f"A4_{name}"
            rec = {
                "run_id": rid, "label": f"无学习基线 {name}", "seed": int(seed),
                "architecture": "none", "prediction_mode": "none", "loss_type": "none",
                "test_mae": m["mae"], "test_rmse": m["rmse"], "test_relative_error": m["relative_error"],
                "mean_bias": m["mean_bias"], "status": "ok",
                "log": os.path.relpath(log_path, WORKSPACE_ROOT),
            }
            records[(rid, seed)] = rec
            baseline_records[(rid, seed)] = rec
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 汇总 ---------------------------------------------------------------
    order = run_ids + sorted({rid for (rid, _) in records if rid.startswith("A4_")})
    csv_path = os.path.join(result_dir, f"{args.tag}_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "label", "seed", "test_relative_error", "test_mae", "test_rmse",
                    "group_same_leaf_relative_error", "group_cross_leaf_relative_error",
                    "wall_seconds"])
        for rid in order:
            for seed in args.seeds:
                rec = records.get((rid, seed))
                if not rec:
                    continue
                w.writerow([rid, rec.get("label", ""), seed,
                            rec.get("test_relative_error", ""), rec.get("test_mae", ""),
                            rec.get("test_rmse", ""),
                            rec.get("group_same_leaf_relative_error", ""),
                            rec.get("group_cross_leaf_relative_error", ""),
                            rec.get("wall_seconds", "")])
    print(f"\n[ablation] 汇总 CSV -> {csv_path}")
    print(f"[ablation] 原始 JSONL -> {jsonl_path}")
    print(build_markdown_table(order, args.seeds, records, args, base_name))
    print(f"[ablation] 总耗时 {time.time() - t_all:.1f}s")
    return 0


def _mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0, 1
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var), len(vals)


def build_markdown_table(order, seeds, records, args, base_name):
    lines = []
    lines.append("")
    lines.append(f"## 消融实验结果（{base_name}, seeds={seeds}, pairs={args.distance_samples}）")
    lines.append("")
    lines.append("| 配置 | 说明 | test rel.err（越低越好） | test MAE | test RMSE | 同分区 rel | 跨分区 rel |")
    lines.append("|---|---|---|---|---|---|---|")
    base_rel = None
    for rid in order:
        recs = [records.get((rid, s)) for s in seeds]
        recs = [r for r in recs if r]
        if not recs:
            continue
        rel = [r.get("test_relative_error") for r in recs]
        mae = [r.get("test_mae") for r in recs]
        rmse = [r.get("test_rmse") for r in recs]
        sl = [r.get("group_same_leaf_relative_error") for r in recs]
        cl = [r.get("group_cross_leaf_relative_error") for r in recs]
        m_rel, s_rel, _ = _mean_std(rel)
        m_mae, _, _ = _mean_std(mae)
        m_rmse, _, _ = _mean_std(rmse)
        m_sl, _, _ = _mean_std(sl)
        m_cl, _, _ = _mean_std(cl)
        if rid == "M0_full":
            base_rel = m_rel

        def fmt(m, s=None):
            if m is None:
                return "-"
            return f"{m:.6f}" if s is None else f"{m:.6f} ± {s:.6f}"

        lines.append("| {id} | {lab} | {rel} | {mae} | {rmse} | {sl} | {cl} |".format(
            id=rid, lab=recs[0].get("label", ""),
            rel=fmt(m_rel, s_rel),
            mae=fmt(m_mae) if m_mae is None else f"{m_mae:.2f}",
            rmse=fmt(m_rmse) if m_rmse is None else f"{m_rmse:.2f}",
            sl=fmt(m_sl), cl=fmt(m_cl),
        ))
    if base_rel:
        lines.append("")
        lines.append("（退化幅度以 M0 为基准，由 report 脚本计算）")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
