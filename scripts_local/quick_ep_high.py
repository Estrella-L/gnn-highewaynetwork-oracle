# -*- coding: utf-8 -*-
"""EP_high 一键复现脚本（快速冒烟 / 完整实验两档）。
用法（需 cloud_tools 的 venv 运行）:
  python quick_ep_high.py --mode quick --port 44589 --pass PASSWORD
  python quick_ep_high.py --mode full  --port 44589 --pass PASSWORD
quick: 1 配置 x 3 epoch（约 8 分钟验证链路）
full : 读取 scripts_local/ep_high_grid.json 全部配置（含 100 轮终版）
"""
import argparse
import os
import sys
import time

import paramiko

LOCAL_PROJ = "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local"
EP_HIGH_OFF = "/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_high/EP_high.off"
REMOTE_PROJ = "/root/autodl-tmp/gnn-euclidean-local"


def run(host, port, pwd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname=host, port=port, username="root", password=pwd, timeout=30, allow_agent=False, look_for_keys=False)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--host", default="connect.westb.seetacloud.com")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--pass", dest="pwd", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--samples", type=int, default=50000)
    args = ap.parse_args()

    c = run(args.host, args.port, args.pwd)
    c.get_transport().set_keepalive(20)
    sf = c.open_sftp()

    def ensure_remote_dir(path):
        parts = path.strip("/").split("/")
        cur = ""
        for p in parts:
            cur += "/" + p
            try:
                sf.stat(cur)
            except FileNotFoundError:
                sf.mkdir(cur)

    # 1) 代码 + 网格配置 + 数据
    files_local = [
        (os.path.join(LOCAL_PROJ, "scripts_local", "cloud_ep_high_runner.py"), "scripts_local/cloud_ep_high_runner.py"),
        (os.path.join(LOCAL_PROJ, "scripts_local", "ep_high_grid.json"), "scripts_local/ep_high_grid.json"),
    ]
    ensure_remote_dir(REMOTE_PROJ + "/scripts_local")
    for local, rel in files_local:
        remote = REMOTE_PROJ + "/" + rel
        if os.path.exists(local):
            sf.put(local, remote)
            print("[up]", rel)
    # 项目核心代码（若云端缺）
    for name in ["main.py", "preprocess.py", "build_highway.py", "gnn.py", "model.py", "baseline.py"]:
        remote = REMOTE_PROJ + "/" + name
        try:
            sf.stat(remote)
        except FileNotFoundError:
            sf.put(os.path.join(LOCAL_PROJ, name), remote)
            print("[up]", name)
    # 数据集（若缺）
    remote_off = REMOTE_PROJ + "/dataset/EP_high/EP_high.off"
    try:
        if sf.stat(remote_off).st_size != os.path.getsize(EP_HIGH_OFF):
            raise FileNotFoundError
    except FileNotFoundError:
        ensure_remote_dir(REMOTE_PROJ + "/dataset/EP_high")
        print("[up] EP_high.off (99MB)")
        sf.put(EP_HIGH_OFF, remote_off)
    sf.close()

    # 2) 写云端一键启动脚本
    if args.mode == "quick":
        grid_json = [
            {"name": "quick_probe", "sample_strategy": "random", "loss_type": "huber",
             "highway_k": 3, "hidden_dim": 64, "out_dim": 32, "transit_k": 8,
             "num_epoch": 3, "early_stop_patience": 100, "batch_size": 16}
        ]
        import json as _j
        sf_put_grid = c.open_sftp()
        sf_put_grid.putfo(_io_StringIO(_j.dumps(grid_json)), None) if False else None
        # 直接写云端 grid 文件覆盖 quick 模式
        stdin, stdout, stderr = c.exec_command(
            "python3 -c \"import json; grid=[{'name':'quick_probe','sample_strategy':'random','loss_type':'huber',"
            "'highway_k':3,'hidden_dim':64,'out_dim':32,'transit_k':8,'num_epoch':3,'early_stop_patience':100,'batch_size':16}]; "
            "open('" + REMOTE_PROJ + "/scripts_local/ep_high_grid.json','w').write(json.dumps(grid))\"", timeout=30)
        print(stdout.read().decode("utf-8", "replace")[:200])

    env_py = "/root/miniconda3/envs/torch_pro6000/bin/python"
    launch = (
        "mkdir -p " + REMOTE_PROJ + "/logs && cd " + REMOTE_PROJ + " && "
        "setsid nohup env CLOUD_LOG=logs/pro6000_runner.log "
        + env_py + " -u scripts_local/cloud_ep_high_runner.py "
        "--off_file dataset/EP_high/EP_high.off --device cuda --workers " + str(args.workers) +
        " --samples " + str(args.samples) + " --chunk 32 --epochs_screen 12 --run_tag ep_high_d3c32 " +
        "> logs/pro6000_console.log 2>&1 & echo PID=$!"
    )
    stdin, stdout, stderr = c.exec_command(launch, timeout=60)
    print("launch:", stdout.read().decode("utf-8", "replace"))
    time.sleep(20)
    stdin, stdout, stderr = c.exec_command("ps aux | grep cloud_ep_high_runner | grep -v grep | wc -l", timeout=30)
    print("runner procs:", stdout.read().decode("utf-8", "replace").strip())
    print("日志: " + REMOTE_PROJ + "/logs/pro6000_console.log")
    c.close()


def _io_StringIO(x):
    import io
    return io.StringIO(x)


if __name__ == "__main__":
    main()
