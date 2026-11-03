# 五个地形对照组：A1/A2/A3 批量训练（Windows PowerShell，在仓库根目录执行）
#
# 用法：
#   .\scripts_local\run_A123_five_terrain.ps1 -terrains BH_high     # 只跑 BH_high（建议先跑）
#   .\scripts_local\run_A123_five_terrain.ps1                       # 四张图全跑（12 个配置）
#
# 协议与归档 M0 对齐：seed=42、100 轮、huber、hidden64/out32、dropout=0.1、
# highway_k=3、transit_k=8、batch=16、plateau(5,0.5)、best 按 val relative_error、
# EUCLID_RESIDUAL_SCALE=1.0、归档固定划分（train 40000 / val 5000 / test 5000）。
#
# 启动后每个 run 必须打印这行才算固定划分生效：
#   [distance] 固定划分（跳过随机切分）：train=40000, val=5000, test=5000
# single_gnn 若 CUDA OOM：在该配置命令后加  --single_gnn_head_batch 2048  重试。

param(
    [string[]]$terrains = @("BH_high", "SF_high", "USGS_CO", "USGS_WA"),
    [string[]]$only = @()   # 例：-only A2,A3 只跑这两类配置（按 tag 前缀匹配）；默认三个配置全跑
)

$ErrorActionPreference = "Continue"
$env:EUCLID_RESIDUAL_SCALE = "1.0"
# 中文输出走 UTF-8，避免日志/控制台乱码
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
New-Item -ItemType Directory -Force logs | Out-Null

# A1：three_stage + direct（去欧氏残差）
# A2：single_gnn + euclidean_residual（去分区+highway）
# A3：single_gnn + direct（两项都去）
# 顺序：先跑快的 A2/A3（single_gnn 无高速预处理，每轮几秒），A1 放最后
# （three_stage 首次需数小时的高速 Dijkstra 预处理，之后走缓存）。
$configs = @(
    @{ tag = "A2_single_res_s42_100ep";    arch = "single_gnn";  mode = "euclidean_residual" },
    @{ tag = "A3_single_direct_s42_100ep"; arch = "single_gnn";  mode = "direct" },
    @{ tag = "A1_no_res_s42_100ep";        arch = "three_stage"; mode = "direct" }
)

foreach ($ds in $terrains) {
    $D = "五个地形实验_标签数据与结果_20261024\$ds"
    if (-not (Test-Path "$D\原始地形\$ds.off")) {
        Write-Host "[跳过] 找不到 $D\原始地形\$ds.off（数据未解压？）"
        continue
    }
    foreach ($c in $configs) {
        $tag = $c.tag
        if ($only.Count -gt 0) {
            $hit = $false
            foreach ($o in $only) { if ($tag.StartsWith($o)) { $hit = $true; break } }
            if (-not $hit) { continue }
        }
        $done = "outputs\results\${ds}_${tag}.txt"
        if (Test-Path $done) {
            Write-Host "[已完成，跳过] $ds $tag"
            continue
        }
        Write-Host "===== $ds $tag 开始 $(Get-Date -Format 'HH:mm:ss') ====="
        python -u main.py `
            --off_file "$D\原始地形\$ds.off" --file_folder . `
            --labels_file "$D\标签与划分\labels.csv" --labels_col true_distance `
            --train_pairs_file "$D\标签与划分\train_pairs.csv" `
            --val_pairs_file   "$D\标签与划分\val_pairs.csv" `
            --test_pairs_file  "$D\标签与划分\test_pairs.csv" `
            --architecture $($c.arch) --prediction_mode $($c.mode) `
            --loss_type huber --hidden_dim 64 --out_dim 32 --dropout_ratio 0.1 `
            --highway_k 3 --transit_k 8 --batch_size 16 `
            --num_epoch 100 --early_stop_patience 100 `
            --lr_scheduler plateau --lr_patience 5 --lr_factor 0.5 `
            --selection_metric relative_error `
            --seed 42 --device cuda `
            --run_tag $tag `
            2>&1 | Tee-Object "logs\${ds}_${tag}.log"
        Write-Host "===== $ds $tag 结束 $(Get-Date -Format 'HH:mm:ss') ====="
    }
}
Write-Host "ALL DONE: $($terrains -join ', ')"
