# AIoTC: Kernel-Aware Inference Cost Prediction for Edge GPU PDR

> **论文**: 面向边缘 GPU 学习式 PDR 的内核感知推理代价预测与预算约束部署  
> **Paper**: Kernel-Aware Inference Cost Prediction and Budget-Constrained Deployment for Edge-GPU Learning-Based PDR

本仓库包含论文完整实验代码、各设备测量数据、以及审稿人修订版补充实验结果。

---

## 仓库结构

```
aiotc/
├── scripts/                          # 所有实验脚本
│   ├── profile_device.py             # 延迟/吞吐/能耗测量（在目标设备上运行）
│   ├── cost_model.py                 # 代价模型主脚本
│   ├── validate_cost_model.py        # LOMO/LOAO/LOPO/LODO 协议验证
│   ├── kernel_profile.py             # CUDA 内核计数 profiling
│   ├── diagnose_bf16_lstm.py         # BF16-LSTM 回退现象诊断
│   ├── run_cost_model_ablation.py    # ★ 实验1: 特征消融（7种 × 4协议）
│   ├── run_fallback_ablation.py      # ★ 实验2: I_fallback 专项消融
│   ├── run_selector_oracle_eval.py   # ★ 实验3: 选择器 oracle/regret 评估
│   └── run_few_shot_calibration.py   # ★ 实验4: 少样本架构族校准
├── results/                          # 实验结果（主机侧，含 Blackwell 基准数据）
│   ├── profile_blackwell.csv         # Blackwell RTX PRO 6000 实测延迟/能耗
│   ├── profile_agx_orin.csv          # AGX Orin 实测数据
│   ├── profile_orin_nano.csv         # Orin Nano 实测数据
│   ├── kernel_profile.json           # CUDA 内核计数（BF16-LSTM: 10→1241, 124×）
│   ├── model_flops.json              # 各模型实测 FLOPs（FlopCounterMode）
│   └── revision_experiments/         # ★ 审稿人修订版补充实验
│       ├── cost_model_ablation.*     # 实验1 (CSV/MD/TEX)
│       ├── fallback_ablation.*       # 实验2
│       ├── selector_oracle_eval.*    # 实验3
│       ├── few_shot_arch_calibration.* # 实验4
│       └── REVISION_REPORT.md        # 综合报告（含结论和数据一致性检查）
├── agx_orin/                         # Jetson AGX Orin 设备侧原始数据
└── orin_nano/                        # Jetson Orin Nano 设备侧原始数据
```

---

## 已验证硬件环境

| 设备 | JetPack / CUDA | PyTorch | TensorRT | GPU Clock | 数据位置 |
|------|----------------|---------|----------|-----------|----------|
| RTX PRO 6000 Blackwell | CUDA 13.0 | 2.13.0 | — | — | `results/` |
| Jetson AGX Orin | debug L4T / CUDA 13.3 | 2.13.0 | 11.1 | 1300.5 MHz | `agx_orin/` |
| Jetson Orin Nano (4GB) | L4T R36.5 / CUDA 12.6 | 2.11.0 | 10.3 | 624.75 MHz | `orin_nano/` |

---

## 快速开始：复现补充实验（仅需 CPU + Python）

```bash
git clone https://github.com/nvSiruiW/aiotc.git
cd aiotc

pip install scipy numpy

# 实验1: 特征消融（7种特征集 × 4种 leave-out 协议）
python3 scripts/run_cost_model_ablation.py \
    --output results/revision_experiments/cost_model_ablation

# 实验2: I_fallback 专项消融
python3 scripts/run_fallback_ablation.py \
    --output results/revision_experiments/fallback_ablation

# 实验3: 部署选择器 oracle/regret 评估
python3 scripts/run_selector_oracle_eval.py \
    --output results/revision_experiments/selector_oracle_eval

# 实验4: 少样本架构族校准
python3 scripts/run_few_shot_calibration.py \
    --output results/revision_experiments/few_shot_arch_calibration
```

### 预期核心结果

| 实验 | 指标 | 预期值 |
|------|------|--------|
| 特征消融 | Params LOMO MAPE | 69.8% |
| 特征消融 | N_exec LOMO MAPE | **35.2%**（内核感知特征比 params 好 1× 以上）|
| Fallback 消融 | Orin Nano BF16/FP16 延迟比 | **29.6×** |
| Fallback 消融 | Orin Nano I_fallback 预测改善 | 97.5% → **7.4%** APE |
| Fallback 消融 | BF16/FP16 内核数比 | **124×**（10 → 1241 launches）|
| 选择器 Oracle | Feasible rate | **100%**（44/44 场景）|
| 选择器 Oracle | ATE regret | **0.0000 m** |
| 少样本校准 | CNN: 0-shot → 1-shot | 27.4% → **15.1%** MAPE |

---

## 在新 Jetson 设备上复现延迟测量

### 前提条件

- Jetson 已安装 JetPack（含 PyTorch、CUDA、tegrastats）
- 本仓库已 `git clone` 到 Jetson 上
- 模型源码（ronin/source、IMUNet/RONIN_torch 等）已放置在对应路径

### 运行 profiling

```bash
# 在 Jetson 上运行（替换 <DEVICE> 为设备标识）
cd aiotc

python3 scripts/profile_device.py \
    --device <DEVICE> \
    --models ronin_resnet18,ronin_tcn,ronin_lstm,imunet,mobilenetv2,mnasnet,\
efficientnet_b0,tinyodom,tlio_resnet,eqnio \
    --precisions fp32,fp16,bf16 \
    --out results/profile_<DEVICE>.csv \
    --iters 500 --dur 10

# 动态能耗测量（需要 tegrastats）
python3 scripts/measure_power.py \
    --device <DEVICE> \
    --out results/power_dynamic_<DEVICE>.csv
```

> **BF16 注意事项**：Orin Nano 上 BF16-LSTM 会触发 fallback（约 95ms vs FP16 3.2ms）。
> 如果测试挂起，只测 `fp32,fp16`，fallback 现象已有现成数据。

### 将新设备数据纳入实验

将新 CSV 传回主机后，在 `scripts/run_cost_model_ablation.py` 开头的 `PROF` 字典中添加：

```python
PROF = {
    "blackwell": load_csv("profile_blackwell.csv"),
    "agx_orin":  load_csv("profile_agx_orin.csv"),
    "orin_nano": load_csv("profile_orin_nano.csv"),
    "<DEVICE>":  load_csv("profile_<DEVICE>.csv"),   # 新增
}
```

`run_fallback_ablation.py` 和 `run_selector_oracle_eval.py` 中同样添加。然后重新运行实验脚本。

---

## BF16-LSTM 回退现象诊断

cuDNN 的 FP16 持久化 LSTM 内核（`RNN_blockPersist_fp_LSTM`）不支持 BF16，导致 BF16 回退到逐时间步展开路径，内核数从 10 增加到 1241（124×）。在启动开销较高的 Jetson 上，这导致延迟增加 26-30×。

```bash
# 在有 CUDA 的设备上运行诊断
python3 scripts/diagnose_bf16_lstm.py all 2>&1 | tee diagnose_bf16_<DEVICE>.log
```

预期输出见 `results/revision_experiments/fallback_ablation.md`。

---

## FLOPs 提取（可选，模型构建器需要 GPU）

```bash
python3 -c "
import sys, json
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'ronin/source')
sys.path.insert(0, 'IMUNet/RONIN_torch')
import torch
from profile_device import MODEL_REGISTRY
from torch.utils.flop_counter import FlopCounterMode
MODELS = ['ronin_resnet18','ronin_tcn','ronin_lstm','imunet','mobilenetv2',
          'mnasnet','efficientnet_b0','tinyodom','tlio_resnet','eqnio']
out = {}
for m in MODELS:
    try:
        net, shp = MODEL_REGISTRY[m](); net = net.float().eval()
        params = sum(p.numel() for p in net.parameters())
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc: net(torch.randn(*shp))
        out[m] = {'params_M': round(params/1e6,3), 'flops_M': round(fc.get_total_flops()/1e6,2)}
        print(f'{m}: {params/1e6:.3f}M params, {fc.get_total_flops()/1e6:.1f}M FLOPs')
    except Exception as e: print(f'[SKIP {m}]: {e}'); out[m] = None
json.dump(out, open('results/model_flops.json','w'), indent=2)
"
```

已预计算结果存于 `results/model_flops.json`（Blackwell 上验证，PyTorch 2.13.0）。

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| BF16 在 Orin Nano 上极慢/挂起 | 正常现象（fallback），只测 fp32,fp16 即可 |
| `from model_temporal import ...` 报错 | 从 `aiotc/` 根目录运行；`ronin/source` 需在路径中 |
| `FlopCounterMode` 不可用 | 需要 PyTorch ≥ 2.0；已预存 `model_flops.json` |
| `tegrastats` 找不到 | `sudo apt install nvidia-jetpack` 或完整 JetPack 安装 |
| EqNIO 在旧 PyTorch 上报错 | 需要 `torch.linalg.cross`（PyTorch ≥ 1.10）|

---

## 数据一致性注意事项

- `results/power_dynamic_*.csv` 中 `dynamic_energy_mJ` = (活跃 - 空闲) × 延迟（增量能耗）
- `results/profile_*.csv` 中 `energy_mJ_per_inf` = 总 GPU 功率 × 延迟（含空闲底功率）
- 两者相差约 **4-5×**，定义不同，请勿混用。论文中统一使用 `dynamic_energy_mJ`。
