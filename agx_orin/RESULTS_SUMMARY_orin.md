# Edge-Tier Reproduction — Jetson AGX Orin

Companion to the Orin Nano run at repo root; this is the **AGX Orin (label `orin`)** tier.

## Measurement conditions (part of the results)
- **Device:** NVIDIA Jetson AGX Orin Developer Kit (SoC tegra234, GPU sm_87)
- **Stack:** non-standard debug L4T (`/etc/nv_tegra_release` = "R00 (debug)"); CUDA userspace sbsa-linux 12.6/13.0/13.3 (default → 13.3)
- **PyTorch:** 2.13.0+cu130 (aarch64); runs on sm_87 via **PTX-JIT** (cosmetic "not built for CC 8.7" warning); `torch.cuda.is_available()=True`, device "Orin"
- **nvpmodel:** mode 0 = **MAXN**
- **jetson_clocks:** ON — GPU pinned **1300.5 MHz**
- **Power boundary:** whole-board via tegrastats **VIN_SYS_5V0** rail (source tier is card-level nvidia-smi → compare trends/energy, not absolute watts)

## Environment fixes applied (non-sudo unless noted) to make torch work
1. torch import failed: system lacked `libnvJitLink.so.12` (needed by torch's bundled `libcusparse.so.12`; no CUDA install on the box ships it). Fixed: `pip install nvidia-nvjitlink-cu12` + `LD_LIBRARY_PATH` to it.
2. numba needed numpy ≤2.4 and llvmlite: pinned `numpy==2.4.6`, installed `llvmlite==0.48`.
3. `scripts/profile_device.py PowerSampler`: the original `check_output(["tegrastats"], timeout=0.5)` never captured anything (tegrastats never exits → always TimeoutExpired → discarded) and looked for `VDD_IN`/`POM_*` rails absent here. Rewrote it to stream a persistent tegrastats and read the board rail `VIN_SYS_5V0`. See `scripts/profile_device_PATCHED.py`.

## Efficiency (NEW numbers) — FP32, AGX Orin (board, MAXN+locked) vs Blackwell (card)
| model | lat_bw ms | lat_orin ms | slowdown | E_bw mJ | E_orin mJ | E_ratio |
|---|---|---|---|---|---|---|
| ronin_resnet18 | 1.19 | 10.87 | 9.1x | 130.8 | 44.4 | 0.34x |
| ronin_tcn | 1.42 | 11.09 | 7.8x | 148.9 | 43.0 | 0.29x |
| ronin_lstm | 1.47 | 1.88 | 1.3x | 153.0 | 5.3 | 0.03x |
| imunet | 1.80 | 15.84 | 8.8x | 180.8 | 59.9 | 0.33x |
| mobilenetv2 | 2.18 | 19.26 | 8.8x | 216.8 | 73.0 | 0.34x |
| mnasnet | 2.31 | 19.47 | 8.4x | 219.7 | 74.5 | 0.34x |
| efficientnet_b0 | 2.28 | 21.35 | 9.4x | 232.6 | 81.1 | 0.35x |
| tlio_resnet | 1.33 | 11.96 | 9.0x | 142.2 | 53.8 | 0.38x |
| tinyodom | 0.94 | 8.44 | 9.0x | 94.7 | 32.7 | 0.35x |
| eqnio | 3.78 | 31.94 | 8.5x | 395.8 | 128.0 | 0.32x |

Story: AGX Orin is ~8–10× slower per inference but ~0.29–0.40× the energy per inference (board 3.8–4.5 W vs card ~105 W). Full CSV incl fp16/bf16: `profile_orin.csv`.

**Honesty caveat on power:** tegrastats telemetry here is live but weakly responsive; these tiny launch-bound models keep the GPU near-idle so board power sits at its ~4 W baseline for every config → `power_W`/`energy_mJ` are baseline-dominated (energy is an upper-ish bound). **Latency/throughput are the reliable efficiency metrics.** `profile_orin_30W_unlocked.csv` is a pre-STEP-3 run (30W, DVFS) kept for reference.

## Accuracy (should MATCH source tier — device-independence check)
All 24 configs (8 backbones × fp32/fp16/bf16) over 53 MagPIE sequences.
**Max |ATE Δ| vs Blackwell = 0.60%** (mnasnet bf16); FP32/FP16 ≤0.1%. **No model exceeds the 5% flag → device-independence CONFIRMED.** CSV: `accuracy_orin.csv`. e.g. RoNIN-ResNet FP32 ATE = 0.9018 m (ref 0.9018 m).

## INT8 / TensorRT (STEP 6) — DONE with a TRT 11.1 rewrite
Installed **TensorRT 11.1.0.106+cuda13.3** (apt, sbsa repo). The shipped `scripts/export_trt.py` does NOT run on TRT 11.1: `IInt8EntropyCalibrator2` **and** `BuilderFlag.FP16/INT8` were removed (TRT 11 is strongly-typed only; `trtexec --fp16/--int8/--calib` also gone). Rewrote the path (`scripts/export_trt_trt11.py`): precision now lives in the ONNX graph — **FP16** = float16 ONNX; **INT8** = ONNX Runtime static QDQ over the *real* IMU calibration windows, forced symmetric (TRT needs zero_point=0) with bias unquantized (TRT rejects Int32 bias QDQ). INT8 accuracy measured with `scripts/eval_trt_accuracy.py` (torch-tensor TRT runtime, reuses `eval_accuracy.recon_traj`+`compute_ate_rte`).

INT8 accuracy is **measured, never assumed equal to FP:**

| model | FP32 ATE | TRT-INT8 ATE | Δ | TRT-FP16 ATE | INT8 lat ms | INT8 qps |
|---|---|---|---|---|---|---|
| ronin_resnet18 | 0.902 | 1.487 | **+65%** | 0.903 | 0.361 | 2759 |
| imunet | 0.970 | 1.326 | +37% | 0.969 | 0.300 | 3324 |
| mobilenetv2 | 0.934 | 1.365 | +46% | 0.935 | 0.445 | 2242 |
| mnasnet | 0.904 | 1.502 | +66% | 0.904 | 0.492 | 2030 |
| efficientnet_b0 | 0.981 | 1.330 | +36% | 0.982 | 0.610 | 1637 |
| tinyodom | 1.207 | 2.192 | **+82%** | 1.206 | 0.443 | 2251 |

Findings: **FP16 within ±0.2% of FP32** (device-independent). **INT8 is materially worse on every model (+35%…+82%)** — PTQ (ORT MinMax symmetric) is not accuracy-free for this velocity-regression task; entropy/percentile calibration or QAT might recover some, not pursued. TRT engines are ~20–40× faster than the PyTorch-eager profile; INT8-vs-FP16 latency is **mixed** (Q/DQ overhead offsets gains on the smallest models). Engines: `engines/*.plan`; latency CSV: `trt_latency_orin.csv`; INT8/FP16 accuracy: `accuracy_int8_orin.csv`.

Scope note: RoNIN-LSTM (recurrent) + TLIO/EqNIO excluded from INT8 (same as the manual's scope: 6 window CNNs).
