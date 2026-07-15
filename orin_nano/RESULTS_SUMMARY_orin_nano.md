# Edge-Tier Reproduction — Jetson Orin Nano

## Measurement conditions (part of the results)
- **Device:** NVIDIA Jetson Orin Nano Developer Kit (Ampere sm_87, ~3.5 GB shared RAM → 4GB-class module)
- **L4T:** R36.5.0 (JetPack 6.x); **CUDA** 12.6; **cuDNN** 9.3.0
- **PyTorch:** 2.11.0 (NVIDIA Jetson cu126 wheel), `torch.cuda.is_available()=True`, device "Orin"
- **nvpmodel:** mode 0 ("NV Power Mode: 10W", MAXN)
- **jetson_clocks:** ON — CPU pinned 1.51 GHz ×6, GPU pinned 624.75 MHz
- **Power boundary:** whole-board via tegrastats **VDD_IN** rail (source tier is card-level nvidia-smi → compare trends/energy, not absolute watts)

## Environment fixes applied (non-sudo) to make torch work
1. `pip install nvidia-cuda-cupti-cu12==12.6.80` + symlink `libcupti.so.12` into torch/lib (wheel dep, absent on system).
2. `pip install nvidia-cudss-cu12==0.5.0.16` + symlink `libcudss.so.0` into torch/lib.
3. cuDSS dragged in pip cuBLAS 12.9 whose RPATH shadowed the system lib; cuBLAS 12.9 > CUDA 12.6 driver → `cublasCreate` failed `CUBLAS_STATUS_ALLOC_FAILED`. Fixed by repointing `nvidia/cublas/lib/libcublas*.so.12` to the system CUDA-12.6 libs.
4. `scripts/profile_device.py PowerSampler._read`: tegrastats never exits so `check_output(timeout=)` was discarding output → power blank. Patched to capture `TimeoutExpired.output`. (Change is documented; a per-board tuning point the manual explicitly anticipates.)

## Efficiency (NEW numbers) — FP32, Orin Nano (board) vs Blackwell (card)
| model | lat_bw ms | lat_orin ms | slowdown | E_bw mJ | E_orin mJ | E_ratio | P_bw W | P_orin W |
|---|---|---|---|---|---|---|---|---|
| ronin_resnet18 | 1.19 | 10.69 | 9.0x | 130.8 | 63.7 | 0.49x | 112.7 | 7.77 |
| ronin_tcn | 1.42 | 8.21 | 5.8x | 148.9 | 52.5 | 0.35x | 105.7 | 6.49 |
| ronin_lstm | 1.47 | 3.30 | 2.2x | 153.0 | 18.1 | 0.12x | 111.9 | 6.37 |
| imunet | 1.80 | 13.77 | 7.6x | 180.8 | 80.8 | 0.45x | 103.0 | 6.49 |
| mobilenetv2 | 2.18 | 16.58 | 7.6x | 216.8 | 100.2 | 0.46x | 100.7 | 6.10 |
| mnasnet | 2.31 | 15.78 | 6.8x | 219.7 | 96.6 | 0.44x | 101.7 | 6.14 |
| efficientnet_b0 | 2.28 | 17.29 | 7.6x | 232.6 | 105.6 | 0.45x | 102.5 | 6.14 |
| tlio_resnet | 1.33 | 11.05 | 8.3x | 142.2 | 71.4 | 0.50x | 110.2 | 7.69 |
| tinyodom | 0.94 | 6.29 | 6.7x | 94.7 | 37.7 | 0.40x | 104.2 | 6.10 |
| eqnio | 3.78 | 24.97 | 6.6x | 395.8 | 163.0 | 0.41x | 105.3 | 6.65 |

Story: Orin Nano is 2.2–9× slower per inference but consumes **0.12–0.50× the energy per inference** (board 6–8 W vs card ~100–113 W). Full CSV incl. fp16/bf16: `results/profile_orin_nano.csv`.
Note: `ronin_lstm bf16` is a pathological outlier (95.5 ms, 10.5 ips) — bf16 LSTM is slow on both tiers (Blackwell bf16 also spiked to 4.4 ms).

## Accuracy (should MATCH source tier — device-independence check)
All 24 configs (8 backbones × fp32/fp16/bf16) over 53 MagPIE test sequences.
**Max |ATE delta| vs Blackwell = 0.59%** (mnasnet bf16). FP32/FP16 ≤0.03%. **No model exceeds the 5% flag threshold → device-independence CONFIRMED.**
Full CSV: `results/accuracy_orin_nano.csv`. Example: RoNIN-ResNet FP32 ATE = 0.9020 m (ref 0.9018 m).

## INT8 / TensorRT (STEP 6) — DONE (TensorRT 10.3.0)
User installed JetPack TensorRT (`nvidia-tensorrt`, `python3-libnvinfer`, `libnvinfer-bin`, `nvidia-l4t-dla-compiler`). Wired into the venv via a `tensorrt` symlink + `LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia` (for `libnvdla_compiler.so`). Also `pip install onnx` for the ONNX export.
`export_trt.py` needed **no** API changes — it was already TensorRT-10-correct. All **12 engines built** (6 window CNNs × FP16/INT8), real entropy calibration on 300 MagPIE IMU windows.

**Latency (trtexec, 500 iters) + accuracy (measured over 53 seqs, batch-1 engine):**
| model | ATE fp32 (py) | ATE trt_fp16 | ATE **int8** | **INT8 ΔATE** | lat py-fp16 ms | lat trt-fp16 ms | lat **int8** ms | int8 speedup vs py-fp16 |
|---|---|---|---|---|---|---|---|---|
| ronin_resnet18 | 0.9020 | 0.9033 | 0.9485 | **+5.2%** | 11.31 | 0.693 | 0.510 | 22.2× |
| imunet | 0.9704 | 0.9691 | 1.0494 | **+8.1%** | 15.10 | 0.848 | 0.744 | 20.3× |
| mobilenetv2 | 0.9340 | 0.9354 | 1.5854 | **+69.7%** ⚠️ | 18.96 | 0.833 | 0.743 | 25.5× |
| mnasnet | 0.9039 | 0.9039 | 2.1914 | **+142.4%** ⚠️ | 17.95 | 0.916 | 0.771 | 23.3× |
| efficientnet_b0 | 0.9810 | 0.9819 | 0.9876 | +0.7% | 19.34 | 1.527 | 1.427 | 13.5× |
| tinyodom | 1.2068 | 1.2053 | 1.2055 | −0.1% | 6.25 | 0.740 | 0.799 | 7.8× |

**Findings (INT8 re-measured, never assumed):**
- **FP16 engines match FP32 within ≤0.2%** → validates the TRT export/runtime path.
- **INT8 accuracy is strongly model-dependent.** Robust: efficientnet_b0, tinyodom (~0%), ronin_resnet18 (+5%); moderate: imunet (+8%); **catastrophic: mobilenetv2 (+70%), mnasnet (+142%)**. The two depthwise-separable nets degrade badly under per-tensor entropy calibration (depthwise channels have very different dynamic ranges) — a known INT8 failure mode, and exactly why INT8 accuracy must be measured.
- **INT8 latency:** fastest of all precisions and 8–26× faster than PyTorch-eager FP16. INT8 beats FP16 on 5/6 models; `tinyodom` INT8 is slightly slower than its FP16 (quantize/dequantize overhead dominates a 0.11M-param net — its INT8 engine is also larger than FP16).
- Engines: `engines/orin_nano/*.plan` (12). INT8 latency CSV: `results/profile_int8_orin_nano.csv`. INT8/FP16 ATE/RTE: `results/accuracy_int8_orin_nano.csv`. Runtime wrapper written: `scripts/eval_int8_accuracy.py` (reuses eval_accuracy's recon_traj + compute_ate_rte).

## Honesty notes
- tegrastats power required a code fix (above) to be non-blank; values are real VDD_IN board watts.
- Benign, non-fatal `NvMapMemAllocInternalTagged ... error 12` lines appeared in the accuracy log (L4T allocator retries under the 4GB memory limit); all 24 rows still produced with n_seq=53 and exit 0.
- Accuracy numbers are for the fixed checkpoints; INT8 was not run.
