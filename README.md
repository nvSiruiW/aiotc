# Edge-Tier PDR Benchmark — Jetson Results

Reproductions of the **edge tier** of the learning-based PDR (pedestrian dead-reckoning)
benchmark on NVIDIA Jetson hardware. Each device lives in its own subdirectory with its
own measured efficiency / accuracy / INT8 results and a full write-up. All numbers are
**measured on-device**, never assumed. The source (card) tier is the Blackwell
RTX PRO 6000 reference shipped with the benchmark bundle.

| Device | Subdir | JetPack / CUDA | PyTorch | TensorRT | GPU clock (locked) | Write-up |
|---|---|---|---|---|---|---|
| **Jetson Orin Nano** (4GB) | [`orin_nano/`](orin_nano) | L4T R36.5 / CUDA 12.6 | 2.11.0 | **10.3** | 624.75 MHz | [summary](orin_nano/RESULTS_SUMMARY_orin_nano.md) |
| **Jetson AGX Orin** | [`agx_orin/`](agx_orin) | debug L4T / CUDA 13.3 | 2.13.0 | **11.1** | 1300.5 MHz | [summary](agx_orin/RESULTS_SUMMARY_orin.md) · [conditions](agx_orin/RUN_CONDITIONS_orin.txt) |

Both under `nvpmodel` MAXN with `jetson_clocks` locked; whole-board power via tegrastats
(Orin Nano rail `VDD_IN`, AGX rail `VIN_SYS_5V0`).

## Shared findings (hold on both devices)
- **Efficiency:** Jetsons are far slower per inference than the Blackwell card but use a
  small fraction of the energy (board 4–8 W vs card ~100–113 W). These window models are
  tiny / launch-bound, so latency & throughput are the reliable metrics.
- **Accuracy is device-independent** for a fixed PyTorch checkpoint: FP32/FP16 match the
  Blackwell reference within FP rounding (≤~0.2%) on both devices. TensorRT **FP16 engines
  also match FP32 within ≤0.2%**, validating the export/runtime path.
- **INT8 accuracy must be measured, never assumed** — and it degrades on both devices,
  but by how much depends on the model *and the quantization method* (see below).

## INT8 is the interesting axis — and the two runs used different TRT stacks
| | Orin Nano | AGX Orin |
|---|---|---|
| TensorRT | 10.3 | 11.1 |
| `export_trt.py` | ran **unmodified** (classic `IInt8EntropyCalibrator2`) | **rewritten** — TRT 11 removed the calibrator + `BuilderFlag.FP16/INT8` (strongly-typed only); INT8 done as **QDQ ONNX** via onnxruntime static quant |
| INT8 calibration | entropy (300 real IMU windows) | ORT MinMax symmetric PTQ |
| INT8 ΔATE vs FP32 | **model-dependent: +0.7% to +142%** | **+35% to +82% on every model** |
| INT8 latency | fastest; 8–26× over PyTorch-eager | mixed vs FP16; 20–40× over PyTorch-eager |

The Orin Nano's entropy calibration leaves efficientnet_b0 / tinyodom / ronin_resnet18
nearly lossless but **collapses the depthwise-separable nets (mobilenetv2 +70%, mnasnet
+142%)**; the AGX's MinMax-symmetric QDQ degrades *all* six models moderately (incl.
tinyodom +82%). Same headline — **INT8 PTQ is not accuracy-free for velocity regression** —
reached two ways. FP16 is the safe edge precision.

## Layout
```
orin_nano/   Orin Nano: profile/accuracy CSVs, INT8 CSVs, engines/ (12 .plan), logs, summary
agx_orin/    AGX Orin:  same shape (+ scripts/, RUN_CONDITIONS, 30W-unlocked profile)
```
See each device's summary for the exact per-model tables and the environment fixes required.
