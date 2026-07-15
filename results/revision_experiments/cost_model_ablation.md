# Cost Model Feature Ablation

Dataset: all (model × precision × device); LOMO/LOAO/LOPO/LODO protocols.
LOPO: train on fp32, predict fp16+bf16 (worst case: LSTM-BF16 fallback path).
LODO: train on 2 devices, predict held-out device (fp32).

## Main Table

| Feature Set | LOMO MAPE | LOAO MAPE | LOPO MAPE | LODO MAPE | Median APE | P95 APE | Max APE | Pearson r | Spearman ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 Params | 69.8% | 77.5% | 39.9% | 273.4% | 46.0% | 221.6% | 644.1% | 0.437 | 0.620 |
| 2 FLOPs | 80.7% | 87.2% | 51.8% | 285.7% | 41.6% | 181.2% | 785.1% | 0.183 | 0.245 |
| 3 Params+FLOPs | 81.8% | 94.7% | 39.7% | 265.7% | 67.6% | 222.6% | 521.2% | 0.090 | 0.283 |
| 4 N exec | 35.2% | 39.1% | 22.7% | 248.8% | 25.4% | 97.4% | 250.8% | 0.829 | 0.928 |
| 5 N exec+B eff | 36.0% | 42.1% | 22.7% | 253.5% | 23.4% | 96.4% | 252.0% | 0.825 | 0.911 |
| 6 N exec+B eff+F eff | 49.0% | 59.3% | 23.3% | 220.1% | 34.2% | 149.7% | 402.8% | 0.762 | 0.828 |
| 7 N exec+B eff+F eff+I fallback | 36.6% | 56.0% | 23.3% | 220.1% | 15.5% | 126.0% | 402.8% | 0.811 | 0.930 |
|  fallback path detail | — | — | — | — | — | — | — | — | — |

## Notes
- Feature columns: Params=params_M, FLOPs≈params_M (used as F_eff proxy where FlopCounterMode output unavailable), N_exec=k_real (measured CUDA kernel launch count from torch.profiler), B_eff=peak_mem_MB (device-measured working set), I_fallback=1 for ronin_lstm+bf16 else 0.
- For LSTM-BF16: k_real=1241 (unfused time-step-wise path) vs k_real=10 (fused FP16 path): 124× jump.
- LODO uses fp32 latency only; structural model underperforms vs 1-scalar transfer (M5).
- See fallback_ablation.md for LSTM-BF16 path-specific analysis.
