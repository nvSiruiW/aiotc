#!/usr/bin/env bash
# Rigorous dynamic-energy measurement on a Jetson, comparable to the datacenter card
# via idle-subtraction: E_dynamic = (P_active - P_idle) / throughput removes the
# whole-board static draw, leaving the model's MARGINAL energy per inference.
#
# Controls (what makes it publishable):
#   * locked clocks   : nvpmodel -m 0 + jetson_clocks  (mode recorded in CSV)
#   * thermal steady  : 60 s sustained warm-up per model until GPU temp plateaus
#   * provenance      : nvpmodel mode, GPU freq, temp_start/temp_end per row
#   * repeats         : 3 trials, reports mean + per-trial std of dynamic energy
#
# Usage:  bash jetson_energy/measure_energy_jetson.sh <orin|orin_nano>
set -e
LABEL="${1:?usage: bash measure_energy_jetson.sh <orin|orin_nano>}"
cd "$(dirname "$0")/.."                      # repo root
OUT="results/power_dynamic_${LABEL}.csv"

echo "== locking clocks for reproducible power =="
sudo nvpmodel -m 0 && sudo jetson_clocks && echo "clocks locked (MAXN)" \
  || echo "WARN: could not lock clocks (need sudo) — RECORD THIS in your report"
nvpmodel -q || true
tegrastats --interval 1000 | head -1 || true   # sanity: tegrastats works

MODELS="ronin_resnet18 tlio_resnet ronin_lstm ronin_tcn tinyodom imunet mobilenetv2 mnasnet efficientnet_b0 eqnio"
rm -f "$OUT"
first=1
for m in $MODELS; do
  A="--device $LABEL --model $m --precisions fp32,fp16,bf16 --dur 10 --runs 3 --thermal_warmup 60 --out $OUT"
  [ $first -eq 0 ] && A="$A --append"
  echo "== $m =="
  python3 scripts/measure_power.py $A
  first=0
done
echo ""
echo "DONE -> $OUT"
echo "Sanity checks before you push:"
echo "  * temp_start_C vs temp_end_C should be close (no thermal throttling drift)."
echo "  * power_mode column should read your nvpmodel mode (not '?')."
echo "  * dynamic_energy_std_mJ should be small relative to dynamic_energy_mJ."
