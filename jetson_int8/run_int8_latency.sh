#!/usr/bin/env bash
# Build INT8 TensorRT engines from the CANONICAL QDQ ONNX (identical on every device)
# and measure GPU latency with trtexec. Run ON EACH JETSON.
#   Usage: bash run_int8_latency.sh <device_label>      e.g.  orin   or   orin_nano
# Output: trt_int8_latency_<label>.csv  (model, int8 GPU median latency, throughput)
#
# Why: INT8 accuracy is already unified on the host (one number per model, from these
# same ONNX). This script re-measures INT8 LATENCY from the IDENTICAL artifact so the
# per-device latency is a clean hardware comparison (not a quantization-method artifact).
set -uo pipefail
cd "$(dirname "$0")"
DEV=${1:?usage: run_int8_latency.sh <device_label>}
TRTEXEC=${TRTEXEC:-$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)}
[ -x "$TRTEXEC" ] || { echo "trtexec not found; set TRTEXEC=/path/to/trtexec"; exit 1; }
OUT=trt_int8_latency_$DEV.csv
echo "model,precision,gpu_lat_med_ms,throughput_qps,status" > "$OUT"
mkdir -p engines
parse(){ echo "$1" | grep "GPU Compute Time" | grep -oE "median = [0-9.]+" | grep -oE "[0-9.]+" | head -1; }
thrpt(){ echo "$1" | grep -iE "Throughput:" | grep -oE "[0-9.]+" | head -1; }

for m in ronin_resnet18 imunet mobilenetv2 mnasnet efficientnet_b0 tinyodom; do
  onnx=onnx/${m}_int8.onnx; eng=engines/${m}_int8.plan
  echo "== $m =="
  # QDQ ONNX -> INT8 engine. Flags vary by TRT version: strongly-typed (TRT 10/11),
  # else --int8 (TRT 8/9). The QDQ nodes carry the quantization either way.
  if   b=$("$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --stronglyTyped 2>&1); then :
  elif b=$("$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --int8 2>&1);         then :
  elif b=$("$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" 2>&1);                then :
  else echo "  BUILD FAILED"; echo "$m,int8,,,build_fail" >> "$OUT"; continue; fi
  meas=$("$TRTEXEC" --loadEngine="$eng" --iterations=500 --avgRuns=100 2>&1)
  lat=$(parse "$meas"); thr=$(thrpt "$meas")
  echo "  int8: GPU median=${lat} ms  throughput=${thr} qps"
  echo "$m,int8,${lat},${thr},ok" >> "$OUT"
done
echo; echo "WROTE $OUT:"; column -s, -t "$OUT" 2>/dev/null || cat "$OUT"
echo "Send this CSV back (commit to the branch, or paste it)."
