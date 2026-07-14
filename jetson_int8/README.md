# jetson-int8 — unified INT8 latency re-measurement

INT8 accuracy is unified on the host (one canonical QDQ ONNX per model, measured once,
device-independent). This branch ships those identical ONNX so each Jetson rebuilds its
INT8 TensorRT engine from the SAME artifact and re-measures LATENCY only — making the
per-device INT8 latency a clean hardware comparison.

- `onnx/<model>_int8.onnx` — 6 canonical QDQ INT8 ONNX (do not re-quantize)
- `run_int8_latency.sh <label>` — trtexec build + measure -> trt_int8_latency_<label>.csv
- `AGENT_PROMPT.md` — paste this to the AI on each Jetson to run it autonomously
