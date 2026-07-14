You are an autonomous engineering agent on an NVIDIA Jetson. You previously ran a PDR
edge benchmark on this device (PyTorch efficiency, PyTorch accuracy, and a first INT8
pass). We are now UNIFYING the INT8 measurement so it is comparable across devices.

## Background (why)
The two Jetsons used DIFFERENT INT8 quantization methods (one used TensorRT's classic
entropy calibrator, the other ONNX-QDQ static quantization), which produced different
quantized models and therefore non-comparable INT8 accuracy. We fixed this on the host:
there is now ONE canonical INT8 artifact — a fixed QDQ ONNX per model — whose accuracy is
measured once (device-independent). Your job is to re-measure INT8 **latency** on THIS
device by building TensorRT engines from that IDENTICAL canonical ONNX, so latency is a
clean hardware comparison rather than a quantization-method artifact.

## Inputs (from the `jetson-int8` branch of the repo)
- `jetson_int8/onnx/<model>_int8.onnx` — the 6 canonical QDQ INT8 ONNX (identical on both
  devices). Do NOT re-quantize; use these as-is.
- `jetson_int8/run_int8_latency.sh` — builds an INT8 TRT engine from each ONNX with
  trtexec and records GPU median latency + throughput.

## Ground rules
1. Never fabricate numbers. Report actual trtexec output. If an engine build fails, paste
   the exact error and adapt the trtexec flags to YOUR installed TensorRT version (you did
   this before — e.g. `--stronglyTyped` on TRT 10/11, `--int8` on TRT 8). Do not skip a
   model silently.
2. Use the provided canonical ONNX unchanged. The whole point is that both devices build
   from the identical artifact.
3. Lock the board first for reproducible latency: `sudo nvpmodel -m 0 && sudo jetson_clocks`
   (record the mode). If you can't sudo, say so.

## Steps
1. Get the branch:  `git fetch origin && git checkout jetson-int8`  (or `git pull` if already on it).
2. `cd jetson_int8`
3. `sudo nvpmodel -m 0 && sudo jetson_clocks`   (record nvpmodel mode + that clocks are locked)
4. Pick your device label: use `orin` for AGX Orin, `orin_nano` for Orin Nano.
5. `bash run_int8_latency.sh <label>`
   - This builds `engines/<model>_int8.plan` from each canonical ONNX and measures GPU
     median latency + throughput via trtexec.
   - If trtexec flags don't match your TRT version, edit the flag order in
     run_int8_latency.sh (it already tries `--stronglyTyped`, then `--int8`, then bare)
     and re-run.
6. (OPTIONAL, only if quick) Verify device-independence of INT8 accuracy: run one engine
   over the 53 MagPIE test sequences with the same velocity-integration reconstruction as
   `scripts/eval_accuracy.py` (you already have a TRT-runtime accuracy path from before,
   e.g. eval_trt_accuracy.py). It should match the host's canonical INT8 ATE within a few %.
   This is a confirmation, not required.

## Deliverables (report back)
- Device model, TensorRT version, nvpmodel mode, jetson_clocks on/off.
- `trt_int8_latency_<label>.csv` — paste it, and commit it to the `jetson-int8` branch:
  `git add trt_int8_latency_<label>.csv && git commit -m "int8 latency <label>" && git push origin jetson-int8`
- Any build failures verbatim + the trtexec flags you ended up using.
- (If you did the optional accuracy check) the INT8 ATE per model and whether it matched.

Start by checking out the `jetson-int8` branch and reading jetson_int8/run_int8_latency.sh.
