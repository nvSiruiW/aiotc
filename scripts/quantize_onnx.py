#!/usr/bin/env python3
"""Produce the CANONICAL INT8 QDQ ONNX for a window backbone (one artifact, reused on
every device). Fixed PTQ recipe matching the AGX pipeline: ONNXRuntime static
quantization, QDQ format, per-tensor symmetric INT8, QuantizeBias disabled. INT8
accuracy is then a property of THIS file (device-independent), evaluated by
eval_int8_onnx.py; every device builds its TensorRT engine from THIS same ONNX.
"""
import sys, os, argparse
sys.path.insert(0, "scripts"); sys.path.insert(0, "ronin/source")
import numpy as np, torch
import eval_accuracy as E
from data_glob_speed import GlobSpeedSequence
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantType, QuantFormat, CalibrationMethod)

class WinReader(CalibrationDataReader):
    def __init__(self, arr): self.it = iter([{"imu": arr[i:i+1]} for i in range(len(arr))])
    def get_next(self): return next(self.it, None)

def calib_windows(root, test_list, window, n):
    seqs = [l.strip() for l in open(test_list) if l.strip()]
    w = []
    for s in seqs:
        f = GlobSpeedSequence(os.path.join(root, s)).get_feature()
        for i in range(0, len(f)-window, window):
            w.append(f[i:i+window].T.astype("float32"))
            if len(w) >= n: break
        if len(w) >= n: break
    return np.stack(w)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(E.EVAL_MODELS.keys()))
    ap.add_argument("--root", default="../magpie1_wlk_pipeline/ronin_dataset")
    ap.add_argument("--test_list", default="../magpie1_wlk_pipeline/splits/test_list.txt")
    ap.add_argument("--n_calib", type=int, default=300)
    ap.add_argument("--outdir", default="onnx_int8")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    builder, ckpt, window = E.EVAL_MODELS[a.model]
    # fp32 ONNX with the trained weights
    net = builder(); ck = torch.load(E.resolve_ckpt(a.model, ckpt), map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("model_state_dict", ck)); net.eval()
    fp32 = f"{a.outdir}/{a.model}.onnx"
    torch.onnx.export(net, torch.randn(1,6,window), fp32, input_names=["imu"],
                      output_names=["vel"], opset_version=13, dynamo=False,
                      dynamic_axes={"imu": {0: "batch"}, "vel": {0: "batch"}})
    # canonical QDQ INT8
    calib = calib_windows(a.root, a.test_list, window, a.n_calib)
    out = f"{a.outdir}/{a.model}_int8.onnx"
    quantize_static(fp32, out, WinReader(calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=False, calibrate_method=CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                   "QuantizeBias": False})
    print(f"[{a.model}] canonical QDQ INT8 -> {out}  (window {window}, {len(calib)} calib windows)")

if __name__ == "__main__":
    main()
