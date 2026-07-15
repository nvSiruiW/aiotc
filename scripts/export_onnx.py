#!/usr/bin/env python3
"""Export a trained window->velocity backbone to ONNX (for TensorRT INT8 on Jetson).

Loads the same trained checkpoint used by eval_accuracy (so the ONNX carries the
real weights), then exports. Scope: the six window CNN backbones with a (1,6,W) input
and a single velocity output. LSTM/TLIO/EqNIO are not exported here.
"""
import sys, os, argparse
sys.path.insert(0, "scripts"); sys.path.insert(0, "ronin/source")
import torch
import eval_accuracy as E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(E.EVAL_MODELS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=13)
    a = ap.parse_args()
    builder, ckpt, window = E.EVAL_MODELS[a.model]
    net = builder()
    ck = torch.load(E.resolve_ckpt(a.model, ckpt), map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("model_state_dict", ck)); net.eval()
    x = torch.randn(1, 6, window)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.onnx.export(net, x, a.out, input_names=["imu"], output_names=["vel"],
                      opset_version=a.opset, dynamo=False)
    # numeric sanity: ONNX vs PyTorch
    try:
        import onnxruntime as ort, numpy as np
        y = net(x).detach().numpy()
        yo = ort.InferenceSession(a.out).run(None, {"imu": x.numpy()})[0]
        print(f"wrote {a.out}  input=(1,6,{window})  onnx-vs-torch max|Δ|={np.abs(y-yo).max():.2e}")
    except Exception:
        print(f"wrote {a.out}  input=(1,6,{window})  (onnxruntime not present; skipped numeric check)")

if __name__ == "__main__":
    main()
