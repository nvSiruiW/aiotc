#!/usr/bin/env python3
"""Measure INT8 accuracy (ATE/RTE) of the CANONICAL QDQ ONNX via ONNXRuntime.

INT8 accuracy is a property of the quantized artifact, so it is device-independent and
evaluated once here (host). Reuses eval_accuracy's velocity-integration reconstruction
and RoNIN ATE/RTE. Also evaluates the FP32 ONNX as a cross-check that ONNX==PyTorch.
"""
import sys, os, csv, argparse, statistics
sys.path.insert(0, "scripts"); sys.path.insert(0, "ronin/source")
import numpy as np
from torch.utils.data import DataLoader
import onnxruntime as ort
import eval_accuracy as E
from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset
from metric import compute_ate_rte

WIN = {"ronin_resnet18":200,"imunet":200,"mobilenetv2":200,"mnasnet":200,"efficientnet_b0":200,"tinyodom":400}

def eval_onnx(onnx_path, ds, seq_ids, bs=256):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    P,S,F = [],[],[]
    for feat,targ,sid,fid in loader:
        y = sess.run(None, {"imu": feat.numpy().astype("float32")})[0]
        P.append(y); S.append(sid.numpy()); F.append(fid.numpy())
    P=np.concatenate(P); S=np.concatenate(S); F=np.concatenate(F)
    ates,rtes=[],[]
    for sid in seq_ids:
        m=S==sid
        if m.sum()<3: continue
        p=P[m]; ind=F[m]; o=np.argsort(ind); p,ind=p[o],ind[o]
        pos=E.recon_traj(ds,p,sid,ind); gt=ds.gt_pos[sid][:,:2]
        a,r=compute_ate_rte(pos,gt); ates.append(a); rtes.append(r)
    return statistics.mean(ates), statistics.mean(rtes), len(ates)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(WIN))
    ap.add_argument("--onnxdir", default="onnx_int8")
    ap.add_argument("--root", default="../magpie1_wlk_pipeline/ronin_dataset")
    ap.add_argument("--test_list", default="../magpie1_wlk_pipeline/splits/test_list.txt")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", default="results/accuracy_int8_canonical.csv")
    args=ap.parse_args()
    seqs=[l.strip() for l in open(args.test_list) if l.strip()]
    rows=[]
    ds_cache={}
    for m in args.models.split(","):
        m=m.strip(); window=WIN[m]
        if window not in ds_cache:
            ds=StridedSequenceDataset(GlobSpeedSequence, args.root, seqs,
                                      cache_path=f"/tmp/pdreval_w{window}.pkl",
                                      step_size=args.step, window_size=window, shuffle=False)
            ds_cache[window]=(ds, sorted(set(i[0] for i in ds.index_map)))
        ds,sids=ds_cache[window]
        for tag,fn in [("onnx_fp32", f"{args.onnxdir}/{m}.onnx"),
                       ("onnx_int8", f"{args.onnxdir}/{m}_int8.onnx")]:
            if not os.path.exists(fn): print(f"  [{m}/{tag}] missing {fn}"); continue
            a,r,n=eval_onnx(fn, ds, sids)
            rows.append(dict(model=m, precision=tag, ate_m=round(a,4), rte_m=round(r,4), n_seq=n))
            print(f"  [{m}/{tag}] ATE={a:.3f}m RTE={r:.3f}m ({n} seqs)", flush=True)
    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.out} ({len(rows)} rows)")

if __name__=="__main__":
    main()
