#!/usr/bin/env python3
"""Measure ATE/RTE of TensorRT engines (fp16 / int8) over the 53 MagPIE sequences.

Reuses the EXACT reconstruction + metric from scripts/eval_accuracy.py (recon_traj,
compute_ate_rte, StridedSequenceDataset) so INT8/FP16 accuracy is comparable to the
PyTorch FP numbers. INT8 accuracy is MEASURED here, never assumed equal to FP.

TRT runtime uses torch CUDA tensors as I/O buffers (no pycuda). Engines are static
batch=1 (as built by export_trt_trt11.py), so windows are fed one at a time.

Usage:
  python eval_trt_accuracy.py --engines engines/orin --precisions fp16,int8 \
      --models ronin_resnet18,imunet,mobilenetv2,mnasnet,efficientnet_b0,tinyodom \
      --root data/ronin_dataset --test_list data/splits/test_list.txt \
      --out results/accuracy_int8_orin.csv
"""
import argparse, os, sys, csv, statistics, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import torch, tensorrt as trt
import eval_accuracy as E
from torch.utils.data import DataLoader

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
_NP2T = {np.float32: torch.float32, np.float16: torch.float16}


class TRTRunner:
    def __init__(self, plan_path):
        rt = trt.Runtime(TRT_LOGGER)
        self.engine = rt.deserialize_cuda_engine(open(plan_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dt = _NP2T[trt.nptype(self.engine.get_tensor_dtype(self.in_name))]
        self.out_dt = _NP2T[trt.nptype(self.engine.get_tensor_dtype(self.out_name))]
        self.out_dim = int(tuple(self.ctx.get_tensor_shape(self.out_name))[-1])
        self.stream = torch.cuda.Stream()

    def run_batch(self, feat_cuda):  # feat_cuda: (Nb,C,W) fp32 on cuda; engine is batch=1
        Nb = feat_cuda.shape[0]
        x = feat_cuda.to(self.in_dt).contiguous()
        out = torch.empty((Nb, self.out_dim), dtype=self.out_dt, device="cuda")
        # enqueue all windows on one stream (distinct I/O addresses), sync ONCE
        for i in range(Nb):
            self.ctx.set_tensor_address(self.in_name, x[i:i+1].data_ptr())
            self.ctx.set_tensor_address(self.out_name, out[i:i+1].data_ptr())
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out.float()


def eval_engine(plan_path, ds, seq_ids_order):
    runner = TRTRunner(plan_path)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4)
    preds_all, sid_all, fid_all = [], [], []
    for feat, targ, sid, fid in loader:
        feat = feat.cuda(non_blocking=True)
        outs = runner.run_batch(feat)
        preds_all.append(outs.cpu().numpy()); sid_all.append(sid.numpy()); fid_all.append(fid.numpy())
    preds_all = np.concatenate(preds_all); sid_all = np.concatenate(sid_all); fid_all = np.concatenate(fid_all)
    ates, rtes = [], []
    for seq_id in seq_ids_order:
        m = sid_all == seq_id
        if m.sum() < 3: continue
        p = preds_all[m]; ind = fid_all[m]
        order = np.argsort(ind); p, ind = p[order], ind[order]
        pos_pred = E.recon_traj(ds, p, seq_id, ind)
        gt = ds.gt_pos[seq_id][:, :2]
        ate, rte = E.compute_ate_rte(pos_pred, gt)
        ates.append(ate); rtes.append(rte)
    return statistics.mean(ates), statistics.mean(rtes), len(ates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="engines/orin")
    ap.add_argument("--models", default="ronin_resnet18,imunet,mobilenetv2,mnasnet,efficientnet_b0,tinyodom")
    ap.add_argument("--precisions", default="fp16,int8")
    ap.add_argument("--root", default="data/ronin_dataset")
    ap.add_argument("--test_list", default="data/splits/test_list.txt")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", default="results/accuracy_int8_orin.csv")
    args = ap.parse_args()

    test_list = E.load_list(args.test_list)
    ds_cache, rows = {}, []
    for name in [m.strip() for m in args.models.split(",")]:
        if name not in E.EVAL_MODELS:
            print(f"[skip] {name} (not a window CNN)"); continue
        window = E.EVAL_MODELS[name][2]
        if window not in ds_cache:
            ds = E.StridedSequenceDataset(E.GlobSpeedSequence, args.root, test_list,
                                          cache_path=f"/tmp/pdreval_w{window}.pkl",
                                          step_size=args.step, window_size=window, shuffle=False)
            seq_ids_order = sorted(set(i[0] for i in ds.index_map))
            ds_cache[window] = (ds, seq_ids_order)
        ds, seq_ids_order = ds_cache[window]
        for prec in [p.strip() for p in args.precisions.split(",")]:
            plan = os.path.join(args.engines, f"{name}_{prec}.plan")
            if not os.path.exists(plan):
                print(f"  [{name}/{prec}] no engine at {plan}"); continue
            try:
                ate, rte, n = eval_engine(plan, ds, seq_ids_order)
                rows.append(dict(model=name, precision=f"trt_{prec}", ate_m=round(ate,4), rte_m=round(rte,4), n_seq=n))
                print(f"  [{name}/trt_{prec}] ATE={ate:.4f}m RTE={rte:.4f}m ({n} seqs)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [{name}/trt_{prec}] FAILED: {type(e).__name__}: {e}", flush=True)
    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
