#!/usr/bin/env python3
"""INT8/FP16 TensorRT accuracy (ATE/RTE) on the MagPIE test set.

Reuses the EXACT reconstruction + metric code from eval_accuracy.py
(StridedSequenceDataset, recon_traj, compute_ate_rte) and only swaps the
PyTorch forward for a TensorRT engine forward. INT8 accuracy is MEASURED here,
never assumed equal to FP.

Engines are the fixed batch-1 (1,6,W) .plan files from export_trt.py, so windows
are fed one at a time. TensorRT 10.x tensor API; torch CUDA tensors as I/O buffers
(no pycuda).

Usage:
  python eval_int8_accuracy.py --models ronin_resnet18,imunet,... \
     --engines-dir engines/orin_nano --precisions int8,fp16 \
     --root data/ronin_dataset --test_list data/splits/test_list.txt \
     --out results/accuracy_int8_orin_nano.csv
"""
import sys, os, csv, argparse, statistics
sys.path.insert(0, "ronin/source"); sys.path.insert(0, "scripts")
import numpy as np, torch, tensorrt as trt
import eval_accuracy as E
from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset


class TRTInfer:
    """Minimal TensorRT-10 batch-1 runner using torch CUDA buffers."""
    def __init__(self, plan_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(plan_path, "rb") as f:
            data = f.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {plan_path}")
        self.context = self.engine.create_execution_context()
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name = n
            else:
                self.out_name = n
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        # persistent output buffer
        self.out = torch.empty(self.out_shape, device="cuda", dtype=torch.float32)

    @torch.no_grad()
    def infer(self, x):
        # x: torch cuda float32 contiguous tensor matching in_shape
        self.context.set_tensor_address(self.in_name, x.data_ptr())
        self.context.set_tensor_address(self.out_name, self.out.data_ptr())
        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        torch.cuda.synchronize()
        return self.out.clone()


@torch.no_grad()
def eval_engine(name, plan_path, ds, seq_ids_order):
    trtnet = TRTInfer(plan_path)
    preds_all, sid_all, fid_all = [], [], []
    for i in range(len(ds)):
        feat, targ, sid, fid = ds[i]
        x = torch.as_tensor(np.ascontiguousarray(feat), device="cuda", dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)                      # (1,6,W)
        out = trtnet.infer(x)                       # (1,2)
        preds_all.append(out.float().cpu().numpy().reshape(1, -1))
        sid_all.append(np.array([sid])); fid_all.append(np.array([fid]))
    preds_all = np.concatenate(preds_all); sid_all = np.concatenate(sid_all); fid_all = np.concatenate(fid_all)
    ates, rtes = [], []
    for seq_id in seq_ids_order:
        m = sid_all == seq_id
        if m.sum() < 3:
            continue
        p = preds_all[m]; ind = fid_all[m]
        order = np.argsort(ind); p, ind = p[order], ind[order]
        pos_pred = E.recon_traj(ds, p, seq_id, ind)
        gt = ds.gt_pos[seq_id][:, :2]
        ate, rte = E.compute_ate_rte(pos_pred, gt)
        ates.append(ate); rtes.append(rte)
    return statistics.mean(ates), statistics.mean(rtes), len(ates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ronin_resnet18,imunet,mobilenetv2,mnasnet,efficientnet_b0,tinyodom")
    ap.add_argument("--engines-dir", required=True)
    ap.add_argument("--precisions", default="int8,fp16")
    ap.add_argument("--test_list", default="data/splits/test_list.txt")
    ap.add_argument("--root", default="data/ronin_dataset")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    test_list = E.load_list(args.test_list)
    ds_cache = {}
    rows = []
    for name in args.models.split(","):
        name = name.strip()
        if name not in E.EVAL_MODELS:
            print(f"[skip] {name} (not a window CNN)"); continue
        window = E.EVAL_MODELS[name][2]
        if window not in ds_cache:
            ds = StridedSequenceDataset(GlobSpeedSequence, args.root, test_list,
                                        cache_path=f"/tmp/pdreval_w{window}.pkl",
                                        step_size=args.step, window_size=window, shuffle=False)
            seq_ids_order = sorted(set(i[0] for i in ds.index_map))
            ds_cache[window] = (ds, seq_ids_order)
        ds, seq_ids_order = ds_cache[window]
        for prec in args.precisions.split(","):
            prec = prec.strip()
            plan = os.path.join(args.engines_dir, f"{name}_{prec}.plan")
            if not os.path.exists(plan):
                print(f"  [{name}/{prec}] MISSING engine {plan}"); continue
            try:
                ate, rte, n = eval_engine(name, plan, ds, seq_ids_order)
                rows.append(dict(model=name, precision=f"trt_{prec}", ate_m=round(ate, 4), rte_m=round(rte, 4), n_seq=n))
                print(f"  [{name}/{prec}] ATE={ate:.4f}m RTE={rte:.4f}m ({n} seqs)", flush=True)
            except Exception as e:
                print(f"  [{name}/{prec}] FAILED: {e}", flush=True)
    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
