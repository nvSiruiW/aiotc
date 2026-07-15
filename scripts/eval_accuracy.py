#!/usr/bin/env python3
"""Localization-accuracy (ATE/RTE) evaluation on the MagPIE test set, at FP32/FP16/BF16.

Reconstructs each test trajectory by integrating predicted global 2-D velocity
(RoNIN protocol) and reports ATE/RTE against tango ground truth. Precision is
device-independent, so the FP16/BF16 delta vs FP32 is the reported contribution.

Covers the window->velocity backbones (RoNIN-ResNet + the five models trained by
train_pdr.py). Seq2seq (LSTM/TCN) and TLIO/EqNIO are handled separately.
"""
import sys, os, csv, argparse, statistics
sys.path.insert(0, "ronin/source"); sys.path.insert(0, "scripts")
import numpy as np, torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset
from metric import compute_ate_rte

AIOTC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL = os.path.join(AIOTC, "..", "magpie1_wlk_pipeline")

# ---------------- model builders + checkpoints ----------------
def _ronin_resnet():
    from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
    cfg = {'fc_dim':512,'in_dim':7,'dropout':0.5,'trans_planes':128}
    return ResNet1D(6,2,BasicBlock1D,[2,2,2,2],base_plane=64,output_block=FCOutputModule,kernel_size=3,**cfg)
def _imunet():
    if "IMUNet/RONIN_torch" not in sys.path: sys.path.insert(0,"IMUNet/RONIN_torch")
    from IMUNet import IMUNet
    return IMUNet(num_classes=2, input_size=(1,6,200), sampling_rate=200, num_T=32, num_S=64, hidden=64, dropout_rate=0.5)
def _ext(name):
    from models_ext import EXT_REGISTRY
    return EXT_REGISTRY[name]()[0]

# fall back to the journal checkpoint until the same-source retrain lands (reproducible now, auto-upgrades)
JOURNAL_CKPT = {
    "ronin_resnet18": f"{JOURNAL}/pdr_models/resnet/checkpoints/checkpoint_best.pt",
    "ronin_lstm":     f"{JOURNAL}/pdr_models/lstm/checkpoints/checkpoint_best.pt",
    "ronin_tcn":      f"{JOURNAL}/pdr_models/tcn/checkpoints/checkpoint_best.pt",
}
def resolve_ckpt(name, primary):
    return primary if os.path.exists(primary) else JOURNAL_CKPT.get(name, primary)

# name -> (builder, checkpoint_path, window)
EVAL_MODELS = {
    "ronin_resnet18":  (_ronin_resnet, f"{AIOTC}/trained_models/ronin_resnet18/checkpoint_best.pt", 200),
    "imunet":          (_imunet,        f"{AIOTC}/trained_models/imunet/checkpoint_best.pt", 200),
    "mobilenetv2":     (lambda:_ext("mobilenetv2"),     f"{AIOTC}/trained_models/mobilenetv2/checkpoint_best.pt", 200),
    "mnasnet":         (lambda:_ext("mnasnet"),         f"{AIOTC}/trained_models/mnasnet/checkpoint_best.pt", 200),
    "efficientnet_b0": (lambda:_ext("efficientnet_b0"), f"{AIOTC}/trained_models/efficientnet_b0/checkpoint_best.pt", 200),
    "tinyodom":        (lambda:_ext("tinyodom"),        f"{AIOTC}/trained_models/tinyodom/checkpoint_best.pt", 400),
}
DT = {"fp32":torch.float32, "fp16":torch.float16, "bf16":torch.bfloat16}

# ---------------- seq2seq (LSTM/TCN): whole-sequence feed + per-frame integration ----------------
def _ronin_lstm(dev):
    import types
    from model_temporal import LSTMSeqNetwork
    net = LSTMSeqNetwork(6, 2, 1, torch.device(dev), lstm_size=100, lstm_layers=3, dropout=0)  # batch=1 for eval
    def init_weights(self):   # hidden state must match model device AND dtype (fp16/bf16)
        p = next(self.parameters())
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size, device=p.device, dtype=p.dtype)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size, device=p.device, dtype=p.dtype)
        return (h0, c0)
    net.init_weights = types.MethodType(init_weights, net)
    return net
def _ronin_tcn(dev):
    from model_temporal import TCNSeqNetwork
    return TCNSeqNetwork(6, 2, 3, [32,64,128,256,72,36], dropout=0.2)
# name -> (builder(dev), checkpoint_path)
SEQ_MODELS = {
    "ronin_lstm": (_ronin_lstm, f"{AIOTC}/trained_models/ronin_lstm/checkpoint_best.pt"),
    "ronin_tcn":  (_ronin_tcn,  f"{AIOTC}/trained_models/ronin_tcn/checkpoint_best.pt"),
}

@torch.no_grad()
def eval_seq_model(name, prec, test_list, root, dev):
    builder, ckpt_path = SEQ_MODELS[name]
    net = builder(dev)
    ck = torch.load(resolve_ckpt(name, ckpt_path), map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("model_state_dict", ck))
    net = net.to(dev).to(DT[prec]).eval()
    ates, rtes = [], []
    for seq_name in test_list:
        seq = GlobSpeedSequence(os.path.join(root, seq_name))
        feat = seq.get_feature(); gt = seq.gt_pos[:, :2]; ts = seq.ts
        x = torch.tensor(feat, device=dev, dtype=DT[prec]).unsqueeze(0)   # (1, N, 6)
        pred = net(x)
        if isinstance(pred, (tuple, list)): pred = pred[0]
        pred = pred[0].float().cpu().numpy()                              # (N, 2) per-frame velocity
        n = min(len(pred), len(gt), len(ts))
        dt = float(np.mean(np.diff(ts[:n])))
        pos = np.cumsum(pred[:n] * dt, axis=0) + gt[0]
        ate, rte = compute_ate_rte(pos, gt[:n])
        ates.append(ate); rtes.append(rte)
    return statistics.mean(ates), statistics.mean(rtes), len(ates)

def recon_traj(ds, preds, seq_id, ind):
    """Integrate predicted global 2-D velocity into a trajectory (RoNIN protocol)."""
    ts = ds.ts[seq_id]; ind = np.asarray(ind, dtype=np.int64)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0]+2, 2])
    pos[0] = ds.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2]*dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0]-1e-6], ts[ind], [ts[-1]+1e-6]])
    return interp1d(ts_ext, pos, axis=0)(ts)

def load_list(p): return [l.strip() for l in open(p) if l.strip()]

@torch.no_grad()
def eval_model(name, prec, ds, seq_ids_order, dev):
    builder, ckpt_path, window = EVAL_MODELS[name]
    net = builder()
    ck = torch.load(resolve_ckpt(name, ckpt_path), map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("model_state_dict", ck))
    net = net.to(dev).to(DT[prec]).eval()
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4)
    preds_all, sid_all, fid_all = [], [], []
    for feat, targ, sid, fid in loader:
        pred = net(feat.to(dev).to(DT[prec]))
        if isinstance(pred, (tuple, list)): pred = pred[0]
        preds_all.append(pred.float().cpu().numpy()); sid_all.append(sid.numpy()); fid_all.append(fid.numpy())
    preds_all = np.concatenate(preds_all); sid_all = np.concatenate(sid_all); fid_all = np.concatenate(fid_all)
    ates, rtes = [], []
    for seq_id in seq_ids_order:
        m = sid_all == seq_id
        if m.sum() < 3: continue
        p = preds_all[m]; ind = fid_all[m]
        order = np.argsort(ind); p, ind = p[order], ind[order]
        pos_pred = recon_traj(ds, p, seq_id, ind)
        gt = ds.gt_pos[seq_id][:, :2]
        ate, rte = compute_ate_rte(pos_pred, gt)
        ates.append(ate); rtes.append(rte)
    return statistics.mean(ates), statistics.mean(rtes), len(ates)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ronin_resnet18")
    ap.add_argument("--precisions", default="fp32,fp16,bf16")
    ap.add_argument("--test_list", default=f"{JOURNAL}/splits/test_list.txt")
    ap.add_argument("--root", default=f"{JOURNAL}/ronin_dataset")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", default=f"{AIOTC}/results/accuracy_blackwell.csv")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_list = load_list(args.test_list)

    # build one dataset per window size, reused across precisions
    ds_cache = {}
    rows = []
    for name in args.models.split(","):
        name = name.strip()
        is_seq = name in SEQ_MODELS
        if not is_seq and name not in EVAL_MODELS: print(f"[skip] {name}"); continue
        if not is_seq:
            window = EVAL_MODELS[name][2]
            if window not in ds_cache:
                ds = StridedSequenceDataset(GlobSpeedSequence, args.root, test_list,
                                            cache_path=f"/tmp/pdreval_w{window}.pkl",
                                            step_size=args.step, window_size=window, shuffle=False)
                seq_ids_order = sorted(set(i[0] for i in ds.index_map))
                ds_cache[window] = (ds, seq_ids_order)
            ds, seq_ids_order = ds_cache[window]
        for prec in args.precisions.split(","):
            prec = prec.strip()
            try:
                if is_seq:
                    ate, rte, n = eval_seq_model(name, prec, test_list, args.root, dev)
                else:
                    ate, rte, n = eval_model(name, prec, ds, seq_ids_order, dev)
                rows.append(dict(model=name, precision=prec, ate_m=round(ate,4), rte_m=round(rte,4), n_seq=n))
                print(f"  [{name}/{prec}] ATE={ate:.3f}m RTE={rte:.3f}m ({n} seqs)", flush=True)
            except Exception as e:
                print(f"  [{name}/{prec}] FAILED: {e}", flush=True)
    if rows:
        out = args.out; append = args.append and os.path.exists(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a" if append else "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not append: w.writeheader()
            w.writerows(rows)
        print(f"{'appended' if append else 'wrote'} {out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
