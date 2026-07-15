#!/usr/bin/env python3
"""Unified RoNIN-style trainer on MagPIE — window (CNN) and seq2seq (LSTM/TCN).

All window->velocity CNN backbones (RoNIN-ResNet, IMUNet, MobileNetV2, MnasNet,
EfficientNet-B0, TinyOdom) share the RoNIN-ResNet recipe (Adam lr=1e-4, batch=256,
step=10, RandomHoriRotate augmentation). The sequence models (RoNIN-LSTM/TCN) use
RoNIN's native seq2seq recipe (batch=72, step=100, window=400, RandomHoriRotateSeq,
per-model lr). Reuses RoNIN's data loaders so everything is same-source and fair.
Saves checkpoint_best.pt (model_state_dict).
"""
import sys, os, time, math, argparse, statistics
sys.path.insert(0, "ronin/source"); sys.path.insert(0, "scripts")
import numpy as np, torch
from torch.utils.data import DataLoader
from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset, SequenceToSequenceDataset
from transformations import RandomHoriRotate, RandomHoriRotateSeq, ComposeTransform

# per-model recipe. seq=False -> window->velocity (RoNIN-ResNet recipe);
# seq=True -> seq2seq (RoNIN's LSTM/TCN recipe, journal per-model lr).
RECIPE = {
    "ronin_resnet18":  dict(window=200, batch=256, lr=1e-4, step=10,  seq=False),
    "imunet":          dict(window=200, batch=256, lr=1e-4, step=10,  seq=False),
    "mobilenetv2":     dict(window=200, batch=256, lr=1e-4, step=10,  seq=False),
    "mnasnet":         dict(window=200, batch=256, lr=1e-4, step=10,  seq=False),
    "efficientnet_b0": dict(window=200, batch=256, lr=1e-4, step=10,  seq=False),
    "tinyodom":        dict(window=400, batch=256, lr=1e-4, step=10,  seq=False),
    "ronin_lstm":      dict(window=400, batch=72,  lr=3e-4, step=100, seq=True),
    "ronin_tcn":       dict(window=400, batch=72,  lr=1e-3, step=100, seq=True),
}

def build_net(name, dev, batch):
    if name == "ronin_resnet18":
        from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
        cfg = {'fc_dim':512,'in_dim':7,'dropout':0.5,'trans_planes':128}
        return ResNet1D(6,2,BasicBlock1D,[2,2,2,2],base_plane=64,output_block=FCOutputModule,kernel_size=3,**cfg)
    if name == "imunet":
        if "IMUNet/RONIN_torch" not in sys.path: sys.path.insert(0,"IMUNet/RONIN_torch")
        from IMUNet import IMUNet
        return IMUNet(num_classes=2, input_size=(1,6,200), sampling_rate=200, num_T=32, num_S=64, hidden=64, dropout_rate=0.5)
    if name == "ronin_lstm":
        from model_temporal import LSTMSeqNetwork
        return LSTMSeqNetwork(6, 2, batch, torch.device(dev), lstm_size=100, lstm_layers=3, dropout=0)
    if name == "ronin_tcn":
        from model_temporal import TCNSeqNetwork
        return TCNSeqNetwork(6, 2, 3, [32,64,128,256,72,36], dropout=0.2)
    from models_ext import EXT_REGISTRY
    return EXT_REGISTRY[name]()[0]

def make_ds(root, lst, window, step, cache, is_train, seq):
    shift = step // 2 if is_train else 0
    if seq:
        tf = ComposeTransform([RandomHoriRotateSeq([0,3,6],[0,2])] if is_train else [])
        return SequenceToSequenceDataset(GlobSpeedSequence, root, lst, cache, step, window,
                                         random_shift=shift, transform=tf, shuffle=is_train,
                                         max_velocity_norm=3.0)
    tf = RandomHoriRotate(math.pi*2) if is_train else None
    return StridedSequenceDataset(GlobSpeedSequence, root, lst, cache_path=cache, step_size=step,
                                  window_size=window, random_shift=shift, transform=tf, shuffle=is_train)

def load_list(p): return [l.strip() for l in open(p) if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RECIPE.keys()))
    ap.add_argument("--root", required=True); ap.add_argument("--train_list", required=True)
    ap.add_argument("--val_list", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--cuda", default="0")
    args = ap.parse_args()
    r = RECIPE[args.model]; window, batch, lr, step, seq = r["window"], r["batch"], r["lr"], r["step"], r["seq"]
    dev = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    net = build_net(args.model, dev, batch).to(dev)
    print(f"[{args.model}] params={sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"window={window} batch={batch} lr={lr} step={step} seq={seq} dev={dev}", flush=True)

    croot = f"/tmp/pdrcache_{'seqw' if seq else 'w'}{window}"
    tr = make_ds(args.root, load_list(args.train_list), window, step, f"{croot}_train.pkl", True,  seq)
    va = make_ds(args.root, load_list(args.val_list),   window, step, f"{croot}_val.pkl",   False, seq)
    trL = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=4, drop_last=True)
    vaL = DataLoader(va, batch_size=batch, shuffle=False, num_workers=4, drop_last=seq)   # LSTM needs fixed batch
    print(f"  train_windows={len(tr)} val_windows={len(va)}", flush=True)

    crit = torch.nn.MSELoss(); opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, bad = float("inf"), 0
    for ep in range(args.epochs):
        net.train(); t0 = time.time()
        for feat, targ, _, _ in trL:
            feat, targ = feat.to(dev), targ.to(dev)
            pred = net(feat)
            if isinstance(pred, (tuple, list)): pred = pred[0]
            loss = crit(pred, targ)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); vs = []
        with torch.no_grad():
            for feat, targ, _, _ in vaL:
                pred = net(feat.to(dev))
                if isinstance(pred, (tuple, list)): pred = pred[0]
                vs.append(crit(pred, targ.to(dev)).item())
        v = statistics.mean(vs); sched.step(v); tag = ""
        if v < best:
            best, bad = v, 0
            torch.save({"model_state_dict": net.state_dict(), "epoch": ep, "val_mse": v,
                        "window": window, "arch": args.model}, args.out); tag = " *"
        else:
            bad += 1
        print(f"  ep{ep:3d} val_mse={v:.5f} lr={opt.param_groups[0]['lr']:.1e} ({time.time()-t0:.0f}s){tag}", flush=True)
        if bad >= args.patience:
            print("  early stop", flush=True); break
    print(f"[{args.model}] DONE best_val_mse={best:.5f} -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
