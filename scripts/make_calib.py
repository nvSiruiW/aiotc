#!/usr/bin/env python3
"""Build a real INT8 calibration set from the MagPIE test IMU windows.

INT8 latency/energy is valid with any calibration data, but INT8 ACCURACY needs a
representative set. This extracts global-frame 6-channel windows (same features the
models consume) and saves them as [N, 6, W] float32 for the TensorRT calibrator.
"""
import sys, os, argparse
sys.path.insert(0, "ronin/source")
import numpy as np
from data_glob_speed import GlobSpeedSequence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ronin_dataset")
    ap.add_argument("--test_list", default="data/splits/test_list.txt")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="calib/imu_calib_200.npy")
    a = ap.parse_args()
    seqs = [l.strip() for l in open(a.test_list) if l.strip()]
    wins = []
    for s in seqs:
        feat = GlobSpeedSequence(os.path.join(a.root, s)).get_feature()   # (N,6)
        for i in range(0, len(feat) - a.window, a.window):
            wins.append(feat[i:i+a.window].T.astype("float32"))            # (6,window)
            if len(wins) >= a.n: break
        if len(wins) >= a.n: break
    arr = np.stack(wins)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out, arr)
    print(f"wrote {a.out}  shape={arr.shape}")

if __name__ == "__main__":
    main()
