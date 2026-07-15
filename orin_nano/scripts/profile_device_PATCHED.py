#!/usr/bin/env python3
"""Portable PyTorch profiler: latency / throughput / power / energy / memory.

Runs on any device with torch (Blackwell GPU now; Jetson via JetPack torch).
Auto-detects power backend: tegrastats (Jetson, whole-board) or nvidia-smi (GPU, card).

Usage (run ON each device):
  python profile_device.py --device blackwell --models ronin_resnet18 --precisions fp32,fp16 \
      --out results/profile_blackwell.csv

Add new models in MODEL_REGISTRY.
"""
import argparse, os, sys, time, threading, subprocess, statistics, csv, shutil
sys.path.insert(0, "ronin/source")
import torch

# ---------------- model registry ----------------
def _ronin_resnet18():
    from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
    cfg = {'fc_dim':512,'in_dim':7,'dropout':0.5,'trans_planes':128}   # window 200 -> in_dim 7
    net = ResNet1D(6,2,BasicBlock1D,[2,2,2,2],base_plane=64,
                   output_block=FCOutputModule,kernel_size=3,**cfg).eval()
    return net, (1,6,200)

def _ronin_tcn():
    from model_temporal import TCNSeqNetwork
    net = TCNSeqNetwork(6, 2, 3, [32,64,128,256,72,36], dropout=0.2).eval()
    return net, (1,200,6)   # forward transposes (1,2) internally

def _ronin_lstm():
    import types
    from model_temporal import LSTMSeqNetwork
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = LSTMSeqNetwork(6, 2, 1, dev, lstm_size=100, lstm_layers=3, dropout=0).eval()
    def init_weights(self):   # hidden state must match model device AND dtype (fp16/fp32)
        p = next(self.parameters())
        h0=torch.zeros(self.num_layers, self.batch_size, self.lstm_size, device=p.device, dtype=p.dtype)
        c0=torch.zeros(self.num_layers, self.batch_size, self.lstm_size, device=p.device, dtype=p.dtype)
        return (h0, c0)
    net.init_weights = types.MethodType(init_weights, net)
    return net, (1,200,6)

def _imunet():
    if "IMUNet/RONIN_torch" not in sys.path: sys.path.insert(0, "IMUNet/RONIN_torch")
    from IMUNet import IMUNet
    net = IMUNet(num_classes=2, input_size=(1,6,200), sampling_rate=200,
                 num_T=32, num_S=64, hidden=64, dropout_rate=0.5).eval()
    return net, (1,6,200)

MODEL_REGISTRY = {
    "ronin_resnet18": _ronin_resnet18,
    "ronin_tcn": _ronin_tcn,
    "ronin_lstm": _ronin_lstm,
    "imunet": _imunet,
}
# merge extra open-source models (mobile CNNs, TLIO, TinyOdom, EqNIO).
# RUN ONE MODEL PER PROCESS (--models <one> --append) to avoid cross-repo module clashes.
try:
    from models_ext import EXT_REGISTRY
    MODEL_REGISTRY.update(EXT_REGISTRY)
except Exception as _e:
    print("  [warn] models_ext not loaded:", _e)

# ---------------- power backends ----------------
def detect_power_backend():
    if shutil.which("tegrastats"): return "tegrastats"
    if shutil.which("nvidia-smi"): return "nvidia-smi"
    return "none"

class PowerSampler:
    """Background power sampler. Returns median steady-state power (W)."""
    def __init__(self, backend):
        self.backend, self.samples, self._stop = backend, [], threading.Event()
    def _read(self):
        try:
            if self.backend == "nvidia-smi":
                o = subprocess.check_output(
                    ["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits","-i","0"]).decode()
                return float(o.splitlines()[0])
            elif self.backend == "tegrastats":
                # parse a POM/VDD power field in mW, e.g. "VDD_IN 5000mW/..." -> take total board if present
                # tegrastats never exits, so check_output always TimeoutExpires; grab its partial output.
                try:
                    o = subprocess.check_output(["tegrastats","--interval","100"], timeout=0.5).decode()
                except subprocess.TimeoutExpired as te:
                    o = te.output.decode() if isinstance(te.output, (bytes, bytearray)) else (te.output or "")
                import re
                m = re.findall(r'(\w*VDD\w*|POM_\w+)\s+(\d+)mW', o)
                if m:  # sum board rails or take VDD_IN if present
                    d = {k:int(v) for k,v in m}
                    tot = d.get("VDD_IN") or sum(v for k,v in d.items())
                    return tot/1000.0
        except Exception:
            pass
        return None
    def _loop(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None: self.samples.append(v)
            time.sleep(0.1)
    def __enter__(self):
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self.t.join(timeout=1)
    def median(self):
        return statistics.median(self.samples) if self.samples else float('nan')

# ---------------- benchmark one config ----------------
def bench(net, in_shape, dev, dtype, iters=500, dur=8.0, power_backend="none"):
    net = net.to(dev).to(dtype); x = torch.randn(*in_shape, device=dev, dtype=dtype)
    cuda = dev.startswith("cuda")
    with torch.no_grad():
        for _ in range(50): net(x)
        if cuda: torch.cuda.synchronize()
        # latency
        ts=[]
        for _ in range(iters):
            if cuda:
                s=torch.cuda.Event(True); e=torch.cuda.Event(True)
                s.record(); net(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
            else:
                t=time.perf_counter(); net(x); ts.append((time.perf_counter()-t)*1000)
        lat_med=statistics.median(ts); lat_p95=sorted(ts)[int(0.95*iters)]
        # throughput + power over `dur` seconds
        with PowerSampler(power_backend) as ps:
            t0=time.perf_counter(); n=0
            while time.perf_counter()-t0 < dur:
                net(x); n+=1
            if cuda: torch.cuda.synchronize()
            dt=time.perf_counter()-t0
            power=ps.median()
    thr=n/dt
    energy=(power/thr*1000) if (thr>0 and power==power) else float('nan')
    # memory
    params=sum(p.numel() for p in net.parameters())
    peak_mb=(torch.cuda.max_memory_allocated(dev)/1e6) if cuda else float('nan')
    return dict(lat_med_ms=round(lat_med,4), lat_p95_ms=round(lat_p95,4),
                throughput_ips=round(thr,1), power_W=round(power,2) if power==power else "",
                energy_mJ_per_inf=round(energy,4) if energy==energy else "",
                params_M=round(params/1e6,3), peak_mem_MB=round(peak_mb,1) if peak_mb==peak_mb else "")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device", required=True, help="label, e.g. blackwell / orin / xaviernx / nano")
    ap.add_argument("--models", default="ronin_resnet18")
    ap.add_argument("--precisions", default="fp32,fp16,bf16")
    ap.add_argument("--cuda-index", default="0")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--power", default="auto", choices=["auto","nvidia-smi","tegrastats","none"])
    ap.add_argument("--append", action="store_true", help="append to --out (write header only if new)")
    ap.add_argument("--out", default=None)
    args=ap.parse_args()

    dev = f"cuda:{args.cuda_index}" if torch.cuda.is_available() else "cpu"
    pb = detect_power_backend() if args.power=="auto" else args.power
    arch = torch.cuda.get_device_name(int(args.cuda_index)) if dev.startswith("cuda") else "cpu"
    boundary = "board" if pb=="tegrastats" else ("card" if pb=="nvidia-smi" else "n/a")
    print(f"device={args.device} arch={arch} power_backend={pb}({boundary}) runtime=pytorch dtype_dev={dev}")

    out = args.out or f"results/profile_{args.device}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows=[]
    dt_map={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}
    for mname in args.models.split(","):
        mname=mname.strip()
        if mname not in MODEL_REGISTRY:
            print(f"  [skip] unknown model {mname}"); continue
        for pname in args.precisions.split(","):
            pname=pname.strip()
            try:
                net,in_shape=MODEL_REGISTRY[mname]()
                if dev.startswith("cuda"): torch.cuda.reset_peak_memory_stats(dev)
                r=bench(net,in_shape,dev,dt_map[pname],args.iters,args.dur,pb)
                row=dict(device=args.device,arch=arch,runtime="pytorch",power_boundary=boundary,
                         model=mname,precision=pname,**r)
                rows.append(row)
                print(f"  [{mname}/{pname}] lat={r['lat_med_ms']}ms thr={r['throughput_ips']}/s "
                      f"P={r['power_W']}W E={r['energy_mJ_per_inf']}mJ mem={r['peak_mem_MB']}MB")
            except Exception as e:
                print(f"  [{mname}/{pname}] FAILED: {e}")
    if rows:
        append = args.append and os.path.exists(out)
        with open(out,"a" if append else "w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            if not append: w.writeheader()
            w.writerows(rows)
        print(f"{'appended' if append else 'wrote'} {out} ({len(rows)} rows)")

if __name__=="__main__":
    main()
