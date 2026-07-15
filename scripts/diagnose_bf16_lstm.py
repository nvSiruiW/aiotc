#!/usr/bin/env python3
"""Rigorous diagnosis of the BF16 RoNIN-LSTM slowdown observation.

Goal: decide whether the ~3x BF16 vs FP16 latency gap is
  (a) a genuine kernel-path phenomenon, or
  (b) an experimental artifact (non-contiguous LSTM weights / flatten_parameters,
      dtype mismatch, timing overhead).

Runs sanity checks + Layer1 (end-to-end) + Layer2 (minimal nn.LSTM sweep)
+ Layer3 (profiler kernel timeline). Prints everything; no strong claims.
"""
import sys, statistics, json
sys.path.insert(0, "ronin/source")
import torch, torch.nn as nn

DEV = "cuda:0"
DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def qstats(ms):
    ms = sorted(ms); n = len(ms)
    med = statistics.median(ms)
    p95 = ms[min(n-1, int(0.95*n))]
    q1, q3 = ms[n//4], ms[(3*n)//4]
    return dict(median=round(med,4), p95=round(p95,4), iqr=round(q3-q1,4),
                min=round(ms[0],4), max=round(ms[-1],4))

def time_cuda(forward, iters=200, warmup=30):
    """Median-based per-call latency (ms) using CUDA events with sync."""
    with torch.no_grad():
        for _ in range(warmup): forward()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); forward(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return ts

# ============================ 0. ENVIRONMENT ============================
def env_probe():
    print("="*70); print("0. ENVIRONMENT / SCOPE")
    print(f"  torch            {torch.__version__}")
    print(f"  cuda (torch)     {torch.version.cuda}")
    print(f"  cudnn            {torch.backends.cudnn.version()}")
    print(f"  cudnn.enabled    {torch.backends.cudnn.enabled}")
    print(f"  device           {torch.cuda.get_device_name(0)}")
    cc = torch.cuda.get_device_capability(0)
    print(f"  compute cap.     {cc[0]}.{cc[1]}")
    print(f"  is_bf16_supported {torch.cuda.is_bf16_supported()}")
    print(f"  autocast enabled (cuda) {torch.is_autocast_enabled()}")

# ============================ RoNIN-LSTM builder =======================
def build_ronin_lstm(dtype, flatten):
    import types
    from model_temporal import LSTMSeqNetwork
    net = LSTMSeqNetwork(6, 2, 1, torch.device(DEV), lstm_size=100, lstm_layers=3, dropout=0).eval()
    def init_weights(self):
        p = next(self.parameters())
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size, device=p.device, dtype=p.dtype)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size, device=p.device, dtype=p.dtype)
        return (h0, c0)
    net.init_weights = types.MethodType(init_weights, net)
    net = net.to(DEV).to(dtype)
    if flatten:
        net.lstm.flatten_parameters()
    return net

# ============================ 1. SANITY CHECKS ========================
def sanity():
    print("="*70); print("1. SANITY CHECKS (dtype plumbing, per precision)")
    for pn, dt in DT.items():
        net = build_ronin_lstm(dt, flatten=True)
        x = torch.randn(1, 200, 6, device=DEV, dtype=dt)
        with torch.no_grad(): y = net(x)
        pd = {p.dtype for p in net.parameters()}
        bd = {b.dtype for b in net.buffers()} or {"(none)"}
        hh = net.init_weights()
        print(f"  [{pn}] param={ [str(d) for d in pd] } buf={ [str(d) for d in bd] } "
              f"in={x.dtype} out={y.dtype} hidden={hh[0].dtype} "
              f"out_finite={torch.isfinite(y).all().item()}")

# ============================ 2. LAYER 1 ==============================
def layer1(runs=5, iters=200, warmup=30):
    print("="*70); print(f"2. LAYER 1  RoNIN-LSTM end-to-end  ({runs} indep. runs x {iters} iters)")
    print(f"  cfg: nn.LSTM(6,100,3), batch=1, seq=200, hidden=100, layers=3")
    print(f"  {'prec':5} {'flatten':8} {'median':>9} {'p95':>8} {'iqr':>7} {'run-to-run medians'}")
    summary = {}
    for flatten in (False, True):
        for pn, dt in DT.items():
            run_meds = []
            allms = []
            for r in range(runs):
                net = build_ronin_lstm(dt, flatten=flatten)
                x = torch.randn(1, 200, 6, device=DEV, dtype=dt)
                ms = time_cuda(lambda: net(x), iters=iters, warmup=warmup)
                run_meds.append(statistics.median(ms)); allms += ms
            st = qstats(allms)
            summary[(pn, flatten)] = st["median"]
            rm = " ".join(f"{m:.3f}" for m in run_meds)
            print(f"  {pn:5} {str(flatten):8} {st['median']:9.3f} {st['p95']:8.3f} {st['iqr']:7.3f}  [{rm}]")
    print("  --- ratios vs FP16 (same flatten) ---")
    for flatten in (False, True):
        f16 = summary[("fp16", flatten)]
        for pn in ("fp32","bf16"):
            print(f"    flatten={flatten}: {pn}/fp16 = {summary[(pn,flatten)]/f16:.2f}x")
    return summary

# ============================ 3. LAYER 2 ==============================
class PureLSTM(nn.Module):
    """Minimal: only nn.LSTM + 2 linears, hidden passed as None (canonical)."""
    def __init__(self, in_size=6, hidden=100, layers=3, out=2):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden, layers, batch_first=True)
        self.l1 = nn.Linear(hidden, out*5); self.l2 = nn.Linear(out*5, out)
    def forward(self, x):
        o,_ = self.lstm(x)          # hx=None -> internal zero states in correct dtype
        return self.l2(self.l1(o))

def build_pure(dtype, hidden, layers, flatten=True):
    net = PureLSTM(6, hidden, layers, 2).eval().to(DEV).to(dtype)
    if flatten: net.lstm.flatten_parameters()
    return net

def layer2(iters=100, warmup=20):
    print("="*70); print("3. LAYER 2  minimal nn.LSTM microbenchmark (flatten_parameters ON)")
    print("   isolates: RoNIN-specific?  shape-dependent?  generic LSTM op?")
    # exact-match config first, then sweep
    configs = []
    # exact RoNIN match
    configs.append((1,200,100,3))
    # sweeps around it (vary one axis at a time to keep it small)
    for b in (1,8): configs.append((b,200,100,3))
    for s in (100,200,400): configs.append((1,s,100,3))
    for h in (50,100,200): configs.append((1,200,h,3))
    seen=set()
    print(f"  {'batch':>5} {'seq':>4} {'hid':>4} {'lyr':>3} | {'fp32':>8} {'fp16':>8} {'bf16':>8} | bf16/fp16")
    for (b,s,h,l) in configs:
        key=(b,s,h,l)
        if key in seen: continue
        seen.add(key)
        lat={}
        for pn,dt in DT.items():
            net=build_pure(dt,h,l); x=torch.randn(b,s,6,device=DEV,dtype=dt)
            lat[pn]=statistics.median(time_cuda(lambda: net(x),iters=iters,warmup=warmup))
        ratio=lat["bf16"]/lat["fp16"]
        flag=" <== >2x" if ratio>2 else ""
        print(f"  {b:5} {s:4} {h:4} {l:3} | {lat['fp32']:8.3f} {lat['fp16']:8.3f} {lat['bf16']:8.3f} | {ratio:6.2f}x{flag}")

# ============================ 4. LAYER 3 ==============================
def _cuda_us(e):
    for a in ("self_device_time_total","self_cuda_time_total","device_time_total","cuda_time_total"):
        v=getattr(e,a,None)
        if v: return float(v)
    return 0.0

def layer3(iters=50, hidden=100):
    print("="*70); print(f"4. LAYER 3  profiler kernel timeline (minimal LSTM, b=1 s=200 h={hidden} l=3)")
    from torch.profiler import profile, ProfilerActivity
    stats={}
    for pn in ("fp16","bf16"):
        dt=DT[pn]; net=build_pure(dt,hidden,3); x=torch.randn(1,200,6,device=DEV,dtype=dt)
        with torch.no_grad():
            for _ in range(20): net(x)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                for _ in range(iters): net(x)
                torch.cuda.synchronize()
        evs=[e for e in prof.key_averages() if _cuda_us(e)>0]
        evs=sorted(evs,key=_cuda_us,reverse=True)
        n_launch=sum(int(e.count) for e in evs)
        tot=sum(_cuda_us(e) for e in evs)
        stats[pn]=(len(evs),n_launch,tot)
        print(f"  --- {pn}: {len(evs)} distinct CUDA kernels, {n_launch} launches, {tot/1000:.2f} ms CUDA / {iters} iters "
              f"({tot/1000/iters:.3f} ms/inf) ---")
        cast=[e for e in evs if any(k in e.key.lower() for k in ('cast','copy','convert','contiguous','transpose','_to_'))]
        rnn=[e for e in evs if any(k in e.key.lower() for k in ('rnn','lstm','cudnn'))]
        print(f"      cudnn/rnn/lstm kernels : {[e.key[:40] for e in rnn][:3] or 'NONE'}")
        print(f"      cast/copy/convert      : {sum(int(e.count) for e in cast)} launches "
              f"{[e.key[:34] for e in cast][:3] or 'none'}")
        for e in evs[:9]:
            print(f"      {_cuda_us(e)/1000:8.3f}ms  x{int(e.count):<5} {e.key[:56]}")
    if "fp16" in stats and "bf16" in stats:
        print(f"  --- launches bf16/fp16 = {stats['bf16'][1]/max(1,stats['fp16'][1]):.2f}x , "
              f"distinct kernels fp16={stats['fp16'][0]} bf16={stats['bf16'][0]} ---")

def layer4_cudnn_toggle(iters=200, warmup=30):
    print("="*70); print("5. LAYER 4a  cuDNN on/off sensitivity (minimal LSTM, b=1 s=200 h=100 l=3)")
    print("   if disabling cuDNN removes the gap -> the fp16-only persistent cuDNN kernel is the cause")
    for enabled in (True, False):
        torch.backends.cudnn.enabled = enabled
        row={}
        for pn,dt in DT.items():
            net=build_pure(dt,100,3); x=torch.randn(1,200,6,device=DEV,dtype=dt)
            row[pn]=statistics.median(time_cuda(lambda: net(x),iters=iters,warmup=warmup))
        print(f"  cudnn.enabled={str(enabled):5}: fp32={row['fp32']:.3f} fp16={row['fp16']:.3f} "
              f"bf16={row['bf16']:.3f} ms | bf16/fp16={row['bf16']/row['fp16']:.2f}x")
    torch.backends.cudnn.enabled = True

def env_line():
    """One-line env stamp for cross-version (Layer 4b) comparison."""
    cc=torch.cuda.get_device_capability(0)
    print(f"STACK torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} "
          f"cc={cc[0]}.{cc[1]} dev={torch.cuda.get_device_name(0)}")

if __name__=="__main__":
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=False
    which = sys.argv[1] if len(sys.argv)>1 else "all"
    if which=="l4":
        env_line(); layer4_cudnn_toggle(); print("="*70); print("DONE."); sys.exit(0)
    if which=="l4b":   # minimal cross-version probe: env + exact-match latency only
        env_line()
        for pn,dt in DT.items():
            net=build_pure(dt,100,3); x=torch.randn(1,200,6,device=DEV,dtype=dt)
            m=statistics.median(time_cuda(lambda: net(x),iters=200,warmup=30))
            print(f"  h100 {pn}: {m:.3f} ms")
        print("DONE."); sys.exit(0)
    if which in ("all","env"): env_probe()
    if which in ("all","sanity"): sanity()
    if which in ("all","l1"): layer1()
    if which in ("all","l2"): layer2()
    if which in ("all","l3"):
        layer3(hidden=100)   # 3x-gap regime
        layer3(hidden=200)   # ~1x regime (controlled contrast)
    print("="*70); print("DONE.")
