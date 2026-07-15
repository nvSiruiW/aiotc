#!/usr/bin/env python3
"""Idle/active dynamic-power protocol with repeats + full provenance.

Per model+precision: run N independent trials; each trial samples board power at
~10 Hz for `dur` s at IDLE (no GPU work) and for `dur` s under a sustained inference
loop (ACTIVE). Reports mean idle/active/dynamic power, throughput, total and
idle-subtracted DYNAMIC energy per inference, the per-trial std of dynamic energy,
the number of trials, per-trial duration and inference count, plus the GPU clock,
temperature and power-limit at start. One model per process; use --append.

Run ONLY on an idle, single GPU (no other load). Power is a card-level nvidia-smi
figure (not wall-plug). Sampling: nvidia-smi power.draw at ~10 Hz, aggregated by median.
"""
import argparse, os, sys, time, csv, subprocess, statistics
sys.path.insert(0, "scripts"); sys.path.insert(0, "ronin/source")
import torch
from profile_device import MODEL_REGISTRY, PowerSampler, detect_power_backend

DT = {"fp32":torch.float32, "fp16":torch.float16, "bf16":torch.bfloat16}

def gpu_static(cuda):
    try:
        o=subprocess.check_output(["nvidia-smi",
            "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.limit,persistence_mode",
            "--format=csv,noheader,nounits","-i",str(cuda)]).decode().strip().split(",")
        return dict(sm_MHz=o[0].strip(), sm_max_MHz=o[1].strip(), temp_C=o[2].strip(),
                    power_limit_W=o[3].strip(), persistence=o[4].strip())
    except Exception:
        return dict(sm_MHz="?", sm_max_MHz="?", temp_C="?", power_limit_W="?", persistence="?")

def sample_idle(backend, dur):
    with PowerSampler(backend) as ps:
        t0=time.time()
        while time.time()-t0 < dur: time.sleep(0.05)
    return ps.median()

def sample_active(net, x, backend, dur):
    with torch.no_grad():
        for _ in range(30): net(x)
        if x.is_cuda: torch.cuda.synchronize()
        with PowerSampler(backend) as ps:
            t0=time.time(); n=0
            while time.time()-t0 < dur:
                net(x); n+=1
            if x.is_cuda: torch.cuda.synchronize()
            dt=time.time()-t0
    return ps.median(), n, dt

def _tegrastats_line():
    """One tegrastats line. `timeout` kills tegrastats and exits 124, so check_output
    raises CalledProcessError — its .output still holds the captured line(s)."""
    try:
        out=subprocess.check_output(["timeout","2","tegrastats"],stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        out=e.output or b""
    except Exception:
        return ""
    txt=out.decode() if isinstance(out,(bytes,bytearray)) else (out or "")
    lines=[l for l in txt.splitlines() if l.strip()]
    return lines[-1] if lines else ""

def board_static(cuda):
    """Jetson provenance: nvpmodel power mode + tegrastats GPU freq/temperature."""
    d=dict(sm_MHz="?", sm_max_MHz="?", temp_C="?", power_limit_W="board", persistence="jetson")
    try:
        q=subprocess.check_output(["nvpmodel","-q"],stderr=subprocess.STDOUT).decode()
        for ln in q.splitlines():
            if "NV Power Mode" in ln or ln.strip().isdigit(): d["power_limit_W"]="nvpmodel:"+ln.strip().split(":")[-1].strip()
    except Exception: pass
    try:
        import re
        s=_tegrastats_line()
        mg=re.search(r"GR3D_FREQ \d+%@?(\d+)?", s); mt=re.search(r"(gpu|GPU)@([\d.]+)C", s)
        if mg and mg.group(1): d["sm_MHz"]=mg.group(1)
        if mt: d["temp_C"]=mt.group(2)
    except Exception: pass
    return d

def read_temp(backend, cuda="0"):
    try:
        if backend=="nvidia-smi":
            return float(subprocess.check_output(["nvidia-smi","--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits","-i",str(cuda)]).decode().strip())
        import re
        s=_tegrastats_line()
        m=re.search(r"[gG][pP][uU]@([\d.]+)C", s); return float(m.group(1)) if m else float("nan")
    except Exception: return float("nan")

def wait_thermal_steady(net, x, backend, cuda="0", warmup=90, window=15, slope_thr=0.15):
    """Run sustained load until GPU temp plateaus (|slope|<slope_thr C/s over `window`s)
    or `warmup`s elapse. Returns final temp. Guards against thermal-throttling drift."""
    import torch as _t
    with _t.no_grad():
        t0=time.time(); temps=[]
        while time.time()-t0 < warmup:
            for _ in range(50): net(x)
            if x.is_cuda: _t.cuda.synchronize()
            temps.append((time.time()-t0, read_temp(backend, cuda)))
            recent=[T for (tt,T) in temps if tt>=temps[-1][0]-window and T==T]
            if len(recent)>=4 and max(recent)-min(recent) < slope_thr*window:
                break
    return temps[-1][1] if temps else float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--precisions", default="fp32,fp16,bf16")
    ap.add_argument("--dur", type=float, default=12.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--thermal_warmup", type=float, default=0.0,
                    help="seconds of sustained load to reach thermal steady state before measuring (Jetson: use 90)")
    ap.add_argument("--cuda", default="0"); ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args=ap.parse_args()
    dev=f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    pb=detect_power_backend(); boundary="board" if pb=="tegrastats" else ("card" if pb=="nvidia-smi" else "n/a")
    arch=torch.cuda.get_device_name(int(args.cuda)) if dev.startswith("cuda") else "cpu"
    gs=board_static(args.cuda) if boundary=="board" else gpu_static(args.cuda)
    # thermal steady-state: warm up under load until temp plateaus (prevents throttling drift)
    temp_start=read_temp(pb, args.cuda)
    if args.thermal_warmup>0:
        _net,_shp=MODEL_REGISTRY[args.model](); _net=_net.to(dev).to(DT[args.precisions.split(',')[0].strip()]).eval()
        _x=torch.randn(*_shp,device=dev,dtype=DT[args.precisions.split(',')[0].strip()])
        tp=wait_thermal_steady(_net,_x,pb,args.cuda,warmup=args.thermal_warmup)
        print(f"[thermal] warmed to steady state: {temp_start}C -> {tp}C", flush=True)
    idle=statistics.mean([sample_idle(pb, args.dur) for _ in range(args.runs)])   # idle baseline, N trials
    print(f"[{args.model}] idle={idle:.2f}W sm={gs['sm_MHz']}/{gs['sm_max_MHz']}MHz "
          f"T={gs['temp_C']}C Plim={gs['power_limit_W']}W ({boundary}, {args.runs}x{args.dur:.0f}s)", flush=True)
    rows=[]
    for pn in args.precisions.split(","):
        pn=pn.strip()
        try:
            acts, thrs, dyn_es, ninfs = [], [], [], []
            for _ in range(args.runs):
                net,shp=MODEL_REGISTRY[args.model](); net=net.to(dev).to(DT[pn]).eval()
                x=torch.randn(*shp,device=dev,dtype=DT[pn])
                a,n,dt=sample_active(net,x,pb,args.dur); thr=n/dt
                acts.append(a); thrs.append(thr); ninfs.append(n)
                dyn_es.append((a-idle)/thr*1000 if thr>0 else float('nan'))
            active=statistics.mean(acts); thr=statistics.mean(thrs); dyn=active-idle
            tot_e=active/thr*1000; dyn_e=statistics.mean(dyn_es)
            dyn_std=statistics.pstdev(dyn_es) if len(dyn_es)>1 else 0.0
            rows.append(dict(device=args.device,arch=arch,power_boundary=boundary,model=args.model,precision=pn,
                idle_W=round(idle,2),active_W=round(active,2),dynamic_W=round(dyn,2),throughput_ips=round(thr,1),
                total_energy_mJ=round(tot_e,2),dynamic_energy_mJ=round(dyn_e,2),dynamic_energy_std_mJ=round(dyn_std,2),
                runs=args.runs,dur_s=args.dur,infer_per_run=round(statistics.mean(ninfs)),
                sm_MHz=gs['sm_MHz'],temp_start_C=temp_start,temp_end_C=read_temp(pb,args.cuda),
                power_mode=gs['power_limit_W'],power_limit_W=gs['power_limit_W']))
            print(f"  [{args.model}/{pn}] active={active:.1f}W dyn={dyn:.1f}W thr={thr:.0f}/s "
                  f"E_tot={tot_e:.1f} E_dyn={dyn_e:.1f}±{dyn_std:.1f} mJ ({args.runs} runs)", flush=True)
        except Exception as e:
            print(f"  [{args.model}/{pn}] FAILED: {e}", flush=True)
    if rows:
        append=args.append and os.path.exists(args.out); os.makedirs(os.path.dirname(args.out),exist_ok=True)
        with open(args.out,"a" if append else "w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            if not append: w.writeheader()
            w.writerows(rows)
        print(f"{'appended' if append else 'wrote'} {args.out}", flush=True)

if __name__=="__main__":
    main()
