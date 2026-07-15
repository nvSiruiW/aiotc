import sys, time, threading, subprocess, statistics
sys.path.insert(0, "ronin/source")
import torch
from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule

def build_ronin_resnet18():
    cfg = {'fc_dim':512, 'in_dim':7, 'dropout':0.5, 'trans_planes':128}  # window=200 -> in_dim=200//32+1=7
    net = ResNet1D(6, 2, BasicBlock1D, [2,2,2,2], base_plane=64,
                   output_block=FCOutputModule, kernel_size=3, **cfg)
    return net.eval()

def nparams(m): return sum(p.numel() for p in m.parameters())

def bench(net, x, dev, dtype, iters=500):
    net = net.to(dev).to(dtype); x = x.to(dev).to(dtype)
    with torch.no_grad():
        for _ in range(50): net(x)                       # warmup
        torch.cuda.synchronize()
        ts=[]
        for _ in range(iters):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True)
            s.record(); net(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))                  # ms
        lat_med=statistics.median(ts); lat_p95=sorted(ts)[int(0.95*iters)]
        pw=[]; stop=threading.Event()
        def sampler():
            while not stop.is_set():
                try:
                    o=subprocess.check_output(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits","-i","0"]).decode().strip()
                    pw.append(float(o.splitlines()[0]))
                except Exception: pass
                time.sleep(0.1)
        th=threading.Thread(target=sampler); th.start()
        t0=time.perf_counter(); n=0
        while time.perf_counter()-t0 < 8.0:
            net(x); n+=1
        torch.cuda.synchronize(); dt=time.perf_counter()-t0
        stop.set(); th.join()
    thr=n/dt
    power=statistics.median(pw) if pw else float('nan')
    energy=power/thr*1000 if thr>0 else float('nan')      # mJ/inf
    return dict(lat_med_ms=lat_med, lat_p95_ms=lat_p95, throughput=thr, power_W=power, energy_mJ=energy)

dev="cuda:0"
net=build_ronin_resnet18(); x=torch.randn(1,6,200)
print(f"RoNIN-ResNet18: params={nparams(net)/1e6:.2f}M  input=(1,6,200)  device={torch.cuda.get_device_name(0)}")
for dt,name in [(torch.float32,"fp32"),(torch.float16,"fp16")]:
    r=bench(build_ronin_resnet18(), x, dev, dt)
    print(f"  {name}: lat_med={r['lat_med_ms']:.3f}ms  P95={r['lat_p95_ms']:.3f}ms  "
          f"throughput={r['throughput']:.0f}/s  power={r['power_W']:.1f}W  energy={r['energy_mJ']:.3f}mJ/inf")
