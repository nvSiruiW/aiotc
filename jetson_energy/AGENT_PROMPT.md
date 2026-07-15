# Task: rigorous dynamic-energy measurement on this Jetson

You are an autonomous engineering agent on an NVIDIA Jetson that already ran this PDR
edge benchmark (you have the repo, the trained models, and a working PyTorch that can
build all 10 backbones — you produced `results/profile_<label>.csv` before). We are now
adding a **publication-grade energy measurement** so a reviewer cannot dismiss it.

## Why (the one thing that matters)
Whole-board (tegrastats) power is NOT comparable to the datacenter card (nvidia-smi).
We fix this with **idle-subtraction**: for each model we measure idle board power and
active board power, and report the MARGINAL dynamic energy per inference
`E_dyn = (P_active − P_idle) / throughput`. That cancels the board's static draw and is
comparable across devices. Your job: produce this under **locked clocks + thermal
steady state**, with full provenance.

## Ground rules
1. Never fabricate numbers. Report actual measured values. If something fails, paste the
   exact error and adapt; do not skip a model silently.
2. Lock the board and RECORD the mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
   If you cannot sudo, say so explicitly in your report (it changes interpretation).
3. Measurements must start from thermal steady state (the script warms up 60 s per model
   until GPU temp plateaus) — do not disable this.

## Steps
1. `git fetch origin && git checkout <branch>` (the branch this file is on), then `git pull`.
2. Pick your label: `orin` for AGX Orin, `orin_nano` for Orin Nano.
3. Run:  `bash jetson_energy/measure_energy_jetson.sh <label>`
   - It locks clocks, then for all 10 backbones × {fp32,fp16,bf16} measures idle/active
     board power (3 trials each) and writes `results/power_dynamic_<label>.csv`.
   - Total run is ~40–70 min (thermal warm-up dominates). Let it finish.
4. If `python3` can't import a model builder, use the SAME environment you used for
   `results/profile_<label>.csv` (same venv / PYTHONPATH). The builders are identical.

## Deliverables (report back + commit)
- `results/power_dynamic_<label>.csv` — paste it, and commit + push to the branch:
  `git add results/power_dynamic_<label>.csv && git commit -m "dynamic energy <label>" && git push`
- Confirm the provenance sanity checks the script prints:
  * `temp_start_C` ≈ `temp_end_C` (no throttling drift). If they diverge >5 °C, say so.
  * `power_mode` column shows your real nvpmodel mode (not `?`).
  * `dynamic_energy_std_mJ` is small vs `dynamic_energy_mJ` (stable).
- State: device model, JetPack / L4T version, nvpmodel mode, whether jetson_clocks locked,
  and the steady-state temperature you measured.

Start by checking out the branch and reading `jetson_energy/measure_energy_jetson.sh`.
