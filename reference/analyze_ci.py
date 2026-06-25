"""Perplexity + bootstrap 95% CIs from cached per-token NLLs.

Paired token-level bootstrap vs the `ref` config (same resampled token indices),
1000 resamples. (Token-level resampling ignores within-sequence correlation; with
a larger corpus, switch to chunk-level — sent.npy carries the grouping.)
"""
import os, sys, numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "_ppl"
B = 1000
rng = np.random.default_rng(0)
ref = np.load(os.path.join(D, "ref.npy"))
n = ref.size
idx = rng.integers(0, n, size=(B, n))
ref_boot = np.exp(ref[idx].mean(1))               # [B]
ref_ppl = float(np.exp(ref.mean()))

order = ["ref", "pv8", "pv4", "pv3", "pv2", "pv_asym3", "pv_asym2",
         "pc4", "pc3", "pc2", "pc_asym3", "pc_asym2", "pc_asym2_prerope",
         "act_absmean", "act_median"]
print(f"corpus dir={D}  tokens={n}  REF ppl={ref_ppl:.3f}")
print(f"{'config':20s} {'ppl':>10s}  {'95% CI':>20s}  {'dPPL':>10s}  {'dPPL 95% CI':>20s}")
for name in order:
    f = os.path.join(D, name + ".npy")
    if not os.path.exists(f):
        continue
    a = np.load(f)
    if a.size != n:
        print(f"{name:20s}  (skip: {a.size} tok != {n})"); continue
    ppl = float(np.exp(a.mean()))
    boot = np.exp(a[idx].mean(1))                  # [B], same idx as ref (paired)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    d = boot - ref_boot
    dlo, dhi = np.percentile(d, [2.5, 97.5])
    print(f"{name:20s} {ppl:10.3f}  [{lo:8.3f},{hi:8.3f}]  {ppl-ref_ppl:+10.3f}  [{dlo:+8.3f},{dhi:+8.3f}]")
