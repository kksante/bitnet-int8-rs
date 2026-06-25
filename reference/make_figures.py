"""Generate publication figures (PDF + PNG) into figures/, reading measured
per-token NLL arrays from _ppl/ and _ppl_wiki/ so the figures regenerate
automatically from whatever runs are present (local full-corpus runs included)."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})


def ppl(d, name):
    p = os.path.join(d, name + ".npy")
    return float(np.exp(np.load(p).mean())) if os.path.exists(p) else None


def dppl_ci(d, name, B=2000):
    ref = np.load(os.path.join(d, "ref.npy")); n = ref.size
    rng = np.random.default_rng(0); idx = rng.integers(0, n, size=(B, n))
    refb = np.exp(ref[idx].mean(1))
    a = np.load(os.path.join(d, name + ".npy"))
    boot = np.exp(a[idx].mean(1)) - refb
    return float(np.exp(a.mean()) - np.exp(ref.mean())), np.percentile(boot, [2.5, 97.5])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# Headline KV results are on the 10,240-token WikiText-2 set (run_ppl2.py output,
# analyze_ci.py). Hardcoded here so the figures match Tables 1/6 exactly. ΔPPL and
# 95% CIs as measured (ref PPL 27.422).
WIKI10K = {  # name -> (dppl, lo, hi)
    "pv8": (0.006, -0.039, 0.050),
    "pv4": (0.084, -0.070, 0.247),
    "pv3": (1.284, 0.961, 1.582),
    "pv2": (265.731, 247.875, 283.665),
    "pc4": (0.245, 0.126, 0.378),
    "pc3": (1.112, 0.824, 1.400),
    "pc2": (264.048, 246.235, 282.720),
    "pc_asym3": (0.858, 0.645, 1.070),
    "pc_asym2": (9.961, 9.107, 10.873),
    "pc_asym2_prerope": (8.172, 7.430, 8.980),
}

# ---- Fig 1: KV ΔPPL vs bit-width by granularity (WikiText-2, 10k tokens) ----
series = {
    "per-vector (sym)":  ("#c0392b", "o", {8: "pv8", 4: "pv4", 3: "pv3", 2: "pv2"}),
    "per-channel (sym)": ("#e67e22", "s", {4: "pc4", 3: "pc3", 2: "pc2"}),
    "per-channel (asym)":("#27ae60", "^", {3: "pc_asym3", 2: "pc_asym2"}),
}
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for lbl, (col, mk, m) in series.items():
    xs = sorted(m, reverse=True)
    ys = [WIKI10K[m[b]][0] for b in xs]
    ax.plot(xs, ys, mk + "-", color=col, label=lbl)
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_yscale("symlog", linthresh=1)
ax.set_xticks([8, 4, 3, 2]); ax.invert_xaxis()
ax.set_xlabel("KV-cache bits"); ax.set_ylabel(r"$\Delta$PPL vs reference (symlog)")
ax.set_title("KV on WikiText-2: only asymmetric survives 2-bit")
ax.legend(frameon=False, fontsize=9)
save(fig, "fig_kv_granularity")

# ---- Fig 1b: WikiText-2 headline ΔPPL with bootstrap CIs (10k tokens) ----
order = [("pv8", "int8 KV"), ("pc_asym2", "2-bit per-ch asym"),
         ("pc_asym2_prerope", "  + pre-RoPE"), ("pc2", "2-bit per-ch sym"),
         ("pv2", "2-bit per-vec")]
labels, vals, los, his = [], [], [], []
for nm, lab in order:
    d, lo, hi = WIKI10K[nm]
    labels.append(lab); vals.append(d); los.append(d - lo); his.append(hi - d)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
y = np.arange(len(labels))
ax.barh(y, vals, xerr=[los, his], color="#2980b9", alpha=0.85, capsize=3)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xscale("symlog", linthresh=1)
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel(r"$\Delta$PPL vs reference (WikiText-2, 10k tok, 95\% CI)")
ax.set_title("KV quantization on WikiText-2")
save(fig, "fig_kv_wikitext")

# ---- Fig 2: mechanism ----
if os.path.exists("_ppl/mech.npz"):
    m = np.load("_ppl/mech.npz"); L = np.arange(len(m["k_bias"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    a1.plot(L, m["k_bias"], "o-", color="#c0392b", ms=3, label="Key")
    a1.plot(L, m["v_bias"], "s-", color="#2980b9", ms=3, label="Value")
    a1.set_xlabel("layer"); a1.set_ylabel(r"per-channel bias $|\mu_c|/\sigma_c$")
    a1.set_title("KV cache: Key channels are off-center"); a1.legend(frameon=False, fontsize=9)
    a2.plot(L, m["act_ratio"], "o-", color="#8e44ad", ms=3)
    a2.set_xlabel("layer"); a2.set_ylabel("activation absmax / absmean")
    a2.set_title("Activation outliers across depth")
    save(fig, "fig_mechanism")

# ---- Fig 3: activation recovery curve ----
mult = [2, 4, 8, 16, 32, 64]; pplv = [4.233e7, 3.134e4, 4165, 270.7, 55.76, 23.28]; ref = 8.269
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(mult, pplv, "o-", color="#c0392b")
ax.axhline(ref, color="#27ae60", ls="--", label=f"absmax reference ({ref:.1f})")
ax.set_yscale("log"); ax.set_xscale("log", base=2)
ax.set_xticks(mult); ax.set_xticklabels(mult)
ax.set_xlabel(r"absmean clip multiplier $k$  (scale $=k\cdot$mean$|x|$)")
ax.set_ylabel("perplexity (log)")
ax.set_title("Robust activation scaling recovers only as it stops clipping")
ax.legend(frameon=False, fontsize=9)
save(fig, "fig_act_recovery")

# ---- Fig 4: sub-int8 activations, naive vs post-hoc rotation ----
abits = [8, 6, 4, 3, 2, 1]
naive = [8.184, 8.856, 12.308, 1877.205, 3.169e7, 1.306e6]
rot = [8.265, 8.154, 9.313, 14.090, 1.042e6, 2.970e5]
best = [8.27, 8.27, 8.41, 9.24, 10.21, 5.0e4]   # rotation + per-group (+ zero-point at 2b)
refb = 8.269
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(abits, naive, "o-", color="#c0392b", label="naive (per-token absmax)")
ax.plot(abits, rot, "^-", color="#e67e22", label="+ rotation (per-token)")
ax.plot(abits, best, "s-", color="#27ae60", label="+ per-group + zero-point")
ax.axhline(refb, color="k", ls=":", lw=0.8, label=f"native int8 ({refb:.1f})")
ax.set_yscale("log"); ax.set_xticks(abits); ax.invert_xaxis()
ax.set_xlabel("activation bits"); ax.set_ylabel("perplexity (log)")
ax.set_title("Sub-int8 activations on the shipped model (no retraining)")
ax.legend(frameon=False, fontsize=9)
save(fig, "fig_actbits")

print("figures:", sorted(os.listdir("figures")))
