"""Emit LaTeX result tables straight from cached per-token NLL arrays + bootstrap.

Writes paper-ready \\input fragments so local full-corpus runs flow into the paper
with no manual editing:

  python3 gen_table.py            # regenerates ../paper/results_kv.tex (+ ablation)

Define which dirs/configs go in each table in TABLES below.
"""
import os, numpy as np

B = 2000
PRETTY = {
    "ref": "reference (absmax, KV float)",
    "pv8": "per-vector 8-bit",
    "pv4": "per-vector 4-bit",
    "pv3": "per-vector 3-bit",
    "pv2": "per-vector 2-bit (sym)",
    "pv_asym2": "per-vector 2-bit (asym)",
    "pc2": "per-channel 2-bit (sym)",
    "pc_asym2": "per-channel 2-bit (asym)",
    "pc_asym2_prerope": "\\quad + pre-RoPE Keys",
    "act_absmean": "absmean activation scale",
}


def stats(d, order):
    ref = np.load(os.path.join(d, "ref.npy"))
    n = ref.size
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(B, n))
    refb = np.exp(ref[idx].mean(1))
    rows = []
    for name in order:
        p = os.path.join(d, name + ".npy")
        if not os.path.exists(p):
            continue
        a = np.load(p)
        if a.size != n:
            continue
        ppl = float(np.exp(a.mean()))
        boot = np.exp(a[idx].mean(1))
        d2 = boot - refb
        dlo, dhi = np.percentile(d2, [2.5, 97.5])
        rows.append((name, ppl, ppl - float(np.exp(ref.mean())), dlo, dhi, n))
    return rows


def fmt_ppl(x):
    return f"{x:,.1f}" if x < 1e4 else f"{x:.2g}"


def emit(dir_, order, caption, label, out):
    rows = stats(dir_, order)
    n = rows[0][5] if rows else 0
    lines = [
        "\\begin{table}[t]\\centering\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Configuration & PPL & $\\Delta$PPL & 95\\% CI on $\\Delta$PPL\\\\",
        "\\midrule",
    ]
    for name, ppl, dppl, dlo, dhi, _ in rows:
        pretty = PRETTY.get(name, name)
        bold = "\\textbf{" if name == "pc_asym2" else ""
        endb = "}" if bold else ""
        lines.append(
            f"{pretty} & {fmt_ppl(ppl)} & {bold}{dppl:+.2f}{endb} & "
            f"$[{dlo:+.2f},\\,{dhi:+.2f}]$\\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\\\[2pt]{{\\footnotesize {n} tokens; paired token-level bootstrap, "
              f"{B} resamples.}}", "\\end{table}"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out, f"({len(rows)} rows, {n} tokens)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    paper = os.path.join(here, "..", "paper")
    order = ["ref", "pv8", "pv2", "pc2", "pc_asym2", "pc_asym2_prerope", "act_absmean"]
    emit("_ppl_wiki", order,
         "Integer-path perplexity on a WikiText-2 test slice. Int8 KV is free; "
         "2-bit per-vector KV collapses while 2-bit per-channel asymmetric KV is "
         "nearly recovered; robust activation scaling is catastrophic.",
         "tab:wiki", os.path.join(paper, "results_kv.tex"))
    order2 = ["pv2", "pv_asym2", "pc2", "pc_asym2"]
    emit("_ppl", order2,
         "2-bit KV-cache ablation on the mini-corpus: per-channel granularity is "
         "necessary, and the zero-point (asymmetry) helps only on top of it.",
         "tab:ablation", os.path.join(paper, "results_ablation.tex"))
