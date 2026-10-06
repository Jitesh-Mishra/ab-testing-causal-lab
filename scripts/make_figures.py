"""reports/figures/{effects,peeking,iv}-{light,dark}.png from reports/ (run after python -m abx.readout)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP, OUT = ROOT / "reports", ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
itt = pd.read_csv(REP / "itt.csv")
pk = pd.read_csv(REP / "peeking.csv")
R = json.loads((REP / "readout.json").read_text())
THEMES = {"light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6e9ed",
                        cols=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]),
          "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#21262d",
                       cols=["#3987e5", "#d95926", "#199e70", "#c98500", "#9085e9"])}
EST = [("unadjusted", "Difference in means"), ("linear_features", "Linear on 12 features (Lin)"),
       ("cupac_lgbm", "CUPAC (LightGBM prediction)"), ("linear_plus_cupac", "Linear + CUPAC"),
       ("aipw_doubly_robust", "AIPW doubly robust")]


def style(ax, th):
    ax.set_facecolor(th["bg"]); ax.grid(axis="x", color=th["grid"], lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


for theme, th in THEMES.items():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": th["ink2"], "axes.labelcolor": th["ink"],
                         "xtick.color": th["ink2"], "ytick.color": th["ink2"], "text.color": th["ink"]})
    # 1. effects with CIs
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), facecolor=th["bg"], sharey=True)
    for a_ in axes: a_.set_axisbelow(True)
    for ax, y in zip(axes, ["visit", "conversion"]):
        g = itt[itt.outcome == y].set_index("estimator")
        base = R["itt"][y]["unadjusted"]["control_mean"]
        for i, (k, lab) in enumerate(EST):
            r = g.loc[k]
            ax.errorbar(100 * r.estimate / base, i, xerr=[[100 * (r.estimate - r.ci_low) / base], [100 * (r.ci_high - r.estimate) / base]],
                        fmt="o", color=th["cols"][0] if k == "unadjusted" else th["cols"][2], ms=6, capsize=3, lw=2)
            vr = r.variance_reduction
            lab2 = "" if k == "unadjusted" else (f"variance −{vr:.0%}" if vr >= 0 else f"variance +{-vr:.0%}")
            ax.text(100 * r.ci_high / base + 0.6, i, lab2,
                    va="center", fontsize=8.5, color=th["ink2"])
        ax.set_yticks(range(len(EST))); ax.set_yticklabels([l for _, l in EST]); ax.invert_yaxis()
        ax.set_title(f"{y}: relative lift vs control, 95% CI", loc="left", color=th["ink"])
        ax.set_xlabel("% lift"); style(ax, th)
    fig.tight_layout(); fig.savefig(OUT / f"effects-{theme}.png", dpi=150, facecolor=th["bg"]); plt.close(fig)

    # 2. peeking false positives
    fig, ax = plt.subplots(figsize=(11.5, 3.4), facecolor=th["bg"])
    meth = [("fixed_horizon_rate", "Fixed horizon (1 look)"), ("naive_reject_rate", "Peek 20x, naive z > 1.96"),
            ("pocock_reject_rate", "Pocock boundary"), ("obrien_fleming_reject_rate", "O'Brien-Fleming boundary"),
            ("msprt_reject_rate", "mSPRT (always valid)")]
    aa = pk[pk["mode"] == "aa"].set_index("outcome")
    w = 0.38
    for j, y in enumerate(["visit", "conversion"]):
        vals = [100 * aa.loc[y, m] for m, _ in meth]
        ax.barh([i + (j - 0.5) * w for i in range(len(meth))], vals, height=w, color=th["cols"][j], label=y)
        for i, v in enumerate(vals):
            ax.text(v + 0.4, i + (j - 0.5) * w, f"{v:.1f}%", va="center", fontsize=8.5, color=th["ink"])
    ax.axvline(5, color=th["ink2"], ls="--", lw=1); ax.text(5.3, len(meth) - 0.45, "α = 5%", color=th["ink2"], fontsize=9)
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(meth))); ax.set_yticklabels([l for _, l in meth]); ax.invert_yaxis()
    ax.set_xlabel("false-positive rate across 2,000 A/A replays of the real data (%)")
    ax.set_title("Peeking at a fixed-horizon test, and three ways to stop safely", loc="left", color=th["ink"])
    ax.legend(frameon=False, loc="lower right", labelcolor=th["ink"]); style(ax, th)
    fig.tight_layout(); fig.savefig(OUT / f"peeking-{theme}.png", dpi=150, facecolor=th["bg"]); plt.close(fig)

    # 3. IV vs naive
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.2), facecolor=th["bg"])
    ax_pad = {"visit": 0.6, "conversion": 0.08}
    for ax, y in zip(axes, ["visit", "conversion"]):
        iv = R["iv"][y]
        ax.set_axisbelow(True)
        items = [("Naive: exposed vs unexposed", iv["naive"]["exposed_vs_unexposed_all"], None, th["cols"][1]),
                 ("IV Wald (assignment as instrument)", iv["wald"]["late"], (iv["wald"]["ci_low"], iv["wald"]["ci_high"]), th["cols"][2]),
                 ("2SLS + 12 features", iv["tsls_with_features"]["late"],
                  (iv["tsls_with_features"]["ci_low"], iv["tsls_with_features"]["ci_high"]), th["cols"][2])]
        for i, (lab, v, ci, c) in enumerate(items):
            ax.barh(i, 100 * v, color=c, height=0.6)
            if ci:
                ax.errorbar(100 * v, i, xerr=[[100 * (v - ci[0])], [100 * (ci[1] - v)]], color=th["ink"], capsize=3, lw=1.2)
            right = 100 * (ci[1] if ci else v)
            ax.text(right + ax_pad[y], i, f"{100*v:.2f}", va="center", fontsize=9, color=th["ink"])
        ax.set_yticks(range(3)); ax.set_yticklabels([x[0] for x in items]); ax.invert_yaxis()
        ax.set_title(f"{y}: effect of seeing an ad, pts", loc="left", color=th["ink"]); style(ax, th)
        ax.set_xlim(0, 100 * iv["naive"]["exposed_vs_unexposed_all"] * 1.18)
    fig.tight_layout(); fig.savefig(OUT / f"iv-{theme}.png", dpi=150, facecolor=th["bg"]); plt.close(fig)
print("ok")
