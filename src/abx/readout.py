"""Experiment readout on the Criteo Uplift v2.1 sample. Writes every number to reports/.

python -m abx.readout                                  # data/criteo_sample.parquet
python -m abx.readout --data other.parquet --out reports/dryrun
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from .estimators import (aipw, diff_in_means, mde_proportion, n_for_mde, naive_exposure_contrasts, regression_adjusted,
                         srm_test, tsls, variance_reduction, wald_iv)
from .sequential import fixed_horizon_rate, peeking_study

ROOT = Path(__file__).resolve().parents[2]
FEATS = [f"f{i}" for i in range(12)]
DESIGNED_TREAT_SHARE = 0.85      # Criteo Uplift v2.1 documentation: treatment ratio 85%
FULL_N = 13_979_592              # rows in the full v2.1 dataset
OUTCOMES = ["visit", "conversion"]
SEED = 42
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def auc(y, s):
    """ROC AUC via the Mann-Whitney rank statistic."""
    r = stats.rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def lgb_params(obj):
    return dict(objective=obj, learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=SEED)


def cross_fit(X, y, folds=2, rounds=300, obj="binary"):
    """Out-of-fold predictions of y from pre-treatment features only (never sees assignment)."""
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, folds, len(y))
    pred = np.zeros(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        m = lgb.train(lgb_params(obj), lgb.Dataset(X[tr], y[tr]), rounds)
        pred[te] = m.predict(X[te])
    return pred


def arm_models(X, y, t, folds=2, rounds=300):
    """Cross-fitted outcome models E[y | X, arm] for AIPW."""
    rng = np.random.default_rng(SEED + 1)
    fold = rng.integers(0, folds, len(y))
    mu1, mu0 = np.zeros(len(y)), np.zeros(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        for arm, out in ((1, mu1), (0, mu0)):
            m = tr & (t == arm)
            mod = lgb.train(lgb_params("binary"), lgb.Dataset(X[m], y[m]), rounds)
            out[te] = mod.predict(X[te])
    return mu1, mu0


def run(data_path: Path, out: Path, reps=1000):
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    df = pd.read_parquet(data_path)
    t = df["treatment"].to_numpy(float)
    d = df["exposure"].to_numpy(float)
    X = df[FEATS].to_numpy(np.float64)
    R = {"n": len(df), "data": str(data_path.name)}
    log(f"rows={len(df):,}")

    # ---------- 1. health checks
    R["srm"] = srm_test(t, DESIGNED_TREAT_SHARE)
    R["rates_by_arm"] = df.groupby("treatment")[["visit", "conversion", "exposure"]].mean().to_dict()
    bal = cross_fit(X, t.astype(int), rounds=150)
    R["balance_auc_t_from_features"] = auc(t, bal)          # ~0.5 if randomization holds
    smd = (X[t == 1].mean(0) - X[t == 0].mean(0)) / X.std(0)
    R["max_abs_std_mean_diff"] = float(np.abs(smd).max())
    R["std_mean_diff"] = {f: float(v) for f, v in zip(FEATS, smd)}
    R["smd_se"] = float(np.sqrt(1 / t.sum() + 1 / (len(t) - t.sum())))
    log(f"SRM p={R['srm']['p_value']:.3g} share={R['srm']['observed_share']:.4f} balance AUC={R['balance_auc_t_from_features']:.4f}")

    # ---------- 2. ITT + variance reduction
    itt, rows = {}, []
    for yname in OUTCOMES:
        y = df[yname].to_numpy(float)
        u = diff_in_means(y, t)
        lin = regression_adjusted(y, t, X)
        pred = cross_fit(X, y.astype(int))
        cupac = regression_adjusted(y, t, pred[:, None])
        both = regression_adjusted(y, t, np.column_stack([X, pred]))
        # doubly robust: cross-fitted outcome model per arm + cross-fitted propensity (handles imbalance)
        mu1, mu0 = arm_models(X, y, t)
        dr = aipw(y, t, mu1, mu0, bal)
        # permuted A/A: under clean randomization, adjustment should not move the estimate
        tp = np.random.default_rng(SEED).permutation(t)
        up, lp = diff_in_means(y, tp), regression_adjusted(y, tp, np.column_stack([X, pred]))
        itt[yname] = {"unadjusted": u, "linear_features": lin, "cupac_lgbm": cupac, "linear_plus_cupac": both,
                      "aipw_doubly_robust": dr, "outcome_auc_from_features": auc(y, pred),
                      "permuted_check": {"unadj_est": up["estimate"], "adj_est": lp["estimate"],
                                         "shift_in_se": (lp["estimate"] - up["estimate"]) / up["se"],
                                         "variance_reduction": variance_reduction(up["se"], lp["se"])}}
        for name, r in [("unadjusted", u), ("linear_features", lin), ("cupac_lgbm", cupac),
                        ("linear_plus_cupac", both), ("aipw_doubly_robust", dr)]:
            rows.append({"outcome": yname, "estimator": name, "estimate": r["estimate"], "se": r["se"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"],
                         "rel_lift": r["estimate"] / u["control_mean"],
                         "variance_reduction": variance_reduction(u["se"], r["se"])})
        log(f"{yname}: unadj {u['estimate']:.5f} ({u['rel_lift']:+.1%}) adj {both['estimate']:.5f} aipw {dr['estimate']:.5f} "
            f"VR both={rows[-2]['variance_reduction']:.1%}; permuted shift {itt[yname]['permuted_check']['shift_in_se']:+.2f} SE, "
            f"permuted VR {itt[yname]['permuted_check']['variance_reduction']:.1%}")
    R["itt"] = itt
    pd.DataFrame(rows).to_csv(out / "itt.csv", index=False, float_format="%.6g")

    # ---------- 3. power / MDE
    n1, n0 = int(t.sum()), int(len(t) - t.sum())
    pw = []
    for yname in OUTCOMES:
        p0 = itt[yname]["unadjusted"]["control_mean"]
        vr = next(r["variance_reduction"] for r in rows if r["outcome"] == yname and r["estimator"] == "linear_plus_cupac")
        for label, a, b in [("this sample", n1, n0), ("full dataset", int(FULL_N * n1 / len(t)), int(FULL_N * n0 / len(t)))]:
            m, m_vr = mde_proportion(p0, a, b), mde_proportion(p0, a, b, var_reduction=vr)
            pw.append({"outcome": yname, "sample": label, "n": a + b, "control_rate": p0, "mde_abs": m,
                       "mde_rel": m / p0, "mde_abs_adjusted": m_vr, "mde_rel_adjusted": m_vr / p0, "var_reduction": vr})
        for rel in (0.01, 0.02, 0.05, 0.10):
            pw.append({"outcome": yname, "sample": f"n needed for {rel:.0%} lift", "control_rate": p0,
                       "n": n_for_mde(p0, rel * p0, n1 / len(t)),
                       "n_adjusted": n_for_mde(p0, rel * p0, n1 / len(t), var_reduction=vr)})
    pd.DataFrame(pw).to_csv(out / "power.csv", index=False, float_format="%.6g")

    # ---------- 4. peeking
    pk = []
    for yname in OUTCOMES:
        y = df[yname].to_numpy(float)
        for mode in ("aa", "bootstrap"):
            r = peeking_study(y, t, K=20, reps=reps, mode=mode, seed=SEED)
            r["fixed_horizon_rate"] = fixed_horizon_rate(y, t, reps=reps, mode=mode, seed=SEED)
            r["outcome"] = yname
            pk.append(r)
            log(f"peeking {yname}/{mode}: naive {r['naive_reject_rate']:.3f} pocock {r['pocock_reject_rate']:.3f} "
                f"obf {r['obrien_fleming_reject_rate']:.3f} msprt {r['msprt_reject_rate']:.3f} fixed {r['fixed_horizon_rate']:.3f}")
    pd.DataFrame(pk).to_csv(out / "peeking.csv", index=False, float_format="%.6g")

    # ---------- 5. IV: effect on users who actually saw an ad
    iv = {}
    for yname in OUTCOMES:
        y = df[yname].to_numpy(float)
        w = wald_iv(y, t, d)
        s = tsls(y, d, t, X)
        nv = naive_exposure_contrasts(y, t, d)
        iv[yname] = {"wald": w, "tsls_with_features": s, "naive": nv,
                     "naive_over_iv": {k: v / w["late"] for k, v in nv.items()}}
        log(f"IV {yname}: LATE {w['late']:.5f} [{w['ci_low']:.5f},{w['ci_high']:.5f}] naive {nv['exposed_vs_unexposed_all']:.5f} "
            f"ratio {nv['exposed_vs_unexposed_all']/w['late']:.2f}x first-stage {w['itt_d']:.4f}")
    R["iv"] = iv
    R["runtime_s"] = time.time() - T0
    (out / "readout.json").write_text(json.dumps(R, indent=2, default=float))
    log("done")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "criteo_sample.parquet"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    run(Path(a.data), Path(a.out), a.reps)
