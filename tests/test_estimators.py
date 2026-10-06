"""Estimators checked against simulations where the true answer is known."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abx.estimators import (diff_in_means, mde_proportion, naive_exposure_contrasts,  # noqa: E402
                            regression_adjusted, srm_test, tsls, wald_iv)
from abx.sequential import peeking_study  # noqa: E402
from abx.simulate import criteo_like  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    return criteo_like(n=400_000, seed=1)


def test_diff_in_means_ci_covers_truth():
    rng = np.random.default_rng(0)
    cover = 0
    for _ in range(400):
        t = (rng.random(4000) < 0.5).astype(float)
        y = rng.normal(1.0 + 0.1 * t, 1.0)
        r = diff_in_means(y, t)
        cover += r["ci_low"] <= 0.1 <= r["ci_high"]
    assert 0.92 <= cover / 400 <= 0.98


def test_regression_adjustment_unbiased_and_tighter(sim):
    ests, ses_u, ses_a = [], [], []
    for s in range(6):
        d = criteo_like(n=120_000, seed=100 + s)
        u = diff_in_means(d["y"], d["t"])
        a = regression_adjusted(d["y"], d["t"], d["X"])
        ests.append(a["estimate"] - d["true_itt"]); ses_u.append(u["se"]); ses_a.append(a["se"])
    assert abs(np.mean(ests)) < 2 * np.mean(ses_a)          # unbiased within noise
    assert np.mean(ses_a) < 0.99 * np.mean(ses_u)             # predictive X reduces variance (weak X here)


def test_iv_recovers_effect_on_exposed_and_naive_is_biased(sim):
    r = wald_iv(sim["y"], sim["t"], sim["d"])
    assert r["control_exposure_rate"] == 0.0
    assert r["ci_low"] <= sim["true_late"] <= r["ci_high"]
    naive = naive_exposure_contrasts(sim["y"], sim["t"], sim["d"])["exposed_vs_unexposed_all"]
    assert naive - sim["true_late"] > 4 * r["se"]          # intent confounds the naive comparison


def test_tsls_matches_wald_without_covariates(sim):
    w = wald_iv(sim["y"], sim["t"], sim["d"])
    s = tsls(sim["y"], sim["d"], sim["t"])
    assert abs(w["late"] - s["late"]) < 1e-9
    assert abs(w["se"] / s["se"] - 1) < 0.02


def test_srm_detects_imbalance():
    rng = np.random.default_rng(0)
    t_ok = (rng.random(500_000) < 0.85).astype(float)
    assert srm_test(t_ok, 0.85)["p_value"] > 1e-3
    t_bad = (rng.random(500_000) < 0.84).astype(float)
    assert srm_test(t_bad, 0.85)["p_value"] < 1e-6


def test_mde_gives_80_percent_power():
    rng = np.random.default_rng(0)
    p0, n1, n0 = 0.05, 20_000, 20_000
    mde = mde_proportion(p0, n1, n0)
    hits = 0
    for _ in range(600):
        y1 = rng.random(n1) < p0 + mde
        y0 = rng.random(n0) < p0
        y = np.concatenate([y1, y0]).astype(float); t = np.r_[np.ones(n1), np.zeros(n0)]
        hits += diff_in_means(y, t)["p_value"] < 0.05
    assert 0.75 <= hits / 600 <= 0.85


def test_peeking_inflates_and_corrections_restore_alpha():
    rng = np.random.default_rng(0)
    y = (rng.random(60_000) < 0.05).astype(float)
    t = (rng.random(60_000) < 0.5).astype(float)
    r = peeking_study(y, t, K=20, reps=600, mode="aa", seed=1)
    assert r["naive_reject_rate"] > 0.15
    for m in ("pocock", "obrien_fleming", "msprt"):
        assert r[f"{m}_reject_rate"] <= 0.075, (m, r[f"{m}_reject_rate"])


def test_aipw_fixes_imbalanced_assignment():
    """Assignment probability depends on X (like pooling tests with different treatment ratios):
    difference in means is biased, AIPW with the true nuisances recovers the effect."""
    from abx.estimators import aipw
    rng = np.random.default_rng(3)
    n = 300_000
    x = rng.standard_normal(n)
    e = 1 / (1 + np.exp(-(1.7 + 0.6 * x)))              # ~85% treated overall, more if x high
    t = (rng.random(n) < e).astype(float)
    p0 = 1 / (1 + np.exp(-(-3 + 0.8 * x)))
    y = (rng.random(n) < p0 + 0.01 * t).astype(float)
    naive = diff_in_means(y, t)["estimate"]
    r = aipw(y, t, p0 + 0.01, p0, e)
    assert abs(naive - 0.01) > 4 * r["se"]
    assert r["ci_low"] <= 0.01 <= r["ci_high"]


def test_lin_estimator_matches_statsmodels_hc2():
    """Cross-check the hand-rolled regression adjustment against statsmodels OLS with HC2 errors."""
    import statsmodels.api as sm
    rng = np.random.default_rng(5)
    n = 5000
    X = rng.standard_normal((n, 3)); t = (rng.random(n) < 0.7).astype(float)
    y = 0.5 * X[:, 0] + 0.2 * t + rng.standard_normal(n)
    r = regression_adjusted(y, t, X)
    Xc = X - X.mean(0)
    D = sm.add_constant(np.column_stack([t, Xc, Xc * t[:, None]]))
    fit = sm.OLS(y, D).fit(cov_type="HC2")
    assert abs(r["estimate"] - fit.params[1]) < 1e-10
    assert abs(r["se"] - fit.bse[1]) < 1e-10
