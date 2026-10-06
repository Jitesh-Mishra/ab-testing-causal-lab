"""Estimators for a randomized experiment readout.

Conventions: t = assignment (0/1, randomized), y = outcome, d = exposure (0/1, NOT randomized).
All functions take numpy arrays and return plain dicts so results drop straight into tables.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

Z95 = stats.norm.ppf(0.975)


# ----------------------------------------------------------------------------- SRM
def srm_test(t: np.ndarray, expected_treat_share: float) -> dict:
    """Chi-square goodness-of-fit of the observed split against the designed split."""
    n1 = int(t.sum()); n = len(t); n0 = n - n1
    e1, e0 = n * expected_treat_share, n * (1 - expected_treat_share)
    chi2 = (n1 - e1) ** 2 / e1 + (n0 - e0) ** 2 / e0
    return {"n": n, "n_treat": n1, "n_control": n0, "observed_share": n1 / n,
            "expected_share": expected_treat_share, "chi2": chi2, "p_value": float(stats.chi2.sf(chi2, 1))}


# ----------------------------------------------------------------------------- ITT
def diff_in_means(y: np.ndarray, t: np.ndarray) -> dict:
    """Difference in means with Welch (unequal-variance) SE, plus relative lift via the delta method."""
    y1, y0 = y[t == 1], y[t == 0]
    m1, m0 = y1.mean(), y0.mean()
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    est, se = m1 - m0, np.sqrt(v1 + v0)
    with np.errstate(divide="ignore", invalid="ignore"):   # relative lift undefined when control mean is 0
        rel = est / m0 if m0 else float("nan")
        rel_se = np.sqrt(v1 / m0 ** 2 + v0 * m1 ** 2 / m0 ** 4) if m0 else float("nan")
    return {"control_mean": m0, "treat_mean": m1, "estimate": est, "se": se,
            "ci_low": est - Z95 * se, "ci_high": est + Z95 * se, "z": est / se,
            "p_value": float(2 * stats.norm.sf(abs(est / se))), "rel_lift": rel,
            "rel_ci_low": rel - Z95 * rel_se, "rel_ci_high": rel + Z95 * rel_se}


# ----------------------------------------------------------------------------- regression adjustment
def _ols_hc2(X: np.ndarray, y: np.ndarray, col: int = 1):
    """OLS coefficient and HC2 robust SE for one column. Memory-light for millions of rows."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # leverages
    w = e ** 2 / (1 - h)
    meat = (X * w[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv
    return float(beta[col]), float(np.sqrt(V[col, col]))


def regression_adjusted(y: np.ndarray, t: np.ndarray, X: np.ndarray) -> dict:
    """Lin (2013) estimator: y ~ 1 + t + Xc + t*Xc with centred covariates and HC2 SEs.
    Unbiased under randomization for any covariates measured before treatment; with a single
    covariate it is the CUPED estimator with treatment-specific slopes."""
    Xc = X - X.mean(0)
    D = np.column_stack([np.ones(len(y)), t, Xc, Xc * t[:, None]])
    est, se = _ols_hc2(D, y, col=1)
    return {"estimate": est, "se": se, "ci_low": est - Z95 * se, "ci_high": est + Z95 * se,
            "p_value": float(2 * stats.norm.sf(abs(est / se)))}


def variance_reduction(se_unadj: float, se_adj: float) -> float:
    """Share of the estimator's variance removed by adjustment = equivalent sample-size saving."""
    return 1 - (se_adj / se_unadj) ** 2


# ----------------------------------------------------------------------------- power / MDE
def mde_proportion(p0: float, n_treat: int, n_control: int, alpha=0.05, power=0.8, var_reduction=0.0) -> float:
    """Minimum detectable absolute effect for a two-sample difference in proportions (normal approx)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    se = np.sqrt(p0 * (1 - p0) * (1 / n_treat + 1 / n_control) * (1 - var_reduction))
    return float((za + zb) * se)


def n_for_mde(p0: float, mde_abs: float, treat_share: float, alpha=0.05, power=0.8, var_reduction=0.0) -> int:
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    k = 1 / treat_share + 1 / (1 - treat_share)
    return int(np.ceil((za + zb) ** 2 * p0 * (1 - p0) * k * (1 - var_reduction) / mde_abs ** 2))


# ----------------------------------------------------------------------------- IV
def wald_iv(y: np.ndarray, t: np.ndarray, d: np.ndarray) -> dict:
    """Effect of exposure on the exposed, using random assignment as the instrument.

    LATE = ITT_y / ITT_d. With one-sided non-compliance (no exposure in control) this is the
    effect on the treated-and-exposed (ATT). SE by the delta method using the covariance of
    the two ITT estimates (both computed on the same units)."""
    i_y, i_d = diff_in_means(y, t), diff_in_means(d, t)
    late = i_y["estimate"] / i_d["estimate"]
    # residual form: SE of 2SLS without covariates (robust)
    first = i_d["estimate"]
    resid = (y - y[t == 0].mean() * (t == 0) - y[t == 1].mean() * (t == 1)) - late * (
        d - d[t == 0].mean() * (t == 0) - d[t == 1].mean() * (t == 1))
    v = np.zeros(2)
    for g in (0, 1):
        m = t == g
        v[g] = resid[m].var(ddof=1) / m.sum()
    se = np.sqrt(v.sum()) / abs(first)
    return {"late": late, "se": se, "ci_low": late - Z95 * se, "ci_high": late + Z95 * se,
            "itt_y": i_y["estimate"], "itt_d": first,
            "compliers_share": first, "control_exposure_rate": float(d[t == 0].mean()),
            "first_stage_F": float((first / i_d["se"]) ** 2)}


def tsls(y: np.ndarray, d: np.ndarray, t: np.ndarray, X: np.ndarray | None = None) -> dict:
    """2SLS of y on d instrumented by t, optional exogenous covariates X, HC1 robust SE."""
    n = len(y)
    W = np.ones((n, 1)) if X is None else np.column_stack([np.ones(n), X - X.mean(0)])
    Z = np.column_stack([W, t])
    R = np.column_stack([W, d])
    PZ_R = Z @ np.linalg.lstsq(Z, R, rcond=None)[0]               # first-stage fitted regressors
    beta = np.linalg.lstsq(PZ_R, y, rcond=None)[0]
    e = y - R @ beta                                               # structural residuals use actual d
    A = np.linalg.inv(PZ_R.T @ PZ_R)
    meat = (PZ_R * (e ** 2)[:, None]).T @ PZ_R
    V = A @ meat @ A * n / (n - R.shape[1])
    k = R.shape[1] - 1
    est, se = float(beta[k]), float(np.sqrt(V[k, k]))
    return {"late": est, "se": se, "ci_low": est - Z95 * se, "ci_high": est + Z95 * se}


def naive_exposure_contrasts(y: np.ndarray, t: np.ndarray, d: np.ndarray) -> dict:
    """The comparisons a dashboard would show, none of which is causal on its own."""
    return {
        "exposed_vs_unexposed_all": y[d == 1].mean() - y[d == 0].mean(),
        "exposed_vs_unexposed_within_treatment": y[(t == 1) & (d == 1)].mean() - y[(t == 1) & (d == 0)].mean(),
        "exposed_treatment_vs_control": y[(t == 1) & (d == 1)].mean() - y[t == 0].mean(),
    }


# ----------------------------------------------------------------------------- doubly robust
def aipw(y: np.ndarray, t: np.ndarray, mu1: np.ndarray, mu0: np.ndarray, e: np.ndarray, clip=0.01) -> dict:
    """Augmented IPW (doubly robust) average treatment effect from cross-fitted nuisances:
    outcome models mu1, mu0 and propensity e = P(t=1 | X). Consistent if either the propensity
    or the outcome model is right; with true randomization e is constant and it reduces to an
    efficient regression-adjusted estimator."""
    e = np.clip(e, clip, 1 - clip)
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    est, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))
    return {"estimate": est, "se": se, "ci_low": est - Z95 * se, "ci_high": est + Z95 * se,
            "p_value": float(2 * stats.norm.sf(abs(est / se)))}
