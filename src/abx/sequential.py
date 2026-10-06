"""Peeking: what happens to the false-positive rate when a fixed-horizon test is checked repeatedly,
and three ways to keep it at alpha (Pocock and O'Brien-Fleming group-sequential boundaries, mSPRT)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _calibrate(K: int, shape: np.ndarray, alpha=0.05, sims=200_000, seed=0) -> float:
    """Constant c so that P(any |Z_k| > c * shape_k) = alpha for Z on K equally spaced looks of a
    Brownian motion (the large-sample distribution of sequential z-statistics)."""
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((sims, K))
    Z = np.cumsum(inc, 1) / np.sqrt(np.arange(1, K + 1))
    ratio = np.max(np.abs(Z) / shape, axis=1)
    return float(np.quantile(ratio, 1 - alpha))


def boundaries(K: int, alpha=0.05) -> dict:
    k = np.arange(1, K + 1)
    pocock_shape = np.ones(K)
    obf_shape = np.sqrt(K / k)
    return {"naive": np.full(K, stats.norm.ppf(1 - alpha / 2)),
            "pocock": _calibrate(K, pocock_shape, alpha) * pocock_shape,
            "obrien_fleming": _calibrate(K, obf_shape, alpha) * obf_shape}


def msprt_reject(theta: np.ndarray, var: np.ndarray, tau2: float, alpha=0.05) -> np.ndarray:
    """Always-valid mixture SPRT (Johari et al. 2017) with a N(0, tau2) mixing prior on the effect.
    Reject at a look when the likelihood ratio exceeds 1/alpha."""
    lam = np.sqrt(var / (var + tau2)) * np.exp(tau2 * theta ** 2 / (2 * var * (var + tau2)))
    return lam >= 1 / alpha


def sequential_z(y: np.ndarray, t: np.ndarray, looks: np.ndarray):
    """Difference in means, its variance and z at each look (prefix of the arrival order)."""
    c1, c0 = np.cumsum(t), np.cumsum(1 - t)
    s1, s0 = np.cumsum(y * t), np.cumsum(y * (1 - t))
    q1, q0 = np.cumsum(y * y * t), np.cumsum(y * y * (1 - t))
    i = looks - 1
    n1, n0 = c1[i], c0[i]
    m1, m0 = s1[i] / n1, s0[i] / n0
    v1 = (q1[i] / n1 - m1 ** 2) * n1 / (n1 - 1)
    v0 = (q0[i] / n0 - m0 ** 2) * n0 / (n0 - 1)
    theta, var = m1 - m0, v1 / n1 + v0 / n0
    return theta, var, theta / np.sqrt(var)


def peeking_study(y, t, K=20, reps=1000, alpha=0.05, mode="aa", seed=0, tau2=None) -> dict:
    """mode='aa': permute assignment (true effect 0) -> false-positive rates.
    mode='bootstrap': resample users with real assignment -> power and average stopping point."""
    rng = np.random.default_rng(seed)
    n = len(y)
    looks = np.linspace(n / K, n, K).astype(int)
    B = boundaries(K, alpha)
    if tau2 is None:   # mixing prior scale = effect size we care about: 5% relative lift (fixed before real data)
        tau2 = (0.05 * y.mean()) ** 2
    rej = {m: 0 for m in list(B) + ["msprt"]}
    stop = {m: [] for m in rej}
    for _ in range(reps):
        if mode == "aa":
            tt, yy = rng.permutation(t), y
        else:
            idx = rng.integers(0, n, n)
            tt, yy = t[idx], y[idx]
        theta, var, z = sequential_z(yy, tt, looks)
        hits = {m: np.abs(z) > b for m, b in B.items()}
        hits["msprt"] = msprt_reject(theta, var, tau2, alpha)
        for m, h in hits.items():
            if h.any():
                rej[m] += 1
                stop[m].append((np.argmax(h) + 1) / K)
    out = {"K": K, "reps": reps, "mode": mode, "tau2": tau2,
           "pocock_c": float(B["pocock"][0]), "obf_final_c": float(B["obrien_fleming"][-1])}
    for m in rej:
        out[f"{m}_reject_rate"] = rej[m] / reps
        out[f"{m}_avg_stop_fraction"] = float(np.mean(stop[m])) if stop[m] else float("nan")
    # fixed-horizon reference: only the final look with the naive threshold
    return out


def fixed_horizon_rate(y, t, reps=1000, mode="aa", seed=0, alpha=0.05) -> float:
    rng = np.random.default_rng(seed)
    n, c, hits = len(y), stats.norm.ppf(1 - alpha / 2), 0
    for _ in range(reps):
        if mode == "aa":
            tt, yy = rng.permutation(t), y
        else:
            idx = rng.integers(0, n, n); tt, yy = t[idx], y[idx]
        _, _, z = sequential_z(yy, tt, np.array([n]))
        hits += abs(z[0]) > c
    return hits / reps
