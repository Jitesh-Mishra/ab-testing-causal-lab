"""Synthetic experiments with known ground truth, shaped like the Criteo data (used only for tests)."""
from __future__ import annotations

import numpy as np


def criteo_like(n=200_000, treat_share=0.85, seed=0, base=0.04, effect_scale=0.03):
    """Randomized assignment t; exposure d only possible in treatment and more likely for high-intent
    users (confounded with the outcome); binary outcome with effect only for exposed users.
    Returns arrays and the true effect on the treated-and-exposed."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    intent = 0.9 * X[:, 0] + 0.5 * X[:, 1]
    t = (rng.random(n) < treat_share).astype(float)
    p_exp = 1 / (1 + np.exp(-(-3.0 + 1.2 * intent)))            # ~6% exposure, rises with intent
    d = t * (rng.random(n) < p_exp)
    p0 = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + 1.0 * intent)))
    tau = effect_scale * (1 + 0.5 * np.tanh(X[:, 2]))          # heterogeneous effect, positive
    y = (rng.random(n) < np.clip(p0 + d * tau, 0, 1)).astype(float)
    att_exposed = float(tau[d == 1].mean())
    return {"X": X, "t": t, "d": d, "y": y, "true_late": att_exposed, "true_itt": float(tau[d == 1].sum() / t.sum())}
