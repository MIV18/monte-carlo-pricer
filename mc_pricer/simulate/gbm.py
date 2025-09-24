"""
Geometric Brownian Motion terminal price sampler under risk-neutral measure.
"""
from __future__ import annotations
import numpy as np

def sample_terminal_prices(s0: float, r: float, sigma: float, t: float, n: int, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    z = rng.standard_normal(n)
    drift = (r - 0.5*sigma**2)*t
    diff = sigma*np.sqrt(t)*z
    sT = s0 * np.exp(drift + diff)
    return sT, z  # return z for antithetic/greeks
