"""
Quasi Monte Carlo helpers (Sobol) for [0,1] and map to N(0,1).
"""
from __future__ import annotations
import numpy as np
try:
    from scipy.stats import qmc, norm
except Exception:
    qmc = None
    norm = None

def sobol_normals(n: int, scramble=True, seed=None):
    if qmc is None or norm is None:
        raise ImportError("SciPy qmc not available.")
    dim = 1
    engine = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    u = engine.random(n)[:,0]
    # Map (0,1) to N(0,1) via inverse CDF
    z = norm.ppf(u.clip(1e-12, 1-1e-12))
    return z
