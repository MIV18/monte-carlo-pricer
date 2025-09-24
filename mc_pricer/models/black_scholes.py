"""
Black–Scholes–Merton closed-form pricing for European options.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from math import log, sqrt, exp
from scipy.stats import norm

@dataclass
class BSMParams:
    s0: float
    k: float
    r: float
    sigma: float
    t: float  # in years

def d1(params: BSMParams) -> float:
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t
    return (np.log(s0/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))

def d2(params: BSMParams) -> float:
    return d1(params) - params.sigma*np.sqrt(params.t)

def euro_call_price(params: BSMParams) -> float:
    D1, D2 = d1(params), d2(params)
    return params.s0*norm.cdf(D1) - params.k*np.exp(-params.r*params.t)*norm.cdf(D2)

def euro_put_price(params: BSMParams) -> float:
    D1, D2 = d1(params), d2(params)
    return params.k*np.exp(-params.r*params.t)*norm.cdf(-D2) - params.s0*norm.cdf(-D1)
