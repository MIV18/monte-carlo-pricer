"""
Monte Carlo pricing for European calls/puts with variance reduction.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
from ..simulate.gbm import sample_terminal_prices
from ..models.black_scholes import BSMParams, euro_call_price, euro_put_price

Kind = Literal["call", "put"]

@dataclass
class MCResult:
    price: float
    stderr: float
    ci95: Tuple[float, float]
    n: int
    method: str
    delta: Optional[float] = None

def payoff(kind: Kind, sT: np.ndarray, k: float) -> np.ndarray:
    if kind == "call":
        return np.maximum(sT - k, 0.0)
    else:
        return np.maximum(k - sT, 0.0)

def plain_mc(kind: Kind, params: BSMParams, n: int, rng=None) -> MCResult:
    sT, _ = sample_terminal_prices(params.s0, params.r, params.sigma, params.t, n, rng=rng)
    g = payoff(kind, sT, params.k)
    disc_g = np.exp(-params.r*params.t) * g
    mean = disc_g.mean()
    se = disc_g.std(ddof=1)/np.sqrt(n)
    ci = (mean - 1.96*se, mean + 1.96*se)
    return MCResult(price=float(mean), stderr=float(se), ci95=ci, n=n, method="plain")

def antithetic(kind: Kind, params: BSMParams, n: int, rng=None) -> MCResult:
    if rng is None:
        rng = np.random.default_rng()
    # half draws, pair z and -z
    m = (n + 1)//2
    z = rng.standard_normal(m)
    s0, r, sigma, t = params.s0, params.r, params.sigma, params.t
    drift = (r - 0.5*sigma**2)*t
    diff = sigma*np.sqrt(t)*z
    sT1 = s0*np.exp(drift + diff)
    sT2 = s0*np.exp(drift - diff)
    g = 0.5*(payoff(kind, sT1, params.k) + payoff(kind, sT2, params.k))
    disc_g = np.exp(-r*t)*g
    mean = disc_g.mean()
    se = disc_g.std(ddof=1)/np.sqrt(m)
    ci = (mean - 1.96*se, mean + 1.96*se)
    return MCResult(price=float(mean), stderr=float(se), ci95=ci, n=2*m, method="antithetic")

def control_variate(kind: Kind, params: BSMParams, n: int, rng=None) -> MCResult:
    if rng is None:
        rng = np.random.default_rng()
    sT, _ = sample_terminal_prices(params.s0, params.r, params.sigma, params.t, n, rng=rng)
    g = payoff(kind, sT, params.k)
    disc_g = np.exp(-params.r*params.t) * g
    # Control variate: use terminal price discounted vs its expectation (s0*exp(rT))
    control = np.exp(-params.r*params.t) * sT
    control_mean = params.s0  # E[exp(-rT) S_T] = S_0 under risk-neutral
    # optimal beta = Cov(Y,X)/Var(X)
    x = control - control_mean
    y = disc_g
    beta = np.cov(y, x, ddof=1)[0,1] / np.var(x, ddof=1)
    y_cv = y - beta * x
    mean = y_cv.mean()
    se = y_cv.std(ddof=1)/np.sqrt(n)
    ci = (mean - 1.96*se, mean + 1.96*se)
    return MCResult(price=float(mean), stderr=float(se), ci95=ci, n=n, method=f"control_variate(beta={beta:.3f})")

def pathwise_delta_call(params: BSMParams, n: int, rng=None) -> float:
    """
    Pathwise estimator for Delta of a European call under BSM.
    Uses identity: Delta = E[ 1_{S_T>K} * S_T / S_0 ] under a simple pathwise derivation.
    More precise derivations include terms with Z; we keep a robust version here.
    """
    if rng is None:
        rng = np.random.default_rng()
    sT, _ = sample_terminal_prices(params.s0, params.r, params.sigma, params.t, n, rng=rng)
    ind = (sT > params.k).astype(float)
    delta_est = np.exp(-params.r*params.t) * np.mean(ind * (sT / params.s0))
    return float(delta_est)

def price(kind: Kind, params: BSMParams, n: int = 100_000, method: str = "antithetic", rng=None) -> MCResult:
    if method == "plain":
        res = plain_mc(kind, params, n, rng=rng)
    elif method == "antithetic":
        res = antithetic(kind, params, n, rng=rng)
    elif method == "control":
        res = control_variate(kind, params, n, rng=rng)
    else:
        raise ValueError("method must be one of: plain, antithetic, control")
    # add delta for calls for demonstration
    if kind == "call":
        res.delta = pathwise_delta_call(params, max(10_000, n//2), rng=rng)
    return res
