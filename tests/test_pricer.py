import numpy as np
from mc_pricer.models.black_scholes import BSMParams, euro_call_price, euro_put_price
from mc_pricer.pricers.euro_vanilla import price

def test_mc_close_to_analytic():
    params = BSMParams(s0=100.0, k=100.0, r=0.01, sigma=0.2, t=1.0)
    res = price("call", params, n=50_000, method="antithetic")
    analytic = euro_call_price(params)
    # within ~3 std errors
    assert abs(res.price - analytic) < 3 * res.stderr

def test_control_variate_improves_std():
    params = BSMParams(s0=100.0, k=100.0, r=0.01, sigma=0.2, t=1.0)
    res_plain = price("call", params, n=30_000, method="plain")
    res_ctrl = price("call", params, n=30_000, method="control")
    assert res_ctrl.stderr < res_plain.stderr
