import os
import numpy as np
import matplotlib.pyplot as plt
from mc_pricer.models.black_scholes import BSMParams, euro_call_price, euro_put_price
from mc_pricer.pricers.euro_vanilla import price

def main():
    params = BSMParams(s0=100.0, k=100.0, r=0.02, sigma=0.20, t=1.0)
    kind = "call"
    analytic = euro_call_price(params) if kind == "call" else euro_put_price(params)

    methods = ["plain", "antithetic", "control"]
    N_list = np.geomspace(1_000, 200_000, num=8, dtype=int)

    os.makedirs("artifacts", exist_ok=True)
    rows = []
    for method in methods:
        errs = []
        ses = []
        for N in N_list:
            res = price(kind, params, n=int(N), method=method)
            err = abs(res.price - analytic)
            errs.append(err)
            ses.append(res.stderr)
            rows.append((method, N, res.price, res.stderr, res.ci95[0], res.ci95[1], res.delta))
        # plot convergence
        plt.figure()
        plt.loglog(N_list, errs, marker="o")
        plt.xlabel("Paths (N)")
        plt.ylabel("|MC - Analytic|")
        plt.title(f"Convergence error — {method}")
        plt.grid(True, which="both")
        plt.savefig(f"artifacts/convergence_{method}.png", bbox_inches="tight")
        plt.close()

    # save a CSV of summary
    import pandas as pd
    df = pd.DataFrame(rows, columns=["method", "N", "price", "stderr", "ci_low", "ci_high", "delta"])
    df.to_csv("artifacts/summary.csv", index=False)
    print(df.head(10))
    print("Analytic price:", analytic)
    print("Wrote plots to artifacts/.")

if __name__ == "__main__":
    main()
