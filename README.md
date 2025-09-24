# Monte Carlo Option Pricer (with Variance Reduction)

A clean, well-documented implementation of Monte Carlo pricing for European options under Black–Scholes–Merton (BSM), including variance-reduction techniques and basic Greeks estimation.

## Features
- Plain Monte Carlo and **antithetic variates**
- **Control variate** using the analytic BSM price
- (Optional) **Quasi–Monte Carlo** with Sobol sequences
- Pathwise estimator for **Delta**
- Reproducible results, unit tests, and plots

## Quick Start
```bash
# create and activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python scripts/run_pricer.py
```

## Project Structure
```
mc_pricer/
  __init__.py
  models/black_scholes.py
  simulate/gbm.py
  pricers/euro_vanilla.py
  utils/qmc.py
scripts/
  run_pricer.py
tests/
  test_pricer.py
```

## Example Output
The script prints the estimated price, standard error, and confidence interval, and writes a PNG of the convergence plot to `artifacts/`.

## Methodology
Under BSM: dS_t = r S_t dt + σ S_t dW_t. Closed-form price exists for European calls/puts.

We estimate price via Monte Carlo by simulating terminal prices:
S_T = S_0 * exp((r - 0.5 σ^2) T + σ √T Z).

Variance reduction:
- **Antithetic variates:** use Z and −Z and average payoffs.
- **Control variate:** use analytic closed-form as control for the discounted payoff indicator; we regress to reduce variance.

Greeks:
- **Delta (pathwise):** for a call, Δ = E[ 1_{S_T>K} * S_T / (S_0 σ √T) * Z ] under lognormal pathwise form, or use likelihood-ratio; here we implement a simple pathwise estimator for a call.

## Reproducing the Paper Results
We include a convergence plot (error vs. paths) and a VR comparison. See `scripts/run_pricer.py`.

## Cite / Learning Value
This repo demonstrates:
- Financial maths rigor (derivation summarized above, with references in code)
- Software engineering hygiene (tests, structure, type hints)
- Practical numerics (variance reduction, QMC)

## License
MIT
