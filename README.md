# sabench

**A benchmark suite for variance-based (Sobol) global sensitivity analysis**

[![PyPI version](https://badge.fury.io/py/sabench.svg)](https://badge.fury.io/py/sabench)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/dhetting/sabench/actions/workflows/ci.yml/badge.svg)](https://github.com/dhetting/sabench/actions)
[![codecov](https://codecov.io/gh/dhetting/sabench/branch/main/graph/badge.svg)](https://codecov.io/gh/dhetting/sabench)

`sabench` provides a curated, extensible library of benchmark functions and output
transformations for testing and validating variance-based global sensitivity analysis
(GSA) methods, in particular Sobol first-order (S₁) and total-effect (Sᵀ) indices
estimated via the Jansen–Saltelli estimator.

## Why sabench?

- **19 scalar benchmarks** (Ishigami, Borehole, Sobol-G, Piston, Morris, …) with
  analytical S₁ where derivable.
- **7 functional / PDE benchmarks** (SIR epidemic, Lorenz-96, 2-compartment PK, …)
  producing vector/field outputs.
- **3 spatial benchmarks** (Campbell 2-D / 3-D, exponential variant) for testing
  spatially-resolved GSA.
- **172 output transformations** spanning environmental, engineering, financial,
  ecological, climate, hydrological, medical, and purely mathematical families,
  each with exhaustive metadata (convexity, monotonicity, differentiability class,
  symmetry, reference DOIs, …).
- **Jansen / Saltelli estimator** with variance-weighted aggregation for
  multi-output fields.
- Two bounded non-commutativity scores: Decision Score *D* (sigmoid threshold
  margin) and Sensitivity Shift Δ (Bray-Curtis dissimilarity).
- **Metadata JSON** for every benchmark and transformation — designed for
  regression modelling of transformation-induced index change.

## Installation

```bash
pip install sabench
```

With notebook extras (matplotlib, Jupyter):

```bash
pip install "sabench[notebook]"
```

### Development install (with pixi)

```bash
git clone https://github.com/dhetting/sabench
cd sabench
pixi install           # creates the full dev+notebook environment
pixi run test          # run test suite
pixi run check-all     # lint + format-check + typecheck + test
```

## Quick start

```python
import numpy as np
from sabench.analysis import jansen_s1_st
from sabench.benchmarks.scalar import Ishigami
from sabench.sampling import saltelli_sample
from sabench.transforms import get_transform, score_transform

# 1. Build a Saltelli sample
model = Ishigami()
X = saltelli_sample(model.d, model.bounds, N=2048, seed=0)

# 2. Evaluate the benchmark
Y = model.evaluate(X)

# 3. Estimate first-order Sobol indices
S1, ST = jansen_s1_st(Y, N=2048, d=model.d)
print("S1:", S1)  # analytical: [0.314, 0.442, 0.000]

# 4. Apply a nonlinear transform and score the non-commutativity
transform = get_transform("tanh_a03")
Y_trans = transform(Y)
S1_trans, _ = jansen_s1_st(Y_trans, N=2048, d=model.d)
scores = score_transform(S1, S1_trans, Y, Y_trans)
print(f"D={scores['D']:.3f}  Δ={scores['delta']:.3f}")
```

## Benchmark catalogue

| Category | Examples |
|----------|---------|
| Scalar | Ishigami, Sobol-G, Borehole, Piston, Morris, Friedman, OakleyOHagan, … |
| Functional | SIR epidemic, Lorenz-96, 2-compartment PK, Lotka-Volterra, heat diffusion, … |
| Spatial | Campbell 2-D/3-D, exponential Campbell |

## Transform catalogue (172 transforms)

| Family | Count | Examples |
|--------|-------|---------|
| Mathematical | 67 | polynomials y⁴–y⁶, orthogonal polynomials, sigmoid/activation, oscillatory, threshold, VST |
| Environmental | 20 | log-shift, Box-Cox, GEV/Fréchet/Gumbel CDF, POT, return period |
| Engineering | 19 | Arrhenius, Weibull, Hill dose-response, fatigue (Miner/Basquin), von Mises |
| Statistical | 18 | rank, standardised anomaly, Yeo-Johnson, quantile-normalise, VST |
| Temporal | 13 | cumsum, bandpass, RMS, autocorrelation, temporal quantiles |
| Spatial | 10 | block-average, Matérn smoothing, Laplacian roughness, isoline |
| Financial | 6 | VaR, CVaR/ES, Sharpe ratio, drawdown, fold-change |
| Climate | 5 | anomaly %, bias-correction, quantile delta mapping, GDD, SPI |
| Ecological | 4 | Hellinger, chord, relative abundance, log-ratio |
| Information | 4 | negentropy, Wasserstein proxy, energy distance, Rényi entropy |
| Hydrology | 3 | Nash-Sutcliffe, log-POT, log-streamflow |
| Medical | 3 | Hill response, Emax, log-AUC |

Each transform entry in `TRANSFORMS` carries: `fn`, `params`, `category`, `name`.
The companion JSON (`sabench/metadata/transforms_metadata.json`) provides 35+ fields
per transform for downstream regression modelling.

## Property sets

```python
from sabench.transforms import (
    POINTWISE_TRANSFORMS,   # 74 element-wise transforms
    NONLOCAL_TRANSFORMS,    # 98 transforms using per-sample statistics
    CONVEX_TRANSFORMS,      # 23 transforms with φ″ ≥ 0
    CONCAVE_TRANSFORMS,     # 30 transforms with φ″ ≤ 0
    MONOTONE_TRANSFORMS,    # 56 monotone transforms
    SMOOTH_TRANSFORMS,      # 90 C² or smoother transforms
    NONSMOOTH_TRANSFORMS,   # 27 C⁰ or discontinuous transforms
)
```

## Analysis notebooks

Two deterministic analysis notebooks implement the empirical and theorem-based
diagnostics described in the manuscript sources.

| Notebook | Purpose |
|---|---|
| `notebooks/noncommutativity_grid_analysis.ipynb` | Empirical noncommutativity metrics across all benchmark × transform pairs |
| `notebooks/bounds_theorem_grid_analysis.ipynb` | Taylor-reference and local-affine diagnostics for theorem-compatible pairs; sample-range diagnostics use empirical support bounds and compare against Taylor reference `V_m` |

### Running the notebooks

```bash
pixi run jupyter notebook
```

Or run headless via nbconvert:

```bash
# Noncommutativity grid (all pairs, default N=128)
pixi run jupyter nbconvert --to notebook --execute \
  notebooks/noncommutativity_grid_analysis.ipynb \
  --output /tmp/noncommutativity_executed.ipynb

# Bounds theorem grid (all compatible pairs, default N=128)
pixi run jupyter nbconvert --to notebook --execute \
  notebooks/bounds_theorem_grid_analysis.ipynb \
  --output /tmp/bounds_executed.ipynb
```

### Configuration

Both notebooks read environment variables:

| Variable | Notebook | Default | Effect |
|---|---|---|---|
| `SABENCH_GRID_N_BASE` | noncommutativity | `128` | Saltelli sample size |
| `SABENCH_GRID_MAX_BENCHMARKS` | noncommutativity | `0` (all) | Limit benchmark count |
| `SABENCH_GRID_MAX_TRANSFORMS` | noncommutativity | `0` (all) | Limit transform count |
| `SABENCH_GRID_OUTPUT_DIR` | noncommutativity | `outputs/noncommutativity_grid_analysis` | Output CSV directory |
| `SABENCH_BOUNDS_N_BASE` | bounds | `128` | Saltelli sample size |
| `SABENCH_BOUNDS_MAX_BENCHMARKS` | bounds | `0` (all) | Limit benchmark count |
| `SABENCH_BOUNDS_MAX_TRANSFORMS` | bounds | `0` (all) | Limit transform count |
| `SABENCH_BOUNDS_OUTPUT_DIR` | bounds | `outputs/bounds_theorem_grid_analysis` | Output CSV directory |

### Output artifacts

Outputs are written to `outputs/` (gitignored) under the repo root.

**Noncommutativity notebook** writes:
- `outputs/noncommutativity_grid_analysis/pair_status.csv` — every candidate pair with status
- `outputs/noncommutativity_grid_analysis/noncommutativity_metrics.csv` — metrics for computed pairs
- `outputs/noncommutativity_grid_analysis/summary_by_transform.csv` — per-transform aggregates
- `outputs/noncommutativity_grid_analysis/summary_by_benchmark.csv` — per-benchmark aggregates

**Bounds notebook** writes:
- `outputs/bounds_theorem_grid_analysis/bounds_pair_status.csv` — every candidate pair with bounds status
- `outputs/bounds_theorem_grid_analysis/taylor_reference_results.csv` — Taylor-reference diagnostics
- `outputs/bounds_theorem_grid_analysis/local_affine_results.csv` — local-affine diagnostics
- `outputs/bounds_theorem_grid_analysis/bounds_summary.csv` — status counts

## Citation

If you use sabench in academic work, please cite:

```bibtex
@article{hettinger2026sabench,
  title   = {sabench: A benchmark suite for variance-based global sensitivity analysis},
  author  = {Hettinger, Dylan},
  journal = {Journal of Open Source Software},
  year    = {2026},
  doi     = {10.21105/joss.XXXXX}
}
```

## License

MIT — see [LICENSE](LICENSE).
