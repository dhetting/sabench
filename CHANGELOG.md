# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2026-03-24

### Added
- **172 output transforms** (expanded from 85), spanning 12 scientific categories:
  polynomial family (y⁴–y⁶, signed powers, Legendre P3, Chebyshev T4, Hermite
  He2/He3, Bernstein B3), neural activation functions (Swish, Mish, SELU, softsign,
  bent identity, hard sigmoid/tanh), oscillatory family (sinc, sin², cos², damped
  sine, sawtooth, square wave, double-sin), threshold/piecewise family (soft/hard
  threshold, ramp, spike, breakpoint, hockey stick, deadzone, bimodal flip, donut),
  variance-stabilising transforms (Anscombe, Freeman-Tukey, arcsinh/IHS, modulus,
  dual power, log₂, log₁₀), extreme-value family (GEV CDF, Pareto tail,
  log-logistic), financial (VaR, CVaR/ES, Sharpe, drawdown, fold-change), ecological
  (Hellinger, chord, relative abundance, log-ratio), climate (anomaly %, bias
  correction, quantile delta mapping, GDD, SPI), hydrological (Nash-Sutcliffe,
  log-POT, log-streamflow), medical (Hill response, Emax model, log-AUC),
  structural engineering (von Mises, safety factor, Miner damage, Basquin S-N),
  statistical summary (variance, skewness, kurtosis, percentiles, IQR),
  information-theoretic (negentropy, Wasserstein proxy, energy distance, Rényi
  entropy H₂).
- **Exhaustive transform metadata** (`transforms_metadata.json`, 172 entries, 35+
  fields each): LaTeX formula, domain/range, all property flags, differentiability
  class, derivative sign changes, symmetry, physical motivation, canonical use case,
  full references with DOIs.
- **Benchmark metadata** (`benchmarks_metadata.json`, 29 entries) with analytical
  S₁ availability, input distributions, interaction order, sensitivity structure.
- **`test_transforms_expanded.py`**: 77 new tests covering all 87 new transforms,
  with numerical verification of boundedness, monotonicity, convexity (midpoint
  inequality), symmetry, and mathematical identities.
- Property sets updated: `CONVEX_TRANSFORMS` (23), `CONCAVE_TRANSFORMS` (30),
  `MONOTONE_TRANSFORMS` (56), `SMOOTH_TRANSFORMS` (90), `NONSMOOTH_TRANSFORMS` (27),
  `POINTWISE_TRANSFORMS` (74), `NONLOCAL_TRANSFORMS` (98).
- `pixi.toml` for reproducible environment management.
- `.pre-commit-config.yaml` with ruff lint/format, mypy, nbstripout, file hygiene.
- `CITATION.cff` and `zenodo.json` for citation metadata.
- `docs/paper/paper.md` JOSS submission skeleton.
- Demonstration Jupyter notebook (`notebooks/demo.ipynb`).

### Fixed
- `SobolG.analytical_S1`: corrected formula from `D_i * prod_rest / Var` to
  `D_i / Var` (sum was 1.07; now correctly < 1.0).
- `TwoCompartmentPK`: fixed NumPy broadcasting shape error `(D/V1[:,None])` →
  `(D/V1)[:,None]`.
- `HeatDiffusion1D`: corrected test interface (accepts `n_x` and `t_obs`, not `n_t`).
- All test key names: registry uses no `t_` prefix.

---

## [0.2.0] — 2026-01-15

### Added
- 19 scalar benchmarks (Ishigami, Sobol-G, Borehole, Piston, WingWeight, OTLCircuit,
  Morris, LinearModel, AdditiveQuadratic, PCETestFunction, Friedman, OakleyOHagan,
  MoonHerrera, CornerPeak, ProductPeak, Rosenbrock, EnvironModel, CSTRReactor,
  DetPep8D).
- 7 functional benchmarks (BoussinesqRecession, DampedOscillator, LotkaVolterra,
  EpidemicSIR, HeatDiffusion1D, Lorenz96, TwoCompartmentPK).
- 3 spatial benchmarks (Campbell2D, ExponentialCampbell2D, Campbell3D).
- 85 output transforms with property classification sets.
- Jansen-Saltelli estimator with variance-weighted aggregation.
- `score_transform` with Decision Score D and Sensitivity Shift Δ metrics.
- Saltelli quasi-random sampling.

---

## [0.1.0] — 2025-11-01

### Added
- Initial release with Ishigami, SobolG, Borehole, and basic Saltelli sampling.
