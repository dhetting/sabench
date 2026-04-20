---
title: 'sabench: A benchmark suite for variance-based global sensitivity analysis'
tags:
  - Python
  - sensitivity analysis
  - Sobol indices
  - uncertainty quantification
  - benchmark functions
  - output transformations
authors:
  - name: Dylan Hettinger
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 24 March 2026
bibliography: paper.bib
---

# Summary

Variance-based global sensitivity analysis (GSA) decomposes the variance of a model
output $Y = f(X_1,\ldots,X_d)$ into contributions from individual inputs
[@saltelli2008primer]. The Sobol first-order index $S_i = \text{Var}_{X_i}[E[Y|X_i]] /
\text{Var}[Y]$ and the total-effect index $S_i^T$ are the canonical measures
[@sobol1993sensitivity; @jansen1999estimation]. A persistent practical challenge is that
GSA practitioners routinely transform model outputs before analysis—applying log
transforms for skewed flood flows, exceedance indicators for risk thresholds, or spatial
block averages for coarse-resolution assessments—yet the effect of these transformations
on the resulting indices is rarely quantified systematically.

`sabench` addresses this gap by providing a curated, extensible Python library that
unifies (1) a benchmark suite of well-characterised test functions with (2) a rich
catalogue of 172 output transformations, (3) exhaustive transformation metadata, and
(4) two bounded non-commutativity metrics for measuring how much a transformation
changes the sensitivity structure.

# Statement of need

Several open-source Python libraries implement GSA estimators (SALib [@herman2017salib],
OpenTURNS [@baudin2017openturns]), but none provide a dedicated benchmark-plus-transform
infrastructure for systematically studying how output transformations affect index values.
Benchmark function collections exist in isolation (e.g., as supplementary material to
papers), but they lack standardised interfaces, metadata, or integration with
transformation pipelines.

`sabench` fills this niche by offering:

1. **A benchmark registry** — 19 scalar, 7 functional, and 3 spatial benchmarks with a
   uniform `evaluate(X)` interface and analytical $S_1$ where derivable.
2. **A transformation registry** — 172 transforms in 12 scientific categories, each with
   boolean property flags (convexity, monotonicity, differentiability class, symmetry),
   numerical metadata (expected rank-change magnitude, tail behaviour), and full
   literature references.
3. **Metadata JSON files** — machine-readable metadata for every benchmark and
   transformation, designed for downstream regression modelling of transformation impact.
4. **Non-commutativity metrics** — the Decision Score $D \in [0,1]$ (soft sigmoid
   threshold margin) and Sensitivity Shift $\Delta \in [0,1]$ (Bray-Curtis
   dissimilarity), bounded and cross-benchmark-comparable.

Together these components enable researchers to: (a) validate new GSA estimators on
standard test functions; (b) systematically characterise how transformation properties
(e.g., convexity, differentiability class) predict index redistribution; and (c) build
regression models that analytically and empirically map transformation metadata to
observed $S_1$ changes.

# Design and implementation

`sabench` is pure Python 3.10+ with NumPy [@harris2020array] and SciPy
[@virtanen2020scipy] as the only runtime dependencies. The package is organised into
five submodules:

- `sabench.benchmarks.scalar` — scalar test functions inheriting `BenchmarkFunction`, each
  implementing `evaluate(X)`, `bounds`, `d`, and optionally `analytical_S1()`.
- `sabench.functional` — PDE/ODE-based models producing vector or field outputs.
- `sabench.spatial` — 2-D and 3-D spatial random field benchmarks.
- `sabench.transforms` — the 172-transform registry with property classification sets
  (`CONVEX_TRANSFORMS`, `MONOTONE_TRANSFORMS`, etc.) and the `apply_transform` /
  `score_transform` API.
- `sabench.analysis` — the Jansen-Saltelli estimator with variance-weighted aggregation
  for multi-output fields.

**Non-commutativity metrics.** The Decision Score is motivated by Proposition 1 of
[@hettinger2026noncommutativity]: affine transforms commute with Sobol indices; all
strictly nonlinear maps do not. For a transform $\phi$, $D$ measures the mean per-input
sigmoid displacement across the keep/discard boundary at threshold $\tau$:

$$D = \frac{1}{d}\sum_{i=1}^{d}\left|\sigma\!\left(\frac{\hat{S}_i(\phi(Y))-\tau}{\tau}\right) - \sigma\!\left(\frac{\hat{S}_i(Y)-\tau}{\tau}\right)\right|$$

The Sensitivity Shift $\Delta$ is the Bray-Curtis dissimilarity [@bray1957ordination]
between the transformed and original index vectors, which avoids the divergence of
relative $\ell_2$ distance when indices are near zero.

# Benchmarks and transforms

The benchmark suite covers a range of sensitivity structures: product-form functions
with analytical indices (SobolG, Ishigami), additive functions (AdditiveQuadratic,
Friedman), high-dimensional engineering models (Borehole, Piston, WingWeight),
polynomial chaos test functions (PCETestFunction, OakleyOHagan), and dynamical systems
(Lorenz-96, SIR epidemic, two-compartment pharmacokinetics).

The 172 transformations span:
(a) purely mathematical families for systematic mathematical-property sweeps
    (polynomial y⁴–y⁶, orthogonal polynomials, neural activations, oscillatory,
     threshold, variance-stabilising, curvature extremes);
(b) domain-specific real-world transformations from hydrology, extreme-value theory,
    financial risk, ecology, climate science, pharmacology, and structural engineering.

Each transform entry carries 35+ metadata fields, making the full dataset suitable as
a feature matrix for regression modelling of the mapping:
*transformation properties* $\to$ *$\Delta S_1$ statistics*.

# Acknowledgements

The author thanks the developers of SALib [@herman2017salib] and NumPy
[@harris2020array] for foundational infrastructure.

# References
