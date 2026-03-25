"""
sabench — Benchmark functions for sensitivity analysis
======================================================

A curated library of scalar, spatial, and functional benchmark functions
for testing and validating variance-based global sensitivity analysis
methods (Sobol indices, Morris screening, etc.).

Quick start
-----------
>>> from sabench.scalar import Ishigami
>>> from sabench.spatial import Campbell2D
>>> from sabench.sampling import saltelli_sample
>>> from sabench.analysis import jansen_s1_st
>>> from sabench.transforms import TRANSFORMS, POINTWISE_TRANSFORMS
>>>
>>> model = Ishigami()
>>> X = saltelli_sample(model.d, model.bounds, N=2048, seed=0)
>>> Y = model.evaluate(X)
>>> S1, ST = jansen_s1_st(Y, N=2048, d=model.d)
"""

from sabench._base import BenchmarkFunction
from sabench.analysis import first_order, jansen_s1_st, total_effect
from sabench.functional import (
    BoussinesqRecession,
    DampedOscillator,
    EpidemicSIR,
    HeatDiffusion1D,
    Lorenz96,
    LotkaVolterra,
    TwoCompartmentPK,
)
from sabench.sampling import saltelli_sample
from sabench.scalar import (
    AdditiveQuadratic,
    Borehole,
    CornerPeak,
    CSTRReactor,
    DetPep8D,
    EnvironModel,
    Friedman,
    Ishigami,
    LinearModel,
    MoonHerrera,
    Morris,
    OakleyOHagan,
    OTLCircuit,
    PCETestFunction,
    Piston,
    ProductPeak,
    Rosenbrock,
    SobolG,
    WingWeight,
)
from sabench.spatial import Campbell2D, Campbell3D, ExponentialCampbell2D
from sabench.transforms import (
    CONCAVE_TRANSFORMS,
    CONVEX_TRANSFORMS,
    LINEAR_TRANSFORMS,
    MONOTONE_TRANSFORMS,
    NONLOCAL_TRANSFORMS,
    NONMONOTONE_TRANSFORMS,
    NONSMOOTH_TRANSFORMS,
    POINTWISE_TRANSFORMS,
    SMOOTH_TRANSFORMS,
    TRANSFORMS,
    apply_transform,
    score_transform,
)

__version__ = "0.3.0"

__all__ = [
    # Original scalar benchmarks
    "Ishigami",
    "SobolG",
    "Borehole",
    "Piston",
    "WingWeight",
    "OTLCircuit",
    "Morris",
    "LinearModel",
    # New scalar benchmarks
    "AdditiveQuadratic",
    "PCETestFunction",
    "Friedman",
    "OakleyOHagan",
    "MoonHerrera",
    "CornerPeak",
    "ProductPeak",
    "Rosenbrock",
    "EnvironModel",
    "CSTRReactor",
    "DetPep8D",
    # Spatial benchmarks
    "Campbell2D",
    "ExponentialCampbell2D",
    "Campbell3D",
    # Functional benchmarks
    "BoussinesqRecession",
    "DampedOscillator",
    "LotkaVolterra",
    "EpidemicSIR",
    "HeatDiffusion1D",
    "Lorenz96",
    "TwoCompartmentPK",
    # Utilities
    "saltelli_sample",
    "jansen_s1_st",
    "first_order",
    "total_effect",
    "BenchmarkFunction",
    # Transforms
    "TRANSFORMS",
    "POINTWISE_TRANSFORMS",
    "LINEAR_TRANSFORMS",
    "NONLOCAL_TRANSFORMS",
    "CONVEX_TRANSFORMS",
    "CONCAVE_TRANSFORMS",
    "MONOTONE_TRANSFORMS",
    "NONMONOTONE_TRANSFORMS",
    "SMOOTH_TRANSFORMS",
    "NONSMOOTH_TRANSFORMS",
    "apply_transform",
    "score_transform",
    "__version__",
]
