"""Canonical typed registry for sabench benchmark definitions."""

from __future__ import annotations

from sabench.benchmarks.base import BenchmarkFunction
from sabench.benchmarks.functional import (
    BoussinesqRecession,
    DampedOscillator,
    EpidemicSIR,
    HeatDiffusion1D,
    Lorenz96,
    LotkaVolterra,
    TwoCompartmentPK,
)
from sabench.benchmarks.scalar import (
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
from sabench.benchmarks.spatial import Campbell2D, Campbell3D, ExponentialCampbell2D
from sabench.benchmarks.types import BenchmarkFamily, BenchmarkSpec, OutputKind

_LEGACY_OUTPUT_KIND_MAP: dict[str, OutputKind] = {
    "scalar": OutputKind.SCALAR,
    "spatial_2d": OutputKind.SPATIAL,
    "spatial_3d": OutputKind.SPATIAL,
    "functional": OutputKind.FUNCTIONAL,
}


def _normalize_output_kind(benchmark_cls: type[BenchmarkFunction]) -> OutputKind:
    legacy_output_type = getattr(benchmark_cls, "output_kind", None)
    if legacy_output_type is None:
        legacy_output_type = getattr(benchmark_cls, "output_type", None)

    if legacy_output_type not in _LEGACY_OUTPUT_KIND_MAP:
        msg = (
            f"Benchmark class {benchmark_cls.__module__}.{benchmark_cls.__name__} "
            f"declares unsupported output label {legacy_output_type!r}."
        )
        raise ValueError(msg)

    return _LEGACY_OUTPUT_KIND_MAP[legacy_output_type]


def _spec(
    benchmark_cls: type[BenchmarkFunction],
    family: BenchmarkFamily,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        key=benchmark_cls.__name__,
        benchmark_cls=benchmark_cls,
        family=family,
        output_kind=_normalize_output_kind(benchmark_cls),
        module=benchmark_cls.__module__,
    )


_BENCHMARK_SPECS = (
    _spec(Ishigami, BenchmarkFamily.SCALAR),
    _spec(SobolG, BenchmarkFamily.SCALAR),
    _spec(Borehole, BenchmarkFamily.SCALAR),
    _spec(Piston, BenchmarkFamily.SCALAR),
    _spec(WingWeight, BenchmarkFamily.SCALAR),
    _spec(OTLCircuit, BenchmarkFamily.SCALAR),
    _spec(Morris, BenchmarkFamily.SCALAR),
    _spec(LinearModel, BenchmarkFamily.SCALAR),
    _spec(AdditiveQuadratic, BenchmarkFamily.SCALAR),
    _spec(PCETestFunction, BenchmarkFamily.SCALAR),
    _spec(Friedman, BenchmarkFamily.SCALAR),
    _spec(OakleyOHagan, BenchmarkFamily.SCALAR),
    _spec(MoonHerrera, BenchmarkFamily.SCALAR),
    _spec(CornerPeak, BenchmarkFamily.SCALAR),
    _spec(ProductPeak, BenchmarkFamily.SCALAR),
    _spec(Rosenbrock, BenchmarkFamily.SCALAR),
    _spec(EnvironModel, BenchmarkFamily.SCALAR),
    _spec(CSTRReactor, BenchmarkFamily.SCALAR),
    _spec(DetPep8D, BenchmarkFamily.SCALAR),
    _spec(Campbell2D, BenchmarkFamily.SPATIAL),
    _spec(ExponentialCampbell2D, BenchmarkFamily.SPATIAL),
    _spec(Campbell3D, BenchmarkFamily.SPATIAL),
    _spec(BoussinesqRecession, BenchmarkFamily.FUNCTIONAL),
    _spec(DampedOscillator, BenchmarkFamily.FUNCTIONAL),
    _spec(LotkaVolterra, BenchmarkFamily.FUNCTIONAL),
    _spec(EpidemicSIR, BenchmarkFamily.FUNCTIONAL),
    _spec(HeatDiffusion1D, BenchmarkFamily.FUNCTIONAL),
    _spec(Lorenz96, BenchmarkFamily.FUNCTIONAL),
    _spec(TwoCompartmentPK, BenchmarkFamily.FUNCTIONAL),
)


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {spec.key: spec for spec in _BENCHMARK_SPECS}
if len(BENCHMARK_SPECS) != len(_BENCHMARK_SPECS):
    raise ValueError("Benchmark registry contains duplicate keys.")


def get_benchmark_spec(key: str) -> BenchmarkSpec:
    """Return the canonical typed spec for a benchmark key."""

    return BENCHMARK_SPECS[key]


def get_benchmark_class(key: str) -> type[BenchmarkFunction]:
    """Return the benchmark class registered under ``key``."""

    return get_benchmark_spec(key).benchmark_cls


def list_benchmark_names() -> list[str]:
    """Return benchmark keys in deterministic sorted order."""

    return sorted(BENCHMARK_SPECS)
