from __future__ import annotations

from typing import get_type_hints

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction
from sabench.benchmarks.scalar.oakley_ohagan import OakleyOHagan


def test_benchmark_sample_seed_annotation_allows_none() -> None:
    seed_hint = get_type_hints(BenchmarkFunction.sample)["seed"]
    assert seed_hint == int | None


def test_oakley_ohagan_analytical_s1_annotation_allows_none() -> None:
    return_hint = get_type_hints(OakleyOHagan.analytical_S1)["return"]
    assert return_hint == np.ndarray | None
