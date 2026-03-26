"""Representative linear transform implementations extracted from the monolith."""

from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import flatten_samples


def t_temporal_cumsum(Y):
    """Cumulative sum / running integral: Z(t) = sum_{s<=t} Y(s).

    Arises in hydrology (cumulative runoff volume), pharmacokinetics (area
    under concentration-time curve), and structural fatigue accumulation.
    The cumulative operator is linear but its composition with nonlinear
    subsequent post-processing (e.g. log of the cumulative sum) is nonlinear.
    Here we return the full cumulative trajectory; note this transform is
    LINEAR so it provides a negative control confirming Proposition 1.
    """
    flat = flatten_samples(Y)
    return np.cumsum(flat, axis=1).reshape(np.asarray(Y).shape)


def t_temporal_bandpass(Y, low_frac=0.05, high_frac=0.30):
    """Bandpass filter retaining frequencies in [low_frac, high_frac] * Nyquist.

    Isolates a frequency band of the time series via zero-phase FFT filtering.
    Arises in spectral analysis of climate variables, seismology (body-wave
    extraction), and signal conditioning. The filtering operator is linear in
    the frequency domain but compresses the contributions of inputs that drive
    the suppressed frequency components.
    NOTE: This transform IS linear (convolution), providing another negative
    control alongside cumsum.
    """
    flat = flatten_samples(Y)
    n_t = flat.shape[1]
    freq = np.fft.rfftfreq(n_t)
    fft_Y = np.fft.rfft(flat, axis=1)
    mask = (freq >= low_frac) & (freq <= high_frac)
    fft_Y[:, ~mask] = 0.0
    return np.fft.irfft(fft_Y, n=n_t, axis=1).reshape(np.asarray(Y).shape)


__all__ = ["t_temporal_bandpass", "t_temporal_cumsum"]
