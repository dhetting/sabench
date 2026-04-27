"""Samplewise output transformations."""

from __future__ import annotations

import numpy as np


def t_temporal_cumsum(Y: np.ndarray) -> np.ndarray:
    """Cumulative sum / running integral: Z(t) = sum_{s<=t} Y(s)."""
    flat = Y.reshape(len(Y), -1)
    return np.cumsum(flat, axis=1).reshape(Y.shape)


def t_temporal_log_cumsum(Y, eps=1.0):
    """Log of the cumulative sum: Z(t) = log(sum_{s<=t} Y(s) + eps).

    Nonlinear composed transform of a cumulative integral.  Arises in
    log-transformed streamflow volume analysis and pharmacokinetic
    log-AUC calculations. Combines the nonlinearity of log with the
    time-integration operator.
    """
    flat = Y.reshape(len(Y), -1)
    flat_pos = flat - flat.min(axis=1, keepdims=True) + eps
    return np.log(np.cumsum(flat_pos, axis=1)).reshape(Y.shape)


def t_temporal_exceedance_duration(Y, quantile=0.75):
    """Cumulative duration spent above threshold: Z(t) = sum_{s<=t} 1[Y(s) > c].

    Standard metric in flood frequency analysis (flood duration), ecology
    (duration above thermal stress threshold), and climate science (heat-wave
    duration). Returns a monotone non-decreasing step function; the nonlinearity
    arises from the indicator threshold.
    """
    flat = Y.reshape(len(Y), -1)
    c = np.quantile(flat, quantile, axis=1, keepdims=True)
    above = (flat > c).astype(float)
    return np.cumsum(above, axis=1).reshape(Y.shape)


def t_temporal_envelope(Y):
    """Running maximum envelope: Z(t) = max_{s<=t} Y(s).

    The running maximum envelope of a time series is used in structural
    dynamics (maximum response up to time t), extreme value analysis, and
    signal processing. It is a nonlinear (idempotent, monotone) operator that
    strongly compresses the sensitivity of late-time inputs.
    """
    flat = Y.reshape(len(Y), -1)
    return np.maximum.accumulate(flat, axis=1).reshape(Y.shape)


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
    flat = Y.reshape(len(Y), -1)
    n_t = flat.shape[1]
    freq = np.fft.rfftfreq(n_t)  # [0, 1/(2)]
    fft_Y = np.fft.rfft(flat, axis=1)
    mask = (freq >= low_frac) & (freq <= high_frac)
    fft_Y[:, ~mask] = 0.0
    return np.fft.irfft(fft_Y, n=n_t, axis=1).reshape(Y.shape)
