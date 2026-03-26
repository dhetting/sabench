"""Representative pointwise transform implementations extracted from the monolith."""

from __future__ import annotations

import numpy as np


def t_affine(Y, a=2.0, b=1.0):
    """Affine (linear) pointwise transform: Z(z) = a · Y(z) + b.

    This is the CANONICAL COMMUTATIVE case: an affine map φ(y) = a·y + b
    satisfies φ ∘ E[·|X_i] = E[φ(·)|X_i] and so preserves Sobol indices exactly.

    Specifically:
      Var[a·Y+b] = a² · Var[Y]
      Var_{X_i}[E[a·Y+b | X_i]] = a² · Var_{X_i}[E[Y|X_i]]
      → S_i(a·Y+b) = S_i(Y)  for all i, all models, all locations.

    This transform is the negative control / positive case for the biconditional
    theorem: it demonstrates that Sobol indices are invariant under affine maps,
    in contrast to all strictly nonlinear pointwise maps.

    Physical contexts
    -----------------
    Unit conversion: converting metres to feet (a=3.281, b=0), Celsius to
      Fahrenheit (a=9/5, b=32), or pascals to psi.
    Baseline shift: adding a fixed background level b to a model output.
    Amplitude scaling: rescaling a signal by a constant gain factor a.

    Parameters
    ----------
    a : float  Scale factor (a ≠ 0 required for invertibility; default 2.0).
    b : float  Offset (default 1.0).
    """
    return a * Y + b


def t_tanh_pointwise(Y, alpha=0.3):
    """Hyperbolic tangent saturation: Z(z) = tanh(α · Y(z)).

    This is a GENUINELY POINTWISE transform: φ(y) = tanh(α·y) depends only
    on the scalar value y, with no reference to any other location z' or to
    any per-sample statistics (_ymin, _safe_range are NOT called).  It is the
    canonical example for demonstrating Sobol index noncommutativity under a
    pointwise nonlinear map.

    Physical contexts
    -----------------
    Neural fields: Amari (1977) and Wilson-Cowan (1972) use Z(z) = tanh(α·Y(z))
      as the local firing-rate function of the synaptic potential field Y(z).
    Receptor/enzyme kinetics: the symmetric Michaelis-Menten approximation maps
      a spatial concentration field through a saturating occupancy curve.
    Sensor saturation: bounded digital readouts of a physical signal Y with
      finite dynamic range.

    Notes
    -----
    α controls the effective nonlinearity relative to the output range.
    Smaller α → near-linear regime (small index shifts).
    Larger α → strong saturation (large flips possible).
    α=0.3 paired with Campbell2D (U[0.5,15] amplitudes) produces robust
    threshold flips for the rate inputs X3, X4 while preserving activity
    structure across the field.
    """
    return np.tanh(alpha * Y)


def t_softplus_pointwise(Y, beta=0.1):
    """Softplus (smooth ReLU): Z(z) = log(1 + exp(β · Y(z))) / β.

    A GENUINELY POINTWISE smooth monotone transform.  In the limit β → ∞
    this approaches ReLU; for moderate β it is a smooth bounded-curvature
    alternative to tanh that does not saturate from above.  Used in neural
    network activations and dose-response models where only the lower
    (near-zero) regime is nonlinear.

    Physical context: Softplus arises as the log-partition function in
    statistical mechanics and as a smooth activation function in deep
    learning systems where the output represents a probability or intensity.
    """
    return np.log1p(np.exp(np.clip(beta * Y, -500, 500))) / beta


__all__ = ["t_affine", "t_tanh_pointwise", "t_softplus_pointwise"]
