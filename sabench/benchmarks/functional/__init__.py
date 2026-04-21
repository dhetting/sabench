from .boussinesq import BoussinesqRecession
from .damped_oscillator import DampedOscillator
from .epidemic_sir import EpidemicSIR
from .heat_diffusion import HeatDiffusion1D
from .lorenz96 import Lorenz96
from .lotka_volterra import LotkaVolterra
from .pharmacokinetics import TwoCompartmentPK

__all__ = [
    "DampedOscillator",
    "LotkaVolterra",
    "BoussinesqRecession",
    "EpidemicSIR",
    "HeatDiffusion1D",
    "Lorenz96",
    "TwoCompartmentPK",
]
