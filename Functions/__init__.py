"""
functions — MPDR Beamformer toolkit.

Modules
-------
steering : Steering vector computation for arbitrary array geometries.
dsp      : Signal generation, adaptive filtering, and consensus algorithms.
mpdr     : MPDR/Capon beamformer solvers (theoretical, estimated, distributed).
plotter  : Beam pattern visualization utilities.
"""

from . import dsp
from . import mpdr
from . import plotter
from .steering import steering_vector

__all__ = ["dsp", "mpdr", "plotter", "steering_vector"]
