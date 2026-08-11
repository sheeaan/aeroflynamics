"""wt - a 2-D wind tunnel: D2Q9 lattice-Boltzmann, polygon bodies, video out.

Nothing in this package is animated by hand. A scene sets a body's position and incidence at
time t and then lets go; every vortex, every separation, every pressure fluctuation on screen
is the solver's output.
"""
from .lbm import LBM, equilibrium
from .gpu import GPU, asnumpy, describe_backend, xp

__all__ = ["LBM", "equilibrium", "GPU", "asnumpy", "describe_backend", "xp"]
__version__ = "0.1.0"
