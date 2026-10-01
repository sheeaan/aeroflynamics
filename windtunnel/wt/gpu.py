"""gpu.py - the array-module shim.

Every hot array op in the solver goes through `xp`, which is CuPy when it is installed and
NumPy otherwise. The solver never imports either directly, so the same source runs on both
without a branch anywhere in the physics.

Setup-time work (building a turbulence patch, rasterising a polygon) deliberately stays on
the host in NumPy/SciPy and is transferred once: it happens a few times per run, not a few
thousand times per frame, and keeping it host-side means SciPy is always available for it.
"""
from __future__ import annotations

import numpy as _np

try:                                    # pragma: no cover - depends on the machine
    import cupy as _cp

    xp = _cp
    GPU = True
except Exception:                       # ImportError, or a CUDA runtime that fails to init
    xp = _np
    GPU = False


def asnumpy(a):
    """Bring an array back to the host, whichever module it came from."""
    if GPU:
        return _cp.asnumpy(a)
    return _np.asarray(a)


def to_device(a, dtype=_np.float32):
    """Host array -> device array of `dtype`."""
    return xp.asarray(_np.ascontiguousarray(a, dtype=dtype))


def describe_backend():
    return "cupy (GPU)" if GPU else "numpy (CPU)"
