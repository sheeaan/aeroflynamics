"""colormap.py - scalar field -> RGB.

Anchor tables are sampled from the matplotlib originals and interpolated at build time into a
256-entry LUT, so mapping a field is one clip, one cast and one fancy-index rather than a
per-pixel evaluation. Nothing here imports matplotlib.

The sequential maps (viridis, inferno) are perceptually uniform and safe for the common forms
of colour-blindness; `coolwarm` is the diverging map, used only for signed fields where zero
is meaningful - vorticity and gauge pressure - and it is always normalised SYMMETRICALLY so
that the neutral colour lands exactly on zero. A diverging map with an off-centre midpoint is
actively misleading: it reads as though the flow has a sign it does not have.
"""
from __future__ import annotations

import numpy as np

from .gpu import asnumpy

_ANCHORS = {
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
                (31, 158, 137), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)],
    "inferno": [(0, 0, 4), (22, 11, 57), (66, 10, 104), (106, 23, 110), (147, 38, 103),
                (188, 55, 84), (221, 81, 58), (243, 120, 25), (252, 165, 10), (252, 255, 164)],
    "magma": [(0, 0, 4), (24, 15, 62), (68, 15, 118), (114, 31, 129), (158, 47, 127),
              (205, 64, 113), (240, 96, 93), (253, 150, 104), (254, 202, 141), (252, 253, 191)],
    "coolwarm": [(59, 76, 192), (98, 130, 234), (141, 176, 254), (184, 208, 249),
                 (221, 221, 221), (245, 196, 173), (244, 154, 123), (222, 96, 77),
                 (180, 4, 38)],
    "gray": [(0, 0, 0), (255, 255, 255)],
}


def build_lut(name, n=256):
    """Anchor list -> (n, 3) uint8 lookup table."""
    a = np.asarray(_ANCHORS[name], dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(a))
    dst = np.linspace(0.0, 1.0, int(n))
    out = np.stack([np.interp(dst, src, a[:, c]) for c in range(3)], axis=1)
    return np.clip(out, 0, 255).astype(np.uint8)


_LUTS = {k: build_lut(k) for k in _ANCHORS}


def available():
    return sorted(_LUTS)


def colorize(field, cmap="viridis", vmin=None, vmax=None, symmetric=False, gamma=1.0):
    """Scalar field -> (ny, nx, 3) uint8.

    vmin/vmax default to the field's own range. `symmetric=True` forces vmin = -vmax, which
    is what any diverging map needs. `gamma` < 1 lifts the low end, which is how a speed field
    keeps its wake visible without blowing out the accelerated region over the suction side.
    """
    a = np.asarray(asnumpy(field), dtype=np.float32)
    if symmetric:
        lim = float(vmax) if vmax is not None else float(np.abs(a).max())
        lim = max(lim, 1e-12)
        lo, hi = -lim, lim
    else:
        lo = float(np.nanmin(a)) if vmin is None else float(vmin)
        hi = float(np.nanmax(a)) if vmax is None else float(vmax)
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    t = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        t = t ** float(gamma)
    lut = _LUTS[cmap]
    idx = (t * (len(lut) - 1) + 0.5).astype(np.int32)
    return lut[idx]


def overlay_mask(rgb, mask, color=(16, 17, 20)):
    """Paint the body silhouette flat over an already-coloured frame."""
    m = np.asarray(asnumpy(mask), dtype=bool)
    out = rgb.copy()
    out[m] = np.asarray(color, dtype=np.uint8)
    return out


def outline_mask(rgb, mask, color=(235, 238, 245), width=1):
    """Draw a thin rim just OUTSIDE the body, so the silhouette reads against a dark field."""
    m = np.asarray(asnumpy(mask), dtype=bool)
    grown = m.copy()
    for _ in range(max(1, int(width))):
        g = grown.copy()
        g[1:, :] |= grown[:-1, :]
        g[:-1, :] |= grown[1:, :]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    rim = grown & ~m
    out = rgb.copy()
    out[rim] = np.asarray(color, dtype=np.uint8)
    return out
