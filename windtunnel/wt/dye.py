"""dye.py - passive scalar transport, as a D2Q5 lattice riding on the flow solver.

WHY A SECOND LATTICE RATHER THAN JUST MOVING A SCALAR ARROUND
-------------------------------------------------------------
The tempting cheap option is semi-Lagrangian advection: trace each cell back along the
velocity field and interpolate. It is one line and it is wrong in a specific, visible way -
the interpolation is a low-pass filter applied every step, so a dye filament smears out over a
few hundred steps and a clip long enough to be worth watching ends as uniform fog. The
diffusion you see is the interpolation, not the fluid, and it cannot be turned down because it
is not a parameter.

A D2Q5 advection-diffusion lattice has the same structure as the flow solver: stream exactly
(a shift, no interpolation, no loss) and collide locally, with the diffusivity set by a
relaxation time you actually choose. A filament stays a filament until the FLOW tears it up,
which is the entire point of putting dye in a tunnel.

D2Q5 rather than D2Q9 because a scalar has no momentum flux to represent - the extra diagonal
populations buy nothing here and cost 80% more memory traffic in a kernel that is already
bandwidth-bound.

MODEL
-----
  weights   w0 = 1/3 at rest, 1/6 on each of the four axial links
  equilibrium  g_k^eq = w_k C (1 + 3 c_k . u)   - linear in u, because a passive scalar has no
               self-advection: it is carried by the flow's u, never by its own gradient
  diffusivity  D = (tau_c - 1/2)/3, exactly as nu = (tau - 1/2)/3 in the flow lattice

Peclet number Pe = U L / D is the knob that matters. High Pe (tau_c near 0.5) keeps filaments
sharp and is what you want visually; too high and the scalar goes unstable in the same way BGK
does, so `tau_c` is floored.
"""
from __future__ import annotations

import numpy as np

from .gpu import asnumpy, to_device, xp

# D2Q5: rest, +x, +y, -x, -y
DX = np.array([0, 1, 0, -1, 0], dtype=np.int32)
DY = np.array([0, 0, 1, 0, -1], dtype=np.int32)
DW = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=np.float32)
DOPP = np.array([0, 3, 4, 1, 2], dtype=np.int32)

TAU_FLOOR = 0.505


class Dye:
    """A passive scalar advected and diffused by a flow solver's velocity field.

    Parameters
    ----------
    nx, ny      lattice size, matching the flow solver
    diffusivity lattice diffusivity D. Small keeps filaments sharp; `tau_c` is floored at
                0.505 so an over-eager value degrades gracefully instead of exploding.
    n_species   how many independent dyes to carry. Several colours cost one lattice each but
                share the velocity field, so a second colour is much cheaper than a second run.
    """

    def __init__(self, nx, ny, diffusivity=0.002, n_species=1):
        self.nx, self.ny = int(nx), int(ny)
        self.n = int(n_species)
        self.D = float(diffusivity)
        self.tau = max(3.0 * self.D + 0.5, TAU_FLOOR)
        self.omega = 1.0 / self.tau

        f32 = xp.float32
        self.dx = to_device(DX.reshape(5, 1, 1))
        self.dy = to_device(DY.reshape(5, 1, 1))
        self.w = to_device(DW.reshape(5, 1, 1))
        self._opp = xp.asarray(DOPP)

        # (species, 5, ny, nx)
        self.g = xp.zeros((self.n, 5, self.ny, self.nx), dtype=f32)
        self.C = xp.zeros((self.n, self.ny, self.nx), dtype=f32)
        self._sources = []          # (species, mask, value, rate)

    # -- setup -----------------------------------------------------------------------------
    def add_source(self, species, mask, value=1.0, rate=0.25):
        """Relax the concentration toward `value` inside `mask` at `rate` per step.

        Relaxation rather than assignment. Hard-setting a block of cells to 1.0 every step is a
        discontinuity the flow then has to absorb: the dye appears as a rectangle with sharp
        corners that shed their own little vortices. Relaxing lets the injected patch take the
        shape the flow gives it.
        """
        self._sources.append((int(species), to_device(np.asarray(asnumpy(mask), np.float32)),
                              float(value), float(rate)))

    def inject_line(self, species, x, y0=None, y1=None, width=2, value=1.0, rate=0.25):
        """A vertical dye rake at column `x` - the classic tunnel streakline source."""
        m = np.zeros((self.ny, self.nx), np.float32)
        j0 = 0 if y0 is None else int(y0)
        j1 = self.ny if y1 is None else int(y1)
        m[j0:j1, int(x):int(x) + int(width)] = 1.0
        self.add_source(species, m, value=value, rate=rate)
        return m

    def inject_bands(self, species, x, n_bands=9, duty=0.45, width=2, value=1.0, rate=0.25):
        """A comb of horizontal bands - streaklines that show shear directly.

        A solid rake fills the whole wake and reads as a cloud. Bands leave gaps, and the gaps
        are what let you see one filament move relative to its neighbour, which is where shear
        and roll-up become legible rather than merely present.
        """
        m = np.zeros((self.ny, self.nx), np.float32)
        pitch = self.ny / float(n_bands)
        for k in range(n_bands):
            j0 = int(k * pitch)
            j1 = int(j0 + pitch * duty)
            m[j0:j1, int(x):int(x) + int(width)] = 1.0
        self.add_source(species, m, value=value, rate=rate)
        return m

    def seed(self, species, field):
        """Set the initial concentration directly (equilibrium at rest)."""
        c = to_device(np.asarray(asnumpy(field), np.float32))
        self.C[species] = c
        self.g[species] = self.w * c

    # -- step --------------------------------------------------------------------------------
    def step(self, ux, uy, solid, substeps=1):
        """Advance the dye using the flow solver's CURRENT velocity field.

        The velocity is read once and reused across `substeps`. That is exact when the flow is
        stepped the same number of times with a frozen field, and a good approximation
        otherwise - the dye is a diagnostic overlay, and re-fetching per substep costs more
        than the error it removes.
        """
        u = xp.asarray(ux)
        v = xp.asarray(uy)
        fluid = ~xp.asarray(solid)
        for _ in range(int(substeps)):
            for s in range(self.n):
                g = self.g[s]
                C = xp.maximum(g.sum(axis=0), 0.0)
                # linear equilibrium: a passive scalar is carried by u, it does not carry itself
                geq = self.w * C * (1.0 + 3.0 * (self.dx * u + self.dy * v))
                gout = g - self.omega * (g - geq)

                for src_s, mask, val, rate in self._sources:
                    if src_s != s:
                        continue
                    dC = rate * mask * (val - C)
                    gout = gout + self.w * dC

                # zero-flux (no-penetration) at solids: reflect rather than absorb, or the body
                # slowly eats the dye and the wake fades for a reason that is not the flow
                back = gout[self._opp]
                gout = xp.where(xp.asarray(solid)[None], back, gout)

                for k in range(1, 5):
                    gout[k] = xp.roll(gout[k], (int(DY[k]), int(DX[k])), axis=(0, 1))

                self.g[s] = gout
                self.C[s] = xp.where(fluid, xp.maximum(gout.sum(axis=0), 0.0), 0.0)

    # -- output --------------------------------------------------------------------------------
    def concentration(self, species=0):
        return self.C[species]

    def composite(self, rgb, colors=None, gain=1.0, gamma=0.75):
        """Paint the dye over an existing RGB frame.

        SCREEN blending, not replacement: dye is something added to the light already there, so
        a filament over a bright region should brighten it rather than punch a hole. Replacing
        makes the dye look like a decal sitting on top of the picture instead of in it.
        """
        if colors is None:
            colors = [(90, 200, 255), (255, 140, 90), (150, 255, 160), (255, 220, 120)]
        out = np.asarray(rgb, dtype=np.float32)
        for s in range(self.n):
            c = np.asarray(asnumpy(self.C[s]), dtype=np.float32)
            a = np.clip(c * gain, 0.0, 1.0) ** gamma
            col = np.asarray(colors[s % len(colors)], dtype=np.float32)
            out = 255.0 - (255.0 - out) * (1.0 - a[..., None] * (col / 255.0))
        return np.clip(out, 0, 255).astype(np.uint8)

    def describe(self):
        return (f"D2Q5 dye | {self.n} species | D={self.D:.5f} | tau_c={self.tau:.4f}"
                + ("  (FLOORED - diffusivity request was too low)"
                   if self.tau <= TAU_FLOOR + 1e-9 else ""))
