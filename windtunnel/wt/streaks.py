"""streaks.py - passive tracer particles.

A colour-mapped scalar field shows you where the flow is fast; it does not show you where the
flow is GOING. Tracers do, and they are what makes a separation bubble legible as a bubble
rather than as a smear of low speed - you can watch a particle enter it, orbit, and leave.

Cost is one bilinear sample per particle per step, so tens of thousands are affordable even
on the CPU path. They are purely diagnostic: nothing here feeds back into the solver.

Seeding is UNIFORM IN Y AT THE INLET, not uniform over the domain. Seeding everywhere looks
better for exactly one frame and then decays, because the domain drains downstream faster than
the interior reseeds; feeding a fixed rate through the inlet reaches a genuine steady state.
"""
from __future__ import annotations

import numpy as np

from .gpu import asnumpy


class Streaks:
    """A cloud of passive tracers advected by the solver's velocity field.

    Parameters
    ----------
    n         particle count
    nx, ny    lattice size, so particles know when they have left
    life      mean lifetime in steps before a particle is recycled to the inlet. Finite life
              is what stops the picture silting up: without it every particle eventually ends
              in a recirculation zone or against the body, and the wake goes bald.
    """

    def __init__(self, n, nx, ny, life=900, seed=3, x0=1.0):
        self.n = int(n)
        self.nx, self.ny = int(nx), int(ny)
        self.life = float(life)
        self.x0 = float(x0)
        self.rng = np.random.default_rng(seed)
        # start scattered across the domain so the first frames are not empty, then let the
        # inlet recycling take over
        self.x = self.rng.uniform(0.0, self.nx, self.n).astype(np.float32)
        self.y = self.rng.uniform(0.0, self.ny, self.n).astype(np.float32)
        self.age = self.rng.uniform(0.0, self.life, self.n).astype(np.float32)

    def _sample(self, fld, x, y):
        """Bilinear sample of an (ny, nx) host array at float coords."""
        x = np.clip(x, 0.0, self.nx - 1.001)
        y = np.clip(y, 0.0, self.ny - 1.001)
        i0 = x.astype(np.int32)
        j0 = y.astype(np.int32)
        fx = x - i0
        fy = y - j0
        i1 = i0 + 1
        j1 = j0 + 1
        return (fld[j0, i0] * (1 - fx) * (1 - fy) + fld[j0, i1] * fx * (1 - fy)
                + fld[j1, i0] * (1 - fx) * fy + fld[j1, i1] * fx * fy)

    def advance(self, sim, steps=1):
        """Advect for `steps` solver steps using the CURRENT velocity field.

        The field is fetched once and reused across the sub-steps: it is a diagnostic overlay,
        and re-syncing the device array per sub-step would cost more than the advection does.
        """
        ux = np.asarray(asnumpy(sim.ux), dtype=np.float32)
        uy = np.asarray(asnumpy(sim.uy), dtype=np.float32)
        solid = np.asarray(asnumpy(sim.solid), dtype=bool)
        for _ in range(int(steps)):
            self.x += self._sample(ux, self.x, self.y)
            self.y += self._sample(uy, self.x, self.y)
        self.age += float(steps)

        # recycle: off the end, aged out, or swallowed by the body
        j = np.clip(self.y.astype(np.int32), 0, self.ny - 1)
        i = np.clip(self.x.astype(np.int32), 0, self.nx - 1)
        dead = ((self.x >= self.nx - 2) | (self.x < 0.0)
                | (self.y >= self.ny - 1) | (self.y < 0.0)
                | (self.age > self.life) | solid[j, i])
        k = int(dead.sum())
        if k:
            self.x[dead] = self.x0 + self.rng.uniform(0.0, 2.0, k)
            self.y[dead] = self.rng.uniform(1.0, self.ny - 1.0, k)
            self.age[dead] = self.rng.uniform(0.0, 0.25 * self.life, k)

    def draw(self, rgb, color=(255, 255, 255), alpha=0.55, fade=True):
        """Composite the tracers onto an RGB frame, in place-safe fashion.

        Additive rather than replacing: a tracer over a bright region should read as a
        highlight, not punch a hole in the field behind it.
        """
        out = rgb.astype(np.float32)
        i = np.clip(self.x.astype(np.int32), 0, self.nx - 1)
        j = np.clip(self.y.astype(np.int32), 0, self.ny - 1)
        a = float(alpha)
        if fade:
            # fade in on birth and out on death, so particles do not pop
            t = np.clip(self.age / max(self.life, 1e-6), 0.0, 1.0)
            w = np.clip(np.minimum(t * 8.0, (1.0 - t) * 4.0), 0.0, 1.0) * a
        else:
            w = np.full(self.n, a, dtype=np.float32)
        col = np.asarray(color, dtype=np.float32)
        np.add.at(out, (j, i), w[:, None] * col)
        return np.clip(out, 0, 255).astype(np.uint8)
