# SPDX-License-Identifier: MIT
# Parts adapted from the wind-tunnel engine in spectrometry.mp4 engines by Ethan Earl,
# https://github.com/ec175/spectrometry_public - Copyright (c) 2026 Ethan Earl.
# Licence text: windtunnel/LICENSES/spectrometry_public-MIT.txt
"""lbm.py - D2Q9 lattice-Boltzmann with a Smagorinsky subgrid closure.

WHY LBM AND NOT A PRESSURE-PROJECTION SOLVER
--------------------------------------------
The entire update is local plus a one-cell shift, so it is a handful of vectorised array ops
with no linear solve anywhere - no pressure Poisson equation, no multigrid, no preconditioner.
That has two consequences that matter here:

  - it maps onto array hardware trivially (swap NumPy for CuPy and nothing else changes);
  - solid walls are HALF-WAY BOUNCE-BACK on a boolean mask, so an arbitrary polygon - one that
    is re-rasterised every frame as the foil rotates - needs no meshing and no remeshing.

Unsteady separation and vortex shedding come out on their own at the viscosities we want.
Nothing in this file scripts them.

MODEL
-----
D2Q9 with the standard weights, BGK collision, and a Smagorinsky eddy viscosity that raises
tau locally where the strain rate is high. The LES term is not optional at these Reynolds
numbers: Re 6000 on a 110-cell chord puts the molecular tau at ~0.504, and plain BGK speckles
and then diverges anywhere below about 0.51. Raising tau exactly where the strain is largest
is both the physically correct closure and precisely where the instability starts.

The strain-rate magnitude comes from the non-equilibrium momentum-flux tensor Pi, which is
available for free from `f - feq` - already computed for the collision. The closure therefore
costs three reductions and a square root, not a separate gradient evaluation.

BOUNDARIES
----------
  inlet  (i=0)      full equilibrium at (rho=1, u=(u0, 0)); `set_inlet` shapes it across the
                    span, `set_inlet_turbulence` adds a convecting divergence-free fluctuation
  outlet (i=nx-1)   zero-gradient on the inward-facing populations, backed by a graded sponge
  sides  (j=0,-1)   free-slip by specular reflection - an inviscid wall that grows no boundary
                    layer and imposes no velocity, so nothing stripes the frame edge
  body              half-way bounce-back on `solid`, with a moving-wall term when it moves

UNITS
-----
Lattice units throughout: dx = dt = 1, cs^2 = 1/3, nu = (tau - 1/2)/3. `u0` is a LATTICE
VELOCITY, not a Mach number - the lattice sound speed is 1/sqrt(3) = 0.577, so the physical
Mach number of a run is u0 * sqrt(3). Keep u0 <= ~0.1 or the O(Ma^2) compressibility error
stops being small.
"""
from __future__ import annotations

import numpy as np

from .gpu import GPU, asnumpy, to_device, xp

# --- D2Q9 lattice -------------------------------------------------------------------
#   6   2   5          index 0 is rest; 1-4 are the axial links (weight 1/9);
#     \ | /            5-8 are the diagonals (weight 1/36).
#   3 - 0 - 1
#     / | \
#   7   4   8
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
              1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
LEFTWARD = np.flatnonzero(CX < 0)      # populations that would stream in through the outlet

CS2 = 1.0 / 3.0
CS = float(np.sqrt(CS2))


def equilibrium(rho, ux, uy, cx, cy, w):
    """f_k^eq = w_k rho (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u^2). Arrays broadcast to (9, ny, nx)."""
    cu = 3.0 * (cx * ux + cy * uy)
    usq = 1.5 * (ux * ux + uy * uy)
    return w * rho * (1.0 + cu + 0.5 * cu * cu - usq)


class LBM:
    """A D2Q9 lattice. `solid` is a (ny, nx) boolean mask that may be replaced every frame.

    Parameters
    ----------
    nx, ny      lattice size in cells
    u0          freestream LATTICE velocity (see the units note above); <= ~0.1
    re          Reynolds number, defined on `ref_len`
    ref_len     the length Re is built on, in cells - the chord, or the cylinder diameter
    csm         Smagorinsky constant; 0.16 is the classic value, 0 disables the closure
    sponge_out  width in cells of the graded outlet absorber
    sponge_side width of an optional side absorber (only used when side_bc="free")
    side_bc     "slip" for free-slip tunnel walls, "free" for a graded far-field band
    wall_clamp  ceiling on the wall speed the bounce-back term is allowed to see
    """

    def __init__(self, nx, ny, u0=0.08, re=6000.0, ref_len=100.0, csm=0.16,
                 sponge_out=16, sponge_side=0, side_bc="slip", wall_clamp=0.10,
                 fast=None):
        self.nx, self.ny = int(nx), int(ny)
        self.u0 = float(u0)
        self.re = float(re)
        self.ref_len = float(ref_len)
        self.nu = max(1e-6, self.u0 * self.ref_len / max(self.re, 1.0))
        self.tau = 3.0 * self.nu + 0.5
        self.csm2 = float(csm) ** 2
        self.side_bc = side_bc
        self.wall_clamp = float(wall_clamp)
        self.steps_done = 0

        f32 = xp.float32
        self.cx = to_device(CX.reshape(9, 1, 1))
        self.cy = to_device(CY.reshape(9, 1, 1))
        self.w = to_device(W.reshape(9, 1, 1))

        # The uniform freestream state, reused for the inlet column and as the sponge target.
        # `set_farfield` may replace this with a (9, ny, 1) spanwise profile, so nothing
        # downstream may assume it is one value per population.
        self._feq_free = equilibrium(xp.ones((1, 1), dtype=f32),
                                     xp.full((1, 1), self.u0, dtype=f32),
                                     xp.zeros((1, 1), dtype=f32),
                                     self.cx, self.cy, self.w).reshape(9, 1, 1)

        # SPONGE: relax f toward the freestream over a graded BAND. A single clamped row is a
        # discontinuity - flow accelerated around a blocking body slams into it and leaves a
        # bright stripe down the frame edge. A squared ramp absorbs the same disturbance with
        # no visible seam, and costs one fused multiply-add per cell per step.
        wi = np.zeros((1, self.nx), np.float32)
        if sponge_out > 0:
            d = (self.nx - 1 - np.arange(self.nx, dtype=np.float32)) / float(sponge_out)
            wi[0, :] = np.clip(1.0 - d, 0.0, 1.0) ** 2
        wj = np.zeros((self.ny, 1), np.float32)
        if sponge_side > 0:
            r = np.arange(self.ny, dtype=np.float32)
            d = np.minimum(r, self.ny - 1 - r) / float(sponge_side)
            wj[:, 0] = np.clip(1.0 - d, 0.0, 1.0) ** 2
        self._sponge_gain = 0.35
        self._sponge = to_device(np.maximum(wi, wj) * self._sponge_gain)[None]

        self.f = xp.broadcast_to(self._feq_free, (9, self.ny, self.nx)).copy()
        self.solid = xp.zeros((self.ny, self.nx), dtype=bool)
        self.rho = xp.ones((self.ny, self.nx), dtype=f32)
        self.ux = xp.full((self.ny, self.nx), self.u0, dtype=f32)
        self.uy = xp.zeros((self.ny, self.nx), dtype=f32)

        self._in_ux = self._in_uy = None        # shaped inlet, see set_inlet
        self._wux = self._wuy = None            # moving-wall velocity, see set_wall_velocity
        self._link = None                       # cached boundary links, see _rebuild_links
        self._fcol = None                       # last post-collision state, for force_field

        # BACKEND. The fused numba kernel is ~40x the numpy path on this machine (see
        # tools/bench.py) because the step is bandwidth-bound and fusing it writes none of the
        # ~30 temporaries numpy materialises. It is only usable on the CPU path: `xp` being
        # cupy means the arrays live on a device numba cannot reach from here.
        from . import lbm_numba
        self._nb = lbm_numba
        auto = lbm_numba.available() and not GPU
        self._fast = bool(auto if fast is None else (fast and lbm_numba.available()))
        self._buf_out = None
        self._buf_col = None
        self._feq_free_2d = None
        self._refresh_feq_free_2d()
        self._turb_on = False                   # see set_inlet_turbulence
        self._turb_du = self._turb_dv = None
        self._turb_amp = 0.0
        self._turb_n = 0
        self._turb_c = 0.0

    # -- geometry ---------------------------------------------------------------------
    def set_solid(self, mask):
        """Swap in a new body mask, REFILLING the cells the wall just vacated.

        Refilling is not optional: a stale population left inside a former solid cell is
        whatever bounce-back put there, and streaming it back into the flow injects a
        momentum spike. But refilling AT REST is only correct for a stationary body - behind
        one that is actually moving it lays down a trail of dead fluid in a live stream, and
        that velocity discontinuity launches a pressure wave every single step.

        So we extrapolate instead: new fluid takes the weighted average velocity of its
        neighbours that were ALREADY fluid last step. Neighbours the body just vacated hold
        garbage, and averaging that back in is how a moving body poisons its own wake.
        """
        new = xp.asarray(mask)
        freed = self.solid & ~new
        if bool(freed.any()):
            valid = ((~new) & (~self.solid)).astype(xp.float32)
            wsum = xp.zeros_like(self.rho)
            usum = xp.zeros_like(self.rho)
            vsum = xp.zeros_like(self.rho)
            for k in range(1, 9):
                sh = (-int(CY[k]), -int(CX[k]))
                wk = xp.roll(valid, sh, axis=(0, 1)) * float(W[k])
                wsum += wk
                usum += wk * xp.roll(self.ux, sh, axis=(0, 1))
                vsum += wk * xp.roll(self.uy, sh, axis=(0, 1))
            inv = 1.0 / xp.maximum(wsum, 1e-6)
            feq = equilibrium(xp.ones_like(self.rho), usum * inv, vsum * inv,
                              self.cx, self.cy, self.w)
            self.f = xp.where(freed[None], feq, self.f)
        self.solid = new
        self._rebuild_links()

    def _rebuild_links(self):
        """Cache the boundary links: `_link[k]` is True at each FLUID node whose neighbour in
        direction c_k is solid.

        Rebuilt only when the mask changes - once a frame - rather than inside `step()`, which
        runs tens of times per frame. Both bounce-back and the force integral need exactly this
        set, and they must agree on it: computing it twice from two slightly different
        expressions is how a boundary condition and its own force integral drift apart.
        """
        s = self.solid
        if not bool(s.any()):
            self._link = None
            return
        link = xp.zeros((9, self.ny, self.nx), dtype=bool)
        for k in range(1, 9):
            ahead = xp.roll(s, (-int(CY[k]), -int(CX[k])), axis=(0, 1))
            link[k] = (~s) & ahead
        self._link = link

    def set_wall_velocity(self, wux, wuy):
        """Velocity of the moving wall at each solid cell, or None for a stationary body.

        Plain bounce-back reflects off a wall assumed to be AT REST. If the mask moves every
        step that assumption is wrong twice: the boundary keeps injecting whatever momentum
        is needed to hold fluid still against a wall that is not still (which pumps energy in
        until the lattice diverges), and the force the body feels is computed against the
        wrong relative velocity. `step()` adds the standard 6 w_k (c_k . u_wall) term.

        The clamp guards against violent transients. Setting it too LOW does harm rather than
        good - a body drifting with a fast current gets its wall pinned to a slower speed, so
        the boundary brakes the very fluid it should be riding.
        """
        if wux is None:
            self._wux = self._wuy = None
            return
        lim = self.wall_clamp
        self._wux = xp.clip(to_device(asnumpy(wux)), -lim, lim)
        self._wuy = xp.clip(to_device(asnumpy(wuy)), -lim, lim)

    # -- forcing ----------------------------------------------------------------------
    def perturb(self, amp=0.02, seed=7, scale=6.0):
        """One-shot smooth transverse velocity - the symmetry breaker.

        A numerically PERFECT symmetric setup (a centred cylinder in a uniform stream) has
        nothing to trigger the von Karman instability: the wake sits as a stable symmetric
        twin bubble essentially forever. That is a numerical artefact, not physics - real
        flow always carries some disturbance. Inject a little once during settling and let
        the instability select and amplify its own mode; after that the shedding is entirely
        self-sustaining and the perturbation is long gone.
        """
        from scipy.ndimage import gaussian_filter
        n = np.random.default_rng(seed).standard_normal((self.ny, self.nx)).astype(np.float32)
        n = gaussian_filter(n, float(scale))
        n = n / (float(np.abs(n).max()) + 1e-9)
        # never fight the sponge: taper the kick to nothing inside the absorber band
        taper = 1.0 - asnumpy(self._sponge[0]) / self._sponge_gain
        duy = to_device(amp * self.u0 * n * taper)
        rho = xp.maximum(self.f.sum(axis=0), 1e-6)
        ux = (self.cx * self.f).sum(axis=0) / rho
        uy = (self.cy * self.f).sum(axis=0) / rho + duy
        self.f = equilibrium(rho, ux, uy, self.cx, self.cy, self.w)

    def set_inlet(self, ux_profile=None, uy_profile=None):
        """Override the inlet velocity across the span - length-ny arrays, or None to reset.

        This is what makes *currents* rather than a uniform stream: a shaped inlet profile
        shears against itself and rolls up into eddies downstream, with no obstacle needed
        and nothing keyframed in the interior.
        """
        self._in_ux = None if ux_profile is None else to_device(ux_profile)
        self._in_uy = None if uy_profile is None else to_device(uy_profile)

    def set_farfield(self, ux_profile=None):
        """Retarget the SPONGE toward a spanwise profile instead of a uniform u0.

        The outlet sponge relaxes f toward the freestream so a vortex leaving the domain does
        not reflect back in. That is only correct while "the freestream" is a single number.
        A scene that shapes the inlet into bands of different speed otherwise gets a pump
        bolted onto its exit plane: the sponge accelerates the slow band back toward u0, and
        that pressure gradient reaches upstream at the lattice sound speed, so the bands wash
        out over the clip while every stability number stays perfect. Feed the same profile
        here that went into `set_inlet`.
        """
        f32 = xp.float32
        if ux_profile is None:
            self._feq_free = equilibrium(xp.ones((1, 1), dtype=f32),
                                         xp.full((1, 1), self.u0, dtype=f32),
                                         xp.zeros((1, 1), dtype=f32),
                                         self.cx, self.cy, self.w).reshape(9, 1, 1)
            self._refresh_feq_free_2d()
            return
        u = to_device(ux_profile).reshape(1, self.ny, 1)
        self._feq_free = equilibrium(xp.ones((1, self.ny, 1), dtype=f32),
                                     u, xp.zeros_like(u), self.cx, self.cy, self.w)
        self._refresh_feq_free_2d()

    def set_inlet_turbulence(self, intensity=0.0, length=12.0, span=4096, seed=11, u_conv=None):
        """Carry freestream turbulence in through the inlet, continuously, for the whole run.

        `perturb()` is a single kick during settling, deliberately gone by frame 0. Real
        tunnel air is not like that: it carries a broadband fluctuation of ~0.1-2% of the
        freestream the entire time the fan is on, and that background is what trips a
        separating shear layer into breaking down at an irregular time and place instead of
        rolling up into the clean, evenly-spaced billows a silent solver produces. A
        noise-free 2-D solve reads as syrup no matter how high `re` goes, because nothing is
        ever there to disturb it.

        METHOD - synthetic inflow by Taylor's frozen-turbulence hypothesis. A (ny, span)
        patch is generated ONCE and convected past the inlet plane at `u_conv` cells/step, so
        the inlet sees a time series with the right length scale and correlation time for
        free: one interpolated column read per step, no RNG and no filter in the hot loop.
        `span` must exceed total_steps * u_conv or the patch recycles - visible as the same
        gust arriving twice.

        The patch is the CURL of a smoothed random streamfunction (u' = dpsi/dy,
        v' = -dpsi/dx), so what gets injected is divergence-free by construction. That is not
        fastidiousness: the inlet is a hard equilibrium BC at rho=1, so any compressive part
        of the fluctuation leaves as an acoustic wave that then rattles around the domain.
        Take the curl and it is never created in the first place.

        `intensity` is |u'|_rms as a fraction of the freestream - a scalar, or an (ny,) array
        for a banded inlet, where each band should carry the same *fraction* of its own speed
        rather than the same absolute wobble.
        """
        arr = np.asarray(intensity, dtype=np.float32)
        if float(np.max(np.abs(arr))) <= 0.0:
            self._turb_on = False
            return
        from scipy.ndimage import gaussian_filter
        n = int(span)
        rng = np.random.default_rng(seed)
        # mode="wrap" makes the patch periodic on both axes: seamless if it ever does recycle,
        # and no filter roll-off against the spanwise edges, which would otherwise leave the
        # rows nearest the tunnel walls quieter than the rest of the inlet.
        psi = gaussian_filter(rng.standard_normal((self.ny, n)).astype(np.float32),
                              float(length), mode="wrap")
        du = np.gradient(psi, axis=0)
        dv = -np.gradient(psi, axis=1)
        rms = float(np.sqrt(np.mean(du * du + dv * dv))) + 1e-12
        self._turb_du = to_device(du / rms)
        self._turb_dv = to_device(dv / rms)
        self._turb_n = n
        amp = arr * self.u0
        self._turb_amp = float(amp) if amp.ndim == 0 else to_device(amp.reshape(-1, 1))
        self._turb_c = float(self.u0 if u_conv is None else u_conv)
        self._turb_on = True

    def init_velocity(self, ux, uy):
        """Start the lattice from a given velocity field, at equilibrium and rho = 1."""
        self.ux = to_device(asnumpy(ux))
        self.uy = to_device(asnumpy(uy))
        self.rho = xp.ones((self.ny, self.nx), dtype=xp.float32)
        self.f = equilibrium(self.rho, self.ux, self.uy, self.cx, self.cy, self.w)

    # -- the step ---------------------------------------------------------------------
    def _refresh_feq_free_2d(self):
        """(9, ny) view of the sponge target, which is what the numba kernel indexes.

        `_feq_free` is (9, 1, 1) for a uniform freestream and (9, ny, 1) for a banded one;
        broadcasting it to a concrete (9, ny) once here keeps the branch out of the hot loop.
        """
        a = asnumpy(self._feq_free)
        self._feq_free_2d = np.ascontiguousarray(
            np.broadcast_to(a.reshape(9, -1), (9, self.ny)), dtype=np.float32)

    def _inlet_columns(self):
        """The (ny,) inlet velocity columns for this step, turbulence included.

        Computed host-side. The turbulence lookup is two interpolated column reads and some
        index arithmetic on `steps_done`; threading that would cost more in dispatch than it
        saves, and keeping it here means the kernel never has to know the patch exists.
        """
        if self._in_ux is None:
            ux = np.full(self.ny, self.u0, dtype=np.float32)
        else:
            ux = np.asarray(asnumpy(self._in_ux), dtype=np.float32).reshape(-1).copy()
        if self._in_uy is None:
            uy = np.zeros(self.ny, dtype=np.float32)
        else:
            uy = np.asarray(asnumpy(self._in_uy), dtype=np.float32).reshape(-1).copy()
        if self._turb_on:
            p = (self.steps_done * self._turb_c) % self._turb_n
            i0 = int(p)
            fr = p - i0
            i1 = (i0 + 1) % self._turb_n
            a = self._turb_amp
            a = a if np.isscalar(a) else np.asarray(asnumpy(a)).reshape(-1)
            du = asnumpy(self._turb_du)
            dv = asnumpy(self._turb_dv)
            ux += (a * ((1.0 - fr) * du[:, i0] + fr * du[:, i1])).reshape(-1)
            uy += (a * ((1.0 - fr) * dv[:, i0] + fr * dv[:, i1])).reshape(-1)
        return ux, uy

    def step(self):
        if self._fast:
            return self._step_numba()
        return self._step_numpy()

    def _step_numba(self):
        f = self.f
        self._edge_bcs(f)
        if self._buf_out is None or self._buf_out.shape != f.shape:
            self._buf_out = np.empty_like(f)
            self._buf_col = np.empty_like(f)
        ux_in, uy_in = self._inlet_columns()
        link = self._link if self._link is not None else np.zeros((9, self.ny, self.nx), bool)
        has_wall = self._wux is not None
        wux = self._wux if has_wall else np.zeros((self.ny, self.nx), np.float32)
        wuy = self._wuy if has_wall else np.zeros((self.ny, self.nx), np.float32)
        self._nb._step(
            f, self._buf_out, self._buf_col, self.rho, self.ux, self.uy,
            self.solid, link, self._sponge[0], self._feq_free_2d,
            ux_in, uy_in, wux, wuy, has_wall,
            np.float32(self.tau), np.float32(self.csm2),
            CX.astype(np.int64), CY.astype(np.int64), W.astype(np.float32),
            OPP.astype(np.int64))
        # rotate the buffers rather than allocating: `f` is dead once streamed, so it becomes
        # next step's scratch and nothing is freed or requested from the allocator per step
        self.f, self._buf_out = self._buf_out, f
        self._fcol = self._buf_col
        self.steps_done += 1

    def _edge_bcs(self, f):
        """Side walls and the outlet - six rows and three columns, so numpy is fine here."""
        if self.side_bc == "slip":
            # FREE-SLIP side walls by specular reflection: a population arriving from outside
            # comes back with its y-component mirrored. This is a real inviscid wall - it
            # grows no boundary layer, and unlike forcing the edge rows to the freestream it
            # imposes no velocity, so flow may legitimately accelerate along it and no stripe
            # appears in the picture.
            f[2, 0, :] = f[4, 0, :]              # bottom: up-going set = down-going set
            f[5, 0, :] = f[8, 0, :]
            f[6, 0, :] = f[7, 0, :]
            f[4, -1, :] = f[2, -1, :]            # top: the mirror image of the above
            f[7, -1, :] = f[6, -1, :]
            f[8, -1, :] = f[5, -1, :]

        # outlet: zero-gradient for the populations that would otherwise stream in from
        # outside the domain. The sponge does the real absorbing; this just stops the
        # boundary inventing information.
        for k in LEFTWARD:
            f[int(k), :, -1] = f[int(k), :, -2]

    def _step_numpy(self):
        """The reference implementation. Slower, but it is the definition of correct.

        `lbm_numba._step` must reproduce this to float32 roundoff; tools/bench.py asserts it
        before reporting any timing. Keep this readable in preference to making it fast - the
        fast path is elsewhere, and having a plain, obviously-correct version to diff against
        is what makes the fast one trustworthy.
        """
        f = self.f
        self._edge_bcs(f)

        rho = xp.maximum(f.sum(axis=0), 1e-6)
        ux = (self.cx * f).sum(axis=0) / rho
        uy = (self.cy * f).sum(axis=0) / rho

        # inlet column: clamp to the freestream before the collision sees it
        ux_in, uy_in = self._inlet_columns()
        rho[:, 0] = 1.0
        ux[:, 0] = to_device(ux_in)
        uy[:, 0] = to_device(uy_in)

        feq = equilibrium(rho, ux, uy, self.cx, self.cy, self.w)
        fneq = f - feq

        # --- Smagorinsky: raise tau where the strain rate is high ------------------------
        if self.csm2 > 0.0:
            pxx = (self.cx * self.cx * fneq).sum(axis=0)
            pyy = (self.cy * self.cy * fneq).sum(axis=0)
            pxy = (self.cx * self.cy * fneq).sum(axis=0)
            pi = xp.sqrt(2.0 * (pxx * pxx + pyy * pyy + 2.0 * pxy * pxy))
            tau_e = 0.5 * (self.tau + xp.sqrt(self.tau * self.tau
                                              + 18.0 * 1.4142135 * self.csm2 * pi / rho))
            omega = 1.0 / tau_e
        else:
            omega = 1.0 / self.tau

        fcol = f - omega * fneq

        # graded far field / outlet absorber, then a hard equilibrium inlet column
        fcol += self._sponge * (self._feq_free - fcol)
        fcol[:, :, 0] = feq[:, :, 0]

        # --- stream ----------------------------------------------------------------------
        fout = xp.empty_like(fcol)
        fout[0] = fcol[0]
        for k in range(1, 9):
            fout[k] = xp.roll(fcol[k], (int(CY[k]), int(CX[k])), axis=(0, 1))

        # --- body: HALF-WAY bounce-back --------------------------------------------------
        # The wall sits midway along each link, so a population leaving a boundary fluid node
        # toward the solid turns around and is back at the SAME node at the end of the same
        # step. It is applied after streaming rather than before, because the reflected
        # population must stay put: streaming it would carry it to x_b - c_k, which is a
        # different cell.
        #
        # This also repairs every slot at a boundary node that would otherwise have been fed
        # from inside the body. A fluid node receives f[j] from x_b - c_j, and that source is
        # solid exactly when j == OPP[k] for one of this node's links k - which is precisely
        # the set of slots overwritten below. Nothing garbage from inside the body survives.
        #
        # (An earlier revision reversed populations INSIDE the solid cells and streamed them
        # back out - full-way bounce-back. That is a legitimate scheme on its own, but it is
        # NOT the one the momentum-exchange force integral below assumes: it delays the
        # reflection by a step and offsets its origin by a cell, and the near-wall
        # non-equilibrium field varies far too sharply for that to wash out. Measured against
        # a control-volume momentum balance it inflated cylinder Cd from 1.13 to 3.94. The
        # boundary condition and its force integral have to be the same scheme.)
        if self._link is not None:
            for k in range(1, 9):
                L = self._link[k]
                refl = fcol[k]
                if self._wux is not None:
                    # moving wall: the reflected population is shifted by the wall's momentum,
                    # -2 w_k rho_w (c_k . u_w) / cs^2, with rho_w taken as 1. The wall velocity
                    # is stored on solid cells, so read it from the solid neighbour.
                    sh = (-int(CY[k]), -int(CX[k]))
                    uwx = xp.roll(self._wux, sh, axis=(0, 1))
                    uwy = xp.roll(self._wuy, sh, axis=(0, 1))
                    refl = refl - 6.0 * float(W[k]) * (float(CX[k]) * uwx + float(CY[k]) * uwy)
                    # positivity floor: a fast wall meeting a low-density cell can otherwise
                    # drive a population negative, which the collision then amplifies
                    refl = xp.maximum(refl, 0.0)
                fout[int(OPP[k])] = xp.where(L, refl, fout[int(OPP[k])])

        self.f = fout
        self._fcol = fcol            # kept for force_field(); no copy, fcol is dead after this
        self.rho, self.ux, self.uy = rho, ux, uy
        self.steps_done += 1

    def run(self, n):
        for _ in range(int(n)):
            self.step()

    # -- derived fields ----------------------------------------------------------------
    def speed(self):
        u = xp.sqrt(self.ux * self.ux + self.uy * self.uy)
        return xp.where(self.solid, 0.0, u)

    def vorticity(self):
        """dv/dx - du/dy by central differences, in lattice units (1/step)."""
        dvdx = (xp.roll(self.uy, -1, axis=1) - xp.roll(self.uy, 1, axis=1)) * 0.5
        dudy = (xp.roll(self.ux, -1, axis=0) - xp.roll(self.ux, 1, axis=0)) * 0.5
        return xp.where(self.solid, 0.0, dvdx - dudy)

    def pressure(self):
        """Gauge static pressure, p = (rho - rho_mean) * cs^2.

        Gauged against the instantaneous FLUID MEAN rather than against 1.0 on purpose. A
        weakly compressible lattice carries acoustic waves, so rho rings domain-wide whenever
        anything changes; plotting (rho - 1)/3 puts every one of those swings into the colour
        of every cell and the whole frame breathes. Subtracting the mean cancels the uniform
        component - the part that makes the picture pulse together - and absorbs any slow
        drift as well.
        """
        m = ~self.solid
        ref = float(asnumpy((self.rho * m).sum() / xp.maximum(m.sum(), 1)))
        return xp.where(self.solid, 0.0, (self.rho - ref) * CS2)

    def q_criterion(self):
        """Q = 0.5(|Omega|^2 - |S|^2); positive inside a vortex core.

        Vorticity alone cannot tell a vortex from a shear layer - a plain boundary layer is
        full of it. Q subtracts the strain contribution, so what survives is rotation that is
        actually winning, which is what isolates the discrete shed cores from the sheet they
        were born in.
        """
        dudx = (xp.roll(self.ux, -1, axis=1) - xp.roll(self.ux, 1, axis=1)) * 0.5
        dudy = (xp.roll(self.ux, -1, axis=0) - xp.roll(self.ux, 1, axis=0)) * 0.5
        dvdx = (xp.roll(self.uy, -1, axis=1) - xp.roll(self.uy, 1, axis=1)) * 0.5
        dvdy = (xp.roll(self.uy, -1, axis=0) - xp.roll(self.uy, 1, axis=0)) * 0.5
        s12 = 0.5 * (dudy + dvdx)
        o12 = 0.5 * (dudy - dvdx)
        q = 0.5 * (2.0 * o12 * o12 - (dudx * dudx + dvdy * dvdy + 2.0 * s12 * s12))
        return xp.where(self.solid, 0.0, q)

    # -- forces -------------------------------------------------------------------------
    def force_field(self):
        """Per-cell momentum handed to the wall, as two (ny, nx) arrays.

        Momentum exchange over the bounce-back links. Across one link the wall receives the
        incident momentum and pays back the reflected one, so it gains
        c_k f_k* - (-c_k f_bb) = c_k (f_k* + f_bb); for a stationary wall f_bb = f_k* and this
        is simply 2 c_k f_k*.

        Two things here are load-bearing, and both were established by measurement rather than
        by reading, because a wrong force integral is smooth, stable and completely believable:

          - `_fcol` is the POST-COLLISION, PRE-STREAMING state, and the reflected population
            is the one this same step's bounce-back actually wrote. Using the post-streaming
            `self.f` instead samples the incident and reflected populations a step apart and a
            cell apart. The density part survives that (it is smooth); the non-equilibrium part
            does not, and it contributed a spurious Cd of ~3.0 on top of a true ~1.1.

          - the sum runs over `_link`, the SAME cached link set the bounce-back used. If the
            boundary condition and its force integral disagree about which links exist, the
            force is an integral over a surface that is not the one the fluid saw.

        tools/validate.py checks the result against a control-volume momentum balance and
        against published cylinder data. Do not change this method without re-running it.

        Left un-summed so a scene with several bodies can integrate over each body's own mask;
        computing it once for the whole lattice is far cheaper than a link loop per body.
        """
        fx = xp.zeros_like(self.rho)
        fy = xp.zeros_like(self.rho)
        if self._link is None or self._fcol is None:
            return fx, fy
        for k in range(1, 9):
            L = self._link[k].astype(xp.float32)
            src = self._fcol[k]
            refl = src
            if self._wux is not None:
                sh = (-int(CY[k]), -int(CX[k]))
                uwx = xp.roll(self._wux, sh, axis=(0, 1))
                uwy = xp.roll(self._wuy, sh, axis=(0, 1))
                refl = xp.maximum(
                    src - 6.0 * float(W[k]) * (float(CX[k]) * uwx + float(CY[k]) * uwy), 0.0)
            amt = (src + refl) * L
            fx += float(CX[k]) * amt
            fy += float(CY[k]) * amt
        return fx, fy

    def force(self, mask=None):
        """Net (Fx, Fy) on the body in lattice units. `mask` restricts to one body."""
        fx, fy = self.force_field()
        if mask is not None:
            m = xp.asarray(mask)
            fx, fy = fx * m, fy * m
        return float(asnumpy(fx.sum())), float(asnumpy(fy.sum()))

    def coefficients(self, ref_len=None, mask=None):
        """(Cd, Cl) = F / (0.5 rho u0^2 L), the usual 2-D non-dimensionalisation.

        NOTE the sign on Cl. Lattice +y is screen-DOWN (see shapes.place), so a nose-up foil
        generating lift feels a force toward -y. Flipping it here means Cl is positive for
        positive incidence, the way anyone reading it expects.
        """
        L = self.ref_len if ref_len is None else float(ref_len)
        fx, fy = self.force(mask=mask)
        q = 0.5 * self.u0 * self.u0 * max(L, 1e-9)
        return fx / q, -fy / q

    # -- diagnostics --------------------------------------------------------------------
    def health(self):
        """Max |u| - a cheap divergence tripwire."""
        return float(asnumpy(xp.sqrt(self.ux ** 2 + self.uy ** 2).max()))

    @property
    def health_limit(self):
        """Abort threshold. The lattice sound speed is 1/sqrt(3) = 0.577, and past about 0.45
        the solve is already gone, so every remaining frame would be noise."""
        return 0.45

    @property
    def mach(self):
        """Physical Mach number of the freestream: a lattice velocity over cs = 1/sqrt(3)."""
        return self.u0 / CS

    def describe(self):
        return (f"D2Q9 LBM + Smagorinsky | {self.nx}x{self.ny} | u0={self.u0:.4f} "
                f"(M={self.mach:.3f}) | Re={self.re:g} on L={self.ref_len:g} | "
                f"nu={self.nu:.6f} | tau={self.tau:.5f} | "
                f"kernel={'numba' if self._fast else 'numpy'}")
