# SPDX-License-Identifier: MIT
# Adapted from the wind-tunnel engine in spectrometry.mp4 engines by Ethan Earl,
# https://github.com/ec175/spectrometry_public - Copyright (c) 2026 Ethan Earl.
# Licence text: windtunnel/LICENSES/spectrometry_public-MIT.txt
"""cns.py - the COMPRESSIBLE solver: 2-D Navier-Stokes, finite volume, MUSCL + HLLC.

`lbm.py` is the right engine for almost everything in this project. This module exists for
the one class of thing it structurally cannot do.

WHY A SECOND SOLVER
-------------------
D2Q9 lattice-Boltzmann is a WEAKLY COMPRESSIBLE, ISOTHERMAL model, and each of the following
is a wall rather than a setting you can turn up:

  - its equation of state is p = rho*cs^2 - an isothermal gas with gamma = 1. There is no
    temperature, no internal energy, no thermodynamics of any kind;
  - its error is O(Ma^2) and it has no shock-capturing mechanism, so it degrades continuously
    and then simply fails somewhere above M ~ 0.3;
  - the lattice sound speed is pinned at 1/sqrt(3) in grid units, so "go faster" always means
    "go closer to the model's own ceiling".

A supersonic pocket, a shock, a stagnation temperature - the phenomena that make air read as
a GAS rather than as a thin liquid - are unreachable in `lbm.py` at any parameter setting.

WHY THIS SCHEME
---------------
The in-keeping answer would be a higher-order lattice (D2Q37), which restores thermodynamics
and lifts the Mach ceiling. It is also research-grade: an order-9 Gauss-Hermite quadrature, a
4th-order Hermite equilibrium, streaming with velocity components of +-3 so it is no longer a
single-cell shift, and it STILL needs a separate shock-capturing filter bolted on to stop
Gibbs oscillations ringing off a discontinuity.

A Godunov finite-volume scheme reaches the same physics by a route that is textbook, robustly
stable, and just as vectorisable:

  - CONSERVATIVE FORM in (rho, rho*u, rho*v, E) with a real ideal-gas EOS, gamma = 1.4, so
    energy and thermodynamics are carried explicitly rather than emerging in a limit;
  - MUSCL reconstruction with a minmod limiter: 2nd order where the flow is smooth, TVD across
    a discontinuity, which is what stops a captured shock from ringing;
  - an HLLC approximate Riemann solver at every face. HLLC restores the contact wave that HLL
    smears, and that matters here specifically - a shear layer and a wake ARE contact
    discontinuities, so an HLL solve would diffuse the vortex sheet this project exists to show;
  - SSP-RK2 in time, which preserves the TVD property of the spatial operator;
  - explicit Navier-Stokes viscous terms (full stress tensor + Fourier conduction, Pr = 0.71)
    plus a Smagorinsky eddy viscosity, so the boundary layer and the wake are real viscous
    features rather than numerical dissipation.

Shocks are then captured by the scheme itself, at the correct jump conditions, with no filter
and no tuning. That is the entire point of a conservative Godunov method: by Lax-Wendroff, a
conservative, consistent, convergent scheme converges to the correct weak solution, so the
Rankine-Hugoniot relations hold automatically rather than being imposed.

UNITS - AND THEY ARE NOT THE LATTICE'S
--------------------------------------
Non-dimensionalised on the freestream: rho_inf = 1 and p_inf = 1/gamma, hence

    c_inf = sqrt(gamma * p_inf / rho_inf) = 1

so A VELOCITY IN THIS SOLVER IS A MACH NUMBER, exactly, with no conversion. `u0 = 0.19` means
M 0.19. Contrast `lbm.py`, where `u0 = 0.055` is a lattice velocity and the Mach number is
`u0 * sqrt(3)`. This is the single most likely thing to get wrong when moving a scene between
the two solvers - the same field name means a different physical quantity in each. Length unit
is one cell; time unit is one cell per freestream sound speed.

THE TIME STEP IS FIXED, ON PURPOSE
----------------------------------
The natural thing for a compressible code is an adaptive dt from the instantaneous CFL. This
one computes dt ONCE and holds it, because the project's timebase rests on steps-per-frame:
apparent on-screen speed is (distance per step x steps per frame), and a dt that wandered with
the solution would make the wind speed in the picture a function of the flow. `dt` is set from
a conservative bound on max(|u| + c) and `cfl_now()` reports what was actually achieved, so a
run can be checked rather than assumed.
"""
from __future__ import annotations

import numpy as np

from .gpu import asnumpy, to_device, xp

GAMMA = 1.4
PR = 0.71                  # Prandtl number for air
NG = 2                     # ghost layers; MUSCL reaches two cells
_TINY = 1e-9


def _minmod(a, b):
    """The limiter. Returns whichever slope is smaller in magnitude, or 0 if they disagree.

    The zero branch is the entire point. At an extremum the two one-sided slopes have opposite
    signs, the reconstruction collapses to 1st order there, and the scheme stays TVD - which is
    what stops a captured shock from ringing. Do not "improve" this to a smoother limiter
    without re-running tools/sod_check.py.
    """
    return xp.where(a * b > 0.0, xp.where(xp.abs(a) < xp.abs(b), a, b), 0.0)


class CNS:
    """Compressible Navier-Stokes on a uniform grid. `solid` may change every frame.

    Arrays carry `NG` ghost cells on every side; `nx`/`ny` are the INTERIOR size and every
    public field returns the interior view only, so nothing outside this file has to know the
    padding exists.

    Parameters
    ----------
    u0        freestream MACH number (c_inf = 1). Not a lattice velocity - see module docstring.
    re        Reynolds number on `ref_len`
    csm       Smagorinsky constant; 0 disables the subgrid model
    cfl       target CFL used to set the fixed dt
    blockage  expected peak speed as a multiple of u0, used to bound dt. Under-estimate this
              and the run violates CFL and dies; over-estimate and it merely renders slower.
    """

    def __init__(self, nx, ny, u0=0.19, re=16000.0, ref_len=100.0, csm=0.20,
                 cfl=0.45, sponge_out=24, side_bc="slip", wall_clamp=0.30, blockage=4.2):
        self.nx, self.ny = int(nx), int(ny)
        self.Nx, self.Ny = self.nx + 2 * NG, self.ny + 2 * NG
        self.u0 = float(u0)
        self.re = float(re)
        self.ref_len = float(ref_len)
        self.nu = max(1e-9, self.u0 * self.ref_len / max(self.re, 1.0))
        self.mu = self.nu                      # rho_inf = 1, so mu and nu coincide
        self.csm2 = float(csm) ** 2
        self.side_bc = side_bc
        self.wall_clamp = float(wall_clamp)
        self.p_inf = 1.0 / GAMMA
        self.steps_done = 0

        # FIXED dt. The term that actually bites is BLOCKAGE, not the freestream: with
        # free-slip walls the stream squeezing past a body at incidence reaches several times
        # the freestream, and a compressible stagnation region raises the local sound speed on
        # top of that.
        self.cfl = float(cfl)
        self.umax_design = float(blockage) * self.u0
        # Peak sound speed is the STAGNATION value, c0/c_inf = sqrt(1 + (gamma-1)/2 * M^2),
        # reached where the flow is brought to rest. Note it and `umax_design` cannot occur in
        # the same cell - one needs high speed, the other needs none - so adding them is
        # already conservative without inflating either. An earlier version wrote (2*u0)^2
        # inside the stagnation relation, which at M 2 bounded c by 4.3 instead of 1.34 and
        # made every compressible run three times slower than it needed to be.
        self.cmax_design = np.sqrt(1.0 + 0.5 * (GAMMA - 1.0) * self.u0 ** 2) + 0.10
        self.dt = self.cfl / (self.umax_design + self.cmax_design)

        f32 = xp.float32
        self.Q = xp.zeros((4, self.Ny, self.Nx), dtype=f32)
        self._solid_g = xp.zeros((self.Ny, self.Nx), dtype=bool)
        self._set_uniform(self.u0)

        self._in_ux = self._in_uy = None
        self._wux = self._wuy = None
        self._turb_on = False
        self._turb_du = self._turb_dv = None
        self._turb_amp = 0.0
        self._turb_n = 0
        self._turb_c = 0.0
        # immersed-boundary ghost cache, rebuilt whenever the mask changes
        self._gy = self._gx = None
        self._giy = self._gix = None
        self._gnx = self._gny = self._gfac = None
        self._gn = 0

        # Graded outlet sponge, same device and same justification as the LBM one: a hard
        # zero-gradient exit reflects, and in a COMPRESSIBLE solve what reflects is an acoustic
        # wave that then crosses the entire picture.
        w = np.zeros((1, 1, self.Nx), np.float32)
        if sponge_out > 0:
            d = (self.Nx - 1 - NG - np.arange(self.Nx, dtype=np.float32)) / float(sponge_out)
            w[0, 0, :] = np.clip(1.0 - d, 0.0, 1.0) ** 2
        self._sponge = to_device(w * 0.25)
        self._q_far = None
        self.set_farfield(None)

        self.rho = self._interior(self.Q[0]).copy()
        self.ux = xp.full((self.ny, self.nx), self.u0, dtype=f32)
        self.uy = xp.zeros((self.ny, self.nx), dtype=f32)
        self._p = xp.full((self.ny, self.nx), self.p_inf, dtype=f32)

    # -- helpers -------------------------------------------------------------------------
    @staticmethod
    def _interior(a):
        return a[..., NG:-NG, NG:-NG]

    def _set_uniform(self, u):
        r = xp.ones((self.Ny, self.Nx), dtype=xp.float32)
        self.Q[0] = r
        self.Q[1] = r * u
        self.Q[2] = 0.0
        self.Q[3] = self.p_inf / (GAMMA - 1.0) + 0.5 * r * u * u

    @staticmethod
    def _prim(Q):
        """(rho, u, v, p) from the conservative state, floored so a transient cannot make NaN."""
        r = xp.maximum(Q[0], 1e-6)
        u = Q[1] / r
        v = Q[2] / r
        p = xp.maximum((GAMMA - 1.0) * (Q[3] - 0.5 * r * (u * u + v * v)), 1e-8)
        return r, u, v, p

    @staticmethod
    def _cons(r, u, v, p):
        return xp.stack([r, r * u, r * v,
                         p / (GAMMA - 1.0) + 0.5 * r * (u * u + v * v)])

    @property
    def solid(self):
        return self._solid_g[NG:-NG, NG:-NG]

    # -- geometry ------------------------------------------------------------------------
    def set_wall_velocity(self, wux, wuy):
        """Velocity of the moving wall at each solid cell (None = stationary)."""
        if wux is None:
            self._wux = self._wuy = None
            return
        lim = self.wall_clamp
        self._wux = xp.clip(to_device(asnumpy(wux)), -lim, lim)
        self._wuy = xp.clip(to_device(asnumpy(wuy)), -lim, lim)

    def set_solid(self, mask):
        """Swap in a new body mask and rebuild the immersed-boundary ghost cache.

        The body is imposed by GHOST CELLS, not by bounce-back: for every solid cell within
        reach of the reconstruction stencil we mirror the fluid state across the surface, so
        the values the scheme reads outside the fluid are the ones that make the wall condition
        hold. Which cells, where their image points are, the surface normal there, and the
        extrapolation lever arm all depend only on the MASK, so they are computed once here
        rather than at every one of the ~100 steps in a frame.

        The signed distance comes from two Euclidean distance transforms and the normal is its
        gradient. Both are exact for any polygon, which preserves the property that made the
        LBM worth using in the first place: an arbitrary shape, re-rasterised as it rotates,
        needs no mesh.
        """
        from scipy.ndimage import distance_transform_edt

        m = np.zeros((self.Ny, self.Nx), bool)
        m[NG:-NG, NG:-NG] = np.asarray(asnumpy(mask), bool)
        self._solid_g = xp.asarray(m)
        if not m.any():
            self._gy = self._gx = None
            self._gn = 0
            return

        d_in = distance_transform_edt(m).astype(np.float32)      # solid -> nearest fluid
        d_out = distance_transform_edt(~m).astype(np.float32)    # fluid -> nearest solid
        phi = d_out - d_in                                       # >0 in fluid, <0 in solid
        gy, gx = np.gradient(phi)                                # points solid -> fluid
        nrm = np.sqrt(gy * gy + gx * gx) + 1e-6
        gy, gx = gy / nrm, gx / nrm

        band = m & (d_in <= NG + 0.5)                            # only what the stencil reaches
        jj, ii = np.nonzero(band)

        # HALF-CELL CORRECTION. `d_in` is the distance from a solid cell centre to the nearest
        # FLUID CELL CENTRE; the wall condition needs the distance to the SURFACE, about half a
        # cell short of that. It is kept because it is the correct definition - but it is NOT a
        # cure for the staircase over-turn, and the measurement in tools/shock_check.py says so
        # explicitly. What remains is the staircase itself: a shallow ramp steps one row every
        # few columns and the EDT normals oscillate between the tread and riser orientations.
        d = np.maximum(d_in[jj, ii] - 0.5, 0.05)
        ny_, nx_ = gy[jj, ii], gx[jj, ii]

        # Image point at a FIXED stand-off h rather than at the mirror distance. The mirror
        # form (h = d) is the textbook one and it breaks on a thin body: near a trailing edge
        # the section is only a few cells thick, so a deep ghost cell's mirror lands on the far
        # side, in the fluid belonging to the OTHER surface. Pinning h >= 1.75 keeps the image
        # in the fluid this cell actually belongs to, and taking h = max(1.75, d) also caps the
        # extrapolation factor d/h at 1, so a deep ghost can never amplify the image state.
        h = np.maximum(1.75, d)
        self._gy = xp.asarray(jj.astype(np.int64))
        self._gx = xp.asarray(ii.astype(np.int64))
        self._gnx = to_device(nx_)
        self._gny = to_device(ny_)
        self._gfac = to_device((d / h).astype(np.float32))
        self._gn = len(jj)
        # image-point coordinates, as one stacked (4, N) gather over the primitive array
        self._giy = to_device((jj + (d + h) * ny_).astype(np.float32))
        self._gix = to_device((ii + (d + h) * nx_).astype(np.float32))

    # -- boundaries / forcing -------------------------------------------------------------
    def set_inlet(self, ux_profile=None, uy_profile=None):
        """Override the inlet velocity across the span (interior rows), or None to reset."""
        self._in_ux = None if ux_profile is None else to_device(ux_profile)
        self._in_uy = None if uy_profile is None else to_device(uy_profile)

    def set_farfield(self, ux_profile=None):
        """Retarget the outlet sponge to a spanwise profile - see LBM.set_farfield, same trap.

        A sponge relaxing toward one uniform u0 is a pump bolted onto the exit plane of any
        scene whose inlet is banded: it erodes the bands over the clip while every stability
        number stays perfect. Feed it the same profile that went into `set_inlet`.
        """
        f32 = xp.float32
        u = xp.full((self.Ny, 1), self.u0, dtype=f32)
        if ux_profile is not None:
            col = to_device(ux_profile).reshape(-1, 1)
            u = xp.concatenate([xp.repeat(col[:1], NG, axis=0), col,
                                xp.repeat(col[-1:], NG, axis=0)], axis=0)
        r = xp.ones((self.Ny, 1), dtype=f32)
        z = xp.zeros((self.Ny, 1), dtype=f32)
        p = xp.full((self.Ny, 1), self.p_inf, dtype=f32)
        self._q_far = self._cons(r, u, z, p)

    def set_inlet_turbulence(self, intensity=0.0, length=12.0, span=4096, seed=11, u_conv=None):
        """Continuous freestream turbulence at the inlet - see LBM.set_inlet_turbulence.

        Identical construction (a frozen divergence-free patch convected past the inlet plane),
        and divergence-free matters MORE here than it did there: this solver resolves acoustics
        properly, so the compressive part of a careless inflow perturbation would not merely
        add noise, it would radiate a pressure wave the scheme is good enough to keep.
        """
        arr = np.asarray(intensity, dtype=np.float32)
        if float(np.max(np.abs(arr))) <= 0.0:
            self._turb_on = False
            return
        from scipy.ndimage import gaussian_filter
        n = int(span)
        rng = np.random.default_rng(seed)
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
        """Start from a given velocity field, at freestream density and pressure."""
        u = xp.zeros((self.Ny, self.Nx), dtype=xp.float32)
        v = xp.zeros((self.Ny, self.Nx), dtype=xp.float32)
        u[NG:-NG, NG:-NG] = to_device(asnumpy(ux))
        v[NG:-NG, NG:-NG] = to_device(asnumpy(uy))
        for a in (u, v):
            a[:NG] = a[NG:NG + 1]
            a[-NG:] = a[-NG - 1:-NG]
            a[:, :NG] = a[:, NG:NG + 1]
            a[:, -NG:] = a[:, -NG - 1:-NG]
        r = xp.ones((self.Ny, self.Nx), dtype=xp.float32)
        p = xp.full((self.Ny, self.Nx), self.p_inf, dtype=xp.float32)
        self.Q = self._cons(r, u, v, p)

    def perturb(self, amp=0.02, seed=7, scale=6.0):
        """One-shot smooth transverse velocity - the symmetry breaker (see LBM.perturb)."""
        if amp <= 0:
            return
        from scipy.ndimage import gaussian_filter
        n = np.random.default_rng(seed).standard_normal((self.Ny, self.Nx)).astype(np.float32)
        n = gaussian_filter(n, float(scale))
        n = to_device(n / (float(np.abs(n).max()) + 1e-9))
        r, u, v, p = self._prim(self.Q)
        self.Q = self._cons(r, u, v + amp * self.u0 * n, p)

    def set_drive(self, *a, **k):
        raise NotImplementedError("set_drive is an LBM device with no meaning in a "
                                  "compressible tunnel")

    def force_field(self):
        raise NotImplementedError(
            "CNS has no VALIDATED body-force integral yet, so scenes that need Cl/Cd must stay "
            "on the LBM. Implementing it means integrating pressure + viscous traction over "
            "the immersed surface and checking it against a known coefficient, the way "
            "tools/validate.py did for the LBM. That check is not optional here: a "
            "plausible-looking wrong force is exactly the bug that made this project's LBM "
            "report Cd 3.94 against a true 1.13, and every rendered frame looked correct.")

    def coefficients(self, ref_len=None, mask=None):
        raise NotImplementedError(self.force_field.__doc__)

    # -- the scheme ------------------------------------------------------------------------
    def _bcs(self, Q):
        """Ghost cells for inlet, outlet and tunnel walls, in place."""
        u_in = self.u0 if self._in_ux is None else self._in_ux.reshape(-1, 1)
        v_in = 0.0 if self._in_uy is None else self._in_uy.reshape(-1, 1)
        if self._turb_on:
            p = (self.steps_done * self._turb_c * self.dt) % self._turb_n
            i0 = int(p)
            fr = p - i0
            i1 = (i0 + 1) % self._turb_n
            a = self._turb_amp
            du = ((1.0 - fr) * self._turb_du[:, i0] + fr * self._turb_du[:, i1]).reshape(-1, 1)
            dv = ((1.0 - fr) * self._turb_dv[:, i0] + fr * self._turb_dv[:, i1]).reshape(-1, 1)
            # `a` is a float for a uniform inlet, or an (ny, 1) column for a banded one;
            # both broadcast against the (ny, 1) fluctuation without a branch
            u_in = u_in + a * du
            v_in = v_in + a * dv

        one = xp.ones((self.ny, 1), dtype=xp.float32)
        qin = self._cons(one,
                         xp.broadcast_to(xp.asarray(u_in, dtype=xp.float32), (self.ny, 1)) * one,
                         xp.broadcast_to(xp.asarray(v_in, dtype=xp.float32), (self.ny, 1)) * one,
                         xp.full((self.ny, 1), self.p_inf, dtype=xp.float32))
        Q[:, NG:-NG, :NG] = qin

        # OUTLET: zero-gradient; the sponge does the absorbing
        Q[:, :, -NG:] = Q[:, :, -NG - 1:-NG]

        # SIDES: free-slip tunnel walls by mirroring with the wall-normal momentum reversed.
        # Same choice and same reason as the LBM: imposing a velocity on the edge rows draws a
        # hard stripe wherever a body blocks the tunnel. Blockage is therefore REAL here too,
        # which is why it feeds the dt estimate.
        for g, src in ((NG - 1, NG), (NG - 2, NG + 1)):
            Q[0, g] = Q[0, src]; Q[1, g] = Q[1, src]
            Q[2, g] = -Q[2, src]; Q[3, g] = Q[3, src]
        for g, src in ((-NG, -NG - 1), (-NG + 1, -NG - 2)):
            Q[0, g] = Q[0, src]; Q[1, g] = Q[1, src]
            Q[2, g] = -Q[2, src]; Q[3, g] = Q[3, src]
        return Q

    def _ibm(self, Q):
        """Impose the body by writing mirrored states into its near-surface cells, in place."""
        if self._gn == 0:
            return Q
        r, u, v, p = self._prim(Q)
        yi = self._giy
        xi = self._gix
        # bilinear gather at the image points (host-independent, no scipy in the hot loop)
        j0 = xp.clip(yi.astype(xp.int64), 0, self.Ny - 2)
        i0 = xp.clip(xi.astype(xp.int64), 0, self.Nx - 2)
        fy = xp.clip(yi - j0.astype(xp.float32), 0.0, 1.0)
        fx = xp.clip(xi - i0.astype(xp.float32), 0.0, 1.0)

        def gather(a):
            return (a[j0, i0] * (1 - fx) * (1 - fy) + a[j0, i0 + 1] * fx * (1 - fy)
                    + a[j0 + 1, i0] * (1 - fx) * fy + a[j0 + 1, i0 + 1] * fx * fy)

        ri, ui, vi, pi = gather(r), gather(u), gather(v), gather(p)
        if self._wux is not None:
            uw = self._wux[self._gy - NG, self._gx - NG]
            vw = self._wuy[self._gy - NG, self._gx - NG]
        else:
            uw = vw = 0.0
        # NO-SLIP by linear extrapolation through the wall: the ghost value is chosen so a
        # straight line from the image point to the ghost cell passes through the wall velocity
        # AT the surface. `_gfac` = d/h <= 1 is the lever arm.
        ug = uw - self._gfac * (ui - uw)
        vg = vw - self._gfac * (vi - vw)
        # ADIABATIC wall, zero normal pressure gradient: carry density and pressure across
        # unchanged. Imposing a temperature instead would be a heat source the scene never
        # asked for - and at these Mach numbers the stagnation heating IS part of the subject.
        Q[:, self._gy, self._gx] = self._cons(ri, ug, vg, pi)
        return Q

    def _visc_mu(self, r, u, v):
        """Molecular + Smagorinsky eddy viscosity at cell centres."""
        ux = 0.5 * (xp.roll(u, -1, axis=1) - xp.roll(u, 1, axis=1))
        uy = 0.5 * (xp.roll(u, -1, axis=0) - xp.roll(u, 1, axis=0))
        vx = 0.5 * (xp.roll(v, -1, axis=1) - xp.roll(v, 1, axis=1))
        vy = 0.5 * (xp.roll(v, -1, axis=0) - xp.roll(v, 1, axis=0))
        sxy = 0.5 * (uy + vx)
        smag = xp.sqrt(2.0 * (ux * ux + vy * vy + 2.0 * sxy * sxy))
        return self.mu + r * self.csm2 * smag

    @staticmethod
    def _hllc(rL, uL, vL, pL, rR, uR, vR, pR):
        """HLLC flux for the 1-D Euler equations; `u` normal, `v` tangential to the face.

        Three waves: the two acoustic waves bounding the fan (SL, SR) and the CONTACT (S*) in
        between. HLL drops the middle one and smears every contact discontinuity over several
        cells - which here would mean smearing the shear layers and the wake, i.e. the subject.
        Wave speeds use the Davis estimate, which is cheap and provably bounding for an ideal
        gas.
        """
        cL = xp.sqrt(GAMMA * pL / rL)
        cR = xp.sqrt(GAMMA * pR / rR)
        SL = xp.minimum(uL - cL, uR - cR)
        SR = xp.maximum(uL + cL, uR + cR)
        EL = pL / (GAMMA - 1.0) + 0.5 * rL * (uL * uL + vL * vL)
        ER = pR / (GAMMA - 1.0) + 0.5 * rR * (uR * uR + vR * vR)
        mL = rL * (SL - uL)
        mR = rR * (SR - uR)
        den = mL - mR
        Ss = (pR - pL + mL * uL - mR * uR) / xp.where(xp.abs(den) < _TINY, _TINY, den)

        FL = xp.stack([rL * uL, rL * uL * uL + pL, rL * uL * vL, (EL + pL) * uL])
        FR = xp.stack([rR * uR, rR * uR * uR + pR, rR * uR * vR, (ER + pR) * uR])
        UL = xp.stack([rL, rL * uL, rL * vL, EL])
        UR = xp.stack([rR, rR * uR, rR * vR, ER])

        def star(rK, uK, vK, pK, EK, SK, mK):
            f = rK * (SK - uK) / xp.where(xp.abs(SK - Ss) < _TINY, _TINY, SK - Ss)
            e = EK / rK + (Ss - uK) * (Ss + pK / xp.where(xp.abs(mK) < _TINY, _TINY, mK))
            return xp.stack([f, f * Ss, f * vK, f * e])

        FsL = FL + SL * (star(rL, uL, vL, pL, EL, SL, mL) - UL)
        FsR = FR + SR * (star(rR, uR, vR, pR, ER, SR, mR) - UR)
        return xp.where(SL >= 0.0, FL,
                        xp.where(Ss >= 0.0, FsL, xp.where(SR >= 0.0, FsR, FR)))

    def _flux_dir(self, r, u, v, p, mu, axis):
        """Net inviscid + viscous flux difference along one axis. `u` is the NORMAL velocity."""
        W = xp.stack([r, u, v, p])
        s = _minmod(W - xp.roll(W, 1, axis=axis + 1), xp.roll(W, -1, axis=axis + 1) - W)
        WL = W + 0.5 * s
        WR = xp.roll(W - 0.5 * s, -1, axis=axis + 1)
        # Floor the RECONSTRUCTED density and pressure, not just the cell averages. minmod
        # keeps the reconstruction monotone but not positive, and one negative pressure inside
        # a strong expansion makes sqrt(gamma p / rho) NaN and takes the whole field with it in
        # a single step. The floor only ever engages where the limiter has already given up.
        WL = xp.concatenate([xp.maximum(WL[:1], 1e-6), WL[1:3], xp.maximum(WL[3:], 1e-8)])
        WR = xp.concatenate([xp.maximum(WR[:1], 1e-6), WR[1:3], xp.maximum(WR[3:], 1e-8)])
        F = self._hllc(WL[0], WL[1], WL[2], WL[3], WR[0], WR[1], WR[2], WR[3])

        # --- viscous flux at the same faces (central, so it stays 2nd order and conservative)
        ax = axis
        other = 1 - ax
        dn_un = xp.roll(u, -1, axis=ax) - u                  # d(u_n)/dn across the face
        dn_ut = xp.roll(v, -1, axis=ax) - v                  # d(u_t)/dn

        def dt_(a):
            d = 0.5 * (xp.roll(a, -1, axis=other) - xp.roll(a, 1, axis=other))
            return 0.5 * (d + xp.roll(d, -1, axis=ax))

        dt_un = dt_(u)
        dt_ut = dt_(v)
        muf = 0.5 * (mu + xp.roll(mu, -1, axis=ax))
        div = dn_un + dt_ut
        tnn = muf * (2.0 * dn_un - (2.0 / 3.0) * div)
        tnt = muf * (dn_ut + dt_un)
        uf = 0.5 * (u + xp.roll(u, -1, axis=ax))
        vf = 0.5 * (v + xp.roll(v, -1, axis=ax))
        # Fourier conduction on T = p/rho (so T_inf = p_inf/rho_inf); cp/(gamma-1) folded in
        tt = p / r
        kf = muf * GAMMA / ((GAMMA - 1.0) * PR)
        q = kf * (xp.roll(tt, -1, axis=ax) - tt)
        G = xp.stack([xp.zeros_like(tnn), tnn, tnt, uf * tnn + vf * tnt + q])

        H = F - G
        return H - xp.roll(H, 1, axis=axis + 1)

    def _rhs(self, Q):
        """dQ/dt for the whole domain, with boundaries and the body already imposed."""
        Q = self._ibm(self._bcs(Q))
        r, u, v, p = self._prim(Q)
        mu = self._visc_mu(r, u, v)
        dQ = -self._flux_dir(r, u, v, p, mu, 1)                  # x sweep
        # y sweep: hand the solver the NORMAL velocity first, then swap the momenta back
        dy = self._flux_dir(r, v, u, p, mu, 0)
        return dQ - xp.stack([dy[0], dy[2], dy[1], dy[3]])

    def step(self):
        """One SSP-RK2 (Heun) step - strong-stability-preserving, so the limiter still holds."""
        Q0 = self.Q
        Q1 = Q0 + self.dt * self._rhs(Q0)
        self.Q = 0.5 * Q0 + 0.5 * (Q1 + self.dt * self._rhs(Q1))
        self.Q = self.Q + self._sponge * (self._q_far - self.Q)
        self.steps_done += 1
        r, u, v, p = self._prim(self._bcs(self.Q))
        self.rho = self._interior(r)
        self.ux = self._interior(u)
        self.uy = self._interior(v)
        self._p = self._interior(p)

    def run(self, n):
        for _ in range(int(n)):
            self.step()

    # -- derived fields ---------------------------------------------------------------------
    def speed(self):
        u = xp.sqrt(self.ux * self.ux + self.uy * self.uy)
        return xp.where(self.solid, 0.0, u)

    def mach(self):
        """Local Mach number - the field this solver exists to be able to compute at all."""
        c = xp.sqrt(GAMMA * self._p / xp.maximum(self.rho, 1e-6))
        return xp.where(self.solid, 0.0, self.speed() / xp.maximum(c, 1e-6))

    def schlieren(self):
        """|grad rho| / rho - the classic compressible-flow visualisation.

        A schlieren image is what a real tunnel shows you of AIR: it is blind to velocity and
        sensitive only to density gradient, so it renders exactly the things only a
        compressible fluid has - shocks as hairline discontinuities, expansion fans, and the
        density wake behind a body. Normalised by rho so a weak wave in thin fluid reads as
        strongly as the same wave in dense fluid.
        """
        gy = 0.5 * (xp.roll(self.rho, -1, axis=0) - xp.roll(self.rho, 1, axis=0))
        gx = 0.5 * (xp.roll(self.rho, -1, axis=1) - xp.roll(self.rho, 1, axis=1))
        g = xp.sqrt(gx * gx + gy * gy) / xp.maximum(self.rho, 1e-6)
        return xp.where(self.solid, 0.0, g)

    def vorticity(self):
        dvdx = (xp.roll(self.uy, -1, axis=1) - xp.roll(self.uy, 1, axis=1)) * 0.5
        dudy = (xp.roll(self.ux, -1, axis=0) - xp.roll(self.ux, 1, axis=0)) * 0.5
        return xp.where(self.solid, 0.0, dvdx - dudy)

    def pressure(self):
        return xp.where(self.solid, 0.0, self._p - self.p_inf)

    def temperature(self):
        return xp.where(self.solid, 0.0, GAMMA * self._p / xp.maximum(self.rho, 1e-6))

    def q_criterion(self):
        dudx = (xp.roll(self.ux, -1, axis=1) - xp.roll(self.ux, 1, axis=1)) * 0.5
        dudy = (xp.roll(self.ux, -1, axis=0) - xp.roll(self.ux, 1, axis=0)) * 0.5
        dvdx = (xp.roll(self.uy, -1, axis=1) - xp.roll(self.uy, 1, axis=1)) * 0.5
        dvdy = (xp.roll(self.uy, -1, axis=0) - xp.roll(self.uy, 1, axis=0)) * 0.5
        s12 = 0.5 * (dudy + dvdx)
        o12 = 0.5 * (dudy - dvdx)
        q = 0.5 * (2.0 * o12 * o12 - (dudx * dudx + dvdy * dvdy + 2.0 * s12 * s12))
        return xp.where(self.solid, 0.0, q)

    # -- diagnostics ---------------------------------------------------------------------------
    def health(self):
        """Max |u| - which in these units IS the peak local Mach number."""
        return float(asnumpy(xp.sqrt(self.ux ** 2 + self.uy ** 2).max()))

    @property
    def health_limit(self):
        """Abort threshold. NOT the LBM's 0.45: here a velocity of 1 IS Mach 1, and a local
        supersonic pocket is the physics this solver was written for rather than a failure.
        What is genuinely wrong is a runaway, so the guard sits well above anything this tunnel
        can legitimately produce."""
        return 3.0

    def cfl_now(self):
        """Achieved CFL this instant. The fixed dt is only safe while this stays under ~1."""
        c = xp.sqrt(GAMMA * self._p / xp.maximum(self.rho, 1e-6))
        return float(asnumpy((xp.abs(self.ux) + xp.abs(self.uy) + c).max())) * self.dt

    def describe(self):
        return (f"CNS compressible (gamma={GAMMA}) | {self.nx}x{self.ny} | "
                f"M_inf={self.u0:.4f} | Re={self.re:g} on L={self.ref_len:g} | "
                f"mu={self.mu:.6f} | dt={self.dt:.4f} | CFL_design={self.cfl:g}")
