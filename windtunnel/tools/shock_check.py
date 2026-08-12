"""shock_check.py - the IMMERSED BOUNDARY against the oblique-shock relation.

sod_check.py validates the scheme (conservation, Riemann solver, limiter) but it contains no
body at all, so it says nothing about how the solid is imposed. This case is the complement:
supersonic flow over a wedge, where the only thing under test is whether the ghost-cell
boundary makes the flow turn through the angle the geometry actually asks for.

The theta-beta-M relation gives the exact shock angle for a given deflection and Mach number,
with no empirical content whatsoever - it follows from the conservation laws across an oblique
shock. So the measured shock angle is a direct read-out of the EFFECTIVE wedge angle the fluid
saw, which is what a staircase-rasterised ramp puts in question.

MEASURED RESULT (M 2, 15 deg wedge, 220x130, inviscid): wave angle 45.81 deg against an exact
45.34, so the boundary over-turns the flow by +0.34 deg - about 2.3%. A staircase-rasterised
ramp was expected to do noticeably worse than that, since the distance-transform normals along
it oscillate between the tread and riser orientations; the ghost-cell construction evidently
averages that out better than the geometry suggests. The number belongs in the record either
way: it is the honest bound on how far this solver's surface pressures can be trusted.

A WARNING ABOUT THIS TOOL, because it produced two confident wrong answers before it produced
a right one, and both looked like solver defects:

  - the shock detector must search for the strongest POSITIVE density difference scanning DOWN
    a column. Scanning for the largest drop instead locks onto the near-wall gradient a few
    rows above the ramp, which tracks the SURFACE - so the tool reports the ramp angle and
    calls it the wave angle;
  - the fit window must stop before the incident shock leaves the top of the domain. Past that
    point the field contains the wall reflection, and fitting one line through incident and
    reflected points tilts the answer steeper - which is indistinguishable from an immersed
    boundary that over-turns.

    python tools/shock_check.py
    python tools/shock_check.py --mach 2.5 --wedge 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wt.cns import CNS, GAMMA, NG      # noqa: E402
from wt.gpu import asnumpy             # noqa: E402


def theta_from_beta(beta, M, g=GAMMA):
    """The theta-beta-M relation: deflection produced by a shock at wave angle `beta` (rad)."""
    s = np.sin(beta)
    num = 2.0 / np.tan(beta) * (M * M * s * s - 1.0)
    den = M * M * (g + np.cos(2.0 * beta)) + 2.0
    return np.arctan(num / den)


def beta_from_theta(theta, M, g=GAMMA):
    """Weak-solution shock angle for a given deflection. Inverts the relation numerically.

    Two roots exist (weak and strong); the weak one is what a wedge in an unbounded supersonic
    stream actually produces, so the bracket is taken from the Mach angle up to the detachment
    angle - the beta that maximises theta - which isolates the weak branch.
    """
    mu = np.arcsin(1.0 / M)                                    # Mach angle: theta = 0 here
    bs = np.linspace(mu + 1e-6, np.pi / 2 - 1e-6, 20000)
    ts = theta_from_beta(bs, M, g)
    b_det = bs[int(np.argmax(ts))]                             # detachment
    if theta > ts.max():
        raise ValueError(f"theta {np.degrees(theta):.2f} deg exceeds detachment "
                         f"{np.degrees(ts.max()):.2f} deg at M={M}")
    return brentq(lambda b: theta_from_beta(b, M, g) - theta, mu + 1e-9, b_det, xtol=1e-12)


def build_wedge(nx, ny, x_ramp, wedge_deg):
    """Solid mask: a flat floor that turns upward into a ramp at `x_ramp`.

    Lattice row 0 is the TOP of the picture, so the floor is at high j and the ramp climbs
    toward lower j as x increases.
    """
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    rise = np.maximum(i - x_ramp, 0) * np.tan(np.radians(wedge_deg))
    floor_j = (ny - 1) - rise                                  # surface height as a row index
    return j > floor_j


def measure_shock_angle(rho, solid, x_ramp, ny, nx, beta_deg, x0_frac=0.12, x1_frac=0.80):
    """Fit the shock line from the density field and return its angle above horizontal (deg).

    At each column downstream of the ramp corner the shock is the strongest upward density
    jump above the wedge surface. Fitting a line through those points is robust to a couple of
    bad columns in a way that any single column is not.

    THE WINDOW IS BOUNDED BY WHERE THE INCIDENT SHOCK LEAVES THE DOMAIN, computed from the
    expected wave angle rather than taken as a fixed fraction of the domain. At M 2 and 15 deg
    the shock climbs about one row per column, so in a domain only a little wider than it is
    tall the wave exits through the top well before the right-hand edge - and everything past
    that point is the wall REFLECTION coming back down. Fitting a line through incident and
    reflected points together tilts the answer steeper, which reads exactly like an immersed
    boundary that over-turns the flow. A fixed 0.25-0.75 window did that here and produced a
    3.4 deg error that was measurement, not physics.
    """
    xs, ys = [], []
    corner_row = ny - 1
    top_guard = int(ny * 0.12)
    # column at which the incident shock reaches the top guard band, minus a safety margin
    rise = np.tan(np.radians(beta_deg))
    i_exit = x_ramp + (corner_row - top_guard) / max(rise, 1e-6)
    i0 = int(x_ramp + (nx - x_ramp) * x0_frac)
    i1 = int(min(x_ramp + (nx - x_ramp) * x1_frac, x_ramp + 0.85 * (i_exit - x_ramp)))
    for i in range(i0, i1):
        col = rho[:, i]
        solid_col = solid[:, i]
        top_solid = np.argmax(solid_col) if solid_col.any() else ny
        # search strictly in the fluid above the surface, and away from the top wall so a
        # reflected shock cannot be mistaken for the incident one
        lo, hi = top_guard, max(top_guard + 3, top_solid - 2)
        if hi - lo < 5:
            continue
        seg = col[lo:hi]
        grad = np.diff(seg)
        # Row index increases DOWNWARD, and the flow above the shock is undisturbed
        # freestream while the flow below it is compressed - so scanning down a column, density
        # JUMPS UP at the shock and the strongest positive difference is the wave.
        # `argmin` here instead finds the largest DROP, which is the near-wall gradient a few
        # rows above the ramp surface: it tracks the surface rather than the shock, so it
        # reports the ramp angle and calls it the wave angle.
        k = int(np.argmax(grad))
        xs.append(i)
        ys.append(lo + k)
    if len(xs) < 8:
        return float("nan"), 0
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    slope, _ = np.polyfit(xs, ys, 1)
    # ys increases downward, so a shock climbing to the right has NEGATIVE slope in j
    return float(np.degrees(np.arctan(-slope))), len(xs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mach", type=float, default=2.0)
    ap.add_argument("--wedge", type=float, default=15.0, help="ramp half-angle, degrees")
    # 220x130 is deliberate: the compressible solver has no numba path, and the cost climbs
    # sharply once the working set falls out of cache (300x180 measured 172 ms/step against
    # 56 ms at this size - 3x the cost for 1.9x the cells). This size converges the wave angle
    # and finishes in about four minutes.
    ap.add_argument("--nx", type=int, default=220)
    ap.add_argument("--ny", type=int, default=130)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)

    theta = np.radians(a.wedge)
    beta_exact = np.degrees(beta_from_theta(theta, a.mach))
    print(f"oblique shock: M={a.mach}, wedge {a.wedge} deg")
    print(f"  exact weak-shock angle (theta-beta-M): {beta_exact:.3f} deg")

    nx, ny = a.nx, a.ny
    x_ramp = int(nx * 0.22)
    sim = CNS(nx, ny, u0=a.mach, re=1e7, ref_len=float(ny), csm=0.0,
              cfl=0.35, sponge_out=0, blockage=1.6)
    sim.mu = 0.0                       # inviscid: the relation is inviscid, so the test is too
    mask = build_wedge(nx, ny, x_ramp, a.wedge)
    sim.set_solid(mask)
    print(f"  {sim.describe()}")

    # Long enough for the shock to establish. Quoted in FLOW-THROUGH times (domain crossings)
    # rather than steps, so changing the resolution does not silently change how converged the
    # answer is. Three crossings is comfortably past the point where the wave angle stops
    # moving; the startup transient washes out in about one.
    steps = a.steps if a.steps else int(3.0 * nx / (a.mach * sim.dt))
    print(f"  running {steps} steps at dt={sim.dt:.4f} ...", flush=True)
    report = max(1, steps // 6)
    for i in range(steps):
        sim.step()
        if i % report == 0:
            print(f"    {i:6d}/{steps}  max M = {sim.health():.3f}  "
                  f"CFL = {sim.cfl_now():.3f}", flush=True)

    rho = np.asarray(asnumpy(sim.rho))
    solid = np.asarray(asnumpy(sim.solid))
    beta_num, npts = measure_shock_angle(rho, solid, x_ramp, ny, nx, beta_exact)

    print()
    if not np.isfinite(beta_num):
        print("  [FAIL]  could not locate a shock front")
        return 1

    # convert the measured wave angle back into the deflection the fluid actually saw
    theta_eff = np.degrees(theta_from_beta(np.radians(beta_num), a.mach))
    over = theta_eff - a.wedge
    print(f"  measured shock angle: {beta_num:.3f} deg   ({npts} columns fitted)")
    print(f"  implied deflection:   {theta_eff:.3f} deg   vs geometric {a.wedge:.2f} deg")
    print(f"  OVER-TURN:            {over:+.3f} deg  ({100 * over / a.wedge:+.1f}%)")
    print()
    print("  The over-turn is the staircase, not the scheme: a rasterised ramp steps one row")
    print("  every 1/tan(theta) columns and the EDT normals oscillate between tread and riser.")
    print("  Removing it needs a cut-cell or level-set surface, not a tweak to the ghost cells.")
    print()

    ok = abs(beta_num - beta_exact) < 2.5
    print(f"  [{'PASS' if ok else 'FAIL'}]  shock angle within 2.5 deg of exact "
          f"(delta {beta_num - beta_exact:+.3f} deg)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
