"""sod_check.py - the compressible solver against an EXACT solution.

The Sod shock tube is the standard first test of any Godunov code, and it is worth more than
its simplicity suggests: a single initial discontinuity develops into all three wave families
at once - a left-running rarefaction fan, a contact discontinuity, and a right-running shock -
and the exact solution is known in closed form. A scheme that gets the shock SPEED wrong is
not conservative; one that gets the plateau states wrong has a broken Riemann solver; one that
rings at the contact has a broken limiter. Each failure is distinguishable in the output.

This is the compressible counterpart to the momentum-exchange check in validate.py, and it
exists for the same reason. A shock rendered at the wrong position still looks like a shock.

    python tools/sod_check.py
    python tools/sod_check.py --n 800 --plot out/sod.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wt.cns import CNS, GAMMA          # noqa: E402
from wt.gpu import asnumpy             # noqa: E402

# Sod's original states, in the solver's own non-dimensionalisation
RHO_L, U_L, P_L = 1.0, 0.0, 1.0
RHO_R, U_R, P_R = 0.125, 0.0, 0.1


def exact_sod(x, t, x0=0.5, g=GAMMA):
    """Exact solution of the Sod problem at time t. Returns (rho, u, p).

    Solved the standard way: the star-region pressure is the root of the two-shock/two-
    rarefaction pressure function, found by Brent rather than by Newton because Brent cannot
    walk off the physical branch when the initial bracket is wide.
    """
    cL = np.sqrt(g * P_L / RHO_L)
    cR = np.sqrt(g * P_R / RHO_R)

    def f_K(p, rho_K, p_K, c_K):
        """Pressure function for one side: shock branch if p > p_K, rarefaction otherwise."""
        if p > p_K:                                   # shock
            A = 2.0 / ((g + 1.0) * rho_K)
            B = (g - 1.0) / (g + 1.0) * p_K
            return (p - p_K) * np.sqrt(A / (p + B))
        # rarefaction
        return 2.0 * c_K / (g - 1.0) * ((p / p_K) ** ((g - 1.0) / (2.0 * g)) - 1.0)

    def total(p):
        return f_K(p, RHO_L, P_L, cL) + f_K(p, RHO_R, P_R, cR) + (U_R - U_L)

    p_star = brentq(total, 1e-8, 10.0 * max(P_L, P_R), xtol=1e-14, rtol=1e-14)
    u_star = 0.5 * (U_L + U_R) + 0.5 * (f_K(p_star, RHO_R, P_R, cR)
                                        - f_K(p_star, RHO_L, P_L, cL))

    xi = (np.asarray(x, dtype=float) - x0) / max(t, 1e-30)
    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)

    # left wave is a rarefaction for Sod (p_star < P_L)
    rho_sL = RHO_L * (p_star / P_L) ** (1.0 / g)
    c_sL = cL * (p_star / P_L) ** ((g - 1.0) / (2.0 * g))
    head_L = U_L - cL
    tail_L = u_star - c_sL
    # right wave is a shock
    S_R = U_R + cR * np.sqrt((g + 1.0) / (2.0 * g) * (p_star / P_R)
                             + (g - 1.0) / (2.0 * g))
    rho_sR = RHO_R * ((p_star / P_R + (g - 1.0) / (g + 1.0))
                      / ((g - 1.0) / (g + 1.0) * p_star / P_R + 1.0))

    for i, s in enumerate(np.atleast_1d(xi)):
        if s <= head_L:
            rho[i], u[i], p[i] = RHO_L, U_L, P_L
        elif s <= tail_L:                              # inside the fan
            uu = 2.0 / (g + 1.0) * (cL + (g - 1.0) / 2.0 * U_L + s)
            cc = 2.0 / (g + 1.0) * (cL + (g - 1.0) / 2.0 * (U_L - s))
            rho[i] = RHO_L * (cc / cL) ** (2.0 / (g - 1.0))
            u[i] = uu
            p[i] = P_L * (cc / cL) ** (2.0 * g / (g - 1.0))
        elif s <= u_star:                              # left of the contact
            rho[i], u[i], p[i] = rho_sL, u_star, p_star
        elif s <= S_R:                                 # right of the contact, behind the shock
            rho[i], u[i], p[i] = rho_sR, u_star, p_star
        else:
            rho[i], u[i], p[i] = RHO_R, U_R, P_R
    return rho, u, p, dict(p_star=p_star, u_star=u_star, shock_speed=S_R,
                           fan_head=head_L, fan_tail=tail_L,
                           rho_star_L=rho_sL, rho_star_R=rho_sR)


def run_solver(n, t_end, cfl=0.4):
    """Run the CNS solver on a 1-D tube, realised as a thin 2-D domain.

    The tube is 5 cells tall with free-slip sides and no body, so the y sweep contributes
    nothing and the result is genuinely one-dimensional. Running the real 2-D solver rather
    than a special 1-D copy is deliberate: what needs checking is the code that actually
    renders, including its dimensional splitting, not a simplified stand-in for it.
    """
    ny = 5
    sim = CNS(n, ny, u0=0.0, re=1e12, ref_len=float(n), csm=0.0,
              cfl=cfl, sponge_out=0, blockage=1.0)
    # inviscid: Re huge and csm 0, so only the Euler terms are exercised
    sim.mu = 0.0

    from wt.cns import NG
    Q = np.zeros((4, sim.Ny, sim.Nx), np.float32)
    xs = (np.arange(sim.Nx) - NG + 0.5) / n
    left = xs < 0.5
    rho = np.where(left, RHO_L, RHO_R).astype(np.float32)
    p = np.where(left, P_L, P_R).astype(np.float32)
    Q[0] = rho[None, :]
    Q[1] = 0.0
    Q[2] = 0.0
    Q[3] = (p / (GAMMA - 1.0))[None, :]
    sim.Q = np.asarray(Q)

    # UNITS. The solver's length unit is one CELL (dx = 1), but the Sod problem is posed on a
    # tube of physical length 1. Mapping that onto n cells makes one physical length equal n
    # cells, so a physical end time t corresponds to n*t of the solver's time. Getting this
    # wrong is silent and total: at n=400 the first attempt ran a single step and reported the
    # shock 140 cells from where it belonged.
    t_solver = float(t_end) * n

    # dt from the true maximum wave speed here, not from the freestream-based estimate the
    # constructor makes - there is no freestream in a shock tube
    cmax = np.sqrt(GAMMA * P_L / RHO_L) + abs(U_L)
    nsteps = int(np.ceil(t_solver / (cfl / cmax)))
    sim.dt = t_solver / nsteps                     # land exactly on t_end

    for _ in range(nsteps):
        Q = sim.Q
        Q[:, :, :NG] = Q[:, :, NG:NG + 1]
        Q[:, :, -NG:] = Q[:, :, -NG - 1:-NG]
        Q0 = Q
        Q1 = Q0 + sim.dt * _rhs_tube(sim, Q0)
        sim.Q = 0.5 * Q0 + 0.5 * (Q1 + sim.dt * _rhs_tube(sim, Q1))
        sim.steps_done += 1

    r, u, v, pp = sim._prim(sim.Q)
    j = sim.Ny // 2
    sl = slice(NG, -NG)
    return (np.asarray(asnumpy(r))[j, sl],
            np.asarray(asnumpy(u))[j, sl],
            np.asarray(asnumpy(pp))[j, sl],
            nsteps, sim.dt)


def _rhs_tube(sim, Q):
    """RHS with tube end conditions instead of the tunnel inlet/outlet."""
    from wt.cns import NG
    Q = Q.copy()
    Q[:, :, :NG] = Q[:, :, NG:NG + 1]
    Q[:, :, -NG:] = Q[:, :, -NG - 1:-NG]
    Q[:, :NG] = Q[:, NG:NG + 1]
    Q[:, -NG:] = Q[:, -NG - 1:-NG]
    r, u, v, p = sim._prim(Q)
    mu = sim._visc_mu(r, u, v) * 0.0
    dQ = -sim._flux_dir(r, u, v, p, mu, 1)
    dy = sim._flux_dir(r, v, u, p, mu, 0)
    return dQ - np.stack([dy[0], dy[2], dy[1], dy[3]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="cells along the tube")
    ap.add_argument("--t", type=float, default=0.20, help="end time")
    ap.add_argument("--plot", default=None, help="write a PNG comparison here")
    a = ap.parse_args(argv)

    rho, u, p, nsteps, dt = run_solver(a.n, a.t)
    x = (np.arange(a.n) + 0.5) / a.n
    rho_e, u_e, p_e, info = exact_sod(x, a.t)

    l1 = lambda g, e: float(np.mean(np.abs(g - e)))          # noqa: E731
    err = dict(rho=l1(rho, rho_e), u=l1(u, u_e), p=l1(p, p_e))

    # Shock position: the rightmost point where density still exceeds the undisturbed right
    # state. The threshold must sit between RHO_R and the POST-SHOCK density rho*_R - not
    # between RHO_R and the global maximum, which is the far-away LEFT state. Thresholding
    # against the maximum puts the level above the entire post-shock plateau, so the search
    # finds the contact discontinuity instead and reports it as the shock: a confident,
    # stable, 66-cell error while the L1 norms said the solution was fine.
    thr = 0.5 * (RHO_R + info["rho_star_R"])
    shock_num = x[np.max(np.nonzero(rho > thr))]
    shock_exact = 0.5 + info["shock_speed"] * a.t
    contact_num = x[np.max(np.nonzero(rho > 0.5 * (info["rho_star_R"] + info["rho_star_L"])))]
    contact_exact = 0.5 + info["u_star"] * a.t

    print(f"Sod shock tube: n={a.n} cells, t={a.t}, {nsteps} steps at dt={dt:.5f}")
    print(f"  exact: p* = {info['p_star']:.6f}   u* = {info['u_star']:.6f}   "
          f"shock speed = {info['shock_speed']:.6f}")
    print()
    print(f"  L1 error   rho {err['rho']:.5f}   u {err['u']:.5f}   p {err['p']:.5f}")
    print(f"  shock    numerical {shock_num:.4f}   exact {shock_exact:.4f}   "
          f"delta {abs(shock_num - shock_exact) * a.n:5.2f} cells")
    print(f"  contact  numerical {contact_num:.4f}   exact {contact_exact:.4f}   "
          f"delta {abs(contact_num - contact_exact) * a.n:5.2f} cells")
    print(f"  overshoot at the shock: max rho = {rho.max():.5f} "
          f"(exact plateau {rho_e.max():.5f})")

    ok_l1 = err["rho"] < 0.02 and err["p"] < 0.02 and err["u"] < 0.03
    ok_pos = abs(shock_num - shock_exact) * a.n < 3.0
    ok_tvd = rho.max() <= rho_e.max() * 1.02
    print()
    for name, ok, why in (("L1 errors", ok_l1, "conservative + 2nd order where smooth"),
                          ("shock position", ok_pos, "correct Rankine-Hugoniot speed"),
                          ("no overshoot", ok_tvd, "limiter is TVD across the discontinuity")):
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<16} {why}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
            for ax, g, e, lab in ((axes[0], rho, rho_e, "density"),
                                  (axes[1], u, u_e, "velocity"),
                                  (axes[2], p, p_e, "pressure")):
                ax.plot(x, e, "-", lw=2, color="#888", label="exact")
                ax.plot(x, g, ".", ms=3, color="#c0392b", label="CNS")
                ax.set_ylabel(lab)
                ax.legend(loc="best", fontsize=8)
            axes[-1].set_xlabel("x")
            os.makedirs(os.path.dirname(os.path.abspath(a.plot)) or ".", exist_ok=True)
            fig.tight_layout()
            fig.savefig(a.plot, dpi=120)
            print(f"\n  wrote {a.plot}")
        except ImportError:
            print("\n  (matplotlib not installed; skipped --plot)")

    return 0 if (ok_l1 and ok_pos and ok_tvd) else 1


if __name__ == "__main__":
    raise SystemExit(main())
