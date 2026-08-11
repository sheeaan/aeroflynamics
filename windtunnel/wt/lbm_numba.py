"""lbm_numba.py - the fused LBM step, JIT-compiled and threaded across CPU cores.

WHY THIS EXISTS, given that lbm.py already works
------------------------------------------------
The numpy path is not slow because the arithmetic is slow. It is slow because of MEMORY: a
single step materialises roughly thirty temporaries of shape (9, ny, nx), and at 640x256 that
is about 170 MB of traffic per step for maybe 25 MFLOP of actual work. The kernel is
bandwidth-bound by a wide margin, so the fix is not a faster processor, it is not writing the
temporaries at all.

Fusing the whole collide into one pass over the cells does exactly that. Measured on this
machine (tools/bench.py, 640x256, 8 threads):

    numpy       79.4 ms/step     baseline
    torch-cpu   14.0 ms/step      5.7x
    torch-xpu    5.4 ms/step     14.7x    (Intel Arc 130V)
    numba        2.0 ms/step     40.3x

The integrated GPU loses to threaded CPU here for the same reason: it shares system memory, so
it is bound by the same bandwidth, and it still has to materialise the temporaries that this
kernel avoids. A discrete card with its own high-bandwidth memory would change that ranking.

WHAT IS AND IS NOT IN THE KERNEL
--------------------------------
In: the collide (equilibrium, Smagorinsky, BGK), the sponge, the inlet column, streaming, and
half-way bounce-back with the moving-wall term.

Out, deliberately: the side and outlet boundary conditions touch six rows and three columns, so
JITing them would add compile time to save nothing measurable; and the inlet turbulence lookup
is two interpolated column reads of host-side Python arithmetic, which is cheaper done once per
step outside the kernel than threaded inside it.

CORRECTNESS
-----------
This kernel must reproduce `LBM._step_numpy` to float32 roundoff, and tools/bench.py asserts
exactly that before it will report a timing. If you change the maths here, change it there too
and re-run both bench.py and validate.py - the momentum-exchange force integral reads `_fcol`,
which this kernel writes.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):                              # type: ignore[misc]
        def deco(f):
            return f
        return deco if not a else a[0]

    prange = range                                  # type: ignore[assignment]


@njit(parallel=True, fastmath=True, cache=True)
def _step(f, fout, fcol, rho_o, ux_o, uy_o, solid, link, sponge, feq_free,
          in_ux, in_uy, wux, wuy, has_wall, tau, csm2, cxi, cyi, wf, oppi):
    ny, nx = solid.shape

    # --- collide: equilibrium, Smagorinsky, BGK, sponge, inlet - all in one pass ----------
    for j in prange(ny):
        for i in range(nx):
            r = 0.0
            mx = 0.0
            my = 0.0
            for k in range(9):
                v = f[k, j, i]
                r += v
                mx += cxi[k] * v
                my += cyi[k] * v
            if r < 1e-6:
                r = 1e-6
            ux = mx / r
            uy = my / r
            if i == 0:
                r = 1.0
                ux = in_ux[j]
                uy = in_uy[j]
            rho_o[j, i] = r
            ux_o[j, i] = ux
            uy_o[j, i] = uy

            usq = 1.5 * (ux * ux + uy * uy)
            # first pass over the populations: the non-equilibrium momentum-flux tensor, which
            # is what the Smagorinsky closure needs and which f - feq gives up for free
            pxx = 0.0
            pyy = 0.0
            pxy = 0.0
            for k in range(9):
                cu = 3.0 * (cxi[k] * ux + cyi[k] * uy)
                fe = wf[k] * r * (1.0 + cu + 0.5 * cu * cu - usq)
                fn = f[k, j, i] - fe
                pxx += cxi[k] * cxi[k] * fn
                pyy += cyi[k] * cyi[k] * fn
                pxy += cxi[k] * cyi[k] * fn
            if csm2 > 0.0:
                pi = np.sqrt(2.0 * (pxx * pxx + pyy * pyy + 2.0 * pxy * pxy))
                tau_e = 0.5 * (tau + np.sqrt(tau * tau + 18.0 * 1.4142135 * csm2 * pi / r))
                om = 1.0 / tau_e
            else:
                om = 1.0 / tau

            # sponge is (ny, nx), not (nx,): with side_bc="free" it grades in y as well
            sp = sponge[j, i]
            for k in range(9):
                cu = 3.0 * (cxi[k] * ux + cyi[k] * uy)
                fe = wf[k] * r * (1.0 + cu + 0.5 * cu * cu - usq)
                v = f[k, j, i] - om * (f[k, j, i] - fe)
                v += sp * (feq_free[k, j] - v)
                if i == 0:
                    v = fe               # inlet is pure equilibrium, and overrides the sponge
                fcol[k, j, i] = v

    # --- stream ---------------------------------------------------------------------------
    for k in range(9):
        cy = cyi[k]
        cx = cxi[k]
        for j in prange(ny):
            js = (j - cy) % ny
            for i in range(nx):
                fout[k, j, i] = fcol[k, js, (i - cx) % nx]

    # --- half-way bounce-back ---------------------------------------------------------------
    # Applied after streaming because the reflected population must STAY at its own node;
    # streaming it would carry it to x_b - c_k. This also overwrites exactly the slots that
    # would otherwise have been fed from inside the body.
    for k in range(1, 9):
        o = oppi[k]
        wk6 = 6.0 * wf[k]
        cx = cxi[k]
        cy = cyi[k]
        for j in prange(ny):
            for i in range(nx):
                if link[k, j, i]:
                    v = fcol[k, j, i]
                    if has_wall:
                        js = (j + cy) % ny
                        iss = (i + cx) % nx
                        v -= wk6 * (cx * wux[js, iss] + cy * wuy[js, iss])
                        if v < 0.0:
                            v = 0.0
                    fout[o, j, i] = v


def available():
    return HAVE_NUMBA


def warmup(nx=32, ny=16):
    """Force compilation now rather than inside the first timed frame.

    Without this the JIT cost lands on frame 0 of a render, which reads as a mysterious
    multi-second stall before anything appears.
    """
    if not HAVE_NUMBA:
        return False
    from .lbm import CX, CY, W, OPP
    z9 = np.zeros((9, ny, nx), np.float32)
    _step(z9 + 0.1, np.empty_like(z9), np.empty_like(z9),
          np.zeros((ny, nx), np.float32), np.zeros((ny, nx), np.float32),
          np.zeros((ny, nx), np.float32),
          np.zeros((ny, nx), np.bool_), np.zeros((9, ny, nx), np.bool_),
          np.zeros((ny, nx), np.float32), np.zeros((9, ny), np.float32),
          np.zeros(ny, np.float32), np.zeros(ny, np.float32),
          np.zeros((ny, nx), np.float32), np.zeros((ny, nx), np.float32),
          False, np.float32(0.6), np.float32(0.0256),
          CX.astype(np.int64), CY.astype(np.int64), W.astype(np.float32),
          OPP.astype(np.int64))
    return True
