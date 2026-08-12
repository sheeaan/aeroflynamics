"""bench.py - measure the LBM hot loop on every backend this machine can offer.

The solver is written against an array module (`wt/gpu.py`), so the backend question is
"which module", and that is an empirical question rather than an architectural one. This
file answers it by running the SAME kernel - collide with the Smagorinsky closure, sponge,
stream, half-way bounce-back - under each candidate and timing it.

It also CHECKS THE ANSWERS AGREE before reporting any timing. A backend that is fast and
wrong is worse than no backend at all, and float32 reductions on a GPU do not associate the
same way they do on a CPU, so "agrees" here means to a stated tolerance rather than bitwise.

Backends:
  numpy      the baseline the solver already runs on
  numba      the same maths as fused parallel loops over 8 CPU threads. The win is not
             arithmetic, it is memory: the numpy path materialises ~30 temporaries of shape
             (9, ny, nx) per step and numba materialises none.
  torch-cpu  sanity reference for the torch path, and occasionally faster than numpy because
             ATen threads its elementwise kernels
  torch-xpu  Intel Arc via torch.xpu

Run:
    python tools/bench.py
    python tools/bench.py --nx 960 --ny 400 --steps 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wt import shapes                                    # noqa: E402
from wt.lbm import CX, CY, W, OPP, equilibrium           # noqa: E402

CSM2 = 0.16 ** 2


# --- problem setup -------------------------------------------------------------------------
def make_case(nx, ny):
    """A representative scene: a NACA section at incidence, blocking the tunnel a bit."""
    chord = ny * 0.55
    poly = shapes.place(shapes.naca4("2412"), chord, nx * 0.33, ny * 0.5, 9.0)
    solid = shapes.rasterize([poly], nx, ny)
    link = np.zeros((9, ny, nx), dtype=bool)
    for k in range(1, 9):
        ahead = np.roll(solid, (-int(CY[k]), -int(CX[k])), axis=(0, 1))
        link[k] = (~solid) & ahead
    # graded outlet sponge, as in LBM.__init__
    d = (nx - 1 - np.arange(nx, dtype=np.float32)) / 16.0
    sponge = (np.clip(1.0 - d, 0.0, 1.0) ** 2 * 0.35).reshape(1, 1, nx)
    return solid, link, sponge, chord


def initial_f(nx, ny, u0):
    rho = np.ones((1, 1), np.float32)
    ux = np.full((1, 1), u0, np.float32)
    uy = np.zeros((1, 1), np.float32)
    feq = equilibrium(rho, ux, uy,
                      CX.reshape(9, 1, 1).astype(np.float32),
                      CY.reshape(9, 1, 1).astype(np.float32),
                      W.reshape(9, 1, 1))
    return np.broadcast_to(feq.reshape(9, 1, 1), (9, ny, nx)).astype(np.float32).copy()


# --- numpy -----------------------------------------------------------------------------------
def step_numpy(f, solid, link, sponge, feq_free, tau, u0):
    cx = CX.reshape(9, 1, 1).astype(np.float32)
    cy = CY.reshape(9, 1, 1).astype(np.float32)
    w = W.reshape(9, 1, 1).astype(np.float32)

    f[2, 0, :] = f[4, 0, :]; f[5, 0, :] = f[8, 0, :]; f[6, 0, :] = f[7, 0, :]
    f[4, -1, :] = f[2, -1, :]; f[7, -1, :] = f[6, -1, :]; f[8, -1, :] = f[5, -1, :]
    for k in (3, 6, 7):
        f[k, :, -1] = f[k, :, -2]

    rho = np.maximum(f.sum(axis=0), 1e-6)
    ux = (cx * f).sum(axis=0) / rho
    uy = (cy * f).sum(axis=0) / rho
    rho[:, 0] = 1.0; ux[:, 0] = u0; uy[:, 0] = 0.0

    feq = equilibrium(rho, ux, uy, cx, cy, w)
    fneq = f - feq
    pxx = (cx * cx * fneq).sum(axis=0)
    pyy = (cy * cy * fneq).sum(axis=0)
    pxy = (cx * cy * fneq).sum(axis=0)
    pi = np.sqrt(2.0 * (pxx * pxx + pyy * pyy + 2.0 * pxy * pxy))
    tau_e = 0.5 * (tau + np.sqrt(tau * tau + 18.0 * 1.4142135 * CSM2 * pi / rho))
    fcol = f - (1.0 / tau_e) * fneq
    fcol += sponge * (feq_free - fcol)
    fcol[:, :, 0] = feq[:, :, 0]

    fout = np.empty_like(fcol)
    fout[0] = fcol[0]
    for k in range(1, 9):
        fout[k] = np.roll(fcol[k], (int(CY[k]), int(CX[k])), axis=(0, 1))
    for k in range(1, 9):
        fout[int(OPP[k])] = np.where(link[k], fcol[k], fout[int(OPP[k])])
    return fout


# --- torch -------------------------------------------------------------------------------------
def make_torch_step(device):
    import torch

    cx = torch.tensor(CX.reshape(9, 1, 1), dtype=torch.float32, device=device)
    cy = torch.tensor(CY.reshape(9, 1, 1), dtype=torch.float32, device=device)
    w = torch.tensor(W.reshape(9, 1, 1), dtype=torch.float32, device=device)

    def teq(rho, ux, uy):
        cu = 3.0 * (cx * ux + cy * uy)
        usq = 1.5 * (ux * ux + uy * uy)
        return w * rho * (1.0 + cu + 0.5 * cu * cu - usq)

    def step(f, solid, link, sponge, feq_free, tau, u0):
        f[2, 0, :] = f[4, 0, :]; f[5, 0, :] = f[8, 0, :]; f[6, 0, :] = f[7, 0, :]
        f[4, -1, :] = f[2, -1, :]; f[7, -1, :] = f[6, -1, :]; f[8, -1, :] = f[5, -1, :]
        for k in (3, 6, 7):
            f[k, :, -1] = f[k, :, -2]

        rho = torch.clamp(f.sum(dim=0), min=1e-6)
        ux = (cx * f).sum(dim=0) / rho
        uy = (cy * f).sum(dim=0) / rho
        rho[:, 0] = 1.0; ux[:, 0] = u0; uy[:, 0] = 0.0

        feq = teq(rho, ux, uy)
        fneq = f - feq
        pxx = (cx * cx * fneq).sum(dim=0)
        pyy = (cy * cy * fneq).sum(dim=0)
        pxy = (cx * cy * fneq).sum(dim=0)
        pi = torch.sqrt(2.0 * (pxx * pxx + pyy * pyy + 2.0 * pxy * pxy))
        tau_e = 0.5 * (tau + torch.sqrt(tau * tau + 18.0 * 1.4142135 * CSM2 * pi / rho))
        fcol = f - (1.0 / tau_e) * fneq
        fcol = fcol + sponge * (feq_free - fcol)
        fcol[:, :, 0] = feq[:, :, 0]

        fout = torch.empty_like(fcol)
        fout[0] = fcol[0]
        for k in range(1, 9):
            fout[k] = torch.roll(fcol[k], (int(CY[k]), int(CX[k])), dims=(0, 1))
        for k in range(1, 9):
            o = int(OPP[k])
            fout[o] = torch.where(link[k], fcol[k], fout[o])
        return fout

    return step


# --- numba ---------------------------------------------------------------------------------------
def make_numba_step():
    from numba import njit, prange

    CXi = CX.astype(np.int64)
    CYi = CY.astype(np.int64)
    Wf = W.astype(np.float32)
    OPPi = OPP.astype(np.int64)

    @njit(parallel=True, fastmath=True, cache=True)
    def _kernel(f, fout, fcol, solid, link, sponge, feq_free, tau, u0, cxi, cyi, wf, oppi):
        ny, nx = solid.shape
        # --- collide (fused: no temporaries materialised anywhere) --------------------
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
                    ux = u0
                    uy = 0.0
                usq = 1.5 * (ux * ux + uy * uy)
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
                pi = np.sqrt(2.0 * (pxx * pxx + pyy * pyy + 2.0 * pxy * pxy))
                tau_e = 0.5 * (tau + np.sqrt(tau * tau + 18.0 * 1.4142135 * CSM2 * pi / r))
                om = 1.0 / tau_e
                sp = sponge[i]
                for k in range(9):
                    cu = 3.0 * (cxi[k] * ux + cyi[k] * uy)
                    fe = wf[k] * r * (1.0 + cu + 0.5 * cu * cu - usq)
                    v = f[k, j, i] - om * (f[k, j, i] - fe)
                    v += sp * (feq_free[k] - v)
                    if i == 0:
                        v = fe
                    fcol[k, j, i] = v
        # --- stream -------------------------------------------------------------------
        for k in range(9):
            ck_y = cyi[k]
            ck_x = cxi[k]
            for j in prange(ny):
                js = (j - ck_y) % ny
                for i in range(nx):
                    fout[k, j, i] = fcol[k, (js), (i - ck_x) % nx]
        # --- half-way bounce-back ------------------------------------------------------
        for k in range(1, 9):
            o = oppi[k]
            for j in prange(ny):
                for i in range(nx):
                    if link[k, j, i]:
                        fout[o, j, i] = fcol[k, j, i]

    def step(f, fout, fcol, solid, link, sponge1d, feq_free1d, tau, u0):
        # side and outlet BCs stay in numpy: they touch 6 rows and 3 columns, so JITing them
        # would add compile time to save nothing measurable
        f[2, 0, :] = f[4, 0, :]; f[5, 0, :] = f[8, 0, :]; f[6, 0, :] = f[7, 0, :]
        f[4, -1, :] = f[2, -1, :]; f[7, -1, :] = f[6, -1, :]; f[8, -1, :] = f[5, -1, :]
        for k in (3, 6, 7):
            f[k, :, -1] = f[k, :, -2]
        _kernel(f, fout, fcol, solid, link, sponge1d, feq_free1d,
                np.float32(tau), np.float32(u0), CXi, CYi, Wf, OPPi)
        return fout

    return step


# --- driver -----------------------------------------------------------------------------------
def run(name, fn, steps, warmup, sync=None):
    """Time `steps` calls after `warmup` untimed ones.

    Warmup counts differ between backends - numba has to JIT, torch has to build and cache its
    kernels - so the STATE each backend ends on after this is not comparable, and the caller
    must not diff it. Correctness is checked separately by `check`, which runs every backend
    the same fixed number of steps from the same initial condition.

    An earlier version diffed the post-timing state and reported torch as deviating by 8e-2,
    which looked exactly like a real numerical disagreement. It was the extra five warmup steps.
    """
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / steps * 1000.0


def check(advance, n):
    """Advance a fresh copy of the initial state exactly `n` steps and return it."""
    for _ in range(n):
        advance()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=640)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--u0", type=float, default=0.08)
    ap.add_argument("--re", type=float, default=6000.0)
    a = ap.parse_args(argv)

    nx, ny = a.nx, a.ny
    solid, link, sponge, chord = make_case(nx, ny)
    nu = a.u0 * chord / a.re
    tau = np.float32(3.0 * nu + 0.5)
    f0 = initial_f(nx, ny, a.u0)
    feq_free = f0[:, :1, :1].copy()

    cells = nx * ny
    print(f"lattice {nx}x{ny} = {cells:,} cells | chord {chord:.0f} | tau {tau:.5f} | "
          f"{a.steps} steps (+{a.warmup} warmup)")
    print(f"solid cells {int(solid.sum()):,} | boundary links {int(link.sum()):,}")
    print()

    results = {}
    devs = {}
    NCHK = 20                       # steps used for the correctness comparison

    # -- numpy (baseline for both timing and correctness)
    state = {"f": f0.copy()}

    def numpy_call():
        state["f"] = step_numpy(state["f"], solid, link, sponge, feq_free, tau, a.u0)
    check(numpy_call, NCHK)
    ref = state["f"].copy()
    ref_scale = float(np.abs(ref).max())

    state["f"] = f0.copy()
    ms = run("numpy", numpy_call, a.steps, a.warmup)
    results["numpy"] = ms
    print(f"  {'numpy':<12} {ms:8.2f} ms/step   {cells/ms/1e3:8.2f} Mcell/s   baseline")

    # -- numba
    try:
        nb_step = make_numba_step()
        sp1 = np.ascontiguousarray(sponge.reshape(-1))
        ff1 = np.ascontiguousarray(feq_free.reshape(-1))
        st = {"f": f0.copy(), "o": np.empty_like(f0)}
        fcol = np.empty_like(f0)

        def numba_call():
            r = nb_step(st["f"], st["o"], fcol, solid, link, sp1, ff1, tau, a.u0)
            st["f"], st["o"] = r, st["f"]
        check(numba_call, NCHK)
        devs["numba"] = float(np.abs(st["f"] - ref).max()) / ref_scale

        st["f"], st["o"] = f0.copy(), np.empty_like(f0)
        ms = run("numba", numba_call, a.steps, max(a.warmup, 2))
        results["numba"] = ms
        print(f"  {'numba':<12} {ms:8.2f} ms/step   {cells/ms/1e3:8.2f} Mcell/s   "
              f"{results['numpy']/ms:5.2f}x   max|dev| {devs['numba']:.2e}")
    except Exception as e:
        print(f"  {'numba':<12} unavailable: {type(e).__name__}: {e}")

    # -- torch
    try:
        import torch
        print(f"\n  torch {torch.__version__}  xpu_available={torch.xpu.is_available()}"
              if hasattr(torch, "xpu") else f"\n  torch {torch.__version__}  (no xpu attr)")
        devices = ["cpu"]
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices.append("xpu")
            try:
                print(f"  xpu device: {torch.xpu.get_device_name(0)}")
            except Exception:
                pass
        for dev in devices:
            try:
                tstep = make_torch_step(dev)
                tsolid = torch.tensor(solid, device=dev)
                tlink = torch.tensor(link, device=dev)
                tsp = torch.tensor(sponge, device=dev)
                tff = torch.tensor(feq_free, device=dev)
                stt = {"f": torch.tensor(f0, device=dev)}

                def torch_call():
                    stt["f"] = tstep(stt["f"], tsolid, tlink, tsp, tff, tau, a.u0)
                check(torch_call, NCHK)
                got = stt["f"].cpu().numpy()
                devs[f"torch-{dev}"] = float(np.abs(got - ref).max()) / ref_scale

                stt["f"] = torch.tensor(f0, device=dev)
                sync = (lambda: torch.xpu.synchronize()) if dev == "xpu" else None
                ms = run(f"torch-{dev}", torch_call, a.steps, max(a.warmup, 10), sync=sync)
                results[f"torch-{dev}"] = ms
                print(f"  {'torch-' + dev:<12} {ms:8.2f} ms/step   {cells/ms/1e3:8.2f} Mcell/s"
                      f"   {results['numpy']/ms:5.2f}x   max|dev| {devs['torch-' + dev]:.2e}")
            except Exception as e:
                print(f"  {'torch-' + dev:<12} failed: {type(e).__name__}: {e}")
    except ImportError:
        print("\n  torch        not installed")

    print()
    bad = {k: v for k, v in devs.items() if v > 1e-4}
    if bad:
        print("DISAGREEMENT - these backends do not reproduce the numpy result:")
        for k, v in bad.items():
            print(f"    {k:<12} max|dev| {v:.2e}  (float32 roundoff over "
                  f"{NCHK} steps should be ~1e-6)")
        print("  Do not choose one of these on speed until it agrees.")
        print()

    ok = {k: v for k, v in results.items() if devs.get(k, 0.0) <= 1e-4}
    best = min(ok, key=ok.get)
    print(f"fastest CORRECT backend: {best} at {results[best]:.2f} ms/step "
          f"({results['numpy'] / results[best]:.2f}x over numpy)")
    print()
    print("A 300-frame clip at 20 steps/frame = 6,000 steps:")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"    {k:<12} {6000 * v / 1000 / 60:6.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
