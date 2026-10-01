"""scenes.py - the compositions, and the CLI that renders them.

A scene owns geometry and nothing else. It answers one question per frame - "where is the
body at time t, and at what incidence" - and hands back a mask. It never touches the flow
field, never nudges a vortex, and never decides what the picture should look like. Everything
downstream of the mask belongs to the solver.

    python scenes.py --list
    python scenes.py aoa_sweep
    python scenes.py karman --frames 300 --field vorticity --tracers 6000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from wt import LBM, describe_backend
from wt import shapes
from wt.cns import CNS
from wt.film import PRESETS as FILM_PRESETS, make as make_film
from wt.render import render_scene, ORIENTATIONS


class Scene:
    """Base: subclasses set `self.mask` in `update()`."""

    name = "scene"
    # WHICH SOLVER. "lbm" is right for almost everything. "cns" is for the phenomena the
    # lattice structurally cannot reach - shocks, real Mach number, stagnation temperature.
    #
    # THE TRAP: `u0` means a different physical quantity in each. In the LBM it is a LATTICE
    # VELOCITY and the Mach number is u0*sqrt(3); in the CNS it IS the Mach number, exactly.
    # Copying a scene between solvers without converting produces a run that is stable,
    # plausible, and at completely the wrong speed.
    solver = "lbm"
    # solver defaults; every one is overridable from the CLI
    nx, ny = 480, 200
    u0 = 0.075
    re = 6000.0
    csm = 0.16
    settle = 500
    steps = 18
    frames = 240
    fps = 30
    field = "vorticity"
    vlim = None
    tracers = 0
    turbulence = 0.0
    ref_len = 100.0
    orientation = "h"
    csm = 0.16
    dye = None                 # a wt.dye.Dye, built by the scene in build()
    film = None                # a preset name from wt.film.PRESETS, or None
    bodies = None              # a wt.bodies.BodySystem, built by the scene in build()

    def reference_length(self):
        """The length Re and the force coefficients are built on - chord, or diameter.

        Answered BEFORE the solver exists, because `re` and `ref_len` together set the
        viscosity, so the solver cannot be constructed without it. Scenes that carry a chord
        or a diameter override this; anything else falls back to the class default.
        """
        for attr in ("chord", "diameter"):
            v = getattr(self, attr, None)
            if v is not None:
                return float(v)
        return float(self.ref_len)

    def build(self, sim):
        """Optional one-time setup once the solver exists."""

    def update(self, sim, t, frame):
        raise NotImplementedError

    def hud(self, sim):
        return []


# --- airfoil scenes ---------------------------------------------------------------------
class AoASweep(Scene):
    """A NACA section swept up through stall and back down.

    The point of sweeping BOTH ways is hysteresis. Stall is not a line the flow crosses
    reversibly: once the boundary layer has separated, reattaching it needs a noticeably lower
    incidence than the one that broke it, because the separated shear layer has to re-close
    against its own wake. Sweep up only and you would never see that - the clip would read as
    a threshold rather than as a loop, and the loop is the physics.
    """

    name = "aoa_sweep"
    nx, ny = 520, 220
    chord = 110.0
    profile = "2412"
    aoa_max = 18.0
    re = 6000.0
    u0 = 0.075
    tracers = 5000
    turbulence = 0.008
    frames = 300

    def build(self, sim):
        self.pts = shapes.naca4(self.profile)
        self.ref_len = self.chord
        self.cx = sim.nx * 0.34
        self.cy = sim.ny * 0.5
        self.aoa = 0.0

    def update(self, sim, t, frame):
        # triangular sweep 0 -> max -> 0 over the whole clip
        span = self.frames / float(self.fps)
        phase = (t / span) * 2.0
        self.aoa = self.aoa_max * (phase if phase <= 1.0 else 2.0 - phase)
        poly = shapes.place(self.pts, self.chord, self.cx, self.cy, self.aoa)
        self.mask = shapes.rasterize([poly], sim.nx, sim.ny)

    def hud(self, sim):
        return [f"NACA {self.profile}   Re {self.re:,.0f}", f"AoA {self.aoa:5.2f} deg"]


class Stall(Scene):
    """Held deep in stall, so the shedding cycle is the subject rather than the transition.

    A fixed high incidence lets the leading-edge separation reach a genuinely periodic state:
    a vortex forms at the nose, convects over the upper surface, sheds into the wake, and the
    next one starts. Sweeping past this angle shows it for a few frames; parking here shows it
    as a rhythm, which is the thing worth looking at.
    """

    name = "stall"
    nx, ny = 520, 220
    chord = 110.0
    profile = "2412"
    aoa = 20.0
    re = 6000.0
    u0 = 0.075
    tracers = 6000
    turbulence = 0.012
    frames = 260

    def build(self, sim):
        self.pts = shapes.naca4(self.profile)
        self.ref_len = self.chord
        poly = shapes.place(self.pts, self.chord, sim.nx * 0.34, sim.ny * 0.5, self.aoa)
        self._mask = shapes.rasterize([poly], sim.nx, sim.ny)

    def update(self, sim, t, frame):
        self.mask = self._mask       # geometry is fixed; only the flow evolves

    def hud(self, sim):
        return [f"NACA {self.profile}   Re {self.re:,.0f}", f"AoA {self.aoa:5.2f} deg  STALLED"]


class TriFoil(Scene):
    """Three sections at different incidence in the same stream, for direct comparison.

    Same freestream, same Reynolds number, same instant - so any difference between the three
    wakes is caused by incidence alone. Running them as three separate clips would not support
    that claim nearly as well, because the turbulence realisation would differ.
    """

    name = "tri_foil"
    nx, ny = 520, 300
    chord = 78.0
    profile = "0012"
    angles = (4.0, 11.0, 18.0)
    re = 5000.0
    u0 = 0.07
    tracers = 6000
    turbulence = 0.010
    frames = 260

    def build(self, sim):
        self.pts = shapes.naca4(self.profile)
        self.ref_len = self.chord
        polys = [shapes.place(self.pts, self.chord, sim.nx * 0.32,
                              sim.ny * f, a)
                 for f, a in zip((0.22, 0.5, 0.78), self.angles)]
        self._mask = shapes.rasterize(polys, sim.nx, sim.ny)

    def update(self, sim, t, frame):
        self.mask = self._mask

    def hud(self, sim):
        a = "  ".join(f"{x:.0f}" for x in self.angles)
        return [f"NACA {self.profile}   Re {self.re:,.0f}", f"AoA {a} deg (top to bottom)"]


# --- bluff-body scenes ------------------------------------------------------------------
class Karman(Scene):
    """A circular cylinder shedding a von Karman street - the canonical validation case.

    Deliberately low Reynolds number. At Re 150 the wake is periodic and two-dimensional, so a
    2-D solver is not lying, and both the drag coefficient and the shedding frequency have
    well-established experimental values to check against. tools/validate.py does exactly that.
    """

    name = "karman"
    nx, ny = 480, 200
    diameter_frac = 0.17          # of the tunnel height, so the blockage is resolution-proof
    re = 150.0
    u0 = 0.07
    csm = 0.0                 # laminar and well resolved: no subgrid model wanted
    tracers = 6000
    frames = 300
    steps = 22
    settle = 2500
    field = "vorticity"
    vlim = 0.035

    def reference_length(self):
        return self.ny * self.diameter_frac

    def build(self, sim):
        self.pts = shapes.cylinder()
        self.diameter = self.ref_len
        poly = shapes.place(self.pts, self.diameter, sim.nx * 0.26, sim.ny * 0.5, 0.0,
                            pivot=0.5)
        self._mask = shapes.rasterize([poly], sim.nx, sim.ny)

    def update(self, sim, t, frame):
        self.mask = self._mask

    def hud(self, sim):
        return [f"cylinder  D {self.diameter:.0f} cells", f"Re {self.re:,.0f}"]


class PlungingFoil(Scene):
    """A section oscillating in heave - an unsteady case the mask approach gets for free.

    A body that MOVES is where the bounce-back boundary earns its keep: the mask is simply
    re-rasterised at the new position each frame, and the moving-wall term supplies the
    momentum. There is no remeshing, no ALE formulation, and no mesh-motion solver, because
    there is no mesh.
    """

    name = "plunge"
    nx, ny = 520, 260
    chord = 96.0
    profile = "0015"
    amplitude = 34.0          # cells, peak-to-mean
    period = 2.4              # seconds
    aoa = 6.0
    re = 4000.0
    u0 = 0.07
    tracers = 6000
    frames = 260

    def build(self, sim):
        self.pts = shapes.naca4(self.profile)
        self.ref_len = self.chord
        self.cx = sim.nx * 0.34
        self.cy0 = sim.ny * 0.5
        self._prev = None

    def update(self, sim, t, frame):
        cy = self.cy0 + self.amplitude * np.sin(2 * np.pi * t / self.period)
        poly = shapes.place(self.pts, self.chord, self.cx, cy, self.aoa)
        self.mask = shapes.rasterize([poly], sim.nx, sim.ny)
        if self._prev is not None:
            wux, wuy = shapes.wall_velocity(self.mask, [self._prev], [poly],
                                            dt_steps=self.steps)
            sim.set_wall_velocity(wux, wuy)
        self._prev = poly
        self.cy = cy

    def hud(self, sim):
        return [f"NACA {self.profile}  plunging", f"AoA {self.aoa:.0f} deg   T {self.period}s"]


# --- compressible scenes (CNS) ------------------------------------------------------------
class MachCone(Scene):
    """A pointed projectile at Mach 2 - bow shock, Mach cone, and the wake behind it.

    This is the scene the lattice-Boltzmann solver cannot produce at any parameter setting.
    D2Q9 is isothermal with an effective gamma of 1 and has no shock-capturing mechanism, so
    it degrades continuously above about M 0.3 and then simply fails. Everything visible here
    - the hairline attached shock at the nose, the expansion at the shoulder, the recompression
    behind the base - exists only because the compressible solver carries real energy and a
    real equation of state.

    Drawn as SCHLIEREN, which is what a real supersonic tunnel shows you: blind to velocity,
    sensitive only to density gradient, so shocks appear as the discontinuities they are
    instead of being smeared into a speed map.
    """

    name = "mach_cone"
    solver = "cns"
    nx, ny = 560, 320
    u0 = 2.0                   # MACH 2 - in this solver u0 IS the Mach number
    re = 2.0e6
    csm = 0.20
    length = 120.0
    field = "schlieren"
    frames = 240
    steps = 26
    settle = 900
    tracers = 0
    blockage = 1.8

    def build(self, sim):
        self.pts = shapes.ogive(caliber=3.4, nose=0.45)
        self.ref_len = self.length
        poly = shapes.place(self.pts, self.length, sim.nx * 0.30, sim.ny * 0.5, 0.0, pivot=0.5)
        self._mask = shapes.rasterize([poly], sim.nx, sim.ny)

    def update(self, sim, t, frame):
        self.mask = self._mask

    def hud(self, sim):
        return [f"ogive  M {self.u0:.2f}", "schlieren  |grad rho| / rho"]


class MachSweep(Scene):
    """A section throttled from subsonic toward sonic - the transonic rise, as it happens.

    Held at fixed incidence and swept in Mach rather than angle, so the only thing changing is
    compressibility. The supercritical pocket appears on the upper surface part-way through and
    terminates in a shock; that shock is captured by the scheme at the correct jump conditions
    rather than drawn.

    NOTE the freestream Mach is changed by rebuilding the inlet each frame, not by editing
    `u0` - `dt` is fixed at construction from a bound on max(|u| + c), and moving u0 underneath
    it would quietly invalidate that bound.
    """

    name = "mach_sweep"
    solver = "cns"
    nx, ny = 560, 300
    u0 = 0.82                  # the DESIGN Mach, used to fix dt; the sweep stays below it
    re = 1.2e6
    csm = 0.20
    chord = 130.0
    aoa = 2.0
    m_lo, m_hi = 0.55, 0.80
    field = "schlieren"
    frames = 240
    steps = 24
    settle = 1200
    blockage = 2.2

    def build(self, sim):
        self.pts = shapes.naca4("0012")
        self.ref_len = self.chord
        poly = shapes.place(self.pts, self.chord, sim.nx * 0.33, sim.ny * 0.5, self.aoa)
        self._mask = shapes.rasterize([poly], sim.nx, sim.ny)
        self.mach = self.m_lo

    def update(self, sim, t, frame):
        self.mask = self._mask
        span = self.frames / float(self.fps)
        phase = min(t / span, 1.0)
        self.mach = self.m_lo + (self.m_hi - self.m_lo) * phase
        prof = np.full(sim.ny, self.mach, dtype=np.float32)
        sim.set_inlet(prof)
        sim.set_farfield(prof)

    def hud(self, sim):
        return [f"NACA 0012  AoA {self.aoa:.0f} deg", f"M {self.mach:5.3f}"]


# --- dye and free-body scenes ---------------------------------------------------------------
class DyeRake(Scene):
    """A comb of dye filaments drawn through a foil - streaklines, not a colour map.

    A scalar field shows you where the flow is fast. Streaklines show you where a given parcel
    of fluid WENT, which is the only way to see that the air over the upper surface and the air
    under it arrive at the trailing edge at different times. The gaps between bands are doing
    as much work as the bands: without them the wake fills in and reads as a cloud.
    """

    name = "dye_rake"
    nx, ny = 560, 240
    chord = 100.0
    profile = "2412"
    aoa = 9.0
    re = 5000.0
    u0 = 0.075
    turbulence = 0.006
    frames = 260
    steps = 18
    field = "speed"
    tracers = 0

    def build(self, sim):
        from wt.dye import Dye
        self.pts = shapes.naca4(self.profile)
        self.ref_len = self.chord
        poly = shapes.place(self.pts, self.chord, sim.nx * 0.38, sim.ny * 0.5, self.aoa)
        self._mask = shapes.rasterize([poly], sim.nx, sim.ny)
        self.dye = Dye(sim.nx, sim.ny, diffusivity=0.0015, n_species=2)
        # two colours from two interleaved combs, so neighbouring filaments stay
        # distinguishable once the shear starts folding them together
        self.dye.inject_bands(0, x=6, n_bands=7, duty=0.30, width=2)
        self.dye.inject_bands(1, x=6, n_bands=7, duty=0.30, width=2)
        self.dye._sources[1] = (1, np.roll(self.dye._sources[1][1],
                                           int(sim.ny / 14), axis=0), 1.0, 0.25)

    def update(self, sim, t, frame):
        self.mask = self._mask

    def hud(self, sim):
        return [f"NACA {self.profile}  AoA {self.aoa:.0f} deg", "dye streaklines"]


class Tumble(Scene):
    """Free bodies dropped into a stream - nothing here has a path, only initial conditions.

    Every position in this clip is the integral of the momentum the fluid handed to a surface.
    That is only worth watching if the force integral is right, which is why tools/validate.py
    exists: a drag 3x too large would look like heavier objects in thicker fluid, and no frame
    would betray it.

    Reynolds number is deliberately modest. Above ~900 the per-step load fluctuations on a
    rasterised body outrun the low-pass filter and the bodies buzz.
    """

    name = "tumble"
    nx, ny = 520, 300
    re = 700.0
    u0 = 0.06
    csm = 0.16
    frames = 300
    steps = 16
    settle = 300
    field = "vorticity"
    tracers = 3000

    def reference_length(self):
        return self.ny * 0.14          # representative body size, for Re and the coefficients

    def build(self, sim):
        from wt.bodies import RigidBody, BodySystem
        rng = np.random.default_rng(4)
        bl = []
        # Sizes are a FRACTION OF THE DOMAIN, not a cell count. Absolute sizes look fine at
        # the scene's default resolution and then blow the solve up the first time someone
        # renders it smaller: the bodies stay the same while the tunnel shrinks around them,
        # blockage climbs, and the flow squeezing past reaches the lattice's speed ceiling.
        # (Rendered at --size 1080x1920 the domain went 520x300 -> 384x216 and this diverged
        # at frame 249.)
        for k in range(4):
            pts = (shapes.naca4("0018") if k % 2 else shapes.ellipse(0.5))
            bl.append(RigidBody(pts, size=sim.ny * (0.125 + 0.018 * k),
                                cx=sim.nx * (0.30 + 0.11 * k),
                                cy=sim.ny * (0.34 + 0.12 * (k % 3)),
                                angle=float(rng.uniform(0, 180)),
                                density=1.5, vx=0.0, vy=0.0,
                                omega=float(rng.uniform(-0.004, 0.004))))
        self.bodies = BodySystem(bl, sim.nx, sim.ny, gravity=0.0,
                                 load_filter=0.08, speed_limit=0.085,
                                 wall_margin=5.0, outlet_margin=34.0,
                                 spawn_x=sim.nx * 0.12, u_stream=sim.u0)
        self.mask = self.bodies.mask

    def update(self, sim, t, frame):
        if frame == 0:
            self.mask = self.bodies.mask
            return
        self.mask = self.bodies.update(sim, dt=float(self.steps))
        sim.set_wall_velocity(*self.bodies.wall_velocity())

    def hud(self, sim):
        return [f"{len(self.bodies.bodies)} free bodies   Re {self.re:,.0f}",
                "no path is scripted"]


SCENES = {c.name: c for c in (AoASweep, Stall, TriFoil, Karman, PlungingFoil,
                              MachCone, MachSweep, DyeRake, Tumble)}


# --- CLI ----------------------------------------------------------------------------------
def build_sim(scene):
    scene.ref_len = scene.reference_length()
    if scene.solver == "cns":
        sim = CNS(scene.nx, scene.ny, u0=scene.u0, re=scene.re,
                  ref_len=scene.ref_len, csm=scene.csm,
                  blockage=getattr(scene, "blockage", 4.2))
    elif scene.solver == "lbm":
        sim = LBM(scene.nx, scene.ny, u0=scene.u0, re=scene.re,
                  ref_len=scene.ref_len, csm=scene.csm)
    else:
        raise ValueError(f"scene.solver must be 'lbm' or 'cns', got {scene.solver!r}")
    scene.build(sim)
    if scene.turbulence:
        total = scene.settle + scene.frames * scene.steps
        sim.set_inlet_turbulence(scene.turbulence, length=scene.ref_len * 0.12,
                                 span=int(total * sim.u0) + 512)
    sim.perturb(amp=0.03)
    return sim


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="scene name")
    ap.add_argument("--list", action="store_true", help="list scenes and exit")
    ap.add_argument("--out", default=None, help="output path (.mp4, or .gif)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="solver steps per frame; this is the APPARENT SPEED knob, not u0")
    ap.add_argument("--settle", type=int, default=None)
    ap.add_argument("--re", type=float, default=None)
    ap.add_argument("--u0", type=float, default=None)
    ap.add_argument("--field", default=None,
                    choices=["vorticity", "speed", "pressure", "q", "mach", "schlieren"])
    ap.add_argument("--orient", default=None, choices=list(ORIENTATIONS),
                    help="h = horizontal; vl/vr = vertical (rotate CCW / CW)")
    ap.add_argument("--film", default=None, choices=sorted(FILM_PRESETS),
                    help="optical post-process: off, clean, crt, wrecked")
    ap.add_argument("--vlim", type=float, default=None)
    ap.add_argument("--tracers", type=int, default=None)
    ap.add_argument("--scale", type=int, default=1, help="integer upscale of the output")
    ap.add_argument("--nx", type=int, default=None, help="lattice width in cells")
    ap.add_argument("--ny", type=int, default=None, help="lattice height in cells")
    ap.add_argument("--size", default=None, metavar="WxH",
                    help="target FRAME size, e.g. 1080x1920. Picks the lattice and integer "
                         "upscale that hit it exactly for the chosen --orient, and says so.")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.list or not a.scene:
        print("scenes:")
        for k, c in SCENES.items():
            doc = (c.__doc__ or "").strip().splitlines()[0]
            print(f"  {k:<10} {doc}")
        return 0
    if a.scene not in SCENES:
        print(f"unknown scene {a.scene!r}; try --list", file=sys.stderr)
        return 2

    scene = SCENES[a.scene]()
    for k in ("frames", "fps", "steps", "settle", "field", "tracers"):
        v = getattr(a, k)
        if v is not None:
            setattr(scene, k, v)
    if a.orient is not None:
        scene.orientation = a.orient
    if a.film is not None:
        scene.film = a.film
    if a.nx is not None:
        scene.nx = a.nx
    if a.ny is not None:
        scene.ny = a.ny

    scale = a.scale
    if a.size:
        # Solve for the lattice that lands exactly on the requested frame size. The solvers
        # compute in cells and the aspect ratio is chosen here, at render time - so hitting a
        # delivery format is an arithmetic problem, not a reason to touch the physics.
        try:
            W, H = (int(v) for v in a.size.lower().split("x"))
        except Exception:
            print(f"--size wants WxH, e.g. 1080x1920 (got {a.size!r})", file=sys.stderr)
            return 2
        # after orientation, frame = (lattice_x, lattice_y) or its transpose, times `scale`
        want_x, want_y = (W, H) if scene.orientation == "h" else (H, W)
        if a.scale != 1:
            scale = a.scale
        else:
            # Pick the LARGEST upscale that still leaves a usable lattice - not the largest
            # upscale full stop. Maximising the scale minimises the lattice, and a lattice
            # smaller than the body in it is not a cheap render, it is a divergent one: at
            # 1080x1920 the naive choice was 128x72 with x15, into which a 110-cell chord does
            # not fit at all. MIN_SHORT is the floor on the short axis in cells.
            MIN_SHORT = 200
            scale = 1
            for s in range(1, 33):
                if want_x % s or want_y % s:
                    continue
                if min(want_x // s, want_y // s) >= MIN_SHORT:
                    scale = s
        scene.nx, scene.ny = want_x // scale, want_y // scale
        got_x, got_y = scene.nx * scale, scene.ny * scale
        frame = (got_x, got_y) if scene.orientation == "h" else (got_y, got_x)
        print(f"size:    lattice {scene.nx}x{scene.ny} x{scale} -> frame "
              f"{frame[0]}x{frame[1]}" + ("" if frame == (W, H) else f"  (asked {W}x{H})"))
    if a.re is not None:
        scene.re = a.re
    if a.u0 is not None:
        scene.u0 = a.u0
    if a.vlim is not None:
        scene.vlim = a.vlim

    sim = build_sim(scene)
    out = a.out or f"out/{scene.name}.mp4"
    print(f"backend: {describe_backend()}")
    print(sim.describe())
    print(f"scene:   {scene.name}  {scene.frames} frames @ {scene.fps}fps, "
          f"{scene.steps} steps/frame")

    # The film chain is built against the FINAL frame size, so it has to know the orientation
    # and the upscale factor - a filter sized to the lattice would stretch its own barrel
    # distortion when the frame is rotated or scaled.
    film = None
    if scene.film:
        fh, fw = (scene.ny, scene.nx) if scene.orientation == "h" else (scene.nx, scene.ny)
        film = make_film(fw * scale, fh * scale, preset=scene.film)
        print(f"film:    {film.describe()}")

    render_scene(sim, scene, out,
                 frames=scene.frames, fps=scene.fps, steps=scene.steps,
                 settle=scene.settle, field=scene.field, vlim=scene.vlim,
                 tracers=scene.tracers, scale=scale,
                 hud=not a.no_hud, quiet=a.quiet,
                 orientation=scene.orientation,
                 dye=scene.dye, film=film)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
