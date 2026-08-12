# windtunnel

A 2-D wind tunnel: lattice-Boltzmann and compressible Navier-Stokes, polygon bodies, video out.

Nothing here is animated by hand. A scene says where a body is at time `t` — or, for the free
bodies, only where it *started* — and then lets go. Every vortex, every separation, every shock
on screen is the solver's output.

```
python scenes.py --list
python scenes.py karman --tracers 8000
python tools/validate.py
```

Runs on NumPy + SciPy + Pillow. `numba` is strongly recommended (40× faster, see below);
`ffmpeg` is optional (without it you get an animated GIF instead of mp4).

---

## Two solvers

|  | `lbm.py` | `cns.py` |
|---|---|---|
| method | D2Q9 lattice-Boltzmann, BGK + Smagorinsky | finite-volume Navier–Stokes |
| scheme | half-way bounce-back on a boolean mask | MUSCL + minmod, HLLC, SSP-RK2 |
| gas | isothermal, effectively γ=1 | real ideal gas, γ=1.4, energy carried |
| ceiling | fails above M ≈ 0.3 | shocks, supersonic, stagnation heating |
| body | bounce-back | EDT ghost-cell immersed boundary |
| forces | validated momentum exchange | **not implemented** — see below |
| speed | numba-fused, ~2 ms/step at 640×256 | numpy only, ~40× slower |

Use the LBM for everything it can reach. The compressible solver exists for the phenomena the
lattice structurally *cannot* produce at any parameter setting — it is isothermal with no
shock-capturing mechanism, so a supersonic pocket is not a matter of turning `u0` up.

### The `u0` trap

`u0` means a **different physical quantity** in each solver:

- **LBM** — a lattice velocity. Sound speed is `1/√3`, so `u0 = 0.08` is M 0.139. Keep it under ~0.1.
- **CNS** — non-dimensionalised so `c∞ = 1`, therefore `u0` **is** the Mach number, exactly.

Copying a scene between solvers without converting gives you a run that is stable, plausible,
and at completely the wrong speed.

### To make a clip look faster, raise `--steps`, never `u0`

Apparent speed is (velocity × steps per frame). Raising `steps` samples the same physics less
often; raising `u0` changes the physics and walks the solve toward its own ceiling.

Frames are driven from real seconds (`t = i/fps`), never the step counter, so a half-resolution
preview at 24 fps and a final at 60 fps are the *same animation* sampled differently.

---

## Layout

```
windtunnel/
├── scenes.py             the compositions + CLI
├── wt/
│   ├── lbm.py            D2Q9 + Smagorinsky, half-way bounce-back, forces
│   ├── lbm_numba.py      the fused JIT kernel (40x); lbm.py falls back to numpy
│   ├── cns.py            compressible: MUSCL/HLLC/SSP-RK2 + viscous + ghost-cell IBM
│   ├── bodies.py         free rigid bodies driven by the validated force integral
│   ├── dye.py            D2Q5 passive-scalar transport (streaklines)
│   ├── film.py           phosphor -> dust -> lens -> camera, in that order
│   ├── shapes.py         naca4, cylinder, ogive, wedge, place, rasterize
│   ├── streaks.py        passive tracer particles
│   ├── colormap.py       perceptual LUTs
│   ├── render.py         orientation, HUD, ffmpeg/GIF writer, frame loop
│   └── gpu.py            numpy/cupy shim
└── tools/
    ├── validate.py       cylinder Cd + Strouhal, foil symmetry/slope, handedness
    ├── sod_check.py      compressible solver vs the exact Sod shock tube
    ├── shock_check.py    immersed boundary vs the theta-beta-M relation
    └── bench.py          backend shoot-out, with a correctness gate
```

## Scenes

| scene | solver | what it shows |
|---|---|---|
| `aoa_sweep` | lbm | NACA 2412 swept 0° → 18° → 0°, through stall and back |
| `stall` | lbm | a section parked at 20°, so the shedding cycle is the subject |
| `tri_foil` | lbm | three sections at 4/11/18° in the same stream, same instant |
| `karman` | lbm | cylinder at Re 150 — the validation case, and the prettiest |
| `plunge` | lbm | a section oscillating in heave (moving-wall boundary) |
| `dye_rake` | lbm | two-colour dye combs drawn through a foil |
| `tumble` | lbm | four free bodies with no scripted path at all |
| `mach_cone` | cns | pointed ogive at M 2 — bow shock, Mach cone, schlieren |
| `mach_sweep` | cns | a section throttled 0.55 → 0.80, transonic shock forming |

Fields: `vorticity`, `speed`, `pressure`, `q` (Q-criterion), plus `mach` and `schlieren` on the
compressible solver.

```
python scenes.py mach_cone --field schlieren
python scenes.py dye_rake --orient vl --scale 3       # vertical 1080x1920
python scenes.py karman --film crt --tracers 8000
```

---

## Performance

`tools/bench.py`, 640×256, this machine (Core Ultra 5 226V, 8 threads; Arc 130V iGPU):

| backend | ms/step | speedup | 300-frame clip |
|---|---|---|---|
| **numba** | **1.97** | **40.3×** | 0.2 min |
| torch-xpu (Arc 130V) | 5.41 | 14.7× | 0.5 min |
| torch-cpu | 14.02 | 5.7× | 1.4 min |
| numpy | 79.38 | — | 7.9 min |

The integrated GPU loses because this kernel is **bandwidth-bound**: the Arc shares system
memory, so it hits the same wall, and it still materialises the ~30 temporaries per step that
the fused numba kernel simply never writes. A discrete card with its own memory would reorder
this. CuPy is CUDA-only and does not apply here at all.

`bench.py` refuses to crown a backend that does not reproduce the numpy result to float32
roundoff — it separates the correctness pass from the timing pass, because an earlier version
gave torch five extra warmup steps and reported the resulting state difference as a numerical
disagreement.

**Known gap:** `cns.py` has no numba path, so compressible scenes run at numpy speed.

---

## Validation

> A CFD picture is persuasive whether or not it is right.

```
python tools/validate.py            # 12 cases
python tools/sod_check.py
python tools/shock_check.py
```

| case | checks |
|---|---|
| `cylinder` | mean Cd and Strouhal at Re 100–200 vs published correlations |
| `symmetry` | a symmetric section at 0° must make exactly zero lift |
| `slope` | Cl odd in incidence; dCl/dα reported as a diagnostic |
| `handedness` | every orientation mapping preserves vortex sense |
| `sod_check` | exact Sod shock tube — wave speeds, plateaus, no overshoot |
| `shock_check` | immersed boundary vs the exact oblique-shock relation |

Latest full run — **12/12**, plus Sod 3/3:

| | measured | reference |
|---|---|---|
| cylinder Strouhal | 0.1907 | 0.1854 |
| cylinder mean Cd | 1.455 | 1.33 (7.1% blockage inflates it) |
| NACA 0012 @ 0°, Cl | −0.0000 | 0 |
| Cl antisymmetry | 0.0000 | 0 |
| Sod shock position | within **0.33 cells** | exact |
| Sod contact position | within 0.70 cells | exact |

### Why the validation harness exists

The first `force_field()` produced a drag coefficient that was smooth, stable, converged, and
**3× too large** — and every rendered frame looked completely fine.

| method | Cd |
|---|---|
| control-volume momentum balance | 1.13 |
| momentum exchange (as first written) | **3.94** |
| momentum exchange (fixed) | 1.37 |
| published, Re 100 unconfined | 1.33 |

The cause: full-way bounce-back (reversing populations *inside* solid cells) paired with the
momentum-exchange integral that belongs to **half-way** bounce-back, so incident and reflected
populations were sampled a step apart and a cell apart. Decomposing the sum localised it
exactly — the density term gave 1.10 (right), the non-equilibrium term contributed a spurious
2.98. The fix was to make the boundary condition and its force integral the *same scheme*, not
to scale the answer.

**So: do not change `LBM.force_field` or the bounce-back in `LBM.step` without re-running
`tools/validate.py`.** `bodies.py` integrates that force directly — a wrong drag there looks
like heavier objects in thicker fluid, and no frame would betray it.

Three principles the harness follows, each learned the same way:

1. **Measure a second, independent way.** The control-volume balance is what caught the force bug.
2. **A validation case must not contain a fitted parameter.** When cylinder Cd read 23% high at
   13.9% blockage, the fix was widening the domain to ~7% so the test matched the unconfined
   reference — not applying an empirical blockage correction to the number.
3. **Don't assert what the setup cannot resolve.** `dCl/dα` is reported against a wide band, not
   against the inviscid 2π: at chord 56 the NACA 0012 leading-edge radius is 0.9 cells, so the
   suction peak is unresolved. (Raising chord to 110 moved the measured slope 1.06 → 2.07,
   confirming the diagnosis.)

The handedness case has a **negative control**: swap `orient` for a `flipud` and it must fail.
A test that cannot fail is decoration.

---

## Install

```
pip install -r requirements.txt
pip install numba          # strongly recommended: 40x
```

`ffmpeg` is a binary, not a package — without it you get an animated GIF. On Windows:
`winget install Gyan.FFmpeg`.

## Relationship to aeroflynamics

The complement of the Unity panel-method solver in `../`, not a competitor:

| | this | aeroflynamics |
|---|---|---|
| method | LBM / compressible FV | 3-D surface panel method |
| viscosity | real (+ LES) | inviscid |
| time | unsteady | steady per frame |
| dimensions | 2-D | 3-D |
| gets you | stall, separation, shedding, shocks | full 3-D Cp, spanwise load, real-time |

Neither can do the other's job — which is exactly why `docs/DESIGN.md` §3 has an empirical
overlay section. Use this to *see* what those stall curves are standing in for.

---

## Credits

This tunnel builds on the wind-tunnel engine in
[spectrometry_public](https://github.com/ec175/spectrometry_public) by Ethan Earl, released
under the MIT licence. Its full text is in [`LICENSES/`](LICENSES/spectrometry_public-MIT.txt).

| file | what came from spectrometry_public |
|---|---|
| `wt/cns.py` | the compressible solver, adapted |
| `wt/lbm.py` | parts of the lattice-Boltzmann solver |
