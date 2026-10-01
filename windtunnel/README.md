# windtunnel

A 2-D wind tunnel: lattice-Boltzmann and compressible Navier-Stokes, polygon bodies, video out.

## Why this exists

I love racing. My IB Math IA was about the downforce an F1 rear wing makes, and this project
extends it. Here a real fluid solver computes the flow, so you can watch the air that makes the
downforce. You can also open DRS, tilt a Porsche GT3 RS flap or thicken a wing, and watch the
numbers respond.

The look is inspired by the physics animations of
[spectrometry.mp4](https://github.com/ec175/spectrometry_public): a jet-coloured speed field,
white streaks that trace the air, and nothing animated by hand. Parts of the solver are adapted
from that project's open-source code; see [Credits](#credits).

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
├── interactive.py        live windows: AoA, rear wings (--wings), thickness (--thickness)
├── wt/
│   ├── wings.py          F1 2025 / F1 2026 / GT3 RS sections, built to the published rules
│   ├── look.py           the spectrometry.mp4 look: jet field, comet streaks, AA bodies
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

## Interactive: rear wings and wing thickness

```
python interactive.py --wings          # F1 2025 / F1 2026 / Porsche 992 GT3 RS
python interactive.py --thickness      # one simple wing, thickness slider
python interactive.py --wings --selftest
python interactive.py --thickness --selftest
```

**`--wings`** puts three real rear wings in the tunnel, one at a time. `Tab` switches wing.
`D` opens DRS on the 2025 F1 wing, switches the 2026 F1 wing to its low-drag X-mode, or
flattens the GT3 RS flap. Up/Down tilts the GT3 RS flap through its 34° range. `[` and `]`
set the road speed for the Newton readout.

| wing | where the geometry comes from |
|---|---|
| F1 2025 | FIA 2025 Technical Regulations Art. 3.10: two sections, 10–15 mm slot, DRS opens it to 85 mm |
| F1 2026 | FIA 2026 Technical Regulations Issue 8 Art. 3.11: three sections, rear two rotate, X-mode slot ≤ 65 mm, trailing-edge angle caps 10° / 40° / 65° |
| GT3 RS | Porsche press kit (two elements, 34° of flap travel); chords and span are a forum estimate from photos |

Teams and Porsche keep their real profiles secret, so every element is a public NACA 6412 shape.
`--wings --selftest` checks that the F1 sections meet every limit they were built to.

**`--thickness`** holds one inverted NACA 44xx wing at a fixed 300 mm chord and 6° angle. A
slider changes **only** its thickness, from 6% to 24% of the chord.

### Results (averaged over 80–120 frames, 250 km/h)

| wing | setting | C<sub>down</sub> | C<sub>D</sub> | downforce | drag |
|---|---|---|---|---|---|
| F1 2025 | DRS closed → open | 4.14 → 2.37 | 0.89 → 0.31 | 4,661 → 2,668 N | 1,002 → 349 N |
| F1 2026 | Z-mode → X-mode | 2.33 → 0.46 | 0.46 → 0.13 | 2,580 → 509 N | 509 → 144 N |
| GT3 RS | flap 40° → flat | 2.84 → 1.94 | 0.46 → 0.13 | 6,132 → 4,189 N | 993 → 281 N |

Thickness sweep, one run stepping upward from 6%:

| thickness | C<sub>down</sub> | C<sub>D</sub> | C<sub>down</sub> / C<sub>D</sub> |
|---|---|---|---|
| 6% | 0.58 | 0.088 | 6.6 |
| 9% | 0.69 | 0.111 | 6.2 |
| 12% | 0.65 | 0.127 | 5.1 |
| 15% | 0.52 | 0.143 | 3.7 |
| 18% | 0.46 | 0.160 | 2.9 |
| 21% | 0.32 | 0.184 | 1.8 |
| 24% | 0.39 | 0.214 | 1.8 |

**What the thickness sweep shows**
- **Drag rises steadily with thickness.** A thicker section blocks more air and leaves a wider
  wake. Two separate runs agree on this.
- **Downforce peaks near 9%, then falls.** That is real low-Reynolds-number behaviour: at
  Re ≈ 4,500 the boundary layer is thick and separates early off a fat section. At a real car's
  Re of about 1.4 million, thicker sections keep their flow attached far better. Treat this trend
  as specific to this simulation.
- **Thick sections are unsteady.** At 24% the downforce read 0.39 in this sweep but 0.87 in the
  self-test, which jumped straight from 6%. The separated flow depends on its history, so only
  the drag trend is solid there.

---

## The math

### 1. The fluid: lattice-Boltzmann

Instead of solving the Navier–Stokes equations directly, the lattice-Boltzmann method tracks how
many particles move in each of 9 directions $\mathbf{c}_i$ at every grid cell (the D2Q9 model).
Each step has two parts: particles **stream** to the neighbouring cell, then **collide**,
relaxing toward a local equilibrium:

$$f_i(\mathbf{x}+\mathbf{c}_i,\ t+1) = f_i(\mathbf{x},t) - \frac{1}{\tau}\Big(f_i(\mathbf{x},t) - f_i^{\text{eq}}(\mathbf{x},t)\Big)$$

$$f_i^{\text{eq}} = w_i\,\rho\left(1 + 3\,\mathbf{c}_i\!\cdot\!\mathbf{u} + \tfrac{9}{2}(\mathbf{c}_i\!\cdot\!\mathbf{u})^2 - \tfrac{3}{2}\,|\mathbf{u}|^2\right)$$

Density and velocity are sums over the 9 directions:

$$\rho = \sum_i f_i, \qquad \rho\,\mathbf{u} = \sum_i \mathbf{c}_i f_i$$

The relaxation time $\tau$ sets the viscosity, $\nu = (\tau - \tfrac12)/3$, so choosing a
Reynolds number fixes $\tau$:

$$\mathrm{Re} = \frac{U L}{\nu} \quad\Rightarrow\quad \tau = \frac{3\,U L}{\mathrm{Re}} + \frac12$$

For the wing tunnel, $U = 0.075$ (grid cells per step), $L = 160$ cells (400 mm) and
$\mathrm{Re} = 6000$. That gives $\nu = 0.002$ and $\tau = 0.506$. With $\tau$ this close to
$\tfrac12$, plain LBM becomes unstable, so a Smagorinsky turbulence model adds viscosity where
the flow shears hardest: $\nu_t = (C_s\Delta)^2\,|S|$ with $C_s = 0.16$.

LBM is only accurate at low Mach number. Here $\mathrm{Ma} = U/c_s = 0.075\sqrt3 = 0.13$,
so the compressibility error, which scales like $\mathrm{Ma}^2$, is about 1.7%.

### 2. From grid units to the real world

One cell is $\Delta x = 2.5$ mm. Lift and drag coefficients have no units, so the grid's speed
never has to equal the car's speed: the real speed only enters at the end, through the dynamic
pressure. For the record, one solver step at 250 km/h stands for

$$\Delta t = \frac{\Delta x \cdot U_{\text{grid}}}{V_{\text{real}}} = \frac{0.0025 \times 0.075}{69.4} \approx 2.7\ \mu\text{s}$$

### 3. Measuring the force on the wing

Walls use **bounce-back**: a particle that would enter the wing is sent back the way it came.
Each bounce reverses its momentum, and the wing absorbs the difference. Summing over every link
$(\mathbf{x}, i)$ that crosses the wing surface gives the total force (momentum exchange):

$$\mathbf{F} = \sum_{\text{boundary links}} \mathbf{c}_i\,\big(f_i^{*}(\mathbf{x}) + f_{\bar i}(\mathbf{x})\big)$$

Here $f_i^{*}$ is the population heading into the wall after collision, and $f_{\bar i}$ is the
one bounced back. The code checks this against a second, independent method (see
[Validation](#validation)). The first version gave a drag 3× too high that looked perfectly
believable.

### 4. Coefficients, then Newtons

$$C_D = \frac{F_x}{\tfrac12 \rho U^2 c}, \qquad C_{\text{down}} = -\,\frac{F_y}{\tfrac12 \rho U^2 c}$$

$c$ is the wing's streamwise length in grid cells. The minus sign is there because a rear wing
pushes **down**. Converting to Newtons uses the real air density, road speed $V$, chord and
span $b$:

$$D = C_{\text{down}} \cdot \tfrac12\,\rho_{\text{air}}\,V^2 \cdot c\,b$$

Worked example: the F1 2025 wing, DRS closed, 250 km/h:

$$V = \frac{250}{3.6} = 69.4\ \text{m/s}, \qquad q = \tfrac12 (1.225)(69.4)^2 = 2954\ \text{Pa}$$

$$D = 4.14 \times 2954 \times (0.397 \times 0.960) = 4661\ \text{N} \approx 475\ \text{kg}$$

Because $D \propto V^2$, doubling the speed quadruples the downforce.

### 5. The wing shapes

Every section is a NACA 4-digit airfoil. With $x$ running from 0 to 1 along the chord, the
thickness either side of the camber line is

$$y_t = 5t\left(0.2969\sqrt{x} - 0.1260\,x - 0.3516\,x^2 + 0.2843\,x^3 - 0.1036\,x^4\right)$$

The camber line $y_c$ is two parabolas that meet at the point of maximum camber $p$:

$$y_c = \begin{cases} \dfrac{m}{p^2}\left(2px - x^2\right) & x < p \\[2mm] \dfrac{m}{(1-p)^2}\left((1-2p) + 2px - x^2\right) & x \ge p \end{cases}$$

The thickness is applied perpendicular to the camber line, with $\theta = \arctan(dy_c/dx)$:

$$x_u = x - y_t\sin\theta,\quad y_u = y_c + y_t\cos\theta, \qquad x_l = x + y_t\sin\theta,\quad y_l = y_c - y_t\cos\theta$$

Each section is then flipped ($z \to -z$, so it pushes down instead of up), rotated by its angle
$\alpha$, and scaled to its chord:

$$\begin{pmatrix} x' \\ z' \end{pmatrix} = c \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix} \begin{pmatrix} x \\ -y \end{pmatrix}$$

**Why the thickness slider never changes the size.** The thickness $t$ only appears as the
factor in front of $y_t$. Chord, camber and angle don't depend on it. The cross-section area
does grow in proportion to $t$:

$$A = c^2 \int_0^1 2\,y_t\,dx = 10\,t\,c^2\left(\tfrac{2}{3}(0.2969) - \tfrac{0.1260}{2} - \tfrac{0.3516}{3} + \tfrac{0.2843}{4} - \tfrac{0.1036}{5}\right) \approx 0.681\,t\,c^2$$

For the 300 mm wing, that is 37 cm² at 6% and 147 cm² at 24%.

**Fitting the rules.** The FIA rules fix gaps and angles, not positions, so the code solves for
the positions. The slot gap $g$ is the shortest distance between two outlines, using
point-to-segment distances. Then **bisection** finds the value that hits each target:

- the height $h$ that puts the flap exactly $g(h) = 12$ mm above the main plane;
- the rotation $\varphi$ that opens DRS to $g(\varphi) = 84$ mm (the rule allows 85);
- for 2026, the angle at which the steepest underside tangent in the last 40 mm sits 1° under
  each cap.

Bisection halves the search interval each step. 60 steps narrow an initial 160 mm range to
$160 / 2^{60} \approx 10^{-16}$ mm.

### 6. The streaks

Each of ~12,000 tracer particles moves with the local flow, stepped with the second-order
midpoint method (RK2):

$$\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta t\;\mathbf{u}\!\left(\mathbf{x}_n + \tfrac{\Delta t}{2}\,\mathbf{u}(\mathbf{x}_n)\right)$$

Each white tail is traced **backward** from the particle through the current velocity field,
22 samples long, and fades along its length. The colour shows how fast the air moves; the
tails show which way.

### 7. What the numbers do and don't mean

| | this simulation | real car |
|---|---|---|
| dimensions | 2-D slice | 3-D wing with endplates |
| Reynolds number | ~4,500–6,400 | ~1.4–2 million |
| tunnel walls | up to ~25% blockage | open road |

At this low Reynolds number the boundary layer is far thicker than on a real wing, which costs
downforce. Being 2-D ignores tip losses, and the walls squeeze the flow; both of those add
downforce. The two errors pull in opposite directions, so the size of the net error is unknown.
For comparison, the 2-D GT3 RS wing alone reads ~625 kg at 250 km/h. Porsche quotes 860 kg for
the **whole car** at 285 km/h, which scales by $V^2$ to ~662 kg at 250 km/h. **Trends** are the
trustworthy output: which way DRS, flap angle or thickness moves downforce and drag.

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

Latest full run — **14/14**, plus Sod 3/3:

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
| `wt/look.py` | the jet / vorticity palettes and the streak constants; the code is a rewrite |

The rest is this project's own: the force validation and the bug it caught, the scenes, the
interactive modes, the FIA-rule wing geometry, the thickness study, and the math write-up.
