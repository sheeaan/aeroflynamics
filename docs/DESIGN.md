# Boeing 787 Airfoil & Wing-Flex Simulator — Design Document

A real-time, interactive 3D simulation of airflow, lift, and structural wing flex
for a Boeing 787-class wing, built in Unity. The goal is a visually compelling and
*physically honest* toy: you can see the air flow around the wing, watch pressure
build below and drop above, and watch the composite wing bend under the load it
actually generates.

---

## 1. Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Platform | **Unity** (URP + VFX Graph; Unity 6 or 2022 LTS) | Best real-time 3D + interactivity; author knows/prefers it |
| Aerodynamics | **Full 3D surface panel method** (source–doublet, Katz & Plotkin) | True surface pressure everywhere in 3D; cross-section views come for free |
| Edge-case realism | **Empirical overlays** | Inviscid solver can't do stall/high-lift; blend with empirical curves so the whole range reads correctly |
| Scope | **Full 3D** | Immersive; wing flex requires 3D anyway |
| Live controls | Angle of attack, airspeed, altitude (→ air density), flaps/slats, wing load (g) | The full set |
| Aeroelastic coupling | **One-way to start** (aero → structure); two-way is a stretch goal | Two-way forces an influence-matrix re-factor every frame |

---

## 2. Aerodynamics — 3D surface panel method

### 2.1 Method
- Discretize the wing surface into quadrilateral panels (target **1,000–3,000** panels).
- Each panel carries a **constant-strength source + doublet** singularity.
- Enforce the **no-penetration** boundary condition at each panel control point → dense
  linear system `A·μ = b`, where `A` = influence coefficients (geometry only),
  `b` = freestream boundary condition (depends on angle of attack & airspeed).
- **Kutta condition** at the trailing edge: shed a **wake** of doublet panels downstream
  so circulation (and therefore lift) is well-defined. A flat/prescribed wake is fine
  for steady flow; the tip vortex roll-up is handled visually (see §5).
- Recover **surface velocity** from the singularity distribution → **pressure coefficient**
  `Cp = 1 − (V/V∞)²` (Bernoulli) on every panel.
- Integrate Cp over the surface → **lift, induced drag, pitching moment**, and the
  **spanwise load distribution** that drives the wing flex.

### 2.2 Real-time strategy (critical)
- `A` depends only on geometry → **LU-factorize once**, cache it.
- Per frame, only `b` changes (AoA/airspeed) → **back-substitution only**, O(N²), trivial.
- **Re-factorize only on geometry change**: flap/slat deployment (or two-way flex, later).
- Implement with Unity **Burst/Jobs**; a hand-rolled LU is fine at this panel count.
  (One-time factor at N=2000 ≈ a few GFLOP; per-frame re-solve ≈ a few MFLOP.)

### 2.3 Velocity field (for streamlines)
- Velocity at any point in space = freestream + summed induced velocity from all panels
  and wake. Sample this on a grid / trace it for streamlines, and on cutting planes for
  cross-section views.

---

## 3. Empirical overlays (viscous effects the solver can't compute)

Potential flow is **inviscid** — no stall, no separation, no true high-lift boost. We blend:

- **Stall / post-stall:** clamp/blend the linear panel-method lift curve into an empirical
  `Cl_max` and a post-stall drop (e.g. a corrected `Cl-α` curve or Viterna-style
  extrapolation). Beyond stall, show separated flow stylistically.
- **Flaps / slats:** model the geometry approximately (deflected trailing-edge / leading-edge
  shape) for the *visual*, and apply an empirical `ΔCl_max` / lift shift per configuration for
  the *numbers*.
- **Transonic (Mach ~0.85 cruise):** out of scope for the solver. Either stay subsonic or add
  a stylized shock indicator on the upper surface; flagged as future work.

---

## 4. Structure & aeroelasticity (wing flex)

- Model the wing as a **spanwise beam** (Euler–Bernoulli) with a stiffness distribution `EI(y)`
  tuned so tip deflection matches reality: **~3 m in normal cruise, ~7.6 m at the 150%
  static-test limit.**
- Spanwise lift → shear → **bending moment** → deflection curve → bend the wing **mesh via a
  bone chain** (skinned mesh) along the span. Add slight twist for realism.
- **One-way** coupling first (fast, stable). **Two-way** (deflected shape re-enters the solve)
  is a stretch goal and requires periodic re-factorization or an incremental update.

---

## 5. Visualization

- **Pressure map:** shade Cp directly onto the wing surface (diverging colormap; blue = suction
  above, red = high pressure below). Accessible/perceptually-uniform palette.
- **Streamlines / particles:** trace the velocity field; particles accelerate over the upper
  surface, slow beneath — lift becomes visible, not just a readout.
- **Wingtip vortices:** the trailing wake + induced swirl; render a stylized rolled-up vortex
  core (VFX Graph) for the signature trailing-tip spiral.
- **Cross-section probe:** a slidable spanwise cutting plane showing the airfoil outline with
  Cp and 2D streamlines — sampled straight from the 3D solution.

---

## 6. Controls & readouts (UI)

**Sliders:** angle of attack · airspeed · altitude (→ ISA air density) · flap/slat setting ·
load factor *g*.

**Live readouts:** `CL`, total lift (N / vs. weight), induced drag, bending moment,
**tip deflection (m)**, Mach number, and a stall warning.

---

## 7. 787 geometry & data assumptions

- Boeing's exact airfoil is proprietary → use a **public representative supercritical section**
  (e.g. NASA **SC(2)-0714** family), thicker at root, thinner toward the tip.
- Planform (787-8/-9 class, approximate):
  - Wingspan **~60 m** (with raked wingtips — not winglets)
  - Wing area **~325 m²**, aspect ratio **~11**
  - Quarter-chord sweep **~32°**
  - Cruise **Mach 0.85**, typical cruise **35,000–43,000 ft**
  - MTOW ~228 t (-8) / ~254 t (-9); use for the "lift vs. weight" readout
- **ISA atmosphere** model for density vs. altitude (sea level ρ = 1.225 kg/m³;
  ~0.38 kg/m³ at 35,000 ft).
- Composite (carbon-fiber) wing → the large, dramatic flex is a headline feature.

---

## 8. Unity architecture (modules)

```
Geometry        → 787 planform + supercritical section; flap/slat reconfiguration; panel mesh
AeroSolver      → build A (Burst), LU factor (cached), per-frame back-sub, Kutta + wake,
                  Cp + forces + spanwise load, velocity-field sampler
Empirical       → stall / post-stall / high-lift correction curves layered on solver output
Structure       → beam model EI(y) → bending → bone-chain mesh deformation (one-way)
Viz             → surface Cp shader, streamline/particle system, tip-vortex VFX, section probe
UI              → sliders + readouts (uGUI or UI Toolkit)
Scene/Camera    → sky, sun, aircraft "cutting across"; orbit/free camera
```

**Data flow (per frame):** UI → freestream `b` → AeroSolver back-sub → Cp / forces /
spanwise load → (Empirical correction) → Structure deflection → mesh bend; velocity field →
Viz streamlines & vortices.

---

## 9. Milestones

- **M1 — Solver core.** 3D panel method on a simple straight wing: build A, LU factor,
  Kutta + wake, correct `CL-α`. *Accept:* lift matches thin-airfoil/known results within a few %.
- **M2 — 787 geometry + pressure map.** Real planform + supercritical section, Cp shaded on
  surface. *Accept:* plausible suction peak on upper surface, moves with AoA.
- **M3 — Flow visualization.** Streamlines/particles + wingtip vortices from the velocity field.
- **M4 — Wing flex.** Spanwise load → beam → mesh bend; tip deflection tracks load (cruise→limit).
- **M5 — Flaps/slats + stall overlay + UI polish.** Empirical corrections, full control set,
  readouts, scene.

**Stretch:** two-way aeroelastic coupling · transonic shock indicator · multiple 787 variants ·
gust/turbulence loads · save/share configurations.

---

## 10. References
- Katz & Plotkin, *Low-Speed Aerodynamics* (2nd ed.) — the definitive panel-method reference.
- Anderson, *Fundamentals of Aerodynamics* — airfoils, lift, stall.
- NASA supercritical airfoil (SC(2)) coordinate data (public).
- Viterna–Corrigan post-stall extrapolation.
