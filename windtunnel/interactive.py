"""interactive.py - the aoa_sweep tunnel, with the angle of attack under your hand.

Same solver, same NACA 2412, same Reynolds number as `scenes.py aoa_sweep`. The only change is
who sets the incidence: the arrow keys and the mouse wheel instead of a triangular sweep.
Everything downstream of the mask still belongs to the solver, and the Cl / Cd on the HUD are
the validated momentum-exchange integral, read live every frame.

    python interactive.py
    python interactive.py --scale 3
    python interactive.py --selftest

Controls:  Up / Down, or the mouse wheel    angle of attack, 1 deg per press
           1  2  3  4                       vorticity / speed / pressure / Q-criterion

WINGS MODE (`--wings`): an F1 2025 DRS wing, an F1 2026 active-aero wing and a Porsche 992
GT3 RS wing, one at a time in free stream, drawn in the spectrometry.mp4 look (wt/look.py).
Geometry and its sources are in wt/wings.py.

    python interactive.py --wings
    python interactive.py --wings --selftest

Controls:  Tab                              next wing
           D or Space                       DRS / X-mode / GT3 RS flat, toggle
           Up / Down, or the mouse wheel    GT3 RS upper element, 2 deg per press
           [  ]                             road speed for the Newton readout
           1  2  3  4                       speed / vorticity / pressure / Q-criterion
"""
from __future__ import annotations

import argparse
import sys
import tkinter

import numpy as np
from PIL import Image, ImageTk

from wt import LBM, describe_backend
from wt import colormap as cm
from wt import shapes
from wt import wings
from wt.look import Look, draw_panel, draw_speed_legend
from wt.render import draw_hud, upscale
from scenes import AoASweep

# The incidence you ASK for jumps a whole degree per key press. The body turns toward it at a
# finite rate instead, because a 1-degree jump moves the trailing edge of a 110-cell chord by
# about two cells in one step, and a step change in the wall radiates a pressure pulse.
# aoa_sweep turns at ~0.12 deg/frame; this is a little faster so the controls feel responsive.
MAX_TURN_DEG_PER_FRAME = 0.2
AOA_STEP_DEG = 1.0
MIN_AOA_DEG = -8.0
# Above this the projected blockage squeezes the flow past the body toward the lattice
# speed ceiling. stall parks at 20 deg and is stable; 22 leaves a little margin past it.
MAX_AOA_DEG = 22.0

# The inlet turbulence patch recycles after span / u0 steps. aoa_sweep sizes it for one clip;
# an interactive run has no end, so this is long enough for about ten minutes at 30 fps
# (24576 / 0.075 = 327,680 steps) before the same gust arrives twice.
TURBULENCE_SPAN = 24576

# The same FIXED colour limits as render_scene, for the same reason: auto-fitting per frame
# would make a wake that is genuinely weakening look exactly as bright as one that is not.
# Copied rather than imported because render_scene keeps them local to the function.
# limit = factor * u0 ** u0_power
FIELDS = {
    "vorticity": {"method": "vorticity", "cmap": "coolwarm", "symmetric": True,
                  "factor": 0.80, "u0_power": 1},
    "speed": {"method": "speed", "cmap": "inferno", "symmetric": False,
              "factor": 2.20, "u0_power": 1},
    "pressure": {"method": "pressure", "cmap": "coolwarm", "symmetric": True,
                 "factor": 0.70, "u0_power": 2},
    "q": {"method": "q_criterion", "cmap": "magma", "symmetric": False,
          "factor": 0.12, "u0_power": 2},
}
FIELD_KEYS = {"1": "vorticity", "2": "speed", "3": "pressure", "4": "q"}

CONTROLS = ["up/down or wheel: AoA", "1 vort  2 speed  3 pressure  4 Q"]


class Tunnel:
    """The solver and the picture, with no window. The window only feeds it key presses."""

    def __init__(self, scale=2):
        self.scene = AoASweep()
        self.scene.ref_len = self.scene.reference_length()
        self.sim = LBM(self.scene.nx, self.scene.ny, u0=self.scene.u0, re=self.scene.re,
                       ref_len=self.scene.ref_len, csm=self.scene.csm)
        self.scene.build(self.sim)
        self.sim.set_inlet_turbulence(self.scene.turbulence,
                                      length=self.scene.ref_len * 0.12,
                                      span=TURBULENCE_SPAN)
        self.sim.perturb(amp=0.03)

        self.scale = int(scale)
        self.field = "vorticity"
        self.target_aoa = 0.0
        self.drag_coefficient = float("nan")
        self.lift_coefficient = float("nan")
        self.diverged_message = None

        # settle at the starting incidence so the window does not open on a start-up transient
        self.place_body()
        self.sim.run(int(self.scene.settle))

    def place_body(self):
        polygon = shapes.place(self.scene.pts, self.scene.chord,
                               self.scene.cx, self.scene.cy, self.scene.aoa)
        mask = shapes.rasterize([polygon], self.sim.nx, self.sim.ny)
        self.sim.set_solid(mask)

    def nudge_target(self, delta_deg):
        new_target = self.target_aoa + delta_deg
        self.target_aoa = min(MAX_AOA_DEG, max(MIN_AOA_DEG, new_target))

    def advance(self):
        """Turn the body one frame's worth toward the target, then run the solver."""
        if self.diverged_message is not None:
            return

        remaining = self.target_aoa - self.scene.aoa
        if abs(remaining) > 1e-9:
            turn = min(MAX_TURN_DEG_PER_FRAME, max(-MAX_TURN_DEG_PER_FRAME, remaining))
            self.scene.aoa += turn
            self.place_body()

        self.sim.run(int(self.scene.steps))

        health = self.sim.health()
        if not np.isfinite(health) or health > self.sim.health_limit:
            self.diverged_message = (f"DIVERGED: max|u| = {health:.3f} > "
                                     f"{self.sim.health_limit}. Restart and stay lower.")
            print(self.diverged_message, file=sys.stderr)
            return

        drag, lift = self.sim.coefficients(ref_len=self.scene.ref_len)
        self.drag_coefficient = drag
        self.lift_coefficient = lift

    def frame(self):
        """The current state as an RGB frame, HUD included."""
        config = FIELDS[self.field]
        limit = config["factor"] * self.sim.u0 ** config["u0_power"]
        values = getattr(self.sim, config["method"])()
        if config["symmetric"]:
            minimum = None
        else:
            minimum = 0.0
        rgb = cm.colorize(values, cmap=config["cmap"], symmetric=config["symmetric"],
                          vmin=minimum, vmax=limit)
        rgb = cm.overlay_mask(rgb, self.sim.solid)
        rgb = cm.outline_mask(rgb, self.sim.solid)
        rgb = np.ascontiguousarray(upscale(rgb, self.scale))

        lines = list(self.scene.hud(self.sim))
        if abs(self.target_aoa - self.scene.aoa) > 1e-9:
            lines.append(f"target {self.target_aoa:5.2f} deg")
        if np.isfinite(self.lift_coefficient):
            lines.append(f"Cl {self.lift_coefficient:+.3f}   Cd {self.drag_coefficient:+.3f}")
        lines.append(f"field: {self.field}")
        if self.diverged_message is not None:
            lines.append(self.diverged_message)
        rgb = draw_hud(rgb, lines, scale=self.scale)
        rgb = draw_hud(rgb, CONTROLS, scale=self.scale, corner="bl")
        return rgb


class Window:
    """A tkinter label that shows one frame per tick and forwards keys to the tunnel."""

    def __init__(self, tunnel, title):
        self.tunnel = tunnel
        self.root = tkinter.Tk()
        self.root.title(title)
        self.label = tkinter.Label(self.root, borderwidth=0)
        self.label.pack()
        self.photo = None
        self.root.bind("<Key>", self.on_key)
        self.root.bind("<MouseWheel>", self.on_wheel)

    def on_key(self, event):
        if event.keysym in ("Up", "Right"):
            self.tunnel.nudge_target(+AOA_STEP_DEG)
        elif event.keysym in ("Down", "Left"):
            self.tunnel.nudge_target(-AOA_STEP_DEG)
        elif event.char in FIELD_KEYS:
            self.tunnel.field = FIELD_KEYS[event.char]

    def on_wheel(self, event):
        if event.delta > 0:
            self.tunnel.nudge_target(+AOA_STEP_DEG)
        else:
            self.tunnel.nudge_target(-AOA_STEP_DEG)

    def tick(self):
        self.tunnel.advance()
        rgb = self.tunnel.frame()
        # keep a reference: tkinter does not, and a collected PhotoImage shows as blank
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=self.photo)
        if self.tunnel.diverged_message is None:
            self.root.after(1, self.tick)

    def run(self):
        self.tick()
        self.root.mainloop()


# --- wings mode -----------------------------------------------------------------------------
# 2.5 mm per cell puts a 12 mm slot across ~5 cells and the biggest wing (GT3 RS, ~465 mm long
# at full tilt) across ~186. Finer would resolve the slot better and cost frame rate as the
# square; this is the compromise that stays interactive.
MM_PER_CELL = 2.5
WING_NX = 640
WING_NY = 320
WING_CENTRE_X_FRACTION = 0.32
WING_U0 = 0.075
# Re is quoted on a fixed 400 mm, so the VISCOSITY is the same for every wing and switching
# wings changes only the geometry. The HUD shows each wing's Re on its own chord.
WING_RE = 6000.0
WING_RE_LENGTH_MM = 400.0
WING_TURBULENCE = 0.008
WING_STEPS_PER_FRAME = 10
WING_SETTLE_STEPS = 600
# How fast a moving element turns toward its commanded position. Not the real actuator speed:
# at 250 km/h one lattice step is ~3 microseconds, so the real 400 ms DRS stroke would take
# 100,000+ steps and look frozen. This only keeps each mask change small.
FLAP_TURN_DEG_PER_FRAME = 0.6
GT3RS_STEP_DEG = 2.0
# The HUD shows a running average, not the instantaneous force. Moving a flap sheds a vortex,
# and while it leaves the instantaneous coefficient can read backwards: the flat GT3 RS read
# C 3.36 one frame after a toggle, against 1.77 once averaged. Each frame moves the average
# this fraction of the way toward the new value - about 30 frames (~3 s here) to settle.
FORCE_AVERAGE_FRACTION = 1.0 / 30.0
# After any geometry change the readout is flagged as settling for this many frames. The wake
# and starting vortex of the OLD shape take thousands of steps to clear a 640-cell tunnel: the
# GT3 RS at full tilt averaged C_down 1.9 measured 60 frames after a wing switch, 3.5 from rest.
READOUT_SETTLE_FRAMES = 150

AIR_DENSITY = 1.225            # kg/m^3, ISA sea level
GRAVITY = 9.81
ROAD_SPEED_DEFAULT_KMH = 250.0
ROAD_SPEED_STEP_KMH = 10.0
ROAD_SPEED_MIN_KMH = 50.0
ROAD_SPEED_MAX_KMH = 350.0

WING_FIELD_KEYS = {"1": "speed", "2": "vorticity", "3": "pressure", "4": "q"}
WING_CONTROLS = ["Tab next wing   D / Space DRS", "Up/Down GT3 RS flap   [ ] speed",
                 "1 speed  2 vort  3 pressure  4 Q"]


class WingTunnel:
    """One lattice, three wings. Switching swaps the mask; the air keeps flowing."""

    def __init__(self, scale=2):
        self.wings = wings.build_all()
        self.wing_index = 0
        self.sim = LBM(WING_NX, WING_NY, u0=WING_U0, re=WING_RE,
                       ref_len=WING_RE_LENGTH_MM / MM_PER_CELL, csm=0.16)
        # The default 4096-column patch, not TURBULENCE_SPAN: the inlet reads one column of the
        # patch per step, and at 320 rows x 24576 columns those reads are scattered over 31 MB,
        # which measured +0.8 to +2.3 ms on a ~3 ms step. 4096 recycles (seamlessly - the patch
        # is periodic) after ~55,000 steps, about five minutes here.
        self.sim.set_inlet_turbulence(WING_TURBULENCE,
                                      length=WING_RE_LENGTH_MM / MM_PER_CELL * 0.12)
        self.sim.perturb(amp=0.03)
        self.look = Look(WING_NX, WING_NY, scale, WING_STEPS_PER_FRAME)

        self.rotation = 0.0
        self.target_rotation = 0.0
        self.road_speed_kmh = ROAD_SPEED_DEFAULT_KMH
        self.field = "speed"
        self.drag_coefficient = float("nan")
        self.lift_coefficient = float("nan")
        self.average_downforce = float("nan")
        self.average_drag = float("nan")
        self.diverged_message = None
        self.polygons = None
        self.slot_gap = 0.0
        self.frames_since_change = 0

        self.place_wing()
        self.sim.run(WING_SETTLE_STEPS)

    @property
    def wing(self):
        return self.wings[self.wing_index]

    def place_wing(self):
        self.polygons = self.wing.lattice_polygons(self.rotation, MM_PER_CELL,
                                                   WING_NX * WING_CENTRE_X_FRACTION,
                                                   WING_NY * 0.5)
        self.sim.set_solid(shapes.rasterize(self.polygons, WING_NX, WING_NY))
        self.slot_gap = self.wing.slot_gap_mm(self.rotation)
        self.frames_since_change = 0

    def next_wing(self):
        self.wing_index = (self.wing_index + 1) % len(self.wings)
        self.rotation = 0.0
        self.target_rotation = 0.0
        self.average_downforce = float("nan")
        self.average_drag = float("nan")
        self.place_wing()

    def toggle(self):
        if abs(self.target_rotation) < 1e-6:
            self.target_rotation = self.wing.toggle_rotation_deg
        else:
            self.target_rotation = 0.0

    def nudge_flap(self, delta_deg):
        """GT3 RS only. The F1 rules allow exactly two positions, so F1 wings ignore this."""
        if not self.wing.continuous:
            return
        new_target = self.target_rotation + delta_deg
        self.target_rotation = min(0.0, max(self.wing.min_rotation_deg, new_target))

    def nudge_speed(self, delta_kmh):
        new_speed = self.road_speed_kmh + delta_kmh
        self.road_speed_kmh = min(ROAD_SPEED_MAX_KMH, max(ROAD_SPEED_MIN_KMH, new_speed))

    def advance(self):
        if self.diverged_message is not None:
            return

        remaining = self.target_rotation - self.rotation
        if abs(remaining) > 1e-9:
            turn = min(FLAP_TURN_DEG_PER_FRAME, max(-FLAP_TURN_DEG_PER_FRAME, remaining))
            self.rotation += turn
            self.place_wing()

        self.sim.run(WING_STEPS_PER_FRAME)
        self.frames_since_change += 1

        health = self.sim.health()
        if not np.isfinite(health) or health > self.sim.health_limit:
            self.diverged_message = (f"DIVERGED: max|u| = {health:.3f} > "
                                     f"{self.sim.health_limit}. Restart.")
            print(self.diverged_message, file=sys.stderr)
            return

        drag, lift = self.sim.coefficients(ref_len=self.wing.ref_chord_mm / MM_PER_CELL)
        self.drag_coefficient = drag
        self.lift_coefficient = lift
        if np.isfinite(self.average_downforce):
            downforce_change = self.downforce_coefficient() - self.average_downforce
            drag_change = drag - self.average_drag
            self.average_downforce += downforce_change * FORCE_AVERAGE_FRACTION
            self.average_drag += drag_change * FORCE_AVERAGE_FRACTION
        else:
            self.average_downforce = self.downforce_coefficient()
            self.average_drag = drag
        self.look.advance(self.sim)

    def downforce_coefficient(self):
        # Cl is positive UP (see LBM.coefficients); a rear wing's job is the opposite
        return -self.lift_coefficient

    def to_newtons(self, coefficient):
        """Coefficient -> force on the WHOLE SPAN at the chosen road speed."""
        speed = self.road_speed_kmh / 3.6
        dynamic_pressure = 0.5 * AIR_DENSITY * speed * speed
        area = (self.wing.ref_chord_mm / 1000.0) * (self.wing.span_mm / 1000.0)
        return coefficient * dynamic_pressure * area

    def frame(self):
        rgb = self.look.frame(self.sim, self.field, self.polygons)

        wing = self.wing
        mode = wing.mode_name(self.rotation)
        if wing.continuous:
            upper_angle = wing.moving[0].angle_deg + self.rotation
            mode = f"{mode}   upper element {upper_angle:4.1f} deg"
        own_re = WING_RE * wing.ref_chord_mm / WING_RE_LENGTH_MM
        lines = [wing.name, f"{mode}   slot {self.slot_gap:4.1f} mm"]
        if abs(self.target_rotation - self.rotation) > 1e-9:
            lines.append("moving ...")
        elif self.frames_since_change < READOUT_SETTLE_FRAMES:
            lines.append("settling - readout still reacting to the change")
        if np.isfinite(self.average_downforce):
            downforce = self.to_newtons(self.average_downforce)
            drag = self.to_newtons(self.average_drag)
            lines.append(f"downforce  C {self.average_downforce:+.2f}   "
                         f"{downforce:7,.0f} N  ({downforce / GRAVITY:5,.0f} kg)   avg")
            lines.append(f"drag       C {self.average_drag:+.2f}   {drag:7,.0f} N   avg")
        lines.append(f"at {self.road_speed_kmh:.0f} km/h over {wing.span_mm:.0f} mm span")
        lines.append(f"2-D, Re {own_re:,.0f} (real ~{self.real_reynolds():.1e}): scaled, NOT real")
        lines.append(f"field: {self.field}")
        if self.diverged_message is not None:
            lines.append(self.diverged_message)
        rgb = draw_panel(rgb, lines, corner="tl")
        rgb = draw_panel(rgb, WING_CONTROLS, corner="bl", size=13)
        rgb = draw_panel(rgb, wing.sources + wing.notes, corner="br", size=13)
        if self.field == "speed":
            full_scale = self.road_speed_kmh * 2.20
            rgb = draw_speed_legend(rgb, f"{full_scale:.0f} km/h")
        return rgb

    def real_reynolds(self):
        kinematic_viscosity_air = 1.5e-5
        speed = self.road_speed_kmh / 3.6
        return speed * (self.wing.ref_chord_mm / 1000.0) / kinematic_viscosity_air


class WingWindow(Window):
    def on_key(self, event):
        if event.keysym == "Tab":
            self.tunnel.next_wing()
        elif event.keysym in ("d", "D", "space"):
            self.tunnel.toggle()
        elif event.keysym in ("Up", "Right"):
            self.tunnel.nudge_flap(+GT3RS_STEP_DEG)
        elif event.keysym in ("Down", "Left"):
            self.tunnel.nudge_flap(-GT3RS_STEP_DEG)
        elif event.keysym == "bracketright":
            self.tunnel.nudge_speed(+ROAD_SPEED_STEP_KMH)
        elif event.keysym == "bracketleft":
            self.tunnel.nudge_speed(-ROAD_SPEED_STEP_KMH)
        elif event.char in WING_FIELD_KEYS:
            self.tunnel.field = WING_FIELD_KEYS[event.char]
        # stop Tk's own Tab binding from moving keyboard focus out of the window
        return "break"

    def on_wheel(self, event):
        if event.delta > 0:
            self.tunnel.nudge_flap(+GT3RS_STEP_DEG)
        else:
            self.tunnel.nudge_flap(-GT3RS_STEP_DEG)


# --- self-test ------------------------------------------------------------------------------
def report(name, passed, detail):
    if passed:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"  {status}  {name}  ({detail})")
    return passed


def selftest():
    print("selftest: settling the tunnel ...")
    tunnel = Tunnel(scale=1)
    results = []

    # End to end 1: six presses up, let the body turn and the flow respond. A cambered 2412
    # at +6 deg is well above its zero-lift angle (about -2 deg), so lift must be positive.
    for _ in range(6):
        tunnel.nudge_target(+AOA_STEP_DEG)
    for _ in range(120):
        tunnel.advance()
    reached = abs(tunnel.scene.aoa - 6.0) < 1e-6
    lifting = np.isfinite(tunnel.lift_coefficient) and tunnel.lift_coefficient > 0.0
    stable = tunnel.diverged_message is None
    results.append(report("AoA control drives the body and the solver makes lift",
                          reached and lifting and stable,
                          f"AoA {tunnel.scene.aoa:.2f} deg, Cl {tunnel.lift_coefficient:+.3f}"))

    # End to end 2: every field key produces a full frame with the body painted on it.
    expected_shape = (tunnel.sim.ny, tunnel.sim.nx, 3)
    all_fields_ok = True
    for key in FIELD_KEYS:
        tunnel.field = FIELD_KEYS[key]
        rgb = tunnel.frame()
        if rgb.shape != expected_shape:
            all_fields_ok = False
    results.append(report("every field key renders a full frame",
                          all_fields_ok, f"{len(FIELD_KEYS)} fields at {expected_shape}"))

    if all(results):
        return 0
    return 1


def rule_violations(wing_list):
    """Every FIA limit the F1 geometry is meant to meet, checked in the rulebook's millimetres."""
    problems = []
    f1_2025 = wing_list[0]
    f1_2026 = wing_list[1]

    closed = f1_2025.slot_gap_mm(0.0)
    opened = f1_2025.slot_gap_mm(f1_2025.toggle_rotation_deg)
    if not 10.0 <= closed <= 15.0:
        problems.append(f"2025 closed gap {closed:.2f} mm not in 10-15")
    if not 9.4 <= opened <= 85.0:
        problems.append(f"2025 DRS gap {opened:.2f} mm not in 9.4-85")
    if not f1_2025.moving[0].chord_mm < f1_2025.fixed[0].chord_mm:
        problems.append("2025 flap chord is not smaller than the main plane's")
    # The box is checked in the design (closed) position only: Art. 3.10.10 d exempts the
    # deployed DRS position from Art. 3.10.1, which is where the box requirement lives.
    box = wings.F1_2025_BOX
    for rotation in (0.0,):
        for point in np.concatenate(f1_2025.polygons_mm(rotation)):
            x, z = float(point[0]), float(point[1])
            # RV-RW-PROFILES top edge: flat at 825 to X 355, ramps to 910 at X 530
            if x < 355.0:
                top = 825.0
            elif x < 530.0:
                top = 825.0 + (x - 355.0) * (910.0 - 825.0) / (530.0 - 355.0)
            else:
                top = 910.0
            if not (box["x_min"] <= x <= box["x_max"] and box["z_min"] <= z <= top):
                problems.append(f"2025 point ({x:.1f}, {z:.1f}) outside RV-RW-PROFILES")
                break
    if not (f1_2025.pivot_mm[0] >= box["x_max"] - 20.0 and f1_2025.pivot_mm[1] >= box["z_max"] - 20.0):
        problems.append("2025 flap axis not within 20 mm of the box's rear-top corner")

    design_2026 = f1_2026.polygons_mm(0.0)
    for index in range(len(design_2026) - 1):
        gap = wings.gap_mm(design_2026[index], design_2026[index + 1])
        if not 10.0 <= gap <= 15.0:
            problems.append(f"2026 gap {index}-{index + 1} is {gap:.2f} mm, not 10-15")
    x_mode_gap = f1_2026.slot_gap_mm(f1_2026.toggle_rotation_deg)
    if not 10.0 <= x_mode_gap <= 65.0:
        problems.append(f"2026 X-mode gap {x_mode_gap:.2f} mm not in 10-65")
    # Design position only, as above: Art. 3.11.6 c ii exempts the rotated position from
    # Art. 3.11.1 (a), the box requirement.
    box = wings.F1_2026_BOX
    for rotation in (0.0,):
        for point in np.concatenate(f1_2026.polygons_mm(rotation)):
            x, z = float(point[0]), float(point[1])
            if not (box["x_min"] <= x <= box["x_max"] and box["z_min"] <= z <= box["z_max"]):
                problems.append(f"2026 point ({x:.1f}, {z:.1f}) outside RV-RW-PROFILES")
                break
    elements_2026 = f1_2026.fixed + f1_2026.moving
    for index in range(len(elements_2026)):
        angle = wings.trailing_edge_underside_angle(elements_2026[index].underside())
        cap = wings.F1_2026_ANGLE_CAPS_DEG[index]
        if angle > cap:
            problems.append(f"2026 section {index} trailing edge {angle:.1f} deg over its {cap} cap")
    if not 450.0 <= f1_2026.pivot_mm[0] <= 525.0:
        problems.append(f"2026 flap axis at XR {f1_2026.pivot_mm[0]:.0f}, not in 450-525")
    return problems


def mean_over_frames(tunnel, frames):
    downforce_total = 0.0
    drag_total = 0.0
    for _ in range(frames):
        tunnel.advance()
        downforce_total += tunnel.downforce_coefficient()
        drag_total += tunnel.drag_coefficient
    return downforce_total / frames, drag_total / frames


def selftest_wings():
    results = []

    problems = rule_violations(wings.build_all())
    if problems:
        detail = "; ".join(problems)
    else:
        detail = "gaps, boxes, 2026 trailing-edge caps, flap axes"
    results.append(report("F1 geometry meets the FIA limits it was built to", not problems, detail))

    print("selftest: settling the wing tunnel ...")
    tunnel = WingTunnel(scale=1)
    # long enough for the vortex shed by the flap moving to leave; see READOUT_SETTLE_FRAMES
    switch_settle_frames = 200
    settle_frames = 60
    average_frames = 80
    every_wing_ok = True
    details = []
    every_frame_ok = True
    for wing_number in range(len(tunnel.wings)):
        if wing_number > 0:
            tunnel.next_wing()
        mean_over_frames(tunnel, switch_settle_frames)
        downforce_closed, drag_closed = mean_over_frames(tunnel, average_frames)
        tunnel.toggle()
        while abs(tunnel.target_rotation - tunnel.rotation) > 1e-9:
            tunnel.advance()
        mean_over_frames(tunnel, settle_frames)
        downforce_open, drag_open = mean_over_frames(tunnel, average_frames)
        if not (downforce_open < downforce_closed and drag_open < drag_closed):
            every_wing_ok = False
        details.append(f"{tunnel.wing.name.split(' rear')[0]}: down {downforce_closed:+.2f}->"
                       f"{downforce_open:+.2f}, drag {drag_closed:+.2f}->{drag_open:+.2f}")
        rgb = tunnel.frame()
        if rgb.shape != (WING_NY, WING_NX, 3):
            every_frame_ok = False
        if tunnel.diverged_message is not None:
            every_wing_ok = False
    results.append(report("opening DRS / X-mode / flat lowers downforce and drag on every wing",
                          every_wing_ok, "; ".join(details)))
    results.append(report("every wing renders a full frame",
                          every_frame_ok, f"{len(tunnel.wings)} wings at {WING_NX}x{WING_NY}"))

    if all(results):
        return 0
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=int, default=2, help="integer upscale of the window")
    parser.add_argument("--selftest", action="store_true", help="run the checks, no window")
    parser.add_argument("--wings", action="store_true",
                        help="F1 2025 / F1 2026 / GT3 RS rear wings instead of the NACA 2412")
    args = parser.parse_args(argv)

    if args.wings and args.selftest:
        return selftest_wings()
    if args.selftest:
        return selftest()
    if args.wings:
        print(f"backend: {describe_backend()}")
        print("settling ...")
        wing_tunnel = WingTunnel(scale=args.scale)
        print(wing_tunnel.sim.describe())
        WingWindow(wing_tunnel, "windtunnel - rear wings").run()
        return 0

    print(f"backend: {describe_backend()}")
    print("settling ...")
    tunnel = Tunnel(scale=args.scale)
    print(tunnel.sim.describe())
    Window(tunnel, f"windtunnel - NACA {tunnel.scene.profile}").run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
