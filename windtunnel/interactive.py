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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=int, default=2, help="integer upscale of the window")
    parser.add_argument("--selftest", action="store_true", help="run the checks, no window")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    print(f"backend: {describe_backend()}")
    print("settling ...")
    tunnel = Tunnel(scale=args.scale)
    print(tunnel.sim.describe())
    Window(tunnel, f"windtunnel - NACA {tunnel.scene.profile}").run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
