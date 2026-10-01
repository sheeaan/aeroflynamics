"""make_figures.py - every figure in the top-level README, generated from the simulation.

    python tools/make_figures.py data       # run the sweeps -> docs/figures/data/*.csv (slow)
    python tools/make_figures.py plots      # CSV -> result plots, light + dark
    python tools/make_figures.py geometry   # wing sections against the FIA reference boxes
    python tools/make_figures.py hero       # the DRS animation (WebP)
    python tools/make_figures.py all

Nothing in a figure is typed in by hand. The plots read only the CSVs, and the CSVs hold only
what the solver measured: for every operating point, the MEAN and STANDARD DEVIATION of the
instantaneous force coefficient over AVERAGE_FRAMES frames, taken after SETTLE_FRAMES frames
for the flow to forget the previous geometry.

The standard deviation is the flow's own unsteadiness - vortex shedding moving the force
around - not a measurement error. It is drawn as a pale band, so a reader can see how much of
a difference between two points is bigger than the flow's own wobble.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDTUNNEL = os.path.dirname(HERE)
ROOT = os.path.dirname(WINDTUNNEL)
FIGURES = os.path.join(ROOT, "docs", "figures")
DATA = os.path.join(FIGURES, "data")
sys.path.insert(0, WINDTUNNEL)

from interactive import ThicknessTunnel, WingTunnel  # noqa: E402
from wt import wings  # noqa: E402
from wt.look import draw_panel  # noqa: E402

SETTLE_FRAMES_AFTER_SWITCH = 300
SETTLE_FRAMES = 150
AVERAGE_FRAMES = 120

GT3RS_FLATTENING_DEG = [40, 34, 28, 22, 16, 10, 6]
GT3RS_TILTING_UP_DEG = [6, 10, 16, 22, 28, 34, 40]
THICKNESS_UP_PERCENT = [6, 9, 12, 15, 18, 21, 24]
THICKNESS_DOWN_PERCENT = [24, 21, 18, 15, 12, 9, 6]

# Validated with the dataviz palette validator: both modes pass every check (CVD dE 24.7
# light / 26.8 dark). Slot 1 = closed / increasing, slot 2 = open / decreasing.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926"]},
}
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


# --- measurement -----------------------------------------------------------------------------
def measure(tunnel, frames):
    """Mean and standard deviation of the instantaneous coefficients over `frames` frames."""
    downforce_samples = []
    drag_samples = []
    for _ in range(frames):
        tunnel.advance()
        downforce_samples.append(tunnel.downforce_coefficient())
        drag_samples.append(tunnel.drag_coefficient)
    if tunnel.diverged_message is not None:
        raise RuntimeError(tunnel.diverged_message)
    return (float(np.mean(downforce_samples)), float(np.std(downforce_samples)),
            float(np.mean(drag_samples)), float(np.std(drag_samples)))


def run_until_still(tunnel):
    """Advance until a moving flap or a changing thickness has reached its target."""
    while True:
        if hasattr(tunnel, "target_rotation"):
            remaining = tunnel.target_rotation - tunnel.rotation
        else:
            remaining = tunnel.target_thickness - tunnel.thickness
        if abs(remaining) <= 1e-9:
            return
        tunnel.advance()


def settle(tunnel, frames):
    for _ in range(frames):
        tunnel.advance()


def write_csv(name, header, rows):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def read_csv(name):
    with open(os.path.join(DATA, name), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_drs():
    print("DRS / X-mode / flat, each wing ...")
    tunnel = WingTunnel(scale=1)
    rows = []
    for wing_number in range(len(tunnel.wings)):
        if wing_number > 0:
            tunnel.next_wing()
        settle(tunnel, SETTLE_FRAMES_AFTER_SWITCH)
        closed = measure(tunnel, AVERAGE_FRAMES)
        tunnel.toggle()
        run_until_still(tunnel)
        settle(tunnel, SETTLE_FRAMES)
        opened = measure(tunnel, AVERAGE_FRAMES)
        name = tunnel.wing.name.replace(" rear wing", "")
        rows.append([name, tunnel.wing.mode_names[0], *closed])
        rows.append([name, tunnel.wing.mode_names[1], *opened])
        print(f"  {name}: C_down {closed[0]:.2f} -> {opened[0]:.2f}")
    write_csv("drs.csv", ["wing", "state", "downforce_mean", "downforce_std",
                          "drag_mean", "drag_std"], rows)


def collect_gt3rs_flap():
    print("GT3 RS flap sweep, flattening then tilting up ...")
    tunnel = WingTunnel(scale=1)
    tunnel.next_wing()
    tunnel.next_wing()
    settle(tunnel, SETTLE_FRAMES_AFTER_SWITCH)
    full_tilt = tunnel.wing.moving[0].angle_deg
    rows = []
    for direction, angles in (("flattening", GT3RS_FLATTENING_DEG),
                              ("tilting up", GT3RS_TILTING_UP_DEG)):
        for angle in angles:
            tunnel.target_rotation = angle - full_tilt
            run_until_still(tunnel)
            settle(tunnel, SETTLE_FRAMES)
            result = measure(tunnel, AVERAGE_FRAMES)
            rows.append([direction, angle, *result])
            print(f"  {direction:10s} {angle:2d} deg: C_down {result[0]:.2f}  C_D {result[2]:.3f}")
    write_csv("gt3rs_flap.csv", ["direction", "flap_deg", "downforce_mean", "downforce_std",
                                 "drag_mean", "drag_std"], rows)


def collect_thickness():
    print("Thickness sweep, thickening then thinning ...")
    tunnel = ThicknessTunnel(scale=1)
    tunnel.set_target_thickness(THICKNESS_UP_PERCENT[0] / 100.0)
    run_until_still(tunnel)
    settle(tunnel, SETTLE_FRAMES_AFTER_SWITCH)
    rows = []
    for direction, values in (("thickening", THICKNESS_UP_PERCENT),
                              ("thinning", THICKNESS_DOWN_PERCENT)):
        for percent in values:
            tunnel.set_target_thickness(percent / 100.0)
            run_until_still(tunnel)
            settle(tunnel, SETTLE_FRAMES)
            result = measure(tunnel, AVERAGE_FRAMES)
            rows.append([direction, percent, *result])
            print(f"  {direction:10s} {percent:2d}%: C_down {result[0]:.2f}  C_D {result[2]:.3f}")
    write_csv("thickness.csv", ["direction", "thickness_pct", "downforce_mean", "downforce_std",
                                "drag_mean", "drag_std"], rows)


# --- plotting --------------------------------------------------------------------------------
def themed_figure(mode, panels, height=4.2, width_ratios=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = THEMES[mode]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10.5,
        "text.color": theme["ink"], "axes.labelcolor": theme["secondary"],
        "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "axes.edgecolor": theme["axis"], "axes.facecolor": theme["surface"],
        "figure.facecolor": theme["surface"], "savefig.facecolor": theme["surface"],
        "axes.grid": True, "grid.color": theme["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })
    figure, axes = plt.subplots(1, panels, figsize=(5.2 * panels, height), dpi=160,
                                gridspec_kw={"width_ratios": width_ratios})
    if panels == 1:
        axes = [axes]
    for axis in axes:
        axis.set_axisbelow(True)
        axis.tick_params(length=0)
    return figure, axes, theme


def save(figure, name, mode):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, f"{name}.{mode}.png")
    figure.savefig(path, bbox_inches="tight", pad_inches=0.25)
    import matplotlib.pyplot as plt
    plt.close(figure)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def sweep_plot(name, csv_name, x_key, x_label, series_names, title, mode, legend_loc):
    rows = read_csv(csv_name)
    figure, axes, theme = themed_figure(mode, 2)
    measures = (("downforce", "Downforce coefficient, $C_{down}$"),
                ("drag", "Drag coefficient, $C_D$"))
    measured_x = sorted({float(row[x_key]) for row in rows})
    for panel_index in range(2):
        axis = axes[panel_index]
        key, label = measures[panel_index]
        lowest = 0.0
        for series_index in range(len(series_names)):
            series = series_names[series_index]
            colour = theme["series"][series_index]
            x_values = []
            means = []
            spreads = []
            for row in rows:
                if row["direction"] == series:
                    x_values.append(float(row[x_key]))
                    means.append(float(row[f"{key}_mean"]))
                    spreads.append(float(row[f"{key}_std"]))
            order = np.argsort(x_values)
            x_sorted = np.array(x_values)[order]
            mean_sorted = np.array(means)[order]
            spread_sorted = np.array(spreads)[order]
            axis.fill_between(x_sorted, mean_sorted - spread_sorted, mean_sorted + spread_sorted,
                              color=colour, alpha=0.10, linewidth=0)
            axis.plot(x_sorted, mean_sorted, color=colour, linewidth=2, solid_capstyle="round",
                      label=series)
            axis.plot(x_sorted, mean_sorted, "o", markersize=6, color=colour,
                      markeredgecolor=theme["surface"], markeredgewidth=2)
            lowest = min(lowest, float((mean_sorted - spread_sorted).min()))
        axis.set_title(label, loc="left", fontsize=11, color=theme["ink"], pad=10)
        axis.set_xlabel(x_label)
        axis.set_xticks(measured_x)
        axis.set_xticklabels([f"{value:g}" for value in measured_x])
        # zero stays on the axis, but nothing measured below it is clipped away
        axis.set_ylim(bottom=lowest * 1.1)
        if lowest < 0:
            axis.axhline(0, color=theme["axis"], linewidth=1)
    axes[0].legend(loc=legend_loc, labelcolor=theme["secondary"])
    figure.suptitle(title, x=0.01, y=1.04, ha="left", fontsize=12.5, fontweight="semibold",
                    color=theme["ink"])
    figure.text(0.01, -0.04, "Line: mean over 120 frames.  Band: ±1 standard deviation "
                "(the flow's own unsteadiness).  2-D lattice-Boltzmann, Re ≈ 4,500–6,400.",
                ha="left", fontsize=8.5, color=theme["muted"])
    save(figure, name, mode)


def drs_plot(mode):
    rows = read_csv("drs.csv")
    figure, axes, theme = themed_figure(mode, 2)
    wing_names = []
    for row in rows:
        if row["wing"] not in wing_names:
            wing_names.append(row["wing"])
    bar_width = 0.30
    gap = 0.02
    measures = (("downforce", "Downforce coefficient, $C_{down}$"),
                ("drag", "Drag coefficient, $C_D$"))
    for panel_index in range(2):
        axis = axes[panel_index]
        key, label = measures[panel_index]
        for wing_index in range(len(wing_names)):
            wing_rows = []
            for row in rows:
                if row["wing"] == wing_names[wing_index]:
                    wing_rows.append(row)
            for state_index in range(2):
                row = wing_rows[state_index]
                offset = (state_index - 0.5) * (bar_width + gap)
                x = wing_index + offset
                value = float(row[f"{key}_mean"])
                spread = float(row[f"{key}_std"])
                colour = theme["series"][state_index]
                if wing_index == 0:
                    legend_label = ("closed / corner mode / full tilt",
                                    "DRS open / straight mode / flat")[state_index]
                else:
                    legend_label = None
                axis.bar(x, value, width=bar_width, color=colour, label=legend_label,
                         linewidth=0)
                axis.errorbar(x, value, yerr=spread, color=theme["muted"], linewidth=1,
                              capsize=0)
                axis.text(x, value + spread + 0.03 * axis.get_ylim()[1] + 0.02,
                          f"{value:.2f}", ha="center", va="bottom", fontsize=9,
                          color=theme["secondary"])
        axis.set_xticks(range(len(wing_names)))
        axis.set_xticklabels(wing_names)
        axis.grid(axis="x", visible=False)
        axis.set_title(label, loc="left", fontsize=11, color=theme["ink"], pad=10)
        top = 0.0
        for row in rows:
            top = max(top, float(row[f"{key}_mean"]) + float(row[f"{key}_std"]))
        axis.set_ylim(0, top * 1.18)
    axes[0].legend(loc="upper right", labelcolor=theme["secondary"], fontsize=9)
    figure.suptitle("Opening the wing: what DRS, X-mode and a flat flap cost in downforce and save "
                    "in drag", x=0.01, y=1.04, ha="left", fontsize=12.5, fontweight="semibold",
                    color=theme["ink"])
    figure.text(0.01, -0.04, "Bar: mean over 120 frames.  Whisker: ±1 standard deviation.  "
                "Coefficients use each wing's own design-position chord.",
                ha="left", fontsize=8.5, color=theme["muted"])
    save(figure, "drs", mode)


def make_plots():
    print("Result plots ...")
    for mode in THEMES:
        drs_plot(mode)
        sweep_plot("gt3rs_flap", "gt3rs_flap.csv", "flap_deg", "Upper-element angle (deg)",
                   ["flattening", "tilting up"],
                   "GT3 RS: tilting the upper element buys downforce, and costs more drag",
                   mode, "upper left")
        sweep_plot("thickness", "thickness.csv", "thickness_pct",
                   "Thickness (% of chord, chord fixed at 300 mm)", ["thickening", "thinning"],
                   "Simple wing: drag grows with thickness, and downforce depends on where the flow has been",
                   mode, "upper right")


# --- geometry --------------------------------------------------------------------------------
def closest_points(polygon_a, polygon_b):
    best = (float("inf"), None, None)
    for point in polygon_a:
        distances = np.hypot(polygon_b[:, 0] - point[0], polygon_b[:, 1] - point[1])
        index = int(np.argmin(distances))
        if distances[index] < best[0]:
            best = (float(distances[index]), point, polygon_b[index])
    return best[1], best[2]


def draw_gap(axis, polygon_a, polygon_b, text, theme, side="left"):
    """The gauge line across the slot, labelled just beside it."""
    start, end = closest_points(polygon_a, polygon_b)
    axis.plot([start[0], end[0]], [start[1], end[1]], color=theme["ink"], linewidth=1.4,
              solid_capstyle="round")
    middle = 0.5 * (start + end)
    if side == "left":
        axis.text(middle[0] - 8, middle[1], text, ha="right", va="center", fontsize=9,
                  color=theme["ink"])
    else:
        # out in the empty air to the upper left, with a hairline leader back to the gap
        axis.annotate(text, middle, xytext=(middle[0] - 70, middle[1] + 22), ha="right",
                      va="center", fontsize=9, color=theme["ink"],
                      arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8,
                                  "shrinkA": 2, "shrinkB": 0})


def geometry_plot(mode):
    built = wings.build_all()
    spans = []
    for wing in built:
        points = np.concatenate(wing.polygons_mm(0.0) + wing.polygons_mm(wing.toggle_rotation_deg))
        spans.append(float(points[:, 0].max() - points[:, 0].min()))
    spans[0] = max(spans[0], 415.0)          # the 2025 box is longer than the wing
    spans[1] = max(spans[1], 390.0)
    figure, axes, theme = themed_figure(mode, 3, height=3.6, width_ratios=spans)
    closed_colour = theme["series"][0]
    open_colour = theme["series"][1]

    f1_2025 = built[0]
    box_2025 = [(140, 670), (140, 825), (355, 825), (530, 910), (555, 910), (555, 670), (140, 670)]
    f1_2026 = built[1]
    box_2026 = [(240, 700), (240, 880), (630, 880), (630, 700), (240, 700)]
    panels = ((f1_2025, box_2025, "F1 2025: DRS", "12 mm", "84 mm", "left"),
              (f1_2026, box_2026, "F1 2026: Z-mode / X-mode", "12 mm", "64 mm", "left"),
              (built[2], None, "GT3 RS: 40° to flat", "12 mm", "44 mm", "above"))
    for panel_index in range(3):
        axis = axes[panel_index]
        wing, box, title, closed_label, open_label, open_side = panels[panel_index]
        if box is not None:
            box_x = [point[0] for point in box]
            box_z = [point[1] for point in box]
            axis.plot(box_x, box_z, color=theme["muted"], linewidth=1)
            axis.text(box_x[0] + 4, max(box_z) + 6, "FIA reference volume", fontsize=8,
                      color=theme["muted"])
        closed = wing.polygons_mm(0.0)
        opened = wing.polygons_mm(wing.toggle_rotation_deg)
        # outline only the sections that MOVE; the fixed main plane is the same in both states
        for index in range(len(wing.fixed), len(opened)):
            outline = opened[index]
            axis.plot(np.append(outline[:, 0], outline[0, 0]),
                      np.append(outline[:, 1], outline[0, 1]),
                      color=open_colour, linewidth=1.6,
                      label="open" if index == len(wing.fixed) else None)
        for index in range(len(closed)):
            outline = closed[index]
            axis.fill(outline[:, 0], outline[:, 1], color=closed_colour, linewidth=0,
                      label="closed" if index == 0 else None)
        axis.plot(*wing.pivot_mm, "o", markersize=6, color=theme["ink"],
                  markeredgecolor=theme["surface"], markeredgewidth=2)
        fixed_index = len(wing.fixed) - 1
        draw_gap(axis, closed[fixed_index], closed[fixed_index + 1], closed_label, theme)
        draw_gap(axis, opened[fixed_index], opened[fixed_index + 1], open_label, theme, open_side)
        axis.set_aspect("equal")
        axis.set_title(title, loc="left", fontsize=11, color=theme["ink"], pad=10)
        axis.set_xlabel("x, mm (flow →)")
        axis.grid(False)
    axes[0].set_ylabel("z, mm")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncols=2, labelcolor=theme["secondary"],
                  fontsize=9.5)
    figure.suptitle("The three wings, built to their rules: closed (filled) and open (outline)",
                    x=0.01, ha="left", fontsize=12.5, fontweight="semibold", color=theme["ink"])
    figure.text(0.01, -0.02, "Dot: flap pivot.  Gaps are the FIA spherical-gauge "
                "distance, solved by bisection.  Sections are NACA 6412 stand-ins; real profiles "
                "are proprietary.", ha="left", fontsize=8.5, color=theme["muted"])
    save(figure, "geometry", mode)


def make_geometry():
    print("Geometry figure ...")
    for mode in THEMES:
        geometry_plot(mode)


# --- hero animation --------------------------------------------------------------------------------
HERO_WIDTH = 760
HERO_FPS = 20
HERO_QUALITY = 72          # animated WebP; GIF's 256-colour palette made this 31 MB
HERO_HOLD_FRAMES = 45


def hero_frame(tunnel):
    rgb = tunnel.look.frame(tunnel.sim, "speed", tunnel.polygons)
    if abs(tunnel.rotation) < 1e-6:
        state = "DRS closed"
    elif abs(tunnel.rotation - tunnel.wing.toggle_rotation_deg) < 1e-6:
        state = "DRS open"
    else:
        state = "DRS moving"
    # No force numbers here: through a flap stroke the instantaneous force carries the
    # mask-motion artifact, and a running average lags the state on screen. The measured
    # numbers live in the plots, where every point is settled and averaged.
    lines = ["F1 2025 rear wing  ·  2-D lattice-Boltzmann CFD",
             f"{state}   slot {tunnel.slot_gap:4.1f} mm"]
    return draw_panel(rgb, lines, corner="tl", size=16)


def make_hero():
    print("Hero animation: F1 2025, DRS closed -> open -> closed ...")
    from PIL import Image
    tunnel = WingTunnel(scale=2)
    settle(tunnel, SETTLE_FRAMES_AFTER_SWITCH)
    frames_dir = tempfile.mkdtemp(prefix="hero_")
    count = 0

    def capture():
        nonlocal count
        image = Image.fromarray(hero_frame(tunnel))
        height = int(round(image.height * HERO_WIDTH / image.width))
        image.resize((HERO_WIDTH, height), Image.LANCZOS).save(
            os.path.join(frames_dir, f"{count:04d}.png"))
        count += 1

    for _ in range(HERO_HOLD_FRAMES):
        tunnel.advance()
        capture()
    tunnel.toggle()
    while abs(tunnel.target_rotation - tunnel.rotation) > 1e-9:
        tunnel.advance()
        capture()
    for _ in range(HERO_HOLD_FRAMES + 15):
        tunnel.advance()
        capture()
    tunnel.toggle()
    while abs(tunnel.target_rotation - tunnel.rotation) > 1e-9:
        tunnel.advance()
        capture()
    for _ in range(HERO_HOLD_FRAMES):
        tunnel.advance()
        capture()

    os.makedirs(FIGURES, exist_ok=True)
    out = os.path.join(FIGURES, "hero_drs.webp")
    images = []
    for index in range(count):
        images.append(Image.open(os.path.join(frames_dir, f"{index:04d}.png")).convert("RGB"))
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / HERO_FPS),
                   loop=0, quality=HERO_QUALITY, method=4)
    shutil.rmtree(frames_dir)
    size_mb = os.path.getsize(out) / 1e6
    print(f"  wrote {os.path.relpath(out, ROOT)}  ({count} frames, {size_mb:.1f} MB)")


def main(argv):
    if len(argv) != 1 or argv[0] not in ("data", "plots", "geometry", "hero", "all"):
        print(__doc__)
        return 2
    stage = argv[0]
    if stage in ("data", "all"):
        collect_drs()
        collect_gt3rs_flap()
        collect_thickness()
    if stage in ("plots", "all"):
        make_plots()
    if stage in ("geometry", "all"):
        make_geometry()
    if stage in ("hero", "all"):
        make_hero()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
