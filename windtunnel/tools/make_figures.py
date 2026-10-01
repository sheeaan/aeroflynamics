"""make_figures.py - every figure in the top-level README, generated from the simulation.

    python tools/make_figures.py data       # run the sweeps -> docs/figures/data/*.csv (slow)
    python tools/make_figures.py plots      # CSV -> result plots, light + dark
    python tools/make_figures.py geometry   # wing anatomy, and the sections against the FIA boxes
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

from interactive import MM_PER_CELL, ThicknessTunnel, WingTunnel  # noqa: E402
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
def themed_figure(mode, panels, height=4.2, width_ratios=None, rows=1):
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
    if rows == 1:
        figure, axes = plt.subplots(1, panels, figsize=(5.2 * panels, height), dpi=160,
                                    gridspec_kw={"width_ratios": width_ratios})
    else:
        figure, axes = plt.subplots(rows, 1, figsize=(10.0, height), dpi=160)
    if panels == 1 and rows == 1:
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
        anatomy_plot(mode)


# --- wing anatomy ----------------------------------------------------------------------------
def camber_line_mm(element):
    """The NACA 6412 mean line of `element`, inverted and placed exactly like its outline."""
    camber = 0.06
    position = 0.4
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 80)))
    y = np.where(x < position,
                 camber / position ** 2 * (2 * position * x - x ** 2),
                 camber / (1 - position) ** 2 * ((1 - 2 * position) + 2 * position * x - x ** 2))
    unit = np.stack([x, y], axis=1)
    return wings.inverted_outline(unit, element.chord_mm, element.le_x_mm, element.le_z_mm,
                                  element.angle_deg)


def label(axis, text, point, text_position, theme):
    axis.annotate(text, point, xytext=text_position, fontsize=9.5, color=theme["ink"],
                  ha="center", va="center",
                  arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8,
                              "shrinkA": 3, "shrinkB": 1})


def anatomy_plot(mode):
    figure, axes, theme = themed_figure(mode, 1, height=4.6)
    axis = axes[0]
    figure.set_size_inches(10.5, 4.6)
    wing = wings.build_all()[0]
    main = wing.fixed[0]
    flap = wing.moving[0]
    outlines = wing.polygons_mm(0.0)
    for outline in outlines:
        axis.fill(outline[:, 0], outline[:, 1], color=theme["series"][0], linewidth=0)

    leading_edge = main.polygon()[wings.PROFILE_LEADING_EDGE_INDEX]
    main_trailing_edge = main.trailing_edge()
    flap_trailing_edge = flap.trailing_edge()

    # chord line, the freestream direction through the leading edge, and the angle between them
    axis.plot([leading_edge[0], main_trailing_edge[0]], [leading_edge[1], main_trailing_edge[1]],
              color=theme["ink"], linewidth=1)
    axis.plot([leading_edge[0], leading_edge[0] + 300], [leading_edge[1], leading_edge[1]],
              color=theme["muted"], linewidth=1)
    arc_radius = 120
    arc_angles = np.radians(np.linspace(0.0, main.angle_deg, 30))
    axis.plot(leading_edge[0] + arc_radius * np.cos(arc_angles),
              leading_edge[1] + arc_radius * np.sin(arc_angles), color=theme["ink"], linewidth=1)
    arc_middle = np.radians(main.angle_deg / 2)
    axis.text(leading_edge[0] + (arc_radius + 7) * np.cos(arc_middle),
              leading_edge[1] + (arc_radius + 7) * np.sin(arc_middle), "α", fontsize=11,
              color=theme["ink"], va="center")
    axis.text(leading_edge[0] + 308, leading_edge[1], "airflow direction:  "
              f"α = {main.angle_deg:.0f}° to the chord", fontsize=9.5,
              color=theme["ink"], va="center")

    camber = camber_line_mm(main)
    axis.plot(camber[:, 0], camber[:, 1], color=theme["series"][1], linewidth=1.6)

    start, end = closest_points(outlines[0], outlines[1])
    axis.plot([start[0], end[0]], [start[1], end[1]], color=theme["ink"], linewidth=1.4)

    label(axis, "main plane", (240, 690), (180, 655), theme)
    label(axis, "flap", (470, 820), (430, 870), theme)
    label(axis, "leading edge", leading_edge, (80, 735), theme)
    label(axis, "flap trailing edge", flap_trailing_edge, (610, 900), theme)
    label(axis, "chord line", (300, 721.8), (300, 790), theme)
    label(axis, "camber line", tuple(camber[22]), (95, 680), theme)
    label(axis, "slot gap, 12 mm", 0.5 * (start + end), (520, 770), theme)

    axis.annotate("", xy=(140, 610), xytext=(40, 610),
                  arrowprops={"arrowstyle": "-|>", "color": theme["ink"], "linewidth": 1.4})
    axis.text(40, 620, "airflow", fontsize=10, color=theme["ink"], va="bottom")
    axis.text(330, 845, "pressure side: higher pressure than underneath", fontsize=9.5,
              color=theme["secondary"], ha="center", style="italic")
    axis.text(330, 640, "suction side: faster air, lower pressure", fontsize=9.5,
              color=theme["secondary"], ha="center", style="italic")
    axis.annotate("", xy=(480, 600), xytext=(480, 650),
                  arrowprops={"arrowstyle": "-|>", "color": theme["ink"], "linewidth": 2})
    axis.text(492, 612, "net force: downforce", fontsize=10, color=theme["ink"],
              fontweight="semibold", va="center")

    axis.set_aspect("equal")
    axis.set_xlim(20, 640)
    axis.set_ylim(590, 915)
    axis.axis("off")
    figure.suptitle("Anatomy of a rear wing (F1 2025 section, DRS closed)", x=0.01, ha="left",
                    fontsize=12.5, fontweight="semibold", color=theme["ink"])
    save(figure, "anatomy", mode)


# --- where the downforce comes from: surface pressure ---------------------------------------
# Pressure is sampled this far outside the wall, along the outward normal: the first fluid
# cells rather than the stair-stepped wall cells themselves, which carry the bounce-back noise.
SURFACE_OFFSET_CELLS = 1.5
FREESTREAM_COLUMNS = (20, 40)       # upstream of the wing, clear of it and of the outlet sponge


def bilinear(field, x, y):
    ny, nx = field.shape
    x = np.clip(x, 0.0, nx - 1.001)
    y = np.clip(y, 0.0, ny - 1.001)
    column = x.astype(np.int32)
    row = y.astype(np.int32)
    fraction_x = x - column
    fraction_y = y - row
    top = field[row, column] * (1 - fraction_x) + field[row, column + 1] * fraction_x
    bottom = field[row + 1, column] * (1 - fraction_x) + field[row + 1, column + 1] * fraction_x
    return top * (1 - fraction_y) + bottom * fraction_y


def surface_segments(polygon):
    """Midpoints, outward unit normals and lengths of every edge of a closed polygon."""
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    edges = ends - starts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1) / np.maximum(lengths, 1e-12)[:, None]
    # Shoelace: with a positive signed area, (dy, -dx) points out of the polygon; flip otherwise.
    signed_area = 0.5 * float(np.sum(starts[:, 0] * ends[:, 1] - ends[:, 0] * starts[:, 1]))
    if signed_area < 0:
        normals = -normals
    return 0.5 * (starts + ends), normals, lengths


def collect_surface_pressure():
    print("Surface pressure on the simple wing (12%) ...")
    from matplotlib.path import Path
    tunnel = ThicknessTunnel(scale=1)
    settle(tunnel, SETTLE_FRAMES_AFTER_SWITCH)
    pressure_sum = None
    downforce_samples = []
    drag_samples = []
    for _ in range(AVERAGE_FRAMES):
        tunnel.advance()
        pressure = np.asarray(tunnel.sim.pressure(), dtype=np.float64)
        if pressure_sum is None:
            pressure_sum = pressure
        else:
            pressure_sum = pressure_sum + pressure
        downforce_samples.append(tunnel.downforce_coefficient())
        drag_samples.append(tunnel.drag_coefficient)
    mean_pressure = pressure_sum / AVERAGE_FRAMES

    polygon = tunnel.polygons[0]
    midpoints, normals, lengths = surface_segments(polygon)
    probes = midpoints + SURFACE_OFFSET_CELLS * normals
    outside = ~Path(polygon).contains_points(probes)
    if outside.mean() < 0.95:
        raise RuntimeError(f"only {outside.mean():.0%} of pressure probes are outside the wing")
    surface_pressure = bilinear(mean_pressure, probes[:, 0], probes[:, 1])

    # Force from pressure alone: F = -sum(p n ds). Lattice y points DOWN, so +Fy is downforce.
    dynamic_pressure = 0.5 * tunnel.sim.u0 ** 2
    chord_cells = wings.SIMPLE_WING_CHORD_MM / MM_PER_CELL
    force_x = -float(np.sum(surface_pressure * normals[:, 0] * lengths))
    force_y = -float(np.sum(surface_pressure * normals[:, 1] * lengths))
    downforce_from_pressure = force_y / (dynamic_pressure * chord_cells)
    drag_from_pressure = force_x / (dynamic_pressure * chord_cells)

    freestream = float(mean_pressure[:, FREESTREAM_COLUMNS[0]:FREESTREAM_COLUMNS[1]].mean())
    pressure_coefficient = (surface_pressure - freestream) / dynamic_pressure
    leading_edge = polygon[wings.PROFILE_LEADING_EDGE_INDEX]
    trailing_edge = polygon[0]
    chord_vector = trailing_edge - leading_edge
    along = ((midpoints - leading_edge) @ chord_vector) / float(chord_vector @ chord_vector)

    rows = []
    for index in range(len(polygon)):
        # The outline runs trailing edge -> leading edge along the ORIGINAL upper surface, which
        # the inversion turned into the underside: the suction side.
        if index < wings.PROFILE_LEADING_EDGE_INDEX:
            side = "suction side (underneath)"
        else:
            side = "pressure side (on top)"
        rows.append([side, float(along[index]), float(pressure_coefficient[index])])
    write_csv("surface_pressure.csv", ["side", "x_over_c", "cp"], rows)
    write_csv("pressure_check.csv", ["quantity", "value"], [
        ["downforce coefficient, momentum exchange", float(np.mean(downforce_samples))],
        ["downforce coefficient, surface pressure integral", downforce_from_pressure],
        ["drag coefficient, momentum exchange", float(np.mean(drag_samples))],
        ["drag coefficient, surface pressure integral", drag_from_pressure],
    ])
    print(f"  C_down: momentum exchange {np.mean(downforce_samples):.3f}, "
          f"pressure integral {downforce_from_pressure:.3f}")
    print(f"  C_D:    momentum exchange {np.mean(drag_samples):.3f}, "
          f"pressure integral {drag_from_pressure:.3f}")


def surface_pressure_plot(mode):
    rows = read_csv("surface_pressure.csv")
    checks = {}
    for row in read_csv("pressure_check.csv"):
        checks[row["quantity"]] = float(row["value"])
    figure, axes, theme = themed_figure(mode, 1, height=4.4)
    figure.set_size_inches(9.5, 4.4)
    axis = axes[0]
    sides = ("suction side (underneath)", "pressure side (on top)")
    curves = {}
    for series_index in range(2):
        side = sides[series_index]
        x_values = []
        cp_values = []
        for row in rows:
            if row["side"] == side and 0.0 <= float(row["x_over_c"]) <= 1.0:
                x_values.append(float(row["x_over_c"]))
                cp_values.append(float(row["cp"]))
        order = np.argsort(x_values)
        curves[side] = (np.array(x_values)[order], np.array(cp_values)[order])
        axis.plot(curves[side][0], curves[side][1], color=theme["series"][series_index],
                  linewidth=2, label=side)
    grid = np.linspace(0.0, 1.0, 400)
    suction = np.interp(grid, *curves[sides[0]])
    pushing = np.interp(grid, *curves[sides[1]])
    axis.fill_between(grid, suction, pushing, color=theme["series"][0], alpha=0.10, linewidth=0)
    area = float(np.trapezoid(pushing - suction, grid))
    axis.invert_yaxis()
    axis.axhline(0, color=theme["axis"], linewidth=1)
    axis.set_xlabel("position along the chord, x / c  (0 = leading edge, 1 = trailing edge)")
    axis.set_ylabel("pressure coefficient $C_p$  (suction ↑)")
    axis.legend(loc="lower right", labelcolor=theme["secondary"])
    momentum = checks["downforce coefficient, momentum exchange"]
    integral = checks["downforce coefficient, surface pressure integral"]
    axis.text(0.40, 0.93, f"shaded area = {area:.2f}\n"
              f"∮ pressure: $C_{{down}}$ = {integral:.2f}\n"
              f"momentum exchange: $C_{{down}}$ = {momentum:.2f}",
              transform=axis.transAxes, fontsize=9.5, color=theme["ink"], va="top")
    figure.suptitle("Where the downforce comes from: pressure along the simple wing (12% thick)",
                    x=0.01, y=1.03, ha="left", fontsize=12.5, fontweight="semibold",
                    color=theme["ink"])
    figure.text(0.01, -0.04, "Time-averaged over 120 frames, sampled 1.5 cells off the wall.  "
                "Shaded area = force normal to the chord.  $C_p$ is relative to the upstream "
                "freestream.\nThe wiggles near the leading edge come from the grid: the curved "
                "wall is made of square cells.", ha="left", fontsize=8.5, color=theme["muted"])
    save(figure, "surface_pressure", mode)


# --- hysteresis, made visible ----------------------------------------------------------------
HYSTERESIS_PERCENT = 21
HYSTERESIS_PATH_UP = [9, 12, 15, 18]


def averaged_flow(tunnel):
    """Mean velocity field and mean coefficients over AVERAGE_FRAMES frames."""
    sum_x = None
    sum_y = None
    downforce_samples = []
    for _ in range(AVERAGE_FRAMES):
        tunnel.advance()
        ux = np.asarray(tunnel.sim.ux, dtype=np.float64)
        uy = np.asarray(tunnel.sim.uy, dtype=np.float64)
        if sum_x is None:
            sum_x = ux
            sum_y = uy
        else:
            sum_x = sum_x + ux
            sum_y = sum_y + uy
        downforce_samples.append(tunnel.downforce_coefficient())
    return sum_x / AVERAGE_FRAMES, sum_y / AVERAGE_FRAMES, float(np.mean(downforce_samples))


def collect_hysteresis():
    """Follow the thickness sweep's own path to 21% from below and from above."""
    print(f"Hysteresis: the {HYSTERESIS_PERCENT}% wing reached from both sides ...")
    tunnel = ThicknessTunnel(scale=1)
    tunnel.set_target_thickness(THICKNESS_UP_PERCENT[0] / 100.0)
    run_until_still(tunnel)
    settle(tunnel, SETTLE_FRAMES_AFTER_SWITCH)
    measure(tunnel, AVERAGE_FRAMES)
    for percent in HYSTERESIS_PATH_UP:
        tunnel.set_target_thickness(percent / 100.0)
        run_until_still(tunnel)
        settle(tunnel, SETTLE_FRAMES)
        measure(tunnel, AVERAGE_FRAMES)
    tunnel.set_target_thickness(HYSTERESIS_PERCENT / 100.0)
    run_until_still(tunnel)
    settle(tunnel, SETTLE_FRAMES)
    up_x, up_y, up_downforce = averaged_flow(tunnel)
    polygon = tunnel.polygons[0].copy()

    tunnel.set_target_thickness(THICKNESS_UP_PERCENT[-1] / 100.0)
    run_until_still(tunnel)
    settle(tunnel, SETTLE_FRAMES)
    measure(tunnel, AVERAGE_FRAMES)
    tunnel.set_target_thickness(HYSTERESIS_PERCENT / 100.0)
    run_until_still(tunnel)
    settle(tunnel, SETTLE_FRAMES)
    down_x, down_y, down_downforce = averaged_flow(tunnel)

    # crop to the wing and its near wake, so the saved fields stay small
    left = int(polygon[:, 0].min()) - 50
    right = int(polygon[:, 0].max()) + 200
    top = int(polygon[:, 1].min()) - 75
    bottom = int(polygon[:, 1].max()) + 75
    crop = (slice(top, bottom), slice(left, right))
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(os.path.join(DATA, "hysteresis_fields.npz"),
                        up_ux=up_x[crop].astype(np.float32), up_uy=up_y[crop].astype(np.float32),
                        down_ux=down_x[crop].astype(np.float32),
                        down_uy=down_y[crop].astype(np.float32),
                        polygon=(polygon - np.array([left, top])).astype(np.float32),
                        freestream=np.float32(tunnel.sim.u0))
    write_csv("hysteresis.csv", ["state", "thickness_pct", "downforce_mean"], [
        ["thickening", HYSTERESIS_PERCENT, up_downforce],
        ["thinning", HYSTERESIS_PERCENT, down_downforce]])
    print(f"  C_down at {HYSTERESIS_PERCENT}%: thickening {up_downforce:.2f}, "
          f"thinning {down_downforce:.2f}")


SPEED_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def hysteresis_plot(mode):
    from matplotlib.colors import LinearSegmentedColormap
    fields = np.load(os.path.join(DATA, "hysteresis_fields.npz"))
    states = {}
    for row in read_csv("hysteresis.csv"):
        states[row["state"]] = float(row["downforce_mean"])
    from matplotlib.path import Path
    figure, axes, theme = themed_figure(mode, 1, height=6.0, rows=2)
    # sequential, one hue: faster air is darker on the light theme and lighter on the dark one
    if mode == "light":
        ramp = SPEED_RAMP
    else:
        ramp = list(reversed(SPEED_RAMP))
    colour_map = LinearSegmentedColormap.from_list("speed", ramp)
    freestream = float(fields["freestream"])
    chord_cells = wings.SIMPLE_WING_CHORD_MM / MM_PER_CELL
    polygon = fields["polygon"]
    leading_edge_x = float(polygon[:, 0].min())
    centre_y = float(polygon[:, 1].mean())
    panels = (("up", "thickening", "Reached by thickening (6% → 21%)"),
              ("down", "thinning", "Reached by thinning (24% → 21%)"))
    image = None
    for panel_index in range(2):
        axis = axes[panel_index]
        prefix, state, title = panels[panel_index]
        ux = fields[f"{prefix}_ux"]
        uy = fields[f"{prefix}_uy"]
        ny, nx = ux.shape
        x = (np.arange(nx) - leading_edge_x) / chord_cells
        y = (centre_y - np.arange(ny)) / chord_cells        # flip: lattice rows run downward
        # Inside the wing the velocity is zero by construction, so mask it out: otherwise the
        # u_x = 0 contour traces the wing's own interior instead of the separated region.
        column_grid, row_grid = np.meshgrid(np.arange(nx), np.arange(ny))
        cell_centres = np.stack([column_grid.ravel(), row_grid.ravel()], axis=1)
        inside = Path(polygon).contains_points(cell_centres).reshape(ny, nx)
        ux_fluid = np.ma.masked_where(inside, ux)
        speed = np.ma.masked_where(inside, np.hypot(ux, uy) / freestream)
        image = axis.pcolormesh(x, y, speed, cmap=colour_map, vmin=0.0, vmax=1.4,
                                shading="gouraud", rasterized=True)
        axis.streamplot(x, y[::-1], ux[::-1] / freestream, -uy[::-1] / freestream,
                        color=theme["ink"], linewidth=0.45, density=1.1, arrowsize=0.55)
        axis.contourf(x, y, ux_fluid, levels=[-1.0, 0.0], colors=[theme["series"][1]],
                      alpha=0.35)
        axis.contour(x, y, ux_fluid, levels=[0.0], colors=[theme["series"][1]], linewidths=1.6)
        axis.fill((polygon[:, 0] - leading_edge_x) / chord_cells,
                  (centre_y - polygon[:, 1]) / chord_cells, color=theme["ink"], linewidth=0)
        axis.set_aspect("equal")
        axis.set_xlim(-0.25, 1.9)
        axis.set_ylim(-0.4, 0.35)
        axis.grid(False)
        axis.set_title(f"{title}:  $C_{{down}}$ = {states[state]:.2f}", loc="left",
                       fontsize=10.5, color=theme["ink"], pad=8)
        axis.set_ylabel("z / chord")
    axes[1].set_xlabel("x / chord")
    bar = figure.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label("flow speed / freestream", color=theme["secondary"])
    bar.outline.set_visible(False)
    figure.suptitle(f"Same wing, same thickness ({HYSTERESIS_PERCENT}%), two different flows",
                    x=0.01, y=1.04, ha="left", fontsize=12.5, fontweight="semibold",
                    color=theme["ink"])
    figure.text(0.01, -0.02, "Time-averaged over 120 frames.  Thin lines: mean streamlines.  "
                "Orange: where the mean flow runs backwards ($u_x$ < 0), the separated region.",
                ha="left", fontsize=8.5, color=theme["muted"])
    save(figure, "hysteresis", mode)


def make_explainers():
    print("Surface-pressure and hysteresis figures ...")
    for mode in THEMES:
        surface_pressure_plot(mode)
        hysteresis_plot(mode)


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
        collect_surface_pressure()
        collect_hysteresis()
    if stage in ("plots", "all"):
        make_plots()
        make_explainers()
    if stage in ("geometry", "all"):
        make_geometry()
    if stage in ("hero", "all"):
        make_hero()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
