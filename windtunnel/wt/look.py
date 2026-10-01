# SPDX-License-Identifier: MIT
# Palettes and streak constants from spectrometry.mp4 engines by Ethan Earl,
# https://github.com/ec175/spectrometry_public - Copyright (c) 2026 Ethan Earl.
# Licence text: windtunnel/LICENSES/spectrometry_public-MIT.txt
"""look.py - the spectrometry.mp4 picture: jet speed field, white comet-tail streaks, clean body.

Technique, palettes and constants adapted from the wind-tunnel engine in spectrometry.mp4's
engines by Ethan Earl (MIT licence) - https://github.com/ec175/spectrometry_public. The code
here is a rewrite against this repo's solver, not a copy.

What makes that picture read as AIR rather than as a heatmap, in order of importance:

  1. The field is SPEED on classic jet - navy in the separated wake, cyan-green freestream,
     yellow-red over the suction peak - normalised to a FIXED full scale (2.2 x freestream), so
     two frames are comparable.
  2. Thousands of massless tracers are drawn as short white comet tails, integrated BACKWARD
     from each tracer's current position through the current field. The colour says how fast;
     the tails say which way, and they make a separation bubble visibly recirculate.
  3. Everything is drawn at SCREEN resolution: the field is bilinearly upsampled and the tails
     and the body outline are splatted / anti-aliased at full size.

Point 3 is a deliberate departure from wt/render.py, whose `upscale` is nearest-neighbour on
the grounds that the lattice IS the resolution. That stays true for the video renderer. This
module exists for the look, and the look needs smooth upsampling.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageDraw
from scipy.ndimage import gaussian_filter

from .gpu import asnumpy
from .render import _font

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

# (position, (r, g, b)) stops, linearly interpolated into a 256-entry table
_STOPS = {
    "jet": [(0.00, (0, 0, 131)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
            (0.625, (255, 255, 0)), (0.875, (255, 0, 0)), (1.00, (128, 0, 0))],
    # diverging, with zero vorticity near BLACK so the shed vortices are what glows
    "vort": [(0.00, (40, 130, 255)), (0.30, (10, 30, 90)), (0.50, (3, 4, 8)),
             (0.70, (110, 25, 20)), (1.00, (255, 140, 40))],
    "inferno": [(0.00, (0, 0, 4)), (0.25, (87, 16, 110)), (0.50, (188, 55, 84)),
                (0.75, (249, 142, 9)), (1.00, (252, 255, 164))],
}

SPEED_FULL_SCALE = 2.20        # speed field: top of the ramp = 2.2 x freestream
FIELD_GAMMA = 0.95

STREAKS_AT_1080x1920 = 32000   # density is quoted at that size and scaled by frame area
STREAK_TAIL_SAMPLES = 22
# Tail length in SOLVER STEPS of travel. spectrometry quotes 4 frames of ~14 steps at ~4.5
# screen px per cell; this tunnel draws ~2 px per cell, so it needs about twice the travel for
# the dashes to come out the same length on screen. At 4 frames they read as dots.
STREAK_SPAN_STEPS = 120.0
STREAK_GAIN = 0.46
STREAK_SPEED_GAIN = 0.38       # how much a fast tracer outshines a slow one
STREAK_BLUR_PX = 0.85
STREAK_LIFE_SECONDS = (0.55, 1.5)
NOMINAL_FPS = 30.0             # tracer lifetimes are counted in frames at this rate

BODY_FILL = (12, 14, 18)
BODY_EDGE = (238, 240, 236)
BODY_EDGE_PX = 3


def build_lut(name):
    stops = _STOPS[name]
    positions = np.array([stop[0] for stop in stops])
    colours = np.array([stop[1] for stop in stops], dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, 256)
    channels = []
    for channel in range(3):
        channels.append(np.interp(ramp, positions, colours[:, channel]))
    return np.clip(np.stack(channels, axis=1), 0, 255).astype(np.uint8)


LUTS = {}
for _name in _STOPS:
    LUTS[_name] = build_lut(_name)


def splat_tails_python(x, y, base, ux, uy, scale, dt_back, samples, out):
    """Trace every tracer's tail backward through (ux, uy) and splat it into `out`.

    One loop over tracers and tail samples, compiled by numba when it is installed. Each tail
    sample is splatted bilinearly into the four screen pixels around it, then the walk steps
    backward along the local velocity.
    """
    height, width = out.shape
    ny, nx = ux.shape
    for particle in range(x.shape[0]):
        px = x[particle]
        py = y[particle]
        for sample_index in range(samples):
            weight = base[particle] * (1.0 - sample_index / samples) ** 1.35
            screen_x = (px + 0.5) * scale - 0.5
            screen_y = (py + 0.5) * scale - 0.5
            if 0.0 <= screen_x <= width - 2 and 0.0 <= screen_y <= height - 2:
                column = int(screen_x)
                row = int(screen_y)
                fraction_x = screen_x - column
                fraction_y = screen_y - row
                out[row, column] += weight * (1.0 - fraction_x) * (1.0 - fraction_y)
                out[row, column + 1] += weight * fraction_x * (1.0 - fraction_y)
                out[row + 1, column] += weight * (1.0 - fraction_x) * fraction_y
                out[row + 1, column + 1] += weight * fraction_x * fraction_y
            sample_x = min(max(px, 0.0), nx - 1.001)
            sample_y = min(max(py, 0.0), ny - 1.001)
            cell_x = int(sample_x)
            cell_y = int(sample_y)
            fraction_x = sample_x - cell_x
            fraction_y = sample_y - cell_y
            velocity_x = ((ux[cell_y, cell_x] * (1.0 - fraction_x)
                           + ux[cell_y, cell_x + 1] * fraction_x) * (1.0 - fraction_y)
                          + (ux[cell_y + 1, cell_x] * (1.0 - fraction_x)
                             + ux[cell_y + 1, cell_x + 1] * fraction_x) * fraction_y)
            velocity_y = ((uy[cell_y, cell_x] * (1.0 - fraction_x)
                           + uy[cell_y, cell_x + 1] * fraction_x) * (1.0 - fraction_y)
                          + (uy[cell_y + 1, cell_x] * (1.0 - fraction_x)
                             + uy[cell_y + 1, cell_x + 1] * fraction_x) * fraction_y)
            px = px + dt_back * velocity_x
            py = py + dt_back * velocity_y


if HAVE_NUMBA:
    splat_tails_compiled = njit(cache=True)(splat_tails_python)
else:
    splat_tails_compiled = None


class Streaks:
    """Massless tracers that die young and respawn uniformly, drawn as backward comet tails.

    Uniform respawn rather than inlet seeding: inlet seeding starves the fast regions and piles
    tracers up in the slow wake within a second, so the dashes thin out exactly where the flow
    is interesting. A short randomised life holds the density flat everywhere.
    """

    def __init__(self, count, nx, ny, steps_per_frame, seed=7):
        self.count = int(count)
        self.nx = int(nx)
        self.ny = int(ny)
        self.steps_per_frame = float(steps_per_frame)
        self.random = np.random.default_rng(seed)
        self.x = self.random.random(self.count, dtype=np.float32) * (self.nx - 1)
        self.y = self.random.random(self.count, dtype=np.float32) * (self.ny - 1)
        self.life = self.new_life(self.count)
        # stagger the starting ages so the whole field does not blink in unison
        self.age = self.random.random(self.count, dtype=np.float32) * self.life

    def new_life(self, how_many):
        shortest, longest = STREAK_LIFE_SECONDS
        return (shortest + (longest - shortest)
                * self.random.random(how_many, dtype=np.float32)).astype(np.float32)

    def sample(self, field, x, y):
        """Bilinear sample of an (ny, nx) field at float lattice coordinates."""
        x = np.clip(x, 0.0, self.nx - 1.001)
        y = np.clip(y, 0.0, self.ny - 1.001)
        column = x.astype(np.int32)
        row = y.astype(np.int32)
        fraction_x = x - column
        fraction_y = y - row
        top = field[row, column] * (1 - fraction_x) + field[row, column + 1] * fraction_x
        bottom = field[row + 1, column] * (1 - fraction_x) + field[row + 1, column + 1] * fraction_x
        return top * (1 - fraction_y) + bottom * fraction_y

    def advance(self, ux, uy, solid, substeps=3):
        """One frame of travel by RK2 midpoint, then age, then respawn the dead."""
        dt = self.steps_per_frame / float(substeps)
        for _ in range(substeps):
            mid_x = self.x + 0.5 * dt * self.sample(ux, self.x, self.y)
            mid_y = self.y + 0.5 * dt * self.sample(uy, self.x, self.y)
            self.x = self.x + dt * self.sample(ux, mid_x, mid_y)
            self.y = self.y + dt * self.sample(uy, mid_x, mid_y)

        self.age += 1.0 / NOMINAL_FPS
        inside_body = self.sample(solid.astype(np.float32), self.x, self.y) > 0.5
        dead = ((self.age > self.life) | inside_body
                | (self.x < 0) | (self.x > self.nx - 1)
                | (self.y < 0) | (self.y > self.ny - 1))
        dead_count = int(dead.sum())
        if dead_count:
            self.x[dead] = self.random.random(dead_count, dtype=np.float32) * (self.nx - 1)
            self.y[dead] = self.random.random(dead_count, dtype=np.float32) * (self.ny - 1)
            self.life[dead] = self.new_life(dead_count)
            self.age[dead] = 0.0

    def brightness(self, ux, uy, freestream):
        """Fade in over the first 15% of life and out over the last 30%; faster is brighter."""
        fraction = np.clip(self.age / np.maximum(self.life, 1e-6), 0.0, 1.0)
        fade = np.clip(fraction / 0.15, 0, 1) * np.clip((1.0 - fraction) / 0.30, 0, 1)
        speed = np.sqrt(self.sample(ux, self.x, self.y) ** 2 + self.sample(uy, self.x, self.y) ** 2)
        relative = np.clip(speed / max(freestream, 1e-6), 0.0, 1.8)
        return fade * (1.0 + STREAK_SPEED_GAIN * (relative - 1.0))

    def draw(self, ux, uy, freestream, scale, width, height):
        """Integrate each tail backward and splat it into an (height, width) float layer."""
        base = self.brightness(ux, uy, freestream).astype(np.float32)
        dt_back = -STREAK_SPAN_STEPS / STREAK_TAIL_SAMPLES
        if splat_tails_compiled is not None:
            layer = np.zeros((height, width), dtype=np.float32)
            splat_tails_compiled(self.x, self.y, base, ux, uy, float(scale), dt_back,
                                 STREAK_TAIL_SAMPLES, layer)
        else:
            layer = self.splat_tails_numpy(base, ux, uy, scale, dt_back, width, height)
        # truncate=2 keeps the kernel at 5 taps; at sigma 0.85 the default 4 adds nothing visible
        return gaussian_filter(layer, sigma=STREAK_BLUR_PX, truncate=2.0)

    def splat_tails_numpy(self, base, ux, uy, scale, dt_back, width, height):
        """The same tail walk as `splat_tails_python`, vectorised over tracers, for machines
        without numba. All tail samples are gathered first and written with ONE bincount;
        writing per sample step would be 22 full-frame accumulations a frame."""
        x = self.x.copy()
        y = self.y.copy()
        all_indices = []
        all_weights = []
        for sample_index in range(STREAK_TAIL_SAMPLES):
            weight = base * (1.0 - sample_index / float(STREAK_TAIL_SAMPLES)) ** 1.35
            screen_x = (x + 0.5) * scale - 0.5
            screen_y = (y + 0.5) * scale - 0.5
            on_screen = ((screen_x >= 0) & (screen_x <= width - 2)
                         & (screen_y >= 0) & (screen_y <= height - 2))
            screen_x = screen_x[on_screen]
            screen_y = screen_y[on_screen]
            weight = weight[on_screen]
            column = screen_x.astype(np.int32)
            row = screen_y.astype(np.int32)
            fraction_x = screen_x - column
            fraction_y = screen_y - row
            corner = row * width + column
            all_indices.extend([corner, corner + 1, corner + width, corner + width + 1])
            all_weights.extend([weight * (1 - fraction_x) * (1 - fraction_y),
                                weight * fraction_x * (1 - fraction_y),
                                weight * (1 - fraction_x) * fraction_y,
                                weight * fraction_x * fraction_y])
            velocity_x = self.sample(ux, x, y)
            velocity_y = self.sample(uy, x, y)
            x = x + dt_back * velocity_x
            y = y + dt_back * velocity_y
        layer = np.bincount(np.concatenate(all_indices), weights=np.concatenate(all_weights),
                            minlength=width * height)
        return layer.reshape(height, width).astype(np.float32)


class Look:
    """Turns solver state into a screen frame at `scale` x the lattice size."""

    def __init__(self, nx, ny, scale, steps_per_frame):
        self.nx = int(nx)
        self.ny = int(ny)
        # Lattice -> screen uses the PIXEL-CENTRE convention, screen = (cell + 0.5) * scale - 0.5,
        # because that is what Pillow's bilinear resize of the field does. The streak splatter
        # and the body outline use the same mapping, so the tails sit on the field they were
        # traced through and the rim sits on the mask the solver sees.
        self.scale = int(scale)
        self.width = self.nx * self.scale
        self.height = self.ny * self.scale
        area_ratio = (self.width * self.height) / (1080.0 * 1920.0)
        self.streaks = Streaks(int(STREAKS_AT_1080x1920 * area_ratio), nx, ny, steps_per_frame)

    def field01(self, sim, field):
        """The chosen field normalised into [0, 1] with a FIXED scale, plus its LUT name."""
        if field == "vorticity":
            values = asnumpy(sim.vorticity())
            return 0.5 + 0.5 * np.clip(values / (sim.u0 * 0.55), -1.0, 1.0), "vort"
        if field == "pressure":
            values = asnumpy(sim.pressure())
            return 0.5 + 0.5 * np.clip(values / (sim.u0 * sim.u0 * 1.6), -1.0, 1.0), "jet"
        if field == "q":
            values = asnumpy(sim.q_criterion())
            return np.clip(values / (0.12 * sim.u0 * sim.u0), 0.0, 1.0), "inferno"
        values = asnumpy(sim.speed())
        normalised = np.clip(values / (sim.u0 * SPEED_FULL_SCALE), 0.0, 1.0)
        return normalised ** FIELD_GAMMA, "jet"

    def advance(self, sim):
        ux = np.asarray(asnumpy(sim.ux), dtype=np.float32)
        uy = np.asarray(asnumpy(sim.uy), dtype=np.float32)
        solid = np.asarray(asnumpy(sim.solid), dtype=bool)
        self.streaks.advance(ux, uy, solid)

    def frame(self, sim, field, polygons):
        """Field -> streaks -> body. `polygons` are the bodies in lattice coordinates."""
        normalised, lut_name = self.field01(sim, field)
        # Colour at lattice resolution, then a bilinear resize to the screen. Interpolating
        # colours rather than the scalar differs only within one cell of a hue boundary, and it
        # is a C resize instead of a full-frame map_coordinates (about 3 ms against 19 ms).
        index = np.clip(normalised * 255.0, 0, 255).astype(np.int32)
        small = Image.fromarray(LUTS[lut_name][index])
        big = small.resize((self.width, self.height), Image.BILINEAR)
        ux = np.asarray(asnumpy(sim.ux), dtype=np.float32)
        uy = np.asarray(asnumpy(sim.uy), dtype=np.float32)
        layer = self.streaks.draw(ux, uy, sim.u0, self.scale, self.width, self.height)
        # Additive white, saturating. spectrometry caps the streak layer at 1.6 x full white,
        # but on an ADD that cap and a cap at 1.0 give identical pixels - any channel plus 255
        # is already 255 - so the layer fits in uint8 and Pillow's C saturating add does it.
        brightness = np.minimum(layer * (STREAK_GAIN * 255.0), 255.0).astype(np.uint8)
        white = Image.fromarray(brightness).convert("RGB")
        rgb = np.asarray(ImageChops.add(big, white))
        return self.draw_bodies(rgb, polygons)

    def draw_bodies(self, rgb, polygons):
        """Filled body + bright rim, drawn at 2x over the bodies' bounding box and box-filtered
        down, so the edge is anti-aliased at screen resolution instead of stair-stepped."""
        supersample = 2
        screen = []
        for polygon in polygons:
            points = np.empty_like(polygon)
            points[:, 0] = (polygon[:, 0] + 0.5) * self.scale - 0.5
            points[:, 1] = (polygon[:, 1] + 0.5) * self.scale - 0.5
            screen.append(points)
        all_points = np.concatenate(screen)
        pad = BODY_EDGE_PX + 3
        left = max(0, int(np.floor(all_points[:, 0].min())) - pad)
        top = max(0, int(np.floor(all_points[:, 1].min())) - pad)
        right = min(self.width, int(np.ceil(all_points[:, 0].max())) + pad)
        bottom = min(self.height, int(np.ceil(all_points[:, 1].max())) + pad)
        box_width = right - left
        box_height = bottom - top

        fill = Image.new("L", (box_width * supersample, box_height * supersample), 0)
        edge = Image.new("L", (box_width * supersample, box_height * supersample), 0)
        fill_draw = ImageDraw.Draw(fill)
        edge_draw = ImageDraw.Draw(edge)
        for points in screen:
            vertices = []
            for point in points:
                vertices.append(((float(point[0]) - left) * supersample,
                                 (float(point[1]) - top) * supersample))
            fill_draw.polygon(vertices, fill=255)
            # No rounded joints: at ~320 vertices per section the bends between segments are
            # under a degree, and joint="curve" draws a pie slice at every one of them - it was
            # the single largest cost of drawing the body.
            edge_draw.line(vertices + [vertices[0]], fill=255, width=BODY_EDGE_PX * supersample)
        fill_alpha = np.asarray(fill.resize((box_width, box_height), Image.BOX),
                                np.float32)[:, :, None] / 255.0
        edge_alpha = np.asarray(edge.resize((box_width, box_height), Image.BOX),
                                np.float32)[:, :, None] / 255.0
        region = rgb[top:bottom, left:right].astype(np.float32)
        region = region * (1 - fill_alpha) + np.asarray(BODY_FILL, np.float32) * fill_alpha
        region = region * (1 - edge_alpha) + np.asarray(BODY_EDGE, np.float32) * edge_alpha
        out = rgb.copy()
        out[top:bottom, left:right] = np.clip(region, 0, 255).astype(np.uint8)
        return out


_FONTS = {}


def cached_font(size):
    """render._font searches the font directories on every call; a frame draws three panels."""
    if size not in _FONTS:
        _FONTS[size] = _font(size)
    return _FONTS[size]


def composite_patch(rgb, left, top, right, bottom, paint):
    """Run `paint(draw, offset_x, offset_y)` on an RGBA overlay of just this patch and blend it in.

    Working on the patch instead of the whole frame is the difference between ~1 ms and ~14 ms
    per panel at 1280x640.
    """
    left = max(0, left)
    top = max(0, top)
    right = min(rgb.shape[1], right)
    bottom = min(rgb.shape[0], bottom)
    patch = Image.fromarray(np.ascontiguousarray(rgb[top:bottom, left:right])).convert("RGBA")
    overlay = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    paint(ImageDraw.Draw(overlay), left, top)
    out = rgb.copy()
    out[top:bottom, left:right] = np.asarray(Image.alpha_composite(patch, overlay).convert("RGB"))
    return out


def draw_panel(rgb, lines, corner="tl", size=15, pad=10):
    """A few lines of text on a translucent dark panel with a faint rim."""
    if not lines:
        return rgb
    font = cached_font(size)
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    line_height = int(size * 1.4)
    text_width = 0
    for line in lines:
        text_width = max(text_width, int(measure.textlength(line, font=font)))
    panel_width = text_width + 2 * pad
    panel_height = line_height * len(lines) + 2 * pad
    frame_height, frame_width = rgb.shape[:2]
    if corner in ("tr", "br"):
        x0 = frame_width - panel_width - pad
    else:
        x0 = pad
    if corner in ("bl", "br"):
        y0 = frame_height - panel_height - pad
    else:
        y0 = pad

    def paint(draw, offset_x, offset_y):
        left = x0 - offset_x
        top = y0 - offset_y
        draw.rounded_rectangle([left, top, left + panel_width, top + panel_height], radius=6,
                               fill=(6, 8, 12, 165), outline=(210, 220, 226, 70), width=1)
        for index, line in enumerate(lines):
            draw.text((left + pad, top + pad + index * line_height), line, font=font,
                      fill=(236, 240, 238, 255))

    return composite_patch(rgb, x0, y0, x0 + panel_width + 1, y0 + panel_height + 1, paint)


def draw_speed_legend(rgb, full_scale_label, size=13, pad=10):
    """Vertical jet colour bar on the right edge: bottom = still air, top = full scale.

    The ramp is drawn with the same gamma the field uses, so a colour on the bar is the colour
    the field takes at that speed, and the labels stay linear in speed.
    """
    font = cached_font(size)
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    frame_height, frame_width = rgb.shape[:2]
    bar_width = 14
    bar_height = int(frame_height * 0.42)
    label_width = int(measure.textlength(full_scale_label, font=font))
    panel_width = bar_width + label_width + 3 * pad
    panel_height = bar_height + 2 * pad
    x0 = frame_width - panel_width - pad
    y0 = (frame_height - bar_height) // 2 - pad
    lut = LUTS["jet"]

    def paint(draw, offset_x, offset_y):
        left = x0 - offset_x
        top = y0 - offset_y
        draw.rounded_rectangle([left, top, left + panel_width, top + panel_height], radius=6,
                               fill=(6, 8, 12, 150), outline=(210, 220, 226, 70), width=1)
        bar_x = left + pad
        bar_top = top + pad
        for row in range(bar_height):
            speed_fraction = 1.0 - row / float(bar_height - 1)
            colour = lut[int(round((speed_fraction ** FIELD_GAMMA) * 255))]
            draw.line([(bar_x, bar_top + row), (bar_x + bar_width - 1, bar_top + row)],
                      fill=(int(colour[0]), int(colour[1]), int(colour[2]), 255))
        draw.rectangle([bar_x, bar_top, bar_x + bar_width - 1, bar_top + bar_height - 1],
                       outline=(226, 232, 230, 120), width=1)
        text_x = bar_x + bar_width + pad
        draw.text((text_x, bar_top - 2), full_scale_label, font=font, fill=(236, 240, 238, 255))
        draw.text((text_x, bar_top + bar_height - size), "0", font=font,
                  fill=(236, 240, 238, 255))

    return composite_patch(rgb, x0, y0, x0 + panel_width + 1, y0 + panel_height + 1, paint)
