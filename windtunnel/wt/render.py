"""render.py - frames out.

Two responsibilities, deliberately separated:

  `VideoWriter`  swallows frames and produces a file. It prefers ffmpeg (piped raw RGB, so
                 there is no PNG round-trip) and falls back to an animated GIF through Pillow
                 when ffmpeg is not on PATH, so a fresh checkout produces a watchable result
                 with nothing installed beyond the three Python packages.

  `render_scene` drives a solver through a scene: advance, colour, composite, write, and
                 watch for divergence.

TIMEBASE. Frames are driven from real seconds (t = i / fps) and never from the step counter,
so a half-resolution preview at 24 fps and a final at 60 fps are the SAME animation sampled
differently. Anything that reads the step counter instead silently becomes a different clip
when you change the frame rate, which makes previewing worthless.

APPARENT SPEED is (lattice velocity) x (steps per frame). To make a clip look faster, raise
`steps` - never `u0`. Raising u0 raises the compressibility error as its square and walks the
solve toward the lattice sound speed; raising steps just samples the same physics less often.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import colormap as cm


def ffmpeg_exe():
    """Path to an ffmpeg binary, or None.

    Prefers one on PATH, then falls back to the static binary that `imageio-ffmpeg` ships
    inside site-packages. The fallback matters more than it looks: ffmpeg is a system package
    rather than a Python one, and without it the writer drops to animated GIF - which at
    1080x1920 and 300 frames is hundreds of megabytes of 256-colour video.
    """
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def have_ffmpeg():
    return ffmpeg_exe() is not None


class VideoWriter:
    """Frames in, one file out.

    Writes to a `.part` temporary and renames only on a clean close, so a file that exists is
    always a complete, playable file - an interrupted render leaves no half-written .mp4 to be
    mistaken for a finished one.
    """

    def __init__(self, path, fps=30, quality=18, prefer="auto"):
        self.path = str(path)
        self.fps = int(fps)
        self.quality = int(quality)
        self.proc = None
        self.frames = []
        self.size = None
        self.count = 0
        root, ext = os.path.splitext(self.path)
        use_ffmpeg = have_ffmpeg() if prefer == "auto" else (prefer == "ffmpeg")
        if use_ffmpeg and ext.lower() in (".mp4", ".mov", ".webm", ".mkv"):
            self.mode = "ffmpeg"
            self.tmp = root + ".part" + ext
        else:
            self.mode = "gif"
            self.path = root + ".gif"
            self.tmp = root + ".part.gif"
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)

    def _open_ffmpeg(self, w, h):
        cmd = [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-preset", "medium",
            "-crf", str(self.quality),
            # yuv420p is what makes the file playable in browsers and QuickTime rather than
            # only in VLC; it needs even dimensions, which `render_scene` guarantees.
            "-pix_fmt", "yuv420p",
            self.tmp,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def add(self, rgb):
        a = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        h, w = a.shape[:2]
        if self.size is None:
            self.size = (w, h)
            if self.mode == "ffmpeg":
                self._open_ffmpeg(w, h)
        elif self.size != (w, h):
            raise ValueError(f"frame size changed {self.size} -> {(w, h)}")
        if self.mode == "ffmpeg":
            self.proc.stdin.write(a.tobytes())
        else:
            self.frames.append(Image.fromarray(a))
        self.count += 1

    def close(self):
        if self.count == 0:
            return None
        if self.mode == "ffmpeg":
            self.proc.stdin.close()
            rc = self.proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited {rc}")
        else:
            # Pillow's GIF encoder quantises to 256 colours per frame; the adaptive palette is
            # noticeably better than the web-safe default on a continuous colour map.
            head, *rest = self.frames
            head.save(self.tmp, save_all=True, append_images=rest,
                      duration=int(1000 / max(self.fps, 1)), loop=0, optimize=True)
        os.replace(self.tmp, self.path)
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        elif self.mode == "ffmpeg" and self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


# --- HUD ------------------------------------------------------------------------------
def _font(size):
    try:
        return ImageFont.truetype("consola.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:      # Pillow < 10
            return ImageFont.load_default()


def draw_hud(rgb, lines, scale=1, corner="tl", pad=8):
    """Burn a few lines of monospace text into a frame. `lines` is a list of strings."""
    if not lines:
        return rgb
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    size = max(11, 11 * int(scale))
    f = _font(size)
    step = int(size * 1.45)
    h = img.height
    w = img.width
    y = pad if corner in ("tl", "tr") else h - pad - step * len(lines)
    for i, ln in enumerate(lines):
        if corner in ("tr", "br"):
            tw = d.textlength(ln, font=f)
            x = w - pad - tw
        else:
            x = pad
        # a 1px dark offset instead of a box: legible over both the bright and dark ends of
        # every colour map, without covering the flow
        d.text((x + 1, y + i * step + 1), ln, font=f, fill=(0, 0, 0))
        d.text((x, y + i * step), ln, font=f, fill=(238, 241, 246))
    return np.asarray(img)


ORIENTATIONS = ("h", "vl", "vr")


def orient(a, mode="h"):
    """Map lattice axes onto frame axes. Works on (ny, nx) scalars and (ny, nx, 3) frames.

    The solver always computes with the flow running along +x; which way that points on screen
    is chosen here, at render time, and is never baked into the physics.

      h    horizontal, flow left to right - the identity
      vl   rotate the picture counter-clockwise; flow runs bottom to top
      vr   rotate the picture clockwise; flow runs top to bottom

    ONLY ROTATIONS ARE ALLOWED HERE, never a flip, and that restriction is the whole point.
    A rotation is orientation-preserving, so a counter-clockwise vortex stays
    counter-clockwise. A mirror is not: it silently reverses the sense of every vortex in the
    frame, and the result still looks like plausible fluid, so nothing about the picture would
    tell you. `np.flipud` is the tempting one-character way to get a vertical clip pointing the
    other way - use "vl" instead of "vr" and rotate.

    tools/validate.py has a regression case that seeds a vortex of known sign and checks it
    survives every mapping here.
    """
    if mode == "h":
        return a
    if mode == "vl":
        return np.rot90(a, 1)
    if mode == "vr":
        return np.rot90(a, 3)
    raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {mode!r}")


def upscale(rgb, factor):
    """Nearest-neighbour upscale. Nearest, not bilinear, on purpose: the lattice IS the
    resolution, and smoothing it invents detail the solver never computed."""
    k = int(factor)
    if k <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)


# --- the driver -------------------------------------------------------------------------
def render_scene(sim, scene, out, frames=240, fps=30, steps=18, settle=400,
                 field="vorticity", cmap=None, vlim=None, gamma=1.0,
                 tracers=0, scale=1, hud=True, quiet=False, orientation="h",
                 dye=None, film=None):
    """Drive `sim` through `scene` and write a clip.

    `scene` is an object with:
        update(sim, t, frame)   -> called once per frame BEFORE the steps; sets geometry
        mask                    -> current boolean solid mask (set by update)
        hud(sim)                -> optional list[str] of overlay lines
        ref_len                 -> optional reference length for the coefficients

    Returns a dict of the run's diagnostics.
    """
    from .streaks import Streaks

    # Colour limits are FIXED for the whole clip and scale with u0, never auto-fitted per
    # frame. Auto-fitting is the obvious thing and it is wrong twice: the picture flickers as
    # the running max wanders, and - worse - it silently renormalises, so a wake that is
    # genuinely weakening looks exactly as bright as one that is not. A fixed scale means two
    # frames are comparable, which is the entire point of drawing a field rather than a mood.
    # Each is expressed against the natural scale of its quantity at this freestream:
    #   vorticity ~ u0 / (a few cells)   pressure ~ 0.5 u0^2 (dynamic)   Q ~ (u0 / delta)^2
    field_cfg = {
        "vorticity": dict(cmap="coolwarm", symmetric=True, lim=lambda s: 0.80 * s.u0),
        "speed": dict(cmap="inferno", symmetric=False, lim=lambda s: 2.20 * s.u0),
        "pressure": dict(cmap="coolwarm", symmetric=True, lim=lambda s: 0.70 * s.u0 ** 2),
        "q": dict(cmap="magma", symmetric=False, lim=lambda s: 0.12 * s.u0 ** 2),
        # compressible only; `mach` is absolute so its scale does NOT follow u0 - the
        # sonic line at M = 1 has to land in the same colour in every clip or the field
        # stops being readable across scenes
        "mach": dict(cmap="inferno", symmetric=False, lim=lambda s: max(1.6, 1.25 * s.u0)),
        "schlieren": dict(cmap="gray", symmetric=False, lim=lambda s: 0.55),
    }
    if field not in field_cfg:
        raise ValueError(f"field must be one of {sorted(field_cfg)}")
    fc = field_cfg[field]
    cmap = cmap or fc["cmap"]

    getters = {
        "vorticity": lambda: sim.vorticity(),
        "speed": lambda: sim.speed(),
        "pressure": lambda: sim.pressure(),
        "q": lambda: sim.q_criterion(),
        "mach": lambda: sim.mach(),
        "schlieren": lambda: sim.schlieren(),
    }
    if not hasattr(sim, {"mach": "mach", "schlieren": "schlieren"}.get(field, "vorticity")):
        raise ValueError(f"field {field!r} needs the compressible solver "
                         f"(set scene.solver = 'cns')")
    getter = getters[field]

    lim = float(vlim) if vlim is not None else float(fc["lim"](sim))

    # settle at frame-0 geometry so the clip does not open on a starting transient
    scene.update(sim, 0.0, 0)
    sim.set_solid(scene.mask)
    if settle:
        if not quiet:
            print(f"  settling {settle} steps ...", flush=True)
        sim.run(int(settle))

    tr = Streaks(int(tracers), sim.nx, sim.ny) if tracers else None
    diag = {"cd": [], "cl": [], "health": [], "t": []}
    t_start = time.time()

    with VideoWriter(out, fps=fps) as vw:
        for i in range(int(frames)):
            t = i / float(fps)
            scene.update(sim, t, i)
            sim.set_solid(scene.mask)
            sim.run(int(steps))
            if tr is not None:
                tr.advance(sim, steps=max(1, int(steps) // 3))
            if dye is not None:
                dye.step(sim.ux, sim.uy, sim.solid, substeps=int(steps))

            h = sim.health()
            if not np.isfinite(h) or h > sim.health_limit:
                raise RuntimeError(
                    f"diverged at frame {i}: max|u| = {h:.4f} > {sim.health_limit}. "
                    f"Lower u0, lower re, or raise the resolution.")

            rgb = cm.colorize(getter(), cmap=cmap,
                              symmetric=fc["symmetric"],
                              vmin=None if fc["symmetric"] else 0.0,
                              vmax=lim, gamma=gamma)
            # Dye goes on BEFORE the body is painted, so a filament passing behind the
            # silhouette is occluded by it rather than drawn over the top.
            if dye is not None:
                rgb = dye.composite(rgb)
            if tr is not None:
                rgb = tr.draw(rgb)
            rgb = cm.overlay_mask(rgb, sim.solid)
            rgb = cm.outline_mask(rgb, sim.solid)
            # orientation BEFORE upscale (cheaper) and before the HUD, so text stays upright
            rgb = np.ascontiguousarray(orient(rgb, orientation))
            rgb = upscale(rgb, scale)

            # Not every solver has a validated force integral - CNS deliberately raises rather
            # than return a plausible-looking wrong number - so a missing Cl/Cd is a normal
            # state here, not an error.
            try:
                cd, cl = sim.coefficients(ref_len=getattr(scene, "ref_len", None))
            except NotImplementedError:
                cd = cl = float("nan")
            diag["cd"].append(cd)
            diag["cl"].append(cl)
            diag["health"].append(h)
            diag["t"].append(t)

            # The film chain models a camera looking at the screen, so it goes LAST - after
            # the HUD, which is also on that screen and should pick up the same lens
            # distortion and grain. Burning text in after the filter makes it float above the
            # image, which is the giveaway that the effect is a filter rather than a camera.
            if hud:
                lines = list(getattr(scene, "hud", lambda s: [])(sim))
                if np.isfinite(cl):
                    lines.append(f"Cl {cl:+.3f}   Cd {cd:+.3f}")
                rgb = draw_hud(rgb, lines, scale=scale)
            if film is not None:
                rgb = film.apply(rgb, t=t)

            vw.add(rgb)
            if not quiet and (i % 20 == 0 or i == frames - 1):
                el = time.time() - t_start
                print(f"  frame {i + 1:4d}/{frames}  max|u|={h:.4f}  "
                      f"Cl={cl:+.3f} Cd={cd:+.3f}  [{el:5.1f}s]", flush=True)
        path = vw.path
        mode = vw.mode

    diag["path"] = path
    diag["mode"] = mode
    diag["seconds"] = time.time() - t_start
    if not quiet:
        print(f"  wrote {path}  ({mode}, {frames} frames, {diag['seconds']:.1f}s)")
        if mode == "gif":
            print("  (ffmpeg not found -> GIF. Install ffmpeg and re-run for mp4.)",
                  file=sys.stderr)
    return diag
