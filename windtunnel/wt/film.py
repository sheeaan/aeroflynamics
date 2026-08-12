"""film.py - the optical chain: CRT phosphor, lens, and camera, applied in that order.

This is a post-process on finished RGB frames and knows nothing about fluid. It can be pointed
at any video.

THE ORDER IS THE WHOLE THING
----------------------------
These stages are not a bag of effects to be toggled in any arrangement. They model a physical
path - light leaves a phosphor, passes through a lens, lands on a sensor - and each stage can
only see what the ones before it produced. Applying them out of order produces artefacts that
look like bugs in the renderer:

  1. PHOSPHOR   grain and bloom belong to the emitting surface, so they must be modulated by
                LOCAL BRIGHTNESS. Grain applied uniformly (i.e. after the lens) appears in the
                black surround, where a real CRT emits nothing and therefore cannot be grainy.
  2. DUST       sits on the glass, between phosphor and lens. It is lit BY the trace, so it
                must multiply light that already exists rather than be added on top - dust
                added after the lens glows in the dark, which reads as snow, not dirt.
  3. LENS       barrel distortion, chromatic aberration, corner defocus. Geometry, so it must
                come after everything painted on the glass and before anything the sensor does.
  4. CAMERA     exposure flicker, mains hum band, sensor noise. These are properties of the
                capture, so they apply to the whole frame uniformly and go last.

ON BLOOM, which is the one that will bite you
----------------------------------------------
Bloom is a wide blur, and a blur averages many elements, so its output changes SLOWLY even
when the underlying image is switching hard. If the source is doing anything that depends on
fast per-channel timing - an RGB channel-delay colour scheme, a strobing trace - the bloom of
it converges toward the mean of all channels, and the mean of all channels is white. So bloom
quietly washes exactly the effect you were trying to show. Keep `bloom` low, or off, whenever
the colour is coming from timing rather than from a palette.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


class Film:
    """The optical chain. Build once per clip - the expensive maps are cached - then call
    `apply(rgb, t)` per frame.

    Every strength is 0..1 and 0 disables that stage cleanly.
    """

    def __init__(self, w, h, seed=17,
                 grain=0.10, bloom=0.06, phosphor_smear=0.35,
                 dust=0.05, scratches=0.02,
                 barrel=0.045, chroma=0.0025, corner_blur=0.55, vignette=0.30,
                 scanlines=0.10, flicker=0.02, hum=0.025, hum_hz=0.7, noise=0.012):
        self.w, self.h = int(w), int(h)
        self.rng = np.random.default_rng(seed)
        self.grain = float(grain)
        self.bloom = float(bloom)
        self.phosphor_smear = float(phosphor_smear)
        self.dust_amt = float(dust)
        self.scratch_amt = float(scratches)
        self.barrel = float(barrel)
        self.chroma = float(chroma)
        self.corner_blur = float(corner_blur)
        self.vignette = float(vignette)
        self.scanlines = float(scanlines)
        self.flicker = float(flicker)
        self.hum = float(hum)
        self.hum_hz = float(hum_hz)
        self.noise = float(noise)

        self._prev = None                       # for phosphor persistence
        self._build_maps()
        self._build_dirt()

    # -- cached, geometry-only -------------------------------------------------------------
    def _build_maps(self):
        """Barrel-distortion sampling coordinates, one set per channel.

        Computed once: the lens does not move. Chromatic aberration is the SAME distortion at
        three slightly different strengths, which is what a real lens does - it is dispersion,
        one refractive index per wavelength, not three independent warps. Faking it by
        translating the channels gives a flat colour fringe everywhere instead of one that
        grows toward the corners.
        """
        h, w = self.h, self.w
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        s = max(cx, cy)
        nx = (xx - cx) / s
        ny = (yy - cy) / s
        r2 = nx * nx + ny * ny
        self._r = np.sqrt(r2)
        self._maps = []
        for ch in range(3):
            k = self.barrel * (1.0 + self.chroma / max(self.barrel, _EPS) * (ch - 1))
            f = 1.0 + k * r2
            self._maps.append((np.clip(cy + ny * f * s, 0, h - 1.001).astype(np.float32),
                               np.clip(cx + nx * f * s, 0, w - 1.001).astype(np.float32)))
        # radial masks
        self._vign = np.clip(1.0 - self.vignette * self._r ** 2.2, 0.0, 1.0).astype(np.float32)
        self._corner = np.clip((self._r - 0.45) / 0.55, 0.0, 1.0).astype(np.float32)

    def _build_dirt(self):
        """Static dust specks and scratches on the glass. Fixed for the whole clip - dirt that
        resamples every frame is snow, and snow reads as a broken sensor rather than a dirty
        lens."""
        h, w = self.h, self.w
        d = np.zeros((h, w), np.float32)
        n_spots = int(0.00035 * h * w * (self.dust_amt > 0))
        for _ in range(n_spots):
            y = self.rng.integers(0, h)
            x = self.rng.integers(0, w)
            r = self.rng.integers(1, 3)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            d[y0:y1, x0:x1] += float(self.rng.uniform(0.3, 1.0))
        n_scr = int(6 * (self.scratch_amt > 0))
        for _ in range(n_scr):
            x = self.rng.integers(0, w)
            length = self.rng.integers(h // 4, h)
            y0 = self.rng.integers(0, max(1, h - length))
            drift = self.rng.uniform(-0.05, 0.05)
            for k in range(length):
                xi = int(x + drift * k)
                if 0 <= xi < w:
                    d[y0 + k, xi] += 0.5
        self._dirt = np.clip(d, 0.0, 1.5)

    # -- per frame -----------------------------------------------------------------------------
    @staticmethod
    def _blur(a, sigma):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma, mode="nearest")

    def _warp(self, img):
        from scipy.ndimage import map_coordinates
        out = np.empty_like(img)
        for ch in range(3):
            yy, xx = self._maps[ch]
            out[..., ch] = map_coordinates(img[..., ch], [yy, xx], order=1, mode="nearest")
        return out

    def apply(self, rgb, t=0.0):
        """One frame through the chain. `t` is seconds, used by the time-varying stages."""
        img = np.asarray(rgb, dtype=np.float32) / 255.0
        if img.shape[:2] != (self.h, self.w):
            raise ValueError(f"frame is {img.shape[1]}x{img.shape[0]}, "
                             f"filter built for {self.w}x{self.h}")
        lum = img.mean(axis=2)

        # --- 1. phosphor ---------------------------------------------------------------------
        if self.phosphor_smear > 0.0 and self._prev is not None:
            # persistence: the trace decays rather than vanishing between frames
            img = np.maximum(img, self._prev * self.phosphor_smear)
        if self.grain > 0.0:
            # modulated by local brightness: a phosphor that is not lit cannot be grainy
            g = self.rng.standard_normal((self.h, self.w)).astype(np.float32)
            img = img * (1.0 + self.grain * g[..., None] * lum[..., None])
        if self.bloom > 0.0:
            hi = np.clip(lum - 0.55, 0.0, None)
            img = img + self.bloom * self._blur(hi, max(self.h, self.w) * 0.012)[..., None]
        self._prev = np.clip(img, 0.0, 4.0)

        # --- 2. dust on the glass, LIT BY the trace ------------------------------------------
        if self.dust_amt > 0.0 or self.scratch_amt > 0.0:
            local = self._blur(img.mean(axis=2), 3.0)
            img = img + (self._dirt * local * self.dust_amt * 3.0)[..., None]

        # --- 3. lens ---------------------------------------------------------------------------
        if self.barrel != 0.0 or self.chroma != 0.0:
            img = self._warp(img)
        if self.corner_blur > 0.0:
            soft = self._blur(img, (1.4, 1.4, 0.0))
            m = (self._corner * self.corner_blur)[..., None]
            img = img * (1.0 - m) + soft * m
        img = img * self._vign[..., None]

        # --- 4. camera ---------------------------------------------------------------------------
        if self.scanlines > 0.0:
            rows = np.arange(self.h, dtype=np.float32)
            img = img * (1.0 - self.scanlines * 0.5
                         * (1.0 + np.cos(np.pi * rows))[:, None, None])
        if self.flicker > 0.0:
            img = img * (1.0 + self.flicker * float(self.rng.standard_normal()))
        if self.hum > 0.0:
            # a mains-beat band drifting slowly down the frame
            rows = np.arange(self.h, dtype=np.float32) / self.h
            band = np.cos(2 * np.pi * (rows - self.hum_hz * t))
            img = img * (1.0 + self.hum * band[:, None, None])
        if self.noise > 0.0:
            img = img + self.noise * self.rng.standard_normal(img.shape).astype(np.float32)

        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def describe(self):
        on = [n for n, v in (("grain", self.grain), ("bloom", self.bloom),
                             ("persist", self.phosphor_smear), ("dust", self.dust_amt),
                             ("barrel", self.barrel), ("chroma", self.chroma),
                             ("corner", self.corner_blur), ("vignette", self.vignette),
                             ("scanlines", self.scanlines), ("flicker", self.flicker),
                             ("hum", self.hum), ("noise", self.noise)) if v]
        return f"film {self.w}x{self.h}: " + ", ".join(on)


PRESETS = {
    "off": dict(grain=0, bloom=0, phosphor_smear=0, dust=0, scratches=0, barrel=0,
                chroma=0, corner_blur=0, vignette=0, scanlines=0, flicker=0, hum=0, noise=0),
    "clean": dict(grain=0.03, bloom=0.04, phosphor_smear=0.0, dust=0.0, scratches=0.0,
                  barrel=0.02, chroma=0.0012, corner_blur=0.25, vignette=0.18,
                  scanlines=0.0, flicker=0.006, hum=0.0, noise=0.004),
    "crt": dict(grain=0.10, bloom=0.06, phosphor_smear=0.35, dust=0.05, scratches=0.02,
                barrel=0.045, chroma=0.0025, corner_blur=0.55, vignette=0.30,
                scanlines=0.10, flicker=0.02, hum=0.025, noise=0.012),
    "wrecked": dict(grain=0.20, bloom=0.10, phosphor_smear=0.45, dust=0.12, scratches=0.06,
                    barrel=0.075, chroma=0.005, corner_blur=0.8, vignette=0.45,
                    scanlines=0.18, flicker=0.05, hum=0.05, noise=0.03),
}


def make(w, h, preset="crt", **over):
    """Build a Film from a named preset, with individual overrides."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    cfg = dict(PRESETS[preset])
    cfg.update(over)
    return Film(w, h, **cfg)
