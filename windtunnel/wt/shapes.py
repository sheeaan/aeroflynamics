"""shapes.py - bodies as closed polygons in unit-chord space.

The solver only ever sees the boolean mask that `rasterize()` produces, so adding a new body
means adding a function that returns an (N, 2) array of points. There is no mesh, no
signed-distance field, and no special case anywhere in the physics - which is the property
that lets a body rotate every single frame for free.

Profiles are generated with x from 0 (leading edge) to 1 (trailing edge) and y up, then
`place()`d into lattice coordinates: scaled by the chord, rotated about the quarter-chord,
and translated.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# --- profile generators, unit chord -------------------------------------------------


def naca4(code="2412", n=160, closed_te=True, thickness=None):
    """NACA 4-digit section. `code` = MPXX: M = max camber (% chord), P = its position
    (tenths of chord), XX = thickness (% chord).

    `thickness`, a fraction of chord, overrides XX when given. It exists for a CONTINUOUS
    thickness: XX is whole percent, and a slider stepping 1% at a time moves the surface of a
    120-cell chord by more than a cell per step. Camber and chord are untouched by it.

    Points use COSINE SPACING, which clusters them toward the leading edge. That is not
    cosmetic: the curvature there is extreme, and uniform spacing either under-resolves the
    nose (so the suction peak is wrong) or wastes points along the nearly-flat aft half.
    """
    code = str(code).zfill(4)
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    if thickness is None:
        t = int(code[2:]) / 100.0
    else:
        t = float(thickness)

    beta = np.linspace(0.0, np.pi, int(n))
    x = 0.5 * (1.0 - np.cos(beta))

    # The last coefficient differs between the open- and closed-trailing-edge variants of the
    # standard thickness polynomial. Closed is what we want: an open TE leaves a one-cell gap
    # that the rasteriser may or may not bridge depending on the angle, so the body would
    # flicker between sealed and leaking as it rotates.
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 + a4 * x ** 4)

    if m > 0.0 and p > 0.0:
        yc = np.where(x < p,
                      m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p,
                       2 * m / p ** 2 * (p - x),
                       2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)

    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    # upper surface TE->LE, then lower surface LE->TE: one closed loop, no repeated endpoint
    return np.concatenate([np.stack([xu[::-1], yu[::-1]], 1),
                           np.stack([xl[1:], yl[1:]], 1)])


def cylinder(n=180):
    """Unit-diameter circle, centred so it occupies the same 0..1 x-range as a chord."""
    a = np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    return np.stack([0.5 + 0.5 * np.cos(a), 0.5 * np.sin(a)], 1)


def flat_plate(thick=0.02):
    """A thin plate - the cleanest test of pure incidence with no thickness effects."""
    return np.array([[0.0, -thick], [1.0, -thick], [1.0, thick], [0.0, thick]])


def ellipse(thick=0.20, n=160):
    a = np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    return np.stack([0.5 + 0.5 * np.cos(a), 0.5 * thick * np.sin(a)], 1)


def ogive(caliber=3.0, nose=0.42, n=200):
    """A pointed projectile: tangent-ogive nose blending into a cylindrical body.

    The nose is the standard tangent ogive - a circular arc tangent to the body at the
    shoulder, which is what actual bullets use and what sets the bow-shock angle. `nose` is
    the nose length as a fraction of total length; `caliber` is length / diameter.
    """
    half_d = 0.5 / float(caliber)
    ln = float(nose)
    # ogive radius from the tangency condition
    rho = (ln * ln + half_d * half_d) / (2.0 * half_d)
    x = np.linspace(0.0, ln, int(n * 0.7))
    y = np.sqrt(np.maximum(rho * rho - (ln - x) ** 2, 0.0)) + half_d - rho
    xb = np.linspace(ln, 1.0, max(int(n * 0.3), 2))[1:]
    yb = np.full_like(xb, half_d)
    xu = np.concatenate([x, xb])
    yu = np.concatenate([y, yb])
    # upper surface nose->tail, then lower surface tail->nose: one closed loop
    return np.concatenate([np.stack([xu, yu], 1),
                           np.stack([xu[::-1][:-1], -yu[::-1][:-1]], 1)])


def wedge(half_deg=15.0):
    """A simple symmetric wedge - the canonical oblique-shock body."""
    t = np.tan(np.radians(float(half_deg)))
    return np.array([[0.0, 0.0], [1.0, t], [1.0, -t]])


PROFILES = {
    "naca": naca4,
    "cylinder": cylinder,
    "plate": flat_plate,
    "ellipse": ellipse,
    "ogive": ogive,
    "wedge": wedge,
}


# --- placement ----------------------------------------------------------------------
def place(pts, chord, cx, cy, aoa_deg=0.0, pivot=0.25):
    """Unit-chord profile -> lattice coordinates.

    Rotation is about `pivot`, the QUARTER-CHORD by default, because that is the conventional
    aerodynamic centre: a foil sweeping through incidence in a real rig pivots about a spar
    near there, not about its centroid. Pivoting about the centroid instead swings the nose
    through a visibly larger arc and changes the effective plunge history the wake sees.

    SIGN CONVENTION: lattice +y is array-row-increasing, which `render` maps to SCREEN-DOWN.
    A positive angle of attack must therefore be a positive rotation here - the leading edge
    sits at x < pivot, and +a sends it to negative y, i.e. upward on screen, nose into the
    flow. Get this backwards and lift flips sign silently, suction side and all.
    """
    a = np.deg2rad(float(aoa_deg))
    p = np.asarray(pts, dtype=np.float64).copy()
    # Profiles are authored the natural way - x aft, y UP - because that is how every airfoil
    # table in existence is written. Lattice row index increases DOWNWARD, so profile +y has to
    # be negated on the way in or the section is placed inverted: camber pointing down, suction
    # side underneath, negative lift at zero incidence.
    #
    # A symmetric section is unchanged by this, which is exactly why it went unnoticed - every
    # validation case used a NACA 0012. It showed up the first time a cambered 2412 was rendered
    # and reported Cl < 0 at low AoA. tools/validate.py now has a camber case.
    p[:, 1] = -p[:, 1]
    p[:, 0] -= float(pivot)
    ca, sa = np.cos(a), np.sin(a)
    r = np.stack([p[:, 0] * ca - p[:, 1] * sa,
                  p[:, 0] * sa + p[:, 1] * ca], 1)
    r *= float(chord)
    r[:, 0] += float(cx)
    r[:, 1] += float(cy)
    return r


def rasterize(polys, nx, ny, supersample=3):
    """Closed polygons in lattice coords -> boolean solid mask, shape (ny, nx).

    Drawn at `supersample`x and area-averaged before thresholding. Without that, a slowly
    rotating body's mask jumps a whole cell at a time as the angle creeps, and every jump is
    a step change in the boundary that radiates a pressure pulse into the solution.
    """
    s = max(1, int(supersample))
    img = Image.new("L", (int(nx) * s, int(ny) * s), 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        p = np.asarray(p)
        if len(p) >= 3:
            d.polygon([(float(x) * s, float(y) * s) for x, y in p], fill=255)
    a = np.asarray(img, dtype=np.float32)
    if s > 1:
        a = a.reshape(int(ny), s, int(nx), s).mean(axis=(1, 3))
    return a >= 128.0


def wall_velocity(mask, polys_prev, polys_now, dt_steps=1):
    """Per-cell wall velocity implied by how far the polygons moved, as two (ny, nx) arrays.

    Only meaningful where `mask` is True. Uses the centroid displacement plus the rigid
    rotation about it, which is exact for the similarity transform `place` applies.
    """
    ny, nx = mask.shape
    wux = np.zeros((ny, nx), np.float32)
    wuy = np.zeros((ny, nx), np.float32)
    c_prev = np.mean(np.concatenate(polys_prev), axis=0)
    c_now = np.mean(np.concatenate(polys_now), axis=0)
    v = (c_now - c_prev) / max(int(dt_steps), 1)
    wux[mask] = v[0]
    wuy[mask] = v[1]
    return wux, wuy
