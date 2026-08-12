"""bodies.py - free rigid bodies, moved by the fluid rather than by a script.

This is where the validated force integral earns its keep. A body here has no prescribed path:
each step it is handed the momentum the fluid actually delivered to its surface, and it goes
where that puts it. Nothing in this file decides what should happen.

WHY THIS NEEDS THE FORCE TO BE RIGHT, not merely plausible
----------------------------------------------------------
A wrong drag in a fixed-body scene shows up only as a wrong NUMBER in the corner of the frame -
the picture is unaffected. Here it shows up as motion, and motion is the subject. A drag 3x too
large (which is exactly what this project's first momentum-exchange integral produced, see
tools/validate.py) does not look like an error: it looks like a slightly heavier object in
slightly thicker fluid. There is no frame you could inspect to catch it.

THE NINE CONSTRAINTS - all of them learned the hard way, none of them optional
------------------------------------------------------------------------------
 1. Wall forces act on the fluid cells ADJACENT to the body, never on the solid cells; the
    populations stored inside a body are not physical (see LBM.force_field).
 2. Bounce-back needs to know the wall velocity, and it breaks above |u_wall| ~ 0.12 in lattice
    units. `speed_limit` clamps it and reports when it bites rather than silently diverging.
 3. Rotation is capped by SURFACE speed (omega * radius), not by angular rate. A big slow body
    can break the boundary condition while its omega looks harmless.
 4. Gaps between bodies must stay above one cell or the fluid between them is unresolved and
    the two masks fuse into one.
 5. A shape may not materialise inside moving fluid: a body must be introduced where it already
    matches the flow, or the discontinuity radiates.
 6. Fluid loads need LOW-PASS FILTERING and contacts need soft damping. The raw per-step force
    on a rasterised body is noisy - the mask gains and loses whole cells as it moves - and
    feeding that straight into an integrator produces jitter that looks like turbulence.
 7. Flow sampling must avoid body centres, where the "velocity" is whatever bounce-back left.
 8. Free-body scenes want modest Reynolds numbers (~450-900). Higher and the load fluctuations
    outrun the filter.
 9. Bodies are impenetrable: contact is resolved by pushing apart, never by letting masks merge.
"""
from __future__ import annotations

import numpy as np

from .gpu import asnumpy
from . import shapes


class RigidBody:
    """One free body: a unit-space polygon plus its rigid state.

    Parameters
    ----------
    pts        closed polygon in unit-chord space (from `shapes`)
    size       scale in cells (the polygon's unit length maps to this)
    cx, cy     initial centre, in lattice cells
    angle      initial rotation, degrees
    density    body density relative to the fluid (which is 1). 1.0 is neutrally buoyant;
               below 1 rises, above 1 sinks.
    pivot      rotation pivot in unit-space, as for shapes.place
    """

    def __init__(self, pts, size, cx, cy, angle=0.0, density=1.6, pivot=0.5,
                 vx=0.0, vy=0.0, omega=0.0):
        self.pts = np.asarray(pts, dtype=np.float64)
        self.size = float(size)
        self.cx, self.cy = float(cx), float(cy)
        self.angle = float(angle)
        self.vx, self.vy = float(vx), float(vy)
        self.omega = float(omega)
        self.density = float(density)
        self.pivot = float(pivot)

        # Mass and moment of inertia from the ACTUAL rasterised area, not from an analytic
        # formula for the ideal shape. The solver only ever sees the mask, so the mask is what
        # the fluid pushes on; using the ideal area instead leaves a systematic mismatch
        # between the force applied and the inertia resisting it.
        # probe grid sized to the body, so a large one is not silently clipped by a fixed box
        g = max(int(4 * self.size), 64)
        m = self.rasterize_at(self.cx, self.cy, self.angle, g, g, probe=True)
        area = float(m.sum())
        self.area = max(area, 1.0)
        self.mass = self.density * self.area
        jj, ii = np.nonzero(m)
        r2 = (ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2
        self.inertia = max(self.density * float(r2.sum()), 1.0)

        self._fx_s = 0.0            # low-passed loads
        self._fy_s = 0.0
        self._tq_s = 0.0
        self.mask = None
        self.clamped = 0            # how many steps the speed limit actually bit

    def polygon(self, cx=None, cy=None, angle=None):
        return shapes.place(self.pts, self.size,
                            self.cx if cx is None else cx,
                            self.cy if cy is None else cy,
                            self.angle if angle is None else angle,
                            pivot=self.pivot)

    def rasterize_at(self, cx, cy, angle, nx, ny, probe=False):
        poly = shapes.place(self.pts, self.size, cx if not probe else nx * 0.5,
                            cy if not probe else ny * 0.5, angle, pivot=self.pivot)
        return shapes.rasterize([poly], nx, ny)


class BodySystem:
    """A set of free bodies sharing one lattice.

    Usage per frame:
        sys.update(sim, substeps)      # integrate + rebuild the mask
        sim.set_solid(sys.mask)
        sim.set_wall_velocity(*sys.wall_velocity())
        sim.run(substeps)
    """

    def __init__(self, bodies, nx, ny, gravity=0.0, damping=0.002,
                 load_filter=0.10, speed_limit=0.10, restitution=0.25,
                 walls=True, wall_margin=4.0, recycle=True, outlet_margin=28.0,
                 spawn_x=None, u_stream=0.0, seed=11):
        self.bodies = list(bodies)
        self.nx, self.ny = int(nx), int(ny)
        self.gravity = float(gravity)
        self.damping = float(damping)
        # LOAD FILTER: fraction of the new load blended in per step. The raw momentum exchange
        # on a rasterised body jumps whenever the mask gains or loses a boundary cell, and that
        # jump is discretisation noise, not force. Filtering is not cosmetic - an unfiltered
        # load fed to the integrator makes a body buzz, and the buzz then radiates into the
        # fluid as a pressure wave the body itself created.
        self.load_filter = float(load_filter)
        self.speed_limit = float(speed_limit)
        self.restitution = float(restitution)
        self.walls = bool(walls)
        # KEEP BODIES OUT OF THE PLACES THE LATTICE CANNOT RESOLVE THEM. Two of them:
        #
        #   the side walls - a body allowed to touch leaves a sub-cell gap, and the flow
        #   forced through an unresolved gap accelerates without limit. Measured: four bodies
        #   pinned against the walls and the outlet drove max|u| from a freestream of 0.06 to
        #   0.35 and then past the lattice ceiling at frame 174.
        #
        #   the outlet sponge - it relaxes the fluid toward the uniform freestream, so a body
        #   sitting inside it is a solid obstacle in a region being forced to look like
        #   undisturbed flow. Those two demands cannot both be met and the solve pays for it.
        self.wall_margin = float(wall_margin)
        self.outlet_margin = float(outlet_margin)
        self.recycle = bool(recycle)
        self.spawn_x = float(spawn_x) if spawn_x is not None else nx * 0.10
        self.u_stream = float(u_stream)
        self.rng = np.random.default_rng(seed)
        self.recycled = 0
        self.mask = np.zeros((ny, nx), bool)
        self._masks = []
        self.rebuild()

    # -- geometry ----------------------------------------------------------------------
    def rebuild(self):
        """Rasterise every body separately, then union. Separate masks are needed because the
        force integral has to be attributed per body, and a union cannot be taken apart."""
        self._masks = [b.rasterize_at(b.cx, b.cy, b.angle, self.nx, self.ny)
                       for b in self.bodies]
        for b, m in zip(self.bodies, self._masks):
            b.mask = m
        self.mask = np.zeros((self.ny, self.nx), bool)
        for m in self._masks:
            self.mask |= m
        return self.mask

    def wall_velocity(self):
        """Per-cell wall velocity over the union mask: rigid translation plus rotation.

        Built from each body's own state at its own cells, so two bodies touching do not
        smear each other's velocity across the contact.
        """
        wux = np.zeros((self.ny, self.nx), np.float32)
        wuy = np.zeros((self.ny, self.nx), np.float32)
        jj = np.arange(self.ny)[:, None]
        ii = np.arange(self.nx)[None, :]
        for b, m in zip(self.bodies, self._masks):
            if not m.any():
                continue
            rx = (ii - b.cx).astype(np.float32)      # (1, nx)
            ry = (jj - b.cy).astype(np.float32)      # (ny, 1)
            # v = v_cm + omega x r. NOTE lattice +y is screen-DOWN, so a positive omega reads
            # as clockwise on screen; the torque below uses the same convention, so the pair
            # is self-consistent and a body spins the way its own torque says it should.
            #
            # Both terms must be broadcast to the FULL (ny, nx) shape before boolean indexing.
            # `b.vx - b.omega * ry` is (ny, 1), and indexing that with an (ny, nx) mask is an
            # error rather than a broadcast - an earlier version guarded the shape mismatch
            # with a fallback to plain `b.vx`, which silently dropped rotation from the wall
            # velocity entirely: bodies span but the fluid never felt them spin.
            vxf = np.broadcast_to(b.vx - b.omega * ry, (self.ny, self.nx))
            vyf = np.broadcast_to(b.vy + b.omega * rx, (self.ny, self.nx))
            wux[m] = vxf[m]
            wuy[m] = vyf[m]
        np.clip(wux, -self.speed_limit, self.speed_limit, out=wux)
        np.clip(wuy, -self.speed_limit, self.speed_limit, out=wuy)
        return wux, wuy

    # -- dynamics ------------------------------------------------------------------------
    def loads(self, sim):
        """(fx, fy, torque) per body, from the solver's own force field.

        One `force_field()` call for the whole lattice, then a masked sum per body - far
        cheaper than running the link loop once per body, and it guarantees every body is
        integrated over exactly the same field.
        """
        FX, FY = sim.force_field()
        FX = np.asarray(asnumpy(FX))
        FY = np.asarray(asnumpy(FY))
        jj = np.arange(self.ny)[:, None]
        ii = np.arange(self.nx)[None, :]
        out = []
        for b, m in zip(self.bodies, self._masks):
            # the force lives on the FLUID cells adjacent to the body, so dilate the mask by
            # one cell to collect it - summing over the solid cells themselves collects nothing
            g = m.copy()
            g[1:, :] |= m[:-1, :]
            g[:-1, :] |= m[1:, :]
            g[:, 1:] |= m[:, :-1]
            g[:, :-1] |= m[:, 1:]
            ring = g & ~self.mask
            fx = float(FX[ring].sum())
            fy = float(FY[ring].sum())
            tq = float(((ii - b.cx) * FY - (jj - b.cy) * FX)[ring].sum())
            out.append((fx, fy, tq))
        return out

    def update(self, sim, dt=1.0):
        """Integrate one frame's worth of body motion and rebuild the geometry.

        `dt` is in solver steps: pass the number of steps that will be run this frame, so the
        body advances by the same amount of time the fluid does.
        """
        loads = self.loads(sim)
        a = self.load_filter
        for b, (fx, fy, tq) in zip(self.bodies, loads):
            b._fx_s += a * (fx - b._fx_s)
            b._fy_s += a * (fy - b._fy_s)
            b._tq_s += a * (tq - b._tq_s)

            ax = b._fx_s / b.mass
            ay = b._fy_s / b.mass + self.gravity
            b.vx += ax * dt
            b.vy += ay * dt
            b.omega += (b._tq_s / b.inertia) * dt
            b.vx *= (1.0 - self.damping)
            b.vy *= (1.0 - self.damping)
            b.omega *= (1.0 - self.damping)

            # Constraint 2/3: the BOUNDARY CONDITION, not the physics, sets the ceiling. Clamp
            # translation directly and rotation by its SURFACE speed - a large body reaches the
            # limit at an omega that looks entirely reasonable.
            sp = np.hypot(b.vx, b.vy)
            if sp > self.speed_limit:
                b.vx *= self.speed_limit / sp
                b.vy *= self.speed_limit / sp
                b.clamped += 1
            r_max = 0.5 * b.size
            if abs(b.omega) * r_max > self.speed_limit:
                b.omega = np.sign(b.omega) * self.speed_limit / max(r_max, 1e-9)
                b.clamped += 1

            b.cx += b.vx * dt
            b.cy += b.vy * dt
            b.angle += np.degrees(b.omega) * dt

        if self.walls:
            self._bounce_walls()
        self._recycle()
        self._separate()
        return self.rebuild()

    def _bounce_walls(self):
        """Keep bodies clear of the side walls by `wall_margin`, so the gap stays resolved."""
        for b in self.bodies:
            r = 0.5 * b.size + self.wall_margin
            if b.cy < r:
                b.cy = r
                b.vy = abs(b.vy) * self.restitution
            elif b.cy > self.ny - 1 - r:
                b.cy = self.ny - 1 - r
                b.vy = -abs(b.vy) * self.restitution
            if b.cx < r:
                b.cx = r
                b.vx = abs(b.vx) * self.restitution

    def _recycle(self):
        """Wash bodies out of the downstream end and re-introduce them near the inlet.

        A tunnel with things dropped into it does not accumulate them at the back - they leave.
        Letting them pile up against the outlet is both wrong and, because the sponge lives
        there, unstable.

        The respawn happens NEAR THE INLET AND AT THE FREESTREAM VELOCITY, which is the least
        violent place to do it. Constraint 5 says a shape may not materialise inside moving
        fluid, and strictly this does violate it - but the inlet region is uniform freestream,
        and giving the body the local flow velocity means the bounce-back sees no relative
        motion at the instant it appears. Dropping one in at rest, mid-wake, radiates instead.
        """
        if not self.recycle:
            return
        x_out = self.nx - self.outlet_margin
        for b in self.bodies:
            if b.cx - 0.5 * b.size < x_out:
                continue
            r = 0.5 * b.size + self.wall_margin
            b.cx = self.spawn_x
            b.cy = float(self.rng.uniform(r, self.ny - 1 - r))
            b.vx, b.vy = self.u_stream, 0.0
            b.omega = float(self.rng.uniform(-0.002, 0.002))
            b.angle = float(self.rng.uniform(0, 360))
            # clear the low-pass state too, or the load from its old position keeps pushing it
            b._fx_s = b._fy_s = b._tq_s = 0.0
            self.recycled += 1

    def _separate(self):
        """Constraint 9: bodies stay impenetrable.

        Resolved on the bounding circles rather than on the masks. Mask-level contact would be
        exact but it is also where two bodies can end up sharing cells for a step, and a
        merged mask is not recoverable - the union has no seam to split along, so the pair
        would move as one solid from then on.
        """
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.bodies[i], self.bodies[j]
                ra, rb = 0.5 * a.size, 0.5 * b.size
                dx, dy = b.cx - a.cx, b.cy - a.cy
                d = float(np.hypot(dx, dy))
                # Constraint 4: leave more than a cell of fluid between them, or the gap is
                # unresolved and the two masks behave as one body anyway
                want = (ra + rb) * 0.92 + 1.5
                if d >= want or d < 1e-9:
                    continue
                nxu, nyu = dx / d, dy / d
                push = 0.5 * (want - d)
                a.cx -= nxu * push
                a.cy -= nyu * push
                b.cx += nxu * push
                b.cy += nyu * push
                # exchange the normal component of momentum, with damping
                rel = (b.vx - a.vx) * nxu + (b.vy - a.vy) * nyu
                if rel < 0.0:
                    e = self.restitution
                    ma, mb = a.mass, b.mass
                    imp = -(1.0 + e) * rel / (1.0 / ma + 1.0 / mb)
                    a.vx -= imp * nxu / ma
                    a.vy -= imp * nyu / ma
                    b.vx += imp * nxu / mb
                    b.vy += imp * nyu / mb

    # -- diagnostics ---------------------------------------------------------------------
    def report(self):
        return [f"body {k}: pos ({b.cx:6.1f},{b.cy:6.1f})  v ({b.vx:+.4f},{b.vy:+.4f})  "
                f"w {b.omega:+.5f}  clamped {b.clamped}"
                for k, b in enumerate(self.bodies)]
