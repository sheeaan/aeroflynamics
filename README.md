# Aeroflynamics

A real-time, interactive 3D **Boeing 787 "Virtual Wind Tunnel"** built in Unity (C#).
Demonstrates real aerodynamics: airflow/streamlines, a pressure field, dynamic
composite **wing flex**, and **Fowler flaps + leading-edge slats** that extend and
change the lift math.

## Layout
```
aeroflynamics/
├── README.md            ← you are here
├── docs/
│   └── DESIGN.md         ← full technical spec (physics, milestones, 787 data)
└── WingTunnel/           ← the Unity project (created via Unity Hub in Phase 1)
```

## Build phases
1. **Setup** — Unity project, camera, wind-tunnel environment & lighting.
2. **Geometry** — procedural wing + flaps + slats generated in C#.
3. **Physics** — aerodynamics solver + wing-flex math.
4. **Visualization** — particle streamlines + wind-tunnel effects.
5. **UI** — Canvas sliders: AoA, airspeed, altitude, flaps/slats, wing load (g).

See `docs/DESIGN.md` for the detailed plan.
