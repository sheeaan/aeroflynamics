# Aeroflynamics

A real-time, interactive 3D **Boeing 787 "Virtual Wind Tunnel"** built in Unity (C#).
Demonstrates real aerodynamics: airflow/streamlines, a pressure field, dynamic
composite **wing flex**, and **Fowler flaps + leading-edge slats** that extend and
change the lift math.

## Why

I love racing, and this project grew out of my IB Math IA on the downforce an F1 rear wing
makes. It runs real aerodynamics so you can see that downforce being made.

The 2-D wind tunnel in [`windtunnel/`](windtunnel/README.md) is already working. It runs F1 2025,
F1 2026 and Porsche GT3 RS rear wings with live DRS / active aero, and a wing-thickness slider
with live downforce and drag. Its README walks through the math.

## Layout
```
aeroflynamics/
├── README.md            ← you are here
├── docs/
│   └── DESIGN.md         ← full technical spec (physics, milestones, 787 data)
├── windtunnel/           ← 2-D CFD wind tunnel (Python): rear wings, DRS, thickness study
└── WingTunnel/           ← the Unity project (planned - not started yet)
```

## Build phases
1. **Setup** — Unity project, camera, wind-tunnel environment & lighting.
2. **Geometry** — procedural wing + flaps + slats generated in C#.
3. **Physics** — aerodynamics solver + wing-flex math.
4. **Visualization** — particle streamlines + wind-tunnel effects.
5. **UI** — Canvas sliders: AoA, airspeed, altitude, flaps/slats, wing load (g).

See `docs/DESIGN.md` for the detailed plan.
