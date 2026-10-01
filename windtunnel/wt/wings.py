"""wings.py - motorsport rear wings as 2-D sections, built to the published rules.

Every dimension here is in MILLIMETRES, x aft and z up, the way the FIA writes its reference
volumes. `Wing.lattice_polygons()` converts to lattice cells at the very end, so the geometry
can be checked against the rulebook in the rulebook's own units.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Real, from the regulations:
  F1 2025   FIA 2025 Technical Regulations Art. 3.10 + RV-RW-PROFILES (Annex 28):
            two sections, flap chord < main chord, closed gap 10-15 mm, DRS opens the gap to
            85 mm, flap axis within 20 mm of the box's rear-top corner, box X 140-555,
            Z 670-910 at Y=0, profiles to Y=480 (span 960).
  F1 2026   FIA 2026 Technical Regulations Issue 8 (24 June 2024) Art. 3.11 + Annex 25:
            three volumes, gaps 10-15 mm, rear two rotate together about an axis in
            XR 450-525, X-mode opens the front gap to at most 65 mm, trailing-edge underside
            tangent within 40 mm of each section's rear point capped at 10 / 40 / 65 deg,
            box XR 240-630, Z 700-880, profiles to Y=575 (span 1150).
            Later issues of the 2026 rules may have changed this.
  GT3 RS    Porsche 992 GT3 RS press kit: fixed main + hydraulically adjusted upper element,
            34 deg of travel, DRS flattens it.

NOT real, and labelled as such on screen:
  - Every profile is the public NACA 6412, inverted. Team and Porsche profiles are
    proprietary.
  - F1 2025 element angles and chords are typical high-downforce values chosen to fit the box.
  - GT3 RS chords (318 / 170 mm), span (1712 mm) and the 6 deg main-element angle are a
    forum estimate from spy photos (Rennlist 992 GT3 RS thread, p.123), as is the upper element
    lying at the main element's angle when flat. The 12 mm gap (set at full tilt) and the
    upper element's pivot at its own mid-chord are assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import shapes

# NACA 6412 from the existing generator: 6% camber at 40% chord, 12% thick. Upper surface
# trailing edge -> leading edge, then lower surface leading edge -> trailing edge.
#
# Not the S1223 high-lift section, which was the first choice: its trailing edge turns up by
# ~42 deg over the last 3% of chord, so the 2026 rule's 10 deg cap on the main plane's trailing
# edge forced a 33 deg nose-up chord angle - a shape no team would build. The 6412's trailing
# edge sits ~19 deg off its chord, which lets all three rule sets produce a believable wing.
PROFILE = shapes.naca4("6412")
# Everything up to and including this index is the ORIGINAL upper surface, which becomes the
# UNDERSIDE once the section is inverted to make downforce.
PROFILE_LEADING_EDGE_INDEX = int(np.argmin(PROFILE[:, 0]))


def inverted_outline(unit_profile, chord_mm, le_x_mm, le_z_mm, angle_deg):
    """Unit-chord profile (x aft, y up) -> inverted, raked, scaled outline in mm.

    Inverted so the suction side faces the ground; `angle_deg` positive is trailing-edge UP.
    """
    unit = unit_profile.copy()
    unit[:, 1] = -unit[:, 1]
    angle = np.radians(angle_deg)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x = unit[:, 0] * cos_angle - unit[:, 1] * sin_angle
    z = unit[:, 0] * sin_angle + unit[:, 1] * cos_angle
    outline = np.stack([x, z], axis=1) * chord_mm
    outline[:, 0] += le_x_mm
    outline[:, 1] += le_z_mm
    return outline


@dataclass(frozen=True)
class Element:
    """One wing section: an inverted NACA 6412 with its leading edge at (le_x, le_z).

    `angle_deg` is the chord angle, positive trailing-edge UP - the direction a downforce
    element is raked.
    """
    chord_mm: float
    le_x_mm: float
    le_z_mm: float
    angle_deg: float

    def polygon(self):
        """Closed outline in mm, as an (N, 2) array of (x, z)."""
        return inverted_outline(PROFILE, self.chord_mm, self.le_x_mm, self.le_z_mm,
                                self.angle_deg)

    def underside(self):
        """The surface visible from below, trailing edge -> leading edge."""
        return self.polygon()[: PROFILE_LEADING_EDGE_INDEX + 1]

    def trailing_edge(self):
        return self.polygon()[0]


def rotate_about(points, pivot, delta_deg):
    """Rotate (N, 2) mm points about `pivot`; positive is trailing-edge-up (counter-clockwise)."""
    angle = np.radians(delta_deg)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    relative_x = points[:, 0] - pivot[0]
    relative_z = points[:, 1] - pivot[1]
    rotated = np.empty_like(points)
    rotated[:, 0] = pivot[0] + relative_x * cos_angle - relative_z * sin_angle
    rotated[:, 1] = pivot[1] + relative_x * sin_angle + relative_z * cos_angle
    return rotated


def point_to_segments_distance(point, starts, ends):
    """Shortest distance from one point to each of many segments."""
    segment = ends - starts
    length_squared = np.maximum((segment * segment).sum(axis=1), 1e-12)
    along = ((point - starts) * segment).sum(axis=1) / length_squared
    along = np.clip(along, 0.0, 1.0)
    nearest = starts + segment * along[:, None]
    return np.sqrt(((nearest - point) ** 2).sum(axis=1))


def gap_mm(polygon_a, polygon_b):
    """Closest distance between two closed outlines - what the FIA's spherical gauge measures."""
    best = float("inf")
    starts_b = polygon_b
    ends_b = np.roll(polygon_b, -1, axis=0)
    for point in polygon_a:
        distance = float(point_to_segments_distance(point, starts_b, ends_b).min())
        if distance < best:
            best = distance
    starts_a = polygon_a
    ends_a = np.roll(polygon_a, -1, axis=0)
    for point in polygon_b:
        distance = float(point_to_segments_distance(point, starts_a, ends_a).min())
        if distance < best:
            best = distance
    return best


def trailing_edge_underside_angle(polygon_underside, within_mm=40.0):
    """Steepest underside tangent within `within_mm` of the section's rearmost point (deg).

    This is the 2026 rule's measurement (Art. 3.11.1 e): angle against the X axis, positive
    pointing aft and up.
    """
    rear_x = float(polygon_underside[:, 0].max())
    steepest = -90.0
    for index in range(len(polygon_underside) - 1):
        aft_point = polygon_underside[index]
        fore_point = polygon_underside[index + 1]
        if rear_x - min(aft_point[0], fore_point[0]) > within_mm:
            continue
        angle = np.degrees(np.arctan2(aft_point[1] - fore_point[1],
                                      aft_point[0] - fore_point[0]))
        if angle > steepest:
            steepest = angle
    return steepest


def bisect(function, low, high, iterations=60):
    """Root of a monotonic `function` between `low` and `high`."""
    value_low = function(low)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        value_middle = function(middle)
        if (value_middle > 0.0) == (value_low > 0.0):
            low = middle
            value_low = value_middle
        else:
            high = middle
    return 0.5 * (low + high)


def place_behind(front, chord_mm, angle_deg, overlap_mm, target_gap_mm):
    """The element that sits behind and above `front` with exactly `target_gap_mm` of slot.

    The leading edge goes `overlap_mm` forward of the front element's trailing edge, then rises
    until the gauge distance between the two sections equals the target.
    """
    front_outline = front.polygon()
    trailing = front.trailing_edge()
    le_x = float(trailing[0]) - overlap_mm

    def excess(rise_mm):
        candidate = Element(chord_mm, le_x, float(trailing[1]) + rise_mm, angle_deg)
        return gap_mm(front_outline, candidate.polygon()) - target_gap_mm

    rise = bisect(excess, -10.0, 150.0)
    return Element(chord_mm, le_x, float(trailing[1]) + rise, angle_deg)


class Wing:
    """A fixed front section plus a group of sections that rotate together about one pivot.

    `rotation_deg` is the rotation of that group away from its design (maximum downforce)
    position. Negative flattens it.
    """

    def __init__(self, name, fixed, moving, pivot_mm, span_mm, min_rotation_deg,
                 toggle_rotation_deg, continuous, mode_names, sources, notes):
        self.name = name
        self.fixed = fixed
        self.moving = moving
        self.pivot_mm = np.asarray(pivot_mm, dtype=np.float64)
        self.span_mm = float(span_mm)
        self.min_rotation_deg = float(min_rotation_deg)
        self.toggle_rotation_deg = float(toggle_rotation_deg)
        self.continuous = continuous
        self.mode_names = mode_names
        self.sources = sources
        self.notes = notes
        design = self.polygons_mm(0.0)
        all_points = np.concatenate(design)
        # Reference chord for the coefficients: the streamwise length of the wing in its design
        # position. Fixed per wing, so a coefficient change is a FLOW change and never a
        # change of yardstick when the flap moves. Newtons do not depend on this choice.
        self.ref_chord_mm = float(all_points[:, 0].max() - all_points[:, 0].min())
        self.design_centre_mm = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))

    def polygons_mm(self, rotation_deg):
        outlines = []
        for element in self.fixed:
            outlines.append(element.polygon())
        for element in self.moving:
            outlines.append(rotate_about(element.polygon(), self.pivot_mm, rotation_deg))
        return outlines

    def slot_gap_mm(self, rotation_deg):
        """Gauge distance between the fixed section and the first moving one."""
        outlines = self.polygons_mm(rotation_deg)
        return gap_mm(outlines[len(self.fixed) - 1], outlines[len(self.fixed)])

    def lattice_polygons(self, rotation_deg, mm_per_cell, centre_x_cell, centre_y_cell):
        """mm (x aft, z up) -> lattice cells (x aft, y DOWN), centred on the design position."""
        cells = []
        for outline in self.polygons_mm(rotation_deg):
            lattice = np.empty_like(outline)
            lattice[:, 0] = centre_x_cell + (outline[:, 0] - self.design_centre_mm[0]) / mm_per_cell
            lattice[:, 1] = centre_y_cell - (outline[:, 1] - self.design_centre_mm[1]) / mm_per_cell
            cells.append(lattice)
        return cells

    def mode_name(self, rotation_deg):
        if abs(rotation_deg) < 1e-6:
            return self.mode_names[0]
        if abs(rotation_deg - self.toggle_rotation_deg) < 1e-6:
            return self.mode_names[1]
        return "moving"


def open_rotation_for_gap(wing, target_gap_mm):
    """The flattening rotation at which the slot opens to exactly `target_gap_mm`."""
    def excess(rotation_deg):
        return wing.slot_gap_mm(rotation_deg) - target_gap_mm
    return bisect(excess, 0.0, -60.0)


# --- the three wings -----------------------------------------------------------------------
F1_2025_BOX = {"x_min": 140.0, "x_max": 555.0, "z_min": 670.0, "z_max": 910.0}
F1_2026_BOX = {"x_min": 240.0, "x_max": 630.0, "z_min": 700.0, "z_max": 880.0}
F1_2025_CLOSED_GAP_MM = 12.0          # rule: 10-15
F1_2025_DRS_GAP_MM = 84.0             # rule: at most 85
F1_2026_CLOSED_GAP_MM = 12.0          # rule: 10-15
F1_2026_X_MODE_GAP_MM = 64.0          # rule: at most 65
F1_2026_ANGLE_CAPS_DEG = (10.0, 40.0, 65.0)
F1_2026_CAP_MARGIN_DEG = 1.0
GT3RS_TRAVEL_DEG = 34.0
GT3RS_GAP_MM = 12.0


def build_f1_2025():
    main = Element(chord_mm=280.0, le_x_mm=145.0, le_z_mm=700.0, angle_deg=8.0)
    flap = place_behind(main, chord_mm=190.0, angle_deg=42.0, overlap_mm=22.0,
                        target_gap_mm=F1_2025_CLOSED_GAP_MM)
    # Art. 3.10.10 b: the axis is no more than 20 mm below the box top and no more than 20 mm
    # forward of its rear face.
    pivot = (F1_2025_BOX["x_max"] - 18.0, F1_2025_BOX["z_max"] - 18.0)
    wing = Wing("F1 2025 rear wing", [main], [flap], pivot, span_mm=960.0,
                min_rotation_deg=-60.0, toggle_rotation_deg=0.0, continuous=False,
                mode_names=("DRS closed", "DRS open"),
                sources=["FIA 2025 Tech Regs Art. 3.10"],
                notes=["profile: NACA 6412 (team sections are proprietary)",
                       "element angles: typical high-downforce, not published"])
    wing.toggle_rotation_deg = open_rotation_for_gap(wing, F1_2025_DRS_GAP_MM)
    wing.min_rotation_deg = wing.toggle_rotation_deg
    return wing


def angle_for_cap(chord_mm, le_x_mm, le_z_mm, cap_deg):
    """Chord angle at which the trailing-edge underside tangent sits just under the 2026 cap."""
    def excess(angle_deg):
        element = Element(chord_mm, le_x_mm, le_z_mm, angle_deg)
        return trailing_edge_underside_angle(element.underside()) - (cap_deg - F1_2026_CAP_MARGIN_DEG)
    return bisect(excess, -60.0, 80.0)


def build_f1_2026():
    caps = F1_2026_ANGLE_CAPS_DEG
    main_angle = angle_for_cap(170.0, 245.0, 738.0, caps[0])
    main = Element(170.0, 245.0, 738.0, main_angle)

    middle_angle = angle_for_cap(125.0, 0.0, 0.0, caps[1])
    middle = place_behind(main, chord_mm=125.0, angle_deg=middle_angle, overlap_mm=15.0,
                          target_gap_mm=F1_2026_CLOSED_GAP_MM)
    rear_angle = angle_for_cap(95.0, 0.0, 0.0, caps[2])
    rear = place_behind(middle, chord_mm=95.0, angle_deg=rear_angle, overlap_mm=12.0,
                        target_gap_mm=F1_2026_CLOSED_GAP_MM)
    # Art. 3.11.6 b: the axis lies between XR 450 and 525, on the moving volumes.
    pivot = (500.0, float(rear.le_z_mm))
    wing = Wing("F1 2026 rear wing", [main], [middle, rear], pivot, span_mm=1150.0,
                min_rotation_deg=-60.0, toggle_rotation_deg=0.0, continuous=False,
                mode_names=("Z-mode (corner)", "X-mode (straight)"),
                sources=["FIA 2026 Tech Regs Iss. 8 Art. 3.11"],
                notes=["profile: NACA 6412 (team sections are proprietary)",
                       "angles set 1 deg under the 10/40/65 deg caps"])
    wing.toggle_rotation_deg = open_rotation_for_gap(wing, F1_2026_X_MODE_GAP_MM)
    wing.min_rotation_deg = wing.toggle_rotation_deg
    return wing


def build_gt3rs():
    main = Element(chord_mm=318.0, le_x_mm=0.0, le_z_mm=0.0, angle_deg=6.0)
    # The 12 mm gap is set at FULL TILT, because that is where a slot has to work: it is what
    # keeps the steep upper element's flow attached. Sizing it at the flat position instead left
    # a 23 mm slot at full tilt, and the upper element there flipped between attached
    # (C_down 3.5) and stalled (2.1) from run to run.
    #
    # The pivot is at the upper element's mid-chord, so flattening lifts its leading edge AWAY
    # from the main plane. Pivoting at the leading edge swings it down onto the main plane
    # instead (it closed the gap to 0.1 mm).
    upper = place_behind(main, chord_mm=170.0, angle_deg=main.angle_deg + GT3RS_TRAVEL_DEG,
                         overlap_mm=20.0, target_gap_mm=GT3RS_GAP_MM)
    upper_angle = np.radians(upper.angle_deg)
    pivot = (upper.le_x_mm + 0.5 * upper.chord_mm * np.cos(upper_angle),
             upper.le_z_mm + 0.5 * upper.chord_mm * np.sin(upper_angle))
    return Wing("Porsche 992 GT3 RS rear wing", [main], [upper], pivot, span_mm=1712.0,
                min_rotation_deg=-GT3RS_TRAVEL_DEG, toggle_rotation_deg=-GT3RS_TRAVEL_DEG,
                continuous=True, mode_names=("High downforce", "DRS (flat)"),
                sources=["Porsche 992 GT3 RS press kit"],
                notes=["chords/span: forum estimate from photos",
                       "profile NACA 6412, 12 mm gap, mid-chord pivot: assumed",
                       "Porsche: WHOLE CAR 409 kg @ 200, 860 kg @ 285 km/h"])


def build_all():
    return [build_f1_2025(), build_f1_2026(), build_gt3rs()]


# --- the simple wing (thickness study) -------------------------------------------------------
# One inverted NACA 44xx section: 4% camber at 40% chord, held at a fixed chord and rake while
# ONLY the thickness changes. Chord, camber and angle are typical of a single-element rear wing,
# not taken from any car.
SIMPLE_WING_CODE = "4412"            # the last two digits are overridden by the slider
SIMPLE_WING_CHORD_MM = 300.0
SIMPLE_WING_ANGLE_DEG = 6.0
SIMPLE_WING_SPAN_MM = 1000.0         # forces are quoted per metre of span
SIMPLE_WING_MIN_THICKNESS = 0.06
SIMPLE_WING_MAX_THICKNESS = 0.24


def simple_wing_mm(thickness_fraction):
    """Outline in mm, leading edge at the origin. Thickness changes; chord does not."""
    unit = shapes.naca4(SIMPLE_WING_CODE, thickness=thickness_fraction)
    return inverted_outline(unit, SIMPLE_WING_CHORD_MM, 0.0, 0.0, SIMPLE_WING_ANGLE_DEG)
