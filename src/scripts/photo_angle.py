"""Turn two points clicked on a photograph into a tip angle in degrees.

Measuring the angle by eye with a protractor works, but reading pixel
coordinates out of any image viewer is faster and repeatable. Most viewers
(GIMP, Preview, even Windows Photos with the cursor readout) show pixel
coordinates on hover.

Usage — plant base at pixel (812, 1540), stem tip at (960, 300):

    python3 src/scripts/photo_angle.py --base 812 1540 --tip 960 300

If the camera was not perfectly level, give two points on a vertical
reference (a taped plumb line, a door frame) and the tilt is removed:

    python3 src/scripts/photo_angle.py --base 812 1540 --tip 960 300 \
        --vertical 200 1500 210 240

Image pixel coordinates run *downward*, which is the single easiest thing to
get wrong here; it is handled below.
"""

import argparse
import math


def angle_from_vertical(base, tip, vertical=None):
    # Flip y so that "up in the photo" is +y, matching the physical scene.
    dx = tip[0] - base[0]
    dy = -(tip[1] - base[1])

    reference = 0.0
    if vertical is not None:
        vx = vertical[2] - vertical[0]
        vy = -(vertical[3] - vertical[1])
        # Angle of the reference line away from image-vertical, subtracted off.
        reference = math.degrees(math.atan2(vx, vy))

    return math.degrees(math.atan2(dx, dy)) - reference


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=float, nargs=2, required=True,
                    metavar=("X", "Y"), help="pixel coords of the stem base")
    ap.add_argument("--tip", type=float, nargs=2, required=True,
                    metavar=("X", "Y"), help="pixel coords of the stem tip")
    ap.add_argument("--vertical", type=float, nargs=4, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="two points on a known-vertical reference line")
    args = ap.parse_args()

    angle = angle_from_vertical(args.base, args.tip, args.vertical)
    print(f"tip angle from vertical: {angle:+.2f} deg")
    print(f"  (magnitude {abs(angle):.2f} deg; sign shows lean direction)")
    print()
    print("Put the magnitude in the tip_angle_deg column of")
    print("Research/field_measurements.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
