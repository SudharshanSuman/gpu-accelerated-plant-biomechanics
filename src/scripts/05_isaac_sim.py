"""Deformable plant canopy inside NVIDIA Isaac Sim.

Run through the launcher, which finds Isaac and fixes up the environment:

    ./run_isaac.sh                              # headless, bakes a USD animation
    ./run_isaac.sh --plants 64 --seconds 10
    ./run_isaac.sh --gui                        # interactive viewport (needs a GPU)

The canopy is driven entirely by this project's Warp solver. Isaac Sim is the
renderer and robot host; PhysX never sees the plants, because it has no model
for a tapered anisotropic stem, which is the reason this solver exists.

Targets Isaac Sim 5.x/6.x (`isaacsim.core.api`). The `omni.isaac.core`
namespace used by older tutorials was removed in Isaac Sim 5.
"""

import argparse
import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plants", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--gui", action="store_true",
                    help="open the viewport (requires a working GPU driver)")
    ap.add_argument("--record", dest="record", action="store_true", default=True,
                    help="bake motion into the exported USD (default)")
    ap.add_argument("--no-record", dest="record", action="store_false")
    ap.add_argument("--out", default=None, help="output .usd path")
    ap.add_argument("--arm", action="store_true",
                    help="use the procedural articulated arm on a driving base "
                         "instead of a single floating effector sphere")
    ap.add_argument("--arm-standoff", type=float, default=0.55,
                    help="lateral distance from the crop row to the vehicle (m)")
    ap.add_argument("--robot-usd", default=None,
                    help="optional manipulator USD to add and drive (UNTESTED: "
                         "Isaac ships no robot meshes locally, so this needs an "
                         "asset downloaded from NVIDIA's content server)")
    ap.add_argument("--effector-radius", type=float, default=0.06)
    ap.add_argument("--effector-height", type=float, default=0.55)
    ap.add_argument("--sweep-speed", type=float, default=0.30,
                    help="end-effector travel speed in m/s. This is the single "
                         "most important knob for contact force: drag a gripper "
                         "through a canopy fast enough and it breaks stems.")
    ap.add_argument("--sweep-start", type=float, default=-0.5,
                    help="x position where the sweep begins (m)")
    ap.add_argument("--stem-failure-n", type=float, default=5.0,
                    help="contact force above which a stem of this size is "
                         "assumed to be damaged (N)")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # SimulationApp must exist before any omni/isaacsim import. Kit enforces
    # this, and violating it fails with a confusing extension error.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.gui})

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane, VisualSphere
    import omni.usd

    from _bootstrap import RESULTS
    from plantsim import SimConfig
    from plantsim.isaac_bridge import PlantCanopyBridge

    cfg = SimConfig()
    cfg.scene.n_plants = args.plants
    cfg.scene.plants_per_row = 8

    world = World(stage_units_in_meters=1.0,
                  physics_dt=cfg.solver.dt,
                  rendering_dt=cfg.solver.dt)
    GroundPlane(prim_path="/World/GroundPlane", size=20.0)

    bridge = PlantCanopyBridge(cfg, n_plants=args.plants, device=args.device)
    print(f"[canopy] {bridge.summary()}")
    bridge.build(omni.usd.get_context().get_stage())

    arm = None
    effector = None
    if args.arm:
        from plantsim.isaac_robot import ProceduralArm
        arm = ProceduralArm(
            base_position=(args.sweep_start, 0.0, -args.arm_standoff),
            tool_radius=args.effector_radius,
        )
        arm.build(omni.usd.get_context().get_stage(), "/World/Arm")
        print(f"[arm] reach {arm.reach:.2f} m, {arm.spheres_per_link} spheres/link, "
              f"coverage_ok={arm.coverage_ok()}")
        if not arm.coverage_ok():
            print("[arm] WARNING: collision spheres do not overlap; a stem "
                  "could pass between them without registering contact.")
    else:
        effector = VisualSphere(
            prim_path="/World/Effector", name="effector",
            position=np.array([-0.5, args.effector_height, 0.3]),
            radius=args.effector_radius, color=np.array([0.25, 0.43, 0.71]),
        )

    articulation = None
    if args.robot_usd:
        # Optional: attach a real manipulator. Kept optional on purpose --
        # Isaac's bundled robot assets live on a content server, and a demo
        # that needs a download is a demo that fails on an offline machine.
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation
        add_reference_to_stage(usd_path=args.robot_usd, prim_path="/World/Robot")
        articulation = SingleArticulation(prim_path="/World/Robot", name="robot")
        world.scene.add(articulation)
        print(f"[robot] loaded {args.robot_usd}")

    world.reset()
    if args.record:
        bridge.start_recording(fps=1.0 / cfg.solver.dt)

    total = int(args.seconds / cfg.solver.dt)
    settle_fraction = 0.15          # let the canopy settle before the robot moves
    settle_frames = int(settle_fraction * total)

    print(f"[run] {total} frames ({args.seconds:.1f} s), "
          f"{'GUI' if args.gui else 'headless'}, record={args.record}")
    print(f"[run] effector sweeping at {args.sweep_speed:.2f} m/s "
          f"from x={args.sweep_start:.2f} m at height {args.effector_height:.2f} m")

    peak_force = 0.0
    frames_over_threshold = 0
    for frame in range(total):
        # Motion defined by speed, not by a fraction of the run, so the
        # contact force does not silently change when --seconds changes.
        elapsed = max(0.0, (frame - settle_frames) * cfg.solver.dt)
        x = args.sweep_start + args.sweep_speed * elapsed

        if articulation is not None:
            pose = articulation.get_world_pose()[0]
            position = np.asarray(pose, dtype=np.float32)
            bridge.set_effector(position, args.effector_radius)

        elif arm is not None:
            # The vehicle drives along the row; the arm holds a reach pose into
            # the canopy. Contact force is then governed by driving speed,
            # which is the parameter an operator actually sets.
            arm.set_base_position((x, 0.0, -args.arm_standoff))
            arm.set_tool_target((x + 0.10, args.effector_height, 0.0))
            centres, radii = arm.collision_spheres()
            bridge.set_effectors(centres, radii)

        else:
            position = np.array([x, args.effector_height, 0.0], dtype=np.float32)
            effector.set_world_pose(position=position)
            bridge.set_effector(position, args.effector_radius)

        bridge.solver.step()

        if args.record:
            bridge.record_frame()
            if arm is not None:
                arm.update_usd(time_code=float(frame))
        else:
            bridge.publish()
            if arm is not None:
                arm.update_usd()

        world.step(render=args.gui)

        # With the arm, several spheres touch at once; the load on the
        # manipulator is the sum over all of them, not any single contact.
        forces = bridge.contact_force()
        force = float(np.linalg.norm(forces.sum(axis=0))) if len(forces) else 0.0
        peak_force = max(peak_force, force)
        if force > args.stem_failure_n:
            frames_over_threshold += 1
        if frame % 60 == 0:
            print(f"  frame {frame:5d}/{total}  x={x:+.2f} m  contact {force:6.3f} N")

    print(f"[run] peak contact force {peak_force:.3f} N "
          f"at {args.sweep_speed:.2f} m/s")
    if frames_over_threshold:
        print(f"[run] WARNING: {frames_over_threshold} frames exceeded the "
              f"{args.stem_failure_n:.1f} N stem-damage threshold. At this "
              f"approach speed the manipulator would damage the crop.")
    else:
        print(f"[run] stayed below the {args.stem_failure_n:.1f} N stem-damage "
              f"threshold throughout.")

    if args.record:
        out = args.out or str(RESULTS / "canopy_isaac.usd")
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        bridge.export_usd(out)
        print(f"[out] wrote {out}")
        print("      open with:  ~/isaac-sim/isaac-sim.sh  (or usdview)")

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
