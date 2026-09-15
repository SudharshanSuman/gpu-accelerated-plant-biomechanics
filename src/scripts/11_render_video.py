"""Render the canopy from inside Isaac Sim with the RTX renderer.

Until the GPU driver was working, every visual in this project came from
matplotlib. This produces actual rendered frames: the scene an agricultural
robot's camera would see, with the plants driven by the Warp solver.

    ./run_isaac.sh --script 11        # via the launcher, or directly:
    ~/isaac-sim/python.sh src/scripts/11_render_video.py --plants 40

Frames are pulled through a replicator annotator rather than a writer, so the
simulation loop stays under our control: the solver steps, the renderer draws
what the solver produced, and only then is the frame read back. Letting
replicator's orchestrator drive would invert that.

Produces:
    results/render/frame_*.png
    results/canopy_render.gif
"""

import argparse
import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=20, help="frames captured per second")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--arm", action="store_true", default=True)
    ap.add_argument("--no-arm", dest="arm", action="store_false")
    ap.add_argument("--sweep-speed", type=float, default=0.35)
    ap.add_argument("--standoff", type=float, default=0.50)
    ap.add_argument("--height-target", type=float, default=0.55)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    from isaacsim import SimulationApp
    # renderer must be on: this script exists to produce pixels
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane, VisualCuboid
    from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt

    from _bootstrap import RESULTS
    from plantsim import SimConfig
    from plantsim.isaac_bridge import PlantCanopyBridge
    from plantsim.isaac_robot import ProceduralArm

    out_dir = pathlib.Path(args.out) if args.out else (RESULTS / "render")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig()
    cfg.scene.n_plants = args.plants
    cfg.scene.plants_per_row = 10

    world = World(stage_units_in_meters=1.0,
                  physics_dt=cfg.solver.dt, rendering_dt=cfg.solver.dt)
    GroundPlane(prim_path="/World/GroundPlane", size=40.0)
    stage = omni.usd.get_context().get_stage()

    # GroundPlane is a collision primitive and renders as nothing, which left
    # the first pass looking like plants floating in a void. VisualCuboid is
    # used rather than a raw UsdGeom.Cube because it binds an actual material:
    # a bare displayColor is ignored by the RTX renderer, which shaded the
    # soil white. Made large enough that its far edge sits outside the frame.
    VisualCuboid(
        prim_path="/World/Soil",
        position=np.array([2.0, -0.05, 0.0]),
        scale=np.array([200.0, 0.1, 200.0]),
        color=np.array([0.30, 0.21, 0.14]),
    )

    # Overcast dome for fill, plus a strong low sun so stems cast shadows and
    # read against the soil. The first pass used a bright bluish dome alone,
    # which flattened everything to the same value.
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(300.0)
    dome.CreateColorAttr(Gf.Vec3f(0.72, 0.80, 0.95))

    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    sun.CreateIntensityAttr(3400.0)
    sun.CreateAngleAttr(0.8)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.88))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-38.0, 125.0, 0.0))

    bridge = PlantCanopyBridge(cfg, n_plants=args.plants)
    print(f"[canopy] {bridge.summary()}")
    bridge.build(stage)

    arm = None
    if args.arm:
        arm = ProceduralArm(base_position=(-0.8, 0.0, -args.standoff),
                            tool_radius=0.06)
        arm.build(stage, "/World/Arm")
        print(f"[arm] reach {arm.reach:.2f} m, "
              f"{len(arm.collision_spheres()[0])} spheres")

    # Camera looking down the row from the side, roughly where a vehicle-mounted
    # sensor would sit.
    # Camera placed from an explicit look-at matrix. rep.create.camera's
    # look_at helper assumes a Z-up stage while this project is Y-up (gravity
    # is -Y), and Isaac's set_camera_view drives the viewport camera, which
    # did not track the render product reliably in headless mode. Building the
    # transform directly is the one approach that behaved predictably.
    #
    # KNOWN ISSUE: the ground plane still meets the background along a
    # diagonal rather than a level horizon, and the soil renders lighter than
    # its assigned albedo. The geometry and motion are correct -- this is a
    # staging problem, not a simulation one -- but the output is not yet
    # publication quality. The matplotlib figures remain the paper's visuals.
    eye = Gf.Vec3d(-1.30, 0.78, -1.55)
    target = Gf.Vec3d(1.30, 0.38, 0.05)
    up = Gf.Vec3d(0.0, 1.0, 0.0)

    view = Gf.Matrix4d().SetLookAt(eye, target, up)
    cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path("/World/RobotCamera"))
    cam_prim.CreateFocalLengthAttr(22.0)
    cam_prim.CreateHorizontalApertureAttr(24.0)
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    UsdGeom.Xformable(cam_prim).AddTransformOp().Set(view.GetInverse())

    render_product = rep.create.render_product(
        "/World/RobotCamera", (args.width, args.height))

    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])

    world.reset()

    total_frames = int(args.seconds / cfg.solver.dt)
    capture_every = max(1, int(round((1.0 / args.fps) / cfg.solver.dt)))
    settle = int(0.15 * total_frames)

    print(f"[run] {total_frames} sim frames, capturing every {capture_every} "
          f"({args.fps} fps), {args.width}x{args.height}")

    try:
        from PIL import Image
    except ImportError:
        print("[error] PIL unavailable inside Isaac's interpreter")
        app.close()
        return 1

    saved = []
    peak_force = 0.0
    for frame in range(total_frames):
        if arm is not None:
            elapsed = max(0.0, (frame - settle) * cfg.solver.dt)
            x = -0.8 + args.sweep_speed * elapsed
            arm.set_base_position((x, 0.0, -args.standoff))
            arm.set_tool_target((x + 0.10, args.height_target, 0.0))
            centres, radii = arm.collision_spheres()
            bridge.set_effectors(centres, radii)

        bridge.solver.step()
        bridge.publish()
        if arm is not None:
            arm.update_usd()

        world.step(render=True)

        forces = bridge.contact_force()
        if len(forces):
            peak_force = max(peak_force, float(np.linalg.norm(forces.sum(axis=0))))

        if frame % capture_every == 0:
            data = rgb.get_data()
            if data is None or getattr(data, "size", 0) == 0:
                continue
            arr = np.asarray(data)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            path = out_dir / f"frame_{len(saved):04d}.png"
            Image.fromarray(arr.astype(np.uint8)).save(path)
            saved.append(path)
            if len(saved) % 20 == 0:
                print(f"  captured {len(saved)} frames")

    print(f"[run] captured {len(saved)} frames, peak contact {peak_force:.3f} N")

    if saved:
        gif = RESULTS / "canopy_render.gif"
        images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in saved]
        images[0].save(gif, save_all=True, append_images=images[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"[out] wrote {gif}")
        print(f"[out] frames in {out_dir}")

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
