"""Test suite. Run with:  python -m pytest src/tests -v"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from plantsim import CPUPlantSolver, SimConfig, build_scene, build_template
from plantsim.solver import PlantSolver
from plantsim.topology import greedy_colour, verify_colouring
from plantsim.validation import cantilever_self_weight, inextensibility


# ---------------------------------------------------------------- topology
def test_colouring_is_conflict_free():
    cfg = SimConfig()
    tpl = build_template(cfg)
    for nodes in (tpl.stretch_nodes, tpl.bend_nodes):
        colours = greedy_colour(nodes, tpl.n_nodes)
        verify_colouring(nodes, colours)          # raises if two share a particle
        assert sum(len(c) for c in colours) == len(nodes)


def test_chain_needs_only_two_stretch_colours():
    """A simple chain should colour optimally, not fall back to many colours."""
    cfg = SimConfig()
    cfg.leaf.enabled = False
    tpl = build_template(cfg)
    assert len(greedy_colour(tpl.stretch_nodes, tpl.n_nodes)) == 2


def test_rigidity_decreases_with_height():
    cfg = SimConfig()
    tpl = build_template(cfg)
    stem_hinges = tpl.bend_compliance[: cfg.stem.n_nodes - 2]
    assert np.all(np.diff(stem_hinges) > 0), "compliance must rise toward the tip"
    ratio = stem_hinges[-1] / stem_hinges[0]
    assert ratio > 10, f"expected a strong taper, got {ratio:.1f}x"


def test_root_is_pinned_and_clamped():
    """The soil anchor must exist, or the plant topples as a rigid pole."""
    cfg = SimConfig()
    tpl = build_template(cfg)
    assert tpl.pinned[0], "root particle must be pinned"
    assert tpl.pinned[-1], "buried anchor particle must be pinned"
    # A bending constraint must hinge at the root between anchor and node 1.
    assert any(tuple(c) == (tpl.n_nodes - 1, 0, 1) for c in tpl.bend_nodes)


def test_scene_tiling_preserves_topology():
    cfg = SimConfig()
    scene = build_scene(cfg, n_plants=5)
    k = scene.template.n_nodes
    assert scene.n_nodes == 5 * k
    # Every constraint must stay inside its own plant.
    for nodes in (scene.stretch_nodes, scene.bend_nodes):
        plant_of = nodes // k
        assert np.all(plant_of == plant_of[:, :1]), "constraint spans two plants"


# ------------------------------------------------------------------ solver
def test_solver_is_stable_and_finite():
    cfg = SimConfig()
    solver = PlantSolver(cfg)
    solver.run(120)
    solver.sync()
    pos = solver.positions()
    assert np.isfinite(pos).all()
    # The buried soil anchor is meant to sit below y=0; every other particle
    # is subject to the ground constraint.
    free = solver.scene.inv_mass > 0.0
    assert pos[free, 1].min() >= -1e-3, "particles fell through the ground"


def test_plant_stays_standing_in_wind():
    cfg = SimConfig()
    cfg.scene.height_jitter = 0.0
    solver = PlantSolver(cfg)
    solver.run(240)
    solver.sync()
    tpl = solver.scene.template
    stem = solver.positions()[solver.scene.plant_slice(0)][tpl.stem_path]
    assert stem[-1, 1] > 0.7 * cfg.stem.length, "plant collapsed"
    assert abs(stem[-1, 0]) > 0.01, "plant did not respond to wind at all"


def test_base_is_stiffer_than_tip():
    """The defining behaviour: the stem must bend as an arc, not a straight lean."""
    cfg = SimConfig()
    cfg.scene.height_jitter = 0.0
    solver = PlantSolver(cfg)
    solver.run(240)
    solver.sync()
    tpl = solver.scene.template
    stem = solver.positions()[solver.scene.plant_slice(0)][tpl.stem_path]
    steps = np.abs(np.diff(stem[:, 0]))
    lower, upper = steps[:4].mean(), steps[-4:].mean()
    assert upper > 3 * lower, (
        f"tip should displace far more per segment than the base "
        f"(base {lower:.4f} m, tip {upper:.4f} m)"
    )


def test_stem_is_inextensible():
    cfg = SimConfig()
    solver = PlantSolver(cfg)
    solver.run(180)
    solver.sync()
    _, _, error = inextensibility(solver)
    assert error < 0.005, f"stem stretched by {100 * error:.2f}%"


def test_root_does_not_drift():
    cfg = SimConfig()
    solver = PlantSolver(cfg)
    start = solver.positions()[0].copy()
    solver.run(180)
    solver.sync()
    assert np.allclose(solver.positions()[0], start, atol=1e-6)


# ------------------------------------------------------- cross-validation
def test_gpu_and_numpy_paths_agree():
    cfg = SimConfig()
    cfg.scene.n_plants = 4
    warp_solver = PlantSolver(cfg, build_scene(cfg, n_plants=4))
    numpy_solver = CPUPlantSolver(cfg, build_scene(cfg, n_plants=4))
    warp_solver.run(90)
    warp_solver.sync()
    numpy_solver.run(90)
    difference = np.abs(warp_solver.positions() - numpy_solver.positions()).max()
    assert difference < 1e-3, f"implementations diverged by {difference:.3e} m"


def test_both_solver_modes_agree():
    """Chain and coloured modes solve the same system, given enough work."""
    results = {}
    for mode, substeps, iterations in [("chain", 64, 8), ("coloured", 64, 64)]:
        cfg = SimConfig()
        cfg.scene.height_jitter = 0.0
        cfg.solver.mode = mode
        cfg.solver.substeps = substeps
        cfg.solver.iterations = iterations
        cfg.wind.gust_amplitude = 0.0
        cfg.wind.turbulence_amplitude = 0.0
        solver = PlantSolver(cfg)
        solver.run(180)
        solver.sync()
        results[mode] = solver.tip_deflection()[0]
    a, b = results["chain"], results["coloured"]
    assert abs(a - b) / max(a, b) < 0.10, f"chain {a:.4f} vs coloured {b:.4f}"


@pytest.mark.slow
def test_matches_beam_theory():
    """Converged deflection must land near qL^4/(8EI)."""
    result = cantilever_self_weight(n_nodes=17, substeps=64, iterations=8)
    assert 0.0 < result.relative_error < 0.25, str(result)


# -------------------------------------------------------------- contacts
def test_robot_contact_pushes_plant_and_reports_force():
    cfg = SimConfig()
    cfg.scene.n_plants = 1
    cfg.scene.height_jitter = 0.0
    cfg.wind.speed = 0.0
    cfg.wind.gust_amplitude = 0.0
    cfg.wind.turbulence_amplitude = 0.0

    solver = PlantSolver(cfg)
    solver.run(30)
    tpl = solver.scene.template
    before = solver.positions()[tpl.stem_path][:, 0].copy()

    # Park a sphere overlapping the stem at mid height and hold it there.
    # Force is sampled across the whole interval rather than at the final
    # frame: a struck stem oscillates, so it makes and breaks contact and any
    # single frame may legitimately read zero.
    peak_force = 0.0
    for _ in range(40):
        solver.set_robot_spheres(np.array([[-0.02, 0.5, 0.0]]), np.array([0.07]))
        solver.step()
        peak_force = max(peak_force, float(np.linalg.norm(solver.reaction_forces()[0])))
    solver.sync()

    after = solver.positions()[tpl.stem_path][:, 0]
    assert after.max() > before.max() + 0.005, "plant was not pushed"
    assert solver.reaction_forces().shape == (1, 3)
    assert peak_force > 1e-3, "contact never reported a reaction force"


def test_no_contact_means_no_force():
    cfg = SimConfig()
    cfg.scene.n_plants = 1
    solver = PlantSolver(cfg)
    solver.set_robot_spheres(np.array([[5.0, 0.5, 5.0]]), np.array([0.05]))
    solver.run(10)
    solver.sync()
    assert np.linalg.norm(solver.reaction_forces()[0]) < 1e-6


# --------------------------------------------------------- Isaac Sim bridge
def test_isaac_bridge_imports_without_isaac_sim():
    """The bridge must be importable outside Isaac Sim.

    Isaac is imported lazily inside the methods that need it. If an `omni` or
    `isaacsim` import ever migrates to module scope, this test fails and the
    whole test suite becomes unrunnable without a 35 GB install.
    """
    from plantsim.isaac_bridge import PlantCanopyBridge

    cfg = SimConfig()
    bridge = PlantCanopyBridge(cfg, n_plants=3)
    assert bridge.n_plants == 3
    assert bridge.n_particles == 3 * bridge.scene.template.n_nodes
    assert "plants" in bridge.summary()


def test_isaac_bridge_solver_is_driveable_without_isaac():
    """Contact input and force output must not depend on Isaac being present."""
    from plantsim.isaac_bridge import PlantCanopyBridge

    cfg = SimConfig()
    cfg.scene.height_jitter = 0.0
    bridge = PlantCanopyBridge(cfg, n_plants=1)

    bridge.set_effector([5.0, 0.5, 5.0], radius=0.05)
    bridge.solver.run(5)
    assert bridge.contact_force_magnitude() < 1e-6, "no contact should mean no force"

    peak = 0.0
    for _ in range(40):
        bridge.set_effector([-0.02, 0.5, 0.0], radius=0.07)
        bridge.solver.step()
        peak = max(peak, bridge.contact_force_magnitude())
    assert peak > 1e-3, "overlapping effector should report a reaction force"


def test_isaac_bridge_usd_methods_need_a_stage():
    """USD calls must fail loudly, not silently no-op, when unbuilt."""
    from plantsim.isaac_bridge import PlantCanopyBridge

    bridge = PlantCanopyBridge(SimConfig(), n_plants=1)
    with pytest.raises(RuntimeError):
        bridge.record_frame()


# ------------------------------------------------------- procedural arm
def test_arm_forward_kinematics_preserves_link_lengths():
    from plantsim.isaac_robot import ProceduralArm

    arm = ProceduralArm()
    for angles in [(0.0, 0.0, 0.0), (0.4, 0.3, -0.8), (-1.2, -0.2, -1.4)]:
        arm.set_joint_angles(*angles)
        origin, elbow, tool = arm.joint_positions()
        assert np.isclose(np.linalg.norm(elbow - origin), arm.upper_length, atol=1e-9)
        assert np.isclose(np.linalg.norm(tool - elbow), arm.fore_length, atol=1e-9)


def test_arm_ik_round_trips_for_reachable_targets():
    """IK must land on the target exactly when the target is in the workspace."""
    from plantsim.isaac_robot import ProceduralArm

    arm = ProceduralArm(base_position=(-0.6, 0.0, 0.0))
    origin = arm.base_position + np.array([0.0, arm.base_height, 0.0])

    for offset in [(0.5, 0.2, 0.0), (0.3, 0.4, 0.2), (0.4, -0.1, -0.3)]:
        target = origin + np.array(offset)
        assert np.linalg.norm(offset) < arm.reach, "test target must be reachable"
        arm.set_tool_target(target)
        assert np.allclose(arm.tool_position(), target, atol=1e-6), (
            f"IK missed {target} by "
            f"{np.linalg.norm(arm.tool_position() - target) * 1000:.2f} mm"
        )


def test_arm_ik_clamps_unreachable_targets_without_raising():
    from plantsim.isaac_robot import ProceduralArm

    arm = ProceduralArm()
    arm.set_tool_target([50.0, 0.5, 0.0])          # far outside the workspace
    origin = arm.base_position + np.array([0.0, arm.base_height, 0.0])
    distance = np.linalg.norm(arm.tool_position() - origin)
    assert distance <= arm.reach + 1e-6, "clamped pose must stay inside reach"
    assert np.isfinite(arm.angles).all()


def test_arm_collision_spheres_overlap():
    """Gaps between proxies would let a stem pass through the arm untouched."""
    from plantsim.isaac_robot import ProceduralArm

    arm = ProceduralArm()
    assert arm.coverage_ok(), (
        f"spacing {arm.sphere_spacing():.3f} m exceeds sphere diameter "
        f"{2 * arm.link_radius:.3f} m"
    )
    centres, radii = arm.collision_spheres()
    assert len(centres) == len(radii) >= 2 * arm.spheres_per_link
    assert np.isfinite(centres).all()

    # Spheres must actually lie on the arm, not drift off it.
    origin, elbow, tool = arm.joint_positions()
    assert np.allclose(centres[-1], tool, atol=1e-9), "last proxy is the tool"


def test_arm_sphere_count_adapts_to_geometry():
    """A longer link must get more proxies, not silently lose coverage."""
    from plantsim.isaac_robot import ProceduralArm

    short = ProceduralArm(upper_length=0.3, fore_length=0.3)
    long = ProceduralArm(upper_length=1.2, fore_length=1.0)
    assert long.spheres_per_link > short.spheres_per_link
    assert short.coverage_ok() and long.coverage_ok()


def test_whole_arm_contact_reports_force_on_multiple_spheres():
    """Multi-sphere contact is the arm's reason to exist; exercise that path."""
    from plantsim.isaac_bridge import PlantCanopyBridge
    from plantsim.isaac_robot import ProceduralArm

    cfg = SimConfig()
    cfg.scene.n_plants = 1
    cfg.scene.height_jitter = 0.0
    cfg.wind.speed = 0.0
    cfg.wind.gust_amplitude = 0.0
    cfg.wind.turbulence_amplitude = 0.0

    bridge = PlantCanopyBridge(cfg, n_plants=1)
    bridge.solver.run(20)

    arm = ProceduralArm(base_position=(-0.55, 0.0, 0.0))
    centres, radii = arm.collision_spheres()

    peak = 0.0
    for step in range(60):
        # Drive the base straight through the plant at the origin.
        arm.set_base_position((-0.55 + 0.01 * step, 0.0, 0.0))
        arm.set_tool_target((-0.05 + 0.01 * step, 0.5, 0.0))
        centres, radii = arm.collision_spheres()
        bridge.set_effectors(centres, radii)
        bridge.solver.step()
        forces = bridge.contact_force()
        assert forces.shape == (len(centres), 3)
        peak = max(peak, float(np.linalg.norm(forces.sum(axis=0))))

    assert peak > 1e-3, "driving an arm through a plant must produce contact"
