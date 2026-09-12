"""Configuration dataclasses for the plant biomechanics simulator.

All physical quantities are SI units (metres, kilograms, seconds, newtons).
Keeping them SI matters: the numbers below can be checked against published
values for real crops, which is what makes the simulation defensible in a
report rather than a pile of tuned magic constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StemParams:
    """Geometry and material properties of a single tapered stem.

    The stem is a cantilever beam clamped at the soil. Its flexural rigidity
    is EI, where E is Young's modulus and I = pi*r^4/4 is the second moment of
    area of a circular cross-section. Because I scales with r^4 and the radius
    tapers from base to tip, rigidity falls off very fast with height -- this
    is the physical origin of "stiffness inversely proportional to distance
    from the ground", not an artificial rule.
    """

    # Defaults describe a row crop with a slender stalk -- canola or a staked
    # tomato at working height, the kind of plant a selective-harvesting arm
    # actually pushes through. A maize stalk is far stouter (radius_base
    # ~0.009 m) and, correctly, barely moves in a breeze: EI goes as r^4, so
    # doubling the base radius makes the plant 16x harder to bend.
    n_nodes: int = 16
    length: float = 0.85          # m, total stem length
    radius_base: float = 0.005    # m, 5 mm at the soil
    radius_tip: float = 0.002     # m, 2 mm at the growing tip
    youngs_modulus: float = 1.0e9 # Pa, green herbaceous stem tissue (~1 GPa)
    density: float = 700.0        # kg/m^3, fresh plant tissue (mostly water + fibre)
    drag_coefficient: float = 1.1 # dimensionless, bluff cylinder in cross-flow

    @property
    def segment_length(self) -> float:
        return self.length / (self.n_nodes - 1)


@dataclass
class LeafParams:
    """Leaves modelled as short 2-node blades hanging off stem nodes.

    Leaves carry most of the aerodynamic load on a real plant (large area,
    negligible mass) which is why they dominate the wind response.
    """

    enabled: bool = True
    nodes_per_leaf: int = 3       # chain length of each leaf, including the joint node
    length: float = 0.14          # m, leaf blade length
    width: float = 0.05           # m, mean blade width (sets drag area)
    mass: float = 0.0025          # kg, per leaf
    youngs_modulus: float = 1.0e8 # Pa, leaf lamina is ~10x more compliant than stem
    thickness: float = 0.0004     # m, leaf lamina thickness
    drag_coefficient: float = 1.4 # flat plate normal to flow
    first_node: int = 4           # lowest stem node that carries a leaf
    stride: int = 3               # attach a leaf every `stride` stem nodes
    alternation: float = 2.399963 # rad, golden-angle phyllotaxis between leaves


@dataclass
class WindParams:
    """Turbulent wind field driving the canopy.

    Wind enters the simulation as aerodynamic *drag*, not as a constant
    acceleration. This distinction is the whole reason the plant reaches an
    equilibrium bend instead of accelerating until it lies flat: drag depends
    on the velocity of the air *relative to the moving plant*, so it falls to
    zero as the plant gives way, and balances against the restoring bending
    stiffness.
    """

    speed: float = 4.0            # m/s, mean wind at reference height
    direction: tuple = (1.0, 0.0, 0.0)
    reference_height: float = 1.0 # m, height at which `speed` is defined
    shear_exponent: float = 0.20  # power-law boundary layer: u(y) = u_ref*(y/y_ref)^a
    gust_amplitude: float = 0.35  # fraction of mean speed
    gust_frequency: float = 0.45  # Hz, dominant gust timescale
    turbulence_amplitude: float = 0.18
    turbulence_frequency: float = 2.7  # Hz, small-scale buffeting
    turbulence_wavelength: float = 1.6 # m, spatial decorrelation of gusts
    air_density: float = 1.225    # kg/m^3


@dataclass
class SolverParams:
    """XPBD solver settings.

    XPBD (extended position-based dynamics) is used rather than plain PBD
    because plain PBD's effective stiffness depends on the iteration count --
    change the iterations and the plant silently changes material. XPBD
    introduces a compliance term so stiffness is a property of the plant and
    the iteration count only affects convergence.
    """

    mode: str = "chain"           # "chain" (per-plant sweep) or "coloured"
    dt: float = 1.0 / 60.0        # s, frame time
    # Substeps, not iterations, are what make XPBD converge on a chain: a
    # measured refinement study (see Documentation/VALIDATION.md) puts these
    # defaults within 0.5% of the fully converged tip deflection, whereas
    # substeps=8/iterations=2 leaves the stem 4x too soft.
    substeps: int = 32            # substeps per frame
    iterations: int = 4           # constraint iterations per substep
    contact_iterations: int = 4   # XPBD contact projections per substep
    # Compliance of the manipulator's contact surface, m/N. A real gripper has
    # a padded jaw rather than a diamond-hard one, and that finite stiffness
    # bounds the peak of an impact. 1e-5 m/N is 100 N/mm -- a firm rubber pad,
    # which penetrates ~50 um under a 5 N load.
    contact_compliance: float = 1.0e-5
    # Plant tissue is strongly viscoelastic: a stem displaced and released
    # settles within about a second rather than ringing like a spring. For a
    # ~1.5 Hz stem a damping ratio of ~0.1 corresponds to a rate of 2*zeta*w
    # ~ 1.9/s. Aerodynamic drag supplies some of that already, so the tissue
    # term carries the rest.
    velocity_damping: float = 1.2    # 1/s, internal viscoelastic damping of tissue
    ground_height: float = 0.0    # m
    ground_friction: float = 0.6
    max_velocity: float = 25.0    # m/s, safety clamp against blow-up


@dataclass
class SceneParams:
    """A field of plants."""

    n_plants: int = 1
    spacing: float = 0.35         # m, between plants along a row
    row_spacing: float = 0.60     # m, between rows
    plants_per_row: int = 10
    height_jitter: float = 0.12   # fractional variation in stem length
    seed: int = 7


@dataclass
class SimConfig:
    stem: StemParams = field(default_factory=StemParams)
    leaf: LeafParams = field(default_factory=LeafParams)
    wind: WindParams = field(default_factory=WindParams)
    solver: SolverParams = field(default_factory=SolverParams)
    scene: SceneParams = field(default_factory=SceneParams)
    gravity: tuple = (0.0, -9.81, 0.0)
    device: str | None = None     # None -> auto-select cuda:0 if present, else cpu
