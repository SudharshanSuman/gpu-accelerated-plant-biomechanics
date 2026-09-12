"""Builds the particle/constraint graph for a plant and for a field of plants.

Two ideas carry most of the weight here.

1.  Compliance is derived from beam theory, not tuned by hand. A stem segment
    of length L and radius r has axial stiffness EA/L and flexural rigidity
    EI with I = pi*r^4/4. Because I goes as r^4 and the radius tapers toward
    the tip, a stem that is 3.6x thicker at the base than the tip is roughly
    170x stiffer there. "Rigidity inversely proportional to distance from the
    ground" therefore falls out of the geometry instead of being imposed.

2.  Constraints are graph-coloured once, at build time. Within one colour no
    two constraints touch a shared particle, so every constraint in a colour
    can be projected in parallel without write conflicts. This is what lets
    the whole solve stay resident on the GPU -- the naive alternative is to
    walk the chain root-to-tip on the CPU, which serialises the solver and
    throws away the speed-up the project exists to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig

EPS = 1e-12


# --------------------------------------------------------------------------
# Constraint colouring
# --------------------------------------------------------------------------
def greedy_colour(constraint_nodes: np.ndarray, n_nodes: int) -> list[np.ndarray]:
    """Partition constraints into conflict-free groups.

    `constraint_nodes` is (C, k) -- the k particles each constraint touches.
    Returns a list of index arrays, one per colour. Greedy colouring is not
    optimal, but for chain-like graphs it lands on the minimum (2 for stretch,
    3 for bending) and the cost is paid once at build time.
    """
    if len(constraint_nodes) == 0:
        return []

    node_to_constraints: list[list[int]] = [[] for _ in range(n_nodes)]
    for ci, nodes in enumerate(constraint_nodes):
        for n in nodes:
            node_to_constraints[int(n)].append(ci)

    colour_of = np.full(len(constraint_nodes), -1, dtype=np.int32)
    for ci, nodes in enumerate(constraint_nodes):
        used = set()
        for n in nodes:
            for other in node_to_constraints[int(n)]:
                if other != ci and colour_of[other] >= 0:
                    used.add(int(colour_of[other]))
        c = 0
        while c in used:
            c += 1
        colour_of[ci] = c

    n_colours = int(colour_of.max()) + 1
    return [np.where(colour_of == c)[0].astype(np.int32) for c in range(n_colours)]


def verify_colouring(constraint_nodes: np.ndarray, colours: list[np.ndarray]) -> None:
    """Assert no colour contains two constraints sharing a particle.

    A silent colouring bug would show up only as a subtly too-soft plant under
    high thread counts, which is exactly the kind of defect that survives to
    the results table. Cheap to check, so always check.
    """
    for c, idx in enumerate(colours):
        touched = constraint_nodes[idx].reshape(-1)
        if len(np.unique(touched)) != len(touched):
            raise AssertionError(f"colour {c} has constraints sharing a particle")


# --------------------------------------------------------------------------
# Template: one plant, unit scale
# --------------------------------------------------------------------------
@dataclass
class PlantTemplate:
    """One plant's topology at nominal size, in local coordinates."""

    positions: np.ndarray      # (N, 3) local, root at origin
    mass: np.ndarray           # (N,)
    pinned: np.ndarray         # (N,) bool -- root particles anchored in soil
    radius: np.ndarray         # (N,) m, for rendering and drag
    drag_area: np.ndarray      # (N,) m^2, tributary area facing the flow
    drag_cd: np.ndarray        # (N,)
    height_param: np.ndarray   # (N,) 0 at soil, 1 at stem tip

    stretch_nodes: np.ndarray      # (S, 2) int32
    stretch_rest: np.ndarray       # (S,)
    stretch_compliance: np.ndarray # (S,) m/N

    bend_nodes: np.ndarray         # (B, 3) int32 -- (a, b, c), b is the hinge
    bend_rest: np.ndarray          # (B,) rest magnitude of b - (a+c)/2
    bend_compliance: np.ndarray    # (B,)

    stem_path: np.ndarray          # node indices along the stem, for plotting
    leaf_paths: list               # list of node-index arrays

    @property
    def n_nodes(self) -> int:
        return len(self.positions)


def build_template(cfg: SimConfig) -> PlantTemplate:
    stem, leaf = cfg.stem, cfg.leaf
    n = stem.n_nodes
    seg = stem.segment_length

    positions = [np.array([0.0, i * seg, 0.0]) for i in range(n)]
    height_param = [i / (n - 1) for i in range(n)]
    radius = [stem.radius_base + (stem.radius_tip - stem.radius_base) * s
              for s in height_param]
    mass = [0.0] * n
    drag_area = [0.0] * n
    drag_cd = [stem.drag_coefficient] * n
    pinned = [False] * n
    pinned[0] = True

    stretch_nodes, stretch_rest, stretch_compliance = [], [], []
    bend_nodes, bend_rest, bend_compliance = [], [], []

    # --- stem segments: lumped mass, projected drag area, axial compliance ---
    for i in range(n - 1):
        r_mid = 0.5 * (radius[i] + radius[i + 1])
        m_seg = stem.density * np.pi * r_mid ** 2 * seg
        mass[i] += 0.5 * m_seg
        mass[i + 1] += 0.5 * m_seg

        a_seg = 2.0 * r_mid * seg            # diameter x length, seen edge-on by wind
        drag_area[i] += 0.5 * a_seg
        drag_area[i + 1] += 0.5 * a_seg

        area = np.pi * radius[i] ** 2
        stretch_nodes.append((i, i + 1))
        stretch_rest.append(seg)
        stretch_compliance.append(seg / (stem.youngs_modulus * area))

    # --- stem bending: hinge at each interior node, stiffness from EI ---
    for i in range(1, n - 1):
        second_moment = np.pi * radius[i] ** 4 / 4.0
        ei = stem.youngs_modulus * second_moment
        bend_nodes.append((i - 1, i, i + 1))
        d = positions[i] - 0.5 * (positions[i - 1] + positions[i + 1])
        bend_rest.append(float(np.linalg.norm(d)))
        bend_compliance.append(seg ** 3 / (4.0 * ei))

    # --- leaves ---
    leaf_paths: list = []
    if leaf.enabled and leaf.nodes_per_leaf >= 2:
        leaf_seg = leaf.length / (leaf.nodes_per_leaf - 1)
        leaf_area = leaf.width * leaf.thickness
        leaf_i = leaf.width * leaf.thickness ** 3 / 12.0
        alpha_stretch_leaf = leaf_seg / (leaf.youngs_modulus * leaf_area)
        alpha_bend_leaf = leaf_seg ** 3 / (4.0 * leaf.youngs_modulus * leaf_i)
        m_node = leaf.mass / (leaf.nodes_per_leaf - 1)

        k = 0
        for attach in range(leaf.first_node, n - 1, leaf.stride):
            angle = k * leaf.alternation
            # Blades rise at ~25 degrees and spread radially around the stem.
            out = np.array([np.cos(angle) * np.cos(np.deg2rad(25.0)),
                            np.sin(np.deg2rad(25.0)),
                            np.sin(angle) * np.cos(np.deg2rad(25.0))])
            path = [attach]
            prev = attach
            for j in range(1, leaf.nodes_per_leaf):
                idx = len(positions)
                positions.append(positions[attach] + out * leaf_seg * j)
                height_param.append(height_param[attach])
                radius.append(0.5 * leaf.width)
                mass.append(m_node)
                drag_area.append(leaf.width * leaf_seg)
                drag_cd.append(leaf.drag_coefficient)
                pinned.append(False)

                stretch_nodes.append((prev, idx))
                stretch_rest.append(leaf_seg)
                stretch_compliance.append(alpha_stretch_leaf)
                path.append(idx)
                prev = idx

            # Bending along the blade.
            for j in range(1, len(path) - 1):
                a, b, c = path[j - 1], path[j], path[j + 1]
                bend_nodes.append((a, b, c))
                d = positions[b] - 0.5 * (positions[a] + positions[c])
                bend_rest.append(float(np.linalg.norm(d)))
                bend_compliance.append(alpha_bend_leaf)

            # Petiole: holds the leaf's attachment angle to the stem. Without
            # it the blade is a free hinge and windmills around the stem node.
            bend_nodes.append((attach - 1, attach, path[1]))
            d = positions[attach] - 0.5 * (positions[attach - 1] + positions[path[1]])
            bend_rest.append(float(np.linalg.norm(d)))
            bend_compliance.append(alpha_bend_leaf * 0.25)

            leaf_paths.append(np.array(path, dtype=np.int32))
            k += 1

    # --- soil clamp -------------------------------------------------------
    # A pinned root particle fixes the stem's *position* but not its
    # *direction*: every bending constraint is still satisfied when the whole
    # stem rotates rigidly about that one point, so the plant topples like a
    # felled pole regardless of how stiff the material is. A real stem is
    # clamped by the soil, which fixes position and tangent both. That second
    # condition needs a second fixed reference, so a pinned anchor particle is
    # buried one segment below the root and a bending constraint is hinged at
    # the root between the anchor and the first free node.
    anchor = len(positions)
    positions.append(np.array([0.0, -seg, 0.0]))
    height_param.append(0.0)
    radius.append(stem.radius_base)
    mass.append(1.0)          # ignored: pinned particles get zero inverse mass
    drag_area.append(0.0)
    drag_cd.append(0.0)
    pinned.append(True)

    second_moment_base = np.pi * radius[0] ** 4 / 4.0
    bend_nodes.append((anchor, 0, 1))
    bend_rest.append(0.0)     # anchor, root and node 1 are collinear at rest
    bend_compliance.append(seg ** 3 / (4.0 * stem.youngs_modulus * second_moment_base))

    stretch_nodes_a = np.array(stretch_nodes, dtype=np.int32).reshape(-1, 2)
    bend_nodes_a = np.array(bend_nodes, dtype=np.int32).reshape(-1, 3)

    return PlantTemplate(
        positions=np.array(positions, dtype=np.float64),
        mass=np.array(mass, dtype=np.float64),
        pinned=np.array(pinned, dtype=bool),
        radius=np.array(radius, dtype=np.float64),
        drag_area=np.array(drag_area, dtype=np.float64),
        drag_cd=np.array(drag_cd, dtype=np.float64),
        height_param=np.array(height_param, dtype=np.float64),
        stretch_nodes=stretch_nodes_a,
        stretch_rest=np.array(stretch_rest, dtype=np.float64),
        stretch_compliance=np.array(stretch_compliance, dtype=np.float64),
        bend_nodes=bend_nodes_a,
        bend_rest=np.array(bend_rest, dtype=np.float64),
        bend_compliance=np.array(bend_compliance, dtype=np.float64),
        stem_path=np.arange(n, dtype=np.int32),
        leaf_paths=leaf_paths,
    )


# --------------------------------------------------------------------------
# Scene: many plants, tiled from the template
# --------------------------------------------------------------------------
@dataclass
class Scene:
    """A whole field, flattened into the arrays the solver consumes."""

    positions: np.ndarray
    inv_mass: np.ndarray
    radius: np.ndarray
    drag_area: np.ndarray
    drag_cd: np.ndarray
    height_param: np.ndarray
    node_plant: np.ndarray

    stretch_nodes: np.ndarray
    stretch_rest: np.ndarray
    stretch_compliance: np.ndarray
    stretch_colours: list

    bend_nodes: np.ndarray
    bend_rest: np.ndarray
    bend_compliance: np.ndarray
    bend_colours: list

    template: PlantTemplate
    n_plants: int

    @property
    def n_nodes(self) -> int:
        return len(self.positions)

    @property
    def n_constraints(self) -> int:
        return len(self.stretch_nodes) + len(self.bend_nodes)

    def plant_slice(self, p: int) -> slice:
        k = self.template.n_nodes
        return slice(p * k, (p + 1) * k)

    def summary(self) -> str:
        return (
            f"{self.n_plants} plants | {self.n_nodes} particles | "
            f"{len(self.stretch_nodes)} stretch + {len(self.bend_nodes)} bend "
            f"constraints | colours: {len(self.stretch_colours)} stretch, "
            f"{len(self.bend_colours)} bend"
        )


def build_scene(cfg: SimConfig, n_plants: int | None = None) -> Scene:
    """Tile the template into a field.

    The template's colouring is computed once and reused for every plant: all
    plants share a topology, so constraint `c` of plant `p` conflicts with the
    same neighbours as constraint `c` of plant 0. Colouring the field directly
    would cost O(total constraints) in Python and dominate setup time for the
    large benchmark scenes.
    """
    tpl = build_template(cfg)
    scene_cfg = cfg.scene
    n_plants = scene_cfg.n_plants if n_plants is None else n_plants
    k = tpl.n_nodes

    tpl_stretch_colours = greedy_colour(tpl.stretch_nodes, k)
    tpl_bend_colours = greedy_colour(tpl.bend_nodes, k)
    verify_colouring(tpl.stretch_nodes, tpl_stretch_colours)
    verify_colouring(tpl.bend_nodes, tpl_bend_colours)

    rng = np.random.default_rng(scene_cfg.seed)
    scale = 1.0 + scene_cfg.height_jitter * rng.uniform(-1.0, 1.0, size=n_plants)
    yaw = rng.uniform(0.0, 2.0 * np.pi, size=n_plants)

    positions = np.empty((n_plants * k, 3), dtype=np.float64)
    inv_mass = np.empty(n_plants * k, dtype=np.float64)
    radius = np.empty(n_plants * k, dtype=np.float64)
    drag_area = np.empty(n_plants * k, dtype=np.float64)
    drag_cd = np.tile(tpl.drag_cd, n_plants)
    height_param = np.tile(tpl.height_param, n_plants)
    node_plant = np.repeat(np.arange(n_plants, dtype=np.int32), k)

    n_s, n_b = len(tpl.stretch_nodes), len(tpl.bend_nodes)
    stretch_nodes = np.empty((n_plants * n_s, 2), dtype=np.int32)
    stretch_rest = np.empty(n_plants * n_s, dtype=np.float64)
    stretch_compliance = np.empty(n_plants * n_s, dtype=np.float64)
    bend_nodes = np.empty((n_plants * n_b, 3), dtype=np.int32)
    bend_rest = np.empty(n_plants * n_b, dtype=np.float64)
    bend_compliance = np.empty(n_plants * n_b, dtype=np.float64)

    for p in range(n_plants):
        f = scale[p]
        row, col = divmod(p, max(1, scene_cfg.plants_per_row))
        base = np.array([col * scene_cfg.spacing, 0.0, row * scene_cfg.row_spacing])

        c, s = np.cos(yaw[p]), np.sin(yaw[p])
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

        sl = slice(p * k, (p + 1) * k)
        positions[sl] = (tpl.positions * f) @ rot.T + base
        # Mass scales with volume (f^3); pinned roots keep zero inverse mass.
        m = tpl.mass * f ** 3
        inv_mass[sl] = np.where(tpl.pinned, 0.0, 1.0 / np.maximum(m, EPS))
        radius[sl] = tpl.radius * f
        drag_area[sl] = tpl.drag_area * f ** 2

        cs = slice(p * n_s, (p + 1) * n_s)
        stretch_nodes[cs] = tpl.stretch_nodes + p * k
        stretch_rest[cs] = tpl.stretch_rest * f
        # alpha = L/(EA): length scales as f, area as f^2, so alpha as 1/f.
        stretch_compliance[cs] = tpl.stretch_compliance / f

        cb = slice(p * n_b, (p + 1) * n_b)
        bend_nodes[cb] = tpl.bend_nodes + p * k
        bend_rest[cb] = tpl.bend_rest * f
        # alpha = L^3/(4EI): length^3 scales as f^3, I as f^4, so alpha as 1/f.
        bend_compliance[cb] = tpl.bend_compliance / f

    def tile_colours(tpl_colours, per_plant, n_plants):
        return [
            np.concatenate([idx + p * per_plant for p in range(n_plants)]).astype(np.int32)
            for idx in tpl_colours
        ]

    return Scene(
        positions=positions,
        inv_mass=inv_mass,
        radius=radius,
        drag_area=drag_area,
        drag_cd=drag_cd,
        height_param=height_param,
        node_plant=node_plant,
        stretch_nodes=stretch_nodes,
        stretch_rest=stretch_rest,
        stretch_compliance=stretch_compliance,
        stretch_colours=tile_colours(tpl_stretch_colours, n_s, n_plants),
        bend_nodes=bend_nodes,
        bend_rest=bend_rest,
        bend_compliance=bend_compliance,
        bend_colours=tile_colours(tpl_bend_colours, n_b, n_plants),
        template=tpl,
        n_plants=n_plants,
    )
