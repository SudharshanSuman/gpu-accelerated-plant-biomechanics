"""A procedural manipulator for canopy-interaction experiments.

Isaac Sim ships no robot meshes in the install itself -- Franka, UR10 and the
rest live on NVIDIA's content server. A demo that needs a download is a demo
that fails on an offline machine and in CI, so the arm here is built from USD
primitives at runtime and driven by closed-form forward kinematics.

The point of the arm is not fidelity to a particular product. It is to give
the plant solver a *swept volume* rather than a single point: a real
manipulator pushes the canopy with its whole forearm, and the contact force a
planner cares about is the sum over everything that touches. Each link is
therefore sampled into several collision spheres.

Joint convention (all angles in radians)::

    yaw       rotation about +Y, i.e. which way the arm points in plan
    shoulder  elevation of the upper arm from horizontal
    elbow     bend between upper arm and forearm (0 = straight)
"""

from __future__ import annotations

import numpy as np


class ProceduralArm:
    """Two-link arm on a yaw base, with sphere-sampled collision proxies."""

    def __init__(self,
                 base_position=(-0.9, 0.0, 0.0),
                 upper_length: float = 0.50,
                 fore_length: float = 0.45,
                 link_radius: float = 0.045,
                 tool_radius: float = 0.06,
                 base_height: float = 0.35,
                 spheres_per_link: int | None = None):
        self.base_position = np.asarray(base_position, dtype=np.float64)
        self.upper_length = upper_length
        self.fore_length = fore_length
        self.link_radius = link_radius
        self.tool_radius = tool_radius
        self.base_height = base_height

        # Proxies must overlap, or a stem can pass between two spheres and
        # register no contact at all. Pick the count from the geometry rather
        # than trusting a hardcoded default to stay valid when links change.
        if spheres_per_link is None:
            longest = max(upper_length, fore_length)
            spheres_per_link = int(np.ceil(longest / link_radius))
        self.spheres_per_link = max(2, int(spheres_per_link))

        self.angles = np.zeros(3)          # yaw, shoulder, elbow
        self._prims = {}
        self._stage = None
        self._root = None

    # ------------------------------------------------------------------
    @property
    def reach(self) -> float:
        return self.upper_length + self.fore_length

    def set_joint_angles(self, yaw: float, shoulder: float, elbow: float):
        self.angles = np.array([yaw, shoulder, elbow], dtype=np.float64)
        return self

    def set_base_position(self, position):
        """Move the whole arm. Models a field robot driving down the row.

        A real agricultural manipulator does not sweep a long arc through the
        canopy; it holds a reach pose while the vehicle drives. That makes the
        vehicle speed the parameter governing contact force, which is the one
        an operator actually controls.
        """
        self.base_position = np.asarray(position, dtype=np.float64)
        return self

    def solve_ik(self, target) -> tuple:
        """Closed-form 2R inverse kinematics for a world-space tool target.

        Returns (yaw, shoulder, elbow). Targets beyond reach are clamped to
        the workspace boundary rather than raising, so a trajectory that
        strays slightly outside still produces a sane, fully extended pose.
        """
        target = np.asarray(target, dtype=np.float64)
        origin = self.base_position + np.array([0.0, self.base_height, 0.0])
        delta = target - origin

        yaw = float(np.arctan2(delta[2], delta[0]))
        reach_in_plane = float(np.hypot(delta[0], delta[2]))
        height = float(delta[1])

        distance = float(np.hypot(reach_in_plane, height))
        limit = 0.999 * self.reach
        if distance > limit:
            scale = limit / max(distance, 1e-9)
            reach_in_plane *= scale
            height *= scale
            distance = limit
        distance = max(distance, abs(self.upper_length - self.fore_length) + 1e-4)

        l1, l2 = self.upper_length, self.fore_length
        cos_elbow = (distance ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        elbow = -float(np.arccos(np.clip(cos_elbow, -1.0, 1.0)))   # elbow-down

        shoulder = float(
            np.arctan2(height, reach_in_plane)
            - np.arctan2(l2 * np.sin(elbow), l1 + l2 * np.cos(elbow))
        )
        return yaw, shoulder, elbow

    def set_tool_target(self, target):
        """Drive the arm so the tool reaches a world-space point."""
        return self.set_joint_angles(*self.solve_ik(target))

    def joint_positions(self) -> np.ndarray:
        """Forward kinematics: shoulder, elbow and tool positions in world space.

        The arm works in a vertical plane that is rotated about +Y by `yaw`,
        which keeps the kinematics closed-form and easy to check by hand.
        """
        yaw, shoulder, elbow = self.angles
        plane = np.array([np.cos(yaw), 0.0, np.sin(yaw)])   # in-plane horizontal
        up = np.array([0.0, 1.0, 0.0])

        origin = self.base_position + up * self.base_height

        upper_dir = plane * np.cos(shoulder) + up * np.sin(shoulder)
        elbow_pos = origin + upper_dir * self.upper_length

        total = shoulder + elbow
        fore_dir = plane * np.cos(total) + up * np.sin(total)
        tool_pos = elbow_pos + fore_dir * self.fore_length

        return np.stack([origin, elbow_pos, tool_pos])

    def tool_position(self) -> np.ndarray:
        return self.joint_positions()[-1]

    # ------------------------------------------------------------------
    def collision_spheres(self):
        """Sphere proxies covering the whole arm, not just the tool.

        Returns (centres, radii). Spheres are spaced along each link so that
        a stem cannot slip between two of them: the spacing is under one
        sphere diameter for the default sampling.
        """
        origin, elbow, tool = self.joint_positions()
        centres, radii = [], []

        for start, end in ((origin, elbow), (elbow, tool)):
            # Skip t=0 on the second link so the elbow is not double-counted.
            for t in np.linspace(0.0, 1.0, self.spheres_per_link + 1)[1:]:
                centres.append(start + (end - start) * t)
                radii.append(self.link_radius)

        centres.append(tool)
        radii.append(self.tool_radius)
        return np.asarray(centres, dtype=np.float32), np.asarray(radii, dtype=np.float32)

    def sphere_spacing(self) -> float:
        """Gap between consecutive proxies, for checking coverage."""
        return max(self.upper_length, self.fore_length) / self.spheres_per_link

    def coverage_ok(self) -> bool:
        """True when consecutive spheres overlap, leaving no gap for a stem."""
        return self.sphere_spacing() < 2.0 * self.link_radius

    # ------------------------------------------------------------------
    def build(self, stage, root_path: str = "/World/Arm"):
        """Author the arm as USD capsules plus a base cylinder and tool sphere."""
        from pxr import Gf, Sdf, UsdGeom, Vt

        self._stage = stage
        self._root = root_path
        UsdGeom.Xform.Define(stage, Sdf.Path(root_path))

        pedestal = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_path}/Base"))
        pedestal.CreateAxisAttr().Set(UsdGeom.Tokens.y)
        pedestal.CreateHeightAttr().Set(float(self.base_height))
        pedestal.CreateRadiusAttr().Set(float(self.link_radius * 1.6))
        pedestal.CreateDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(0.25, 0.27, 0.31)])
        )
        base_xform = UsdGeom.Xformable(pedestal)
        base_xform.AddTranslateOp().Set(
            Gf.Vec3d(*(self.base_position + np.array([0.0, self.base_height / 2, 0.0])))
        )

        for name, radius, colour in (
            ("UpperArm", self.link_radius, (0.24, 0.42, 0.70)),
            ("Forearm", self.link_radius * 0.85, (0.30, 0.50, 0.78)),
        ):
            capsule = UsdGeom.Capsule.Define(stage, Sdf.Path(f"{root_path}/{name}"))
            capsule.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            capsule.CreateRadiusAttr().Set(float(radius))
            capsule.CreateHeightAttr().Set(1.0)     # rewritten every update
            capsule.CreateDisplayColorAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(*colour)])
            )
            xform = UsdGeom.Xformable(capsule)
            self._prims[name] = {
                "prim": capsule,
                "translate": xform.AddTranslateOp(),
                "orient": xform.AddOrientOp(),
            }

        tool = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{root_path}/Tool"))
        tool.CreateRadiusAttr().Set(float(self.tool_radius))
        tool.CreateDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(0.85, 0.42, 0.20)])
        )
        self._prims["Tool"] = {
            "prim": tool,
            "translate": UsdGeom.Xformable(tool).AddTranslateOp(),
        }
        self.update_usd()
        return self

    # ------------------------------------------------------------------
    def update_usd(self, time_code=None):
        """Move the USD prims to match the current joint angles."""
        if not self._prims:
            return
        from pxr import Gf

        origin, elbow, tool = self.joint_positions()

        for name, (start, end) in (("UpperArm", (origin, elbow)),
                                   ("Forearm", (elbow, tool))):
            entry = self._prims[name]
            segment = end - start
            length = float(np.linalg.norm(segment))
            midpoint = 0.5 * (start + end)

            entry["prim"].GetHeightAttr().Set(length)
            # A USD capsule with axis=Z runs along +Z, so the link is oriented
            # by the rotation taking +Z onto the link direction.
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1),
                                   Gf.Vec3d(*(segment / max(length, 1e-9))))
            quat = Gf.Quatf(rotation.GetQuat())

            if time_code is None:
                entry["translate"].Set(Gf.Vec3d(*midpoint))
                entry["orient"].Set(quat)
            else:
                entry["translate"].Set(Gf.Vec3d(*midpoint), time_code)
                entry["orient"].Set(quat, time_code)

        tool_entry = self._prims["Tool"]
        if time_code is None:
            tool_entry["translate"].Set(Gf.Vec3d(*tool))
        else:
            tool_entry["translate"].Set(Gf.Vec3d(*tool), time_code)

    # ------------------------------------------------------------------
    def sweep_pose(self, t: float, period: float = 6.0,
                   shoulder_range=(0.35, -0.10), elbow_range=(-0.9, -0.35),
                   yaw_range=(-0.5, 0.5)):
        """A scripted reach-and-sweep trajectory through the canopy.

        Deliberately simple and deterministic: the experiment being run is
        about contact force versus approach speed, so the trajectory needs to
        be repeatable, not clever.
        """
        phase = np.clip(t / period, 0.0, 1.0)
        eased = 0.5 - 0.5 * np.cos(np.pi * phase)      # smooth start and stop

        yaw = yaw_range[0] + (yaw_range[1] - yaw_range[0]) * eased
        shoulder = shoulder_range[0] + (shoulder_range[1] - shoulder_range[0]) * eased
        elbow = elbow_range[0] + (elbow_range[1] - elbow_range[0]) * eased
        return yaw, shoulder, elbow
