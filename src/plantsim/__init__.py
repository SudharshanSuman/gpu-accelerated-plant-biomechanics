"""GPU-accelerated plant biomechanics for agricultural robot simulation.

Public surface:

    SimConfig, StemParams, LeafParams, WindParams, SolverParams, SceneParams
    build_scene, build_template
    PlantSolver       -- Warp XPBD solver (GPU when a CUDA driver is present)
    CPUPlantSolver    -- NumPy reference of the same algorithm
"""

from .config import (
    LeafParams,
    SceneParams,
    SimConfig,
    SolverParams,
    StemParams,
    WindParams,
)
from .cpu_reference import CPUPlantSolver
from .device import describe_device, device_report, resolve_device
from .isaac_bridge import PlantCanopyBridge
from .solver import PlantSolver
from .topology import Scene, build_scene, build_template

__all__ = [
    "SimConfig", "StemParams", "LeafParams", "WindParams", "SolverParams",
    "SceneParams", "Scene", "build_scene", "build_template",
    "PlantSolver", "CPUPlantSolver", "PlantCanopyBridge",
    "resolve_device", "describe_device", "device_report",
]

__version__ = "0.1.0"
