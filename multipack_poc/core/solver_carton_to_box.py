"""
Optimisation logic for packing cartons into a single shipping box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    from ortools.sat.python import cp_model  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Google OR-Tools is required for the carton-to-box solver. "
        "Install it via `pip install ortools`."
    ) from exc

from multipack_poc.core.utils_geometry import (
    Placement,
    boxes_overlap,
    footprint_coverage,
    generate_axis_orientations,
    iter_candidate_positions,
    validate_clearances,
    volume_utilization,
)
from multipack_poc.models.box import Box
from multipack_poc.models.package import Package


@dataclass
class CartonBoxPackingResult:
    items_per_box: int
    placements: List[Placement]
    orientation_usage: Dict[str, int]
    volume_utilisation_pct: float
    weight_utilisation_pct: float
    footprint_utilisation_pct: float
    solver_status: str
    fallback_used: bool

    def to_dict(self) -> dict:
        return {
            "items_per_box": self.items_per_box,
            "volume_utilization": self.volume_utilisation_pct,
            "weight_utilization": self.weight_utilisation_pct,
            "footprint_utilization": self.footprint_utilisation_pct,
            "positions": [placement.as_dict() for placement in self.placements],
            "orientation_usage": self.orientation_usage,
            "solver_status": self.solver_status,
            "fallback_used": self.fallback_used,
        }


def pack_cartons_into_box(
    package: Package,
    box: Box,
    time_limit_sec: float = 5.0,
) -> CartonBoxPackingResult:
    """
    Solve the 3D packing problem for cartons within a shipping box using CP-SAT.
    """
    available_weight = box.max_weight - box.tare_weight
    if package.weight > available_weight:
        return CartonBoxPackingResult(
            items_per_box=0,
            placements=[],
            orientation_usage={},
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - package exceeds box weight limit",
            fallback_used=False,
        )

    max_items_by_weight = available_weight // package.weight
    orientations = [
        orientation
        for orientation in generate_axis_orientations(package.dimensions)
        if all(orientation[i] <= box.dimensions[i] for i in range(3))
    ]

    if not orientations or max_items_by_weight == 0:
        return CartonBoxPackingResult(
            items_per_box=0,
            placements=[],
            orientation_usage={},
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - no orientations fit in box",
            fallback_used=False,
        )

    candidates: List[Placement] = []
    for orientation_index, orientation in enumerate(orientations):
        for (x, y, z) in iter_candidate_positions(box.dimensions, orientation):
            candidates.append(
                Placement(
                    x=x,
                    y=y,
                    z=z,
                    orientation=orientation,
                    item_index=orientation_index,
                )
            )

    if not candidates:
        return _fallback_grid_fit(package, box, orientations)

    model = cp_model.CpModel()
    decision_vars: List[cp_model.IntVar] = []
    for idx, candidate in enumerate(candidates):
        decision_vars.append(model.NewBoolVar(f"place_{idx}"))

    # Non-overlap constraints
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if boxes_overlap(
                (candidates[i].x, candidates[i].y, candidates[i].z),
                candidates[i].orientation,
                (candidates[j].x, candidates[j].y, candidates[j].z),
                candidates[j].orientation,
            ):
                model.Add(decision_vars[i] + decision_vars[j] <= 1)

    model.Add(sum(decision_vars) <= max_items_by_weight)
    model.Maximize(sum(decision_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _fallback_grid_fit(package, box, orientations, status_name=solver.StatusName(status))

    placements: List[Placement] = []
    for idx, candidate in enumerate(candidates):
        if solver.Value(decision_vars[idx]):
            placements.append(candidate)

    item_count = len(placements)
    total_weight = item_count * package.weight

    orientation_usage: Dict[str, int] = {}
    for placement in placements:
        key = str(placement.orientation)
        orientation_usage[key] = orientation_usage.get(key, 0) + 1

    footprint_pct = footprint_coverage(placements, box.length, box.width)
    validate_clearances(placements)  # Ensures SciPy-based validation is executed.

    return CartonBoxPackingResult(
        items_per_box=item_count,
        placements=placements,
        orientation_usage=orientation_usage,
        volume_utilisation_pct=volume_utilization(item_count * package.volume, box.inner_volume),
        weight_utilisation_pct=volume_utilization(total_weight, available_weight),
        footprint_utilisation_pct=footprint_pct,
        solver_status=solver.StatusName(status),
        fallback_used=False,
    )


def _fallback_grid_fit(
    package: Package,
    box: Box,
    orientations: List[tuple[int, int, int]],
    status_name: str = "Fallback",
) -> CartonBoxPackingResult:
    """
    Simple heuristic placing cartons in a uniform grid using the best orientation.
    """
    available_weight = box.max_weight - box.tare_weight
    best_count = 0
    best_orientation = orientations[0]

    for orientation in orientations:
        count = (
            (box.length // orientation[0])
            * (box.width // orientation[1])
            * (box.height // orientation[2])
        )
        if count > best_count:
            best_count = count
            best_orientation = orientation

    if best_count == 0:
        return CartonBoxPackingResult(
            items_per_box=0,
            placements=[],
            orientation_usage={},
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status=status_name,
            fallback_used=True,
        )

    max_items_by_weight = available_weight // package.weight
    item_count = min(best_count, max_items_by_weight)
    placements: List[Placement] = []
    index = 0

    for (x, y, z) in iter_candidate_positions(box.dimensions, best_orientation):
        if index >= item_count:
            break
        placements.append(
            Placement(
                x=x,
                y=y,
                z=z,
                orientation=best_orientation,
                item_index=0,
            )
        )
        index += 1

    total_weight = item_count * package.weight
    orientation_usage = {str(best_orientation): item_count}
    footprint_pct = footprint_coverage(placements, box.length, box.width)
    validate_clearances(placements)

    return CartonBoxPackingResult(
        items_per_box=item_count,
        placements=placements,
        orientation_usage=orientation_usage,
        volume_utilisation_pct=volume_utilization(item_count * package.volume, box.inner_volume),
        weight_utilisation_pct=volume_utilization(total_weight, available_weight),
        footprint_utilisation_pct=footprint_pct,
        solver_status=status_name,
        fallback_used=True,
    )

