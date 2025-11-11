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


DIMENSION_SCALE = 100  # work in hundredths of a millimetre for solver precision


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

    scale = DIMENSION_SCALE
    box_dims = box.scaled_dimensions(scale)
    package_dims = package.scaled_dimensions(scale)

    max_items_by_weight = int(available_weight // package.weight)
    orientations = [
        orientation
        for orientation in generate_axis_orientations(package_dims)
        if all(orientation[i] <= box_dims[i] for i in range(3))
    ]
    orientations = _respect_thickness_constraint(package, orientations, scale)

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

    if package.thickness_axis is not None:
        sheet_result = _pack_sheet_cartons(
            package,
            box,
            orientations,
            max_items_by_weight,
            scale,
            box_dims,
        )
        if sheet_result is not None:
            return sheet_result

    candidates = _generate_candidate_placements(box_dims, orientations)

    if not candidates:
        return _fallback_grid_fit(package, box, orientations, scale, box_dims)

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
        return _fallback_grid_fit(
            package,
            box,
            orientations,
            scale,
            box_dims,
            status_name=solver.StatusName(status),
        )

    placements_scaled: List[Placement] = []
    for idx, candidate in enumerate(candidates):
        if solver.Value(decision_vars[idx]):
            placements_scaled.append(candidate)

    placements = [_scaled_to_real(placement, scale) for placement in placements_scaled]

    item_count = len(placements)
    total_weight = item_count * package.weight

    orientation_usage: Dict[str, int] = {}
    for placement in placements:
        key = str(tuple(round(d, 2) for d in placement.orientation))
        orientation_usage[key] = orientation_usage.get(key, 0) + 1

    footprint_pct = footprint_coverage(placements, box.length, box.width)
    validate_clearances(placements)

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
    scale: int,
    box_dims: tuple[int, int, int],
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
            (box_dims[0] // orientation[0])
            * (box_dims[1] // orientation[1])
            * (box_dims[2] // orientation[2])
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

    max_items_by_weight = int(available_weight // package.weight)
    item_count = min(best_count, max_items_by_weight)
    placements: List[Placement] = []
    index = 0

    for (x, y, z) in iter_candidate_positions(box_dims, best_orientation):
        if index >= item_count:
            break
        placements.append(_scaled_to_real(Placement(x, y, z, best_orientation, 0), scale))
        index += 1

    total_weight = item_count * package.weight
    orientation_usage = {str(tuple(round(d, 2) for d in _scale_orientation(best_orientation, scale))): item_count}
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


def _respect_thickness_constraint(
    package: Package,
    orientations: List[tuple[int, int, int]],
    scale: int,
) -> List[tuple[int, int, int]]:
    """
    Filter orientations so that the configured thickness axis remains vertical.
    """
    thickness = package.scaled_thickness(scale)
    if thickness is None:
        return orientations

    filtered = [orientation for orientation in orientations if orientation[2] == thickness]
    return filtered


def _generate_candidate_placements(
    box_dims: tuple[int, int, int],
    orientations: List[tuple[int, int, int]],
) -> List[Placement]:
    """
    Generate candidate placements using reachable coordinate sums to reduce gaps.
    Falls back to basic grid generation if the search space becomes too large.
    """
    if not orientations:
        return []

    x_positions = _generate_axis_positions(box_dims[0], [ori[0] for ori in orientations])
    y_positions = _generate_axis_positions(box_dims[1], [ori[1] for ori in orientations])
    z_positions = _generate_axis_positions(box_dims[2], [ori[2] for ori in orientations])

    candidates: List[Placement] = []
    for orientation_index, orientation in enumerate(orientations):
        feasible_x = [x for x in x_positions if x + orientation[0] <= box_dims[0]]
        feasible_y = [y for y in y_positions if y + orientation[1] <= box_dims[1]]
        feasible_z = [z for z in z_positions if z + orientation[2] <= box_dims[2]]

        # If the combinatorial space is too large, revert to regular grid for this orientation.
        if len(feasible_x) * len(feasible_y) * len(feasible_z) > 20000:
            for (x, y, z) in iter_candidate_positions(box_dims, orientation):
                candidates.append(
                    Placement(
                        x=x,
                        y=y,
                        z=z,
                        orientation=orientation,
                        item_index=orientation_index,
                    )
                )
            continue

        for x in feasible_x:
            for y in feasible_y:
                for z in feasible_z:
                    candidates.append(
                        Placement(
                            x=x,
                            y=y,
                            z=z,
                            orientation=orientation,
                            item_index=orientation_index,
                        )
                    )
    return candidates


def _generate_axis_positions(max_length: int, sizes: List[int]) -> List[int]:
    """
    Compute all reachable coordinate starts by combining item sizes up to the container limit.
    """
    sizes = [size for size in sizes if size > 0]
    if not sizes:
        return [0]

    reachable = {0}
    queue = [0]
    while queue:
        current = queue.pop()
        for size in sizes:
            new_value = current + size
            if new_value <= max_length and new_value not in reachable:
                reachable.add(new_value)
                queue.append(new_value)

    return sorted(reachable)


def _scale_orientation(orientation: tuple[int, int, int], scale: int) -> tuple[float, float, float]:
    return tuple(dim / scale for dim in orientation)


def _scaled_to_real(placement: Placement, scale: int) -> Placement:
    return Placement(
        x=placement.x / scale,
        y=placement.y / scale,
        z=placement.z / scale,
        orientation=_scale_orientation(placement.orientation, scale),
        item_index=placement.item_index,
    )


def _pack_sheet_cartons(
    package: Package,
    box: Box,
    orientations: List[tuple[int, int, int]],
    max_items_by_weight: int,
    scale: int,
    box_dims: tuple[int, int, int],
) -> CartonBoxPackingResult | None:
    """
    Deterministic packing strategy for sheet-like items where one axis is locked
    as the thickness and must stay vertical. This reduces the problem to tiling
    the base area and stacking layers.
    """
    if not orientations:
        return None

    box_base_area = box.length * box.width
    best_choice: tuple[tuple[int, int, int], int, int, float] | None = None

    for orientation in orientations:
        x_dim, y_dim, z_dim = orientation
        if z_dim <= 0:
            continue

        x_count = box_dims[0] // x_dim
        y_count = box_dims[1] // y_dim
        if x_count == 0 or y_count == 0:
            continue

        per_layer = x_count * y_count
        layers_by_height = box_dims[2] // z_dim
        if layers_by_height == 0 or per_layer == 0:
            continue

        coverage_area = per_layer * (x_dim * y_dim) / (scale * scale)
        coverage_ratio = coverage_area / box_base_area if box_base_area else 0.0

        candidate = (orientation, x_count, y_count, coverage_ratio)
        if best_choice is None:
            best_choice = candidate
            continue

        _, _, _, best_ratio = best_choice
        if coverage_ratio > best_ratio:
            best_choice = candidate
        elif coverage_ratio == best_ratio:
            # Tie-breaker: prefer arrangement with more items per layer.
            _, best_x, best_y, _ = best_choice
            if per_layer > best_x * best_y:
                best_choice = candidate

    if best_choice is None:
        return None

    orientation, x_count, y_count, _ = best_choice
    x_dim, y_dim, z_dim = orientation

    per_layer = x_count * y_count
    layers_by_height = box_dims[2] // z_dim
    max_items_geometry = per_layer * layers_by_height
    if max_items_geometry == 0:
        return None

    items_to_place = min(max_items_geometry, max_items_by_weight)
    if items_to_place <= 0:
        return CartonBoxPackingResult(
            items_per_box=0,
            placements=[],
            orientation_usage={},
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Sheet heuristic - limited by weight",
            fallback_used=False,
        )

    placements: List[Placement] = []
    items_remaining = items_to_place
    layer = 0
    while items_remaining > 0 and layer < layers_by_height:
        for ix in range(x_count):
            for iy in range(y_count):
                if items_remaining <= 0:
                    break
                placements.append(
                    _scaled_to_real(
                        Placement(
                            x=ix * x_dim,
                            y=iy * y_dim,
                            z=layer * z_dim,
                            orientation=orientation,
                            item_index=0,
                        ),
                        scale,
                    )
                )
                items_remaining -= 1
            if items_remaining <= 0:
                break
        layer += 1

    item_count = len(placements)
    total_weight = item_count * package.weight
    used_volume = item_count * package.volume
    orientation_usage = {
        str(tuple(round(d, 2) for d in _scale_orientation(orientation, scale))): item_count
    }

    footprint_pct = footprint_coverage(placements, box.length, box.width)
    validate_clearances(placements)

    return CartonBoxPackingResult(
        items_per_box=item_count,
        placements=placements,
        orientation_usage=orientation_usage,
        volume_utilisation_pct=volume_utilization(used_volume, box.inner_volume),
        weight_utilisation_pct=volume_utilization(total_weight, box.max_weight - box.tare_weight),
        footprint_utilisation_pct=footprint_pct,
        solver_status="Sheet heuristic",
        fallback_used=False,
    )

