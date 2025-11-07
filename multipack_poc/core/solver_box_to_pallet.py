"""
Optimisation logic for stacking filled boxes onto a pallet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    from ortools.sat.python import cp_model  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Google OR-Tools is required for the box-to-pallet solver. "
        "Install it via `pip install ortools`."
    ) from exc

from multipack_poc.core.utils_geometry import (
    Placement,
    boxes_overlap,
    footprint_coverage,
    iter_candidate_positions,
    unique_horizontal_orientations,
    validate_clearances,
    volume_utilization,
)
from multipack_poc.models.box import Box
from multipack_poc.models.pallet import Pallet


@dataclass
class BoxPalletPackingResult:
    boxes_per_layer: int
    layers_per_pallet: int
    total_boxes: int
    total_items: int
    placements: List[Placement]
    volume_utilisation_pct: float
    weight_utilisation_pct: float
    footprint_utilisation_pct: float
    solver_status: str
    fallback_used: bool

    def to_dict(self) -> dict:
        return {
            "boxes_per_layer": self.boxes_per_layer,
            "layers_per_pallet": self.layers_per_pallet,
            "total_boxes": self.total_boxes,
            "total_items_per_pallet": self.total_items,
            "volume_utilization": self.volume_utilisation_pct,
            "weight_utilization": self.weight_utilisation_pct,
            "footprint_utilization": self.footprint_utilisation_pct,
            "positions": [placement.as_dict() for placement in self.placements],
            "solver_status": self.solver_status,
            "fallback_used": self.fallback_used,
        }


def pack_boxes_on_pallet(
    box: Box,
    pallet: Pallet,
    items_per_box: int,
    filled_box_weight: int,
    time_limit_sec: float = 5.0,
) -> BoxPalletPackingResult:
    """
    Stack filled boxes onto a pallet using CP-SAT optimisation.
    """
    if filled_box_weight <= 0:
        raise ValueError("filled_box_weight must be positive")

    if filled_box_weight > pallet.max_weight:
        return BoxPalletPackingResult(
            boxes_per_layer=0,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - box weight exceeds pallet capacity",
            fallback_used=False,
        )

    orientations = unique_horizontal_orientations(box.dimensions)
    orientations = [
        orientation
        for orientation in orientations
        if orientation[0] <= pallet.length and orientation[1] <= pallet.width
    ]

    if not orientations:
        return BoxPalletPackingResult(
            boxes_per_layer=0,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - no orientations fit pallet footprint",
            fallback_used=False,
        )

    container_dims = (pallet.length, pallet.width, orientations[0][2])
    candidates: List[Placement] = []

    for orientation_index, orientation in enumerate(orientations):
        for (x, y, z) in iter_candidate_positions(container_dims, orientation):
            candidates.append(
                Placement(
                    x=x,
                    y=y,
                    z=0,
                    orientation=orientation,
                    item_index=orientation_index,
                )
            )

    if not candidates:
        return _fallback_grid_fit(box, pallet, orientations, items_per_box, filled_box_weight)

    model = cp_model.CpModel()
    decision_vars: List[cp_model.IntVar] = []
    for idx in range(len(candidates)):
        decision_vars.append(model.NewBoolVar(f"place_box_{idx}"))

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if boxes_overlap(
                (candidates[i].x, candidates[i].y, candidates[i].z),
                candidates[i].orientation,
                (candidates[j].x, candidates[j].y, candidates[j].z),
                candidates[j].orientation,
            ):
                model.Add(decision_vars[i] + decision_vars[j] <= 1)

    max_boxes_by_weight = pallet.max_weight // filled_box_weight
    height_layers_limit = pallet.usable_height() // box.height if box.height > 0 else 0

    model.Add(sum(decision_vars) <= max_boxes_by_weight)
    model.Maximize(sum(decision_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _fallback_grid_fit(
            box,
            pallet,
            orientations,
            items_per_box,
            filled_box_weight,
            status_name=solver.StatusName(status),
        )

    base_layer: List[Placement] = []
    for idx, candidate in enumerate(candidates):
        if solver.Value(decision_vars[idx]):
            base_layer.append(candidate)

    boxes_per_layer = len(base_layer)
    if boxes_per_layer == 0:
        return _fallback_grid_fit(box, pallet, orientations, items_per_box, filled_box_weight)

    layers_by_weight = max_boxes_by_weight // boxes_per_layer if boxes_per_layer else 0
    possible_layers = min(height_layers_limit, layers_by_weight)

    layers_per_pallet = max(1, possible_layers) if possible_layers > 0 else 0
    total_boxes = boxes_per_layer * layers_per_pallet

    if layers_per_pallet == 0:
        return BoxPalletPackingResult(
            boxes_per_layer=boxes_per_layer,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status=solver.StatusName(status),
            fallback_used=False,
        )

    placements: List[Placement] = []
    for layer in range(layers_per_pallet):
        for placement in base_layer:
            placements.append(
                Placement(
                    x=placement.x,
                    y=placement.y,
                    z=layer * box.height,
                    orientation=placement.orientation,
                    item_index=placement.item_index,
                )
            )

    total_items = total_boxes * items_per_box
    used_volume = total_boxes * box.length * box.width * box.height
    pallet_volume = pallet.length * pallet.width * pallet.usable_height()
    pallet_weight_used = total_boxes * filled_box_weight
    footprint_pct = footprint_coverage(base_layer, pallet.length, pallet.width)
    validate_clearances(base_layer)

    return BoxPalletPackingResult(
        boxes_per_layer=boxes_per_layer,
        layers_per_pallet=layers_per_pallet,
        total_boxes=total_boxes,
        total_items=total_items,
        placements=placements,
        volume_utilisation_pct=volume_utilization(used_volume, pallet_volume),
        weight_utilisation_pct=volume_utilization(pallet_weight_used, pallet.max_weight),
        footprint_utilisation_pct=footprint_pct,
        solver_status=solver.StatusName(status),
        fallback_used=False,
    )


def _fallback_grid_fit(
    box: Box,
    pallet: Pallet,
    orientations: List[tuple[int, int, int]],
    items_per_box: int,
    filled_box_weight: int,
    status_name: str = "Fallback",
) -> BoxPalletPackingResult:
    """
    Greedy fallback that tiles the best fitting orientation across the pallet.
    """
    best_orientation = orientations[0]
    best_count = 0
    for orientation in orientations:
        count = (pallet.length // orientation[0]) * (pallet.width // orientation[1])
        if count > best_count:
            best_count = count
            best_orientation = orientation

    if best_count == 0:
        return BoxPalletPackingResult(
            boxes_per_layer=0,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status=status_name,
            fallback_used=True,
        )

    max_boxes_by_weight = pallet.max_weight // filled_box_weight
    layers_height_limit = pallet.usable_height() // box.height if box.height else 0
    layers_possible = min(
        layers_height_limit,
        max_boxes_by_weight // best_count if best_count else 0,
    )
    layers_per_pallet = max(1, layers_possible) if layers_possible > 0 else 0
    total_boxes = best_count * layers_per_pallet

    placements: List[Placement] = []
    for (x, y, _z) in iter_candidate_positions(
        (pallet.length, pallet.width, best_orientation[2]),
        best_orientation,
    ):
        placements.append(
            Placement(
                x=x,
                y=y,
                z=0,
                orientation=best_orientation,
                item_index=0,
            )
        )
        if len(placements) >= best_count:
            break

    all_placements: List[Placement] = []
    for layer in range(layers_per_pallet):
        for placement in placements:
            all_placements.append(
                Placement(
                    x=placement.x,
                    y=placement.y,
                    z=layer * box.height,
                    orientation=placement.orientation,
                    item_index=placement.item_index,
                )
            )

    total_items = total_boxes * items_per_box
    used_volume = total_boxes * box.length * box.width * box.height
    pallet_volume = pallet.length * pallet.width * pallet.usable_height()
    pallet_weight_used = total_boxes * filled_box_weight
    footprint_pct = footprint_coverage(placements, pallet.length, pallet.width)
    validate_clearances(placements)

    return BoxPalletPackingResult(
        boxes_per_layer=best_count,
        layers_per_pallet=layers_per_pallet,
        total_boxes=total_boxes,
        total_items=total_items,
        placements=all_placements,
        volume_utilisation_pct=volume_utilization(used_volume, pallet_volume),
        weight_utilisation_pct=volume_utilization(pallet_weight_used, pallet.max_weight),
        footprint_utilisation_pct=footprint_pct,
        solver_status=status_name,
        fallback_used=True,
    )

