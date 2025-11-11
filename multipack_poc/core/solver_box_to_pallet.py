"""
Optimisation logic for stacking filled boxes onto a pallet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from multipack_poc.core.utils_geometry import (
    Placement,
    footprint_coverage,
    unique_horizontal_orientations,
    validate_clearances,
    volume_utilization,
)
from multipack_poc.models.box import Box
from multipack_poc.models.pallet import Pallet


DIMENSION_SCALE = 100  # consistency with carton solver (0.01 mm resolution)


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
    Deterministic packing strategy that tiles the pallet footprint and stacks layers.
    """
    del time_limit_sec  # CP-SAT parameters retained for API compatibility.

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

    scale = DIMENSION_SCALE
    pallet_length = int(round(pallet.length * scale))
    pallet_width = int(round(pallet.width * scale))
    usable_height = int(round(pallet.usable_height() * scale))

    orientations_scaled: List[Tuple[int, int, int]] = []
    for orientation in unique_horizontal_orientations(box.dimensions):
        length, width, height = orientation
        orientation_scaled = (
            int(round(length * scale)),
            int(round(width * scale)),
            int(round(height * scale)),
        )
        if orientation_scaled[0] <= pallet_length and orientation_scaled[1] <= pallet_width:
            orientations_scaled.append(orientation_scaled)

    feasible: List[Tuple[Tuple[int, int, int], int, int, float]] = []

    for orientation in orientations_scaled:
        length, width, height = orientation
        if length <= 0 or width <= 0 or height <= 0:
            continue
        x_count = pallet_length // length
        y_count = pallet_width // width
        per_layer = x_count * y_count
        if per_layer <= 0:
            continue
        coverage = per_layer * (length * width) / (scale * scale)
        feasible.append((orientation, x_count, y_count, coverage))

    if not feasible:
        return BoxPalletPackingResult(
            boxes_per_layer=0,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - box footprint does not fit pallet",
            fallback_used=False,
        )

    # Select the orientation that maximises footprint coverage; break ties by per-layer count.
    orientation, x_count, y_count, _ = max(
        feasible,
        key=lambda entry: (entry[3], entry[1] * entry[2]),
    )

    per_layer = x_count * y_count
    if per_layer == 0:
        return BoxPalletPackingResult(
            boxes_per_layer=0,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="No boxes per layer could be arranged",
            fallback_used=False,
        )

    layers_by_height = usable_height // orientation[2]
    if layers_by_height == 0:
        return BoxPalletPackingResult(
            boxes_per_layer=per_layer,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - pallet height insufficient",
            fallback_used=False,
        )

    max_boxes_by_weight = int(pallet.max_weight // filled_box_weight)
    layers_by_weight = max_boxes_by_weight // per_layer

    layers_per_pallet = min(layers_by_height, layers_by_weight)

    if layers_per_pallet <= 0:
        return BoxPalletPackingResult(
            boxes_per_layer=per_layer,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - pallet weight limit reached",
            fallback_used=False,
        )

    total_boxes = per_layer * layers_per_pallet
    total_items = total_boxes * items_per_box

    placements: List[Placement] = []
    for layer in range(layers_per_pallet):
        for ix in range(x_count):
            for iy in range(y_count):
                scaled = Placement(
                    x=ix * orientation[0],
                    y=iy * orientation[1],
                    z=layer * orientation[2],
                    orientation=orientation,
                    item_index=0,
                )
                placements.append(_scaled_to_real(scaled, scale))

    used_volume = total_boxes * box.length * box.width * box.height
    pallet_volume = pallet.length * pallet.width * pallet.usable_height()
    pallet_weight_used = total_boxes * filled_box_weight
    base_placements = placements[:per_layer]
    footprint_pct = footprint_coverage(base_placements, pallet.length, pallet.width)
    validate_clearances(base_placements)

    return BoxPalletPackingResult(
        boxes_per_layer=per_layer,
        layers_per_pallet=layers_per_pallet,
        total_boxes=total_boxes,
        total_items=total_items,
        placements=placements,
        volume_utilisation_pct=volume_utilization(used_volume, pallet_volume),
        weight_utilisation_pct=volume_utilization(pallet_weight_used, pallet.max_weight),
        footprint_utilisation_pct=footprint_pct,
        solver_status="Deterministic tiling heuristic",
        fallback_used=False,
    )


def _scaled_to_real(placement: Placement, scale: int) -> Placement:
    return Placement(
        x=placement.x / scale,
        y=placement.y / scale,
        z=placement.z / scale,
        orientation=tuple(dim / scale for dim in placement.orientation),
        item_index=placement.item_index,
    )

