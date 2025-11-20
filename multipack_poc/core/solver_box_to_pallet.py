"""
Optimisation logic for stacking filled boxes onto a pallet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from ortools.sat.python import cp_model  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Google OR-Tools is required for the box-to-pallet solver. "
        "Install it via `pip install ortools`."
    ) from exc

from multipack_poc.core.utils_geometry import (
    Placement,
    footprint_coverage,
    generate_axis_orientations,
    iter_candidate_positions,
    rects_overlap_1d,
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


def _rectangles_overlap_2d(
    a_x: int, a_y: int, a_length: int, a_width: int,
    b_x: int, b_y: int, b_length: int, b_width: int,
) -> bool:
    """Check if two 2D rectangles overlap."""
    return (
        rects_overlap_1d(a_x, a_length, b_x, b_length) and
        rects_overlap_1d(a_y, a_width, b_y, b_width)
    )


def _generate_axis_positions_2d(max_length: int, sizes: List[int]) -> List[int]:
    """
    Compute all reachable coordinate starts by combining item sizes up to the container limit.
    This allows boxes to be placed at any position that can be reached by multiples of box dimensions,
    enabling more flexible and efficient packing.
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


def _generate_2d_candidate_placements(
    pallet_length: int,
    pallet_width: int,
    box_length: int,
    box_width: int,
) -> List[Tuple[int, int]]:
    """
    Generate candidate 2D positions for a box on the pallet footprint.
    Uses reachable coordinate sums to allow flexible positioning and improve packing efficiency.
    For identical rectangles, optimal solutions are usually grid-aligned, but we allow some flexibility.
    """
    candidates: List[Tuple[int, int]] = []
    
    # Generate reachable positions on each axis using box dimensions
    # This creates positions at multiples of box dimensions, allowing efficient tiling
    x_positions = _generate_axis_positions_2d(pallet_length, [box_length])
    y_positions = _generate_axis_positions_2d(pallet_width, [box_width])
    
    # Generate all valid combinations
    for x in x_positions:
        if x + box_length > pallet_length:
            continue
        for y in y_positions:
            if y + box_width > pallet_width:
                continue
            candidates.append((x, y))
    
    return candidates


def _generate_2d_candidates_with_orientations(
    pallet_length: int,
    pallet_width: int,
    orientations: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, Tuple[int, int, int]]]:
    """
    Generate 2D candidate placements for all orientations.
    Returns list of (x, y, orientation) tuples.
    """
    candidates: List[Tuple[int, int, Tuple[int, int, int]]] = []
    
    # Generate reachable positions for all orientations
    all_lengths = [ori[0] for ori in orientations]
    all_widths = [ori[1] for ori in orientations]
    
    x_positions = _generate_axis_positions_2d(pallet_length, all_lengths)
    y_positions = _generate_axis_positions_2d(pallet_width, all_widths)
    
    # Generate candidates for each orientation
    for orientation in orientations:
        length, width, height = orientation
        feasible_x = [x for x in x_positions if x + length <= pallet_length]
        feasible_y = [y for y in y_positions if y + width <= pallet_width]
        
        for x in feasible_x:
            for y in feasible_y:
                candidates.append((x, y, orientation))
    
    return candidates


def _solve_2d_packing_mixed_orientations(
    pallet_length: int,
    pallet_width: int,
    orientations: List[Tuple[int, int, int]],
    max_boxes: int,
    time_limit_sec: float,
) -> Tuple[List[Tuple[int, int, Tuple[int, int, int]]], str]:
    """
    Solve 2D bin packing with mixed orientations using OR-Tools CP-SAT.
    Allows each box to have a different orientation to maximize placement.
    Returns list of (x, y, orientation) tuples and solver status.
    """
    candidates = _generate_2d_candidates_with_orientations(
        pallet_length, pallet_width, orientations
    )
    
    if not candidates:
        return [], "No candidate positions"
    
    # Limit candidates if too many
    if len(candidates) > 5000:
        # Prioritize grid-aligned positions for each orientation
        grid_candidates = []
        other_candidates = []
        
        for x, y, orientation in candidates:
            length, width, _ = orientation
            if x % length == 0 and y % width == 0:
                grid_candidates.append((x, y, orientation))
            else:
                other_candidates.append((x, y, orientation))
        
        # Keep all grid positions, sample others
        if len(other_candidates) > 2000:
            step = max(1, len(other_candidates) // 2000)
            other_candidates = other_candidates[::step]
        
        candidates = grid_candidates + other_candidates
        
        # Final limit
        if len(candidates) > 5000:
            candidates = grid_candidates[:3000] + other_candidates[:2000]
    
    model = cp_model.CpModel()
    decision_vars: List[cp_model.IntVar] = []
    for idx in range(len(candidates)):
        decision_vars.append(model.NewBoolVar(f"place_{idx}"))
    
    # Non-overlap constraints for 2D rectangles with different orientations
    overlap_pairs: List[Tuple[int, int]] = []
    for i in range(len(candidates)):
        x_i, y_i, orientation_i = candidates[i]
        length_i, width_i, _ = orientation_i
        for j in range(i + 1, len(candidates)):
            x_j, y_j, orientation_j = candidates[j]
            length_j, width_j, _ = orientation_j
            if _rectangles_overlap_2d(
                x_i, y_i, length_i, width_i,
                x_j, y_j, length_j, width_j,
            ):
                overlap_pairs.append((i, j))
    
    # Add non-overlap constraints
    for i, j in overlap_pairs:
        model.Add(decision_vars[i] + decision_vars[j] <= 1)
    
    # Limit by max boxes
    model.Add(sum(decision_vars) <= max_boxes)
    
    # Maximize number of boxes, with slight preference for orientations with smaller heights
    # This encourages the solver to prefer boxes that allow more layers when stacked
    # Use a small weight (1/1000 of max height) to prefer smaller heights without sacrificing box count
    max_height = max(ori[2] for ori in orientations)
    height_penalty = sum(
        decision_vars[idx] * (candidates[idx][2][2] * 1000 // max_height)
        for idx in range(len(candidates))
    )
    # Maximize boxes minus small height penalty (prefer smaller heights)
    model.Maximize(sum(decision_vars) * 10000 - height_penalty)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False
    solver.parameters.linearization_level = 2
    
    status = solver.Solve(model)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], solver.StatusName(status)
    
    placements: List[Tuple[int, int, Tuple[int, int, int]]] = []
    for idx, candidate in enumerate(candidates):
        if solver.Value(decision_vars[idx]):
            placements.append(candidate)
    
    return placements, solver.StatusName(status)


def _solve_2d_packing(
    pallet_length: int,
    pallet_width: int,
    box_length: int,
    box_width: int,
    max_boxes: int,
    time_limit_sec: float,
) -> Tuple[List[Tuple[int, int]], str]:
    """
    Solve 2D bin packing problem using OR-Tools CP-SAT (single orientation).
    Returns list of (x, y) positions and solver status.
    """
    candidates = _generate_2d_candidate_placements(
        pallet_length, pallet_width, box_length, box_width
    )
    
    if not candidates:
        return [], "No candidate positions"
    
    # For identical rectangles, grid-aligned positions are usually optimal
    grid_candidates = [
        (x, y) for x, y in candidates
        if x % box_length == 0 and y % box_width == 0
    ]
    
    if len(grid_candidates) >= max_boxes:
        candidates = grid_candidates
    elif len(candidates) > 3000:
        other_candidates = [
            (x, y) for x, y in candidates
            if not (x % box_length == 0 and y % box_width == 0)
        ]
        if len(other_candidates) > 1000:
            step = max(1, len(other_candidates) // 1000)
            other_candidates = other_candidates[::step]
        candidates = grid_candidates + other_candidates
    
    if len(candidates) > 3000:
        candidates = grid_candidates[:2000] + candidates[len(grid_candidates):len(grid_candidates)+1000]
    
    model = cp_model.CpModel()
    decision_vars: List[cp_model.IntVar] = []
    for idx in range(len(candidates)):
        decision_vars.append(model.NewBoolVar(f"place_{idx}"))
    
    overlap_pairs: List[Tuple[int, int]] = []
    for i in range(len(candidates)):
        x_i, y_i = candidates[i]
        for j in range(i + 1, len(candidates)):
            x_j, y_j = candidates[j]
            if _rectangles_overlap_2d(
                x_i, y_i, box_length, box_width,
                x_j, y_j, box_length, box_width,
            ):
                overlap_pairs.append((i, j))
    
    for i, j in overlap_pairs:
        model.Add(decision_vars[i] + decision_vars[j] <= 1)
    
    model.Add(sum(decision_vars) <= max_boxes)
    model.Maximize(sum(decision_vars))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False
    solver.parameters.linearization_level = 2
    
    status = solver.Solve(model)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], solver.StatusName(status)
    
    placements: List[Tuple[int, int]] = []
    for idx, candidate in enumerate(candidates):
        if solver.Value(decision_vars[idx]):
            placements.append(candidate)
    
    return placements, solver.StatusName(status)


def pack_boxes_on_pallet(
    box: Box,
    pallet: Pallet,
    items_per_box: int,
    filled_box_weight: int,
    time_limit_sec: float = 5.0,
) -> BoxPalletPackingResult:
    """
    Optimize box placement on pallet using OR-Tools 2D bin packing.
    Solves 2D placement for the pallet footprint, then stacks vertically.
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

    scale = DIMENSION_SCALE
    pallet_length = int(round(pallet.length * scale))
    pallet_width = int(round(pallet.width * scale))
    usable_height = int(round(pallet.usable_height() * scale))

    # Generate all 6 possible orientations using OUTER dimensions (including wall thickness)
    # Outer dimensions are used because boxes are placed on pallets, not inside them
    box_dims_scaled = box.scaled_outer_dimensions(scale)
    all_orientations = generate_axis_orientations(box_dims_scaled)
    
    # Filter orientations that fit on pallet footprint and within height
    feasible_orientations: List[Tuple[int, int, int]] = []
    for orientation in all_orientations:
        length, width, height = orientation
        if (length <= pallet_length and width <= pallet_width and 
            height <= usable_height and height > 0):
            feasible_orientations.append(orientation)

    if not feasible_orientations:
        return BoxPalletPackingResult(
            boxes_per_layer=0,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - box does not fit pallet in any orientation",
            fallback_used=False,
        )

    # Calculate max boxes by weight
    max_boxes_by_weight = int(pallet.max_weight // filled_box_weight)
    
    # Filter orientations by height constraint for stacking
    # We need orientations where height fits in usable height
    # Sort by height (smallest first) to prioritize orientations that allow more layers
    stacking_orientations: List[Tuple[int, int, int]] = []
    for orientation in feasible_orientations:
        length, width, height = orientation
        if height <= usable_height and height > 0:
            stacking_orientations.append(orientation)
    
    if not stacking_orientations:
        return BoxPalletPackingResult(
            boxes_per_layer=0,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - no orientations fit pallet height",
            fallback_used=False,
        )
    
    # Sort orientations by height (smallest first) to maximize stacking potential
    # This ensures we prioritize orientations that allow more layers
    stacking_orientations.sort(key=lambda o: (o[2], -o[0] * o[1]))  # Height first, then footprint area (descending)
    
    # Use mixed orientations approach for better packing, especially on smaller pallets
    # This allows each box to have a different orientation to maximize placement
    
    # Calculate max layers based on minimum height (most restrictive)
    min_height = min(ori[2] for ori in stacking_orientations)
    max_layers_by_height = usable_height // min_height if min_height > 0 else 0
    
    if max_layers_by_height == 0:
        return BoxPalletPackingResult(
            boxes_per_layer=0,
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
    
    # Optimize for TOTAL boxes (boxes_per_layer * layers_per_pallet), not just boxes per layer
    # Try different height groups and find the combination that maximizes total boxes
    best_overall_result: Tuple[
        List[Tuple[int, int, Tuple[int, int, int]]],  # placements
        int,  # boxes_per_layer
        int,  # layers_per_pallet
        int,  # total_boxes
        int,  # height
        str,  # status
    ] | None = None
    
    # Group orientations by height to optimize each height group separately
    orientations_by_height: Dict[int, List[Tuple[int, int, int]]] = {}
    for orientation in stacking_orientations:
        height = orientation[2]
        if height not in orientations_by_height:
            orientations_by_height[height] = []
        orientations_by_height[height].append(orientation)
    
    # Try each height group and optimize for total boxes
    # For each height, we need to find the optimal balance between boxes_per_layer and layers_per_pallet
    # Allocate time across height groups
    time_per_height_group = (time_limit_sec * 0.7) / max(len(orientations_by_height), 1)
    
    for height, height_orientations in sorted(orientations_by_height.items()):
        layers_by_height = usable_height // height
        if layers_by_height == 0:
            continue
        
        # Calculate theoretical maximums
        max_boxes_by_weight_for_height = max_boxes_by_weight // layers_by_height if layers_by_height > 0 else max_boxes_by_weight
        theoretical_max_boxes = (pallet_length * pallet_width) // min(ori[0] * ori[1] for ori in height_orientations) * 2
        max_boxes_per_layer_for_height = min(max_boxes_by_weight_for_height, theoretical_max_boxes)
        
        if max_boxes_per_layer_for_height <= 0:
            continue
        
        # Try solving 2D packing for this height group
        # The solver will maximize boxes per layer, but we'll evaluate based on total boxes
        layer_placements_mixed, status = _solve_2d_packing_mixed_orientations(
            pallet_length,
            pallet_width,
            height_orientations,
            max_boxes_per_layer_for_height,
            time_per_height_group,
        )
        
        if not layer_placements_mixed:
            continue
        
        # Calculate total boxes for this height group
        boxes_per_layer = len(layer_placements_mixed)
        layers_by_weight = max_boxes_by_weight // boxes_per_layer if boxes_per_layer > 0 else 0
        layers_per_pallet = min(layers_by_height, layers_by_weight)
        
        if layers_per_pallet <= 0:
            continue
        
        total_boxes = boxes_per_layer * layers_per_pallet
        
        # Also try reducing boxes per layer to see if we can get more layers
        # This helps when weight is the limiting factor
        if layers_by_weight < layers_by_height and boxes_per_layer > 1:
            # Try with fewer boxes per layer to see if we can stack more layers
            # Only try if we're weight-limited (not height-limited)
            for reduced_boxes in range(boxes_per_layer - 1, max(1, boxes_per_layer // 2), -1):
                if reduced_boxes <= 0:
                    break
                # Quick check: if reducing boxes doesn't increase layers, skip
                potential_layers = min(layers_by_height, max_boxes_by_weight // reduced_boxes if reduced_boxes > 0 else 0)
                if potential_layers <= layers_per_pallet:
                    break  # No benefit, skip
                
                # Try solving with reduced target
                reduced_placements, reduced_status = _solve_2d_packing_mixed_orientations(
                    pallet_length,
                    pallet_width,
                    height_orientations,
                    reduced_boxes,
                    time_per_height_group * 0.3,  # Use less time for these trials
                )
                
                if reduced_placements:
                    reduced_boxes_count = len(reduced_placements)
                    reduced_layers = min(layers_by_height, max_boxes_by_weight // reduced_boxes_count if reduced_boxes_count > 0 else 0)
                    reduced_total = reduced_boxes_count * reduced_layers
                    
                    if reduced_total > total_boxes:
                        # Better total boxes with fewer boxes per layer but more layers
                        layer_placements_mixed = reduced_placements
                        boxes_per_layer = reduced_boxes_count
                        layers_per_pallet = reduced_layers
                        total_boxes = reduced_total
                        status = reduced_status
                        break  # Found better solution, no need to try fewer
        
        # Select best result: maximize total boxes
        if best_overall_result is None:
            best_overall_result = (layer_placements_mixed, boxes_per_layer, layers_per_pallet, total_boxes, height, status)
        else:
            _, _, _, best_total, _, _ = best_overall_result
            if total_boxes > best_total:
                best_overall_result = (layer_placements_mixed, boxes_per_layer, layers_per_pallet, total_boxes, height, status)
            elif total_boxes == best_total:
                # Tie-breaker: prefer smaller height for more layers
                _, _, _, _, best_h, _ = best_overall_result
                if height < best_h:
                    best_overall_result = (layer_placements_mixed, boxes_per_layer, layers_per_pallet, total_boxes, height, status)
    
    if best_overall_result:
        layer_placements_mixed, boxes_per_layer, layers_per_pallet, total_boxes, height, status = best_overall_result
        
        # Use the best result found
        total_items = total_boxes * items_per_box
        
        # Generate 3D placements by stacking the 2D arrangement
        placements_3d: List[Placement] = []
        for layer in range(layers_per_pallet):
            for x, y, orientation in layer_placements_mixed:
                scaled = Placement(
                    x=x,
                    y=y,
                    z=layer * height,
                    orientation=orientation,
                    item_index=0,
                )
                placements_3d.append(_scaled_to_real(scaled, scale))
        
        # Calculate used volume from placements (already in real scale)
        used_volume = sum(
            placement.orientation[0] * placement.orientation[1] * placement.orientation[2]
            for placement in placements_3d
        )
        pallet_volume = pallet.length * pallet.width * pallet.usable_height()
        pallet_weight_used = total_boxes * filled_box_weight
        base_placements = placements_3d[:boxes_per_layer]
        footprint_pct = footprint_coverage(base_placements, pallet.length, pallet.width)
        validate_clearances(base_placements)
        
        return BoxPalletPackingResult(
            boxes_per_layer=boxes_per_layer,
            layers_per_pallet=layers_per_pallet,
            total_boxes=total_boxes,
            total_items=total_items,
            placements=placements_3d,
            volume_utilisation_pct=volume_utilization(used_volume, pallet_volume),
            weight_utilisation_pct=volume_utilization(pallet_weight_used, pallet.max_weight),
            footprint_utilisation_pct=footprint_pct,
            solver_status=f"OR-Tools 2D (mixed orientations, optimized for total boxes): {status}",
            fallback_used=False,
        )
    
    # Fallback if no mixed orientation solution found
    if not best_overall_result:
        # Fallback: try single-orientation approach
        best_single_result: Tuple[Tuple[int, int, int], List[Tuple[int, int]], int, float, str, int] | None = None
        
        # Try orientations in order of height (smallest first) to maximize layers
        for orientation in stacking_orientations:
            length, width, height = orientation
            layers_by_height = usable_height // height
            if layers_by_height == 0:
                continue
            
            max_boxes_this_orientation = min(
                max_boxes_by_weight // layers_by_height if layers_by_height > 0 else max_boxes_by_weight,
                (pallet_length // length) * (pallet_width // width) * 2
            )
            
            if max_boxes_this_orientation <= 0:
                continue
            
            layer_placements, single_status = _solve_2d_packing(
                pallet_length,
                pallet_width,
                length,
                width,
                max_boxes_this_orientation,
                time_limit_sec * 0.3 / len(stacking_orientations),
            )
            
            if not layer_placements:
                continue
            
            boxes_per_layer = len(layer_placements)
            total_layers = min(layers_by_height, max_boxes_by_weight // boxes_per_layer if boxes_per_layer > 0 else 0)
            
            if total_layers <= 0:
                continue
            
            total_boxes = boxes_per_layer * total_layers
            coverage = boxes_per_layer * (length * width) / (scale * scale)
            footprint_util = coverage / (pallet.length * pallet.width) * 100
            
            if best_single_result is None:
                best_single_result = (orientation, layer_placements, total_boxes, footprint_util, single_status, height)
            else:
                _, _, best_total, best_util, _, best_h = best_single_result
                # Prefer more boxes, or same boxes with smaller height
                if total_boxes > best_total or (total_boxes == best_total and height < best_h):
                    best_single_result = (orientation, layer_placements, total_boxes, footprint_util, single_status, height)
        
        if best_single_result:
            orientation, layer_placements, total_boxes, footprint_util, single_status, _ = best_single_result
            length, width, height = orientation
            boxes_per_layer = len(layer_placements)
            layers_by_height = usable_height // height
            layers_by_weight = max_boxes_by_weight // boxes_per_layer if boxes_per_layer > 0 else 0
            layers_per_pallet = min(layers_by_height, layers_by_weight)
            
            if layers_per_pallet <= 0:
                return BoxPalletPackingResult(
                    boxes_per_layer=boxes_per_layer,
                    layers_per_pallet=0,
                    total_boxes=0,
                    total_items=0,
                    placements=[],
                    volume_utilisation_pct=0.0,
                    weight_utilisation_pct=0.0,
                    footprint_utilisation_pct=0.0,
                    solver_status="Infeasible - pallet height or weight limit reached",
                    fallback_used=False,
                )
            
            total_boxes = boxes_per_layer * layers_per_pallet
            total_items = total_boxes * items_per_box
            
            placements: List[Placement] = []
            for layer in range(layers_per_pallet):
                for x, y in layer_placements:
                    scaled = Placement(
                        x=x,
                        y=y,
                        z=layer * height,
                        orientation=orientation,
                        item_index=0,
                    )
                    placements.append(_scaled_to_real(scaled, scale))
            
            # Use outer dimensions for volume calculation (boxes on pallet use outer dimensions)
            used_volume = total_boxes * box.outer_length * box.outer_width * box.outer_height
            pallet_volume = pallet.length * pallet.width * pallet.usable_height()
            pallet_weight_used = total_boxes * filled_box_weight
            base_placements = placements[:boxes_per_layer]
            footprint_pct = footprint_coverage(base_placements, pallet.length, pallet.width)
            validate_clearances(base_placements)
            
            return BoxPalletPackingResult(
                boxes_per_layer=boxes_per_layer,
                layers_per_pallet=layers_per_pallet,
                total_boxes=total_boxes,
                total_items=total_items,
                placements=placements,
                volume_utilisation_pct=volume_utilization(used_volume, pallet_volume),
                weight_utilisation_pct=volume_utilization(pallet_weight_used, pallet.max_weight),
                footprint_utilisation_pct=footprint_pct,
                solver_status=f"OR-Tools 2D (single orientation): {single_status}",
                fallback_used=False,
            )
        
        return BoxPalletPackingResult(
            boxes_per_layer=0,
            layers_per_pallet=0,
            total_boxes=0,
            total_items=0,
            placements=[],
            volume_utilisation_pct=0.0,
            weight_utilisation_pct=0.0,
            footprint_utilisation_pct=0.0,
            solver_status="Infeasible - no valid arrangement found",
            fallback_used=False,
        )
    
    # Should not reach here, but handle gracefully
    return BoxPalletPackingResult(
        boxes_per_layer=0,
        layers_per_pallet=0,
        total_boxes=0,
        total_items=0,
        placements=[],
        volume_utilisation_pct=0.0,
        weight_utilisation_pct=0.0,
        footprint_utilisation_pct=0.0,
        solver_status="Infeasible - no valid arrangement found",
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

