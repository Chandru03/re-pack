"""
Geometry helper utilities shared across solver modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import sqrt
from typing import Iterable, List, Sequence, Tuple


Orientation = Tuple[float, float, float]


def generate_axis_orientations(dimensions: Sequence[int]) -> List[Tuple[int, int, int]]:
    """
    Return the six right-handed axis-aligned orientations for a rectangular item.
    """
    dims = tuple(int(value) for value in dimensions)
    unique: set[Tuple[int, int, int]] = set(permutations(dims))
    return list(unique)


def unique_horizontal_orientations(dimensions: Sequence[float]) -> List[Orientation]:
    """
    Return the orientations that preserve height (no sideways flipping).

    This is used when stacking boxes on pallets where tipping the box is not allowed.
    """
    l, w, h = (float(value) for value in dimensions)
    return [(l, w, h), (w, l, h)] if l != w else [(l, w, h)]


def rects_overlap_1d(a_start: int, a_len: int, b_start: int, b_len: int) -> bool:
    """
    Determine if two line segments on the same axis overlap (non-strict).
    """
    return not (a_start + a_len <= b_start or b_start + b_len <= a_start)


def boxes_overlap(
    a_origin: Tuple[int, int, int],
    a_dims: Orientation,
    b_origin: Tuple[int, int, int],
    b_dims: Orientation,
) -> bool:
    """
    Check whether two axis-aligned cuboids intersect.
    """
    return (
        rects_overlap_1d(a_origin[0], a_dims[0], b_origin[0], b_dims[0])
        and rects_overlap_1d(a_origin[1], a_dims[1], b_origin[1], b_dims[1])
        and rects_overlap_1d(a_origin[2], a_dims[2], b_origin[2], b_dims[2])
    )


@dataclass(frozen=True)
class Placement:
    """
    Represents the placement of an item within a container.
    """

    x: float
    y: float
    z: float
    orientation: Orientation
    item_index: int

    def as_dict(self) -> dict:
        return {
            "item_index": self.item_index,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "orientation": self.orientation,
        }


def iter_candidate_positions(
    container_dims: Orientation,
    item_dims: Orientation,
) -> Iterable[Tuple[int, int, int]]:
    """
    Yield candidate (x, y, z) positions on a discrete grid obtained by packing
    the item size repeatedly within the container dimensions.
    """
    max_x = container_dims[0] // item_dims[0]
    max_y = container_dims[1] // item_dims[1]
    max_z = container_dims[2] // item_dims[2]
    for ix in range(max_x):
        for iy in range(max_y):
            for iz in range(max_z):
                yield ix * item_dims[0], iy * item_dims[1], iz * item_dims[2]


def volume_utilization(used_volume: int, container_volume: int) -> float:
    """
    Simple volume utilisation metric expressed as a percentage (0.0 - 100.0).
    """
    if container_volume <= 0:
        return 0.0
    return float(used_volume) / float(container_volume) * 100.0


def footprint_coverage(
    placements: Sequence[Placement],
    container_length: int,
    container_width: int,
) -> float:
    """
    Compute the percentage of footprint covered in the XY plane using a plane sweep.
    """
    if not placements:
        return 0.0

    container_area = float(container_length * container_width)
    if container_area <= 0:
        return 0.0

    # Axis-aligned rectangle union via plane sweep.
    x_edges: List[int] = []
    rects: List[Tuple[int, int, int, int]] = []
    for placement in placements:
        x0, y0 = placement.x, placement.y
        dx, dy = placement.orientation[0], placement.orientation[1]
        if dx <= 0 or dy <= 0:
            continue
        x1, y1 = x0 + dx, y0 + dy
        x_edges.extend([x0, x1])
        rects.append((x0, x1, y0, y1))

    if not rects:
        return 0.0

    x_edges = sorted(set(x_edges))
    area = 0.0
    for i in range(len(x_edges) - 1):
        x_start, x_end = x_edges[i], x_edges[i + 1]
        if x_end <= x_start:
            continue
        # Collect y-intervals for rectangles spanning this x-slice.
        intervals: List[Tuple[int, int]] = []
        for x0, x1, y0, y1 in rects:
            if x0 <= x_start and x1 >= x_end:
                intervals.append((y0, y1))

        if not intervals:
            continue

        # Merge Y-intervals.
        intervals.sort()
        merged: List[Tuple[int, int]] = []
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if end <= start:
                continue
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))

        slice_width = x_end - x_start
        for y_start, y_end in merged:
            if y_end <= y_start:
                continue
            area += slice_width * (y_end - y_start)

    # Clip to container area.
    area = min(area, container_area)
    return area / container_area * 100.0


def validate_clearances(
    placements: Sequence[Placement],
    minimum_clearance: float = 0.0,
) -> bool:
    """
    Optional helper that checks the minimum centre-to-centre distance between placements.
    Implemented with straightforward pairwise checks to avoid heavy dependencies.
    """
    if len(placements) <= 1:
        return True

    centers = [
        (
            placement.x + placement.orientation[0] / 2.0,
            placement.y + placement.orientation[1] / 2.0,
            placement.z + placement.orientation[2] / 2.0,
        )
        for placement in placements
    ]

    min_distance = float("inf")
    for i in range(len(centers)):
        x1, y1, z1 = centers[i]
        for j in range(i + 1, len(centers)):
            x2, y2, z2 = centers[j]
            distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            if distance < min_distance:
                min_distance = distance
            if min_distance < minimum_clearance:
                return False

    return bool(min_distance >= minimum_clearance)

