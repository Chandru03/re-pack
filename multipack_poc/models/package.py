"""
Data model representing a finished goods carton/package.

The packing logic assumes all dimensions are expressed in millimetres (mm)
and weight in grams (g). Validation is intentionally strict to prevent
invalid optimisation inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


def _require_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


@dataclass(frozen=True)
class Package:
    """Immutable representation of a shippable carton."""

    length: float
    width: float
    height: float
    weight: float  # grams
    name: str = field(default="Carton")
    thickness_axis: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", float(_require_positive("length", self.length)))
        object.__setattr__(self, "width", float(_require_positive("width", self.width)))
        object.__setattr__(self, "height", float(_require_positive("height", self.height)))
        object.__setattr__(self, "weight", float(_require_positive("weight", self.weight)))
        if self.thickness_axis is not None:
            normalized = self.thickness_axis.lower()
            if normalized not in {"length", "width", "height"}:
                raise ValueError(
                    "thickness_axis must be one of {'length', 'width', 'height'} or None"
                )
            object.__setattr__(self, "thickness_axis", normalized)

    @property
    def volume(self) -> float:
        """Return the cubic volume of a single package in mm^3."""
        return self.length * self.width * self.height

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Expose dimensions as an (L, W, H) tuple."""
        return self.length, self.width, self.height

    @property
    def thickness_value(self) -> Optional[float]:
        """Return the dimension value that represents thickness (if configured)."""
        if self.thickness_axis is None:
            return None
        return float(getattr(self, self.thickness_axis))

    def to_dict(self) -> Dict[str, float | str]:
        """Serialize the package for reporting."""
        return {
            "name": self.name,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "weight": self.weight,
            "volume": self.volume,
            "thickness_axis": self.thickness_axis or "",
        }

    def scaled_dimensions(self, scale: int) -> Tuple[int, int, int]:
        """Return scaled integer dimensions for solver usage."""
        return tuple(int(round(value * scale)) for value in self.dimensions)

    def scaled_thickness(self, scale: int) -> Optional[int]:
        value = self.thickness_value
        if value is None:
            return None
        return int(round(value * scale))

