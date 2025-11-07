"""
Data model representing a finished goods carton/package.

The packing logic assumes all dimensions are expressed in millimetres (mm)
and weight in grams (g). Validation is intentionally strict to prevent
invalid optimisation inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


def _require_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


@dataclass(frozen=True)
class Package:
    """Immutable representation of a shippable carton."""

    length: int
    width: int
    height: int
    weight: int  # grams
    name: str = field(default="Carton")

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", int(_require_positive("length", self.length)))
        object.__setattr__(self, "width", int(_require_positive("width", self.width)))
        object.__setattr__(self, "height", int(_require_positive("height", self.height)))
        object.__setattr__(self, "weight", int(_require_positive("weight", self.weight)))

    @property
    def volume(self) -> int:
        """Return the cubic volume of a single package in mm^3."""
        return self.length * self.width * self.height

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Expose dimensions as an (L, W, H) tuple."""
        return self.length, self.width, self.height

    def to_dict(self) -> Dict[str, int | str]:
        """Serialize the package for reporting."""
        return {
            "name": self.name,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "weight": self.weight,
            "volume": self.volume,
        }

