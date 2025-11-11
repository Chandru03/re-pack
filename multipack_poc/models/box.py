"""
Data model representing a shipping box.

Dimensions are internal dimensions in millimetres (mm) and max weight is in
grams (g). All checks are performed eagerly to surface invalid configurations
before invoking the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


def _require_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


@dataclass(frozen=True)
class Box:
    """Immutable shipping box used as the bin for cartons."""

    length: float
    width: float
    height: float
    max_weight: float
    tare_weight: float = field(default=0)
    name: str = field(default="Box")

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", float(_require_positive("length", self.length)))
        object.__setattr__(self, "width", float(_require_positive("width", self.width)))
        object.__setattr__(self, "height", float(_require_positive("height", self.height)))
        object.__setattr__(self, "max_weight", float(_require_positive("max_weight", self.max_weight)))
        if self.tare_weight < 0:
            raise ValueError("tare_weight cannot be negative")
        object.__setattr__(self, "tare_weight", float(self.tare_weight))

    @property
    def inner_volume(self) -> float:
        """Return usable internal volume in mm^3."""
        return self.length * self.width * self.height

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return self.length, self.width, self.height

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "name": self.name,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "max_weight": self.max_weight,
            "tare_weight": self.tare_weight,
            "inner_volume": self.inner_volume,
        }

    def scaled_dimensions(self, scale: int) -> Tuple[int, int, int]:
        return tuple(int(round(value * scale)) for value in self.dimensions)

