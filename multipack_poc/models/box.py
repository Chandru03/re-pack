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

    length: int
    width: int
    height: int
    max_weight: int
    tare_weight: int = field(default=0)
    name: str = field(default="Box")

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", int(_require_positive("length", self.length)))
        object.__setattr__(self, "width", int(_require_positive("width", self.width)))
        object.__setattr__(self, "height", int(_require_positive("height", self.height)))
        object.__setattr__(self, "max_weight", int(_require_positive("max_weight", self.max_weight)))
        if self.tare_weight < 0:
            raise ValueError("tare_weight cannot be negative")
        object.__setattr__(self, "tare_weight", int(self.tare_weight))

    @property
    def inner_volume(self) -> int:
        """Return usable internal volume in mm^3."""
        return self.length * self.width * self.height

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        return self.length, self.width, self.height

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "name": self.name,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "max_weight": self.max_weight,
            "tare_weight": self.tare_weight,
            "inner_volume": self.inner_volume,
        }

