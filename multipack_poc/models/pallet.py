"""
Pallet model describing the stacking surface used for shipping boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


def _require_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


@dataclass(frozen=True)
class Pallet:
    """Immutable representation of a pallet suitable for optimisation inputs."""

    length: float
    width: float
    height: float  # deck height only
    max_height: float
    max_weight: float
    name: str = field(default="Pallet")

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", float(_require_positive("length", self.length)))
        object.__setattr__(self, "width", float(_require_positive("width", self.width)))
        object.__setattr__(self, "height", float(_require_positive("height", self.height)))
        object.__setattr__(self, "max_height", float(_require_positive("max_height", self.max_height)))
        object.__setattr__(self, "max_weight", float(_require_positive("max_weight", self.max_weight)))
        if self.height > self.max_height:
            raise ValueError("max_height must be >= pallet height")

    @property
    def footprint(self) -> float:
        """Return pallet footprint area in mm^2."""
        return self.length * self.width

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return self.length, self.width, self.height

    def usable_height(self) -> float:
        """Return the maximum allowable stack height (deck height excluded)."""
        return self.max_height - self.height

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "name": self.name,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "max_height": self.max_height,
            "max_weight": self.max_weight,
            "usable_height": self.usable_height(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, int | str]) -> "Pallet":
        """Instantiate from a raw configuration dictionary."""
        return cls(
            name=str(payload.get("name", "Pallet")),
            length=float(payload["length"]),
            width=float(payload["width"]),
            height=float(payload["height"]),
            max_height=float(payload["max_height"]),
            max_weight=float(payload["max_weight"]),
        )

    def scaled_dimensions(self, scale: int) -> Tuple[int, int, int]:
        return tuple(int(round(value * scale)) for value in self.dimensions)

