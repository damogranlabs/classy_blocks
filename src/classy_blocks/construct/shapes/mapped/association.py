"""What each point of a mapped shape is asked to stick to"""

import enum
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from classy_blocks.cbtyping import IndexType, NPPointListType
from classy_blocks.construct.shapes.mapped.features import Features, Names, Target


class Kind(enum.Enum):
    FREE = 0
    SURFACE = 1
    CURVE = 2
    CORNER = 3


@dataclass(frozen=True)
class Label:
    # surfaces of the patches the point is on
    names: Names
    # what Features.find found for them at the point's position
    target: Optional[Target]

    @property
    def kind(self) -> Kind:
        """A point whose feature is missing is free"""
        return Kind(min(len(self.names), 3) * (self.target is not None))

    @property
    def error(self) -> str:
        missing = bool(self.names) and self.target is None
        return f"No feature where {', '.join(sorted(self.names))} meet" * missing


def associate(points: NPPointListType, patches: dict[str, list[IndexType]], features: Features) -> list[Label]:
    """Labels for every point; patches without a surface of the same name don't count"""
    names: dict[int, set[str]] = defaultdict(set)
    surfaced = {name: faces for name, faces in patches.items() if name in features.surfaces}
    for name, faces in surfaced.items():
        for point in np.unique(np.asarray(faces, dtype=int)):
            names[int(point)].add(name)

    found = [frozenset(names[point]) for point in range(len(points))]
    return [Label(point_names, features.find(point_names, point)) for point_names, point in zip(found, points)]
